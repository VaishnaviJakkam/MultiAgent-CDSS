from datetime import datetime, timezone

import pytest

from src.database.repositories import InMemoryDatabase, LabReportRepository, PatientRepository, WorkflowEventRepository
from src.database.lab_reports import ReportProcessingStatus
from src.events import EventBus, EventType, WorkflowEvent
from src.input.report_event_workflow import GenerateResultRequestService, LabReportSubmissionService, ReportProcessingEventHandler


class FakeReportAgent:
    def __init__(self, repository: LabReportRepository | None = None, should_fail: bool = False) -> None:
        self.repository = repository
        self.should_fail = should_fail
        self.calls: list[str] = []

    def analyze(self, storage_reference: str) -> dict:
        self.calls.append(storage_reference)
        if self.repository is not None:
            report = self.repository.get_report("R001")
            assert report is not None
            assert report["processing_status"] == ReportProcessingStatus.PROCESSING.value
        if self.should_fail:
            raise RuntimeError("OCR unavailable")
        return {"lab_results": {"WBC": {"value": 8.5, "unit": "thou/mm3"}}}


@pytest.fixture
def context() -> tuple[LabReportRepository, WorkflowEventRepository, EventBus]:
    database = InMemoryDatabase()
    patients = PatientRepository(database)
    patients.create_patient("P001")
    patients.create_patient("P002")
    patients.create_admission("P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc))
    patients.create_admission("P002", "A002", datetime(2026, 1, 1, tzinfo=timezone.utc))
    event_repository = WorkflowEventRepository(database)
    return LabReportRepository(database), event_repository, EventBus(event_repository)


def submit(repository: LabReportRepository, event_bus: EventBus, report_id: str = "R001", patient_id: str = "P001", admission_id: str = "A001") -> dict:
    return LabReportSubmissionService(repository, event_bus).submit_report(
        report_id=report_id,
        patient_id=patient_id,
        admission_id=admission_id,
        filename="labs.pdf",
        content_type="application/pdf",
        storage_reference="reports/labs.pdf",
    )


def test_submit_persists_before_publishing_upload_event(context: tuple[LabReportRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, event_repository, event_bus = context
    observed: list[dict] = []

    def observe(event: WorkflowEvent) -> None:
        persisted = repository.get_report(event.related_report_id or "")
        assert persisted is not None
        assert persisted["processing_status"] == ReportProcessingStatus.UPLOADED.value
        observed.append(event.to_dict())

    event_bus.subscribe(EventType.REPORT_UPLOADED, observe)
    report = submit(repository, event_bus)

    assert report["processing_status"] == ReportProcessingStatus.UPLOADED.value
    assert len(observed) == 1
    persisted_event = event_repository.list_events(event_type=EventType.REPORT_UPLOADED)[0]
    assert persisted_event["event_id"] == observed[0]["event_id"]
    assert persisted_event["related_report_id"] == "R001"
    assert persisted_event["patient_id"] == "P001"
    assert persisted_event["admission_id"] == "A001"
    assert persisted_event["payload"]["report_id"] == "R001"


def test_uploaded_event_reaches_handler_and_success_persists_processed_event(context: tuple[LabReportRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, event_repository, event_bus = context
    agent = FakeReportAgent(repository)
    ReportProcessingEventHandler(repository, event_bus, agent)

    submit(repository, event_bus)

    report = repository.get_report("R001")
    assert report is not None
    assert agent.calls == ["reports/labs.pdf"]
    assert report["processing_status"] == ReportProcessingStatus.PROCESSED.value
    assert report["extracted_data"]["WBC"]["value"] == 8.5
    processed_events = event_repository.list_events(event_type=EventType.REPORT_PROCESSED)
    assert len(processed_events) == 1
    assert processed_events[0]["payload"]["report_id"] == "R001"
    assert processed_events[0]["payload"]["processing_status"] == "PROCESSED"


def test_processing_failure_marks_report_failed_without_processed_event(context: tuple[LabReportRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, event_repository, event_bus = context
    agent = FakeReportAgent(should_fail=True)
    ReportProcessingEventHandler(repository, event_bus, agent)

    submit(repository, event_bus)

    report = repository.get_report("R001")
    assert report is not None
    assert report["processing_status"] == ReportProcessingStatus.FAILED.value
    assert report["metadata"]["processing_error"] == "OCR unavailable"
    assert event_repository.list_events(event_type=EventType.REPORT_PROCESSED) == []


def test_duplicate_uploaded_delivery_does_not_reprocess_processed_report(context: tuple[LabReportRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, event_repository, event_bus = context
    agent = FakeReportAgent()
    ReportProcessingEventHandler(repository, event_bus, agent)
    submit(repository, event_bus)
    uploaded_event = event_repository.list_events(event_type=EventType.REPORT_UPLOADED)[0]

    event_bus.publish(WorkflowEvent(
        event_type=EventType.REPORT_UPLOADED,
        event_id=uploaded_event["event_id"],
        patient_id="P001",
        admission_id="A001",
        related_report_id="R001",
        payload={"report_id": "R001"},
        timestamp=uploaded_event["timestamp"],
    ))

    assert agent.calls == ["reports/labs.pdf"]
    assert len(event_repository.list_events(event_type=EventType.REPORT_PROCESSED)) == 1


def test_report_scope_mismatch_cannot_process_another_patient_report(context: tuple[LabReportRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, event_repository, event_bus = context
    submit(repository, event_bus, patient_id="P001", admission_id="A001")
    agent = FakeReportAgent()
    handler = ReportProcessingEventHandler(repository, event_bus, agent)

    result = handler.handle(WorkflowEvent(
        event_type=EventType.REPORT_UPLOADED,
        patient_id="P002",
        admission_id="A002",
        related_report_id="R001",
        payload={"report_id": "R001"},
    ))

    assert result is not None
    assert result["processing_status"] == ReportProcessingStatus.FAILED.value
    assert agent.calls == []
    assert len(event_repository.list_events(event_type=EventType.REPORT_PROCESSED)) == 0


def test_handler_without_report_reference_does_nothing(context: tuple[LabReportRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, _, event_bus = context
    agent = FakeReportAgent()
    handler = ReportProcessingEventHandler(repository, event_bus, agent)

    assert handler.handle(WorkflowEvent(event_type=EventType.REPORT_UPLOADED)) is None
    assert agent.calls == []


def test_lab_user_generate_request_is_explicit_and_requires_processed_report(context: tuple[LabReportRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    report = submit(repository, bus)
    service = GenerateResultRequestService(repository, bus)

    assert service.request_result("P001", "A001", report["report_id"]) is None
    repository.update_report(report["report_id"], processing_status=ReportProcessingStatus.PROCESSED, extracted_data={"WBC": {"value": 8.5}})
    service.request_result("P001", "A001", report["report_id"])

    generated = events.list_events(event_type=EventType.GENERATE_RESULT_REQUESTED)
    assert len(generated) == 1
    assert generated[0]["payload"]["report_id"] == report["report_id"]
