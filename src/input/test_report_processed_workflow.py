from datetime import datetime, timezone

import pytest

from src.database.lab_reports import ReportProcessingStatus
from src.database.repositories import (
    InMemoryDatabase,
    LabReportRepository,
    PatientRepository,
    WorkflowEventRepository,
    WorkflowTaskRepository,
)
from src.database.workflow_tasks import TaskStatus, TaskType
from src.events import EventBus, EventType, WorkflowEvent
from src.input.report_event_workflow import ReportProcessedEventHandler


@pytest.fixture
def context() -> tuple[LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]:
    database = InMemoryDatabase()
    patients = PatientRepository(database)
    patients.create_patient("P001")
    patients.create_patient("P002")
    patients.create_admission("P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc))
    patients.create_admission("P002", "A002", datetime(2026, 1, 1, tzinfo=timezone.utc))
    events = WorkflowEventRepository(database)
    return LabReportRepository(database), WorkflowTaskRepository(database), events, EventBus(events)


def create_processed_report(repository: LabReportRepository, extracted_data: dict, report_id: str = "R001", patient_id: str = "P001", admission_id: str = "A001") -> dict:
    return repository.create_report(
        report_id=report_id,
        patient_id=patient_id,
        admission_id=admission_id,
        report_date=None,
        filename="labs.pdf",
        content_type="application/pdf",
        storage_reference="reports/labs.pdf",
        processing_status=ReportProcessingStatus.PROCESSED,
        extracted_data=extracted_data,
    )


def processed_event(report: dict, event_id: str = "processed-1") -> WorkflowEvent:
    return WorkflowEvent(
        event_id=event_id,
        event_type=EventType.GENERATE_RESULT_REQUESTED,
        patient_id=report["patient_id"],
        admission_id=report["admission_id"],
        related_report_id=report["report_id"],
        payload={"report_id": report["report_id"], "processing_status": "PROCESSED"},
    )


def all_sepsis_fields() -> dict:
    return {
        "HR": 90,
        "O2Sat": 97,
        "Temp": 37.1,
        "SBP": 120,
        "MAP": 85,
        "Resp": 18,
        "WBC": 8.5,
        "Lactate": 1.2,
    }


def test_sufficient_report_publishes_observation_ready_without_task(context: tuple[LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    reports, tasks, events, bus = context
    report = create_processed_report(reports, all_sepsis_fields())
    ReportProcessedEventHandler(reports, tasks, bus)

    bus.publish(processed_event(report))

    assert tasks.get_patient_tasks("P001", "A001") == []
    ready = events.list_events(event_type=EventType.OBSERVATION_READY)
    assert len(ready) == 1
    assert ready[0]["related_report_id"] == "R001"
    assert ready[0]["patient_id"] == "P001"
    assert ready[0]["admission_id"] == "A001"
    assert ready[0]["payload"]["available_fields"] == sorted(all_sepsis_fields())


def test_missing_report_data_creates_pending_nurse_task_and_event(context: tuple[LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    reports, tasks, events, bus = context
    report = create_processed_report(reports, {"WBC": {"value": 8.5}})
    ReportProcessedEventHandler(reports, tasks, bus)

    bus.publish(processed_event(report))

    created = tasks.get_patient_tasks("P001", "A001")
    assert len(created) == 1
    task = created[0]
    assert task["task_type"] == TaskType.NURSE_INPUT.value
    assert task["status"] == TaskStatus.PENDING.value
    assert task["required_fields"] == ["HR", "O2Sat", "Temp", "BP", "Resp", "Lactate"]
    assert task["provided_fields"] == ["WBC"]
    assert task["metadata"]["report_id"] == "R001"

    required = events.list_events(event_type=EventType.NURSE_INPUT_REQUIRED)
    assert len(required) == 1
    assert required[0]["related_task_id"] == task["task_id"]
    assert required[0]["related_report_id"] == "R001"
    assert required[0]["payload"]["missing_fields"] == task["required_fields"]
    assert events.list_events(event_type=EventType.OBSERVATION_READY) == []


def test_task_exists_before_nurse_required_event_is_published(context: tuple[LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    reports, tasks, _, bus = context
    report = create_processed_report(reports, {"WBC": {"value": 8.5}})
    observed: list[str] = []

    def verify_task(event: WorkflowEvent) -> None:
        assert event.related_task_id is not None
        assert tasks.get_task(event.related_task_id) is not None
        observed.append(event.related_task_id)

    ReportProcessedEventHandler(reports, tasks, bus)
    bus.subscribe(EventType.NURSE_INPUT_REQUIRED, verify_task)
    bus.publish(processed_event(report))

    assert len(observed) == 1


def test_duplicate_report_processed_reuses_task_and_event(context: tuple[LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    reports, tasks, events, bus = context
    report = create_processed_report(reports, {"WBC": {"value": 8.5}})
    ReportProcessedEventHandler(reports, tasks, bus)

    bus.publish(processed_event(report, "processed-1"))
    bus.publish(processed_event(report, "processed-2"))

    assert len(tasks.get_patient_tasks("P001", "A001")) == 1
    assert len(events.list_events(event_type=EventType.NURSE_INPUT_REQUIRED)) == 1


def test_existing_pending_task_is_reused(context: tuple[LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    reports, tasks, events, bus = context
    report = create_processed_report(reports, {"WBC": {"value": 8.5}})
    existing = tasks.create_task(
        task_type=TaskType.NURSE_INPUT,
        patient_id="P001",
        admission_id="A001",
        required_fields=["HR", "O2Sat"],
        provided_fields=["WBC"],
        metadata={"report_id": "R001"},
        task_id="existing-task",
    )
    ReportProcessedEventHandler(reports, tasks, bus)

    bus.publish(processed_event(report))

    assert tasks.get_task("existing-task") == existing
    assert len(tasks.get_patient_tasks("P001", "A001")) == 1
    required = events.list_events(event_type=EventType.NURSE_INPUT_REQUIRED)
    assert len(required) == 1
    assert required[0]["related_task_id"] == "existing-task"


def test_report_processed_scope_mismatch_creates_no_task_or_readiness_event(context: tuple[LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    reports, tasks, events, bus = context
    report = create_processed_report(reports, {"WBC": {"value": 8.5}})
    ReportProcessedEventHandler(reports, tasks, bus)

    bus.publish(WorkflowEvent(
        event_id="wrong-scope",
        event_type=EventType.REPORT_PROCESSED,
        patient_id="P002",
        admission_id="A002",
        related_report_id=report["report_id"],
        payload={"report_id": report["report_id"]},
    ))

    assert tasks.get_patient_tasks("P001", "A001") == []
    assert tasks.get_patient_tasks("P002", "A002") == []
    assert events.list_events(event_type=EventType.NURSE_INPUT_REQUIRED) == []
    assert events.list_events(event_type=EventType.OBSERVATION_READY) == []


def test_duplicate_sufficient_events_do_not_duplicate_readiness(context: tuple[LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    reports, _, events, bus = context
    report = create_processed_report(reports, all_sepsis_fields())
    ReportProcessedEventHandler(reports, WorkflowTaskRepository(context[0].database), bus)

    bus.publish(processed_event(report, "processed-1"))
    bus.publish(processed_event(report, "processed-2"))

    assert len(events.list_events(event_type=EventType.OBSERVATION_READY)) == 1
