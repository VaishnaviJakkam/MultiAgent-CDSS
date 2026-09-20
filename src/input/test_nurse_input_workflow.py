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
from src.input.nurse_input_workflow import NurseInputReceivedEventHandler, NurseInputSubmissionService


@pytest.fixture
def context() -> tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]:
    database = InMemoryDatabase()
    patients = PatientRepository(database)
    patients.create_patient("P001")
    patients.create_patient("P002")
    patients.create_admission("P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc))
    patients.create_admission("P002", "A002", datetime(2026, 1, 1, tzinfo=timezone.utc))
    reports = LabReportRepository(database)
    reports.create_report(
        report_id="R001",
        patient_id="P001",
        admission_id="A001",
        report_date=None,
        filename="labs.pdf",
        content_type="application/pdf",
        processing_status=ReportProcessingStatus.PROCESSED,
        extracted_data={"WBC": {"value": 8.5}, "Lactate": {"value": 1.2}},
    )
    events = WorkflowEventRepository(database)
    return patients, reports, WorkflowTaskRepository(database), events, EventBus(events)


def create_task(tasks: WorkflowTaskRepository, task_id: str = "T001", patient_id: str = "P001", admission_id: str = "A001") -> dict:
    return tasks.create_task(
        task_id=task_id,
        task_type=TaskType.NURSE_INPUT,
        patient_id=patient_id,
        admission_id=admission_id,
        required_fields=["HR", "O2Sat", "Temp", "BP", "Resp"],
        provided_fields=["WBC", "Lactate"],
        metadata={"report_id": "R001"},
    )


def nurse_values() -> dict:
    return {
        "HR": 90,
        "O2Sat": 97,
        "Temp": 37.1,
        "BP": {"SBP": 120, "DBP": 65},
        "Resp": 18,
    }


def attach_handler(context: tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]:
    patients, reports, tasks, events, bus = context
    NurseInputReceivedEventHandler(tasks, reports, patients, bus)
    return context


def test_valid_submission_merges_values_creates_observation_and_completes_task(context: tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    patients, reports, tasks, events, bus = attach_handler(context)
    create_task(tasks)
    service = NurseInputSubmissionService(tasks, reports, bus)

    service.submit_nurse_input("T001", "P001", "A001", nurse_values())

    task = tasks.get_task("T001")
    assert task is not None
    assert task["status"] == TaskStatus.COMPLETED.value
    assert task["completed_at"] is not None
    observations = patients.get_patient_observations("P001", "A001")
    assert len(observations) == 1
    observation = observations[0]
    assert observation["clinical_parameters"]["WBC"] == 8.5
    assert observation["clinical_parameters"]["HR"] == 90
    assert observation["clinical_parameters"]["MAP"] == 83.33
    assert observation["source"] == "merged_report_nurse"
    assert observation["task_id"] == "T001"
    assert observation["report_id"] == "R001"
    assert events.list_events(event_type=EventType.NURSE_INPUT_RECEIVED)[0]["payload"]["provided_fields"] == ["DBP", "HR", "MAP", "O2Sat", "Resp", "SBP", "Temp"]
    ready = events.list_events(event_type=EventType.OBSERVATION_READY)
    assert len(ready) == 1
    assert ready[0]["payload"]["observation_id"] == observation["observation_id"]


def test_submission_moves_task_to_in_progress_before_handler_completes(context: tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    _, reports, tasks, events, bus = context
    create_task(tasks)
    service = NurseInputSubmissionService(tasks, reports, bus)

    service.submit_nurse_input("T001", "P001", "A001", nurse_values())

    task = tasks.get_task("T001")
    assert task is not None
    assert task["status"] == TaskStatus.IN_PROGRESS.value
    assert events.list_events(event_type=EventType.NURSE_INPUT_RECEIVED)[0]["related_task_id"] == "T001"

    # The handler can be attached independently by the application composition root.
    NurseInputReceivedEventHandler(tasks, reports, context[0], bus)
    bus.publish(WorkflowEvent(
        event_id="nurse-retry",
        event_type=EventType.NURSE_INPUT_RECEIVED,
        patient_id="P001",
        admission_id="A001",
        related_report_id="R001",
        related_task_id="T001",
        payload={"task_id": "T001", "report_id": "R001"},
    ))
    assert tasks.get_task("T001")["status"] == TaskStatus.COMPLETED.value


def test_invalid_or_terminal_tasks_are_rejected(context: tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    _, reports, tasks, _, bus = context
    service = NurseInputSubmissionService(tasks, reports, bus)

    with pytest.raises(ValueError, match="Task not found"):
        service.submit_nurse_input("missing", "P001", "A001", nurse_values())

    task = create_task(tasks)
    tasks.update_task(task["task_id"], status=TaskStatus.CANCELLED)
    with pytest.raises(ValueError, match="CANCELLED"):
        service.submit_nurse_input(task["task_id"], "P001", "A001", nurse_values())

    task = create_task(tasks, "T002")
    tasks.update_task(task["task_id"], status=TaskStatus.COMPLETED)
    with pytest.raises(ValueError, match="COMPLETED"):
        service.submit_nurse_input(task["task_id"], "P001", "A001", nurse_values())


def test_patient_admission_and_report_mismatches_are_rejected(context: tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    _, reports, tasks, _, bus = context
    service = NurseInputSubmissionService(tasks, reports, bus)
    create_task(tasks)

    with pytest.raises(ValueError, match="do not match"):
        service.submit_nurse_input("T001", "P002", "A002", nurse_values())

    tasks.update_task("T001", metadata={"report_id": "missing-report"})
    with pytest.raises(ValueError, match="Report not found"):
        service.submit_nurse_input("T001", "P001", "A001", nurse_values())


def test_partial_input_does_not_publish_or_complete_task(context: tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    _, reports, tasks, events, bus = context
    create_task(tasks)
    service = NurseInputSubmissionService(tasks, reports, bus)

    service.submit_nurse_input("T001", "P001", "A001", {"HR": 90})

    task = tasks.get_task("T001")
    assert task["status"] == TaskStatus.IN_PROGRESS.value
    assert task["metadata"]["answers"] == {"HR": 90}
    assert len(events.list_events(event_type=EventType.NURSE_INPUT_RECEIVED)) == 1
    assert events.list_events(event_type=EventType.OBSERVATION_READY) == []


def test_duplicate_nurse_event_creates_one_observation_and_one_ready_event(context: tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    patients, reports, tasks, events, bus = attach_handler(context)
    create_task(tasks)
    service = NurseInputSubmissionService(tasks, reports, bus)
    service.submit_nurse_input("T001", "P001", "A001", nurse_values())
    received = events.list_events(event_type=EventType.NURSE_INPUT_RECEIVED)[0]

    bus.publish(WorkflowEvent(
        event_id=received["event_id"],
        event_type=EventType.NURSE_INPUT_RECEIVED,
        patient_id="P001",
        admission_id="A001",
        related_report_id="R001",
        related_task_id="T001",
        payload={"task_id": "T001", "report_id": "R001"},
        timestamp=received["timestamp"],
    ))

    assert len(patients.get_patient_observations("P001", "A001")) == 1
    assert tasks.get_task("T001")["status"] == TaskStatus.COMPLETED.value
    assert len(events.list_events(event_type=EventType.OBSERVATION_READY)) == 1


def test_processing_failure_leaves_task_in_progress_without_observation_or_ready_event(context: tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    patients, reports, tasks, events, bus = context
    create_task(tasks)

    class FailingProcessor:
        def process(self, **_: object) -> dict:
            raise RuntimeError("observation failure")

    NurseInputReceivedEventHandler(tasks, reports, patients, bus, data_processor=FailingProcessor())
    NurseInputSubmissionService(tasks, reports, bus).submit_nurse_input("T001", "P001", "A001", nurse_values())

    task = tasks.get_task("T001")
    assert task is not None
    assert task["status"] == TaskStatus.IN_PROGRESS.value
    assert task["metadata"]["processing_error"] == "observation failure"
    assert patients.get_patient_observations("P001", "A001") == []
    assert events.list_events(event_type=EventType.OBSERVATION_READY) == []


def test_observation_and_completed_task_exist_before_ready_event(context: tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    patients, reports, tasks, _, bus = context
    create_task(tasks)

    def verify_ready(event: WorkflowEvent) -> None:
        assert patients.get_observation_by_task("P001", "A001", "T001") is not None
        assert tasks.get_task("T001")["status"] == TaskStatus.COMPLETED.value

    NurseInputReceivedEventHandler(tasks, reports, patients, bus)
    bus.subscribe(EventType.OBSERVATION_READY, verify_ready)
    NurseInputSubmissionService(tasks, reports, bus).submit_nurse_input("T001", "P001", "A001", nurse_values())


def test_audio_answers_are_presented_and_processed_one_question_at_a_time(context: tuple[PatientRepository, LabReportRepository, WorkflowTaskRepository, WorkflowEventRepository, EventBus]) -> None:
    patients, reports, tasks, events, bus = attach_handler(context)
    create_task(tasks)
    service = NurseInputSubmissionService(tasks, reports, bus)

    class FakeAudioAgent:
        values = {
            "hr.wav": {"HR": 90},
            "o2.wav": {"O2Sat": 97},
            "temp.wav": {"Temp": 37.1},
            "bp.wav": {"SBP": 120, "DBP": 65, "MAP": 83.33},
            "resp.wav": {"Resp": 18},
        }

        def analyze(self, audio_path: str) -> dict:
            return {"extracted_parameters": self.values[audio_path]}

    audio = FakeAudioAgent()
    bus.publish(WorkflowEvent(
        event_type=EventType.NURSE_INPUT_REQUIRED,
        patient_id="P001",
        admission_id="A001",
        related_report_id="R001",
        related_task_id="T001",
        payload={"task_id": "T001", "report_id": "R001", "question_index": 0, "question_field": "HR"},
    ))
    for audio_path in ("hr.wav", "o2.wav", "temp.wav", "bp.wav", "resp.wav"):
        service.submit_nurse_audio("T001", "P001", "A001", audio_path, audio)

    task = tasks.get_task("T001")
    assert task["status"] == TaskStatus.COMPLETED.value
    assert patients.get_patient_observations("P001", "A001")
    questions = events.list_events(event_type=EventType.NURSE_INPUT_REQUIRED)
    assert [item["payload"]["question_field"] for item in questions] == ["HR", "O2Sat", "Temp", "BP", "Resp"]
    assert len(events.list_events(event_type=EventType.OBSERVATION_READY)) == 1
