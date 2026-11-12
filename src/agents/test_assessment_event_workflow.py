from datetime import datetime, timedelta, timezone

import pytest

from src.agents.assessment_event_workflow import ObservationReadyAssessmentHandler
from src.agents.disease_assessment_agent import DiseaseAssessmentAgent
from src.database.repositories import InMemoryDatabase, PatientRepository, WorkflowEventRepository
from src.events import EventBus, EventType, WorkflowEvent


SUCCESS_RESULT = {
    "prediction": 0,
    "probability": 0.2,
    "threshold": 0.5,
    "risk_level": "LOWER",
    "observation_count": 1,
    "status": "success",
    "message": "test",
}


class RecordingAssessmentAgent:
    def __init__(self, repository: PatientRepository, sepsis_status: str = "success", aki_status: str = "success") -> None:
        self.repository = repository
        self.calls: list[tuple[str, str, str, datetime]] = []
        self.sepsis_status = sepsis_status
        self.aki_status = aki_status

    def analyze(self, patient_id: str, admission_id: str, observation_id: str, assessment_time: datetime) -> dict:
        history = self.repository.get_patient_history(patient_id, admission_id)
        self.calls.append((patient_id, admission_id, observation_id, assessment_time))
        sepsis = {**SUCCESS_RESULT, "disease": "Sepsis", "model": "stub", "status": self.sepsis_status}
        aki = {**SUCCESS_RESULT, "disease": "AKI", "model": "stub", "status": self.aki_status}
        if self.aki_status != "success":
            aki["message"] = "AKI fields unavailable"
        self.repository.add_assessment(
            patient_id=patient_id,
            admission_id=admission_id,
            assessment_time=assessment_time,
            sepsis_result=sepsis,
            aki_result=aki,
            observation_id=observation_id,
        )
        return {
            "status": "completed" if self.sepsis_status == "success" and self.aki_status == "success" else "partial_failure",
            "sepsis": sepsis,
            "aki": aki,
            "history_observation_count": len(history["observations"]),
        }


@pytest.fixture
def context() -> tuple[PatientRepository, WorkflowEventRepository, EventBus]:
    database = InMemoryDatabase()
    repository = PatientRepository(database)
    repository.create_patient("P001")
    repository.create_patient("P002")
    repository.create_admission("P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc))
    repository.create_admission("P001", "A002", datetime(2026, 1, 2, tzinfo=timezone.utc))
    repository.create_admission("P002", "A003", datetime(2026, 1, 3, tzinfo=timezone.utc))
    events = WorkflowEventRepository(database)
    return repository, events, EventBus(events)


def add_observation(repository: PatientRepository, patient_id: str = "P001", admission_id: str = "A001", task_id: str = "T001", report_id: str = "R001") -> dict:
    return repository.add_observation(
        patient_id=patient_id,
        admission_id=admission_id,
        observation_time=datetime(2026, 1, 4, tzinfo=timezone.utc),
        clinical_parameters={"HR": 90, "WBC": 8.5},
        source="merged_report_nurse",
        task_id=task_id,
        report_id=report_id,
    )


def observation_event(observation: dict, event_id: str = "ready-1", patient_id: str | None = None, admission_id: str | None = None) -> WorkflowEvent:
    return WorkflowEvent(
        event_id=event_id,
        event_type=EventType.OBSERVATION_READY,
        patient_id=patient_id or observation["patient_id"],
        admission_id=admission_id or observation["admission_id"],
        related_report_id=observation.get("report_id"),
        related_task_id=observation.get("task_id"),
        payload={"observation_id": observation["observation_id"], "report_id": observation.get("report_id"), "task_id": observation.get("task_id")},
    )


def test_valid_observation_retrieves_history_persists_assessment_and_publishes_completion(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    observation = add_observation(repository)
    agent = RecordingAssessmentAgent(repository)
    observed_order: list[bool] = []

    def verify_persisted(event: WorkflowEvent) -> None:
        stored = repository.get_assessment_by_observation("P001", "A001", observation["observation_id"])
        observed_order.append(stored is not None)

    bus.subscribe(EventType.ASSESSMENT_COMPLETED, verify_persisted)
    ObservationReadyAssessmentHandler(repository, bus, agent)

    bus.publish(observation_event(observation))

    assessment = repository.get_assessment_by_observation("P001", "A001", observation["observation_id"])
    assert assessment is not None
    assert assessment["patient_id"] == "P001"
    assert assessment["admission_id"] == "A001"
    assert assessment["observation_id"] == observation["observation_id"]
    assert assessment["sepsis"]["disease"] == "Sepsis"
    assert assessment["aki"]["disease"] == "AKI"
    assert len(agent.calls) == 1
    assert agent.calls[0][3] == observation["observation_time"]
    completion = events.list_events(event_type=EventType.ASSESSMENT_COMPLETED)
    assert len(completion) == 1
    assert completion[0]["payload"]["assessment_id"] == assessment["assessment_id"]
    assert observed_order == [True]


def test_history_is_patient_and_admission_specific(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, _, bus = context
    first = add_observation(repository, "P001", "A001", "T001", "R001")
    add_observation(repository, "P001", "A002", "T002", "R002")
    add_observation(repository, "P002", "A003", "T003", "R003")
    agent = RecordingAssessmentAgent(repository)
    ObservationReadyAssessmentHandler(repository, bus, agent)

    bus.publish(observation_event(first))

    assert agent.calls == [("P001", "A001", first["observation_id"], first["observation_time"])]
    assert repository.get_assessment_by_observation("P001", "A002", first["observation_id"]) is None
    assert repository.get_patient_assessments("P002", "A003") == []


def test_duplicate_observation_ready_is_idempotent(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    observation = add_observation(repository)
    agent = RecordingAssessmentAgent(repository)
    ObservationReadyAssessmentHandler(repository, bus, agent)

    bus.publish(observation_event(observation, "ready-1"))
    bus.publish(observation_event(observation, "ready-2"))

    assert len(agent.calls) == 1
    assert len(repository.get_patient_assessments("P001", "A001")) == 1
    assert len(events.list_events(event_type=EventType.ASSESSMENT_COMPLETED)) == 1


def test_scope_mismatch_is_rejected_without_assessment(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    observation = add_observation(repository)
    agent = RecordingAssessmentAgent(repository)
    ObservationReadyAssessmentHandler(repository, bus, agent)

    bus.publish(observation_event(observation, patient_id="P002", admission_id="A003"))

    assert agent.calls == []
    assert repository.get_patient_assessments("P001", "A001") == []
    assert events.list_events(event_type=EventType.ASSESSMENT_COMPLETED) == []


def test_partial_failure_preserves_results_without_claiming_full_completion(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    observation = add_observation(repository)
    agent = RecordingAssessmentAgent(repository, aki_status="insufficient_data")
    ObservationReadyAssessmentHandler(repository, bus, agent)

    bus.publish(observation_event(observation))

    assessment = repository.get_assessment_by_observation("P001", "A001", observation["observation_id"])
    assert assessment is not None
    assert assessment["sepsis"]["status"] == "success"
    assert assessment["aki"]["status"] == "insufficient_data"
    assert events.list_events(event_type=EventType.ASSESSMENT_COMPLETED) == []


def test_real_disease_orchestrator_reports_insufficient_data_without_padding(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    observation = add_observation(repository)

    class SepsisStub:
        def __init__(self) -> None:
            self.history_lengths: list[int] = []

        def analyze(self, history: list[dict]) -> dict:
            self.history_lengths.append(len(history))
            raise ValueError("Sepsis LSTM requires at least 12 real observations; received 1")

    class AkiStub:
        def analyze(self, data: dict) -> dict:
            return {"disease": "AKI", "status": "insufficient_data", "message": "Missing required model features"}

    sepsis = SepsisStub()
    agent = DiseaseAssessmentAgent(repository, sepsis, AkiStub())
    ObservationReadyAssessmentHandler(repository, bus, agent)

    bus.publish(observation_event(observation))

    assessment = repository.get_assessment_by_observation("P001", "A001", observation["observation_id"])
    assert assessment is not None
    assert assessment["sepsis"]["status"] == "insufficient_data"
    assert assessment["aki"]["status"] == "insufficient_data"
    assert sepsis.history_lengths == [1]
    assert events.list_events(event_type=EventType.ASSESSMENT_COMPLETED) == []


def test_missing_observation_event_reference_is_ignored(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    agent = RecordingAssessmentAgent(repository)
    ObservationReadyAssessmentHandler(repository, bus, agent)

    bus.publish(WorkflowEvent(
        event_type=EventType.OBSERVATION_READY,
        patient_id="P001",
        admission_id="A001",
        payload={},
    ))

    assert agent.calls == []
    assert events.list_events(event_type=EventType.ASSESSMENT_COMPLETED) == []
