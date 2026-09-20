from datetime import datetime, timedelta, timezone

import pytest

from src.agents.trend_event_workflow import AssessmentCompletedTrendHandler
from src.database.repositories import InMemoryDatabase, PatientRepository, WorkflowEventRepository
from src.events import EventBus, EventType, WorkflowEvent


def result(disease: str, probability: float, status: str = "success") -> dict:
    return {
        "patient_id": "P001",
        "disease": disease,
        "model": "stub",
        "prediction": int(probability >= 0.5),
        "probability": probability,
        "threshold": 0.5,
        "risk_level": "ELEVATED" if probability >= 0.5 else "LOWER",
        "observation_count": 1,
        "assessment_time": None,
        "status": status,
        "message": "test",
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


def add_assessment(repository: PatientRepository, patient_id: str, admission_id: str, assessment_time: datetime, sepsis_probability: float, aki_probability: float) -> dict:
    return repository.add_assessment(
        patient_id=patient_id,
        admission_id=admission_id,
        assessment_time=assessment_time,
        sepsis_result=result("Sepsis", sepsis_probability),
        aki_result=result("AKI", aki_probability),
    )


def assessment_event(assessment: dict, patient_id: str | None = None, admission_id: str | None = None, event_id: str = "assessment-event") -> WorkflowEvent:
    return WorkflowEvent(
        event_id=event_id,
        event_type=EventType.ASSESSMENT_COMPLETED,
        patient_id=patient_id or assessment["patient_id"],
        admission_id=admission_id or assessment["admission_id"],
        payload={"assessment_id": assessment["assessment_id"]},
    )


def test_valid_assessment_persists_sepsis_and_aki_trends_and_publishes_completion(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    first = add_assessment(repository, "P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc), 0.2, 0.7)
    latest = add_assessment(repository, "P001", "A001", datetime(2026, 1, 2, tzinfo=timezone.utc), 0.8, 0.2)
    observed_order: list[bool] = []

    def verify_persisted(event: WorkflowEvent) -> None:
        observed_order.append(len(repository.get_trends_by_assessment("P001", "A001", latest["assessment_id"])) == 2)

    bus.subscribe(EventType.TREND_ANALYSIS_COMPLETED, verify_persisted)
    AssessmentCompletedTrendHandler(repository, bus)
    bus.publish(assessment_event(latest))

    trends = repository.get_trends_by_assessment("P001", "A001", latest["assessment_id"])
    assert len(trends) == 2
    by_disease = {trend["disease"]: trend for trend in trends}
    assert by_disease["Sepsis"]["trend"] == "WORSENING"
    assert by_disease["AKI"]["trend"] == "IMPROVING"
    assert all(trend["patient_id"] == "P001" and trend["admission_id"] == "A001" for trend in trends)
    assert all(trend["assessment_id"] == latest["assessment_id"] for trend in trends)
    completion = events.list_events(event_type=EventType.TREND_ANALYSIS_COMPLETED)
    assert len(completion) == 1
    assert completion[0]["payload"]["assessment_id"] == latest["assessment_id"]
    assert observed_order == [True]


def test_history_is_scoped_to_patient_and_admission(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    add_assessment(repository, "P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc), 0.2, 0.2)
    latest = add_assessment(repository, "P001", "A001", datetime(2026, 1, 2, tzinfo=timezone.utc), 0.8, 0.8)
    add_assessment(repository, "P001", "A002", datetime(2026, 1, 1, tzinfo=timezone.utc), 0.1, 0.1)
    add_assessment(repository, "P002", "A003", datetime(2026, 1, 1, tzinfo=timezone.utc), 0.1, 0.1)
    AssessmentCompletedTrendHandler(repository, bus)

    bus.publish(assessment_event(latest))

    trends = repository.get_trends_by_assessment("P001", "A001", latest["assessment_id"])
    assert {trend["trend"] for trend in trends} == {"WORSENING"}
    assert repository.get_patient_trends("P001", "A002") == []
    assert repository.get_patient_trends("P002", "A003") == []
    assert len(events.list_events(event_type=EventType.TREND_ANALYSIS_COMPLETED)) == 1


def test_chronological_assessment_history_is_passed_to_agent(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, _, bus = context
    first = add_assessment(repository, "P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc), 0.2, 0.2)
    latest = add_assessment(repository, "P001", "A001", datetime(2026, 1, 3, tzinfo=timezone.utc), 0.8, 0.8)
    middle = add_assessment(repository, "P001", "A001", datetime(2026, 1, 2, tzinfo=timezone.utc), 0.5, 0.5)
    seen: list[list[datetime]] = []

    class RecordingTrendAgent:
        def analyze(self, history: dict) -> dict:
            seen.append([item["assessment_time"] for item in history["assessments"]])
            return {"status": "success", "disease_trends": {"Sepsis": {"trend": "WORSENING", "first": 0.2, "latest": 0.8, "change": 0.6, "count": 3}, "AKI": {"trend": "WORSENING", "first": 0.2, "latest": 0.8, "change": 0.6, "count": 3}}}

    AssessmentCompletedTrendHandler(repository, bus, RecordingTrendAgent())
    bus.publish(assessment_event(latest))

    assert seen == [[first["assessment_time"], middle["assessment_time"], latest["assessment_time"]]]


def test_insufficient_history_persists_structured_insufficient_trends_without_completion(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    assessment = add_assessment(repository, "P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc), 0.2, 0.2)
    AssessmentCompletedTrendHandler(repository, bus)

    bus.publish(assessment_event(assessment))

    trends = repository.get_trends_by_assessment("P001", "A001", assessment["assessment_id"])
    assert len(trends) == 2
    assert {trend["trend"] for trend in trends} == {"INSUFFICIENT_DATA"}
    assert events.list_events(event_type=EventType.TREND_ANALYSIS_COMPLETED) == []


def test_duplicate_assessment_event_is_idempotent(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    first = add_assessment(repository, "P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc), 0.2, 0.2)
    latest = add_assessment(repository, "P001", "A001", datetime(2026, 1, 2, tzinfo=timezone.utc), 0.8, 0.8)
    AssessmentCompletedTrendHandler(repository, bus)

    bus.publish(assessment_event(latest, event_id="event-1"))
    bus.publish(assessment_event(latest, event_id="event-2"))

    assert len(repository.get_trends_by_assessment("P001", "A001", latest["assessment_id"])) == 2
    assert len(events.list_events(event_type=EventType.TREND_ANALYSIS_COMPLETED)) == 1


def test_mismatched_assessment_reference_is_rejected(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    assessment = add_assessment(repository, "P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc), 0.2, 0.2)
    AssessmentCompletedTrendHandler(repository, bus)

    bus.publish(assessment_event(assessment, patient_id="P002", admission_id="A003"))

    assert repository.get_patient_trends("P001", "A001") == []
    assert events.list_events(event_type=EventType.TREND_ANALYSIS_COMPLETED) == []


def test_persistence_failure_does_not_publish_completion(context: tuple[PatientRepository, WorkflowEventRepository, EventBus]) -> None:
    repository, events, bus = context
    assessment = add_assessment(repository, "P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc), 0.2, 0.2)
    original = repository.add_trend

    def fail_once(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("trend persistence failed")

    repository.add_trend = fail_once  # type: ignore[method-assign]
    AssessmentCompletedTrendHandler(repository, bus).handle(assessment_event(assessment))
    repository.add_trend = original  # type: ignore[method-assign]

    assert events.list_events(event_type=EventType.TREND_ANALYSIS_COMPLETED) == []
