from datetime import datetime, timezone

import pytest

from src.agents.risk_prioritization_event_workflow import TrendCompletedRiskPrioritizationHandler
from src.database.repositories import InMemoryDatabase, PatientRepository, WorkflowEventRepository
from src.events import EventBus, EventType, WorkflowEvent


SUCCESS = {"probability": 0.7, "status": "success"}


@pytest.fixture
def context() -> tuple[PatientRepository, WorkflowEventRepository, EventBus]:
    return make_context()


def make_context() -> tuple[PatientRepository, WorkflowEventRepository, EventBus]:
    database = InMemoryDatabase()
    repository = PatientRepository(database)
    for patient_id, admission_id in (("P001", "A001"), ("P001", "A002"), ("P002", "A003")):
        if repository.get_patient(patient_id) is None:
            repository.create_patient(patient_id)
        repository.create_admission(patient_id, admission_id, datetime(2026, 1, 1, tzinfo=timezone.utc))
    events = WorkflowEventRepository(database)
    return repository, events, EventBus(events)


def assessment(repository: PatientRepository, patient_id: str = "P001", admission_id: str = "A001", day: int = 1, sepsis: float = 0.7, aki: float = 0.6) -> dict:
    return repository.add_assessment(
        patient_id,
        admission_id,
        datetime(2026, 1, day, tzinfo=timezone.utc),
        sepsis_result={**SUCCESS, "probability": sepsis, "disease": "Sepsis"},
        aki_result={**SUCCESS, "probability": aki, "disease": "AKI"},
    )


def trends(repository: PatientRepository, current: dict, trend: str = "WORSENING") -> list[dict]:
    return [
        repository.add_trend("P001", "A001", disease, current["assessment_time"], trend, 0.2, latest, latest - 0.2, 2, current["assessment_id"], {"trend": trend})
        for disease, latest in (("Sepsis", 0.7), ("AKI", 0.6))
    ]


def event(patient_id: str, admission_id: str, trend_records: list[dict], event_id: str = "trend-event") -> WorkflowEvent:
    return WorkflowEvent(
        event_id=event_id,
        event_type=EventType.TREND_ANALYSIS_COMPLETED,
        patient_id=patient_id,
        admission_id=admission_id,
        payload={"trend_ids": [record["_id"] for record in trend_records]},
    )


def test_valid_flow_persists_before_completion_and_preserves_scope_and_reason(context) -> None:
    repository, events, bus = context
    current = assessment(repository)
    records = trends(repository, current)
    observed: list[bool] = []
    bus.subscribe(EventType.RISK_PRIORITIZATION_COMPLETED, lambda _: observed.append(repository.get_latest_prioritization("P001", "A001") is not None))
    TrendCompletedRiskPrioritizationHandler(repository, bus)

    bus.publish(event("P001", "A001", records))

    stored = repository.get_latest_prioritization("P001", "A001")
    assert stored["patient_id"] == "P001"
    assert stored["admission_id"] == "A001"
    assert stored["source_trend_id"] == records[0]["_id"]
    assert stored["reason"]
    assert observed == [True]
    assert len(events.list_events(event_type=EventType.RISK_PRIORITIZATION_COMPLETED)) == 1


def test_history_is_patient_and_admission_specific(context) -> None:
    repository, events, bus = context
    current = assessment(repository)
    records = trends(repository, current)
    other_patient = assessment(repository, "P002", "A003", 2)
    trends(repository, other_patient)
    other_admission = assessment(repository, "P001", "A002", 2)
    trends(repository, other_admission)
    TrendCompletedRiskPrioritizationHandler(repository, bus)

    bus.publish(event("P001", "A001", records))

    assert repository.get_patient_history("P001", "A001")["latest_prioritization"]["admission_id"] == "A001"
    assert repository.get_latest_prioritization("P002", "A003") is None
    assert repository.get_latest_prioritization("P001", "A002") is None
    assert len(events.list_events(event_type=EventType.RISK_PRIORITIZATION_COMPLETED)) == 1


def test_current_and_previous_assessment_and_trend_history_are_passed(context) -> None:
    repository, _, bus = context
    previous = assessment(repository, day=1, sepsis=0.2, aki=0.2)
    current = assessment(repository, day=2, sepsis=0.8, aki=0.7)
    old_trends = trends(repository, previous, "STABLE")
    current_trends = trends(repository, current, "WORSENING")
    seen: list[tuple[list[str], list[str]]] = []

    class RecordingAgent:
        def prioritize_history(self, assessments, trend_records):
            seen.append(([item["assessment_id"] for item in assessments], [item["_id"] for item in trend_records]))
            return {"status": "success", "priority": "HIGH", "highest_risk_disease": "Sepsis", "highest_probability": 0.8, "worsening_diseases": ["Sepsis"], "reasons": ["current and trajectory"]}

    TrendCompletedRiskPrioritizationHandler(repository, bus, RecordingAgent())
    bus.publish(event("P001", "A001", current_trends))

    assert seen == [([previous["assessment_id"], current["assessment_id"]], [item["_id"] for item in old_trends + current_trends])]


def test_worsening_and_improving_trajectories_are_deterministic(context) -> None:
    repository, _, bus = context
    current = assessment(repository)
    worsening = trends(repository, current, "WORSENING")
    TrendCompletedRiskPrioritizationHandler(repository, bus)
    bus.publish(event("P001", "A001", worsening))
    worsening_result = repository.get_latest_prioritization("P001", "A001")
    assert "worsening" in worsening_result["reason"].lower()

    repository2, events2, bus2 = make_context()
    current2 = assessment(repository2)
    improving = trends(repository2, current2, "IMPROVING")
    TrendCompletedRiskPrioritizationHandler(repository2, bus2)
    bus2.publish(event("P001", "A001", improving))
    improving_result = repository2.get_latest_prioritization("P001", "A001")
    assert "worsening" not in improving_result["reason"].lower()
    assert len(events2.list_events(event_type=EventType.RISK_PRIORITIZATION_COMPLETED)) == 1


def test_missing_history_or_malformed_reference_does_not_complete(context) -> None:
    repository, events, bus = context
    TrendCompletedRiskPrioritizationHandler(repository, bus)
    bus.publish(event("P001", "A001", [{"_id": "missing"}]))
    bus.publish(WorkflowEvent(event_type=EventType.TREND_ANALYSIS_COMPLETED, patient_id="P001", admission_id="A001", payload={"trend_id": "missing"}))
    assert repository.get_latest_prioritization("P001", "A001") is None
    assert events.list_events(event_type=EventType.RISK_PRIORITIZATION_COMPLETED) == []


def test_insufficient_assessment_data_is_persisted_without_success_event(context) -> None:
    repository, events, bus = context
    current = repository.add_assessment("P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc), sepsis_result={"status": "insufficient_data"})
    records = [repository.add_trend("P001", "A001", "Sepsis", current["assessment_time"], "INSUFFICIENT_DATA", 0.0, 0.0, 0.0, 1, current["assessment_id"])]
    TrendCompletedRiskPrioritizationHandler(repository, bus)
    bus.publish(event("P001", "A001", records))
    assert repository.get_latest_prioritization("P001", "A001")["priority_level"] == "INSUFFICIENT_DATA"
    assert events.list_events(event_type=EventType.RISK_PRIORITIZATION_COMPLETED) == []


def test_duplicate_delivery_is_idempotent_and_persistence_failure_completes_never(context) -> None:
    repository, events, bus = context
    current = assessment(repository)
    records = trends(repository, current)
    TrendCompletedRiskPrioritizationHandler(repository, bus)
    bus.publish(event("P001", "A001", records, "first"))
    bus.publish(event("P001", "A001", records, "second"))
    assert len(repository.get_patient_history("P001", "A001")["assessments"]) == 1
    assert len(events.list_events(event_type=EventType.RISK_PRIORITIZATION_COMPLETED)) == 1

    repository2, events2, bus2 = make_context()
    current2 = assessment(repository2)
    records2 = trends(repository2, current2)
    repository2.add_prioritization = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("persist failed"))
    TrendCompletedRiskPrioritizationHandler(repository2, bus2)
    bus2.publish(event("P001", "A001", records2))
    assert events2.list_events(event_type=EventType.RISK_PRIORITIZATION_COMPLETED) == []
