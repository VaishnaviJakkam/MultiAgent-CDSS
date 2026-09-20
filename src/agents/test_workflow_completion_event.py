from datetime import datetime, timezone

from src.agents.risk_prioritization_event_workflow import TrendCompletedRiskPrioritizationHandler
from src.database.repositories import InMemoryDatabase, NotificationRepository, PatientRepository, WorkflowEventRepository
from src.events import EventBus, EventType, WorkflowEvent
from src.events.notifications import NotificationEventHandler


class SuccessfulPriorityAgent:
    def prioritize_history(self, assessments, trends):
        return {
            "status": "success",
            "priority": "HIGH",
            "highest_risk_disease": "Sepsis",
            "highest_probability": 0.8,
            "worsening_diseases": ["Sepsis"],
            "reasons": ["Sepsis trajectory is worsening"],
        }


def test_successful_prioritization_publishes_workflow_completed() -> None:
    database = InMemoryDatabase()
    repository = PatientRepository(database)
    repository.create_patient("P001")
    repository.create_admission("P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc))
    assessment = repository.add_assessment(
        "P001",
        "A001",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
        sepsis_result={"status": "success", "probability": 0.8},
        aki_result={"status": "success", "probability": 0.2},
    )
    trend = repository.add_trend(
        "P001",
        "A001",
        "Sepsis",
        assessment["assessment_time"],
        "WORSENING",
        0.2,
        0.8,
        0.6,
        2,
        assessment["assessment_id"],
    )
    events = WorkflowEventRepository(database)
    notifications = NotificationRepository(database)
    bus = EventBus(events)
    NotificationEventHandler(notifications, bus)
    TrendCompletedRiskPrioritizationHandler(repository, bus, SuccessfulPriorityAgent())

    bus.publish(
        WorkflowEvent(
            event_type=EventType.TREND_ANALYSIS_COMPLETED,
            patient_id="P001",
            admission_id="A001",
            payload={"trend_ids": [trend["_id"]]},
        )
    )

    completed = events.list_events(event_type=EventType.WORKFLOW_COMPLETED)
    assert len(completed) == 1
    assert completed[0]["status"] == "COMPLETED"
    assert completed[0]["payload"]["workflow_status"] == "COMPLETED"
    assert completed[0]["payload"]["prioritization_id"]
    assert len(notifications.list_notifications("DOCTOR")) == 1
