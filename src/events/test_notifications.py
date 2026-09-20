from src.database.repositories import InMemoryDatabase, NotificationRepository, WorkflowEventRepository
from src.events import EventBus, EventType, WorkflowEvent
from src.events.notifications import NotificationEventHandler


def test_notification_handler_persists_required_role_notifications() -> None:
    database = InMemoryDatabase()
    event_repository = WorkflowEventRepository(database)
    notification_repository = NotificationRepository(database)
    bus = EventBus(event_repository)
    NotificationEventHandler(notification_repository, bus)

    events = [
        WorkflowEvent(
            event_type=EventType.NURSE_INPUT_REQUIRED,
            patient_id="P001",
            admission_id="A001",
            payload={"missing_fields": ["HR"]},
        ),
        WorkflowEvent(
            event_type=EventType.ASSESSMENT_COMPLETED,
            patient_id="P001",
            admission_id="A001",
        ),
        WorkflowEvent(
            event_type=EventType.RISK_PRIORITIZATION_COMPLETED,
            patient_id="P001",
            admission_id="A001",
            payload={"priority_level": "HIGH"},
        ),
    ]

    for event in events:
        bus.publish(event)

    nurse_notifications = notification_repository.list_notifications("NURSE")
    doctor_notifications = notification_repository.list_notifications("DOCTOR")
    assert [item["notification_type"] for item in nurse_notifications] == ["NURSE_INPUT_REQUIRED"]
    assert {item["notification_type"] for item in doctor_notifications} == {
        "ASSESSMENT_COMPLETED",
        "RISK_PRIORITIZATION_COMPLETED",
    }
    assert nurse_notifications[0]["event_id"] == events[0].event_id
    assert doctor_notifications[0]["patient_id"] == "P001"
