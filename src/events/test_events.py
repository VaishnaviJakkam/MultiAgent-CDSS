from datetime import datetime, timedelta, timezone

from src.database.repositories import InMemoryDatabase, WorkflowEventRepository
from src.events import EventBus, EventType, WorkflowEvent


def test_workflow_event_serializes_typed_event_and_scope() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event = WorkflowEvent(
        event_type=EventType.REPORT_UPLOADED,
        patient_id="P001",
        admission_id="A001",
        related_report_id="R001",
        payload={"file_name": "report.pdf"},
        timestamp=timestamp,
    )

    document = event.to_dict()

    assert document["event_id"] == event.event_id
    assert document["event_type"] == "REPORT_UPLOADED"
    assert document["patient_id"] == "P001"
    assert document["timestamp"] == timestamp


def test_event_repository_is_idempotent_and_scoped() -> None:
    repository = WorkflowEventRepository(InMemoryDatabase())
    first = WorkflowEvent(
        event_type=EventType.REPORT_UPLOADED,
        patient_id="P001",
        admission_id="A001",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    duplicate = repository.append(first.to_dict())
    repeated = repository.append(first.to_dict())

    assert repeated == duplicate
    assert len(repository.list_events("P001", "A001")) == 1
    assert repository.list_events("P002", "A001") == []


def test_event_repository_orders_events_by_timestamp() -> None:
    repository = WorkflowEventRepository(InMemoryDatabase())
    later = WorkflowEvent(
        event_type=EventType.REPORT_PROCESSED,
        timestamp=datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
    )
    earlier = WorkflowEvent(
        event_type=EventType.REPORT_UPLOADED,
        timestamp=later.timestamp - timedelta(hours=1),
    )

    repository.append(later.to_dict())
    repository.append(earlier.to_dict())

    assert [event["event_type"] for event in repository.list_events()] == [
        "REPORT_UPLOADED",
        "REPORT_PROCESSED",
    ]


def test_event_bus_persists_before_dispatching_handlers() -> None:
    repository = WorkflowEventRepository(InMemoryDatabase())
    bus = EventBus(repository)
    received: list[str] = []

    def handle(event: WorkflowEvent) -> str:
        received.append(event.event_id)
        assert repository.get(event.event_id) is not None
        return "handled"

    bus.subscribe(EventType.REPORT_UPLOADED, handle)
    event = WorkflowEvent(event_type=EventType.REPORT_UPLOADED)

    assert bus.publish(event) == ["handled"]
    assert received == [event.event_id]
    assert repository.get(event.event_id)["event_type"] == "REPORT_UPLOADED"


def test_event_bus_does_not_register_duplicate_handler() -> None:
    bus = EventBus()
    received: list[str] = []

    def handle(event: WorkflowEvent) -> None:
        received.append(event.event_id)

    bus.subscribe(EventType.REPORT_UPLOADED, handle)
    bus.subscribe(EventType.REPORT_UPLOADED, handle)
    bus.publish(WorkflowEvent(event_type=EventType.REPORT_UPLOADED))

    assert len(received) == 1
