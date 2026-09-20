from datetime import datetime, timezone

import pytest

from src.database.repositories import InMemoryDatabase, WorkflowEventRepository
from src.events import EventBus, EventType, WorkflowEvent


def test_event_lifecycle_persists_processing_and_completion_timestamps() -> None:
    repository = WorkflowEventRepository(InMemoryDatabase())
    bus = EventBus(repository)
    event = WorkflowEvent(
        event_type=EventType.REPORT_UPLOADED,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    bus.subscribe(EventType.REPORT_UPLOADED, lambda _: "handled")

    assert bus.publish(event) == ["handled"]
    stored = repository.get(event.event_id)
    assert stored is not None
    assert stored["status"] == "COMPLETED"
    assert stored["created_at"]
    assert stored["processing_at"]
    assert stored["completed_at"]
    assert stored["failed_at"] is None
    assert stored["error"] is None


def test_event_lifecycle_records_handler_failure() -> None:
    repository = WorkflowEventRepository(InMemoryDatabase())
    bus = EventBus(repository)
    event = WorkflowEvent(event_type=EventType.REPORT_UPLOADED)

    def fail(_: WorkflowEvent) -> None:
        raise RuntimeError("handler failed")

    bus.subscribe(EventType.REPORT_UPLOADED, fail)
    with pytest.raises(RuntimeError, match="handler failed"):
        bus.publish(event)

    stored = repository.get(event.event_id)
    assert stored is not None
    assert stored["status"] == "FAILED"
    assert stored["processing_at"]
    assert stored["failed_at"]
    assert stored["completed_at"] is None
    assert stored["error"] == "handler failed"
