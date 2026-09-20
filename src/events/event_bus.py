from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .event import WorkflowEvent
from .event_types import EventType

EventHandler = Callable[[WorkflowEvent], Any]


class EventBus:
    """Synchronous in-process event bus with optional durable event storage."""

    def __init__(self, event_repository: Any | None = None) -> None:
        self.event_repository = event_repository
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        event_type = EventType(event_type)
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def publish(self, event: WorkflowEvent) -> list[Any]:
        """Persist before dispatching so transitions remain auditable."""
        if self.event_repository is not None:
            self.event_repository.append(event.to_dict())

        return [handler(event) for handler in self._handlers[event.event_type]]
