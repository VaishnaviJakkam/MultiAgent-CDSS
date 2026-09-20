from __future__ import annotations

from typing import Any

from src.database.repositories import NotificationRepository

from .event import WorkflowEvent
from .event_bus import EventBus
from .event_types import EventType


class NotificationEventHandler:
    """Persist role-targeted notifications for important workflow events."""

    NOTIFICATION_ROLES = {
        EventType.NURSE_INPUT_REQUIRED: "NURSE",
        EventType.ASSESSMENT_COMPLETED: "DOCTOR",
        EventType.RISK_PRIORITIZATION_COMPLETED: "DOCTOR",
    }

    def __init__(self, repository: NotificationRepository, event_bus: EventBus) -> None:
        self.repository = repository
        self.event_bus = event_bus
        for event_type in self.NOTIFICATION_ROLES:
            event_bus.subscribe(event_type, self.handle)

    def handle(self, event: WorkflowEvent) -> dict[str, Any]:
        role = self.NOTIFICATION_ROLES[event.event_type]
        message = self._message(event)
        return self.repository.create_notification(
            role=role,
            notification_type=event.event_type.value,
            message=message,
            patient_id=event.patient_id,
            admission_id=event.admission_id,
            event_id=event.event_id,
            payload=event.payload,
        )

    @staticmethod
    def _message(event: WorkflowEvent) -> str:
        if event.event_type == EventType.NURSE_INPUT_REQUIRED:
            missing = event.payload.get("missing_fields", [])
            return f"Nurse input required: {', '.join(str(item) for item in missing)}"
        if event.event_type == EventType.ASSESSMENT_COMPLETED:
            return "Disease assessment completed."
        return f"Risk prioritization completed: {event.payload.get('priority_level', 'PENDING')}"