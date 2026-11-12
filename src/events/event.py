from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .event_types import EventType


@dataclass(frozen=True)
class WorkflowEvent:
    """A serializable workflow transition shared by agents and persistence."""

    event_type: EventType
    patient_id: str | None = None
    admission_id: str | None = None
    related_report_id: str | None = None
    related_task_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    event_id: str = field(default_factory=lambda: f"event_{uuid4().hex}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            object.__setattr__(self, "event_type", EventType(self.event_type))
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        """Return the MongoDB-compatible event document."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "patient_id": self.patient_id,
            "admission_id": self.admission_id,
            "related_report_id": self.related_report_id,
            "related_task_id": self.related_task_id,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
            "status": self.status,
        }
