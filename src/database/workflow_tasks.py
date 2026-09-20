from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class TaskType(StrEnum):
    NURSE_INPUT = "NURSE_INPUT"


@dataclass(frozen=True)
class WorkflowTask:
    """Native representation of a human-action workflow task."""

    task_id: str
    task_type: TaskType
    patient_id: str
    admission_id: str
    status: TaskStatus = TaskStatus.PENDING
    required_fields: list[str] = field(default_factory=list)
    provided_fields: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
