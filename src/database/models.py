from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

COLLECTIONS = (
    "patients",
    "admissions",
    "observations",
    "assessments",
    "trends",
    "prioritizations",
    "workflow_events",
    "workflow_tasks",
    "lab_reports",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value.strip()


def require_timestamp(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    return value


def with_created_updated(document: dict[str, Any]) -> dict[str, Any]:
    timestamp = utc_now()
    document.setdefault("created_at", timestamp)
    document.setdefault("updated_at", timestamp)
    return document
