from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ReportProcessingStatus(StrEnum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class ReportSource(StrEnum):
    LAB_UPLOAD = "LAB_UPLOAD"


@dataclass(frozen=True)
class LabReport:
    """Native representation of an uploaded laboratory report."""

    report_id: str
    patient_id: str
    admission_id: str
    report_date: datetime | None
    uploaded_at: datetime
    source: ReportSource
    filename: str
    content_type: str
    storage_reference: str | None = None
    processing_status: ReportProcessingStatus = ReportProcessingStatus.UPLOADED
    extracted_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)