from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LabUploadResponse(BaseModel):
    report_id: str
    status: str
    workflow_started: bool
    task_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    mongodb: bool


class PatientHistoryResponse(BaseModel):
    patient: dict[str, Any] | None
    admission: dict[str, Any]
    observations: list[dict[str, Any]]
    assessments: list[dict[str, Any]]
    trends: list[dict[str, Any]]
    latest_prioritization: dict[str, Any] | None = None


class NotificationResponse(BaseModel):
    notification_id: str
    role: str
    notification_type: str
    message: str
    patient_id: str | None = None
    admission_id: str | None = None
    event_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    read: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class WorkflowTaskResponse(BaseModel):
    task_id: str
    task_type: str
    patient_id: str
    admission_id: str
    status: str
    required_fields: list[str] = Field(default_factory=list)
    provided_fields: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NurseAudioResponse(BaseModel):
    completed: bool
    next_question: str | None = None
    question_index: int | None = None
    total_questions: int | None = None
    observation_id: str | None = None
    assessment_started: bool = False


class DoctorPatientResponse(BaseModel):
    patient: dict[str, Any]
    admission: dict[str, Any]
    observations: list[dict[str, Any]]
    assessments: list[dict[str, Any]]
    trends: list[dict[str, Any]]
    prioritizations: list[dict[str, Any]]
    reports: list[dict[str, Any]]
    tasks: list[dict[str, Any]]


class DashboardPatientSummary(BaseModel):
    patient_id: str
    admission_id: str
    observation_count: int
    assessment_count: int
    sepsis: dict[str, Any]
    aki: dict[str, Any]
    priority: dict[str, Any]
    latest_parameters: dict[str, Any]
    latest_report: dict[str, Any] | None = None
    pending_task_count: int = 0
    latest_assessment: dict[str, Any] | None = None
    latest_trend: dict[str, Any] | None = None
    latest_prioritization: dict[str, Any] | None = None


class DashboardResponse(BaseModel):
    status: str
    patient_count: int
    patients: list[DashboardPatientSummary]