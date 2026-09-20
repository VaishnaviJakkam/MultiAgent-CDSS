from __future__ import annotations

import shutil
import tempfile
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Query
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.database.mongodb import MongoDatabase
from src.database.repositories import (
    LabReportRepository,
    NotificationRepository,
    PatientRepository,
    WorkflowEventRepository,
    WorkflowTaskRepository,
)
from src.database.workflow_tasks import TaskStatus
from src.events import EventBus, EventType, WorkflowEvent
from src.events.notifications import NotificationEventHandler

from src.agents.sepsis_agent import (
    SepsisDetectionAgent,
)

from src.agents.aki_agent import (
    AKIDetectionAgent,
)

from src.agents.disease_assessment_agent import DiseaseAssessmentAgent
from src.agents.assessment_event_workflow import ObservationReadyAssessmentHandler
from src.agents.risk_prioritization_event_workflow import TrendCompletedRiskPrioritizationHandler
from src.agents.trend_event_workflow import AssessmentCompletedTrendHandler
from src.input.nurse_input_workflow import NurseInputReceivedEventHandler, NurseInputSubmissionService
from src.input.report_event_workflow import (
    GenerateResultRequestService,
    LabReportSubmissionService,
    ReportProcessedEventHandler,
    ReportProcessingEventHandler,
)
from src.api.schemas import (
    DashboardResponse,
    DoctorPatientResponse,
    HealthResponse,
    LabUploadResponse,
    NurseAudioResponse,
    NotificationResponse,
    PatientHistoryResponse,
    WorkflowTaskResponse,
)


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Multi-Agent Clinical Deterioration API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATABASE + EVENT WORKFLOW
# =========================================================

database: Any = None
repository: PatientRepository
event_repository: WorkflowEventRepository
task_repository: WorkflowTaskRepository
lab_report_repository: LabReportRepository
notification_repository: NotificationRepository
event_bus: EventBus
report_submission_service: LabReportSubmissionService
generate_result_service: GenerateResultRequestService
nurse_submission_service: NurseInputSubmissionService


def configure_workflow(
    database_instance: Any,
    sepsis_agent_instance: Any,
    aki_agent_instance: Any,
    report_agent_instance: Any | None = None,
) -> None:
    """Configure the in-process workflow and its persistence dependencies."""
    global database, repository, event_repository, task_repository
    global lab_report_repository, notification_repository, event_bus, report_submission_service
    global generate_result_service, nurse_submission_service

    database = database_instance
    repository = PatientRepository(database)
    event_repository = WorkflowEventRepository(database)
    task_repository = WorkflowTaskRepository(database)
    lab_report_repository = LabReportRepository(database)
    notification_repository = NotificationRepository(database)
    event_bus = EventBus(event_repository)

    assessment_agent = DiseaseAssessmentAgent(
        repository=repository,
        sepsis_tool=sepsis_agent_instance,
        aki_tool=aki_agent_instance,
    )
    ReportProcessingEventHandler(
        lab_report_repository,
        event_bus,
        report_agent=report_agent_instance,
    )
    ReportProcessedEventHandler(
        lab_report_repository,
        task_repository,
        event_bus,
    )
    NurseInputReceivedEventHandler(
        task_repository,
        lab_report_repository,
        repository,
        event_bus,
    )
    ObservationReadyAssessmentHandler(repository, event_bus, assessment_agent)
    AssessmentCompletedTrendHandler(repository, event_bus)
    TrendCompletedRiskPrioritizationHandler(repository, event_bus)
    NotificationEventHandler(notification_repository, event_bus)

    def publish_generate_result_request(event: WorkflowEvent) -> None:
        event_bus.publish(
            WorkflowEvent(
                event_type=EventType.GENERATE_RESULT_REQUESTED,
                patient_id=event.patient_id,
                admission_id=event.admission_id,
                related_report_id=event.related_report_id,
                payload={"report_id": event.related_report_id},
            )
        )

    event_bus.subscribe(EventType.REPORT_PROCESSED, publish_generate_result_request)
    report_submission_service = LabReportSubmissionService(lab_report_repository, event_bus)
    generate_result_service = GenerateResultRequestService(lab_report_repository, event_bus)
    nurse_submission_service = NurseInputSubmissionService(task_repository, lab_report_repository, event_bus)


if os.getenv("MONGODB_URI"):
    configure_workflow(
        MongoDatabase(),
        SepsisDetectionAgent(model="advanced"),
        AKIDetectionAgent(),
    )


# =========================================================
# =========================================================
# TEMP FILE UTILITY
# =========================================================

def save_upload_temporarily(
    upload: UploadFile,
) -> str:

    suffix = Path(
        upload.filename or ""
    ).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        shutil.copyfileobj(
            upload.file,
            temp_file,
        )

        return temp_file.name


# =========================================================
# IDENTIFIER / ADMISSION HELPERS
# =========================================================

def clean_identifier(
    value: str,
    field_name: str,
) -> str:

    cleaned = str(value).strip()

    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} cannot be empty.",
        )

    return cleaned


def require_existing_admission(
    patient_id: str,
    admission_id: str,
) -> dict[str, Any]:

    admission = repository.get_admission(
        patient_id,
        admission_id,
    )

    if admission is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Admission not found: "
                f"{patient_id} / {admission_id}"
            ),
        )

    return admission


# =========================================================
# DASHBOARD HELPERS
# =========================================================

def get_latest_item(
    items: list[dict[str, Any]],
) -> dict[str, Any] | None:

    if not items:
        return None

    return items[-1]


def get_latest_disease_trend(
    trends: list[dict[str, Any]],
    disease: str,
) -> dict[str, Any] | None:

    disease_trends = [
        trend
        for trend in trends
        if str(
            trend.get(
                "disease",
                "",
            )
        ).lower()
        == disease.lower()
    ]

    if not disease_trends:
        return None

    return disease_trends[-1]


def probability_from_assessment(
    assessment: dict[str, Any] | None,
    disease: str,
) -> float | None:

    if not assessment:
        return None

    disease_result = (
        assessment.get(
            disease.lower()
        )
        or assessment.get(
            f"{disease.lower()}_result"
        )
    )

    if not disease_result:
        return None

    probability = disease_result.get(
        "probability"
    )

    if isinstance(
        probability,
        (int, float),
    ):
        return float(probability)

    return None


def risk_level_from_assessment(
    assessment: dict[str, Any] | None,
    disease: str,
) -> str | None:

    if not assessment:
        return None

    disease_result = (
        assessment.get(
            disease.lower()
        )
        or assessment.get(
            f"{disease.lower()}_result"
        )
    )

    if not disease_result:
        return None

    return disease_result.get(
        "risk_level"
    )


def assessment_status(
    assessment: dict[str, Any] | None,
    disease: str,
) -> str:

    if not assessment:
        return "no_data"

    disease_result = (
        assessment.get(
            disease.lower()
        )
        or assessment.get(
            f"{disease.lower()}_result"
        )
    )

    if not disease_result:
        return "no_data"

    return str(
        disease_result.get(
            "status",
            "unknown",
        )
    )


def priority_rank(
    priority: str,
) -> int:

    order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
        "PENDING": 3,
        "UNKNOWN": 4,
    }

    return order.get(
        str(priority).upper(),
        5,
    )


# =========================================================
# HEALTH
# =========================================================

@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Dashboard"],
    summary="Check API and MongoDB connectivity",
)
def health() -> HealthResponse:

    try:
        connected = database.ping()

    except Exception:
        connected = False

    return HealthResponse(status="ok", mongodb=connected)


@app.get(
    "/api/notifications/{role}",
    response_model=list[NotificationResponse],
    tags=["Dashboard"],
    summary="List workflow notifications for a role",
)
def get_notifications(role: str) -> list[dict[str, Any]]:
    try:
        return notification_repository.list_notifications(role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# =========================================================
# EVENT-DRIVEN LAB UPLOAD
# =========================================================

@app.post(
    "/api/lab/upload-report",
    response_model=LabUploadResponse,
    tags=["Lab"],
    summary="Upload a laboratory report and start the workflow",
)
def upload_lab_report(
    patient_id: str = Form(...),
    admission_id: str = Form(...),
    report_file: UploadFile = File(...),
) -> LabUploadResponse:
    patient_id = clean_identifier(patient_id, "patient_id")
    admission_id = clean_identifier(admission_id, "admission_id")
    require_existing_admission(patient_id, admission_id)

    temporary_path = save_upload_temporarily(report_file)
    try:
        report = report_submission_service.submit_report(
            patient_id=patient_id,
            admission_id=admission_id,
            filename=report_file.filename or "report",
            content_type=report_file.content_type or "application/octet-stream",
            storage_reference=temporary_path,
        )
        persisted_report = lab_report_repository.get_report(report["report_id"]) or report
        tasks = task_repository.get_patient_tasks(patient_id, admission_id)
        task = next(
            (
                item for item in reversed(tasks)
                if item.get("metadata", {}).get("report_id") == report["report_id"]
            ),
            None,
        )
        return LabUploadResponse(
            report_id=report["report_id"],
            status=persisted_report["processing_status"],
            workflow_started=True,
            task_id=task["task_id"] if task else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        path = Path(temporary_path)
        if path.exists():
            path.unlink()


# =========================================================
# NURSE TASKS
# =========================================================

@app.get(
    "/api/nurse/tasks",
    response_model=list[WorkflowTaskResponse],
    tags=["Nurse"],
    summary="List nurse workflow tasks",
)
def get_nurse_tasks(
    patient_id: str | None = Query(default=None),
    admission_id: str | None = Query(default=None),
    status: TaskStatus | None = Query(default=None),
) -> list[dict[str, Any]]:
    try:
        return task_repository.list_tasks(patient_id, admission_id, status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/api/nurse/tasks/{task_id}",
    response_model=WorkflowTaskResponse,
    tags=["Nurse"],
    summary="Get a nurse task and its current question",
)
def get_nurse_task(task_id: str) -> dict[str, Any]:
    task = task_repository.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    metadata = task.get("metadata", {})
    fields = metadata.get("question_fields", task.get("required_fields", []))
    index = int(metadata.get("question_index", 0))
    task["metadata"] = {
        **metadata,
        "missing_fields": list(fields[index:]),
        "current_question": (
            f"Please provide {fields[index]}." if index < len(fields) else None
        ),
        "question_index": index,
        "report_id": metadata.get("report_id"),
    }
    return task


@app.post(
    "/api/nurse/tasks/{task_id}/audio",
    response_model=NurseAudioResponse,
    tags=["Nurse"],
    summary="Submit the next nurse questionnaire answer by audio",
)
def submit_nurse_audio(
    task_id: str,
    audio_file: UploadFile = File(...),
) -> NurseAudioResponse:
    task = task_repository.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    temporary_path = save_upload_temporarily(audio_file)
    try:
        updated_task = nurse_submission_service.submit_nurse_audio(
            task_id=task_id,
            patient_id=task["patient_id"],
            admission_id=task["admission_id"],
            audio_path=temporary_path,
        )
        current = task_repository.get_task(task_id) or updated_task
        metadata = current.get("metadata", {})
        fields = metadata.get("question_fields", current.get("required_fields", []))
        index = int(metadata.get("question_index", 0))
        observation = repository.get_observation_by_task(
            current["patient_id"], current["admission_id"], task_id
        )
        completed = current["status"] == TaskStatus.COMPLETED.value
        return NurseAudioResponse(
            completed=completed,
            next_question=(f"Please provide {fields[index]}." if index < len(fields) else None),
            question_index=index,
            total_questions=len(fields),
            observation_id=observation.get("observation_id") if observation else None,
            assessment_started=observation is not None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        path = Path(temporary_path)
        if path.exists():
            path.unlink()


# =========================================================
# DOCTOR PATIENT HISTORY
# =========================================================

@app.get(
    "/api/doctor/patient/{patient_id}",
    response_model=DoctorPatientResponse,
    tags=["Doctor"],
    summary="Get the complete patient and admission workflow history",
)
def get_doctor_patient(
    patient_id: str,
    admission_id: str | None = Query(default=None),
) -> dict[str, Any]:
    patient_id = clean_identifier(patient_id, "patient_id")
    if admission_id is None:
        admissions = list(database["admissions"].find({"patient_id": patient_id}))
        if not admissions:
            raise HTTPException(status_code=404, detail="Patient has no admissions")
        if len(admissions) > 1:
            raise HTTPException(status_code=400, detail="admission_id is required for patients with multiple admissions")
        admission_id = admissions[0]["admission_id"]
    admission_id = clean_identifier(admission_id, "admission_id")
    try:
        history = repository.get_patient_history(patient_id, admission_id)
        reports = lab_report_repository.get_patient_reports(patient_id, admission_id)
        tasks = task_repository.get_patient_tasks(patient_id, admission_id)
        return {
            "patient": history["patient"],
            "admission": history["admission"],
            "observations": history["observations"],
            "assessments": history["assessments"],
            "trends": history["trends"],
            "prioritizations": repository.get_patient_prioritizations(patient_id, admission_id),
            "reports": reports,
            "tasks": tasks,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# =========================================================
# PATIENT HISTORY
# =========================================================

@app.get(
    "/api/patients/"
    "{patient_id}/admissions/"
    "{admission_id}/history",
    response_model=PatientHistoryResponse,
    tags=["Doctor"],
    summary="Get patient admission history",
)
def get_patient_history(
    patient_id: str,
    admission_id: str,
) -> dict[str, Any]:

    patient_id = clean_identifier(
        patient_id,
        "patient_id",
    )

    admission_id = clean_identifier(
        admission_id,
        "admission_id",
    )

    try:

        return repository.get_patient_history(
            patient_id,
            admission_id,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


# =========================================================
# MULTI-PATIENT CLINICAL DASHBOARD
# =========================================================

@app.get(
    "/api/dashboard/patients",
    response_model=DashboardResponse,
    tags=["Dashboard"],
    summary="Summarize active admissions for clinical monitoring",
)
def get_dashboard_patients():

    """
    Returns one dashboard summary for every active
    admission.

    Used by the Clinical Dashboard to compare and
    prioritize multiple patients.
    """

    try:

        admissions_collection = database[
            "admissions"
        ]

        active_admissions = list(
            admissions_collection.find(
                {
                    "status": "active",
                }
            )
        )

        dashboard_patients: list[
            dict[str, Any]
        ] = []

        for admission in active_admissions:

            patient_id = admission.get(
                "patient_id"
            )

            admission_id = admission.get(
                "admission_id"
            )

            if (
                not patient_id
                or not admission_id
            ):
                continue

            history = (
                repository
                .get_patient_history(
                    patient_id,
                    admission_id,
                )
            )

            observations = (
                history.get(
                    "observations",
                    [],
                )
                or []
            )

            assessments = (
                history.get(
                    "assessments",
                    [],
                )
                or []
            )

            trends = (
                history.get(
                    "trends",
                    [],
                )
                or []
            )

            reports = lab_report_repository.get_patient_reports(
                patient_id,
                admission_id,
            )
            tasks = task_repository.get_patient_tasks(
                patient_id,
                admission_id,
            )

            latest_priority = (
                history.get(
                    "latest_prioritization"
                )
            )

            latest_observation = (
                get_latest_item(
                    observations
                )
            )

            latest_assessment = (
                get_latest_item(
                    assessments
                )
            )

            latest_report = get_latest_item(reports)
            latest_trend = get_latest_item(trends)

            latest_sepsis_trend = (
                get_latest_disease_trend(
                    trends,
                    "Sepsis",
                )
            )

            latest_aki_trend = (
                get_latest_disease_trend(
                    trends,
                    "AKI",
                )
            )

            sepsis_probability = (
                probability_from_assessment(
                    latest_assessment,
                    "Sepsis",
                )
            )

            aki_probability = (
                probability_from_assessment(
                    latest_assessment,
                    "AKI",
                )
            )

            sepsis_risk_level = (
                risk_level_from_assessment(
                    latest_assessment,
                    "Sepsis",
                )
            )

            priority_level = (
                latest_priority.get(
                    "priority_level",
                    "PENDING",
                )
                if latest_priority
                else "PENDING"
            )

            highest_risk_disease = (
                latest_priority.get(
                    "highest_risk_disease"
                )
                if latest_priority
                else None
            )

            priority_reason = (
                latest_priority.get(
                    "reason"
                )
                if latest_priority
                else None
            )

            worsening_diseases = (
                latest_priority.get(
                    "worsening_diseases",
                    [],
                )
                if latest_priority
                else []
            )

            latest_parameters = {}

            if latest_observation:

                latest_parameters = (
                    latest_observation.get(
                        "clinical_parameters",
                        {},
                    )
                    or {}
                )

            dashboard_patients.append(
                {
                    "patient_id":
                        patient_id,

                    "admission_id":
                        admission_id,

                    "observation_count":
                        len(observations),

                    "assessment_count":
                        len(assessments),

                    "sepsis": {
                        "status":
                            assessment_status(
                                latest_assessment,
                                "Sepsis",
                            ),

                        "probability":
                            sepsis_probability,

                        "risk_level":
                            sepsis_risk_level,

                        "trend":
                            (
                                latest_sepsis_trend
                                or {}
                            ).get(
                                "trend",
                                "INSUFFICIENT_DATA",
                            ),

                        "first_probability":
                            (
                                latest_sepsis_trend
                                or {}
                            ).get(
                                "first_probability"
                            ),

                        "latest_probability":
                            (
                                latest_sepsis_trend
                                or {}
                            ).get(
                                "latest_probability"
                            ),

                        "probability_change":
                            (
                                latest_sepsis_trend
                                or {}
                            ).get(
                                "probability_change"
                            ),
                    },

                    "aki": {
                        "status":
                            assessment_status(
                                latest_assessment,
                                "AKI",
                            ),

                        "probability":
                            aki_probability,

                        "trend":
                            (
                                latest_aki_trend
                                or {}
                            ).get(
                                "trend",
                                "INSUFFICIENT_DATA",
                            ),
                    },

                    "priority": {
                        "level":
                            priority_level,

                        "highest_risk_disease":
                            highest_risk_disease,

                        "reason":
                            priority_reason,

                        "worsening_diseases":
                            worsening_diseases,
                    },

                    "latest_parameters":
                        latest_parameters,

                    "latest_report":
                        latest_report,

                    "pending_task_count":
                        sum(
                            task.get("status") == TaskStatus.PENDING.value
                            for task in tasks
                        ),

                    "latest_assessment":
                        latest_assessment,

                    "latest_trend":
                        latest_trend,

                    "latest_prioritization":
                        latest_priority,
                }
            )

        dashboard_patients.sort(
            key=lambda patient: (
                priority_rank(
                    patient[
                        "priority"
                    ][
                        "level"
                    ]
                ),
                -(
                    patient[
                        "sepsis"
                    ][
                        "probability"
                    ]
                    or 0.0
                ),
            )
        )

        return {
            "status":
                "success",

            "patient_count":
                len(
                    dashboard_patients
                ),

            "patients":
                dashboard_patients,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# =========================================================
# SHUTDOWN
# =========================================================

@app.on_event("shutdown")
def shutdown_event():
    if database is not None:
        database.close()