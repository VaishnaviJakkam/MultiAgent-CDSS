from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.database.mongodb import MongoDatabase
from src.database.repositories import (
    LabReportRepository,
    PatientRepository,
    WorkflowEventRepository,
    WorkflowTaskRepository,
)
from src.database.workflow_tasks import TaskStatus
from src.events import EventBus, EventType

from src.input.nurse_input_workflow import NurseInputSubmissionService, NurseInputReceivedEventHandler
from src.input.nurse_audio_agent import NurseAudioAgent
from src.input.report_event_workflow import (
    GenerateResultRequestService,
    GenerateResultRequestedEventHandler,
    LabReportSubmissionService,
    ReportProcessingEventHandler,
)
from src.agents.assessment_event_workflow import ObservationReadyAssessmentHandler
from src.agents.trend_event_workflow import AssessmentCompletedTrendHandler
from src.agents.risk_prioritization_event_workflow import TrendCompletedRiskPrioritizationHandler
from src.agents.nurse_agent import NurseAgent
from src.api.workflow_routes import WorkflowAPIContext, create_workflow_router


# =========================================================
# APP
# =========================================================

@asynccontextmanager
async def lifespan(_: FastAPI):
    nurse_input_service.set_audio_agent(NurseAudioAgent())
    yield

app = FastAPI(
    title="Multi-Agent Clinical Deterioration API",
    version="1.0.0",
    lifespan=lifespan,
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
# DATABASE + AGENTS
# =========================================================

database = MongoDatabase()
repository = PatientRepository(database)
report_repository = LabReportRepository(database)
task_repository = WorkflowTaskRepository(database)
event_repository = WorkflowEventRepository(database)
event_bus = EventBus(event_repository)
nurse_agent = NurseAgent()

ReportProcessingEventHandler(report_repository, event_bus)
GenerateResultRequestedEventHandler(
    report_repository,
    task_repository,
    event_bus,
    patient_repository=repository,
    nurse_agent=nurse_agent,
)
NurseInputReceivedEventHandler(task_repository, report_repository, repository, event_bus)
ObservationReadyAssessmentHandler(repository, event_bus)
AssessmentCompletedTrendHandler(repository, event_bus)
TrendCompletedRiskPrioritizationHandler(repository, event_bus)

report_submission_service = LabReportSubmissionService(report_repository, event_bus)
generate_result_service = GenerateResultRequestService(report_repository, event_bus)
nurse_input_service = NurseInputSubmissionService(
    task_repository,
    report_repository,
    event_bus,
    nurse_agent=nurse_agent,
)

# =========================================================
# REQUEST MODELS
# =========================================================

class ConfirmationRequest(BaseModel):
    confirmed_parameters: dict[str, float]


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

@app.get("/api/health")
def health():

    try:
        connected = database.ping()

    except Exception:
        connected = False

    return {
        "status": "ok",
        "mongodb": connected,
    }


# =========================================================
# REPORT UPLOAD
# =========================================================

@app.post("/api/legacy/input/report", include_in_schema=False)
def upload_report(
    patient_id: str = Form(...),
    admission_id: str = Form(...),
    report: UploadFile = File(...),
):
    raise HTTPException(status_code=410, detail="Use POST /api/reports")


# =========================================================
# AUDIO UPLOAD
# =========================================================

@app.post(
    "/api/legacy/input/{input_id}/audio",
    include_in_schema=False,
)
def upload_nurse_audio(
    input_id: str,
    audio: UploadFile = File(...),
):
    raise HTTPException(status_code=410, detail="Use POST /api/nurse/tasks/{task_id}/audio")


# =========================================================
# CONFIRM NURSE VALUES
# =========================================================

@app.post(
    "/api/legacy/input/{input_id}/confirm",
    include_in_schema=False,
)
def confirm_nurse_values(
    input_id: str,
    request: ConfirmationRequest,
):
    raise HTTPException(status_code=410, detail="Use the persistent nurse task workflow")


# =========================================================
# PATIENT HISTORY
# =========================================================

@app.get(
    "/api/patients/"
    "{patient_id}/admissions/"
    "{admission_id}/history"
)
def get_patient_history(
    patient_id: str,
    admission_id: str,
):

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
    "/api/dashboard/patients"
)
def get_dashboard_patients():

    """
    Returns one dashboard summary for every active
    admission.

    Used by the doctor workspace to compare and
    prioritize patients.
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

app.include_router(
    create_workflow_router(
        WorkflowAPIContext(
            repository=repository,
            report_repository=report_repository,
            task_repository=task_repository,
            report_submission_service=report_submission_service,
            generate_result_service=generate_result_service,
            nurse_input_service=nurse_input_service,
        )
    )
)

@app.on_event("shutdown")
def shutdown_event():

    database.close()