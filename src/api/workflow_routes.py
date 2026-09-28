from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from src.database.repositories import LabReportRepository, PatientRepository, WorkflowTaskRepository
from src.database.workflow_tasks import TaskStatus
from src.input.input_orchestrator import InputOrchestrator
from src.input.document_text_extractor import DocumentTextExtractor
from src.input.deepseek_report_extractor import DeepSeekReportExtractor
from src.input.nurse_input_workflow import NurseInputSubmissionService
from src.input.report_event_workflow import GenerateResultRequestService, LabReportSubmissionService


@dataclass(frozen=True)
class WorkflowAPIContext:
    repository: PatientRepository
    report_repository: LabReportRepository
    task_repository: WorkflowTaskRepository
    report_submission_service: LabReportSubmissionService
    generate_result_service: GenerateResultRequestService
    nurse_input_service: NurseInputSubmissionService


def _save_upload_temporarily(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(upload.file, temp_file)
        return temp_file.name


def _clean_identifier(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty")
    return cleaned


def _serialize_task(task: dict[str, Any]) -> dict[str, Any]:
    metadata = task.get("metadata") or {}
    fields = list(metadata.get("question_fields", task.get("required_fields", [])))
    question_index = int(metadata.get("question_index", 0))
    current_field = fields[question_index] if question_index < len(fields) else None
    display_name = InputOrchestrator.DISPLAY_NAMES.get(current_field, current_field)
    agent_decision = metadata.get("nurse_agent_decision", {})
    question = agent_decision.get("question") if isinstance(agent_decision, dict) else None
    return {
        "task_id": task["task_id"],
        "patient_id": task["patient_id"],
        "admission_id": task["admission_id"],
        "status": task["status"],
        "questionnaire_status": task["status"],
        "current_question": (
            {"field": current_field, "text": question or f"Please provide {display_name}."}
            if current_field
            else None
        ),
        "question_index": question_index,
        "total_questions": len(fields),
        "collected_fields": sorted((metadata.get("answers") or {}).keys()),
        "required_fields": fields,
        "next_action": "ANSWER_CURRENT_QUESTION" if current_field else "COMPLETED",
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }


def create_workflow_router(context: WorkflowAPIContext) -> APIRouter:
    router = APIRouter(prefix="/api")

    def require_admission(patient_id: str, admission_id: str) -> None:
        if context.repository.get_admission(patient_id, admission_id) is None:
            raise HTTPException(status_code=404, detail="Admission not found")

    @router.get("/patients/{patient_id}/admissions")
    def list_patient_admissions(patient_id: str) -> dict[str, Any]:
        patient_id = _clean_identifier(patient_id, "patient_id")
        if context.repository.get_patient(patient_id) is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        records = context.repository.admissions.find({"patient_id": patient_id})
        admissions = [
            {
                "patient_id": item["patient_id"],
                "admission_id": item["admission_id"],
                "admission_time": item.get("admission_time"),
                "status": item.get("status"),
            }
            for item in records
        ]
        return {"patient_id": patient_id, "admissions": admissions}

    @router.post("/patients")
    def create_patient_and_admission(payload: dict[str, Any]) -> dict[str, Any]:
        patient_id = _clean_identifier(str(payload.get("patient_id", "")), "patient_id")
        admission_id = _clean_identifier(str(payload.get("admission_id", "")), "admission_id")
        demographics = payload.get("demographics") or {}
        admission_time_value = payload.get("admission_time")
        try:
            admission_time = (
                datetime.fromisoformat(admission_time_value.replace("Z", "+00:00"))
                if isinstance(admission_time_value, str) and admission_time_value
                else datetime.now(timezone.utc)
            )
            if admission_time.tzinfo is None:
                admission_time = admission_time.replace(tzinfo=timezone.utc)
            if context.repository.get_patient(patient_id) is not None:
                raise HTTPException(status_code=409, detail="Patient already exists")
            patient = context.repository.create_patient(patient_id, demographics=demographics)
            admission = context.repository.create_admission(patient_id, admission_id, admission_time)
            return {"patient": patient, "admission": admission}
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def scoped_task(task_id: str, patient_id: str, admission_id: str) -> dict[str, Any]:
        task = context.task_repository.get_task(task_id)
        if task is None or task["patient_id"] != patient_id or task["admission_id"] != admission_id:
            raise HTTPException(status_code=404, detail="Nurse task not found")
        return task

    @router.post("/reports")
    def submit_lab_report(
        patient_id: str = Form(...),
        admission_id: str = Form(...),
        report: UploadFile = File(...),
    ) -> dict[str, Any]:
        patient_id = _clean_identifier(patient_id, "patient_id")
        admission_id = _clean_identifier(admission_id, "admission_id")
        require_admission(patient_id, admission_id)
        temporary_path = _save_upload_temporarily(report)
        try:
            DocumentTextExtractor().validate(
                temporary_path,
                filename=report.filename or "report",
                content_type=report.content_type,
            )
            document_text = DocumentTextExtractor().extract(
                temporary_path,
                filename=report.filename or "report",
                content_type=report.content_type,
            )
            DocumentTextExtractor.validate_report_context(
                document_text,
                patient_id=patient_id,
                admission_id=admission_id,
            )
            extracted_data = DeepSeekReportExtractor().extract(document_text)
            persisted = context.report_submission_service.submit_report(
                patient_id=patient_id,
                admission_id=admission_id,
                filename=report.filename or "report",
                content_type=report.content_type or "application/octet-stream",
                storage_reference=temporary_path,
                extracted_data=extracted_data,
                raw_text=document_text,
            )
            current = context.report_repository.get_report(persisted["report_id"]) or persisted
            return {
                "report_id": current["report_id"],
                "patient_id": current["patient_id"],
                "admission_id": current["admission_id"],
                "processing_status": current["processing_status"],
                "extracted_data": current.get("extracted_data", {}),
                "task_status": "WAITING_FOR_EXPLICIT_GENERATE_RESULT",
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if Path(temporary_path).exists():
                Path(temporary_path).unlink()

    @router.get("/reports/{report_id}")
    def get_report_status(report_id: str) -> dict[str, Any]:
        report = context.report_repository.get_report(report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        return {
            "report_id": report["report_id"],
            "patient_id": report["patient_id"],
            "admission_id": report["admission_id"],
            "processing_status": report["processing_status"],
            "extracted_data": report.get("extracted_data", {}),
            "error": (report.get("metadata") or {}).get("processing_error"),
        }

    @router.post("/reports/{report_id}/generate-result")
    def generate_result(
        report_id: str,
        patient_id: str = Query(...),
        admission_id: str = Query(...),
    ) -> dict[str, Any]:
        patient_id = _clean_identifier(patient_id, "patient_id")
        admission_id = _clean_identifier(admission_id, "admission_id")
        report = context.report_repository.get_report(report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        if report["patient_id"] != patient_id or report["admission_id"] != admission_id:
            raise HTTPException(status_code=404, detail="Report not found")
        if report["processing_status"] != "PROCESSED":
            raise HTTPException(status_code=409, detail="Report has not been processed")
        try:
            context.generate_result_service.request_result(patient_id, admission_id, report_id)
            tasks = context.task_repository.get_patient_tasks(patient_id, admission_id)
            task = next((item for item in reversed(tasks) if item.get("metadata", {}).get("report_id") == report_id), None)
            observations = context.repository.get_patient_observations(patient_id, admission_id)
            observation = next((item for item in reversed(observations) if item.get("report_id") == report_id), None)
            return {
                "report_id": report_id,
                "patient_id": patient_id,
                "admission_id": admission_id,
                "status": "NURSE_INPUT_REQUIRED" if task and task["status"] != TaskStatus.COMPLETED.value else "OBSERVATION_READY",
                "task": _serialize_task(task) if task else None,
                "observation_id": observation.get("observation_id") if observation else None,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get("/nurse/tasks")
    def list_nurse_tasks(patient_id: str | None = None, admission_id: str | None = None) -> dict[str, Any]:
        if (patient_id is None) != (admission_id is None):
            raise HTTPException(status_code=400, detail="patient_id and admission_id must be provided together")
        tasks = (
            context.task_repository.get_patient_tasks(patient_id, admission_id)
            if patient_id is not None and admission_id is not None
            else context.task_repository.get_pending_tasks()
        )
        return {"tasks": [_serialize_task(task) for task in tasks]}

    @router.post("/nurse/tasks/{task_id}/start")
    def start_nurse_task(task_id: str, patient_id: str, admission_id: str) -> dict[str, Any]:
        task = scoped_task(task_id, patient_id, admission_id)
        try:
            return _serialize_task(context.task_repository.start_task(task["task_id"]))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/nurse/tasks/{task_id}")
    def get_nurse_task(task_id: str, patient_id: str, admission_id: str) -> dict[str, Any]:
        return _serialize_task(scoped_task(task_id, patient_id, admission_id))

    @router.get("/nurse/tasks/{task_id}/question")
    def get_current_nurse_question(task_id: str, patient_id: str, admission_id: str) -> dict[str, Any]:
        return _serialize_task(scoped_task(task_id, patient_id, admission_id))

    @router.post("/nurse/tasks/{task_id}/answer")
    def submit_structured_nurse_answer(
        task_id: str,
        patient_id: str,
        admission_id: str,
        provided_values: dict[str, Any],
    ) -> dict[str, Any]:
        scoped_task(task_id, patient_id, admission_id)
        try:
            updated = context.nurse_input_service.submit_nurse_input(
                task_id=task_id,
                patient_id=patient_id,
                admission_id=admission_id,
                provided_values=provided_values,
                source="nurse_structured_input",
            )
            return _serialize_task(updated)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"status": "retry", "message": str(exc)}) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/nurse/tasks/{task_id}/audio")
    def submit_nurse_audio(
        task_id: str,
        patient_id: str = Form(...),
        admission_id: str = Form(...),
        audio: UploadFile = File(...),
    ) -> dict[str, Any]:
        task = scoped_task(task_id, patient_id, admission_id)
        if task["status"] in {TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value}:
            raise HTTPException(status_code=409, detail=f"Task is already {task['status']}")
        temporary_path = _save_upload_temporarily(audio)
        try:
            updated = context.nurse_input_service.submit_nurse_audio(task_id, patient_id, admission_id, temporary_path)
            if (updated.get("metadata") or {}).get("processing_error"):
                raise HTTPException(
                    status_code=500,
                    detail="Nurse answer was persisted but observation processing failed",
                )
            return _serialize_task(updated)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"status": "retry", "message": str(exc)}) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if Path(temporary_path).exists():
                Path(temporary_path).unlink()

    @router.get("/patients/{patient_id}/admissions/{admission_id}")
    def get_patient_admission(patient_id: str, admission_id: str) -> dict[str, Any]:
        patient_id = _clean_identifier(patient_id, "patient_id")
        admission_id = _clean_identifier(admission_id, "admission_id")
        require_admission(patient_id, admission_id)
        return context.repository.get_patient_history(patient_id, admission_id)

    @router.get("/patients/{patient_id}/admissions/{admission_id}/assessments")
    def get_patient_assessments(patient_id: str, admission_id: str) -> list[dict[str, Any]]:
        require_admission(patient_id, admission_id)
        return context.repository.get_patient_assessments(patient_id, admission_id)

    @router.get("/patients/{patient_id}/admissions/{admission_id}/trends")
    def get_patient_trends(patient_id: str, admission_id: str) -> list[dict[str, Any]]:
        require_admission(patient_id, admission_id)
        return context.repository.get_patient_trends(patient_id, admission_id)

    @router.get("/patients/{patient_id}/admissions/{admission_id}/prioritization")
    def get_patient_prioritization(patient_id: str, admission_id: str) -> dict[str, Any] | None:
        require_admission(patient_id, admission_id)
        return context.repository.get_latest_prioritization(patient_id, admission_id)

    return router
