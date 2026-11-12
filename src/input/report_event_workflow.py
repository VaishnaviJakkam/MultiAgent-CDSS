from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from src.database.lab_reports import ReportProcessingStatus
from src.database.repositories import LabReportRepository, PatientRepository, WorkflowTaskRepository
from src.events import EventBus, EventType, WorkflowEvent
from src.input.data_processing_agent import DataProcessingAgent
from src.input.deepseek_report_extractor import DeepSeekReportExtractor
from src.input.document_text_extractor import DocumentTextExtractor
from src.input.input_orchestrator import InputOrchestrator
from src.agents.nurse_agent import NurseAgent


class LabReportSubmissionService:
    """Persist a lab report before publishing its upload event."""

    def __init__(self, repository: LabReportRepository, event_bus: EventBus) -> None:
        self.repository = repository
        self.event_bus = event_bus

    def submit_report(
        self,
        patient_id: str,
        admission_id: str,
        filename: str,
        content_type: str,
        storage_reference: str,
        report_date: datetime | None = None,
        source: Any = "LAB_UPLOAD",
        metadata: dict[str, Any] | None = None,
        report_id: str | None = None,
        uploaded_at: datetime | None = None,
        extracted_data: dict[str, Any] | None = None,
        raw_text: str = "",
    ) -> dict[str, Any]:
        report = self.repository.create_report(
            patient_id=patient_id,
            admission_id=admission_id,
            report_date=report_date,
            filename=filename,
            content_type=content_type,
            storage_reference=storage_reference,
            source=source,
            metadata=metadata,
            report_id=report_id,
            uploaded_at=uploaded_at,
            processing_status=(ReportProcessingStatus.PROCESSED if extracted_data is not None else ReportProcessingStatus.UPLOADED),
            extracted_data=extracted_data,
            raw_text=raw_text,
        )
        event = WorkflowEvent(
            event_type=EventType.REPORT_UPLOADED,
            patient_id=report["patient_id"],
            admission_id=report["admission_id"],
            related_report_id=report["report_id"],
            payload={
                "report_id": report["report_id"],
                "storage_reference": report["storage_reference"],
                "filename": report["filename"],
            },
        )
        self.event_bus.publish(event)
        if extracted_data is not None:
            self.event_bus.publish(
                WorkflowEvent(
                    event_type=EventType.REPORT_PROCESSED,
                    patient_id=report["patient_id"],
                    admission_id=report["admission_id"],
                    related_report_id=report["report_id"],
                    payload={
                        "report_id": report["report_id"],
                        "processing_status": ReportProcessingStatus.PROCESSED.value,
                        "extracted_fields": sorted(extracted_data),
                    },
                )
            )
        return report


class GenerateResultRequestService:
    """Publish the lab user's explicit request to generate a clinical result."""

    def __init__(self, repository: LabReportRepository, event_bus: EventBus) -> None:
        self.repository = repository
        self.event_bus = event_bus

    def request_result(self, patient_id: str, admission_id: str, report_id: str) -> dict[str, Any] | None:
        report = self.repository.get_report(report_id)
        if report is None:
            return None
        if report["patient_id"] != patient_id or report["admission_id"] != admission_id:
            return None
        if report["processing_status"] != ReportProcessingStatus.PROCESSED.value:
            return None
        self.event_bus.publish(
            WorkflowEvent(
                event_type=EventType.GENERATE_RESULT_REQUESTED,
                patient_id=patient_id,
                admission_id=admission_id,
                related_report_id=report_id,
                payload={"report_id": report_id},
            )
        )
        return report


class ReportProcessingEventHandler:
    """Process persisted lab reports when REPORT_UPLOADED is delivered."""

    def __init__(
        self,
        repository: LabReportRepository,
        event_bus: EventBus,
        report_agent: Any | None = None,
    ) -> None:
        self.repository = repository
        self.event_bus = event_bus
        self.report_agent = report_agent
        self.text_extractor = DocumentTextExtractor()
        self.event_bus.subscribe(EventType.REPORT_UPLOADED, self.handle)

    def handle(self, event: WorkflowEvent) -> dict[str, Any] | None:
        """Process once and convert extraction failures into FAILED state."""
        report_id = event.related_report_id or event.payload.get("report_id")
        if not isinstance(report_id, str) or not report_id.strip():
            return None

        report = self.repository.get_report(report_id)
        if report is None:
            return None
        if report["patient_id"] != event.patient_id or report["admission_id"] != event.admission_id:
            return self._fail(report, "Event scope does not match the persisted report.")
        if report["processing_status"] == ReportProcessingStatus.PROCESSED.value:
            return report

        try:
            processing = self.repository.update_report(
                report_id,
                processing_status=ReportProcessingStatus.PROCESSING,
            )
            storage_reference = processing.get("storage_reference")
            if not storage_reference:
                raise ValueError("Lab report has no storage_reference")

            if self.report_agent is None:
                raw_text = self.text_extractor.extract(
                    storage_reference,
                    filename=processing["filename"],
                    content_type=processing["content_type"],
                )
                extracted_data = DeepSeekReportExtractor().extract(raw_text)
                result = {"extracted_data": extracted_data, "raw_text": raw_text}
            else:
                result = self.report_agent.analyze(storage_reference)
            extracted_data = result.get("extracted_data") or result.get("lab_results") or {}
            if not isinstance(extracted_data, dict):
                raise ValueError("Report extraction did not return structured lab results")

            processed = self.repository.update_report(
                report_id,
                processing_status=ReportProcessingStatus.PROCESSED,
                extracted_data=extracted_data,
                raw_text=str(result.get("raw_text", "")),
            )
            self.event_bus.publish(
                WorkflowEvent(
                    event_type=EventType.REPORT_PROCESSED,
                    patient_id=processed["patient_id"],
                    admission_id=processed["admission_id"],
                    related_report_id=processed["report_id"],
                    payload={
                        "report_id": processed["report_id"],
                        "processing_status": processed["processing_status"],
                        "extracted_fields": sorted(processed["extracted_data"]),
                    },
                )
            )
            return processed
        except Exception as exc:
            return self._fail(report, str(exc))

    def _fail(self, report: dict[str, Any], message: str) -> dict[str, Any]:
        metadata = deepcopy(report.get("metadata") or {})
        metadata["processing_error"] = message
        return self.repository.update_report(
            report["report_id"],
            processing_status=ReportProcessingStatus.FAILED,
            metadata=metadata,
        )


class GenerateResultRequestedEventHandler:
    """Create the next input state after an explicit result-generation request."""

    def __init__(
        self,
        report_repository: LabReportRepository,
        task_repository: WorkflowTaskRepository,
        event_bus: EventBus,
        input_orchestrator: InputOrchestrator | None = None,
        patient_repository: PatientRepository | None = None,
        data_processor: DataProcessingAgent | None = None,
        nurse_agent: NurseAgent | None = None,
    ) -> None:
        self.report_repository = report_repository
        self.task_repository = task_repository
        self.patient_repository = patient_repository or PatientRepository(report_repository.database)
        self.event_bus = event_bus
        self.input_orchestrator = input_orchestrator or InputOrchestrator()
        self.data_processor = data_processor or DataProcessingAgent()
        self.nurse_agent = nurse_agent or NurseAgent()
        self.event_bus.subscribe(EventType.GENERATE_RESULT_REQUESTED, self.handle)

    def handle(self, event: WorkflowEvent) -> dict[str, Any] | None:
        """Route a processed report to readiness or a persistent nurse task."""
        report_id = event.related_report_id or event.payload.get("report_id")
        if not isinstance(report_id, str) or not report_id.strip():
            return None

        report = self.report_repository.get_report(report_id)
        if report is None:
            return None
        if report["patient_id"] != event.patient_id or report["admission_id"] != event.admission_id:
            return None

        report_result = self._report_result(report)
        nurse_request = self.input_orchestrator.create_nurse_request(report_result)
        provided_fields = sorted(report_result["sepsis_parameters"])

        decision = self.nurse_agent.decide(
            patient_id=report["patient_id"],
            admission_id=report["admission_id"],
            lab_values=report_result["sepsis_parameters"],
            nurse_values={},
            current_task_state={"status": "PENDING", "report_id": report["report_id"]},
        )

        if decision.action == "COMPLETE":
            existing_observation = self.patient_repository.get_observation_by_report(
                report["patient_id"],
                report["admission_id"],
                report["report_id"],
            )
            if existing_observation is not None:
                self._publish_ready(report, existing_observation)
                return existing_observation
            observation = self.data_processor.process(
                patient_id=report["patient_id"],
                admission_id=report["admission_id"],
                report_result=report_result,
                observation_time=datetime.now().astimezone(),
            )
            if not observation["ready_for_sepsis_assessment"]:
                raise ValueError(f"Missing clinical parameters: {observation['missing_sepsis_parameters']}")
            stored_observation = self.patient_repository.add_observation(
                patient_id=report["patient_id"],
                admission_id=report["admission_id"],
                observation_time=observation["observation_time"],
                clinical_parameters=observation["clinical_parameters"],
                source="lab_report",
                report_id=report["report_id"],
                parameter_sources=observation["parameter_sources"],
            )
            self._publish_ready(report, stored_observation)
            return stored_observation

        existing_task = self._find_task(report)
        if existing_task is not None:
            self._publish_nurse_required(report, existing_task, nurse_request, allow_duplicate=False)
            return existing_task

        task = self.task_repository.create_task(
            task_type="NURSE_INPUT",
            patient_id=report["patient_id"],
            admission_id=report["admission_id"],
            required_fields=nurse_request["requested_nurse_fields"],
            provided_fields=provided_fields,
            metadata={
                "report_id": report["report_id"],
                "missing_model_features": nurse_request["missing_model_features"],
                "request_message": nurse_request["message"],
                "question_fields": [decision.parameter],
                "question_index": 0,
                "answers": {},
                "nurse_agent_decision": decision.as_dict(),
            },
        )
        self._publish_nurse_required(report, task, nurse_request, allow_duplicate=False)
        return task

    @staticmethod
    def _report_result(report: dict[str, Any]) -> dict[str, Any]:
        extracted_data = report.get("extracted_data") or {}
        sepsis_parameters: dict[str, Any] = {}
        for parameter, value in extracted_data.items():
            if isinstance(value, dict) and "value" in value:
                sepsis_parameters[parameter] = value["value"]
            else:
                sepsis_parameters[parameter] = value
        return {"sepsis_parameters": sepsis_parameters}

    def _find_task(self, report: dict[str, Any]) -> dict[str, Any] | None:
        from src.database.workflow_tasks import TaskStatus, TaskType

        tasks = self.task_repository.get_patient_tasks(
            report["patient_id"],
            report["admission_id"],
        )
        for task in tasks:
            if (
                task["task_type"] == TaskType.NURSE_INPUT.value
                and task["status"] in {
                    TaskStatus.PENDING.value,
                    TaskStatus.IN_PROGRESS.value,
                    TaskStatus.COMPLETED.value,
                }
                and task.get("metadata", {}).get("report_id") == report["report_id"]
            ):
                return task
        return None

    def _event_exists(self, event_type: EventType, report: dict[str, Any], task_id: str | None = None) -> bool:
        event_repository = self.event_bus.event_repository
        if event_repository is None:
            return False
        events = event_repository.list_events(
            patient_id=report["patient_id"],
            admission_id=report["admission_id"],
            event_type=event_type,
        )
        for existing in events:
            payload = existing.get("payload", {})
            if payload.get("report_id") != report["report_id"]:
                continue
            if task_id is None or payload.get("task_id") == task_id:
                return True
        return False

    def _publish_nurse_required(
        self,
        report: dict[str, Any],
        task: dict[str, Any],
        nurse_request: dict[str, Any],
        allow_duplicate: bool,
    ) -> None:
        if not allow_duplicate and self._event_exists(
            EventType.NURSE_INPUT_REQUIRED,
            report,
            task["task_id"],
        ):
            return
        self.event_bus.publish(
            WorkflowEvent(
                event_type=EventType.NURSE_INPUT_REQUIRED,
                patient_id=report["patient_id"],
                admission_id=report["admission_id"],
                related_report_id=report["report_id"],
                related_task_id=task["task_id"],
                payload={
                    "report_id": report["report_id"],
                    "task_id": task["task_id"],
                    "missing_fields": task["required_fields"],
                    "missing_model_features": nurse_request["missing_model_features"],
                    **self._question_payload(task, nurse_request),
                },
            )
        )

    @staticmethod
    def _question_payload(task: dict[str, Any], nurse_request: dict[str, Any]) -> dict[str, Any]:
        fields = task.get("required_fields", [])
        index = int(task.get("metadata", {}).get("question_index", 0))
        field = fields[index] if index < len(fields) else None
        display_name = InputOrchestrator.DISPLAY_NAMES.get(field, field)
        return {
            "question_index": index,
            "total_questions": len(fields),
            "question_field": field,
            "question": f"Please provide {display_name}." if display_name else nurse_request["message"],
        }

    def _publish_ready(self, report: dict[str, Any], observation: dict[str, Any]) -> None:
        if not self._event_exists(EventType.OBSERVATION_READY, report):
            self.event_bus.publish(
                WorkflowEvent(
                    event_type=EventType.OBSERVATION_READY,
                    patient_id=report["patient_id"],
                    admission_id=report["admission_id"],
                    related_report_id=report["report_id"],
                    payload={
                        "report_id": report["report_id"],
                        "observation_id": observation["observation_id"],
                        "processing_status": report["processing_status"],
                        "available_fields": sorted(report.get("extracted_data", {})),
                    },
                )
            )


ReportProcessedEventHandler = GenerateResultRequestedEventHandler
GenerateResultEventHandler = GenerateResultRequestedEventHandler
