from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from src.database.repositories import LabReportRepository, PatientRepository, WorkflowTaskRepository
from src.database.workflow_tasks import TaskStatus, TaskType
from src.events import EventBus, EventType, WorkflowEvent
from src.input.data_processing_agent import DataProcessingAgent
from src.input.input_orchestrator import InputOrchestrator


class NurseInputSubmissionService:
    """Validate and persist structured nurse input before publishing an event."""

    def __init__(
        self,
        task_repository: WorkflowTaskRepository,
        report_repository: LabReportRepository,
        event_bus: EventBus,
        input_orchestrator: InputOrchestrator | None = None,
    ) -> None:
        self.task_repository = task_repository
        self.report_repository = report_repository
        self.event_bus = event_bus
        self.input_orchestrator = input_orchestrator or InputOrchestrator()

    def submit_nurse_input(
        self,
        task_id: str,
        patient_id: str,
        admission_id: str,
        provided_values: dict[str, Any],
        source: str = "nurse_confirmed",
    ) -> dict[str, Any]:
        task = self.task_repository.get_task(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        if task["task_type"] != TaskType.NURSE_INPUT.value:
            raise ValueError("Task is not a NURSE_INPUT task")
        if task["patient_id"] != patient_id or task["admission_id"] != admission_id:
            raise ValueError("patient_id and admission_id do not match the task")
        if task["status"] in {TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value}:
            raise ValueError(f"Task is already {task['status']}")
        if not isinstance(provided_values, dict):
            raise ValueError("provided_values must be a dictionary")

        report_id = task.get("metadata", {}).get("report_id")
        if not isinstance(report_id, str) or not report_id.strip():
            raise ValueError("NURSE_INPUT task does not reference a report")
        report = self.report_repository.get_report(report_id)
        if report is None:
            raise ValueError(f"Report not found: {report_id}")
        if report["patient_id"] != patient_id or report["admission_id"] != admission_id:
            raise ValueError("Report scope does not match the task")

        normalized_values = self._normalize_values(provided_values)
        question_fields = list(task.get("metadata", {}).get("question_fields", task["required_fields"]))
        answers = deepcopy(task.get("metadata", {}).get("answers", {}))
        requested_fields = question_fields if question_fields else task["required_fields"]
        validation = self.input_orchestrator.validate_nurse_response(
            nurse_request={"requested_nurse_fields": requested_fields},
            nurse_result={"extracted_parameters": normalized_values},
        )
        question_index = int(task.get("metadata", {}).get("question_index", 0))
        current_field = requested_fields[question_index] if question_index < len(requested_fields) else None
        current_validation = self.input_orchestrator.validate_nurse_response(
            nurse_request={"requested_nurse_fields": [current_field] if current_field else []},
            nurse_result={"extracted_parameters": normalized_values},
        )
        if not validation["all_requested_values_received"] and not current_validation["all_requested_values_received"]:
            raise ValueError(f"Missing requested nurse fields: {validation['still_missing']}")

        answers.update(normalized_values)
        next_index = question_index
        while next_index < len(requested_fields):
            check = self.input_orchestrator.validate_nurse_response(
                nurse_request={"requested_nurse_fields": [requested_fields[next_index]]},
                nurse_result={"extracted_parameters": answers},
            )
            if not check["all_requested_values_received"]:
                break
            next_index += 1

        metadata = deepcopy(task.get("metadata") or {})
        metadata["nurse_input"] = deepcopy(answers)
        metadata["answers"] = deepcopy(answers)
        metadata["question_index"] = next_index
        metadata["nurse_input_source"] = source
        updated_task = self.task_repository.update_task(
            task_id,
            status=TaskStatus.IN_PROGRESS,
            provided_fields=sorted(set(task.get("provided_fields", [])) | set(answers)),
            metadata=metadata,
        )
        self.event_bus.publish(
            WorkflowEvent(
                event_type=EventType.NURSE_INPUT_RECEIVED,
                patient_id=patient_id,
                admission_id=admission_id,
                related_report_id=report_id,
                related_task_id=task_id,
                payload={
                    "task_id": task_id,
                    "report_id": report_id,
                    "provided_fields": sorted(normalized_values),
                    "question_index": next_index,
                    "complete": next_index >= len(requested_fields),
                },
            )
        )
        return updated_task

    def submit_nurse_audio(
        self,
        task_id: str,
        patient_id: str,
        admission_id: str,
        audio_path: str,
        audio_agent: Any | None = None,
    ) -> dict[str, Any]:
        if audio_agent is None:
            from src.input.nurse_audio_agent import NurseAudioAgent

            audio_agent = NurseAudioAgent()
        result = audio_agent.analyze(audio_path)
        extracted = result.get("extracted_parameters")
        if not isinstance(extracted, dict):
            raise ValueError("Nurse audio did not produce structured clinical values")
        return self.submit_nurse_input(
            task_id=task_id,
            patient_id=patient_id,
            admission_id=admission_id,
            provided_values=extracted,
            source="nurse_audio",
        )

    def submit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.submit_nurse_input(*args, **kwargs)

    @staticmethod
    def _normalize_values(provided_values: dict[str, Any]) -> dict[str, Any]:
        values = deepcopy(provided_values)
        blood_pressure = values.pop("BP", None)
        if isinstance(blood_pressure, dict):
            if "SBP" in blood_pressure:
                values.setdefault("SBP", blood_pressure["SBP"])
            if "DBP" in blood_pressure:
                values.setdefault("DBP", blood_pressure["DBP"])
        if "SBP" in values and "DBP" in values and "MAP" not in values:
            values["MAP"] = round((values["SBP"] + (2 * values["DBP"])) / 3, 2)
        return values


class NurseInputReceivedEventHandler:
    """Merge nurse input with the original report and create one observation."""

    def __init__(
        self,
        task_repository: WorkflowTaskRepository,
        report_repository: LabReportRepository,
        patient_repository: PatientRepository,
        event_bus: EventBus,
        data_processor: DataProcessingAgent | None = None,
    ) -> None:
        self.task_repository = task_repository
        self.report_repository = report_repository
        self.patient_repository = patient_repository
        self.event_bus = event_bus
        self.data_processor = data_processor or DataProcessingAgent()
        self.event_bus.subscribe(EventType.NURSE_INPUT_RECEIVED, self.handle)

    def handle(self, event: WorkflowEvent) -> dict[str, Any] | None:
        task_id = event.related_task_id or event.payload.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            return None
        task = self.task_repository.get_task(task_id)
        if task is None or task["task_type"] != TaskType.NURSE_INPUT.value:
            return None
        if task["patient_id"] != event.patient_id or task["admission_id"] != event.admission_id:
            return None

        report_id = task.get("metadata", {}).get("report_id")
        event_report_id = event.related_report_id or event.payload.get("report_id")
        if not isinstance(report_id, str) or report_id != event_report_id:
            return None
        report = self.report_repository.get_report(report_id)
        if report is None:
            return None
        if report["patient_id"] != task["patient_id"] or report["admission_id"] != task["admission_id"]:
            return None

        existing_observation = self.patient_repository.get_observation_by_task(
            task["patient_id"],
            task["admission_id"],
            task_id,
        )
        if existing_observation is not None:
            return self._finish_existing(task, report, existing_observation)
        if task["status"] in {TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value}:
            return None

        try:
            nurse_values = task.get("metadata", {}).get("nurse_input")
            if not isinstance(nurse_values, dict):
                raise ValueError("Nurse input is not persisted on the task")
            question_fields = list(task.get("metadata", {}).get("question_fields", task["required_fields"]))
            question_index = int(task.get("metadata", {}).get("question_index", 0))
            if question_index < len(question_fields):
                self._publish_next_question(task, report)
                return task
            observation = self.data_processor.process(
                patient_id=task["patient_id"],
                admission_id=task["admission_id"],
                report_result=self._report_result(report),
                nurse_result={
                    "extracted_parameters": nurse_values,
                    "metadata": {"source": task.get("metadata", {}).get("nurse_input_source", "nurse_confirmed")},
                },
                observation_time=datetime.now(timezone.utc),
            )
            if not observation["ready_for_sepsis_assessment"]:
                raise ValueError(f"Missing clinical parameters: {observation['missing_sepsis_parameters']}")

            stored_observation = self.patient_repository.add_observation(
                patient_id=task["patient_id"],
                admission_id=task["admission_id"],
                observation_time=observation["observation_time"],
                clinical_parameters=observation["clinical_parameters"],
                source="merged_report_nurse",
                task_id=task_id,
                report_id=report_id,
                parameter_sources=observation["parameter_sources"],
            )
            completed_task = self.task_repository.complete_task(task_id)
            self._publish_observation_ready(completed_task, report, stored_observation)
            return stored_observation
        except Exception as exc:
            self._record_failure(task, str(exc))
            return None

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

    def _finish_existing(self, task: dict[str, Any], report: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
        completed_task = task
        if task["status"] != TaskStatus.COMPLETED.value:
            completed_task = self.task_repository.complete_task(task["task_id"])
        self._publish_observation_ready(completed_task, report, observation)
        return observation

    def _publish_observation_ready(self, task: dict[str, Any], report: dict[str, Any], observation: dict[str, Any]) -> None:
        event_repository = self.event_bus.event_repository
        if event_repository is not None:
            existing = event_repository.list_events(
                patient_id=task["patient_id"],
                admission_id=task["admission_id"],
                event_type=EventType.OBSERVATION_READY,
            )
            if any(item.get("payload", {}).get("task_id") == task["task_id"] for item in existing):
                return
        self.event_bus.publish(
            WorkflowEvent(
                event_type=EventType.OBSERVATION_READY,
                patient_id=task["patient_id"],
                admission_id=task["admission_id"],
                related_report_id=report["report_id"],
                related_task_id=task["task_id"],
                payload={
                    "observation_id": observation["observation_id"],
                    "report_id": report["report_id"],
                    "task_id": task["task_id"],
                },
            )
        )

    def _publish_next_question(self, task: dict[str, Any], report: dict[str, Any]) -> None:
        fields = list(task.get("metadata", {}).get("question_fields", task["required_fields"]))
        index = int(task.get("metadata", {}).get("question_index", 0))
        if index >= len(fields):
            return
        field = fields[index]
        display_name = InputOrchestrator.DISPLAY_NAMES.get(field, field)
        event_repository = self.event_bus.event_repository
        if event_repository is not None:
            existing = event_repository.list_events(
                patient_id=task["patient_id"],
                admission_id=task["admission_id"],
                event_type=EventType.NURSE_INPUT_REQUIRED,
            )
            if any(
                item.get("payload", {}).get("task_id") == task["task_id"]
                and item.get("payload", {}).get("question_index") == index
                for item in existing
            ):
                return
        self.event_bus.publish(
            WorkflowEvent(
                event_type=EventType.NURSE_INPUT_REQUIRED,
                patient_id=task["patient_id"],
                admission_id=task["admission_id"],
                related_report_id=report["report_id"],
                related_task_id=task["task_id"],
                payload={
                    "task_id": task["task_id"],
                    "report_id": report["report_id"],
                    "missing_fields": fields[index:],
                    "question_index": index,
                    "total_questions": len(fields),
                    "question_field": field,
                    "question": f"Please provide {display_name}.",
                },
            )
        )

    def _record_failure(self, task: dict[str, Any], message: str) -> None:
        metadata = deepcopy(task.get("metadata") or {})
        metadata["processing_error"] = message
        self.task_repository.update_task(task["task_id"], metadata=metadata)
