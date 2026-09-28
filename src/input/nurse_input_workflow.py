from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from src.database.repositories import LabReportRepository, PatientRepository, WorkflowTaskRepository
from src.database.workflow_tasks import TaskStatus, TaskType
from src.events import EventBus, EventType, WorkflowEvent
from src.agents.nurse_agent import NurseAgent
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
        audio_agent: Any | None = None,
        nurse_agent: NurseAgent | None = None,
    ) -> None:
        self.task_repository = task_repository
        self.report_repository = report_repository
        self.event_bus = event_bus
        self.input_orchestrator = input_orchestrator or InputOrchestrator()
        self.audio_agent = audio_agent
        self.nurse_agent = nurse_agent or NurseAgent()

    def set_audio_agent(self, audio_agent: Any) -> None:
        self.audio_agent = audio_agent

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
        question_index = int(task.get("metadata", {}).get("question_index", 0))
        current_field = requested_fields[question_index] if question_index < len(requested_fields) else None
        current_validation = self.input_orchestrator.validate_nurse_response(
            nurse_request={"requested_nurse_fields": [current_field] if current_field else []},
            nurse_result={"extracted_parameters": normalized_values},
            allow_explicit_none=bool(
                current_field == "BP"
                and normalized_values.get("SBP") is None
                and normalized_values.get("DBP") is None
            ) or bool(current_field and current_field in normalized_values and normalized_values[current_field] is None),
        )
        if not current_validation["all_requested_values_received"]:
            if current_field == "BP":
                raise ValueError("Could not understand the blood pressure value. Please try again.")
            raise ValueError(f"Could not understand the {current_field} value. Please try again.")

        answers.update(normalized_values)
        decision = self.nurse_agent.decide(
            patient_id=patient_id,
            admission_id=admission_id,
            lab_values=self._report_values(report),
            nurse_values=answers,
            unavailable_parameters=self._unavailable_parameters(answers),
            current_task_state={
                "status": task.get("status"),
                "question_index": question_index,
                "current_parameter": current_field,
            },
        )
        next_index = question_index
        next_index = 1 if decision.action == "COMPLETE" else 0

        metadata = deepcopy(task.get("metadata") or {})
        metadata["nurse_input"] = deepcopy(answers)
        metadata["answers"] = deepcopy(answers)
        metadata["question_index"] = next_index
        metadata["question_fields"] = [] if decision.action == "COMPLETE" else [decision.parameter]
        metadata["nurse_agent_decision"] = decision.as_dict()
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
                    "question_field": decision.parameter,
                    "question": decision.question,
                    "complete": decision.action == "COMPLETE",
                },
            )
        )
        # Downstream handlers may complete the task synchronously after the
        # final answer creates OBSERVATION_READY. Return the persisted state.
        return self.task_repository.get_task(task_id) or updated_task

    def submit_nurse_audio(
        self,
        task_id: str,
        patient_id: str,
        admission_id: str,
        audio_path: str,
        audio_agent: Any | None = None,
    ) -> dict[str, Any]:
        active_audio_agent = audio_agent or self.audio_agent
        if active_audio_agent is None:
            raise RuntimeError("Nurse audio agent is not initialized")
        result = active_audio_agent.analyze(audio_path)
        extracted = result.get("extracted_parameters")
        if not isinstance(extracted, dict):
            raise ValueError("Nurse audio did not produce structured clinical values")
        if result.get("no_value"):
            task = self.task_repository.get_task(task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")
            fields = list(task.get("metadata", {}).get("question_fields", task["required_fields"]))
            index = int(task.get("metadata", {}).get("question_index", 0))
            current_field = fields[index] if index < len(fields) else None
            if current_field == "BP":
                extracted = {"SBP": None, "DBP": None}
            elif current_field:
                extracted = {current_field: None}
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
        if (
            "SBP" in values
            and "DBP" in values
            and values["SBP"] is not None
            and values["DBP"] is not None
            and "MAP" not in values
        ):
            values["MAP"] = round((values["SBP"] + (2 * values["DBP"])) / 3, 2)
        return values

    @staticmethod
    def _report_values(report: dict[str, Any]) -> dict[str, Any]:
        extracted_data = report.get("extracted_data") or {}
        return {
            parameter: value.get("value") if isinstance(value, dict) and "value" in value else value
            for parameter, value in extracted_data.items()
        }

    @staticmethod
    def _unavailable_parameters(answers: dict[str, Any]) -> set[str]:
        unavailable = {key for key, value in answers.items() if value is None}
        if answers.get("SBP") is None and answers.get("DBP") is None and ("SBP" in answers or "DBP" in answers):
            unavailable.add("BP")
        return unavailable


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
        agent_decision = task.get("metadata", {}).get("nurse_agent_decision", {})
        question = agent_decision.get("question") if isinstance(agent_decision, dict) else None
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
                    "question": question or f"Please provide {display_name}.",
                },
            )
        )

    def _record_failure(self, task: dict[str, Any], message: str) -> None:
        metadata = deepcopy(task.get("metadata") or {})
        metadata["processing_error"] = message
        self.task_repository.update_task(task["task_id"], metadata=metadata)
