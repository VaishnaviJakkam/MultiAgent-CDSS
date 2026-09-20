from __future__ import annotations

from datetime import datetime
from typing import Any

from src.agents.disease_assessment_agent import DiseaseAssessmentAgent
from src.events import EventBus, EventType, WorkflowEvent
from src.database.repositories import PatientRepository


class ObservationReadyAssessmentHandler:
    """Run the existing disease orchestrator for one persisted observation."""

    def __init__(
        self,
        repository: PatientRepository,
        event_bus: EventBus,
        assessment_agent: DiseaseAssessmentAgent | Any | None = None,
    ) -> None:
        self.repository = repository
        self.event_bus = event_bus
        if assessment_agent is None:
            from src.agents.aki_agent import AKIDetectionAgent
            from src.agents.sepsis_agent import SepsisDetectionAgent

            assessment_agent = DiseaseAssessmentAgent(
                repository=repository,
                sepsis_tool=SepsisDetectionAgent(model="advanced"),
                aki_tool=AKIDetectionAgent(model="advanced"),
            )
        self.assessment_agent = assessment_agent
        self.event_bus.subscribe(EventType.OBSERVATION_READY, self.handle)

    def handle(self, event: WorkflowEvent) -> dict[str, Any] | None:
        observation_id = event.payload.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id.strip():
            return None
        patient_id = event.patient_id
        admission_id = event.admission_id
        if not isinstance(patient_id, str) or not isinstance(admission_id, str):
            return None

        observation = self.repository.get_observation(patient_id, admission_id, observation_id)
        if observation is None:
            return None
        if not self._event_matches_observation(event, observation):
            return None

        existing = self.repository.get_assessment_by_observation(
            patient_id,
            admission_id,
            observation_id,
        )
        if existing is not None:
            return existing

        try:
            result = self.assessment_agent.analyze(
                patient_id=patient_id,
                admission_id=admission_id,
                observation_id=observation_id,
                assessment_time=observation["observation_time"],
            )
            assessment = self.repository.get_assessment_by_observation(
                patient_id,
                admission_id,
                observation_id,
            )
            if assessment is None:
                assessment = self._persist_agent_result(
                    patient_id,
                    admission_id,
                    observation_id,
                    observation["observation_time"],
                    result,
                )
            if assessment is None:
                return result if isinstance(result, dict) else None

            if not self._is_fully_successful(result, assessment):
                return result if isinstance(result, dict) else assessment

            self._publish_completed(event, observation, assessment, result)
            return assessment
        except Exception as exc:
            return {
                "patient_id": patient_id,
                "admission_id": admission_id,
                "observation_id": observation_id,
                "status": "error",
                "message": str(exc),
            }

    @staticmethod
    def _event_matches_observation(event: WorkflowEvent, observation: dict[str, Any]) -> bool:
        for event_field, observation_field in (
            ("related_report_id", "report_id"),
            ("related_task_id", "task_id"),
        ):
            event_value = getattr(event, event_field)
            if event_value is not None and observation.get(observation_field) != event_value:
                return False
        payload = event.payload
        for payload_field in ("report_id", "task_id"):
            if payload_field in payload and observation.get(payload_field) != payload[payload_field]:
                return False
        return True

    def _persist_agent_result(
        self,
        patient_id: str,
        admission_id: str,
        observation_id: str,
        assessment_time: datetime,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(result, dict):
            return None
        sepsis_result = result.get("sepsis")
        aki_result = result.get("aki")
        if not isinstance(sepsis_result, dict) and not isinstance(aki_result, dict):
            return None
        return self.repository.add_assessment(
            patient_id=patient_id,
            admission_id=admission_id,
            assessment_time=assessment_time,
            sepsis_result=sepsis_result if isinstance(sepsis_result, dict) else None,
            aki_result=aki_result if isinstance(aki_result, dict) else None,
            observation_id=observation_id,
        )

    @staticmethod
    def _is_fully_successful(result: dict[str, Any], assessment: dict[str, Any]) -> bool:
        if result.get("status") != "completed":
            return False
        return (
            isinstance(assessment.get("sepsis"), dict)
            and assessment["sepsis"].get("status") == "success"
            and isinstance(assessment.get("aki"), dict)
            and assessment["aki"].get("status") == "success"
        )

    def _publish_completed(
        self,
        source_event: WorkflowEvent,
        observation: dict[str, Any],
        assessment: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        event_repository = self.event_bus.event_repository
        if event_repository is not None:
            existing = event_repository.list_events(
                patient_id=source_event.patient_id,
                admission_id=source_event.admission_id,
                event_type=EventType.ASSESSMENT_COMPLETED,
            )
            if any(item.get("payload", {}).get("observation_id") == observation["observation_id"] for item in existing):
                return
        self.event_bus.publish(
            WorkflowEvent(
                event_type=EventType.ASSESSMENT_COMPLETED,
                patient_id=observation["patient_id"],
                admission_id=observation["admission_id"],
                related_report_id=observation.get("report_id"),
                related_task_id=observation.get("task_id"),
                payload={
                    "observation_id": observation["observation_id"],
                    "assessment_id": assessment["assessment_id"],
                    "assessment_status": result.get("status"),
                    "disease_statuses": {
                        "sepsis": assessment.get("sepsis", {}).get("status"),
                        "aki": assessment.get("aki", {}).get("status"),
                    },
                },
            )
        )
