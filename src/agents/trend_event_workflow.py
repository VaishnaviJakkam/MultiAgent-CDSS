from __future__ import annotations

from datetime import datetime
from typing import Any

from src.agents.trend_analysis_agent import TrendAnalysisAgent
from src.database.repositories import PatientRepository
from src.events import EventBus, EventType, WorkflowEvent


class AssessmentCompletedTrendHandler:
    """Analyze actual scoped assessment history after assessment completion."""

    def __init__(
        self,
        repository: PatientRepository,
        event_bus: EventBus,
        trend_agent: TrendAnalysisAgent | Any | None = None,
    ) -> None:
        self.repository = repository
        self.event_bus = event_bus
        self.trend_agent = trend_agent or TrendAnalysisAgent()
        self.event_bus.subscribe(EventType.ASSESSMENT_COMPLETED, self.handle)

    def handle(self, event: WorkflowEvent) -> dict[str, Any] | None:
        patient_id = event.patient_id
        admission_id = event.admission_id
        assessment_id = event.payload.get("assessment_id")
        if not all(isinstance(value, str) and value.strip() for value in (patient_id, admission_id, assessment_id)):
            return None

        assessment = self._get_scoped_assessment(patient_id, admission_id, assessment_id)
        if assessment is None:
            return None
        event_observation_id = event.payload.get("observation_id")
        if event_observation_id is not None and assessment.get("observation_id") != event_observation_id:
            return None

        existing_trends = self.repository.get_trends_by_assessment(
            patient_id,
            admission_id,
            assessment_id,
        )
        if existing_trends:
            return {"status": "already_processed", "trends": existing_trends}

        try:
            history = self.repository.get_patient_history(patient_id, admission_id)
            result = self.trend_agent.analyze(history)
            disease_trends = result.get("disease_trends", {})
            if not isinstance(disease_trends, dict):
                raise ValueError("Trend analysis did not return disease trends")

            persisted_trends: list[dict[str, Any]] = []
            for disease in ("Sepsis", "AKI"):
                disease_result = disease_trends.get(disease)
                if not isinstance(disease_result, dict):
                    raise ValueError(f"Trend analysis did not return {disease} trend")
                persisted_trends.append(
                    self.repository.add_trend(
                        patient_id=patient_id,
                        admission_id=admission_id,
                        disease=disease,
                        assessment_time=assessment["assessment_time"],
                        trend=disease_result.get("trend", "INSUFFICIENT_DATA"),
                        first_probability=disease_result.get("first"),
                        latest_probability=disease_result.get("latest"),
                        probability_change=disease_result.get("change"),
                        observation_count=disease_result.get("count", 0),
                        assessment_id=assessment_id,
                        trend_result=disease_result,
                    )
                )

            if result.get("status") != "success":
                return {"status": result.get("status", "insufficient_data"), "trends": persisted_trends}

            self._publish_completed(event, assessment, result, persisted_trends)
            return {"status": "success", "trends": persisted_trends, "analysis": result}
        except Exception as exc:
            return {
                "patient_id": patient_id,
                "admission_id": admission_id,
                "assessment_id": assessment_id,
                "status": "error",
                "message": str(exc),
            }

    def _get_scoped_assessment(self, patient_id: str, admission_id: str, assessment_id: str) -> dict[str, Any] | None:
        return self.repository.get_assessment(patient_id, admission_id, assessment_id)

    def _publish_completed(
        self,
        source_event: WorkflowEvent,
        assessment: dict[str, Any],
        result: dict[str, Any],
        trends: list[dict[str, Any]],
    ) -> None:
        event_repository = self.event_bus.event_repository
        if event_repository is not None:
            existing = event_repository.list_events(
                patient_id=source_event.patient_id,
                admission_id=source_event.admission_id,
                event_type=EventType.TREND_ANALYSIS_COMPLETED,
            )
            if any(item.get("payload", {}).get("assessment_id") == assessment["assessment_id"] for item in existing):
                return
        self.event_bus.publish(
            WorkflowEvent(
                event_type=EventType.TREND_ANALYSIS_COMPLETED,
                patient_id=assessment["patient_id"],
                admission_id=assessment["admission_id"],
                payload={
                    "assessment_id": assessment["assessment_id"],
                    "trend_ids": [trend["_id"] for trend in trends],
                    "analysis_status": result.get("status"),
                },
            )
        )
