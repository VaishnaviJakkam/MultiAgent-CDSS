from __future__ import annotations

from datetime import datetime
from typing import Any

from src.agents.risk_prioritization_agent import RiskPrioritizationAgent
from src.database.repositories import PatientRepository
from src.events import EventBus, EventType, WorkflowEvent


class TrendCompletedRiskPrioritizationHandler:
    """Prioritize the exact patient/admission history after trend persistence."""

    def __init__(self, repository: PatientRepository, event_bus: EventBus, agent: RiskPrioritizationAgent | Any | None = None) -> None:
        self.repository = repository
        self.event_bus = event_bus
        self.agent = agent or RiskPrioritizationAgent()
        self.event_bus.subscribe(EventType.TREND_ANALYSIS_COMPLETED, self.handle)

    def handle(self, event: WorkflowEvent) -> dict[str, Any] | None:
        patient_id = event.patient_id
        admission_id = event.admission_id
        trend_ids = self._trend_ids(event.payload)
        if not all(isinstance(value, str) and value.strip() for value in (patient_id, admission_id)) or not trend_ids:
            return None

        try:
            trends = [self.repository.get_trend(patient_id, admission_id, trend_id) for trend_id in trend_ids]
            if any(trend is None for trend in trends):
                return {"status": "error", "message": "triggering trend was not found"}
            stored_trends = [trend for trend in trends if trend is not None]
            history = self.repository.get_patient_history(patient_id, admission_id)
            if any(trend.get("_id") not in {item.get("_id") for item in history["trends"]} for trend in stored_trends):
                return {"status": "error", "message": "triggering trend is outside patient admission history"}

            source_trend_id = trend_ids[0]
            if self.repository.get_prioritization_by_source_trend(patient_id, admission_id, source_trend_id) is not None:
                return {"status": "already_processed"}

            result = self.agent.prioritize_history(history["assessments"], history["trends"])
            priority = result.get("priority") or "INSUFFICIENT_DATA"
            reasons = result.get("reasons") or ["Insufficient persisted history for prioritization"]
            current_trend = max(stored_trends, key=lambda item: item["assessment_time"])
            current_assessment = next((item for item in history["assessments"] if item.get("assessment_id") == current_trend.get("assessment_id")), None)
            persisted = self.repository.add_prioritization(
                patient_id=patient_id,
                admission_id=admission_id,
                priority_level=priority,
                highest_risk_disease=result.get("highest_risk_disease", "Unknown"),
                highest_probability=float(result.get("highest_probability") or 0.0),
                worsening_diseases=list(result.get("worsening_diseases", [])),
                reason="; ".join(reasons),
                assessment_time=(current_assessment or history["assessments"][-1])["assessment_time"] if history["assessments"] else datetime.now().astimezone(),
                source_trend_id=source_trend_id,
                assessment_ids=[item["assessment_id"] for item in history["assessments"]],
                trend_ids=[item["_id"] for item in history["trends"]],
            )
            if result.get("status") != "success":
                return {"status": result.get("status", "insufficient_data"), "prioritization": persisted}
            self.event_bus.publish(WorkflowEvent(
                event_type=EventType.RISK_PRIORITIZATION_COMPLETED,
                patient_id=patient_id,
                admission_id=admission_id,
                payload={"prioritization_id": persisted["_id"], "source_trend_id": source_trend_id, "priority_level": priority},
            ))
            self.event_bus.publish(WorkflowEvent(
                event_type=EventType.WORKFLOW_COMPLETED,
                patient_id=patient_id,
                admission_id=admission_id,
                payload={
                    "prioritization_id": persisted["_id"],
                    "source_trend_id": source_trend_id,
                    "priority_level": priority,
                    "workflow_status": "COMPLETED",
                },
            ))
            return {"status": "success", "prioritization": persisted}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @staticmethod
    def _trend_ids(payload: dict[str, Any]) -> list[str]:
        value = payload.get("trend_id")
        if isinstance(value, str) and value.strip():
            return [value]
        values = payload.get("trend_ids")
        if isinstance(values, list) and values and all(isinstance(item, str) and item.strip() for item in values):
            return values
        return []


RiskPrioritizationEventHandler = TrendCompletedRiskPrioritizationHandler
TrendAnalysisCompletedRiskPrioritizationHandler = TrendCompletedRiskPrioritizationHandler