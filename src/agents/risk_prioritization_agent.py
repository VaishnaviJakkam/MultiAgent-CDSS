from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

SUPPORTED_DISEASES = frozenset({"Sepsis", "AKI"})
VALID_TRENDS = frozenset({"WORSENING", "IMPROVING", "STABLE", "INSUFFICIENT_DATA"})
HIGH_PROBABILITY_THRESHOLD = 0.70
MODERATE_PROBABILITY_THRESHOLD = 0.40
WORSENING_HIGH_PROBABILITY_THRESHOLD = 0.50


@dataclass(frozen=True)
class RiskPrioritizationResult:
    patient_id: str
    priority_level: str
    diseases_assessed: list[str]
    highest_probability: float
    highest_risk_disease: str
    worsening_diseases: list[str]
    available_trends: dict[str, str]
    reason: str
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RiskPrioritizationAgent:
    """Prototype rule-based prioritization over disease results and trends."""

    def _normalize(self, record: Any, kind: str) -> dict[str, Any]:
        if hasattr(record, "to_dict") and callable(record.to_dict):
            result = record.to_dict()
        elif isinstance(record, dict):
            result = dict(record)
        else:
            raise ValueError(f"Each {kind} must be a compatible result dictionary or schema object")
        if "patient_id" not in result and "patient_reference" in result:
            result["patient_id"] = result["patient_reference"]
        return result

    def _validate_assessments(self, assessments: Iterable[Any]) -> list[dict[str, Any]]:
        if assessments is None:
            raise ValueError("assessments cannot be None")
        records = [self._normalize(record, "assessment") for record in assessments]
        if not records:
            raise ValueError("assessments cannot be empty")
        required = {"patient_id", "disease", "probability"}
        for record in records:
            missing = sorted(required.difference(record))
            if missing:
                raise ValueError(f"Assessment is missing required fields: {missing}")
            if record["disease"] not in SUPPORTED_DISEASES:
                raise ValueError(f"Unsupported disease: {record['disease']}")
            probability = record["probability"]
            if isinstance(probability, bool) or not isinstance(probability, Real):
                raise ValueError("Assessment probability must be numeric")
            if not 0.0 <= float(probability) <= 1.0:
                raise ValueError("Assessment probability must be between 0 and 1")
            record["probability"] = float(probability)
        patient_ids = {record["patient_id"] for record in records}
        if len(patient_ids) != 1:
            raise ValueError("All assessments must belong to the same patient")
        diseases = [record["disease"] for record in records]
        if len(set(diseases)) != len(diseases):
            raise ValueError("Duplicate disease assessments are not supported")
        return records

    def _validate_trends(self, trends: Iterable[Any] | None, patient_id: Any) -> dict[str, str]:
        if trends is None:
            return {}
        trend_records = [self._normalize(record, "trend") for record in trends]
        available: dict[str, str] = {}
        for record in trend_records:
            required = {"patient_id", "disease", "trend"}
            missing = sorted(required.difference(record))
            if missing:
                raise ValueError(f"Trend is missing required fields: {missing}")
            if record["patient_id"] != patient_id:
                raise ValueError("Trend patient_id does not match assessment patient_id")
            if record["disease"] not in SUPPORTED_DISEASES:
                raise ValueError(f"Unsupported trend disease: {record['disease']}")
            if record["trend"] not in VALID_TRENDS:
                raise ValueError(f"Unsupported trend value: {record['trend']}")
            if record["disease"] in available:
                raise ValueError("Duplicate trend results are not supported")
            available[record["disease"]] = record["trend"]
        return available

    def prioritize(self, assessments: Iterable[Any], trends: Iterable[Any] | None = None) -> dict[str, Any]:
        records = self._validate_assessments(assessments)
        patient_id = records[0]["patient_id"]
        available_trends = self._validate_trends(trends, patient_id)
        probabilities = {record["disease"]: record["probability"] for record in records}
        diseases = list(probabilities)
        highest_risk_disease = max(diseases, key=probabilities.get)
        highest_probability = probabilities[highest_risk_disease]
        worsening = [disease for disease in diseases if available_trends.get(disease) == "WORSENING"]
        has_both_moderate = len(probabilities) == 2 and all(value >= MODERATE_PROBABILITY_THRESHOLD for value in probabilities.values())
        high_probability = highest_probability >= HIGH_PROBABILITY_THRESHOLD
        high_worsening = any(probabilities[disease] >= WORSENING_HIGH_PROBABILITY_THRESHOLD for disease in worsening)
        high_combined = has_both_moderate and bool(worsening)

        if high_probability or high_worsening or high_combined:
            priority = "HIGH"
            if high_probability:
                reason = f"High {highest_risk_disease} risk with probability {highest_probability:.2f}."
            elif high_worsening:
                disease = next(disease for disease in worsening if probabilities[disease] >= WORSENING_HIGH_PROBABILITY_THRESHOLD)
                reason = f"Moderate-to-high {disease} risk with worsening trend."
            else:
                reason = "Both diseases have moderate risk and at least one has a worsening trend."
        elif highest_probability >= MODERATE_PROBABILITY_THRESHOLD or worsening or has_both_moderate:
            priority = "MEDIUM"
            if worsening:
                reason = f"{worsening[0]} has a worsening trend without meeting the high-priority rule."
            elif has_both_moderate:
                reason = "Both assessed diseases have moderate risk without a high-priority rule."
            else:
                reason = f"Moderate {highest_risk_disease} risk; no high-priority rule met."
        else:
            priority = "LOW"
            reason = "Low disease risk with stable available trends."

        trend_message = "Trend information unavailable." if not available_trends else "Available trends were incorporated."
        return RiskPrioritizationResult(
            patient_id=str(patient_id),
            priority_level=priority,
            diseases_assessed=diseases,
            highest_probability=highest_probability,
            highest_risk_disease=highest_risk_disease,
            worsening_diseases=worsening,
            available_trends=available_trends,
            reason=reason,
            status="success",
            message=f"Prototype prioritization only; not clinical guidance. {trend_message}",
        ).to_dict()

    def analyze(self, assessments: Iterable[Any], trends: Iterable[Any] | None = None) -> dict[str, Any]:
        return self.prioritize(assessments, trends)
