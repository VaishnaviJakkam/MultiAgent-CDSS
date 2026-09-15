from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any


SUPPORTED_DISEASES = frozenset({"Sepsis", "AKI"})
PROBABILITY_CHANGE_THRESHOLD = 0.05


@dataclass(frozen=True)
class TrendAnalysisResult:
    patient_id: str
    disease: str
    trend: str
    first_probability: float | None
    latest_probability: float | None
    probability_change: float | None
    observation_count: int
    assessment_time_start: Any | None
    assessment_time_end: Any | None
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrendAnalysisAgent:
    """Rule-based trend analysis over repeated model assessment results."""

    def __init__(self, change_threshold: float = PROBABILITY_CHANGE_THRESHOLD):
        if change_threshold <= 0.0:
            raise ValueError("change_threshold must be positive")
        self.change_threshold = float(change_threshold)

    def _normalize_record(self, record: Any) -> dict[str, Any]:
        if isinstance(record, TrendAnalysisResult):
            return record.to_dict()
        if hasattr(record, "to_dict") and callable(record.to_dict):
            normalized = record.to_dict()
        elif isinstance(record, dict):
            normalized = dict(record)
        else:
            raise ValueError("Each assessment must be a PatientAssessmentResult or compatible dictionary")

        if "patient_id" not in normalized and "patient_reference" in normalized:
            normalized["patient_id"] = normalized["patient_reference"]
        required = {"patient_id", "disease", "probability"}
        missing = sorted(required.difference(normalized))
        if missing:
            raise ValueError(f"Assessment is missing required fields: {missing}")
        return normalized

    def _validate_and_order(self, assessments: Iterable[Any]) -> list[dict[str, Any]]:
        if assessments is None:
            raise ValueError("assessments cannot be None")
        records = [self._normalize_record(record) for record in assessments]
        if not records:
            raise ValueError("assessments cannot be empty")

        patient_ids = {record["patient_id"] for record in records}
        if len(patient_ids) != 1:
            raise ValueError("All assessments must belong to the same patient")
        diseases = {record["disease"] for record in records}
        if len(diseases) != 1:
            raise ValueError("All assessments must belong to the same disease")
        disease = next(iter(diseases))
        if disease not in SUPPORTED_DISEASES:
            raise ValueError(f"Unsupported disease: {disease}")

        for record in records:
            probability = record["probability"]
            if isinstance(probability, bool) or not isinstance(probability, Real):
                raise ValueError("Assessment probability must be numeric")
            probability = float(probability)
            if not 0.0 <= probability <= 1.0:
                raise ValueError("Assessment probability must be between 0 and 1")
            record["probability"] = probability

        times = [record.get("assessment_time") for record in records]
        if all(time is None for time in times):
            return records
        if any(time is None for time in times):
            raise ValueError("Either all assessment times must be present or all must be absent")
        try:
            return sorted(records, key=lambda record: record["assessment_time"])
        except TypeError as exc:
            raise ValueError("assessment_time values must be mutually orderable") from exc

    def analyze(self, assessments: Iterable[Any]) -> dict[str, Any]:
        records = self._validate_and_order(assessments)
        patient_id = str(records[0]["patient_id"])
        disease = records[0]["disease"]
        count = len(records)
        first_probability = records[0]["probability"]
        latest_probability = records[-1]["probability"]
        start_time = records[0].get("assessment_time")
        end_time = records[-1].get("assessment_time")

        if count < 2:
            return TrendAnalysisResult(
                patient_id=patient_id,
                disease=disease,
                trend="INSUFFICIENT_DATA",
                first_probability=first_probability,
                latest_probability=latest_probability,
                probability_change=None,
                observation_count=count,
                assessment_time_start=start_time,
                assessment_time_end=end_time,
                status="INSUFFICIENT_DATA",
                message="At least two assessments are required to determine a trend.",
            ).to_dict()

        probability_change = latest_probability - first_probability
        if probability_change > self.change_threshold:
            trend = "WORSENING"
        elif probability_change < -self.change_threshold:
            trend = "IMPROVING"
        else:
            trend = "STABLE"

        return TrendAnalysisResult(
            patient_id=patient_id,
            disease=disease,
            trend=trend,
            first_probability=first_probability,
            latest_probability=latest_probability,
            probability_change=probability_change,
            observation_count=count,
            assessment_time_start=start_time,
            assessment_time_end=end_time,
            status="success",
            message="Prototype probability trend; not a clinical deterioration rule.",
        ).to_dict()
