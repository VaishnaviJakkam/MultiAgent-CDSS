from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PatientAssessmentResult:
    """Unified model-output contract for disease detection agents."""

    patient_id: str
    disease: str
    model: str
    prediction: int
    probability: float
    threshold: float
    risk_level: str
    observation_count: int
    assessment_time: Any | None
    status: str
    message: str

    def __post_init__(self) -> None:
        if self.prediction not in {0, 1}:
            raise ValueError("prediction must be binary")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.observation_count < 0:
            raise ValueError("observation_count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
