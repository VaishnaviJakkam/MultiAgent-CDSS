import pytest

from src.agents.result_schema import PatientAssessmentResult


def test_result_schema_serializes_required_fields() -> None:
    result = PatientAssessmentResult(
        patient_id="P1",
        disease="Sepsis",
        model="LSTM",
        prediction=1,
        probability=0.8,
        threshold=0.6,
        risk_level="ELEVATED",
        observation_count=12,
        assessment_time=11,
        status="success",
        message="Prototype output; not a medical diagnosis.",
    ).to_dict()
    assert set(result) == {"patient_id", "disease", "model", "prediction", "probability", "threshold", "risk_level", "observation_count", "assessment_time", "status", "message"}


@pytest.mark.parametrize("field,value", [("prediction", 2), ("probability", 1.1), ("threshold", -0.1), ("observation_count", -1)])
def test_result_schema_rejects_invalid_values(field: str, value: object) -> None:
    values = {
        "patient_id": "P1", "disease": "AKI", "model": "XGBoost", "prediction": 0,
        "probability": 0.2, "threshold": 0.5, "risk_level": "LOWER", "observation_count": 1,
        "assessment_time": None, "status": "success", "message": "Prototype output.",
    }
    values[field] = value
    with pytest.raises(ValueError):
        PatientAssessmentResult(**values)