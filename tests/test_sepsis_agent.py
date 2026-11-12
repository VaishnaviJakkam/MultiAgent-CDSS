import pytest

from src.agents.sepsis_agent import SepsisDetectionAgent


def synthetic_history(count: int = 12) -> list[dict[str, float | str]]:
    return [
        {
            "Patient_ID": "TEST-SEPSIS",
            "Hour": hour,
            "HR": 70.0 + hour,
            "O2Sat": 98.0 - hour * 0.1,
            "Temp": 36.8 + hour * 0.02,
            "SBP": 120.0 - hour * 0.2,
            "MAP": 80.0 - hour * 0.1,
            "Resp": 16.0 + hour * 0.1,
            "WBC": 7.0 + hour * 0.05,
            "Lactate": 1.0 + hour * 0.01,
        }
        for hour in reversed(range(count))
    ]


def test_advanced_lstm_model_loads() -> None:
    agent = SepsisDetectionAgent()
    assert agent.model_name == "advanced"
    assert agent.sequence_length == 12
    assert agent.feature_columns == ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "WBC", "Lactate"]


def test_missing_required_lstm_feature_is_rejected() -> None:
    history = synthetic_history()
    for observation in history:
        observation.pop("Lactate")
    with pytest.raises(ValueError, match="Missing required LSTM features"):
        SepsisDetectionAgent().analyze(history)


def test_patient_id_consistency_is_rejected() -> None:
    history = synthetic_history()
    history[-1]["Patient_ID"] = "OTHER"
    with pytest.raises(ValueError, match="Inconsistent Patient_ID"):
        SepsisDetectionAgent().analyze(history)


def test_chronological_ordering_and_lstm_inference() -> None:
    result = SepsisDetectionAgent().analyze(synthetic_history())
    assert result["status"] == "success"
    assert result["model"] == "LSTM"
    assert result["number_of_observations_used"] == 12


def test_insufficient_sequence_is_rejected_without_padding() -> None:
    with pytest.raises(ValueError, match="at least 12 real observations"):
        SepsisDetectionAgent().analyze(synthetic_history(11))


def test_probability_threshold_and_structured_result() -> None:
    result = SepsisDetectionAgent().analyze(synthetic_history())
    assert 0.0 <= result["probability"] <= 1.0
    assert result["patient_id"] == "TEST-SEPSIS"
    assert result["disease"] == "Sepsis"
    assert result["model"] == "LSTM"
    assert result["observation_count"] == 12
    assert result["assessment_time"] == 11
    assert result["status"] == "success"
    assert "message" in result
    assert result["prediction"] == int(result["probability"] >= result["threshold"])
    assert {"disease", "model", "probability", "threshold", "prediction", "risk_level", "number_of_observations_used"}.issubset(result)


def test_explicit_baseline_model_remains_available() -> None:
    result = SepsisDetectionAgent(model="baseline").analyze(synthetic_history())
    assert result["status"] == "success"
    assert result["model"] == "Temporal Logistic Regression"
