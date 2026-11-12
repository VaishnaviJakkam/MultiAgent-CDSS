from src.aki.preprocessing import get_strict_feature_columns, load_raw_dataset
from src.agents.aki_agent import AKIDetectionAgent


def synthetic_patient_data() -> dict[str, float | str]:
    data: dict[str, float | str] = {"patient_reference": "TEST-AKI"}
    for index, feature in enumerate(get_strict_feature_columns(load_raw_dataset())):
        data[feature] = float(index + 1)
    return data


def test_advanced_xgboost_model_loads() -> None:
    agent = AKIDetectionAgent()
    assert agent.model_name == "advanced"
    assert len(agent.feature_names) == 60
    assert agent.threshold == 0.6


def test_valid_patient_data_produces_xgboost_result() -> None:
    result = AKIDetectionAgent().analyze(synthetic_patient_data())
    assert result["status"] == "success"
    assert result["disease"] == "AKI"
    assert result["model"] == "XGBoost"


def test_probability_and_threshold_application() -> None:
    result = AKIDetectionAgent().analyze(synthetic_patient_data())
    assert 0.0 <= result["probability"] <= 1.0
    assert result["patient_id"] == "TEST-AKI"
    assert result["disease"] == "AKI"
    assert result["model"] == "XGBoost"
    assert result["observation_count"] == 1
    assert result["assessment_time"] is None
    assert result["status"] == "success"
    assert "message" in result
    assert result["prediction"] == int(result["probability"] >= result["threshold"])


def test_missing_feature_is_rejected() -> None:
    data = synthetic_patient_data()
    data.pop("ANIONGAP_min")
    result = AKIDetectionAgent().analyze(data)
    assert result["status"] == "error"
    assert "Missing required model features" in result["message"]


def test_forbidden_leakage_fields_are_rejected() -> None:
    data = synthetic_patient_data()
    data["creat"] = 1.0
    result = AKIDetectionAgent().analyze(data)
    assert result["status"] == "error"
    assert "Forbidden fields supplied" in result["message"]


def test_structured_result_contains_advanced_model_identity() -> None:
    result = AKIDetectionAgent().analyze(synthetic_patient_data())
    assert {"disease", "model", "probability", "threshold", "prediction", "risk_level"}.issubset(result)


def test_explicit_baseline_model_remains_available() -> None:
    result = AKIDetectionAgent(model="baseline").analyze(synthetic_patient_data())
    assert result["status"] == "success"
    assert result["model"] == "Baseline"
