from __future__ import annotations

from src.agents.aki_agent import AKIDetectionAgent


def synthetic_patient_data() -> dict[str, float | str]:
    base = {
        "patient_reference": "PTEST1",
        "ANIONGAP_min": 10.0,
        "ANIONGAP_max": 12.0,
        "ALBUMIN_min": 3.5,
        "ALBUMIN_max": 4.0,
        "BANDS_min": 0.1,
        "BANDS_max": 0.2,
        "BICARBONATE_min": 22.0,
        "BICARBONATE_max": 24.0,
        "BILIRUBIN_min": 0.4,
        "BILIRUBIN_max": 0.7,
        "CHLORIDE_min": 101.0,
        "CHLORIDE_max": 104.0,
        "GLUCOSE_min": 90.0,
        "GLUCOSE_max": 105.0,
        "HEMATOCRIT_min": 37.0,
        "HEMATOCRIT_max": 41.0,
        "HEMOGLOBIN_min": 12.0,
        "HEMOGLOBIN_max": 13.0,
        "LACTATE_min": 1.0,
        "LACTATE_max": 1.2,
        "PLATELET_min": 180.0,
        "PLATELET_max": 210.0,
        "POTASSIUM_min": 3.9,
        "POTASSIUM_max": 4.1,
        "PTT_min": 28.0,
        "PTT_max": 32.0,
        "INR_min": 1.0,
        "INR_max": 1.1,
        "PT_min": 12.0,
        "PT_max": 14.0,
        "SODIUM_min": 137.0,
        "SODIUM_max": 139.0,
        "BUN_min": 13.0,
        "BUN_max": 15.0,
        "WBC_min": 6.5,
        "WBC_max": 8.5,
        "HeartRate_Min": 70.0,
        "HeartRate_Max": 88.0,
        "HeartRate_Mean": 79.0,
        "SysBP_Min": 110.0,
        "SysBP_Max": 128.0,
        "SysBP_Mean": 118.0,
        "DiasBP_Min": 66.0,
        "DiasBP_Max": 78.0,
        "DiasBP_Mean": 72.0,
        "MeanBP_Min": 82.0,
        "MeanBP_Max": 94.0,
        "MeanBP_Mean": 87.0,
        "RespRate_Min": 14.0,
        "RespRate_Max": 18.0,
        "RespRate_Mean": 16.0,
        "TempC_Min": 36.7,
        "TempC_Max": 37.2,
        "TempC_Mean": 36.95,
        "SpO2_Min": 96.0,
        "SpO2_Max": 99.0,
        "SpO2_Mean": 97.5,
        "Glucose_Min_1": 93.0,
        "Glucose_Max_1": 104.0,
        "Glucose_Mean": 98.5,
    }
    return base


def test_agent_initializes() -> None:
    agent = AKIDetectionAgent()
    assert agent.threshold == 0.5
    assert agent.feature_names


def test_model_loads() -> None:
    agent = AKIDetectionAgent()
    assert hasattr(agent.pipeline, "predict_proba")


def test_valid_patient_data_produces_result() -> None:
    agent = AKIDetectionAgent()
    data = synthetic_patient_data()
    result = agent.analyze(data)
    assert result["status"] == "success"
    assert result["agent"] == "AKIDetectionAgent"
    assert result["disease"] == "AKI"


def test_probability_is_between_0_and_1() -> None:
    agent = AKIDetectionAgent()
    result = agent.analyze(synthetic_patient_data())
    assert 0.0 <= result["probability"] <= 1.0


def test_prediction_is_0_or_1() -> None:
    agent = AKIDetectionAgent()
    result = agent.analyze(synthetic_patient_data())
    assert result["prediction"] in {0, 1}


def test_threshold_is_between_0_and_1() -> None:
    agent = AKIDetectionAgent()
    result = agent.analyze(synthetic_patient_data())
    assert 0.0 <= result["threshold"] <= 1.0


def test_missing_feature_produces_clear_error() -> None:
    agent = AKIDetectionAgent()
    data = synthetic_patient_data()
    data.pop("ANIONGAP_min")
    result = agent.analyze(data)
    assert result["status"] == "error"
    assert "Missing required model features" in result["message"]


def test_target_column_cannot_be_passed() -> None:
    agent = AKIDetectionAgent()
    data = synthetic_patient_data()
    data["AKI_TARGET"] = 1
    result = agent.analyze(data)
    assert result["status"] == "error"
    assert "Forbidden fields supplied" in result["message"]


def test_excluded_leakage_variables_are_not_passed() -> None:
    agent = AKIDetectionAgent()
    data = synthetic_patient_data()
    data["creat"] = 1.0
    result = agent.analyze(data)
    assert result["status"] == "error"
    assert "Forbidden fields supplied" in result["message"]


def test_output_contains_expected_fields() -> None:
    agent = AKIDetectionAgent()
    result = agent.analyze(synthetic_patient_data())
    expected = {"agent", "patient_reference", "disease", "prediction", "probability", "threshold", "risk_level", "status"}
    assert expected.issubset(result.keys())


if __name__ == "__main__":
    test_agent_initializes()
    test_model_loads()
    test_valid_patient_data_produces_result()
    test_probability_is_between_0_and_1()
    test_prediction_is_0_or_1()
    test_threshold_is_between_0_and_1()
    test_missing_feature_produces_clear_error()
    test_target_column_cannot_be_passed()
    test_excluded_leakage_variables_are_not_passed()
    test_output_contains_expected_fields()
    print("AKI agent tests passed")
