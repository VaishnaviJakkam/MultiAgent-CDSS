import pytest

from src.agents.risk_prioritization_agent import RiskPrioritizationAgent


def assessment(patient_id: str, disease: str, probability: float) -> dict[str, object]:
    return {"patient_id": patient_id, "disease": disease, "probability": probability}


def trend(patient_id: str, disease: str, value: str) -> dict[str, object]:
    return {"patient_id": patient_id, "disease": disease, "trend": value}


def test_high_sepsis_probability() -> None:
    result = RiskPrioritizationAgent().analyze([assessment("P001", "Sepsis", 0.82)])
    assert result["priority_level"] == "HIGH"
    assert "High Sepsis risk" in result["reason"]


def test_moderate_sepsis_with_worsening_is_high() -> None:
    result = RiskPrioritizationAgent().analyze([assessment("P001", "Sepsis", 0.55)], [trend("P001", "Sepsis", "WORSENING")])
    assert result["priority_level"] == "HIGH"
    assert "worsening" in result["reason"]


def test_moderate_aki_is_medium() -> None:
    result = RiskPrioritizationAgent().analyze([assessment("P001", "AKI", 0.45)])
    assert result["priority_level"] == "MEDIUM"


def test_low_diseases_with_stable_trends_are_low() -> None:
    assessments = [assessment("P001", "Sepsis", 0.20), assessment("P001", "AKI", 0.25)]
    trends = [trend("P001", "Sepsis", "STABLE"), trend("P001", "AKI", "STABLE")]
    result = RiskPrioritizationAgent().analyze(assessments, trends)
    assert result["priority_level"] == "LOW"


def test_both_diseases_and_one_worsening_are_high() -> None:
    assessments = [assessment("P001", "Sepsis", 0.42), assessment("P001", "AKI", 0.44)]
    trends = [trend("P001", "AKI", "WORSENING")]
    result = RiskPrioritizationAgent().analyze(assessments, trends)
    assert result["priority_level"] == "HIGH"
    assert result["worsening_diseases"] == ["AKI"]


def test_only_sepsis_is_valid() -> None:
    result = RiskPrioritizationAgent().analyze([assessment("P001", "Sepsis", 0.35)])
    assert result["diseases_assessed"] == ["Sepsis"]


def test_only_aki_is_valid() -> None:
    result = RiskPrioritizationAgent().analyze([assessment("P001", "AKI", 0.35)])
    assert result["diseases_assessed"] == ["AKI"]


def test_missing_trends_are_explicit() -> None:
    result = RiskPrioritizationAgent().analyze([assessment("P001", "Sepsis", 0.35)])
    assert result["available_trends"] == {}
    assert "unavailable" in result["message"]


def test_mixed_patient_ids_fail() -> None:
    with pytest.raises(ValueError, match="same patient"):
        RiskPrioritizationAgent().analyze([assessment("P001", "Sepsis", 0.3), assessment("P002", "AKI", 0.3)])


def test_invalid_probability_fails() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        RiskPrioritizationAgent().analyze([assessment("P001", "Sepsis", 1.2)])


def test_invalid_disease_fails() -> None:
    with pytest.raises(ValueError, match="Unsupported disease"):
        RiskPrioritizationAgent().analyze([assessment("P001", "Other", 0.3)])


def test_empty_input_fails() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        RiskPrioritizationAgent().analyze([])


def test_reason_explains_priority_and_result_serializes() -> None:
    result = RiskPrioritizationAgent().analyze([assessment("P001", "Sepsis", 0.82)])
    assert result["reason"]
    assert result["to_dict"] if False else isinstance(result, dict)
    assert {"patient_id", "priority_level", "diseases_assessed", "highest_probability", "highest_risk_disease", "worsening_diseases", "available_trends", "reason", "status", "message"}.issubset(result)


def test_duplicate_disease_assessments_fail() -> None:
    with pytest.raises(ValueError, match="Duplicate disease"):
        RiskPrioritizationAgent().analyze([assessment("P001", "Sepsis", 0.3), assessment("P001", "Sepsis", 0.4)])


def test_mismatched_trend_patient_fails() -> None:
    with pytest.raises(ValueError, match="does not match"):
        RiskPrioritizationAgent().analyze([assessment("P001", "Sepsis", 0.3)], [trend("P002", "Sepsis", "WORSENING")])
