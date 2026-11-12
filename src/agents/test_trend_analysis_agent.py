import pytest

from src.agents.trend_analysis_agent import TrendAnalysisAgent


def assessment(patient_id: str, disease: str, probability: float, assessment_time: int | None) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "disease": disease,
        "model": "test-model",
        "prediction": int(probability >= 0.5),
        "probability": probability,
        "threshold": 0.5,
        "risk_level": "ELEVATED" if probability >= 0.5 else "LOWER",
        "observation_count": 1,
        "assessment_time": assessment_time,
        "status": "success",
        "message": "Synthetic test assessment",
    }


def test_increasing_sepsis_probability_is_worsening() -> None:
    result = TrendAnalysisAgent().analyze([assessment("P001", "Sepsis", 0.20, 2), assessment("P001", "Sepsis", 0.40, 1), assessment("P001", "Sepsis", 0.70, 3)])
    assert result["trend"] == "WORSENING"
    assert result["first_probability"] == 0.40
    assert result["latest_probability"] == 0.70
    assert result["probability_change"] == pytest.approx(0.30)
    assert result["assessment_time_start"] == 1
    assert result["assessment_time_end"] == 3


def test_decreasing_sepsis_probability_is_improving() -> None:
    result = TrendAnalysisAgent().analyze([assessment("P001", "Sepsis", 0.80, 1), assessment("P001", "Sepsis", 0.30, 2)])
    assert result["trend"] == "IMPROVING"


def test_nearly_unchanged_probability_is_stable() -> None:
    result = TrendAnalysisAgent().analyze([assessment("P001", "Sepsis", 0.50, 1), assessment("P001", "Sepsis", 0.54, 2)])
    assert result["trend"] == "STABLE"


def test_single_assessment_is_insufficient() -> None:
    result = TrendAnalysisAgent().analyze([assessment("P001", "Sepsis", 0.50, 1)])
    assert result["trend"] == "INSUFFICIENT_DATA"
    assert result["status"] == "INSUFFICIENT_DATA"


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        TrendAnalysisAgent().analyze([])


def test_mixed_patient_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="same patient"):
        TrendAnalysisAgent().analyze([assessment("P001", "Sepsis", 0.2, 1), assessment("P002", "Sepsis", 0.4, 2)])


def test_mixed_diseases_are_rejected() -> None:
    with pytest.raises(ValueError, match="same disease"):
        TrendAnalysisAgent().analyze([assessment("P001", "Sepsis", 0.2, 1), assessment("P001", "AKI", 0.4, 2)])


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        TrendAnalysisAgent().analyze([assessment("P001", "Sepsis", 1.2, 1)])


def test_missing_times_use_supplied_sequence_order() -> None:
    result = TrendAnalysisAgent().analyze([assessment("P002", "AKI", 0.80, None), assessment("P002", "AKI", 0.30, None)])
    assert result["trend"] == "IMPROVING"
    assert result["assessment_time_start"] is None
    assert result["assessment_time_end"] is None


def test_aki_trend_is_independent_of_sepsis() -> None:
    result = TrendAnalysisAgent().analyze([assessment("P002", "AKI", 0.20, 1), assessment("P002", "AKI", 0.80, 2)])
    assert result["disease"] == "AKI"
    assert result["trend"] == "WORSENING"
