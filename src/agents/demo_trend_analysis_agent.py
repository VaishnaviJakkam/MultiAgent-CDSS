"""Synthetic/test-data demo for TrendAnalysisAgent."""

from pprint import pprint

from src.agents.trend_analysis_agent import TrendAnalysisAgent


def result(patient_id: str, disease: str, probability: float, assessment_time: int | None) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "disease": disease,
        "model": "synthetic-test-model",
        "prediction": int(probability >= 0.5),
        "probability": probability,
        "threshold": 0.5,
        "risk_level": "ELEVATED" if probability >= 0.5 else "LOWER",
        "observation_count": 1,
        "assessment_time": assessment_time,
        "status": "success",
        "message": "Synthetic demonstration data",
    }


def run_demo() -> None:
    agent = TrendAnalysisAgent()
    print("TREND ANALYSIS AGENT DEMO (synthetic/test data only)")
    print("Patient P001, Sepsis: expected WORSENING")
    pprint(agent.analyze([result("P001", "Sepsis", 0.25, 1), result("P001", "Sepsis", 0.72, 2)]))
    print("Patient P002, AKI: expected IMPROVING")
    pprint(agent.analyze([result("P002", "AKI", 0.78, None), result("P002", "AKI", 0.31, None)]))


if __name__ == "__main__":
    run_demo()
