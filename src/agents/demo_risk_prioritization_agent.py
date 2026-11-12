"""Synthetic/test-data demo for RiskPrioritizationAgent."""

from pprint import pprint

from src.agents.risk_prioritization_agent import RiskPrioritizationAgent


def assessment(patient_id: str, disease: str, probability: float) -> dict[str, object]:
    return {"patient_id": patient_id, "disease": disease, "probability": probability}


def trend(patient_id: str, disease: str, value: str) -> dict[str, object]:
    return {"patient_id": patient_id, "disease": disease, "trend": value}


def run_demo() -> None:
    agent = RiskPrioritizationAgent()
    print("RISK PRIORITIZATION AGENT DEMO (synthetic/test data only)")
    print("CASE 1: P001, high Sepsis risk with worsening trend; expected HIGH")
    pprint(agent.analyze(
        [assessment("P001", "Sepsis", 0.82), assessment("P001", "AKI", 0.45)],
        [trend("P001", "Sepsis", "WORSENING"), trend("P001", "AKI", "STABLE")],
    ))
    print("CASE 2: P002, low stable Sepsis and AKI risk; expected LOW")
    pprint(agent.analyze(
        [assessment("P002", "Sepsis", 0.20), assessment("P002", "AKI", 0.25)],
        [trend("P002", "Sepsis", "STABLE"), trend("P002", "AKI", "STABLE")],
    ))


if __name__ == "__main__":
    run_demo()
