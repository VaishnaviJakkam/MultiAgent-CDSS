"""Synthetic software demo for the advanced Sepsis LSTM agent."""

from pprint import pprint

from src.agents.sepsis_agent import SepsisDetectionAgent


def synthetic_sequence() -> list[dict[str, float | str]]:
    observations = []
    for hour in range(12):
        observations.append({
            "Patient_ID": "SYNTHETIC-TEST-001",
            "Hour": hour,
            "HR": 78.0 + hour * 0.4,
            "O2Sat": 98.0 - hour * 0.05,
            "Temp": 36.8 + hour * 0.02,
            "SBP": 122.0 - hour * 0.3,
            "MAP": 84.0 - hour * 0.2,
            "Resp": 16.0 + hour * 0.15,
            "WBC": 7.5 + hour * 0.03,
            "Lactate": 1.0 + hour * 0.01,
        })
    return list(reversed(observations))


def run_demo() -> None:
    agent = SepsisDetectionAgent()
    print("SEPSIS LSTM AGENT DEMO (synthetic/test data only)")
    result = agent.analyze(synthetic_sequence())
    pprint({key: result[key] for key in ("patient_id", "disease", "model", "prediction", "probability", "threshold", "risk_level", "observation_count", "status", "message")})


if __name__ == "__main__":
    run_demo()
