"""Demo for the SepsisDetectionAgent using synthetic data.

This demo shows two examples:
- New patient with a single observation
- Existing patient with multiple historical observations

All demonstration data is synthetic and for prototype testing only.
"""

from pprint import pprint
from pathlib import Path

from src.agents.sepsis_agent import SepsisDetectionAgent


def run_demo():
    model_path = Path("models/temporal/sepsis_temporal_pipeline.pkl")
    agent = SepsisDetectionAgent(model_path)

    print("========================================")
    print("SEPSIS DETECTION AGENT")
    print("========================================")

    # Example 1: New patient with one observation
    new_patient = [
        {
            "Patient_ID": "P1001",
            "Hour": 1,
            "HR": 80,
            "O2Sat": 98,
            "Temp": 37.0,
            "SBP": 120,
            "MAP": 85,
            "Resp": 18,
            "WBC": 8.0,
            "Lactate": 1.0,
        }
    ]

    print("\n--- New patient (single observation) ---\n")
    res1 = agent.analyze(new_patient)
    pprint(res1)

    # Example 2: Existing patient with multiple observations (out-of-order input)
    existing_patient = [
        {"Patient_ID": "P2001", "Hour": 2, "HR": 88, "O2Sat": 96, "Temp": 38.0, "SBP": 110, "MAP": 75, "Resp": 20, "WBC": 10.0, "Lactate": 1.5},
        {"Patient_ID": "P2001", "Hour": 0, "HR": 82, "O2Sat": 97, "Temp": 37.5, "SBP": 115, "MAP": 78, "Resp": 19, "WBC": 9.0, "Lactate": 1.2},
        {"Patient_ID": "P2001", "Hour": 1, "HR": 85, "O2Sat": 96, "Temp": 37.8, "SBP": 112, "MAP": 76, "Resp": 19, "WBC": 9.5, "Lactate": 1.3},
    ]

    print("\n--- Existing patient (multiple observations) ---\n")
    res2 = agent.analyze(existing_patient)
    pprint(res2)


if __name__ == "__main__":
    run_demo()
