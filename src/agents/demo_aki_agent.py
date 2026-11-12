"""Synthetic software demo for the advanced AKI XGBoost agent."""

from pprint import pprint

from src.agents.aki_agent import AKIDetectionAgent
from src.aki.preprocessing import get_strict_feature_columns, load_raw_dataset


def synthetic_patient_data() -> dict[str, float | str]:
    data: dict[str, float | str] = {"patient_reference": "SYNTHETIC-TEST-001"}
    for index, feature in enumerate(get_strict_feature_columns(load_raw_dataset())):
        data[feature] = float(index + 1)
    return data


def run_demo() -> None:
    agent = AKIDetectionAgent()
    print("AKI XGBOOST AGENT DEMO (synthetic/test data only)")
    result = agent.analyze(synthetic_patient_data())
    pprint({key: result[key] for key in ("patient_id", "disease", "model", "prediction", "probability", "threshold", "risk_level", "observation_count", "status", "message")})


if __name__ == "__main__":
    run_demo()
