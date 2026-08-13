"""Demo for the AKIDetectionAgent using synthetic patient data."""

from pprint import pprint
from pathlib import Path

from src.agents.aki_agent import AKIDetectionAgent


def run_demo() -> None:
    model_path = Path("models/aki/aki_pipeline.pkl")
    agent = AKIDetectionAgent(model_path)

    print("========================================")
    print("        AKI DETECTION AGENT")
    print("========================================")

    patient_report = {
        "patient_reference": "P1001",
        "ANIONGAP_min": 8.0,
        "ANIONGAP_max": 12.0,
        "ALBUMIN_min": 3.8,
        "ALBUMIN_max": 4.2,
        "BANDS_min": 0.1,
        "BANDS_max": 0.3,
        "BICARBONATE_min": 22.0,
        "BICARBONATE_max": 24.0,
        "BILIRUBIN_min": 0.5,
        "BILIRUBIN_max": 0.8,
        "CHLORIDE_min": 100.0,
        "CHLORIDE_max": 104.0,
        "GLUCOSE_min": 90.0,
        "GLUCOSE_max": 110.0,
        "HEMATOCRIT_min": 38.0,
        "HEMATOCRIT_max": 42.0,
        "HEMOGLOBIN_min": 12.5,
        "HEMOGLOBIN_max": 13.5,
        "LACTATE_min": 1.0,
        "LACTATE_max": 1.3,
        "PLATELET_min": 180.0,
        "PLATELET_max": 220.0,
        "POTASSIUM_min": 3.8,
        "POTASSIUM_max": 4.2,
        "PTT_min": 28.0,
        "PTT_max": 32.0,
        "INR_min": 1.0,
        "INR_max": 1.1,
        "PT_min": 12.0,
        "PT_max": 14.0,
        "SODIUM_min": 136.0,
        "SODIUM_max": 140.0,
        "BUN_min": 12.0,
        "BUN_max": 16.0,
        "WBC_min": 6.0,
        "WBC_max": 8.5,
        "HeartRate_Min": 72.0,
        "HeartRate_Max": 88.0,
        "HeartRate_Mean": 80.0,
        "SysBP_Min": 110.0,
        "SysBP_Max": 130.0,
        "SysBP_Mean": 120.0,
        "DiasBP_Min": 65.0,
        "DiasBP_Max": 80.0,
        "DiasBP_Mean": 72.0,
        "MeanBP_Min": 80.0,
        "MeanBP_Max": 95.0,
        "MeanBP_Mean": 88.0,
        "RespRate_Min": 14.0,
        "RespRate_Max": 18.0,
        "RespRate_Mean": 16.0,
        "TempC_Min": 36.6,
        "TempC_Max": 37.2,
        "TempC_Mean": 36.9,
        "SpO2_Min": 95.0,
        "SpO2_Max": 99.0,
        "SpO2_Mean": 97.0,
        "Glucose_Min_1": 92.0,
        "Glucose_Max_1": 108.0,
        "Glucose_Mean": 100.0,
    }

    print("\nPatient Reference: P1001")
    print("\nInput:")
    print("Structured patient report (synthetic)")
    print("----------------------------------------")
    pprint(patient_report)

    print("\n----------------------------------------")
    print("AKI RISK RESULT")
    print("----------------------------------------")
    result = agent.analyze(patient_report)
    pprint(result)
    print("----------------------------------------")


if __name__ == "__main__":
    run_demo()
