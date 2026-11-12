from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from typing import Any

from src.agents.result_schema import PatientAssessmentResult
from src.database.connection import get_client
from src.database.repositories import PatientRepository

PATIENT_ID = "DEMO-P001"
ADMISSION_ID = "DEMO-A001"
DEMO_SOURCE = "synthetic_initialization_demo"
DEMO_ASSESSMENT_MESSAGE = "Synthetic database initialization assessment"


def _synthetic_assessment() -> dict[str, Any]:
    return PatientAssessmentResult(
        patient_id=PATIENT_ID,
        disease="Sepsis",
        model="synthetic-demo",
        prediction=0,
        probability=0.21,
        threshold=0.60,
        risk_level="LOWER",
        observation_count=1,
        assessment_time=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        status="success",
        message=DEMO_ASSESSMENT_MESSAGE,
    ).to_dict()


def initialize() -> dict[str, int | str | bool]:
    client = None
    try:
        client = get_client()
        client.admin.command("ping")
        database = client[os.getenv("MONGODB_DATABASE", "multiagent_cdss")]
        repository = PatientRepository(database)

        admission_time = datetime(2026, 1, 1, 8, tzinfo=timezone.utc)
        first_observation_time = admission_time
        second_observation_time = admission_time + timedelta(hours=4)
        assessment_time = admission_time + timedelta(hours=8)

        if repository.get_patient(PATIENT_ID) is None:
            repository.create_patient(
                PATIENT_ID,
                demographics={"synthetic": True, "display_name": "Synthetic demonstration patient"},
                status="demo",
            )

        if repository.get_admission(PATIENT_ID, ADMISSION_ID) is None:
            repository.create_admission(PATIENT_ID, ADMISSION_ID, admission_time, status="demo")

        observations = repository.get_patient_observations(PATIENT_ID, ADMISSION_ID)
        if not any(item.get("source") == DEMO_SOURCE and item.get("clinical_parameters", {}).get("demo_observation") == 1 for item in observations):
            repository.add_observation(PATIENT_ID, ADMISSION_ID, first_observation_time, {"demo_observation": 1, "HR": 78.0, "Temp": 36.8, "synthetic": True}, DEMO_SOURCE)
        if not any(item.get("source") == DEMO_SOURCE and item.get("clinical_parameters", {}).get("demo_observation") == 2 for item in observations):
            repository.add_observation(PATIENT_ID, ADMISSION_ID, second_observation_time, {"demo_observation": 2, "HR": 82.0, "Temp": 37.1, "synthetic": True}, DEMO_SOURCE)

        assessments = repository.get_patient_assessments(PATIENT_ID, ADMISSION_ID)
        if not any(item.get("sepsis", {}).get("message") == DEMO_ASSESSMENT_MESSAGE for item in assessments):
            repository.add_assessment(PATIENT_ID, ADMISSION_ID, assessment_time, sepsis_result=_synthetic_assessment())

        history = repository.get_patient_history(PATIENT_ID, ADMISSION_ID)
        scoped = all(
            record.get("patient_id") == PATIENT_ID and record.get("admission_id") == ADMISSION_ID
            for collection in ("observations", "assessments", "trends")
            for record in history[collection]
        )
        if history["patient"] is None or history["admission"] is None or not scoped:
            raise RuntimeError("Synthetic patient history verification failed")
        return {
            "connection_successful": True,
            "patient_id": PATIENT_ID,
            "admission_id": ADMISSION_ID,
            "observation_count": len(history["observations"]),
            "assessment_count": len(history["assessments"]),
            "retrieval_successful": True,
        }
    finally:
        if client is not None:
            client.close()


def main() -> int:
    try:
        result = initialize()
    except Exception:
        print("Database initialization failed.")
        return 1
    print("Connection successful.")
    print(f"Patient ID: {result['patient_id']}")
    print(f"Admission ID: {result['admission_id']}")
    print(f"Observation count: {result['observation_count']}")
    print(f"Assessment count: {result['assessment_count']}")
    print("Retrieval successful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
