import json
from datetime import datetime

from src.database.repositories import (
    InMemoryDatabase,
    PatientRepository,
)
from src.agents.gemini_planner import GeminiPlanner
from src.agents.llm_deterioration_agent import LLMDeteriorationAgent


def main():

    # =====================================================
    # SETUP IN-MEMORY DATABASE
    # =====================================================

    database = InMemoryDatabase()
    repository = PatientRepository(database)

    patient_id = "TEST-PATIENT-001"
    admission_id = "ADM-001"

    # =====================================================
    # CREATE PATIENT
    # =====================================================

    repository.create_patient(
        patient_id=patient_id,
        demographics={
            "name": "Test Patient",
        },
    )

    # =====================================================
    # CREATE ADMISSION
    # =====================================================

    repository.create_admission(
        patient_id=patient_id,
        admission_id=admission_id,
        admission_time=datetime(2026, 9, 13, 8, 0),
    )

    # =====================================================
    # ADD LONGITUDINAL OBSERVATIONS
    # =====================================================

    observations = [
        (
            datetime(2026, 9, 13, 8, 0),
            {
                "HR": 90,
                "MAP": 82,
                "SBP": 118,
                "O2Sat": 97,
                "Resp": 18,
                "Temp": 37.0,
                "WBC": 9.0,
                "Lactate": 1.4,
            },
        ),
        (
            datetime(2026, 9, 13, 9, 0),
            {
                "HR": 104,
                "MAP": 74,
                "SBP": 108,
                "O2Sat": 95,
                "Resp": 22,
                "Temp": 37.6,
                "WBC": 11.0,
                "Lactate": 2.1,
            },
        ),
        (
            datetime(2026, 9, 13, 10, 0),
            {
                "HR": 118,
                "MAP": 65,
                "SBP": 96,
                "O2Sat": 91,
                "Resp": 27,
                "Temp": 38.2,
                "WBC": 14.0,
                "Lactate": 3.0,
            },
        ),
    ]

    for observation_time, parameters in observations:

        repository.add_observation(
            patient_id,
            admission_id,
            observation_time,
            parameters,
            source="test",
        )

    # =====================================================
    # ADD MOCK AGENT 1 RESULTS
    #
    # These simulate Agent 1 having been executed at
    # three different times.
    # =====================================================

    assessment_values = [
        (
            datetime(2026, 9, 13, 8, 0),
            0.31,
            0.22,
        ),
        (
            datetime(2026, 9, 13, 9, 0),
            0.48,
            0.35,
        ),
        (
            datetime(2026, 9, 13, 10, 0),
            0.67,
            0.61,
        ),
    ]

    for (
        assessment_time,
        sepsis_probability,
        aki_probability,
    ) in assessment_values:

        repository.add_assessment(
            patient_id,
            admission_id,
            assessment_time,
            sepsis_result={
                "status": "success",
                "probability": sepsis_probability,
            },
            aki_result={
                "status": "success",
                "probability": aki_probability,
            },
        )

    # =====================================================
    # SHOW HISTORY BEFORE AGENT 2
    # =====================================================

    print("\n========== PATIENT HISTORY ==========")

    history = repository.get_patient_history(
        patient_id,
        admission_id,
    )

    print(
        json.dumps(
            history,
            indent=2,
            default=str,
        )
    )

    # =====================================================
    # LOAD GEMINI
    # =====================================================

    print("\nLoading Gemini planner...")

    planner = GeminiPlanner()

    # =====================================================
    # CREATE AGENT 2
    # =====================================================

    agent = LLMDeteriorationAgent(
        repository=repository,
        planner=planner,
    )

    # =====================================================
    # RUN AGENT 2
    # =====================================================

    print("\nRunning Agent 2...\n")

    result = agent.run(
        patient_id=patient_id,
        admission_id=admission_id,
    )

    # =====================================================
    # PRINT RESULT
    # =====================================================

    print("\n========== AGENT 2 RESULT ==========")

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )

    # =====================================================
    # PRINT REACT TRACE
    # =====================================================

    print("\n========== REACT TRACE ==========")

    for step in result.get("react_trace", []):
        print(step)

    # =====================================================
    # VERIFY AGENT 2 STORAGE
    # =====================================================

    print("\n========== DATABASE AFTER AGENT 2 ==========")

    updated_history = repository.get_patient_history(
        patient_id,
        admission_id,
    )

    print(
        json.dumps(
            {
                "trends": updated_history["trends"],
                "latest_prioritization":
                    updated_history["latest_prioritization"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

