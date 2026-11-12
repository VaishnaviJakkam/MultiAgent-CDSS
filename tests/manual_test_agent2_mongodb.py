import json

from src.database.mongodb import MongoDatabase
from src.database.repositories import PatientRepository

from src.agents.gemini_planner import GeminiPlanner
from src.agents.llm_deterioration_agent import LLMDeteriorationAgent


def print_json(data):
    print(
        json.dumps(
            data,
            indent=2,
            default=str,
        )
    )


def main():

    print("\nConnecting to MongoDB...")

    database = MongoDatabase()

    try:
        # ---------------------------------------------------------
        # VERIFY DATABASE CONNECTION
        # ---------------------------------------------------------

        if database.ping():
            print("MongoDB connected successfully.")

        repository = PatientRepository(database)

        patient_id = "TEST-PATIENT-MONGO-001"
        admission_id = "ADM-MONGO-001"

        # ---------------------------------------------------------
        # SHOW CURRENT HISTORY
        # ---------------------------------------------------------

        history_before = repository.get_patient_history(
            patient_id,
            admission_id,
        )

        print("\n========== HISTORY BEFORE AGENT 2 ==========")

        print(
            "Observations:",
            len(history_before.get("observations", [])),
        )

        print(
            "Assessments:",
            len(history_before.get("assessments", [])),
        )

        print(
            "Trends:",
            len(history_before.get("trends", [])),
        )

        print(
            "Latest prioritization:",
            history_before.get("latest_prioritization"),
        )

        # ---------------------------------------------------------
        # CREATE GEMINI PLANNER
        # ---------------------------------------------------------

        print("\nCreating Gemini planner...")

        planner = GeminiPlanner()

        # ---------------------------------------------------------
        # CREATE AGENT 2
        # ---------------------------------------------------------

        agent = LLMDeteriorationAgent(
            repository=repository,
            planner=planner,
        )

        # ---------------------------------------------------------
        # RUN AGENT 2
        # ---------------------------------------------------------

        print("\nRunning Agent 2...")

        result = agent.run(
            patient_id=patient_id,
            admission_id=admission_id,
        )

        # ---------------------------------------------------------
        # SHOW AGENT RESULT
        # ---------------------------------------------------------

        print("\n========== AGENT 2 RESULT ==========")

        print_json(result)

        # ---------------------------------------------------------
        # VERIFY DATABASE AFTER AGENT 2
        # ---------------------------------------------------------

        history_after = repository.get_patient_history(
            patient_id,
            admission_id,
        )

        print("\n========== DATABASE AFTER AGENT 2 ==========")

        print("\nStored trends:")

        print_json(
            history_after.get(
                "trends",
                [],
            )
        )

        print("\nLatest prioritization:")

        print_json(
            history_after.get(
                "latest_prioritization",
            )
        )

        # ---------------------------------------------------------
        # TRACE
        # ---------------------------------------------------------

        print("\n========== REACT TRACE ==========")

        print_json(
            result.get(
                "react_trace",
                [],
            )
        )

        print(
            "\nSUCCESS: Agent 2 ran using MongoDB history."
        )

    finally:
        database.close()


if __name__ == "__main__":
    main()