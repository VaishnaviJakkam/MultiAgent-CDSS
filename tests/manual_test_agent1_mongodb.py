import json

from src.database.mongodb import MongoDatabase
from src.database.repositories import PatientRepository

from src.agents.sepsis_agent import SepsisDetectionAgent
from src.agents.aki_agent import AKIDetectionAgent
from src.agents.disease_assessment_agent import DiseaseAssessmentAgent


def main():

    print("\nConnecting to MongoDB...")

    database = MongoDatabase()

    try:
        if database.ping():
            print("MongoDB connected successfully.")

        repository = PatientRepository(database)

        patient_id = "TEST-PATIENT-MONGO-001"
        admission_id = "ADM-MONGO-001"

        # -----------------------------------------
        # LOAD ML TOOLS
        # -----------------------------------------

        print("\nLoading Sepsis model...")

        sepsis_tool = SepsisDetectionAgent(
            model="advanced"
        )

        print("Loading AKI model...")

        aki_tool = AKIDetectionAgent()

        # -----------------------------------------
        # CREATE AGENT 1
        # -----------------------------------------

        agent = DiseaseAssessmentAgent(
            repository=repository,
            sepsis_tool=sepsis_tool,
            aki_tool=aki_tool,
        )

        # -----------------------------------------
        # SHOW HISTORY COUNT
        # -----------------------------------------

        history = repository.get_patient_history(
            patient_id,
            admission_id,
        )

        print(
            "\nNumber of observations in MongoDB:",
            len(history["observations"]),
        )

        # -----------------------------------------
        # RUN AGENT 1
        # -----------------------------------------

        print("\nRunning Disease Assessment Agent...")

        result = agent.analyze(
            patient_id=patient_id,
            admission_id=admission_id,
        )

        print(
            "\n========== AGENT 1 RESULT =========="
        )

        print(
            json.dumps(
                result,
                indent=2,
                default=str,
            )
        )

        # -----------------------------------------
        # VERIFY DATABASE
        # -----------------------------------------

        updated_history = (
            repository.get_patient_history(
                patient_id,
                admission_id,
            )
        )

        print(
            "\n========== STORED ASSESSMENTS =========="
        )

        print(
            json.dumps(
                updated_history["assessments"],
                indent=2,
                default=str,
            )
        )

        print(
            "\nSUCCESS: Agent 1 ran using MongoDB history."
        )

    finally:
        database.close()


if __name__ == "__main__":
    main()