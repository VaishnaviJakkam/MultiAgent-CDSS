import json

from src.database.mongodb import MongoDatabase
from src.database.repositories import PatientRepository

from src.agents.sepsis_agent import SepsisDetectionAgent
from src.agents.aki_agent import AKIDetectionAgent
from src.agents.gemini_planner import GeminiPlanner

from src.input.input_pipeline import SepsisWorkflowPipeline


PATIENT_ID = "TEST-PATIENT-MONGO-001"
ADMISSION_ID = "ADM-MONGO-001"


def main():

    db = MongoDatabase()

    repository = PatientRepository(db)

    sepsis = SepsisDetectionAgent(model="advanced")

    aki = AKIDetectionAgent()

    planner = GeminiPlanner()

    pipeline = SepsisWorkflowPipeline(
        repository=repository,
        sepsis_agent=sepsis,
        aki_agent=aki,
        gemini_planner=planner,
    )

    result = pipeline.run(
        patient_id=PATIENT_ID,
        admission_id=ADMISSION_ID,
        report_path="data/test_reports/report2.png",
        nurse_audio_path="data/test_audio/nurse_input.wav",
    )

    print("\n========== COMPLETE PIPELINE OUTPUT ==========\n")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()