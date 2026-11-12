from datetime import datetime, timezone

from src.agents.disease_assessment_agent import DiseaseAssessmentAgent
from src.database.repositories import InMemoryDatabase, PatientRepository


class FakeSepsisTool:
    def analyze(self, patient_history):
        return {
            "disease": "Sepsis",
            "status": "success",
            "prediction": 1,
            "probability": 0.82,
            "risk_level": "ELEVATED",
        }


class FakeAKITool:
    def analyze(self, patient_data):
        return {
            "disease": "AKI",
            "status": "success",
            "prediction": 0,
            "probability": 0.21,
            "risk_level": "LOWER",
        }


def test_disease_assessment_agent():

    database = InMemoryDatabase()
    repository = PatientRepository(database)

    repository.create_patient(
        patient_id="P001",
        demographics={
            "age": 65,
        },
    )

    repository.create_admission(
        patient_id="P001",
        admission_id="A001",
        admission_time=datetime.now(timezone.utc),
    )

    repository.add_observation(
        patient_id="P001",
        admission_id="A001",
        observation_time=datetime.now(timezone.utc),
        clinical_parameters={
            "HR": 110,
            "Temp": 38.5,
            "SBP": 95,
            "MAP": 70,
            "Resp": 24,
            "O2Sat": 94,
            "WBC": 14.0,
            "Lactate": 2.8,
        },
    )

    agent = DiseaseAssessmentAgent(
        repository=repository,
        sepsis_tool=FakeSepsisTool(),
        aki_tool=FakeAKITool(),
    )

    result = agent.analyze(
        patient_id="P001",
        admission_id="A001",
    )

    # Basic agent result checks
    assert result["status"] == "completed"

    assert result["sepsis"]["status"] == "success"
    assert result["aki"]["status"] == "success"

    assert result["sepsis"]["probability"] == 0.82
    assert result["aki"]["probability"] == 0.21

    # Check ReAct-style trace
    assert len(result["trace"]) > 0

    assert result["trace"][0] == {
        "step": "reason",
        "decision": "assess_sepsis",
    }

    assert result["trace"][-1] == {
        "step": "reason",
        "decision": "finish",
    }

    # Check expected reasoning sequence
    decisions = [
        entry["decision"]
        for entry in result["trace"]
        if entry["step"] == "reason"
    ]

    assert decisions == [
        "assess_sepsis",
        "assess_aki",
        "finish",
    ]

    # Check that assessment was stored in repository
    assessments = repository.get_patient_assessments(
        "P001",
        "A001",
    )

    assert len(assessments) == 1

    stored = assessments[0]

    assert stored["sepsis"]["status"] == "success"
    assert stored["aki"]["status"] == "success"