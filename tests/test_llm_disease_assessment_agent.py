from datetime import datetime, timezone

from src.agents.llm_disease_assessment_agent import (
    LLMDiseaseAssessmentAgent,
)

from src.database.repositories import (
    InMemoryDatabase,
    PatientRepository,
)


class FakeGeminiPlanner:
    """
    Fake Gemini planner for unit testing.

    This prevents unit tests from making real API calls.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        prompt: str,
    ) -> str:

        self.call_count += 1

        if self.call_count == 1:

            return """
            {
                "action": "assess_sepsis",
                "reason": "Sepsis assessment is pending."
            }
            """

        if self.call_count == 2:

            return """
            {
                "action": "assess_aki",
                "reason": "AKI assessment is pending."
            }
            """

        return """
        {
            "action": "finish",
            "reason": "Both assessments are complete."
        }
        """


class FakeSepsisTool:

    def analyze(
        self,
        patient_history,
    ):

        return {
            "disease": "Sepsis",
            "status": "success",
            "prediction": 1,
            "probability": 0.82,
            "risk_level": "ELEVATED",
        }


class FakeAKITool:

    def analyze(
        self,
        patient_data,
    ):

        return {
            "disease": "AKI",
            "status": "success",
            "prediction": 0,
            "probability": 0.21,
            "risk_level": "LOWER",
        }


def test_llm_disease_assessment_agent():

    database = InMemoryDatabase()

    repository = PatientRepository(
        database
    )

    repository.create_patient(
        patient_id="P001",
        demographics={
            "age": 65,
        },
    )

    repository.create_admission(
        patient_id="P001",
        admission_id="A001",
        admission_time=datetime.now(
            timezone.utc
        ),
    )

    repository.add_observation(
        patient_id="P001",
        admission_id="A001",
        observation_time=datetime.now(
            timezone.utc
        ),
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

    agent = LLMDiseaseAssessmentAgent(
        repository=repository,
        sepsis_tool=FakeSepsisTool(),
        aki_tool=FakeAKITool(),
        llm=FakeGeminiPlanner(),
    )

    result = agent.analyze(
        patient_id="P001",
        admission_id="A001",
    )

    assert result["status"] == "completed"

    assert (
        result["sepsis"]["status"]
        == "success"
    )

    assert (
        result["aki"]["status"]
        == "success"
    )

    assert (
        result["sepsis"]["probability"]
        == 0.82
    )

    assert (
        result["aki"]["probability"]
        == 0.21
    )

    reason_steps = [
        entry
        for entry in result["trace"]
        if entry["step"] == "reason"
    ]

    actions = [
        entry["validated_action"]
        for entry in reason_steps
    ]

    assert actions == [
        "assess_sepsis",
        "assess_aki",
        "finish",
    ]

    assert all(
        entry["planner"] == "gemini"
        for entry in reason_steps
    )

    stored_assessments = (
        repository.get_patient_assessments(
            "P001",
            "A001",
        )
    )

    assert (
        len(stored_assessments)
        == 1
    )

    stored = (
        stored_assessments[0]
    )

    assert (
        stored["sepsis"]["status"]
        == "success"
    )

    assert (
        stored["aki"]["status"]
        == "success"
    )