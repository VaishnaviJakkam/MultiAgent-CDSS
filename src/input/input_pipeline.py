from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.database.repositories import PatientRepository

from src.input.report_input_agent import ReportInputAgent
from src.input.nurse_audio_agent import NurseAudioAgent
from src.input.input_orchestrator import InputOrchestrator
from src.input.data_processing_agent import DataProcessingAgent

from src.agents.disease_assessment_agent import (
    DiseaseAssessmentAgent,
)
from src.agents.llm_deterioration_agent import (
    LLMDeteriorationAgent,
)


class SepsisWorkflowPipeline:
    """
    Complete longitudinal clinical workflow.

    Stage 1:
        Report -> OCR -> determine missing values

    Stage 2:
        Nurse audio -> extract structured values

    Stage 3:
        Nurse confirms/edits values
        -> merge report + nurse input
        -> store observation
        -> Agent 1
        -> Agent 2
    """

    def __init__(
        self,
        repository: PatientRepository,
        sepsis_agent: Any,
        aki_agent: Any,
        gemini_planner: Any,
    ) -> None:

        self.repository = repository

        self.report_agent = ReportInputAgent()
        self.audio_agent = NurseAudioAgent()
        self.orchestrator = InputOrchestrator()
        self.processor = DataProcessingAgent()

        self.assessment_agent = DiseaseAssessmentAgent(
            repository=repository,
            sepsis_tool=sepsis_agent,
            aki_tool=aki_agent,
        )

        self.deterioration_agent = LLMDeteriorationAgent(
            repository=repository,
            planner=gemini_planner,
        )

    # =========================================================
    # REPORT STAGE
    # =========================================================

    def process_report(
        self,
        report_path: str,
    ) -> dict[str, Any]:

        report_result = self.report_agent.analyze(
            report_path
        )

        nurse_request = (
            self.orchestrator.create_nurse_request(
                report_result
            )
        )

        return {
            "status": (
                "awaiting_nurse_input"
                if nurse_request["status"]
                == "input_required"
                else "report_complete"
            ),
            "report_result": report_result,
            "nurse_request": nurse_request,
        }

    # =========================================================
    # AUDIO STAGE
    # =========================================================

    def process_nurse_audio(
        self,
        audio_path: str,
    ) -> dict[str, Any]:

        nurse_result = self.audio_agent.analyze(
            audio_path
        )

        return {
            "status": "awaiting_confirmation",
            "nurse_result": nurse_result,
        }

    # =========================================================
    # FINAL CONFIRMATION STAGE
    # =========================================================

    def complete_observation(
        self,
        patient_id: str,
        admission_id: str,
        report_result: dict[str, Any],
        confirmed_parameters: dict[str, Any],
    ) -> dict[str, Any]:

        nurse_request = (
            self.orchestrator.create_nurse_request(
                report_result
            )
        )

        nurse_result = {
            "status": "success",
            "source": "nurse_confirmed",
            "extracted_parameters":
                confirmed_parameters,
            "metadata": {},
            "requires_confirmation": False,
        }

        validation = (
            self.orchestrator.validate_nurse_response(
                nurse_request=nurse_request,
                nurse_result=nurse_result,
            )
        )

        if not validation[
            "all_requested_values_received"
        ]:
            return {
                "status": "incomplete",
                "validation": validation,
                "nurse_request": nurse_request,
            }

        observation = self.processor.process(
            patient_id=patient_id,
            admission_id=admission_id,
            report_result=report_result,
            nurse_result=nurse_result,
            observation_time=datetime.now(
                timezone.utc
            ),
        )

        if not observation[
            "ready_for_sepsis_assessment"
        ]:
            return {
                "status": "incomplete_observation",
                "observation": observation,
            }

        stored_observation = (
            self.repository.add_observation(
                patient_id=patient_id,
                admission_id=admission_id,
                observation_time=observation[
                    "observation_time"
                ],
                clinical_parameters=observation[
                    "clinical_parameters"
                ],
                source="nurse_confirmed",
            )
        )

        assessment = (
            self.assessment_agent.analyze(
                patient_id=patient_id,
                admission_id=admission_id,
            )
        )

        deterioration = (
            self.deterioration_agent.run(
                patient_id=patient_id,
                admission_id=admission_id,
            )
        )

        return {
            "status": "completed",
            "observation": observation,
            "stored_observation":
                stored_observation,
            "assessment": assessment,
            "deterioration": deterioration,
        }