from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class DiseaseAssessmentAgent:
    """
    Higher-level orchestration agent for Sepsis and AKI assessment.

    The agent retrieves patient history, invokes the appropriate
    disease-assessment tools, observes their outputs, handles partial
    failures, stores the assessment, and returns a combined result.
    """

    def __init__(
        self,
        repository: Any,
        sepsis_tool: Any,
        aki_tool: Any,
    ) -> None:
        self.repository = repository
        self.sepsis_tool = sepsis_tool
        self.aki_tool = aki_tool

    def _prepare_sepsis_history(
        self,
        patient_id: str,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert repository observations into the format expected
        by the Sepsis assessment tool.
        """

        prepared: list[dict[str, Any]] = []

        for index, observation in enumerate(observations):
            clinical = observation.get("clinical_parameters", {})

            row = {
                "Patient_ID": patient_id,
                "Hour": index,
            }

            row.update(clinical)
            prepared.append(row)

        return prepared

    def _prepare_aki_data(
        self,
        patient_id: str,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Prepare the latest patient observation for the AKI tool.

        Exact feature validation is performed by the AKI model tool.
        """

        if not observations:
            return None

        latest = observations[-1]
        clinical = latest.get("clinical_parameters", {})

        return {
            "patient_reference": patient_id,
            **clinical,
        }

    def _run_sepsis(
        self,
        patient_id: str,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Execute the Sepsis assessment tool.
        """

        if not observations:
            return {
                "disease": "Sepsis",
                "status": "insufficient_data",
                "message": "No patient observations are available.",
            }

        prepared_history = self._prepare_sepsis_history(
            patient_id,
            observations,
        )

        try:
            return self.sepsis_tool.analyze(prepared_history)

        except ValueError as exc:
            # Missing observations/features are treated as insufficient data.
            return {
                "disease": "Sepsis",
                "status": "insufficient_data",
                "message": str(exc),
            }

        except Exception as exc:
            return {
                "disease": "Sepsis",
                "status": "error",
                "message": str(exc),
            }

    def _run_aki(
        self,
        patient_id: str,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Execute the AKI assessment tool.
        """

        prepared = self._prepare_aki_data(
            patient_id,
            observations,
        )

        if prepared is None:
            return {
                "disease": "AKI",
                "status": "insufficient_data",
                "message": "No patient observations are available.",
            }

        try:
            result = self.aki_tool.analyze(prepared)

            # The existing AKI tool reports missing model features
            # using status="error". Convert missing-data errors into
            # an insufficient_data state for the higher-level agent.
            if result.get("status") == "error":
                message = result.get("message", "")

                if (
                    "Missing required model features" in message
                    or "must be numeric" in message
                    or "must be finite" in message
                ):
                    return {
                        **result,
                        "status": "insufficient_data",
                    }

            return result

        except Exception as exc:
            return {
                "disease": "AKI",
                "status": "error",
                "message": str(exc),
            }

    def _reason(self, state: dict[str, Any]) -> str:
        """
        REASON:
        Determine which assessment is still pending and select
        the next action.
        """

        if state["sepsis"]["status"] == "pending":
            return "assess_sepsis"

        if state["aki"]["status"] == "pending":
            return "assess_aki"

        return "finish"

    def _observe(
        self,
        state: dict[str, Any],
        disease: str,
        result: dict[str, Any],
    ) -> None:
        """
        OBSERVE:
        Update the agent's internal state using the result returned
        by a disease-assessment tool.
        """

        state[disease] = result

    def analyze(
        self,
        patient_id: str,
        admission_id: str,
    ) -> dict[str, Any]:
        """
        Run the complete Disease Assessment Agent.

        ReAct-style cycle:

        REASON -> ACT -> OBSERVE -> REASON -> ...

        The cycle terminates after both Sepsis and AKI have reached
        a terminal state such as success, insufficient_data, or error.
        """

        # ---------------------------------------------------------
        # GET PATIENT CONTEXT
        # ---------------------------------------------------------

        history = self.repository.get_patient_history(
            patient_id,
            admission_id,
        )

        observations = history.get("observations", [])

        # ---------------------------------------------------------
        # INITIAL AGENT STATE
        # ---------------------------------------------------------

        state: dict[str, Any] = {
            "patient_id": patient_id,
            "admission_id": admission_id,
            "sepsis": {
                "status": "pending",
            },
            "aki": {
                "status": "pending",
            },
            "trace": [],
        }

        # ---------------------------------------------------------
        # REASON -> ACT -> OBSERVE LOOP
        # ---------------------------------------------------------

        while True:

            # -------------------------
            # REASON
            # -------------------------

            action = self._reason(state)

            state["trace"].append(
                {
                    "step": "reason",
                    "decision": action,
                }
            )

            # -------------------------
            # FINISH
            # -------------------------

            if action == "finish":
                break

            # -------------------------
            # ACT: SEPSIS TOOL
            # -------------------------

            if action == "assess_sepsis":

                state["trace"].append(
                    {
                        "step": "act",
                        "tool": "sepsis_assessment",
                    }
                )

                result = self._run_sepsis(
                    patient_id,
                    observations,
                )

                # -------------------------
                # OBSERVE
                # -------------------------

                self._observe(
                    state,
                    "sepsis",
                    result,
                )

                state["trace"].append(
                    {
                        "step": "observe",
                        "tool": "sepsis_assessment",
                        "status": result.get("status"),
                    }
                )

                continue

            # -------------------------
            # ACT: AKI TOOL
            # -------------------------

            if action == "assess_aki":

                state["trace"].append(
                    {
                        "step": "act",
                        "tool": "aki_assessment",
                    }
                )

                result = self._run_aki(
                    patient_id,
                    observations,
                )

                # -------------------------
                # OBSERVE
                # -------------------------

                self._observe(
                    state,
                    "aki",
                    result,
                )

                state["trace"].append(
                    {
                        "step": "observe",
                        "tool": "aki_assessment",
                        "status": result.get("status"),
                    }
                )

                continue

            # Safety guard in case _reason() ever returns
            # an unsupported action.
            raise RuntimeError(
                f"Unsupported agent action: {action}"
            )

        # ---------------------------------------------------------
        # STORE COMBINED ASSESSMENT
        # ---------------------------------------------------------

        assessment_time = datetime.now(timezone.utc)

        stored_assessment = self.repository.add_assessment(
            patient_id=patient_id,
            admission_id=admission_id,
            assessment_time=assessment_time,
            sepsis_result=state["sepsis"],
            aki_result=state["aki"],
        )

        # ---------------------------------------------------------
        # FINAL AGENT OUTPUT
        # ---------------------------------------------------------

        return {
            "agent": self.__class__.__name__,
            "patient_id": patient_id,
            "admission_id": admission_id,
            "goal": "Obtain Sepsis and AKI assessments",
            "sepsis": state["sepsis"],
            "aki": state["aki"],
            "trace": state["trace"],
            "assessment_id": stored_assessment["assessment_id"],
            "assessment_time": assessment_time,
            "status": "completed",
        }