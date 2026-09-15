from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol


class LLMPlanner(Protocol):

    def generate(
        self,
        prompt: str,
    ) -> str:
        ...


class LLMDiseaseAssessmentAgent:
    """
    Gemini-guided ReAct-style orchestration agent.

    Gemini performs workflow reasoning and tool selection.

    Disease-specific ML models remain responsible for
    generating numerical Sepsis and AKI predictions.
    """

    ALLOWED_ACTIONS = {
        "assess_sepsis",
        "assess_aki",
        "finish",
    }

    TERMINAL_STATUSES = {
        "success",
        "insufficient_data",
        "error",
    }

    def __init__(
        self,
        repository: Any,
        sepsis_tool: Any,
        aki_tool: Any,
        llm: LLMPlanner,
        max_steps: int = 6,
    ) -> None:

        self.repository = repository
        self.sepsis_tool = sepsis_tool
        self.aki_tool = aki_tool
        self.llm = llm
        self.max_steps = max_steps

    # =========================================================
    # INPUT PREPARATION
    # =========================================================

    def _prepare_sepsis_history(
        self,
        patient_id: str,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        prepared: list[dict[str, Any]] = []

        for index, observation in enumerate(
            observations
        ):

            clinical = observation.get(
                "clinical_parameters",
                {},
            )

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

        if not observations:
            return None

        latest = observations[-1]

        clinical = latest.get(
            "clinical_parameters",
            {},
        )

        return {
            "patient_reference": patient_id,
            **clinical,
        }

    # =========================================================
    # SEPSIS TOOL
    # =========================================================

    def _run_sepsis_tool(
        self,
        patient_id: str,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:

        if not observations:

            return {
                "disease": "Sepsis",
                "status": "insufficient_data",
                "message": (
                    "No patient observations are available."
                ),
            }

        prepared_history = (
            self._prepare_sepsis_history(
                patient_id,
                observations,
            )
        )

        try:

            result = self.sepsis_tool.analyze(
                prepared_history
            )

            return result

        except ValueError as exc:

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

    # =========================================================
    # AKI TOOL
    # =========================================================

    def _run_aki_tool(
        self,
        patient_id: str,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:

        prepared = self._prepare_aki_data(
            patient_id,
            observations,
        )

        if prepared is None:

            return {
                "disease": "AKI",
                "status": "insufficient_data",
                "message": (
                    "No patient observations are available."
                ),
            }

        try:

            result = self.aki_tool.analyze(
                prepared
            )

            if result.get("status") == "error":

                message = result.get(
                    "message",
                    "",
                )

                missing_data_messages = (
                    "Missing required model features",
                    "must be numeric",
                    "must be finite",
                )

                if any(
                    text in message
                    for text in missing_data_messages
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

    # =========================================================
    # STATE HELPERS
    # =========================================================

    def _is_terminal(
        self,
        disease_state: dict[str, Any],
    ) -> bool:

        return (
            disease_state.get("status")
            in self.TERMINAL_STATUSES
        )

    def _goal_complete(
        self,
        state: dict[str, Any],
    ) -> bool:

        return (
            self._is_terminal(
                state["sepsis"]
            )
            and self._is_terminal(
                state["aki"]
            )
        )

    # =========================================================
    # GEMINI PROMPT
    # =========================================================

    def _build_prompt(
        self,
        state: dict[str, Any],
        observation_count: int,
    ) -> str:

        return f"""
You are the orchestration controller for an academic
clinical decision-support prototype.

Your goal is to obtain terminal assessments for BOTH:
- Sepsis
- AKI

You must choose exactly ONE next action.

AVAILABLE ACTIONS:

assess_sepsis
Runs the trained Sepsis machine-learning model.

assess_aki
Runs the trained AKI machine-learning model.

finish
Ends the workflow.

CURRENT STATE:

Number of patient observations:
{observation_count}

Sepsis status:
{state["sepsis"].get("status")}

AKI status:
{state["aki"].get("status")}

TERMINAL STATUSES:

success
insufficient_data
error

RULES:

1. Never calculate a clinical probability yourself.
2. Never invent patient values.
3. Never infer missing laboratory values.
4. Do not reassess a disease that already has a
   terminal status.
5. If Sepsis is pending, assess_sepsis may be used.
6. If AKI is pending, assess_aki may be used.
7. Use finish only when both diseases have reached
   terminal states.
8. Numerical predictions must come only from the
   registered ML tools.

Return the next action and a short reason.
""".strip()

    # =========================================================
    # GEMINI REASONING
    # =========================================================

    def _ask_llm(
        self,
        state: dict[str, Any],
        observation_count: int,
    ) -> dict[str, str]:

        prompt = self._build_prompt(
            state,
            observation_count,
        )

        raw_response = self.llm.generate(
            prompt
        )

        try:

            response = json.loads(
                raw_response
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "LLM returned invalid JSON"
            ) from exc

        action = response.get(
            "action"
        )

        reason = response.get(
            "reason",
            "",
        )

        if action not in self.ALLOWED_ACTIONS:

            raise ValueError(
                f"Unsupported LLM action: {action}"
            )

        return {
            "action": action,
            "reason": str(reason),
        }

    # =========================================================
    # DETERMINISTIC GUARDRAILS
    # =========================================================

    def _fallback_action(
        self,
        state: dict[str, Any],
    ) -> str:

        if not self._is_terminal(
            state["sepsis"]
        ):
            return "assess_sepsis"

        if not self._is_terminal(
            state["aki"]
        ):
            return "assess_aki"

        return "finish"

    def _validate_action(
        self,
        action: str,
        state: dict[str, Any],
    ) -> str:

        if action == "assess_sepsis":

            if self._is_terminal(
                state["sepsis"]
            ):

                return self._fallback_action(
                    state
                )

            return action

        if action == "assess_aki":

            if self._is_terminal(
                state["aki"]
            ):

                return self._fallback_action(
                    state
                )

            return action

        if action == "finish":

            if not self._goal_complete(
                state
            ):

                return self._fallback_action(
                    state
                )

            return "finish"

        return self._fallback_action(
            state
        )

    # =========================================================
    # MAIN REACT LOOP
    # =========================================================

    def analyze(
        self,
        patient_id: str,
        admission_id: str,
    ) -> dict[str, Any]:

        history = (
            self.repository.get_patient_history(
                patient_id,
                admission_id,
            )
        )

        observations = history.get(
            "observations",
            [],
        )

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

        for cycle in range(
            1,
            self.max_steps + 1,
        ):

            # =================================================
            # REASON
            # =================================================

            try:

                decision = self._ask_llm(
                    state,
                    len(observations),
                )

                requested_action = (
                    decision["action"]
                )

                reason = (
                    decision["reason"]
                )

                planner = "gemini"

            except Exception as exc:

                requested_action = (
                    self._fallback_action(
                        state
                    )
                )

                reason = (
                    "Gemini planner failed. "
                    "Deterministic fallback selected. "
                    f"{exc}"
                )

                planner = (
                    "deterministic_fallback"
                )

            action = self._validate_action(
                requested_action,
                state,
            )

            state["trace"].append(
                {
                    "cycle": cycle,
                    "step": "reason",
                    "planner": planner,
                    "requested_action": (
                        requested_action
                    ),
                    "validated_action": action,
                    "reason": reason,
                }
            )

            # =================================================
            # FINISH
            # =================================================

            if action == "finish":

                state["trace"].append(
                    {
                        "cycle": cycle,
                        "step": "finish",
                        "message": (
                            "Sepsis and AKI assessments "
                            "have reached terminal states."
                        ),
                    }
                )

                break

            # =================================================
            # ACT -> SEPSIS
            # =================================================

            if action == "assess_sepsis":

                state["trace"].append(
                    {
                        "cycle": cycle,
                        "step": "act",
                        "tool": (
                            "sepsis_model"
                        ),
                    }
                )

                result = (
                    self._run_sepsis_tool(
                        patient_id,
                        observations,
                    )
                )

                # =============================================
                # OBSERVE
                # =============================================

                state["sepsis"] = result

                state["trace"].append(
                    {
                        "cycle": cycle,
                        "step": "observe",
                        "tool": (
                            "sepsis_model"
                        ),
                        "status": (
                            result.get(
                                "status"
                            )
                        ),
                    }
                )

                continue

            # =================================================
            # ACT -> AKI
            # =================================================

            if action == "assess_aki":

                state["trace"].append(
                    {
                        "cycle": cycle,
                        "step": "act",
                        "tool": (
                            "aki_model"
                        ),
                    }
                )

                result = (
                    self._run_aki_tool(
                        patient_id,
                        observations,
                    )
                )

                # =============================================
                # OBSERVE
                # =============================================

                state["aki"] = result

                state["trace"].append(
                    {
                        "cycle": cycle,
                        "step": "observe",
                        "tool": (
                            "aki_model"
                        ),
                        "status": (
                            result.get(
                                "status"
                            )
                        ),
                    }
                )

                continue

        else:

            state["trace"].append(
                {
                    "step": "stop",
                    "reason": (
                        "Maximum number of "
                        "agent cycles reached."
                    ),
                }
            )

        assessment_time = datetime.now(
            timezone.utc
        )

        stored_assessment = (
            self.repository.add_assessment(
                patient_id=patient_id,
                admission_id=admission_id,
                assessment_time=assessment_time,
                sepsis_result=state["sepsis"],
                aki_result=state["aki"],
            )
        )

        return {

            "agent": self.__class__.__name__,

            "architecture": (
                "Gemini-guided ReAct-style "
                "tool orchestration"
            ),

            "patient_id": patient_id,

            "admission_id": admission_id,

            "goal": (
                "Obtain Sepsis and AKI "
                "assessments"
            ),

            "sepsis": state["sepsis"],

            "aki": state["aki"],

            "trace": state["trace"],

            "assessment_id": (
                stored_assessment[
                    "assessment_id"
                ]
            ),

            "assessment_time": (
                assessment_time
            ),

            "status": (
                "completed"
                if self._goal_complete(
                    state
                )
                else "incomplete"
            ),
        }