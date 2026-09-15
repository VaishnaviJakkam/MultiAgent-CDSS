from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.agents.gemini_planner import GeminiPlanner
from src.agents.trend_analysis_agent import TrendAnalysisAgent
from src.agents.risk_prioritization_agent import RiskPrioritizationAgent


class LLMDeteriorationAgent:
    """
    Agent 2:
    Longitudinal Deterioration & Patient Prioritization Agent.

    Gemini controls tool orchestration.

    Deterministic tools perform:
    - trend analysis
    - deterioration detection
    - patient prioritization

    Results are stored through PatientRepository.
    """

    def __init__(
        self,
        repository,
        planner: GeminiPlanner,
        max_cycles: int = 6,
    ):
        self.repository = repository
        self.planner = planner
        self.max_cycles = max_cycles

        self.trend_tool = TrendAnalysisAgent()
        self.priority_tool = RiskPrioritizationAgent()

    # =========================================================
    # STORE DISEASE TRENDS
    # =========================================================

    def _store_trends(
        self,
        patient_id: str,
        admission_id: str,
        analysis_time: datetime,
        trend_result: dict[str, Any],
    ) -> None:

        disease_trends = trend_result.get(
            "disease_trends",
            {},
        )

        for disease in ("Sepsis", "AKI"):

            disease_result = disease_trends.get(disease)

            if not disease_result:
                continue

            # Cannot store a meaningful disease trend
            # without at least two probability values.
            if disease_result.get("trend") == "INSUFFICIENT_DATA":
                continue

            first_probability = disease_result.get("first")
            latest_probability = disease_result.get("latest")
            probability_change = disease_result.get("change")

            if (
                first_probability is None
                or latest_probability is None
                or probability_change is None
            ):
                continue

            self.repository.add_trend(
                patient_id=patient_id,
                admission_id=admission_id,
                disease=disease,
                assessment_time=analysis_time,
                trend=disease_result["trend"],
                first_probability=float(first_probability),
                latest_probability=float(latest_probability),
                probability_change=float(probability_change),
                observation_count=int(
                    disease_result.get("count", 0)
                ),
            )

    # =========================================================
    # STORE PRIORITIZATION
    # =========================================================

    def _store_prioritization(
        self,
        patient_id: str,
        admission_id: str,
        analysis_time: datetime,
        trend_result: dict[str, Any],
        priority_result: dict[str, Any],
    ) -> None:

        disease_trends = trend_result.get(
            "disease_trends",
            {},
        )

        # -----------------------------------------------------
        # Find disease with highest latest probability
        # -----------------------------------------------------

        probabilities = {}

        for disease in ("Sepsis", "AKI"):

            latest = (
                disease_trends
                .get(disease, {})
                .get("latest")
            )

            if latest is not None:
                probabilities[disease] = float(latest)

        if probabilities:

            highest_risk_disease = max(
                probabilities,
                key=probabilities.get,
            )

            highest_probability = probabilities[
                highest_risk_disease
            ]

        else:

            highest_risk_disease = "Unknown"
            highest_probability = 0.0

        # -----------------------------------------------------
        # Find diseases whose risk is worsening
        # -----------------------------------------------------

        worsening_diseases = []

        for disease in ("Sepsis", "AKI"):

            if (
                disease_trends
                .get(disease, {})
                .get("trend")
                == "WORSENING"
            ):
                worsening_diseases.append(disease)

        # -----------------------------------------------------
        # Convert reasons list into repository reason string
        # -----------------------------------------------------

        reasons = priority_result.get(
            "reasons",
            [],
        )

        reason = "; ".join(reasons)

        if not reason:
            reason = "Priority calculated from longitudinal assessment."

        # -----------------------------------------------------
        # Store
        # -----------------------------------------------------

        self.repository.add_prioritization(
            patient_id=patient_id,
            admission_id=admission_id,
            priority_level=priority_result["priority"],
            highest_risk_disease=highest_risk_disease,
            highest_probability=highest_probability,
            worsening_diseases=worsening_diseases,
            reason=reason,
            assessment_time=analysis_time,
        )

    # =========================================================
    # MAIN AGENT
    # =========================================================

    def run(
        self,
        patient_id: str,
        admission_id: str,
    ) -> dict[str, Any]:

        state = {
            "history_loaded": False,
            "trend_completed": False,
            "priority_completed": False,
        }

        history = None
        trend_result = None
        priority_result = None

        trace = []

        analysis_time = datetime.now(timezone.utc)

        # =====================================================
        # REACT LOOP
        # =====================================================

        for cycle in range(
            1,
            self.max_cycles + 1,
        ):

            prompt = f"""
You are the controller for a longitudinal clinical
deterioration and patient prioritization workflow.

Patient ID: {patient_id}
Admission ID: {admission_id}

Current workflow state:
{state}

Available actions:

1. get_history
   Load longitudinal patient observations and
   previous Sepsis and AKI assessments.

2. analyze_trends
   Use the deterministic trend-analysis tool to
   analyze disease risk and clinical changes.

3. calculate_priority
   Use the deterministic prioritization tool to
   calculate patient priority.

4. finish
   Finish only after trend analysis and
   prioritization have completed.

Rules:

- Do not calculate disease probabilities yourself.
- Do not invent patient values.
- Do not calculate clinical trends yourself.
- Do not calculate priority scores yourself.
- Use the available tools.
- Follow the workflow in logical order.
- Return exactly one next action.
"""

            decision = self.planner.decide(
                prompt=prompt,
                allowed_actions=[
                    "get_history",
                    "analyze_trends",
                    "calculate_priority",
                    "finish",
                ],
            )

            requested_action = decision["action"]
            action = requested_action

            # =================================================
            # GUARDRAILS
            # =================================================

            if (
                action == "analyze_trends"
                and not state["history_loaded"]
            ):
                action = "get_history"

            elif (
                action == "calculate_priority"
                and not state["trend_completed"]
            ):

                if not state["history_loaded"]:
                    action = "get_history"
                else:
                    action = "analyze_trends"

            elif (
                action == "finish"
                and not state["priority_completed"]
            ):

                if not state["history_loaded"]:
                    action = "get_history"

                elif not state["trend_completed"]:
                    action = "analyze_trends"

                else:
                    action = "calculate_priority"

            # =================================================
            # TOOL 1 — GET HISTORY
            # =================================================

            if action == "get_history":

                history = (
                    self.repository.get_patient_history(
                        patient_id,
                        admission_id,
                    )
                )

                state["history_loaded"] = True

                trace.append({
                    "cycle": cycle,
                    "requested_action": requested_action,
                    "action": action,
                    "status": "success",
                })

            # =================================================
            # TOOL 2 — ANALYZE TRENDS
            # =================================================

            elif action == "analyze_trends":

                trend_result = (
                    self.trend_tool.analyze(
                        history
                    )
                )

                state["trend_completed"] = True

                # Store Sepsis + AKI trends
                if trend_result.get("status") == "success":

                    self._store_trends(
                        patient_id=patient_id,
                        admission_id=admission_id,
                        analysis_time=analysis_time,
                        trend_result=trend_result,
                    )

                trace.append({
                    "cycle": cycle,
                    "requested_action": requested_action,
                    "action": action,
                    "status": trend_result.get("status"),
                })

            # =================================================
            # TOOL 3 — CALCULATE PRIORITY
            # =================================================

            elif action == "calculate_priority":

                priority_result = (
                    self.priority_tool.prioritize(
                        trend_result
                    )
                )

                state["priority_completed"] = True

                # Store prioritization
                if priority_result.get("status") == "success":

                    self._store_prioritization(
                        patient_id=patient_id,
                        admission_id=admission_id,
                        analysis_time=analysis_time,
                        trend_result=trend_result,
                        priority_result=priority_result,
                    )

                trace.append({
                    "cycle": cycle,
                    "requested_action": requested_action,
                    "action": action,
                    "status": priority_result.get("status"),
                })

            # =================================================
            # FINISH
            # =================================================

            elif action == "finish":

                trace.append({
                    "cycle": cycle,
                    "requested_action": requested_action,
                    "action": "finish",
                    "status": "completed",
                })

                break

        # =====================================================
        # FINAL RESULT
        # =====================================================

        completed = (
            state["trend_completed"]
            and state["priority_completed"]
        )

        return {
            "patient_id": patient_id,
            "admission_id": admission_id,
            "analysis_time": analysis_time,
            "status": (
                "completed"
                if completed
                else "incomplete"
            ),
            "trend": trend_result,
            "prioritization": priority_result,
            "react_trace": trace,
        }