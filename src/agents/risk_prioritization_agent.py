from __future__ import annotations

from typing import Any


class RiskPrioritizationAgent:
    """
    Deterministic patient-prioritization tool.

    Uses:
    - latest Sepsis probability
    - latest AKI probability
    - longitudinal deterioration signals

    Gemini does NOT calculate the priority score.
    """

    def prioritize(
        self,
        trend_result: dict[str, Any],
    ) -> dict[str, Any]:

        if trend_result.get("status") != "success":
            return {
                "status": "insufficient_data",
                "priority": None,
                "priority_score": None,
                "reasons": [
                    "Insufficient longitudinal data"
                ],
            }

        score = 0
        reasons = []

        disease_trends = trend_result.get(
            "disease_trends", {}
        )

        # -----------------------------------------
        # Current disease risk
        # -----------------------------------------

        for disease in ("Sepsis", "AKI"):

            result = disease_trends.get(disease, {})
            probability = result.get("latest")

            if probability is None:
                continue

            if probability >= 0.60:
                score += 3
                reasons.append(
                    f"{disease} probability elevated "
                    f"({probability:.2f})"
                )

            elif probability >= 0.40:
                score += 2
                reasons.append(
                    f"{disease} probability moderate "
                    f"({probability:.2f})"
                )

            # Extra weight if disease risk is increasing
            if result.get("trend") == "WORSENING":
                score += 2
                reasons.append(
                    f"{disease} risk is worsening"
                )

        # -----------------------------------------
        # Clinical deterioration
        # -----------------------------------------

        clinical_trends = trend_result.get(
            "clinical_trends", {}
        )

        worsening_parameters = []

        for parameter, result in clinical_trends.items():

            if result.get("trend") == "WORSENING":
                worsening_parameters.append(parameter)

        if worsening_parameters:

            score += min(
                len(worsening_parameters),
                3,
            )

            reasons.append(
                "Worsening clinical parameters: "
                + ", ".join(worsening_parameters)
            )

        # -----------------------------------------
        # Overall deterioration
        # -----------------------------------------

        if trend_result.get("overall_trend") == "WORSENING":
            score += 2
            reasons.append(
                "Overall longitudinal trend is worsening"
            )

        # -----------------------------------------
        # Priority category
        # -----------------------------------------

        if score >= 8:
            priority = "HIGH"

        elif score >= 4:
            priority = "MEDIUM"

        else:
            priority = "LOW"

        return {
            "status": "success",
            "patient_id": trend_result.get("patient_id"),
            "priority": priority,
            "priority_score": score,
            "reasons": reasons,
            "worsening_signals": trend_result.get(
                "worsening_signals", []
            ),
        }