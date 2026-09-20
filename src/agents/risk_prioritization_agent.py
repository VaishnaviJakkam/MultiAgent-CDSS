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

    def prioritize_history(
        self,
        assessments: list[dict[str, Any]],
        trends: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Prioritize one patient/admission from persisted longitudinal history."""
        if not assessments:
            return {
                "status": "insufficient_data",
                "priority": None,
                "priority_score": None,
                "reasons": ["No persisted disease assessments are available"],
            }

        current: dict[str, dict[str, Any]] = {}
        for assessment in assessments:
            for disease in ("sepsis", "aki"):
                result = assessment.get(disease)
                if isinstance(result, dict) and result.get("status") == "success":
                    current[disease.title()] = result

        disease_trends: dict[str, dict[str, Any]] = {}
        for trend in trends:
            disease = trend.get("disease")
            if disease in ("Sepsis", "AKI"):
                disease_trends[disease] = {
                    "trend": trend.get("trend", "INSUFFICIENT_DATA"),
                    "first": trend.get("first_probability"),
                    "latest": trend.get("latest_probability"),
                    "change": trend.get("probability_change"),
                }

        trend_result = {
            "status": "success" if current else "insufficient_data",
            "patient_id": assessments[-1].get("patient_id"),
            "disease_trends": disease_trends,
            "clinical_trends": {},
            "overall_trend": self._overall_trend(disease_trends),
            "worsening_signals": [
                f"{disease}_risk"
                for disease, result in disease_trends.items()
                if result.get("trend") == "WORSENING"
            ],
        }
        for disease, result in current.items():
            disease_trends.setdefault(disease, {})["latest"] = result.get("probability")

        prioritized = self.prioritize(trend_result)
        reasons = list(prioritized.get("reasons", []))
        if current:
            reasons.insert(0, "Current persisted assessments used: " + ", ".join(sorted(current)))
        if len(assessments) > 1:
            reasons.append(f"Previous persisted assessment history reviewed ({len(assessments) - 1} prior assessments)")
        if trends:
            reasons.append(f"Persisted trend history reviewed ({len(trends)} trends)")
        if not trends:
            reasons.append("Persisted trend history unavailable")
        prioritized["reasons"] = reasons
        return prioritized

    @staticmethod
    def _overall_trend(disease_trends: dict[str, dict[str, Any]]) -> str:
        states = {result.get("trend") for result in disease_trends.values()}
        if "WORSENING" in states:
            return "WORSENING"
        if "IMPROVING" in states:
            return "IMPROVING"
        if "STABLE" in states:
            return "STABLE"
        return "INSUFFICIENT_DATA"

    def analyze(
        self,
        assessments: list[dict[str, Any]],
        trends: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Preserve the legacy deterministic assessment/trend contract."""
        if not assessments:
            raise ValueError("assessment history cannot be empty")
        patient_ids = {assessment.get("patient_id") for assessment in assessments}
        if len(patient_ids) != 1:
            raise ValueError("assessments must belong to the same patient")
        diseases = {assessment.get("disease") for assessment in assessments}
        if any(disease not in {"Sepsis", "AKI"} for disease in diseases):
            raise ValueError("Unsupported disease")
        if len(diseases) != len(assessments):
            raise ValueError("Duplicate disease")
        if trends:
            trend_patients = {item.get("patient_id") for item in trends}
            if trend_patients != patient_ids:
                raise ValueError("trend patient does not match assessment patient")
        ordered = sorted(assessments, key=lambda item: item.get("assessment_time") or 0)
        probabilities = [assessment.get("probability") for assessment in ordered]
        if any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("probability must be between 0 and 1")
        latest_by_disease = {
            assessment["disease"]: float(assessment["probability"])
            for assessment in ordered
        }
        score = 0
        reasons = []
        for disease, latest in latest_by_disease.items():
            if latest >= 0.60:
                score += 3
                reasons.append(f"High {disease} risk ({latest:.2f})")
            elif latest >= 0.40:
                score += 2
                reasons.append(f"Moderate {disease} risk ({latest:.2f})")
        worsening = [item.get("disease") for item in trends or [] if item.get("trend") == "WORSENING"]
        if worsening:
            score += 2 * len(worsening)
            reasons.append("worsening risk: " + ", ".join(worsening))
        priority = "HIGH" if score >= 3 else "MEDIUM" if score >= 2 else "LOW"
        highest_risk_disease = max(latest_by_disease, key=latest_by_disease.get)
        return {
            "status": "success",
            "patient_id": next(iter(patient_ids)),
            "priority_level": priority,
            "priority": priority,
            "priority_score": score,
            "diseases_assessed": sorted(diseases),
            "highest_probability": latest_by_disease[highest_risk_disease],
            "highest_risk_disease": highest_risk_disease,
            "worsening_diseases": worsening,
            "available_trends": {item.get("disease"): item.get("trend") for item in trends or []},
            "reason": "; ".join(reasons),
            "reasons": reasons,
            "message": "Trend history unavailable" if not trends else "",
        }