from __future__ import annotations

from numbers import Real
from typing import Any


# Parameters whose increase generally represents deterioration
INCREASE_WORSE = {
    "HR",
    "Resp",
    "WBC",
    "Lactate",
}

# Parameters whose decrease generally represents deterioration
DECREASE_WORSE = {
    "O2Sat",
    "SBP",
    "MAP",
}

# Temperature is handled separately because both
# unusually high and unusually low values can matter.
SPECIAL_PARAMETERS = {
    "Temp",
}

CLINICAL_PARAMETERS = (
    INCREASE_WORSE
    | DECREASE_WORSE
    | SPECIAL_PARAMETERS
)

PROBABILITY_CHANGE_THRESHOLD = 0.05
CLINICAL_CHANGE_THRESHOLD = 0.05


class TrendAnalysisAgent:
    """
    Deterministic longitudinal trend-analysis tool.

    Analyzes:
    1. Sepsis probability over time
    2. AKI probability over time
    3. Available clinical parameters over time

    This tool does not use an LLM to calculate trends.
    """

    def __init__(
        self,
        probability_threshold: float = PROBABILITY_CHANGE_THRESHOLD,
        clinical_threshold: float = CLINICAL_CHANGE_THRESHOLD,
    ) -> None:

        self.probability_threshold = probability_threshold
        self.clinical_threshold = clinical_threshold

    # =========================================================
    # DISEASE PROBABILITY TREND
    # =========================================================

    def _probability_trend(
        self,
        values: list[float],
    ) -> dict[str, Any]:

        if len(values) < 2:
            return {
                "trend": "INSUFFICIENT_DATA",
                "first": values[0] if values else None,
                "latest": values[-1] if values else None,
                "change": None,
                "count": len(values),
            }

        first = float(values[0])
        latest = float(values[-1])

        change = latest - first

        if change > self.probability_threshold:
            trend = "WORSENING"

        elif change < -self.probability_threshold:
            trend = "IMPROVING"

        else:
            trend = "STABLE"

        return {
            "trend": trend,
            "first": first,
            "latest": latest,
            "change": change,
            "count": len(values),
        }

    # =========================================================
    # CLINICAL PARAMETER TREND
    # =========================================================

    def _clinical_trend(
        self,
        parameter: str,
        values: list[float],
    ) -> dict[str, Any]:

        if len(values) < 2:
            return {
                "trend": "INSUFFICIENT_DATA",
                "first": values[0] if values else None,
                "latest": values[-1] if values else None,
                "change": None,
                "count": len(values),
            }

        first = float(values[0])
        latest = float(values[-1])

        absolute_change = latest - first

        denominator = max(abs(first), 1e-8)

        relative_change = (
            absolute_change / denominator
        )

        if parameter in INCREASE_WORSE:

            if relative_change > self.clinical_threshold:
                trend = "WORSENING"

            elif relative_change < -self.clinical_threshold:
                trend = "IMPROVING"

            else:
                trend = "STABLE"

        elif parameter in DECREASE_WORSE:

            if relative_change < -self.clinical_threshold:
                trend = "WORSENING"

            elif relative_change > self.clinical_threshold:
                trend = "IMPROVING"

            else:
                trend = "STABLE"

        elif parameter == "Temp":

            # For temperature, direction alone is not enough.
            # We only report whether the value is moving.
            if abs(relative_change) <= self.clinical_threshold:
                trend = "STABLE"
            else:
                trend = "CHANGING"

        else:

            trend = "UNKNOWN"

        return {
            "trend": trend,
            "first": first,
            "latest": latest,
            "change": absolute_change,
            "relative_change": relative_change,
            "count": len(values),
        }

    # =========================================================
    # EXTRACT DISEASE HISTORY
    # =========================================================

    def _extract_disease_probabilities(
        self,
        assessments: list[dict[str, Any]],
        disease: str,
    ) -> list[float]:

        values = []

        key = disease.lower()

        for assessment in assessments:

            result = assessment.get(key)

            if not isinstance(result, dict):
                continue

            if result.get("status") != "success":
                continue

            probability = result.get("probability")

            if (
                isinstance(probability, Real)
                and not isinstance(probability, bool)
            ):
                values.append(
                    float(probability)
                )

        return values

    # =========================================================
    # EXTRACT CLINICAL HISTORY
    # =========================================================

    def _extract_clinical_values(
        self,
        observations: list[dict[str, Any]],
        parameter: str,
    ) -> list[float]:

        values = []

        for observation in observations:

            clinical = observation.get(
                "clinical_parameters",
                {},
            )

            value = clinical.get(parameter)

            if (
                isinstance(value, Real)
                and not isinstance(value, bool)
            ):
                values.append(
                    float(value)
                )

        return values

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        history: dict[str, Any],
    ) -> dict[str, Any]:

        observations = history.get(
            "observations",
            [],
        )

        assessments = history.get(
            "assessments",
            [],
        )

        patient = history.get(
            "patient",
            {},
        )

        patient_id = patient.get(
            "patient_id",
        )

        # -----------------------------------------------------
        # Disease probability trends
        # -----------------------------------------------------

        disease_trends = {}

        for disease in (
            "Sepsis",
            "AKI",
        ):

            probabilities = (
                self._extract_disease_probabilities(
                    assessments,
                    disease,
                )
            )

            disease_trends[disease] = (
                self._probability_trend(
                    probabilities
                )
            )

        # -----------------------------------------------------
        # Clinical parameter trends
        # -----------------------------------------------------

        clinical_trends = {}

        for parameter in sorted(
            CLINICAL_PARAMETERS
        ):

            values = (
                self._extract_clinical_values(
                    observations,
                    parameter,
                )
            )

            if values:

                clinical_trends[parameter] = (
                    self._clinical_trend(
                        parameter,
                        values,
                    )
                )

        # -----------------------------------------------------
        # Overall deterioration
        # -----------------------------------------------------

        worsening_signals = []

        improving_signals = []

        for disease, result in disease_trends.items():

            if result["trend"] == "WORSENING":
                worsening_signals.append(
                    f"{disease}_risk"
                )

            elif result["trend"] == "IMPROVING":
                improving_signals.append(
                    f"{disease}_risk"
                )

        for parameter, result in clinical_trends.items():

            if result["trend"] == "WORSENING":
                worsening_signals.append(
                    parameter
                )

            elif result["trend"] == "IMPROVING":
                improving_signals.append(
                    parameter
                )

        if worsening_signals:

            overall = "WORSENING"

        elif improving_signals:

            overall = "IMPROVING"

        else:

            enough_data = (
                len(observations) >= 2
                or len(assessments) >= 2
            )

            overall = (
                "STABLE"
                if enough_data
                else "INSUFFICIENT_DATA"
            )

        return {
            "patient_id": patient_id,
            "overall_trend": overall,
            "disease_trends": disease_trends,
            "clinical_trends": clinical_trends,
            "worsening_signals": worsening_signals,
            "improving_signals": improving_signals,
            "observation_count": len(observations),
            "assessment_count": len(assessments),
            "status": (
                "success"
                if overall != "INSUFFICIENT_DATA"
                else "insufficient_data"
            ),
        }