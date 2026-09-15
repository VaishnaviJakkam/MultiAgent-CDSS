from datetime import datetime, timezone
from typing import Optional


class DataProcessingAgent:
    """
    Combines report-extracted lab values and confirmed
    nurse-input values into one structured patient observation.

    Current focus: Sepsis.
    """

    SEPSIS_REQUIRED_FEATURES = [
        "HR",
        "O2Sat",
        "Temp",
        "SBP",
        "MAP",
        "Resp",
        "WBC",
        "Lactate",
    ]

    def process(
        self,
        patient_id: str,
        admission_id: str,
        report_result: Optional[dict] = None,
        nurse_result: Optional[dict] = None,
        observation_time: Optional[datetime] = None,
    ) -> dict:

        if observation_time is None:
            observation_time = datetime.now(timezone.utc)

        parameters = {}
        parameter_sources = {}

        # --------------------------------
        # 1. REPORT VALUES
        # --------------------------------

        if report_result:

            report_parameters = report_result.get(
                "sepsis_parameters",
                {},
            )

            for name, value in report_parameters.items():

                parameters[name] = value

                parameter_sources[name] = {
                    "source": "lab_report"
                }

        # --------------------------------
        # 2. NURSE VALUES
        # --------------------------------

        if nurse_result:

            nurse_parameters = nurse_result.get(
                "extracted_parameters",
                {},
            )

            nurse_metadata = nurse_result.get(
                "metadata",
                {},
            )

            for name, value in nurse_parameters.items():

                # Nurse-confirmed data can fill values that
                # were absent from the report.
                #
                # For now, if the same value exists from both
                # sources, preserve report value unless the
                # nurse value is explicitly used later as a
                # correction.
                if name not in parameters:
                    parameters[name] = value

                    parameter_sources[name] = {
                        "source": "nurse_audio"
                    }

                if name in nurse_metadata:
                    parameter_sources[name].update(
                        nurse_metadata[name]
                    )

        # --------------------------------
        # 3. CHECK SEPSIS REQUIREMENTS
        # --------------------------------

        missing = [
            feature
            for feature in self.SEPSIS_REQUIRED_FEATURES
            if feature not in parameters
            or parameters[feature] is None
        ]

        complete = len(missing) == 0

        # Only exact Sepsis model features.
        sepsis_features = {
            feature: parameters[feature]
            for feature in self.SEPSIS_REQUIRED_FEATURES
            if feature in parameters
        }

        return {
            "status": (
                "ready"
                if complete
                else "incomplete"
            ),

            "patient_id": patient_id,

            "admission_id": admission_id,

            "observation_time": observation_time,

            "clinical_parameters": parameters,

            "sepsis_features": sepsis_features,

            "parameter_sources": parameter_sources,

            "missing_sepsis_parameters": missing,

            "ready_for_sepsis_assessment": complete,
        }