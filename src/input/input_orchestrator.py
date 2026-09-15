class InputOrchestrator:
    """
    Coordinates the input stage.

    Responsibilities:
    1. Inspect values extracted from the report.
    2. Determine which Sepsis inputs are still missing.
    3. Convert missing model features into values the nurse can provide.
    4. Generate a nurse-facing request.
    5. Check whether nurse input satisfied the request.

    This does NOT run the Sepsis model.
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

    # Model feature -> what we actually ask nurse for.
    #
    # SBP + MAP do not need two separate questions.
    # Nurse gives BP and MAP can be derived.
    FEATURE_TO_NURSE_REQUEST = {
        "HR": "HR",
        "O2Sat": "O2Sat",
        "Temp": "Temp",
        "SBP": "BP",
        "MAP": "BP",
        "Resp": "Resp",
        "WBC": "WBC",
        "Lactate": "Lactate",
    }

    DISPLAY_NAMES = {
        "HR": "heart rate",
        "O2Sat": "oxygen saturation",
        "Temp": "temperature",
        "BP": "blood pressure",
        "Resp": "respiratory rate",
        "WBC": "latest WBC result",
        "Lactate": "latest lactate result",
    }

    def determine_missing_features(
        self,
        report_result: dict,
    ) -> list[str]:
        """
        Determine which Sepsis features were NOT obtained
        from the uploaded report.
        """

        report_parameters = report_result.get(
            "sepsis_parameters",
            {},
        )

        missing = []

        for feature in self.SEPSIS_REQUIRED_FEATURES:
            if (
                feature not in report_parameters
                or report_parameters[feature] is None
            ):
                missing.append(feature)

        return missing

    def create_nurse_request(
        self,
        report_result: dict,
    ) -> dict:
        """
        Generate a structured request for the nurse.

        Example:
        report contains WBC only.

        Missing model features:
        HR, O2Sat, Temp, SBP, MAP, Resp, Lactate

        Nurse request becomes:
        HR, O2Sat, Temp, BP, Resp, Lactate

        BP appears only once because SBP is obtained from BP
        and MAP can be derived from SBP/DBP.
        """

        missing_features = self.determine_missing_features(
            report_result
        )

        nurse_fields = []

        for feature in missing_features:

            request_field = (
                self.FEATURE_TO_NURSE_REQUEST.get(feature)
            )

            if (
                request_field
                and request_field not in nurse_fields
            ):
                nurse_fields.append(request_field)

        display_values = [
            self.DISPLAY_NAMES[field]
            for field in nurse_fields
        ]

        if display_values:
            request_text = (
                "Please provide "
                + ", ".join(display_values[:-1])
            )

            if len(display_values) == 1:
                request_text = (
                    "Please provide "
                    + display_values[0]
                )

            elif len(display_values) > 1:
                request_text += (
                    " and "
                    + display_values[-1]
                )

            request_text += "."

        else:
            request_text = (
                "No additional nurse input is required."
            )

        return {
            "status": (
                "input_required"
                if nurse_fields
                else "complete"
            ),

            "missing_model_features": missing_features,

            "requested_nurse_fields": nurse_fields,

            "message": request_text,
        }

    def validate_nurse_response(
        self,
        nurse_request: dict,
        nurse_result: dict,
    ) -> dict:
        """
        Check whether the nurse response supplied all
        requested values.
        """

        requested_fields = nurse_request.get(
            "requested_nurse_fields",
            [],
        )

        extracted = nurse_result.get(
            "extracted_parameters",
            {},
        )

        still_missing = []

        for field in requested_fields:

            # Blood pressure is considered present only
            # when systolic and diastolic values exist.
            if field == "BP":

                if (
                    "SBP" not in extracted
                    or "DBP" not in extracted
                ):
                    still_missing.append("BP")

            else:

                if (
                    field not in extracted
                    or extracted[field] is None
                ):
                    still_missing.append(field)

        return {
            "status": (
                "complete"
                if not still_missing
                else "incomplete"
            ),

            "requested_fields": requested_fields,

            "still_missing": still_missing,

            "all_requested_values_received": (
                len(still_missing) == 0
            ),
        }