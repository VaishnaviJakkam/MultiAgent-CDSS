import json

from src.input.report_input_agent import (
    ReportInputAgent,
)

from src.input.nurse_audio_agent import (
    NurseAudioAgent,
)

from src.input.input_orchestrator import (
    InputOrchestrator,
)

from src.input.data_processing_agent import (
    DataProcessingAgent,
)


def main():

    print("\nLoading agents...")

    report_agent = ReportInputAgent()
    nurse_agent = NurseAudioAgent()
    orchestrator = InputOrchestrator()
    processing_agent = DataProcessingAgent()

    # -----------------------------------------
    # TEMPORARY DEVELOPMENT FILES
    # Later these come from the Nurse UI.
    # -----------------------------------------

    report_path = (
        r"data/test_reports/report2.webp"
    )

    audio_path = (
        r"data/test_audio/nurse_sepsis_test.mp3"
    )

    # -----------------------------------------
    # STEP 1 — REPORT INPUT AGENT
    # -----------------------------------------

    print("\nProcessing report...")

    report_result = report_agent.analyze(
        report_path
    )

    print(
        "\n========== REPORT VALUES =========="
    )

    print(
        json.dumps(
            report_result.get(
                "sepsis_parameters",
                {},
            ),
            indent=2,
        )
    )

    # -----------------------------------------
    # STEP 2 — ORCHESTRATOR
    # Detect missing values
    # -----------------------------------------

    print(
        "\n========== ORCHESTRATOR =========="
    )

    nurse_request = (
        orchestrator.create_nurse_request(
            report_result
        )
    )

    print(
        json.dumps(
            nurse_request,
            indent=2,
        )
    )

    print(
        "\nNURSE NOTIFICATION:"
    )

    print(
        nurse_request["message"]
    )

    # -----------------------------------------
    # STEP 3 — NURSE AUDIO INPUT
    # -----------------------------------------

    if (
        nurse_request["status"]
        == "input_required"
    ):

        print(
            "\nProcessing nurse audio..."
        )

        nurse_result = (
            nurse_agent.analyze(
                audio_path
            )
        )

    else:

        nurse_result = {
            "status": "not_required",
            "source": "nurse_audio",
            "transcript": "",
            "extracted_parameters": {},
            "metadata": {},
            "requires_confirmation": False,
        }

    print(
        "\n========== NURSE TRANSCRIPT =========="
    )

    print(
        nurse_result.get(
            "transcript",
            "",
        )
    )

    print(
        "\n========== NURSE VALUES =========="
    )

    print(
        json.dumps(
            nurse_result.get(
                "extracted_parameters",
                {},
            ),
            indent=2,
        )
    )

    # -----------------------------------------
    # STEP 4 — VALIDATE NURSE RESPONSE
    # -----------------------------------------

    validation = (
        orchestrator.validate_nurse_response(
            nurse_request,
            nurse_result,
        )
    )

    print(
        "\n========== NURSE RESPONSE VALIDATION =========="
    )

    print(
        json.dumps(
            validation,
            indent=2,
        )
    )

    # -----------------------------------------
    # STEP 5 — MERGE INPUTS
    # -----------------------------------------

    print(
        "\nCombining inputs..."
    )

    final_result = (
        processing_agent.process(
            patient_id="TEST-PATIENT-001",
            admission_id="ADM-001",
            report_result=report_result,
            nurse_result=nurse_result,
        )
    )

    print(
        "\n========== FINAL OBSERVATION =========="
    )

    print(
        json.dumps(
            final_result,
            indent=2,
            default=str,
        )
    )

    # -----------------------------------------
    # FINAL PIPELINE STATUS
    # -----------------------------------------

    print(
        "\n========== INPUT PIPELINE STATUS =========="
    )

    if final_result[
        "ready_for_sepsis_assessment"
    ]:

        print(
            "SUCCESS: Observation is ready "
            "for Sepsis assessment."
        )

    else:

        print(
            "INCOMPLETE: Missing Sepsis inputs:"
        )

        print(
            final_result[
                "missing_sepsis_parameters"
            ]
        )


if __name__ == "__main__":
    main()