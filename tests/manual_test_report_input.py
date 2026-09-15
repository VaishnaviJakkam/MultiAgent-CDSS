import json

from src.input.report_input_agent import (
    ReportInputAgent,
)


def main():

    agent = ReportInputAgent()

    # Temporary developer test only.
    # Later this path comes automatically
    # from the UI upload API.
    report_path = (
        r"C:\Users\varsh\MultiAgent-CDSS\data\test_reports\report2.webp"
       
    )

    result = agent.analyze(
        report_path
    )

    print(
        "\n========== OCR TEXT =========="
    )

    print(
        result["raw_text"]
    )

    print(
        "\n========== ALL LAB VALUES =========="
    )

    print(
        json.dumps(
            result[
                "lab_results"
            ],
            indent=2,
        )
    )

    print(
        "\n========== SEPSIS VALUES =========="
    )

    print(
        json.dumps(
            {
                "sepsis_parameters":
                    result[
                        "sepsis_parameters"
                    ],

                "missing":
                    result[
                        "missing_sepsis_parameters"
                    ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()