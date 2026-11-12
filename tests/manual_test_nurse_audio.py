import json

from src.input.nurse_audio_agent import NurseAudioAgent


def main():
    print("\nLoading speech-to-text model...")

    agent = NurseAudioAgent()

    # Temporary test file.
    # Later this comes directly from the nurse UI.
    audio_path = r"data/test_audio/nurse_sepsis_test.mp3"

    print("\nProcessing nurse audio...")

    result = agent.analyze(audio_path)

    print("\n========== TRANSCRIPT ==========")
    print(result["transcript"])

    print("\n========== EXTRACTED PARAMETERS ==========")
    print(
        json.dumps(
            result["extracted_parameters"],
            indent=2,
        )
    )

    print("\n========== METADATA ==========")
    print(
        json.dumps(
            result["metadata"],
            indent=2,
        )
    )

    print("\n========== CONFIRMATION REQUIRED ==========")
    print(result["requires_confirmation"])


if __name__ == "__main__":
    main()