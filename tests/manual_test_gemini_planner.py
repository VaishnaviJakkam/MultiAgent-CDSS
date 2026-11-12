from src.agents.gemini_planner import (
    GeminiPlanner,
)


def main() -> None:

    planner = GeminiPlanner()

    prompt = """
You are controlling a clinical assessment workflow.

Sepsis status: pending
AKI status: pending

Available actions:
assess_sepsis
assess_aki
finish

Both diseases must eventually reach a terminal state.

Choose exactly one next action.
"""

    result = planner.generate(
        prompt
    )

    print(
        "Gemini response:"
    )

    print(
        result
    )


if __name__ == "__main__":
    main()