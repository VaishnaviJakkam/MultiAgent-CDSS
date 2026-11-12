from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai


# Load environment variables from .env
load_dotenv()


class GeminiPlanner:
    """
    Reusable Gemini planner for multiple agents.

    Gemini only decides the next workflow action.
    It does NOT calculate clinical probabilities,
    trends, or priority scores.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.6-flash",
    ):
        # Use explicitly supplied key, otherwise read from .env
        api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Add it to your .env file."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def decide(
        self,
        prompt: str,
        allowed_actions: list[str],
    ) -> dict[str, Any]:
        """
        Ask Gemini to choose exactly one action
        from the actions supplied by the calling agent.
        """

        if not allowed_actions:
            raise ValueError(
                "allowed_actions cannot be empty."
            )

        action_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": allowed_actions,
                },
                "reason": {
                    "type": "string",
                },
            },
            "required": [
                "action",
                "reason",
            ],
            "additionalProperties": False,
        }

        response = self.client.interactions.create(
            model=self.model,
            input=prompt,
            response_format=action_schema,
        )

        text = response.output_text

        if not text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        try:
            decision = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini returned invalid JSON: {text}"
            ) from exc

        action = decision.get("action")

        # Deterministic guardrail:
        # Gemini cannot call anything outside the
        # tools/actions exposed by the current agent.
        if action not in allowed_actions:
            raise ValueError(
                f"Gemini returned invalid action: {action}"
            )

        return decision