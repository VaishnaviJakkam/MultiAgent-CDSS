from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from src.agents.nurse_agent import NurseAgentError, OpenRouterProvider

load_dotenv()


class ReportExtractionError(ValueError):
    """Raised when structured LLM report extraction cannot be trusted."""


class DeepSeekReportExtractor:
    """Extract every recognizable report row from already extracted raw text."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None, provider: Any | None = None) -> None:
        self.api_key = api_key or os.getenv("REPORT_EXTRACTION_LLM_API_KEY")
        self.model = model or os.getenv("REPORT_EXTRACTION_LLM_MODEL", "deepseek/deepseek-chat")
        self.provider = provider or OpenRouterProvider(api_key=self.api_key, model=self.model)

    def extract(self, raw_text: str) -> dict[str, dict[str, Any]]:
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ReportExtractionError("Report extraction received no raw text.")
        if not self.api_key:
            raise ReportExtractionError("Report extraction LLM is unavailable: set REPORT_EXTRACTION_LLM_API_KEY.")

        try:
            response_text = self.provider.generate(self._build_prompt(raw_text))
        except NurseAgentError as exc:
            raise ReportExtractionError(str(exc)) from exc
        payload = self._parse_json(response_text)
        return self._validate(payload, raw_text)

    @staticmethod
    def _parse_json(response_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ReportExtractionError("Report extraction returned malformed JSON.") from exc
        if not isinstance(payload, dict):
            raise ReportExtractionError("Report extraction response must be a JSON object.")
        return payload

    @staticmethod
    def _build_prompt(raw_text: str) -> str:
        return f"""Extract all recognizable clinical parameter/value/unit rows from this laboratory report text.

Rules:
- Extract every recognizable row, not only disease-model or vital-sign fields.
- Parameter keys must be names grounded in the report text; do not invent parameters.
- Preserve the report label in source_label and preserve raw_line when available.
- Use numeric values only when explicitly present in the report.
- Preserve explicit null as null.
- If a value is missing or uncertain, use null; never guess or infer.
- Do not filter by Sepsis, AKI, nurse, or any model feature list.
- Return ONLY JSON in this exact shape:
{{"parameters": {{"ReportParameterName": {{"value": 1.2, "unit": "mg/dL", "source_label": "Creatinine", "raw_line": "Creatinine | 1.2 | mg/dL"}}}}}}
- Each value must be a number or null. Each unit must be a string or null. source_label must be a non-empty string. raw_line must be a string or null.

RAW REPORT TEXT:
{raw_text}
""".strip()

    @classmethod
    def _validate(cls, payload: dict[str, Any], raw_text: str) -> dict[str, dict[str, Any]]:
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ReportExtractionError("Report extraction JSON must contain an object named parameters.")

        normalized_text = cls._search_text(raw_text)
        validated: dict[str, dict[str, Any]] = {}
        for name, item in parameters.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(item, dict):
                raise ReportExtractionError("Report extraction contains an invalid parameter entry.")
            source_label = item.get("source_label")
            if not isinstance(source_label, str) or not source_label.strip():
                raise ReportExtractionError(f"Parameter {name!r} has no valid source_label.")
            if cls._search_text(source_label) not in normalized_text:
                raise ReportExtractionError(f"Parameter {name!r} is not grounded in the report text.")

            value = item.get("value")
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ReportExtractionError(f"Parameter {name!r} has an invalid value.")
            unit = item.get("unit")
            if unit is not None and not isinstance(unit, str):
                raise ReportExtractionError(f"Parameter {name!r} has an invalid unit.")
            raw_line = item.get("raw_line")
            if raw_line is not None and not isinstance(raw_line, str):
                raise ReportExtractionError(f"Parameter {name!r} has an invalid raw_line.")

            validated[name.strip()] = {
                "canonical_name": name.strip(),
                "source_name": source_label.strip(),
                "value": value,
                "unit": unit,
                "raw_line": raw_line,
            }
        return validated

    @staticmethod
    def _search_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())
