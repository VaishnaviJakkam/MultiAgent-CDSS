import json

import pytest

from src.input.deepseek_report_extractor import DeepSeekReportExtractor, ReportExtractionError


class FakeProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        assert "RAW REPORT TEXT:" in prompt
        return self.response


class FakeDeepSeekExtractor(DeepSeekReportExtractor):
    def __init__(self, response: str) -> None:
        super().__init__(api_key="test", model="deepseek/deepseek-chat", provider=FakeProvider(response))


def test_validates_complete_unrestricted_parameter_response() -> None:
    raw_text = "HR | 79 | bpm\nCreatinine | 0.8 | mg/dL"
    response = json.dumps({
        "parameters": {
            "HR": {"value": 79, "unit": "bpm", "source_label": "HR", "raw_line": "HR | 79 | bpm"},
            "Creatinine": {"value": 0.8, "unit": "mg/dL", "source_label": "Creatinine", "raw_line": "Creatinine | 0.8 | mg/dL"},
        }
    })
    extracted = FakeDeepSeekExtractor(response).extract(raw_text)
    assert extracted["HR"]["value"] == 79
    assert extracted["Creatinine"]["value"] == 0.8


def test_rejects_malformed_json() -> None:
    with pytest.raises(ReportExtractionError, match="malformed JSON"):
        FakeDeepSeekExtractor("not json").extract("HR | 79 | bpm")


def test_rejects_parameter_not_grounded_in_raw_text() -> None:
    response = json.dumps({
        "parameters": {
            "Invented": {"value": 1, "unit": None, "source_label": "Invented", "raw_line": None},
        }
    })
    with pytest.raises(ReportExtractionError, match="not grounded"):
        FakeDeepSeekExtractor(response).extract("HR | 79 | bpm")


def test_preserves_explicit_null() -> None:
    response = json.dumps({
        "parameters": {
            "Lactate": {"value": None, "unit": "mmol/L", "source_label": "Lactate", "raw_line": "Lactate | null | mmol/L"},
        }
    })
    extracted = FakeDeepSeekExtractor(response).extract("Lactate | null | mmol/L")
    assert extracted["Lactate"]["value"] is None
