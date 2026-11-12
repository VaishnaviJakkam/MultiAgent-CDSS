import json

import pytest

from src.agents.nurse_agent import NurseAgent, NurseAgentError


class FakeProvider:
    def __init__(self, response: dict[str, object] | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return json.dumps(self.response)


def ask(provider: FakeProvider, lab: dict[str, object], nurse: dict[str, object] | None = None, unavailable: set[str] | None = None):
    return NurseAgent(provider).decide(
        patient_id="P001",
        admission_id="A001",
        lab_values=lab,
        nurse_values=nurse or {},
        unavailable_parameters=unavailable,
    )


def test_lactate_already_present_is_not_selected() -> None:
    provider = FakeProvider({"action": "ASK_NURSE", "parameter": "O2Sat", "question": "O2Sat?", "reason": "O2Sat is missing."})
    decision = ask(provider, {"Lactate": 3.7})
    assert decision.parameter == "O2Sat"
    assert '"Lactate": 3.7' in provider.prompts[0]


def test_all_required_parameters_available_returns_complete_without_llm_call() -> None:
    provider = FakeProvider({})
    values = {"HR": 115, "O2Sat": 97, "Temp": 38.6, "SBP": 120, "DBP": 70, "Resp": 20, "WBC": 15.2, "Lactate": 3.7}
    decision = ask(provider, values)
    assert decision.action == "COMPLETE"
    assert provider.prompts == []


def test_only_lactate_missing_requests_lactate() -> None:
    provider = FakeProvider({"action": "ASK_NURSE", "parameter": "Lactate", "question": "Please provide lactate.", "reason": "Lactate is missing."})
    values = {"HR": 115, "O2Sat": 97, "Temp": 38.6, "SBP": 120, "DBP": 70, "Resp": 20, "WBC": 15.2}
    decision = ask(provider, values)
    assert decision.action == "ASK_NURSE"
    assert decision.parameter == "Lactate"


def test_lab_and_nurse_values_leave_only_bp_and_resp_unresolved() -> None:
    provider = FakeProvider({"action": "ASK_NURSE", "parameter": "BP", "question": "Please provide blood pressure.", "reason": "BP is missing."})
    decision = ask(provider, {"HR": 115, "Temp": 38.6, "WBC": 15.2, "Lactate": 3.7}, {"O2Sat": 97})
    prompt = provider.prompts[0]
    assert '"unresolved_parameters": ["BP", "Resp"]' in prompt
    assert decision.parameter == "BP"


def test_unavailable_parameter_is_resolved_and_not_requested_again() -> None:
    provider = FakeProvider({"action": "ASK_NURSE", "parameter": "Resp", "question": "Please provide respiratory rate.", "reason": "Resp is missing."})
    decision = ask(provider, {"HR": 90}, {"Lactate": None}, {"Lactate"})
    assert decision.parameter == "Resp"
    assert '"Lactate"' in provider.prompts[0]
    assert '"unresolved_parameters": ["O2Sat", "Temp", "BP", "Resp", "WBC"]' in provider.prompts[0]


def test_invalid_llm_parameter_is_rejected() -> None:
    provider = FakeProvider({"action": "ASK_NURSE", "parameter": "Glucose", "question": "Glucose?", "reason": "Missing."})
    with pytest.raises(NurseAgentError, match="invalid parameter"):
        ask(provider, {"HR": 90})


def test_invalid_llm_action_is_rejected() -> None:
    provider = FakeProvider({"action": "CALL_TOOL", "parameter": None, "question": None, "reason": "No."})
    with pytest.raises(NurseAgentError, match="invalid action"):
        ask(provider, {"HR": 90})


def test_llm_unavailable_is_a_controlled_error() -> None:
    provider = FakeProvider(error=RuntimeError("provider offline"))
    with pytest.raises(NurseAgentError, match="provider offline"):
        ask(provider, {"HR": 90})
