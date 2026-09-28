from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv()


class NurseAgentError(ValueError):
    """Controlled error raised when nurse-agent reasoning cannot be completed."""


class NurseAgentProvider(Protocol):
    def generate(self, prompt: str) -> str:
        """Return the provider response as text."""


class OpenRouterProvider:
    """OpenAI-compatible OpenRouter adapter for Nurse Agent decisions."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("NURSE_AGENT_LLM_API_KEY")
        self.model = model or os.getenv("NURSE_AGENT_LLM_MODEL", "openrouter/free")

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise NurseAgentError(
                "Nurse Agent LLM is unavailable: set NURSE_AGENT_LLM_API_KEY."
            )

        request_body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = Request(
            self.BASE_URL,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                response_payload = json.load(response)
        except HTTPError as exc:
            raise NurseAgentError(
                f"Nurse Agent LLM request failed: OpenRouter returned HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            raise NurseAgentError(
                f"Nurse Agent LLM request failed: OpenRouter is unavailable ({exc.reason})."
            ) from exc
        except Exception as exc:
            raise NurseAgentError(f"Nurse Agent LLM request failed: {exc}") from exc

        try:
            text = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise NurseAgentError("OpenRouter returned an invalid response shape.") from exc

        if not isinstance(text, str) or not text.strip():
            raise NurseAgentError("OpenRouter returned an empty response.")
        return text


@dataclass(frozen=True)
class NurseAgentDecision:
    action: str
    parameter: str | None
    question: str | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "parameter": self.parameter,
            "question": self.question,
            "reason": self.reason,
        }


class NurseAgent:
    """LLM decision layer for selecting the next unresolved nurse parameter."""

    ACTIONS = {"ASK_NURSE", "COMPLETE"}
    REQUIRED_PARAMETERS = ("HR", "O2Sat", "Temp", "BP", "Resp", "WBC", "Lactate")
    DISPLAY_NAMES = {
        "HR": "heart rate",
        "O2Sat": "oxygen saturation (SpO2)",
        "Temp": "temperature",
        "BP": "blood pressure",
        "Resp": "respiratory rate",
        "WBC": "white blood cell count",
        "Lactate": "lactate",
    }

    def __init__(self, provider: NurseAgentProvider | None = None) -> None:
        self.provider = provider or self._configured_provider()

    @staticmethod
    def _configured_provider() -> NurseAgentProvider:
        provider_name = os.getenv("NURSE_AGENT_LLM_PROVIDER", "openrouter").strip().lower()
        if provider_name != "openrouter":
            raise NurseAgentError(
                f"Unsupported NURSE_AGENT_LLM_PROVIDER: {provider_name}."
            )
        return OpenRouterProvider()

    def decide(
        self,
        *,
        patient_id: str,
        admission_id: str,
        lab_values: dict[str, Any],
        nurse_values: dict[str, Any],
        unavailable_parameters: set[str] | None = None,
        current_task_state: dict[str, Any] | None = None,
    ) -> NurseAgentDecision:
        unavailable = set(unavailable_parameters or set())
        state = self._observe(
            patient_id=patient_id,
            admission_id=admission_id,
            lab_values=lab_values,
            nurse_values=nurse_values,
            unavailable_parameters=unavailable,
            current_task_state=current_task_state or {},
        )
        if not state["unresolved_parameters"]:
            return NurseAgentDecision(
                action="COMPLETE",
                parameter=None,
                question=None,
                reason="All required assessment parameters are already resolved.",
            )

        try:
            raw_response = self.provider.generate(self._build_prompt(state))
        except NurseAgentError:
            raise
        except Exception as exc:
            raise NurseAgentError(f"Nurse Agent LLM is unavailable: {exc}") from exc
        decision = self._parse_and_validate(raw_response, state)
        return decision

    def _observe(
        self,
        *,
        patient_id: str,
        admission_id: str,
        lab_values: dict[str, Any],
        nurse_values: dict[str, Any],
        unavailable_parameters: set[str],
        current_task_state: dict[str, Any],
    ) -> dict[str, Any]:
        resolved: list[str] = []
        unresolved: list[str] = []
        resolved_as_unavailable = set(unavailable_parameters)
        for parameter in self.REQUIRED_PARAMETERS:
            if parameter in unavailable_parameters or self._has_value(parameter, lab_values, nurse_values):
                resolved.append(parameter)
            else:
                unresolved.append(parameter)

        return {
            "patient_id": patient_id,
            "admission_id": admission_id,
            "lab_values": lab_values,
            "nurse_values": nurse_values,
            "required_parameters": list(self.REQUIRED_PARAMETERS),
            "resolved_parameters": resolved,
            "resolved_as_unavailable": sorted(resolved_as_unavailable),
            "unresolved_parameters": unresolved,
            "current_task_state": current_task_state,
        }

    @staticmethod
    def _has_value(parameter: str, lab_values: dict[str, Any], nurse_values: dict[str, Any]) -> bool:
        values = {**lab_values, **nurse_values}
        if parameter == "BP":
            return (
                (values.get("SBP") is not None and (
                    values.get("DBP") is not None
                    or values.get("MAP") is not None
                ))
                or values.get("BP") is not None
            )
        return values.get(parameter) is not None

    def _build_prompt(self, state: dict[str, Any]) -> str:
        return f"""You are the Nurse Agent decision-maker.

Goal: collect the clinical information required for the downstream disease assessment for this patient.

Observe the current state below and choose exactly one action. Existing lab values are authoritative inputs. Nurse-collected values are also inputs. Do not ask for information already resolved. A parameter marked resolved-as-unavailable is resolved and must not be requested again. Missing is different from resolved-as-unavailable. Never invent or infer a clinical value. You only decide; you do not modify data, call tools, or execute functions.

Allowed actions: ASK_NURSE, COMPLETE.
Allowed parameters: {list(self.REQUIRED_PARAMETERS)}.
If unresolved parameters remain, return ASK_NURSE and select exactly one parameter from unresolved_parameters. Generate a concise nurse-facing question. If none remain, return COMPLETE with null parameter and question.
Return ONLY valid JSON with exactly these keys: action, parameter, question, reason.

Current state:
{json.dumps(state, sort_keys=True, default=str)}
""".strip()

    def _parse_and_validate(self, raw_response: str, state: dict[str, Any]) -> NurseAgentDecision:
        try:
            payload = json.loads(raw_response)
        except (TypeError, json.JSONDecodeError) as exc:
            raise NurseAgentError("Nurse Agent LLM returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise NurseAgentError("Nurse Agent LLM response must be a JSON object.")

        action = payload.get("action")
        parameter = payload.get("parameter")
        question = payload.get("question")
        reason = payload.get("reason")
        if action not in self.ACTIONS:
            raise NurseAgentError(f"Nurse Agent returned invalid action: {action}")
        if not isinstance(reason, str) or not reason.strip():
            raise NurseAgentError("Nurse Agent response must include a reason.")

        unresolved = set(state["unresolved_parameters"])
        if action == "COMPLETE":
            if parameter is not None or question is not None:
                raise NurseAgentError("COMPLETE decisions must use null parameter and question.")
            if unresolved:
                raise NurseAgentError("Nurse Agent returned COMPLETE while parameters remain unresolved.")
        else:
            if parameter not in self.REQUIRED_PARAMETERS:
                raise NurseAgentError(f"Nurse Agent returned invalid parameter: {parameter}")
            if parameter not in unresolved:
                raise NurseAgentError(f"Nurse Agent selected already resolved parameter: {parameter}")
            if not isinstance(question, str) or not question.strip():
                raise NurseAgentError("ASK_NURSE decisions must include a question.")

        return NurseAgentDecision(
            action=action,
            parameter=parameter,
            question=question,
            reason=reason,
        )
