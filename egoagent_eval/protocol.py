"""Single, auditable model-response parser and protocol normalizer.

This module is deliberately independent of the VLM transport and controllers.
It performs only deterministic envelope unwrapping and strict typed validation;
it never invents timestamps or turns prose evidence into frame references.
"""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

from .decisions import (AnswerDecision, Decision, DecisionMetadata, ProtocolError,
                        ProviderRefusal, RawModelResponse, ToolDecision)
from .schema import (ACTION_ADAPTER, FinalAction, FinalAnswer, InspectFrame,
                     SampleInterval, SelectCandidate)
from .video import Observation


# These fields describe the model's reasoning context but never control tool
# execution.  The whitelist is intentionally small: unknown fields remain
# schema errors rather than being silently discarded.
AUDIT_ONLY_FIELDS = frozenset({
    "confidence", "objects", "explanation", "reason", "elapsed_seconds", "tool_calls",
})


def _load_json_object(text: str) -> dict[str, Any]:
    def pairs(items):
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("JSON output must be a nonempty string")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError("Unclosed Markdown JSON code block")
        cleaned = "\n".join(lines[1:-1]).strip()

    decoder = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=invalid_constant)
    try:
        result = decoder.decode(cleaned)
    except json.JSONDecodeError:
        result = None
        for index, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                result = candidate
                break
        if result is None:
            raise ValueError("JSON output is not a complete object")
    if not isinstance(result, dict):
        raise ValueError("JSON output must be an object")
    return result


def _action_from_tool_call(value: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    name = value.get("name")
    arguments = value.get("arguments")
    tool_id = value.get("id") if isinstance(value.get("id"), str) else None
    steps: list[str] = []
    if isinstance(value.get("function"), dict):
        function = value["function"]
        name = function.get("name", name)
        arguments = function.get("arguments", arguments)
        steps.append("unwrap_function_call")
    if not isinstance(name, str):
        return None, tool_id, steps
    if isinstance(arguments, str):
        arguments = _load_json_object(arguments)
        steps.append("decode_function_arguments")
    if not isinstance(arguments, dict):
        raise ValueError("Tool definition did not contain JSON arguments")
    if name not in {"inspect_frame", "sample_interval", "select_candidate", "final_answer"}:
        raise ValueError(f"Unsupported tool action: {name}")
    return {**arguments, "action": name}, tool_id, steps


def _unwrap_action(value: dict[str, Any], depth: int = 0) -> tuple[dict[str, Any], list[str], str | None]:
    if depth > 4:
        raise ValueError("Action wrapper nesting is too deep")
    if isinstance(value.get("action"), dict):
        payload, steps, tool_id = _unwrap_action(value["action"], depth + 1)
        return payload, ["unwrap_action_object", *steps], tool_id
    if isinstance(value.get("action"), str):
        return value, [], None

    for key in ("output", "result", "response", "payload", "data"):
        nested = value.get(key)
        if isinstance(nested, dict):
            payload, steps, tool_id = _unwrap_action(nested, depth + 1)
            return payload, [f"unwrap_{key}", *steps], tool_id
        if isinstance(nested, str) and nested.strip():
            payload, steps, tool_id = _unwrap_action(_load_json_object(nested), depth + 1)
            return payload, [f"decode_{key}", *steps], tool_id

    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list):
        if len(tool_calls) != 1 or not isinstance(tool_calls[0], dict):
            raise ValueError("Expected exactly one tool call")
        payload, tool_id, steps = _action_from_tool_call(tool_calls[0])
        if payload is None:
            raise ValueError("Tool call did not contain a supported function")
        return payload, ["unwrap_tool_calls", *steps], tool_id

    payload, tool_id, steps = _action_from_tool_call(value)
    if payload is not None:
        return payload, steps, tool_id

    if any(key in value for key in ("$defs", "oneOf", "discriminator")):
        raise ValueError("Model returned a JSON schema instead of an action")
    if "parameters" in value and isinstance(value.get("parameters"), dict):
        raise ValueError("Model returned a tool definition instead of an action")
    return value, [], None


def parse_json_object(text: str) -> dict[str, Any]:
    """Backwards-compatible safe object parser used by existing callers/tests."""
    return _unwrap_action(_load_json_object(text))[0]


def _metadata(raw: RawModelResponse, stage: str, steps: list[str], payload: dict[str, Any] | None = None,
              tool_call_id: str | None = None, auxiliary: dict[str, Any] | None = None) -> DecisionMetadata:
    return DecisionMetadata(parse_stage=stage, normalization_steps=tuple(steps),
                            raw_content=raw.content, payload=payload, tool_call_id=tool_call_id,
                            auxiliary=auxiliary or {})


def _protocol_error(raw: RawModelResponse, code: str, message: str, stage: str,
                    steps: list[str] | None = None, payload: dict[str, Any] | None = None,
                    auxiliary: dict[str, Any] | None = None) -> ProtocolError:
    return ProtocolError(code=code, message=message,
                         metadata=_metadata(raw, stage, steps or [], payload, auxiliary=auxiliary))


def _split_auxiliary(payload: dict[str, Any], steps: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    auxiliary = {key: payload[key] for key in AUDIT_ONLY_FIELDS if key in payload}
    if auxiliary:
        payload = {key: value for key, value in payload.items() if key not in AUDIT_ONLY_FIELDS}
        steps.append("isolate_audit_metadata:" + ",".join(sorted(auxiliary)))
    return payload, auxiliary


def _evidence_alias(payload: dict[str, Any], steps: list[str]) -> dict[str, Any]:
    if payload.get("action") == "final_answer" and "evidence_timestamps" not in payload and "evidence" in payload:
        value = payload.get("evidence")
        if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, (int, float))
                                              or not math.isfinite(float(item)) for item in value):
            raise ValueError("evidence must be a numeric timestamp array; prose evidence is not executable")
        payload = dict(payload)
        payload["evidence_timestamps"] = payload.pop("evidence")
        steps.append("rename_numeric_evidence_to_evidence_timestamps")
    return payload


def normalize_model_response(raw: RawModelResponse,
                             observed: Sequence[Observation] | None = None) -> Decision:
    """Convert exactly one raw response into a typed, auditable decision."""
    if raw.refusal:
        return ProviderRefusal(raw.refusal, _metadata(raw, "provider_refusal", []))
    if not raw.content and raw.native_tool_calls:
        payload: dict[str, Any] | None = None
        tool_id: str | None = None
        steps = ["read_native_tool_calls"]
        try:
            if len(raw.native_tool_calls) != 1:
                return _protocol_error(raw, "tool_call_count", "Expected exactly one native tool call", "native_tools", steps)
            payload, tool_id, more = _action_from_tool_call(raw.native_tool_calls[0])
            steps.extend(more)
            if payload is None:
                return _protocol_error(raw, "tool_call_shape", "Native tool call did not contain a supported function", "native_tools", steps)
        except ValueError as exc:
            return _protocol_error(raw, "tool_call_arguments", str(exc), "native_tools", steps)
    else:
        if not raw.content:
            return _protocol_error(raw, "missing_content", "Model response contains no executable content", "response")
        try:
            payload, steps, tool_id = _unwrap_action(_load_json_object(raw.content))
        except ValueError as exc:
            message = str(exc)
            code = "duplicate_json_key" if message.startswith("Duplicate JSON key") else "invalid_json"
            return _protocol_error(raw, code, message, "json_decode")

    if not isinstance(payload, dict):
        return _protocol_error(raw, "not_object", "Model response must resolve to an object", "action_unwrap")

    payload, auxiliary = _split_auxiliary(payload, steps)
    status = payload.get("status")
    action = payload.get("action")
    if action is None:
        if status == "refused":
            return ProviderRefusal("Model returned an explicit refusal", _metadata(raw, "status", steps, payload,
                                                                                  auxiliary=auxiliary))
        if status == "answered" and "answer" in payload and "evidence_timestamps" in payload:
            payload = {**payload, "action": "final_answer", "status": "ok"}
            steps.append("infer_final_action_from_complete_answer")
        elif status in {"ok", "failed"} and "answer" in payload and "evidence_timestamps" in payload:
            payload = {**payload, "action": "final_answer"}
            steps.append("infer_final_action_from_complete_answer")
        else:
            return _protocol_error(raw, "missing_action", "Executable response is missing action", "action_dispatch", steps,
                                   payload, auxiliary)

    if payload.get("action") == "final_answer":
        if payload.get("status") == "refused":
            return ProviderRefusal("Model returned an explicit refusal", _metadata(raw, "status", steps, payload, tool_id,
                                                                                    auxiliary))
        if payload.get("status") == "answered":
            payload = {**payload, "status": "ok"}
            steps.append("normalize_answered_status")
        elif payload.get("status") == "success":
            return _protocol_error(raw, "unsupported_status", "status=success is not an accepted protocol status", "status", steps,
                                   payload, auxiliary)
        try:
            payload = _evidence_alias(payload, steps)
        except ValueError as exc:
            return _protocol_error(raw, "evidence_type", str(exc), "evidence", steps, payload, auxiliary)

    try:
        action_value = ACTION_ADAPTER.validate_python(payload)
    except (TypeError, ValueError) as exc:
        return _protocol_error(raw, "schema_validation", str(exc), "schema", steps, payload, auxiliary)

    metadata = _metadata(raw, "typed_decision", steps, payload, tool_id, auxiliary)
    if isinstance(action_value, (InspectFrame, SampleInterval, SelectCandidate)):
        return ToolDecision(action_value, metadata)
    if isinstance(action_value, FinalAction):
        final = FinalAnswer(answer=action_value.answer,
                            evidence_timestamps=action_value.evidence_timestamps,
                            status=action_value.status)
        return AnswerDecision(final, metadata)
    return _protocol_error(raw, "unsupported_action", "Action type is not supported", "schema", steps, payload)


def evidence_coverage(decision: Decision, frames: Sequence[Observation],
                      observed_before_response: Sequence[Observation] | None = None) -> dict[str, Any] | None:
    if not isinstance(decision, AnswerDecision):
        return None
    cited = decision.answer.evidence_timestamps
    observed = [frame.record.timestamp for frame in frames]
    all_observed = [frame.record.timestamp for frame in (observed_before_response or frames)]
    matched = [any(abs(timestamp - candidate) <= 1e-6 for candidate in observed) for timestamp in cited]
    matched_observed = [any(abs(timestamp - candidate) <= 1e-6 for candidate in all_observed) for timestamp in cited]
    return {"cited_timestamps": cited, "observed_timestamps_in_request": observed,
            "matched_citation_count": sum(matched),
            "all_cited_in_request": bool(cited) and all(matched),
            "observed_timestamps_before_response": all_observed,
            "all_cited_in_observed": bool(cited) and all(matched_observed)}
