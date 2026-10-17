"""Typed outcomes produced by the shared model-response protocol layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .schema import CallRecord, FinalAnswer, InspectFrame, SampleInterval, SelectCandidate


@dataclass(frozen=True)
class RawModelResponse:
    """Provider output before any action or answer parsing.

    The provider owns transport and API-envelope validation.  The content of
    the model message remains opaque until ProtocolNormalizer handles it.
    """

    content: str | None
    raw_response: str | None = None
    refusal: str | None = None
    finish_reason: str | None = None
    native_tool_calls: tuple[dict[str, Any], ...] = ()
    call: CallRecord | None = None

    @classmethod
    def from_call(cls, call: CallRecord) -> "RawModelResponse":
        native = getattr(call, "native_tool_calls", None) or []
        return cls(content=call.content, raw_response=call.raw_response,
                   refusal=call.error.message if call.error and call.error.code == "refused" else None,
                   finish_reason=call.finish_reason,
                   native_tool_calls=tuple(native), call=call)


@dataclass(frozen=True)
class DecisionMetadata:
    parse_stage: str
    normalization_steps: tuple[str, ...] = ()
    raw_content: str | None = None
    payload: dict[str, Any] | None = None
    tool_call_id: str | None = None
    # Whitelisted non-execution fields are retained for auditability but are
    # deliberately removed before the strict action schema is validated.
    auxiliary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_stage": self.parse_stage,
            "normalization_steps": list(self.normalization_steps),
            "raw_content": self.raw_content,
            "payload": self.payload,
            "tool_call_id": self.tool_call_id,
            "auxiliary": self.auxiliary,
        }


@dataclass(frozen=True)
class ToolDecision:
    action: InspectFrame | SampleInterval | SelectCandidate
    metadata: DecisionMetadata

    @property
    def kind(self) -> Literal["tool"]:
        return "tool"


@dataclass(frozen=True)
class AnswerDecision:
    answer: FinalAnswer
    metadata: DecisionMetadata

    @property
    def kind(self) -> Literal["answer"]:
        return "answer"


@dataclass(frozen=True)
class ProviderRefusal:
    reason: str
    metadata: DecisionMetadata

    @property
    def kind(self) -> Literal["provider_refusal"]:
        return "provider_refusal"


@dataclass(frozen=True)
class ProtocolError:
    code: str
    message: str
    metadata: DecisionMetadata

    @property
    def kind(self) -> Literal["protocol_error"]:
        return "protocol_error"


Decision = ToolDecision | AnswerDecision | ProviderRefusal | ProtocolError


def decision_trace(decision: Decision) -> dict[str, Any]:
    """Return a JSON-safe append-only representation for protocol.jsonl."""
    data: dict[str, Any] = {"kind": decision.kind, "metadata": decision.metadata.to_dict()}
    if isinstance(decision, ToolDecision):
        data["action"] = decision.action.model_dump()
    elif isinstance(decision, AnswerDecision):
        data["answer"] = decision.answer.model_dump()
    elif isinstance(decision, ProviderRefusal):
        data["reason"] = decision.reason
    else:
        data.update({"code": decision.code, "message": decision.message})
    return data
