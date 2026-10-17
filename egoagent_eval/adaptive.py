"""Lightweight adaptive Active policy with explicit visual-context controls.

This module is intentionally separate from :mod:`egoagent_eval.agent` so the
original Active-4+4 implementation and its historical traces remain stable.
"""

import json
from typing import Literal

from .records import ArtifactStore
from .decisions import (AnswerDecision, ProtocolError, ProviderRefusal,
                        RawModelResponse, ToolDecision, decision_trace)
from .protocol import evidence_coverage, normalize_model_response
from .schema import Budget, Question, RunResult
from .video import Observation, Video
from .vlm import VLMProvider
from .harness import run_agent_harness
from .core.controller import AdaptiveRetrieval


ContextPolicy = Literal["full_context", "selective_context", "compact_context"]


def _validate_action(content: str):
    """Backwards-compatible helper returning only a typed executable action."""
    decision = normalize_model_response(RawModelResponse(content=content))
    if isinstance(decision, ToolDecision):
        return decision.action
    if isinstance(decision, AnswerDecision):
        return decision.answer
    if isinstance(decision, ProviderRefusal):
        raise ValueError(decision.reason)
    raise ValueError(f"{decision.code}: {decision.message}")


def _context_frames(policy, initial, observed, new_frames):
    from .core.controller import ContextObservationPolicy
    return ContextObservationPolicy(policy).select(initial, observed, new_frames)


def _needs_multiple_evidence(question):
    from .core.controller import EvidenceValidator
    return EvidenceValidator.needs_multiple(question)


def _evidence_is_structurally_sufficient(question, action, observed):
    from .core.controller import EvidenceValidator
    duration = max((f.record.timestamp for f in observed), default=0.0) + 1.0
    return EvidenceValidator().validate(question, action, observed, duration).structurally_complete


def _record_protocol(store: ArtifactStore | None, call, decision, frames):
    coverage = evidence_coverage(decision, frames)
    call.evidence_coverage = coverage
    if store:
        store.append_protocol({"call_index": call.index,
                               "decision": decision_trace(decision),
                               "evidence_coverage": coverage})


def _history_summary(history: list[dict], observed: list[Observation]) -> str:
    timestamps = ", ".join(f"{f.record.timestamp:g}s/{f.record.sha256[:8]}"
                           for f in observed)
    actions = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
    return f"Observed timestamps and frame hashes: [{timestamps}]\nPrior actions: {actions}"


def _prompt(question: Question, video: Video, policy: ContextPolicy,
            remaining_calls: int, remaining_frames: int, remaining_tool_calls: int,
            history: list[dict], observed: list[Observation]) -> str:
    if remaining_frames > 0 and remaining_tool_calls > 0:
        contract = (
            'Return exactly one executable JSON object. Allowed actions: '
            '{"action":"inspect_frame","timestamp":12.5}; '
            '{"action":"sample_interval","start":10,"end":20,"count":4}; '
            '{"action":"final_answer","answer":"...",'
            '"evidence_timestamps":[12.5,20.0],"status":"ok"}. '
        )
    else:
        contract = (
            'Observation budget is exhausted. Return exactly one executable '
            'final_answer JSON object; do not request a tool. You may use '
            'status="refused" with an empty answer and empty evidence_timestamps '
            'when the observed evidence is insufficient. '
        )
    contract += ('Do not return a JSON Schema, tool definition, Markdown, or prose wrapper. '
                 'Cite only timestamps from frames you have actually observed.')
    return (f"You are an active video observer. Use only supplied frames. {contract} "
            f"Video duration={video.duration:g}s. Context policy={policy}. "
            f"Remaining model calls={remaining_calls}; remaining extra frames={remaining_frames}; "
            f"remaining tool calls={remaining_tool_calls}. "
            f"Question: {json.dumps(question.question, ensure_ascii=False)}\n"
            + _history_summary(history, observed))


def _harness_prompt(question, video, policy, remaining_calls, remaining_frames,
                    remaining_tools, history, observed, retrieval_hint, final_only):
    return _prompt(question, video, policy.name, remaining_calls, remaining_frames,
                   remaining_tools, history, observed)


def run_adaptive_active(question: Question, video: Video, provider: VLMProvider,
                        initial: list[Observation] | None = None,
                        budget: Budget | None = None,
                        store: ArtifactStore | None = None,
                        context_policy: ContextPolicy = "selective_context") -> RunResult:
    """Run Adaptive through the shared v2 harness."""
    return run_agent_harness(question, video, provider, method="adaptive_active",
                             initial=initial, budget=budget, store=store,
                             context_policy=context_policy, retrieval=AdaptiveRetrieval(), prompt_builder=_harness_prompt)
