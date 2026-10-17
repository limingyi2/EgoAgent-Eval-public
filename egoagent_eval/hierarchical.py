"""Independent hierarchical Active policy using a local OpenCV time index."""

import json

from .adaptive import _history_summary
from .index import VideoTimeIndex, build_time_index
from .records import ArtifactStore
from .schema import Budget, Question, RunResult
from .video import Observation, Video
from .vlm import VLMProvider
from .harness import IndexRetrievalStrategy, run_agent_harness
from .core.controller import (CandidateFirstHierarchicalRetrieval,
                              CandidateHierarchicalRetrieval, StagedObservationRetrieval,
                              default_prompt)


def _interval_text(windows) -> str:
    return "; ".join(f"[{window.start:.2f},{window.end:.2f}]"
                      for window in windows) or "none"


def _hierarchical_prompt(question: Question, video: Video, index: VideoTimeIndex,
                         remaining_calls: int, remaining_frames: int,
                         remaining_tool_calls: int, history: list[dict],
                         observed: list[Observation]) -> str:
    budget_line = ("Observation budget is exhausted. Return exactly one executable "
                   "final_answer JSON object; do not request a tool. "
                   if remaining_frames == 0 or remaining_tool_calls == 0 else
                   "You may request one sample_interval or inspect_frame action. ")
    contract = (
        'Return exactly one executable JSON object. Allowed actions are '
        '{"action":"inspect_frame","timestamp":12.5}, '
        '{"action":"sample_interval","start":10,"end":20,"count":4}, or '
        '{"action":"final_answer","answer":"...",'
        '"evidence_timestamps":[12.5,20.0],"status":"ok"}. '
        'Do not return a JSON Schema, Markdown, or prose wrapper. '
        'Cite only timestamps from frames actually observed. '
    )
    if remaining_frames == 0 or remaining_tool_calls == 0:
        contract = ('Return exactly one executable final_answer JSON object. '
                    'Use status="refused" with empty answer and evidence when evidence is insufficient. '
                    'Do not request a tool. ')
    return ("You are a hierarchical active video observer. Use only supplied frames. "
            + budget_line + contract
            + "Local index hints are retrieval hints only: visual change is not proof of an action. "
            "Keep global time coverage and search before or after observed points when needed. "
            f"Video duration={video.duration:g}s. Remaining model calls={remaining_calls}; "
            f"remaining extra frames={remaining_frames}; remaining tool calls={remaining_tool_calls}. "
            f"Uniform coverage windows: {_interval_text(index.uniform_windows)}. "
            f"Change/motion candidate windows: {_interval_text(index.candidate_windows)}. "
            f"Question: {json.dumps(question.question, ensure_ascii=False)}\n"
            + _history_summary(history, observed))


def _candidate_match(start: float, end: float, index: VideoTimeIndex) -> bool:
    return any(start < window.end and end > window.start for window in index.candidate_windows)


def _make_harness_prompt(index):
    # The existing hierarchical prompt remains the strategy-specific prompt;
    # the harness owns the loop and all state/budget transitions.
    def _harness_prompt(question, video, policy, remaining_calls, remaining_frames,
                        remaining_tools, history, observed, retrieval_hint, final_only):
        return _hierarchical_prompt(question, video, index, remaining_calls,
                                    remaining_frames, remaining_tools, history, observed)
    return _harness_prompt


def run_hierarchical_active(question: Question, video: Video, provider: VLMProvider,
                            initial: list[Observation] | None = None,
                            budget: Budget | None = None,
                            store: ArtifactStore | None = None,
                            index: VideoTimeIndex | None = None) -> RunResult:
    """Run hierarchical retrieval through the shared v2 harness."""
    index = index or build_time_index(video)
    return run_agent_harness(question, video, provider, method="hierarchical_active",
                             initial=initial, budget=budget, store=store,
                             context_policy="compact_context",
                             retrieval=IndexRetrievalStrategy(index),
                             prompt_builder=_make_harness_prompt(index))


def run_hierarchical_active_v2(question: Question, video: Video, provider: VLMProvider,
                               initial: list[Observation] | None = None,
                               budget: Budget | None = None,
                               store: ArtifactStore | None = None,
                               index: VideoTimeIndex | None = None) -> RunResult:
    """Run candidate-ID retrieval through the shared P1 AgentHarness loop."""
    retrieval = CandidateHierarchicalRetrieval(index)
    return run_agent_harness(question, video, provider, method="hierarchical_active_v2",
                             initial=initial, budget=budget, store=store,
                             context_policy="compact_context", retrieval=retrieval)


def _candidate_first_prompt(question, video, policy, remaining_calls, remaining_frames,
                            remaining_tools, history, observed, retrieval_hint, final_only):
    prompt = default_prompt(question, video, policy, remaining_calls, remaining_frames,
                            remaining_tools, history, observed, retrieval_hint, final_only)
    if not history and not final_only and remaining_frames > 0 and remaining_tools > 0:
        general = ('Return exactly one executable JSON object. Allowed actions: '
                   '{"action":"inspect_frame","timestamp":12.5}; '
                   '{"action":"select_candidate","candidate_id":"C<pool>-01","count":4}; '
                   '{"action":"sample_interval","start":10,"end":20,"count":4} '
                   '(freeform fallback only when no listed candidate fits); '
                   '{"action":"final_answer","answer":"...",'
                   '"evidence_timestamps":[12.5,20.0],"status":"ok"}. ')
        initial = ('Initial decision: select a listed candidate, answer, or refuse. '
                   'Do not request an arbitrary frame or interval yet. '
                   'Return exactly one executable JSON object. Allowed actions: '
                   '{"action":"select_candidate","candidate_id":"C<pool>-01","count":4}; '
                   '{"action":"final_answer","answer":"...",'
                   '"evidence_timestamps":[12.5,20.0],"status":"ok"}. ')
        if general not in prompt:
            raise ValueError("Candidate-first prompt could not locate the shared action contract")
        prompt = prompt.replace(general, initial, 1)
    return prompt


def run_hierarchical_active_v3_candidate_first(
        question: Question, video: Video, provider: VLMProvider,
        initial: list[Observation] | None = None,
        budget: Budget | None = None,
        store: ArtifactStore | None = None,
        index: VideoTimeIndex | None = None,
        candidate_sampling_policy: str = "uniform",
        candidate_context_padding: float = 0.0,
        local_index_samples: int = 24) -> RunResult:
    """Run candidate-first retrieval through the shared P1 AgentHarness loop."""
    retrieval = CandidateFirstHierarchicalRetrieval(index, candidate_sampling_policy,
                                                    candidate_context_padding, local_index_samples)
    return run_agent_harness(
        question, video, provider, method="hierarchical_active_v3_candidate_first",
        initial=initial, budget=budget, store=store,
        context_policy="compact_context", retrieval=retrieval,
        prompt_builder=_candidate_first_prompt)


_V4_CANDIDATE_SELECTION_POLICIES = {
    "baseline", "temporal_complementarity", "temporal_complementarity_ranked"
}


def _staged_observation_prompt(question, video, policy, remaining_calls, remaining_frames,
                               remaining_tools, history, observed, retrieval_hint, final_only,
                               candidate_selection_policy="baseline"):
    if candidate_selection_policy not in _V4_CANDIDATE_SELECTION_POLICIES:
        raise ValueError(f"Unknown v4 candidate selection policy: {candidate_selection_policy}")
    allowed = (remaining_frames >= 2 and remaining_tools > 0 and not final_only)
    if allowed:
        if history:
            contract = ('Return exactly one JSON object: select_candidate with a listed candidate_id and count=2; '
                        'or sample_interval with a legal start/end and count=2 only when no listed candidate fits; '
                        'or final_answer with evidence_timestamps. Do not supply start/end alongside candidate_id. ')
        else:
            contract = ('Return exactly one JSON object: select_candidate with a listed candidate_id and count=2; '
                        'or final_answer with evidence_timestamps. Do not supply candidate start/end times. ')
    else:
        contract = ('No further observation action is available. Return final_answer or refused only. ')
    timestamps = ", ".join(f"{frame.record.timestamp:g}" for frame in observed)
    wire_format = (
        ' Every response must be a single JSON object with a top-level string field "action"; '
        'do not wrap the action in a key named after the action. Use these exact field shapes: '
        '{"action":"select_candidate","candidate_id":"<exact listed candidate ID>","count":2}; '
        '{"action":"sample_interval","start":1.0,"end":2.0,"count":2}; '
        '{"action":"final_answer","answer":"REPLACE_WITH_ANSWER","evidence_timestamps":[1.0],"status":"ok"}; '
        'for refusal use {"action":"final_answer","answer":"","evidence_timestamps":[],"status":"refused"}. '
        'The candidate ID and answer strings are placeholders; replace them. The interval and evidence times are examples: '
        'choose a legal interval and cite only timestamps actually observed. Do not copy example values blindly. '
    )
    selection_guidance = ""
    if candidate_selection_policy == "temporal_complementarity":
        selection_guidance = (
            "\nCandidate selection policy: temporal_complementarity. Use the question, prior actions, "
            "observed frames, timestamps, and listed candidate metadata to identify what requested "
            "event or step remains visually unresolved. Compare each candidate interval and its listed "
            "remaining sampling positions with the observed time coverage. Prefer a candidate that can "
            "add useful coverage in a relevant unobserved portion or complement earlier observations; "
            "for multi-event or order questions, consider whether its interval and remaining positions "
            "can help inspect more than one unresolved part. Do not automatically prefer the widest "
            "interval, a partially observed candidate, or a high change score. Change scores are only "
            "visual-change hints and are not semantic relevance. Candidate state and remaining positions "
            "describe sampling progress, not evidence sufficiency. Continue only when the displayed frame, "
            "tool, and request budgets allow a useful observation and a later decision. If no legal, useful "
            "observation can be completed within the remaining budgets, refuse rather than imply that the "
            "actions were seen."
        )
    elif candidate_selection_policy == "temporal_complementarity_ranked":
        selection_guidance = (
            "\nCandidate selection policy: temporal_complementarity_ranked. The listed order is a "
            "deterministic Python retrieval ranking hint; you retain control of whether to select, "
            "answer, or refuse. Candidate scores are not evidence confidence. "
        )
    return ("You are a staged active video observer. First inspect two frames from a listed candidate. "
            + contract + wire_format + "Candidate windows are search leads, not action proof. Change scores are visual change/motion only. "
            + "Cite only timestamps from actually observed frames. "
            + retrieval_hint
            + selection_guidance
            + f"Video duration={video.duration:g}s; remaining model calls={remaining_calls}; "
            + f"remaining extra frames={remaining_frames}; remaining tool calls={remaining_tools}. "
            + f"Question: {json.dumps(question.question, ensure_ascii=False)}\n"
            + _history_summary(history, observed)
            + f"\nUnified observed timestamps: [{timestamps}]")


def run_hierarchical_active_v4_staged_observation(
        question: Question, video: Video, provider: VLMProvider,
        initial: list[Observation] | None = None,
        budget: Budget | None = None,
        store: ArtifactStore | None = None,
        index: VideoTimeIndex | None = None,
        candidate_selection_policy: str = "baseline") -> RunResult:
    """Experimental two-stage candidate observation using the sole P1 harness."""
    if candidate_selection_policy not in _V4_CANDIDATE_SELECTION_POLICIES:
        raise ValueError(f"Unknown v4 candidate selection policy: {candidate_selection_policy}")
    retrieval = StagedObservationRetrieval(index, candidate_selection_policy)
    def prompt_builder(*args):
        return _staged_observation_prompt(
            *args, candidate_selection_policy=candidate_selection_policy)

    return run_agent_harness(
        question, video, provider,
        method="hierarchical_active_v4_staged_observation",
        initial=initial, budget=budget, store=store,
        context_policy="full_context", retrieval=retrieval,
        prompt_builder=prompt_builder)
