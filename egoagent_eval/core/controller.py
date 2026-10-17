"""Independent policies, retrieval, budget enforcement and evidence checks."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
from ..schema import Budget, FinalAnswer, InspectFrame, Question, SampleInterval, SelectCandidate
from ..video import Observation, Video

class RetrievalStrategy(Protocol):
    """Retrieval hints only; implementations must not read Oracle data."""

    def initial_frames(self, video: Video) -> list[Observation]: ...

    def prepare(self, video: Video, question: Question) -> Any: ...

    def prompt_context(self, context: Any, observed=None) -> str: ...

    def candidate_match(self, action: Any, context: Any) -> bool | None: ...

    def resolve_candidate(self, action: SelectCandidate, context: Any): ...


class ObservationPolicy(Protocol):
    name: str

    def select(self, initial: list[Observation], observed: list[Observation],
               new_frames: list[Observation]) -> list[Observation]: ...


ContextPolicy = ObservationPolicy


def _dedupe(frames: Sequence[Observation]) -> list[Observation]:
    seen: set[str] = set()
    output: list[Observation] = []
    for frame in frames:
        if frame.record.sha256 not in seen:
            output.append(frame)
            seen.add(frame.record.sha256)
    return output


class ContextObservationPolicy:
    """Full, selective and compact visual context policies."""

    def __init__(self, name: str = "selective_context"):
        if name not in {"full_context", "selective_context", "compact_context"}:
            raise ValueError(f"Unknown context policy: {name}")
        self.name = name

    def select(self, initial: list[Observation], observed: list[Observation],
               new_frames: list[Observation]) -> list[Observation]:
        if not new_frames and len(observed) == len(initial):
            return list(initial)
        if self.name == "full_context":
            return list(observed)
        if self.name == "selective_context":
            return _dedupe(initial[:1] + initial[-1:] + new_frames)
        return _dedupe(new_frames or (initial[:1] + initial[-1:]))


class UniformRetrieval:
    """Uniform global anchors; the model still decides whether to inspect."""
    def initial_frames(self, video: Video) -> list[Observation]:
        return video.uniform_frames(4)

    def prepare(self, video: Video, question: Question) -> None:
        return None

    def prompt_context(self, context: Any, observed=None) -> str:
        return ""

    def candidate_match(self, action: Any, context: Any) -> bool | None:
        return None


class AdaptiveRetrieval(UniformRetrieval):
    """Global anchors followed by model-selected observation actions."""


class HierarchicalRetrieval(AdaptiveRetrieval):
    """Adapter for an existing local VideoTimeIndex."""

    def __init__(self, index=None):
        self.index = index

    def prepare(self, video: Video, question: Question):
        if self.index is None:
            from ..index import build_time_index
            return build_time_index(video)
        return self.index

    def prompt_context(self, context: Any, observed=None) -> str:
        if context is None:
            return ""
        uniform = "; ".join(f"[{w.start:.2f},{w.end:.2f}]" for w in context.uniform_windows) or "none"
        candidates = "; ".join(f"[{w.start:.2f},{w.end:.2f}]" for w in context.candidate_windows) or "none"
        return ("Local index hints are retrieval hints only: visual change is not proof of an action. "
                "Keep global time coverage and search before or after observed points when needed. "
                f"Uniform coverage windows: {uniform}. Change/motion candidate windows: {candidates}. ")

    def candidate_match(self, action: Any, context: Any) -> bool | None:
        if context is None or not hasattr(action, "start") or not hasattr(action, "end"):
            return None
        return any(action.start < w.end and action.end > w.start for w in context.candidate_windows)

    def resolve_candidate(self, action: SelectCandidate, context: Any):
        raise ValueError("Hierarchical v1 does not execute candidate IDs")


class CandidateHierarchicalRetrieval(HierarchicalRetrieval):
    """Candidate pool strategy used only by hierarchical_active_v2."""

    def prepare(self, video: Video, question: Question):
        from ..index import build_time_index
        from ..retrieval import build_candidate_pool
        from ..video import file_sha256
        # A freshly built index already hashes the exact active video. Avoid
        # reading the entire file a second time; externally supplied snapshots
        # still require a freshness check before their candidate IDs are used.
        index = build_time_index(video) if self.index is None else self.index
        if Path(index.video_path).resolve() != Path(video.path).resolve():
            raise ValueError("Index video does not match the active video")
        if self.index is not None and index.video_sha256 != file_sha256(video.path):
            raise ValueError("Index snapshot is stale for the active video content")
        self.index = index
        return build_candidate_pool(index,
                                   change_context_padding=getattr(self, "change_context_padding", 0.0))

    def prompt_context(self, context: Any, observed=None) -> str:
        seen = observed or []
        uniform = sorted((c for c in context.candidates if c.source == "uniform_coverage"),
                         key=lambda c: (c.start, c.end, c.candidate_id))
        changes = sorted((c for c in context.candidates if c.source != "uniform_coverage"),
                         key=lambda c: (-(c.score if c.score is not None else -math.inf),
                                        c.start, c.end, c.candidate_id))

        def observed_status(candidate):
            return "seen" if any(candidate.start <= frame.record.timestamp <= candidate.end
                                 for frame in seen) else "new"

        uniform_entries = []
        for candidate in uniform:
            uniform_entries.append(
                f"{candidate.candidate_id} [{candidate.start:.2f},{candidate.end:.2f}] "
                f"{observed_status(candidate)}")
        change_entries = []
        for rank, candidate in enumerate(changes, start=1):
            score = "n/a" if candidate.score is None else f"{candidate.score:.4f}"
            change_entries.append(
                f"{rank}. {candidate.candidate_id} [{candidate.start:.2f},{candidate.end:.2f}] "
                f"score={score} {observed_status(candidate)}")
        uniform_listing = "; ".join(uniform_entries) or "none"
        change_listing = "; ".join(change_entries) or "none"
        return ("Candidate intervals are search leads, not action labels or proof. "
                "Prefer a listed candidate when its interval suits the missing evidence; "
                "use freeform sample_interval only when no listed candidate fits. "
                "You may also answer or refuse. Do not invent candidate IDs. "
                "Change scores indicate visual change/motion only, not semantic relevance. "
                f"Pool={context.candidate_pool_id}. Uniform coverage: {uniform_listing}. "
                f"Change/motion ranked by score descending: {change_listing}.\n")

    def resolve_candidate(self, action: SelectCandidate, context: Any):
        if context is None or not hasattr(context, "get"):
            raise ValueError("Candidate pool is unavailable")
        candidate = context.get(action.candidate_id)
        if candidate is None:
            raise ValueError("candidate_id is not present in this run's candidate pool")
        if not (0 <= candidate.start < candidate.end <= context.video_duration):
            raise ValueError("Candidate lies outside this video's duration")
        return candidate

    def sample_candidate(self, tools, action: SelectCandidate, candidate, context):
        points = getattr(self.index, "coarse_points", ()) if self.index is not None else ()
        return tools.sample_candidate(action, candidate, context.candidate_pool_id,
                                      sampling_policy=getattr(self, "sampling_policy", "uniform"),
                                      change_points=points,
                                      local_index_samples=getattr(self, "local_index_samples", 24))

    def candidate_match(self, action: Any, context: Any) -> bool | None:
        if context is None or not hasattr(action, "start") or not hasattr(action, "end"):
            return None
        return any(action.start < candidate.end and action.end > candidate.start
                   for candidate in context.candidates)


class CandidateFirstHierarchicalRetrieval(CandidateHierarchicalRetrieval):
    """Candidate-first v3 policy; freeform observation is enabled after a tool."""

    def __init__(self, index=None, sampling_policy: str = "uniform",
                 change_context_padding: float = 0.0, local_index_samples: int = 24):
        if sampling_policy not in {"uniform", "change_peaks", "hybrid_change"}:
            raise ValueError(f"Unknown candidate sampling policy: {sampling_policy}")
        if (isinstance(change_context_padding, bool)
                or not isinstance(change_context_padding, (int, float))
                or not math.isfinite(change_context_padding)
                or not 0 <= change_context_padding <= 2):
            raise ValueError("change_context_padding must be in [0, 2]")
        if type(local_index_samples) is not int or not 4 <= local_index_samples <= 128:
            raise ValueError("local_index_samples must be in [4, 128]")
        super().__init__(index)
        self.sampling_policy = sampling_policy
        self.change_context_padding = float(change_context_padding)
        self.local_index_samples = local_index_samples

    def prompt_context(self, context: Any, observed=None) -> str:
        return (super().prompt_context(context, observed)
                + "For multi-action or order questions, prefer a candidate interval broad enough to cover the full sequence. "
                + "Uniform windows preserve broader time coverage; change/motion scores only describe visual variation. ")

    def action_allowed(self, action: Any, history: list[dict]) -> bool:
        # The initial decision may choose a candidate, answer, or refuse. Final
        # answers are handled before this gate; arbitrary frame tools are not.
        if not history:
            return isinstance(action, SelectCandidate)
        return True


class StagedObservationRetrieval(CandidateHierarchicalRetrieval):
    """Candidate-ID retrieval for the isolated two-stage observation prototype."""

    staged_observation = True

    def __init__(self, index=None, candidate_selection_policy: str = "baseline"):
        if candidate_selection_policy not in {"baseline", "temporal_complementarity",
                                               "temporal_complementarity_ranked"}:
            raise ValueError(f"Unknown v4 candidate selection policy: {candidate_selection_policy}")
        super().__init__(index)
        self.candidate_selection_policy = candidate_selection_policy

    def action_allowed(self, action: Any, history: list[dict]) -> bool:
        # Stage one must be anchored to the shared candidate pool. After that,
        # a bounded Python-validated interval is a controlled fallback.
        if not history:
            return isinstance(action, SelectCandidate)
        return isinstance(action, (SelectCandidate, SampleInterval))

    def rank_candidates(self, context: Any, observed=None, *, remaining_frames: int = 4,
                        remaining_tools: int = 2) -> list[dict[str, Any]]:
        """Score executable candidates from runtime-only progress and pool metadata.

        Scores are deterministic retrieval hints. No question labels, Oracle data,
        or action annotations are accepted by this function.
        """
        seen = list(observed or [])
        fps = float(getattr(self.index, "fps", 0) or 0)
        if fps <= 0:
            raise ValueError("Candidate ranking requires a prepared video index with FPS")
        if (getattr(self.index, "video_sha256", None)
                and self.index.video_sha256 != context.video_sha256):
            raise ValueError("Candidate pool video hash does not match the active index")
        duration = max(float(getattr(context, "video_duration", 0) or 0), 1e-9)
        requestable = 2 if remaining_frames >= 2 and remaining_tools > 0 else 0
        snapshots = {candidate.candidate_id: self.candidate_observation_snapshot(
            candidate, seen, fps=fps, video_sha256=context.video_sha256)
            for candidate in context.candidates}
        scores = [candidate.score for candidate in context.candidates if candidate.score is not None]
        low, high = (min(scores), max(scores)) if scores else (0.0, 0.0)
        rows = []
        anchors = [float(frame.record.timestamp) for frame in seen]
        for candidate in context.candidates:
            if context.get(candidate.candidate_id) != candidate:
                raise ValueError("Candidate ID is not present in the active pool")
            if not (0 <= candidate.start < candidate.end <= duration):
                raise ValueError("Candidate interval lies outside the active video's bounds")
            snapshot = snapshots[candidate.candidate_id]
            plan = snapshot["sampling_plan"]
            remaining = snapshot["remaining_sampling_positions"]
            availability = len(remaining) / max(len(plan), 1)
            if anchors and remaining:
                scale = max(duration * 0.1, 1e-9)
                novelty = sum(min(1.0, min(abs(float(slot["timestamp"]) - point)
                                           for point in anchors) / scale)
                              for slot in remaining) / len(remaining)
            else:
                novelty = 1.0 if remaining else 0.0
            budget_fit = (min(len(remaining), requestable) / 2.0) if requestable else 0.0
            change = 0.5 if candidate.score is None or high == low else (
                (float(candidate.score) - low) / (high - low))
            eligible = len(remaining) >= 2 and requestable == 2
            score = 0.45 * novelty + 0.25 * availability + 0.20 * budget_fit + 0.10 * change
            rows.append({
                "candidate_id": candidate.candidate_id,
                "candidate_pool_id": context.candidate_pool_id,
                "video_sha256": context.video_sha256,
                "start": float(candidate.start), "end": float(candidate.end),
                "source": candidate.source, "change_score": candidate.score,
                "state": snapshot["state"],
                "remaining_positions": len(remaining),
                "remaining_timestamps": [float(slot["timestamp"]) for slot in remaining],
                "temporal_novelty": round(novelty, 8),
                "availability": round(availability, 8),
                "budget_fit": round(budget_fit, 8),
                "normalized_change": round(change, 8),
                "rank_score": round(score, 8), "eligible": eligible,
            })
        return sorted(rows, key=lambda row: (-row["rank_score"], row["start"],
                                             row["end"], row["candidate_id"]))

    def select_candidate_python(self, context: Any, observed=None, *,
                                remaining_frames: int, remaining_tools: int):
        """Return the highest-ranked legal candidate for offline policy comparison."""
        from ..schema import SelectCandidate
        ranked = self.rank_candidates(context, observed, remaining_frames=remaining_frames,
                                      remaining_tools=remaining_tools)
        first = next((row for row in ranked if row["eligible"]), None)
        return (SelectCandidate(action="select_candidate", candidate_id=first["candidate_id"], count=2)
                if first else None)

    def prompt_context(self, context: Any, observed=None, *, ranking=None) -> str:
        seen = observed or []
        if self.candidate_selection_policy == "temporal_complementarity_ranked":
            rows = ranking or self.rank_candidates(context, seen)
            entries = []
            for rank, row in enumerate(rows, 1):
                positions = ",".join(f"{timestamp:.2f}" for timestamp in row["remaining_timestamps"])
                score = ("n/a" if row["change_score"] is None
                         else f"{row['change_score']:.4f}")
                entries.append(
                    f"{rank}. {row['candidate_id']} [{row['start']:.2f},{row['end']:.2f}] "
                    f"source={row['source']} change={score} rank_score={row['rank_score']:.4f} "
                    f"state={row['state']} remaining={row['remaining_positions']} "
                    f"eligible={str(row['eligible']).lower()} positions=[{positions}]")
            return ("Candidate intervals are search leads, not action labels or proof. "
                    "The ordered list is a deterministic Python ranking hint; the model still chooses "
                    "whether to select a candidate, answer, or refuse. Scores use temporal novelty, "
                    "remaining legal positions, requestable observation budget, and normalized visual "
                    "change only. Candidate state describes sampling progress, not evidence sufficiency. "
                    f"Pool={context.candidate_pool_id}. Python-ranked candidates: "
                    + ("; ".join(entries) or "none") + ".\n")
        uniform = sorted((c for c in context.candidates if c.source == "uniform_coverage"),
                         key=lambda c: (c.start, c.end, c.candidate_id))
        changes = sorted((c for c in context.candidates if c.source != "uniform_coverage"),
                         key=lambda c: (-(c.score if c.score is not None else -math.inf),
                                        c.start, c.end, c.candidate_id))

        def entry(candidate, rank=None):
            snapshot = self.candidate_observation_snapshot(candidate, seen)
            prefix = f"{rank}. " if rank is not None else ""
            score = (f" score={candidate.score:.4f}" if rank is not None and candidate.score is not None
                     else " score=n/a" if rank is not None else "")
            positions = ",".join(f"{slot['timestamp']:.2f}"
                                  for slot in snapshot["remaining_sampling_positions"])
            return (f"{prefix}{candidate.candidate_id} [{candidate.start:.2f},{candidate.end:.2f}]"
                    f"{score} state={snapshot['state']} remaining={len(snapshot['remaining_sampling_positions'])}"
                    f" positions=[{positions}]")

        uniform_listing = "; ".join(entry(candidate) for candidate in uniform) or "none"
        change_listing = "; ".join(entry(candidate, rank) for rank, candidate in enumerate(changes, 1)) or "none"
        return ("Candidate intervals are search leads, not action labels or proof. "
                "Prefer a listed candidate when its interval suits the missing evidence; "
                "use freeform sample_interval only when no listed candidate fits. "
                "You may also answer or refuse. Do not invent candidate IDs. "
                "Change scores indicate visual change/motion only, not semantic relevance. "
                f"Pool={context.candidate_pool_id}. Uniform coverage: {uniform_listing}. "
                f"Change/motion ranked by score descending: {change_listing}.\n"
                "Candidate state describes sampling progress only, not evidence sufficiency. "
                "partially_observed candidates remain selectable while planned positions remain; "
                "exhausted means no executable planned position remains. "
                "This is a two-stage observation run. Each select_candidate action must use count=2. "
                "Candidate IDs and their exact boundaries are validated by Python. "
                "After the first two frames, choose another listed candidate, answer, or refuse. ")

    @staticmethod
    def _sampling_plan(candidate, fps: float) -> list[dict[str, float | int]]:
        """Return a fixed, reproducible 16-cell interior plan per candidate.

        The ordinary first two uniform midpoints (fractions .25 and .75) are
        included in this plan. The remaining fractions provide legal,
        deterministic positions for a later selection of the same candidate.
        """
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Candidate sampling plan requires a positive finite video FPS")
        by_frame: dict[int, dict[str, float | int]] = {}
        for slot in range(1, 16):
            fraction = slot / 16
            timestamp = candidate.start + fraction * (candidate.end - candidate.start)
            if not candidate.start <= timestamp < candidate.end:
                continue
            frame_index = math.floor(timestamp * fps)
            by_frame.setdefault(frame_index, {"timestamp": float(timestamp),
                                              "frame_index": int(frame_index),
                                              "fraction": fraction})
        return sorted(by_frame.values(), key=lambda item: (item["timestamp"], item["frame_index"]))

    def candidate_observation_snapshot(self, candidate, observed, *, fps: float | None = None,
                                       video_sha256: str | None = None) -> dict[str, Any]:
        """Derive progress and remaining legal plan positions without Oracle data."""
        fps = fps if fps is not None else getattr(self.index, "fps", None)
        if fps is None:
            raise ValueError("Candidate sampling state requires the prepared video index")
        plan = self._sampling_plan(candidate, float(fps))
        observed_indices = {frame.record.frame_index for frame in observed}
        in_candidate = [frame for frame in observed
                        if candidate.start <= frame.record.timestamp < candidate.end]
        excluded = [slot for slot in plan if slot["frame_index"] in observed_indices]
        remaining = [slot for slot in plan if slot["frame_index"] not in observed_indices]
        state = ("exhausted" if not remaining else
                 "partially_observed" if in_candidate else "unseen")
        return {
            "candidate_id": candidate.candidate_id,
            "video_sha256": (video_sha256 if video_sha256 is not None
                             else getattr(self.index, "video_sha256", None)),
            "original_interval": [float(candidate.start), float(candidate.end)],
            "state": state,
            "previous_observed_timestamps": sorted({float(frame.record.timestamp)
                                                     for frame in in_candidate}),
            "sampling_plan": [dict(slot) for slot in plan],
            "duplicate_excluded_positions": [dict(slot) for slot in excluded],
            "remaining_sampling_positions": [dict(slot) for slot in remaining],
        }

    @staticmethod
    def _select_remaining_positions(snapshot: dict[str, Any], observed,
                                    count: int) -> list[float]:
        remaining = list(snapshot["remaining_sampling_positions"])
        if count < 1 or len(remaining) < count:
            return []
        if snapshot["state"] == "unseen":
            preferred_fractions = (0.25, 0.75)
            preferred = [slot for fraction in preferred_fractions
                         for slot in remaining
                         if math.isclose(float(slot["fraction"]), fraction, abs_tol=1e-12)]
            if len(preferred) >= count:
                return [float(slot["timestamp"]) for slot in preferred[:count]]

        # Farthest-first over the fixed candidate plan. This explores unobserved
        # portions deterministically while excluding every already decoded frame.
        anchors = [float(frame.record.timestamp) for frame in observed]
        selected: list[dict[str, float | int]] = []
        while remaining and len(selected) < count:
            reference = anchors + [float(slot["timestamp"]) for slot in selected]
            if reference:
                choice = max(remaining, key=lambda slot: (
                    min(abs(float(slot["timestamp"]) - point) for point in reference),
                    -float(slot["timestamp"])))
            else:
                choice = remaining[0]
            selected.append(choice)
            remaining.remove(choice)
        return [float(slot["timestamp"]) for slot in selected]

    def sample_candidate(self, tools, action: SelectCandidate, candidate, context, *, observed=()):
        """Sample two remaining positions from a finite per-candidate plan."""
        before = self.candidate_observation_snapshot(candidate, observed,
                                                      fps=tools.video.fps,
                                                      video_sha256=context.video_sha256)
        selected = self._select_remaining_positions(before, observed, action.count)
        audit = {**before,
                 "previous_observed_frame_indices": sorted({int(frame.record.frame_index)
                                                             for frame in observed
                                                             if candidate.start <= frame.record.timestamp < candidate.end}),
                 "new_sampling_positions": selected,
                 "duplicate_frames_excluded": bool(before["duplicate_excluded_positions"]),
                 "duplicate_reason": ("decoded frame index already observed; omitted from remaining plan"
                                      if before["duplicate_excluded_positions"] else None),
                 "execution_error": None}
        if not selected:
            code = ("candidate_exhausted" if not before["remaining_sampling_positions"]
                    else "insufficient_remaining_positions")
            message = ("Candidate sampling plan is exhausted" if code == "candidate_exhausted"
                       else "Fewer than two executable positions remain in candidate sampling plan")
            audit["execution_error"] = code
            output = tools.reject_candidate(action, code, message, context.candidate_pool_id)
            output.record.arguments["candidate_observation"] = audit
            return output
        arguments = {"candidate_id": action.candidate_id,
                     "candidate_pool_id": context.candidate_pool_id,
                     "start": candidate.start, "end": candidate.end,
                     "count": action.count, "source": candidate.source, "score": candidate.score,
                     "sampling_policy": "uniform_progressive_16_cell",
                     "requested_timestamps": selected,
                     "duplicate_avoidance": "exclude_observed_frame_indices",
                     "duplicate_reason": audit["duplicate_reason"],
                     "candidate_observation": audit}
        from ..schema import SampleInterval
        output = tools.execute(SampleInterval(action="sample_interval", start=candidate.start,
                                              end=candidate.end, count=action.count),
                               record_tool="select_candidate", record_arguments=arguments,
                               requested_timestamps=selected)
        after = self.candidate_observation_snapshot(candidate, list(observed) + output.frames,
                                                     fps=tools.video.fps,
                                                     video_sha256=context.video_sha256)
        audit["state_after"] = after["state"]
        audit["remaining_sampling_positions_after"] = after["remaining_sampling_positions"]
        audit["execution_error"] = (output.record.error.code if output.record.error else None)
        output.record.arguments["candidate_observation"] = audit
        return output

    def sample_interval(self, tools, action: SampleInterval, *, observed=()):
        """Execute the stage-two freeform fallback inside Python-checked video bounds."""
        selected, adjustment = self._novel_uniform_points(tools, action.start, action.end, observed)
        if not selected:
            return tools.reject_action(action, "duplicate_observation_unavoidable",
                                       "Could not find two novel video frames in requested interval")
        return tools.execute(action, record_arguments={"start": action.start, "end": action.end,
                             "count": 2, "sampling_policy": "uniform",
                             "requested_timestamps": selected, "source": "freeform_fallback",
                             "duplicate_avoidance": adjustment, "duplicate_reason": None},
                             requested_timestamps=selected)

    @staticmethod
    def _novel_uniform_points(tools, start: float, end: float, observed):
        from ..sampling import sample_interval_timestamps
        # Start with the ordinary two-bin uniform points. If video frame
        # quantization or a narrow candidate collides with prior observations,
        # search a denser uniform lattice and keep the most separated novel
        # pair. This is a generic duplicate-avoidance rule, not task-specific.
        points = sample_interval_timestamps(start, end, 2)
        existing_indices = {frame.record.frame_index for frame in observed}

        def novel(timestamp):
            return math.floor(timestamp * tools.video.fps) not in existing_indices

        lattice = sample_interval_timestamps(start, end, 8)
        alternatives = sorted({point for point in points + lattice if novel(point)})
        pairs = [(left, right) for i, left in enumerate(alternatives)
                 for right in alternatives[i + 1:]
                 if math.floor(left * tools.video.fps) != math.floor(right * tools.video.fps)]
        selected = list(min(pairs, key=lambda pair: (abs(pair[0] - points[0])
                                                     + abs(pair[1] - points[1]), pair))) if pairs else []
        adjustment = "uniform_lattice_8" if selected and list(points) != selected else "none"
        return selected, adjustment


PromptBuilder = Callable[[Question, Video, ObservationPolicy, int, int, int, list[dict],
                          list[Observation], str, bool], str]


def default_prompt(question: Question, video: Video, policy: ObservationPolicy,
                   remaining_calls: int, remaining_frames: int, remaining_tools: int,
                   history: list[dict], observed: list[Observation], retrieval_hint: str,
                   final_only: bool) -> str:
    if final_only or remaining_frames == 0 or remaining_tools == 0:
        contract = ('Return exactly one executable final_answer JSON object. Use status="refused" '
                    'with an empty answer and empty evidence_timestamps when evidence is insufficient. ')
    elif "Candidate pool=" in retrieval_hint or "Pool=" in retrieval_hint:
        contract = ('Return exactly one executable JSON object. Allowed actions: '
                    '{"action":"inspect_frame","timestamp":12.5}; '
                    '{"action":"select_candidate","candidate_id":"C<pool>-01","count":4}; '
                    '{"action":"sample_interval","start":10,"end":20,"count":4} '
                    '(freeform fallback only when no listed candidate fits); '
                    '{"action":"final_answer","answer":"...",'
                    '"evidence_timestamps":[12.5,20.0],"status":"ok"}. ')
    else:
        contract = ('Return exactly one executable JSON object. Allowed actions: '
                    '{"action":"inspect_frame","timestamp":12.5}; '
                    '{"action":"sample_interval","start":10,"end":20,"count":4}; '
                    '{"action":"final_answer","answer":"...",'
                    '"evidence_timestamps":[12.5,20.0],"status":"ok"}. ')
    timestamps = ", ".join(f"{f.record.timestamp:g}s/{f.record.sha256[:8]}" for f in observed)
    actions = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
    return ("You are an active video observer. Use only supplied frames. " + contract +
            "Do not return a JSON Schema, tool definition, Markdown, or prose wrapper. "
            "Cite only timestamps from frames you have actually observed. " + retrieval_hint +
            f"Video duration={video.duration:g}s. Context policy={policy.name}. "
            f"Remaining model calls={remaining_calls}; remaining extra frames={remaining_frames}; "
            f"remaining tool calls={remaining_tools}. "
            f"Question: {json.dumps(question.question, ensure_ascii=False)}\n"
            f"Observed timestamps and frame hashes: [{timestamps}]\nPrior actions: {actions}")




class FullContext(ContextObservationPolicy):
    def __init__(self):
        super().__init__("full_context")


class SelectiveContext(ContextObservationPolicy):
    def __init__(self):
        super().__init__("selective_context")


class CompactContext(ContextObservationPolicy):
    def __init__(self):
        super().__init__("compact_context")


@dataclass(frozen=True)
class BudgetSnapshot:
    model_calls: int
    tool_calls: int
    extra_frames: int

    def as_dict(self) -> dict[str, int]:
        return {"model_calls": self.model_calls, "tool_calls": self.tool_calls,
                "extra_frames": self.extra_frames}


class BudgetManager:
    """Authoritative per-run counters, independent of provider record length.

    Four initial plus max_extra_frames is the observation acquisition limit.
    Re-sent images count as transfer cost, not new observations; each request
    still has an eight-image cap. Rejected attempts are recorded separately.
    """
    def __init__(self, budget: Budget, max_request_images: int = 8):
        self.budget = budget
        self.max_request_images = max_request_images
        self.requests = 0
        self.tool_attempts = 0
        self.reserved_frames = 0

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(max(0, self.budget.max_model_calls - self.requests),
                              max(0, self.budget.max_tool_calls - self.tool_attempts),
                              max(0, self.budget.max_extra_frames - self.reserved_frames))

    def consume_request(self, frames) -> None:
        if self.snapshot().model_calls == 0:
            raise ValueError("Model request budget exhausted")
        if not 1 <= len(frames) <= self.max_request_images:
            raise ValueError("Request image budget exceeded")
        if any(max(f.record.width, f.record.height) > self.budget.max_image_side for f in frames):
            raise ValueError("Request image dimensions exceed budget")
        self.requests += 1

    def tool_rejection(self, action) -> str | None:
        count = 1 if isinstance(action, InspectFrame) else action.count
        if self.snapshot().tool_calls == 0:
            return "tool_call_budget"
        if count > self.snapshot().extra_frames:
            return "frame_budget"
        return None

    def record_rejected_tool(self) -> None:
        self.tool_attempts += 1

    def reserve_tool(self, action) -> None:
        rejection = self.tool_rejection(action)
        if rejection:
            raise ValueError(rejection)
        self.tool_attempts += 1
        self.reserved_frames += 1 if isinstance(action, InspectFrame) else action.count


@dataclass(frozen=True)
class EvidenceCheck:
    traceable: bool
    structurally_complete: bool
    reason: str


class EvidenceValidator:
    """Validate membership and structural coverage, never visual semantics.

    The optional required_action_count is a runtime structural requirement,
    not an Oracle interval. Legacy English prompts default to a two-stage
    gate for sequence/state changes. Actual action visibility requires review.
    """
    @staticmethod
    def needs_multiple(question: str) -> bool:
        text = question.lower()
        return any(marker in text for marker in (
            "order", "sequence", "then", "after attaching", "what does the wearer do",
            "tap", "turned on", "turned off", "state change",
        ))

    def validate(self, question: Question, answer: FinalAnswer, observed,
                 duration: float, required_action_count: int | None = None) -> EvidenceCheck:
        timestamps = answer.evidence_timestamps
        available = [f.record.timestamp for f in observed]
        if any(isinstance(t, bool) or not math.isfinite(t) or t < 0 or t >= duration
               or not any(abs(t - actual) <= 1e-6 for actual in available) for t in timestamps):
            return EvidenceCheck(False, False, "invalid or unobserved timestamp")
        required = required_action_count or (2 if self.needs_multiple(question.question) else 1)
        complete = len(set(timestamps)) >= required
        return EvidenceCheck(True, complete, "structural coverage only" if complete else "missing evidence stages")
