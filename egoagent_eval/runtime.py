"""Shared result validation and accounting; no correctness grading is inferred."""

from time import perf_counter

from .records import utc_now
from .schema import ErrorRecord, FinalAnswer, Question, RunResult, Usage
from .video import Observation, Video, file_sha256
from .vlm import VLMProvider


def start_result(question: Question, method: str, provider: VLMProvider, video: Video,
                 initial: list[Observation]) -> RunResult:
    return RunResult(answer="", evidence_timestamps=[], status="failed", method=method,
                     question=question, provider=provider.name, model=provider.model, is_fake=provider.is_fake,
                     started_at_utc=utc_now(), video_sha256=file_sha256(video.path), video_duration=video.duration,
                     video_fps=video.fps, initial_frames=[f.record.model_copy() for f in initial],
                     observed_frames=[f.record.model_copy() for f in initial])


def apply_final(result: RunResult, final: FinalAnswer):
    observed = [f.timestamp for f in result.observed_frames]
    normalized = []
    for timestamp in final.evidence_timestamps:
        match = next((t for t in observed if abs(t - timestamp) <= 1e-6), None)
        if match is None:
            raise ValueError(f"Evidence timestamp {timestamp} was never observed")
        normalized.append(match)
    result.answer = final.answer
    result.evidence_timestamps = normalized
    result.status = final.status
    if final.status != "ok":
        result.errors.append(ErrorRecord(code="refused" if final.status == "refused" else "model_failed",
                                         message="Model reported " + final.status))


def finish_result(result: RunResult, provider: VLMProvider, first_call: int, started: float) -> RunResult:
    result.calls = provider.calls[first_call:]
    all_hashes = [frame.sha256 for call in result.calls for frame in call.frames]
    result.total_image_count = len(all_hashes)
    result.run_unique_image_count = len(set(all_hashes))
    seen: set[str] = set()
    result.run_reused_image_count = 0
    for call in result.calls:
        # Count only blocks repeated from an earlier request. Repeated blocks
        # inside the same request are covered by image_count vs unique count.
        call_hashes = [frame.sha256 for frame in call.frames]
        result.run_reused_image_count += sum(frame_hash in seen for frame_hash in call_hashes)
        seen.update(call_hashes)
    usage = {}
    for field in ("input_tokens", "output_tokens", "cached_tokens", "cost_usd"):
        values = [getattr(call.usage, field) for call in result.calls]
        usage[field] = sum(values) if values and all(v is not None for v in values) else None
    result.usage = Usage(**usage)
    result.latency_seconds = perf_counter() - started
    # Fake records and ungraded real records must never be treated as model scores.
    result.correct = None
    return result
