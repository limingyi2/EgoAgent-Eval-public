"""Deterministic candidate pools derived only from a local video time index."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .index import CandidateWindow, VideoTimeIndex


@dataclass(frozen=True)
class CandidateInterval:
    candidate_id: str
    start: float
    end: float
    source: str
    score: float | None

    def __post_init__(self):
        if not self.candidate_id or not self.source:
            raise ValueError("candidate_id and source are required")
        if not math.isfinite(self.start) or not math.isfinite(self.end) or not self.start < self.end:
            raise ValueError("Candidate interval must have finite start < end")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("Candidate score must be finite")


@dataclass(frozen=True)
class CandidatePool:
    candidate_pool_id: str
    index_version: str
    video_path: str
    video_sha256: str
    video_duration: float
    candidates: tuple[CandidateInterval, ...]
    scanned_frames: int
    cpu_seconds: float
    context_padding_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_pool_id": self.candidate_pool_id,
            "index_version": self.index_version,
            "video_path": self.video_path,
            "video_sha256": self.video_sha256,
            "video_duration": self.video_duration,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "scanned_frames": self.scanned_frames,
            "cpu_seconds": self.cpu_seconds,
            "context_padding_ratio": self.context_padding_ratio,
            "score_semantics": "relative OpenCV visual change/motion only; not semantic confidence",
        }

    def get(self, candidate_id: str) -> CandidateInterval | None:
        return next((c for c in self.candidates if c.candidate_id == candidate_id), None)


def build_candidate_pool(index: VideoTimeIndex, *, change_context_padding: float = 0.0) -> CandidatePool:
    """Build a stable pool and IDs scoped to this exact index snapshot."""
    if (isinstance(change_context_padding, bool) or not isinstance(change_context_padding, (int, float))
            or not math.isfinite(change_context_padding) or not 0 <= change_context_padding <= 2):
        raise ValueError("change_context_padding must be in [0, 2]")
    path = str(Path(index.video_path).resolve())
    ordered: list[CandidateWindow] = []
    ordered.extend(sorted(index.uniform_windows, key=lambda w: (w.start, w.end, w.source)))
    ordered.extend(sorted(index.candidate_windows, key=lambda w: (w.start, w.end, w.source, -w.score)))
    payload = {
        "video_path": path,
        "video_sha256": index.video_sha256,
        "duration": index.duration,
        "fps": index.fps,
        "coarse_points": [asdict(p) for p in index.coarse_points],
        "uniform": [asdict(w) for w in index.uniform_windows],
        "change_motion": [asdict(w) for w in index.candidate_windows],
        "index_version": "opencv-time-index-v1",
    }
    if change_context_padding:
        payload["change_context_padding"] = float(change_context_padding)
    pool_id = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode("utf-8")).hexdigest()[:16]
    prefix = f"C{pool_id[:8]}"
    candidates = []
    for idx, window in enumerate(ordered, start=1):
        start = max(0.0, min(float(window.start), index.duration))
        end = max(0.0, min(float(window.end), index.duration))
        if window.source != "uniform_coverage" and change_context_padding:
            padding = (end - start) * float(change_context_padding)
            start = max(0.0, start - padding)
            end = min(index.duration, end + padding)
        if end <= start:
            continue
        score = float(window.score) if window.score is not None else None
        candidates.append(CandidateInterval(
            candidate_id=f"{prefix}-{idx:02d}", start=start, end=end,
            source=window.source, score=score))
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs are not unique")
    return CandidatePool(pool_id, "opencv-time-index-v1", path, index.video_sha256, float(index.duration),
                         tuple(candidates), int(index.scanned_frames), float(index.cpu_seconds),
                         float(change_context_padding))
