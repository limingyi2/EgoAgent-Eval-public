"""Local, lightweight video time indexing for hierarchical retrieval.

The index is a retrieval hint only. It never sees task answers, Oracle data,
or event annotations, and its scan frames are not counted as VLM observations.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .video import Video, file_sha256


@dataclass(frozen=True)
class IndexPoint:
    timestamp: float
    frame_index: int
    change_score: float
    motion_score: float


@dataclass(frozen=True)
class CandidateWindow:
    start: float
    end: float
    score: float
    source: str


@dataclass(frozen=True)
class VideoTimeIndex:
    video_path: str
    video_sha256: str
    duration: float
    fps: float
    coarse_points: list[IndexPoint]
    uniform_windows: list[CandidateWindow]
    candidate_windows: list[CandidateWindow]
    scanned_frames: int
    cpu_seconds: float

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "video_sha256": self.video_sha256,
            "duration": self.duration,
            "fps": self.fps,
            "coarse_points": [asdict(point) for point in self.coarse_points],
            "uniform_windows": [asdict(window) for window in self.uniform_windows],
            "candidate_windows": [asdict(window) for window in self.candidate_windows],
            "scanned_frames": self.scanned_frames,
            "cpu_seconds": self.cpu_seconds,
            "scan_is_local_only": True,
            "scores_are_retrieval_hints": True,
        }


def _gray_small(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)


def _uniform_windows(duration: float, count: int) -> list[CandidateWindow]:
    count = max(1, min(count, 16))
    windows: list[CandidateWindow] = []
    for i in range(count):
        start = duration * i / count
        end = duration * (i + 1) / count
        windows.append(CandidateWindow(start, end, 0.0, "uniform_coverage"))
    return windows


def _merge_overlapping(windows: list[CandidateWindow]) -> list[CandidateWindow]:
    merged: list[CandidateWindow] = []
    for window in sorted(windows, key=lambda item: (item.start, item.end)):
        if merged and window.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = CandidateWindow(previous.start, max(previous.end, window.end),
                                          max(previous.score, window.score), "change_motion")
        else:
            merged.append(window)
    return merged


def build_time_index(video: Video, coarse_samples: int = 24,
                     candidate_limit: int = 6) -> VideoTimeIndex:
    """Scan a small number of local frames and return retrieval hints.

    Every scan frame is decoded by OpenCV in a separate capture. The returned
    frame count and CPU time are recorded separately from VLM observations.
    """
    if type(coarse_samples) is not int or not 4 <= coarse_samples <= 256:
        raise ValueError("coarse_samples must be an integer in [4, 256]")
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= 16:
        raise ValueError("candidate_limit must be an integer in [1, 16]")
    started = perf_counter()
    video_digest = file_sha256(video.path)
    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise ValueError("Could not open video for local indexing")
    points: list[IndexPoint] = []
    previous: np.ndarray | None = None
    try:
        for i in range(coarse_samples):
            timestamp = min(video.duration - 1.0 / video.fps,
                            video.duration * (i + 0.5) / coarse_samples)
            frame_index = min(video.frame_count - 1, math.floor(timestamp * video.fps))
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise ValueError(f"Could not seek index frame {frame_index}")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"Could not decode index frame {frame_index}")
            gray = _gray_small(image)
            if previous is None:
                change = 0.0
                motion = 0.0
            else:
                difference = cv2.absdiff(gray, previous).astype(np.float32) / 255.0
                change = float(np.mean(difference))
                motion = float(np.percentile(difference, 75))
            points.append(IndexPoint(float(timestamp), int(frame_index), change, motion))
            previous = gray
    finally:
        capture.release()

    coverage_count = min(8, max(2, coarse_samples // 4))
    uniform = _uniform_windows(video.duration, coverage_count)
    half_window = video.duration / coarse_samples
    scored = sorted(points[1:], key=lambda point: point.change_score + point.motion_score,
                    reverse=True)
    candidates = []
    for point in scored[:candidate_limit]:
        score = point.change_score + point.motion_score
        start = max(0.0, point.timestamp - half_window)
        end = min(video.duration, point.timestamp + half_window)
        if end > start:
            candidates.append(CandidateWindow(start, end, score, "change_motion"))
    candidates = _merge_overlapping(candidates)
    elapsed = perf_counter() - started
    return VideoTimeIndex(str(video.path), video_digest, video.duration, video.fps, points, uniform,
                          candidates, len(points), elapsed)


def scan_interval_change_points(video: Video, start: float, end: float,
                                samples: int = 24) -> tuple[list[IndexPoint], int, float]:
    """Build a finer local OpenCV change profile for one selected interval.

    These decoded frames are index-only. They are never returned as model
    observations or counted against the VLM image budget.
    """
    if (not math.isfinite(start) or not math.isfinite(end)
            or not 0 <= start < end <= video.duration):
        raise ValueError("Local index interval must lie within the active video")
    if type(samples) is not int or not 4 <= samples <= 128:
        raise ValueError("Local index samples must be an integer in [4, 128]")
    started = perf_counter()
    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise ValueError("Could not open video for local interval indexing")
    points: list[IndexPoint] = []
    previous: np.ndarray | None = None
    try:
        for i in range(samples):
            timestamp = start + (i + 0.5) * (end - start) / samples
            frame_index = min(video.frame_count - 1, math.floor(timestamp * video.fps))
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise ValueError(f"Could not seek local index frame {frame_index}")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"Could not decode local index frame {frame_index}")
            gray = _gray_small(image)
            if previous is None:
                change = motion = 0.0
            else:
                difference = cv2.absdiff(gray, previous).astype(np.float32) / 255.0
                change = float(np.mean(difference))
                motion = float(np.percentile(difference, 75))
            points.append(IndexPoint(float(timestamp), int(frame_index), change, motion))
            previous = gray
    finally:
        capture.release()
    return points, len(points), perf_counter() - started
