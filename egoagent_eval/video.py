"""On-demand OpenCV decoding; only requested frames enter the model context."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from .schema import FrameRecord


@dataclass(frozen=True)
class Observation:
    record: FrameRecord
    jpeg: bytes


def resize_image(image: np.ndarray, max_side: int = 768) -> np.ndarray:
    if type(max_side) is not int or not 1 <= max_side <= 1024:
        raise ValueError("max_side must be an integer in [1, 1024]")
    height, width = image.shape[:2]
    if min(height, width) <= 0:
        raise ValueError("empty image")
    scale = min(1.0, max_side / max(height, width))
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1 else image


class Video:
    def __init__(self, path: str | Path, max_side: int = 768):
        if type(max_side) is not int or not 1 <= max_side <= 1024:
            raise ValueError("max_side must be an integer in [1, 1024]")
        self.path = Path(path).resolve()
        self.max_side = max_side
        if not self.path.is_file():
            raise ValueError(f"Video file does not exist: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        raw_count = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.capture.isOpened() or not math.isfinite(self.fps) or self.fps <= 0 or not math.isfinite(raw_count) or raw_count < 1:
            self.capture.release()
            raise ValueError("Video has invalid or unsupported frame metadata")
        self.frame_count = int(raw_count)
        self.duration = self.frame_count / self.fps

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.capture.release()

    def validate_timestamp(self, timestamp: float):
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or not 0 <= timestamp < self.duration:
            raise ValueError(f"timestamp must satisfy 0 <= t < {self.duration}")

    def uniform_timestamps(self, count: int, start: float = 0.0, end: float | None = None) -> list[float]:
        end = self.duration if end is None else end
        self.validate_timestamp(start)
        if isinstance(end, bool) or not isinstance(end, (int, float)) or not math.isfinite(end) or not start < end <= self.duration:
            raise ValueError("interval must satisfy 0 <= start < end <= duration")
        if type(count) is not int or not 1 <= count <= 8:
            raise ValueError("count must be an integer in [1, 8]")
        return [float(start + (i + 0.5) * (end - start) / count) for i in range(count)]

    def read_frame(self, timestamp: float) -> Observation:
        self.validate_timestamp(timestamp)
        index = min(self.frame_count - 1, math.floor(timestamp * self.fps))
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError(f"Could not seek frame {index}")
        ok, image = self.capture.read()
        if not ok or image is None:
            raise ValueError(f"Could not decode frame {index}")
        decoded_index = round(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if decoded_index != index:
            raise ValueError(f"Decoder returned frame {decoded_index}, requested {index}")
        image = resize_image(image, self.max_side)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("JPEG encoding failed")
        jpeg = encoded.tobytes()
        return Observation(FrameRecord(requested_timestamp=float(timestamp), timestamp=index / self.fps,
                                       frame_index=index, width=image.shape[1], height=image.shape[0],
                                       sha256=hashlib.sha256(jpeg).hexdigest()), jpeg)

    def sample_interval(self, start: float, end: float, count: int) -> list[Observation]:
        return [self.read_frame(t) for t in self.uniform_timestamps(count, start, end)]

    def uniform_frames(self, count: int) -> list[Observation]:
        return [self.read_frame(t) for t in self.uniform_timestamps(count)]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
