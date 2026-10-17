"""Task validation and reproducible experiment configuration."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def interval(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(number(x) for x in value) and 0 <= value[0] <= value[1])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def strict_loads(text):
    def reject(value):
        raise ValueError(f"nonstandard JSON constant: {value}")
    return json.loads(text, parse_constant=reject)


@dataclass(frozen=True)
class Config:
    run_id: str
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 512
    seed: int | None = None
    tolerance_seconds: float = 0.5
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    def __post_init__(self):
        if (not isinstance(self.run_id, str) or not isinstance(self.model, str)
                or not self.run_id.strip() or not self.model.strip()):
            raise ValueError("run_id and model must be nonempty")
        if not number(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not isinstance(self.max_output_tokens, int) or isinstance(self.max_output_tokens, bool) or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("seed must be an integer or null")
        if not number(self.tolerance_seconds) or self.tolerance_seconds < 0:
            raise ValueError("tolerance_seconds must be finite and nonnegative")
        prices = (self.input_price_per_million, self.output_price_per_million)
        if (prices[0] is None) != (prices[1] is None):
            raise ValueError("provide both input and output prices, or neither")
        if any(x is not None and (not number(x) or x < 0) for x in prices):
            raise ValueError("prices must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)


def normalize_dataset_task(task):
    """Adapt the repository dataset contract without importing dataset or Agent code."""
    if "answer" not in task or "standard_answer" in task:
        return task, False
    task = dict(task)
    answer = task["answer"]
    review = task.get("verification") or {}
    if not isinstance(review, dict):
        raise ValueError("verification must be an object or null")
    if isinstance(answer, list):
        if not answer or any(not isinstance(item, str) or not item.strip() for item in answer):
            raise ValueError("structured reference answer must contain nonempty strings")
        # No guessed string serialization is accepted as a correct sequence answer.
        task["standard_answer"] = json.dumps(answer, ensure_ascii=False)
        task["aliases"] = []
        task["answer_mode"] = "manual"
    else:
        task["standard_answer"] = answer
        task["aliases"] = review.get("accepted_answers", [])
        task.setdefault("answer_mode", "closed")
    task["question_type"] = task.get("answer_type")
    task["event_time_range"] = [task.get("event_start"), task.get("event_end")]
    source = task.get("source") or {}
    if not isinstance(source, dict):
        raise ValueError("source must be an object")
    task.setdefault("duration_seconds", source.get("video_end"))
    task["human_verified"] = task.get("verification_status") == "verified" and bool(review)
    task["fixture"] = source.get("is_synthetic") is True
    if task["human_verified"]:
        if review.get("decision") != "approve" or not review.get("reviewer") or not review.get("reviewed_at"):
            raise ValueError("verified dataset task requires approval, reviewer and reviewed_at")
        if any(review.get(key) != task.get(key) for key in ("question", "answer", "event_start", "event_end")):
            raise ValueError("dataset task differs from approved human review")
        if not task.get("video_sha256") or review.get("watched_video_sha256") != task["video_sha256"]:
            raise ValueError("human review must bind to the video checksum")
    return task, True


def load_tasks(path, *, mock=False, video_root=None):
    path = Path(path).resolve()
    tasks, seen = [], set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                task = strict_loads(line)
                if not isinstance(task, dict):
                    raise ValueError("task must be an object")
                task, dataset_format = normalize_dataset_task(task)
                for key in ("task_id", "question", "question_type", "video_path"):
                    if not isinstance(task.get(key), str) or not task[key].strip():
                        raise ValueError(f"missing/non-string {key}")
                if task["task_id"] in seen:
                    raise ValueError("duplicate task_id")
                if task.get("answer_mode", "manual") not in ("closed", "manual"):
                    raise ValueError("answer_mode must be closed or manual")
                task.setdefault("answer_mode", "manual")
                if not isinstance(task.get("standard_answer"), str) or not task["standard_answer"].strip():
                    raise ValueError("standard_answer must be a nonempty string")
                aliases = task.setdefault("aliases", [])
                if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
                    raise ValueError("aliases must be nonempty strings")
                windows = task.get("event_windows", [task.get("event_time_range")])
                if not isinstance(windows, list) or not windows or not all(interval(w) for w in windows):
                    raise ValueError("event_windows/event_time_range must contain valid event intervals")
                task["event_windows"] = windows
                if not number(task.get("duration_seconds")) or task["duration_seconds"] <= 0:
                    raise ValueError("duration_seconds must be positive and finite")
                if any(w[1] > task["duration_seconds"] for w in windows):
                    raise ValueError("event window exceeds video duration")
                source = task.get("source") or {}
                if not isinstance(source, dict):
                    raise ValueError("source must be an object")
                if not mock and (task.get("human_verified") is not True or task.get("fixture") is True
                                 or source.get("is_synthetic") is True
                                 or task.get("verification_status", "verified") != "verified"):
                    raise ValueError("real evaluation requires human_verified=true and forbids fixtures")
                video = Path(task["video_path"])
                if not video.is_absolute():
                    base = Path(video_root) if video_root is not None else (Path(__file__).resolve().parents[1] if dataset_format else path.parent)
                    video = base / video
                task["video_path"] = str(video.resolve())
                if not mock:
                    sha = hashlib.sha256()
                    with video.open("rb") as video_stream:
                        for chunk in iter(lambda: video_stream.read(1024 * 1024), b""):
                            sha.update(chunk)
                    actual_hash = sha.hexdigest()
                    if task.get("video_sha256") is not None and task["video_sha256"] != actual_hash:
                        raise ValueError("video checksum differs from human-verified task")
                    task["video_sha256"] = actual_hash
                else:
                    task["video_sha256"] = "mock:" + digest(task["video_path"])
                seen.add(task["task_id"])
                tasks.append(task)
            except (ValueError, TypeError, OSError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not tasks:
        raise ValueError("tasks file is empty")
    return tasks


def runner_task(task):
    """Never expose reference answers or human evidence windows to a runner."""
    return {key: task[key] for key in ("task_id", "question", "question_type", "video_path",
                                      "video_sha256", "duration_seconds")}


def uniform_times(duration, count):
    """Bin centers avoid requesting the out-of-range frame at duration itself."""
    return [duration * (i + 0.5) / count for i in range(count)]
