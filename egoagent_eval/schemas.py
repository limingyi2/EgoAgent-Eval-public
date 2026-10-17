"""Public data schemas shared by the project workstreams."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import Literal


@dataclass(frozen=True, slots=True)
class Task:
    """A question and its ground truth over a local video."""

    video_path: Path
    question: str
    standard_answer: str
    event_time_range: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_path", Path(self.video_path))
        start, end = self.event_time_range
        if start < 0 or end < start:
            raise ValueError("event_time_range must satisfy 0 <= start <= end")
        if not self.question.strip():
            raise ValueError("question must not be empty")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation in an agent's observation/action trace."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Answer plus the video evidence and actions used to produce it."""

    answer: str
    evidence_timestamps: tuple[float, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    status: Literal["ok", "refused", "failed"] = "ok"

    def __post_init__(self) -> None:
        if self.status not in {"ok", "refused", "failed"}:
            raise ValueError("status must be ok, refused or failed")


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """One evaluated task result with resource usage and failure details."""

    task_id: str
    prediction: str
    correct: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0 or self.latency_seconds < 0:
            raise ValueError("token counts and latency must be non-negative")
