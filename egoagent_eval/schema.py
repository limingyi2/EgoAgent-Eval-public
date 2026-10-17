"""Strict runtime contracts. Ground truth deliberately has no input field here."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Timestamp = Annotated[float, Field(ge=0)]


class Question(StrictModel):
    video_path: NonEmpty
    question: NonEmpty


class ErrorRecord(StrictModel):
    code: str
    message: str


class FrameRecord(StrictModel):
    requested_timestamp: Timestamp
    timestamp: Timestamp
    frame_index: Annotated[int, Field(ge=0)]
    width: Annotated[int, Field(gt=0, le=1024)]
    height: Annotated[int, Field(gt=0, le=1024)]
    sha256: str
    file: str | None = None


class FinalAnswer(StrictModel):
    answer: str
    evidence_timestamps: list[Timestamp]
    status: Literal["ok", "refused", "failed"]

    @model_validator(mode="after")
    def successful_answer_needs_evidence(self):
        if self.status == "ok" and (not self.answer.strip() or not self.evidence_timestamps):
            raise ValueError("ok requires a nonempty answer and observed evidence")
        return self


class InspectFrame(StrictModel):
    action: Literal["inspect_frame"]
    timestamp: Timestamp


class SampleInterval(StrictModel):
    action: Literal["sample_interval"]
    start: Timestamp
    end: Timestamp
    # Upper bound is enforced by the tool so oversized requests are logged as rejections.
    # MiMo sometimes omits count in a sample_interval action. Four is the
    # maximum Active extra-frame budget and keeps the default deterministic.
    count: Annotated[int, Field(gt=0)] = 4


class SelectCandidate(StrictModel):
    action: Literal["select_candidate"]
    candidate_id: NonEmpty
    count: Annotated[int, Field(gt=0)] = 4


class FinalAction(FinalAnswer):
    action: Literal["final_answer"]
    # The compact public action form may omit status; successful final answers
    # default to the same status used by FinalAnswer callers.
    status: Literal["ok", "refused", "failed"] = "ok"


Action = Annotated[InspectFrame | SampleInterval | SelectCandidate | FinalAction, Field(discriminator="action")]
ACTION_ADAPTER = TypeAdapter(Action)


class Usage(StrictModel):
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    cached_tokens: Annotated[int, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0)] | None = None


class CallRecord(StrictModel):
    index: int
    # Explicit aliases make per-request cost audits easier while preserving
    # the original index field used by existing traces.
    request_index: int | None = None
    started_at_utc: str
    provider: str
    model: str
    is_fake: bool
    network_attempted: bool = False
    latency_seconds: Annotated[float, Field(ge=0)]
    frames: list[FrameRecord]
    request: dict[str, Any]
    raw_response: str | None = None
    content: str | None = None
    # Native provider tool calls are preserved separately from text content.
    # The protocol layer, not the transport, validates their arguments.
    native_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    response_model: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None
    image_count: int | None = None
    unique_image_count: int | None = None
    reused_image_count: int | None = None
    estimated_cost: Annotated[float, Field(ge=0)] | None = None
    evidence_coverage: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    error: ErrorRecord | None = None


class ToolRecord(StrictModel):
    index: int
    tool: str
    arguments: dict[str, Any]
    requested_timestamps: list[Timestamp] = Field(default_factory=list)
    frames: list[FrameRecord] = Field(default_factory=list)
    latency_seconds: Annotated[float, Field(ge=0)] = 0.0
    status: Literal["ok", "rejected", "failed"] = "ok"
    error: ErrorRecord | None = None
    index_scanned_frames: int = 0
    index_cpu_seconds: Annotated[float, Field(ge=0)] = 0.0


class StateEvent(StrictModel):
    """Append-only state transition record for the v2 harness."""

    index: Annotated[int, Field(ge=0)]
    state_from: str
    state_to: str
    reason: str
    request_index: int | None = None
    observed_frame_hashes: list[str] = Field(default_factory=list)
    observed_timestamps: list[Timestamp] = Field(default_factory=list)
    budget_before: dict[str, int] = Field(default_factory=dict)
    budget_after: dict[str, int] = Field(default_factory=dict)
    error_code: str | None = None


class RunResult(FinalAnswer):
    schema_version: str = "1.0"
    method: Literal["direct4", "uniform8", "active", "adaptive_active", "hierarchical_active",
                    "hierarchical_active_v2", "hierarchical_active_v3_candidate_first",
                    "hierarchical_active_v4_staged_observation"]
    question: Question
    provider: str
    model: str
    is_fake: bool
    started_at_utc: str
    video_sha256: str | None = None
    video_duration: float | None = None
    video_fps: float | None = None
    initial_frames: list[FrameRecord] = Field(default_factory=list)
    observed_frames: list[FrameRecord] = Field(default_factory=list)
    tool_calls: list[ToolRecord] = Field(default_factory=list)
    calls: list[CallRecord] = Field(default_factory=list)
    # Task-level image accounting. Per-request fields remain on CallRecord.
    total_image_count: int = 0
    run_unique_image_count: int = 0
    run_reused_image_count: int = 0
    index_scanned_frames: int = 0
    index_cpu_seconds: float = 0.0
    index_cache_reused: bool = False
    index_lookup_cpu_seconds: float = 0.0
    index_candidate_windows: list[dict[str, Any]] = Field(default_factory=list)
    candidate_pool_id: str | None = None
    retrieval_trace: list[dict[str, Any]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_seconds: float = 0.0
    errors: list[ErrorRecord] = Field(default_factory=list)
    termination_reason: str | None = None
    terminal_state: Literal["ANSWERED", "INSUFFICIENT_EVIDENCE", "PROTOCOL_ERROR",
                            "TOOL_ERROR", "BUDGET_EXHAUSTED", "PROVIDER_ERROR"] | None = None
    state_events: list[StateEvent] = Field(default_factory=list)
    correct: bool | None = None


class Budget(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    max_tool_calls: Annotated[int, Field(ge=0, le=2)] = 2
    max_extra_frames: Annotated[int, Field(ge=0, le=4)] = 4
    max_model_calls: Annotated[int, Field(ge=1, le=3)] = 3
    max_image_side: Annotated[int, Field(ge=1, le=1024)] = 768
