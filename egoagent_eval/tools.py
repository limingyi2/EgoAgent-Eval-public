"""Bounded on-demand video tools. Failed and rejected attempts consume call slots."""

from dataclasses import dataclass
from time import perf_counter

from .records import ArtifactStore
from .sampling import sample_interval_timestamps
from .schema import Budget, ErrorRecord, InspectFrame, SampleInterval, SelectCandidate, ToolRecord
from .video import Observation, Video


@dataclass
class ToolOutput:
    frames: list[Observation]
    record: ToolRecord


class ObservationTools:
    def __init__(self, video: Video, budget: Budget | None = None, store: ArtifactStore | None = None):
        self.video = video
        self.budget = budget or Budget()
        if video.max_side > self.budget.max_image_side:
            raise ValueError("Video image size exceeds tool budget")
        self.store = store
        self.records: list[ToolRecord] = []
        self.reserved_frames = 0

    def execute(self, action: InspectFrame | SampleInterval, *,
                record_tool: str | None = None,
                record_arguments: dict | None = None,
                requested_timestamps: list[float] | None = None,
                index_scanned_frames: int = 0,
                index_cpu_seconds: float = 0.0) -> ToolOutput:
        started = perf_counter()
        args = action.model_dump(exclude={"action"})
        record = ToolRecord(index=len(self.records) + 1, tool=record_tool or action.action,
                            arguments=record_arguments or args,
                            index_scanned_frames=index_scanned_frames,
                            index_cpu_seconds=index_cpu_seconds)
        self.records.append(record)
        frames = []
        count = 1 if isinstance(action, InspectFrame) else action.count
        try:
            if record.index > self.budget.max_tool_calls:
                record.status = "rejected"
                record.error = ErrorRecord(code="tool_call_budget", message="Maximum tool call attempts exceeded")
            elif self.reserved_frames + count > self.budget.max_extra_frames:
                record.status = "rejected"
                record.error = ErrorRecord(code="frame_budget", message="Additional frame budget exceeded")
            else:
                if isinstance(action, InspectFrame):
                    record.requested_timestamps = [action.timestamp]
                    self.video.validate_timestamp(action.timestamp)
                else:
                    record.requested_timestamps = (requested_timestamps if requested_timestamps is not None
                                                   else self.video.uniform_timestamps(action.count, action.start, action.end))
                    if len(record.requested_timestamps) != action.count:
                        raise ValueError("Sampler returned a different frame count than requested")
                    if any(not action.start <= timestamp < action.end
                           for timestamp in record.requested_timestamps):
                        raise ValueError("Sampler returned a timestamp outside the selected interval")
                    for timestamp in record.requested_timestamps:
                        self.video.validate_timestamp(timestamp)
                # Reserve before reading. Decode failures cannot bypass the cumulative limit.
                self.reserved_frames += count
                for timestamp in record.requested_timestamps:
                    frame = self.video.read_frame(timestamp)
                    if self.store:
                        self.store.save_frame(frame)
                    frames.append(frame)
                    record.frames.append(frame.record.model_copy())
        except ValueError as exc:
            record.status = "failed"
            record.error = ErrorRecord(code="video_tool_error", message=str(exc))
        finally:
            record.latency_seconds = perf_counter() - started
            if self.store:
                # Persist each attempt immediately, even if a later model request fails.
                with (self.store.path / "tools.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(record.model_dump_json() + "\n")
        return ToolOutput(frames, record)

    def reject_action(self, action: InspectFrame | SampleInterval, code: str, message: str) -> ToolOutput:
        """Record a bounded action rejection without touching video frames.

        The v2 harness calls this after its deterministic budget gate.  It is
        intentionally separate from ``execute`` so an exhausted budget cannot
        accidentally decode or reserve another frame.
        """
        record = ToolRecord(index=len(self.records) + 1, tool=action.action,
                            arguments=action.model_dump(exclude={"action"}),
                            status="rejected",
                            error=ErrorRecord(code=code, message=message))
        self.records.append(record)
        if self.store:
            with (self.store.path / "tools.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(record.model_dump_json() + "\n")
        return ToolOutput([], record)

    def reject_candidate(self, action: SelectCandidate, code: str, message: str,
                         candidate_pool_id: str | None) -> ToolOutput:
        record = ToolRecord(index=len(self.records) + 1, tool="select_candidate",
                            arguments={"candidate_id": action.candidate_id, "count": action.count,
                                       "candidate_pool_id": candidate_pool_id},
                            status="rejected", error=ErrorRecord(code=code, message=message))
        self.records.append(record)
        if self.store:
            with (self.store.path / "tools.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(record.model_dump_json() + "\n")
        return ToolOutput([], record)

    def sample_candidate(self, action: SelectCandidate, candidate,
                         candidate_pool_id: str | None, *,
                         sampling_policy: str = "uniform",
                         change_points=(), local_index_samples: int = 24) -> ToolOutput:
        """Sample only the Python-resolved interval; the model cannot override it."""
        scanned = 0
        index_cpu = 0.0
        if sampling_policy != "uniform":
            from .index import scan_interval_change_points
            try:
                change_points, scanned, index_cpu = scan_interval_change_points(
                    self.video, candidate.start, candidate.end, local_index_samples)
            except (ValueError, OSError) as exc:
                return self.reject_candidate(action, "local_index_error", str(exc), candidate_pool_id)
        timestamps = sample_interval_timestamps(
            candidate.start, candidate.end, action.count,
            policy=sampling_policy, change_points=change_points)
        arguments = {"candidate_id": action.candidate_id,
                     "candidate_pool_id": candidate_pool_id,
                     "start": candidate.start, "end": candidate.end,
                     "count": action.count, "source": candidate.source,
                     "score": candidate.score, "sampling_policy": sampling_policy,
                     "requested_timestamps": timestamps,
                     "local_index_scanned_frames": scanned,
                     "local_index_cpu_seconds": index_cpu}
        return self.execute(SampleInterval(action="sample_interval", start=candidate.start,
                                           end=candidate.end, count=action.count),
                            record_tool="select_candidate", record_arguments=arguments,
                            requested_timestamps=timestamps,
                            index_scanned_frames=scanned, index_cpu_seconds=index_cpu)

    def inspect_frame(self, timestamp: float) -> ToolOutput:
        return self.execute(InspectFrame(action="inspect_frame", timestamp=timestamp))

    def sample_interval(self, start: float, end: float, count: int) -> ToolOutput:
        return self.execute(SampleInterval(action="sample_interval", start=start, end=end, count=count))
