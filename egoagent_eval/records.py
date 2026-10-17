"""Append-only per-run artifacts; credentials and image data URLs are excluded."""

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .schema import CallRecord
from .video import Observation


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactStore:
    def __init__(self, root: str | Path, is_fake: bool, label: str, run_id: str | None = None):
        run_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-") + uuid.uuid4().hex[:12])
        self.run_id = run_id
        self.path = Path(root).resolve() / ("fake" if is_fake else "real") / f"{run_id}-{label}"
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "frames").mkdir()

    def save_frame(self, frame: Observation):
        relative = f"frames/{frame.record.sha256}.jpg"
        target = self.path / relative
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(frame.jpeg)
        frame.record.file = relative

    def append_call(self, record: CallRecord):
        with (self.path / "calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(record.model_dump_json() + "\n")

    def append_protocol(self, record: dict):
        """Append normalized protocol output without rewriting raw call traces."""
        with (self.path / "protocol.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def append_state_event(self, record: dict):
        """Append one immutable Agent v2 state transition."""
        with (self.path / "state_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def append_retrieval(self, record: dict):
        """Append candidate selection and execution details."""
        with (self.path / "retrieval.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def write_json(self, name: str, value):
        if Path(name).name != name:
            raise ValueError("Artifact name must be a filename")
        with (self.path / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
