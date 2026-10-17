"""Execution and append-only JSONL logging with per-task failure isolation."""

import json
import os
import platform
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import STRATEGIES
from .data import digest, number, runner_task, strict_loads, uniform_times
from .scoring import parse_prediction, score_answer, score_evidence


class BudgetExceeded(RuntimeError):
    pass


class ProtocolError(RuntimeError):
    pass


class Recorder:
    """All provider requests and video tool calls must pass through this recorder."""

    def __init__(self, task, strategy, config, runner):
        self.task, self.strategy, self.config = task, strategy, config
        self.mode = "mock" if runner.is_mock else "real"
        self.provider_config = {"model": config.model, "temperature": config.temperature,
                                "max_output_tokens": config.max_output_tokens,
                                "seed": config.seed if runner.supports_seed else None}
        self.frame_limit = 4 if strategy == "Direct-4" else 8
        self.request_limit = 2 if strategy == "Active-4+4" else 1
        self.requests, self.tools, self.observations = [], [], []
        self.violations = []

    def deny(self, exception):
        self.violations.append(exception)
        raise exception

    def frames(self, timestamps, images):
        if len(timestamps) != len(images) or any(not number(t) or t < 0 or t > self.task["duration_seconds"] for t in timestamps):
            raise ProtocolError("frames require matched images and in-video timestamps")
        if len(self.observations) + len(images) > self.frame_limit:
            self.deny(BudgetExceeded(f"observation budget {self.frame_limit} exceeded"))
        if not self.observations:
            count = 8 if self.strategy == "Uniform-8" else 4
            expected = uniform_times(self.task["duration_seconds"], count)
            if len(timestamps) != count or any(abs(a - b) > 1e-6 for a, b in zip(timestamps, expected)):
                raise ProtocolError("initial observations must match the configured uniform frame plan")
        if self.observations and (self.strategy != "Active-4+4" or len(self.requests) != 1):
            self.deny(ProtocolError("extra observations require Active-4+4 selection request"))
        new_frames = []
        for timestamp, image in zip(timestamps, images):
            if not isinstance(image, str):
                raise ProtocolError("image references must be strings")
            if self.mode == "real":
                path = Path(image)
                if not path.is_file():
                    raise ProtocolError("real frame images must be readable local files")
                import hashlib
                image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                image_hash = None
            new_frames.append({"timestamp_seconds": timestamp, "image": image, "sha256": image_hash})
        self.observations.extend(new_frames)
        return {"timestamps": timestamps, "images": images, "image_count": len(images)}

    def observation_images(self):
        return [frame["image"] for frame in self.observations]

    def tool(self, name, arguments, callback):
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            raise ProtocolError("tool must have a nonempty name and object arguments")
        json.dumps(arguments, allow_nan=False)
        entry = {"tool": name, "arguments": json.loads(json.dumps(arguments)), "result": None, "error": None}
        self.tools.append(entry)
        start = time.perf_counter()
        try:
            result = callback()
            json.dumps(result, allow_nan=False)
            entry["result"] = json.loads(json.dumps(result))
            return entry["result"]
        except Exception as exc:
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            entry["elapsed_seconds"] = time.perf_counter() - start

    def request(self, model_input, callback):
        if sum(request["sent"] for request in self.requests) >= self.request_limit:
            self.deny(BudgetExceeded(f"model request budget {self.request_limit} exceeded"))
        if not isinstance(model_input, dict) or not isinstance(model_input.get("images"), list):
            raise ProtocolError("model input must include an images list and the full prompt/messages")
        if not self.observations or model_input["images"] != self.observation_images():
            raise ProtocolError("each model request must include all currently observed images in order")
        json.dumps(model_input, allow_nan=False)
        entry = {"input": json.loads(json.dumps(model_input)), "config": self.provider_config.copy(), "output": None, "usage": None,
                 "image_count": len(model_input["images"]), "sent": True, "error": None,
                 "provider_kind": self.mode}
        self.requests.append(entry)
        start = time.perf_counter()
        try:
            response = callback()
            if not isinstance(response, dict) or "output" not in response:
                raise ProtocolError("provider must return {output, usage, provider_kind}")
            declared_kind = response.get("provider_kind", "mock" if self.mode == "mock" else None)
            if declared_kind != self.mode:
                entry["rejected_provider_kind"] = declared_kind if isinstance(declared_kind, (str, type(None))) else type(declared_kind).__name__
                raise ProtocolError("provider provenance must match run mode; fake output forbidden in real runs")
            json.dumps(response["output"], allow_nan=False)
            entry["output"] = json.loads(json.dumps(response["output"]))
            usage = response.get("usage")
            if usage is not None:
                if not isinstance(usage, dict) or any(
                    not isinstance(usage.get(key), int) or isinstance(usage.get(key), bool) or usage[key] < 0
                    for key in ("input_tokens", "output_tokens")
                ):
                    raise ProtocolError("usage must contain nonnegative integer input_tokens/output_tokens")
                json.dumps(usage, allow_nan=False)
                entry["usage"] = json.loads(json.dumps(usage))
                if usage["output_tokens"] > self.config.max_output_tokens:
                    raise BudgetExceeded("provider output exceeded max_output_tokens")
            return response["output"]
        except Exception as exc:
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            entry["elapsed_seconds"] = time.perf_counter() - start

    def usage(self):
        complete = bool(self.requests) and all(request["usage"] is not None for request in self.requests)
        return {"input_tokens": sum(r["usage"]["input_tokens"] for r in self.requests) if complete else None,
                "output_tokens": sum(r["usage"]["output_tokens"] for r in self.requests) if complete else None,
                "complete": complete}


def evaluate_one(task, strategy, config, runner):
    recorder = Recorder(task, strategy, config, runner)
    start = time.perf_counter()
    prediction, answer_score, evidence_score, error = None, None, None, None
    failures = []
    try:
        raw = runner.run(runner_task(task), strategy, recorder.provider_config.copy(), recorder)
        if not recorder.requests or recorder.requests[-1]["output"] != raw:
            raise ProtocolError("final output must originate from a recorded provider request")
        if recorder.violations:
            raise recorder.violations[0]
        if any(tool["error"] for tool in recorder.tools) or any(request["error"] for request in recorder.requests):
            raise ProtocolError("runner swallowed a tool/provider failure")
        if len(recorder.observations) < (8 if strategy == "Uniform-8" else 4):
            raise ProtocolError("strategy did not observe its required initial frames")
        if strategy == "Active-4+4" and (len(recorder.observations) != 8 or len(recorder.requests) != 2):
            raise ProtocolError("Active-4+4 requires a selection request, four extra frames and an answer request")
        if strategy == "Active-4+4":
            selection_raw = recorder.requests[0]["output"]
            try:
                selection = strict_loads(selection_raw) if isinstance(selection_raw, str) else selection_raw
            except (ValueError, TypeError) as exc:
                failures.append({"code": "invalid_json", "source": "automatic", "detail": f"selection: {exc}"})
                raise
            selected = selection.get("observe_timestamps") if isinstance(selection, dict) else None
            actual = [frame["timestamp_seconds"] for frame in recorder.observations[4:]]
            if selected != actual:
                raise ProtocolError("extra frames must match the model's observe_timestamps selection")
        try:
            prediction = parse_prediction(raw, task["duration_seconds"])
        except (ValueError, TypeError) as exc:
            failures.append({"code": "invalid_json", "source": "automatic", "detail": str(exc)})
            raise
        answer_score = score_answer(task, prediction)
        evidence_score = score_evidence(task, prediction, config.tolerance_seconds)
        if answer_score["correct"] is False:
            failures.append({"code": "incorrect", "source": "automatic", "detail": "exact/alias mismatch"})
        if answer_score["correct"] is None:
            failures.append({"code": "manual_review", "source": "requires_human", "detail": "answer_mode=manual"})
        if evidence_score["missing"]:
            failures.append({"code": "insufficient_evidence", "source": "automatic", "detail": "no predicted temporal evidence"})
        elif not evidence_score["timestamp_hit"]:
            failures.append({"code": "evidence_miss", "source": "automatic", "detail": "prediction misses annotated windows"})
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc),
                 "traceback": [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                               for frame in traceback.extract_tb(exc.__traceback__)]}
        if not failures:
            code = ("budget_exceeded" if isinstance(exc, BudgetExceeded) else
                    "tool_error" if any(tool["error"] for tool in recorder.tools) else
                    "protocol_error" if isinstance(exc, ProtocolError) else "runner_error")
            failures.append({"code": code, "source": "automatic", "detail": str(exc)})
    elapsed = time.perf_counter() - start
    usage = recorder.usage()
    cost = None
    if usage["complete"] and config.input_price_per_million is not None:
        cost = (usage["input_tokens"] * config.input_price_per_million
                + usage["output_tokens"] * config.output_price_per_million) / 1_000_000
    return {"kind": "evaluation", "schema_version": 1, "run_id": config.run_id,
            "mode": recorder.mode, "runner": runner.name, "strategy": strategy, "task_id": task["task_id"],
            "task": task, "task_fingerprint": digest(task), "config": config.to_dict(),
            "seed_supported": runner.supports_seed, "effective_seed": recorder.provider_config["seed"],
            "prediction": prediction, "answer_score": answer_score, "evidence_score": evidence_score,
            "requests": recorder.requests, "tool_calls": recorder.tools, "frames": recorder.observations,
            "request_count": len(recorder.requests), "observed_frame_count": len(recorder.observations),
            "model_input_image_count": sum(r["image_count"] for r in recorder.requests),
            "usage": usage, "estimated_cost": cost, "elapsed_seconds": elapsed,
            "error": error, "failures": failures,
            "created_at": datetime.now(timezone.utc).isoformat()}


def run_experiment(tasks, config, runner, results_dir, *, mode):
    if mode not in ("real", "mock") or (mode == "mock") != runner.is_mock:
        raise ValueError("run mode and runner provenance disagree")
    if mode == "real" and any(t.get("human_verified") is not True or t.get("fixture") is True for t in tasks):
        raise ValueError("real runs require human-verified, non-fixture tasks")
    output_dir = Path(results_dir) / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", config.run_id)[:80] or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_dir / f"{safe_id}_{stamp}_{uuid.uuid4().hex[:8]}.jsonl"
    manifest = {"kind": "manifest", "schema_version": 1, "run_id": config.run_id, "mode": mode,
                "runner": runner.name, "config": config.to_dict(), "tasks": tasks,
                "strategies": list(STRATEGIES), "task_set_fingerprint": digest(tasks),
                "seed_supported": runner.supports_seed, "created_at": datetime.now(timezone.utc).isoformat(),
                "environment": {"python": platform.python_version(), "platform": platform.platform()},
                "runner_class": f"{type(runner).__module__}:{type(runner).__qualname__}",
                "protocol": {"initial_sampling": "uniform bin centers across the complete input video",
                             "active_selection_key": "observe_timestamps", "active_extra_frames": 4,
                             "request_limits": {"Direct-4": 1, "Uniform-8": 1, "Active-4+4": 2}}}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        def write(record):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write(manifest)
        for task in tasks:
            for strategy in STRATEGIES:
                record = evaluate_one(task, strategy, config, runner)
                write(record)
        write({"kind": "completion", "run_id": config.run_id, "expected_records": len(tasks) * len(STRATEGIES)})
    return output.resolve()
