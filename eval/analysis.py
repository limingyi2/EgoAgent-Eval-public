"""Recompute reports exclusively from saved logs, retaining the planned denominator."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from . import STRATEGIES
from .data import digest, number, strict_loads, uniform_times
from .scoring import parse_prediction, score_answer, score_evidence


def mean_present(values):
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def missing_record(manifest, task, strategy):
    return {"kind": "evaluation", "schema_version": 1, "run_id": manifest["run_id"],
            "mode": manifest["mode"], "strategy": strategy, "task_id": task["task_id"], "task": task,
            "config": manifest["config"], "prediction": None, "answer_score": None, "evidence_score": None,
            "requests": None, "tool_calls": None, "frames": None, "request_count": None,
            "observed_frame_count": None, "model_input_image_count": None,
            "usage": {"input_tokens": None, "output_tokens": None, "complete": False},
            "elapsed_seconds": None, "estimated_cost": None, "error": {"type": "MissingRecord", "message": "planned execution absent from saved log"},
            "failures": [{"code": "missing_record", "source": "automatic", "detail": "interrupted or incomplete run"}]}


def load_logs(paths, *, mode, run_id=None):
    manifests, records, notes = [], [], []
    seen_runs = set()
    for input_path in paths:
        path = Path(input_path)
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        rows = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = strict_loads(line)
            except ValueError as exc:
                if i == len(lines) - 1 and rows:
                    notes.append(f"{path.name}: incomplete final JSON line; missing planned records retained")
                    break
                raise ValueError(f"{path}:{i + 1}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{i + 1}: log row must be an object")
            rows.append(row)
        if not rows or rows[0].get("kind") != "manifest":
            raise ValueError(f"{path}: missing experiment manifest; denominator cannot be audited")
        manifest = rows[0]
        if run_id is not None and manifest.get("run_id") != run_id:
            continue
        if manifest.get("mode") != mode:
            raise ValueError("cannot mix real and mock logs; select an isolated mode directory")
        if manifest.get("schema_version") != 1 or manifest.get("strategies") != list(STRATEGIES):
            raise ValueError("unsupported schema or experiment strategy matrix")
        if manifest["run_id"] in seen_runs:
            raise ValueError("duplicate run_id across input logs; choose one file or use unique run IDs")
        seen_runs.add(manifest["run_id"])
        tasks = manifest["tasks"]
        if not tasks or len({t["task_id"] for t in tasks}) != len(tasks) or digest(tasks) != manifest["task_set_fingerprint"]:
            raise ValueError("invalid task manifest fingerprint or duplicate tasks")
        if mode == "real" and any(t.get("fixture") is True or t.get("human_verified") is not True
                                  or (t.get("source") or {}).get("is_synthetic") is True
                                  or t.get("verification_status", "verified") != "verified"
                                  or not isinstance(t.get("video_sha256"), str) or len(t["video_sha256"]) != 64
                                  for t in tasks):
            raise ValueError("real logs contain unverified/fixture tasks")
        expected = {(t["task_id"], s): t for t in tasks for s in STRATEGIES}
        found = {}
        completed = False
        for row in rows[1:]:
            if row.get("run_id") != manifest["run_id"]:
                raise ValueError("record run_id disagrees with manifest")
            if row.get("kind") == "completion":
                completed = True
                continue
            if row.get("kind") != "evaluation":
                raise ValueError("unexpected log row kind")
            key = (row.get("task_id"), row.get("strategy"))
            if key not in expected or key in found:
                raise ValueError("unexpected or duplicate task/strategy result")
            if (row.get("mode") != mode or row.get("task") != expected[key]
                    or row.get("task_fingerprint") != digest(expected[key]) or row.get("config") != manifest["config"]):
                raise ValueError("result differs from manifest task/config/mode")
            effective = manifest["config"]["seed"] if manifest["seed_supported"] else None
            if row.get("seed_supported") != manifest["seed_supported"] or row.get("effective_seed") != effective:
                raise ValueError("inconsistent provider seed support")
            row = recompute(row)
            found[key] = row
        if not completed:
            notes.append(f"{path.name}: run has no completion marker")
        for key, task in expected.items():
            records.append(found.get(key) or missing_record(manifest, task, key[1]))
        manifests.append(manifest)
    if not manifests:
        raise ValueError("no selected experiments")
    comparison = lambda m: (m["task_set_fingerprint"], {k: v for k, v in m["config"].items() if k != "run_id"}, m["seed_supported"])
    if any(comparison(m) != comparison(manifests[0]) for m in manifests[1:]):
        raise ValueError("comparison requires identical task/video contents, model config, tolerance/prices and seed support")
    return records, manifests, notes


def recompute(row):
    """Do not trust saved correctness or aggregate resource totals."""
    row = dict(row)
    requests, frames = row["requests"], row["frames"]
    if not isinstance(requests, list) or not isinstance(frames, list) or not isinstance(row.get("tool_calls"), list):
        raise ValueError("invalid request/frame/tool trace")
    if not number(row.get("elapsed_seconds")) or row["elapsed_seconds"] < 0:
        raise ValueError("invalid end-to-end elapsed time")
    for request in requests:
        if request.get("provider_kind") != row["mode"]:
            raise ValueError("request provider provenance differs from run mode")
        if request.get("config") != {"model": row["config"]["model"], "temperature": row["config"]["temperature"],
                                     "max_output_tokens": row["config"]["max_output_tokens"], "seed": row["effective_seed"]}:
            raise ValueError("request provider configuration differs from experiment")
        images = request.get("input", {}).get("images")
        if not isinstance(images, list) or request.get("image_count") != len(images):
            raise ValueError("invalid logged model input image count")
        if type(request.get("sent")) is not bool:
            raise ValueError("request sent status must be a boolean")
        usage = request.get("usage")
        if usage is not None and (not isinstance(usage, dict) or any(
                not isinstance(usage.get(k), int) or isinstance(usage.get(k), bool) or usage[k] < 0
                for k in ("input_tokens", "output_tokens"))):
            raise ValueError("invalid logged token usage")
    if any(not isinstance(f, dict) or not number(f.get("timestamp_seconds"))
           or not 0 <= f["timestamp_seconds"] <= row["task"]["duration_seconds"] for f in frames):
        raise ValueError("invalid logged frame timestamps")
    if not row["error"]:
        initial_count = 8 if row["strategy"] == "Uniform-8" else 4
        frame_count = 4 if row["strategy"] == "Direct-4" else 8
        request_count = 2 if row["strategy"] == "Active-4+4" else 1
        expected = uniform_times(row["task"]["duration_seconds"], initial_count)
        actual = [f["timestamp_seconds"] for f in frames[:initial_count]]
        if (len(frames) != frame_count or len(requests) != request_count or len(actual) != initial_count
                or any(abs(a - b) > 1e-6 for a, b in zip(actual, expected))
                or any(r.get("error") or not r["sent"] for r in requests)
                or any(t.get("error") for t in row["tool_calls"])):
            raise ValueError("successful record violates strategy/trace protocol")
        for index, request in enumerate(requests):
            visible = frames[:4] if row["strategy"] == "Active-4+4" and index == 0 else frames
            if request["input"]["images"] != [f["image"] for f in visible]:
                raise ValueError("model input images differ from recorded observations")
        if row["strategy"] == "Active-4+4":
            selection_raw = requests[0]["output"]
            selection = strict_loads(selection_raw) if isinstance(selection_raw, str) else selection_raw
            if not isinstance(selection, dict) or selection.get("observe_timestamps") != [f["timestamp_seconds"] for f in frames[4:]]:
                raise ValueError("active observations differ from recorded model selection")
    row["request_count"] = sum(bool(r["sent"]) for r in requests)
    row["observed_frame_count"] = len(frames)
    row["model_input_image_count"] = sum(r["image_count"] for r in requests)
    complete_usage = bool(requests) and all(r.get("usage") is not None for r in requests)
    row["usage"] = {key: sum(r["usage"][key] for r in requests) if complete_usage else None
                    for key in ("input_tokens", "output_tokens")}
    row["usage"]["complete"] = complete_usage
    row["estimated_cost"] = None
    config = row["config"]
    if complete_usage and config["input_price_per_million"] is not None:
        row["estimated_cost"] = (row["usage"]["input_tokens"] * config["input_price_per_million"]
                                  + row["usage"]["output_tokens"] * config["output_price_per_million"]) / 1_000_000
    row["answer_score"], row["evidence_score"] = None, None
    row["prediction"] = None
    failures = [f for f in row["failures"] if f["code"] not in
                ("incorrect", "manual_review", "insufficient_evidence", "evidence_miss")]
    if not row["error"]:
        try:
            prediction = parse_prediction(requests[-1]["output"], row["task"]["duration_seconds"])
        except (ValueError, TypeError, IndexError) as exc:
            failures.append({"code": "invalid_json", "source": "automatic", "detail": str(exc)})
        else:
            row["prediction"] = prediction
            row["answer_score"] = score_answer(row["task"], prediction)
            row["evidence_score"] = score_evidence(row["task"], prediction, config["tolerance_seconds"])
            if row["answer_score"]["correct"] is False:
                failures.append({"code": "incorrect", "source": "automatic", "detail": "exact/alias mismatch"})
            if row["answer_score"]["correct"] is None:
                failures.append({"code": "manual_review", "source": "requires_human", "detail": "answer_mode=manual"})
            if row["evidence_score"]["missing"]:
                failures.append({"code": "insufficient_evidence", "source": "automatic", "detail": "no temporal evidence"})
            elif not row["evidence_score"]["timestamp_hit"]:
                failures.append({"code": "evidence_miss", "source": "automatic", "detail": "no annotated-window match"})
    row["failures"] = failures
    return row


def aggregate(records):
    n = len(records)
    if not n:
        raise ValueError("cannot aggregate empty records")
    correct = sum((r.get("answer_score") or {}).get("correct") is True for r in records)
    hits = sum((r.get("evidence_score") or {}).get("timestamp_hit") is True for r in records)
    missing_checks = {
        "execution": lambda r: r["requests"] is None,
        "prediction": lambda r: r.get("prediction") is None,
        "evidence": lambda r: r.get("evidence_score") is None or r["evidence_score"]["missing"],
        "token_usage": lambda r: not r["usage"]["complete"],
        "latency": lambda r: r["elapsed_seconds"] is None,
        "cost": lambda r: r["estimated_cost"] is None,
    }
    missing_counts = {field: sum(check(r) for r in records) for field, check in missing_checks.items()}
    groups = defaultdict(list)
    for row in records:
        groups[row["task"]["question_type"]].append(row)
    return {"n": n, "correct_count": correct, "accuracy": correct / n,
            "timestamp_hit_count": hits, "evidence_hit_rate": hits / n,
            "manual_review_count": sum((r.get("answer_score") or {}).get("correct") is None
                                       and r.get("answer_score") is not None for r in records),
            "scored_count": sum(r.get("answer_score") is not None and r["answer_score"]["correct"] is not None for r in records),
            "mean_requests": mean_present([r["request_count"] for r in records]),
            "mean_observed_frames": mean_present([r["observed_frame_count"] for r in records]),
            "mean_model_input_images": mean_present([r["model_input_image_count"] for r in records]),
            "mean_elapsed_seconds": mean_present([r["elapsed_seconds"] for r in records]),
            "mean_input_tokens": mean_present([r["usage"]["input_tokens"] for r in records]),
            "mean_output_tokens": mean_present([r["usage"]["output_tokens"] for r in records]),
            "mean_estimated_cost": mean_present([r["estimated_cost"] for r in records]),
            "mean_tiou": mean_present([(r.get("evidence_score") or {}).get("tiou") for r in records]),
            "tiou_n": sum((r.get("evidence_score") or {}).get("tiou") is not None for r in records),
            "latency_n": n - missing_counts["latency"], "usage_n": n - missing_counts["token_usage"],
            "cost_n": n - missing_counts["cost"],
            "missing_counts": missing_counts, "missing_ratios": {k: v / n for k, v in missing_counts.items()},
            "any_data_missing_ratio": sum(any(check(r) for field, check in missing_checks.items() if field != "cost")
                                          for r in records) / n,
            "failure_counts": dict(Counter(f["code"] for r in records for f in r["failures"])),
            "by_question_type": {kind: {"n": len(rows), "correct_count": sum((r.get("answer_score") or {}).get("correct") is True for r in rows),
                                        "accuracy": sum((r.get("answer_score") or {}).get("correct") is True for r in rows) / len(rows)}
                                 for kind, rows in sorted(groups.items())}}
