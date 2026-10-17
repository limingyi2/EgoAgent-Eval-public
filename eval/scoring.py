"""Deterministic answer and temporal evidence scoring, with audit details."""

import json
import unicodedata

from .data import interval, number, strict_loads


NORMALIZATION = "Unicode NFKC + casefold + collapse whitespace; preserve punctuation"


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def parse_prediction(output, duration):
    value = strict_loads(output) if isinstance(output, str) else output
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
        raise ValueError("prediction must be a JSON object with nonempty string answer")
    timestamps = value.get("evidence_timestamps", [])
    if not isinstance(timestamps, list) or any(not number(t) or t < 0 or t > duration for t in timestamps):
        raise ValueError("evidence_timestamps must be finite in-video seconds")
    predicted_interval = value.get("evidence_interval")
    if predicted_interval is not None and (not interval(predicted_interval) or predicted_interval[1] > duration):
        raise ValueError("evidence_interval must be a valid in-video interval")
    return {"answer": value["answer"], "evidence_timestamps": timestamps,
            "evidence_interval": predicted_interval}


def score_answer(task, prediction):
    candidates = [task["standard_answer"], *task.get("aliases", [])]
    audit = {"normalization": NORMALIZATION, "normalized_prediction": normalize(prediction["answer"]),
             "accepted_answers": candidates, "normalized_answers": [normalize(a) for a in candidates]}
    if task.get("answer_mode") == "manual":
        return {**audit, "correct": None, "verdict": "manual_review", "matched_answer": None}
    for i, candidate in enumerate(audit["normalized_answers"]):
        if audit["normalized_prediction"] == candidate:
            return {**audit, "correct": True, "verdict": "exact" if i == 0 else "alias",
                    "matched_answer": candidates[i]}
    return {**audit, "correct": False, "verdict": "incorrect", "matched_answer": None}


def tiou(predicted, event):
    intersection = max(0.0, min(predicted[1], event[1]) - max(predicted[0], event[0]))
    union = max(predicted[1], event[1]) - min(predicted[0], event[0])
    return intersection / union if union else float(predicted == event)


def score_evidence(task, prediction, tolerance):
    windows = task["event_windows"]
    timestamps = prediction["evidence_timestamps"]
    predicted = prediction["evidence_interval"]
    timestamp_matches = [any(max(0, a - tolerance) <= t <= b + tolerance for a, b in windows)
                         for t in timestamps]
    interval_matches = ([predicted[1] >= max(0, a - tolerance) and predicted[0] <= b + tolerance
                         for a, b in windows] if predicted is not None else [])
    return {"timestamp_hit": any(timestamp_matches) or any(interval_matches),
            "timestamp_matches": timestamp_matches, "interval_matches": interval_matches,
            "event_windows": windows, "tolerance_seconds": tolerance,
            "hit_rule": "any point in padded window OR predicted interval overlaps padded window (inclusive)",
            "tiou": max(tiou(predicted, window) for window in windows) if predicted is not None else None,
            "tiou_rule": "maximum IoU against unpadded human event windows",
            "missing": not timestamps and predicted is None}
