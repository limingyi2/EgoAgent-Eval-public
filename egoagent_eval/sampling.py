"""Deterministic candidate interval samplers.

Change scores come from local OpenCV frame differences and motion statistics;
they are not action or semantic scores.
"""

from __future__ import annotations

import math
from typing import Iterable


def sample_interval_timestamps(start: float, end: float, count: int, *,
                                policy: str = "uniform",
                                change_points: Iterable[object] = ()) -> list[float]:
    """Select bounded timestamps with uniform, peak-only, or hybrid sampling.

    ``hybrid_change`` reserves one sample in every equal-time bin and replaces
    that bin's midpoint with its strongest indexed visual-change point when
    one exists. This preserves temporal coverage while using the available
    local index signal. ``change_peaks`` is retained as an ablation and may
    cluster observations around changes.
    """
    if (not math.isfinite(start) or not math.isfinite(end) or start < 0
            or not start < end or type(count) is not int or not 1 <= count <= 8):
        raise ValueError("interval and count are invalid")
    if policy not in {"uniform", "change_peaks", "hybrid_change"}:
        raise ValueError(f"Unknown candidate sampling policy: {policy}")

    width = (end - start) / count
    midpoints = [start + (i + 0.5) * width for i in range(count)]
    if policy == "uniform":
        return midpoints

    points: list[tuple[float, float]] = []
    for point in change_points:
        timestamp = getattr(point, "timestamp", None)
        change = getattr(point, "change_score", None)
        motion = getattr(point, "motion_score", None)
        if (isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
                and isinstance(change, (int, float)) and isinstance(motion, (int, float))
                and math.isfinite(timestamp) and math.isfinite(change) and math.isfinite(motion)
                and start <= timestamp < end):
            points.append((float(timestamp), float(change + motion)))

    if policy == "hybrid_change":
        selected = []
        for index, midpoint in enumerate(midpoints):
            left = start + index * width
            right = start + (index + 1) * width
            in_bin = [(timestamp, score) for timestamp, score in points
                      if left <= timestamp < right]
            selected.append(max(in_bin, key=lambda item: (item[1], -item[0]))[0]
                            if in_bin else midpoint)
        return selected

    # Change-only ablation: select separated peaks first, then fill with
    # uniform anchors when the coarse index has too few local points.
    minimum_spacing = width / 4
    peaks = sorted(points, key=lambda item: (-item[1], item[0]))
    selected: list[float] = []
    for timestamp, score in peaks:
        if score <= 0:
            continue
        if all(abs(timestamp - existing) >= minimum_spacing for existing in selected):
            selected.append(timestamp)
        if len(selected) == count:
            break
    for midpoint in midpoints:
        if len(selected) == count:
            break
        if all(abs(midpoint - existing) >= minimum_spacing for existing in selected):
            selected.append(midpoint)
    if len(selected) < count:
        for midpoint in midpoints:
            if midpoint not in selected:
                selected.append(midpoint)
            if len(selected) == count:
                break
    return sorted(selected)
