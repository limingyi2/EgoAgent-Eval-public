from dataclasses import dataclass

import pytest

from egoagent_eval.sampling import sample_interval_timestamps


@dataclass(frozen=True)
class Point:
    timestamp: float
    change_score: float
    motion_score: float


def test_uniform_sampler_matches_existing_interval_midpoints():
    assert sample_interval_timestamps(10, 20, 4) == [11.25, 13.75, 16.25, 18.75]


def test_change_peak_sampler_uses_visual_peaks_and_fills_remaining_budget():
    points = [Point(12.0, 0.2, 0.1), Point(18.0, 0.3, 0.1)]
    selected = sample_interval_timestamps(10, 20, 4, policy="change_peaks", change_points=points)
    assert selected == [11.25, 12.0, 13.75, 18.0]


def test_hybrid_sampler_keeps_one_sample_per_time_bin_and_uses_local_peak():
    points = [Point(11.8, 0.1, 0.1), Point(12.1, 0.8, 0.1), Point(17.0, 0.5, 0.2)]
    selected = sample_interval_timestamps(10, 20, 4, policy="hybrid_change", change_points=points)
    assert selected == [12.1, 13.75, 17.0, 18.75]
    assert all(10 <= timestamp < 20 for timestamp in selected)
    assert len(selected) == len(set(selected)) == 4


@pytest.mark.parametrize("policy", ["uniform", "change_peaks", "hybrid_change"])
def test_sampling_policies_respect_candidate_interval_and_exact_budget(policy):
    points = [Point(5.0, 1.0, 1.0), Point(10.0, 0.4, 0.2), Point(20.0, 0.8, 0.1)]
    selected = sample_interval_timestamps(10, 20, 4, policy=policy, change_points=points)
    assert len(selected) == 4
    assert all(10 <= timestamp < 20 for timestamp in selected)


def test_sampling_rejects_unknown_policy_and_invalid_count():
    with pytest.raises(ValueError, match="Unknown candidate sampling"):
        sample_interval_timestamps(0, 10, 4, policy="semantic_guess")
    with pytest.raises(ValueError, match="interval and count"):
        sample_interval_timestamps(0, 10, 0)
