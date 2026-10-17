import json
from pathlib import Path
import hashlib

import pytest

from eval import STRATEGIES
from eval.analysis import aggregate, load_logs
from eval.data import Config, digest, load_tasks, runner_task, uniform_times
from eval.pipeline import BudgetExceeded, Recorder, evaluate_one, run_experiment
from eval.runners import FakeRunner
from eval.scoring import parse_prediction, score_answer, score_evidence, tiou


FIXTURE = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "tasks.jsonl"


@pytest.fixture
def tasks():
    return load_tasks(FIXTURE, mock=True)


@pytest.fixture
def config():
    return Config("unit-mock", "fake-vlm", seed=42)


def rewrite(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exact_alias_audit_and_punctuation(tasks):
    task = tasks[0]
    exact = score_answer(task, {"answer": " ＢＬＵＥ\n "})
    assert exact["correct"] and exact["verdict"] == "exact"
    alias = score_answer(task, {"answer": "AZURE"})
    assert alias["verdict"] == "alias" and alias["matched_answer"] == "azure"
    assert score_answer(task, {"answer": "blue."})["correct"] is False
    assert "preserve punctuation" in alias["normalization"]


def test_manual_never_automatically_correct(tasks):
    assert score_answer(tasks[8], {"answer": "blue"})["correct"] is None


@pytest.mark.parametrize("timestamps,expected", [([3.5], True), ([3.49], False), ([6.5], True), ([6.51], False), ([], False)])
def test_timestamp_hit_boundaries(tasks, timestamps, expected):
    result = score_evidence(tasks[0], {"evidence_timestamps": timestamps, "evidence_interval": None}, 0.5)
    assert result["timestamp_hit"] is expected
    assert result["event_windows"] == [[4, 6]] and result["tolerance_seconds"] == 0.5


def test_interval_hit_and_tiou_unpadded(tasks):
    result = score_evidence(tasks[0], {"evidence_timestamps": [], "evidence_interval": [3, 5]}, 0.5)
    assert result["timestamp_hit"] and result["tiou"] == pytest.approx(1 / 3)
    result = score_evidence(tasks[0], {"evidence_timestamps": [], "evidence_interval": [3, 3.5]}, 0.5)
    assert result["timestamp_hit"] and result["tiou"] == 0
    assert tiou([4, 4], [4, 4]) == 1


def test_multiple_event_windows(tasks):
    task = {**tasks[0], "event_windows": [[1, 2], [4, 6]]}
    result = score_evidence(task, {"evidence_timestamps": [1.5], "evidence_interval": [4, 6]}, 0)
    assert result["timestamp_hit"] and result["tiou"] == 1


@pytest.mark.parametrize("output", ["{bad", "[]", '{}', '{"answer":2}', '{"answer":""}',
                                      '{"answer":"blue","evidence_timestamps":[true]}',
                                      '{"answer":"blue","evidence_timestamps":[NaN]}',
                                      '{"answer":"blue","evidence_timestamps":[13]}',
                                      '{"answer":"blue","evidence_interval":[5,4]}'])
def test_invalid_prediction(output):
    with pytest.raises((ValueError, TypeError)):
        parse_prediction(output, 12)


@pytest.mark.parametrize("kwargs", [{"temperature": float("nan")}, {"max_output_tokens": 0},
                                    {"seed": True}, {"tolerance_seconds": -1},
                                    {"input_price_per_million": 1}, {"temperature": True}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config("test", "model", **kwargs)


def test_fixture_rejected_for_real():
    with pytest.raises(ValueError, match="human_verified"):
        load_tasks(FIXTURE)


def test_real_task_hash_and_validation(tmp_path):
    video = tmp_path / "clip.bin"
    video.write_bytes(b"test video contents")
    task = {"task_id": "real-1", "video_path": "clip.bin", "question": "Q?", "question_type": "closed",
            "standard_answer": "A", "event_time_range": [1, 2], "duration_seconds": 3, "human_verified": True}
    path = tmp_path / "tasks.jsonl"
    rewrite(path, [task])
    loaded = load_tasks(path)
    assert len(loaded[0]["video_sha256"]) == 64
    assert loaded[0]["video_path"] == str(video.resolve())
    rewrite(path, [task, task])
    with pytest.raises(ValueError, match="duplicate"):
        load_tasks(path)


def test_ground_truth_not_exposed(tasks):
    visible = runner_task(tasks[0])
    assert not {"standard_answer", "aliases", "event_windows", "answer_mode", "event_time_range"} & visible.keys()


def test_full_pipeline_keeps_all_failures_and_denominators(tmp_path, tasks, config):
    path = run_experiment(tasks, config, FakeRunner(), tmp_path, mode="mock")
    rows, manifests, notes = load_logs([path], mode="mock")
    assert len(rows) == len(tasks) * 3 and not notes
    for strategy in STRATEGIES:
        group = [r for r in rows if r["strategy"] == strategy]
        metrics = aggregate(group)
        assert metrics["n"] == 10
        assert metrics["accuracy"] == 0.4  # exact, alias, missing evidence, missing usage
        assert metrics["evidence_hit_rate"] == 0.5  # includes incorrect and manual answers with valid evidence
        assert metrics["manual_review_count"] == 1
        assert metrics["by_question_type"]["color"]["accuracy"] == 0.75
        assert metrics["missing_ratios"]["token_usage"] == 0.4
        assert metrics["missing_ratios"]["cost"] == 1
        for code in ("incorrect", "invalid_json", "tool_error", "budget_exceeded", "insufficient_evidence", "manual_review", "runner_error"):
            assert metrics["failure_counts"][code] == 1
    exact = [r for r in rows if r["task_id"] == "fixture-exact"]
    assert [r["request_count"] for r in exact] == [1, 1, 2]
    assert [r["observed_frame_count"] for r in exact] == [4, 8, 8]
    assert [r["model_input_image_count"] for r in exact] == [4, 8, 12]
    assert exact[0]["frames"] == exact[2]["frames"][:4]
    assert all(r["effective_seed"] == 42 for r in rows)
    assert all(r["elapsed_seconds"] >= 0 for r in rows)
    assert manifests[0]["config"]["model"] == "fake-vlm"


def test_resource_means_and_cost(tmp_path, tasks):
    config = Config("prices", "fake-vlm", input_price_per_million=2, output_price_per_million=4)
    path = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    rows, _, _ = load_logs([path], mode="mock")
    assert rows[0]["estimated_cost"] == pytest.approx(0.00036)
    assert rows[2]["estimated_cost"] == pytest.approx(0.0006)
    metrics = aggregate(rows)
    assert metrics["mean_requests"] == pytest.approx(4 / 3)
    assert metrics["mean_observed_frames"] == pytest.approx(20 / 3)
    assert metrics["usage_n"] == 3 and metrics["cost_n"] == 3


def test_never_overwrites_history(tmp_path, tasks, config):
    first = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    original = first.read_bytes()
    second = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    assert first != second and first.read_bytes() == original
    with pytest.raises(ValueError, match="duplicate run_id"):
        load_logs([first, second], mode="mock")


def test_incomplete_run_restores_missing_denominator(tmp_path, tasks, config):
    path = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    rewrite(path, read_rows(path)[:2])
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"partial":')
    rows, _, notes = load_logs([path], mode="mock")
    assert len(rows) == 3
    assert aggregate(rows)["accuracy"] == pytest.approx(1 / 3)
    assert aggregate(rows)["missing_counts"]["execution"] == 2
    assert aggregate(rows)["mean_requests"] == 1  # absent execution is unknown, not zero
    assert any("incomplete final" in note for note in notes)


def test_recompute_scores_and_totals(tmp_path, tasks, config):
    path = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    rows = read_rows(path)
    rows[1]["answer_score"]["correct"] = False
    rows[1]["request_count"] = 900
    rows[1]["estimated_cost"] = 999
    rewrite(path, rows)
    restored, _, _ = load_logs([path], mode="mock")
    assert restored[0]["answer_score"]["correct"] is True
    assert restored[0]["request_count"] == 1 and restored[0]["estimated_cost"] is None


def test_mixing_modes_and_fake_real_runner_rejected(tmp_path, tasks, config):
    with pytest.raises(ValueError, match="provenance"):
        run_experiment(tasks, config, FakeRunner(), tmp_path, mode="real")
    path = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    with pytest.raises(ValueError, match="mix real and mock"):
        load_logs([path], mode="real")


def test_compare_config_and_dataset_constraints(tmp_path, tasks, config):
    first = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    second = run_experiment(tasks[:1], Config("other", "different-model"), FakeRunner(), tmp_path, mode="mock")
    with pytest.raises(ValueError, match="identical"):
        load_logs([first, second], mode="mock")
    selected, _, _ = load_logs([first, second], mode="mock", run_id="unit-mock")
    assert len(selected) == 3


def test_duplicate_matrix_entry_rejected(tmp_path, tasks, config):
    path = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    rows = read_rows(path)
    rows.insert(2, rows[1])
    rewrite(path, rows)
    with pytest.raises(ValueError, match="duplicate task"):
        load_logs([path], mode="mock")


def test_seed_unsupported_recorded(tmp_path, tasks, config):
    runner = FakeRunner()
    runner.supports_seed = False
    path = run_experiment(tasks[:1], config, runner, tmp_path, mode="mock")
    rows, _, _ = load_logs([path], mode="mock")
    assert all(r["effective_seed"] is None and r["config"]["seed"] == 42 for r in rows)


def test_request_and_output_budgets(tasks, config):
    recorder = Recorder(tasks[0], "Direct-4", config, FakeRunner())
    timestamps = uniform_times(12, 4)
    recorder.frames(timestamps, [f"mock://{i}" for i in range(4)])
    model_input = {"question": "Q", "images": recorder.observation_images()}
    recorder.request(model_input, lambda: {"output": "{}", "usage": None})
    with pytest.raises(BudgetExceeded, match="request budget"):
        recorder.request(model_input, lambda: {"output": "{}"})
    recorder = Recorder(tasks[0], "Direct-4", config, FakeRunner())
    recorder.frames(timestamps, [f"mock://{i}" for i in range(4)])
    with pytest.raises(BudgetExceeded, match="max_output_tokens"):
        recorder.request(model_input, lambda: {"output": "{}", "usage": {"input_tokens": 1, "output_tokens": 513}})


def test_malformed_usage_is_logged_failure_not_pipeline_crash(tasks, config):
    class BadUsage(FakeRunner):
        def run(self, task, strategy, config, recorder):
            timestamps = uniform_times(12, 4)
            recorder.frames(timestamps, [f"mock://{i}" for i in range(4)])
            return recorder.request({"question": "Q", "images": recorder.observation_images()},
                                    lambda: {"output": '{"answer":"blue"}', "usage": {"input_tokens": "bad"}})
    row = evaluate_one(tasks[0], "Direct-4", config, BadUsage())
    assert row["failures"][0]["code"] == "protocol_error"
    assert not row["usage"]["complete"]
    json.dumps(row, allow_nan=False)


def test_provider_provenance_rejected_without_logging_fake_output(tasks, config, tmp_path):
    class RealDeclared:
        is_mock = False
        supports_seed = False
        name = "declared-real-test-only"
        def run(self, task, strategy, config, recorder):
            image = tmp_path / "frame.jpg"
            image.write_bytes(b"test-frame")
            recorder.frames(uniform_times(12, 4), [str(image)] * 4)
            return recorder.request({"question": "Q", "images": recorder.observation_images()},
                                    lambda: {"output": "fake-output-must-not-be-stored", "provider_kind": "mock"})
    row = evaluate_one(tasks[0], "Direct-4", config, RealDeclared())
    assert row["failures"][0]["code"] == "protocol_error"
    assert row["requests"][0]["output"] is None
    assert "fake-output-must-not-be-stored" not in json.dumps(row)


def test_cli_pipeline_and_report(tmp_path):
    from scripts.run_eval import main as run_main
    from scripts.analyze import main as analyze_main
    path = run_main(["--fixture", "--mode", "mock", "--runner", "fake", "--run-id", "cli",
                     "--model", "fake-vlm", "--results-dir", str(tmp_path / "results")])
    summary = analyze_main(["--logs", str(path), "--mode", "mock", "--report-dir", str(tmp_path / "reports")])
    assert "MOCK / DRY RUN" in summary.read_text(encoding="utf-8")
    for name in ("strategy_accuracy.png", "evidence_hit.png", "resources_cost.png"):
        assert (summary.parent / name).read_bytes().startswith(b"\x89PNG")
    saved = json.loads((summary.parent / "metrics.json").read_text(encoding="utf-8"))
    assert saved["overall"]["n"] == 30 and saved["overall"]["accuracy"] == 0.4
    assert len(saved["inputs"][0]["sha256"]) == 64
    failures = (summary.parent / "failures.md").read_text(encoding="utf-8")
    assert "requires_human" in failures and "未评审" in failures
    assert len(list((summary.parent / "runs").glob("*/summary.md"))) == 1


def test_cli_forbids_fixture_real():
    from scripts.run_eval import main
    with pytest.raises(SystemExit):
        main(["--fixture", "--mode", "real", "--runner", "fake", "--run-id", "bad", "--model", "model"])


def test_unknown_answer_mode_defaults_manual(tmp_path):
    task = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    task.pop("answer_mode")
    path = tmp_path / "tasks.jsonl"
    rewrite(path, [task])
    assert load_tasks(path, mock=True)[0]["answer_mode"] == "manual"


def test_dataset_task_contract_is_adapted_without_agent_dependency(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"unit-test media, not a real video")
    sha = hashlib.sha256(media.read_bytes()).hexdigest()
    task = {"task_id": "unit-dataset-task", "video_path": "clip.mp4", "question": "Unit-test Q?",
            "answer": "cup", "answer_type": "object_interaction", "event_start": 1, "event_end": 2,
            "source": {"video_end": 3, "is_synthetic": False}, "verification_status": "verified",
            "video_sha256": sha,
            "verification": {"decision": "approve", "reviewer": "unit-test-only", "reviewed_at": "2026-10-11T21:00:00+08:00",
                             "watched_video_sha256": sha, "question": "Unit-test Q?", "answer": "cup",
                             "event_start": 1, "event_end": 2, "accepted_answers": ["mug"]}}
    path = tmp_path / "tasks.jsonl"
    rewrite(path, [task])
    loaded = load_tasks(path, video_root=tmp_path)[0]
    assert loaded["human_verified"] and loaded["question_type"] == "object_interaction"
    assert loaded["aliases"] == ["mug"] and loaded["duration_seconds"] == 3
    assert loaded["video_sha256"] == sha
    task["answer"] = ["pick up cup", "put down cup"]
    task["answer_type"] = "action_sequence"
    task["verification"]["answer"] = task["answer"]
    task["verification"]["accepted_answers"] = []
    rewrite(path, [task])
    loaded = load_tasks(path, video_root=tmp_path)[0]
    assert loaded["answer_mode"] == "manual"
    task["verification_status"] = "pending"
    task["verification"] = None
    rewrite(path, [task])
    with pytest.raises(ValueError, match="human_verified"):
        load_tasks(path, video_root=tmp_path)


def test_video_checksum_is_not_silently_replaced(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"unit-test-only")
    task = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    task.update(video_path="clip.mp4", fixture=False, human_verified=True, video_sha256="0" * 64)
    path = tmp_path / "tasks.jsonl"
    rewrite(path, [task])
    with pytest.raises(ValueError, match="checksum"):
        load_tasks(path)


def test_swallowed_budget_failure_does_not_become_correct(tasks, config):
    class SwallowedBudget(FakeRunner):
        def run(self, task, strategy, config, recorder):
            timestamps = uniform_times(12, 4)
            recorder.frames(timestamps, [f"mock://{i}" for i in range(4)])
            try:
                recorder.frames([1], ["mock://extra"])
            except BudgetExceeded:
                pass
            return recorder.request({"question": "Q", "images": recorder.observation_images()},
                                    lambda: {"output": '{"answer":"blue"}', "usage": None})
    row = evaluate_one(tasks[0], "Direct-4", config, SwallowedBudget())
    assert row["failures"][0]["code"] == "budget_exceeded" and row["answer_score"] is None


def test_saved_success_cannot_conceal_tool_failure(tmp_path, tasks, config):
    path = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    rows = read_rows(path)
    rows[1]["tool_calls"][0]["error"] = {"type": "OSError", "message": "unit-test tampering"}
    rewrite(path, rows)
    with pytest.raises(ValueError, match="violates strategy/trace"):
        load_logs([path], mode="mock")


def test_interrupted_at_manifest_still_has_entire_denominator(tmp_path, tasks, config):
    path = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    rewrite(path, read_rows(path)[:1])
    rows, _, notes = load_logs([path], mode="mock")
    metrics = aggregate(rows)
    assert metrics["n"] == 3 and metrics["accuracy"] == 0
    assert metrics["mean_elapsed_seconds"] is None and metrics["mean_requests"] is None
    assert metrics["missing_ratios"]["execution"] == 1


def test_successful_compatible_repeated_runs_pool_attempts(tmp_path, tasks, config):
    first = run_experiment(tasks[:1], config, FakeRunner(), tmp_path, mode="mock")
    second = run_experiment(tasks[:1], Config("another", "fake-vlm", seed=42), FakeRunner(), tmp_path, mode="mock")
    rows, manifests, _ = load_logs([first, second], mode="mock")
    assert len(manifests) == 2 and aggregate(rows)["n"] == 6
    assert aggregate(rows)["accuracy"] == 1
