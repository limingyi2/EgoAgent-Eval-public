import csv
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from egoagent_eval.experiment_ledger import (
    ANSWER_STATUSES, EVIDENCE_STATUSES, RESULT_COLUMNS, BudgetExceeded,
    ExperimentLedger, LedgerError,
)
from egoagent_eval.cli import main as cli_main
from egoagent_eval.video import Video
from egoagent_eval.vlm import FakeVLMProvider, MiMoVLMProvider, ProviderFailure


def make_ledger(tmp_path):
    experiment_dir = tmp_path / "experiment"
    return ExperimentLedger.initialize(experiment_dir / "request_ledger.sqlite3",
                                       experiment_id="p2.8.21-test",
                                       snapshot_id="snapshot-test-sha256",
                                       _registry_path=tmp_path / "identity_registry.sqlite3")


def register(ledger, strategy, task, run_id=None):
    run_id = run_id or f"{strategy}-{task}"
    ledger.register_run(strategy=strategy, task_id=task, run_id=run_id,
                        video_sha256="a" * 64, model_id="mimo-v2.6-flash")
    return run_id


def _competing_reservation(args):
    path, strategy, task = args
    ledger = ExperimentLedger(path, experiment_id="p2.8.21-test", snapshot_id="snapshot-test-sha256")
    run_id = f"{strategy}-{task}"
    try:
        attempt = ledger.reserve_request(run_id=run_id)
        ledger.mark_attempting(attempt)
        ledger.finish_attempt(attempt, http_status=200, error_type=None)
        return "reserved"
    except BudgetExceeded:
        return "blocked"


def test_strategy_budgets_global_budget_and_restart_persistence(tmp_path):
    ledger = make_ledger(tmp_path)
    for strategy, prefix, count in (("uniform8", "U", 4), ("v4_staged", "V", 10)):
        for index in range(count):
            register(ledger, strategy, f"{prefix}{index}")
    for index in range(3):
        attempt = ledger.reserve_request(run_id=f"uniform8-U{index}")
        ledger.mark_attempting(attempt)
        ledger.finish_attempt(attempt, http_status=200, error_type=None)
    with pytest.raises(BudgetExceeded, match="uniform8"):
        ledger.reserve_request(run_id="uniform8-U3")
    for index in range(9):
        attempt = ledger.reserve_request(run_id=f"v4_staged-V{index}")
        ledger.mark_attempting(attempt)
        ledger.finish_attempt(attempt, http_status=503, error_type="http_error")
    assert ledger.counts() == {"uniform8": 3, "v4_staged": 9, "global": 12}
    with pytest.raises(BudgetExceeded, match="Global"):
        ledger.reserve_request(run_id="v4_staged-V9")
    reopened = ExperimentLedger(ledger.path, experiment_id=ledger.experiment_id,
                                snapshot_id=ledger.snapshot_id)
    assert reopened.counts()["global"] == 12


def test_fake_provider_failure_consumes_a_durable_reservation(video_path, tmp_path):
    ledger = make_ledger(tmp_path)
    run_id = register(ledger, "uniform8", "task-a", "failed-run")
    fake = FakeVLMProvider([{"__fake_error__": "timeout"}], request_gate=ledger, run_id=run_id)
    with Video(video_path) as video, pytest.raises(ProviderFailure):
        fake.complete("fake only", video.uniform_frames(1))
    assert ledger.counts()["uniform8"] == 1
    attempts = ledger.attempts_for_run(run_id)
    assert attempts == [(503, "response_received", "timeout")]
    reopened = ExperimentLedger(ledger.path, experiment_id=ledger.experiment_id,
                                snapshot_id=ledger.snapshot_id)
    assert reopened.counts()["global"] == 1


def test_processes_cannot_race_past_independent_group_limits(tmp_path):
    ledger = make_ledger(tmp_path)
    for strategy, prefix in (("uniform8", "U"), ("v4_staged", "V")):
        for index in range(10):
            register(ledger, strategy, f"{prefix}{index}")
    jobs = [(str(ledger.path), strategy, f"{prefix}{index}")
            for strategy, prefix in (("uniform8", "U"), ("v4_staged", "V"))
            for index in range(10)]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        outcomes = pool.map(_competing_reservation, jobs)
    assert outcomes.count("reserved") == 12
    assert outcomes.count("blocked") == 8
    assert ledger.counts() == {"uniform8": 3, "v4_staged": 9, "global": 12}


def test_unknown_after_crash_stays_consumed_and_duplicate_run_is_refused(tmp_path):
    ledger = make_ledger(tmp_path)
    run_id = register(ledger, "uniform8", "task-a", "crashed")
    attempt = ledger.reserve_request(run_id=run_id)
    ledger.mark_attempting(attempt)
    reopened = ExperimentLedger(ledger.path, experiment_id=ledger.experiment_id,
                                snapshot_id=ledger.snapshot_id)
    assert reopened.counts()["global"] == 1
    assert reopened.attempts_for_run(run_id) == [(None, "attempting", None)]
    with pytest.raises(LedgerError, match="Duplicate"):
        register(reopened, "uniform8", "task-a", "new-rerun")
    with pytest.raises(LedgerError, match="Refusing to (overwrite|reuse)"):
        ExperimentLedger.initialize(ledger.path, experiment_id=ledger.experiment_id,
                                    snapshot_id=ledger.snapshot_id)
    assert reopened.counts()["global"] == 1


def test_experiment_id_cannot_be_reinitialized_under_a_second_path(tmp_path):
    registry = tmp_path / "identity_registry.sqlite3"
    first = ExperimentLedger.initialize(tmp_path / "first" / "request_ledger.sqlite3",
                                        experiment_id="same-experiment",
                                        snapshot_id="snapshot-test-sha256",
                                        _registry_path=registry)
    attempt_run = register(first, "uniform8", "task-a", "unique-run")
    first.reserve_request(run_id=attempt_run)
    with pytest.raises(LedgerError, match="already exists in the machine registry"):
        ExperimentLedger.initialize(tmp_path / "second" / "request_ledger.sqlite3",
                                    experiment_id="same-experiment",
                                    snapshot_id="snapshot-test-sha256",
                                    _registry_path=registry)
    assert ExperimentLedger(first.path, experiment_id="same-experiment",
                            snapshot_id="snapshot-test-sha256").counts()["global"] == 1


def test_corrupt_or_unpersistable_ledger_fails_before_provider_send(video_path, tmp_path, monkeypatch):
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not a sqlite database")
    with pytest.raises(LedgerError):
        ExperimentLedger(corrupt, experiment_id="p2.8.21-test", snapshot_id="snapshot-test-sha256")

    ledger = make_ledger(tmp_path / "good")
    run_id = register(ledger, "uniform8", "task-a", "blocked-send")
    def fail_reservation(*, run_id):
        raise LedgerError("simulated read-only ledger")
    monkeypatch.setattr(ledger, "reserve_request", fail_reservation)
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: pytest.fail("network must not be opened"))
    provider = MiMoVLMProvider(api_key="not-a-real-key", request_gate=ledger, run_id=run_id)
    with Video(video_path) as video, pytest.raises(ProviderFailure):
        provider.complete("never send", video.uniform_frames(1))
    assert provider.calls[0].network_attempted is False
    assert provider.calls[0].error.code == "budget_ledger_error"


def test_live_provider_without_gate_is_closed_before_network(video_path, monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: pytest.fail("unguarded network path"))
    provider = MiMoVLMProvider(api_key="not-a-real-key")
    with Video(video_path) as video, pytest.raises(ProviderFailure):
        provider.complete("guard required", video.uniform_frames(1))
    assert provider.calls[0].network_attempted is False
    assert provider.calls[0].error.code == "budget_ledger_error"
    custom_transport = MiMoVLMProvider(api_key="not-a-real-key", transport=httpx.HTTPTransport())
    assert custom_transport.is_fake is False
    with Video(video_path) as video, pytest.raises(ProviderFailure):
        custom_transport.complete("custom transport is still guarded", video.uniform_frames(1))
    assert custom_transport.calls[0].network_attempted is False


def test_real_cli_requires_the_shared_experiment_ledger(video_path):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--video", str(video_path), "--question", "offline", "--provider", "mimo",
                  "--method", "uniform8"])
    assert exc.value.code == 2


def test_real_cli_rejects_outputs_outside_new_experiment_directory(video_path, tmp_path):
    experiment = tmp_path / "fresh-experiment"
    experiment.mkdir()
    ledger_path = experiment / "request_ledger.sqlite3"
    ExperimentLedger.initialize(ledger_path, experiment_id="p2.8.21-test",
                                snapshot_id="snapshot-test-sha256",
                                _registry_path=experiment / "identity_registry.sqlite3")
    with pytest.raises(SystemExit) as exc:
        cli_main(["--video", str(video_path), "--question", "offline", "--provider", "mimo",
                  "--method", "uniform8", "--experiment-ledger", str(ledger_path),
                  "--experiment-id", "p2.8.21-test", "--snapshot-id", "snapshot-test-sha256",
                  "--task-id", "task-a", "--output", str(tmp_path / "historical-output")])
    assert exc.value.code == 2
    assert ExperimentLedger(ledger_path, experiment_id="p2.8.21-test",
                            snapshot_id="snapshot-test-sha256").counts()["global"] == 0


def test_exhausted_global_budget_blocks_real_provider_before_http(video_path, tmp_path, monkeypatch):
    ledger = make_ledger(tmp_path)
    for strategy, prefix, count in (("uniform8", "U", 4), ("v4_staged", "V", 10)):
        for index in range(count):
            register(ledger, strategy, f"{prefix}{index}")
    for strategy, prefix, count in (("uniform8", "U", 3), ("v4_staged", "V", 9)):
        for index in range(count):
            attempt = ledger.reserve_request(run_id=f"{strategy}-{prefix}{index}")
            ledger.mark_attempting(attempt)
            ledger.finish_attempt(attempt, http_status=200, error_type=None)
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: pytest.fail("13th request must not reach HTTP"))
    provider = MiMoVLMProvider(api_key="not-a-real-key", request_gate=ledger, run_id="v4_staged-V9")
    with Video(video_path) as video, pytest.raises(ProviderFailure):
        provider.complete("blocked before network", video.uniform_frames(1))
    assert provider.calls[0].error.code == "budget_exhausted"
    assert provider.calls[0].network_attempted is False
    assert ledger.counts()["global"] == 12


def test_six_paired_fake_cli_runs_share_budget_and_result_ledger(video_path, tmp_path):
    experiment_dir = tmp_path / "paired-fake"
    ledger_path = experiment_dir / "request_ledger.sqlite3"
    ledger = ExperimentLedger.initialize(
        ledger_path, experiment_id="p2.8.23-fake-six-run",
        snapshot_id="offline-test-snapshot",
        _registry_path=tmp_path / "identity_registry.sqlite3")
    plan = [
        ("task-a", "uniform8"), ("task-a", "hierarchical_active_v4_staged_observation"),
        ("task-b", "hierarchical_active_v4_staged_observation"), ("task-b", "uniform8"),
        ("task-c", "uniform8"), ("task-c", "hierarchical_active_v4_staged_observation"),
    ]
    output_roots = []
    for task_id, method in plan:
        label = "uniform8" if method == "uniform8" else "v4_staged"
        output = experiment_dir / "runs" / f"{task_id}-{label}"
        output_roots.append(output)
        assert cli_main([
            "--video", str(video_path), "--question", f"Fake-only {task_id}",
            "--method", method, "--v4-candidate-selection-policy", "baseline",
            "--provider", "fake", "--experiment-ledger", str(ledger_path),
            "--experiment-id", "p2.8.23-fake-six-run",
            "--snapshot-id", "offline-test-snapshot", "--task-id", task_id,
            "--output", str(output),
        ]) in (0, 1)
    assert len(set(output_roots)) == 6
    assert all(path.is_dir() for path in output_roots)
    assert ledger.counts() == {"uniform8": 3, "v4_staged": 9, "global": 12}
    rows = list(csv.DictReader(ledger.csv_path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 6
    assert {row["task_id"] for row in rows} == {"task-a", "task-b", "task-c"}
    assert sum(int(row["request_attempts"]) for row in rows) == 12
    assert all(row["model_id"] == "fake-scripted-v1" for row in rows)


def test_results_csv_has_fixed_schema_distinct_statuses_and_review_guard(tmp_path):
    ledger = make_ledger(tmp_path)
    rows = [
        ("U", "correct", "ok", "answered"),
        ("V", "refused", "refused", "provider_refusal"),
        ("U", "protocol_error", "failed", "protocol_error"),
        ("V", "insufficient", "refused", "insufficient_evidence"),
        ("U", "run_failure", "failed", "provider_failure"),
    ]
    row_runs = []
    for index, (prefix, label, status, reason) in enumerate(rows):
        strategy = "uniform8" if prefix == "U" else "v4_staged"
        run_id = register(ledger, strategy, f"case-{index}", f"run-{index}")
        fake_result = SimpleNamespace(
            status=("refused" if status == "insufficient" else status), termination_reason=reason, errors=[],
            observed_frames=[object()], latency_seconds=1.25,
            usage=SimpleNamespace(input_tokens=10, output_tokens=3),
        )
        ledger.complete_run(run_id, fake_result)
        row_runs.append((run_id, label))
    ledger.record_answer_review(row_runs[0][0], answer_status="correct", answer_correct=True)
    ledger.record_answer_review(row_runs[2][0], answer_status="protocol_error", answer_correct=None)
    ledger.record_evidence_review(row_runs[0][0], evidence_status="sufficient", reviewer="reviewer-1",
                                  independent=True, blinded=True)
    with pytest.raises(LedgerError, match="independent blinded"):
        ledger.record_evidence_review(row_runs[4][0], evidence_status="sufficient", reviewer="self",
                                      independent=False, blinded=False)
    with ledger.csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        materialized = list(reader)
        assert tuple(reader.fieldnames) == RESULT_COLUMNS
    assert len(materialized) == 5
    by_run = {row["run_id"]: row for row in materialized}
    assert by_run[row_runs[0][0]]["answer_status"] == "correct"
    assert by_run[row_runs[0][0]]["answer_correct"] == "true"
    assert by_run[row_runs[0][0]]["evidence_status"] == "sufficient"
    assert by_run[row_runs[1][0]]["answer_status"] == "refused"
    assert by_run[row_runs[2][0]]["answer_status"] == "protocol_error"
    assert by_run[row_runs[3][0]]["evidence_status"] == "insufficient"
    assert by_run[row_runs[4][0]]["run_status"] == "failed"
    assert by_run[row_runs[4][0]]["answer_status"] == "unknown"
    assert ANSWER_STATUSES == {"correct", "incorrect", "refused", "protocol_error", "unscorable", "unknown"}
    assert EVIDENCE_STATUSES == {"sufficient", "insufficient", "manual_review_required", "unknown"}


def test_invalid_result_state_and_duplicate_csv_rows_fail_validation(tmp_path):
    ledger = make_ledger(tmp_path)
    run_id = register(ledger, "uniform8", "task-a")
    with pytest.raises(LedgerError, match="Invalid answer_status"):
        ledger._update_result(run_id, {"answer_status": "maybe"})
    with ledger.csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        row = next(reader)
    with ledger.csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)
    with pytest.raises(LedgerError, match="duplicate run_id"):
        ledger.validate_results_csv()


def test_result_write_failure_does_not_make_run_replayable(tmp_path, monkeypatch):
    ledger = make_ledger(tmp_path)
    run_id = register(ledger, "uniform8", "task-a")
    attempt = ledger.reserve_request(run_id=run_id)
    ledger.mark_attempting(attempt)
    ledger.finish_attempt(attempt, http_status=200, error_type=None)
    result = SimpleNamespace(status="ok", termination_reason="answered", errors=[],
                             observed_frames=[object()], latency_seconds=1.0,
                             usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    monkeypatch.setattr(ledger, "_export_results", lambda: (_ for _ in ()).throw(LedgerError("disk full")))
    with pytest.raises(LedgerError, match="disk full"):
        ledger.complete_run(run_id, result)
    assert ledger.counts()["global"] == 1
    with pytest.raises(LedgerError, match="Duplicate"):
        register(ledger, "uniform8", "task-a", "result-write-retry")
