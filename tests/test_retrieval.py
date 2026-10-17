"""Candidate-pool and candidate-action tests; no real provider calls."""
import json
from pathlib import Path

import pytest

from egoagent_eval.core.agent import AgentHarness
from egoagent_eval.core.controller import (CandidateFirstHierarchicalRetrieval,
                                          CandidateHierarchicalRetrieval, CompactContext,
                                          default_prompt)
from egoagent_eval.cli import main
from egoagent_eval.index import build_time_index
from egoagent_eval.retrieval import build_candidate_pool
from egoagent_eval.sampling import sample_interval_timestamps
from egoagent_eval.schema import Budget, Question
from egoagent_eval.video import Video
from egoagent_eval.vlm import FakeVLMProvider
from egoagent_eval.hierarchical import run_hierarchical_active_v3_candidate_first
from scripts.make_test_video import create_video


def answer(timestamp):
    return {"action": "final_answer", "answer": "A synthetic test answer",
            "evidence_timestamps": [timestamp], "status": "ok"}


def test_candidate_ids_are_unique_reproducible_and_cover_video(video_path):
    with Video(video_path) as video:
        first = build_candidate_pool(build_time_index(video, coarse_samples=8, candidate_limit=3))
        second = build_candidate_pool(build_time_index(video, coarse_samples=8, candidate_limit=3))
    assert first.candidate_pool_id == second.candidate_pool_id
    assert [c.candidate_id for c in first.candidates] == [c.candidate_id for c in second.candidates]
    assert len({c.candidate_id for c in first.candidates}) == len(first.candidates)
    assert any(c.source == "uniform_coverage" for c in first.candidates)
    assert all(0 <= c.start < c.end <= first.video_duration for c in first.candidates)


def test_change_context_padding_is_generic_bounded_and_changes_pool_identity(video_path):
    with Video(video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=3)
        base = build_candidate_pool(index)
        padded = build_candidate_pool(index, change_context_padding=0.5)
    assert base.candidate_pool_id != padded.candidate_pool_id
    base_uniform = [c for c in base.candidates if c.source == "uniform_coverage"]
    padded_uniform = [c for c in padded.candidates if c.source == "uniform_coverage"]
    assert [(c.start, c.end) for c in base_uniform] == [(c.start, c.end) for c in padded_uniform]
    base_changes = [c for c in base.candidates if c.source != "uniform_coverage"]
    padded_changes = [c for c in padded.candidates if c.source != "uniform_coverage"]
    assert len(base_changes) == len(padded_changes)
    assert all(new.start <= old.start and new.end >= old.end
               for old, new in zip(base_changes, padded_changes))
    assert all(0 <= c.start < c.end <= padded.video_duration for c in padded_changes)


def test_candidate_prompt_is_compact_score_ranked_and_candidate_first(question):
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=4)
        strategy = CandidateHierarchicalRetrieval(index)
        pool = strategy.prepare(video, question)
        initial = video.uniform_frames(4)
        hint = strategy.prompt_context(pool, initial)
        prompt = default_prompt(question, video, CompactContext(), 2, 4, 2,
                               [], initial, hint, False)

    assert "search leads, not action labels or proof" in hint
    assert "visual change/motion only, not semantic relevance" in hint
    assert "freeform sample_interval only when no listed candidate fits" in hint
    assert "Change/motion ranked by score descending" in hint
    for candidate in pool.candidates:
        assert hint.count(candidate.candidate_id) == 1

    changes = sorted((c for c in pool.candidates if c.source != "uniform_coverage"),
                     key=lambda c: (-(c.score if c.score is not None else float("-inf")),
                                    c.start, c.end, c.candidate_id))
    for rank, candidate in enumerate(changes, start=1):
        assert f"{rank}. {candidate.candidate_id}" in hint
    if changes:
        assert changes[0].score >= changes[-1].score

    allowed = prompt.split("Allowed actions:", 1)[1].split("Do not return", 1)[0]
    assert allowed.index('"action":"select_candidate"') < allowed.index('"action":"sample_interval"')
    assert "fallback only when no listed candidate fits" in allowed
    assert '"action":"final_answer"' in allowed


def test_candidate_first_prompt_advises_sequence_coverage_without_semantic_scores(question):
    from egoagent_eval.core.controller import CandidateFirstHierarchicalRetrieval
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=4)
        strategy = CandidateFirstHierarchicalRetrieval(index)
        pool = strategy.prepare(video, question)
        hint = strategy.prompt_context(pool, video.uniform_frames(4))
    assert "multi-action or order questions" in hint
    assert "interval broad enough to cover the full sequence" in hint
    assert "scores only describe visual variation" in hint


def test_pool_is_bound_to_video_and_stale_candidate_is_rejected(video_path, tmp_path):
    second_path = tmp_path / "other.avi"
    create_video(second_path)
    with Video(video_path) as first_video, Video(second_path) as second_video:
        first_pool = build_candidate_pool(build_time_index(first_video, coarse_samples=8))
        second_pool = build_candidate_pool(build_time_index(second_video, coarse_samples=8))
        assert first_pool.candidate_pool_id != second_pool.candidate_pool_id
        strategy = CandidateHierarchicalRetrieval(build_time_index(second_video, coarse_samples=8))
        with pytest.raises(ValueError, match="does not match"):
            strategy.prepare(first_video, Question(video_path=str(video_path), question="What is visible?"))
        from egoagent_eval.schema import SelectCandidate
        with pytest.raises(ValueError, match="not present"):
            CandidateHierarchicalRetrieval().resolve_candidate(
                SelectCandidate(action="select_candidate", candidate_id=first_pool.candidates[0].candidate_id, count=1),
                second_pool)


def test_index_snapshot_rejects_changed_video_content(video_path):
    with Video(video_path) as video:
        stale_index = build_time_index(video, coarse_samples=8)
    with Path(video_path).open("ab") as stream:
        stream.write(b"changed-after-index")
    with Video(video_path) as changed_video:
        strategy = CandidateHierarchicalRetrieval(stale_index)
        with pytest.raises(ValueError, match="snapshot is stale"):
            strategy.prepare(changed_video, Question(video_path=str(video_path), question="What is visible?"))


def test_candidate_action_executes_python_resolved_interval_and_records_frames(question, tmp_path):
    store_dir = tmp_path / "run"
    from egoagent_eval.records import ArtifactStore
    store = ArtifactStore(store_dir, True, "hierarchical_active_v2")
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=2)
        pool = build_candidate_pool(index)
        selected = pool.candidates[0]
        timestamps = video.uniform_timestamps(2, selected.start, selected.end)
        frames = [video.read_frame(t) for t in timestamps]
        cited = frames[0].record.timestamp
        initial = video.uniform_frames(4)
        result = AgentHarness(retrieval=CandidateHierarchicalRetrieval(index)).run(
            question, video,
            FakeVLMProvider([{"action": "select_candidate", "candidate_id": selected.candidate_id, "count": 2},
                             answer(cited)]),
            method="hierarchical_active_v2", initial=initial,
            budget=Budget(max_extra_frames=2), store=store)
    assert result.terminal_state == "ANSWERED"
    assert result.candidate_pool_id == pool.candidate_pool_id
    assert result.index_scanned_frames == 8
    assert len(result.calls[0].frames) == 4
    event = result.retrieval_trace[0]
    assert event["candidate_id"] == selected.candidate_id
    assert event["selected"] and event["executed"]
    assert len(event["actual_observed_timestamps"]) == 2
    assert len(event["frame_hashes"]) == 2
    assert len(result.observed_frames) == 6
    assert result.tool_calls[0].tool == "select_candidate"
    assert result.tool_calls[0].arguments["start"] == selected.start
    assert result.tool_calls[0].arguments["end"] == selected.end
    assert len((store.path / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_candidate_missing_id_is_rejected_and_audited(question):
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8)
        result = AgentHarness(retrieval=CandidateHierarchicalRetrieval(index)).run(
            question, video,
            FakeVLMProvider([{"action": "select_candidate", "candidate_id": "from-another-pool", "count": 2}]),
            method="hierarchical_active_v2")
    assert result.terminal_state == "TOOL_ERROR"
    assert result.retrieval_trace[0]["executed"] is False
    assert result.retrieval_trace[0]["execution_error"]
    assert result.tool_calls[0].status == "rejected"
    assert len(result.observed_frames) == 4


def test_freeform_interval_fallback_is_explicitly_traced(question):
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8)
        frames = video.sample_interval(1.0, 3.0, 2)
        result = AgentHarness(retrieval=CandidateHierarchicalRetrieval(index)).run(
            question, video,
            FakeVLMProvider([{"action": "sample_interval", "start": 1.0, "end": 3.0, "count": 2},
                             answer(frames[0].record.timestamp)]),
            method="hierarchical_active_v2", budget=Budget(max_extra_frames=2))
    assert result.terminal_state == "ANSWERED"
    event = result.retrieval_trace[0]
    assert event["selection_mode"] == "freeform_interval"
    assert event["used_freeform_interval"] is True and event["used_candidate"] is False
    assert event["executed"] is True
    assert len(event["actual_observed_timestamps"]) == 2


def test_candidate_count_over_budget_never_decodes_frames(question, monkeypatch):
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8)
        candidate = build_candidate_pool(index).candidates[0]
        read_calls = []
        original_read = video.read_frame
        def counted_read(timestamp):
            read_calls.append(timestamp)
            return original_read(timestamp)
        monkeypatch.setattr(video, "read_frame", counted_read)
        result = AgentHarness(retrieval=CandidateHierarchicalRetrieval(index)).run(
            question, video,
            FakeVLMProvider([{"action": "select_candidate", "candidate_id": candidate.candidate_id, "count": 5}]),
            method="hierarchical_active_v2", budget=Budget(max_model_calls=1, max_extra_frames=4))
    assert result.terminal_state == "BUDGET_EXHAUSTED"
    assert result.retrieval_trace[0]["executed"] is False
    assert result.retrieval_trace[0]["execution_error"] == "frame_budget"
    assert len(read_calls) == 4  # initial observations only
    assert len(result.observed_frames) == 4


def test_conflicting_interval_fields_are_rejected_by_protocol(question):
    with Video(question.video_path) as video:
        result = AgentHarness(retrieval=CandidateHierarchicalRetrieval()).run(
            question, video,
            FakeVLMProvider([{"action": "select_candidate", "candidate_id": "C-any-01", "count": 2,
                              "start": 1.0, "end": 2.0}]),
            method="hierarchical_active_v2")
    assert result.terminal_state == "PROTOCOL_ERROR"
    assert result.retrieval_trace == []


def test_candidate_can_be_safely_refused_when_evidence_is_insufficient(question):
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8)
        selected = build_candidate_pool(index).candidates[0]
        result = AgentHarness(retrieval=CandidateHierarchicalRetrieval(index)).run(
            question, video,
            FakeVLMProvider([{"action": "select_candidate", "candidate_id": selected.candidate_id, "count": 2},
                             {"action": "final_answer", "answer": "", "evidence_timestamps": [],
                              "status": "refused"}]),
            method="hierarchical_active_v2", budget=Budget(max_extra_frames=2))
    assert result.terminal_state == "INSUFFICIENT_EVIDENCE"
    assert result.status == "refused"
    assert result.retrieval_trace[0]["executed"] is True


def test_hierarchical_v2_fake_cli_smoke(question, tmp_path, capsys):
    exit_code = main(["--video", question.video_path, "--question", question.question,
                      "--method", "hierarchical_active_v2", "--provider", "fake",
                      "--output", str(tmp_path / "artifacts")])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"method": "hierarchical_active_v2"' in output
    assert '"is_fake": true' in output
    assert '"status": "ok"' in output


def test_candidate_first_prompt_excludes_freeform_actions_until_candidate_observed(question):
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=2)
        pool = CandidateHierarchicalRetrieval(index).prepare(video, question)
        initial = video.uniform_frames(4)
        hint = CandidateFirstHierarchicalRetrieval(index).prompt_context(pool, initial)
        from egoagent_eval.hierarchical import _candidate_first_prompt
        first = _candidate_first_prompt(question, video, CompactContext(), 2, 4, 2,
                                        [], initial, hint, False)
        later = _candidate_first_prompt(question, video, CompactContext(), 1, 2, 1,
                                        [{"tool": "select_candidate"}], initial, hint, False)
    first_actions = first.split("Allowed actions:", 1)[1].split("Do not return", 1)[0]
    later_actions = later.split("Allowed actions:", 1)[1].split("Do not return", 1)[0]
    assert '"action":"select_candidate"' in first_actions
    assert '"action":"final_answer"' in first_actions
    assert '"action":"sample_interval"' not in first_actions
    assert '"action":"inspect_frame"' not in first_actions
    assert '"action":"sample_interval"' in later_actions
    assert "freeform fallback only when no listed candidate fits" in later_actions


def test_candidate_first_harness_executes_candidate_then_answers(question):
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=2)
        pool = build_candidate_pool(index)
        candidate = pool.candidates[0]
        times = video.uniform_timestamps(2, candidate.start, candidate.end)
        frames = [video.read_frame(t) for t in times]
        provider = FakeVLMProvider([
            {"action": "select_candidate", "candidate_id": candidate.candidate_id, "count": 2},
            {"action": "final_answer", "answer": "Candidate observed", "evidence_timestamps": [
                f.record.timestamp for f in frames], "status": "ok"},
        ])
        result = run_hierarchical_active_v3_candidate_first(
            question, video, provider, index=index, budget=Budget(max_extra_frames=2))
    assert result.method == "hierarchical_active_v3_candidate_first"
    assert result.terminal_state == "ANSWERED"
    assert result.retrieval_trace[0]["candidate_id"] == candidate.candidate_id
    assert result.retrieval_trace[0]["executed"] is True
    assert [len(call.frames) for call in provider.calls] == [4, 2]
    assert '"action":"sample_interval"' not in provider.calls[0].request["prompt"].split("Allowed actions:",1)[1].split("Do not return",1)[0]


def test_candidate_first_rejects_freeform_without_observing_extra_frames(question):
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=2)
        result = run_hierarchical_active_v3_candidate_first(
            question, video, FakeVLMProvider([
                {"action": "sample_interval", "start": 1, "end": 3, "count": 2}
            ]), index=index, budget=Budget(max_model_calls=1, max_extra_frames=4))
    assert result.terminal_state == "TOOL_ERROR"
    assert len(result.observed_frames) == 4
    assert result.retrieval_trace[0]["action_allowed"] is False
    assert result.retrieval_trace[0]["executed"] is False
    assert result.tool_calls[0].status == "rejected"
    assert result.tool_calls[0].error.code == "action_not_allowed"


def test_candidate_first_allows_controlled_fallback_after_candidate(question):
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=2)
        pool = build_candidate_pool(index)
        candidate = pool.candidates[0]
        candidate_frames = video.sample_interval(candidate.start, candidate.end, 2)
        fallback_frames = video.sample_interval(5.0, 7.0, 2)
        result = run_hierarchical_active_v3_candidate_first(
            question, video,
            FakeVLMProvider([
                {"action": "select_candidate", "candidate_id": candidate.candidate_id, "count": 2},
                {"action": "sample_interval", "start": 5.0, "end": 7.0, "count": 2},
                {"action": "final_answer", "answer": "Observed two points",
                 "evidence_timestamps": [candidate_frames[0].record.timestamp,
                                         fallback_frames[0].record.timestamp], "status": "ok"},
            ]), index=index, budget=Budget(max_extra_frames=4, max_tool_calls=2, max_model_calls=3))
    assert result.terminal_state == "ANSWERED"
    assert result.retrieval_trace[0]["used_candidate"] is True
    assert result.retrieval_trace[1]["used_freeform_interval"] is True
    assert result.retrieval_trace[1]["executed"] is True


def test_candidate_first_hybrid_sampling_is_traced_and_budgeted(question):
    from egoagent_eval.index import scan_interval_change_points
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=2)
        strategy = CandidateFirstHierarchicalRetrieval(index, "hybrid_change", 0.5, 8)
        pool = strategy.prepare(video, question)
        candidate = pool.candidates[0]
        points, _, _ = scan_interval_change_points(video, candidate.start, candidate.end, 8)
        expected = sample_interval_timestamps(candidate.start, candidate.end, 4,
                                              policy="hybrid_change",
                                              change_points=points)
        cited_timestamp = video.read_frame(expected[0]).record.timestamp
        result = run_hierarchical_active_v3_candidate_first(
            question, video,
            FakeVLMProvider([
                {"action": "select_candidate", "candidate_id": candidate.candidate_id, "count": 4},
                {"action": "final_answer", "answer": "Observed", "evidence_timestamps": [cited_timestamp], "status": "ok"},
            ]), index=index, budget=Budget(max_extra_frames=4),
            candidate_sampling_policy="hybrid_change", candidate_context_padding=0.5,
            local_index_samples=8)
    assert result.terminal_state == "ANSWERED"
    assert result.retrieval_trace[0]["sampling_policy"] == "hybrid_change"
    with Video(question.video_path) as video:
        expected_actual = [video.read_frame(t).record.timestamp for t in expected]
    assert result.retrieval_trace[0]["actual_observed_timestamps"] == expected_actual
    assert result.tool_calls[0].arguments["sampling_policy"] == "hybrid_change"
    assert len(result.observed_frames) == 8
    assert result.tool_calls[0].index_scanned_frames == 8
    assert result.index_scanned_frames == index.scanned_frames + 8


@pytest.mark.parametrize("candidate_id,count,error_code", [
    ("C-from-another-pool-01", 2, "invalid_candidate"),
    ("valid", 5, "frame_budget"),
])
def test_candidate_first_rejects_invalid_candidate_or_budget(question, candidate_id, count, error_code):
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=2)
        valid = build_candidate_pool(index).candidates[0]
        selected_id = valid.candidate_id if candidate_id == "valid" else candidate_id
        result = run_hierarchical_active_v3_candidate_first(
            question, video,
            FakeVLMProvider([{"action": "select_candidate", "candidate_id": selected_id,
                              "count": count}]), index=index,
            budget=Budget(max_model_calls=1, max_extra_frames=4))
    assert result.terminal_state in {"TOOL_ERROR", "BUDGET_EXHAUSTED"}
    assert result.retrieval_trace[0]["executed"] is False
    if error_code == "invalid_candidate":
        assert "not present" in result.retrieval_trace[0]["execution_error"]
    else:
        assert result.retrieval_trace[0]["execution_error"] == error_code
    assert len(result.observed_frames) == 4


def test_candidate_first_fake_cli_smoke(question, tmp_path, capsys):
    exit_code = main(["--video", question.video_path, "--question", question.question,
                      "--method", "hierarchical_active_v3_candidate_first", "--provider", "fake",
                      "--output", str(tmp_path / "artifacts")])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"method": "hierarchical_active_v3_candidate_first"' in output
    assert '"is_fake": true' in output


def test_staged_observation_fake_cli_smoke_and_budget(question, tmp_path, capsys):
    output_root = tmp_path / "artifacts"
    exit_code = main(["--video", question.video_path, "--question", question.question,
                      "--method", "hierarchical_active_v4_staged_observation",
                      "--provider", "fake", "--env-file", str(tmp_path / "no-env-file"),
                      "--output", str(output_root)])
    assert exit_code == 0
    cli_record = json.loads(capsys.readouterr().out)
    assert cli_record["method"] == "hierarchical_active_v4_staged_observation"
    assert cli_record["is_fake"] is True
    assert cli_record["status"] == "ok"

    artifact_dir = Path(cli_record["artifact_dir"])
    result = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    calls = [json.loads(line) for line in (artifact_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["method"] == "hierarchical_active_v4_staged_observation"
    assert len(result["calls"]) == len(calls) == 3
    assert len(result["tool_calls"]) == 2
    assert [event["stage"] for event in result["retrieval_trace"]] == [1, 2]
    assert [len(event["actual_observed_timestamps"]) for event in result["retrieval_trace"]] == [2, 2]
    assert sum(len(tool["frames"]) for tool in result["tool_calls"]) <= 4
    assert "Prior actions" in calls[1]["request"]["prompt"]
    assert all(f"{timestamp:g}s/" in calls[1]["request"]["prompt"]
               for timestamp in result["retrieval_trace"][0]["actual_observed_timestamps"])
    assert result["status"] == "ok"


def test_staged_observation_temporal_policy_fake_cli_opt_in(question, tmp_path, capsys):
    output_root = tmp_path / "artifacts"
    exit_code = main(["--video", question.video_path, "--question", question.question,
                      "--method", "hierarchical_active_v4_staged_observation",
                      "--provider", "fake", "--env-file", str(tmp_path / "no-env-file"),
                      "--v4-candidate-selection-policy", "temporal_complementarity",
                      "--output", str(output_root)])
    assert exit_code == 0
    cli_record = json.loads(capsys.readouterr().out)
    artifact_dir = Path(cli_record["artifact_dir"])
    result = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    calls = [json.loads(line) for line in (artifact_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    environment = json.loads((artifact_dir / "environment.json").read_text(encoding="utf-8"))

    assert cli_record["is_fake"] is True
    assert cli_record["status"] == "ok"
    assert environment["v4_candidate_selection_policy"] == "temporal_complementarity"
    prompts = [call["request"]["prompt"] for call in calls]
    assert all("Candidate selection policy: temporal_complementarity" in prompt for prompt in prompts)
    assert all("state=" in prompt and "positions=[" in prompt for prompt in prompts)
    assert len(result["calls"]) == 3
    assert len(result["tool_calls"]) == 2
    assert sum(len(tool["frames"]) for tool in result["tool_calls"]) == 4
    assert all(event["action_allowed"] and event["executed"] for event in result["retrieval_trace"])
    for event, call, tool in zip(result["retrieval_trace"], calls, result["tool_calls"]):
        action = json.loads(call["content"])
        assert action["candidate_id"] == event["candidate_id"]
        assert action["candidate_id"] in prompts[call["index"] - 1]
        snapshot = tool["arguments"]["candidate_observation"]
        assert snapshot["candidate_id"] == event["candidate_id"]
        assert snapshot["state"] in {"unseen", "partially_observed"}
        assert snapshot["remaining_sampling_positions"]


def test_staged_observation_ranked_policy_fake_cli_opt_in(question, tmp_path, capsys):
    output_root = tmp_path / "artifacts"
    exit_code = main(["--video", question.video_path, "--question", question.question,
                      "--method", "hierarchical_active_v4_staged_observation",
                      "--provider", "fake", "--env-file", str(tmp_path / "no-env-file"),
                      "--v4-candidate-selection-policy", "temporal_complementarity_ranked",
                      "--output", str(output_root)])
    assert exit_code == 0
    cli_record = json.loads(capsys.readouterr().out)
    artifact_dir = Path(cli_record["artifact_dir"])
    result = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    environment = json.loads((artifact_dir / "environment.json").read_text(encoding="utf-8"))
    calls = [json.loads(line) for line in
             (artifact_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]

    assert environment["v4_candidate_selection_policy"] == "temporal_complementarity_ranked"
    assert len([event for event in result["retrieval_trace"]
                if event.get("event") == "candidate_ranking"]) == len(calls)
    assert all(event["selection_mode"] == "model_choice_after_python_ranking"
               for event in result["retrieval_trace"] if event.get("event") == "candidate_ranking")
    assert all("Python-ranked candidates:" in call["request"]["prompt"] for call in calls)
    selected_events = [event for event in result["retrieval_trace"]
                       if event.get("decision_source") == "model"]
    assert selected_events
    assert all(event["selection_policy"] == "temporal_complementarity_ranked"
               for event in selected_events)
