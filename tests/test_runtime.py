import json
from pathlib import Path

import pytest

from egoagent_eval.agent import run_active
from egoagent_eval.cli import main
from egoagent_eval.direct import run_direct
from egoagent_eval.records import ArtifactStore
from egoagent_eval.schema import Budget, Question
from egoagent_eval.tools import ObservationTools
from egoagent_eval.video import Video
from egoagent_eval.vlm import FakeVLMProvider


def final(timestamp=1.0, action=False, status="ok"):
    value = {"answer": "fake answer", "evidence_timestamps": [timestamp], "status": status}
    if action:
        value["action"] = "final_answer"
    return value


def test_direct_active_initial_evidence_is_identical(question):
    with Video(question.video_path) as video:
        direct = run_direct(question, video, FakeVLMProvider([final()]))
    with Video(question.video_path) as video:
        active = run_active(question, video, FakeVLMProvider([final(action=True)]))
    assert direct.status == active.status == "ok"
    assert [(f.timestamp, f.sha256) for f in direct.initial_frames] == [(f.timestamp, f.sha256) for f in active.initial_frames]
    assert len(direct.initial_frames) == 4
    assert direct.is_fake and active.is_fake and direct.correct is None and active.correct is None


def test_uniform8_has_exactly_eight_observations(question):
    with Video(question.video_path) as video:
        result = run_direct(question, video, FakeVLMProvider([final(0.5)]), "uniform8")
    assert result.status == "ok"
    assert len(result.observed_frames) == 8
    assert [f.timestamp for f in result.observed_frames] == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    assert len(result.calls) == 1


def test_active_two_tools_four_extra_frames_and_final(question, tmp_path):
    store = ArtifactStore(tmp_path, True, "active-test")
    provider = FakeVLMProvider([
        {"action": "inspect_frame", "timestamp": 2.0},
        {"action": "sample_interval", "start": 4.0, "end": 7.0, "count": 3},
        final(6.5, action=True)], store=store)
    with Video(question.video_path) as video:
        result = run_active(question, video, provider, store=store)
    assert result.status == "ok"
    assert len(result.calls) == 3 and len(result.tool_calls) == 2 and len(result.observed_frames) == 8
    assert [len(c.frames) for c in result.calls] == [4, 5, 8]
    assert result.tool_calls[1].arguments == {"start": 4.0, "end": 7.0, "count": 3}
    assert [f.timestamp for f in result.tool_calls[1].frames] == [4.5, 5.5, 6.5]
    for record in result.tool_calls:
        assert record.latency_seconds >= 0 and record.error is None
    assert len((store.path / "tools.jsonl").read_text().splitlines()) == 2
    assert len((store.path / "calls.jsonl").read_text().splitlines()) == 3
    assert all((store.path / f.file).is_file() for f in result.observed_frames)
    assert "standard_answer" not in json.dumps([c.request for c in result.calls])
    assert result.usage.input_tokens is None and result.usage.cost_usd is None


def test_frame_budget_rejected_before_reading(question, monkeypatch):
    with Video(question.video_path) as video:
        initial = video.uniform_frames(4)
        reads = []
        monkeypatch.setattr(video, "read_frame", lambda t: reads.append(t))
        result = run_active(question, video, FakeVLMProvider([
            {"action": "sample_interval", "start": 0.0, "end": 8.0, "count": 5}]), initial)
    assert result.status == "failed" and reads == []
    assert result.tool_calls[0].status == "rejected"
    assert result.tool_calls[0].error.code == "frame_budget"
    assert len(result.observed_frames) == 4 and len(result.calls) == 1


def test_cumulative_frame_budget(question):
    provider = FakeVLMProvider([
        {"action": "sample_interval", "start": 0.0, "end": 3.0, "count": 3},
        {"action": "sample_interval", "start": 4.0, "end": 8.0, "count": 2}])
    with Video(question.video_path) as video:
        result = run_active(question, video, provider)
    assert result.status == "failed" and result.tool_calls[1].error.code == "frame_budget"
    assert len(result.observed_frames) == 7


def test_third_tool_call_is_rejected_and_recorded(question):
    provider = FakeVLMProvider([{"action": "inspect_frame", "timestamp": t} for t in (2.0, 4.0, 6.0)])
    with Video(question.video_path) as video:
        result = run_active(question, video, provider)
    assert result.status == "failed"
    assert len(result.calls) == 3 and len(result.tool_calls) == 3
    assert result.tool_calls[-1].status == "rejected" and result.tool_calls[-1].error.code == "tool_call_budget"
    assert len(result.observed_frames) == 6


def test_invalid_tool_timestamp_is_recorded(question, tmp_path):
    store = ArtifactStore(tmp_path, True, "bad-tool")
    with Video(question.video_path) as video:
        tools = ObservationTools(video, store=store)
        output = tools.inspect_frame(8.0)
        assert output.record.requested_timestamps == [8.0]
        assert output.record.status == "failed" and output.record.error.code == "video_tool_error"
        tools.inspect_frame(2.0)
        third = tools.inspect_frame(3.0)
        assert third.record.status == "rejected"
    assert len((store.path / "tools.jsonl").read_text().splitlines()) == 3


def test_decode_failure_reserves_frames_and_preserves_partial_trace(question, monkeypatch):
    with Video(question.video_path) as video:
        tools = ObservationTools(video)
        original = video.read_frame
        def fail_second(t):
            if t > 2.0:
                raise ValueError("injected decode failure")
            return original(t)
        monkeypatch.setattr(video, "read_frame", fail_second)
        output = tools.sample_interval(0.0, 8.0, 4)
        assert output.record.status == "failed" and len(output.frames) == 1
        assert tools.reserved_frames == 4
        assert tools.inspect_frame(1.0).record.error.code == "frame_budget"


@pytest.mark.parametrize("response", ["not JSON", '```json\n{}\n```', {}, {"answer": "x", "status": "ok"},
                                      {"answer": "x", "evidence_timestamps": [2.0], "status": "ok"},
                                      {"answer": "", "evidence_timestamps": [], "status": "ok"},
                                      {"answer": "x", "evidence_timestamps": [1.0], "status": "ok", "correct": True}])
def test_invalid_final_answer_cannot_be_success(question, response):
    with Video(question.video_path) as video:
        result = run_direct(question, video, FakeVLMProvider([response]))
    assert result.status == "failed" and result.correct is None and result.errors
    assert result.calls[0].content is not None


@pytest.mark.parametrize("response", [{"action": "unknown"}, {"action": "inspect_frame", "timestamp": "1"},
                                      {"action": "sample_interval", "start": 0.0, "end": 8.0, "count": True},
                                      {"action": "final_answer", **final(2.0)}])
def test_invalid_action_fails_without_tool_execution(question, response):
    with Video(question.video_path) as video:
        result = run_active(question, video, FakeVLMProvider([response]))
    assert result.status == "failed" and result.tool_calls == [] and result.errors


def test_refusal_is_failure_for_scoring(question):
    with Video(question.video_path) as video:
        result = run_direct(question, video, FakeVLMProvider([final(status="refused")]))
    assert result.status == "refused" and result.correct is None and result.errors[0].code == "refused"


def test_smaller_model_call_budget_stops_without_final(question):
    with Video(question.video_path) as video:
        result = run_active(question, video, FakeVLMProvider([{"action": "inspect_frame", "timestamp": 2.0}]),
                            budget=Budget(max_model_calls=1))
    assert result.status == "failed" and len(result.calls) == 1 and result.errors[-1].code == "model_call_budget"


def test_runtime_question_forbids_ground_truth_fields(question):
    with pytest.raises(ValueError):
        Question.model_validate({**question.model_dump(), "standard_answer": "leak"})
    with pytest.raises(ValueError):
        Budget(max_extra_frames=5)


def test_cli_all_preserves_results_and_same_initial_frames(question, tmp_path, capsys):
    assert main(["--video", question.video_path, "--question", question.question, "--provider", "fake",
                 "--method", "all", "--output", str(tmp_path)]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    results = {line["method"]: json.loads((Path(line["artifact_dir"]) / "result.json").read_text()) for line in lines}
    for result in results.values():
        assert result["is_fake"] is True and result["correct"] is None and result["status"] == "ok"
    assert results["direct4"]["initial_frames"] == results["active"]["initial_frames"]
    assert len(results["active"]["observed_frames"]) == 8


def test_cli_missing_video_records_failure(tmp_path, capsys):
    assert main(["--video", str(tmp_path / "missing.avi"), "--question", "Q", "--provider", "fake",
                 "--method", "direct4", "--output", str(tmp_path)]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "failed" and output["errors"][0]["code"] == "setup_error"
