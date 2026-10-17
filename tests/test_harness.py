import json

from egoagent_eval.agent import run_active
from egoagent_eval.adaptive import run_adaptive_active
from egoagent_eval.records import ArtifactStore
from egoagent_eval.schema import Budget
from egoagent_eval.video import Video
from egoagent_eval.vlm import FakeVLMProvider


def _final(timestamp=1.0):
    return {"action": "final_answer", "answer": "observed answer",
            "evidence_timestamps": [timestamp], "status": "ok"}


def test_harness_records_state_events_and_termination(question, tmp_path):
    store = ArtifactStore(tmp_path, True, "state-events")
    provider = FakeVLMProvider([_final()], store=store)
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, provider, store=store)
    assert result.termination_reason == "answered"
    assert result.state_events[0].state_to == "OBSERVE"
    assert result.state_events[-1].state_to == "FINALIZE"
    state_lines = (store.path / "state_events.jsonl").read_text().splitlines()
    assert len(state_lines) == len(result.state_events)
    assert json.loads(state_lines[-1])["reason"] == "final answer verified"


def test_harness_budget_gate_does_not_execute_video_tool(question, monkeypatch):
    provider = FakeVLMProvider([
        {"action": "sample_interval", "start": 0.0, "end": 4.0, "count": 4},
        {"action": "sample_interval", "start": 0.0, "end": 4.0, "count": 4},
        _final(0.5),
    ])
    with Video(question.video_path) as video:
        reads = []
        original = video.read_frame
        def tracked(timestamp):
            reads.append(timestamp)
            return original(timestamp)
        monkeypatch.setattr(video, "read_frame", tracked)
        result = run_adaptive_active(question, video, provider)
    assert result.status == "ok"
    assert result.termination_reason == "answered"
    assert result.tool_calls[1].status == "rejected"
    assert result.tool_calls[1].error.code == "frame_budget"
    assert result.tool_calls[1].frames == []
    # Four initial frames plus four frames from the first accepted tool call;
    # the second, budget-rejected action reads no additional frame.
    assert len(reads) == 8
    assert any(e.code == "budget_exhausted_action" for e in result.errors)


def test_harness_distinguishes_protocol_error_and_refusal(question):
    with Video(question.video_path) as video:
        protocol = run_active(question, video, FakeVLMProvider(["not JSON"]))
    with Video(question.video_path) as video:
        refusal = run_active(question, video, FakeVLMProvider([
            {"action": "final_answer", "status": "refused", "answer": "", "evidence_timestamps": []}
        ]))
    assert protocol.termination_reason == "protocol_error"
    assert refusal.termination_reason == "provider_refusal"
    assert protocol.status == "failed" and refusal.status == "refused"


def test_harness_rejects_unobserved_timestamp(question):
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, FakeVLMProvider([_final(0.12345)]))
    assert result.termination_reason == "protocol_error"
    assert result.errors[-1].code == "invalid_final_answer"


def test_harness_respects_model_budget_terminal_reason(question):
    with Video(question.video_path) as video:
        result = run_active(question, video, FakeVLMProvider([
            {"action": "inspect_frame", "timestamp": 2.0}
        ]), budget=Budget(max_model_calls=1))
    assert result.termination_reason == "budget_exhausted"
    assert result.errors[-1].code == "model_call_budget"
