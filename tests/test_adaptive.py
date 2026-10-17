import json

import pytest

from egoagent_eval.adaptive import run_adaptive_active
from egoagent_eval.schema import Budget, Question
from egoagent_eval.video import Video
from egoagent_eval.vlm import FakeVLMProvider


def final(timestamps, answer="fake adaptive answer"):
    return {"action": "final_answer", "answer": answer,
            "evidence_timestamps": timestamps, "status": "ok"}


@pytest.mark.parametrize(
    "policy,expected_second_images,expected_reused",
    [("full_context", 8, 4), ("selective_context", 6, 2), ("compact_context", 4, 0)],
)
def test_adaptive_context_policies_bound_repeated_visual_input(
    question, policy, expected_second_images, expected_reused
):
    provider = FakeVLMProvider([
        {"action": "sample_interval", "start": 0.0, "end": 4.0, "count": 4},
        final([0.5, 1.5]),
    ])
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, provider, context_policy=policy)
    assert result.status == "ok"
    assert len(result.calls) == 2
    assert [len(call.frames) for call in result.calls] == [4, expected_second_images]
    assert result.calls[1].image_count == expected_second_images
    assert result.calls[1].unique_image_count == expected_second_images
    assert result.calls[1].reused_image_count == expected_reused
    assert result.calls[1].evidence_coverage["all_cited_in_request"] is True
    expected_total_images = 4 + expected_second_images
    assert result.total_image_count == expected_total_images
    assert result.run_unique_image_count == 8
    assert result.run_reused_image_count == expected_total_images - 8


def test_adaptive_can_finish_on_first_request(question):
    provider = FakeVLMProvider([final([1.0])])
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, provider)
    assert result.status == "ok"
    assert len(result.calls) == 1
    assert result.tool_calls == []


def test_adaptive_rejects_weak_sequence_evidence_then_observes(question):
    sequence_question = Question(video_path=question.video_path,
                                 question="In what order does the wearer move the object, then place it?")
    provider = FakeVLMProvider([
        final([1.0]),
        {"action": "sample_interval", "start": 0.0, "end": 4.0, "count": 4},
        final([0.5, 1.5]),
    ])
    with Video(question.video_path) as video:
        result = run_adaptive_active(sequence_question, video, provider)
    assert result.status == "ok"
    assert len(result.calls) == 3
    assert result.tool_calls[0].tool == "sample_interval"
    assert not any(error.code == "insufficient_evidence" for error in result.errors)


def test_adaptive_cannot_cite_unobserved_frame(question):
    provider = FakeVLMProvider([final([0.5])])
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, provider)
    assert result.status == "failed"
    assert result.errors[-1].code == "invalid_final_answer"


def test_adaptive_trace_has_context_policy_and_no_ground_truth(question):
    provider = FakeVLMProvider([final([1.0])])
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, provider, context_policy="compact_context")
    prompt = result.calls[0].request["prompt"]
    assert "Context policy=compact_context" in prompt
    assert "standard_answer" not in json.dumps([call.request for call in result.calls])
    assert result.calls[0].request_index == 1
    assert result.calls[0].stop_reason == "stop"


def test_adaptive_exhausted_frame_budget_forces_final_only_response(question):
    provider = FakeVLMProvider([
        {"action": "sample_interval", "start": 0.0, "end": 4.0, "count": 4},
        {"action": "sample_interval", "start": 0.0, "end": 4.0, "count": 4},
        final([0.5, 1.5], "answer after budget rejection"),
    ])
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, provider)
    assert result.status == "ok"
    assert len(result.calls) == 3
    assert len(result.tool_calls) == 2
    assert result.tool_calls[1].status == "rejected"
    assert result.tool_calls[1].error.code == "frame_budget"
    assert any(error.code == "budget_exhausted_action" for error in result.errors)
    assert "Observation budget is exhausted" in result.calls[2].request["prompt"]


def test_adaptive_exhausted_budget_returns_explicit_insufficient_evidence(question):
    provider = FakeVLMProvider([
        {"action": "sample_interval", "start": 0.0, "end": 4.0, "count": 4},
        {"action": "sample_interval", "start": 0.0, "end": 4.0, "count": 4},
        {"action": "sample_interval", "start": 0.0, "end": 4.0, "count": 4},
    ])
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, provider)
    assert result.status == "refused"
    assert result.answer == ""
    assert any(error.code == "insufficient_evidence" for error in result.errors)
    assert len(result.calls) == 3


def test_adaptive_accepts_explicit_answered_status_alias(question):
    provider = FakeVLMProvider([{
        "action": "final_answer", "answer": "answered alias", "evidence_timestamps": [1.0],
        "status": "answered",
    }])
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, provider)
    assert result.status == "ok"
    assert result.answer == "answered alias"


def test_adaptive_accepts_refusal_envelope_without_action(question):
    provider = FakeVLMProvider([{
        "answer": "insufficient visual evidence", "evidence_timestamps": ["20s"],
        "status": "refused",
    }])
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, provider)
    assert result.status == "refused"
    assert result.answer == ""
    assert result.evidence_timestamps == []


def test_adaptive_accepts_answered_envelope_without_action(question):
    provider = FakeVLMProvider([{
        "answer": "answered envelope", "evidence_timestamps": [1.0],
        "status": "answered",
    }])
    with Video(question.video_path) as video:
        result = run_adaptive_active(question, video, provider)
    assert result.status == "ok"
    assert result.answer == "answered envelope"
