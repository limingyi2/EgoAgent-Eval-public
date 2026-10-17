import json

from egoagent_eval.hierarchical import run_hierarchical_active
from egoagent_eval.index import build_time_index
from egoagent_eval.video import Video
from egoagent_eval.vlm import FakeVLMProvider


def test_hierarchical_active_uses_index_without_counting_scan_as_observation(question):
    provider = FakeVLMProvider([
        {"action": "sample_interval", "start": 0.0, "end": 4.0, "count": 4},
        {"action": "final_answer", "answer": "shapes", "evidence_timestamps": [0.5], "status": "ok"},
    ])
    with Video(question.video_path) as video:
        index = build_time_index(video, coarse_samples=8)
        result = run_hierarchical_active(question, video, provider, index=index)
    assert result.status == "ok"
    assert result.method == "hierarchical_active"
    assert result.index_scanned_frames == 8
    assert result.index_candidate_windows is not None
    assert len(result.observed_frames) == 8
    assert [len(call.frames) for call in result.calls] == [4, 4]
    prompt = result.calls[0].request["prompt"]
    assert "Uniform coverage windows" in prompt
    assert "visual change is not proof" in prompt
    assert "standard_answer" not in json.dumps([call.request for call in result.calls])


def test_hierarchical_rejects_unobserved_evidence(question):
    provider = FakeVLMProvider([
        {"action": "final_answer", "answer": "guess", "evidence_timestamps": [0.12345], "status": "ok"},
    ])
    with Video(question.video_path) as video:
        result = run_hierarchical_active(question, video, provider)
    assert result.status == "failed"
    assert result.errors[-1].code == "invalid_final_answer"
