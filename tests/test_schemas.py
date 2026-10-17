import pytest

from egoagent_eval import AgentResult, EvalRecord, Task, ToolCall


def test_task_normalizes_video_path_and_keeps_event_range():
    task = Task("clip.mp4", "What happened?", "opened a drawer", (1.2, 3.4))
    assert str(task.video_path) == "clip.mp4"
    assert task.event_time_range == (1.2, 3.4)


def test_task_rejects_invalid_event_range_and_blank_question():
    with pytest.raises(ValueError):
        Task("clip.mp4", "Q", "A", (4, 2))
    with pytest.raises(ValueError):
        Task("clip.mp4", "  ", "A", (0, 1))


def test_agent_result_stores_evidence_and_tool_trace():
    result = AgentResult("I picked up a cup", (2.5,), (ToolCall("inspect_frame", {"time": 2.5}, "cup visible"),))
    assert result.evidence_timestamps == (2.5,)
    assert result.tool_calls[0].tool == "inspect_frame"


def test_eval_record_rejects_negative_usage():
    with pytest.raises(ValueError):
        EvalRecord("task-1", "answer", input_tokens=-1)
