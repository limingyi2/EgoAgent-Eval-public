import base64
import json

import httpx
import pytest

from egoagent_eval.direct import run_direct
from egoagent_eval.records import ArtifactStore
from egoagent_eval.schema import ACTION_ADAPTER, FinalAction, InspectFrame, SampleInterval
from egoagent_eval.video import Video
from egoagent_eval.vlm import MiMoVLMProvider, ProviderFailure, parse_json_object


def envelope(content=None, finish_reason="stop", **message_fields):
    return {"model": "mimo-v2.6-flash", "choices": [{"finish_reason": finish_reason,
            "message": {"content": content or '{"answer":"square","evidence_timestamps":[1.0],"status":"ok"}', **message_fields}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 10}}}


@pytest.mark.parametrize("count", [1, 4, 8])
def test_adapter_sends_actual_base64_images_and_json_mode(video_path, tmp_path, count):
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer dummy-secret"
        return httpx.Response(200, json=envelope())
    store = ArtifactStore(tmp_path, True, "transport")
    provider = MiMoVLMProvider(api_key="dummy-secret", transport=httpx.MockTransport(handler), store=store,
                              input_price=1.0, output_price=2.0, cached_price=0.1)
    with Video(video_path) as video:
        frames = video.uniform_frames(count)
        record = provider.complete("Return only JSON.", frames)
    request = captured[0]
    blocks = [b for b in request["messages"][0]["content"] if b["type"] == "image_url"]
    assert len(blocks) == count
    for block, frame in zip(blocks, frames):
        assert base64.b64decode(block["image_url"]["url"].split(",", 1)[1]) == frame.jpeg
    assert request["model"] == "mimo-v2.6-flash" and request["response_format"] == {"type": "json_object"}
    assert request["thinking"] == {"type": "disabled"}
    assert record.is_fake and provider.is_fake
    assert record.usage.input_tokens == 100 and record.usage.output_tokens == 20
    assert record.usage.cost_usd == pytest.approx(0.000131)
    logged = (store.path / "calls.jsonl").read_text()
    assert "dummy-secret" not in logged and "base64," not in logged
    assert json.loads(logged)["raw_response"] is not None


def test_sampling_parameters_are_explicit_and_recorded(video_path):
    captured = []
    provider = MiMoVLMProvider(api_key="dummy", temperature=0.25, max_completion_tokens=768,
        transport=httpx.MockTransport(lambda request: (captured.append(json.loads(request.content)) or
                                                        httpx.Response(200, json=envelope()))))
    with Video(video_path) as video:
        provider.complete("JSON", video.uniform_frames(1))
    assert captured[0]["temperature"] == 0.25
    assert captured[0]["max_completion_tokens"] == 768


@pytest.mark.parametrize("kwargs", [
    {"temperature": -0.1}, {"temperature": 2.1}, {"max_completion_tokens": 0},
    {"max_completion_tokens": True},
])
def test_sampling_parameter_bounds_are_validated(kwargs):
    with pytest.raises(ValueError):
        MiMoVLMProvider(**kwargs)


@pytest.mark.parametrize("payload,expected", [
    (envelope("not JSON"), "protocol_error"),
    (envelope(finish_reason="length"), "incomplete_response"),
    (envelope(refusal="refused"), "refused"),
    (envelope(finish_reason="content_filter"), "refused"),
    ({"choices": []}, "invalid_response"),
    ({**envelope(), "model": "text-only-model"}, "model_mismatch"),
    ({**envelope(), "usage": {"prompt_tokens": -1}}, "invalid_response"),
])
def test_api_response_failures_are_recorded(question, payload, expected):
    provider = MiMoVLMProvider(api_key="test", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    with Video(question.video_path) as video:
        result = run_direct(question, video, provider)
    assert result.status == ("refused" if expected == "refused" else "failed")
    assert result.errors[0].code == expected
    assert result.calls[0].raw_response is not None and result.correct is None and result.is_fake


@pytest.mark.parametrize("case,expected", [("timeout", "timeout"), ("http", "http_error"), ("network", "network_error")])
def test_transport_errors_are_preserved(video_path, case, expected):
    def handler(request):
        if case == "timeout":
            raise httpx.ReadTimeout("test timeout", request=request)
        if case == "network":
            raise httpx.ConnectError("test connect failure", request=request)
        return httpx.Response(429, text="rate limited")
    provider = MiMoVLMProvider(api_key="test", transport=httpx.MockTransport(handler))
    with Video(video_path) as video:
        with pytest.raises(ProviderFailure):
            provider.complete("JSON", video.uniform_frames(1))
    record = provider.calls[0]
    assert record.error.code == expected and record.latency_seconds >= 0
    if case == "http":
        assert record.http_status == 429 and record.raw_response == "rate limited"
    assert len(provider.calls) == 1  # No automatic retry can change the experiment budget.


def test_missing_key_is_preflight_failure_not_api_success(video_path, monkeypatch):
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    provider = MiMoVLMProvider()
    with Video(video_path) as video:
        with pytest.raises(ProviderFailure):
            provider.complete("JSON", video.uniform_frames(1))
    assert provider.calls[0].error.code == "missing_credentials"
    assert provider.calls[0].http_status is None and provider.calls[0].raw_response is None
    assert provider.calls[0].network_attempted is False


def test_unknown_cost_and_tokens_stay_null(video_path):
    payload = envelope()
    del payload["usage"]
    provider = MiMoVLMProvider(api_key="test", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    with Video(video_path) as video:
        result = provider.complete("JSON", video.uniform_frames(1))
    assert result.usage.input_tokens is None and result.usage.cost_usd is None


def test_key_echo_is_redacted_from_raw_records(video_path, tmp_path):
    store = ArtifactStore(tmp_path, True, "redaction")
    payload = envelope('{"answer":"dummy-secret","evidence_timestamps":[1.0],"status":"ok"}')
    provider = MiMoVLMProvider(api_key="dummy-secret", store=store,
                              transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    with Video(video_path) as video:
        provider.complete("Return JSON dummy-secret", video.uniform_frames(1))
    assert "dummy-secret" not in (store.path / "calls.jsonl").read_text()


@pytest.mark.parametrize("value", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '[]'])
def test_json_parser_rejects_non_json_or_ambiguous_output(value):
    with pytest.raises(ValueError):
        parse_json_object(value)


@pytest.mark.parametrize("value,expected", [
    ('{"action":"inspect_frame","timestamp":12.5}', InspectFrame),
    ('```json\n{"action":"inspect_frame","timestamp":12.5}\n```', InspectFrame),
    ('{"output":{"action":"sample_interval","start":10,"end":20}}', SampleInterval),
    ('{"tool_calls":[{"function":{"name":"final_answer","arguments":"{\\"answer\\":\\"cup\\",\\"evidence_timestamps\\":[12.5]}"}}]}', FinalAction),
])
def test_active_action_parser_accepts_mimo_wrappers(value, expected):
    action = ACTION_ADAPTER.validate_python(parse_json_object(value))
    assert isinstance(action, expected)
    if isinstance(action, SampleInterval):
        assert action.count == 4
    if isinstance(action, FinalAction):
        assert action.status == "ok"


def test_active_action_parser_rejects_schema_document_explicitly():
    schema_response = '{"$defs":{"FinalAction":{"properties":{"action":{"const":"final_answer"}}}},"oneOf":[]}'
    with pytest.raises(ValueError, match="JSON schema"):
        parse_json_object(schema_response)


def test_first_release_rejects_other_or_text_only_models():
    with pytest.raises(ValueError):
        MiMoVLMProvider(model="qwen-text")


def test_mock_transport_cannot_write_real_records(tmp_path):
    store = ArtifactStore(tmp_path, False, "real")
    with pytest.raises(ValueError):
        MiMoVLMProvider(api_key="test", store=store, transport=httpx.MockTransport(lambda _: httpx.Response(200)))
