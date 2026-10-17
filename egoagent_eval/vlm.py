"""MiMo image+JSON API with raw records, and a clearly labeled test provider."""

import base64
import json
import math
import re
import os
from time import perf_counter
from typing import Any, Protocol, Sequence

import httpx

from .records import ArtifactStore, utc_now
from .protocol import parse_json_object
from .schema import CallRecord, ErrorRecord, Usage
from .video import Observation
from .experiment_ledger import BudgetExceeded, ExperimentLedger, LedgerError


class ProviderFailure(RuntimeError):
    def __init__(self, record: CallRecord):
        self.record = record
        super().__init__(record.error.message if record.error else "VLM call failed")


class VLMProvider(Protocol):
    name: str
    model: str
    is_fake: bool
    calls: list[CallRecord]

    def complete(self, prompt: str, frames: Sequence[Observation],
                 tools: Sequence[dict[str, Any]] | None = None) -> CallRecord: ...


class MiMoVLMProvider:
    name = "mimo"
    is_fake = False

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: float | None = None,
                 store: ArtifactStore | None = None, transport: httpx.BaseTransport | None = None,
                 input_price: float | None = None, output_price: float | None = None,
                 cached_price: float | None = None, temperature: float = 0.0,
                 max_completion_tokens: int = 2048,
                 request_gate: ExperimentLedger | None = None,
                 run_id: str | None = None):
        self.api_key = api_key or os.environ.get("MIMO_API_KEY", "")
        self.base_url = (base_url or os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")).rstrip("/")
        self.model = (model or os.environ.get("MIMO_MODEL", "mimo-v2.6-flash")).lower()
        self.timeout = float(timeout if timeout is not None else os.environ.get("MIMO_TIMEOUT", "60"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("Temperature must be finite and in [0, 2]")
        if type(max_completion_tokens) is not int or max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be a positive integer")
        self.temperature = float(temperature)
        self.max_completion_tokens = max_completion_tokens
        if self.model != "mimo-v2.6-flash":
            raise ValueError("The first version supports only MiMo-V2.6-Flash")
        self.store = store
        self.transport = transport
        self.request_gate = request_gate
        self.run_id = run_id
        # Only httpx.MockTransport is a closed in-memory test path. Other
        # injected transports must pass the real request gate as well.
        self.is_fake = isinstance(transport, httpx.MockTransport)
        self.name = "mimo-transport-test" if self.is_fake else "mimo"
        if store and store.path.parent.name != ("fake" if self.is_fake else "real"):
            raise ValueError("Transport tests must use fake artifact storage")
        self.prices = (input_price, output_price, cached_price)
        for price in self.prices:
            if price is not None and (not math.isfinite(price) or price < 0):
                raise ValueError("Prices must be finite and nonnegative")
        self.calls: list[CallRecord] = []
        self._sent_frame_hashes: set[str] = set()

    def _redact(self, text: str) -> str:
        return text.replace(self.api_key, "[REDACTED]") if self.api_key else text

    def _usage(self, data: dict) -> Usage:
        usage = data.get("usage") or {}
        values = {}
        for target, source in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
            value = usage.get(source)
            if value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError("Invalid usage counters")
                values[target] = value
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is not None:
            if type(cached) is not int or cached < 0 or (values.get("input_tokens") is not None and cached > values["input_tokens"]):
                raise ValueError("Invalid cached token counter")
            values["cached_tokens"] = cached
        in_price, out_price, cache_price = self.prices
        if in_price is not None and out_price is not None and "input_tokens" in values and "output_tokens" in values:
            if cached == 0:
                values["cost_usd"] = (values["input_tokens"] * in_price + values["output_tokens"] * out_price) / 1e6
            elif cached is not None and cache_price is not None:
                values["cost_usd"] = ((values["input_tokens"] - cached) * in_price + cached * cache_price + values["output_tokens"] * out_price) / 1e6
        return Usage(**values)

    def complete(self, prompt: str, frames: Sequence[Observation],
                 tools: Sequence[dict[str, Any]] | None = None) -> CallRecord:
        started = perf_counter()
        started_at = utc_now()
        content = [{"type": "text", "text": prompt}]
        for frame in frames:
            if self.store:
                self.store.save_frame(frame)
            content.extend([
                {"type": "text", "text": f"Observed frame timestamp={frame.record.timestamp!r} seconds"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode("ascii")}},
            ])
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "max_completion_tokens": self.max_completion_tokens, "temperature": self.temperature, "stream": False,
                   "thinking": {"type": "disabled"}}
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        else:
            payload["response_format"] = {"type": "json_object"}
        logged_content = []
        for block in content:
            if block["type"] == "image_url":
                logged_content.append({"type": "image_url", "image_url": {"url": "[LOCAL_JPEG: see frame manifest]"}})
            else:
                logged_content.append({**block, "text": self._redact(block["text"])})
        frame_hashes = [frame.record.sha256 for frame in frames]
        unique_hashes = set(frame_hashes)
        reused_count = sum(frame_hash in self._sent_frame_hashes for frame_hash in frame_hashes)
        record = CallRecord(index=len(self.calls) + 1, request_index=len(self.calls) + 1,
                            started_at_utc=started_at, provider=self.name, model=self.model,
                            is_fake=self.is_fake, latency_seconds=0.0, frames=[f.record.model_copy() for f in frames],
                            image_count=len(frames), unique_image_count=len(unique_hashes),
                            reused_image_count=reused_count,
                            request={**payload, "messages": [{"role": "user", "content": logged_content}]})
        attempt_id = None
        response_received = False
        try:
            if not 1 <= len(frames) <= 8:
                raise ValueError("VLM requests require 1..8 images")
            if any(max(f.record.width, f.record.height) > 1024 or len(f.jpeg) > 2_000_000 for f in frames):
                raise ValueError("Image exceeds adapter size limit")
            if not self.api_key:
                record.error = ErrorRecord(code="missing_credentials", message="Set MIMO_API_KEY in the environment or local .env")
            else:
                if not self.is_fake:
                    if self.request_gate is None or not self.run_id:
                        raise LedgerError("A registered experiment request gate is required before real API calls")
                    attempt_id = self.request_gate.reserve_request(run_id=self.run_id)
                    # Persist the sending state before network I/O. A crash after
                    # this point stays ambiguous and permanently consumes budget.
                    self.request_gate.mark_attempting(attempt_id)
                record.network_attempted = True
                with httpx.Client(timeout=self.timeout, transport=self.transport, follow_redirects=False) as client:
                    response = client.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json=payload)
                response_received = True
                record.http_status = response.status_code
                record.raw_response = self._redact(response.text)
                response.raise_for_status()
                data = parse_json_object(response.text)
                record.response_model = data.get("model")
                record.usage = self._usage(data)
                choice = data["choices"][0]
                record.finish_reason = choice.get("finish_reason")
                message = choice["message"]
                record.content = self._redact(message.get("content")) if isinstance(message.get("content"), str) else None
                native_tool_calls = message.get("tool_calls")
                if isinstance(native_tool_calls, list):
                    record.native_tool_calls = native_tool_calls
                if message.get("refusal") or record.finish_reason == "content_filter":
                    record.error = ErrorRecord(code="refused", message="Model/API refused the request")
                elif record.finish_reason != "stop":
                    record.error = ErrorRecord(code="incomplete_response", message=f"finish_reason={record.finish_reason}")
                elif record.response_model and record.response_model.lower() != self.model:
                    record.error = ErrorRecord(code="model_mismatch", message=f"API returned unexpected model: {record.response_model}")
                elif record.content is None and not record.native_tool_calls:
                    raise ValueError("Response contains no text content")
        except BudgetExceeded as exc:
            record.error = ErrorRecord(code="budget_exhausted", message=str(exc))
        except LedgerError as exc:
            record.error = ErrorRecord(code="budget_ledger_error", message=str(exc))
        except httpx.TimeoutException:
            record.error = ErrorRecord(code="timeout", message="VLM request timed out; no automatic retry")
        except httpx.HTTPStatusError:
            record.error = ErrorRecord(code="http_error", message=f"HTTP {record.http_status}; see raw_response")
        except httpx.RequestError as exc:
            record.error = ErrorRecord(code="network_error", message=self._redact(str(exc)))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            record.error = ErrorRecord(code="invalid_response", message=self._redact(str(exc)))
        if attempt_id is not None:
            try:
                if response_received:
                    self.request_gate.finish_attempt(attempt_id, http_status=record.http_status,
                                                     error_type=record.error.code if record.error else None)
                else:
                    self.request_gate.mark_unknown(attempt_id,
                                                   error_type=record.error.code if record.error else "result_unknown")
            except LedgerError as exc:
                record.error = ErrorRecord(code="budget_ledger_error", message=str(exc))
        record.latency_seconds = perf_counter() - started
        record.stop_reason = record.finish_reason
        record.estimated_cost = record.usage.cost_usd
        self._sent_frame_hashes.update(frame_hashes)
        self.calls.append(record)
        if self.store:
            self.store.append_call(record)
        if record.error:
            raise ProviderFailure(record)
        return record


class FakeVLMProvider:
    """Scriptable fake. Never measures semantic accuracy or claims MiMo usage."""

    name = "fake"
    model = "fake-scripted-v1"
    is_fake = True

    def __init__(self, responses: list[dict | str] | None = None, store: ArtifactStore | None = None,
                 active_demo: bool = False, request_gate: ExperimentLedger | None = None,
                 run_id: str | None = None):
        self.responses = responses
        self.store = store
        self.active_demo = active_demo
        self.calls: list[CallRecord] = []
        self._sent_frame_hashes: set[str] = set()
        self._candidate_first_demo = False
        self._staged_first_candidate: str | None = None
        # Optional ledger injection is a test-only way to exercise the exact
        # persistent reservation path without constructing a real API client.
        self.request_gate = request_gate
        self.run_id = run_id

    def complete(self, prompt: str, frames: Sequence[Observation],
                 tools: Sequence[dict[str, Any]] | None = None) -> CallRecord:
        started = perf_counter()
        attempt_id = None
        if self.request_gate is not None:
            if not self.run_id:
                raise LedgerError("Fake ledger test requires a registered run_id")
            attempt_id = self.request_gate.reserve_request(run_id=self.run_id)
            self.request_gate.mark_attempting(attempt_id)
        if self.store:
            for frame in frames:
                self.store.save_frame(frame)
        if self.responses is not None:
            value = self.responses[len(self.calls)] if len(self.calls) < len(self.responses) else "FAKE_SCRIPT_EXHAUSTED"
        elif self.active_demo and "You are a staged active video observer." in prompt:
            candidate_ids = list(dict.fromkeys(re.findall(r"C[a-f0-9]{8}-\d{2}", prompt)))
            if len(self.calls) == 0 and candidate_ids:
                self._staged_first_candidate = candidate_ids[0]
                value = {"action": "select_candidate", "candidate_id": candidate_ids[0], "count": 2}
            elif len(self.calls) == 1 and candidate_ids:
                selected = next((candidate_id for candidate_id in candidate_ids
                                 if candidate_id != self._staged_first_candidate), None)
                prior_times = [f"{frame.record.timestamp:g}s/" for frame in frames[-2:]]
                if selected and "Prior actions" in prompt and all(ts in prompt for ts in prior_times):
                    value = {"action": "select_candidate", "candidate_id": selected, "count": 2}
                else:
                    value = {"action": "final_answer", "answer": "", "evidence_timestamps": [],
                             "status": "refused"}
            else:
                value = {"action": "final_answer", "answer": "FAKE ONLY: no visual interpretation performed",
                         "evidence_timestamps": [frames[-1].record.timestamp], "status": "ok"}
        elif "Initial decision: select a listed candidate" in prompt:
            self._candidate_first_demo = True
            match = re.search(r"C[a-f0-9]{8}-\d{2}", prompt)
            if match:
                value = {"action": "select_candidate", "candidate_id": match.group(0), "count": 2}
            else:
                value = {"action": "final_answer", "answer": "", "evidence_timestamps": [], "status": "refused"}
        elif self.active_demo and self._candidate_first_demo and len(self.calls) == 1:
            available = re.search(r"remaining extra frames=(\d+)", prompt)
            count = max(1, min(2, int(available.group(1)) if available else 1))
            times = [f.record.timestamp for f in frames]
            start, end = min(times), max(times)
            if start == end:
                start, end = max(0.0, start - 0.25), start + 0.25
            value = {"action": "sample_interval", "start": start, "end": end, "count": count}
        elif self.active_demo and len(self.calls) == 0:
            value = {"action": "inspect_frame", "timestamp": frames[0].record.timestamp}
        elif self.active_demo and len(self.calls) == 1:
            times = [f.record.timestamp for f in frames]
            start, end = min(times), max(times)
            if start == end:
                start, end = max(0.0, start - 0.25), start + 0.25
            value = {"action": "sample_interval", "start": start, "end": end, "count": 3}
        else:
            value = {"answer": "FAKE ONLY: no visual interpretation performed", "evidence_timestamps": [frames[0].record.timestamp], "status": "ok"}
            if self.active_demo:
                value["action"] = "final_answer"
        fake_error = value.get("__fake_error__") if isinstance(value, dict) else None
        content = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
        frame_hashes = [frame.record.sha256 for frame in frames]
        unique_hashes = set(frame_hashes)
        reused_count = sum(frame_hash in self._sent_frame_hashes for frame_hash in frame_hashes)
        record = CallRecord(index=len(self.calls) + 1, request_index=len(self.calls) + 1,
                            started_at_utc=utc_now(), provider=self.name, model=self.model, is_fake=True,
                            latency_seconds=perf_counter() - started, frames=[f.record.model_copy() for f in frames],
                            image_count=len(frames), unique_image_count=len(unique_hashes),
                            reused_image_count=reused_count,
                            request={"prompt": prompt}, raw_response=content, content=content,
                            usage=Usage(), finish_reason="stop", stop_reason="stop")
        if fake_error:
            record.error = ErrorRecord(code=str(fake_error), message="Fake provider injected failure")
        self._sent_frame_hashes.update(frame_hashes)
        self.calls.append(record)
        if self.store:
            self.store.append_call(record)
        if attempt_id is not None:
            if fake_error:
                self.request_gate.finish_attempt(attempt_id, http_status=503, error_type=str(fake_error))
            else:
                self.request_gate.finish_attempt(attempt_id, http_status=200, error_type=None)
        if fake_error:
            raise ProviderFailure(record)
        return record
