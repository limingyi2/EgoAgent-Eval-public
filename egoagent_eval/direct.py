"""Direct-4 and Uniform-8 use one image-grounded model call."""

import json
from time import perf_counter

from .decisions import AnswerDecision, ProtocolError, ProviderRefusal, RawModelResponse, decision_trace
from .protocol import evidence_coverage, normalize_model_response
from .runtime import apply_final, finish_result, start_result
from .schema import ErrorRecord, FinalAnswer, Question, RunResult
from .video import Observation, Video
from .vlm import ProviderFailure, VLMProvider


def run_direct(question: Question, video: Video, provider: VLMProvider,
               method: str = "direct4", initial: list[Observation] | None = None) -> RunResult:
    if method not in ("direct4", "uniform8"):
        raise ValueError("Direct method must be direct4 or uniform8")
    started = perf_counter()
    first_call = len(provider.calls)
    frames = initial if initial is not None else video.uniform_frames(4 if method == "direct4" else 8)
    if len(frames) != (4 if method == "direct4" else 8):
        raise ValueError("Incorrect fixed-frame count")
    result = start_result(question, method, provider, video, frames)
    prompt = ("Answer the question using only the supplied video frames. Do not infer hidden annotations. "
              "Cite only supplied observed timestamps in seconds. Return only JSON matching this schema: "
              + json.dumps(FinalAnswer.model_json_schema()) + "\nQuestion: " + json.dumps(question.question, ensure_ascii=False))
    try:
        call = provider.complete(prompt, frames)
        decision = normalize_model_response(RawModelResponse.from_call(call), frames)
        call.evidence_coverage = evidence_coverage(decision, frames)
        store = getattr(provider, "store", None)
        if store:
            store.append_protocol({"call_index": call.index,
                                   "decision": decision_trace(decision),
                                   "evidence_coverage": call.evidence_coverage})
        if isinstance(decision, AnswerDecision):
            apply_final(result, decision.answer)
        elif isinstance(decision, ProviderRefusal):
            result.errors.append(ErrorRecord(code="refused", message=decision.reason))
            result.status = "refused"
        elif isinstance(decision, ProtocolError):
            result.errors.append(ErrorRecord(code="protocol_error",
                                             message=f"{decision.code}: {decision.message}"))
            result.status = "failed"
        else:
            result.errors.append(ErrorRecord(code="protocol_error",
                                             message="Direct method received a tool decision"))
            result.status = "failed"
    except ProviderFailure as exc:
        result.errors.append(exc.record.error)
        result.status = "refused" if exc.record.error.code == "refused" else "failed"
    except ValueError as exc:
        result.errors.append(ErrorRecord(code="invalid_final_answer", message=str(exc)))
        result.status = "failed"
    return finish_result(result, provider, first_call, started)
