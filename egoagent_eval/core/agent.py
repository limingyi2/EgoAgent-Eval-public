"""The sole active-observation loop. Legacy method adapters call this harness."""
from __future__ import annotations
from time import perf_counter
from typing import Any
from ..decisions import AnswerDecision, ProtocolError, ProviderRefusal, RawModelResponse, decision_trace
from ..protocol import evidence_coverage, normalize_model_response
from ..records import ArtifactStore
from ..runtime import apply_final, finish_result, start_result
from ..schema import Budget, ErrorRecord, Question, RunResult, SampleInterval, SelectCandidate
from ..tools import ObservationTools
from ..video import Observation, Video
from ..vlm import ProviderFailure, VLMProvider
from .state import AgentState, TerminationReason, terminal_for_reason
from .transitions import TransitionRecorder
from .controller import (BudgetManager, EvidenceValidator, ObservationPolicy, RetrievalStrategy,
                         ContextObservationPolicy, AdaptiveRetrieval, PromptBuilder,
                         default_prompt, CandidateHierarchicalRetrieval)

class AgentHarness:
    """Single bounded loop used by Active, Adaptive and Hierarchical modes."""

    def __init__(self, *, policy: ObservationPolicy | None = None,
                 retrieval: RetrievalStrategy | None = None,
                 prompt_builder: PromptBuilder = default_prompt):
        self.policy = policy or ContextObservationPolicy("selective_context")
        self.retrieval = retrieval or AdaptiveRetrieval()
        self.prompt_builder = prompt_builder

    def run(self, question: Question, video: Video, provider: VLMProvider,
            *, method: str, initial: list[Observation] | None = None,
            budget: Budget | None = None, store: ArtifactStore | None = None) -> RunResult:
        started = perf_counter()
        first_call = len(provider.calls)
        budget = budget or Budget()
        initial_frames = list(initial if initial is not None else self.retrieval.initial_frames(video))
        if len(initial_frames) != 4 or any(max(f.record.width, f.record.height) > budget.max_image_side
                                           for f in initial_frames):
            raise ValueError("Active harness requires exactly 4 initial frames within image budget")
        result = start_result(question, method, provider, video, initial_frames)
        context = self.retrieval.prepare(video, question)
        if hasattr(context, "scanned_frames"):
            result.index_scanned_frames = context.scanned_frames
            result.index_cpu_seconds = context.cpu_seconds
            result.candidate_pool_id = getattr(context, "candidate_pool_id", None)
            windows = getattr(context, "candidate_windows", None)
            if windows is None:
                windows = getattr(context, "candidates", ())
            result.index_candidate_windows = [
                ({"candidate_id": w.candidate_id, "start": w.start, "end": w.end,
                  "score": w.score, "source": w.source} if hasattr(w, "candidate_id") else
                 {"start": w.start, "end": w.end, "score": w.score, "source": w.source})
                for w in windows
            ]
            if store:
                store.write_json("index.json", context.to_dict())
        tools = ObservationTools(video, budget, store)
        observed = list(initial_frames)
        presented: list[Observation] = []
        new_frames: list[Observation] = []
        history: list[dict] = []
        manager = BudgetManager(budget)
        validator = EvidenceValidator()
        machine = TransitionRecorder(result, observed, manager, store)
        final_only = False
        snapshot = manager.snapshot
        transition = machine.transition

        transition(AgentState.OBSERVE, "initialized")
        for _ in range(budget.max_model_calls):
            before_call = snapshot()
            transition(AgentState.DECIDE, "prepare_model_decision")
            remaining_calls = manager.snapshot().model_calls - 1
            remaining_frames = manager.snapshot().extra_frames
            remaining_tools = manager.snapshot().tool_calls
            send_frames = self.policy.select(initial_frames, observed, new_frames)
            candidate_ranking = None
            selection_policy = getattr(self.retrieval, "candidate_selection_policy", None)
            if selection_policy == "temporal_complementarity_ranked":
                candidate_ranking = self.retrieval.rank_candidates(
                    context, observed, remaining_frames=remaining_frames,
                    remaining_tools=remaining_tools)
                ranking_event = {"event": "candidate_ranking", "selection_policy": selection_policy,
                                 "selection_mode": "model_choice_after_python_ranking",
                                 "candidate_pool_id": result.candidate_pool_id,
                                 "remaining_frames": remaining_frames,
                                 "remaining_tools": remaining_tools,
                                 "ranking": candidate_ranking}
                result.retrieval_trace.append(ranking_event)
                if store:
                    store.append_retrieval(ranking_event)
            prompt = self.prompt_builder(question, video, self.policy, remaining_calls,
                                         remaining_frames, remaining_tools, history, observed,
                                         (self.retrieval.prompt_context(context, observed,
                                                                        ranking=candidate_ranking)
                                          if candidate_ranking is not None else
                                          self.retrieval.prompt_context(context, observed)), final_only)
            try:
                if any(not any(f.record.timestamp == known.record.timestamp and
                               f.record.sha256 == known.record.sha256 for known in observed)
                       for f in send_frames):
                    raise ValueError("Context policy returned a frame outside the observed set")
                manager.consume_request(send_frames)
                call = provider.complete(prompt, send_frames)
                presented.extend(send_frames)
                decision = normalize_model_response(RawModelResponse.from_call(call), send_frames)
                call.evidence_coverage = evidence_coverage(
                    decision, send_frames, presented)
                if store:
                    store.append_protocol({"call_index": call.index,
                                           "decision": decision_trace(decision),
                                           "evidence_coverage": call.evidence_coverage})
            except ProviderFailure as exc:
                result.errors.append(exc.record.error)
                reason = (TerminationReason.PROVIDER_REFUSAL if exc.record.error.code == "refused"
                          else TerminationReason.PROVIDER_FAILURE)
                result.status = "refused" if reason == TerminationReason.PROVIDER_REFUSAL else "failed"
                result.termination_reason = reason.value
                transition(AgentState.FINALIZE, "provider failure", exc.record.error.code,
                           exc.record.index, before_call)
                break
            except ValueError as exc:
                result.errors.append(ErrorRecord(code="protocol_error", message=str(exc)))
                result.status = "failed"
                result.termination_reason = TerminationReason.PROTOCOL_ERROR.value
                transition(AgentState.FINALIZE, "provider record error", "protocol_error", None, before_call)
                break

            if isinstance(decision, ProtocolError):
                result.errors.append(ErrorRecord(code="protocol_error",
                                                 message=f"{decision.code}: {decision.message}"))
                result.status = "failed"
                result.termination_reason = TerminationReason.PROTOCOL_ERROR.value
                transition(AgentState.FINALIZE, "protocol normalization failed", decision.code,
                           call.index, before_call)
                break
            if isinstance(decision, ProviderRefusal):
                result.errors.append(ErrorRecord(code="refused", message=decision.reason))
                result.status = "refused"
                result.termination_reason = TerminationReason.PROVIDER_REFUSAL.value
                transition(AgentState.FINALIZE, "provider refusal", "refused", call.index, before_call)
                break
            if isinstance(decision, AnswerDecision):
                transition(AgentState.VERIFY, "final answer received", request_index=call.index,
                           before=before_call)
                cited = decision.answer.evidence_timestamps
                verification = validator.validate(question, decision.answer, presented, video.duration)
                if not verification.traceable:
                    result.errors.append(ErrorRecord(code="invalid_final_answer",
                                                     message="Evidence timestamp was never observed"))
                    result.status = "failed"
                    result.termination_reason = TerminationReason.PROTOCOL_ERROR.value
                    transition(AgentState.FINALIZE, "unobserved evidence timestamp", "invalid_final_answer",
                               call.index, before_call)
                    break
                if decision.answer.status == "ok" and not verification.structurally_complete:
                    history.append({"final_rejected": "insufficient_evidence",
                                    "cited_timestamps": cited})
                    if remaining_calls > 0 and remaining_frames > 0 and remaining_tools > 0:
                        final_only = False
                        transition(AgentState.OBSERVE, "final answer needs more evidence", before=before_call)
                        new_frames = []
                        continue
                    result.errors.append(ErrorRecord(code="insufficient_evidence",
                                                     message="Final answer did not cite enough observed evidence"))
                    result.status = "refused"
                    result.answer = ""
                    result.evidence_timestamps = []
                    result.termination_reason = TerminationReason.INSUFFICIENT_EVIDENCE.value
                    transition(AgentState.FINALIZE, "insufficient evidence", "insufficient_evidence",
                               call.index, before_call)
                    break
                try:
                    apply_final(result, decision.answer)
                    result.termination_reason = (TerminationReason.ANSWERED.value
                                                 if decision.answer.status == "ok"
                                                 else TerminationReason.INSUFFICIENT_EVIDENCE.value)
                    transition(AgentState.FINALIZE, "final answer verified", request_index=call.index,
                               before=before_call)
                except ValueError as exc:
                    result.errors.append(ErrorRecord(code="invalid_final_answer", message=str(exc)))
                    result.status = "failed"
                    result.termination_reason = TerminationReason.PROTOCOL_ERROR.value
                    transition(AgentState.FINALIZE, "final answer application failed", "invalid_final_answer",
                               call.index, before_call)
                break

            action = decision.action
            action_gate = getattr(self.retrieval, "action_allowed", None)
            staged = bool(getattr(self.retrieval, "staged_observation", False))
            staged_action_invalid = staged and (not isinstance(action, (SelectCandidate, SampleInterval))
                                                 or not action_gate(action, history))
            if staged_action_invalid or (action_gate is not None and not staged
                                         and not action_gate(action, history)):
                transition(AgentState.EXECUTE_TOOL, "strategy action gate rejected tool",
                           "action_not_allowed", call.index, before_call)
                message = ("Staged observation permits only listed candidate selection or a final answer"
                           if staged else
                           "Initial candidate-first decision permits only candidate selection or a final answer")
                output = tools.reject_action(action, "action_not_allowed", message)
                manager.record_rejected_tool()
                event = {"candidate_pool_id": result.candidate_pool_id,
                         "candidate_id": None, "source": None, "score": None,
                         "start": getattr(action, "start", None),
                         "end": getattr(action, "end", None),
                         "selected": False, "executed": False,
                         "selection_mode": "candidate_first_gate",
                         "used_candidate": False, "used_freeform_interval": False,
                         "action_allowed": False, "action": action.model_dump(),
                         "execution_error": "action_not_allowed",
                         "actual_observed_timestamps": [], "frame_hashes": []}
                result.retrieval_trace.append(event)
                if store:
                    store.append_retrieval(event)
                result.errors.append(ErrorRecord(code="action_not_allowed", message=message))
                result.status = "failed"
                result.termination_reason = TerminationReason.TOOL_ERROR.value
                transition(AgentState.FINALIZE, "strategy action rejected", "action_not_allowed",
                           call.index, before_call)
                break
            candidate = None
            candidate_error = None
            if isinstance(action, SelectCandidate):
                try:
                    if staged and action.count != 2:
                        raise ValueError("Staged observation requires exactly two frames per candidate selection")
                    if not isinstance(self.retrieval, CandidateHierarchicalRetrieval):
                        raise ValueError("select_candidate is enabled only for hierarchical_active_v2")
                    candidate = self.retrieval.resolve_candidate(action, context)
                except ValueError as exc:
                    candidate_error = str(exc)
            if staged and isinstance(action, SampleInterval) and action.count != 2:
                candidate_error = "Staged observation requires exactly two frames per interval fallback"
            if staged and isinstance(action, SampleInterval) and not (
                    0 <= action.start < action.end <= video.duration):
                candidate_error = "Fallback interval must satisfy 0 <= start < end <= video duration"
            if isinstance(action, SelectCandidate) and candidate_error:
                transition(AgentState.EXECUTE_TOOL, "candidate selection rejected", "invalid_candidate",
                           call.index, before_call)
                output = tools.reject_candidate(action, "invalid_candidate", candidate_error,
                                                result.candidate_pool_id)
                manager.record_rejected_tool()
                retrieval_event = {"candidate_pool_id": result.candidate_pool_id,
                                   "candidate_id": action.candidate_id, "source": None, "score": None,
                                   "start": None, "end": None, "selected": True, "action_allowed": True,
                                   "executed": False,
                                   "selection_mode": "candidate_id",
                                   "selection_policy": getattr(self.retrieval, "candidate_selection_policy", "baseline"),
                                   "decision_source": "model", "used_candidate": True,
                                   "used_freeform_interval": False,
                                   "execution_error": candidate_error, "actual_observed_timestamps": [],
                                   "frame_hashes": []}
                result.retrieval_trace.append(retrieval_event)
                if store:
                    store.append_retrieval(retrieval_event)
                result.errors.append(ErrorRecord(code="invalid_candidate", message=candidate_error))
                result.status = "failed"
                result.termination_reason = TerminationReason.TOOL_ERROR.value
                transition(AgentState.FINALIZE, "candidate selection failed", "invalid_candidate",
                           call.index, before_call)
                break
            rejection = manager.tool_rejection(action)
            if staged and isinstance(action, SampleInterval) and candidate_error:
                rejection = "invalid_interval"
            if rejection in {"frame_budget", "tool_call_budget"} and (remaining_frames == 0 or remaining_tools == 0):
                transition(AgentState.EXECUTE_TOOL, "budget gate rejected tool", "budget_exhausted_action",
                           call.index, before_call)
                reject_code = rejection
                manager.record_rejected_tool()
                output = (tools.reject_candidate(action, reject_code,
                                                 "Candidate observation rejected because budget was exhausted",
                                                 result.candidate_pool_id) if isinstance(action, SelectCandidate)
                          else tools.reject_action(action, reject_code,
                                             "Tool action rejected before execution because the budget was exhausted")
                          )
                if isinstance(action, SelectCandidate):
                    event = {"candidate_pool_id": result.candidate_pool_id,
                             "candidate_id": action.candidate_id,
                             "source": candidate.source if candidate else None,
                             "score": candidate.score if candidate else None,
                             "start": candidate.start if candidate else None,
                             "end": candidate.end if candidate else None,
                             "selected": True, "action_allowed": True, "executed": False,
                             "selection_mode": "candidate_id",
                             "selection_policy": getattr(self.retrieval, "candidate_selection_policy", "baseline"),
                             "decision_source": "model", "used_candidate": True,
                             "used_freeform_interval": False,
                             "execution_error": reject_code, "actual_observed_timestamps": [],
                             "frame_hashes": []}
                    if staged and candidate is not None:
                        event["candidate_observation"] = self.retrieval.candidate_observation_snapshot(
                            candidate, observed)
                    result.retrieval_trace.append(event)
                    if store:
                        store.append_retrieval(event)
                elif isinstance(action, SampleInterval) and isinstance(self.retrieval, CandidateHierarchicalRetrieval):
                    matching = [item.candidate_id for item in context.candidates
                                if action.start < item.end and action.end > item.start]
                    event = {"candidate_pool_id": result.candidate_pool_id,
                             "candidate_id": None, "candidate_ids_overlapped": matching,
                             "source": "freeform_fallback", "score": None,
                             "start": action.start, "end": action.end, "count": action.count,
                             "selected": False, "action_allowed": True, "executed": False,
                             "selection_mode": "freeform_interval", "used_candidate": False,
                             "used_freeform_interval": True,
                             "execution_error": reject_code, "actual_observed_timestamps": [],
                             "frame_hashes": []}
                    result.retrieval_trace.append(event)
                    if store:
                        store.append_retrieval(event)
                history.append({"action": action.model_dump(), "tool": output.record.tool,
                                "status": output.record.status, "timestamps": []})
                result.errors.append(ErrorRecord(code="budget_exhausted_action",
                                                 message=f"Tool action rejected before execution: {reject_code}"))
                final_only = True
                if remaining_calls > 0:
                    transition(AgentState.OBSERVE, "request final-only decision after budget gate",
                               before=before_call)
                    new_frames = []
                    continue
                result.errors.append(ErrorRecord(code="insufficient_evidence",
                                                 message="No observation budget remained for a supported final answer"))
                result.status = "refused"
                result.answer = ""
                result.evidence_timestamps = []
                result.termination_reason = TerminationReason.BUDGET_EXHAUSTED.value
                transition(AgentState.FINALIZE, "budget exhausted", "budget_exhausted_action",
                           call.index, before_call)
                break

            transition(AgentState.EXECUTE_TOOL, "execute validated tool", request_index=call.index,
                       before=before_call)
            if rejection:
                manager.record_rejected_tool()
                output = (tools.reject_candidate(action, rejection, "Candidate action exceeds bounded budget",
                                                 result.candidate_pool_id) if isinstance(action, SelectCandidate)
                          else tools.reject_action(action, rejection, "Tool action exceeds bounded budget"))
                if isinstance(action, SelectCandidate):
                    event = {"candidate_pool_id": result.candidate_pool_id,
                             "candidate_id": action.candidate_id,
                             "source": candidate.source if candidate else None,
                             "score": candidate.score if candidate else None,
                             "start": candidate.start if candidate else None,
                             "end": candidate.end if candidate else None,
                             "selected": True, "action_allowed": True, "executed": False,
                             "selection_mode": "candidate_id",
                             "selection_policy": getattr(self.retrieval, "candidate_selection_policy", "baseline"),
                             "decision_source": "model", "used_candidate": True,
                             "used_freeform_interval": False,
                             "execution_error": rejection, "actual_observed_timestamps": [],
                             "frame_hashes": []}
                    if staged and candidate is not None:
                        event["candidate_observation"] = self.retrieval.candidate_observation_snapshot(
                            candidate, observed)
                    result.retrieval_trace.append(event)
                    if store:
                        store.append_retrieval(event)
                elif isinstance(action, SampleInterval) and isinstance(self.retrieval, CandidateHierarchicalRetrieval):
                    matching = [item.candidate_id for item in context.candidates
                                if action.start < item.end and action.end > item.start]
                    event = {"candidate_pool_id": result.candidate_pool_id,
                             "candidate_id": None, "candidate_ids_overlapped": matching,
                             "source": "freeform_fallback", "score": None,
                             "start": action.start, "end": action.end, "count": action.count,
                             "selected": False, "action_allowed": True, "executed": False,
                             "selection_mode": "freeform_interval", "used_candidate": False,
                             "used_freeform_interval": True,
                             "execution_error": rejection, "actual_observed_timestamps": [],
                             "frame_hashes": []}
                    result.retrieval_trace.append(event)
                    if store:
                        store.append_retrieval(event)
            else:
                manager.reserve_tool(action)
                if isinstance(action, SelectCandidate):
                    sample_candidate = getattr(self.retrieval, "sample_candidate", None)
                    output = (sample_candidate(tools, action, candidate, context, observed=observed)
                              if staged and sample_candidate is not None
                              else sample_candidate(tools, action, candidate, context)
                              if sample_candidate is not None
                              else tools.sample_candidate(action, candidate, result.candidate_pool_id))
                else:
                    output = (self.retrieval.sample_interval(tools, action, observed=observed)
                              if staged and isinstance(action, SampleInterval)
                              else tools.execute(action))
            new_frames = output.frames
            result.index_scanned_frames += output.record.index_scanned_frames
            result.index_cpu_seconds += output.record.index_cpu_seconds
            observed.extend(new_frames)
            result.observed_frames = [frame.record.model_copy() for frame in observed]
            transition(AgentState.VERIFY, "verify tool observation", request_index=call.index,
                       before=before_call)
            if isinstance(action, SelectCandidate):
                event = {"candidate_pool_id": result.candidate_pool_id,
                         "candidate_id": action.candidate_id, "source": candidate.source,
                         "score": candidate.score, "start": candidate.start, "end": candidate.end,
                         "selected": True, "action_allowed": True,
                         "sampling_policy": getattr(self.retrieval, "sampling_policy", "uniform"),
                         "stage": len([item for item in history if item.get("status") == "ok"]) + 1 if staged else None,
                         "local_index_scanned_frames": output.record.index_scanned_frames,
                         "local_index_cpu_seconds": output.record.index_cpu_seconds,
                         "executed": output.record.status == "ok",
                         "selection_mode": "candidate_id",
                         "selection_policy": getattr(self.retrieval, "candidate_selection_policy", "baseline"),
                         "decision_source": "model", "used_candidate": True,
                         "used_freeform_interval": False,
                         "execution_error": output.record.error.code if output.record.error else None,
                         "actual_observed_timestamps": [frame.record.timestamp for frame in new_frames],
                         "frame_hashes": [frame.record.sha256 for frame in new_frames]}
                if staged and isinstance(output.record.arguments, dict):
                    candidate_observation = output.record.arguments.get("candidate_observation")
                    if candidate_observation is not None:
                        event["candidate_observation"] = candidate_observation
                result.retrieval_trace.append(event)
                if store:
                    store.append_retrieval(event)
            elif isinstance(action, SampleInterval) and isinstance(self.retrieval, CandidateHierarchicalRetrieval):
                matching = [item.candidate_id for item in context.candidates
                            if action.start < item.end and action.end > item.start]
                event = {"candidate_pool_id": result.candidate_pool_id,
                         "candidate_id": None, "candidate_ids_overlapped": matching,
                         "source": "freeform_fallback", "score": None,
                         "start": action.start, "end": action.end, "count": action.count,
                         "selected": False, "action_allowed": True,
                         "executed": output.record.status == "ok",
                         "selection_mode": "freeform_interval", "used_candidate": False,
                         "used_freeform_interval": True,
                         "execution_error": output.record.error.code if output.record.error else None,
                         "actual_observed_timestamps": [frame.record.timestamp for frame in new_frames],
                         "frame_hashes": [frame.record.sha256 for frame in new_frames]}
                result.retrieval_trace.append(event)
                if store:
                    store.append_retrieval(event)
            elif staged and isinstance(action, SampleInterval):
                event = {"candidate_pool_id": result.candidate_pool_id,
                         "candidate_id": None, "source": "freeform_fallback",
                         "start": action.start, "end": action.end, "count": action.count,
                         "selected": False, "action_allowed": True,
                         "executed": output.record.status == "ok",
                         "selection_mode": "freeform_interval", "used_candidate": False,
                         "used_freeform_interval": True,
                         "execution_error": output.record.error.code if output.record.error else None,
                         "actual_observed_timestamps": [frame.record.timestamp for frame in new_frames],
                         "frame_hashes": [frame.record.sha256 for frame in new_frames]}
                result.retrieval_trace.append(event)
                if store:
                    store.append_retrieval(event)
            if output.record.error:
                result.errors.append(output.record.error)
                result.status = "failed"
                result.termination_reason = (TerminationReason.BUDGET_EXHAUSTED.value
                                             if rejection in {"frame_budget", "tool_call_budget"}
                                             else TerminationReason.TOOL_ERROR.value)
                transition(AgentState.FINALIZE, "tool execution failed", output.record.error.code,
                           call.index, before_call)
                break
            history.append({"action": action.model_dump(), "tool": output.record.tool,
                            "status": output.record.status,
                            "timestamps": [frame.record.timestamp for frame in new_frames],
                            "candidate_match": self.retrieval.candidate_match(action, context)})
            transition(AgentState.OBSERVE, "tool observation completed", request_index=call.index,
                       before=before_call)
        else:
            result.errors.append(ErrorRecord(code="model_call_budget",
                                             message="No final answer within model call budget"))
            result.status = "failed"
            result.termination_reason = TerminationReason.BUDGET_EXHAUSTED.value
            transition(AgentState.FINALIZE, "model call budget exhausted", "model_call_budget")

        result.tool_calls = tools.records
        if result.termination_reason is None:
            result.termination_reason = TerminationReason.BUDGET_EXHAUSTED.value
        result.terminal_state = terminal_for_reason(result.termination_reason).value
        return finish_result(result, provider, first_call, started)


def run_agent_harness(question: Question, video: Video, provider: VLMProvider, *,
                      method: str = "adaptive_active", initial: list[Observation] | None = None,
                      budget: Budget | None = None, store: ArtifactStore | None = None,
                      context_policy: str = "selective_context",
                      retrieval: RetrievalStrategy | None = None,
                      prompt_builder: PromptBuilder = default_prompt) -> RunResult:
    return AgentHarness(policy=ContextObservationPolicy(context_policy), retrieval=retrieval,
                        prompt_builder=prompt_builder).run(
                            question, video, provider, method=method, initial=initial,
                            budget=budget, store=store)
