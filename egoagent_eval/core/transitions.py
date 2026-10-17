"""Validated transitions with append-only state event recording."""
from ..schema import StateEvent
from .state import AgentState


ALLOWED_TRANSITIONS = {
    AgentState.INITIALIZE: {AgentState.OBSERVE, AgentState.FINALIZE},
    AgentState.OBSERVE: {AgentState.DECIDE, AgentState.FINALIZE},
    AgentState.DECIDE: {AgentState.EXECUTE_TOOL, AgentState.VERIFY, AgentState.FINALIZE},
    AgentState.EXECUTE_TOOL: {AgentState.VERIFY, AgentState.OBSERVE, AgentState.FINALIZE},
    AgentState.VERIFY: {AgentState.OBSERVE, AgentState.FINALIZE},
    AgentState.FINALIZE: set(),
}


class TransitionRecorder:
    def __init__(self, result, observed, budget, store=None):
        self.state = AgentState.INITIALIZE
        self.result = result
        self.observed = observed
        self.budget = budget
        self.store = store

    def transition(self, next_state, reason, error_code=None, request_index=None, before=None):
        if next_state not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"Invalid state transition: {self.state.value} -> {next_state.value}")
        after = self.budget.snapshot()
        event = StateEvent(
            index=len(self.result.state_events), state_from=self.state.value,
            state_to=next_state.value, reason=reason, request_index=request_index,
            observed_frame_hashes=[f.record.sha256 for f in self.observed],
            observed_timestamps=[f.record.timestamp for f in self.observed],
            budget_before=(before or after).as_dict(), budget_after=after.as_dict(),
            error_code=error_code,
        )
        self.result.state_events.append(event)
        if self.store:
            self.store.append_state_event(event.model_dump())
        self.state = next_state
