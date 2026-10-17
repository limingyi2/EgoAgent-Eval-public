"""States and terminal outcomes, independent of transport and strategies."""
from enum import Enum


class AgentState(str, Enum):
    INITIALIZE = "INITIALIZE"
    OBSERVE = "OBSERVE"
    DECIDE = "DECIDE"
    EXECUTE_TOOL = "EXECUTE_TOOL"
    VERIFY = "VERIFY"
    FINALIZE = "FINALIZE"


class TerminalState(str, Enum):
    ANSWERED = "ANSWERED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    PROVIDER_ERROR = "PROVIDER_ERROR"


class TerminationReason(str, Enum):
    # Preserve historical lowercase codes and distinguish an explicit refusal.
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROTOCOL_ERROR = "protocol_error"
    TOOL_ERROR = "tool_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PROVIDER_REFUSAL = "provider_refusal"
    PROVIDER_FAILURE = "provider_failure"


def terminal_for_reason(reason: str) -> TerminalState:
    return {
        "answered": TerminalState.ANSWERED,
        "insufficient_evidence": TerminalState.INSUFFICIENT_EVIDENCE,
        "protocol_error": TerminalState.PROTOCOL_ERROR,
        "tool_error": TerminalState.TOOL_ERROR,
        "budget_exhausted": TerminalState.BUDGET_EXHAUSTED,
        "provider_refusal": TerminalState.INSUFFICIENT_EVIDENCE,
        "provider_failure": TerminalState.PROVIDER_ERROR,
    }[reason]
