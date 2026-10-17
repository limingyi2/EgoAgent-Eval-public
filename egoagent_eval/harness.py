"""Compatibility imports; the only Agent loop lives in core.agent."""
from .core.agent import AgentHarness, run_agent_harness
from .core.state import AgentState, TerminationReason
from .core.controller import (ContextObservationPolicy, ObservationPolicy, RetrievalStrategy,
                              UniformRetrieval, AdaptiveRetrieval, HierarchicalRetrieval,
                              FullContext, SelectiveContext, CompactContext, default_prompt)
NullRetrievalStrategy = AdaptiveRetrieval
IndexRetrievalStrategy = HierarchicalRetrieval
