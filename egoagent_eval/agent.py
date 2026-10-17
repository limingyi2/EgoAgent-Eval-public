"""Active-4+4: strict JSON action loop, at most two tools and three VLM calls."""


from .harness import run_agent_harness
from .core.controller import UniformRetrieval
from .records import ArtifactStore
from .schema import Budget, Question, RunResult
from .video import Observation, Video
from .vlm import VLMProvider


def run_active(question: Question, video: Video, provider: VLMProvider,
               initial: list[Observation] | None = None, budget: Budget | None = None,
               store: ArtifactStore | None = None) -> RunResult:
    result = run_agent_harness(question, video, provider, method="active",
                               initial=initial, budget=budget, store=store,
                               context_policy="full_context", retrieval=UniformRetrieval())
    # Preserve the legacy Active scoring status while exposing the explicit v2
    # termination_reason and state_events fields.
    if result.termination_reason == "budget_exhausted" and result.status == "refused":
        result.status = "failed"
    return result
