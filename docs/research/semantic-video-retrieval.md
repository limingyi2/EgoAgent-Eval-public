# Semantic video retrieval: design notes

The current retrieval stack combines temporal coverage with lightweight appearance-change signals. These signals help propose candidate windows but do not encode the semantic relation between a question and a frame, and they are not action probabilities.

A low-risk research extension is an offline semantic reranker behind the existing retrieval interface: sample a fixed low-frequency frame index, compute image/text embeddings with a locally installed encoder, aggregate neighboring frame relevance into candidate windows, and preserve global uniform candidates. The reranker should return ranking metadata and frame hashes without changing the observation controller or provider budget.

Any encoder addition requires a pinned model and preprocessing version, verified weight hashes, dependency and hardware checks, and a separate offline evaluation. A similarity score is only a retrieval signal; it cannot establish that a brief action happened, preserve action order, or prove that the selected frames support an answer. Short events can be missed between indexed frames, while captioning or learned temporal grounding requires additional models, weights, and validation.

The public repository intentionally includes no private task identifiers, answers, event times, videos, or experiment traces. No semantic encoder or model weights are bundled.
