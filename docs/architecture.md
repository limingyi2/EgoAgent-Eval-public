# Architecture overview

The package separates video access, retrieval, observation control, provider I/O, protocol validation, and experiment recording.

- `egoagent_eval/video.py` decodes frames and attaches timestamps and hashes.
- `egoagent_eval/index.py`, `retrieval.py`, and `sampling.py` produce and sample candidate temporal windows.
- `egoagent_eval/core/` and `harness.py` run the observation policy under explicit frame, tool, and request budgets.
- `egoagent_eval/vlm.py` adapts the provider response and records usage/errors. Real provider calls require an injected registered request gate; fake tests use in-memory transports.
- `egoagent_eval/protocol.py`, `schema.py`, and `schemas.py` validate structured actions and answers.
- `egoagent_eval/records.py` and `experiment_ledger.py` record execution artifacts and experiment state.
- `eval/` provides a separate runner/scoring/reporting pipeline with synthetic fixture support.

The architecture treats retrieval scores as candidate-selection signals, not proof that an action occurred. A model answer, timestamp match, and evidence sufficiency are separate evaluation outcomes. Evidence sufficiency requires review of the actual observed frames.

The public test suite is offline. It creates temporary synthetic media and uses `httpx.MockTransport`; it does not make external provider calls.
