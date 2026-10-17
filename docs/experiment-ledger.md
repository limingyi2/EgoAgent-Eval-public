# Experiment ledger contract

An experiment run has a stable experiment identifier, snapshot identifier, strategy, task identifier, and unique run identifier. The request ledger is authoritative for provider attempts; the result table is a separate analysis view and must not initiate or replay requests.

The run status is one of `success`, `failed`, `incomplete`, or `unknown`. Answer status is one of `correct`, `incorrect`, `refused`, `protocol_error`, `unscorable`, or `unknown`. Evidence status is one of `sufficient`, `insufficient`, `manual_review_required`, or `unknown`.

Each real provider attempt must be reserved before network I/O. Reservations consume budget even if a request fails, times out, returns a non-success HTTP status, or has an unknown outcome. Repeated runs must have new run identifiers and may not reset the ledger. No automatic retry or result-write replay is permitted. Credentials must never be recorded in the ledger or request trace.

Answer correctness and visual evidence sufficiency are independent fields. A correct answer does not automatically establish sufficient evidence. Unknown values remain unknown until a reliable record or authorized review resolves them.
