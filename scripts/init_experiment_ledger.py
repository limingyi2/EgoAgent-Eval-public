"""Create a new immutable P2.8.21 experiment ledger (never overwrites)."""

import argparse
from pathlib import Path

from egoagent_eval.experiment_ledger import ExperimentLedger, LedgerError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True,
                        help="New SQLite file, usually <experiment-dir>/request_ledger.sqlite3")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--snapshot-id", required=True,
                        help="Frozen snapshot_manifest.csv SHA-256")
    args = parser.parse_args(argv)
    try:
        ledger = ExperimentLedger.initialize(args.ledger, experiment_id=args.experiment_id,
                                             snapshot_id=args.snapshot_id)
    except (LedgerError, OSError) as exc:
        parser.error(str(exc))
    print(f"Initialized experiment {ledger.experiment_id}; ledger={ledger.path}; results={ledger.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
