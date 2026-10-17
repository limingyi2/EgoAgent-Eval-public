"""Analyze saved, compatible experiment logs and render auditable reports."""

import argparse
import glob
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.reporting import generate_reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True, help="JSONL files or quoted glob patterns")
    parser.add_argument("--mode", required=True, choices=("mock", "real"))
    parser.add_argument("--run-id")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    paths = sorted({Path(path).resolve() for pattern in args.logs for path in glob.glob(pattern)})
    if not paths:
        parser.error("no log files matched")
    try:
        summary, archive, _ = generate_reports(paths, args.report_dir, mode=args.mode, run_id=args.run_id)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        parser.error(str(exc))
    print(f"{args.mode.upper()}: {summary}\nArchived report: {archive}")
    return summary


if __name__ == "__main__":
    main()
