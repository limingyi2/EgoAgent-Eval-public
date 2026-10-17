"""PowerShell-friendly standalone evaluation entry point."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data import Config, load_tasks
from eval.pipeline import run_experiment
from eval.runners import load_runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--video-root", type=Path, help="base for relative video paths; dataset format defaults to repository root")
    parser.add_argument("--fixture", action="store_true", help="built-in synthetic tasks; requires --mode mock --runner fake")
    parser.add_argument("--mode", required=True, choices=("mock", "real"))
    parser.add_argument("--runner", required=True, help="fake or module:factory")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tolerance-seconds", type=float, default=0.5)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    if args.fixture and (args.tasks is not None or args.mode != "mock" or args.runner != "fake"):
        parser.error("--fixture requires mock/fake and cannot be combined with --tasks")
    if not args.fixture and args.tasks is None:
        parser.error("specify --tasks or --fixture")
    try:
        config = Config(args.run_id, args.model, args.temperature, args.max_output_tokens, args.seed,
                        args.tolerance_seconds, args.input_price_per_million, args.output_price_per_million)
        runner = load_runner(args.runner)
        task_path = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "tasks.jsonl" if args.fixture else args.tasks
        tasks = load_tasks(task_path, mock=args.mode == "mock", video_root=args.video_root)
        output = run_experiment(tasks, config, runner, args.results_dir, mode=args.mode)
    except (ValueError, OSError, ImportError, AttributeError) as exc:
        parser.error(str(exc))
    print(f"{'MOCK / DRY RUN' if args.mode == 'mock' else 'REAL'}: {output}")
    return output


if __name__ == "__main__":
    main()
