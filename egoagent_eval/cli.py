"""PowerShell-compatible single-video experiments, with preserved per-run records."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

from dotenv import load_dotenv

from .agent import run_active
from .adaptive import run_adaptive_active
from .hierarchical import (run_hierarchical_active, run_hierarchical_active_v2,
                           run_hierarchical_active_v3_candidate_first,
                           run_hierarchical_active_v4_staged_observation)
from .direct import run_direct
from .experiment_ledger import ExperimentLedger, LedgerError, new_run_id
from .records import ArtifactStore, utc_now
from .schema import Budget, ErrorRecord, Question, RunResult
from .video import Video, file_sha256
from .vlm import FakeVLMProvider, MiMoVLMProvider


def configured_price(name):
    value = os.environ.get(name)
    return float(value) if value else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--method", choices=["direct4", "uniform8", "active", "adaptive", "hierarchical_active", "hierarchical_active_v2", "hierarchical_active_v3_candidate_first", "hierarchical_active_v4_staged_observation", "all"], default="all")
    parser.add_argument("--adaptive-context", choices=["full_context", "selective_context", "compact_context"],
                        default="selective_context")
    parser.add_argument("--provider", choices=["mimo", "fake"], default="mimo")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--experiment-ledger", type=Path,
                        help="Existing shared experiment request ledger (P2.8.23 budget profile)")
    parser.add_argument("--experiment-id", help="Immutable experiment identity stored in the request ledger")
    parser.add_argument("--snapshot-id", help="Frozen code snapshot ID, normally the snapshot manifest SHA-256")
    parser.add_argument("--task-id", help="Unique frozen task ID, such as task-a, task-b, or task-c")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="MiMo sampling temperature shared across experiment runs (default: 0)")
    parser.add_argument("--max-completion-tokens", type=int, default=2048,
                        help="Maximum completion tokens per request (default: 2048)")
    parser.add_argument("--candidate-sampling-policy",
                        choices=["uniform", "change_peaks", "hybrid_change"],
                        default="uniform",
                        help="Candidate-frame sampler for hierarchical_active_v3_candidate_first")
    parser.add_argument("--candidate-context-padding", type=float, default=0.5,
                        help="Change-window padding as a fraction of the original interval width (default: 0.5)")
    parser.add_argument("--local-index-samples", type=int, default=24,
                        help="OpenCV-only local scan points used by non-uniform candidate samplers (default: 24)")
    parser.add_argument("--v4-candidate-selection-policy",
                        choices=["baseline", "temporal_complementarity",
                                 "temporal_complementarity_ranked"], default="baseline",
                        help="v4 candidate policy; ranked adds deterministic Python ordering (default: baseline)")
    args = parser.parse_args(argv)
    load_dotenv(args.env_file, override=False)
    try:
        question = Question(video_path=str(args.video.resolve()), question=args.question)
        budget = Budget(max_image_side=args.max_image_side)
    except ValueError as exc:
        parser.error(str(exc))
    # Keep the historical --method all baseline unchanged; adaptive is opt-in.
    methods = ["direct4", "uniform8", "active"] if args.method == "all" else [args.method]
    ledger = None
    ledger_args = (args.experiment_ledger, args.experiment_id, args.snapshot_id, args.task_id)
    if any(ledger_args) and not all(ledger_args):
        parser.error("Ledger-backed runs require --experiment-ledger, --experiment-id, --snapshot-id, and --task-id together")
    if args.provider == "mimo" or args.experiment_ledger:
        if args.provider == "mimo" and args.method not in {"uniform8", "hierarchical_active_v4_staged_observation"}:
            parser.error("Real MiMo runs require exactly one guarded strategy: uniform8 or hierarchical_active_v4_staged_observation")
        if not all(ledger_args):
            parser.error("Real MiMo requires --experiment-ledger, --experiment-id, --snapshot-id, and --task-id")
        try:
            ledger = ExperimentLedger(args.experiment_ledger, experiment_id=args.experiment_id,
                                      snapshot_id=args.snapshot_id)
        except LedgerError as exc:
            parser.error(str(exc))
        experiment_root = ledger.path.parent
        output_root = args.output.resolve()
        try:
            relative_output = output_root.relative_to(experiment_root)
        except ValueError:
            parser.error("Ledger-backed run --output must be a new subdirectory inside the experiment ledger directory")
        if not relative_output.parts or output_root.exists():
            parser.error("Ledger-backed run --output must be a non-existing per-run directory inside the experiment directory")
    exit_code = 0
    for method in methods:
        started = perf_counter()
        run_id = new_run_id() if ledger else None
        if ledger:
            strategy = "uniform8" if method == "uniform8" else "v4_staged"
            try:
                ledger.register_run(strategy=strategy, task_id=args.task_id, run_id=run_id,
                                    video_sha256=file_sha256(question.video_path),
                                    model_id=(os.environ.get("MIMO_MODEL", "mimo-v2.6-flash").lower()
                                              if args.provider == "mimo" else FakeVLMProvider.model))
            except (LedgerError, OSError) as exc:
                parser.error(str(exc))
        store = ArtifactStore(args.output, args.provider == "fake", method, run_id=run_id)
        provider = None
        result_method = "adaptive_active" if method == "adaptive" else method
        try:
            if args.provider == "fake":
                provider = FakeVLMProvider(
                    store=store,
                    active_demo=method in ("active", "adaptive", "hierarchical_active", "hierarchical_active_v2",
                                           "hierarchical_active_v3_candidate_first", "hierarchical_active_v4_staged_observation"),
                    request_gate=ledger, run_id=store.run_id if ledger else None)
            else:
                provider = MiMoVLMProvider(
                    store=store, input_price=configured_price("MIMO_INPUT_USD_PER_MILLION"),
                    output_price=configured_price("MIMO_OUTPUT_USD_PER_MILLION"),
                    cached_price=configured_price("MIMO_CACHED_USD_PER_MILLION"),
                    temperature=args.temperature, max_completion_tokens=args.max_completion_tokens,
                    request_gate=ledger, run_id=store.run_id if ledger else None)
            with Video(question.video_path, budget.max_image_side) as video:
                # Deterministic same video, midpoint policy, decoder and JPEG settings for Direct/Active.
                initial = video.uniform_frames(8 if method == "uniform8" else 4)
                for frame in initial:
                    store.save_frame(frame)
                if method == "active":
                    result = run_active(question, video, provider, initial, budget, store)
                elif method == "adaptive":
                    result = run_adaptive_active(question, video, provider, initial, budget, store,
                                                 context_policy=args.adaptive_context)
                elif method == "hierarchical_active":
                    result = run_hierarchical_active(question, video, provider, initial, budget, store)
                elif method == "hierarchical_active_v2":
                    result = run_hierarchical_active_v2(question, video, provider, initial, budget, store)
                elif method == "hierarchical_active_v3_candidate_first":
                    result = run_hierarchical_active_v3_candidate_first(
                        question, video, provider, initial, budget, store,
                        candidate_sampling_policy=args.candidate_sampling_policy,
                        candidate_context_padding=args.candidate_context_padding,
                        local_index_samples=args.local_index_samples)
                elif method == "hierarchical_active_v4_staged_observation":
                    result = run_hierarchical_active_v4_staged_observation(
                        question, video, provider, initial, budget, store,
                        candidate_selection_policy=args.v4_candidate_selection_policy)
                else:
                    result = run_direct(question, video, provider, method, initial)
        except (ValueError, OSError) as exc:
            result = RunResult(answer="", evidence_timestamps=[], status="failed", method=result_method, question=question,
                               provider=provider.name if provider else args.provider,
                               model=provider.model if provider else os.environ.get("MIMO_MODEL", "mimo-v2.6-flash"),
                               is_fake=args.provider == "fake", started_at_utc=utc_now(),
                               calls=provider.calls if provider else [], errors=[ErrorRecord(code="setup_error", message=str(exc))])
        result.latency_seconds = perf_counter() - started
        store.write_json("result.json", result.model_dump())
        store.write_json("environment.json", {"python": platform.python_version(),
                         "packages": {name: importlib.metadata.version(name) for name in ("opencv-python", "pydantic", "httpx", "python-dotenv")},
                         "budget": budget.model_dump(), "frame_sampling": "uniform bin midpoints; frame index=floor(t*fps)",
                         "argv": sys.argv if argv is None else argv,
                         "model": provider.model if provider else None,
                         "base_url": getattr(provider, "base_url", None),
                         "timeout_seconds": getattr(provider, "timeout", None),
                         "adaptive_context": args.adaptive_context if method == "adaptive" else None,
                         "v4_candidate_selection_policy": (args.v4_candidate_selection_policy
                                                            if method == "hierarchical_active_v4_staged_observation"
                                                            else None),
                         "sampling": {"temperature": args.temperature,
                                      "max_completion_tokens": args.max_completion_tokens},
                         "cost": "Configured USD/token estimate; null when unknown. Not an invoice."})
        if ledger:
            ledger.complete_run(store.run_id, result)
        print(json.dumps({"method": method, "answer": result.answer, "evidence_timestamps": result.evidence_timestamps,
                          "status": result.status, "is_fake": result.is_fake, "correct": result.correct,
                          "initial_frames": len(result.initial_frames), "observed_frames": len(result.observed_frames),
                          "model_calls": len(result.calls), "tool_calls": len(result.tool_calls),
                          "artifact_dir": str(store.path), "errors": [e.model_dump() for e in result.errors]}, ensure_ascii=False))
        if result.status != "ok":
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
