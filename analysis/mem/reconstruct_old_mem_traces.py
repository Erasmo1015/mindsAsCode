#!/usr/bin/env python3
"""Reconstruct synthetic mem_trace.jsonl from old (pre--mem_trace) TEH/PICS runs.

Writes traces compatible with analysis/mem/annotate_edits.py → build_dataset.py → fit_mem.py.
Does not modify or rerun teh.py. Rescores initial_pool_from_global on each participant's
train/val split (cached) using the same selection-score formula as TEH.

Example (write under careAIDrive, one participant smoke test):

  python analysis/mem/reconstruct_old_mem_traces.py \\
    --run_dir generated_outputs_old/psych101_train/teh/1peterson2021using/run_260706_211034 \\
    --output_run_dir analysis_2026Sep/mem/old_run_traces/run_260706_211034 \\
    --participants 0 \\
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mem.reconstruct_old_run import (  # noqa: E402
    reconstruct_run,
    validate_artifacts_for_reconstruction,
)


def _parse_participants(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True, help="Old TEH run directory")
    parser.add_argument(
        "--output_run_dir",
        type=str,
        default=None,
        help="If set, write participant_*/mem_trace.jsonl here instead of in-place",
    )
    parser.add_argument(
        "--participants",
        type=str,
        default=None,
        help="Comma list / ranges, e.g. 0 or 0,1,2 or 0-5. Default: all.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Override; default: parse log/command.txt (TEH default 0)",
    )
    parser.add_argument("--split_ratio", type=float, default=None)
    parser.add_argument("--n_eval_seeds", type=int, default=None)
    parser.add_argument("--psych_dataset_split", type=str, default=None)
    parser.add_argument("--evolution_selection_score", type=str, default=None)
    parser.add_argument(
        "--elite_pool_size",
        type=int,
        default=None,
        help="Override; default: parse log/command.txt or 50",
    )
    parser.add_argument(
        "--rescore_cache_dir",
        type=str,
        default=None,
        help="Directory for initial_pool participant score caches (default: under output_run_dir)",
    )
    parser.add_argument(
        "--force_rescore",
        action="store_true",
        help="Ignore existing rescore caches and recompute",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only print artifact readiness report (no writes)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (_REPO_ROOT / run_dir).resolve()

    report = validate_artifacts_for_reconstruction(run_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.inspect_only:
        raise SystemExit(0 if report.get("ok") else 1)
    if not report.get("ok"):
        raise SystemExit("Artifacts incomplete; refusing to reconstruct.")

    out_run = Path(args.output_run_dir) if args.output_run_dir else None
    if out_run is not None and not out_run.is_absolute():
        out_run = (_REPO_ROOT / out_run).resolve()
    cache_dir = Path(args.rescore_cache_dir) if args.rescore_cache_dir else None
    if cache_dir is not None and not cache_dir.is_absolute():
        cache_dir = (_REPO_ROOT / cache_dir).resolve()

    stats = reconstruct_run(
        run_dir,
        dataset=args.dataset,
        split_seed=args.split_seed,
        split_ratio=args.split_ratio,
        n_eval_seeds=args.n_eval_seeds,
        psych_dataset_split=args.psych_dataset_split,
        evolution_selection_score=args.evolution_selection_score,
        elite_pool_size=args.elite_pool_size,
        participant_ids=_parse_participants(args.participants),
        output_run_dir=out_run,
        rescore_cache_dir=cache_dir,
        force_rescore=bool(args.force_rescore),
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "participants": stats.participants,
                "iterations": stats.iterations,
                "candidates_written": stats.candidates_written,
                "runtime_valid_written": stats.runtime_valid_written,
                "fresh_runtime_valid_written": stats.fresh_runtime_valid_written,
                "fresh_added_to_pool": stats.fresh_added_to_pool,
                "initial_pool_rescored": stats.initial_pool_rescored,
                "skipped_missing_code": stats.skipped_missing_code,
                "warnings": stats.warnings,
                "output_run_dir": str(out_run) if out_run else str(run_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
