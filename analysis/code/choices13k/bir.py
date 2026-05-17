#!/usr/bin/env python3
"""Compute Behavioral Inconsistency Rate (BIR) per participant and train/val/test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.code.choices13k.one_phase.val import (  # noqa: E402
    DEFAULT_LOCAL_DATASET,
    split_trials_te_dr,
)

DEFAULT_OUTPUT_CSV = REPO_ROOT / "analysis/data/choices13k/bir_by_split.csv"
DEFAULT_SPLIT_RATIO = 0.6
DEFAULT_SPLIT_SEED = 0

CSV_FIELDS = [
    "participant_id",
    "train_BIR",
    "val_BIR",
    "test_BIR",
    "train_val_BIR",
    "all_BIR",
    "train_num_problem_groups",
    "val_num_problem_groups",
    "test_num_problem_groups",
    "train_val_num_problem_groups",
    "all_num_problem_groups",
    "train_num_inconsistent_problem_groups",
    "val_num_inconsistent_problem_groups",
    "test_num_inconsistent_problem_groups",
    "train_val_num_inconsistent_problem_groups",
    "all_num_inconsistent_problem_groups",
]


def make_problem_key(problem: Dict[str, Any]) -> str:
    """Deterministic key for a Choice13k gamble problem (matches te_dr.make_problem_key)."""
    payload = {
        "gamble_A": {
            "probs": problem["gamble_A"].get("probs"),
            "rewards": problem["gamble_A"].get("rewards"),
        },
        "gamble_B": {
            "probs": problem["gamble_B"].get("probs"),
            "rewards": problem["gamble_B"].get("rewards"),
        },
        "option_keys": problem.get("option_keys"),
        "has_feedback": bool(problem.get("has_feedback", False)),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def compute_bir(trials: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    """
    BIR over trials: inconsistent groups / total groups.

    A problem group is inconsistent if both actions 0 and 1 appear.
    Returns (bir, num_problem_groups, num_inconsistent_problem_groups).
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for trial in trials:
        prob = trial["problem"]
        if isinstance(prob, dict) and "gamble_A" in prob and "gamble_B" in prob:
            key = make_problem_key(prob)
        else:
            key = json.dumps(prob, sort_keys=True, default=str)
        grouped.setdefault(key, []).append(trial)

    num_groups = len(grouped)
    if num_groups == 0:
        return 0.0, 0, 0

    inconsistent = 0
    for group in grouped.values():
        actions = {int(t["action"]) for t in group}
        if 0 in actions and 1 in actions:
            inconsistent += 1

    bir = float(inconsistent) / float(num_groups)
    return bir, num_groups, inconsistent


def _mean_finite(vals: List[float]) -> float:
    good = [v for v in vals if np.isfinite(v)]
    return float(np.mean(good)) if good else float("nan")


def _bir_row(participant_id: int, splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    train_val = train + val
    all_trials = train + val + test

    metrics: Dict[str, Tuple[float, int, int]] = {}
    for name, trial_list in (
        ("train", train),
        ("val", val),
        ("test", test),
        ("train_val", train_val),
        ("all", all_trials),
    ):
        metrics[name] = compute_bir(trial_list)

    row: Dict[str, Any] = {"participant_id": participant_id}
    for name in ("train", "val", "test", "train_val", "all"):
        bir, n_groups, n_incon = metrics[name]
        row[f"{name}_BIR"] = round(bir, 4)
        row[f"{name}_num_problem_groups"] = n_groups
        row[f"{name}_num_inconsistent_problem_groups"] = n_incon
    return row


def _print_table(rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    col_w = [6] + [10] * (len(fieldnames) - 1)
    sep = "  "
    header = sep.join(
        f"{name:>{col_w[i] if i else 6}}" if i else f"{name:<6}"
        for i, name in enumerate(fieldnames)
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        parts = []
        for i, name in enumerate(fieldnames):
            val = row.get(name, "")
            if i == 0:
                parts.append(f"{str(val):<6}")
            elif isinstance(val, float):
                parts.append(f"{val:>{col_w[i]}.4f}")
            else:
                parts.append(f"{str(val):>{col_w[i]}}")
        print(sep.join(parts))


def run(
    output_csv: Path,
    n_participants: int,
    split_ratio: float,
    split_seed: int,
    local_dataset: Path | None,
) -> None:
    from data_modules.choice13k import get_choice13k_experiments

    local = str(local_dataset) if local_dataset is not None else None
    experiments = get_choice13k_experiments(n_participants=n_participants, local_dataset=local)

    rows: List[Dict[str, Any]] = []
    for pid, exp in enumerate(experiments):
        train_trials, val_trials, test_trials = split_trials_te_dr(
            exp, split_ratio=split_ratio, split_seed=split_seed
        )
        rows.append(
            _bir_row(
                pid,
                {"train": train_trials, "val": val_trials, "test": test_trials},
            )
        )

    avg_row: Dict[str, Any] = {"participant_id": "avg"}
    for name in ("train", "val", "test", "train_val", "all"):
        avg_row[f"{name}_BIR"] = round(
            _mean_finite([float(r[f"{name}_BIR"]) for r in rows]), 4
        )
        avg_row[f"{name}_num_problem_groups"] = int(
            round(np.mean([r[f"{name}_num_problem_groups"] for r in rows]))
        )
        avg_row[f"{name}_num_inconsistent_problem_groups"] = int(
            round(np.mean([r[f"{name}_num_inconsistent_problem_groups"] for r in rows]))
        )
    rows.append(avg_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(
        f"BIR by split (split_ratio={split_ratio}, split_seed={split_seed}, "
        f"n_participants={len(experiments)})\n"
    )
    _print_table(rows, CSV_FIELDS)
    print(f"\nWrote {len(rows)} rows (incl. avg) -> {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV.relative_to(REPO_ROOT)})",
    )
    parser.add_argument("--n-participants", type=int, default=10)
    parser.add_argument("--split-ratio", type=float, default=DEFAULT_SPLIT_RATIO)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--local-dataset", type=Path, default=DEFAULT_LOCAL_DATASET)
    args = parser.parse_args()

    output_csv = args.output_csv
    if not output_csv.is_absolute():
        output_csv = REPO_ROOT / output_csv

    local = args.local_dataset
    if local is not None:
        local = local if local.is_absolute() else REPO_ROOT / local
        if not local.exists():
            print(f"Warning: local dataset not found at {local}; will try HF download.")
            local = None

    run(
        output_csv=output_csv,
        n_participants=args.n_participants,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        local_dataset=local,
    )


if __name__ == "__main__":
    main()
