#!/usr/bin/env python3
"""
Export last-iteration log-likelihood summaries from a Choice13k run.

This reads each participant directory under a run folder, finds the highest
iteration_* directory, and records that iteration's best_train_loglik /
best_test_loglik values from metrics.json.

Outputs:
  - participant_details_loglik_last_iter.csv
  - summary_loglik_last_iter.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional, Tuple


def _parse_iteration_dir(path: Path) -> Optional[int]:
    match = re.fullmatch(r"iteration_(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _load_last_iter_metrics(participant_dir: Path) -> Tuple[Optional[int], Optional[dict]]:
    iter_dirs = []
    for cand in participant_dir.iterdir():
        if not cand.is_dir():
            continue
        iter_id = _parse_iteration_dir(cand)
        if iter_id is not None:
            iter_dirs.append((iter_id, cand))
    if not iter_dirs:
        return None, None
    iter_dirs.sort(key=lambda x: x[0])
    last_iter, last_dir = iter_dirs[-1]
    metrics_path = last_dir / "metrics.json"
    if not metrics_path.is_file():
        return last_iter, None
    with metrics_path.open(encoding="utf-8") as f:
        return last_iter, json.load(f)


def _round_opt(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Path to run folder")
    args = parser.parse_args()
    run_dir: Path = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    rows = []
    for participant_dir in sorted(run_dir.glob("participant_*")):
        if not participant_dir.is_dir():
            continue
        try:
            participant_id = int(participant_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        last_iter, metrics = _load_last_iter_metrics(participant_dir)
        if metrics is None:
            rows.append(
                {
                    "participant_id": participant_id,
                    "train_loglik": None,
                    "test_loglik": None,
                }
            )
            continue
        rows.append(
            {
                "participant_id": participant_id,
                "train_loglik": _round_opt(metrics.get("best_train_loglik")),
                "test_loglik": _round_opt(metrics.get("best_test_loglik")),
            }
        )

    details_path = run_dir / "participant_details_loglik_last_iter.csv"
    with details_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_id", "train_loglik", "test_loglik"])
        writer.writeheader()
        writer.writerows(rows)

    train_vals = [r["train_loglik"] for r in rows if r["train_loglik"] is not None]
    test_vals = [r["test_loglik"] for r in rows if r["test_loglik"] is not None]
    avg_train = (sum(train_vals) / len(train_vals)) if train_vals else None
    avg_test = (sum(test_vals) / len(test_vals)) if test_vals else None
    summary_path = run_dir / "summary_loglik_last_iter.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["num_of_participants", "avg_train_loglik", "avg_test_loglik"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "num_of_participants": len(rows),
                "avg_train_loglik": _round_opt(avg_train),
                "avg_test_loglik": _round_opt(avg_test),
            }
        )

    print(f"Wrote: {details_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
