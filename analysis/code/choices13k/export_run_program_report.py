#!/usr/bin/env python3
"""
Load participant log-likelihood CSV from a choice13k evolution run and print
bucketed summaries for proposal / inspection workflows.

Usage:
  python export_run_program_report.py /path/to/run_YYYYMMDD_HHMMSS
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(run_dir: Path) -> list[dict[str, str]]:
    csv_path = run_dir / "participant_details_loglik.csv"
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Path to run folder (contains participant_details_loglik.csv)")
    args = p.parse_args()
    run_dir: Path = args.run_dir.expanduser().resolve()
    rows = load_rows(run_dir)
    rnd = -0.6931471805599453  # log(0.5) for Bernoulli coin flip

    def fkey(r: dict[str, str]) -> tuple[float, float, int]:
        tid = int(r["participant_id"])
        tr = float(r["train_loglik"])
        te = float(r["test_loglik"])
        return (-te, -tr, tid)  # sort by test desc, then train desc

    rows_sorted = sorted(rows, key=fkey)

    print(f"Run: {run_dir}")
    print(f"Participants: {len(rows)}")
    print()
    print("--- Sorted by test loglik (best first) ---")
    for r in rows_sorted:
        print(
            f"  id={r['participant_id']:>3}  train={r['train_loglik']:>8}  test={r['test_loglik']:>8}"
        )

    print()
    print("--- Near random (test within 1e-3 of log(0.5)) ---")
    for r in rows:
        te = float(r["test_loglik"])
        if abs(te - rnd) < 1e-3:
            print(f"  id={r['participant_id']}  train={r['train_loglik']}  test={r['test_loglik']}")

    print()
    print("--- Possible overfit (train much better than test; heuristic: train - test > 0.25 nats) ---")
    for r in rows:
        tr, te = float(r["train_loglik"]), float(r["test_loglik"])
        gap = tr - te  # positive => train fit better than test
        if gap > 0.25:
            print(f"  id={r['participant_id']}  train={tr:.4f}  test={te:.4f}  gap(train-test)={gap:.4f}")

    print()
    print("Best program glob (first match per participant):")
    for r in rows:
        pid = r["participant_id"]
        d = run_dir / f"participant_{pid}"
        matches = sorted(d.glob("best_program*.py"))
        rel = matches[0].relative_to(run_dir) if matches else "(none)"
        print(f"  participant_{pid}: {rel}")


if __name__ == "__main__":
    main()
