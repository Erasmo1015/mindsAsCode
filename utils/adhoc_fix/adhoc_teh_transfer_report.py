#!/usr/bin/env python3
"""
Backfill ``summary_csv/test_loglik.csv`` with ``transfer_1st`` for a TEH transfer run.

The iteration-1 transfer test loglik is not stored in the original run summary CSV.
This script re-evaluates the saved iteration-1 pool-best candidate (or reads a cached
value from ``transfer/results.json``) and writes the updated CSV.

Usage:
  python utils/adhoc_fix/adhoc_teh_transfer_report.py \\
    --run_dir generated_outputs_transfer/teh_transfer/run_260604_094117

  # Inspect current state without writing:
  python utils/adhoc_fix/adhoc_teh_transfer_report.py \\
    --run_dir generated_outputs_transfer/teh_transfer/run_260604_094117 --status

  # Backfill one dataset at a time (merges into existing CSV):
  python utils/adhoc_fix/adhoc_teh_transfer_report.py \\
    --run_dir generated_outputs_transfer/teh_transfer/run_260604_094117 \\
    --datasets 11enkavi2019recentprobes
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.teh_transfer.evolution import (  # noqa: E402
    _read_phase_avg_test_loglik,
    _read_run_test_loglik_rows,
    backfill_run_test_loglik_summary_csv,
)


def _load_run_datasets(run_dir: Path) -> List[Dict[str, Any]]:
    config_path = run_dir / "transfer_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing transfer config: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return list(payload.get("datasets") or [])


def _cached_transfer_1st_test(transfer_dir: Path) -> Optional[float]:
    results_path = transfer_dir / "results.json"
    if not results_path.is_file():
        return None
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    cached = payload.get("first_iteration_best_test_loglik")
    if cached in (None, ""):
        return None
    return float(cached)


def _csv_has_transfer_1st_column(csv_path: Path) -> bool:
    if not csv_path.is_file():
        return False
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    return header is not None and "transfer_1st" in header


def print_run_status(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    csv_path = run_dir / "summary_csv" / "test_loglik.csv"
    datasets = _load_run_datasets(run_dir)

    print(f"Run directory: {run_dir}")
    print(f"test_loglik.csv: {csv_path}")
    print(
        "CSV has transfer_1st column:"
        f" {'yes' if _csv_has_transfer_1st_column(csv_path) else 'no'}"
    )
    print("")
    print("dataset,cached_transfer_1st_test,csv_transfer_1st")
    existing_rows = {str(r.get("dataset", "")): r for r in _read_run_test_loglik_rows(csv_path)}
    for entry in datasets:
        config_key = str(entry["config_key"])
        transfer_dir = run_dir / config_key / "transfer"
        cached = _cached_transfer_1st_test(transfer_dir) if transfer_dir.is_dir() else None
        csv_val = (existing_rows.get(config_key) or {}).get("transfer_1st", "")
        cached_str = f"{cached:.6f}" if cached is not None else ""
        print(f"{config_key},{cached_str},{csv_val}")


def _backup_csv(csv_path: Path) -> Optional[Path]:
    if not csv_path.is_file():
        return None
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    backup_path = csv_path.with_name(f"{csv_path.stem}.bak_{ts}{csv_path.suffix}")
    shutil.copy2(csv_path, backup_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill transfer_1st test loglik into summary_csv/test_loglik.csv."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Transfer run folder, e.g. generated_outputs_transfer/teh_transfer/run_260604_094117",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional config_key subset to backfill (merges into existing CSV).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current backfill status and exit without writing.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup of the existing test_loglik.csv.",
    )
    parser.add_argument(
        "--no-results-cache",
        action="store_true",
        help="Do not write first_iteration_best_test_loglik into transfer/results.json.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (_REPO_ROOT / run_dir).resolve()
    else:
        run_dir = run_dir.resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    if args.status:
        print_run_status(run_dir)
        return

    csv_path = run_dir / "summary_csv" / "test_loglik.csv"
    if not args.no_backup:
        backup_path = _backup_csv(csv_path)
        if backup_path is not None:
            print(f"[backup] {backup_path}")

    out_path = backfill_run_test_loglik_summary_csv(
        run_dir,
        datasets=args.datasets,
        verbose=True,
        write_results_cache=not args.no_results_cache,
    )
    print(f"[OK] Wrote {out_path}")
    print("")
    print_run_status(run_dir)


if __name__ == "__main__":
    main()
