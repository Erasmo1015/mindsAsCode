"""Shared I/O and CLI helpers for teh_psych baselines."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def round_floats_for_csv_row(row: Dict[str, Any], ndigits: int = 4) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (float, np.floating)):
            x = float(v)
            out[k] = round(x, ndigits) if math.isfinite(x) else x
        else:
            out[k] = v
    return out


def _is_finite_number(v: Any) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def finite_mean(vals: Sequence[Any]) -> Optional[float]:
    xs = [float(v) for v in vals if _is_finite_number(v)]
    return float(np.mean(xs)) if xs else None


def trial_weighted_mean_loglik(
    participant_results: Sequence[Dict[str, Any]],
    *,
    split: str,
) -> Optional[float]:
    """
    Mean loglik over trials in ``split`` (train|val|test), weighting each
    participant by ``n_{split}``. Skips empty splits (avoids nan poisoning).
    """
    key_ll = f"{split}_mean_loglik"
    key_n = f"n_{split}"
    num = 0.0
    den = 0
    for r in participant_results:
        n = int(r.get(key_n) or 0)
        ll = r.get(key_ll)
        if n > 0 and _is_finite_number(ll):
            num += float(ll) * n
            den += n
    return (num / den) if den else None


def write_experiment_loglik_csvs(
    base_run_dir: Path,
    participant_details_loglik: List[Dict[str, Any]],
) -> None:
    """TEH-compatible participant_details_loglik.csv + summary.csv (4 dp)."""
    base = Path(base_run_dir)
    base.mkdir(parents=True, exist_ok=True)
    details_fields = ["participant_id", "train_loglik", "val_loglik", "test_loglik"]
    with (base / "participant_details_loglik.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            [round_floats_for_csv_row(r) for r in participant_details_loglik]
        )

    # Skip non-finite (empty split → nan) so summary.csv is usable.
    train_vals = [d["train_loglik"] for d in participant_details_loglik]
    val_vals = [d["val_loglik"] for d in participant_details_loglik]
    test_vals = [d["test_loglik"] for d in participant_details_loglik]
    summary_row = {
        "num_of_participants": len(participant_details_loglik),
        "avg_train_loglik": finite_mean(train_vals),
        "avg_test_loglik": finite_mean(test_vals),
        "avg_val_loglik": finite_mean(val_vals),
        "avg_gated_test_loglik": None,
    }
    summary_fields = [
        "num_of_participants",
        "avg_train_loglik",
        "avg_test_loglik",
        "avg_val_loglik",
        "avg_gated_test_loglik",
    ]
    with (base / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(round_floats_for_csv_row(summary_row))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


# Columns aligned with teh_psych prototype_summary/dataset_results.csv for easy joins.
TEH_PSYCH_COMPARABLE_CSV_COLUMNS = [
    "experiment_id",
    "method",
    "status",
    "n_rows_used",
    "n_prediction_trials",
    "n_train",
    "n_val",
    "n_test",
    "num_actions_min",
    "num_actions_max",
    "is_variable_k",
    "train_loglik",
    "val_loglik",
    "test_loglik",
    "parse_plan_path",
    "parse_plan_sha256",
    "num_participants_fit",
    "notes",
]


def summary_to_comparable_row(summary: Dict[str, Any], *, method: str) -> Dict[str, Any]:
    """Map baseline experiment summary → teh_psych-comparable CSV row."""
    status = str(summary.get("status") or "unknown")
    notes = summary.get("prospect_support_reason") or summary.get("error") or summary.get("notes") or ""
    return {
        "experiment_id": summary.get("experiment_id", ""),
        "method": method,
        "status": status,
        "n_rows_used": summary.get("n_rows_used", ""),
        "n_prediction_trials": summary.get("n_prediction_trials", ""),
        "n_train": summary.get("n_train", ""),
        "n_val": summary.get("n_val", ""),
        "n_test": summary.get("n_test", ""),
        "num_actions_min": summary.get("num_actions_min", ""),
        "num_actions_max": summary.get("num_actions_max", ""),
        "is_variable_k": summary.get("is_variable_k", ""),
        # Same names as teh_psych dataset_results.csv
        "train_loglik": summary.get("avg_train_loglik"),
        "val_loglik": summary.get("avg_val_loglik"),
        "test_loglik": summary.get("avg_test_loglik"),
        "parse_plan_path": summary.get("parse_plan_path", ""),
        "parse_plan_sha256": summary.get("parse_plan_sha256", ""),
        "num_participants_fit": summary.get("num_participants_fit", ""),
        "notes": notes,
    }


def write_comparable_dataset_results_csv(
    out_root: Path,
    summaries: Sequence[Dict[str, Any]],
    *,
    method: str,
) -> Path:
    """
    Write run-level dataset_results.csv joinable with teh_psych
    ``prototype_summary/dataset_results.csv`` on ``experiment_id``.
    """
    out_root = Path(out_root)
    path = out_root / "dataset_results.csv"
    rows = [summary_to_comparable_row(s, method=method) for s in summaries]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TEH_PSYCH_COMPARABLE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows([round_floats_for_csv_row(r) for r in rows])

    # Slim table for quick LL comparison.
    slim_path = out_root / "loglik_comparison.csv"
    slim_fields = ["experiment_id", "method", "status", "train_loglik", "val_loglik", "test_loglik"]
    with slim_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=slim_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(round_floats_for_csv_row({k: r.get(k) for k in slim_fields}))
    return path


def parse_experiment_ids(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "" or str(raw).strip().lower() == "all":
        return None
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def add_teh_psych_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment_ids",
        type=str,
        default=None,
        help="Comma-separated HF experiment ids (default: discover from split).",
    )
    parser.add_argument("--psych_dataset_split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--local_dataset", type=str, default=None)
    parser.add_argument("--max_experiments", type=str, default="all")
    parser.add_argument("--max_participants_per_experiment", type=int, default=50)
    parser.add_argument("--range_start_ordinal", type=int, default=None)
    parser.add_argument("--range_end_ordinal", type=int, default=None)
    parser.add_argument("--min_pooled_prediction_trials", type=int, default=50)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--reuse_parse_plan_cache",
        action="store_true",
        default=False,
        help="Load parse plans from --parse_plan_cache_dir (required for these baselines).",
    )
    parser.add_argument(
        "--parse_plan_cache_dir",
        type=str,
        default=None,
        help="Directory containing <safe_id>/parse_plan.json entries.",
    )
    parser.add_argument(
        "--require_cached_parse_plan",
        action="store_true",
        default=False,
        help="Fail loudly if a plan is missing/invalid (never call the LLM).",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fitness_metric", type=str, default="loglik", choices=["loglik", "accuracy"])
    parser.add_argument(
        "--show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm bars for parse/fit stages (default: on).",
    )


def participant_key(trial: Dict[str, Any]) -> str:
    meta = trial.get("_meta") or {}
    pid = meta.get("participant")
    if pid is None:
        pid = meta.get("row_index")
    return str(pid)


def group_trials_by_participant(
    trials: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for t in trials:
        out.setdefault(participant_key(t), []).append(t)
    return out
