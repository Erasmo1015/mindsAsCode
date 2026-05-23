#!/usr/bin/env python3
"""
Summarize dataset-4 (4wulff2018description) parsed trial counts by 50-participant
ordinal buckets, using TEH Psych-101 parser and within-participant split settings.

Usage:
  python analysis/code/psych-101/dataset4_trial_count_by_50.py

Outputs:
  analysis/data/psych101_dataset4_audit/trial_counts_by_50_participants.csv
  analysis/data/psych101_dataset4_audit/trial_counts_by_50_summary.csv
  analysis/data/psych101_dataset4_audit/trial_counts_by_50_report.txt
"""

from __future__ import annotations

import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    experiment_to_trial_dicts,
    get_filtered_psych101_split,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
    parse_psych101_binary_row,
    split_psych_experiment,
)

_DATASET = "4wulff2018description"
_PSYCH_SPLIT = DEFAULT_PSYCH_DATASET_SPLIT
_SPLIT_RATIO = 0.6
_SPLIT_SEED = 0
_BUCKET_SIZE = 50
_OUT_DIR = "analysis/data/psych101_dataset4_audit"
_MOST_THRESHOLD = 45  # of 50 participants for "most/all" bucket recommendations


def _repo_root() -> Path:
    return _REPO_ROOT


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _stats(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"min": "", "mean": "", "median": "", "max": ""}
    return {
        "min": min(values),
        "mean": round(statistics.mean(values), 4),
        "median": statistics.median(values),
        "max": max(values),
    }


def _participant_rows(
    alias: str,
    filtered,
    *,
    split_ratio: float,
    split_seed: int,
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for ordinal in range(len(filtered)):
        raw = dict(filtered[ordinal])
        hf_participant = raw.get("participant")
        split_error = ""
        n_blocks = n_trials = n_train = n_val = n_test = 0
        try:
            exp = parse_psych101_binary_row(raw, alias)
            n_blocks = len(exp.blocks)
            n_trials = sum(len(b.trials) for b in exp.blocks)
            _ = experiment_to_trial_dicts(exp, dataset_alias=alias)
            train_trials, val_trials, test_trials, _ = split_psych_experiment(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
            n_train = len(train_trials)
            n_val = len(val_trials)
            n_test = len(test_trials)
        except Exception as exc:
            split_error = f"{type(exc).__name__}: {exc}"
            try:
                exp = parse_psych101_binary_row(raw, alias)
                n_blocks = len(exp.blocks)
                n_trials = sum(len(b.trials) for b in exp.blocks)
            except Exception:
                pass

        rows_out.append(
            {
                "ordinal": ordinal,
                "teh_participant_id": ordinal,
                "hf_participant": hf_participant,
                "parsed_total_trials": n_trials,
                "train_trials": n_train,
                "val_trials": n_val,
                "test_trials": n_test,
                "n_blocks": n_blocks,
                "split_valid": split_error == "",
                "split_error": split_error,
            }
        )
    return rows_out


def _bucketize(
    participants: Sequence[Mapping[str, Any]], bucket_size: int
) -> List[Tuple[int, int, List[Mapping[str, Any]]]]:
    buckets: List[Tuple[int, int, List[Mapping[str, Any]]]] = []
    n = len(participants)
    for start in range(0, n, bucket_size):
        end = min(start + bucket_size - 1, n - 1)
        buckets.append((start, end, list(participants[start : end + 1])))
    return buckets


def _summarize_bucket(
    ordinal_start: int,
    ordinal_end: int,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    total_vals = [int(r["parsed_total_trials"]) for r in rows]
    train_vals = [int(r["train_trials"]) for r in rows]
    val_vals = [int(r["val_trials"]) for r in rows]
    test_vals = [int(r["test_trials"]) for r in rows]

    total_stats = _stats(total_vals)
    train_stats = _stats(train_vals)
    val_stats = _stats(val_vals)
    test_stats = _stats(test_vals)

    return {
        "ordinal_start": ordinal_start,
        "ordinal_end": ordinal_end,
        "n_participants": len(rows),
        "total_trials_min": total_stats["min"],
        "total_trials_mean": total_stats["mean"],
        "total_trials_median": total_stats["median"],
        "total_trials_max": total_stats["max"],
        "train_min": train_stats["min"],
        "train_mean": train_stats["mean"],
        "train_median": train_stats["median"],
        "train_max": train_stats["max"],
        "val_min": val_stats["min"],
        "val_mean": val_stats["mean"],
        "val_median": val_stats["median"],
        "val_max": val_stats["max"],
        "test_min": test_stats["min"],
        "test_mean": test_stats["mean"],
        "test_median": test_stats["median"],
        "test_max": test_stats["max"],
        "participants_with_train_1": sum(1 for v in train_vals if v == 1),
        "participants_with_train_lt_4": sum(1 for v in train_vals if v < 4),
        "participants_with_train_lt_6": sum(1 for v in train_vals if v < 6),
        "participants_with_train_lt_10": sum(1 for v in train_vals if v < 10),
        "participants_with_train_ge_4": sum(1 for v in train_vals if v >= 4),
        "participants_with_train_ge_6": sum(1 for v in train_vals if v >= 6),
        "participants_with_train_ge_10": sum(1 for v in train_vals if v >= 10),
        "n_split_valid": sum(1 for r in rows if r.get("split_valid")),
    }


def _bucket_label(start: int, end: int) -> str:
    return f"{start}-{end}"


def _print_terminal_table(summaries: Sequence[Mapping[str, Any]]) -> None:
    header = (
        "bucket | n | train_min | train_mean | train_median | train_max | "
        "train<4 | train>=6 | train>=10"
    )
    print()
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{_bucket_label(int(s['ordinal_start']), int(s['ordinal_end'])):>11} | "
            f"{int(s['n_participants']):3d} | "
            f"{s['train_min']:>9} | "
            f"{s['train_mean']:>10} | "
            f"{s['train_median']:>12} | "
            f"{s['train_max']:>9} | "
            f"{int(s['participants_with_train_lt_4']):>7} | "
            f"{int(s['participants_with_train_ge_6']):>8} | "
            f"{int(s['participants_with_train_ge_10']):>9}"
        )


def _bucket_score(s: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    """Higher is better for choosing a 50-participant subset."""
    n = int(s["n_participants"])
    if n < _BUCKET_SIZE:
        return (-1.0, -1.0, -1.0, -1.0)
    train_min = float(s["train_min"]) if s["train_min"] != "" else -1.0
    train_mean = float(s["train_mean"]) if s["train_mean"] != "" else -1.0
    ge10 = int(s["participants_with_train_ge_10"])
    ge4 = int(s["participants_with_train_ge_4"])
    return (train_min, train_mean, ge10 / n, ge4 / n)


def _recommendations(summaries: Sequence[Mapping[str, Any]]) -> List[str]:
    lines: List[str] = []
    full = [s for s in summaries if int(s["n_participants"]) == _BUCKET_SIZE]

    lines.append("RECOMMENDATIONS")
    lines.append("-" * 60)

    if not full:
        lines.append("No full 50-participant buckets found.")
        return lines

    # 1. Best 50-participant range with enough train trials
    ranked = sorted(full, key=_bucket_score, reverse=True)
    best = ranked[0]
    lines.append(
        "1. Best 50-participant ordinal range (enough train trials): "
        f"{_bucket_label(int(best['ordinal_start']), int(best['ordinal_end']))}"
    )
    lines.append(
        f"   train_min={best['train_min']}, train_mean={best['train_mean']}, "
        f"train_median={best['train_median']}, train_max={best['train_max']}; "
        f"train>=4: {best['participants_with_train_ge_4']}/50, "
        f"train>=10: {best['participants_with_train_ge_10']}/50"
    )

    # 2. First bucket where most participants have train >= 4
    ge4_bucket: Optional[Mapping[str, Any]] = None
    for s in full:
        if int(s["participants_with_train_ge_4"]) >= _MOST_THRESHOLD:
            ge4_bucket = s
            break
    lines.append("")
    if ge4_bucket is None:
        lines.append(
            f"2. First bucket with >={_MOST_THRESHOLD}/50 participants train>=4: "
            "none found in full buckets."
        )
    else:
        lines.append(
            f"2. First bucket with >={_MOST_THRESHOLD}/50 participants train>=4: "
            f"{_bucket_label(int(ge4_bucket['ordinal_start']), int(ge4_bucket['ordinal_end']))} "
            f"({ge4_bucket['participants_with_train_ge_4']}/50 train>=4, "
            f"train_min={ge4_bucket['train_min']}, train_mean={ge4_bucket['train_mean']})"
        )

    # 3. First bucket where most participants have train >= 10
    ge10_bucket: Optional[Mapping[str, Any]] = None
    for s in full:
        if int(s["participants_with_train_ge_10"]) >= _MOST_THRESHOLD:
            ge10_bucket = s
            break
    lines.append("")
    if ge10_bucket is None:
        lines.append(
            f"3. First bucket with >={_MOST_THRESHOLD}/50 participants train>=10: "
            "none found in full buckets."
        )
        # Report first bucket with any train>=10 majority
        alt = next(
            (s for s in full if int(s["participants_with_train_ge_10"]) >= 25),
            None,
        )
        if alt is not None:
            lines.append(
                f"   Closest: {_bucket_label(int(alt['ordinal_start']), int(alt['ordinal_end']))} "
                f"({alt['participants_with_train_ge_10']}/50 train>=10)"
            )
    else:
        lines.append(
            f"3. First bucket with >={_MOST_THRESHOLD}/50 participants train>=10: "
            f"{_bucket_label(int(ge10_bucket['ordinal_start']), int(ge10_bucket['ordinal_end']))} "
            f"({ge10_bucket['participants_with_train_ge_10']}/50 train>=10, "
            f"train_min={ge10_bucket['train_min']}, train_mean={ge10_bucket['train_mean']})"
        )

    return lines


def main() -> None:
    repo = _repo_root()
    out_dir = repo / _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    alias = normalize_psych101_dataset_alias(_DATASET)
    psych_split = normalize_psych_dataset_split(_PSYCH_SPLIT)
    filtered = get_filtered_psych101_split(alias, split=psych_split, local_dataset=None)

    participants = _participant_rows(
        alias,
        filtered,
        split_ratio=_SPLIT_RATIO,
        split_seed=_SPLIT_SEED,
    )
    buckets = _bucketize(participants, _BUCKET_SIZE)
    summaries = [_summarize_bucket(start, end, rows) for start, end, rows in buckets]

    participant_fields = [
        "ordinal",
        "teh_participant_id",
        "hf_participant",
        "parsed_total_trials",
        "train_trials",
        "val_trials",
        "test_trials",
        "n_blocks",
        "split_valid",
        "split_error",
    ]
    summary_fields = [
        "ordinal_start",
        "ordinal_end",
        "n_participants",
        "total_trials_min",
        "total_trials_mean",
        "total_trials_median",
        "total_trials_max",
        "train_min",
        "train_mean",
        "train_median",
        "train_max",
        "val_min",
        "val_mean",
        "val_median",
        "val_max",
        "test_min",
        "test_mean",
        "test_median",
        "test_max",
        "participants_with_train_1",
        "participants_with_train_lt_4",
        "participants_with_train_lt_6",
        "participants_with_train_lt_10",
        "participants_with_train_ge_4",
        "participants_with_train_ge_6",
        "participants_with_train_ge_10",
        "n_split_valid",
    ]

    participants_csv = out_dir / "trial_counts_by_50_participants.csv"
    summary_csv = out_dir / "trial_counts_by_50_summary.csv"
    report_path = out_dir / "trial_counts_by_50_report.txt"

    _write_csv(participants_csv, participants, participant_fields)
    _write_csv(summary_csv, summaries, summary_fields)

    lines: List[str] = [
        "DATASET 4 TRIAL COUNTS BY 50-PARTICIPANT ORDINAL BUCKETS",
        f"Generated: {datetime.now().isoformat()}",
        f"Repo: {repo}",
        f"Dataset: {alias}",
        f"psych_dataset_split: {psych_split}",
        f"split_mode: within_participant",
        f"split_ratio: {_SPLIT_RATIO}",
        f"split_seed: {_SPLIT_SEED}",
        "",
        f"Total filtered participants: {len(participants)}",
        f"Split-valid (>=3 blocks): {sum(1 for p in participants if p['split_valid'])}",
        f"Split-invalid: {sum(1 for p in participants if not p['split_valid'])}",
        "",
        "Terminal table:",
    ]
    for s in summaries:
        lines.append(
            f"  {_bucket_label(int(s['ordinal_start']), int(s['ordinal_end'])):>11} | "
            f"n={int(s['n_participants']):3d} | "
            f"train min/mean/med/max = "
            f"{s['train_min']}/{s['train_mean']}/{s['train_median']}/{s['train_max']} | "
            f"train<4={int(s['participants_with_train_lt_4'])} | "
            f"train>=6={int(s['participants_with_train_ge_6'])} | "
            f"train>=10={int(s['participants_with_train_ge_10'])}"
        )
    lines.append("")
    lines.extend(_recommendations(summaries))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Dataset: {alias} ({len(participants)} participants)")
    print(
        f"Split: within_participant, ratio={_SPLIT_RATIO}, seed={_SPLIT_SEED}, "
        f"psych_dataset_split={psych_split}"
    )
    _print_terminal_table(summaries)
    print()
    for line in _recommendations(summaries):
        print(line)
    print()
    print(f"Wrote {participants_csv.relative_to(repo)} ({len(participants)} rows)")
    print(f"Wrote {summary_csv.relative_to(repo)} ({len(summaries)} rows)")
    print(f"Wrote {report_path.relative_to(repo)}")


if __name__ == "__main__":
    main()
