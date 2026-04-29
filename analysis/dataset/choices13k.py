#!/usr/bin/env python3
"""
Inspect Choice13k train/test trial counts under OpenEvolve split logic.

Default behavior mirrors current OpenEvolve configuration:
- HuggingFace dataset: marcelbinz/Psych-101-test
- Experiment filter: peterson2021using/exp1.csv
- Split ratio: 0.9
- Split seed: 0
- Ordinal range: 0-9 from datasets/choice13k/valid_participant_ids.json

Example:
  python analysis/dataset/choices13k.py
  python analysis/dataset/choices13k.py --hf_dataset marcelbinz/Psych-101 --analysis_split train
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _hf_token_for_datasets() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


@dataclass
class ParticipantTrialStats:
    split_name: str
    ordinal: int
    hf_row_id: int
    raw_participant_id: int
    n_blocks: int
    total_trials: int
    train_blocks: int
    test_blocks: int
    train_trials: int
    test_trials: int


def _split_block_indices(n_blocks: int, split_ratio: float, split_seed: int) -> Tuple[set[int], set[int]]:
    if n_blocks < 2:
        raise ValueError(f"Need at least 2 blocks; got {n_blocks}")
    rng = np.random.default_rng(int(split_seed))
    perm = np.arange(n_blocks)
    rng.shuffle(perm)
    split_idx = int(n_blocks * float(split_ratio))
    split_idx = max(1, min(split_idx, n_blocks - 1))
    train_blocks = set(perm[:split_idx].tolist())
    test_blocks = set(perm[split_idx:].tolist())
    return train_blocks, test_blocks


def _participant_stats(
    *,
    row: Dict,
    split_name: str,
    ordinal: int,
    hf_row_id: int,
    split_ratio: float,
    split_seed: int,
) -> ParticipantTrialStats:
    from data_modules.choice13k import _convert_to_experiment

    exp = _convert_to_experiment(row)
    n_blocks = len(exp.blocks)
    train_blocks, test_blocks = _split_block_indices(n_blocks, split_ratio, split_seed)
    train_trials = sum(len(exp.blocks[i].trials) for i in train_blocks)
    test_trials = sum(len(exp.blocks[i].trials) for i in test_blocks)
    total_trials = train_trials + test_trials
    return ParticipantTrialStats(
        split_name=split_name,
        ordinal=int(ordinal),
        hf_row_id=int(hf_row_id),
        raw_participant_id=int(row["participant"]),
        n_blocks=int(n_blocks),
        total_trials=int(total_trials),
        train_blocks=int(len(train_blocks)),
        test_blocks=int(len(test_blocks)),
        train_trials=int(train_trials),
        test_trials=int(test_trials),
    )


def _fmt_row(items: Sequence[object], widths: Sequence[int]) -> str:
    return " | ".join(str(v).ljust(w) for v, w in zip(items, widths))


def _print_table(rows: List[ParticipantTrialStats], title: str) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no rows)")
        return
    header = [
        "ordinal",
        "hf_row_id",
        "raw_pid",
        "blocks",
        "total_trials",
        "train_trials",
        "test_trials",
        "train_blocks",
        "test_blocks",
    ]
    body = [
        [
            r.ordinal,
            r.hf_row_id,
            r.raw_participant_id,
            r.n_blocks,
            r.total_trials,
            r.train_trials,
            r.test_trials,
            r.train_blocks,
            r.test_blocks,
        ]
        for r in rows
    ]
    widths = [max(len(str(x)) for x in [h] + [b[i] for b in body]) for i, h in enumerate(header)]
    print(_fmt_row(header, widths))
    print("-+-".join("-" * w for w in widths))
    for b in body:
        print(_fmt_row(b, widths))


def _summary_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"n": 0.0, "min": math.nan, "max": math.nan, "mean": math.nan, "median": math.nan}
    return {
        "n": float(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
    }


def _print_distribution_summary(rows: List[ParticipantTrialStats], label: str) -> None:
    tr = [r.train_trials for r in rows]
    te = [r.test_trials for r in rows]
    tt = [r.total_trials for r in rows]
    tr_s = _summary_stats(tr)
    te_s = _summary_stats(te)
    tt_s = _summary_stats(tt)
    print(f"\nDistribution summary: {label}")
    print("metric       | n    | min   | max   | mean   | median")
    print("-------------+------+-------+-------+--------+--------")
    print(
        f"train_trials | {int(tr_s['n']):<4} | {tr_s['min']:<5.0f} | {tr_s['max']:<5.0f} | "
        f"{tr_s['mean']:<6.2f} | {tr_s['median']:<6.2f}"
    )
    print(
        f"test_trials  | {int(te_s['n']):<4} | {te_s['min']:<5.0f} | {te_s['max']:<5.0f} | "
        f"{te_s['mean']:<6.2f} | {te_s['median']:<6.2f}"
    )
    print(
        f"total_trials | {int(tt_s['n']):<4} | {tt_s['min']:<5.0f} | {tt_s['max']:<5.0f} | "
        f"{tt_s['mean']:<6.2f} | {tt_s['median']:<6.2f}"
    )


def _load_valid_ids(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("valid_participant_ids", [])
    if not isinstance(ids, list):
        raise ValueError(f"valid_participant_ids missing/invalid in {path}")
    return [int(x) for x in ids]


def _collect_split_stats(
    *,
    dataset,
    split_name: str,
    experiment_filter: str,
    participant_row_ids: List[int],
    split_ratio: float,
    split_seed: int,
) -> List[ParticipantTrialStats]:
    split = dataset[split_name]
    filtered = split.filter(lambda ex: ex["experiment"] == experiment_filter)
    rows: List[ParticipantTrialStats] = []
    for ordinal, row_id in enumerate(participant_row_ids):
        try:
            row = filtered[int(row_id)]
            rows.append(
                _participant_stats(
                    row=row,
                    split_name=split_name,
                    ordinal=ordinal,
                    hf_row_id=int(row_id),
                    split_ratio=split_ratio,
                    split_seed=split_seed,
                )
            )
        except Exception:
            # Keep behavior aligned with participant validation scripts:
            # invalid rows (e.g., <2 blocks) are skipped.
            continue
    return rows


def _collect_run_dir_stats(run_dir: Path) -> List[ParticipantTrialStats]:
    rows: List[ParticipantTrialStats] = []
    for pdir in sorted(run_dir.glob("participant_*")):
        if not pdir.is_dir():
            continue
        name = pdir.name
        if not name.startswith("participant_"):
            continue
        try:
            pid = int(name.split("_", 1)[1])
        except ValueError:
            continue
        results_path = pdir / "results.json"
        if not results_path.exists():
            continue
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        base = payload.get("baseline", {})
        tr = int(base.get("train_total", 0))
        te = int(base.get("test_total", 0))
        rows.append(
            ParticipantTrialStats(
                split_name="run_artifact",
                ordinal=pid,
                hf_row_id=pid,
                raw_participant_id=pid,
                n_blocks=-1,
                total_trials=tr + te,
                train_blocks=-1,
                test_blocks=-1,
                train_trials=tr,
                test_trials=te,
            )
        )
    rows.sort(key=lambda r: r.ordinal)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_dataset", default="marcelbinz/Psych-101-test")
    parser.add_argument("--experiment_filter", default="peterson2021using/exp1.csv")
    parser.add_argument(
        "--analysis_split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which HF split to analyze for the ordinal table.",
    )
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help=(
            "Optional local run directory (e.g. generated_outputs/choice13k/non_strict/run_xxx). "
            "If set, report train/test trial counts from participant_*/results.json baseline totals."
        ),
    )
    parser.add_argument(
        "--valid_ids_json",
        default=str(REPO_ROOT / "datasets" / "choice13k" / "valid_participant_ids.json"),
    )
    parser.add_argument("--range_start_ordinal", type=int, default=0)
    parser.add_argument("--range_end_ordinal", type=int, default=9)
    parser.add_argument(
        "--compare_train_split",
        action="store_true",
        default=True,
        help="Also report distribution over train split when available.",
    )
    args = parser.parse_args()

    if args.range_start_ordinal < 0 or args.range_end_ordinal < args.range_start_ordinal:
        raise ValueError("Invalid ordinal range.")

    print("Choice13k trial-count inspection (OpenEvolve split logic)")

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_stats = _collect_run_dir_stats(run_dir)
        selected = [r for r in run_stats if args.range_start_ordinal <= r.ordinal <= args.range_end_ordinal]
        print(f"Run directory source: {run_dir}")
        print(f"Requested ordinal range: [{args.range_start_ordinal}, {args.range_end_ordinal}]")
        _print_table(selected, title="Selected participants (from local run artifacts)")
        _print_distribution_summary(selected, "selected ordinals from local run artifacts")
        _print_distribution_summary(run_stats, "all participants found in local run artifacts")
        print(
            "\nNote: local run artifacts cannot verify Psych-101 train-split comparison; "
            "use online mode for that."
        )
        return

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "datasets package is required for online Psych-101 inspection. "
            "Either install it or run with --run_dir to use local artifacts."
        ) from e

    tok = _hf_token_for_datasets()
    ds_kw = {"token": tok} if tok else {}
    dataset = load_dataset(args.hf_dataset, **ds_kw)

    if args.analysis_split not in dataset:
        raise ValueError(
            f"Requested analysis_split={args.analysis_split!r} not found. "
            f"Available splits: {list(dataset.keys())}"
        )

    analysis_filtered = dataset[args.analysis_split].filter(
        lambda ex: ex["experiment"] == args.experiment_filter
    )
    n_analysis = len(analysis_filtered)
    if n_analysis == 0:
        raise RuntimeError(
            f"No rows after filtering split={args.analysis_split!r} by experiment={args.experiment_filter!r}"
        )

    # valid_participant_ids.json is defined for Psych-101-test in this repo.
    # Reuse only when analyzing test; otherwise use direct row ordinals for that split.
    if args.analysis_split == "test":
        valid_ids = _load_valid_ids(Path(args.valid_ids_json))
        if args.range_end_ordinal >= len(valid_ids):
            raise ValueError(
                f"range_end_ordinal={args.range_end_ordinal} out of bounds for valid list size {len(valid_ids)}"
            )
        selected_row_ids = valid_ids[args.range_start_ordinal : args.range_end_ordinal + 1]
        full_row_ids = valid_ids
        participant_source = "datasets/choice13k/valid_participant_ids.json"
    else:
        if args.range_end_ordinal >= n_analysis:
            raise ValueError(
                f"range_end_ordinal={args.range_end_ordinal} out of bounds for "
                f"{args.analysis_split} filtered size {n_analysis}"
            )
        selected_row_ids = list(range(args.range_start_ordinal, args.range_end_ordinal + 1))
        full_row_ids = list(range(n_analysis))
        participant_source = f"direct ordinals in HF split '{args.analysis_split}' after experiment filtering"

    selected_stats = _collect_split_stats(
        dataset=dataset,
        split_name=args.analysis_split,
        experiment_filter=args.experiment_filter,
        participant_row_ids=selected_row_ids,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
    )
    full_test_stats = _collect_split_stats(
        dataset=dataset,
        split_name=args.analysis_split,
        experiment_filter=args.experiment_filter,
        participant_row_ids=full_row_ids,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
    )

    print(f"HF dataset: {args.hf_dataset}")
    print(f"Experiment filter: {args.experiment_filter}")
    print(f"Analysis split: {args.analysis_split}")
    print(f"Participant source for ordinal range: {participant_source}")
    print(f"Requested ordinal range: [{args.range_start_ordinal}, {args.range_end_ordinal}]")
    print(f"Split used: split_ratio={args.split_ratio}, split_seed={args.split_seed}")

    _print_table(selected_stats, title=f"Selected participants (Psych-101 {args.analysis_split} split)")
    _print_distribution_summary(selected_stats, f"selected ordinals on Psych-101 {args.analysis_split} split")
    _print_distribution_summary(full_test_stats, f"all participants on Psych-101 {args.analysis_split} split")

    if args.compare_train_split and "train" in dataset:
        train_split = dataset["train"]
        filtered_train = train_split.filter(lambda ex: ex["experiment"] == args.experiment_filter)
        train_row_ids = list(range(len(filtered_train)))
        train_stats = _collect_split_stats(
            dataset=dataset,
            split_name="train",
            experiment_filter=args.experiment_filter,
            participant_row_ids=train_row_ids,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
        )
        _print_distribution_summary(train_stats, "all participants on Psych-101 train split")
    elif args.compare_train_split:
        print("\nNo 'train' split found in dataset; skipped train comparison.")


if __name__ == "__main__":
    main()
