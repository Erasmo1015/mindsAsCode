#!/usr/bin/env python3
"""
Load Psych-101 training and test Hugging Face datasets separately and write which
experiment problems (the ``experiment`` column) each contains to analysis/data/psych-101/.

Centaur finetunes on marcelbinz/Psych-101 (train split). Held-out participant
evaluation uses marcelbinz/Psych-101-test (test split). This repo's local benchmarks
map to Psych-101 experiments as follows:

  choice13k     -> peterson2021using/exp1.csv
  cpc18         -> plonsky2018when/exp1.csv
  mixed_gambles -> (not included in Psych-101; local CSV only)

Example:
  python analysis/code/psych-101/extract_dataset_problems.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "data" / "psych-101"

TRAIN_HF_ID = "marcelbinz/Psych-101"
TRAIN_SPLIT = "train"
TEST_HF_ID = "marcelbinz/Psych-101-test"
TEST_SPLIT = "test"

REPO_BENCHMARK_TO_EXPERIMENT: Dict[str, str] = {
    "choice13k": "peterson2021using/exp1.csv",
    "cpc18": "plonsky2018when/exp1.csv",
}


@dataclass(frozen=True)
class ExperimentCounts:
    experiment_id: str
    n_participant_rows: int
    repo_benchmark: str = ""


def _hf_token_for_datasets() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_hf_split(hf_id: str, split_name: str, *, local_path: Optional[Path] = None):
    if local_path is not None:
        from datasets import load_from_disk

        dataset = load_from_disk(str(local_path.expanduser().resolve()))
        if split_name not in dataset:
            raise ValueError(
                f"Local dataset at {local_path} has splits {list(dataset.keys())}, "
                f"expected {split_name!r}"
            )
        return dataset[split_name]

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Install the `datasets` package or pass --train_local / --test_local."
        ) from e

    tok = _hf_token_for_datasets()
    ds_kw = {"token": tok} if tok else {}
    dataset = load_dataset(hf_id, **ds_kw)
    if split_name not in dataset:
        raise ValueError(
            f"{hf_id} has splits {list(dataset.keys())}, expected {split_name!r}"
        )
    return dataset[split_name]


def _experiment_to_repo_benchmark(experiment_id: str) -> str:
    for name, exp_id in REPO_BENCHMARK_TO_EXPERIMENT.items():
        if exp_id == experiment_id:
            return name
    return ""


def _collect_experiment_counts(split_ds) -> List[ExperimentCounts]:
    counter = Counter(split_ds["experiment"])
    rows = [
        ExperimentCounts(
            experiment_id=exp_id,
            n_participant_rows=count,
            repo_benchmark=_experiment_to_repo_benchmark(exp_id),
        )
        for exp_id, count in counter.items()
    ]
    rows.sort(key=lambda r: (r.repo_benchmark == "", r.repo_benchmark, r.experiment_id))
    return rows


def _write_experiments_csv(path: Path, rows: Sequence[ExperimentCounts], split_label: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["split", "experiment_id", "repo_benchmark", "n_participant_rows"],
        )
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "split": split_label,
                    "experiment_id": row.experiment_id,
                    "repo_benchmark": row.repo_benchmark,
                    "n_participant_rows": row.n_participant_rows,
                }
            )


def _write_merged_csv(
    path: Path,
    train_counts: Sequence[ExperimentCounts],
    test_counts: Sequence[ExperimentCounts],
) -> None:
    train_map = {r.experiment_id: r for r in train_counts}
    test_map = {r.experiment_id: r for r in test_counts}
    all_ids = sorted(set(train_map) | set(test_map))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_id",
                "repo_benchmark",
                "train_n_participant_rows",
                "test_n_participant_rows",
                "in_train_hf",
                "in_test_hf",
            ],
        )
        w.writeheader()
        for exp_id in all_ids:
            tr = train_map.get(exp_id)
            te = test_map.get(exp_id)
            w.writerow(
                {
                    "experiment_id": exp_id,
                    "repo_benchmark": (tr or te).repo_benchmark if (tr or te) else "",
                    "train_n_participant_rows": tr.n_participant_rows if tr else 0,
                    "test_n_participant_rows": te.n_participant_rows if te else 0,
                    "in_train_hf": int(tr is not None),
                    "in_test_hf": int(te is not None),
                }
            )


def _write_repo_benchmarks_csv(
    path: Path,
    train_counts: Sequence[ExperimentCounts],
    test_counts: Sequence[ExperimentCounts],
) -> None:
    train_map = {r.experiment_id: r.n_participant_rows for r in train_counts}
    test_map = {r.experiment_id: r.n_participant_rows for r in test_counts}

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "repo_benchmark",
                "experiment_id",
                "train_n_participant_rows",
                "test_n_participant_rows",
                "in_psych_101",
            ],
        )
        w.writeheader()
        for bench, exp_id in REPO_BENCHMARK_TO_EXPERIMENT.items():
            w.writerow(
                {
                    "repo_benchmark": bench,
                    "experiment_id": exp_id,
                    "train_n_participant_rows": train_map.get(exp_id, 0),
                    "test_n_participant_rows": test_map.get(exp_id, 0),
                    "in_psych_101": int(exp_id in train_map or exp_id in test_map),
                }
            )
        w.writerow(
            {
                "repo_benchmark": "mixed_gambles",
                "experiment_id": "",
                "train_n_participant_rows": 0,
                "test_n_participant_rows": 0,
                "in_psych_101": 0,
            }
        )


def _build_summary(
    *,
    train_hf: str,
    test_hf: str,
    train_split: str,
    test_split: str,
    train_counts: Sequence[ExperimentCounts],
    test_counts: Sequence[ExperimentCounts],
) -> dict:
    train_ids = {r.experiment_id for r in train_counts}
    test_ids = {r.experiment_id for r in test_counts}
    return {
        "train_hf_id": train_hf,
        "test_hf_id": test_hf,
        "train_split": train_split,
        "test_split": test_split,
        "n_unique_experiments_train": len(train_counts),
        "n_unique_experiments_test": len(test_counts),
        "n_participant_rows_train": sum(r.n_participant_rows for r in train_counts),
        "n_participant_rows_test": sum(r.n_participant_rows for r in test_counts),
        "experiment_ids_train_only": sorted(train_ids - test_ids),
        "experiment_ids_test_only": sorted(test_ids - train_ids),
        "repo_benchmark_mapping": REPO_BENCHMARK_TO_EXPERIMENT,
        "notes": {
            "mixed_gambles": "Not in Psych-101; local datasets/mixed_gambles/ only.",
            "holdout": (
                "Train and test HF repos share the same experiment ids; "
                "Psych-101-test holds out ~10% of participants per experiment."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_hf",
        default=TRAIN_HF_ID,
        help=f"Hugging Face id for training data (default: {TRAIN_HF_ID})",
    )
    parser.add_argument(
        "--test_hf",
        default=TEST_HF_ID,
        help=f"Hugging Face id for test data (default: {TEST_HF_ID})",
    )
    parser.add_argument(
        "--train_local",
        type=Path,
        default=None,
        help="Optional local path from datasets.load_from_disk for train split",
    )
    parser.add_argument(
        "--test_local",
        type=Path,
        default=None,
        help="Optional local path from datasets.load_from_disk for test split",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    out_dir = args.output_dir.expanduser()
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_source = str(args.train_local) if args.train_local else args.train_hf
    test_source = str(args.test_local) if args.test_local else args.test_hf

    train_ds = _load_hf_split(args.train_hf, TRAIN_SPLIT, local_path=args.train_local)
    test_ds = _load_hf_split(args.test_hf, TEST_SPLIT, local_path=args.test_local)

    train_counts = _collect_experiment_counts(train_ds)
    test_counts = _collect_experiment_counts(test_ds)

    paths = {
        "train_experiments.csv": out_dir / "train_experiments.csv",
        "test_experiments.csv": out_dir / "test_experiments.csv",
        "experiments_merged.csv": out_dir / "experiments_merged.csv",
        "repo_benchmarks.csv": out_dir / "repo_benchmarks.csv",
        "summary.json": out_dir / "summary.json",
    }

    _write_experiments_csv(paths["train_experiments.csv"], train_counts, TRAIN_SPLIT)
    _write_experiments_csv(paths["test_experiments.csv"], test_counts, TEST_SPLIT)
    _write_merged_csv(paths["experiments_merged.csv"], train_counts, test_counts)
    _write_repo_benchmarks_csv(paths["repo_benchmarks.csv"], train_counts, test_counts)

    summary = _build_summary(
        train_hf=train_source,
        test_hf=test_source,
        train_split=TRAIN_SPLIT,
        test_split=TEST_SPLIT,
        train_counts=train_counts,
        test_counts=test_counts,
    )
    paths["summary.json"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote Psych-101 problem inventory to {out_dir}/")
    for name in paths:
        print(f"  {name}")


if __name__ == "__main__":
    main()
