#!/usr/bin/env python3
"""
Validate TEH Psych-101 parsers for all registered binary dataset aliases.

Writes:
  analysis/data/psych-101/teh_parser_validation.csv
  analysis/data/psych-101/teh_parser_validation_examples.json

Example:
  python analysis/code/psych-101/validate_teh_all_datasets.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_modules.psych101_binary import (
    PSYCH101_BINARY_DATASETS,
    experiment_to_trial_dicts,
    get_psych101_binary_experiments,
    parse_coverage_stats,
    split_psych_experiment,
)
from datasets import load_dataset

TRAIN_HF = "marcelbinz/Psych-101"
TEST_HF = "marcelbinz/Psych-101-test"
DEFAULT_OUT = REPO_ROOT / "analysis" / "data" / "psych-101"


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_split(hf_id: str, split: str):
    tok = _hf_token()
    kw = {"token": tok} if tok else {}
    return load_dataset(hf_id, **kw)[split]


def _pick_indices(n: int, k: int = 3) -> List[int]:
    if n == 0:
        return []
    if n <= k:
        return list(range(n))
    return [0, n // 2, n - 1]


def _trivial_choose(problem: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    return 0.5


def _eval_loglik(trials: List[Dict[str, Any]]) -> float:
    import math

    ll = 0.0
    for t in trials:
        p = max(1e-6, min(1.0 - 1e-6, _trivial_choose(t["problem"], t["history"])))
        y = int(t["action"])
        ll += math.log(p) if y == 1 else math.log(1.0 - p)
    return ll / max(1, len(trials))


def _validate_participant(
    dataset_alias: str,
    row: Dict[str, Any],
    *,
    split_ratio: float,
    split_seed: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "participant": row.get("participant"),
        "parse_ok": False,
        "error": None,
        "n_blocks": 0,
        "n_trials": 0,
        "parse_coverage": 0.0,
        "option_keys": [],
        "is_binary": False,
        "split_ok": False,
        "loglik_ok": False,
    }
    try:
        from data_modules.psych101_binary import _parse_row

        exp = _parse_row(row, dataset_alias)
        stats = parse_coverage_stats(row["text"], exp)
        trials = experiment_to_trial_dicts(exp)
        keys = set()
        for b in exp.blocks:
            keys.update(b.option_keys)
        out["parse_ok"] = len(trials) > 0
        out["n_blocks"] = len(exp.blocks)
        out["n_trials"] = len(trials)
        out["parse_coverage"] = stats["parse_coverage"]
        out["option_keys"] = sorted(keys)
        out["is_binary"] = len(keys) == 2
        if trials:
            out["sample_problem"] = trials[0]["problem"]
            out["sample_history"] = trials[min(1, len(trials) - 1)].get("history", [])
            out["sample_action"] = trials[0]["action"]
            out["sample_action_key"] = trials[0]["problem"]["option_keys"][trials[0]["action"]]
        tr, va, te, _ = split_psych_experiment(
            exp, split_ratio=split_ratio, split_seed=split_seed
        )
        out["split_ok"] = bool(tr) and bool(te)
        out["n_train_trials"] = len(tr)
        out["n_test_trials"] = len(te)
        if tr:
            _eval_loglik(tr)
            out["loglik_ok"] = True
    except Exception as e:
        out["error"] = str(e)
        out["traceback"] = traceback.format_exc()[-500:]
    return out


def run_validation(
    output_dir: Path,
    *,
    samples_per_split: int = 3,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading HF splits...")
    train_ds = _load_split(TRAIN_HF, "train")
    test_ds = _load_split(TEST_HF, "test")

    csv_rows: List[Dict[str, Any]] = []
    examples: Dict[str, Any] = {}

    for alias, spec in PSYCH101_BINARY_DATASETS.items():
        exp_id = spec["experiment_id"]
        print(f"Validating {alias} ({exp_id})...")
        train_rows = [dict(r) for r in train_ds if r["experiment"] == exp_id]
        test_rows = [dict(r) for r in test_ds if r["experiment"] == exp_id]
        train_pick = [train_rows[i] for i in _pick_indices(len(train_rows), samples_per_split)]
        test_pick = [test_rows[i] for i in _pick_indices(len(test_rows), samples_per_split)]

        train_results = [
            _validate_participant(alias, r, split_ratio=split_ratio, split_seed=split_seed)
            for r in train_pick
        ]
        test_results = [
            _validate_participant(alias, r, split_ratio=split_ratio, split_seed=split_seed)
            for r in test_pick
        ]

        parsed_train = sum(1 for r in train_results if r["parse_ok"])
        parsed_test = sum(1 for r in test_results if r["parse_ok"])
        coverages = [r["parse_coverage"] for r in train_results + test_results if r["parse_ok"]]
        trial_counts = [r["n_trials"] for r in train_results + test_results if r["parse_ok"]]
        med_cov = sorted(coverages)[len(coverages) // 2] if coverages else 0.0
        med_trials = sorted(trial_counts)[len(trial_counts) // 2] if trial_counts else 0

        implemented = spec.get("implemented") and med_cov >= 0.95 and all(
            r.get("is_binary") for r in train_results + test_results if r["parse_ok"]
        )

        csv_rows.append(
            {
                "dataset_alias": alias,
                "experiment_id": exp_id,
                "schema_type": spec["schema_type"],
                "parser": spec["parser"],
                "implemented": int(implemented),
                "train_n_participants": len(train_rows),
                "test_n_participants": len(test_rows),
                "parsed_train_samples": parsed_train,
                "parsed_test_samples": parsed_test,
                "median_parse_coverage": med_cov,
                "median_trials_per_participant": med_trials,
                "all_samples_binary": int(
                    all(r.get("is_binary") for r in train_results + test_results if r["parse_ok"])
                ),
            }
        )
        examples[alias] = {
            "train_samples": train_results,
            "test_samples": test_results,
        }

    csv_path = output_dir / "teh_parser_validation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    json_path = output_dir / "teh_parser_validation_examples.json"
    json_path.write_text(json.dumps(examples, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples_per_split", type=int, default=3)
    args = parser.parse_args()
    out = args.output_dir.expanduser()
    out = out.resolve() if out.is_absolute() else (REPO_ROOT / out).resolve()
    run_validation(out, samples_per_split=args.samples_per_split)


if __name__ == "__main__":
    main()
