#!/usr/bin/env python3
"""
Validate that TEH peterson2021using parsing matches legacy choice13k loader.

  python analysis/code/psych-101/validate_teh_peterson.py
  python analysis/code/psych-101/validate_teh_peterson.py --participant 0 5 10
  python analysis/code/psych-101/validate_teh_peterson.py --psych_dataset_split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from data_modules.choice13k import get_choice13k_experiments
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    get_psych101_binary_experiments,
    hf_id_for_psych_dataset_split,
    parse_coverage_stats,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument(
        "--psych_dataset_split",
        type=str,
        default=DEFAULT_PSYCH_DATASET_SPLIT,
        choices=["train", "test"],
        help="Psych-101 HF corpus for TEH loader (default: train).",
    )
    args = parser.parse_args()

    def _exp_action_sequence(exp):
        actions = []
        for b in exp.blocks:
            for t in b.trials:
                actions.append((b.option_keys, t.action, t.feedback))
        return actions

    ok = True
    for pid in args.participant:
        legacy = get_choice13k_experiments(n_participants=pid + 1)[pid]
        teh_exp = get_psych101_binary_experiments(
            "1peterson2021using",
            n_participants=pid + 1,
            split=args.psych_dataset_split,
        )[pid]
        if _exp_action_sequence(legacy) != _exp_action_sequence(teh_exp):
            print(f"FAIL participant {pid}: trial sequence mismatch")
            ok = False
        elif len(legacy.blocks) != len(teh_exp.blocks):
            print(f"FAIL participant {pid}: block count {len(legacy.blocks)} vs {len(teh_exp.blocks)}")
            ok = False
        else:
            n_trials = sum(len(b.trials) for b in teh_exp.blocks)
            print(f"OK participant {pid}: blocks={len(teh_exp.blocks)} trials={n_trials}")

    # plonsky coverage smoke
    try:
        pexp = get_psych101_binary_experiments(
            "plonsky2018when", n_participants=1, split=args.psych_dataset_split
        )[0]
        from datasets import load_dataset
        import os

        psych_split = args.psych_dataset_split
        hf_id = hf_id_for_psych_dataset_split(psych_split)
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ds = load_dataset(hf_id, token=tok if tok else None)[psych_split]
        row = next(r for r in ds if r["experiment"] == "plonsky2018when/exp1.csv")
        stats = parse_coverage_stats(row["text"], pexp)
        print(
            f"plonsky2018when participant 0 ({args.psych_dataset_split}): "
            f"coverage={stats['parse_coverage']} "
            f"({stats['n_trials_parsed']}/{stats['n_presses_in_text']} presses)"
        )
        if stats["parse_coverage"] < 0.95:
            print("WARN: plonsky coverage below 0.95 — check extended trial regex")
            ok = False
    except Exception as e:
        print(f"plonsky check skipped/failed: {e}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
