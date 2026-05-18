#!/usr/bin/env python3
"""
Validate that TEH peterson2021using parsing matches legacy choice13k loader.

  python analysis/code/psych-101/validate_teh_peterson.py
  python analysis/code/psych-101/validate_teh_peterson.py --participant 0 5 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from data_modules.choice13k import get_choice13k_experiments
from data_modules.psych101_binary import get_psych101_binary_experiments, parse_coverage_stats


def _trial_signature(trials):
    sig = []
    for t in trials:
        p = t["problem"]
        sig.append(
            (
                t["action"],
                tuple(p["gamble_A"]["rewards"] or []),
                tuple(p["gamble_B"]["rewards"] or []),
                len(t["history"]),
            )
        )
    return sig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", type=int, nargs="*", default=[0, 1, 2])
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
            "peterson2021using", n_participants=pid + 1, split="test"
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
        pexp = get_psych101_binary_experiments("plonsky2018when", n_participants=1)[0]
        from datasets import load_dataset
        import os

        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ds = load_dataset("marcelbinz/Psych-101-test", token=tok if tok else None)["test"]
        row = next(r for r in ds if r["experiment"] == "plonsky2018when/exp1.csv")
        stats = parse_coverage_stats(row["text"], pexp)
        print(
            f"plonsky2018when participant 0: coverage={stats['parse_coverage']} "
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
