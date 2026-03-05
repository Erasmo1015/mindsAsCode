#!/usr/bin/env python3
"""
Plot Mixed Gambles figures for the logistic MLE baseline (Section 4.2).

Same figures as plot_mixed_gambles.py (Fig 4.3, 4.5, 4.6), but the "model" is the
fitted gain-loss logistic: utility = G - omega*L, P(accept) = sigmoid(lambda*utility).
Parameters (omega, lambda) are read from analysis/models/mixed_gambles/logistic_MLE_results.csv.

Usage:
  python analysis/code/mixed_gambles/plot_mixed_gambles_MLE.py --participant_id 101
  python analysis/code/mixed_gambles/plot_mixed_gambles_MLE.py --participant_id 101 --split both --grid_mode full
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# Allow importing from sibling analysis/code/
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from plot_mixed_gambles import (
    ACCEPT_ACTION,
    load_mixed_gambles_data,
    make_problem_for_gl,
    plot_fig43_behavior,
    plot_fig45_design_heatmap,
    plot_fig46_generalization_heatmap,
    trial_to_gl,
)

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MLE_CSV = REPO_ROOT / "analysis/models/mixed_gambles/logistic_MLE_results.csv"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def load_mle_params(mle_csv_path: Path, participant_id: int) -> Tuple[float, float]:
    """Load omega and lambda for participant_id from logistic_MLE_results.csv."""
    with open(mle_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["participant_id"]) == participant_id:
                return float(row["omega"]), float(row["lambda"])
    raise ValueError(f"Participant {participant_id} not found in {mle_csv_path}")


def make_mle_choose(omega: float, lam: float) -> Callable:
    """Return choose(problem, history) that implements logistic MLE: 1 if P(accept) >= 0.5 else 0."""

    def choose(problem: Dict[str, Any], history: List[Any]) -> int:
        r = problem["gamble_A"]["rewards"]
        G = float(r[0])
        L = abs(float(r[1]))
        utility = G - omega * L
        p = sigmoid(lam * utility)
        return ACCEPT_ACTION if p >= 0.5 else (1 - ACCEPT_ACTION)

    return choose


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Mixed Gambles figures for MLE baseline")
    parser.add_argument("--participant_id", type=int, required=True, help="Participant (subject) id")
    parser.add_argument("--mle_csv", type=str, default=str(DEFAULT_MLE_CSV), help="Path to logistic_MLE_results.csv")
    parser.add_argument("--output_dir", type=str, default="analysis/figures", help="Output directory for PNGs and metadata")
    parser.add_argument("--tag", type=str, default="", help="Optional extra filename tag")
    parser.add_argument("--split", type=str, choices=["none", "train", "test", "both"], default="none", help="Which split to plot")
    parser.add_argument("--grid_mode", type=str, choices=["none", "design", "full"], default="full", help="Fig 4.6: none=skip, design=design points only, full=dense grid")
    parser.add_argument("--loss_max", type=int, default=32, help="Max loss for axes and full grid")
    parser.add_argument("--gain_max", type=int, default=16, help="Max gain for axes and full grid")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--csv_path", type=str, default="datasets/mixed_gambles/data_all_2021-01-08.csv", help="Path to mixed_gambles CSV (relative to cwd or repo root)")
    parser.add_argument("--filter_mixed_gambles", action="store_true", help="Keep only gain_loss trial type (Section 4.2). Default: disabled (use all trial types).")
    args = parser.parse_args()

    repo_root = Path(os.getcwd())
    if not (repo_root / "Template_evo_non_strict.py").exists():
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
    csv_path = repo_root / args.csv_path
    if not csv_path.exists():
        csv_path_alt = Path(__file__).resolve().parent.parent.parent.parent / args.csv_path
        if csv_path_alt.exists():
            csv_path = csv_path_alt
        else:
            raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    mle_csv_path = Path(args.mle_csv)
    if not mle_csv_path.is_absolute():
        mle_csv_path = repo_root / args.mle_csv
    if not mle_csv_path.exists():
        raise FileNotFoundError(f"MLE results not found: {mle_csv_path}")

    omega, lam = load_mle_params(mle_csv_path, args.participant_id)
    choose_fn = make_mle_choose(omega, lam)

    train_trials, test_trials, _ = load_mixed_gambles_data(str(csv_path), args.participant_id, filter_gain_loss_only=args.filter_mixed_gambles)
    print(f"[Split] Train: {len(train_trials)}, Test: {len(test_trials)} (seed=42)")
    print(f"[MLE] participant_id={args.participant_id} omega={omega:.4f} lambda={lam:.4f}")

    base_folder = f"mixed_gambles_MLE_p{args.participant_id}"
    if args.tag:
        base_folder += f"_{args.tag}"
    out_dir = Path(args.output_dir) / base_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: List[Tuple[str, List[Dict[str, Any]]]] = []
    if args.split == "none":
        all_trials = train_trials + test_trials
        splits.append(("none", all_trials))
    else:
        if args.split in ("train", "both"):
            splits.append(("train", train_trials))
        if args.split in ("test", "both"):
            splits.append(("test", test_trials))

    for split_name, trials in splits:
        if not trials:
            continue
        file_prefix = "" if args.split == "none" else f"{split_name}_"
        plot_fig43_behavior(
            trials,
            str(out_dir / f"{file_prefix}fig43_behavior.png"),
            gain_max=float(args.gain_max),
            loss_max=float(args.loss_max),
            dpi=args.dpi,
        )
        plot_fig45_design_heatmap(
            trials,
            choose_fn,
            str(out_dir / f"{file_prefix}fig45_design_heatmap.png"),
            gain_max=float(args.gain_max),
            loss_max=float(args.loss_max),
            dpi=args.dpi,
        )
        if args.grid_mode != "none":
            plot_fig46_generalization_heatmap(
                trials,
                choose_fn,
                str(out_dir / f"{file_prefix}fig46_generalization_heatmap.png"),
                gain_max=args.gain_max,
                loss_max=args.loss_max,
                grid_mode=args.grid_mode,
                dpi=args.dpi,
            )

    n_train, n_test = len(train_trials), len(test_trials)
    train_accept = sum(1 for t in train_trials if t["action"] == ACCEPT_ACTION)
    test_accept = sum(1 for t in test_trials if t["action"] == ACCEPT_ACTION)
    train_pred_accept = sum(1 for t in train_trials if choose_fn(t["problem"], t["history"]) == ACCEPT_ACTION)

    meta = {
        "participant_id": args.participant_id,
        "model": "logistic_MLE",
        "omega": omega,
        "lambda": lam,
        "split": args.split,
        "mle_csv": str(mle_csv_path),
        "dataset_info": {
            "gamble_A_probs": [0.5, 0.5],
            "gamble_B_probs": [1.0],
            "accept_action": ACCEPT_ACTION,
            "gain_max": args.gain_max,
            "loss_max": args.loss_max,
        },
        "n_train": n_train,
        "n_test": n_test,
        "train_observed_accept_rate": train_accept / n_train if n_train else 0,
        "test_observed_accept_rate": test_accept / n_test if n_test else 0,
        "train_program_accept_rate": train_pred_accept / n_train if n_train else 0,
    }
    if args.split == "none":
        meta["n_total"] = n_train + n_test
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    unique_gl_train = len(set((round(trial_to_gl(t)[0], 4), round(trial_to_gl(t)[1], 4)) for t in train_trials))
    unique_gl_test = len(set((round(trial_to_gl(t)[0], 4), round(trial_to_gl(t)[1], 4)) for t in test_trials))
    print("Trials loaded: train =", n_train, ", test =", n_test)
    if args.split == "none":
        print("Effective split: none (all trials combined), n_total =", meta["n_total"])
    print("Unique (G,L) points: train =", unique_gl_train, ", test =", unique_gl_test)
    print("Participant accept rate: train =", f"{train_accept/n_train:.3f}" if n_train else "N/A", ", test =", f"{test_accept/n_test:.3f}" if n_test else "N/A")
    print("MLE accept rate (train) =", f"{train_pred_accept/n_train:.3f}" if n_train else "N/A")
    print("Saved figures and metadata to:", out_dir)
    print("Metadata:", meta_path)
    for split_name, trials in splits:
        pfx = "" if args.split == "none" else f"{split_name}_"
        print("  ", out_dir / f"{pfx}fig43_behavior.png")
        print("  ", out_dir / f"{pfx}fig45_design_heatmap.png")
        if args.grid_mode != "none":
            print("  ", out_dir / f"{pfx}fig46_generalization_heatmap.png")


if __name__ == "__main__":
    main()
