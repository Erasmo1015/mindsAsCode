#!/usr/bin/env python3
"""
Plot Mixed Gambles figures in the style of Twelve Angry Models (Section 4.2).

For ONE participant and ONE evaluated program (choose(problem, history)), produces:
- Fig 4.3-like: Behavioral data scatter in G–L space (green = accept gamble, red = reject).
- Fig 4.5-like: Predictive probability heatmap over design points (program 0/1 → red/green).
- Fig 4.6-like: Generalization heatmap over full G–L grid.

Dataset and split logic match Template_evo_non_strict.py (mixed_gambles):
- gamble_A = risky [gain, loss], probs [0.5, 0.5]; gamble_B = certain [cert], probs [1.0].
- TE option index: action 0 = choose gamble_A (accept gamble); action 1 = choose gamble_B (certain / reject gamble).
- Raw CSV `took_gamble` is converted at load: action = 1 - took_gamble.
- 80/20 train/test split with RNG seed 42 (shuffled indices), not raw row order.

Usage examples:
  # Default: all trials combined, save to analysis/figures/mixed_gambles_p{id}_iter{iter}_cand{cand}/
  python analysis/code/plot_mixed_gambles.py --participant_id 101 --program_path path/to/candidate_0.py

  python analysis/code/plot_mixed_gambles.py --dataset mixed_gambles --participant_id 114 \\
    --program_path generated_outputs/mixed_gambles/non_strict/run_XXX/participant_114/iteration_2/candidates/candidate_8.py \\
    --split both --grid_mode full

  python analysis/code/plot_mixed_gambles.py --participant_id 101 --program_path path/to/candidate_0.py --split train --output_dir analysis/figures
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# -----------------------------------------------------------------------------
# Dataset: same format and split as Template_evo_non_strict.py
# -----------------------------------------------------------------------------

# TE option index (mixed gambles): 0 = gamble_A (accept), 1 = gamble_B (certain). Loader sets action = 1 - took_gamble.
ACCEPT_ACTION = 0  # predicted / observed "accept gamble" matches action == 0


def load_mixed_gambles_data(
    csv_path: str,
    participant_id: int,
    filter_gain_loss_only: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """Load mixed_gambles CSV, filter by subject == participant_id, 80/20 split.
    Each row: Option A (gamble) = [gain, loss] probs [0.5, 0.5]; Option B (certain) = [cert] probs [1.0].
    Raw CSV `took_gamble`: 1 = chose gamble, 0 = chose certain. Stored action = 1 - took_gamble (0 = gamble_A, 1 = gamble_B).

    Args:
        filter_gain_loss_only: If True, keep only gamble_type == "gain_loss" trials (Section 4.2).
            If False (default), include all trial types.
    """
    option_keys = [0, 1]
    all_trials = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["subject"]) != participant_id:
                continue
            # Optional: use only mixed-gamble trials (gain_loss). Section 4.2 models 165 mixed gambles per participant.
            if filter_gain_loss_only and row.get("gamble_type") != "gain_loss":
                continue
            gain, loss, cert = float(row["gain"]), float(row["loss"]), float(row["cert"])
            took_gamble = int(row["took_gamble"])
            action = 1 - took_gamble
            all_trials.append({
                "problem": {
                    "gamble_A": {"rewards": [gain, loss], "probs": [0.5, 0.5]},
                    "gamble_B": {"rewards": [cert], "probs": [1.0]},
                    "option_keys": option_keys,
                    "has_feedback": False,
                },
                "history": [],
                "options": option_keys,
                "action": action,
            })
    if len(all_trials) == 0:
        raise ValueError(f"No rows found for subject {participant_id} in {csv_path}")
    if filter_gain_loss_only and not getattr(load_mixed_gambles_data, "_printed_gain_loss", False):
        print("[Mixed Gambles] Using gain_loss trials only.")
        load_mixed_gambles_data._printed_gain_loss = True
    # Random train/test split (reproducible).
    # The dataset is ordered by stimulus structure,
    # so row-order split creates artificial distribution shift.
    rng = np.random.default_rng(42)
    indices = np.arange(len(all_trials))
    rng.shuffle(indices)
    split_point = int(len(all_trials) * 0.8)
    train_trials = [all_trials[i] for i in indices[:split_point]]
    test_trials = [all_trials[i] for i in indices[split_point:]]
    return train_trials, test_trials, option_keys


def trial_to_gl(trial: Dict[str, Any]) -> Tuple[float, float]:
    """Extract (Gain, Loss magnitude) from a trial for G–L space.
    gamble_A rewards = [gain, loss] with loss typically negative in CSV.
    """
    r = trial["problem"]["gamble_A"]["rewards"]
    G = float(r[0])
    L = abs(float(r[1]))
    return G, L


def make_problem_for_gl(gain: float, loss_mag: float, cert: float = 0.0) -> Dict[str, Any]:
    """Synthetic problem for (G, L): risky [gain, -loss_mag] 0.5/0.5, certain [cert] 1.0."""
    return {
        "gamble_A": {"rewards": [gain, -loss_mag], "probs": [0.5, 0.5]},
        "gamble_B": {"rewards": [cert], "probs": [1.0]},
        "option_keys": [0, 1],
        "has_feedback": False,
    }


def load_choose_function(program_path: str) -> Callable:
    """Load choose(problem, history) from a Python file."""
    path = Path(program_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Program file not found: {program_path}")
    spec = importlib.util.spec_from_file_location("program_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {program_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    choose_fn = getattr(mod, "choose", None)
    if not callable(choose_fn):
        raise AttributeError(f"No callable 'choose' in {program_path}")
    return choose_fn


def parse_program_path(program_path: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse iteration and candidate id from path like .../iteration_2/candidates/candidate_8.py."""
    s = Path(program_path).name
    iter_m = re.search(r"iteration_(\d+)", program_path)
    cand_m = re.search(r"candidate_(\d+)", program_path)
    iter_id = int(iter_m.group(1)) if iter_m else None
    cand_id = int(cand_m.group(1)) if cand_m else None
    return iter_id, cand_id


# -----------------------------------------------------------------------------
# Figure 4.3-like: "Summary of behavioral data... each trial is characterized by two numbers, G and L... visualized as a two-dimensional plot."
# Behavioral interpretation: "Participant B always accepts gambles for which the potential gain is as large or greater than the potential loss... Participant L is highly and consistently risk averse... losses loom large."
# -----------------------------------------------------------------------------
def plot_fig43_behavior(
    trials: List[Dict[str, Any]],
    output_path: str,
    gain_max: float,
    loss_max: float,
    dpi: int = 300,
) -> None:
    """Behavioral scatter in G–L space: green circle = accepted gamble, red x = rejected. Dashed line Gain = Loss."""
    Gs, Ls = [], []
    accept = []
    for t in trials:
        g, l = trial_to_gl(t)
        Gs.append(g)
        Ls.append(l)
        accept.append(t["action"] == ACCEPT_ACTION)
    Gs = np.array(Gs)
    Ls = np.array(Ls)
    accept = np.array(accept)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(Ls[~accept], Gs[~accept], marker="x", c="red", s=40, label="Rejected", zorder=2)
    ax.scatter(Ls[accept], Gs[accept], marker="o", facecolors="none", edgecolors="green", s=50, linewidths=1.5, label="Accepted", zorder=2)
    # Dashed boundary Gain = Loss
    lim = max(gain_max, loss_max)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.7, linewidth=1, label="Gain = Loss", zorder=1)
    ax.set_xlabel("Loss")
    ax.set_ylabel("Gain")
    ax.set_xlim(0, loss_max)
    ax.set_ylim(0, gain_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Behavioral data (G–L space)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Fig 4.5 descriptive adequacy: "At each combination of gain and loss that defines a trial, the color of the square represents the posterior predictive probability of accepting the gamble... black markers are overlaid on the color squares for those gambles that were accepted by the participant."
# We use program predicted probability (0/1) for deterministic programs.
# -----------------------------------------------------------------------------
def plot_fig45_design_heatmap(
    trials: List[Dict[str, Any]],
    choose_fn: Callable,
    output_path: str,
    gain_max: float,
    loss_max: float,
    dpi: int = 300,
) -> None:
    """Heatmap over unique (G,L) design points: color = P(predict accept) with red=0, green=1 (accept = pred == ACCEPT_ACTION). Black dots = observed accepted."""
    # Unique (G,L) and program prediction
    seen = {}
    for t in trials:
        g, l = trial_to_gl(t)
        key = (round(g, 4), round(l, 4))
        if key not in seen:
            try:
                pred = choose_fn(t["problem"], t["history"])
                p = 1 if (pred is not None and pred == ACCEPT_ACTION) else 0
            except Exception:
                p = 0
            seen[key] = {"G": g, "L": l, "p": p, "accepted": t["action"] == ACCEPT_ACTION}
    points = list(seen.values())
    if not points:
        plt.figure(figsize=(5, 5))
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return
    Gs = np.array([x["G"] for x in points])
    Ls = np.array([x["L"] for x in points])
    Ps = np.array([x["p"] for x in points])
    accepted = np.array([x["accepted"] for x in points])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Colored squares: red (0) to green (1)
    sc = ax.scatter(Ls, Gs, c=Ps, cmap="RdYlGn", vmin=0, vmax=1, s=80, marker="s", edgecolors="gray", linewidths=0.3)
    ax.scatter(Ls[accepted], Gs[accepted], facecolors="none", edgecolors="black", s=120, linewidths=1.5, marker="o", zorder=3, label="Observed accept")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Gain")
    ax.set_xlim(0, loss_max)
    ax.set_ylim(0, gain_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Program prediction at design points")
    plt.colorbar(sc, ax=ax, label="P(accept)", shrink=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Fig 4.6 generalization: "posterior predictive distribution is now inferred for all gains from $1 to $16 combined with all losses from $1 to $32... colors represent probability of accepting... black markers indicate gambles accepted in experiment."
# We use program predicted probability (0/1) over a dense grid.
# -----------------------------------------------------------------------------
def plot_fig46_generalization_heatmap(
    trials: List[Dict[str, Any]],
    choose_fn: Callable,
    output_path: str,
    gain_max: int,
    loss_max: int,
    grid_mode: str,
    dpi: int = 300,
) -> None:
    """Heatmap over G–L grid. grid_mode: 'design' = only design points; 'full' = dense grid."""
    G_design = np.array([trial_to_gl(t)[0] for t in trials])
    L_design = np.array([trial_to_gl(t)[1] for t in trials])
    accepted_obs = np.array([t["action"] == ACCEPT_ACTION for t in trials])

    # Visual swap experiment:
    # - background (model decision regions) uses the lighter symbol palette
    # - observed symbols use the stronger RdYlGn-like colors
    pastel_green = "#b7f0b7"
    pastel_red = "#f2b1b1"
    from matplotlib.colors import ListedColormap
    swapped_cmap = ListedColormap([pastel_red, pastel_green])

    if grid_mode == "design":
        # Use unique design points only
        unique_gl = {}
        for t in trials:
            g, l = trial_to_gl(t)
            key = (round(g, 4), round(l, 4))
            if key not in unique_gl:
                try:
                    pred = choose_fn(t["problem"], t["history"])
                    p = 1 if (pred is not None and pred == ACCEPT_ACTION) else 0
                except Exception:
                    p = 0
                unique_gl[key] = p
        G_vals = np.array([k[0] for k in unique_gl])
        L_vals = np.array([k[1] for k in unique_gl])
        P_vals = np.array([unique_gl[k] for k in unique_gl])
        # For "heatmap" over design we still do scatter with color
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        sc = ax.scatter(L_vals, G_vals, c=P_vals, cmap=swapped_cmap, vmin=0, vmax=1, s=100, marker="s", edgecolors="gray")
        # Observed gambles overlay (match Fig 4.3 styling): accepted = green circles, rejected = red crosses.
        ax.scatter(
            L_design[accepted_obs],
            G_design[accepted_obs],
            facecolors="none",
            edgecolors="green",
            s=100,
            linewidths=1.5,
            marker="o",
            zorder=3,
            label="Observed accept",
        )
        ax.scatter(
            L_design[~accepted_obs],
            G_design[~accepted_obs],
            marker="x",
            c="red",
            s=60,
            linewidths=1.5,
            zorder=3,
            label="Observed reject",
        )
    else:
        # Full grid: G in [1, gain_max], L in [1, loss_max]
        G_grid = np.arange(1, gain_max + 1, dtype=float)
        L_grid = np.arange(1, loss_max + 1, dtype=float)
        GG, LL = np.meshgrid(G_grid, L_grid)
        PP = np.zeros_like(GG)
        for i in range(GG.shape[0]):
            for j in range(GG.shape[1]):
                g, l = float(GG[i, j]), float(LL[i, j])
                prob = make_problem_for_gl(g, l)
                try:
                    pred = choose_fn(prob, [])
                    PP[i, j] = 1 if (pred is not None and pred == ACCEPT_ACTION) else 0
                except Exception:
                    PP[i, j] = 0
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        # For shading='flat', Z.shape must be (len(G_edges)-1, len(L_edges)-1)
        L_edges = np.arange(0, loss_max + 1)
        G_edges = np.arange(0, gain_max + 1)
        im = ax.pcolormesh(L_edges, G_edges, PP.T, cmap=swapped_cmap, vmin=0, vmax=1, shading="flat")
        # Observed gambles overlay (match Fig 4.3 styling): accepted = green circles, rejected = red crosses.
        ax.scatter(
            L_design[accepted_obs],
            G_design[accepted_obs],
            facecolors="none",
            edgecolors="green",
            s=40,
            linewidths=1,
            marker="o",
            zorder=3,
            label="Observed accept",
        )
        ax.scatter(
            L_design[~accepted_obs],
            G_design[~accepted_obs],
            marker="x",
            c="red",
            s=40,
            linewidths=1,
            zorder=3,
            label="Observed reject",
        )
        plt.colorbar(im, ax=ax, label="P(accept)", shrink=0.7)

    # Dashed boundary Gain = Loss (G = L) in G–L space.
    # Axes are Loss on x in [0, loss_max], Gain on y in [0, gain_max], so the correct diagonal
    # is from (0,0) up to (min(gain_max, loss_max), min(gain_max, loss_max)).
    lim = min(gain_max, loss_max)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.7, linewidth=1, label="Gain = Loss")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Gain")
    ax.set_xlim(0, loss_max)
    ax.set_ylim(0, gain_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Generalization (program prediction)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Mixed Gambles figures (Twelve Angry Models style)")
    parser.add_argument("--dataset", type=str, default="mixed_gambles", help="Dataset (only mixed_gambles supported)")
    parser.add_argument("--participant_id", type=int, required=True, help="Participant (subject) id")
    parser.add_argument("--program_path", type=str, required=True, help="Relative path to .py file containing choose(problem, history)")
    parser.add_argument("--output_dir", type=str, default="analysis/figures", help="Output directory for PNGs and metadata")
    parser.add_argument("--tag", type=str, default="", help="Optional extra filename tag")
    parser.add_argument("--split", type=str, choices=["none", "train", "test", "both"], default="none", help="Which split to plot (none=all trials combined, no train/test suffix)")
    parser.add_argument("--grid_mode", type=str, choices=["none", "design", "full"], default="full", help="Fig 4.6: none=skip, design=design points only, full=dense grid")
    parser.add_argument("--loss_max", type=int, default=32, help="Max loss for axes and full grid (verified from dataset range)")
    parser.add_argument("--gain_max", type=int, default=16, help="Max gain for axes and full grid (book uses 1..16)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--csv_path", type=str, default="datasets/mixed_gambles/data_all_2021-01-08.csv", help="Path to mixed_gambles CSV (relative to cwd)")
    parser.add_argument("--filter_mixed_gambles", action="store_true", help="Keep only gain_loss trial type (Section 4.2). Default: disabled (use all trial types).")
    args = parser.parse_args()

    if args.dataset != "mixed_gambles":
        print("Only dataset=mixed_gambles is supported.")
        sys.exit(1)

    # Resolve paths from repo root (assume cwd is repo root)
    repo_root = Path(os.getcwd())
    csv_path = repo_root / args.csv_path
    if not csv_path.exists():
        csv_path_alt = Path(__file__).resolve().parent.parent.parent / args.csv_path
        if csv_path_alt.exists():
            csv_path = csv_path_alt
        else:
            raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    program_path = repo_root / args.program_path
    if not program_path.exists():
        program_path = Path(args.program_path).resolve()
    if not program_path.exists():
        raise FileNotFoundError(f"Program not found: {args.program_path}")

    # Load data (same random split as Template_evo_non_strict, seed=42)
    train_trials, test_trials, _ = load_mixed_gambles_data(str(csv_path), args.participant_id, filter_gain_loss_only=args.filter_mixed_gambles)
    print(f"[Split] Train: {len(train_trials)}, Test: {len(test_trials)} (seed=42)")
    choose_fn = load_choose_function(str(program_path))
    iter_id, cand_id = parse_program_path(str(program_path))
    iter_s = str(iter_id) if iter_id is not None else "NA"
    cand_s = str(cand_id) if cand_id is not None else "NA"

    # Output dir: participant/program-specific folder
    base_folder = f"mixed_gambles_p{args.participant_id}_iter{iter_s}_cand{cand_s}"
    if args.tag:
        base_folder += f"_{args.tag}"
    out_dir = Path(args.output_dir) / base_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build splits list and filename prefix per split
    splits = []
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
        # Filename: when split=none use short names; otherwise use split prefix
        if args.split == "none":
            file_prefix = ""
        else:
            file_prefix = f"{split_name}_"  # train_ or test_
        # Fig 4.3
        plot_fig43_behavior(
            trials,
            str(out_dir / f"{file_prefix}fig43_behavior.png"),
            gain_max=float(args.gain_max),
            loss_max=float(args.loss_max),
            dpi=args.dpi,
        )
        # Fig 4.5
        plot_fig45_design_heatmap(
            trials,
            choose_fn,
            str(out_dir / f"{file_prefix}fig45_design_heatmap.png"),
            gain_max=float(args.gain_max),
            loss_max=float(args.loss_max),
            dpi=args.dpi,
        )
        # Fig 4.6
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

    # Metadata
    n_train, n_test = len(train_trials), len(test_trials)
    train_accept = sum(1 for t in train_trials if t["action"] == ACCEPT_ACTION)
    test_accept = sum(1 for t in test_trials if t["action"] == ACCEPT_ACTION)
    # Program acceptance on train (for self-check)
    train_pred_accept = 0
    for t in train_trials:
        try:
            p = choose_fn(t["problem"], t["history"])
            if p is not None and p == ACCEPT_ACTION:
                train_pred_accept += 1
        except Exception:
            pass
    meta = {
        "participant_id": args.participant_id,
        "split": args.split,
        "effective_split": args.split,
        "program_path": str(program_path),
        "iteration_id": iter_id,
        "candidate_id": cand_id,
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
        meta["n_total"] = len(train_trials) + len(test_trials)
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Self-check print
    unique_gl_train = len(set((round(trial_to_gl(t)[0], 4), round(trial_to_gl(t)[1], 4)) for t in train_trials))
    unique_gl_test = len(set((round(trial_to_gl(t)[0], 4), round(trial_to_gl(t)[1], 4)) for t in test_trials))
    print("Trials loaded: train =", n_train, ", test =", n_test)
    if args.split == "none":
        print("Effective split: none (all trials combined), n_total =", meta["n_total"])
    print("Unique (G,L) points: train =", unique_gl_train, ", test =", unique_gl_test)
    print("Participant accept rate: train =", f"{train_accept/n_train:.3f}" if n_train else "N/A", ", test =", f"{test_accept/n_test:.3f}" if n_test else "N/A")
    print("Program accept rate (train) =", f"{train_pred_accept/n_train:.3f}" if n_train else "N/A")
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
