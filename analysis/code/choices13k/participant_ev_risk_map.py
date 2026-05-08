#!/usr/bin/env python3
"""Qualitative reward-space decision map for evolved Choices13k participant programs."""

## Use argument --participant_id 2 or 4 to plot the decision map for participant 2 or 4.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def expected_utility(gamble: Dict[str, Any]) -> float:
    probs = gamble["probs"]
    rewards = gamble["rewards"]
    return float(sum(float(p) * float(r) for p, r in zip(probs, rewards)))


def variance(gamble: Dict[str, Any]) -> float:
    ev = expected_utility(gamble)
    probs = gamble["probs"]
    rewards = gamble["rewards"]
    return float(sum(float(p) * (float(r) - ev) ** 2 for p, r in zip(probs, rewards)))


def positive_reward(gamble: Dict[str, Any]) -> float:
    vals = [float(r) for r in gamble["rewards"] if float(r) > 0]
    return max(vals) if vals else 0.0


def negative_reward(gamble: Dict[str, Any]) -> float:
    vals = [abs(float(r)) for r in gamble["rewards"] if float(r) < 0]
    return max(vals) if vals else 0.0


def choose_participant_2(problem: Dict[str, Any], history: Sequence[Dict[str, Any]]) -> float:
    """Embedded final evolved program for participant 2."""
    util_A = expected_utility(problem["gamble_A"])
    util_B = expected_utility(problem["gamble_B"])
    var_A = variance(problem["gamble_A"])
    var_B = variance(problem["gamble_B"])

    base_prob_B = 0.5 + 0.5 * (util_B - util_A) / (abs(util_A) + abs(util_B) + 1e-6)

    variance_adjustment = (var_A - var_B) / (var_A + var_B + 1e-6)
    base_prob_B += 0.2 * variance_adjustment

    if history:
        recent_actions = [int(h["action"]) for h in history[-5:]]
        count_B = recent_actions.count(1)
        count_A = recent_actions.count(0)
        if count_B > count_A:
            base_prob_B += 0.3
        elif count_A > count_B:
            base_prob_B -= 0.3

        if int(history[-1]["action"]) == 1:
            base_prob_B += 0.2
        elif int(history[-1]["action"]) == 0:
            base_prob_B -= 0.2

    if problem.get("has_feedback", False) and history:
        last_feedback = history[-1].get("feedback")
        if last_feedback is not None:
            if float(last_feedback) > 0:
                base_prob_B += 0.15
            else:
                base_prob_B -= 0.15

    if util_B < util_A:
        base_prob_B *= 0.8
    else:
        base_prob_B *= 1.2

    return max(1e-6, min(1 - 1e-6, float(base_prob_B)))


def choose_participant_4(problem: Dict[str, Any], history: Sequence[Dict[str, Any]]) -> float:
    """Embedded final evolved program for participant 4."""

    def expected_utility_p4(gamble: Dict[str, Any]) -> float:
        if gamble["probs"] is None:
            return sum(gamble["rewards"]) / len(gamble["rewards"])
        return sum(float(p) * float(r) for p, r in zip(gamble["probs"], gamble["rewards"]))

    utility_A = expected_utility_p4(problem["gamble_A"])
    utility_B = expected_utility_p4(problem["gamble_B"])

    if utility_B > utility_A:
        base_prob = 0.8
    elif utility_B < utility_A:
        base_prob = 0.2
    else:
        base_prob = 0.5

    b_weight = 1.0
    a_weight = 1.0
    for entry in history:
        if int(entry["action"]) == 0:
            a_weight += 1.0
        else:
            b_weight += 1.0

    weighted_ev_A = utility_A * a_weight
    weighted_ev_B = utility_B * b_weight
    total_weighted_ev = weighted_ev_A + weighted_ev_B
    if total_weighted_ev > 0:
        prob_B = weighted_ev_B / total_weighted_ev
    else:
        prob_B = base_prob

    if history:
        count_A = sum(1 for entry in history if int(entry["action"]) == 0)
        count_B = sum(1 for entry in history if int(entry["action"]) == 1)
        total_actions = count_A + count_B
        if total_actions > 0:
            freq_B = count_B / total_actions
            prob_B = (prob_B + freq_B) / 2.0

    return max(1e-6, min(1 - 1e-6, float(prob_B)))


def make_problem_from_delta_rewards(
    delta_positive: float,
    delta_negative: float,
    has_feedback: bool = True,
) -> Dict[str, Any]:
    """Construct synthetic problem in delta reward space."""
    gamble_A = {"probs": [1.0], "rewards": [0.0]}
    b_positive = max(0.0, float(delta_positive))
    b_negative = max(0.0, float(delta_negative))
    gamble_B = {"probs": [0.5, 0.5], "rewards": [-b_negative, b_positive]}

    return {
        "gamble_A": gamble_A,
        "gamble_B": gamble_B,
        "option_keys": ["A", "B"],
        "has_feedback": has_feedback,
    }


def load_trials(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def trial_delta_rewards(trial: Dict[str, Any]) -> Tuple[float, float, int]:
    gA = trial["problem"]["gamble_A"]
    gB = trial["problem"]["gamble_B"]
    delta_positive = positive_reward(gB) - positive_reward(gA)
    delta_negative = negative_reward(gB) - negative_reward(gA)
    action = int(trial["action"])  # 1 -> B, 0 -> A
    return delta_positive, delta_negative, action


def build_prediction_grid(
    choose_fn,
    history: Sequence[Dict[str, Any]],
    x_min: float = -40.0,
    x_max: float = 40.0,
    y_min: float = -40.0,
    y_max: float = 40.0,
    x_n: int = 321,
    y_n: int = 321,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vals = np.linspace(x_min, x_max, x_n)
    y_vals = np.linspace(y_min, y_max, y_n)
    Z = np.zeros((y_n, x_n), dtype=float)
    for y_idx, delta_negative in enumerate(y_vals):
        for x_idx, delta_positive in enumerate(x_vals):
            problem = make_problem_from_delta_rewards(float(delta_positive), float(delta_negative))
            Z[y_idx, x_idx] = float(choose_fn(problem, history))
    return x_vals, y_vals, Z


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot participant reward-space decision map for Choices13k.")
    parser.add_argument("--participant_id", type=int, default=2)
    parser.add_argument("--train_trials_json", type=Path, default=None)
    parser.add_argument("--test_trials_json", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/analysis_plot/proposal"),
    )
    parser.add_argument("--save_pdf", action="store_true")
    args = parser.parse_args()

    data_dir = Path("analysis/data/choices13k")
    train_trials_json = (
        args.train_trials_json
        if args.train_trials_json is not None
        else data_dir / f"participant_{args.participant_id}_train_trials.json"
    )
    test_trials_json = (
        args.test_trials_json
        if args.test_trials_json is not None
        else data_dir / f"participant_{args.participant_id}_test_trials.json"
    )

    if args.participant_id == 2:
        choose_fn = choose_participant_2
        fixed_history = [
            {"action": 0, "feedback": -1.0},
            {"action": 0, "feedback": -1.0},
            {"action": 0, "feedback": -1.0},
        ]
    elif args.participant_id == 4:
        choose_fn = choose_participant_4
        fixed_history = [
            {"action": 1, "feedback": 1.0},
            {"action": 1, "feedback": 1.0},
            {"action": 1, "feedback": 1.0},
        ]
    else:
        raise ValueError("Embedded programs are available for participant_id in {2, 4}.")

    train_trials = load_trials(train_trials_json)
    test_trials = load_trials(test_trials_json)
    all_trials = train_trials + test_trials
    points = np.array([trial_delta_rewards(t) for t in all_trials], dtype=float)
    x_delta_positive = points[:, 0]
    y_delta_negative = points[:, 1]
    actions = points[:, 2].astype(int)

    # Use observed data bounds with padding for a tighter, cleaner view.
    x_min = float(np.min(x_delta_positive))
    x_max = float(np.max(x_delta_positive))
    y_min = float(np.min(y_delta_negative))
    y_max = float(np.max(y_delta_negative))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_pad = 0.12 * x_span
    y_pad = 0.12 * y_span
    grid_x_min = x_min - x_pad
    grid_x_max = x_max + x_pad
    grid_y_min = y_min - y_pad
    grid_y_max = y_max + y_pad
    x_vals, y_vals, Z = build_prediction_grid(
        choose_fn,
        fixed_history,
        x_min=grid_x_min,
        x_max=grid_x_max,
        y_min=grid_y_min,
        y_max=grid_y_max,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.output_dir / f"participant{args.participant_id}_reward_map.png"
    out_pdf = args.output_dir / f"participant{args.participant_id}_reward_map.pdf"

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]
    display_Z = np.clip(Z, 0.02, 0.98)
    im = ax.imshow(
        display_Z,
        origin="lower",
        extent=extent,
        cmap="RdYlGn",
        interpolation="bilinear",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        alpha=0.45,
    )
    mask_B = actions == 1
    mask_A = actions == 0
    sc_a = ax.scatter(
        x_delta_positive[mask_A],
        y_delta_negative[mask_A],
        marker="x",
        s=70,
        c="red",
        alpha=0.85,
        linewidths=1.5,
        label="Actual A",
    )
    sc_b = ax.scatter(
        x_delta_positive[mask_B],
        y_delta_negative[mask_B],
        marker="o",
        s=80,
        facecolors="none",
        edgecolors="green",
        alpha=0.88,
        linewidths=1.8,
        label="Actual B",
    )
    ax.contour(
        x_vals,
        y_vals,
        Z,
        levels=[0.5],
        colors=["#2f2f2f"],
        linewidths=1.0,
        alpha=0.7,
    )
    ax.set_title(f"Participant {args.participant_id}: reward-space decision map", fontsize=12)
    ax.set_xlabel("ΔPositiveReward", fontsize=10)
    ax.set_ylabel("ΔNegativeReward", fontsize=10)
    ax.set_xlim(grid_x_min, grid_x_max)
    ax.set_ylim(grid_y_min, grid_y_max)
    ax.legend(
        handles=[sc_a, sc_b],
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="#dddddd",
        fontsize=9,
    )
    ax.grid(True, alpha=0.12, linewidth=0.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("Predicted P(B)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if args.save_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    if args.save_pdf:
        print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
