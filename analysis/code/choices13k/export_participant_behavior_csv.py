#!/usr/bin/env python3
"""Export participant-level behavior evidence CSVs for proposal workflows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_OURS_DIR = Path("generated_outputs/choice13k/non_strict/run_260430_013702")
DEFAULT_CENTAUR_DIR = Path("generated_outputs/choice13k/centaur/run_260416_000815")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def expected_value(gamble: Dict[str, Any]) -> float:
    probs = gamble.get("probs")
    rewards = gamble["rewards"]
    if probs is None:
        return float(sum(float(r) for r in rewards) / len(rewards))
    return float(sum(float(p) * float(r) for p, r in zip(probs, rewards)))


def variance(gamble: Dict[str, Any]) -> float:
    ev = expected_value(gamble)
    probs = gamble.get("probs")
    rewards = [float(r) for r in gamble["rewards"]]
    if probs is None:
        return float(sum((r - ev) ** 2 for r in rewards) / len(rewards))
    return float(sum(float(p) * (r - ev) ** 2 for p, r in zip(probs, rewards)))


def choose_participant_2(problem: Dict[str, Any], history: Sequence[Dict[str, Any]]) -> float:
    """Finalized evolved program for participant 2."""
    util_a = expected_value(problem["gamble_A"])
    util_b = expected_value(problem["gamble_B"])
    var_a = variance(problem["gamble_A"])
    var_b = variance(problem["gamble_B"])

    prob_b = 0.5 + 0.5 * (util_b - util_a) / (abs(util_a) + abs(util_b) + 1e-6)
    prob_b += 0.2 * ((var_a - var_b) / (var_a + var_b + 1e-6))

    if history:
        recent_actions = [int(h["action"]) for h in history[-5:]]
        count_b = recent_actions.count(1)
        count_a = recent_actions.count(0)
        if count_b > count_a:
            prob_b += 0.3
        elif count_a > count_b:
            prob_b -= 0.3

        if int(history[-1]["action"]) == 1:
            prob_b += 0.2
        else:
            prob_b -= 0.2

    if bool(problem.get("has_feedback", False)) and history:
        last_feedback = history[-1].get("feedback")
        if last_feedback is not None:
            if float(last_feedback) > 0:
                prob_b += 0.15
            else:
                prob_b -= 0.15

    if util_b < util_a:
        prob_b *= 0.8
    else:
        prob_b *= 1.2

    return float(max(1e-6, min(1 - 1e-6, prob_b)))


def choose_participant_4(problem: Dict[str, Any], history: Sequence[Dict[str, Any]]) -> float:
    """Finalized evolved program for participant 4."""
    utility_a = expected_value(problem["gamble_A"])
    utility_b = expected_value(problem["gamble_B"])

    if utility_b > utility_a:
        base_prob = 0.8
    elif utility_b < utility_a:
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

    weighted_ev_a = utility_a * a_weight
    weighted_ev_b = utility_b * b_weight
    denom = weighted_ev_a + weighted_ev_b
    if denom > 0:
        prob_b = weighted_ev_b / denom
    else:
        prob_b = base_prob

    if history:
        count_a = sum(1 for entry in history if int(entry["action"]) == 0)
        count_b = sum(1 for entry in history if int(entry["action"]) == 1)
        total_actions = count_a + count_b
        if total_actions > 0:
            freq_b = count_b / total_actions
            prob_b = (prob_b + freq_b) / 2.0

    return float(max(1e-6, min(1 - 1e-6, prob_b)))


def problem_key(problem: Dict[str, Any]) -> str:
    return json.dumps(problem, sort_keys=True)


def load_trials(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_trials_path(participant_id: int, split: str) -> Path:
    name = f"participant_{participant_id}_{split}_trials.json"
    root = repo_root()
    candidates = [
        root / "analysis" / "code" / "choices13k" / name,
        root / "analysis" / "data" / "choices13k" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {name}. Checked: " + ", ".join(str(c) for c in candidates)
    )


def binary_loglik(prob_b: float, action: int) -> float:
    p = min(max(float(prob_b), 1e-12), 1.0 - 1e-12)
    return math.log(p) if int(action) == 1 else math.log(1.0 - p)


def compute_sequence_rows(
    participant_id: int,
    split: str,
    trials: List[Dict[str, Any]],
    choose_fn: Callable[[Dict[str, Any], Sequence[Dict[str, Any]]], float],
) -> Tuple[List[Dict[str, Any]], float]:
    rows: List[Dict[str, Any]] = []
    loglik_values: List[float] = []
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    order: List[str] = []

    for idx, trial in enumerate(trials):
        key = problem_key(trial["problem"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((idx, trial))

    for key in order:
        block = grouped[key]
        prev_action: Optional[int] = None
        for i, (_, trial) in enumerate(block, start=1):
            action = int(trial["action"])
            actual_label = "B" if action == 1 else "A"
            problem = trial["problem"]
            has_feedback = bool(problem.get("has_feedback", False))
            feedback = trial.get("feedback")
            history = trial.get("history", [])

            pred_prob_b = choose_fn(problem, history)
            pred_label = "B" if pred_prob_b >= 0.5 else "A"
            loglik_values.append(binary_loglik(pred_prob_b, action))

            if prev_action is None:
                repeat_prev = ""
            else:
                repeat_prev = bool(action == prev_action)

            rows.append(
                {
                    "participant_id": participant_id,
                    "split": split,
                    "problem_id": key,
                    "trial_within_problem": i,
                    "actual_action": action,
                    "actual_action_label": actual_label,
                    "predicted_prob_B": pred_prob_b,
                    "predicted_choice_label": pred_label,
                    "has_feedback": has_feedback,
                    "feedback": feedback,
                    "repeat_previous_action": repeat_prev,
                    "gambleA_expected_value": expected_value(problem["gamble_A"]),
                    "gambleB_expected_value": expected_value(problem["gamble_B"]),
                    "gambleA_variance": variance(problem["gamble_A"]),
                    "gambleB_variance": variance(problem["gamble_B"]),
                }
            )
            prev_action = action

    return rows, float(sum(loglik_values) / len(loglik_values))


def repeat_stats(trials: List[Dict[str, Any]]) -> Dict[str, float]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for trial in trials:
        key = problem_key(trial["problem"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(trial)

    total_pairs = 0
    repeats = 0
    switches = 0
    neg_total = 0
    neg_repeats = 0
    nofb_pairs = 0
    nofb_repeats = 0

    for key in order:
        block = grouped[key]
        if len(block) < 2:
            continue
        has_feedback = bool(block[0]["problem"].get("has_feedback", False))
        for idx in range(1, len(block)):
            prev = block[idx - 1]
            cur = block[idx]
            prev_action = int(prev["action"])
            cur_action = int(cur["action"])
            is_repeat = cur_action == prev_action

            total_pairs += 1
            if is_repeat:
                repeats += 1
            else:
                switches += 1

            if not has_feedback:
                nofb_pairs += 1
                if is_repeat:
                    nofb_repeats += 1

            prev_feedback = prev.get("feedback")
            if prev_feedback is not None and float(prev_feedback) < 0:
                neg_total += 1
                if is_repeat:
                    neg_repeats += 1

    def safe_rate(n: int, d: int) -> float:
        return float(n) / float(d) if d > 0 else float("nan")

    return {
        "repeat_rate": safe_rate(repeats, total_pairs),
        "switch_count": float(switches),
        "after_negative_repeat_rate": safe_rate(neg_repeats, neg_total),
        "no_feedback_repeat_rate": safe_rate(nofb_repeats, nofb_pairs),
    }


def load_test_loglik_map(run_dir: Path) -> Dict[int, float]:
    csv_path = run_dir / "participant_details_loglik.csv"
    df = pd.read_csv(csv_path)
    return {int(row["participant_id"]): float(row["test_loglik"]) for _, row in df.iterrows()}


def main() -> None:
    root = repo_root()
    out_dir = root / "analysis" / "analysis_plot" / "proposal"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "participant_summary.csv"
    sequence_path = out_dir / "participant_sequences.csv"

    participant_meta = {
        2: {
            "choose_fn": choose_participant_2,
            "program_mechanism": "Recent-action persistence: last-5 majority ±0.3; last action ±0.2; feedback ±0.15",
            "main_interpretation": "Strong within-problem persistence; recent actions can dominate EV.",
        },
        4: {
            "choose_fn": choose_participant_4,
            "program_mechanism": "Choice-count persistence: past action counts used twice; no feedback variable used",
            "main_interpretation": "Self-reinforcing choice habit; repeats choices even after bad outcomes.",
        },
    }

    ours_test_loglik_map = load_test_loglik_map(root / DEFAULT_OURS_DIR)
    centaur_test_loglik_map = load_test_loglik_map(root / DEFAULT_CENTAUR_DIR)

    all_sequence_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for participant_id, meta in participant_meta.items():
        train_trials = load_trials(resolve_trials_path(participant_id, "train"))
        test_trials = load_trials(resolve_trials_path(participant_id, "test"))
        choose_fn = meta["choose_fn"]

        train_rows, _ = compute_sequence_rows(participant_id, "train", train_trials, choose_fn)
        test_rows, test_loglik_program = compute_sequence_rows(participant_id, "test", test_trials, choose_fn)
        all_sequence_rows.extend(train_rows)
        all_sequence_rows.extend(test_rows)

        train_stats = repeat_stats(train_trials)
        test_stats = repeat_stats(test_trials)

        ours_test_loglik = ours_test_loglik_map.get(participant_id, test_loglik_program)
        centaur_test_loglik = centaur_test_loglik_map.get(participant_id)
        if centaur_test_loglik is None:
            raise KeyError(f"Missing centaur test loglik for participant {participant_id}")

        summary_rows.append(
            {
                "participant_id": participant_id,
                "ours_test_loglik": ours_test_loglik,
                "centaur_test_loglik": centaur_test_loglik,
                "heldout_delta": ours_test_loglik - centaur_test_loglik,
                "train_repeat_rate": train_stats["repeat_rate"],
                "test_repeat_rate": test_stats["repeat_rate"],
                "train_switch_count": int(train_stats["switch_count"]),
                "test_switch_count": int(test_stats["switch_count"]),
                "train_after_negative_repeat_rate": train_stats["after_negative_repeat_rate"],
                "test_after_negative_repeat_rate": test_stats["after_negative_repeat_rate"],
                "train_no_feedback_repeat_rate": train_stats["no_feedback_repeat_rate"],
                "test_no_feedback_repeat_rate": test_stats["no_feedback_repeat_rate"],
                "program_mechanism": meta["program_mechanism"],
                "main_interpretation": meta["main_interpretation"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("participant_id").reset_index(drop=True)
    seq_df = pd.DataFrame(all_sequence_rows).reset_index(drop=True)

    summary_df.to_csv(summary_path, index=False)
    seq_df.to_csv(sequence_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Shape: {summary_df.shape}")
    print(summary_df.head())
    print()
    print(f"Saved: {sequence_path}")
    print(f"Shape: {seq_df.shape}")
    print(seq_df.head())


if __name__ == "__main__":
    main()
