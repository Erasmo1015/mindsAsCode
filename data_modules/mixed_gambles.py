"""
Local mixed-gambles benchmark loader (not in Psych-101).

Trials use the same structured schema as choice13k-style binary loglik:
  choose(problem, history) -> float = P(action=1)
  action=0: gamble / Option A; action=1: certain / Option B.
"""
from __future__ import annotations

import csv
from typing import Any, Dict, List, Tuple

import numpy as np

DEFAULT_CSV_PATH = "datasets/mixed_gambles/data_all_2021-01-08.csv"
DATASET_NAME = "mixed_gambles"


def three_way_unit_counts(n_units: int, split_ratio: float) -> Tuple[int, int, int]:
    """Train/val/test unit counts (same rules as te_dr / Centaur split_trials)."""
    if n_units < 3:
        raise ValueError(
            f"train/val/test split requires at least 3 units; got {n_units}."
        )
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")

    n_train = int(n_units * split_ratio)
    n_train = max(1, min(n_train, n_units - 2))
    n_rem = n_units - n_train
    n_val = (n_rem + 1) // 2
    n_test = n_rem - n_val
    if n_val < 1:
        n_val = 1
        n_test = max(1, n_rem - 1)
        n_train = n_units - n_val - n_test
        n_train = max(1, n_train)
    return n_train, n_val, n_test


def load_mixed_gambles_trials(
    participant_id: int,
    *,
    csv_path: str = DEFAULT_CSV_PATH,
    filter_gain_loss_only: bool = False,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """
    Load mixed_gambles CSV for subject == participant_id; split by unique (gain, loss, cert).

    Returns (train_trials, val_trials, test_trials, option_keys).
    """
    option_keys = [0, 1]
    all_trials: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["subject"]) != participant_id:
                continue
            if filter_gain_loss_only and row.get("gamble_type") != "gain_loss":
                continue
            gain, loss, cert = float(row["gain"]), float(row["loss"]), float(row["cert"])
            took_gamble = int(row["took_gamble"])
            action = 1 - took_gamble
            all_trials.append(
                {
                    "problem": {
                        "gamble_A": {"rewards": [gain, loss], "probs": [0.5, 0.5]},
                        "gamble_B": {"rewards": [cert], "probs": [1.0]},
                        "option_keys": option_keys,
                        "has_feedback": False,
                        "dataset_alias": DATASET_NAME,
                    },
                    "history": [],
                    "options": option_keys,
                    "action": action,
                    "problem_signature": (gain, loss, cert),
                }
            )
    if not all_trials:
        raise ValueError(f"No rows found for subject {participant_id} in {csv_path}")

    if filter_gain_loss_only and not getattr(load_mixed_gambles_trials, "_printed_gain_loss", False):
        print("[Mixed Gambles] Using gain_loss trials only.")
        load_mixed_gambles_trials._printed_gain_loss = True  # type: ignore[attr-defined]

    signatures = sorted({t["problem_signature"] for t in all_trials})
    if len(signatures) < 3:
        raise ValueError(
            f"mixed_gambles participant {participant_id} has <3 unique problem signatures; "
            "cannot build disjoint train/val/test."
        )
    rng = np.random.default_rng(split_seed)
    shuffled = list(signatures)
    rng.shuffle(shuffled)
    n_train, n_val, n_test = three_way_unit_counts(len(shuffled), split_ratio)
    train_sigs = set(shuffled[:n_train])
    val_sigs = set(shuffled[n_train : n_train + n_val])
    test_sigs = set(shuffled[n_train + n_val :])
    train_trials = [t for t in all_trials if t["problem_signature"] in train_sigs]
    val_trials = [t for t in all_trials if t["problem_signature"] in val_sigs]
    test_trials = [t for t in all_trials if t["problem_signature"] in test_sigs]
    for t in train_trials + val_trials + test_trials:
        t.pop("problem_signature", None)
    return train_trials, val_trials, test_trials, option_keys


# Backward-compatible alias used by collect_participant_ids.py
load_mixed_gambles_data = load_mixed_gambles_trials
