"""Internal train/val/test split for pooled categorical trials."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


def _trial_group_key(trial: Dict[str, Any]) -> str:
    meta = trial.get("_meta") or {}
    row_idx = meta.get("row_index", 0)
    block_id = meta.get("block_id", 0)
    return f"row{row_idx}_block{block_id}"


def split_pooled_categorical_trials(
    trials: List[Dict[str, Any]],
    *,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split pooled prediction trials by (participant row, block) groups.

    Callers should pass only prediction-target trials. Context-only trials are
    excluded defensively if any slip through.
    """
    prediction_trials = [t for t in trials if t.get("is_prediction_target", True)]
    if not prediction_trials:
        return [], [], []

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in prediction_trials:
        groups[_trial_group_key(t)].append(t)

    group_keys = sorted(groups.keys())
    n_groups = len(group_keys)
    if n_groups < 3:
        raise ValueError(
            f"Internal split requires at least 3 trial groups (row/block); got {n_groups}"
        )
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")

    rng = np.random.default_rng(split_seed)
    perm = list(group_keys)
    rng.shuffle(perm)

    n_train = max(1, int(n_groups * split_ratio))
    n_train = min(n_train, n_groups - 2)
    n_rem = n_groups - n_train
    n_val = max(1, (n_rem + 1) // 2)
    n_test = max(1, n_rem - n_val)
    if n_train + n_val + n_test > n_groups:
        n_test = max(1, n_groups - n_train - n_val)

    train_keys = set(perm[:n_train])
    val_keys = set(perm[n_train : n_train + n_val])
    test_keys = set(perm[n_train + n_val : n_train + n_val + n_test])

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for key, group_trials in groups.items():
        if key in train_keys:
            train.extend(group_trials)
        elif key in val_keys:
            val.extend(group_trials)
        elif key in test_keys:
            test.extend(group_trials)

    if not train or not val or not test:
        raise ValueError(
            f"Internal split produced empty partition: train={len(train)}, "
            f"val={len(val)}, test={len(test)}"
        )
    return train, val, test
