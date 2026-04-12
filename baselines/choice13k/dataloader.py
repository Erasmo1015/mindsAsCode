from typing import List, Dict, Any, Tuple

import numpy as np

from data_modules.choice13k import get_choice13k_experiments


def load_choice13k(args) -> List[Dict[str, Any]]:
    """Load experiments for requested participants."""
    return get_choice13k_experiments(n_participants=args.num_agents_to_sample)


def trials_from_blocks_chronological(exp, block_indices: set) -> List[Dict[str, Any]]:
    """Trials from selected blocks in original order; history only within each block (no cross-problem leakage)."""
    out: List[Dict[str, Any]] = []
    for bi, block in enumerate(exp.blocks):
        if bi not in block_indices:
            continue
        options = block.option_keys
        history_accum: List[Dict[str, Any]] = []
        for trial in block.trials:
            out.append(
                {
                    "problem": {
                        "gamble_A": {
                            "probs": block.gamble_A.probs,
                            "rewards": block.gamble_A.rewards,
                        },
                        "gamble_B": {
                            "probs": block.gamble_B.probs,
                            "rewards": block.gamble_B.rewards,
                        },
                        "option_keys": options,
                        "has_feedback": block.has_feedback,
                    },
                    "history": list(history_accum),
                    "options": options,
                    "action": trial.action,
                }
            )
            history_accum.append({"action": trial.action, "feedback": trial.feedback})
    return out


def split_trials(exp) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], list]:
    """Split by **problem (block)** 80/20, seed 42; disjoint gamble pairs (aligned with Template_evo)."""
    n_blocks = len(exp.blocks)
    if n_blocks < 2:
        raise ValueError(
            f"Choice13k within-participant split requires at least 2 problems (blocks); got {n_blocks}."
        )
    rng = np.random.default_rng(42)
    perm = np.arange(n_blocks)
    rng.shuffle(perm)
    split_idx = int(n_blocks * 0.8)
    split_idx = max(1, min(split_idx, n_blocks - 1))
    train_blocks = set(perm[:split_idx].tolist())
    test_blocks = set(perm[split_idx:].tolist())
    train_trials = trials_from_blocks_chronological(exp, train_blocks)
    test_trials = trials_from_blocks_chronological(exp, test_blocks)
    options = exp.blocks[0].option_keys
    return train_trials, test_trials, options

