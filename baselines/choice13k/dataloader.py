from typing import List, Dict, Any, Tuple
from data_modules.choice13k import get_choice13k_experiments


def load_choice13k(args) -> List[Dict[str, Any]]:
    """Load experiments for requested participants."""
    return get_choice13k_experiments(n_participants=args.num_agents_to_sample)


def split_trials(exp) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], list]:
    """Split trials into train/test 80/20 (fixed split, matching ROTE); return (train, test, options)."""
    options = exp.blocks[0].option_keys
    all_trials = []
    history_accum = []
    for block in exp.blocks:
        for trial in block.trials:
            history_entry = {"action": trial.action, "feedback": trial.feedback}
            all_trials.append(
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
            history_accum.append(history_entry)
    # Fixed 80:20 split (matching ROTE's approach in plot_and_eval.py)
    split_point = int(len(all_trials) * 0.8)
    train_trials = all_trials[:split_point]
    test_trials = all_trials[split_point:]
    return train_trials, test_trials, options

