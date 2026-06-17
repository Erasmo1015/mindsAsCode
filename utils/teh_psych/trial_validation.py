"""Validation helpers for categorical Psych-101 trials."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from utils.teh_psych.categorical_eval import valid_action_ids_from_problem


def summarize_trial_action_space(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    ks: List[int] = []
    for t in trials:
        ids = valid_action_ids_from_problem(t.get("problem") or {})
        if ids:
            ks.append(len(ids))
    if not ks:
        return {
            "num_actions_min": None,
            "num_actions_max": None,
            "is_variable_k": False,
            "n_trials": len(trials),
        }
    return {
        "num_actions_min": min(ks),
        "num_actions_max": max(ks),
        "is_variable_k": min(ks) != max(ks),
        "n_trials": len(trials),
    }


def validate_categorical_trials(
    trials: List[Dict[str, Any]],
    *,
    min_options: int = 2,
) -> Tuple[List[str], int]:
    """
    Validate categorical trials.

    Returns (error_messages, n_prediction_trials).

    A prediction trial has options, valid consecutive action ids, and valid target.
    """
    errors: List[str] = []
    n_prediction = 0
    for i, t in enumerate(trials):
        problem = t.get("problem") or {}
        options = problem.get("options")
        if not isinstance(options, list) or len(options) < min_options:
            errors.append(f"trial {i}: problem['options'] missing or has < {min_options} options")
            continue
        action_ids: List[int] = []
        for j, opt in enumerate(options):
            if not isinstance(opt, dict) or "action" not in opt:
                errors.append(f"trial {i}: option {j} missing action id")
                action_ids = []
                break
            action_ids.append(int(opt["action"]))
        if not action_ids:
            continue
        expected = list(range(len(action_ids)))
        if sorted(action_ids) != expected:
            errors.append(
                f"trial {i}: action ids must be consecutive 0..{len(action_ids)-1}, got {action_ids}"
            )
            continue
        target = t.get("target_action", t.get("action"))
        if target is None:
            errors.append(f"trial {i}: missing target_action/action")
            continue
        y = int(target)
        if y not in action_ids:
            errors.append(f"trial {i}: target_action {y} not in valid actions {action_ids}")
            continue
        n_prediction += 1
    return errors, n_prediction
