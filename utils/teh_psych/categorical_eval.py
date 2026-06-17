"""
Categorical choose(problem, history) -> dict[int, float] evaluation.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

LOGLIK_EPS = 1e-9


def valid_action_ids_from_problem(problem: Dict[str, Any]) -> List[int]:
    options = problem.get("options") or []
    ids: List[int] = []
    for opt in options:
        if isinstance(opt, dict) and "action" in opt:
            ids.append(int(opt["action"]))
    return ids


def coerce_choose_output(
    probs_raw: Any,
    valid_action_ids: List[int],
) -> Tuple[Dict[int, float], List[str]]:
    """
    Normalize choose() output to a probability dict over valid_action_ids.

    Returns (probs_dict, warnings).
    """
    warnings: List[str] = []
    K = len(valid_action_ids)
    if K < 1:
        raise ValueError("valid_action_ids must be non-empty")

    expected = set(valid_action_ids)

    if isinstance(probs_raw, dict):
        raw_dict = probs_raw
    elif K == 2 and isinstance(probs_raw, (float, int, np.integer, bool)):
        p1 = float(probs_raw)
        if not math.isfinite(p1):
            warnings.append(f"legacy float return non-finite: {probs_raw!r}")
            p1 = 0.5
        p1 = min(max(p1, 0.0), 1.0)
        raw_dict = {0: 1.0 - p1, 1: p1}
    else:
        warnings.append(
            f"choose() returned {type(probs_raw).__name__}, expected dict[int,float] for K={K}"
        )
        uniform = 1.0 / K
        return {aid: uniform for aid in valid_action_ids}, warnings

    probs: Dict[int, float] = {aid: 0.0 for aid in valid_action_ids}
    for key, val in raw_dict.items():
        try:
            aid = int(key)
        except (TypeError, ValueError):
            continue
        if aid not in expected:
            continue
        try:
            p = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(p) and p >= 0.0:
            probs[aid] = p

    total = sum(probs.values())
    if total <= 0.0:
        warnings.append("all probabilities invalid or sum to 0; using uniform fallback")
        uniform = 1.0 / K
        return {aid: uniform for aid in valid_action_ids}, warnings

    return {aid: probs[aid] / total for aid in valid_action_ids}, warnings


def _clamp_prob(p: float) -> float:
    return min(max(float(p), LOGLIK_EPS), 1.0 - LOGLIK_EPS)


def evaluate_categorical_program(
    choose_fn: Callable,
    trials: List[Dict[str, Any]],
    *,
    verbose: bool = False,
    n_seeds: int = 1,
    compute_accuracy: bool = True,
) -> Dict[str, Any]:
    """Evaluate categorical log-likelihood: log p[target_action] per trial."""
    total = len(trials)
    seed_avg_logliks: List[float] = []
    seed_avg_accs: List[float] = []
    total_warnings = 0
    total_errors = 0
    first_error: Optional[str] = None

    def _one_pass(seed_idx: int) -> Tuple[float, float, int, int]:
        nonlocal first_error
        loglik_acc = 0.0
        correct = 0
        errors = 0
        warnings = 0
        for t in trials:
            problem = t.get("problem") or {}
            target = t.get("target_action", t.get("action"))
            if target is None:
                errors += 1
                continue
            y = int(target)
            valid_ids = valid_action_ids_from_problem(problem)
            if not valid_ids:
                errors += 1
                continue
            try:
                probs_raw = choose_fn(problem, t.get("history", []))
                probs, coerce_warnings = coerce_choose_output(probs_raw, valid_ids)
                warnings += len(coerce_warnings)
                if verbose and coerce_warnings and seed_idx == 0:
                    for w in coerce_warnings[:2]:
                        print(f"  Eval warning: {w}")
            except Exception as exc:
                errors += 1
                if first_error is None:
                    first_error = str(exc)
                if verbose and errors <= 3 and seed_idx == 0:
                    print(f"  Evaluation error: {exc}")
                K = len(valid_ids)
                uniform = 1.0 / K
                probs = {aid: uniform for aid in valid_ids}
                warnings += 1

            p_y = probs.get(y, 0.0)
            p_clamped = _clamp_prob(p_y)
            loglik_acc += math.log(p_clamped)
            if compute_accuracy:
                pred = max(probs, key=probs.get)
                correct += int(pred == y)

        avg_ll = loglik_acc / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_ll, acc, errors, warnings

    for seed in range(max(1, n_seeds)):
        ll, acc, errs, warns = _one_pass(seed)
        seed_avg_logliks.append(ll)
        seed_avg_accs.append(acc)
        total_errors += errs
        total_warnings += warns

    n_seeds_used = max(1, n_seeds)
    return {
        "avg_loglik": float(np.mean(seed_avg_logliks)),
        "avg_accuracy": float(np.mean(seed_avg_accs)),
        "errors": total_errors // n_seeds_used,
        "warnings": total_warnings // n_seeds_used,
        "n_trials": total,
        "first_error": first_error,
    }
