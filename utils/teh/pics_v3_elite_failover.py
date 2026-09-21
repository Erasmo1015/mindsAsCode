"""PICS v3 held-out test execution: rank-1 + uniform fallback (OE-matched default).

Default (``structure_aware_v3``): evaluate only the frozen rank-1 program. On
exception or invalid/non-finite returned probability for a test trial, emit
uniform ``0.5`` (binary) or ``1/K`` over valid actions, and **include** that
trial in the participant log-likelihood (never omit the person).

Optional: ``elite_failover=True`` retries frozen ranks 2–3 (cap
``MAX_ELITE_FAILOVER_RANKS``) before uniform. Test labels / likelihoods never
select or reorder programs.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.teh.prompt_snapshots import sanitize_problem_for_choose

LOGLIK_EPS = 1e-9
# Optional elite retry: at most rank-1..rank-3, then uniform.
MAX_ELITE_FAILOVER_RANKS = 3
# Official default: rank-1 only, then uniform (matches OpenEvolve evaluator).
DEFAULT_MAX_FAILOVER_RANKS = 1
# Back-compat alias used by older imports/tests.
MAX_FAILOVER_ELITE_RANKS = MAX_ELITE_FAILOVER_RANKS


def _action_cardinality(problem: Dict[str, Any], trial: Dict[str, Any]) -> int:
    options = trial.get("options") or problem.get("option_keys") or problem.get("options")
    if isinstance(options, list) and len(options) >= 2:
        return len(options)
    n_arms = problem.get("n_arms")
    if isinstance(n_arms, int) and n_arms >= 2:
        return int(n_arms)
    return 2


def _uniform_prediction(k: int) -> Any:
    if k <= 2:
        return 0.5
    return {i: 1.0 / float(k) for i in range(k)}


def _binary_output_valid(p_raw: Any) -> Tuple[bool, Optional[float]]:
    try:
        if isinstance(p_raw, bool) or (
            isinstance(p_raw, (int, np.integer)) and int(p_raw) in (0, 1)
        ):
            return True, 1.0 if int(p_raw) == 1 else 0.0
        if isinstance(p_raw, float):
            if not math.isfinite(p_raw):
                return False, None
            if not (0.0 <= p_raw <= 1.0):
                return False, None
            return True, float(p_raw)
        return False, None
    except Exception:
        return False, None


def _categorical_output_valid(
    p_raw: Any, k: int
) -> Tuple[bool, Optional[Dict[int, float]]]:
    if not isinstance(p_raw, dict) or k < 2:
        return False, None
    probs: Dict[int, float] = {}
    for key, val in p_raw.items():
        try:
            aid = int(key)
            p = float(val)
        except (TypeError, ValueError):
            return False, None
        if not math.isfinite(p) or p < 0.0:
            return False, None
        if 0 <= aid < k:
            probs[aid] = p
    if len(probs) < k:
        # Missing action mass is invalid (no silent repair).
        for aid in range(k):
            if aid not in probs:
                return False, None
    total = sum(probs.values())
    if not math.isfinite(total) or total <= 0.0:
        return False, None
    return True, {aid: probs[aid] / total for aid in range(k)}


def try_elite_prediction(
    choose_fn: Callable,
    trial: Dict[str, Any],
    *,
    categorical: bool,
) -> Tuple[bool, Any, Optional[str]]:
    """Return (ok, prediction_or_None, error_type). Never reads the test label."""
    problem = sanitize_problem_for_choose(trial.get("problem") or {})
    history = trial.get("history") or []
    k = _action_cardinality(problem, trial)
    try:
        raw = choose_fn(problem, history)
    except Exception as exc:
        return False, None, type(exc).__name__
    if categorical or k > 2:
        ok, probs = _categorical_output_valid(raw, k)
        return (ok, probs, None if ok else "invalid_categorical_probs")
    ok, p = _binary_output_valid(raw)
    return (ok, p, None if ok else "invalid_binary_prob")


def resolve_max_failover_ranks(
    *,
    elite_failover: bool = False,
    max_failover_ranks: Optional[int] = None,
) -> int:
    """Default=1 (uniform after rank-1); optional elite retry caps at 3."""
    if max_failover_ranks is not None:
        return max(0, int(max_failover_ranks))
    if elite_failover:
        return int(MAX_ELITE_FAILOVER_RANKS)
    return int(DEFAULT_MAX_FAILOVER_RANKS)


def evaluate_trials_with_frozen_elite_failover(
    elite_programs: Sequence[Tuple[str, str, Callable]],
    trials: Sequence[Dict[str, Any]],
    *,
    categorical: bool = False,
    n_seeds: int = 1,
    elite_failover: bool = False,
    max_failover_ranks: Optional[int] = None,
) -> Dict[str, Any]:
    """Score held-out trials with frozen elite order and failure fallback.

    ``elite_programs`` is ``(program_id, code, choose_fn)`` in observed-union
    elite order (rank-1 first).

    Default (``elite_failover=False``): try rank-1 only; on failure use uniform
    ``0.5`` / ``1/K`` and include the trial in LL.

    Optional (``elite_failover=True``): try ranks 1–3 then uniform. Test labels
    are used only for scoring after a prediction is chosen.
    """
    del n_seeds  # deterministic choose; keep signature aligned with evaluators
    elite_all = list(elite_programs)
    cap = resolve_max_failover_ranks(
        elite_failover=elite_failover, max_failover_ranks=max_failover_ranks
    )
    elite = elite_all[:cap] if cap else []
    mode = "elite_failover" if cap > 1 else "uniform_rank1"
    total = len(trials)
    loglik_acc = 0.0
    correct = 0
    primary_failures = 0
    resolved_by_other = 0
    uniform_fallback_trials = 0
    fallback_depths: List[int] = []
    first_error: Optional[str] = None

    for trial in trials:
        problem = sanitize_problem_for_choose(trial.get("problem") or {})
        k = _action_cardinality(problem, trial)
        y = int(trial.get("action") if trial.get("action") is not None else 0)
        used_pred: Any = None
        depth = -1
        for rank, (_pid, _code, choose_fn) in enumerate(elite):
            ok, pred, err = try_elite_prediction(
                choose_fn, trial, categorical=categorical or k > 2
            )
            if ok:
                used_pred = pred
                depth = rank
                if rank > 0:
                    resolved_by_other += 1
                break
            if rank == 0:
                primary_failures += 1
            if first_error is None and err:
                first_error = err
        else:
            used_pred = _uniform_prediction(k)
            depth = len(elite)  # past last attempted elite
            uniform_fallback_trials += 1
            if not elite:
                primary_failures += 1
        fallback_depths.append(depth)

        # Score only (label access after routing). Uniform trials are included.
        if isinstance(used_pred, dict):
            p_y = float(used_pred.get(y, 0.0))
            p_y = min(max(p_y, LOGLIK_EPS), 1.0 - LOGLIK_EPS)
            loglik_acc += math.log(p_y)
            pred_action = max(used_pred.items(), key=lambda kv: kv[1])[0]
            correct += int(int(pred_action) == y)
        else:
            p = min(max(float(used_pred), LOGLIK_EPS), 1.0 - LOGLIK_EPS)
            loglik_acc += y * math.log(p) + (1 - y) * math.log(1.0 - p)
            pred_bit = 1 if float(used_pred) >= 0.5 else 0
            correct += int(pred_bit == y)

    avg_ll = loglik_acc / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    rate = float(uniform_fallback_trials) / float(total) if total > 0 else 0.0
    diagnostics = {
        "enabled": True,
        "mode": mode,
        "elite_failover": bool(cap > 1),
        "n_elite": len(elite_all),
        "max_failover_ranks": int(cap),
        "n_elite_attempted": len(elite),
        "primary_failures": int(primary_failures),
        "rank1_failures": int(primary_failures),
        "resolved_by_other_elite": int(resolved_by_other),
        "uniform_fallback_trials": int(uniform_fallback_trials),
        "uniform_fallback_rate": float(rate),
        # Alias for older readers / OE-style "errors" count.
        "all_elite_failures": int(uniform_fallback_trials),
        "errors": int(uniform_fallback_trials),
        "mean_fallback_depth": (
            float(sum(fallback_depths) / len(fallback_depths))
            if fallback_depths
            else 0.0
        ),
        "fallback_depths": fallback_depths,
        "deployment_time_only": True,
        "not_model_selection": True,
        "test_label_used_for_routing": False,
        "uniform_fallback_after_cap": True,
        "includes_uniform_trials_in_loglik": True,
    }
    return {
        "avg_loglik": float(avg_ll),
        "accuracy": float(acc),
        "total": total,
        "correct": int(correct),
        "errors": int(uniform_fallback_trials),
        "first_error": first_error,
        "test_time_fallback": diagnostics,
        # Back-compat key (same payload).
        "elite_failover": diagnostics,
    }


def freeze_elite_program_fns(
    elite_parents: Sequence[Tuple[Any, ...]],
    *,
    compile_program: Callable[[str], Any],
    max_ranks: Optional[int] = None,
) -> List[Tuple[str, str, Any]]:
    """Compile frozen elite tuples ``(code, fitness, ..., program_id, ...)``.

    When ``max_ranks`` is set, only the first ``max_ranks`` parents are compiled.
    """
    parents = list(elite_parents)
    if max_ranks is not None:
        parents = parents[: max(0, int(max_ranks))]
    out: List[Tuple[str, str, Any]] = []
    for parent in parents:
        code = parent[0] if parent else ""
        program_id = str(parent[3]) if len(parent) > 3 else f"elite_{len(out)}"
        choose_fn = compile_program(code or "")
        if choose_fn is None:
            continue
        out.append((program_id, str(code or ""), choose_fn))
    return out
