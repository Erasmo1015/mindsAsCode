"""PICS v3 observed-union contract helpers (train ∪ val is one training set).

Paper-facing training data is the chronologically ordered union of the loader's
legacy ``train`` and ``val`` slices. There is no validation role in PICS v3.
These helpers are used only when ``limited_data_protocol == structure_aware_v3``.
v1 / preliminary-v2 behavior is unchanged.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.teh.limited_data_registry import (
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3,
    normalize_limited_data_protocol,
)

# Outcome-like history keys that may be omitted depending on trial/stage.
OPTIONAL_HISTORY_OUTCOME_FIELDS = frozenset(
    {
        "feedback",
        "reward",
        "outcome",
        "outcome_marker",
        "treasure",
        "exploded",
        "was_correct",
        "weather_outcome",
        "correct_category",
        "points_received",
        "response_key",
    }
)


def uses_pics_v3_observed_union(limited_data_protocol: object) -> bool:
    """True only for final ICLR PICS v3 data protocol."""
    try:
        return (
            normalize_limited_data_protocol(limited_data_protocol)
            == LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3
        )
    except Exception:
        return str(limited_data_protocol).strip() == LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3


def trial_observed_identity(trial: Dict[str, Any]) -> str:
    """Stable identity for one observed trial (order-independent equality)."""
    problem = trial.get("problem") or {}
    history = trial.get("history") or []
    payload = {
        "action": trial.get("action"),
        "problem": problem,
        "history": history,
        "options": trial.get("options"),
    }
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _chronological_sort_key(trial: Dict[str, Any], fallback_idx: int) -> Tuple[Any, ...]:
    problem = trial.get("problem") or {}
    block = problem.get("block_index")
    if block is None:
        block = trial.get("block_index")
    hist = trial.get("history") or []
    hist_len = len(hist) if isinstance(hist, list) else 0
    return (
        block if block is not None else 10**9,
        hist_len,
        fallback_idx,
        trial_observed_identity(trial),
    )


def merge_observed_trials(
    train_trials: Sequence[Dict[str, Any]],
    val_trials: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Merge legacy train/val into one chronologically ordered observed union.

    Does not mutate inputs. Deduplicates by trial identity while preserving the
    first chronological occurrence. Test trials must never be passed here.
    """
    pooled: List[Dict[str, Any]] = []
    for t in list(train_trials or []):
        pooled.append(t)
    for t in list(val_trials or []):
        pooled.append(t)
    if not pooled:
        return []
    indexed = list(enumerate(pooled))
    indexed.sort(key=lambda iv: _chronological_sort_key(iv[1], iv[0]))
    out: List[Dict[str, Any]] = []
    seen = set()
    for _i, trial in indexed:
        key = trial_observed_identity(trial)
        if key in seen:
            continue
        seen.add(key)
        out.append(trial)
    return out


def observed_trial_identities(trials: Sequence[Dict[str, Any]]) -> List[str]:
    return [trial_observed_identity(t) for t in trials]


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def count_pooled_observed_loglik(
    train_loglik: Any,
    val_loglik: Any,
    n_train: int,
    n_val: int,
) -> Optional[float]:
    """Count-pooled mean LL over the observed union; None if any non-empty split fails.

    Empty legacy val (``n_val==0``) means all observed trials live in train — that
    is still a valid full union, not a train-only fallback over a crashing val.
    Non-finite or missing scores on a non-empty legacy split reject the candidate.
    """
    n_tr = max(0, int(n_train))
    n_vl = max(0, int(n_val))
    if n_tr + n_vl <= 0:
        return None
    train_ll = _finite_or_none(train_loglik) if n_tr > 0 else None
    val_ll = _finite_or_none(val_loglik) if n_vl > 0 else None
    if n_tr > 0 and train_ll is None:
        return None
    if n_vl > 0 and val_ll is None:
        return None
    if n_tr == 0:
        return val_ll
    if n_vl == 0:
        return train_ll
    return (n_tr * float(train_ll) + n_vl * float(val_ll)) / float(n_tr + n_vl)


def observed_union_runtime_valid(
    *,
    train_errors: int,
    val_errors: int,
    n_train: int,
    n_val: int,
    train_loglik: Any,
    val_loglik: Any,
) -> bool:
    """Candidate is valid iff every observed trial ran without error and scores are finite."""
    if int(train_errors) > 0 or int(val_errors) > 0:
        return False
    return count_pooled_observed_loglik(train_loglik, val_loglik, n_train, n_val) is not None


def mean_test_loglik_with_failure_policy(
    test_logliks: Iterable[Any],
) -> Dict[str, Any]:
    """Reporting helper: never silently average finite-only people.

    If any person has a missing/non-finite test loglik, ``mean`` is ``None`` and
    ``n_nonfinite`` / ``n_finite`` are recorded. Callers must not substitute a
    finite-only mean.
    """
    values = list(test_logliks)
    finite: List[float] = []
    n_nonfinite = 0
    for raw in values:
        number = _finite_or_none(raw)
        if number is None:
            n_nonfinite += 1
        else:
            finite.append(number)
    n = len(values)
    if n == 0:
        return {
            "mean": None,
            "n": 0,
            "n_finite": 0,
            "n_nonfinite": 0,
            "policy": "require_all_finite",
        }
    if n_nonfinite > 0:
        return {
            "mean": None,
            "n": n,
            "n_finite": len(finite),
            "n_nonfinite": n_nonfinite,
            "policy": "require_all_finite",
        }
    return {
        "mean": float(sum(finite) / float(len(finite))),
        "n": n,
        "n_finite": len(finite),
        "n_nonfinite": 0,
        "policy": "require_all_finite",
    }
