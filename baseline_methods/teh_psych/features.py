"""
Feature extractors for teh_psych categorical baselines.

MLE model (documented)
----------------------
For each trial with options ``k = 0..K-1`` and history of prior
``{action, feedback}`` steps within the block:

* ``f_k`` = mean numeric feedback observed for action ``k`` in ``history``
  (0.0 if never observed). Non-numeric feedback is ignored for that step.
* If the stimulus dict contains a numeric value whose key matches an option
  label / raw_key (case-insensitive), that value is **added** to ``f_k``
  (description cues that align with options). Other stimulus fields are not
  mapped to options (no invented rewards/probabilities).

**K = 2 (binary):** same logistic form as ``baseline_methods/MLE.py``::

    x = f_1 - f_0
    P(action=1) = sigmoid(beta * x + bias)

Fit ``(beta, bias)`` with L-BFGS-B, init ``[1, 0]``, bounds ``[-50, 50]^2``.

**K > 2:** multinomial logit with shared inverse temperature::

    u_k = beta * f_k
    P(action=k) = softmax(u)_k

Fit scalar ``beta`` with L-BFGS-B, init ``1.0``, bounds ``[-50, 50]``.
No per-action intercepts (avoids free parameters growing with K).

Prospect Theory
---------------
Supported only when each of two options has an explicit ``(rewards, probs)``
gamble in ``problem["stimulus"]`` under keys ``gamble_0``/``gamble_1`` or
``gamble_A``/``gamble_B`` with ``rewards`` lists and optional ``probs``.
Otherwise the experiment is marked unsupported — we do not invent outcomes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.teh_psych.categorical_eval import valid_action_ids_from_problem

Gamble = Tuple[List[float], Optional[List[float]]]


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        x = float(value)
        return x if np.isfinite(x) else None
    if isinstance(value, str):
        try:
            x = float(value.strip())
        except ValueError:
            return None
        return x if np.isfinite(x) else None
    return None


def empirical_mean_feedback_by_action(
    history: Optional[Sequence[Dict[str, Any]]],
    action_ids: Sequence[int],
) -> Dict[int, float]:
    sums = {int(a): 0.0 for a in action_ids}
    counts = {int(a): 0 for a in action_ids}
    for h in history or []:
        if not isinstance(h, dict) or "action" not in h:
            continue
        a = int(h["action"])
        if a not in sums:
            continue
        fb = _as_float(h.get("feedback"))
        if fb is None:
            continue
        sums[a] += fb
        counts[a] += 1
    return {a: (sums[a] / counts[a] if counts[a] > 0 else 0.0) for a in action_ids}


def _stimulus_label_boosts(
    problem: Dict[str, Any],
    action_ids: Sequence[int],
) -> Dict[int, float]:
    """Add stimulus numerics keyed by option label/raw_key."""
    stim = problem.get("stimulus") or {}
    if not isinstance(stim, dict):
        return {int(a): 0.0 for a in action_ids}
    boosts = {int(a): 0.0 for a in action_ids}
    label_to_action: Dict[str, int] = {}
    for opt in problem.get("options") or []:
        if not isinstance(opt, dict) or "action" not in opt:
            continue
        aid = int(opt["action"])
        for key in (opt.get("label"), opt.get("raw_key")):
            if key is None:
                continue
            label_to_action[str(key).strip().lower()] = aid
    for sk, sv in stim.items():
        num = _as_float(sv)
        if num is None:
            continue
        sk_l = str(sk).strip().lower()
        if sk_l in label_to_action:
            boosts[label_to_action[sk_l]] += num
            continue
        # keys like option_a / lottery_w matching label suffix
        for lab, aid in label_to_action.items():
            if sk_l.endswith("_" + lab) or sk_l == lab:
                boosts[aid] += num
    return boosts


def option_feature_vector(
    problem: Dict[str, Any],
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[float]:
    """Return ``f_k`` aligned with ``valid_action_ids_from_problem`` order."""
    action_ids = valid_action_ids_from_problem(problem)
    if not action_ids:
        return []
    means = empirical_mean_feedback_by_action(history, action_ids)
    boosts = _stimulus_label_boosts(problem, action_ids)
    return [float(means[a] + boosts[a]) for a in action_ids]


def binary_feature_diff(
    problem: Dict[str, Any],
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> float:
    """``f_1 - f_0`` for K=2; raises if not binary."""
    feats = option_feature_vector(problem, history)
    if len(feats) != 2:
        raise ValueError(f"binary_feature_diff requires K=2, got K={len(feats)}")
    return float(feats[1] - feats[0])


def _parse_gamble_dict(obj: Any) -> Optional[Gamble]:
    if not isinstance(obj, dict):
        return None
    rewards = obj.get("rewards")
    if not isinstance(rewards, (list, tuple)) or not rewards:
        return None
    try:
        r = [float(x) for x in rewards]
    except (TypeError, ValueError):
        return None
    probs = obj.get("probs")
    if probs is None:
        return r, None
    if not isinstance(probs, (list, tuple)) or len(probs) != len(r):
        return None
    try:
        p = [float(x) for x in probs]
    except (TypeError, ValueError):
        return None
    if abs(sum(p) - 1.0) > 1e-3 and sum(p) > 0:
        s = sum(p)
        p = [x / s for x in p]
    return r, p


def extract_explicit_binary_gambles(
    problem: Dict[str, Any],
) -> Optional[Tuple[Gamble, Gamble]]:
    """
    Return (gamble_action0, gamble_action1) when stimulus carries explicit gambles.

    Accepted layouts under ``problem["stimulus"]``:
    - ``gamble_0`` / ``gamble_1``
    - ``gamble_A`` / ``gamble_B``
    - ``option_0`` / ``option_1`` with nested rewards/probs
    """
    stim = problem.get("stimulus")
    if not isinstance(stim, dict):
        return None
    pairs = [
        ("gamble_0", "gamble_1"),
        ("gamble_A", "gamble_B"),
        ("option_0", "option_1"),
        ("gamble_a", "gamble_b"),
    ]
    for ka, kb in pairs:
        if ka in stim and kb in stim:
            ga = _parse_gamble_dict(stim[ka])
            gb = _parse_gamble_dict(stim[kb])
            if ga is not None and gb is not None:
                return ga, gb
    # Top-level problem keys (rare for auto-parse, kept for compatibility)
    for ka, kb in (("gamble_A", "gamble_B"), ("gamble_0", "gamble_1")):
        if ka in problem and kb in problem:
            ga = _parse_gamble_dict(problem[ka])
            gb = _parse_gamble_dict(problem[kb])
            if ga is not None and gb is not None:
                return ga, gb
    return None


def prospect_theory_support_reason(problem: Dict[str, Any]) -> Tuple[bool, str]:
    ids = valid_action_ids_from_problem(problem)
    if len(ids) != 2:
        return False, f"Prospect Theory baseline requires K=2; got K={len(ids)}"
    if extract_explicit_binary_gambles(problem) is None:
        return (
            False,
            "No explicit (rewards, probs) gambles in stimulus "
            "(gamble_0/1 or gamble_A/B); refusing to invent outcomes",
        )
    return True, "Explicit binary gambles present in stimulus"


def experiment_prospect_support(
    prediction_trials: Sequence[Dict[str, Any]],
    *,
    min_fraction: float = 0.95,
) -> Tuple[bool, str, float]:
    """Majority of prediction trials must carry explicit gambles."""
    if not prediction_trials:
        return False, "no prediction trials", 0.0
    ok = 0
    last_reason = ""
    for t in prediction_trials:
        supported, reason = prospect_theory_support_reason(t.get("problem") or {})
        last_reason = reason
        if supported:
            ok += 1
    frac = ok / len(prediction_trials)
    if frac >= min_fraction:
        return True, f"explicit gambles on {ok}/{len(prediction_trials)} trials", frac
    return False, last_reason or "insufficient gamble structure", frac
