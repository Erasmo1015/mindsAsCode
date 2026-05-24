"""
Schema-aware option-value features for Psych-101 binary baselines (MLE, prospect theory).

Maps structured trial ``problem`` fields (and bandit ``history``) to:
- a scalar feature x with P(action=1) = sigmoid(beta*x + bias), action 0 = option A;
- or (rewards, probs) pairs for two-option prospect-theory fitting.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

GambleGetter = Callable[[Dict[str, Any], Optional[List[Dict[str, Any]]]], Tuple[List[float], Optional[List[float]]]]


def _task_kind(problem: Dict[str, Any]) -> str:
    schema = str(problem.get("schema_type", "A"))
    if "gamble_A" in problem and "gamble_B" in problem:
        return "gamble"
    if "pHa" in problem:
        return "gamble_cpc18"
    if "memory_set_letters" in problem and "probe_letter" in problem:
        return "memory_probe"
    if "stimulus_features" in problem and "correct_category" in problem:
        return "category_learning"
    if (
        "balloon_id" in problem
        and "pump_count_before" in problem
        and "accumulated_points_before" in problem
    ):
        return "balloon_risk"
    if "n_cards_remaining" in problem:
        return "cct"
    if schema == "D":
        return "cct"
    if schema == "C" or "machine_options" in problem:
        return "bandit"
    if "tree_features" in problem:
        return "tree"
    if "ratings_A" in problem and "ratings_B" in problem:
        return "product"
    if "cards" in problem:
        return "weather"
    raise ValueError(
        f"Unsupported Psych-101 problem for baseline features (schema_type={schema!r}, "
        f"keys={sorted(problem.keys())})."
    )


def expected_value_from_gamble(gamble: Dict[str, Any]) -> float:
    rewards = gamble.get("rewards", [])
    probs = gamble.get("probs", None)
    if not rewards:
        return 0.0
    if probs is None:
        return float(rewards[0])
    return float(np.sum(np.asarray(probs, dtype=np.float64) * np.asarray(rewards, dtype=np.float64)))


def _gamble_ev_diff(problem: Dict[str, Any]) -> float:
    if "gamble_A" in problem and "gamble_B" in problem:
        return expected_value_from_gamble(problem["gamble_B"]) - expected_value_from_gamble(problem["gamble_A"])
    if "pHa" in problem:
        ev_a = float(problem["pHa"] * problem["Ha"] + (1.0 - problem["pHa"]) * problem["La"])
        ev_b = float(problem["pHb"] * problem["Hb"] + (1.0 - problem["pHb"]) * problem["Lb"])
        return ev_b - ev_a
    return 0.0


def _ratings_sum(ratings: Any) -> float:
    if ratings is None:
        return 0.0
    return float(sum(int(x) for x in ratings))


def _bandit_empirical_means(
    history: Optional[List[Dict[str, Any]]],
    n_options: int = 2,
) -> List[float]:
    sums = [0.0] * n_options
    counts = [0] * n_options
    for h in history or []:
        a = int(h.get("action", -1))
        if a < 0 or a >= n_options:
            continue
        if "feedback" in h:
            sums[a] += float(h["feedback"])
            counts[a] += 1
    return [sums[i] / counts[i] if counts[i] > 0 else 0.0 for i in range(n_options)]


def _cct_one_step_ev(problem: Dict[str, Any]) -> Tuple[float, float]:
    """Expected score after one flip vs cashing out at current_score."""
    n_rem = max(1.0, float(problem.get("n_cards_remaining", 1)))
    n_loss = max(0.0, float(problem.get("n_loss_cards", 0)))
    gain = float(problem.get("gain_amount", 0))
    loss = float(problem.get("loss_amount", 0))
    cur = float(problem.get("current_score", 0))
    p_loss = min(1.0, max(0.0, n_loss / n_rem))
    ev_continue = cur + (1.0 - p_loss) * gain - p_loss * loss
    ev_stop = cur
    return ev_stop, ev_continue


def _balloon_explosion_prob(pump_count_before: float) -> float:
    """Simple monotone hazard proxy for BART-like tasks."""
    p = (float(pump_count_before) + 1.0) / 64.0
    return float(min(0.95, max(0.02, p)))


def _balloon_one_step_ev(problem: Dict[str, Any]) -> Tuple[float, float]:
    """
    Expected utility of stop vs one more pump.

    action=0 -> pump, action=1 -> stop.
    """
    cur = max(0.0, float(problem.get("accumulated_points_before", 0.0)))
    pump_count = max(0.0, float(problem.get("pump_count_before", 0.0)))
    p_explode = _balloon_explosion_prob(pump_count)
    ev_pump = (1.0 - p_explode) * (cur + 1.0) + p_explode * 0.0
    ev_stop = cur
    return ev_stop, ev_pump


def option_b_feature_diff(
    problem: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """Scalar x with P(action=1) = sigmoid(beta*x + bias); action 0 = option A, 1 = option B."""
    kind = _task_kind(problem)
    if kind in ("gamble", "gamble_cpc18"):
        return _gamble_ev_diff(problem)
    if kind == "cct":
        ev_stop, ev_continue = _cct_one_step_ev(problem)
        # action 0 = flip, 1 = stop; positive x favors stop
        return ev_stop - ev_continue
    if kind == "balloon_risk":
        ev_stop, ev_pump = _balloon_one_step_ev(problem)
        # action 0 = pump, 1 = stop; positive x favors stop
        return ev_stop - ev_pump
    if kind == "bandit":
        means = _bandit_empirical_means(history)
        return means[1] - means[0]
    if kind == "product":
        return _ratings_sum(problem.get("ratings_B")) - _ratings_sum(problem.get("ratings_A"))
    if kind == "tree":
        tf = problem.get("tree_features") or {}
        accept_u = float(tf.get("leafiness", 0)) + float(tf.get("branchiness", 0))
        return accept_u
    if kind == "weather":
        cards = problem.get("cards") or []
        if not cards:
            return 0.0
        return float(sum(cards)) / float(len(cards))
    if kind == "memory_probe":
        # action 1 corresponds to "yes/in-set" for supported recent-probe parser.
        probe_in_set = bool(problem.get("probe_in_set", False))
        return 1.0 if probe_in_set else -1.0
    if kind == "category_learning":
        # Binary category choice with observed correct category in trial feedback.
        keys = problem.get("option_keys") or []
        if len(keys) < 2:
            return 0.0
        correct = str(problem.get("correct_category", "")).upper()
        key_a = str(keys[0]).upper()
        key_b = str(keys[1]).upper()
        if correct == key_b:
            return 1.0
        if correct == key_a:
            return -1.0
        return 0.0
    return 0.0


def _gamble_getters_gamble_ab() -> Tuple[GambleGetter, GambleGetter]:
    def gamble_a(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        return (list(p["gamble_A"]["rewards"]), p["gamble_A"].get("probs"))

    def gamble_b(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        return (list(p["gamble_B"]["rewards"]), p["gamble_B"].get("probs"))

    return gamble_a, gamble_b


def _gamble_getters_cpc18() -> Tuple[GambleGetter, GambleGetter]:
    def gamble_a(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        return (
            [float(p["Ha"]), float(p["La"])],
            [float(p["pHa"]), float(1.0 - p["pHa"])],
        )

    def gamble_b(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        return (
            [float(p["Hb"]), float(p["Lb"])],
            [float(p["pHb"]), float(1.0 - p["pHb"])],
        )

    return gamble_a, gamble_b


def _gamble_getters_cct() -> Tuple[GambleGetter, GambleGetter]:
    def flip(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        _ev_stop, ev_continue = _cct_one_step_ev(p)
        cur = float(p.get("current_score", 0))
        gain = float(p.get("gain_amount", 0))
        loss = float(p.get("loss_amount", 0))
        n_rem = max(1.0, float(p.get("n_cards_remaining", 1)))
        n_loss = max(0.0, float(p.get("n_loss_cards", 0)))
        p_loss = min(1.0, max(0.0, n_loss / n_rem))
        return (
            [cur + gain, cur - loss],
            [1.0 - p_loss, p_loss],
        )

    def stop(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        cur = float(p.get("current_score", 0))
        return ([cur], [1.0])

    return flip, stop


def _gamble_getters_bandit() -> Tuple[GambleGetter, GambleGetter]:
    def arm_a(p: Dict[str, Any], h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        m = _bandit_empirical_means(h)
        return ([m[0]], [1.0])

    def arm_b(p: Dict[str, Any], h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        m = _bandit_empirical_means(h)
        return ([m[1]], [1.0])

    return arm_a, arm_b


def _gamble_getters_product() -> Tuple[GambleGetter, GambleGetter]:
    def prod_a(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        return ([_ratings_sum(p.get("ratings_A"))], [1.0])

    def prod_b(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        return ([_ratings_sum(p.get("ratings_B"))], [1.0])

    return prod_a, prod_b


def _gamble_getters_tree() -> Tuple[GambleGetter, GambleGetter]:
    def reject(_p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        return ([0.0], [1.0])

    def accept(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        tf = p.get("tree_features") or {}
        u = float(tf.get("leafiness", 0)) + float(tf.get("branchiness", 0))
        return ([u], [1.0])

    return reject, accept


def _gamble_getters_weather() -> Tuple[GambleGetter, GambleGetter]:
    def opt_a(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        cards = p.get("cards") or []
        s = float(sum(cards)) / max(1, len(cards)) if cards else 0.0
        return ([s], [1.0])

    def opt_b(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        cards = p.get("cards") or []
        s = float(sum(cards)) / max(1, len(cards)) if cards else 0.0
        return ([-s], [1.0])

    return opt_a, opt_b


def _gamble_getters_balloon_risk() -> Tuple[GambleGetter, GambleGetter]:
    # option A/action=0 -> pump, option B/action=1 -> stop
    def pump(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        cur = max(0.0, float(p.get("accumulated_points_before", 0.0)))
        pump_count = max(0.0, float(p.get("pump_count_before", 0.0)))
        p_explode = _balloon_explosion_prob(pump_count)
        return ([cur + 1.0, 0.0], [1.0 - p_explode, p_explode])

    def stop(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        cur = max(0.0, float(p.get("accumulated_points_before", 0.0)))
        return ([cur], [1.0])

    return pump, stop


def _gamble_getters_memory_probe() -> Tuple[GambleGetter, GambleGetter]:
    # option A/action=0 -> "no", option B/action=1 -> "yes"
    def no_opt(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        probe_in_set = bool(p.get("probe_in_set", False))
        return ([-1.0 if probe_in_set else 1.0], [1.0])

    def yes_opt(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        probe_in_set = bool(p.get("probe_in_set", False))
        return ([1.0 if probe_in_set else -1.0], [1.0])

    return no_opt, yes_opt


def _gamble_getters_category_learning() -> Tuple[GambleGetter, GambleGetter]:
    # option A/action=0 and option B/action=1 mapped by option_keys.
    def opt_a(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        keys = p.get("option_keys") or []
        if len(keys) < 2:
            return ([0.0], [1.0])
        correct = str(p.get("correct_category", "")).upper()
        return ([1.0 if str(keys[0]).upper() == correct else -1.0], [1.0])

    def opt_b(p: Dict[str, Any], _h: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[float], Optional[List[float]]]:
        keys = p.get("option_keys") or []
        if len(keys) < 2:
            return ([0.0], [1.0])
        correct = str(p.get("correct_category", "")).upper()
        return ([1.0 if str(keys[1]).upper() == correct else -1.0], [1.0])

    return opt_a, opt_b


def prospect_gamble_getters(
    problem: Dict[str, Any],
) -> Tuple[GambleGetter, GambleGetter]:
    """Return (option_A, option_B) getters for two-option prospect-theory fitting."""
    kind = _task_kind(problem)
    if kind == "gamble":
        return _gamble_getters_gamble_ab()
    if kind == "gamble_cpc18":
        return _gamble_getters_cpc18()
    if kind == "cct":
        return _gamble_getters_cct()
    if kind == "balloon_risk":
        return _gamble_getters_balloon_risk()
    if kind == "bandit":
        return _gamble_getters_bandit()
    if kind == "product":
        return _gamble_getters_product()
    if kind == "tree":
        return _gamble_getters_tree()
    if kind == "weather":
        return _gamble_getters_weather()
    if kind == "memory_probe":
        return _gamble_getters_memory_probe()
    if kind == "category_learning":
        return _gamble_getters_category_learning()
    raise ValueError(f"Unsupported task kind for prospect getters: {kind!r}")
