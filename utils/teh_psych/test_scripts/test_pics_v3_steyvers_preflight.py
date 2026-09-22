"""CPU regressions for Steyvers interface preflight vs reward-learning programs."""
from __future__ import annotations

import math
from typing import Any, Dict, List

from utils.teh.pics_v3_contract_preflight import (
    build_admissible_preflight_cases,
    run_interface_contract_preflight,
)
from utils.teh.pics_v3_observed import count_pooled_observed_loglik

# Representative observed-union trials matching the Steyvers loader contract.
_STEYVERS_OBSERVED: List[Dict[str, Any]] = [
    {
        "problem": {
            "dataset_alias": "steyvers_2009_bandit",
            "schema_type": "categorical_bandit",
            "game": 1,
            "trial": 3,
            "n_arms": 4,
            "options": [{"action": i} for i in range(4)],
            "option_keys": [0, 1, 2, 3],
            "has_feedback": True,
            "raw_choice_coding": "1-4",
            "internal_action_coding": "0-3",
        },
        "history": [
            {"action": 0, "reward": 1},
            {"action": 2, "reward": 0},
            {"action": 0, "reward": 1},
        ],
        "action": 1,
        "options": [0, 1, 2, 3],
    },
    {
        "problem": {
            "dataset_alias": "steyvers_2009_bandit",
            "schema_type": "categorical_bandit",
            "game": 1,
            "trial": 1,
            "n_arms": 4,
            "options": [{"action": i} for i in range(4)],
            "option_keys": [0, 1, 2, 3],
            "has_feedback": True,
        },
        "history": [],
        "action": 0,
        "options": [0, 1, 2, 3],
    },
]


SAFE_REWARD_LEARNER = '''
def choose(problem, history):
    options = problem["options"]
    actions = [opt["action"] for opt in options]
    counts = {a: 1 for a in actions}
    rewards = {a: 0.0 for a in actions}
    usable = 0
    for entry in history:
        a = entry.get("action")
        r = entry.get("reward")
        if a in counts and r is not None:
            counts[a] += 1
            rewards[a] += float(r)
            usable += 1
    if usable == 0:
        u = 1.0 / float(len(actions))
        return {a: u for a in actions}
    probs = {a: (rewards[a] + 1.0) / float(counts[a] + 1) for a in actions}
    total = sum(probs.values())
    if total <= 0:
        u = 1.0 / float(len(actions))
        return {a: u for a in actions}
    return {a: probs[a] / total for a in actions}
'''

FRAGILE_LIST_INDEX = '''
def choose(problem, history):
    options = problem["options"]
    K = len(options)
    action_counts = [1] * K
    reward_sums = [0] * K
    for entry in history:
        action = entry.get("action")
        reward = entry.get("reward", 0)
        if action is not None:
            action_counts[action] += 1
            reward_sums[action] += reward
    expected_values = [reward_sums[i] / float(action_counts[i]) for i in range(K)]
    total_value = sum(expected_values)
    probabilities = {
        option["action"]: expected_values[i] / total_value
        for i, option in enumerate(options)
    }
    return probabilities
'''

FRAGILE_ZERO_DIV = '''
def choose(problem, history):
    options = problem["options"]
    probs = {option["action"]: 0.0 for option in options}
    for entry in history:
        a = entry.get("action")
        r = entry.get("reward")
        if a in probs and r is not None:
            probs[a] += float(r)
    total = sum(probs.values())
    return {a: probs[a] / total for a in probs}
'''


def _compile_choose(src: str):
    ns: Dict[str, Any] = {}
    exec(src, ns)  # noqa: S102 — test-only compile of fixture programs
    return ns["choose"]


def test_steyvers_preflight_cases_include_optional_reward_absence() -> None:
    """Interface contract still admits missing/null reward variants (not loader bugs)."""
    cases = build_admissible_preflight_cases(_STEYVERS_OBSERVED)
    labels = {c["label"] for c in cases}
    assert "empty_history" in labels
    assert "action_only_no_outcomes" in labels
    assert "null_optional_outcomes" in labels
    # Real loader histories always carry reward, but synthetic variants strip it.
    stripped = [
        c
        for c in cases
        if c["history"]
        and any(
            ("reward" not in e) or (e.get("reward") is None) for e in c["history"]
        )
    ]
    assert stripped, "expected defensive missing-reward synthetic cases"


def test_safe_steyvers_reward_learner_passes_preflight() -> None:
    choose = _compile_choose(SAFE_REWARD_LEARNER)
    report = run_interface_contract_preflight(choose, _STEYVERS_OBSERVED)
    assert report["ok"] is True
    assert report["failures"] == []


def test_fragile_list_index_and_zerodiv_fail_preflight() -> None:
    for src in (FRAGILE_LIST_INDEX, FRAGILE_ZERO_DIV):
        choose = _compile_choose(src)
        report = run_interface_contract_preflight(choose, _STEYVERS_OBSERVED)
        assert report["ok"] is False
        assert report["failures"]


def test_safe_learner_finite_observed_union_score() -> None:
    """Safe program yields finite categorical loglik on observed trials (no policy hard-code)."""
    choose = _compile_choose(SAFE_REWARD_LEARNER)
    logliks = []
    for trial in _STEYVERS_OBSERVED:
        dist = choose(trial["problem"], trial["history"])
        assert isinstance(dist, dict)
        assert set(dist.keys()) == set(trial["options"])
        total = sum(float(v) for v in dist.values())
        assert total > 0
        action = int(trial["action"])
        p = float(dist[action]) / total
        p = min(max(p, 1e-12), 1.0)
        logliks.append(math.log(p))
    mean = sum(logliks) / len(logliks)
    assert math.isfinite(mean)
    # count-pooled helper accepts finite train-only score
    pooled = count_pooled_observed_loglik(mean, None, len(logliks), 0)
    assert pooled is not None
    assert math.isfinite(pooled)
