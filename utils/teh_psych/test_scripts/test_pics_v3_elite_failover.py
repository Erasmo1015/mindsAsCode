"""PICS v3 elite failover + history robustness regressions (CPU only)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.teh.pics_v3_contract_preflight import (
    build_admissible_history_variants,
    build_admissible_preflight_cases,
    preflight_invalidates_candidate,
)
from utils.teh.pics_v3_elite_failover import (
    evaluate_trials_with_frozen_elite_failover,
    try_elite_prediction,
)
from utils.teh.pics_v3_prompt_robustness import (
    HISTORY_ROBUSTNESS_BLOCK,
    HISTORY_ROBUSTNESS_MARKER,
    ensure_history_robustness_block,
    maybe_attach_history_robustness_after_task_description,
    pics_v3_prompt_robustness_scope,
)


def _trial(action: int = 1, hist=None):
    return {
        "action": action,
        "problem": {"option_keys": ["A", "B"]},
        "history": list(hist or []),
        "options": ["A", "B"],
    }


def test_history_robustness_block_injected_after_task_description_only_under_v3_scope():
    task = "Dataset-adaptive task description goes here."
    with pics_v3_prompt_robustness_scope(False):
        assert maybe_attach_history_robustness_after_task_description(task) == task
    with pics_v3_prompt_robustness_scope(True):
        out = maybe_attach_history_robustness_after_task_description(task)
    assert out.startswith(task)
    assert HISTORY_ROBUSTNESS_MARKER in out
    assert "`history` may be empty" in out
    assert ".get(...)" in out
    # Idempotent
    assert ensure_history_robustness_block(out) == out
    assert HISTORY_ROBUSTNESS_BLOCK in out


def test_preflight_covers_empty_action_only_hetero_null_and_required_fields():
    observed = [
        {
            "action": 1,
            "problem": {
                "option_keys": ["L", "R"],
                "schema_type": "A",
                "stage": "choice",
                "has_feedback": True,
            },
            "history": [
                {"action": 0, "feedback": 1.0},
                {"action": 1, "feedback": -1.0, "reward": 2.0},
            ],
        }
    ]
    labels = {lab for lab, _ in build_admissible_history_variants(observed[0]["history"])}
    assert "empty_history" in labels
    assert "action_only_no_outcomes" in labels
    assert "heterogeneous_entries" in labels
    assert "null_optional_outcomes" in labels
    assert "missing_feedback_and_reward" in labels

    cases = build_admissible_preflight_cases(observed)
    case_labels = {c["label"] for c in cases}
    assert "empty_history" in case_labels
    assert "required_problem_fields_empty_history" in case_labels
    # Required schema/stage preserved on problem; no placeholder injection into trials
    for c in cases:
        if c["label"] == "required_problem_fields_empty_history":
            assert c["problem"].get("schema_type") == "A"
            assert c["history"] == []


def test_poor_but_valid_rank1_never_bypassed():
    """Rank-1 returns a valid but poor probability; must not failover to rank-2."""

    def rank1(problem, history):
        return 0.01  # valid, low likelihood for y=1

    def rank2(problem, history):
        return 0.99

    elite = [
        ("r1", "code1", rank1),
        ("r2", "code2", rank2),
    ]
    trials = [_trial(action=1) for _ in range(5)]
    # Spy: ensure routing never reads action for failovers — wrap trials
    class Guarded(dict):
        def __getitem__(self, key):
            if key == "action":
                raise AssertionError("test label accessed during routing")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key == "action":
                # allow only after prediction chosen — try_elite must not call this
                raise AssertionError("test label accessed during routing")
            return super().get(key, default)

    # Unit: try_elite_prediction itself must not touch action
    t = _trial(1)
    ok, pred, _err = try_elite_prediction(rank1, t, categorical=False)
    assert ok and pred == 0.01

    result = evaluate_trials_with_frozen_elite_failover(elite, trials, categorical=False)
    fo = result["elite_failover"]
    assert fo["resolved_by_other_elite"] == 0
    assert fo["all_elite_failures"] == 0
    assert fo["fallback_depths"] == [0] * 5
    assert fo["test_label_used_for_routing"] is False
    # Score uses y=1 with p=0.01 → much worse than 0.99, proving we kept rank-1
    assert result["avg_loglik"] == pytest.approx(math.log(0.01), rel=0, abs=1e-6)


def test_exception_from_rank1_invokes_rank2_in_frozen_order():
    calls = []

    def rank1(problem, history):
        calls.append("r1")
        raise KeyError("feedback")

    def rank2(problem, history):
        calls.append("r2")
        return 0.8

    def rank3(problem, history):
        calls.append("r3")
        return 0.2

    elite = [("r1", "c1", rank1), ("r2", "c2", rank2), ("r3", "c3", rank3)]
    result = evaluate_trials_with_frozen_elite_failover(
        elite, [_trial(1), _trial(0)], categorical=False
    )
    fo = result["elite_failover"]
    assert fo["primary_failures"] == 2
    assert fo["resolved_by_other_elite"] == 2
    assert fo["fallback_depths"] == [1, 1]
    assert "r3" not in calls  # never skip to rank-3 when rank-2 works
    assert calls.count("r1") == 2
    assert calls.count("r2") == 2


def test_routing_never_reads_test_label():
    def bad(problem, history):
        raise RuntimeError("boom")

    def good(problem, history):
        return 0.6

    class NoLabelTrial(dict):
        def __getitem__(self, key):
            if key == "action":
                # Only evaluate_trials may read action after routing via .get on plain dict;
                # this object is used only inside try_elite_prediction.
                raise AssertionError("label leak")
            return dict.__getitem__(self, key)

        def get(self, key, default=None):
            if key == "action":
                raise AssertionError("label leak")
            return dict.get(self, key, default)

    # try_elite_prediction must not touch action
    t = NoLabelTrial(
        problem={"option_keys": ["A", "B"]},
        history=[],
        options=["A", "B"],
        action=1,
    )
    ok, _pred, err = try_elite_prediction(bad, t, categorical=False)
    assert not ok and err == "RuntimeError"
    ok2, pred2, _ = try_elite_prediction(good, t, categorical=False)
    assert ok2 and pred2 == 0.6


def test_all_elite_failure_produces_uniform_prediction():
    def boom(problem, history):
        raise ValueError("fail")

    elite = [("a", "c", boom), ("b", "c", boom)]
    trials = [_trial(1), _trial(0), _trial(1)]
    result = evaluate_trials_with_frozen_elite_failover(
        elite, trials, categorical=False
    )
    fo = result["elite_failover"]
    assert fo["all_elite_failures"] == 3
    assert fo["fallback_depths"] == [2, 2, 2]
    # Uniform 0.5 → loglik = log(0.5) for every trial
    assert result["avg_loglik"] == pytest.approx(math.log(0.5))


def test_elite_order_frozen_from_caller_not_reordered_by_test():
    order = []

    def make(pid, p):
        def choose(problem, history):
            order.append(pid)
            if pid == "rank1":
                raise RuntimeError("x")
            return p

        return choose

    elite = [
        ("rank1", "c", make("rank1", 0.1)),
        ("rank2", "c", make("rank2", 0.9)),
    ]
    evaluate_trials_with_frozen_elite_failover(elite, [_trial(1)], categorical=False)
    assert order == ["rank1", "rank2"]


def test_preflight_rejects_strict_feedback_on_hetero_histories():
    def bad(problem, history):
        if history:
            return float(history[-1]["feedback"])
        return 0.5

    observed = [
        {
            "action": 0,
            "problem": {"option_keys": ["A", "B"]},
            "history": [{"action": 0, "feedback": 1.0}],
        }
    ]
    invalid, report = preflight_invalidates_candidate(bad, observed)
    assert invalid
    assert report["test_trials_used"] is False
    assert report.get("placeholders_injected") is False
