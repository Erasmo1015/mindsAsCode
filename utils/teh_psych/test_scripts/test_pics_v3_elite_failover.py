"""PICS v3 test-time fallback + history robustness regressions (CPU only)."""
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
    DEFAULT_MAX_FAILOVER_RANKS,
    MAX_ELITE_FAILOVER_RANKS,
    evaluate_trials_with_frozen_elite_failover,
    resolve_max_failover_ranks,
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


def test_default_cap_is_rank1_only():
    assert DEFAULT_MAX_FAILOVER_RANKS == 1
    assert resolve_max_failover_ranks() == 1
    assert resolve_max_failover_ranks(elite_failover=False) == 1
    assert resolve_max_failover_ranks(elite_failover=True) == MAX_ELITE_FAILOVER_RANKS == 3


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
    assert ensure_history_robustness_block(out) == out
    # Unaffected / unspecified datasets keep the legacy generic block verbatim.
    assert HISTORY_ROBUSTNESS_BLOCK in out
    with pics_v3_prompt_robustness_scope(True, dataset="1peterson2021using"):
        out_p = maybe_attach_history_robustness_after_task_description(
            task, dataset="1peterson2021using"
        )
    assert HISTORY_ROBUSTNESS_BLOCK in out_p
    assert out_p == out


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
    for c in cases:
        if c["label"] == "required_problem_fields_empty_history":
            assert c["problem"].get("schema_type") == "A"
            assert c["history"] == []


def test_default_never_calls_ranks_2_or_3_uses_uniform_05():
    """Official default: rank-1 only; on failure uniform 0.5; ranks 2–3 never called."""
    calls = []

    def boom(problem, history):
        calls.append("r1")
        raise KeyError("feedback")

    def rank2(problem, history):
        calls.append("r2")
        return 0.99

    def rank3(problem, history):
        calls.append("r3")
        return 0.01

    elite = [("r1", "c1", boom), ("r2", "c2", rank2), ("r3", "c3", rank3)]
    trials = [_trial(1), _trial(0), _trial(1)]
    result = evaluate_trials_with_frozen_elite_failover(
        elite, trials, categorical=False
    )  # default elite_failover=False
    fo = result["test_time_fallback"]
    assert fo["mode"] == "uniform_rank1"
    assert fo["elite_failover"] is False
    assert fo["max_failover_ranks"] == 1
    assert fo["n_elite_attempted"] == 1
    assert fo["uniform_fallback_trials"] == 3
    assert fo["uniform_fallback_rate"] == pytest.approx(1.0)
    assert fo["resolved_by_other_elite"] == 0
    assert fo["includes_uniform_trials_in_loglik"] is True
    assert calls == ["r1", "r1", "r1"]
    assert "r2" not in calls and "r3" not in calls
    # Uniform included in LL → mean log(0.5), never -inf / omitted
    assert math.isfinite(result["avg_loglik"])
    assert result["avg_loglik"] == pytest.approx(math.log(0.5))
    assert result["total"] == 3


def test_default_categorical_uniform_one_over_k_included_in_ll():
    calls = []

    def boom(problem, history):
        calls.append("r1")
        raise ValueError("x")

    def rank2(problem, history):
        calls.append("r2")
        return {0: 0.9, 1: 0.05, 2: 0.05, 3: 0.0}

    elite = [("r1", "c", boom), ("r2", "c", rank2)]
    trial = {
        "action": 2,
        "problem": {"option_keys": ["A", "B", "C", "D"], "n_arms": 4},
        "history": [],
        "options": ["A", "B", "C", "D"],
    }
    result = evaluate_trials_with_frozen_elite_failover(
        elite, [trial], categorical=True
    )
    assert calls == ["r1"]
    assert "r2" not in calls
    fo = result["test_time_fallback"]
    assert fo["uniform_fallback_trials"] == 1
    assert result["avg_loglik"] == pytest.approx(math.log(0.25))
    assert math.isfinite(result["avg_loglik"])


def test_poor_but_valid_rank1_never_bypassed():
    def rank1(problem, history):
        return 0.01

    def rank2(problem, history):
        return 0.99

    elite = [("r1", "code1", rank1), ("r2", "code2", rank2)]
    trials = [_trial(action=1) for _ in range(5)]
    t = _trial(1)
    ok, pred, _err = try_elite_prediction(rank1, t, categorical=False)
    assert ok and pred == 0.01

    result = evaluate_trials_with_frozen_elite_failover(elite, trials, categorical=False)
    fo = result["test_time_fallback"]
    assert fo["resolved_by_other_elite"] == 0
    assert fo["uniform_fallback_trials"] == 0
    assert fo["fallback_depths"] == [0] * 5
    assert result["avg_loglik"] == pytest.approx(math.log(0.01), rel=0, abs=1e-6)


def test_optional_elite_failover_invokes_rank2():
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
        elite, [_trial(1), _trial(0)], categorical=False, elite_failover=True
    )
    fo = result["test_time_fallback"]
    assert fo["mode"] == "elite_failover"
    assert fo["elite_failover"] is True
    assert fo["max_failover_ranks"] == 3
    assert fo["primary_failures"] == 2
    assert fo["resolved_by_other_elite"] == 2
    assert fo["fallback_depths"] == [1, 1]
    assert "r3" not in calls
    assert calls.count("r1") == 2
    assert calls.count("r2") == 2


def test_optional_elite_failover_stops_at_third_then_uniform():
    calls = []

    def boom(pid):
        def choose(problem, history):
            calls.append(pid)
            raise RuntimeError(pid)

        return choose

    def good_rank4(problem, history):
        calls.append("r4")
        return 0.99

    elite = [
        ("r1", "c", boom("r1")),
        ("r2", "c", boom("r2")),
        ("r3", "c", boom("r3")),
        ("r4", "c", good_rank4),
    ]
    result = evaluate_trials_with_frozen_elite_failover(
        elite, [_trial(1)], categorical=False, elite_failover=True
    )
    fo = result["test_time_fallback"]
    assert fo["uniform_fallback_trials"] == 1
    assert fo["fallback_depths"] == [3]
    assert calls == ["r1", "r2", "r3"]
    assert result["avg_loglik"] == pytest.approx(math.log(0.5))


def test_routing_never_reads_test_label():
    def bad(problem, history):
        raise RuntimeError("boom")

    def good(problem, history):
        return 0.6

    class NoLabelTrial(dict):
        def __getitem__(self, key):
            if key == "action":
                raise AssertionError("label leak")
            return dict.__getitem__(self, key)

        def get(self, key, default=None):
            if key == "action":
                raise AssertionError("label leak")
            return dict.get(self, key, default)

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
