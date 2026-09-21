"""PICS v3 observed-union + interface-preflight contract regressions (CPU only)."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.teh.pics_v3_contract_preflight import (
    build_admissible_history_variants,
    preflight_invalidates_candidate,
    run_interface_contract_preflight,
)
from utils.teh.pics_v3_observed import (
    count_pooled_observed_loglik,
    mean_test_loglik_with_failure_policy,
    merge_observed_trials,
    observed_trial_identities,
    observed_union_runtime_valid,
    trial_observed_identity,
    uses_pics_v3_observed_union,
)
from utils.teh.t_pics_gated_transfer import (
    decide_gate,
    participant_train_val_loglik,
)


def _trial(action: int, block: int, hist: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    return {
        "action": action,
        "problem": {
            "option_keys": ["A", "B"],
            "block_index": block,
            "schema_type": "toy",
        },
        "history": list(hist or []),
        "options": ["A", "B"],
    }


def _partition(union: List[Dict[str, Any]], n_train: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return list(union[:n_train]), list(union[n_train:])


def test_uses_pics_v3_observed_union_only_for_v3():
    assert uses_pics_v3_observed_union("structure_aware_v3")
    assert not uses_pics_v3_observed_union("structure_aware_v2")
    assert not uses_pics_v3_observed_union("structure_aware")
    assert not uses_pics_v3_observed_union("off")


def test_merge_observed_trials_partition_invariant_and_chronological():
    union = [
        _trial(0, 0, []),
        _trial(1, 0, [{"action": 0, "feedback": 1.0}]),
        _trial(0, 1, []),
        _trial(1, 1, [{"action": 0}]),
        _trial(1, 2, [{"action": 1, "feedback": -1.0}]),
    ]
    partitions = [
        _partition(union, 4),
        _partition(union, 2),
        _partition(union, 1),
        _partition(union, 0),  # all in val
        (union[::2], union[1::2]),  # interleaved artificial split
    ]
    ref = merge_observed_trials(*partitions[0])
    ref_ids = observed_trial_identities(ref)
    assert len(ref_ids) == len(union)
    # Chronological by block then history length
    blocks = [t["problem"]["block_index"] for t in ref]
    assert blocks == sorted(blocks)
    for train, val in partitions:
        merged = merge_observed_trials(train, val)
        assert observed_trial_identities(merged) == ref_ids
        assert [trial_observed_identity(t) for t in merged] == ref_ids


def test_count_pooled_observed_score_partition_invariant():
    # Same per-trial LLs: train mean -0.5 over 3, val mean -1.0 over 1 → union -0.625
    score_a = count_pooled_observed_loglik(-0.5, -1.0, 3, 1)
    score_b = count_pooled_observed_loglik(-0.625, None, 4, 0)
    score_c = count_pooled_observed_loglik(-1.0, -0.5, 1, 3)
    assert score_a == pytest.approx(-0.625)
    assert score_b == pytest.approx(-0.625)
    assert score_c == pytest.approx(-0.625)
    # Error / nonfinite on either non-empty legacy split rejects
    assert count_pooled_observed_loglik(-0.5, float("-inf"), 3, 1) is None
    assert count_pooled_observed_loglik(float("-inf"), -1.0, 3, 1) is None
    assert participant_train_val_loglik(-0.5, float("-inf"), 3, 1) is None
    assert participant_train_val_loglik(-0.5, -1.0, 3, 1) == pytest.approx(-0.625)


def test_observed_union_runtime_valid_rejects_either_split_error():
    assert observed_union_runtime_valid(
        train_errors=0,
        val_errors=0,
        n_train=3,
        n_val=1,
        train_loglik=-0.4,
        val_loglik=-0.5,
    )
    assert not observed_union_runtime_valid(
        train_errors=0,
        val_errors=1,
        n_train=3,
        n_val=1,
        train_loglik=-0.4,
        val_loglik=float("-inf"),
    )
    assert not observed_union_runtime_valid(
        train_errors=2,
        val_errors=0,
        n_train=3,
        n_val=1,
        train_loglik=float("-inf"),
        val_loglik=-0.5,
    )


def test_legacy_train_only_fallback_rejected_under_strict_evolution_score():
    import teh as teh_mod

    # Legacy: -inf val → train-only fallback
    legacy = teh_mod._evolution_selection_score(
        -0.1,
        float("-inf"),
        10,
        5,
        evolution_selection_score="train_val",
        strict_observed_union=False,
    )
    assert legacy == pytest.approx(-0.1)
    # PICS v3: rejected
    strict = teh_mod._evolution_selection_score(
        -0.1,
        float("-inf"),
        10,
        5,
        evolution_selection_score="train_val",
        strict_observed_union=True,
    )
    assert strict == float("-inf")
    fitness, sel = teh_mod._apply_evolution_candidate_selection_fitness(
        train_loglik=-0.1,
        val_loglik=float("-inf"),
        train_acc=0.9,
        fitness_metric="loglik",
        n_train=10,
        n_val=5,
        evolution_selection_score="train_val",
        use_train_val_selection=True,
        warn_key="test",
        runtime_valid=True,
        strict_observed_union=True,
    )
    assert sel == float("-inf")
    assert fitness == float("-inf")


def test_gate_decision_uses_count_pooled_scores_and_rejects_nonfinite():
    # Transfer wins on count-pooled objective
    decision = decide_gate(control_score=-0.50, transfer_score=-0.40)
    assert decision.selected_arm == "transfer"
    # Missing / nonfinite transfer → keep control
    decision2 = decide_gate(control_score=-0.50, transfer_score=None)
    assert decision2.selected_arm == "control"
    decision3 = decide_gate(control_score=-0.50, transfer_score=float("-inf"))
    assert decision3.selected_arm == "control"


def test_prompt_cap_uses_shared_union_not_separate_caps():
    import teh as teh_mod

    train = [_trial(i % 2, i // 2) for i in range(40)]
    val = [_trial(i % 2, 100 + i) for i in range(20)]
    capped_train, capped_val = teh_mod._cap_prompt_train_and_val_trials(
        train,
        val,
        max_trials=10,
        max_trials_per_problem=5,
        subsample_seed=0,
    )
    assert len(capped_train) + len(capped_val) == 10
    # Separate caps would allow up to 10+10; union cap forbids that.
    assert len(capped_train) <= 10
    assert len(capped_val) <= 10


def test_no_test_trials_in_prompt_or_selection_helpers():
    train = [_trial(0, 0)]
    val = [_trial(1, 1)]
    test = [_trial(0, 99, [{"action": 1, "feedback": 9.0}])]
    merged = merge_observed_trials(train, val)
    assert all(t["problem"]["block_index"] != 99 for t in merged)
    # Selection helpers do not take test
    assert participant_train_val_loglik(-0.2, -0.3, 1, 1) is not None
    score = count_pooled_observed_loglik(-0.2, -0.3, 1, 1)
    assert score == pytest.approx(-0.25)
    del test  # explicit: unused by selection


def test_interface_preflight_rejects_strict_feedback_access():
    def bad_choose(problem, history):
        if history:
            return 0.7 if history[-1]["feedback"] > 0 else 0.3
        return 0.5

    def good_choose(problem, history):
        if history:
            fb = history[-1].get("feedback")
            if fb is None:
                return 0.5
            return 0.7 if fb > 0 else 0.3
        return 0.5

    observed = [
        _trial(1, 0, [{"action": 0, "feedback": 1.0}]),
        _trial(0, 1, [{"action": 1}]),  # action-only history present in observed
    ]
    invalid_bad, report_bad = preflight_invalidates_candidate(bad_choose, observed)
    assert invalid_bad
    assert report_bad["test_trials_used"] is False
    assert any("feedback" in (f.get("error_message") or "") or f.get("error_type") == "KeyError"
               for f in report_bad["failures"])

    invalid_good, report_good = preflight_invalidates_candidate(good_choose, observed)
    assert not invalid_good
    assert report_good["ok"] is True

    variants = build_admissible_history_variants([{"action": 0, "feedback": 1.0}])
    labels = {lab for lab, _ in variants}
    assert "action_only_no_outcomes" in labels
    assert "missing_feedback_and_reward" in labels


def test_mean_test_loglik_never_silently_finite_only():
    ok = mean_test_loglik_with_failure_policy([-0.5, -0.4, -0.6])
    assert ok["mean"] == pytest.approx(-0.5)
    assert ok["n_nonfinite"] == 0
    bad = mean_test_loglik_with_failure_policy([-0.5, float("-inf"), -0.6])
    assert bad["mean"] is None
    assert bad["n_nonfinite"] == 1
    assert bad["n_finite"] == 2
    assert bad["policy"] == "require_all_finite"


def test_valid_seed_keeps_pool_nonempty_semantics():
    """Seed with finite observed score remains selectable when others fail."""
    import teh as teh_mod

    seed_sel = teh_mod._evolution_selection_score(
        -0.55,
        -0.60,
        30,
        10,
        evolution_selection_score="train_val",
        strict_observed_union=True,
    )
    crash_sel = teh_mod._evolution_selection_score(
        -0.01,
        float("-inf"),
        30,
        10,
        evolution_selection_score="train_val",
        strict_observed_union=True,
    )
    assert math.isfinite(seed_sel)
    assert crash_sel == float("-inf")
    assert seed_sel > crash_sel


def test_elite_ordering_prefers_finite_observed_over_train_only_crash():
    import teh as teh_mod

    rows = [
        {
            "idx": 0,
            "fitness": teh_mod._evolution_selection_score(
                -0.01, float("-inf"), 20, 10,
                evolution_selection_score="train_val",
                strict_observed_union=True,
            ),
            "runtime_valid": False,
        },
        {
            "idx": 1,
            "fitness": teh_mod._evolution_selection_score(
                -0.40, -0.50, 20, 10,
                evolution_selection_score="train_val",
                strict_observed_union=True,
            ),
            "runtime_valid": True,
        },
    ]
    valid = [r for r in rows if r["runtime_valid"] and math.isfinite(r["fitness"])]
    assert len(valid) == 1
    assert valid[0]["idx"] == 1


def test_runtime_contract_documents_optional_history_outcomes():
    from utils.teh.prompt_context import build_deterministic_runtime_contract

    trials = [_trial(0, 0, [{"action": 0, "feedback": 1.0}])]
    contract = build_deterministic_runtime_contract(trials)
    assert "feedback" in contract.lower()
    assert "ABSENT" in contract or "absent" in contract.lower()
    assert ".get()" in contract
