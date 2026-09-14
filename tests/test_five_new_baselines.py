from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("openai", MagicMock())

import numpy as np

from baseline_methods.MLE import (
    eval_mean_loglik_softmax_arm_means,
    fit_softmax_arm_means,
    predict_action_softmax_arm_means,
)
from baseline_methods.psych101_features import (
    _task_kind,
    option_b_feature_diff,
    prospect_gamble_getters,
)
from baseline_methods.prospect_theory import (
    eval_mean_loglik_categorical_prospect,
    fit_prospect_theory_categorical,
)
from utils.teh.teh_datasets import is_categorical_output_dataset


def test_new_dataset_output_types() -> None:
    assert not is_categorical_output_dataset("bergert_nosofsky_2007")
    assert not is_categorical_output_dataset("guan_2020_stopping")
    assert is_categorical_output_dataset("steyvers_2009_bandit")
    assert is_categorical_output_dataset("13schulz2020finding")
    assert not is_categorical_output_dataset("14kool2016when")


def test_bergert_features_action1_is_option_a() -> None:
    problem = {
        "schema_type": "bergert_pairwise",
        "option_A": {"cues": {"cue1": 1, "cue2": 1, "cue3": 0, "cue4": 0, "cue5": 0, "cue6": 0}},
        "option_B": {"cues": {"cue1": 0, "cue2": 0, "cue3": 0, "cue4": 0, "cue5": 0, "cue6": 0}},
        "action_means_option_A_when_1": True,
    }
    assert _task_kind(problem) == "bergert_pairwise"
    # More cues on A, action=1 selects A → positive x.
    assert option_b_feature_diff(problem) == 2.0
    ga, gb = prospect_gamble_getters(problem)
    assert ga(problem, None)[0] == [0.0]
    assert gb(problem, None)[0] == [2.0]


def test_guan_stop_feature_uses_last_observed() -> None:
    problem = {
        "schema_type": "guan_stopping",
        "values_observed": [1.0, 4.0, 9.0],
    }
    assert _task_kind(problem) == "guan_stopping"
    assert option_b_feature_diff(problem) == 9.0
    ga, gb = prospect_gamble_getters(problem)
    assert ga(problem, None)[0] == [0.0]
    assert gb(problem, None)[0] == [9.0]


def test_kool_stage2_uses_matching_letter_rewards() -> None:
    problem = {
        "schema_type": "kool_twostep",
        "stage": 2,
        "option_keys": ["G", "R"],
        "alien_options": ["G", "R"],
    }
    history = [
        {"stage": 2, "action": 1, "option_keys": ["G", "R"], "reward": 1},
        {"stage": 2, "action": 0, "option_keys": ["G", "R"], "reward": 0},
        {"stage": 2, "action": 1, "option_keys": ["G", "R"], "reward": 1},
    ]
    assert _task_kind(problem) == "kool_twostep"
    # R mean=1, G mean=0 → x = 1 - 0
    assert option_b_feature_diff(problem, history) == 1.0
    ga, gb = prospect_gamble_getters(problem)
    assert ga(problem, history)[0] == [0.0]
    assert gb(problem, history)[0] == [1.0]


def test_categorical_softmax_mle_prefers_rich_arm() -> None:
    trials = []
    for _ in range(12):
        trials.append(
            {
                "problem": {"schema_type": "categorical_bandit", "n_arms": 4, "option_keys": [0, 1, 2, 3]},
                "history": [
                    {"action": 0, "reward": 0},
                    {"action": 1, "reward": 1},
                    {"action": 1, "reward": 1},
                    {"action": 2, "reward": 0},
                ],
                "action": 1,
            }
        )
    assert _task_kind(trials[0]["problem"]) == "categorical_bandit"
    beta = fit_softmax_arm_means(trials)
    pred = predict_action_softmax_arm_means(beta, trials[0]["problem"], trials[0]["history"])
    assert pred == 1
    ll = eval_mean_loglik_softmax_arm_means(trials, beta)
    assert np.isfinite(ll)
    assert ll > np.log(0.25)


def test_categorical_prospect_fits() -> None:
    trials = [
        {
            "problem": {"schema_type": "categorical_bandit", "n_arms": 3, "option_keys": [0, 1, 2]},
            "history": [{"action": 2, "reward": 1}, {"action": 0, "reward": 0}],
            "action": 2,
        }
        for _ in range(8)
    ]
    params = fit_prospect_theory_categorical(trials, dataset="steyvers_2009_bandit", participant_id=0)
    ll = eval_mean_loglik_categorical_prospect(trials, params)
    assert np.isfinite(ll)
    assert all(k in params for k in ("alpha", "lambda", "gamma", "beta"))


FORBIDDEN_PROBLEM_KEYS = {
    "action",
    "choice",
    "reward",
    "correct",
    "correct_category",
    "cue_validities",
    "log_odds_weight",
    "_full_values",
    "_full_values_len",
    "_stop_position",
    "stop_position",
    "full_values",
    "cond",
    "rcond",
}


def _feature_payload(trial: dict):
    problem = trial["problem"]
    history = trial.get("history")
    kind = _task_kind(problem)
    x = option_b_feature_diff(problem, history)
    if kind == "categorical_bandit":
        from baseline_methods.psych101_features import categorical_arm_means

        return kind, x, tuple(categorical_arm_means(problem, history))
    ga, gb = prospect_gamble_getters(problem)
    return kind, x, (ga(problem, history), gb(problem, history))


def _assert_no_current_action_leak(trials: list) -> None:
    for trial in trials:
        problem = trial["problem"]
        assert FORBIDDEN_PROBLEM_KEYS.isdisjoint(problem), problem.keys()
        assert "action" not in problem
        before = _feature_payload(trial)
        flipped = dict(trial)
        flipped["action"] = int(trial["action"]) + 1
        after = _feature_payload(flipped)
        assert before == after


def test_lm_pt_features_ignore_current_action_label() -> None:
    """x / PT getters may use problem+history only; the current action is the label."""
    _assert_no_current_action_leak(
        [
            {
                "problem": {
                    "schema_type": "bergert_pairwise",
                    "option_A": {"cues": {"cue1": 1, "cue2": 0}},
                    "option_B": {"cues": {"cue1": 0, "cue2": 1}},
                    "action_means_option_A_when_1": True,
                },
                "history": [],
                "action": 1,
            },
            {
                "problem": {"schema_type": "guan_stopping", "values_observed": [3.0, 8.0]},
                "history": [{"action": 0, "position": 1, "value": 3.0}],
                "action": 1,
            },
            {
                "problem": {
                    "schema_type": "kool_twostep",
                    "stage": 2,
                    "option_keys": ["G", "R"],
                    "alien_options": ["G", "R"],
                    "stage1_action": 0,
                },
                "history": [
                    {"stage": 1, "action": 0, "option_keys": ["B", "L"]},
                    {"stage": 2, "action": 1, "option_keys": ["G", "R"], "reward": 1},
                ],
                "action": 0,
            },
            {
                "problem": {
                    "schema_type": "categorical_bandit",
                    "n_arms": 4,
                    "option_keys": [0, 1, 2, 3],
                },
                "history": [{"action": 1, "reward": 1}, {"action": 0, "reward": 0}],
                "action": 3,
            },
        ]
    )


def test_live_loaders_match_dataset_chat_leakage_contract() -> None:
    """Same TEH trial dicts as Dataset implementation chat: action is top-level only."""
    from pathlib import Path

    import pytest

    from baseline_methods.MLE import resolve_participants_for_scope, trials_for_participant

    repo = Path("/common/home/users/z/zichang.ge.2023/repo/mindsAsCode")
    datasets = [
        "bergert_nosofsky_2007",
        "guan_2020_stopping",
        "steyvers_2009_bandit",
    ]
    for ds in datasets:
        try:
            pids = resolve_participants_for_scope(
                dataset=ds,
                repo_root=repo,
                participant_scope="range",
                single_participant_id=0,
                range_start_ordinal=0,
                range_end_ordinal=0,
                all_max_participants=None,
                participant_ordinals=None,
                filter_mixed_gambles=False,
                split_ratio=0.6,
                split_seed=0,
            )
            train, val, test = trials_for_participant(
                ds,
                pids[0],
                split_ratio=0.6,
                split_seed=0,
                filter_mixed_gambles=False,
                psych_dataset_split="train",
                local_dataset=None,
                mixed_gambles_csv="",
            )
        except (FileNotFoundError, ValueError) as e:
            pytest.skip(f"{ds}: {e}")
        trials = train + val + test
        assert trials
        _assert_no_current_action_leak(trials)
        if ds == "bergert_nosofsky_2007":
            assert all(t["history"] == [] for t in trials)
        if ds == "guan_2020_stopping":
            for t in trials:
                p = t["problem"]
                assert len(p["values_observed"]) == int(p["position"])

