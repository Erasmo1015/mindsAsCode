from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock

import pytest

from baseline_methods.psych101_features import (
    _category_correct_from_history,
    _task_kind,
    option_b_feature_diff,
    prospect_gamble_getters,
)
from data_modules.psych101_binary import (
    experiment_to_trial_dicts,
    get_psych101_binary_experiment,
    split_psych_experiment,
)


def _load_format_trial_compact():
    """Import OpenEvolve trial formatter without requiring the openevolve package."""
    for name in (
        "openevolve",
        "openevolve.config",
        "openevolve.process_parallel",
        "utils.psych101_openevolve_pool",
    ):
        sys.modules.setdefault(name, MagicMock())
    if "utils.psych101_openevolve_pool" in sys.modules:
        sys.modules["utils.psych101_openevolve_pool"].WORKER_VANILLA = "vanilla"
    from baseline_methods.Psych101.run_openevolve import format_trial_compact

    return format_trial_compact


format_trial_compact = _load_format_trial_compact()


FORBIDDEN_CURRENT = {
    "2plonsky2018when": set(),
    "5speekenbrink2008learning": {"weather_outcome", "was_correct"},
    "10frey2017risk": {"outcome_marker", "exploded"},
    "12badham2017deficits": {"response_key", "correct_category"},
}


def _participant0(alias: str):
    exp = get_psych101_binary_experiment(alias, 0, split="train")
    trials = experiment_to_trial_dicts(exp)
    splits = split_psych_experiment(exp, split_ratio=0.6, split_seed=0)
    return exp, trials, splits


@pytest.mark.parametrize(
    "alias",
    [
        "2plonsky2018when",
        "5speekenbrink2008learning",
        "10frey2017risk",
        "12badham2017deficits",
    ],
)
def test_baselines_do_not_consume_current_post_choice_fields(alias: str) -> None:
    _, trials, _ = _participant0(alias)
    forbidden = FORBIDDEN_CURRENT[alias]
    for trial in trials:
        problem = trial["problem"]
        assert forbidden.isdisjoint(problem)
        kind = _task_kind(problem)
        feature = option_b_feature_diff(problem, trial.get("history"))
        assert isinstance(feature, float)
        ga, gb = prospect_gamble_getters(problem)
        ga(problem, trial.get("history"))
        gb(problem, trial.get("history"))
        compact = format_trial_compact(trial, "train")
        assert "correct=" not in compact
        assert "outcome=" not in compact
        for field in forbidden:
            assert field not in problem
        assert kind


def test_badham_baselines_never_read_current_correct_category_or_response_key() -> None:
    from baseline_methods import psych101_features as features

    feature_source = inspect.getsource(features.option_b_feature_diff)
    getters_source = inspect.getsource(features._gamble_getters_category_learning)
    history_source = inspect.getsource(features._category_correct_from_history)
    oe_source = inspect.getsource(format_trial_compact)

    problem = {
        "schema_type": "B",
        "task": "category_learning",
        "option_keys": ["E", "K"],
        "stimulus_features": {"size": "big", "color": "black", "shape": "square"},
        "correct_category": "K",
        "response_key": "E",
    }
    history = [
        {
            "action": 1,
            "stimulus_features": {"size": "big", "color": "black", "shape": "square"},
            "feedback": {"correct_category": "E", "is_correct": False},
            "correct_category": "E",
            "response_key": "K",
        }
    ]
    assert _task_kind(problem) == "category_learning"
    assert _category_correct_from_history(problem, history) == "E"
    assert option_b_feature_diff(problem, history) == -1.0
    assert option_b_feature_diff(problem, []) == 0.0
    ga, gb = prospect_gamble_getters(problem)
    assert ga(problem, history) == ([1.0], [1.0])
    assert gb(problem, history) == ([-1.0], [1.0])
    assert ga(problem, []) == ([0.0], [1.0])
    assert gb(problem, []) == ([0.0], [1.0])

    compact = format_trial_compact(
        {"problem": problem, "history": history, "action": 0},
        "train",
    )
    assert "correct=K" not in compact
    assert "response_key" not in compact
    assert "stim=(big,black,square)" in compact
    assert "corr=E" in compact

    assert 'problem.get("correct_category"' not in feature_source
    assert 'p.get("correct_category"' not in getters_source
    assert 'problem.get("correct_category"' not in history_source
    assert "correct=" not in oe_source
    assert "outcome=" not in oe_source

def test_badham_matched_features_use_earlier_feedback_only() -> None:
    _, trials, splits = _participant0("12badham2017deficits")
    seen: dict[tuple, int] = {}
    matched = None
    for i, trial in enumerate(trials):
        sf = tuple(sorted((trial["problem"].get("stimulus_features") or {}).items()))
        if sf in seen:
            matched = (seen[sf], i, trial)
            break
        seen[sf] = i
    assert matched is not None
    _, _, trial = matched
    assert "correct_category" not in trial["problem"]
    assert "response_key" not in trial["problem"]
    prior = _category_correct_from_history(trial["problem"], trial["history"])
    assert prior
    assert prior == next(
        (
            str(
                (
                    entry.get("feedback") or {}
                ).get("correct_category")
                or entry.get("correct_category")
            ).upper()
            for entry in reversed(trial["history"])
            if (entry.get("stimulus_features") or {})
            == trial["problem"]["stimulus_features"]
        ),
        "",
    )
    keys = trial["problem"]["option_keys"]
    expected = 1.0 if prior == str(keys[1]).upper() else -1.0
    assert option_b_feature_diff(trial["problem"], trial["history"]) == expected
    assert [len(s) for s in splits[:3]] == [192, 96, 96]


@pytest.mark.parametrize(
    ("alias", "kind", "counts"),
    [
        ("2plonsky2018when", "gamble", [450, 150, 150]),
        ("5speekenbrink2008learning", "weather", [104, 48, 48]),
        ("10frey2017risk", "balloon_risk", [571, 251, 265]),
        ("12badham2017deficits", "category_learning", [192, 96, 96]),
    ],
)
def test_repaired_dataset_baseline_kinds_and_splits(
    alias: str, kind: str, counts: list[int]
) -> None:
    _, trials, splits = _participant0(alias)
    assert _task_kind(trials[0]["problem"]) == kind
    assert [len(s) for s in splits[:3]] == counts
    compact = format_trial_compact(trials[0], "train")
    if alias == "5speekenbrink2008learning":
        assert "cards=" in compact
        assert "weather_outcome" not in compact
    if alias == "10frey2017risk":
        assert "pump_n=" in compact
        assert "outcome=" not in compact
    if alias == "2plonsky2018when":
        assert "A=(" in compact
        assert "fb=1" in compact or "fb=0" in compact
