"""Lightweight smoke checks for categorical TEH Psych utilities."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.teh_psych.adapters import binary_trial_to_categorical, convert_binary_trials_to_categorical
from utils.teh_psych.categorical_eval import coerce_choose_output, evaluate_categorical_program


def _trial_k2(target: int = 1):
    return {
        "problem": {
            "options": [{"action": 0, "label": "A"}, {"action": 1, "label": "B"}],
        },
        "history": [],
        "action": target,
        "target_action": target,
    }


def _trial_k3(target: int = 2):
    return {
        "problem": {
            "options": [
                {"action": 0},
                {"action": 1},
                {"action": 2},
            ],
        },
        "history": [],
        "action": target,
        "target_action": target,
    }


def test_coerce_k2_dict():
    probs, warnings = coerce_choose_output({0: 0.35, 1: 0.65}, [0, 1])
    assert warnings == []
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs[1] == pytest.approx(0.65)


def test_coerce_k3_dict():
    probs, warnings = coerce_choose_output({0: 0.2, 1: 0.5, 2: 0.3}, [0, 1, 2])
    assert not warnings
    assert len(probs) == 3


def test_legacy_float_k2():
    probs, warnings = coerce_choose_output(0.7, [0, 1])
    assert not warnings
    assert probs[0] == pytest.approx(0.3)
    assert probs[1] == pytest.approx(0.7)


def test_invalid_return_fallback():
    probs, warnings = coerce_choose_output("bad", [0, 1, 2])
    assert warnings
    assert len(probs) == 3
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_evaluate_k2_trial():
    def choose(problem, history):
        return {0: 0.4, 1: 0.6}

    metrics = evaluate_categorical_program(choose, [_trial_k2(1)])
    assert metrics["avg_loglik"] > -1.0
    assert metrics["avg_accuracy"] == 1.0


def test_evaluate_k3_trial():
    def choose(problem, history):
        return {0: 0.1, 1: 0.2, 2: 0.7}

    metrics = evaluate_categorical_program(choose, [_trial_k3(2)])
    assert metrics["avg_accuracy"] == 1.0


def test_legacy_float_eval_k2():
    def choose(problem, history):
        return 0.8

    metrics = evaluate_categorical_program(choose, [_trial_k2(1)])
    assert metrics["errors"] == 0
    assert metrics["avg_loglik"] == pytest.approx(__import__("math").log(0.8), rel=1e-3)


def test_invalid_eval_does_not_crash():
    def choose(problem, history):
        return 0.5

    metrics = evaluate_categorical_program(choose, [_trial_k3(1)])
    assert metrics["n_trials"] == 1
    assert metrics["warnings"] >= 0


def test_binary_to_categorical_adapter():
    binary = {
        "problem": {
            "option_keys": ["P", "U"],
            "gamble_A": {"probs": [0.5, 0.5], "rewards": [10, -5]},
            "gamble_B": {"probs": [1.0], "rewards": [3]},
            "has_feedback": True,
            "schema_type": "A",
        },
        "history": [],
        "options": ["P", "U"],
        "action": 1,
    }
    cat = binary_trial_to_categorical(binary)
    opts = cat["problem"]["options"]
    assert len(opts) == 2
    assert opts[0]["action"] == 0
    assert opts[1]["action"] == 1
    assert cat["target_action"] == 1
    batch = convert_binary_trials_to_categorical([binary])
    assert batch[0]["problem"]["options"][1]["label"] == "U"
