"""Centaur SA40 prefix: no negative wrap; score test on the obs+test timeline."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baseline_methods" / "Psych101"))

from Centaur import (  # noqa: E402
    _centaur_history_span,
    _centaur_prompt_timeline,
    build_centaur_prompt_prefix_indexed,
)


def _weather_trial(cards: int, history, action: int = 0):
    return {
        "problem": {
            "schema_type": "B",
            "cards": [cards],
            "option_keys": ["f", "s"],
            "dataset_alias": "5speekenbrink2008learning",
        },
        "history": list(history),
        "action": action,
    }


def test_history_span_rejects_test_only_list_with_obs_history():
    test_only = [_weather_trial(222, [{"action": 0}] * 5)]
    with pytest.raises(ValueError, match="history_len=5 exceeds trial_index=0"):
        _centaur_history_span(test_only, 0)
    with pytest.raises(ValueError, match="negative wrap"):
        build_centaur_prompt_prefix_indexed(test_only, 0)


def test_concat_timeline_uses_obs_problem_not_later_test():
    obs = _weather_trial(111, [])
    later_test = _weather_trial(999, [{"action": 0}, {"action": 1}])
    test0 = _weather_trial(222, [{"action": 0}])
    prompt, score = _centaur_prompt_timeline([obs], [], [test0, later_test])
    assert score == [1, 2]
    prefix = build_centaur_prompt_prefix_indexed(prompt, 1)
    assert "card 111" in prefix
    assert "card 999" not in prefix
    assert "card 222" in prefix
    assert prefix.rstrip().endswith("You press")


def test_prompt_timeline_scores_only_test_tail():
    train = [_weather_trial(i, []) for i in range(3)]
    val = [_weather_trial(10 + i, []) for i in range(2)]
    test = [_weather_trial(20 + i, []) for i in range(4)]
    prompt, score = _centaur_prompt_timeline(train, val, test)
    assert len(prompt) == 9
    assert score == [5, 6, 7, 8]
    assert prompt[score[0]]["problem"]["cards"] == [20]
