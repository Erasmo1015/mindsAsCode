"""CPU tests: Centaur SA40-fair timeline has zero extra visible pretest rows."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "baseline_methods" / "Psych101"))

from Centaur import (  # noqa: E402
    _centaur_history_span,
    _centaur_prompt_timeline,
    _centaur_prompt_timeline_sa40_fair,
    _centaur_prompt_timeline_v2,
    _centaur_uses_sa40_fair_timeline,
)
from utils.teh.limited_data_registry import (  # noqa: E402
    CONTINUOUS_SESSION,
    INDEPENDENT_TRIAL,
    RESETTING_UNIT,
)


def _row(tag: str, action: int = 0, history=None):
    return {
        "problem": {"dataset_alias": "synthetic", "tag": tag, "option_keys": ["A", "B"]},
        "history": list(history or []),
        "action": int(action),
    }


def _pretest_tags_in_history_spans(prompt, score_indices, n_pretest: int):
    """Tags of pretest rows that appear in any scored trial's history span."""
    seen = set()
    for idx in score_indices:
        start, hist = _centaur_history_span(prompt, idx)
        for j in range(len(hist)):
            pos = start + j
            if pos < n_pretest:
                seen.add(prompt[pos]["problem"]["tag"])
    return seen


def test_independent_timeline_is_retained_only():
    retained = [_row(f"ret{i}") for i in range(4)]
    # omitted raw TV that must NOT appear
    omitted = [_row(f"omit{i}") for i in range(6)]
    test = [_row(f"te{i}", history=[]) for i in range(3)]
    prompt, scores = _centaur_prompt_timeline_sa40_fair(
        retained[:3],
        retained[3:],
        test,
        category=INDEPENDENT_TRIAL,
        raw_train=retained + omitted[:3],
        raw_val=omitted[3:],
    )
    n_pre = len(prompt) - len(test)
    assert n_pre == len(retained)
    assert scores == [4, 5, 6]
    pretest_tags = {t["problem"]["tag"] for t in prompt[:n_pre]}
    assert pretest_tags == {f"ret{i}" for i in range(4)}
    assert not any(tag.startswith("omit") for tag in pretest_tags)
    # Empty histories → zero pretest rows enter the serialized history span.
    assert _pretest_tags_in_history_spans(prompt, scores, n_pre) == set()


def test_resetting_timeline_retained_plus_within_unit_test_history():
    retained = [_row(f"ret{i}") for i in range(4)]
    omitted = [_row(f"omit{i}") for i in range(5)]
    # One held-out unit of 3 trials; history only within the unit.
    t0 = _row("u0_0", history=[])
    t1 = _row("u0_1", history=[{"action": 0}])
    t2 = _row("u0_2", history=[{"action": 0}, {"action": 1}])
    test = [t0, t1, t2]
    prompt, scores = _centaur_prompt_timeline_sa40_fair(
        retained,
        [],
        test,
        category=RESETTING_UNIT,
        raw_train=retained + omitted,
        raw_val=[],
    )
    n_pre = len(prompt) - len(test)
    assert n_pre == 4
    assert not any(t["problem"]["tag"].startswith("omit") for t in prompt[:n_pre])
    # History spans for test trials must land on prior TEST rows, not retained/omitted.
    for idx in scores[1:]:
        start, hist = _centaur_history_span(prompt, idx)
        assert start >= n_pre
        for j in range(len(hist)):
            assert prompt[start + j]["problem"]["tag"].startswith("u0_")


def test_continuous_includes_omitted_session_priors_without_duplicating_retained():
    # Chrono session: 6 raw TV (first 2 = retained SA40), then 2 test trials.
    raw_tv = [_row(f"s{i}", action=i % 2) for i in range(6)]
    retained = raw_tv[:2]
    test0_hist = [{"action": t["action"]} for t in raw_tv]  # all pretest in PICS history
    test0 = _row("te0", history=test0_hist)
    test1_hist = test0_hist + [{"action": 0}]
    test1 = _row("te1", history=test1_hist)
    prompt, scores = _centaur_prompt_timeline_sa40_fair(
        retained,
        [],
        [test0, test1],
        category=CONTINUOUS_SESSION,
        raw_train=raw_tv[:4],
        raw_val=raw_tv[4:],
    )
    n_pre = len(prompt) - 2
    assert n_pre == 6  # full chrono pretest once
    assert scores == [6, 7]
    # No duplicated retained tags in pretest section.
    pretest_tags = [t["problem"]["tag"] for t in prompt[:n_pre]]
    assert pretest_tags == [f"s{i}" for i in range(6)]
    assert len(pretest_tags) == len(set(pretest_tags))
    # History span for first test covers exactly the pretest (required), no extras beyond.
    start, hist = _centaur_history_span(prompt, scores[0])
    assert start == 0 and len(hist) == 6
    visible_pretest = _pretest_tags_in_history_spans(prompt, scores, n_pre)
    assert visible_pretest == {f"s{i}" for i in range(6)}


def test_unknown_category_rejected():
    with pytest.raises(ValueError, match="Unknown limited-data category"):
        _centaur_prompt_timeline_sa40_fair(
            [_row("a")],
            [],
            [_row("t")],
            category="not_a_category",
        )


def test_sa40_fair_timeline_activated_only_for_structure_aware_v3():
    """Regression: fair gate is v3-only; v1/off/v2 keep legacy timelines."""
    assert _centaur_uses_sa40_fair_timeline("structure_aware_v3") is True
    assert _centaur_uses_sa40_fair_timeline("structure_aware") is False
    assert _centaur_uses_sa40_fair_timeline("structure_aware_v2") is False
    assert _centaur_uses_sa40_fair_timeline("off") is False

    retained = [_row(f"ret{i}") for i in range(4)]
    omitted = [_row(f"omit{i}") for i in range(6)]
    test = [_row(f"te{i}", history=[]) for i in range(2)]
    raw_train = retained + omitted[:3]
    raw_val = omitted[3:]

    # v3 fair (independent): retained only — no omitted TV.
    fair, fair_scores = _centaur_prompt_timeline_sa40_fair(
        retained[:3],
        retained[3:],
        test,
        category=INDEPENDENT_TRIAL,
        raw_train=raw_train,
        raw_val=raw_val,
    )
    assert _centaur_uses_sa40_fair_timeline("structure_aware_v3")
    assert len(fair) - len(test) == len(retained)
    assert not any(t["problem"]["tag"].startswith("omit") for t in fair[:-2])
    assert fair_scores == [4, 5]

    # Legacy structure_aware / off: retained (or full-off) concat unchanged.
    v1, v1_scores = _centaur_prompt_timeline(retained[:3], retained[3:], test)
    assert v1 == retained + test
    assert v1_scores == [4, 5]
    assert len(v1) == len(fair)  # same retained+test shape as fair for independent

    # Legacy structure_aware_v2: raw TV dump still includes omitted rows.
    v2, v2_scores = _centaur_prompt_timeline_v2(raw_train, raw_val, test)
    assert len(v2) - len(test) == len(raw_train) + len(raw_val)
    assert any(t["problem"]["tag"].startswith("omit") for t in v2[:-2])
    assert v2_scores == [10, 11]
    assert len(v2) > len(fair)
