"""CPC18 prompt-example feedback-regime coverage (PICS-only; no loader mutation)."""
from __future__ import annotations

from utils.teh.pics_v3_cpc18_prompt import (
    classify_cpc18_feedback_regime,
    ensure_cpc18_feedback_regime_examples,
)


def _trial(block: int, hist_fb) -> dict:
    history = []
    for i, fb in enumerate(hist_fb):
        entry = {"action": i % 2}
        if fb is not None:
            entry["feedback"] = fb
        history.append(entry)
    return {
        "problem": {
            "dataset_alias": "2plonsky2018when",
            "block_index": block,
            "gamble_A": {"probs": [1.0], "rewards": [1.0]},
            "gamble_B": {"probs": [1.0], "rewards": [0.0]},
            "has_feedback": True,
            "option_keys": ["A", "B"],
            "schema_type": "A",
        },
        "history": history,
        "action": 0,
    }


def test_non_cpc18_noop() -> None:
    pool = [_trial(0, [None]), _trial(0, [1.0])]
    selected = [pool[1]]
    out, diag = ensure_cpc18_feedback_regime_examples(
        selected, pool, dataset="3frey2017cct"
    )
    assert out == selected
    assert diag["applied"] is False


def test_ensures_both_regimes_when_pool_has_both() -> None:
    no_fb = _trial(0, [None, None])
    has_fb = _trial(0, [None, 2.5])
    pool = [no_fb, has_fb, _trial(1, [3.0])]
    # Selected only feedback regime
    out, diag = ensure_cpc18_feedback_regime_examples(
        [has_fb], pool, dataset="2plonsky2018when"
    )
    assert diag["applied"] is True
    regimes = {classify_cpc18_feedback_regime(t) for t in out}
    assert regimes == {"no_feedback", "feedback"}


def test_already_covered_unchanged() -> None:
    no_fb = _trial(0, [])
    has_fb = _trial(0, [1.0])
    selected = [no_fb, has_fb]
    out, diag = ensure_cpc18_feedback_regime_examples(
        selected, selected, dataset="2plonsky2018when"
    )
    assert diag["action"] == "already_covered"
    assert out == selected


def test_pool_missing_regime_noop() -> None:
    only_fb = [_trial(0, [1.0]), _trial(1, [2.0])]
    out, diag = ensure_cpc18_feedback_regime_examples(
        only_fb[:1], only_fb, dataset="2plonsky2018when"
    )
    assert diag["action"] == "pool_missing_regime"
    assert out == only_fb[:1]
