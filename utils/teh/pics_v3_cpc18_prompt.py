"""CPC18 (2plonsky2018when) PICS-only prompt example coverage helpers.

Does **not** mutate shared loaders, baseline-visible trial fields, or invent
forgone-outcome keys. Ensures selected prompt examples cover both early /
no-chosen-feedback and later / chosen-feedback regimes when both exist in
the observed pool.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

CPC18_ALIAS = "2plonsky2018when"


def is_cpc18_alias(dataset: Optional[str]) -> bool:
    return str(dataset or "").strip() == CPC18_ALIAS


def history_has_chosen_feedback(trial: Dict[str, Any]) -> bool:
    """True if any history entry exposes a non-None chosen-option ``feedback``."""
    history = trial.get("history") or []
    if not isinstance(history, list):
        return False
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if entry.get("feedback") is not None:
            return True
    return False


def classify_cpc18_feedback_regime(trial: Dict[str, Any]) -> str:
    """``feedback`` if chosen payoff appears in history; else ``no_feedback``."""
    return "feedback" if history_has_chosen_feedback(trial) else "no_feedback"


def ensure_cpc18_feedback_regime_examples(
    selected: Sequence[Dict[str, Any]],
    pool: Sequence[Dict[str, Any]],
    *,
    dataset: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Ensure ≥1 no-feedback and ≥1 feedback example when both exist in ``pool``.

    Preserves within-problem chronological order as much as possible: needed
    regimes are inserted by replacing a surplus example of the other regime
    (or appending if under capacity), preferring pool trials that keep block
    chronology. No-op for non-CPC18 aliases.
    """
    selected_list = list(selected)
    pool_list = list(pool)
    diag: Dict[str, Any] = {
        "dataset": str(dataset or ""),
        "applied": False,
        "pool_no_feedback": 0,
        "pool_feedback": 0,
        "selected_no_feedback": 0,
        "selected_feedback": 0,
        "action": "noop",
    }
    if not is_cpc18_alias(dataset):
        return selected_list, diag

    def _count(rows: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
        n_no = sum(1 for t in rows if classify_cpc18_feedback_regime(t) == "no_feedback")
        n_fb = sum(1 for t in rows if classify_cpc18_feedback_regime(t) == "feedback")
        return n_no, n_fb

    pool_no, pool_fb = _count(pool_list)
    sel_no, sel_fb = _count(selected_list)
    diag.update(
        {
            "pool_no_feedback": pool_no,
            "pool_feedback": pool_fb,
            "selected_no_feedback": sel_no,
            "selected_feedback": sel_fb,
        }
    )
    if pool_no == 0 or pool_fb == 0:
        diag["action"] = "pool_missing_regime"
        return selected_list, diag
    if sel_no > 0 and sel_fb > 0:
        diag["action"] = "already_covered"
        return selected_list, diag

    need = "no_feedback" if sel_no == 0 else "feedback"
    # Candidates from pool not already selected (by identity).
    selected_ids = {id(t) for t in selected_list}
    candidates = [
        t
        for t in pool_list
        if id(t) not in selected_ids and classify_cpc18_feedback_regime(t) == need
    ]
    if not candidates:
        # Allow reuse of a pool trial already conceptually present via equal content.
        candidates = [
            t for t in pool_list if classify_cpc18_feedback_regime(t) == need
        ]
    if not candidates:
        diag["action"] = "no_candidate"
        return selected_list, diag

    pick = candidates[0]
    if not selected_list:
        out = [pick]
        diag["action"] = "seed_only"
        diag["applied"] = True
    else:
        # Replace only when the opposite regime has surplus (≥2); otherwise append
        # so we never drop the sole example of a covered regime.
        opposite = "feedback" if need == "no_feedback" else "no_feedback"
        opp_indices = [
            i
            for i, t in enumerate(selected_list)
            if classify_cpc18_feedback_regime(t) == opposite
        ]
        out = list(selected_list)
        if len(opp_indices) >= 2:
            replace_idx = opp_indices[-1]
            out[replace_idx] = pick
            diag["action"] = f"replace_idx_{replace_idx}"
        else:
            out.append(pick)
            diag["action"] = "append"
        diag["applied"] = True

    # Re-sort lightly by block_index then history length to preserve chronology.
    def _chrono_key(t: Dict[str, Any]) -> Tuple[Any, ...]:
        p = t.get("problem") or {}
        block = p.get("block_index")
        hist = t.get("history") or []
        return (
            block if block is not None else 10**9,
            len(hist) if isinstance(hist, list) else 0,
        )

    out_sorted = sorted(out, key=_chrono_key)
    sel_no2, sel_fb2 = _count(out_sorted)
    diag["selected_no_feedback"] = sel_no2
    diag["selected_feedback"] = sel_fb2
    return out_sorted, diag
