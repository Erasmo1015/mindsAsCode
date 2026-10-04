"""Primary-analysis guards for schema-v5 construct-effect fits.

The five constructs are history, value, probability_used, feedback, and
learning. ``risk_*`` is not a default and is not ``probability_used``.

Primary focal multiplicity for a uniform-v8 run is Holm on the predeclared
focal family. Benjamini–Hochberg remains available as a diagnostic and is not
the paper correction.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FIVE_CONSTRUCTS: Tuple[str, ...] = (
    "history",
    "value",
    "probability_used",
    "feedback",
    "learning",
)
PRIMARY_DIRECTIONS: Tuple[str, ...] = ("added", "removed", "modified")
PRIMARY_CORRECTION = "holm"
DIAGNOSTIC_CORRECTION = "bh"
GROUPING_KEY = "dataset::run_id::participant_id"

# Support gates applied before a primary joint fit. They match the checks
# named for schema-v5 joint blocks: events, within-participant contrast,
# rank, separation, and more than one dataset.
JOINT_MIN_ROWS = 5
JOINT_MIN_GROUPS_BOTH_LEVELS = 2
JOINT_MIN_DATASETS = 3
JOINT_RATE_LOW = 0.01
JOINT_RATE_HIGH = 0.99


class MemEffectPolicyError(ValueError):
    """A primary construct-effect specification is not safe to fit."""


def directional_effect(construct: str, direction: str) -> str:
    if construct not in FIVE_CONSTRUCTS:
        raise MemEffectPolicyError(
            f"construct {construct!r} is outside the five-construct vocabulary"
        )
    if direction not in PRIMARY_DIRECTIONS:
        raise MemEffectPolicyError(f"direction {direction!r} is not added/removed/modified")
    return f"{construct}_{direction}"


def parse_effect_list(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def require_explicit_effects(effects: Sequence[str]) -> List[str]:
    """Primary fits name their effects. There is no risk_* default."""
    cleaned = [str(effect).strip() for effect in effects if str(effect).strip()]
    if not cleaned:
        raise MemEffectPolicyError(
            "explicit --effects is required. The vocabulary is "
            + ", ".join(FIVE_CONSTRUCTS)
            + ". risk_added and risk_modified are not defaults and are not probability_used."
        )
    banned = [effect for effect in cleaned if effect.startswith("risk_")]
    if banned:
        raise MemEffectPolicyError(
            "risk_* effects are refused: "
            + ", ".join(banned)
            + ". Use probability_used for probability use."
        )
    return cleaned


def same_construct_add_modify_pairs(effects: Sequence[str]) -> List[Tuple[str, str]]:
    present = set(effects)
    pairs: List[Tuple[str, str]] = []
    for construct in FIVE_CONSTRUCTS:
        added = f"{construct}_added"
        modified = f"{construct}_modified"
        if added in present and modified in present:
            pairs.append((added, modified))
    return pairs


def reject_same_construct_add_and_modify(
    effects: Sequence[str],
    *,
    allow_same_construct_add_and_modify: bool = False,
) -> None:
    pairs = same_construct_add_modify_pairs(effects)
    if not pairs:
        return
    if allow_same_construct_add_and_modify:
        return
    rendered = ", ".join(f"{a}+{m}" for a, m in pairs)
    raise MemEffectPolicyError(
        "a primary block cannot contain addition and modification of the same "
        f"construct ({rendered}). Pass --allow_same_construct_add_and_modify "
        "only for a documented non-primary diagnostic."
    )


def holm_adjust(pvals: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm step-down adjusted p-values. None and non-finite inputs stay None."""
    out: List[Optional[float]] = [None] * len(pvals)
    indexed: List[Tuple[int, float]] = []
    for i, raw in enumerate(pvals):
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            indexed.append((i, value))
    if not indexed:
        return out
    order = sorted(range(len(indexed)), key=lambda j: indexed[j][1])
    m = len(indexed)
    running = 0.0
    adjusted = [0.0] * m
    for rank, pos in enumerate(order):
        raw = indexed[pos][1]
        candidate = (m - rank) * raw
        running = max(running, candidate)
        adjusted[pos] = min(1.0, running)
    # Enforce monotonicity in the sorted order (Holm is nondecreasing).
    for rank in range(1, m):
        prev = adjusted[order[rank - 1]]
        cur = adjusted[order[rank]]
        if cur < prev:
            adjusted[order[rank]] = prev
    for pos, value in enumerate(adjusted):
        out[indexed[pos][0]] = float(min(1.0, value))
    return out


def joint_support_failures(
    frame: pd.DataFrame,
    effects: Sequence[str],
    *,
    group_col: str,
) -> List[str]:
    """Return reasons a joint block is not supported. Empty means it may be fit."""
    reasons: List[str] = []
    if group_col not in frame.columns:
        return [f"missing grouping column {group_col}"]
    if len(frame) < JOINT_MIN_ROWS:
        reasons.append(f"n_rows {len(frame)} < {JOINT_MIN_ROWS}")
    if "dataset" in frame.columns and int(frame["dataset"].nunique()) < JOINT_MIN_DATASETS:
        reasons.append(
            f"n_datasets {int(frame['dataset'].nunique())} < {JOINT_MIN_DATASETS}"
        )
    columns = []
    for effect in effects:
        if effect == "iteration":
            continue
        if effect not in frame.columns:
            reasons.append(f"missing effect column {effect}")
            continue
        values = pd.to_numeric(frame[effect], errors="coerce")
        if values.isna().any():
            reasons.append(f"{effect} has non-numeric values")
            continue
        columns.append(effect)
        n_pos = int((values == 1).sum())
        n_neg = int((values == 0).sum())
        rate = float(values.mean()) if len(values) else 0.0
        both = int(
            frame.assign(_v=values)
            .groupby(group_col)["_v"]
            .nunique()
            .ge(2)
            .sum()
        )
        if n_pos < JOINT_MIN_ROWS:
            reasons.append(f"{effect} positive rows {n_pos} < {JOINT_MIN_ROWS}")
        if both < JOINT_MIN_GROUPS_BOTH_LEVELS:
            reasons.append(
                f"{effect} groups with both levels {both} < {JOINT_MIN_GROUPS_BOTH_LEVELS}"
            )
        if rate < JOINT_RATE_LOW or rate > JOINT_RATE_HIGH:
            reasons.append(f"{effect} event rate {rate:.4f} is outside (0.01, 0.99)")
        if n_neg < 1:
            reasons.append(f"{effect} has no zero rows")
    if columns:
        design = np.column_stack(
            [np.ones(len(frame))]
            + [pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float) for col in columns]
        )
        rank = int(np.linalg.matrix_rank(design))
        if rank < design.shape[1]:
            reasons.append(f"design rank {rank} < {design.shape[1]}")
    return reasons
