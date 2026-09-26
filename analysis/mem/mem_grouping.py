"""Global MEM random-effects grouping: dataset::run_id::participant_id.

Raw ``participant_id`` ordinals restart per dataset and MUST NOT be used as
MixedLM groups. Both focal and joint fitters call ``attach_validated_mem_group``
before fitting.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

GROUPING_KEY_NAME = "dataset::run_id::participant_id"
MEM_GROUP_COL = "_mem_group"


class MemGroupingError(ValueError):
    """Raised when RE grouping cannot be safely constructed."""


def make_global_participant_key(df: pd.DataFrame) -> pd.Series:
    """Build ``dataset::run_id::participant_id`` strings (no validation)."""
    for col in ("dataset", "run_id", "participant_id"):
        if col not in df.columns:
            raise MemGroupingError(
                f"MEM RE grouping requires column {col!r}; "
                f"refusing to fall back to raw participant_id"
            )
    ds = df["dataset"].astype(str)
    run = df["run_id"].astype(str)
    pid = df["participant_id"].astype(str)
    if ds.isna().any() or (ds == "nan").any() or (ds == "").any():
        raise MemGroupingError("dataset has missing/empty values")
    if run.isna().any() or (run == "nan").any() or (run == "").any():
        raise MemGroupingError("run_id has missing/empty values")
    if pid.isna().any() or (pid == "nan").any() or (pid == "").any():
        raise MemGroupingError("participant_id has missing/empty values")
    return ds + "::" + run + "::" + pid


def raw_participant_id_collision_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect whether raw participant_id alone would collide across datasets."""
    if "dataset" not in df.columns or "participant_id" not in df.columns:
        return {
            "raw_pid_would_collide": True,
            "reason": "missing dataset or participant_id",
            "n_raw_pids_spanning_gt1_dataset": None,
        }
    g = df.groupby(df["participant_id"].astype(str))["dataset"].nunique()
    n_multi = int((g > 1).sum())
    return {
        "raw_pid_would_collide": n_multi > 0,
        "n_raw_pids_spanning_gt1_dataset": n_multi,
        "frac_rows_affected": float(
            (df["participant_id"].astype(str).map(g) > 1).mean()
        )
        if len(df)
        else 0.0,
    }


def validate_mem_group_uniqueness(
    df: pd.DataFrame, group_col: str = MEM_GROUP_COL
) -> Dict[str, Any]:
    """Every RE group must map to exactly one (dataset, run_id, participant_id)."""
    if group_col not in df.columns:
        raise MemGroupingError(f"missing {group_col}")
    trip = df[["dataset", "run_id", "participant_id"]].astype(str)
    # group -> unique triples
    n_trip_per_group = (
        trip.assign(_g=df[group_col].astype(str))
        .drop_duplicates()
        .groupby("_g")
        .size()
    )
    bad_groups = n_trip_per_group[n_trip_per_group != 1]
    if len(bad_groups):
        raise MemGroupingError(
            f"{len(bad_groups)} RE group(s) map to >1 dataset/run/participant; "
            f"examples: {list(bad_groups.head(5).index)}"
        )
    # triple -> unique group
    keyed = trip.assign(_g=df[group_col].astype(str)).drop_duplicates()
    n_g_per_trip = keyed.groupby(["dataset", "run_id", "participant_id"])["_g"].nunique()
    bad_trips = n_g_per_trip[n_g_per_trip != 1]
    if len(bad_trips):
        raise MemGroupingError(
            f"{len(bad_trips)} dataset/run/participant triple(s) map to >1 RE group"
        )
    return {
        "n_groups": int(df[group_col].nunique()),
        "n_rows": int(len(df)),
        "ok": True,
    }


def attach_validated_mem_group(
    df: pd.DataFrame,
    *,
    fail_if_raw_pid_would_collide_without_global_key: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Attach ``_mem_group`` and validate uniqueness / collision policy.

    Always uses ``dataset::run_id::participant_id``. Raises ``MemGroupingError`` if
    required columns are missing (would force unsafe raw-pid grouping) or if any
    group is not a unique (dataset, run, participant) map.
    """
    work = df.copy()
    collision = raw_participant_id_collision_report(work)
    # Missing dataset/run_id is itself a failure (cannot build safe key).
    key = make_global_participant_key(work)
    work[MEM_GROUP_COL] = key
    uniq = validate_mem_group_uniqueness(work)
    # If raw pids collide across datasets, global key is mandatory — already used.
    # Fail loudly if someone strips dataset (caught above) or if key equals raw pid.
    if fail_if_raw_pid_would_collide_without_global_key and collision["raw_pid_would_collide"]:
        # Ensure key is not identical to raw participant_id (would ignore dataset).
        if (work[MEM_GROUP_COL].astype(str) == work["participant_id"].astype(str)).all():
            raise MemGroupingError(
                "raw participant_id collides across datasets but _mem_group "
                "equals raw participant_id; refusing unsafe RE grouping"
            )
    meta = {
        "grouping_key": GROUPING_KEY_NAME,
        "group_column": MEM_GROUP_COL,
        "collision_report": collision,
        "uniqueness": uniq,
    }
    return work, meta
