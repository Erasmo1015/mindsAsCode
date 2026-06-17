"""Psych-101 experiment discovery and row sampling for teh_psych prototype."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PSYCH101_BINARY_DATASETS,
    _load_hf_split,
    normalize_psych_dataset_split,
)


def safe_experiment_id_for_path(experiment_id: str) -> str:
    """Filesystem-safe slug from raw HF experiment id."""
    slug = re.sub(r"[^\w.\-]+", "_", str(experiment_id).strip())
    return slug.strip("_") or "unknown_experiment"


def discover_psych101_train_experiments(
    split_ds,
    *,
    experiment_ids: Optional[Sequence[str]] = None,
    max_experiments: Optional[int] = None,
) -> List[str]:
    """
    Unique experiment ids in the dataset's native row order (first occurrence).
    """
    allow = set(experiment_ids) if experiment_ids else None
    seen: set = set()
    ordered: List[str] = []
    for row in split_ds:
        eid = str(row["experiment"])
        if allow is not None and eid not in allow:
            continue
        if eid in seen:
            continue
        seen.add(eid)
        ordered.append(eid)
        if max_experiments is not None and len(ordered) >= max_experiments:
            break
    return ordered


def load_psych101_split(
    split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    *,
    local_dataset: Optional[str] = None,
):
    split = normalize_psych_dataset_split(split)
    return _load_hf_split(split, local_dataset=local_dataset)


def filter_rows_for_experiment(split_ds, experiment_id: str):
    """HF rows for one raw experiment id."""
    if str(experiment_id).endswith("/"):
        prefix = str(experiment_id)
        return split_ds.filter(lambda ex, p=prefix: str(ex["experiment"]).startswith(p))
    return split_ds.filter(lambda ex, e=experiment_id: ex["experiment"] == e)


def alias_for_experiment_id(experiment_id: str) -> Optional[str]:
    """Map raw HF experiment id to implemented PSYCH101_BINARY_DATASETS alias, if any."""
    eid = str(experiment_id)
    for alias, spec in PSYCH101_BINARY_DATASETS.items():
        spec_id = str(spec["experiment_id"])
        if not spec.get("implemented"):
            continue
        if spec_id.endswith("/"):
            if eid.startswith(spec_id):
                return alias
        elif eid == spec_id:
            return alias
    return None


def spec_for_experiment_id(experiment_id: str) -> Optional[Dict[str, Any]]:
    alias = alias_for_experiment_id(experiment_id)
    if alias is None:
        return None
    spec = dict(PSYCH101_BINARY_DATASETS[alias])
    spec["alias"] = alias
    return spec


def sample_row_indices(
    n_rows: int,
    *,
    max_participants: int,
    range_start_ordinal: Optional[int] = None,
    range_end_ordinal: Optional[int] = None,
) -> Tuple[List[int], str]:
    """
    Choose 0-based row indices into filtered experiment rows.

    Returns (indices, notes).
    """
    if n_rows <= 0:
        return [], "no rows available"
    if range_start_ordinal is not None or range_end_ordinal is not None:
        if range_start_ordinal is None or range_end_ordinal is None:
            raise ValueError(
                "range_start_ordinal and range_end_ordinal must both be set when using a range"
            )
        start = max(0, int(range_start_ordinal))
        end = min(n_rows - 1, int(range_end_ordinal))
        if start > end:
            return [], f"empty ordinal range [{start}, {end}] for n_rows={n_rows}"
        indices = list(range(start, end + 1))
        return indices, f"ordinal range [{start}, {end}]"

    n_use = min(max_participants, n_rows)
    return list(range(n_use)), f"first {n_use} of {n_rows} rows"


def sample_raw_rows(
    filtered_ds,
    indices: List[int],
    *,
    max_preview: int = 5,
) -> List[Dict[str, Any]]:
    """Materialize selected HF rows as plain dicts."""
    out: List[Dict[str, Any]] = []
    for i in indices:
        if 0 <= i < len(filtered_ds):
            out.append(dict(filtered_ds[i]))
    return out[:max_preview] if max_preview else out
