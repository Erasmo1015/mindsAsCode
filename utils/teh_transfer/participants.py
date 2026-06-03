"""Participant selection for TEH transfer (with per-dataset range clamping)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import teh


def resolve_participants_for_transfer(
    *,
    dataset: str,
    repo_root: Path,
    participant_scope: str,
    single_participant_id: int,
    range_start_ordinal: Optional[int],
    range_end_ordinal: Optional[int],
    all_max_participants: Optional[int],
    participant_ordinals: Optional[List[int]],
    filter_mixed_gambles: bool,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    config_key: Optional[str] = None,
) -> List[int]:
    """
    Resolve participant ids for one transfer dataset.

    When ``participant_scope=range`` and ``range_end_ordinal`` exceeds the valid
    ordinal range for this dataset, clamp to the last valid ordinal and log a warning
    instead of raising.
    """
    label = config_key or dataset

    if participant_scope != "range":
        return teh.resolve_participants_for_scope(
            dataset=dataset,
            repo_root=repo_root,
            participant_scope=participant_scope,
            single_participant_id=single_participant_id,
            range_start_ordinal=range_start_ordinal,
            range_end_ordinal=range_end_ordinal,
            all_max_participants=all_max_participants,
            participant_ordinals=participant_ordinals,
            filter_mixed_gambles=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
        )

    if range_start_ordinal is None or range_end_ordinal is None:
        return teh.resolve_participants_for_scope(
            dataset=dataset,
            repo_root=repo_root,
            participant_scope=participant_scope,
            single_participant_id=single_participant_id,
            range_start_ordinal=range_start_ordinal,
            range_end_ordinal=range_end_ordinal,
            all_max_participants=all_max_participants,
            participant_ordinals=participant_ordinals,
            filter_mixed_gambles=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
        )

    valid = teh.load_valid_participant_ids_from_json(
        dataset,
        repo_root,
        filter_mixed_gambles,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
    n_valid = len(valid)
    start = int(range_start_ordinal)
    end = int(range_end_ordinal)

    if start < 0 or start > end:
        return teh.resolve_participants_for_scope(
            dataset=dataset,
            repo_root=repo_root,
            participant_scope=participant_scope,
            single_participant_id=single_participant_id,
            range_start_ordinal=range_start_ordinal,
            range_end_ordinal=range_end_ordinal,
            all_max_participants=all_max_participants,
            participant_ordinals=participant_ordinals,
            filter_mixed_gambles=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
        )

    if n_valid == 0:
        print(
            f"Warning: [teh_transfer] {label}: no valid participants for "
            f"{dataset} (split={psych_dataset_split}); using empty participant pool."
        )
        return []

    clamped_end = end
    if start >= n_valid:
        print(
            f"Warning: [teh_transfer] {label}: range_start_ordinal={start} is outside "
            f"valid ordinals [0, {n_valid - 1}] (count={n_valid}); using 0 participants."
        )
        return []

    if end >= n_valid:
        clamped_end = n_valid - 1
        print(
            f"Warning: [teh_transfer] {label}: clamped participant ordinal range "
            f"[{start}, {end}] -> [{start}, {clamped_end}] "
            f"(valid count={n_valid}, split={psych_dataset_split})."
        )

    return teh.resolve_participants_for_scope(
        dataset=dataset,
        repo_root=repo_root,
        participant_scope=participant_scope,
        single_participant_id=single_participant_id,
        range_start_ordinal=start,
        range_end_ordinal=clamped_end,
        all_max_participants=all_max_participants,
        participant_ordinals=participant_ordinals,
        filter_mixed_gambles=filter_mixed_gambles,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
