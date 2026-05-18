"""
TEH dataset registry: Psych-101 binary aliases + local mixed_gambles.
"""
from __future__ import annotations

from pathlib import Path
from typing import FrozenSet

from data_modules.psych101_binary import (
    PSYCH101_BINARY_DATASETS,
    DEFAULT_PSYCH_DATASET_SPLIT,
    normalize_psych_dataset_split,
    is_psych101_dataset,
    experiment_id_for_alias,
)
from data_modules import mixed_gambles as mixed_gambles_module

MIXED_GAMBLES = mixed_gambles_module.DATASET_NAME

VALID_PARTICIPANT_IDS_JSON = "valid_participant_ids.json"

IMPLEMENTED_PSYCH101_ALIASES: FrozenSet[str] = frozenset(
    k for k, v in PSYCH101_BINARY_DATASETS.items() if v.get("implemented")
)

PARTICIPANT_DATASETS: FrozenSet[str] = IMPLEMENTED_PSYCH101_ALIASES | {MIXED_GAMBLES}

LOGlik_VAL_SPLIT_DATASETS: FrozenSet[str] = PARTICIPANT_DATASETS


def is_mixed_gambles_dataset(dataset: str) -> bool:
    return dataset == MIXED_GAMBLES


def is_binary_loglik_dataset(dataset: str) -> bool:
    """True for TEH Psych-101 binary aliases (implemented) and mixed_gambles."""
    if is_mixed_gambles_dataset(dataset):
        return True
    return dataset in IMPLEMENTED_PSYCH101_ALIASES


def uses_train_val_test_loglik_split(
    dataset: str, fitness_metric: str, *, cpc18_official_mse: bool = False
) -> bool:
    if fitness_metric != "loglik":
        return False
    if cpc18_official_mse:
        return False
    return is_binary_loglik_dataset(dataset)


def psych101_metadata_root(repo_root: Path, psych_dataset_split: str) -> Path:
    """Root for TEH metadata JSON per HF corpus: datasets/psych101_train or psych101_test."""
    split = normalize_psych_dataset_split(psych_dataset_split)
    return repo_root / "datasets" / f"psych101_{split}"


def valid_participant_ids_path(
    dataset: str,
    repo_root: Path,
    *,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
) -> Path:
    return valid_participant_ids_path_with_filter(
        dataset, repo_root, psych_dataset_split=psych_dataset_split
    )


def valid_participant_ids_path_with_filter(
    dataset: str,
    repo_root: Path,
    *,
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
) -> Path:
    if is_mixed_gambles_dataset(dataset):
        name = (
            "valid_participant_ids_gain_loss.json"
            if filter_mixed_gambles
            else VALID_PARTICIPANT_IDS_JSON
        )
        return repo_root / "datasets" / "mixed_gambles" / name
    return psych101_metadata_root(repo_root, psych_dataset_split) / dataset / VALID_PARTICIPANT_IDS_JSON


def teh_output_base_dir(
    dataset: str,
    timestamp: str,
    *,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
) -> str:
    if is_mixed_gambles_dataset(dataset):
        return f"generated_outputs/mixed_gambles/teh/run_{timestamp}"
    split = normalize_psych_dataset_split(psych_dataset_split)
    return f"generated_outputs/psych101_{split}/teh/{dataset}/run_{timestamp}"


def dataset_display_name(dataset: str) -> str:
    if is_mixed_gambles_dataset(dataset):
        return "Mixed gambles (local CSV)"
    return PSYCH101_BINARY_DATASETS[dataset]["display_name"]
