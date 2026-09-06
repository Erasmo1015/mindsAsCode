"""
TEH dataset registry: Psych-101 binary aliases + local mixed_gambles + external adapters.
"""
from __future__ import annotations

from pathlib import Path
from typing import FrozenSet, Optional

from data_modules.psych101_binary import (
    PSYCH101_BINARY_DATASETS,
    DEFAULT_PSYCH_DATASET_SPLIT,
    normalize_psych_dataset_split,
    normalize_psych101_dataset_alias,
    is_psych101_dataset,
    experiment_id_for_alias,
)
from data_modules import mixed_gambles as mixed_gambles_module
from data_modules.external import (
    BERGERT_NOSOFSKY_2007,
    GUAN_2020_STOPPING,
    STEYVERS_2009_BANDIT,
    EXTERNAL_DATASETS,
    EXTERNAL_DATASET_META,
    external_dataset_display_name,
    external_default_data_dir,
    is_bergert_nosofsky_2007_dataset,
    is_guan_2020_stopping_dataset,
    is_steyvers_2009_bandit_dataset,
    is_external_dataset,
)

MIXED_GAMBLES = mixed_gambles_module.DATASET_NAME

VALID_PARTICIPANT_IDS_JSON = "valid_participant_ids.json"

IMPLEMENTED_PSYCH101_ALIASES: FrozenSet[str] = frozenset(
    k for k, v in PSYCH101_BINARY_DATASETS.items() if v.get("implemented")
)

PARTICIPANT_DATASETS: FrozenSet[str] = (
    IMPLEMENTED_PSYCH101_ALIASES | {MIXED_GAMBLES} | EXTERNAL_DATASETS
)

LOGlik_VAL_SPLIT_DATASETS: FrozenSet[str] = PARTICIPANT_DATASETS


def is_mixed_gambles_dataset(dataset: str) -> bool:
    return dataset == MIXED_GAMBLES


def dataset_output_type(dataset: str) -> str:
    """Return 'bernoulli' or 'categorical' for TEH loglik evaluation."""
    if is_external_dataset(dataset):
        return str(EXTERNAL_DATASET_META[dataset].get("output_type", "bernoulli"))
    alias = normalize_psych101_dataset_alias(dataset)
    if alias in PSYCH101_BINARY_DATASETS:
        return str(PSYCH101_BINARY_DATASETS[alias].get("output_type", "bernoulli"))
    return "bernoulli"


def is_teh_loglik_dataset(dataset: str) -> bool:
    """True for TEH participant loglik datasets (Bernoulli or categorical)."""
    if is_mixed_gambles_dataset(dataset):
        return True
    if is_external_dataset(dataset):
        return True
    return normalize_psych101_dataset_alias(dataset) in IMPLEMENTED_PSYCH101_ALIASES


def is_binary_loglik_dataset(dataset: str) -> bool:
    """
    TEH participant loglik eligibility (Bernoulli or categorical).

    Name is historical: originally all TEH loglik datasets were Bernoulli.
    Use dataset_output_type(...) / is_categorical_output_dataset(...) for
    output-contract branches. Bernoulli evaluation paths must key off
    dataset_output_type == 'bernoulli', not this predicate alone.
    """
    return is_teh_loglik_dataset(dataset)


def is_categorical_output_dataset(dataset: str) -> bool:
    return dataset_output_type(dataset) == "categorical"


def is_bernoulli_output_dataset(dataset: str) -> bool:
    return is_teh_loglik_dataset(dataset) and dataset_output_type(dataset) == "bernoulli"


def uses_train_val_test_loglik_split(
    dataset: str, fitness_metric: str, *, cpc18_official_mse: bool = False
) -> bool:
    if fitness_metric != "loglik":
        return False
    if cpc18_official_mse:
        return False
    return is_teh_loglik_dataset(dataset)


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
    if is_external_dataset(dataset):
        return (
            repo_root
            / "datasets"
            / "external"
            / dataset
            / VALID_PARTICIPANT_IDS_JSON
        )
    alias = normalize_psych101_dataset_alias(dataset)
    return psych101_metadata_root(repo_root, psych_dataset_split) / alias / VALID_PARTICIPANT_IDS_JSON


def teh_output_base_dir(
    dataset: str,
    timestamp: str,
    *,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    ablation: Optional[str] = None,
) -> str:
    output_root = "generated_outputs_ablation" if ablation else "generated_outputs"
    run_dir = ablation if ablation else f"run_{timestamp}"
    if is_mixed_gambles_dataset(dataset):
        return f"{output_root}/mixed_gambles/teh/{run_dir}"
    if is_external_dataset(dataset):
        return f"{output_root}/external/{dataset}/teh/{run_dir}"
    split = normalize_psych_dataset_split(psych_dataset_split)
    alias = normalize_psych101_dataset_alias(dataset)
    return f"{output_root}/psych101_{split}/teh/{alias}/{run_dir}"


def dataset_display_name(dataset: str) -> str:
    if is_mixed_gambles_dataset(dataset):
        return "Mixed gambles (local CSV)"
    if is_external_dataset(dataset):
        return external_dataset_display_name(dataset)
    alias = normalize_psych101_dataset_alias(dataset)
    return PSYCH101_BINARY_DATASETS[alias]["display_name"]
