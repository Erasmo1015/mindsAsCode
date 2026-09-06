"""External (non-Psych-101) TEH dataset adapters."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from data_modules.external import bergert_nosofsky_2007 as bergert_mod
from data_modules.external import guan_2020_stopping as guan_mod
from data_modules.external import steyvers_2009_bandit as steyvers_mod

BERGERT_NOSOFSKY_2007 = bergert_mod.DATASET_ALIAS
GUAN_2020_STOPPING = guan_mod.DATASET_ALIAS
STEYVERS_2009_BANDIT = steyvers_mod.DATASET_ALIAS

EXTERNAL_DATASET_META: Dict[str, Dict[str, Any]] = {
    BERGERT_NOSOFSKY_2007: {
        "display_name": bergert_mod.DISPLAY_NAME,
        "task_description": bergert_mod.TASK_DESCRIPTION,
        "output_type": bergert_mod.OUTPUT_TYPE,
        "n_actions": bergert_mod.N_ACTIONS,
        "split_unit": bergert_mod.SPLIT_UNIT,
        "source": "michael_behavioralDataRepository",
        "default_data_dir": bergert_mod.DEFAULT_DATA_DIR,
        "reference_prompt": "prompts/external/bergert_nosofsky_2007.txt",
    },
    GUAN_2020_STOPPING: {
        "display_name": guan_mod.DISPLAY_NAME,
        "task_description": guan_mod.TASK_DESCRIPTION,
        "output_type": guan_mod.OUTPUT_TYPE,
        "n_actions": guan_mod.N_ACTIONS,
        "split_unit": guan_mod.SPLIT_UNIT,
        "source": "author_riskproject_mat",
        "default_data_dir": guan_mod.DEFAULT_DATA_DIR,
        "reference_prompt": "prompts/external/guan_2020_stopping.txt",
    },
    STEYVERS_2009_BANDIT: {
        "display_name": steyvers_mod.DISPLAY_NAME,
        "task_description": steyvers_mod.TASK_DESCRIPTION,
        "output_type": steyvers_mod.OUTPUT_TYPE,
        "n_actions": steyvers_mod.N_ACTIONS,
        "split_unit": steyvers_mod.SPLIT_UNIT,
        "source": "michael_behavioralDataRepository",
        "default_data_dir": steyvers_mod.DEFAULT_DATA_DIR,
        "reference_prompt": "prompts/external/steyvers_2009_bandit.txt",
        "seed_program": "persona_code_example/teh/categorical_uniform.py",
    },
}

EXTERNAL_DATASETS: FrozenSet[str] = frozenset(EXTERNAL_DATASET_META.keys())


def is_external_dataset(dataset: str) -> bool:
    return dataset in EXTERNAL_DATASET_META


def is_bergert_nosofsky_2007_dataset(dataset: str) -> bool:
    return dataset == BERGERT_NOSOFSKY_2007


def is_guan_2020_stopping_dataset(dataset: str) -> bool:
    return dataset == GUAN_2020_STOPPING


def is_steyvers_2009_bandit_dataset(dataset: str) -> bool:
    return dataset == STEYVERS_2009_BANDIT


def external_dataset_display_name(dataset: str) -> str:
    return str(EXTERNAL_DATASET_META[dataset]["display_name"])


def external_dataset_task_description(dataset: str) -> str:
    return str(EXTERNAL_DATASET_META[dataset]["task_description"])


def external_default_data_dir(dataset: str) -> str:
    return str(EXTERNAL_DATASET_META[dataset]["default_data_dir"])


def load_external_loglik_trials(
    dataset: str,
    participant_id: int,
    *,
    data_dir: Optional[str] = None,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    resolved = data_dir or external_default_data_dir(dataset)
    if is_bergert_nosofsky_2007_dataset(dataset):
        return bergert_mod.load_bergert_nosofsky_2007_trials(
            participant_id,
            data_dir=resolved,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
    if is_guan_2020_stopping_dataset(dataset):
        return guan_mod.load_guan_2020_stopping_trials(
            participant_id,
            data_dir=resolved,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
    if is_steyvers_2009_bandit_dataset(dataset):
        return steyvers_mod.load_steyvers_2009_bandit_trials(
            participant_id,
            data_dir=resolved,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
    raise ValueError(f"Unsupported external TEH dataset: {dataset!r}")
