"""TEH utilities: dataset registry and run setup."""

from utils.teh.teh_datasets import (
    IMPLEMENTED_PSYCH101_ALIASES,
    LOGlik_VAL_SPLIT_DATASETS,
    MIXED_GAMBLES,
    PARTICIPANT_DATASETS,
    dataset_display_name,
    is_binary_loglik_dataset,
    is_mixed_gambles_dataset,
    teh_output_base_dir,
    uses_train_val_test_loglik_split,
    valid_participant_ids_path,
    valid_participant_ids_path_with_filter,
)
from utils.teh.participant_ids import (
    collect_and_write_valid_participant_ids,
    ensure_valid_participant_ids_prepared,
    load_valid_participant_ids,
)
from utils.teh.teh_runtime import (
    DEFAULT_SEED_PROGRAM,
    TEH_WANDB_PROJECT,
    setup_teh_run_prompts,
    teh_wandb_run_name,
)

__all__ = [
    "collect_and_write_valid_participant_ids",
    "DEFAULT_SEED_PROGRAM",
    "ensure_valid_participant_ids_prepared",
    "IMPLEMENTED_PSYCH101_ALIASES",
    "LOGlik_VAL_SPLIT_DATASETS",
    "MIXED_GAMBLES",
    "PARTICIPANT_DATASETS",
    "TEH_WANDB_PROJECT",
    "dataset_display_name",
    "is_binary_loglik_dataset",
    "is_mixed_gambles_dataset",
    "load_valid_participant_ids",
    "setup_teh_run_prompts",
    "teh_output_base_dir",
    "teh_wandb_run_name",
    "uses_train_val_test_loglik_split",
    "valid_participant_ids_path",
    "valid_participant_ids_path_with_filter",
]
