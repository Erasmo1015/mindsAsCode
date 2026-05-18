"""TEH utilities: dataset registry and run setup."""

from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PSYCH_DATASET_SPLITS,
    hf_id_for_psych_dataset_split,
    normalize_psych_dataset_split,
)
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
    psych101_metadata_root,
    valid_participant_ids_path,
    valid_participant_ids_path_with_filter,
    VALID_PARTICIPANT_IDS_JSON,
)
from utils.teh.participant_ids import (
    Psych101ParticipantStats,
    collect_and_write_valid_participant_ids,
    collect_psych101_valid_participants,
    ensure_valid_participant_ids_prepared,
    load_valid_participant_details,
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
    "DEFAULT_PSYCH_DATASET_SPLIT",
    "DEFAULT_SEED_PROGRAM",
    "PSYCH_DATASET_SPLITS",
    "ensure_valid_participant_ids_prepared",
    "hf_id_for_psych_dataset_split",
    "IMPLEMENTED_PSYCH101_ALIASES",
    "LOGlik_VAL_SPLIT_DATASETS",
    "MIXED_GAMBLES",
    "normalize_psych_dataset_split",
    "PARTICIPANT_DATASETS",
    "Psych101ParticipantStats",
    "TEH_WANDB_PROJECT",
    "VALID_PARTICIPANT_IDS_JSON",
    "collect_psych101_valid_participants",
    "dataset_display_name",
    "is_binary_loglik_dataset",
    "is_mixed_gambles_dataset",
    "load_valid_participant_details",
    "load_valid_participant_ids",
    "psych101_metadata_root",
    "setup_teh_run_prompts",
    "teh_output_base_dir",
    "teh_wandb_run_name",
    "uses_train_val_test_loglik_split",
    "valid_participant_ids_path",
    "valid_participant_ids_path_with_filter",
]
