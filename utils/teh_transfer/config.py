"""Load transfer dataset lists from YAML config."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import yaml

from data_modules.psych101_binary import normalize_psych101_dataset_alias, normalize_psych_dataset_split
from utils.teh.teh_datasets import is_mixed_gambles_dataset


@dataclass(frozen=True)
class TransferDatasetSpec:
    """One dataset entry from configs/teh_transfer.yaml."""

    config_key: str
    dataset_alias: str
    psych_dataset_split: str


def parse_config_dataset_key(config_key: str) -> tuple[str, str]:
    """
    Map YAML key to (dataset_alias, psych_dataset_split).

    Keys ending in ``_test`` use Psych-101-test; others default to train.
    """
    key = str(config_key).strip()
    if is_mixed_gambles_dataset(key):
        return key, "train"
    if key.endswith("_test"):
        base = key[: -len("_test")]
        return normalize_psych101_dataset_alias(base), "test"
    return normalize_psych101_dataset_alias(key), "train"


def load_transfer_datasets(config_path: Path | str) -> List[TransferDatasetSpec]:
    """Load dataset list from transfer YAML (top-level ``datasets`` mapping keys)."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Transfer config not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Transfer config must be a mapping, got {type(raw).__name__}")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Transfer config must contain a top-level 'datasets' mapping.")
    specs: List[TransferDatasetSpec] = []
    for config_key in datasets:
        alias, split = parse_config_dataset_key(str(config_key))
        specs.append(
            TransferDatasetSpec(
                config_key=str(config_key).strip(),
                dataset_alias=alias,
                psych_dataset_split=normalize_psych_dataset_split(split),
            )
        )
    if not specs:
        raise ValueError(f"No datasets listed in transfer config: {path}")
    return specs


def filter_transfer_specs(
    specs: Sequence[TransferDatasetSpec],
    *,
    only: Sequence[str] | None = None,
) -> List[TransferDatasetSpec]:
    """Restrict to config keys or dataset aliases when ``only`` is set."""
    if not only:
        return list(specs)
    wanted = {str(x).strip() for x in only}
    out: List[TransferDatasetSpec] = []
    for spec in specs:
        if spec.config_key in wanted or spec.dataset_alias in wanted:
            out.append(spec)
    if not out:
        raise ValueError(f"No transfer datasets matched filter: {sorted(wanted)}")
    return out
