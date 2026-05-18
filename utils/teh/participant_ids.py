"""
Collect and auto-prepare valid_participant_ids.json for TEH datasets.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials
from data_modules.psych101_binary import (
    PSYCH101_BINARY_DATASETS,
    experiment_id_for_alias,
    get_psych101_binary_experiments,
)
from utils.teh.teh_datasets import (
    MIXED_GAMBLES,
    is_mixed_gambles_dataset,
    is_psych101_dataset,
    valid_participant_ids_path_with_filter,
)


def _load_collect_participant_ids_module():
    path = Path(__file__).resolve().parent.parent / "tools" / "collect_participant_ids.py"
    spec = importlib.util.spec_from_file_location("collect_participant_ids", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_psych101_valid_ids(
    dataset_alias: str,
    *,
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str] = None,
) -> List[int]:
    """0-based row indices into Psych-101-test with non-empty train+test after TEH split."""
    if not PSYCH101_BINARY_DATASETS[dataset_alias].get("implemented"):
        raise ValueError(f"Parser not implemented for {dataset_alias!r}")

    from teh import split_trials

    valid: List[int] = []
    n = 0
    while True:
        try:
            experiments = get_psych101_binary_experiments(
                dataset_alias,
                n_participants=n + 1,
                split="test",
                local_dataset=local_dataset,
            )
        except Exception:
            break
        if len(experiments) <= n:
            break
        try:
            train_trials, val_trials, test_trials, _ = split_trials(
                experiments[n], split_ratio=split_ratio, split_seed=split_seed
            )
            if train_trials and test_trials:
                valid.append(n)
        except Exception:
            pass
        n += 1
        if n > 5000:
            break
    return valid


def collect_mixed_gambles_valid_ids(
    csv_path: Path,
    *,
    filter_gain_loss_only: bool = False,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> List[int]:
    collect_mod = _load_collect_participant_ids_module()
    return collect_mod.collect_mixed_gambles(
        csv_path,
        filter_gain_loss_only,
        load_mixed_gambles_trials,
    )


def build_valid_participant_ids_payload(
    dataset: str,
    valid_ids: List[int],
    *,
    repo_root: Path,
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: Optional[Path] = None,
    filter_mixed_gambles: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset": dataset,
        "valid_participant_ids": valid_ids,
        "count": len(valid_ids),
        "min_id": min(valid_ids) if valid_ids else None,
        "max_id": max(valid_ids) if valid_ids else None,
    }
    if is_mixed_gambles_dataset(dataset):
        csv_path = mixed_gambles_csv or (repo_root / DEFAULT_CSV_PATH)
        try:
            csv_rel = str(csv_path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            csv_rel = str(csv_path.resolve())
        payload["data_source"] = {
            "kind": "local",
            "csv_path": csv_rel,
            "filter_gain_loss_only": bool(filter_mixed_gambles),
            "split_ratio": float(split_ratio),
            "split_seed": int(split_seed),
        }
        payload["id_semantics"] = "CSV column 'subject' (integer participant id)."
    else:
        exp_id = experiment_id_for_alias(dataset)
        payload["data_source"] = {
            "kind": "psych101",
            "experiment_id": exp_id,
            "split_ratio": float(split_ratio),
            "split_seed": int(split_seed),
        }
        if local_dataset:
            payload["data_source"]["local_dataset"] = local_dataset
        payload["id_semantics"] = (
            "0-based row index into Psych-101-test filtered by experiment id "
            f"({exp_id})."
        )
    return payload


def write_valid_participant_ids_json(
    path: Path,
    payload: Dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def collect_and_write_valid_participant_ids(
    dataset: str,
    repo_root: Path,
    *,
    split_ratio: float = 0.8,
    split_seed: int = 42,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
    output_path: Optional[Path] = None,
) -> Path:
    """Scan data, write valid_participant_ids JSON; return output path."""
    out_path = output_path or valid_participant_ids_path_with_filter(
        dataset, repo_root, filter_mixed_gambles=filter_mixed_gambles
    )

    if is_mixed_gambles_dataset(dataset):
        csv_path = (
            Path(mixed_gambles_csv)
            if Path(mixed_gambles_csv).is_absolute()
            else (repo_root / mixed_gambles_csv)
        ).resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Mixed gambles CSV not found: {csv_path}")
        valid_ids = collect_mixed_gambles_valid_ids(
            csv_path, filter_gain_loss_only=filter_mixed_gambles
        )
    elif is_psych101_dataset(dataset):
        valid_ids = collect_psych101_valid_ids(
            dataset,
            split_ratio=split_ratio,
            split_seed=split_seed,
            local_dataset=local_dataset,
        )
    else:
        raise ValueError(f"Cannot collect valid participant ids for dataset {dataset!r}")

    if not valid_ids:
        raise ValueError(
            f"No valid participants found for {dataset!r} "
            f"(split_ratio={split_ratio}, split_seed={split_seed}). "
            "Check data availability or split settings."
        )

    payload = build_valid_participant_ids_payload(
        dataset,
        valid_ids,
        repo_root=repo_root,
        split_ratio=split_ratio,
        split_seed=split_seed,
        local_dataset=local_dataset,
        mixed_gambles_csv=Path(mixed_gambles_csv),
        filter_mixed_gambles=filter_mixed_gambles,
    )
    write_valid_participant_ids_json(out_path, payload)
    return out_path


def ensure_valid_participant_ids_prepared(
    dataset: str,
    repo_root: Path,
    *,
    split_ratio: float = 0.8,
    split_seed: int = 42,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
    force_regenerate: bool = False,
) -> Path:
    """
    Ensure valid_participant_ids JSON exists for this dataset (create if missing).

    Uses the same split settings as the TEH run so participant scope matches evolution splits.
    """
    path = valid_participant_ids_path_with_filter(
        dataset, repo_root, filter_mixed_gambles=filter_mixed_gambles
    )
    if path.is_file() and not force_regenerate:
        return path

    if path.is_file() and force_regenerate:
        print(f"[TEH] Regenerating valid participant list: {path}")
    else:
        print(
            f"[TEH] No valid participant list at {path}; "
            f"scanning {dataset!r} (split_ratio={split_ratio}, split_seed={split_seed})..."
        )

    out = collect_and_write_valid_participant_ids(
        dataset,
        repo_root,
        split_ratio=split_ratio,
        split_seed=split_seed,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
        output_path=path,
    )
    n = len(json.loads(out.read_text(encoding="utf-8"))["valid_participant_ids"])
    print(f"[TEH] Wrote {n} valid participant ids -> {out}")
    return out


def load_valid_participant_ids(
    dataset: str,
    repo_root: Path,
    *,
    filter_mixed_gambles: bool = False,
    split_ratio: float = 0.8,
    split_seed: int = 42,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    auto_prepare: bool = True,
) -> List[int]:
    """Load valid ids, auto-generating the JSON file first when missing (if auto_prepare)."""
    if auto_prepare:
        ensure_valid_participant_ids_prepared(
            dataset,
            repo_root,
            split_ratio=split_ratio,
            split_seed=split_seed,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            filter_mixed_gambles=filter_mixed_gambles,
        )
    path = valid_participant_ids_path_with_filter(
        dataset, repo_root, filter_mixed_gambles=filter_mixed_gambles
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing valid participant list: {path}. "
            f"Run: python utils/tools/collect_teh_participant_ids.py --dataset {dataset!r}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["valid_participant_ids"])
