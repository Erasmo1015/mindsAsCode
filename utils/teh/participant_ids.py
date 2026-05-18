"""
Collect and auto-prepare valid participant id JSON for TEH datasets.

Psych-101 aliases (peterson2021using, plonsky2018when, …) store metadata only under:
  datasets/psych101_train/<dataset_alias>/valid_participant_ids.json
  datasets/psych101_test/<dataset_alias>/valid_participant_ids.json
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PSYCH101_BINARY_DATASETS,
    experiment_id_for_alias,
    get_psych101_binary_experiments,
    hf_id_for_psych_dataset_split,
    normalize_psych_dataset_split,
)
from utils.teh.teh_datasets import (
    MIXED_GAMBLES,
    is_mixed_gambles_dataset,
    is_psych101_dataset,
    psych101_metadata_root,
    valid_participant_ids_path_with_filter,
)


@dataclass(frozen=True)
class Psych101ParticipantStats:
    participant_id: int
    n_problems: int
    n_train_trials: int
    n_val_trials: int
    n_test_trials: int
    n_trials_total: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "participant_id": int(self.participant_id),
            "n_problems": int(self.n_problems),
            "n_train_trials": int(self.n_train_trials),
            "n_val_trials": int(self.n_val_trials),
            "n_test_trials": int(self.n_test_trials),
            "n_trials_total": int(self.n_trials_total),
        }


def _load_collect_participant_ids_module():
    path = Path(__file__).resolve().parent.parent / "tools" / "collect_participant_ids.py"
    spec = importlib.util.spec_from_file_location("collect_participant_ids", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_psych101_valid_participants(
    dataset_alias: str,
    *,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
) -> List[Psych101ParticipantStats]:
    """
    Scan Psych-101 HF rows for one experiment id.

    Returns stats for participants with non-empty train and test after TEH within-participant split.
    participant_id is the 0-based row index in the filtered HF split.
    """
    if not PSYCH101_BINARY_DATASETS[dataset_alias].get("implemented"):
        raise ValueError(f"Parser not implemented for {dataset_alias!r}")

    split = normalize_psych_dataset_split(psych_dataset_split)
    from teh import split_trials

    records: List[Psych101ParticipantStats] = []
    n = 0
    while True:
        try:
            experiments = get_psych101_binary_experiments(
                dataset_alias,
                n_participants=n + 1,
                split=split,
                local_dataset=local_dataset,
            )
        except Exception:
            break
        if len(experiments) <= n:
            break
        try:
            exp = experiments[n]
            train_trials, val_trials, test_trials, _ = split_trials(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
            if train_trials and test_trials:
                n_problems = len(exp.blocks)
                n_trials_total = sum(len(b.trials) for b in exp.blocks)
                records.append(
                    Psych101ParticipantStats(
                        participant_id=n,
                        n_problems=n_problems,
                        n_train_trials=len(train_trials),
                        n_val_trials=len(val_trials),
                        n_test_trials=len(test_trials),
                        n_trials_total=n_trials_total,
                    )
                )
        except Exception:
            pass
        n += 1
        if n > 5000:
            break
    return records


def collect_psych101_valid_ids(
    dataset_alias: str,
    *,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
) -> List[int]:
    """Backward-compatible: valid participant row indices only."""
    return [
        r.participant_id
        for r in collect_psych101_valid_participants(
            dataset_alias,
            split_ratio=split_ratio,
            split_seed=split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
        )
    ]


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


def _psych101_summary_stats(records: List[Psych101ParticipantStats]) -> Dict[str, Any]:
    if not records:
        return {
            "median_n_problems": 0,
            "median_n_trials_total": 0,
            "median_n_train_trials": 0,
            "median_n_val_trials": 0,
            "median_n_test_trials": 0,
        }

    def _median(vals: List[int]) -> float:
        s = sorted(vals)
        m = len(s) // 2
        return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2.0

    return {
        "median_n_problems": _median([r.n_problems for r in records]),
        "median_n_trials_total": _median([r.n_trials_total for r in records]),
        "median_n_train_trials": _median([r.n_train_trials for r in records]),
        "median_n_val_trials": _median([r.n_val_trials for r in records]),
        "median_n_test_trials": _median([r.n_test_trials for r in records]),
    }


def build_valid_participant_ids_payload(
    dataset: str,
    valid_ids: List[int],
    *,
    repo_root: Path,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: Optional[Path] = None,
    filter_mixed_gambles: bool = False,
    psych101_participants: Optional[List[Psych101ParticipantStats]] = None,
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
        split = normalize_psych_dataset_split(psych_dataset_split)
        exp_id = experiment_id_for_alias(dataset)
        meta_root = psych101_metadata_root(repo_root, split)
        try:
            meta_dir_rel = str(meta_root.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            meta_dir_rel = str(meta_root.resolve())
        payload["data_source"] = {
            "kind": "psych101",
            "experiment_id": exp_id,
            "psych_dataset_split": split,
            "hf_id": hf_id_for_psych_dataset_split(split),
            "metadata_dir": meta_dir_rel,
            "split_ratio": float(split_ratio),
            "split_seed": int(split_seed),
        }
        if local_dataset:
            payload["data_source"]["local_dataset"] = local_dataset
        payload["id_semantics"] = (
            f"0-based row index into Psych-101 {split} HF split filtered by experiment id "
            f"({exp_id})."
        )
        records = psych101_participants or []
        payload["participants"] = [r.to_dict() for r in records]
        payload["summary"] = _psych101_summary_stats(records)
        payload["trial_split_note"] = (
            "n_train_trials / n_val_trials / n_test_trials are after TEH within-participant "
            "block split (split_ratio, split_seed). n_problems is gamble-pair block count. "
            "n_trials_total is all parsed choice trials in the transcript."
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
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
    output_path: Optional[Path] = None,
) -> Path:
    """Scan data, write valid_participant_ids.json; return output path."""
    out_path = output_path or valid_participant_ids_path_with_filter(
        dataset,
        repo_root,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
    )

    psych_records: Optional[List[Psych101ParticipantStats]] = None
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
        psych_records = collect_psych101_valid_participants(
            dataset,
            split_ratio=split_ratio,
            split_seed=split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
        )
        valid_ids = [r.participant_id for r in psych_records]
    else:
        raise ValueError(f"Cannot collect valid participant ids for dataset {dataset!r}")

    if not valid_ids:
        raise ValueError(
            f"No valid participants found for {dataset!r} "
            f"(psych_dataset_split={psych_dataset_split!r}, "
            f"split_ratio={split_ratio}, split_seed={split_seed}). "
            "Check data availability or split settings."
        )

    payload = build_valid_participant_ids_payload(
        dataset,
        valid_ids,
        repo_root=repo_root,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=Path(mixed_gambles_csv),
        filter_mixed_gambles=filter_mixed_gambles,
        psych101_participants=psych_records,
    )
    write_valid_participant_ids_json(out_path, payload)
    return out_path


def ensure_valid_participant_ids_prepared(
    dataset: str,
    repo_root: Path,
    *,
    split_ratio: float = 0.8,
    split_seed: int = 42,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
    force_regenerate: bool = False,
) -> Path:
    """
    Ensure valid_participant_ids.json exists under datasets/psych101_{train|test}/<dataset>/.

    Uses the same split settings as the TEH run so participant scope matches evolution splits.
    """
    path = valid_participant_ids_path_with_filter(
        dataset,
        repo_root,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
    )
    if path.is_file() and not force_regenerate:
        return path

    if path.is_file() and force_regenerate:
        print(f"[TEH] Regenerating valid participant list: {path}")
    else:
        print(
            f"[TEH] No valid participant list at {path}; "
            f"scanning {dataset!r} (psych_dataset_split={psych_dataset_split}, "
            f"split_ratio={split_ratio}, split_seed={split_seed})..."
        )

    out = collect_and_write_valid_participant_ids(
        dataset,
        repo_root,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
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
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
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
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            filter_mixed_gambles=filter_mixed_gambles,
        )
    path = valid_participant_ids_path_with_filter(
        dataset,
        repo_root,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing valid participant list: {path}. "
            f"Run: python utils/tools/collect_teh_participant_ids.py "
            f"--dataset {dataset!r} --psych_dataset_split {psych_dataset_split!r}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["valid_participant_ids"])


def load_valid_participant_details(
    dataset: str,
    repo_root: Path,
    *,
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
) -> List[Dict[str, Any]]:
    """Load per-participant trial/problem stats from valid_participant_ids.json (Psych-101 only)."""
    path = valid_participant_ids_path_with_filter(
        dataset,
        repo_root,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
    )
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("participants", []))
