#!/usr/bin/env python3
"""
Collect valid participant IDs for choice13k / cpc18 / mixed_gambles using the same
validation rules as baseline_methods/prospect_theory.py (--all_data path).

Writes a JSON file next to the dataset location (for choice13k: datasets/choice13k/)
and prints count and min/max id.

Examples (run from repo root):
  python utils/tools/collect_participant_ids.py --dataset choice13k
  python utils/tools/collect_participant_ids.py --dataset cpc18 --data_path datasets/cpc18
  python utils/tools/collect_participant_ids.py --dataset mixed_gambles \\
      --mixed_gambles_csv datasets/mixed_gambles/data_all_2021-01-08.csv
  # Optional: restrict to gain_loss trials only (fewer valid participants):
  python utils/tools/collect_participant_ids.py --dataset mixed_gambles --filter_mixed_gambles
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DATA_MODULES_DIR = _REPO_ROOT / "data_modules"


def _load_data_module(module_filename: str, module_name: str) -> Any:
    """Load data_modules/<module_filename> without importing data_modules/__init__.py."""
    module_path = _DATA_MODULES_DIR / module_filename
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relative_to_repo(p: Path) -> str:
    try:
        return str(p.relative_to(_REPO_ROOT))
    except ValueError:
        return str(p)


def collect_choice13k(
    hf_dataset: str,
    experiment_filter: str,
    split_trials_fn,
    split_ratio: float,
    split_seed: int,
    allowed_raw_participant_ids: Optional[Set[int]] = None,
) -> List[int]:
    choice13k_mod = _load_data_module("choice13k.py", "choice13k_data_module")
    dataset = choice13k_mod.load_dataset(hf_dataset)
    test_split = dataset["test"]
    choices13k_ds = test_split.filter(lambda ex: ex["experiment"] == experiment_filter)
    valid: List[int] = []
    for participant_id in range(len(choices13k_ds)):
        try:
            row = choices13k_ds[participant_id]
            raw_participant_id = int(row["participant"])
            if (
                allowed_raw_participant_ids is not None
                and raw_participant_id not in allowed_raw_participant_ids
            ):
                continue
            exp = choice13k_mod._convert_to_experiment(row)
            train_trials, test_trials, _ = split_trials_fn(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
            if len(train_trials) > 0 and len(test_trials) > 0:
                valid.append(participant_id)
        except Exception:
            continue
    return valid


def collect_cpc18(data_path: Path, cpc18_mod: Any) -> List[int]:
    raw_file = data_path / "raw-comp-set-data-Track-2.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"CPC18 raw data not found: {raw_file}")
    import csv

    unique_subj_ids: set = set()
    with open(raw_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unique_subj_ids.add(int(row["SubjID"]))
    n = len(unique_subj_ids)
    valid: List[int] = []
    for participant_id in range(n):
        try:
            participant_data = cpc18_mod.load_cpc18_track2_data(
                data_path=str(data_path), participant_id=participant_id
            )
            train_trials, test_trials, _ = cpc18_mod.split_cpc18_trials(
                participant_data, train_ratio=0.8
            )
            if len(train_trials) > 0 and len(test_trials) > 0:
                valid.append(participant_id)
        except Exception:
            continue
    return valid


def collect_mixed_gambles(
    csv_path: Path,
    filter_gain_loss_only: bool,
    load_mixed_gambles_data_fn,
) -> List[int]:
    import csv

    unique_subjects: set = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unique_subjects.add(int(row["subject"]))
    valid: List[int] = []
    for participant_id in sorted(unique_subjects):
        try:
            train_trials, test_trials, _ = load_mixed_gambles_data_fn(
                str(csv_path), participant_id, filter_gain_loss_only=filter_gain_loss_only
            )
            if len(train_trials) > 0 and len(test_trials) > 0:
                valid.append(participant_id)
        except Exception:
            continue
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["choice13k", "cpc18", "mixed_gambles"],
        help="Which dataset to enumerate (same validation as prospect_theory --all_data).",
    )
    parser.add_argument(
        "--data_path",
        default="datasets/cpc18",
        help="CPC18 directory containing raw-comp-set-data-Track-2.csv (cpc18 only).",
    )
    parser.add_argument(
        "--mixed_gambles_csv",
        default="datasets/mixed_gambles/data_all_2021-01-08.csv",
        help="Mixed gambles CSV path (mixed_gambles only).",
    )
    parser.add_argument(
        "--filter_mixed_gambles",
        action="store_true",
        default=False,
        help=(
            "Mixed gambles only: if set, keep only gamble_type==gain_loss trials (fewer participants). "
            "Default False: use all trial types so the valid-id list is as large as possible."
        ),
    )
    parser.add_argument(
        "--hf_dataset",
        default="marcelbinz/Psych-101-test",
        help="HuggingFace dataset id for choice13k.",
    )
    parser.add_argument(
        "--experiment_filter",
        default="peterson2021using/exp1.csv",
        help="choice13k: experiment field value to filter rows.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output JSON path (default: next to that dataset in the repo).",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.9,
        help="Train split ratio for validity checks (choice13k only).",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Split seed for validity checks (choice13k only).",
    )
    parser.add_argument(
        "--choice13k_source_ids_json",
        default=None,
        help=(
            "Optional JSON file containing source raw participant IDs under key "
            "'participant_ids'. If set, choice13k valid ids are filtered to rows "
            "whose raw participant id is in that list."
        ),
    )
    args = parser.parse_args()

    valid_ids: List[int]
    out_path: Path
    payload: Dict[str, Any]

    if args.dataset == "choice13k":
        from baseline_methods.prospect_theory import split_trials

        allowed_raw_participant_ids: Optional[Set[int]] = None
        source_ids_path: Optional[Path] = None
        source_ids_hash: Optional[str] = None
        if args.choice13k_source_ids_json:
            source_ids_path = (
                Path(args.choice13k_source_ids_json)
                if Path(args.choice13k_source_ids_json).is_absolute()
                else (_REPO_ROOT / args.choice13k_source_ids_json)
            ).resolve()
            source_payload = json.loads(source_ids_path.read_text(encoding="utf-8"))
            source_ids = source_payload.get("participant_ids")
            if not isinstance(source_ids, list):
                raise ValueError(
                    "choice13k_source_ids_json must contain key 'participant_ids' as a list."
                )
            allowed_raw_participant_ids = {int(x) for x in source_ids}
            source_ids_hash = hashlib.sha256(source_ids_path.read_bytes()).hexdigest()

        try:
            valid_ids = collect_choice13k(
                args.hf_dataset,
                args.experiment_filter,
                split_trials,
                split_ratio=float(args.split_ratio),
                split_seed=int(args.split_seed),
                allowed_raw_participant_ids=allowed_raw_participant_ids,
            )
        except Exception as e:
            raise RuntimeError(
                "choice13k requires a working HuggingFace `datasets` stack (and often `torch`). "
                "Fix `import datasets` / metadata errors in your environment, then retry."
            ) from e
        out_dir = _REPO_ROOT / "datasets" / "choice13k"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.output) if args.output else out_dir / "valid_participant_ids.json"
        payload = {
            "dataset": "choice13k",
            "data_source": {
                "kind": "huggingface",
                "hf_dataset": args.hf_dataset,
                "hf_split": "test",
                "experiment_filter": args.experiment_filter,
                "split_ratio": float(args.split_ratio),
                "split_seed": int(args.split_seed),
            },
            "id_semantics": "0-based row index into the filtered HuggingFace test split (experiment == experiment_filter).",
            "valid_participant_ids": valid_ids,
        }
        if source_ids_path is not None:
            payload["data_source"]["source_raw_participant_ids_json"] = _relative_to_repo(source_ids_path)
            payload["data_source"]["source_raw_participant_ids_json_sha256"] = source_ids_hash
            payload["data_source"]["source_raw_participant_ids_count"] = (
                len(allowed_raw_participant_ids) if allowed_raw_participant_ids is not None else 0
            )
    elif args.dataset == "cpc18":
        cpc18_mod = _load_data_module("cpc18.py", "cpc18_data_module")
        data_path = (_REPO_ROOT / args.data_path).resolve() if not Path(args.data_path).is_absolute() else Path(args.data_path).resolve()
        valid_ids = collect_cpc18(data_path, cpc18_mod)
        out_path = Path(args.output) if args.output else data_path / "valid_participant_ids.json"
        payload = {
            "dataset": "cpc18",
            "data_source": {
                "kind": "local",
                "data_path": _relative_to_repo(data_path),
                "raw_file": "raw-comp-set-data-Track-2.csv",
            },
            "id_semantics": "0-based index into sorted unique SubjID (same as load_cpc18_track2_data participant_id).",
            "valid_participant_ids": valid_ids,
        }
    else:
        from baseline_methods.prospect_theory import load_mixed_gambles_data

        csv_path = (_REPO_ROOT / args.mixed_gambles_csv).resolve() if not Path(args.mixed_gambles_csv).is_absolute() else Path(args.mixed_gambles_csv).resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Mixed gambles CSV not found: {csv_path}")
        filter_gain_loss_only = bool(getattr(args, "filter_mixed_gambles", False))
        valid_ids = collect_mixed_gambles(csv_path, filter_gain_loss_only, load_mixed_gambles_data)
        suffix = "_gain_loss" if filter_gain_loss_only else ""
        default_name = f"valid_participant_ids{suffix}.json"
        out_path = Path(args.output) if args.output else csv_path.parent / default_name
        payload = {
            "dataset": "mixed_gambles",
            "data_source": {
                "kind": "local",
                "csv_path": _relative_to_repo(csv_path),
                "filter_gain_loss_only": filter_gain_loss_only,
            },
            "id_semantics": "CSV column 'subject' (integer), same as participant_id for mixed_gambles baselines.",
            "valid_participant_ids": valid_ids,
        }

    n = len(valid_ids)
    if n == 0:
        lo = hi = None
    else:
        lo, hi = min(valid_ids), max(valid_ids)

    payload["count"] = n
    payload["min_id"] = lo
    payload["max_id"] = hi

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"dataset={args.dataset}")
    print(f"count={n}")
    print(f"min_id={lo} max_id={hi}")
    if n:
        print(f"ordinal_index 0..{n - 1} -> valid_participant_ids[ordinal_index] (list order below)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
