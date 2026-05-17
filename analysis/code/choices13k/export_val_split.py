#!/usr/bin/env python3
"""Export exact train/val/test block splits and per-split trial JSON for Choice13k.

Participant indices are 0-based ordinals (same as non_strict ``participant_{N}`` folders and
``get_choice13k_experiments()[N]`` when using the default first-N participants).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.code.choices13k.one_phase.val import (  # noqa: E402
    DEFAULT_LOCAL_DATASET,
    split_block_assignment,
    split_trials_te_dr,
    trials_from_blocks_with_metadata,
)


def _trial_to_jsonable(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Copy trial dict for JSON export (includes block metadata)."""
    return dict(trial)


def _export_split_trials(
    *,
    exp: Any,
    block_indices: Set[int],
    participant_ordinal: int,
    out_path: Path,
) -> List[Dict[str, Any]]:
    tagged = trials_from_blocks_with_metadata(exp, block_indices)
    for t in tagged:
        t["participant_id"] = participant_ordinal
    trial_json = [_trial_to_jsonable(t) for t in tagged]
    out_path.write_text(json.dumps(trial_json, indent=2), encoding="utf-8")
    return trial_json


def export_split(
    output_dir: Path,
    n_participants: int,
    split_ratio: float,
    split_seed: int,
    local_dataset: Path | None,
) -> None:
    from data_modules.choice13k import get_choice13k_experiments

    output_dir = output_dir.resolve()
    train_dir = output_dir / "train_trials"
    val_dir = output_dir / "val_trials"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    local = str(local_dataset) if local_dataset is not None else None
    experiments = get_choice13k_experiments(n_participants=n_participants, local_dataset=local)

    split_rows: List[Dict[str, Any]] = []
    all_train: List[Dict[str, Any]] = []
    all_val: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for pid, exp in enumerate(experiments):
        n_blocks = len(exp.blocks)
        perm, train_blocks, val_blocks, test_blocks = split_block_assignment(
            n_blocks, split_ratio=split_ratio, split_seed=split_seed
        )
        train_trials, val_trials, test_trials = split_trials_te_dr(
            exp, split_ratio=split_ratio, split_seed=split_seed
        )

        for bi in range(n_blocks):
            if bi in train_blocks:
                split_name = "train"
            elif bi in val_blocks:
                split_name = "val"
            else:
                split_name = "test"
            split_rows.append(
                {
                    "participant_id": pid,
                    "block_index": bi,
                    "split": split_name,
                    "perm_position": perm.index(bi),
                }
            )

        train_json = _export_split_trials(
            exp=exp,
            block_indices=train_blocks,
            participant_ordinal=pid,
            out_path=train_dir / f"participant_{pid}_train_trials.json",
        )
        all_train.extend(train_json)

        val_json = _export_split_trials(
            exp=exp,
            block_indices=val_blocks,
            participant_ordinal=pid,
            out_path=val_dir / f"participant_{pid}_val_trials.json",
        )
        all_val.extend(val_json)

        summary_rows.append(
            {
                "participant_id": pid,
                "n_blocks": n_blocks,
                "n_train_blocks": len(train_blocks),
                "n_val_blocks": len(val_blocks),
                "n_test_blocks": len(test_blocks),
                "train_block_indices": ",".join(str(b) for b in sorted(train_blocks)),
                "val_block_indices": ",".join(str(b) for b in sorted(val_blocks)),
                "test_block_indices": ",".join(str(b) for b in sorted(test_blocks)),
                "block_permutation": ",".join(str(b) for b in perm),
                "n_train_trials": len(train_trials),
                "n_val_trials": len(val_trials),
                "n_test_trials": len(test_trials),
            }
        )

    manifest = {
        "split_ratio": split_ratio,
        "split_seed": split_seed,
        "n_participants": len(experiments),
        "participant_id_semantics": (
            "0-based ordinal into get_choice13k_experiments() / non_strict participant_{N} "
            "(for default first-N runs, equals HF row index in valid_participant_ids.json)."
        ),
        "description": (
            "Within-participant block split: shuffle block indices with split_seed, "
            "first n_train blocks -> train, next n_val -> val, rest -> test. "
            "Trials are chronological within each block; history does not cross blocks."
        ),
        "trial_exports": {
            "train": "train_trials/participant_{ordinal}_train_trials.json",
            "val": "val_trials/participant_{ordinal}_val_trials.json",
        },
        "source": "analysis/code/choices13k/one_phase/val.py split_block_assignment",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    with (output_dir / "participant_block_splits.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["participant_id", "block_index", "split", "perm_position"],
        )
        w.writeheader()
        w.writerows(split_rows)

    with (output_dir / "participant_split_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()) if summary_rows else [],
        )
        w.writeheader()
        w.writerows(summary_rows)

    (output_dir / "all_participants_train_trials.json").write_text(
        json.dumps(all_train, indent=2), encoding="utf-8"
    )
    (output_dir / "all_participants_val_trials.json").write_text(
        json.dumps(all_val, indent=2), encoding="utf-8"
    )

    print(f"Wrote split export -> {output_dir}")
    print(f"  manifest: split_manifest.json")
    print(f"  block rows: participant_block_splits.csv ({len(split_rows)} rows)")
    print(f"  per-participant train: train_trials/participant_*_train_trials.json")
    print(f"  pooled train: all_participants_train_trials.json ({len(all_train)} trials)")
    print(f"  per-participant val: val_trials/participant_*_val_trials.json")
    print(f"  pooled val: all_participants_val_trials.json ({len(all_val)} trials)")


def main() -> None:
    default_out = REPO_ROOT / "analysis/data/choices13k/split_ratio060_seed0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--n-participants", type=int, default=10)
    parser.add_argument("--split-ratio", type=float, default=0.6)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--local-dataset", type=Path, default=DEFAULT_LOCAL_DATASET)
    args = parser.parse_args()

    local = args.local_dataset
    if local is not None:
        local = local if local.is_absolute() else REPO_ROOT / local
        if not local.exists():
            print(f"Warning: local dataset not found at {local}; will try HF download.")
            local = None

    export_split(
        output_dir=args.output_dir,
        n_participants=args.n_participants,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        local_dataset=local,
    )


if __name__ == "__main__":
    main()
