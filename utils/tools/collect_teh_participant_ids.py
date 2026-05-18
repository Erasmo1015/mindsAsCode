#!/usr/bin/env python3
"""
Collect valid participant IDs for TEH-supported datasets.

Writes JSON under datasets/psych101_train/<dataset>/ or datasets/psych101_test/<dataset>/
(see utils.teh.teh_datasets). mixed_gambles stays under datasets/mixed_gambles/.
teh.py also auto-creates this file on first run when it is missing.

Examples (from repo root):
  python utils/tools/collect_teh_participant_ids.py --dataset peterson2021using
  python utils/tools/collect_teh_participant_ids.py --dataset plonsky2018when --psych_dataset_split test
  python utils/tools/collect_teh_participant_ids.py --dataset mixed_gambles
  python utils/tools/collect_teh_participant_ids.py --dataset mixed_gambles --filter_mixed_gambles
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.psych101_binary import DEFAULT_PSYCH_DATASET_SPLIT
from utils.teh.teh_datasets import IMPLEMENTED_PSYCH101_ALIASES, MIXED_GAMBLES, is_mixed_gambles_dataset
from utils.teh.participant_ids import (
    collect_and_write_valid_participant_ids,
    ensure_valid_participant_ids_prepared,
)


def main() -> None:
    choices = sorted(IMPLEMENTED_PSYCH101_ALIASES | {MIXED_GAMBLES})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=choices)
    parser.add_argument(
        "--psych_dataset_split",
        type=str,
        default=DEFAULT_PSYCH_DATASET_SPLIT,
        choices=["train", "test"],
        help=(
            "Psych-101 HF corpus for id collection (default: train). "
            "Ignored for mixed_gambles."
        ),
    )
    parser.add_argument(
        "--local_dataset",
        default=None,
        help="Optional datasets.load_from_disk path for Psych-101 rows.",
    )
    parser.add_argument(
        "--mixed_gambles_csv",
        default="datasets/mixed_gambles/data_all_2021-01-08.csv",
        help="CSV path for mixed_gambles only.",
    )
    parser.add_argument(
        "--filter_mixed_gambles",
        action="store_true",
        help="mixed_gambles: only gain_loss trials; writes valid_participant_ids_gain_loss.json",
    )
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split-specific valid participant ids JSON if present.",
    )
    args = parser.parse_args()

    psych_split = args.psych_dataset_split
    if is_mixed_gambles_dataset(args.dataset):
        psych_split = DEFAULT_PSYCH_DATASET_SPLIT

    if args.output:
        out_path = collect_and_write_valid_participant_ids(
            args.dataset,
            _REPO_ROOT,
            split_ratio=float(args.split_ratio),
            split_seed=int(args.split_seed),
            psych_dataset_split=psych_split,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
            filter_mixed_gambles=bool(args.filter_mixed_gambles),
            output_path=Path(args.output),
        )
    else:
        out_path = ensure_valid_participant_ids_prepared(
            args.dataset,
            _REPO_ROOT,
            split_ratio=float(args.split_ratio),
            split_seed=int(args.split_seed),
            psych_dataset_split=psych_split,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
            filter_mixed_gambles=bool(args.filter_mixed_gambles),
            force_regenerate=bool(args.force),
        )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    valid_ids = payload["valid_participant_ids"]
    print(f"Wrote {len(valid_ids)} valid participant ids -> {out_path}")
    if valid_ids:
        print(f"  id range: {min(valid_ids)} .. {max(valid_ids)}")
    if args.dataset in IMPLEMENTED_PSYCH101_ALIASES:
        print(f"  psych_dataset_split: {psych_split}")


if __name__ == "__main__":
    main()
