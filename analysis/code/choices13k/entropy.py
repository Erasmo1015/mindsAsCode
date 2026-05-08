#!/usr/bin/env python3
"""
Standalone Choice13k participant entropy diagnostics (train split only).

This script is analysis-only:
- It computes entropy/inconsistency diagnostics from participant train trials.
- It does NOT modify TE fitness/selection/evolution behavior.

Split behavior matches TE Choice13k within-participant splitting:
- Split by problem blocks (not individual trials).
- All repeated trials for the same problem remain together.
- Entropy is computed on train blocks only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from datasets import load_from_disk


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_modules.choice13k import _convert_to_experiment  # pylint: disable=wrong-import-position


DEFAULT_EXPERIMENT_FILTER = "peterson2021using/exp1.csv"
VALID_IDS_JSON = REPO_ROOT / "datasets" / "choice13k" / "valid_participant_ids.json"


def round_floats(obj: Any, ndigits: int = 2) -> Any:
    """Recursively round all float values in nested structures."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits=ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits=ndigits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(round_floats(v, ndigits=ndigits) for v in obj)
    return obj


def binary_entropy(p: float, normalize: bool = True) -> float:
    """Binary entropy of Bernoulli(p); optional normalization to [0, 1]."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    h = -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
    if normalize:
        h /= math.log(2.0)
    return float(h)


def make_problem_key(problem: Dict[str, Any]) -> str:
    """
    Deterministic key for the underlying gamble problem.

    Includes:
    - gamble_A probs/rewards
    - gamble_B probs/rewards
    - option_keys
    - has_feedback

    Ignores:
    - history/action
    - options (redundant with option_keys)
    """
    payload = {
        "gamble_A": {
            "probs": problem["gamble_A"].get("probs"),
            "rewards": problem["gamble_A"].get("rewards"),
        },
        "gamble_B": {
            "probs": problem["gamble_B"].get("probs"),
            "rewards": problem["gamble_B"].get("rewards"),
        },
        "option_keys": problem.get("option_keys"),
        "has_feedback": bool(problem.get("has_feedback", False)),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def split_block_indices(n_blocks: int, split_ratio: float, split_seed: int) -> Tuple[set[int], set[int]]:
    """TE-consistent split by block indices."""
    if n_blocks < 2:
        raise ValueError(f"Need at least 2 blocks for train/test split; got {n_blocks}.")
    rng = np.random.default_rng(int(split_seed))
    perm = np.arange(n_blocks)
    rng.shuffle(perm)
    split_idx = int(n_blocks * float(split_ratio))
    split_idx = max(1, min(split_idx, n_blocks - 1))
    train_blocks = set(perm[:split_idx].tolist())
    test_blocks = set(perm[split_idx:].tolist())
    return train_blocks, test_blocks


def load_valid_participant_ids_from_json(path: Path) -> List[int]:
    """Load TE-compatible valid participant raw ids for Choice13k."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing valid participant list: {path}. "
            "Generate it with: python utils/tools/collect_participant_ids.py --dataset choice13k"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("valid_participant_ids")
    if not isinstance(values, list):
        raise ValueError(f"Invalid valid_participant_ids format in: {path}")
    return [int(v) for v in values]


def resolve_participants_for_scope(
    *,
    participant_scope: str,
    range_start_ordinal: Optional[int],
    range_end_ordinal: Optional[int],
) -> List[int]:
    """
    TE-consistent participant resolution from valid_participant_ids.json.

    For this entropy analysis script we support the same scope argument names,
    with `range` as the default/expected mode.
    """
    valid = load_valid_participant_ids_from_json(VALID_IDS_JSON)
    if participant_scope == "range":
        if range_start_ordinal is None or range_end_ordinal is None:
            raise ValueError(
                "--participant_scope range requires --range_start_ordinal and --range_end_ordinal (inclusive)."
            )
        if (
            range_start_ordinal < 0
            or range_end_ordinal >= len(valid)
            or range_start_ordinal > range_end_ordinal
        ):
            raise ValueError(
                f"Invalid ordinal range [{range_start_ordinal}, {range_end_ordinal}] "
                f"for valid list length {len(valid)} (0-based inclusive end)."
            )
        return valid[range_start_ordinal : range_end_ordinal + 1]
    raise ValueError(
        "This analysis script currently supports --participant_scope range only. "
        "Use TE for single/all experiment runs."
    )


def trials_from_blocks_chronological(exp: Any, block_indices: set[int]) -> List[Dict[str, Any]]:
    """Build trials from selected blocks; history is kept within each block only."""
    out: List[Dict[str, Any]] = []
    for bi, block in enumerate(exp.blocks):
        if bi not in block_indices:
            continue
        options = list(block.option_keys)
        history_accum: List[Dict[str, Any]] = []
        for trial in block.trials:
            out.append(
                {
                    "problem": {
                        "gamble_A": {
                            "probs": block.gamble_A.probs,
                            "rewards": block.gamble_A.rewards,
                        },
                        "gamble_B": {
                            "probs": block.gamble_B.probs,
                            "rewards": block.gamble_B.rewards,
                        },
                        "option_keys": options,
                        "has_feedback": block.has_feedback,
                    },
                    "history": list(history_accum),
                    "options": options,
                    "action": int(trial.action),
                }
            )
            history_accum.append({"action": int(trial.action), "feedback": trial.feedback})
    return out


def group_by_problem_key(trials: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group trials by deterministic problem identity, preserving first-seen order."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for trial in trials:
        key = make_problem_key(trial["problem"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(trial)
    return [grouped[k] for k in order]


def compute_train_entropy_diagnostics(
    participant_id: int,
    participant_scope: str,
    range_start_ordinal: Optional[int],
    range_end_ordinal: Optional[int],
    split_mode: str,
    split_ratio: float,
    split_seed: int,
    train_trials: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Main diagnostic: weighted within-problem action entropy on train trials.

    This estimates behavioral inconsistency across repeated trials of the same
    gamble problem. Entropy near 0 means consistent choices per problem;
    near 1 means approximately 50/50 choices within repeated problems.
    """
    groups = group_by_problem_key(train_trials)
    num_train_trials = len(train_trials)
    num_groups = len(groups)
    mean_group_size = (float(num_train_trials) / float(num_groups)) if num_groups > 0 else 0.0

    weighted_entropy_sum = 0.0
    inconsistent_groups = 0

    for g in groups:
        actions = [int(t["action"]) for t in g]
        p_b = float(sum(1 for a in actions if a == 1)) / float(len(actions))
        h_g = binary_entropy(p_b, normalize=True)
        weighted_entropy_sum += float(len(g)) * h_g
        if len(set(actions)) > 1:
            inconsistent_groups += 1

    within_problem_entropy = (weighted_entropy_sum / float(num_train_trials)) if num_train_trials > 0 else 0.0

    if num_train_trials > 0:
        p_b_all = float(sum(1 for t in train_trials if int(t["action"]) == 1)) / float(num_train_trials)
    else:
        p_b_all = 0.0
    overall_entropy = binary_entropy(p_b_all, normalize=True)
    frac_inconsistent_groups = (float(inconsistent_groups) / float(num_groups)) if num_groups > 0 else 0.0

    return {
        "participant_ordinal": int(participant_id),
        "participant_scope": str(participant_scope),
        "range_start_ordinal": range_start_ordinal,
        "range_end_ordinal": range_end_ordinal,
        "split_mode": str(split_mode),
        "split_ratio": float(split_ratio),
        "split_seed": int(split_seed),
        "num_train_trials": int(num_train_trials),
        "num_train_problem_groups": int(num_groups),
        "mean_train_group_size": float(mean_group_size),
        "within_problem_action_entropy": float(within_problem_entropy),
        "overall_action_entropy": float(overall_entropy),
        "num_inconsistent_problem_groups": int(inconsistent_groups),
        "frac_inconsistent_problem_groups": float(frac_inconsistent_groups),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write rows to CSV with explicit field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local_dataset",
        type=str,
        required=True,
        help="Path to local Choice13k dataset saved with datasets.save_to_disk.",
    )
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument(
        "--participant_scope",
        type=str,
        default="range",
        choices=["range"],
        help="Participant scope semantics aligned with TE. This script supports range mode.",
    )
    parser.add_argument(
        "--range_start_ordinal",
        type=int,
        default=0,
        help="0-based start index into datasets/choice13k/valid_participant_ids.json.",
    )
    parser.add_argument(
        "--range_end_ordinal",
        type=int,
        default=9,
        help="0-based inclusive end index into datasets/choice13k/valid_participant_ids.json.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="within_participant",
        choices=["within_participant", "across_participants"],
        help="Choice13k split mode semantics aligned with TE.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/data/entropy/choices13k"),
    )
    args = parser.parse_args()

    if not (0.0 < args.split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {args.split_ratio}")
    if args.split_mode != "within_participant":
        raise ValueError(
            "Entropy analysis currently supports --split_mode within_participant only."
        )

    dataset_path = Path(args.local_dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {dataset_path}")

    dataset = load_from_disk(str(dataset_path))
    if "test" not in dataset:
        raise ValueError("Local Choice13k dataset must contain 'test' split.")

    filtered = dataset["test"].filter(lambda ex: ex["experiment"] == DEFAULT_EXPERIMENT_FILTER)
    n_available = len(filtered)
    if n_available == 0:
        raise RuntimeError("No rows found after Choice13k experiment filtering on local dataset.")

    participant_ids = resolve_participants_for_scope(
        participant_scope=args.participant_scope,
        range_start_ordinal=args.range_start_ordinal,
        range_end_ordinal=args.range_end_ordinal,
    )
    if not participant_ids:
        raise RuntimeError("No valid participants selected after filtering.")

    all_rows: List[Dict[str, Any]] = []
    skipped = 0
    for pid in participant_ids:
        if int(pid) < 0 or int(pid) >= n_available:
            skipped += 1
            print(
                f"[WARN] Skipping participant {pid}: out of range for filtered dataset "
                f"(size={n_available})."
            )
            continue
        row = filtered[int(pid)]
        exp = _convert_to_experiment(row)
        try:
            train_blocks, _ = split_block_indices(len(exp.blocks), args.split_ratio, args.split_seed)
        except ValueError as exc:
            skipped += 1
            print(f"[WARN] Skipping participant {pid}: {exc}")
            continue
        train_trials = trials_from_blocks_chronological(exp, train_blocks)
        diag = compute_train_entropy_diagnostics(
            participant_id=pid,
            participant_scope=args.participant_scope,
            range_start_ordinal=args.range_start_ordinal,
            range_end_ordinal=args.range_end_ordinal,
            split_mode=args.split_mode,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            train_trials=train_trials,
        )
        all_rows.append(diag)

    if not all_rows:
        raise RuntimeError("No participants processed successfully.")

    all_rows = sorted(all_rows, key=lambda r: int(r["participant_ordinal"]))
    all_rows = [round_floats(row, ndigits=2) for row in all_rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    within_path = args.output_dir / "within_problem_action_entropy.csv"
    overall_path = args.output_dir / "overall_action_entropy.csv"
    inconsistency_path = args.output_dir / "inconsistency_rate.csv"
    combined_path = args.output_dir / "all_entropy_diagnostics.csv"

    common_fields = [
        "participant_ordinal",
        "num_train_trials",
        "num_train_problem_groups",
        "mean_train_group_size",
        "num_inconsistent_problem_groups",
        "frac_inconsistent_problem_groups",
    ]

    within_rows = []
    overall_rows = []
    inconsistency_rows = []
    for row in all_rows:
        within_rows.append(
            {
                **{k: row[k] for k in common_fields},
                "within_problem_action_entropy": row["within_problem_action_entropy"],
            }
        )
        overall_rows.append(
            {
                **{k: row[k] for k in common_fields},
                "overall_action_entropy": row["overall_action_entropy"],
            }
        )
        inconsistency_rows.append(
            {
                **{k: row[k] for k in common_fields},
                "inconsistency_rate": row["frac_inconsistent_problem_groups"],
            }
        )

    write_csv(within_path, within_rows, common_fields + ["within_problem_action_entropy"])
    write_csv(overall_path, overall_rows, common_fields + ["overall_action_entropy"])
    write_csv(inconsistency_path, inconsistency_rows, common_fields + ["inconsistency_rate"])
    write_csv(
        combined_path,
        all_rows,
        common_fields + ["within_problem_action_entropy", "overall_action_entropy"],
    )

    by_entropy = sorted(all_rows, key=lambda r: float(r["within_problem_action_entropy"]), reverse=True)
    top_n = min(10, len(by_entropy))
    p3_row = next((r for r in all_rows if int(r["participant_ordinal"]) == 3), None)

    print(f"Participants processed: {len(all_rows)} (skipped: {skipped})")
    print(
        "Selection/split config: participant_scope={scope}, range=[{start},{end}], "
        "split_mode={mode}, split_ratio={ratio}, split_seed={seed}".format(
            scope=args.participant_scope,
            start=args.range_start_ordinal,
            end=args.range_end_ordinal,
            mode=args.split_mode,
            ratio=args.split_ratio,
            seed=args.split_seed,
        )
    )
    print(f"Output: {within_path}")
    print(f"Output: {overall_path}")
    print(f"Output: {inconsistency_path}")
    print(f"Output: {combined_path}")
    print()
    print("Top participants by within_problem_action_entropy:")
    for row in by_entropy[:top_n]:
        print(
            "  participant_ordinal={pid} entropy={h:.2f} groups={g} "
            "inconsistency_rate={ir:.2f}".format(
                pid=int(row["participant_ordinal"]),
                h=float(row["within_problem_action_entropy"]),
                g=int(row["num_train_problem_groups"]),
                ir=float(row["frac_inconsistent_problem_groups"]),
            )
        )
    print()
    if p3_row is not None:
        print("Participant 3 diagnostics:")
        print(json.dumps(p3_row, indent=2))
    else:
        print("Participant 3 not present in processed participants.")


if __name__ == "__main__":
    main()
