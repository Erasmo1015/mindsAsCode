#!/usr/bin/env python3
"""Evaluate Choice13k evolved programs on train/val/test with optional external consistency gate.

Default: read best programs from an experiment folder in place; write ``adhoc_val_summary.csv``
there with train/val/test and gated_test log-likelihoods (external gate only).

Other modes: ``--legacy-copy`` (copy to part_X.py), ``--gated-only`` (score part_*.py).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RUN_DIR = REPO_ROOT / "generated_outputs/choice13k/non_strict/run_260430_013702"
DEFAULT_LOCAL_DATASET = REPO_ROOT / "datasets/downloaded/choices13k/Psych-101-test"
PROGRAMS_DIR = REPO_ROOT / "analysis/data/choices13k/one_phase/programs"
RESULTS_DIR = REPO_ROOT / "analysis/data/choices13k/one_phase/results"
OUTPUT_CSV = RESULTS_DIR / "val_train_val_test_scores.csv"
ADHOC_SUMMARY_CSV = "adhoc_val_summary.csv"
DEFAULT_CENTAUR_LOGLIK_CSV = (
    REPO_ROOT
    / "generated_outputs/choice13k/centaur/run_260517_013408/participant_details_loglik.csv"
)
CENTAUR_TEST_LOGLIK_COL = "centaur_test_loglik"

SPLIT_RATIO = 0.4
SPLIT_SEED = 0
VAL_LOGLIK_THRESHOLD = -0.45
_LOGLIK_CLAMP_EPS = 1e-9


def _trials_from_blocks_chronological(exp: Any, block_indices: Set[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bi, block in enumerate(exp.blocks):
        if bi not in block_indices:
            continue
        options = block.option_keys
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
                    "action": trial.action,
                }
            )
            history_accum.append({"action": trial.action, "feedback": trial.feedback})
    return out


def split_block_assignment(
    n_blocks: int,
    split_ratio: float = SPLIT_RATIO,
    split_seed: int = SPLIT_SEED,
) -> Tuple[List[int], Set[int], Set[int], Set[int]]:
    """
    Deterministic train/val/test block assignment (same logic as Template_evo / te_dr).

    Returns (block_permutation, train_blocks, val_blocks, test_blocks) where block indices
    refer to original experiment.blocks order.
    """
    if n_blocks < 3:
        raise ValueError(
            f"Choice13k train/val/test split requires at least 3 blocks; got {n_blocks}."
        )
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")

    rng = np.random.default_rng(split_seed)
    perm = np.arange(n_blocks)
    rng.shuffle(perm)

    n_train = int(n_blocks * split_ratio)
    n_train = max(1, min(n_train, n_blocks - 2))
    n_rem = n_blocks - n_train
    n_val = (n_rem + 1) // 2
    n_test = n_rem - n_val
    if n_val < 1:
        n_val = 1
        n_test = max(1, n_rem - 1)
        n_train = n_blocks - n_val - n_test
        n_train = max(1, n_train)
        n_rem = n_blocks - n_train
        n_val = n_rem // 2
        n_test = n_rem - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n_rem - 1)
        n_train = n_blocks - n_val - n_test
        n_train = max(1, n_train)

    train_blocks = set(perm[:n_train].tolist())
    val_blocks = set(perm[n_train : n_train + n_val].tolist())
    test_blocks = set(perm[n_train + n_val :].tolist())
    return perm.tolist(), train_blocks, val_blocks, test_blocks


def split_trials_te_dr(
    exp: Any,
    split_ratio: float = SPLIT_RATIO,
    split_seed: int = SPLIT_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match te_dr.split_trials: train/val/test by block with split_ratio as train fraction."""
    _, train_blocks, val_blocks, test_blocks = split_block_assignment(
        len(exp.blocks), split_ratio=split_ratio, split_seed=split_seed
    )
    train_trials = _trials_from_blocks_chronological(exp, train_blocks)
    val_trials = _trials_from_blocks_chronological(exp, val_blocks)
    test_trials = _trials_from_blocks_chronological(exp, test_blocks)
    return train_trials, val_trials, test_trials


def trials_from_blocks_with_metadata(
    exp: Any, block_indices: Set[int]
) -> List[Dict[str, Any]]:
    """Like _trials_from_blocks_chronological but tags participant_id and block_index."""
    out: List[Dict[str, Any]] = []
    for bi, block in enumerate(exp.blocks):
        if bi not in block_indices:
            continue
        options = block.option_keys
        history_accum: List[Dict[str, Any]] = []
        for ti, trial in enumerate(block.trials):
            out.append(
                {
                    "participant_id": None,
                    "block_index": bi,
                    "trial_index_in_block": ti,
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
                    "action": trial.action,
                    "feedback": trial.feedback,
                }
            )
            history_accum.append({"action": trial.action, "feedback": trial.feedback})
    return out


def _load_choose_fn(program_path: Path) -> Callable:
    mod_name = f"part_program_{program_path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, str(program_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load program module from: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    choose = getattr(module, "choose", None)
    if choose is None or not callable(choose):
        raise RuntimeError(f"'choose(problem, history)' not found in: {program_path}")
    return choose


def _parse_choose_output(p_raw: Any) -> float:
    if isinstance(p_raw, bool) or (
        isinstance(p_raw, (int, np.integer)) and int(p_raw) in (0, 1)
    ):
        return 1.0 if int(p_raw) == 1 else 0.0
    if isinstance(p_raw, float):
        p_use = p_raw
    else:
        raise TypeError(f"choose must return float or 0/1, got {type(p_raw)}")
    if not (0.0 <= p_use <= 1.0):
        raise ValueError(f"invalid probability: {p_use!r}")
    return p_use


def _clamp_probability(p: float) -> float:
    return min(max(p, _LOGLIK_CLAMP_EPS), 1.0 - _LOGLIK_CLAMP_EPS)


def apply_consistency_gate(raw_p: float, val_loglik: float) -> float:
    clamped = _clamp_probability(raw_p)
    if val_loglik < VAL_LOGLIK_THRESHOLD:
        consistency = max(0.0, 1.0 - (VAL_LOGLIK_THRESHOLD - val_loglik))
    else:
        consistency = 1.0
    return consistency * clamped + (1.0 - consistency) * 0.5


def wrap_choose_with_consistency_gate(
    choose_fn: Callable,
    val_loglik: float,
) -> Callable:
    def gated_choose(problem: Any, history: Any) -> float:
        p_raw = choose_fn(problem, history)
        raw_p = _parse_choose_output(p_raw)
        return apply_consistency_gate(raw_p, val_loglik)

    return gated_choose


def evaluate_split(choose_fn: Callable, trials: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean log-likelihood and accuracy (matches te_dr.evaluate_choice13k_program)."""
    total = len(trials)
    if total == 0:
        return {"avg_loglik": float("nan"), "accuracy": float("nan"), "errors": 0, "n_trials": 0}

    loglik_acc = 0.0
    correct = 0
    errors = 0
    for t in trials:
        y = int(t["action"])
        try:
            p_raw = choose_fn(t["problem"], t["history"])
            p_use = _parse_choose_output(p_raw)
            p = _clamp_probability(p_use)
            loglik_acc += y * math.log(p) + (1 - y) * math.log(1.0 - p)
            if isinstance(p_raw, float):
                pred = 1 if p_raw >= 0.5 else 0
            else:
                pred = 1 if int(p_raw) == 1 else 0
            correct += int(pred == y)
        except Exception:
            errors += 1

    if errors > 0:
        avg_ll = float("-inf")
    else:
        avg_ll = loglik_acc / total
    acc = correct / total if total > 0 else 0.0
    return {
        "avg_loglik": float(avg_ll),
        "accuracy": float(acc),
        "errors": int(errors),
        "n_trials": int(total),
    }


def _find_best_program(participant_dir: Path) -> Optional[Path]:
    """Resolve the final evolved program for a participant (never the gated copy)."""
    results_path = participant_dir / "results.json"
    if results_path.is_file():
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
            for key in ("overall_best_train", "overall_best_test"):
                block = data.get(key) or {}
                program_file = block.get("program_file")
                if program_file:
                    candidate = participant_dir / str(program_file)
                    if candidate.is_file():
                        return candidate
        except (json.JSONDecodeError, OSError):
            pass

    iter_pat = re.compile(r"^best_program_fr_iter(\d+)_cand(\d+)\.py$")
    iter_matches: List[Tuple[Tuple[int, int], Path]] = []
    for p in sorted(participant_dir.glob("best_program*.py")):
        if p.name in {"best_program_gated.py"} or p.name.endswith("_gated.py"):
            continue
        m = iter_pat.match(p.name)
        if m:
            iter_matches.append(((int(m.group(1)), int(m.group(2))), p))

    if iter_matches:
        return max(iter_matches, key=lambda x: x[0])[1]

    other = sorted(
        p
        for p in participant_dir.glob("best_program*.py")
        if p.name not in {"best_program_gated.py"} and not p.name.endswith("_gated.py")
    )
    if other:
        return other[0]
    return None


def _discover_participant_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    pat = re.compile(r"^participant_(\d+)$")
    for p in sorted(run_dir.iterdir()):
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out


def _mean_finite(vals: List[float]) -> float:
    good = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(good)) if good else float("nan")


TRAIN_VAL_LOGLIK_COL = "train_val_loglik"

ADHOC_FIELDNAMES_BASE = [
    "participant_id",
    "train_loglik",
    "val_loglik",
    TRAIN_VAL_LOGLIK_COL,
    "test_loglik",
    "gated_test_loglik",
]


def _adhoc_fieldnames(compare_centaur: bool) -> List[str]:
    names = list(ADHOC_FIELDNAMES_BASE)
    if compare_centaur:
        names.append(CENTAUR_TEST_LOGLIK_COL)
    return names


def _load_centaur_test_loglik(centaur_csv: Path) -> Dict[int, float]:
    """Load Centaur test_loglik keyed by participant_id."""
    if not centaur_csv.is_file():
        raise FileNotFoundError(f"Centaur loglik CSV not found: {centaur_csv}")
    out: Dict[int, float] = {}
    with centaur_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        if "participant_id" not in reader.fieldnames:
            raise ValueError(
                f"{centaur_csv}: missing participant_id column (got {reader.fieldnames})"
            )
        if "test_loglik" not in reader.fieldnames:
            raise ValueError(
                f"{centaur_csv}: missing test_loglik column (got {reader.fieldnames})"
            )
        for row in reader:
            raw = row.get("participant_id")
            if raw is None or str(raw).strip() == "":
                continue
            pid = int(float(raw))
            tl = row.get("test_loglik")
            if tl is None or str(tl).strip() == "":
                continue
            out[pid] = float(tl)
    return out


def _evaluate_participant_splits(
    program_path: Path,
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Ungated train/val/train+val/test loglik plus externally gated test loglik."""
    choose_fn = _load_choose_fn(program_path)
    tr = evaluate_split(choose_fn, train_trials)
    va = evaluate_split(choose_fn, val_trials)
    tv = evaluate_split(choose_fn, train_trials + val_trials)
    te = evaluate_split(choose_fn, test_trials)
    val_ll = float(va["avg_loglik"])
    gated_fn = wrap_choose_with_consistency_gate(choose_fn, val_ll)
    te_gated = evaluate_split(gated_fn, test_trials)
    return {
        "train_loglik": float(tr["avg_loglik"]),
        "val_loglik": val_ll,
        TRAIN_VAL_LOGLIK_COL: float(tv["avg_loglik"]),
        "test_loglik": float(te["avg_loglik"]),
        "gated_test_loglik": float(te_gated["avg_loglik"]),
    }


def _format_loglik_cell(value: Any, width: int) -> str:
    if value is None or value == "":
        return f"{'':>{width}}"
    if isinstance(value, str):
        return f"{value:>{width}}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):>{width}d}"
    if isinstance(value, float) and not np.isfinite(value):
        return f"{'n/a':>{width}}"
    return f"{float(value):>{width}.4f}"


def _better_than_centaur_row(
    data_rows: List[Dict[str, Any]],
    fieldnames: List[str],
    centaur_test: Dict[int, float],
) -> Dict[str, Any]:
    """Count participants per column with loglik strictly greater than Centaur test."""
    row: Dict[str, Any] = {"participant_id": "better"}
    compare_cols = [
        c
        for c in fieldnames
        if c not in ("participant_id", CENTAUR_TEST_LOGLIK_COL)
    ]
    for col in compare_cols:
        count = 0
        for r in data_rows:
            pid = r["participant_id"]
            if pid not in centaur_test:
                continue
            ours = r.get(col)
            centaur_ll = centaur_test[pid]
            if ours is None or not np.isfinite(float(ours)):
                continue
            if float(ours) > centaur_ll:
                count += 1
        row[col] = count
    if CENTAUR_TEST_LOGLIK_COL in fieldnames:
        row[CENTAUR_TEST_LOGLIK_COL] = ""
    return row


def _adhoc_column_widths(fieldnames: List[str]) -> List[int]:
    widths: List[int] = []
    for name in fieldnames:
        if name == "participant_id":
            widths.append(16)
        elif name in ("gated_test_loglik", CENTAUR_TEST_LOGLIK_COL):
            widths.append(18)
        elif name == TRAIN_VAL_LOGLIK_COL:
            widths.append(20)
        else:
            widths.append(14)
    return widths


def _print_adhoc_table(rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    col_w = _adhoc_column_widths(fieldnames)
    labels = fieldnames
    sep = "  "
    header_parts = [f"{labels[0]:<{col_w[0]}}"] + [
        f"{labels[i]:>{col_w[i]}}" for i in range(1, len(labels))
    ]
    header = sep.join(header_parts)
    print(header)
    print("-" * (sum(col_w[: len(labels)]) + len(sep) * max(0, len(labels) - 1)))
    for row in rows:
        pid = str(row["participant_id"])
        parts = [f"{pid:<{col_w[0]}}"]
        for i, name in enumerate(labels[1:], start=1):
            parts.append(_format_loglik_cell(row.get(name), col_w[i]))
        print(sep.join(parts))


def run_adhoc_experiment(
    experiment_dir: Path,
    local_dataset: Optional[Path],
    split_ratio: float,
    split_seed: int,
    output_csv: Path,
    *,
    compare_centaur: bool = True,
    centaur_loglik_csv: Optional[Path] = None,
) -> None:
    """Evaluate each participant's best program in an experiment folder (no source changes)."""
    from data_modules.choice13k import get_choice13k_experiments

    participants = _discover_participant_dirs(experiment_dir)
    if not participants:
        raise FileNotFoundError(f"No participant_* directories in {experiment_dir}")

    max_pid = max(pid for pid, _ in participants)
    local = str(local_dataset) if local_dataset is not None else None
    experiments = get_choice13k_experiments(n_participants=max_pid + 1, local_dataset=local)

    centaur_test: Dict[int, float] = {}
    if compare_centaur:
        centaur_path = centaur_loglik_csv or DEFAULT_CENTAUR_LOGLIK_CSV
        centaur_test = _load_centaur_test_loglik(centaur_path)
        print(f"Centaur test loglik: {centaur_path}")

    fieldnames = _adhoc_fieldnames(compare_centaur)
    rows: List[Dict[str, Any]] = []
    skipped: List[int] = []
    for pid, pdir in participants:
        if pid >= len(experiments):
            raise IndexError(f"participant {pid} not in dataset (only {len(experiments)} experiments)")

        program_path = _find_best_program(pdir)
        if program_path is None:
            skipped.append(pid)
            print(f"Warning: skipping participant {pid} (no best_program*.py in {pdir.name})")
            continue

        train_trials, val_trials, test_trials = split_trials_te_dr(
            experiments[pid], split_ratio=split_ratio, split_seed=split_seed
        )
        metrics = _evaluate_participant_splits(program_path, train_trials, val_trials, test_trials)
        row: Dict[str, Any] = {
            "participant_id": pid,
            "train_loglik": round(metrics["train_loglik"], 4),
            "val_loglik": round(metrics["val_loglik"], 4),
            TRAIN_VAL_LOGLIK_COL: round(metrics[TRAIN_VAL_LOGLIK_COL], 4),
            "test_loglik": round(metrics["test_loglik"], 4),
            "gated_test_loglik": round(metrics["gated_test_loglik"], 4),
        }
        if compare_centaur:
            centaur_ll = centaur_test.get(pid)
            row[CENTAUR_TEST_LOGLIK_COL] = (
                round(centaur_ll, 4) if centaur_ll is not None else float("nan")
            )
        rows.append(row)

    if skipped:
        print(f"Skipped {len(skipped)} participant(s) without a best program: {skipped}")
    if not rows:
        raise FileNotFoundError(
            f"No evaluable participants in {experiment_dir} (all missing best_program*.py)"
        )

    data_rows = rows
    avg_row: Dict[str, Any] = {
        "participant_id": "avg",
        "train_loglik": round(_mean_finite([r["train_loglik"] for r in data_rows]), 4),
        "val_loglik": round(_mean_finite([r["val_loglik"] for r in data_rows]), 4),
        TRAIN_VAL_LOGLIK_COL: round(
            _mean_finite([r[TRAIN_VAL_LOGLIK_COL] for r in data_rows]), 4
        ),
        "test_loglik": round(_mean_finite([r["test_loglik"] for r in data_rows]), 4),
        "gated_test_loglik": round(_mean_finite([r["gated_test_loglik"] for r in data_rows]), 4),
    }
    if compare_centaur:
        centaur_vals = [
            centaur_test[pid]
            for pid in (r["participant_id"] for r in data_rows)
            if pid in centaur_test
        ]
        avg_row[CENTAUR_TEST_LOGLIK_COL] = (
            round(_mean_finite(centaur_vals), 4) if centaur_vals else float("nan")
        )
        rows.append(avg_row)
        rows.append(_better_than_centaur_row(data_rows, fieldnames, centaur_test))
    else:
        rows.append(avg_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nAd-hoc evaluation: {experiment_dir}")
    print(f"split_ratio={split_ratio}, split_seed={split_seed}\n")
    _print_adhoc_table(rows, fieldnames)
    footer = "avg, better" if compare_centaur else "avg"
    print(f"\nWrote {len(rows)} rows (incl. {footer}) -> {output_csv}")


def _part_path(programs_dir: Path, pid: int) -> Path:
    return programs_dir / f"part_{pid}.py"


def _discover_programs(programs_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    pat = re.compile(r"^part_(\d+)\.py$")
    for p in sorted(programs_dir.glob("part_*.py")):
        m = pat.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out


def run_gated_only(
    local_dataset: Optional[Path],
    split_ratio: float,
    split_seed: int,
    programs_dir: Path,
    output_csv: Path,
) -> None:
    """part_X train/val/test loglik + externally gated test-only (new_test_loglik)."""
    from data_modules.choice13k import get_choice13k_experiments

    programs = _discover_programs(programs_dir)
    if not programs:
        raise FileNotFoundError(f"No part_*.py files in {programs_dir}")

    max_pid = max(pid for pid, _ in programs)
    local = str(local_dataset) if local_dataset is not None else None
    experiments = get_choice13k_experiments(n_participants=max_pid + 1, local_dataset=local)

    fieldnames = [
        "participant_id",
        "program",
        "split_ratio",
        "split_seed",
        "n_train_trials",
        "n_val_trials",
        "n_test_trials",
        "train_loglik",
        "val_loglik",
        "test_loglik",
        "new_test_loglik",
    ]
    rows: List[Dict[str, Any]] = []

    for pid, part_path in programs:
        if pid >= len(experiments):
            raise IndexError(f"participant {pid} not in dataset (only {len(experiments)} experiments)")

        train_trials, val_trials, test_trials = split_trials_te_dr(
            experiments[pid], split_ratio=split_ratio, split_seed=split_seed
        )
        choose_fn = _load_choose_fn(part_path)
        tr = evaluate_split(choose_fn, train_trials)
        va = evaluate_split(choose_fn, val_trials)
        te = evaluate_split(choose_fn, test_trials)
        val_ll = float(va["avg_loglik"])
        gated_fn = wrap_choose_with_consistency_gate(choose_fn, val_ll)
        te_gated = evaluate_split(gated_fn, test_trials)

        row = {
            "participant_id": pid,
            "program": part_path.name,
            "split_ratio": split_ratio,
            "split_seed": split_seed,
            "n_train_trials": tr["n_trials"],
            "n_val_trials": va["n_trials"],
            "n_test_trials": te["n_trials"],
            "train_loglik": round(tr["avg_loglik"], 4),
            "val_loglik": round(va["avg_loglik"], 4),
            "test_loglik": round(te["avg_loglik"], 4),
            "new_test_loglik": round(te_gated["avg_loglik"], 4),
        }
        rows.append(row)
        print(
            f"participant {pid}: train_ll={row['train_loglik']:.4f} "
            f"val_ll={row['val_loglik']:.4f} test_ll={row['test_loglik']:.4f} "
            f"new_test_ll={row['new_test_loglik']:.4f}"
        )

    avg_row: Dict[str, Any] = {
        "participant_id": "avg",
        "program": "",
        "split_ratio": split_ratio,
        "split_seed": split_seed,
        "n_train_trials": int(round(np.mean([r["n_train_trials"] for r in rows]))),
        "n_val_trials": int(round(np.mean([r["n_val_trials"] for r in rows]))),
        "n_test_trials": int(round(np.mean([r["n_test_trials"] for r in rows]))),
        "train_loglik": round(_mean_finite([r["train_loglik"] for r in rows]), 4),
        "val_loglik": round(_mean_finite([r["val_loglik"] for r in rows]), 4),
        "test_loglik": round(_mean_finite([r["test_loglik"] for r in rows]), 4),
        "new_test_loglik": round(_mean_finite([r["new_test_loglik"] for r in rows]), 4),
    }
    rows.append(avg_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows (incl. avg) -> {output_csv}")
    print(f"split_ratio={split_ratio}, split_seed={split_seed}")


def run(
    run_dir: Path,
    local_dataset: Optional[Path],
    split_ratio: float,
    split_seed: int,
    programs_dir: Path,
    output_csv: Path,
) -> None:
    from data_modules.choice13k import get_choice13k_experiments

    participants = _discover_participant_dirs(run_dir)
    if not participants:
        raise FileNotFoundError(f"No participant_* directories in {run_dir}")

    programs_dir.mkdir(parents=True, exist_ok=True)

    max_pid = max(pid for pid, _ in participants)
    local = str(local_dataset) if local_dataset is not None else None
    experiments = get_choice13k_experiments(n_participants=max_pid + 1, local_dataset=local)

    fieldnames = [
        "participant_id",
        "program",
        "participant_val_loglik",
        "n_train_trials",
        "n_val_trials",
        "n_test_trials",
        "train_loglik",
        "val_loglik",
        "test_loglik",
        "new_test_loglik",
        "train_acc",
        "val_acc",
        "test_acc",
        "train_errors",
        "val_errors",
        "test_errors",
        "gated_test_errors",
    ]
    rows: List[Dict[str, Any]] = []
    skipped: List[int] = []

    for pid, pdir in participants:
        if pid >= len(experiments):
            raise IndexError(f"participant {pid} not in dataset (only {len(experiments)} experiments)")

        src_path = _find_best_program(pdir)
        if src_path is None:
            skipped.append(pid)
            print(f"Warning: skipping participant {pid} (no best_program*.py in {pdir.name})")
            continue

        part_path = _part_path(programs_dir, pid)
        part_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")

        train_trials, val_trials, test_trials = split_trials_te_dr(
            experiments[pid], split_ratio=split_ratio, split_seed=split_seed
        )

        choose_fn = _load_choose_fn(part_path)
        tr = evaluate_split(choose_fn, train_trials)
        va = evaluate_split(choose_fn, val_trials)
        te = evaluate_split(choose_fn, test_trials)
        val_ll = float(va["avg_loglik"])
        gated_fn = wrap_choose_with_consistency_gate(choose_fn, val_ll)
        te_gated = evaluate_split(gated_fn, test_trials)

        row = {
            "participant_id": pid,
            "program": part_path.name,
            "participant_val_loglik": round(val_ll, 4),
            "n_train_trials": tr["n_trials"],
            "n_val_trials": va["n_trials"],
            "n_test_trials": te["n_trials"],
            "train_loglik": round(tr["avg_loglik"], 4),
            "val_loglik": round(va["avg_loglik"], 4),
            "test_loglik": round(te["avg_loglik"], 4),
            "new_test_loglik": round(te_gated["avg_loglik"], 4),
            "train_acc": round(tr["accuracy"], 4),
            "val_acc": round(va["accuracy"], 4),
            "test_acc": round(te["accuracy"], 4),
            "train_errors": tr["errors"],
            "val_errors": va["errors"],
            "test_errors": te["errors"],
            "gated_test_errors": te_gated["errors"],
        }
        rows.append(row)
        print(
            f"participant {pid}: train_ll={row['train_loglik']:.4f} "
            f"val_ll={row['val_loglik']:.4f} test_ll={row['test_loglik']:.4f} "
            f"new_test_ll={row['new_test_loglik']:.4f}"
        )

    if skipped:
        print(f"Skipped {len(skipped)} participant(s) without a best program: {skipped}")
    if not rows:
        raise FileNotFoundError(
            f"No evaluable participants in {run_dir} (all missing best_program*.py)"
        )

    data_rows = rows
    avg_row: Dict[str, Any] = {
        "participant_id": "avg",
        "program": "",
        "participant_val_loglik": round(_mean_finite([r["participant_val_loglik"] for r in data_rows]), 4),
        "n_train_trials": int(round(np.mean([r["n_train_trials"] for r in data_rows]))),
        "n_val_trials": int(round(np.mean([r["n_val_trials"] for r in data_rows]))),
        "n_test_trials": int(round(np.mean([r["n_test_trials"] for r in data_rows]))),
        "train_loglik": round(_mean_finite([r["train_loglik"] for r in data_rows]), 4),
        "val_loglik": round(_mean_finite([r["val_loglik"] for r in data_rows]), 4),
        "test_loglik": round(_mean_finite([r["test_loglik"] for r in data_rows]), 4),
        "new_test_loglik": round(_mean_finite([r["new_test_loglik"] for r in data_rows]), 4),
        "train_acc": round(_mean_finite([r["train_acc"] for r in data_rows]), 4),
        "val_acc": round(_mean_finite([r["val_acc"] for r in data_rows]), 4),
        "test_acc": round(_mean_finite([r["test_acc"] for r in data_rows]), 4),
        "train_errors": int(sum(r["train_errors"] for r in data_rows)),
        "val_errors": int(sum(r["val_errors"] for r in data_rows)),
        "test_errors": int(sum(r["test_errors"] for r in data_rows)),
        "gated_test_errors": int(sum(r["gated_test_errors"] for r in data_rows)),
    }
    rows.append(avg_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nPrograms -> {programs_dir}")
    print(f"Wrote {len(rows)} rows (incl. avg) -> {output_csv}")
    print(f"split_ratio={split_ratio}, split_seed={split_seed}, run_dir={run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        "--run-dir",
        dest="experiment_dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Evolution experiment folder with participant_* subdirs.",
    )
    parser.add_argument(
        "--local-dataset",
        type=Path,
        default=DEFAULT_LOCAL_DATASET,
        help="Choice13k HF dataset on disk (default: datasets/downloaded/.../Psych-101-test)",
    )
    parser.add_argument("--split-ratio", type=float, default=SPLIT_RATIO)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--programs-dir", type=Path, default=PROGRAMS_DIR)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--legacy-copy",
        action="store_true",
        help="Copy programs to --programs-dir and write the legacy wide CSV (not ad-hoc).",
    )
    parser.add_argument(
        "--gated-only",
        action="store_true",
        help="Score existing part_*.py with external consistency gate (no experiment copy).",
    )
    parser.add_argument(
        "--compare-centaur",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add centaur_test_loglik from Centaur participant_details_loglik.csv (default: on).",
    )
    parser.add_argument(
        "--centaur-loglik-csv",
        type=Path,
        default=DEFAULT_CENTAUR_LOGLIK_CSV,
        help=f"Centaur participant_details_loglik.csv (default: {DEFAULT_CENTAUR_LOGLIK_CSV.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    experiment_dir = (
        args.experiment_dir
        if args.experiment_dir.is_absolute()
        else REPO_ROOT / args.experiment_dir
    )
    local = args.local_dataset
    if local is not None:
        local = local if local.is_absolute() else REPO_ROOT / local
        if not local.exists():
            print(f"Warning: local dataset not found at {local}; will try HF download.")
            local = None
    programs_dir = args.programs_dir if args.programs_dir.is_absolute() else REPO_ROOT / args.programs_dir

    if args.gated_only:
        output_csv = args.output_csv
        if output_csv is None:
            tag = f"{args.split_ratio:.2f}".replace(".", "")
            output_csv = RESULTS_DIR / f"val_gated_split{tag}_scores.csv"
        elif not output_csv.is_absolute():
            output_csv = REPO_ROOT / output_csv
        run_gated_only(
            local_dataset=local,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            programs_dir=programs_dir,
            output_csv=output_csv,
        )
        return

    if args.legacy_copy:
        output_csv = args.output_csv
        if output_csv is None:
            output_csv = OUTPUT_CSV
        elif not output_csv.is_absolute():
            output_csv = REPO_ROOT / output_csv
        run(
            run_dir=experiment_dir,
            local_dataset=local,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            programs_dir=programs_dir,
            output_csv=output_csv,
        )
        return

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = experiment_dir / ADHOC_SUMMARY_CSV
    elif not output_csv.is_absolute():
        output_csv = REPO_ROOT / output_csv

    centaur_csv = args.centaur_loglik_csv
    if not centaur_csv.is_absolute():
        centaur_csv = REPO_ROOT / centaur_csv

    run_adhoc_experiment(
        experiment_dir=experiment_dir,
        local_dataset=local,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        output_csv=output_csv,
        compare_centaur=args.compare_centaur,
        centaur_loglik_csv=centaur_csv,
    )


if __name__ == "__main__":
    main()
