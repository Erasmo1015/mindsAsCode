#!/usr/bin/env python3
"""Copy best programs to part_X.py, evaluate train/val/test, build gated variants, report scores."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
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

SPLIT_RATIO = 0.4
SPLIT_SEED = 0
VAL_LOGLIK_THRESHOLD = -0.45
_VAL_CONST = "val_loglik"


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


def split_trials_te_dr(
    exp: Any,
    split_ratio: float = SPLIT_RATIO,
    split_seed: int = SPLIT_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match te_dr.split_trials: train/val/test by block with split_ratio as train fraction."""
    n_blocks = len(exp.blocks)
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

    train_trials = _trials_from_blocks_chronological(exp, train_blocks)
    val_trials = _trials_from_blocks_chronological(exp, val_blocks)
    test_trials = _trials_from_blocks_chronological(exp, test_blocks)
    return train_trials, val_trials, test_trials


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
            if isinstance(p_raw, bool) or (
                isinstance(p_raw, (int, np.integer)) and int(p_raw) in (0, 1)
            ):
                p_use = 1.0 if int(p_raw) == 1 else 0.0
            elif isinstance(p_raw, float):
                p_use = p_raw
            else:
                raise TypeError(f"choose must return float or 0/1, got {type(p_raw)}")
            if not (0.0 <= p_use <= 1.0):
                raise ValueError(f"invalid probability: {p_use!r}")
            p = min(max(p_use, 1e-9), 1.0 - 1e-9)
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


def _find_best_program(participant_dir: Path) -> Path:
    patterns = ["best_program*.py", "**/best_program*.py"]
    for pattern in patterns:
        matches = sorted(participant_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No best_program*.py under {participant_dir}")


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


def _choose_top_level_return_lines(source: str) -> List[int]:
    """Line numbers of return statements directly in choose() (not nested helpers)."""
    tree = ast.parse(source)
    choose_fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "choose"),
        None,
    )
    if choose_fn is None:
        raise ValueError("No choose(problem, history) function found.")

    def walk(stmts: List[ast.stmt]) -> List[int]:
        out: List[int] = []
        for stmt in stmts:
            if isinstance(stmt, ast.FunctionDef):
                continue
            if isinstance(stmt, ast.Return) and stmt.lineno is not None:
                out.append(int(stmt.lineno))
            elif isinstance(stmt, ast.If):
                out.extend(walk(stmt.body))
                out.extend(walk(stmt.orelse))
            elif isinstance(stmt, (ast.For, ast.While, ast.With)):
                out.extend(walk(stmt.body))
                out.extend(walk(getattr(stmt, "orelse", [])))
        return out

    return walk(choose_fn.body)


def inject_consistency_gate(source: str, val_loglik: float) -> str:
    """Inject consistency gate before each top-level return in choose()."""
    return_lines = set(_choose_top_level_return_lines(source))
    if not return_lines:
        raise ValueError("No return statement found in choose().")

    header = f"{_VAL_CONST} = {float(val_loglik)!r}\n\n"
    out_lines: List[str] = []
    for i, line in enumerate(source.splitlines(), start=1):
        if i not in return_lines:
            out_lines.append(line)
            continue
        stripped = line.strip()
        if not stripped.startswith("return "):
            out_lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        expr = stripped[len("return ") :]
        out_lines.append(f"{indent}p = {expr}")
        out_lines.append(f"{indent}if {_VAL_CONST} < -0.45:")
        out_lines.append(f"{indent}    consistency = max(0.0, 1.0 - (-0.45 - {_VAL_CONST}))")
        out_lines.append(f"{indent}else:")
        out_lines.append(f"{indent}    consistency = 1.0")
        out_lines.append(f"{indent}p = consistency * p + (1.0 - consistency) * 0.5")
        out_lines.append(f"{indent}return p")

    body = "\n".join(out_lines).rstrip() + "\n"
    if body.lstrip().startswith(f"{_VAL_CONST} ="):
        return body
    return header + body


def _part_paths(programs_dir: Path, pid: int) -> Tuple[Path, Path]:
    base = programs_dir / f"part_{pid}.py"
    gated = programs_dir / f"part_{pid}_gated.py"
    return base, gated


def _discover_gated_programs(programs_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    pat = re.compile(r"^part_(\d+)_gated\.py$")
    for p in sorted(programs_dir.glob("part_*_gated.py")):
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
    """Original part_X train/val/test loglik + gated test-only (new_test_loglik) at given split."""
    from data_modules.choice13k import get_choice13k_experiments

    gated_programs = _discover_gated_programs(programs_dir)
    if not gated_programs:
        raise FileNotFoundError(f"No part_*_gated.py files in {programs_dir}")

    max_pid = max(pid for pid, _ in gated_programs)
    local = str(local_dataset) if local_dataset is not None else None
    experiments = get_choice13k_experiments(n_participants=max_pid + 1, local_dataset=local)

    fieldnames = [
        "participant_id",
        "program",
        "gated_program",
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

    for pid, gated_path in gated_programs:
        if pid >= len(experiments):
            raise IndexError(f"participant {pid} not in dataset (only {len(experiments)} experiments)")

        part_path, _ = _part_paths(programs_dir, pid)
        if not part_path.is_file():
            raise FileNotFoundError(f"Missing original program: {part_path}")

        train_trials, val_trials, test_trials = split_trials_te_dr(
            experiments[pid], split_ratio=split_ratio, split_seed=split_seed
        )
        choose_fn = _load_choose_fn(part_path)
        tr = evaluate_split(choose_fn, train_trials)
        va = evaluate_split(choose_fn, val_trials)
        te = evaluate_split(choose_fn, test_trials)
        te_gated = evaluate_split(_load_choose_fn(gated_path), test_trials)

        row = {
            "participant_id": pid,
            "program": part_path.name,
            "gated_program": gated_path.name,
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
        "gated_program": "",
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


def _read_baked_val_loglik(gated_path: Path) -> float:
    """Parse module-level val_loglik constant from a gated program file."""
    for line in gated_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("val_loglik ="):
            return float(stripped.split("=", 1)[1].strip())
    return float("nan")


def run(
    run_dir: Path,
    local_dataset: Optional[Path],
    split_ratio: float,
    split_seed: int,
    programs_dir: Path,
    output_csv: Path,
    use_existing_gated: bool = False,
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
        "gated_program",
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

    for pid, pdir in participants:
        if pid >= len(experiments):
            raise IndexError(f"participant {pid} not in dataset (only {len(experiments)} experiments)")

        src_path = _find_best_program(pdir)
        part_path, gated_path = _part_paths(programs_dir, pid)
        part_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")

        train_trials, val_trials, test_trials = split_trials_te_dr(
            experiments[pid], split_ratio=split_ratio, split_seed=split_seed
        )

        choose_fn = _load_choose_fn(part_path)
        tr = evaluate_split(choose_fn, train_trials)
        va = evaluate_split(choose_fn, val_trials)
        te = evaluate_split(choose_fn, test_trials)

        val_ll = float(va["avg_loglik"])
        if use_existing_gated:
            if not gated_path.is_file():
                raise FileNotFoundError(
                    f"--use-existing-gated: missing {gated_path.name} for participant {pid}"
                )
        else:
            gated_source = inject_consistency_gate(
                part_path.read_text(encoding="utf-8"), val_ll
            )
            gated_path.write_text(gated_source, encoding="utf-8")

        gated_fn = _load_choose_fn(gated_path)
        te_gated = evaluate_split(gated_fn, test_trials)

        row = {
            "participant_id": pid,
            "program": part_path.name,
            "gated_program": gated_path.name,
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

    data_rows = rows
    avg_row: Dict[str, Any] = {
        "participant_id": "avg",
        "program": "",
        "gated_program": "",
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
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Evolution run root (default: {DEFAULT_RUN_DIR.relative_to(REPO_ROOT)})",
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
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--use-existing-gated",
        action="store_true",
        help="Score part_*_gated.py as-is; do not inject consistency gate before return.",
    )
    parser.add_argument(
        "--gated-only",
        action="store_true",
        help="Only score existing part_*_gated.py on train/val/test (no copy/inject).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    local = args.local_dataset
    if local is not None:
        local = local if local.is_absolute() else REPO_ROOT / local
        if not local.exists():
            print(f"Warning: local dataset not found at {local}; will try HF download.")
            local = None
    programs_dir = args.programs_dir if args.programs_dir.is_absolute() else REPO_ROOT / args.programs_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else REPO_ROOT / args.output_csv

    if args.gated_only:
        if args.output_csv == OUTPUT_CSV:
            tag = f"{args.split_ratio:.2f}".replace(".", "")
            output_csv = RESULTS_DIR / f"val_gated_split{tag}_scores.csv"
        run_gated_only(
            local_dataset=local,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            programs_dir=programs_dir,
            output_csv=output_csv,
        )
        return

    run(
        run_dir=run_dir,
        local_dataset=local,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        programs_dir=programs_dir,
        output_csv=output_csv,
        use_existing_gated=args.use_existing_gated,
    )


if __name__ == "__main__":
    main()
