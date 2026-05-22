#!/usr/bin/env python3
"""
Check whether TEH training log-likelihood has converged near the end of evolution.

Usage:
  python analysis/code/psych-101/iter.py --all_in
  python analysis/code/psych-101/iter.py --dataset 1peterson2021using

Reads per-iteration training loglik from participant wandb_metrics.jsonl under the
newest TEH run_* for each dataset (same layout as analysis/code/utils/compare.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PETERSON2021USING_ALIAS,
    normalize_psych101_dataset_alias,
)
from utils.teh.teh_datasets import is_mixed_gambles_dataset

_ALL_IN_DATASETS: Tuple[str, ...] = (
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "5speekenbrink2008learning",
    "6sadeghiyeh2020temporal",
    "7hilbig2014generalized",
    "8flesch2018comparing",
    "mixed_gambles",
)

_DEFAULT_OUT = "generated_outputs/psych101_train/teh/iteration_convergence.csv"
_GENERATED_OUTPUTS_DIR = "generated_outputs"
_WANDB_METRICS_NAME = "wandb_metrics.jsonl"
_PARTICIPANT_DIR_RE = re.compile(r"^participant_(\d+)$")

# train_loglik JSON keys tried in order (first finite float wins per row)
_TRAIN_LOGLIK_KEY_SUFFIXES: Tuple[str, ...] = (
    "train_loglik",
    "best_train_loglik",
    "pool_best_train_loglik",
    "iter_best_train_loglik",
    "train_fitness",
)

_CSV_FIELDS = (
    "dataset",
    "run_name",
    "participant_id",
    "metric_key",
    "n_points",
    "first_train_loglik",
    "final_train_loglik",
    "final_improvement",
    "tail_converged_steps",
    "converged_ratio",
    "probably_enough",
)


@dataclass(frozen=True)
class _ParticipantConvergence:
    dataset: str
    run_name: str
    participant_id: int
    metric_key: str
    n_points: int
    first_train_loglik: Optional[float]
    final_train_loglik: Optional[float]
    final_improvement: Optional[float]
    tail_converged_steps: int
    converged_ratio: Optional[float]
    probably_enough: bool
    error: Optional[str] = None


@dataclass
class _DatasetSummary:
    dataset: str
    run_name: str
    n_participants: int = 0
    avg_final_train_loglik: Optional[float] = None
    avg_tail_converged_steps: Optional[float] = None
    n_probably_enough: int = 0
    pct_probably_enough: Optional[float] = None
    error: Optional[str] = None


def _repo_root() -> Path:
    return _REPO_ROOT


def _normalize_dataset(dataset: str) -> str:
    key = str(dataset).strip()
    legacy = {"choice13k": PETERSON2021USING_ALIAS, "cpc18": "2plonsky2018when"}
    if key in legacy:
        return legacy[key]
    if key == "mixed_gambles":
        return key
    return normalize_psych101_dataset_alias(key)


def _psych101_outputs_split() -> str:
    return f"psych101_{DEFAULT_PSYCH_DATASET_SPLIT}"


def _teh_search_root(repo: Path, dataset: str) -> Path:
    alias = _normalize_dataset(dataset)
    gen = repo / _GENERATED_OUTPUTS_DIR
    if is_mixed_gambles_dataset(alias):
        return gen / "mixed_gambles" / "teh"
    return gen / _psych101_outputs_split() / "teh" / alias


def _run_sort_key(path: Path) -> Tuple[str, float]:
    return path.name, path.stat().st_mtime


def _run_has_wandb_metrics(run_dir: Path) -> bool:
    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
        return False
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _PARTICIPANT_DIR_RE.match(child.name)
        if m is None:
            continue
        if (child / _WANDB_METRICS_NAME).is_file():
            return True
        if (child / "refinement" / _WANDB_METRICS_NAME).is_file():
            return True
    return False


def _auto_discover_teh_run(repo: Path, dataset: str) -> Optional[Path]:
    """Newest run_* under teh/ that has at least one participant wandb_metrics.jsonl."""
    root = _teh_search_root(repo, dataset)
    if not root.is_dir():
        return None
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and child.name.startswith("run_") and _run_has_wandb_metrics(child)
    ]
    if not candidates:
        return None
    return max(candidates, key=_run_sort_key)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isfinite(v):
        return v
    return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _train_loglik_from_row(
    row: Mapping[str, Any], participant_id: int
) -> Tuple[Optional[float], str]:
    """Return (value, metric_key_used)."""
    pid = int(participant_id)
    prefixed = [f"p{pid}_{suffix}" for suffix in _TRAIN_LOGLIK_KEY_SUFFIXES]
    global_keys = list(_TRAIN_LOGLIK_KEY_SUFFIXES)
    for key in prefixed + global_keys:
        val = _safe_float(row.get(key))
        if val is not None:
            return val, key
    return None, ""


def _collect_train_loglik_series(
    participant_dir: Path, participant_id: int
) -> Tuple[List[Tuple[int, float]], str]:
    """
    Evolution + refinement trajectories merged by iteration.

    Returns sorted (iteration, train_loglik) pairs and the metric key label used.
    """
    pid = int(participant_id)
    sources: List[Tuple[str, Path]] = [
        ("evolution", participant_dir / _WANDB_METRICS_NAME),
        ("refinement", participant_dir / "refinement" / _WANDB_METRICS_NAME),
    ]
    by_iteration: Dict[int, float] = {}
    metric_keys: List[str] = []
    phases_used: List[str] = []

    for phase, jsonl_path in sources:
        rows = _read_jsonl(jsonl_path)
        if not rows:
            continue
        phase_metric = ""
        for row in rows:
            iteration = row.get("iteration")
            if iteration is None:
                continue
            it = int(iteration)
            val, key = _train_loglik_from_row(row, pid)
            if val is None:
                continue
            by_iteration[it] = val
            if key:
                phase_metric = key
        if phase_metric:
            metric_keys.append(phase_metric)
            phases_used.append(phase)

    if not by_iteration:
        return [], ""

    series = sorted(by_iteration.items(), key=lambda t: t[0])
    if len(phases_used) == 1:
        label = metric_keys[0]
    elif len(phases_used) == 2:
        label = f"{metric_keys[0]}+{metric_keys[1]} ({'+'.join(phases_used)})"
    else:
        label = metric_keys[0] if metric_keys else ""
    return series, label


def _iteration_changes(values: Sequence[float]) -> List[float]:
    if len(values) < 2:
        return []
    return [abs(values[i] - values[i - 1]) for i in range(1, len(values))]


def _tail_converged_steps(changes: Sequence[float], threshold: float) -> int:
    count = 0
    for delta in reversed(changes):
        if delta < threshold:
            count += 1
        else:
            break
    return count


def _converged_ratio_tail_fraction(
    changes: Sequence[float], threshold: float, tail_fraction: float = 0.2
) -> Optional[float]:
    if not changes:
        return None
    n = len(changes)
    k = max(1, math.ceil(n * tail_fraction))
    tail = changes[-k:]
    return sum(1 for d in tail if d < threshold) / len(tail)


def _probably_enough(
    tail_steps: int,
    changes: Sequence[float],
    threshold: float,
    *,
    min_tail_steps: int = 3,
    tail_fraction: float = 0.2,
) -> bool:
    if tail_steps >= min_tail_steps:
        return True
    ratio = _converged_ratio_tail_fraction(changes, threshold, tail_fraction)
    return ratio is not None and ratio >= 1.0 - 1e-12


def _analyze_participant(
    *,
    dataset: str,
    run_name: str,
    participant_id: int,
    participant_dir: Path,
    threshold: float,
) -> _ParticipantConvergence:
    series, metric_key = _collect_train_loglik_series(participant_dir, participant_id)
    if not series:
        return _ParticipantConvergence(
            dataset=dataset,
            run_name=run_name,
            participant_id=participant_id,
            metric_key="",
            n_points=0,
            first_train_loglik=None,
            final_train_loglik=None,
            final_improvement=None,
            tail_converged_steps=0,
            converged_ratio=None,
            probably_enough=False,
            error="no train loglik points in wandb_metrics.jsonl",
        )

    values = [v for _, v in series]
    changes = _iteration_changes(values)
    tail_steps = _tail_converged_steps(changes, threshold)
    conv_ratio = _converged_ratio_tail_fraction(changes, threshold)
    final_improvement: Optional[float] = None
    if len(values) >= 2:
        final_improvement = values[-1] - values[-2]

    return _ParticipantConvergence(
        dataset=dataset,
        run_name=run_name,
        participant_id=participant_id,
        metric_key=metric_key,
        n_points=len(values),
        first_train_loglik=values[0],
        final_train_loglik=values[-1],
        final_improvement=final_improvement,
        tail_converged_steps=tail_steps,
        converged_ratio=conv_ratio,
        probably_enough=_probably_enough(tail_steps, changes, threshold),
    )


def _list_participant_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _PARTICIPANT_DIR_RE.match(child.name)
        if m is None:
            continue
        out.append((int(m.group(1)), child))
    return out


def _analyze_dataset(
    repo: Path,
    dataset: str,
    *,
    threshold: float,
    quiet: bool,
) -> Tuple[List[_ParticipantConvergence], _DatasetSummary]:
    alias = _normalize_dataset(dataset)
    run_dir = _auto_discover_teh_run(repo, alias)
    if run_dir is None:
        root = _teh_search_root(repo, alias)
        return [], _DatasetSummary(
            dataset=alias,
            run_name="",
            error=f"no TEH run with wandb_metrics under {root.relative_to(repo)}",
        )

    if not quiet:
        print(
            f"Auto-selected TEH for {alias}: {run_dir.relative_to(repo)} "
            f"(newest run_* in {_teh_search_root(repo, alias).relative_to(repo)})"
        )

    participants = _list_participant_dirs(run_dir)
    if not participants:
        return [], _DatasetSummary(
            dataset=alias,
            run_name=run_dir.name,
            error=f"no participant_* directories in {run_dir.relative_to(repo)}",
        )

    rows: List[_ParticipantConvergence] = []
    for pid, pdir in participants:
        rows.append(
            _analyze_participant(
                dataset=alias,
                run_name=run_dir.name,
                participant_id=pid,
                participant_dir=pdir,
                threshold=threshold,
            )
        )

    ok = [r for r in rows if r.error is None and r.final_train_loglik is not None]
    finals = [r.final_train_loglik for r in ok if r.final_train_loglik is not None]
    tails = [r.tail_converged_steps for r in ok]
    n_enough = sum(1 for r in ok if r.probably_enough)

    summary = _DatasetSummary(
        dataset=alias,
        run_name=run_dir.name,
        n_participants=len(rows),
        avg_final_train_loglik=(
            statistics.mean(finals) if finals else None
        ),
        avg_tail_converged_steps=(
            statistics.mean(tails) if tails else None
        ),
        n_probably_enough=n_enough,
        pct_probably_enough=(
            (100.0 * n_enough / len(ok)) if ok else None
        ),
    )
    return rows, summary


def _format_float(v: Optional[float], ndigits: int = 6) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.{ndigits}f}"


def _row_to_csv_dict(row: _ParticipantConvergence) -> Dict[str, str]:
    return {
        "dataset": row.dataset,
        "run_name": row.run_name,
        "participant_id": str(row.participant_id),
        "metric_key": row.metric_key,
        "n_points": str(row.n_points),
        "first_train_loglik": _format_float(row.first_train_loglik),
        "final_train_loglik": _format_float(row.final_train_loglik),
        "final_improvement": _format_float(row.final_improvement),
        "tail_converged_steps": str(row.tail_converged_steps),
        "converged_ratio": _format_float(row.converged_ratio, ndigits=4),
        "probably_enough": str(row.probably_enough),
    }


def _write_csv(path: Path, rows: Sequence[_ParticipantConvergence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_CSV_FIELDS))
        w.writeheader()
        for row in rows:
            w.writerow(_row_to_csv_dict(row))


def _print_dataset_summary(summary: _DatasetSummary) -> None:
    if summary.error:
        print(f"  {summary.dataset}: ERROR — {summary.error}")
        return
    avg_ll = summary.avg_final_train_loglik
    avg_tail = summary.avg_tail_converged_steps
    pct = summary.pct_probably_enough
    print(f"  {summary.dataset} (run {summary.run_name}, n={summary.n_participants})")
    print(f"    avg final_train_loglik: {_format_float(avg_ll, ndigits=4)}")
    print(f"    avg tail_converged_steps: {_format_float(avg_tail, ndigits=2)}")
    print(
        f"    probably_enough: {summary.n_probably_enough}/{summary.n_participants} "
        f"({pct:.1f}%)" if pct is not None else "    probably_enough: (none)"
    )


def _print_all_in_summary(summaries: Sequence[_DatasetSummary]) -> None:
    print("\n=== iter.py dataset summary ===")
    for s in summaries:
        _print_dataset_summary(s)
    ok = [s for s in summaries if not s.error and s.n_participants > 0]
    if len(ok) > 1:
        all_finals: List[float] = []
        all_tails: List[float] = []
        total_enough = 0
        total_n = 0
        for s in ok:
            if s.avg_final_train_loglik is not None:
                all_finals.append(s.avg_final_train_loglik)
            if s.avg_tail_converged_steps is not None:
                all_tails.append(s.avg_tail_converged_steps)
            total_enough += s.n_probably_enough
            total_n += s.n_participants
        print("\n  [all datasets pooled]")
        if all_finals:
            print(
                f"    mean of per-dataset avg final_train_loglik: "
                f"{_format_float(statistics.mean(all_finals), ndigits=4)}"
            )
        if all_tails:
            print(
                f"    mean of per-dataset avg tail_converged_steps: "
                f"{_format_float(statistics.mean(all_tails), ndigits=2)}"
            )
        if total_n:
            print(
                f"    probably_enough overall: {total_enough}/{total_n} "
                f"({100.0 * total_enough / total_n:.1f}%)"
            )


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description=(
            "Check TEH training log-likelihood convergence near the end of evolution "
            "from wandb_metrics.jsonl."
        )
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Psych-101 train dataset alias or mixed_gambles. "
            "Required unless --all_in is set."
        ),
    )
    p.add_argument(
        "--all_in",
        action="store_true",
        help=(
            "Run all train Psych-101 datasets plus mixed_gambles; auto-select newest "
            "TEH run_* per dataset."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Max |delta train_loglik| between consecutive iterations to count as converged.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(_DEFAULT_OUT),
        help=f"Output CSV path (default: {_DEFAULT_OUT}).",
    )
    args = p.parse_args()

    if args.threshold <= 0:
        raise SystemExit(f"--threshold must be positive, got {args.threshold}")

    if not args.all_in and args.dataset is None:
        raise SystemExit("Provide --dataset or use --all_in.")

    out_path = Path(args.out).expanduser()
    out_path = out_path.resolve() if out_path.is_absolute() else (repo / out_path).resolve()

    datasets = list(_ALL_IN_DATASETS) if args.all_in else [_normalize_dataset(args.dataset)]

    all_rows: List[_ParticipantConvergence] = []
    summaries: List[_DatasetSummary] = []

    for ds in datasets:
        alias = _normalize_dataset(ds)
        if args.all_in:
            print(f"[--all_in] {alias} ...", flush=True)
        try:
            rows, summary = _analyze_dataset(
                repo, alias, threshold=float(args.threshold), quiet=args.all_in
            )
            all_rows.extend(rows)
            summaries.append(summary)
        except (OSError, ValueError) as exc:
            msg = str(exc) or type(exc).__name__
            print(f"ERROR {alias}: {msg}", file=sys.stderr)
            summaries.append(_DatasetSummary(dataset=alias, run_name="", error=msg))

    _write_csv(out_path, all_rows)
    print(f"\nWrote {len(all_rows)} participant row(s) -> {out_path}")
    _print_all_in_summary(summaries)


if __name__ == "__main__":
    main()
