#!/usr/bin/env python3
"""
Check TEH run convergence from logged metrics (wandb_metrics.jsonl preferred).

Usage:
  python analysis/code/psych-101/check_run_convergence.py \\
    --dataset mixed_gambles \\
    --runs run_260523_001625 run_260523_001751 run_260523_010432 run_260523_010700 \\
    --tail 15 --eps 0.001
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.teh.teh_datasets import is_mixed_gambles_dataset

_GENERATED_OUTPUTS_DIR = "generated_outputs"
_WANDB_METRICS_NAME = "wandb_metrics.jsonl"
_PARTICIPANT_DIR_RE = re.compile(r"^participant_(\d+)$")
_DEFAULT_OUT_DIR = "analysis/data/mixed_gambles_convergence"

_TRAIN_LOGLIK_KEY_SUFFIXES: Tuple[str, ...] = (
    "pool_best_train_loglik",
    "train_loglik",
    "best_train_loglik",
    "iter_best_train_loglik",
    "train_fitness",
)

_VAL_LOGLIK_KEY_SUFFIXES: Tuple[str, ...] = (
    "pool_best_val_loglik",
    "val_loglik",
    "best_val_loglik",
)

_TEST_LOGLIK_KEY_SUFFIXES: Tuple[str, ...] = (
    "pool_best_test_loglik",
    "test_loglik",
    "best_test_loglik",
)

_PARTICIPANT_CSV_FIELDS = (
    "dataset",
    "run_name",
    "participant_id",
    "metric_source",
    "metric_key",
    "n_iterations",
    "final_iteration",
    "best_iteration",
    "final_train_loglik",
    "final_val_loglik",
    "final_test_loglik",
    "final_gated_test_loglik",
    "tail_converged_steps",
    "tail_window_size",
    "tail_converged",
    "improvement_last_tail",
    "best_from_fresh_candidate",
    "needs_more_iterations",
    "planned_total_iterations",
    "stopped_early",
    "error",
)

_RUN_SUMMARY_FIELDS = (
    "dataset",
    "run_name",
    "has_global_phase",
    "n_participants",
    "n_with_metrics",
    "n_missing_metrics",
    "n_incomplete",
    "avg_final_train_loglik",
    "avg_final_val_loglik",
    "avg_final_test_loglik",
    "avg_final_gated_test_loglik",
    "avg_tail_converged_steps",
    "converged_count",
    "not_converged_count",
    "convergence_rate",
    "needs_more_iterations_count",
    "needs_more_iterations_rate",
    "avg_improvement_last_tail",
    "modal_final_iteration",
    "error",
)


@dataclass
class _IterationPoint:
    iteration: int
    train_loglik: float
    val_loglik: Optional[float] = None
    test_loglik: Optional[float] = None
    best_from_fresh_candidate: Optional[str] = None


@dataclass
class _ParticipantResult:
    dataset: str
    run_name: str
    participant_id: int
    metric_source: str = ""
    metric_key: str = ""
    n_iterations: int = 0
    final_iteration: Optional[int] = None
    best_iteration: Optional[int] = None
    final_train_loglik: Optional[float] = None
    final_val_loglik: Optional[float] = None
    final_test_loglik: Optional[float] = None
    final_gated_test_loglik: Optional[float] = None
    tail_converged_steps: int = 0
    tail_window_size: int = 0
    tail_converged: bool = False
    improvement_last_tail: Optional[float] = None
    best_from_fresh_candidate: Optional[str] = None
    needs_more_iterations: bool = False
    planned_total_iterations: Optional[int] = None
    stopped_early: bool = False
    error: Optional[str] = None


@dataclass
class _RunSummary:
    dataset: str
    run_name: str
    has_global_phase: bool = False
    n_participants: int = 0
    n_with_metrics: int = 0
    n_missing_metrics: int = 0
    n_incomplete: int = 0
    avg_final_train_loglik: Optional[float] = None
    avg_final_val_loglik: Optional[float] = None
    avg_final_test_loglik: Optional[float] = None
    avg_final_gated_test_loglik: Optional[float] = None
    avg_tail_converged_steps: Optional[float] = None
    converged_count: int = 0
    not_converged_count: int = 0
    convergence_rate: Optional[float] = None
    needs_more_iterations_count: int = 0
    needs_more_iterations_rate: Optional[float] = None
    avg_improvement_last_tail: Optional[float] = None
    modal_final_iteration: Optional[int] = None
    error: Optional[str] = None


def _repo_root() -> Path:
    return _REPO_ROOT


def _teh_search_root(repo: Path, dataset: str) -> Path:
    if is_mixed_gambles_dataset(dataset):
        return repo / _GENERATED_OUTPUTS_DIR / "mixed_gambles" / "teh"
    return repo / _GENERATED_OUTPUTS_DIR / f"psych101_train" / "teh" / dataset


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


def _metric_from_row(
    row: Mapping[str, Any],
    participant_id: int,
    suffixes: Sequence[str],
) -> Tuple[Optional[float], str]:
    pid = int(participant_id)
    prefixed = [f"p{pid}_{suffix}" for suffix in suffixes]
    for key in prefixed + list(suffixes):
        val = _safe_float(row.get(key))
        if val is not None:
            return val, key
    return None, ""


def _fresh_candidate_from_row(row: Mapping[str, Any]) -> Optional[str]:
    val = row.get("best_from_fresh_candidate")
    if val is None:
        return None
    s = str(val).strip()
    return s if s and s.lower() != "null" else None


def _collect_from_wandb(participant_dir: Path, participant_id: int) -> Tuple[List[_IterationPoint], str]:
    by_iteration: Dict[int, _IterationPoint] = {}
    metric_key = ""
    sources = [
        participant_dir / _WANDB_METRICS_NAME,
        participant_dir / "refinement" / _WANDB_METRICS_NAME,
    ]
    for jsonl_path in sources:
        for row in _read_jsonl(jsonl_path):
            iteration = row.get("iteration")
            if iteration is None:
                continue
            it = int(iteration)
            if it < 1:
                continue
            train_ll, key = _metric_from_row(row, participant_id, _TRAIN_LOGLIK_KEY_SUFFIXES)
            if train_ll is None:
                continue
            val_ll, _ = _metric_from_row(row, participant_id, _VAL_LOGLIK_KEY_SUFFIXES)
            test_ll, _ = _metric_from_row(row, participant_id, _TEST_LOGLIK_KEY_SUFFIXES)
            fresh = _fresh_candidate_from_row(row)
            if key:
                metric_key = key
            by_iteration[it] = _IterationPoint(
                iteration=it,
                train_loglik=train_ll,
                val_loglik=val_ll,
                test_loglik=test_ll,
                best_from_fresh_candidate=fresh,
            )
    if not by_iteration:
        return [], metric_key
    series = [by_iteration[it] for it in sorted(by_iteration)]
    return series, metric_key


def _collect_from_evolution_csv(participant_dir: Path, run_dir: Path, participant_id: int) -> List[_IterationPoint]:
    candidates = [
        run_dir / "csvs" / f"participant_{participant_id}_evolution_loglik.csv",
        participant_dir / f"participant_{participant_id}_evolution_loglik.csv",
    ]
    for csv_path in candidates:
        if not csv_path.is_file():
            continue
        points: List[_IterationPoint] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                it_raw = row.get("iteration")
                if it_raw is None:
                    continue
                it = int(float(it_raw))
                if it < 1:
                    continue
                train_ll = _safe_float(row.get("train_loglik"))
                if train_ll is None:
                    continue
                points.append(
                    _IterationPoint(
                        iteration=it,
                        train_loglik=train_ll,
                        val_loglik=_safe_float(row.get("val_loglik")),
                        test_loglik=_safe_float(row.get("test_loglik")),
                    )
                )
        if points:
            return sorted(points, key=lambda p: p.iteration)
    return []


def _collect_from_iteration_metrics(participant_dir: Path) -> Tuple[List[_IterationPoint], Optional[int]]:
    points: List[_IterationPoint] = []
    planned_total: Optional[int] = None
    for iter_dir in sorted(participant_dir.glob("iteration_*")):
        metrics_path = iter_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        it = data.get("iteration")
        if it is None:
            m2 = re.match(r"^iteration_(\d+)$", iter_dir.name)
            it = int(m2.group(1)) if m2 else None
        if it is None or int(it) < 1:
            continue
        it = int(it)
        train_ll = _safe_float(
            data.get("pool_best_train_loglik", data.get("best_train_loglik"))
        )
        if train_ll is None:
            continue
        total = data.get("total_iterations")
        if total is not None:
            planned_total = int(total)
        fresh = data.get("best_from_fresh_candidate")
        fresh_s = None if fresh is None else str(fresh).strip() or None
        points.append(
            _IterationPoint(
                iteration=it,
                train_loglik=train_ll,
                val_loglik=_safe_float(data.get("best_val_loglik", data.get("pool_best_val_loglik"))),
                test_loglik=_safe_float(data.get("best_test_loglik", data.get("pool_best_test_loglik"))),
                best_from_fresh_candidate=fresh_s,
            )
        )
    return sorted(points, key=lambda p: p.iteration), planned_total


def _collect_iteration_series(
    participant_dir: Path, run_dir: Path, participant_id: int
) -> Tuple[List[_IterationPoint], str, Optional[int]]:
    wandb_series, metric_key = _collect_from_wandb(participant_dir, participant_id)
    if wandb_series:
        _, planned = _collect_from_iteration_metrics(participant_dir)
        return wandb_series, f"wandb_metrics.jsonl ({metric_key})", planned

    csv_series = _collect_from_evolution_csv(participant_dir, run_dir, participant_id)
    if csv_series:
        _, planned = _collect_from_iteration_metrics(participant_dir)
        return csv_series, "evolution_loglik.csv", planned

    json_series, planned = _collect_from_iteration_metrics(participant_dir)
    if json_series:
        return json_series, "iteration_*/metrics.json", planned

    return [], "", None


def _read_results_finals(participant_dir: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "final_val_loglik": None,
        "final_test_loglik": None,
        "final_gated_test_loglik": None,
    }
    path = participant_dir / "results.json"
    if not path.is_file():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    best = data.get("overall_best_train") or data.get("overall_best_test") or {}
    out["final_val_loglik"] = _safe_float(best.get("val_loglik"))
    out["final_test_loglik"] = _safe_float(best.get("test_loglik"))
    out["final_gated_test_loglik"] = _safe_float(best.get("gated_test_loglik"))
    return out


def _tail_convergence(
    series: Sequence[_IterationPoint], tail: int, eps: float
) -> Tuple[int, int, bool, Optional[float]]:
    if not series:
        return 0, 0, False, None
    tail_window = series[-min(tail, len(series)) :]
    final_ll = tail_window[-1].train_loglik
    within = sum(1 for p in tail_window if abs(p.train_loglik - final_ll) <= eps)
    tail_converged = within == len(tail_window)
    start_ll = tail_window[0].train_loglik
    improvement = final_ll - start_ll
    return within, len(tail_window), tail_converged, improvement


def _analyze_participant(
    *,
    dataset: str,
    run_name: str,
    participant_id: int,
    participant_dir: Path,
    run_dir: Path,
    tail: int,
    eps: float,
) -> _ParticipantResult:
    series, metric_source, planned_total = _collect_iteration_series(
        participant_dir, run_dir, participant_id
    )
    results_finals = _read_results_finals(participant_dir)

    if not series:
        return _ParticipantResult(
            dataset=dataset,
            run_name=run_name,
            participant_id=participant_id,
            error="no iteration metrics found",
        )

    best_point = max(series, key=lambda p: p.train_loglik)
    final_point = series[-1]
    tail_steps, tail_size, tail_converged, improvement = _tail_convergence(series, tail, eps)

    final_fresh = final_point.best_from_fresh_candidate
    if final_fresh is None:
        for p in reversed(series):
            if p.best_from_fresh_candidate is not None:
                final_fresh = p.best_from_fresh_candidate
                break

    stopped_early = False
    if planned_total is not None and final_point.iteration < planned_total:
        stopped_early = True

    needs_more = not tail_converged
    if tail_converged and improvement is not None and improvement > eps:
        if best_point.iteration >= final_point.iteration - 1:
            needs_more = True

    return _ParticipantResult(
        dataset=dataset,
        run_name=run_name,
        participant_id=participant_id,
        metric_source=metric_source,
        metric_key=metric_source,
        n_iterations=len(series),
        final_iteration=final_point.iteration,
        best_iteration=best_point.iteration,
        final_train_loglik=final_point.train_loglik,
        final_val_loglik=final_point.val_loglik or results_finals["final_val_loglik"],
        final_test_loglik=final_point.test_loglik or results_finals["final_test_loglik"],
        final_gated_test_loglik=results_finals["final_gated_test_loglik"],
        tail_converged_steps=tail_steps,
        tail_window_size=tail_size,
        tail_converged=tail_converged,
        improvement_last_tail=improvement,
        best_from_fresh_candidate=final_fresh,
        needs_more_iterations=needs_more,
        planned_total_iterations=planned_total,
        stopped_early=stopped_early,
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


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None and math.isfinite(v)]
    if not nums:
        return None
    return statistics.mean(nums)


def _modal(values: Sequence[int]) -> Optional[int]:
    if not values:
        return None
    counts: Dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: (counts[k], k))


def _analyze_run(
    repo: Path,
    dataset: str,
    run_name: str,
    *,
    tail: int,
    eps: float,
) -> Tuple[List[_ParticipantResult], _RunSummary]:
    run_dir = _teh_search_root(repo, dataset) / run_name
    if not run_dir.is_dir():
        return [], _RunSummary(
            dataset=dataset,
            run_name=run_name,
            error=f"run directory not found: {run_dir.relative_to(repo)}",
        )

    participants = _list_participant_dirs(run_dir)
    if not participants:
        return [], _RunSummary(
            dataset=dataset,
            run_name=run_name,
            error=f"no participant_* directories in {run_dir.relative_to(repo)}",
        )

    rows: List[_ParticipantResult] = []
    for pid, pdir in participants:
        rows.append(
            _analyze_participant(
                dataset=dataset,
                run_name=run_name,
                participant_id=pid,
                participant_dir=pdir,
                run_dir=run_dir,
                tail=tail,
                eps=eps,
            )
        )

    ok = [r for r in rows if r.error is None]
    missing = [r for r in rows if r.error is not None]
    converged = [r for r in ok if r.tail_converged]
    needs_more = [r for r in ok if r.needs_more_iterations]
    n_incomplete = sum(
        1 for _pid, pdir in participants if not (pdir / "results.json").is_file()
    )

    summary = _RunSummary(
        dataset=dataset,
        run_name=run_name,
        has_global_phase=(run_dir / "global_phase").is_dir(),
        n_participants=len(rows),
        n_with_metrics=len(ok),
        n_missing_metrics=len(missing),
        n_incomplete=n_incomplete,
        avg_final_train_loglik=_mean([r.final_train_loglik for r in ok]),
        avg_final_val_loglik=_mean([r.final_val_loglik for r in ok]),
        avg_final_test_loglik=_mean([r.final_test_loglik for r in ok]),
        avg_final_gated_test_loglik=_mean([r.final_gated_test_loglik for r in ok]),
        avg_tail_converged_steps=_mean([float(r.tail_converged_steps) for r in ok]),
        converged_count=len(converged),
        not_converged_count=len(ok) - len(converged),
        convergence_rate=(len(converged) / len(ok)) if ok else None,
        needs_more_iterations_count=len(needs_more),
        needs_more_iterations_rate=(len(needs_more) / len(ok)) if ok else None,
        avg_improvement_last_tail=_mean([r.improvement_last_tail for r in ok]),
        modal_final_iteration=_modal([r.final_iteration for r in ok if r.final_iteration is not None]),
    )
    return rows, summary


def _format_float(v: Optional[float], ndigits: int = 6) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.{ndigits}f}"


def _participant_to_csv(row: _ParticipantResult) -> Dict[str, str]:
    return {
        "dataset": row.dataset,
        "run_name": row.run_name,
        "participant_id": str(row.participant_id),
        "metric_source": row.metric_source,
        "metric_key": row.metric_key,
        "n_iterations": str(row.n_iterations),
        "final_iteration": "" if row.final_iteration is None else str(row.final_iteration),
        "best_iteration": "" if row.best_iteration is None else str(row.best_iteration),
        "final_train_loglik": _format_float(row.final_train_loglik),
        "final_val_loglik": _format_float(row.final_val_loglik),
        "final_test_loglik": _format_float(row.final_test_loglik),
        "final_gated_test_loglik": _format_float(row.final_gated_test_loglik),
        "tail_converged_steps": str(row.tail_converged_steps),
        "tail_window_size": str(row.tail_window_size),
        "tail_converged": str(row.tail_converged),
        "improvement_last_tail": _format_float(row.improvement_last_tail),
        "best_from_fresh_candidate": row.best_from_fresh_candidate or "",
        "needs_more_iterations": str(row.needs_more_iterations),
        "planned_total_iterations": (
            "" if row.planned_total_iterations is None else str(row.planned_total_iterations)
        ),
        "stopped_early": str(row.stopped_early),
        "error": row.error or "",
    }


def _summary_to_csv(row: _RunSummary) -> Dict[str, str]:
    return {
        "dataset": row.dataset,
        "run_name": row.run_name,
        "has_global_phase": str(row.has_global_phase),
        "n_participants": str(row.n_participants),
        "n_with_metrics": str(row.n_with_metrics),
        "n_missing_metrics": str(row.n_missing_metrics),
        "n_incomplete": str(row.n_incomplete),
        "avg_final_train_loglik": _format_float(row.avg_final_train_loglik, 4),
        "avg_final_val_loglik": _format_float(row.avg_final_val_loglik, 4),
        "avg_final_test_loglik": _format_float(row.avg_final_test_loglik, 4),
        "avg_final_gated_test_loglik": _format_float(row.avg_final_gated_test_loglik, 4),
        "avg_tail_converged_steps": _format_float(row.avg_tail_converged_steps, 2),
        "converged_count": str(row.converged_count),
        "not_converged_count": str(row.not_converged_count),
        "convergence_rate": _format_float(row.convergence_rate, 4),
        "needs_more_iterations_count": str(row.needs_more_iterations_count),
        "needs_more_iterations_rate": _format_float(row.needs_more_iterations_rate, 4),
        "avg_improvement_last_tail": _format_float(row.avg_improvement_last_tail, 6),
        "modal_final_iteration": (
            "" if row.modal_final_iteration is None else str(row.modal_final_iteration)
        ),
        "error": row.error or "",
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _write_report(
    path: Path,
    *,
    dataset: str,
    runs: Sequence[str],
    tail: int,
    eps: float,
    summaries: Sequence[_RunSummary],
    participant_rows: Sequence[_ParticipantResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("TEH run convergence report")
    lines.append(f"dataset={dataset}")
    lines.append(f"runs={', '.join(runs)}")
    lines.append(f"tail={tail}, eps={eps}")
    lines.append("")

    global_runs = [s for s in summaries if s.has_global_phase and not s.error]
    local_runs = [s for s in summaries if not s.has_global_phase and not s.error]
    if global_runs and local_runs:
        g_rate = _mean([s.convergence_rate for s in global_runs])
        l_rate = _mean([s.convergence_rate for s in local_runs])
        g_need = _mean([s.needs_more_iterations_rate for s in global_runs])
        l_need = _mean([s.needs_more_iterations_rate for s in local_runs])
        lines.append("Global vs non-global phase (from logged run artifacts):")
        lines.append(
            f"  global_phase runs ({len(global_runs)}): "
            f"avg convergence_rate={_format_float(g_rate, 4)}, "
            f"avg needs_more_rate={_format_float(g_need, 4)}"
        )
        lines.append(
            f"  non-global runs ({len(local_runs)}): "
            f"avg convergence_rate={_format_float(l_rate, 4)}, "
            f"avg needs_more_rate={_format_float(l_need, 4)}"
        )
        lines.append("")

    for summary in summaries:
        lines.append(f"Run: {summary.run_name}")
        if summary.error:
            lines.append(f"  ERROR: {summary.error}")
            lines.append("")
            continue
        status = "complete"
        if summary.n_incomplete:
            status = f"INCOMPLETE ({summary.n_incomplete} participants missing results.json)"
        elif summary.n_missing_metrics:
            status = f"partial metrics ({summary.n_missing_metrics} participants missing iteration metrics)"
        lines.append(f"  status: {status}")
        lines.append(f"  has_global_phase: {summary.has_global_phase}")
        lines.append(f"  participants: {summary.n_participants} (metrics ok: {summary.n_with_metrics})")
        lines.append(f"  avg final_train_loglik: {_format_float(summary.avg_final_train_loglik, 4)}")
        lines.append(f"  avg final_val_loglik: {_format_float(summary.avg_final_val_loglik, 4)}")
        lines.append(f"  avg final_test_loglik: {_format_float(summary.avg_final_test_loglik, 4)}")
        lines.append(
            f"  avg final_gated_test_loglik: {_format_float(summary.avg_final_gated_test_loglik, 4)}"
        )
        lines.append(
            f"  convergence: {summary.converged_count}/{summary.n_with_metrics} "
            f"({_format_float((summary.convergence_rate or 0) * 100, 1)}%)"
        )
        lines.append(
            f"  needs more iterations: {summary.needs_more_iterations_count}/{summary.n_with_metrics} "
            f"({_format_float((summary.needs_more_iterations_rate or 0) * 100, 1)}%)"
        )
        lines.append(
            f"  avg tail_converged_steps: {_format_float(summary.avg_tail_converged_steps, 2)} "
            f"(of tail={tail})"
        )
        lines.append(
            f"  avg improvement over last {tail} iterations: "
            f"{_format_float(summary.avg_improvement_last_tail, 6)}"
        )
        lines.append(f"  modal final iteration: {summary.modal_final_iteration}")
        need_pids = sorted(
            r.participant_id
            for r in participant_rows
            if r.run_name == summary.run_name and r.needs_more_iterations and r.error is None
        )
        if need_pids:
            lines.append(f"  participants needing more iterations: {need_pids}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_terminal_ranking(summaries: Sequence[_RunSummary]) -> None:
    ok = [s for s in summaries if not s.error]
    if not ok:
        print("No valid runs to rank.")
        return

    def test_score(s: _RunSummary) -> float:
        if s.avg_final_gated_test_loglik is not None:
            return s.avg_final_gated_test_loglik
        if s.avg_final_test_loglik is not None:
            return s.avg_final_test_loglik
        return float("-inf")

    print("\n=== Run ranking (logged metrics only) ===")
    print("\n1) By avg final test / gated test loglik (higher is better):")
    for i, s in enumerate(sorted(ok, key=test_score, reverse=True), 1):
        gated = _format_float(s.avg_final_gated_test_loglik, 4) or "n/a"
        test = _format_float(s.avg_final_test_loglik, 4) or "n/a"
        global_tag = " [global_phase]" if s.has_global_phase else ""
        print(
            f"  {i}. {s.run_name}{global_tag}: "
            f"gated_test={gated}, test={test}"
        )

    print("\n2) By convergence rate (tail converged within eps):")
    for i, s in enumerate(
        sorted(ok, key=lambda x: (x.convergence_rate or 0.0, test_score(x)), reverse=True),
        1,
    ):
        rate_pct = 100.0 * (s.convergence_rate or 0.0)
        print(
            f"  {i}. {s.run_name}: {s.converged_count}/{s.n_with_metrics} "
            f"({rate_pct:.1f}%) converged"
        )

    print("\n3) By need for more iterations (lower is better):")
    for i, s in enumerate(
        sorted(
            ok,
            key=lambda x: (
                x.needs_more_iterations_rate or 0.0,
                -(x.convergence_rate or 0.0),
            ),
        ),
        1,
    ):
        need_pct = 100.0 * (s.needs_more_iterations_rate or 0.0)
        verdict = "likely converged" if (s.needs_more_iterations_rate or 1.0) < 0.2 else "more iterations may help"
        print(
            f"  {i}. {s.run_name}: {s.needs_more_iterations_count}/{s.n_with_metrics} "
            f"({need_pct:.1f}%) need more — {verdict}"
        )


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(description="Check TEH run convergence from logged metrics.")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--runs", nargs="+", required=True, help="Run folder names (run_YYMMDD_HHMMSS).")
    p.add_argument("--tail", type=int, default=15, help="Tail window for convergence check.")
    p.add_argument("--eps", type=float, default=0.001, help="Tolerance for pool-best train loglik.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(_DEFAULT_OUT_DIR),
        help=f"Output directory (default: {_DEFAULT_OUT_DIR}).",
    )
    args = p.parse_args()

    if args.tail <= 0:
        raise SystemExit(f"--tail must be positive, got {args.tail}")
    if args.eps <= 0:
        raise SystemExit(f"--eps must be positive, got {args.eps}")

    dataset = args.dataset.strip()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir

    all_participants: List[_ParticipantResult] = []
    summaries: List[_RunSummary] = []

    for run_name in args.runs:
        rows, summary = _analyze_run(
            repo,
            dataset,
            run_name.strip(),
            tail=int(args.tail),
            eps=float(args.eps),
        )
        all_participants.extend(rows)
        summaries.append(summary)

    summary_csv = out_dir / "check_run_convergence_summary.csv"
    participants_csv = out_dir / "check_run_convergence_participants.csv"
    report_txt = out_dir / "check_run_convergence_report.txt"

    _write_csv(summary_csv, _RUN_SUMMARY_FIELDS, [_summary_to_csv(s) for s in summaries])
    _write_csv(
        participants_csv,
        _PARTICIPANT_CSV_FIELDS,
        [_participant_to_csv(r) for r in all_participants],
    )
    _write_report(
        report_txt,
        dataset=dataset,
        runs=[r.strip() for r in args.runs],
        tail=int(args.tail),
        eps=float(args.eps),
        summaries=summaries,
        participant_rows=all_participants,
    )

    print(f"Wrote summary -> {summary_csv}")
    print(f"Wrote participants -> {participants_csv}")
    print(f"Wrote report -> {report_txt}")
    _print_terminal_ranking(summaries)


if __name__ == "__main__":
    main()
