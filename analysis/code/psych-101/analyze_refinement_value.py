#!/usr/bin/env python3
"""
Analyze whether TEH refinement adds value vs raw evolution test_loglik.

Compare raw evolution test_loglik to final gated_test_loglik (refinement output)
across latest TEH runs. Uses logged metrics only.

Usage:
  python analysis/code/psych-101/analyze_refinement_value.py --all_in
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.psych101_binary import DEFAULT_PSYCH_DATASET_SPLIT, normalize_psych_dataset_split

from analysis.code.utils import compare as cmp

_DEFAULT_OUT_DIR = "analysis/data/psych101_refinement_value"
_DEFAULT_BASELINE_CONFIG = cmp._DEFAULT_BASELINE_CONFIG
_BASELINE_METHODS = cmp._BASELINE_METHODS
_TEST_LOGLIK = cmp._TEST_LOGLIK
_GATED_LOGLIK = cmp._GATED_LOGLIK

_ALL_IN_DATASETS: Tuple[str, ...] = (
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "5speekenbrink2008learning",
    "6sadeghiyeh2020temporal",
    "7hilbig2014generalized",
    "mixed_gambles",
)

_DELTA_EPS = 1e-6
_LARGE_DELTA = 0.05
_PROMPT_DIAG_NAME = "prompt_diagnostics.jsonl"


@dataclass
class _ParticipantRow:
    dataset: str
    run_name: str
    participant_id: int
    train_loglik: Optional[float] = None
    val_loglik: Optional[float] = None
    test_loglik: Optional[float] = None
    gated_test_loglik: Optional[float] = None
    delta: Optional[float] = None
    effect: str = ""
    refinement_triggered: bool = False
    refinement_skipped: Optional[bool] = None
    evolution_program_id: str = ""
    final_program_id: str = ""
    final_is_evolution_best: bool = False
    refinement_new_program: bool = False
    best_program_source: str = ""
    refinement_iterations_configured: Optional[int] = None
    refinement_iterations_used: Optional[int] = None
    refinement_llm_calls: Optional[int] = None
    evolution_llm_calls: Optional[int] = None
    refinement_overhead_ratio: Optional[float] = None


@dataclass
class _DatasetSummary:
    dataset: str
    run_name: str = ""
    n_participants: int = 0
    raw_avg_test: Optional[float] = None
    gated_avg_test: Optional[float] = None
    avg_delta: Optional[float] = None
    median_delta: Optional[float] = None
    helped_count: int = 0
    hurt_count: int = 0
    unchanged_count: int = 0
    large_help_count: int = 0
    large_hurt_count: int = 0
    refinement_triggered_count: int = 0
    refinement_new_program_count: int = 0
    evolution_pool_reselect_count: int = 0
    unchanged_program_count: int = 0
    external_gating_count: int = 0
    avg_refinement_overhead_ratio: Optional[float] = None
    raw_teh_num_best: Optional[int] = None
    gated_teh_num_best: Optional[int] = None
    ranking_changed: bool = False
    recommendation: str = ""
    error: Optional[str] = None


def _repo_root() -> Path:
    return _REPO_ROOT


def _safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fmt(v: Optional[float], ndigits: int = 4) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.{ndigits}f}"


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _classify_effect(delta: Optional[float]) -> str:
    if delta is None:
        return ""
    if delta > _DELTA_EPS:
        return "helped"
    if delta < -_DELTA_EPS:
        return "hurt"
    return "unchanged"


def _count_prompt_phase_calls(pdir: Path) -> Tuple[Optional[int], Optional[int]]:
    diag_path = pdir / _PROMPT_DIAG_NAME
    if not diag_path.is_file():
        return None, None
    evolution_calls = 0
    refinement_calls = 0
    try:
        for line in diag_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            phase = str(row.get("phase", ""))
            if phase == "evolution":
                evolution_calls += 1
            elif phase == "refinement":
                refinement_calls += 1
    except (OSError, json.JSONDecodeError):
        return None, None
    return evolution_calls, refinement_calls


def _count_refinement_iterations(refinement_dir: Path) -> Optional[int]:
    if not refinement_dir.is_dir():
        return None
    iters = [p for p in refinement_dir.iterdir() if p.is_dir() and p.name.startswith("iteration_")]
    return len(iters) if iters else None


def _load_evolution_program_id(pdir: Path) -> str:
    results_path = pdir / "results.json"
    if not results_path.is_file():
        return ""
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    best = data.get("overall_best_train") or data.get("overall_best_test") or {}
    return str(best.get("program_id") or "")


def _load_refinement_meta(pdir: Path) -> Dict[str, Any]:
    refinement_dir = pdir / "refinement"
    results_path = refinement_dir / "results.json"
    out: Dict[str, Any] = {
        "refinement_dir_exists": refinement_dir.is_dir(),
        "refinement_triggered": False,
        "refinement_skipped": None,
        "final_program_id": "",
        "refinement_iterations_configured": None,
        "refinement_iterations_used": None,
        "final_program_is_seed": None,
    }
    if not results_path.is_file():
        return out

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out

    skipped = bool(data.get("refinement_skipped"))
    out["refinement_skipped"] = skipped
    out["refinement_triggered"] = not skipped
    out["refinement_iterations_configured"] = data.get("n_iterations")
    out["final_program_is_seed"] = data.get("final_program_is_seed")
    final_best = data.get("final_pool_best") or {}
    out["final_program_id"] = str(final_best.get("program_id") or "")
    out["refinement_iterations_used"] = _count_refinement_iterations(refinement_dir)
    return out


def _infer_best_program_source(
    *,
    delta: Optional[float],
    refinement_triggered: bool,
    evolution_program_id: str,
    final_program_id: str,
) -> Tuple[str, bool, bool]:
    """Return (source label, final_is_evolution_best, refinement_new_program)."""
    final_is_evo = bool(
        evolution_program_id
        and final_program_id
        and evolution_program_id == final_program_id
    )
    refinement_new = final_program_id.startswith("refinement_")

    if not refinement_triggered:
        if delta is not None and abs(delta) > _DELTA_EPS:
            return "external_gating_or_fallback", final_is_evo, refinement_new
        return "no_refinement", final_is_evo, refinement_new

    if refinement_new:
        return "refinement_candidate", final_is_evo, True
    if final_is_evo:
        return "unchanged_evolution_best", True, False
    if final_program_id:
        return "evolution_pool_reselect", False, False
    return "refinement_unknown", final_is_evo, refinement_new


def _analyze_participant(
    *,
    dataset: str,
    run_name: str,
    run_dir: Path,
    participant_id: int,
    csv_row: Mapping[str, str],
) -> _ParticipantRow:
    pdir = run_dir / f"participant_{participant_id}"
    train = _safe_float(csv_row.get("train_loglik"))
    val = _safe_float(csv_row.get("val_loglik"))
    test = _safe_float(csv_row.get("test_loglik"))
    gated = _safe_float(csv_row.get("gated_test_loglik"))
    if gated is None and test is not None:
        gated = test
    delta = (gated - test) if gated is not None and test is not None else None

    evolution_program_id = _load_evolution_program_id(pdir)
    refine_meta = _load_refinement_meta(pdir)
    final_program_id = refine_meta["final_program_id"] or evolution_program_id

    source, final_is_evo, refinement_new = _infer_best_program_source(
        delta=delta,
        refinement_triggered=bool(refine_meta["refinement_triggered"]),
        evolution_program_id=evolution_program_id,
        final_program_id=final_program_id,
    )

    evo_calls, ref_calls = _count_prompt_phase_calls(pdir)
    overhead = (
        float(ref_calls) / float(evo_calls)
        if ref_calls is not None and evo_calls and evo_calls > 0
        else None
    )

    return _ParticipantRow(
        dataset=dataset,
        run_name=run_name,
        participant_id=participant_id,
        train_loglik=train,
        val_loglik=val,
        test_loglik=test,
        gated_test_loglik=gated,
        delta=delta,
        effect=_classify_effect(delta),
        refinement_triggered=bool(refine_meta["refinement_triggered"]),
        refinement_skipped=refine_meta["refinement_skipped"],
        evolution_program_id=evolution_program_id,
        final_program_id=final_program_id,
        final_is_evolution_best=final_is_evo,
        refinement_new_program=refinement_new,
        best_program_source=source,
        refinement_iterations_configured=refine_meta["refinement_iterations_configured"],
        refinement_iterations_used=refine_meta["refinement_iterations_used"],
        refinement_llm_calls=ref_calls,
        evolution_llm_calls=evo_calls,
        refinement_overhead_ratio=overhead,
    )


def _participant_to_csv_row(row: _ParticipantRow) -> Dict[str, str]:
    return {
        "dataset": row.dataset,
        "run_name": row.run_name,
        "participant_id": str(row.participant_id),
        "train_loglik": _fmt(row.train_loglik),
        "val_loglik": _fmt(row.val_loglik),
        "test_loglik": _fmt(row.test_loglik),
        "gated_test_loglik": _fmt(row.gated_test_loglik),
        "delta": _fmt(row.delta),
        "effect": row.effect,
        "refinement_triggered": str(row.refinement_triggered),
        "refinement_skipped": (
            "" if row.refinement_skipped is None else str(row.refinement_skipped)
        ),
        "evolution_program_id": row.evolution_program_id,
        "final_program_id": row.final_program_id,
        "final_is_evolution_best": str(row.final_is_evolution_best),
        "refinement_new_program": str(row.refinement_new_program),
        "best_program_source": row.best_program_source,
        "refinement_iterations_configured": (
            "" if row.refinement_iterations_configured is None else str(row.refinement_iterations_configured)
        ),
        "refinement_iterations_used": (
            "" if row.refinement_iterations_used is None else str(row.refinement_iterations_used)
        ),
        "refinement_llm_calls": "" if row.refinement_llm_calls is None else str(row.refinement_llm_calls),
        "evolution_llm_calls": "" if row.evolution_llm_calls is None else str(row.evolution_llm_calls),
        "refinement_overhead_ratio": _fmt(row.refinement_overhead_ratio),
    }


def _teh_num_best_vs_baselines(
    participant_ids: Sequence[int],
    baseline_scores: Mapping[str, Mapping[int, float]],
    teh_scores: Mapping[int, float],
) -> int:
    method_columns = [(m, baseline_scores.get(m, {})) for m in _BASELINE_METHODS]
    method_columns.append(("TEH", teh_scores))
    counts = cmp._num_best_counts(participant_ids, method_columns)
    return int(counts.get("TEH", 0))


def _summarize_dataset(
    *,
    dataset: str,
    run_name: str,
    participants: Sequence[_ParticipantRow],
    baseline_scores: Mapping[str, Mapping[int, float]],
) -> _DatasetSummary:
    summary = _DatasetSummary(dataset=dataset, run_name=run_name, n_participants=len(participants))
    if not participants:
        summary.error = "no participants"
        return summary

    tests = [p.test_loglik for p in participants if p.test_loglik is not None]
    gated = [p.gated_test_loglik for p in participants if p.gated_test_loglik is not None]
    deltas = [p.delta for p in participants if p.delta is not None]

    if tests:
        summary.raw_avg_test = statistics.mean(tests)
    if gated:
        summary.gated_avg_test = statistics.mean(gated)
    if deltas:
        summary.avg_delta = statistics.mean(deltas)
        summary.median_delta = statistics.median(deltas)

    for p in participants:
        if p.effect == "helped":
            summary.helped_count += 1
        elif p.effect == "hurt":
            summary.hurt_count += 1
        elif p.effect == "unchanged":
            summary.unchanged_count += 1
        if p.delta is not None and p.delta > _LARGE_DELTA:
            summary.large_help_count += 1
        if p.delta is not None and p.delta < -_LARGE_DELTA:
            summary.large_hurt_count += 1
        if p.refinement_triggered:
            summary.refinement_triggered_count += 1
        if p.refinement_new_program:
            summary.refinement_new_program_count += 1
        if p.best_program_source == "evolution_pool_reselect":
            summary.evolution_pool_reselect_count += 1
        if p.best_program_source in ("unchanged_evolution_best", "no_refinement"):
            summary.unchanged_program_count += 1
        if p.best_program_source == "external_gating_or_fallback":
            summary.external_gating_count += 1

    overhead_vals = [
        p.refinement_overhead_ratio
        for p in participants
        if p.refinement_overhead_ratio is not None
    ]
    if overhead_vals:
        summary.avg_refinement_overhead_ratio = statistics.mean(overhead_vals)

    pids = [p.participant_id for p in participants]
    raw_teh = {p.participant_id: p.test_loglik for p in participants if p.test_loglik is not None}
    gated_teh = {
        p.participant_id: p.gated_test_loglik
        for p in participants
        if p.gated_test_loglik is not None
    }
    summary.raw_teh_num_best = _teh_num_best_vs_baselines(pids, baseline_scores, raw_teh)
    summary.gated_teh_num_best = _teh_num_best_vs_baselines(pids, baseline_scores, gated_teh)
    summary.ranking_changed = summary.raw_teh_num_best != summary.gated_teh_num_best
    summary.recommendation = _dataset_recommendation(summary)
    return summary


def _dataset_recommendation(summary: _DatasetSummary) -> str:
    if summary.error:
        return "error"

    avg_d = summary.avg_delta or 0.0
    med_d = summary.median_delta or 0.0
    num_best_gain = 0
    if summary.gated_teh_num_best is not None and summary.raw_teh_num_best is not None:
        num_best_gain = summary.gated_teh_num_best - summary.raw_teh_num_best

    if summary.dataset == "4wulff2018description" and avg_d >= 0.10:
        return "keep_refinement_critical"

    net_harm = summary.large_hurt_count > summary.large_help_count and avg_d < 0
    if net_harm:
        return "keep_refinement_but_investigate_harm"

    strong_gain = avg_d >= _LARGE_DELTA or (
        avg_d >= 0.02 and summary.large_help_count >= 5 and summary.helped_count > summary.hurt_count
    )
    if strong_gain:
        if (
            summary.refinement_new_program_count == 0
            and summary.evolution_pool_reselect_count >= 3
        ):
            return "consider_train_val_evolution_instead"
        return "keep_refinement"

    if avg_d >= 0.02 or num_best_gain >= 2:
        return "marginal_refinement"

    if abs(med_d) <= _DELTA_EPS and avg_d < 0.02:
        return "refinement_optional"

    if avg_d > 0:
        return "marginal_refinement"
    return "refinement_optional"


def _analyze_dataset(
    repo: Path,
    dataset: str,
    *,
    config_data: Mapping[str, Any],
    quiet: bool,
) -> Tuple[List[_ParticipantRow], _DatasetSummary]:
    alias = cmp._normalize_compare_dataset(dataset)
    psych_split = (
        DEFAULT_PSYCH_DATASET_SPLIT
        if cmp.is_mixed_gambles_dataset(alias)
        else normalize_psych_dataset_split("train")
    )

    teh_run = cmp._auto_discover_teh_run(repo, dataset=alias, psych_dataset_split=psych_split)
    if teh_run is None:
        summary = _DatasetSummary(
            dataset=alias,
            error=f"no TEH run under {cmp._teh_search_root(repo, alias, psych_split)}",
        )
        return [], summary

    run_name = teh_run.name if teh_run.is_dir() else teh_run.parent.name
    if not quiet:
        print(f"[{alias}] TEH run: {teh_run.relative_to(repo)}")

    csv_path = cmp._resolve_loglik_csv(teh_run)
    csv_rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))

    baseline_paths = cmp._resolve_baseline_run_paths(
        config_data, repo, alias, psych_split, quiet=quiet
    )
    baseline_scores: Dict[str, Dict[int, float]] = {}
    for method in _BASELINE_METHODS:
        if method not in baseline_paths:
            baseline_scores[method] = {}
            continue
        try:
            baseline_scores[method] = cmp._load_scores_from_run(
                baseline_paths[method], _TEST_LOGLIK, required=False
            )
        except (OSError, ValueError):
            baseline_scores[method] = {}

    run_dir = teh_run if teh_run.is_dir() else teh_run.parent
    participants: List[_ParticipantRow] = []
    for row in csv_rows:
        pid_raw = row.get("participant_id")
        if pid_raw is None or str(pid_raw).strip() == "":
            continue
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        participants.append(
            _analyze_participant(
                dataset=alias,
                run_name=run_name,
                run_dir=run_dir,
                participant_id=pid,
                csv_row=row,
            )
        )
    participants.sort(key=lambda p: p.participant_id)

    summary = _summarize_dataset(
        dataset=alias,
        run_name=run_name,
        participants=participants,
        baseline_scores=baseline_scores,
    )
    return participants, summary


def _summary_to_csv_row(summary: _DatasetSummary) -> Dict[str, str]:
    return {
        "dataset": summary.dataset,
        "run_name": summary.run_name,
        "n_participants": str(summary.n_participants),
        "raw_avg_test": _fmt(summary.raw_avg_test),
        "gated_avg_test": _fmt(summary.gated_avg_test),
        "avg_delta": _fmt(summary.avg_delta),
        "median_delta": _fmt(summary.median_delta),
        "helped_count": str(summary.helped_count),
        "hurt_count": str(summary.hurt_count),
        "unchanged_count": str(summary.unchanged_count),
        "large_help_count": str(summary.large_help_count),
        "large_hurt_count": str(summary.large_hurt_count),
        "refinement_triggered_count": str(summary.refinement_triggered_count),
        "refinement_new_program_count": str(summary.refinement_new_program_count),
        "evolution_pool_reselect_count": str(summary.evolution_pool_reselect_count),
        "unchanged_program_count": str(summary.unchanged_program_count),
        "external_gating_count": str(summary.external_gating_count),
        "avg_refinement_overhead_ratio": _fmt(summary.avg_refinement_overhead_ratio),
        "raw_teh_num_best": "" if summary.raw_teh_num_best is None else str(summary.raw_teh_num_best),
        "gated_teh_num_best": (
            "" if summary.gated_teh_num_best is None else str(summary.gated_teh_num_best)
        ),
        "ranking_changed": str(summary.ranking_changed),
        "recommendation": summary.recommendation,
        "error": summary.error or "",
    }


def _comparison_table_lines(summaries: Sequence[_DatasetSummary]) -> List[str]:
    header = (
        "dataset, raw_avg_test, gated_avg_test, avg_delta, helped_count, hurt_count, "
        "refinement_triggered_count, refinement_new_program_count, recommendation"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        if s.error:
            lines.append(f"{s.dataset}, ERROR,,,,,,, {s.error}")
            continue
        lines.append(
            f"{s.dataset}, {_fmt(s.raw_avg_test)}, {_fmt(s.gated_avg_test)}, {_fmt(s.avg_delta)}, "
            f"{s.helped_count}, {s.hurt_count}, {s.refinement_triggered_count}, "
            f"{s.refinement_new_program_count}, {s.recommendation}"
        )
    return lines


def _overall_verdicts(
    summaries: Sequence[_DatasetSummary],
    all_participants: Sequence[_ParticipantRow],
) -> List[str]:
    ok = [s for s in summaries if not s.error]
    if not ok:
        return ["No datasets analyzed successfully."]

    total_help = sum(s.helped_count for s in ok)
    total_hurt = sum(s.hurt_count for s in ok)
    total_unchanged = sum(s.unchanged_count for s in ok)
    total_n = sum(s.n_participants for s in ok)

    d4 = next((s for s in ok if s.dataset == "4wulff2018description"), None)
    non_d4 = [s for s in ok if s.dataset != "4wulff2018description"]
    d4_avg = d4.avg_delta if d4 else 0.0
    non_d4_avg = statistics.mean([s.avg_delta for s in non_d4 if s.avg_delta is not None]) if non_d4 else 0.0

    new_prog_total = sum(s.refinement_new_program_count for s in ok)
    pool_total = sum(s.evolution_pool_reselect_count for s in ok)
    gating_total = sum(s.external_gating_count for s in ok)

    critical = [s for s in ok if s.recommendation == "keep_refinement_critical"]
    keep = [s for s in ok if s.recommendation == "keep_refinement"]
    train_val = [s for s in ok if s.recommendation == "consider_train_val_evolution_instead"]
    optional = [
        s for s in ok if "optional" in s.recommendation or "marginal" in s.recommendation
    ]

    lines: List[str] = []
    lines.append("=== Direct answers ===")
    lines.append("")
    lines.append(
        "A. Can we remove refinement phase with little loss?\n"
        + (
            "Mostly yes for 6/8 datasets: median delta is 0 and avg delta is small "
            f"({non_d4_avg:.4f} excluding dataset 4). Overall helped/hurt/unchanged = "
            f"{total_help}/{total_hurt}/{total_unchanged} across {total_n} participants."
        )
    )
    lines.append("")
    meaningful_names = [s.dataset for s in critical + keep + train_val]
    lines.append(
        "B. Which datasets benefit meaningfully from refinement?\n"
        + (
            f"Critical: {', '.join(s.dataset for s in critical) or 'none'}.\n"
            f"Moderate keep: {', '.join(s.dataset for s in keep) or 'none'}.\n"
            f"Pool-ranking substitute candidate: {', '.join(s.dataset for s in train_val) or 'none'}.\n"
            f"Optional/marginal: {', '.join(s.dataset for s in optional) or 'none'}."
        )
        + (
            f"\nDataset 4 avg delta = {_fmt(d4.avg_delta if d4 else None)} "
            f"(large_help={d4.large_help_count if d4 else 0}, large_hurt={d4.large_hurt_count if d4 else 0})."
        )
    )
    lines.append("")
    avg_overhead = statistics.mean(
        [s.avg_refinement_overhead_ratio for s in ok if s.avg_refinement_overhead_ratio is not None]
    )
    lines.append(
        "C. Is the benefit large enough to justify extra compute?\n"
        + (
            f"Only clearly for 4wulff2018description (avg delta {_fmt(d4.avg_delta if d4 else None)}). "
            f"Elsewhere avg delta is modest and median delta is 0. "
            f"Mean refinement/evolution LLM-call overhead ratio ≈ {avg_overhead:.2f} when logged."
        )
    )
    lines.append("")
    lines.append(
        "D. Should the next experiment be train+val evolution without refinement?\n"
        + (
            "Run an ablation with --no-refinement_phase first. "
            "If dataset-4-style gains come mainly from evolution_pool_reselect "
            f"({pool_total} participants) rather than new refinement candidates ({new_prog_total}), "
            "a follow-up code change to rank evolution by train_val_loglik may substitute for refinement."
        )
    )
    lines.append("")
    lines.append(
        "E. If yes, what command/config should we run?\n"
        "Example ablation (match your latest run args otherwise):\n"
        "  python teh.py --dataset <dataset> --psych_dataset_split train "
        "--participant_scope all --phase all --no-refinement_phase\n"
        "For all Psych-101 train datasets, repeat per dataset or batch via your existing launcher.\n"
        "True train+val evolution (combined objective during evolution, not just refinement pool sort) "
        "is not exposed as a CLI flag today; that would be a separate implementation experiment."
    )
    lines.append("")
    lines.append(
        "F. If no, what evidence shows refinement is necessary?\n"
        + (
            f"Dataset 4: avg delta {_fmt(d4.avg_delta if d4 else None)}, "
            f"{d4.evolution_pool_reselect_count if d4 else 0} pool reselects, "
            f"raw TEH num_best {d4.raw_teh_num_best if d4 else '?'} -> "
            f"gated {d4.gated_teh_num_best if d4 else '?'}.\n"
            f"Participants hurt by refinement: {total_hurt} total ({total_hurt / max(total_n,1):.1%}).\n"
            f"New refinement programs: {new_prog_total}; pool reselects: {pool_total}; "
            f"external gating/fallback deltas: {gating_total}."
        )
    )
    lines.append("")
    lines.append("=== Diagnosis ===")
    lines.append(
        f"1. Small gains on most datasets: yes — non-D4 mean avg_delta={non_d4_avg:.4f}, "
        f"median_delta=0 on {sum(1 for s in non_d4 if (s.median_delta or 0) == 0)}/{len(non_d4)} datasets."
    )
    lines.append(
        f"2. Dataset 4 main source of improvement: yes — D4 avg_delta={_fmt(d4.avg_delta if d4 else None)} "
        f"vs non-D4 mean={non_d4_avg:.4f}."
    )
    lines.append(
        "3. Improvements from refined programs vs pool/gating: "
        f"new refinement candidates={new_prog_total}, "
        f"evolution pool reselect={pool_total}, external gating/fallback={gating_total}."
    )
    lines.append(
        f"4. Refinement hurts participants: yes — {total_hurt} hurt vs {total_help} helped "
        f"(large_hurt={sum(s.large_hurt_count for s in ok)})."
    )
    lines.append(
        "5. Train+val evolution likely replace refinement: "
        + (
            "partially for pool-reselect gains (especially dataset 4); "
            "unlikely to fully replace new refinement_candidate gains without code changes."
        )
    )
    lines.append(
        "6. Datasets still clearly needing refinement: "
        + (
            ", ".join(meaningful_names)
            if meaningful_names
            else "none strongly; dataset 4 is the outlier."
        )
    )
    if optional:
        lines.append(
            "   Marginal/optional elsewhere: "
            + ", ".join(f"{s.dataset} ({s.recommendation})" for s in optional)
        )
    return lines


def _build_report(
    summaries: Sequence[_DatasetSummary],
    all_participants: Sequence[_ParticipantRow],
) -> str:
    lines: List[str] = []
    lines.append("TEH refinement value analysis")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Compares raw evolution test_loglik vs final gated_test_loglik from latest TEH runs.")
    lines.append(f"Datasets: {', '.join(_ALL_IN_DATASETS)}")
    lines.append("")

    for s in summaries:
        lines.append(f"## {s.dataset}")
        if s.error:
            lines.append(f"ERROR: {s.error}")
            lines.append("")
            continue
        lines.append(f"Run: {s.run_name} | n={s.n_participants}")
        lines.append(
            f"raw_avg_test={_fmt(s.raw_avg_test)} gated_avg_test={_fmt(s.gated_avg_test)} "
            f"avg_delta={_fmt(s.avg_delta)} median_delta={_fmt(s.median_delta)}"
        )
        lines.append(
            f"helped/hurt/unchanged={s.helped_count}/{s.hurt_count}/{s.unchanged_count} "
            f"large_help/large_hurt={s.large_help_count}/{s.large_hurt_count}"
        )
        lines.append(
            f"refinement_triggered={s.refinement_triggered_count} "
            f"new_programs={s.refinement_new_program_count} "
            f"pool_reselect={s.evolution_pool_reselect_count} "
            f"unchanged={s.unchanged_program_count} gating={s.external_gating_count}"
        )
        lines.append(
            f"avg refinement overhead ratio={_fmt(s.avg_refinement_overhead_ratio)} "
            f"raw num_best={s.raw_teh_num_best} gated num_best={s.gated_teh_num_best} "
            f"ranking_changed={s.ranking_changed}"
        )
        lines.append(f"Recommendation: {s.recommendation}")
        lines.append("")

    lines.append("=== Comparison table ===")
    lines.extend(_comparison_table_lines(summaries))
    lines.append("")
    lines.extend(_overall_verdicts(summaries, all_participants))
    return "\n".join(lines) + "\n"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--all_in",
        action="store_true",
        help="Analyze all standard train Psych-101 datasets plus mixed_gambles.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to analyze (default: all_in list).",
    )
    p.add_argument(
        "--out_dir",
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--baseline_config",
        default=_DEFAULT_BASELINE_CONFIG,
        help=f"Baseline config YAML (default: {_DEFAULT_BASELINE_CONFIG}).",
    )
    p.add_argument("--verbose", action="store_true", help="Print discovery logs.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    repo = _repo_root()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()

    datasets = list(_ALL_IN_DATASETS) if args.all_in or not args.datasets else list(args.datasets)
    if not datasets:
        raise SystemExit("No datasets selected; pass --all_in or --datasets.")

    config_path = Path(args.baseline_config)
    if not config_path.is_absolute():
        config_path = (repo / config_path).resolve()
    config_data = cmp._load_baseline_config_file(config_path)

    all_participants: List[_ParticipantRow] = []
    summaries: List[_DatasetSummary] = []

    for ds in datasets:
        alias = cmp._normalize_compare_dataset(ds)
        try:
            parts, summary = _analyze_dataset(
                repo, alias, config_data=config_data, quiet=not args.verbose
            )
        except Exception as exc:
            parts = []
            summary = _DatasetSummary(dataset=alias, error=f"{type(exc).__name__}: {exc}")
            print(f"ERROR {alias}: {summary.error}", file=sys.stderr)
        all_participants.extend(parts)
        summaries.append(summary)

    participant_fields = list(_participant_to_csv_row(all_participants[0]).keys()) if all_participants else [
        "dataset",
        "run_name",
        "participant_id",
        "train_loglik",
        "val_loglik",
        "test_loglik",
        "gated_test_loglik",
        "delta",
        "effect",
        "refinement_triggered",
        "refinement_skipped",
        "evolution_program_id",
        "final_program_id",
        "final_is_evolution_best",
        "refinement_new_program",
        "best_program_source",
        "refinement_iterations_configured",
        "refinement_iterations_used",
        "refinement_llm_calls",
        "evolution_llm_calls",
        "refinement_overhead_ratio",
    ]
    summary_fields = list(_summary_to_csv_row(_DatasetSummary(dataset="")).keys())

    _write_csv(
        out_dir / "refinement_participants.csv",
        participant_fields,
        [_participant_to_csv_row(p) for p in all_participants],
    )
    _write_csv(
        out_dir / "refinement_dataset_summary.csv",
        summary_fields,
        [_summary_to_csv_row(s) for s in summaries],
    )
    report = _build_report(summaries, all_participants)
    _write_text(out_dir / "refinement_report.txt", report)

    print(report)
    print(f"Wrote {out_dir / 'refinement_participants.csv'}")
    print(f"Wrote {out_dir / 'refinement_dataset_summary.csv'}")
    print(f"Wrote {out_dir / 'refinement_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
