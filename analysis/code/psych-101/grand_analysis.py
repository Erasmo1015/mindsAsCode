#!/usr/bin/env python3
"""
Grand analysis of latest TEH psych-101 train experiments vs baselines.

Usage:
  python analysis/code/psych-101/grand_analysis.py --all_in
  python analysis/code/psych-101/grand_analysis.py --dataset 8flesch2018comparing

Reads newest TEH run_* per dataset, participant_details_loglik.csv, baselines via
compare.py discovery, iteration_convergence.csv (from iter.py), and per-participant
prompt_diagnostics.jsonl for trial/token truncation analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.psych101_binary import DEFAULT_PSYCH_DATASET_SPLIT
from utils.teh.teh_datasets import is_mixed_gambles_dataset

from analysis.code.utils import compare as cmp

_ALL_IN_DATASETS = cmp._ALL_IN_DATASETS
_BASELINE_METHODS = cmp._BASELINE_METHODS
_TEST_LOGLIK = cmp._TEST_LOGLIK
_GATED_LOGLIK = cmp._GATED_LOGLIK

_DEFAULT_OUT_DIR = "analysis/data/psych101_grand_analysis"
_DEFAULT_CONVERGENCE_CSV = (
    "generated_outputs/psych101_train/teh/iteration_convergence.csv"
)
_DEFAULT_BASELINE_CONFIG = cmp._DEFAULT_BASELINE_CONFIG
_PSYCH_SPLIT = DEFAULT_PSYCH_DATASET_SPLIT

_OVERFIT_TRAIN_TEST_GAP = 0.15
_WEAK_TRAIN_LOGLIK = -0.65
_MIN_TAIL_CONVERGED = 3
_SEVERE_TRUNCATION_RATIO = 0.5
_PROMPT_NEAR_LIMIT_FRAC = 0.85
_TRUNCATION_CORRELATION_MIN = 0.08
_PROMPT_DIAGNOSTICS_NAME = "prompt_diagnostics.jsonl"
_PARTICIPANT_DIR_RE = re.compile(r"^participant_(\d+)$")

_PARTICIPANT_FIELDS = (
    "dataset",
    "run_name",
    "participant_id",
    "teh_train_loglik",
    "teh_val_loglik",
    "teh_test_loglik",
    "teh_gated_test_loglik",
    "best_baseline_method",
    "best_baseline_score",
    "gap_to_best_baseline",
    "teh_is_best",
    "teh_within_margin",
    "teh_win",
    "tail_converged_steps",
    "probably_enough",
    "n_points",
    "final_train_loglik",
    "first_train_loglik",
    "final_improvement",
    "search_budget_failure",
    "converged_but_losing",
    "possible_overfit",
    "weak_train_fit",
    "failure_category",
    "total_train_trials",
    "included_train_trials",
    "excluded_train_trials",
    "truncation_ratio",
    "included_fraction",
    "prompt_char_count",
    "prompt_token_estimate",
    "unique_problems_included",
    "truncation_occurred",
    "per_problem_cap_clipping",
    "prompt_near_limit",
    "severe_truncation",
    "has_prompt_diagnostics",
)

_DATASET_SUMMARY_FIELDS = (
    "dataset",
    "run_name",
    "n_participants",
    "avg_teh_test",
    "avg_teh_gated_test",
    "avg_best_baseline",
    "avg_gap_to_best",
    "teh_win_count",
    "teh_win_rate",
    "within_margin_count",
    "within_margin_rate",
    "search_budget_failure_count",
    "converged_but_losing_count",
    "possible_overfit_count",
    "weak_train_fit_count",
    "avg_tail_converged_steps",
    "probably_enough_rate",
    "dominant_baseline_method",
    "dominant_baseline_count",
    "avg_truncation_ratio",
    "avg_included_fraction",
    "severe_truncation_count",
    "n_with_prompt_diagnostics",
)

_FAILURE_FIELDS = (
    "dataset",
    "run_name",
    "participant_id",
    "teh_test_loglik",
    "best_baseline_method",
    "best_baseline_score",
    "gap_to_best_baseline",
    "tail_converged_steps",
    "probably_enough",
    "final_train_loglik",
    "failure_category",
    "possible_overfit",
    "weak_train_fit",
    "truncation_ratio",
    "included_fraction",
    "severe_truncation",
    "prompt_near_limit",
)


@dataclass
class _ParticipantRow:
    dataset: str
    run_name: str
    participant_id: int
    teh_train_loglik: Optional[float] = None
    teh_val_loglik: Optional[float] = None
    teh_test_loglik: Optional[float] = None
    teh_gated_test_loglik: Optional[float] = None
    best_baseline_method: str = ""
    best_baseline_score: Optional[float] = None
    gap_to_best_baseline: Optional[float] = None
    teh_is_best: bool = False
    teh_within_margin: bool = False
    teh_win: bool = False
    tail_converged_steps: int = 0
    probably_enough: bool = False
    n_points: int = 0
    final_train_loglik: Optional[float] = None
    first_train_loglik: Optional[float] = None
    final_improvement: Optional[float] = None
    search_budget_failure: bool = False
    converged_but_losing: bool = False
    possible_overfit: bool = False
    weak_train_fit: bool = False
    failure_category: str = ""
    baseline_scores: Dict[str, float] = field(default_factory=dict)
    total_train_trials: Optional[int] = None
    included_train_trials: Optional[int] = None
    excluded_train_trials: Optional[int] = None
    truncation_ratio: Optional[float] = None
    included_fraction: Optional[float] = None
    prompt_char_count: Optional[int] = None
    prompt_token_estimate: Optional[int] = None
    unique_problems_included: Optional[int] = None
    truncation_occurred: bool = False
    per_problem_cap_clipping: bool = False
    prompt_near_limit: bool = False
    severe_truncation: bool = False
    has_prompt_diagnostics: bool = False


@dataclass
class _DatasetSummary:
    dataset: str
    run_name: str = ""
    n_participants: int = 0
    avg_teh_test: Optional[float] = None
    avg_teh_gated_test: Optional[float] = None
    avg_best_baseline: Optional[float] = None
    avg_gap_to_best: Optional[float] = None
    teh_win_count: int = 0
    teh_win_rate: Optional[float] = None
    within_margin_count: int = 0
    within_margin_rate: Optional[float] = None
    search_budget_failure_count: int = 0
    converged_but_losing_count: int = 0
    possible_overfit_count: int = 0
    weak_train_fit_count: int = 0
    avg_tail_converged_steps: Optional[float] = None
    probably_enough_rate: Optional[float] = None
    dominant_baseline_method: str = ""
    dominant_baseline_count: int = 0
    avg_truncation_ratio: Optional[float] = None
    avg_included_fraction: Optional[float] = None
    severe_truncation_count: int = 0
    n_with_prompt_diagnostics: int = 0
    error: Optional[str] = None


@dataclass
class _TruncationCorrelation:
    n_with_diagnostics: int = 0
    losing_mean_truncation: Optional[float] = None
    winning_mean_truncation: Optional[float] = None
    conv_losing_mean_truncation: Optional[float] = None
    conv_losing_mean_included: Optional[float] = None
    other_losing_mean_truncation: Optional[float] = None
    corr_included_fraction_test_loglik: Optional[float] = None
    corr_truncation_ratio_gap: Optional[float] = None
    f8_avg_truncation: Optional[float] = None
    global_avg_truncation: Optional[float] = None


def _repo_root() -> Path:
    return _REPO_ROOT


def _normalize_dataset(dataset: str) -> str:
    return cmp._normalize_compare_dataset(dataset)


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


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _load_convergence_csv(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Key (dataset, participant_id) -> convergence fields."""
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = str(row.get("dataset", "")).strip()
            pid_raw = row.get("participant_id")
            if not ds or pid_raw is None or str(pid_raw).strip() == "":
                continue
            pid = int(float(pid_raw))
            out[(ds, pid)] = {
                "run_name": str(row.get("run_name", "")),
                "tail_converged_steps": int(float(row.get("tail_converged_steps") or 0)),
                "probably_enough": _parse_bool(str(row.get("probably_enough", ""))),
                "n_points": int(float(row.get("n_points") or 0)),
                "final_train_loglik": _safe_float(row.get("final_train_loglik")),
                "first_train_loglik": _safe_float(row.get("first_train_loglik")),
                "final_improvement": _safe_float(row.get("final_improvement")),
            }
    return out


def _read_prompt_diagnostics_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _step_indicates_per_problem_cap(steps: Sequence[Any]) -> bool:
    for step in steps:
        s = str(step)
        if "per_problem_cap" in s:
            return True
    return False


def _summarize_prompt_diagnostics(participant_dir: Path) -> Dict[str, Any]:
    """
    Aggregate prompt_diagnostics.jsonl for one participant (worst-case trial omission).

    Trial omission ratio uses train_trials_before vs train_trials_after per row.
    Token ``truncated`` is separate from max_prompt_train_trials capping (often
    truncated=false while trials are still clipped).
    """
    rows = _read_prompt_diagnostics_jsonl(participant_dir / _PROMPT_DIAGNOSTICS_NAME)
    if not rows:
        return {}

    max_ratio = 0.0
    worst_before = 0
    worst_after = 0
    truncation_occurred = False
    per_problem_cap_clipping = False
    prompt_near_limit = False
    max_tokens = 0
    max_chars = 0
    token_cap: Optional[int] = None
    unique_problems: Optional[int] = None

    for row in rows:
        before = int(row.get("train_trials_before") or 0)
        after = int(row.get("train_trials_after") or 0)
        if before > 0:
            ratio = 1.0 - (after / before)
            if ratio > max_ratio:
                max_ratio = ratio
                worst_before = before
                worst_after = after

        if row.get("truncated"):
            truncation_occurred = True
        steps = row.get("truncation_steps") or []
        if steps:
            truncation_occurred = True
        if before > after:
            truncation_occurred = True
        if _step_indicates_per_problem_cap(steps):
            per_problem_cap_clipping = True

        cap = row.get("hard_prompt_token_cap")
        if cap is not None:
            try:
                token_cap = int(cap)
            except (TypeError, ValueError):
                pass
        tok = row.get("prompt_tokens_after_truncation")
        if tok is None:
            tok = row.get("prompt_tokens_before_truncation")
        tok_i = int(tok) if tok is not None else 0
        if tok_i > max_tokens:
            max_tokens = tok_i
        chars = row.get("prompt_char_count") or row.get("final_prompt_char_length")
        if chars is not None:
            try:
                max_chars = max(max_chars, int(chars))
            except (TypeError, ValueError):
                pass
        elif tok_i > 0:
            max_chars = max(max_chars, tok_i * 4)

        if token_cap and tok_i >= _PROMPT_NEAR_LIMIT_FRAC * token_cap:
            prompt_near_limit = True

        for key in (
            "unique_problems_included",
            "n_unique_problems_included",
            "number_of_unique_problems_included",
        ):
            if key in row and row[key] is not None:
                try:
                    unique_problems = int(row[key])
                except (TypeError, ValueError):
                    pass

    excluded = max(0, worst_before - worst_after)
    included_fraction = (worst_after / worst_before) if worst_before > 0 else 1.0
    severe = max_ratio > _SEVERE_TRUNCATION_RATIO

    return {
        "total_train_trials": worst_before if worst_before > 0 else None,
        "included_train_trials": worst_after if worst_before > 0 else None,
        "excluded_train_trials": excluded if worst_before > 0 else None,
        "truncation_ratio": max_ratio if worst_before > 0 else 0.0,
        "included_fraction": included_fraction,
        "prompt_char_count": max_chars if max_chars > 0 else None,
        "prompt_token_estimate": max_tokens if max_tokens > 0 else None,
        "unique_problems_included": unique_problems,
        "truncation_occurred": truncation_occurred,
        "per_problem_cap_clipping": per_problem_cap_clipping,
        "prompt_near_limit": prompt_near_limit,
        "severe_truncation": severe,
        "has_prompt_diagnostics": True,
    }


def _apply_prompt_diagnostics(row: _ParticipantRow, diag: Mapping[str, Any]) -> None:
    if not diag:
        return
    row.has_prompt_diagnostics = True
    row.total_train_trials = diag.get("total_train_trials")
    row.included_train_trials = diag.get("included_train_trials")
    row.excluded_train_trials = diag.get("excluded_train_trials")
    row.truncation_ratio = _safe_float(diag.get("truncation_ratio"))
    row.included_fraction = _safe_float(diag.get("included_fraction"))
    row.prompt_char_count = diag.get("prompt_char_count")
    row.prompt_token_estimate = diag.get("prompt_token_estimate")
    row.unique_problems_included = diag.get("unique_problems_included")
    row.truncation_occurred = bool(diag.get("truncation_occurred"))
    row.per_problem_cap_clipping = bool(diag.get("per_problem_cap_clipping"))
    row.prompt_near_limit = bool(diag.get("prompt_near_limit"))
    row.severe_truncation = bool(diag.get("severe_truncation"))


def _participant_diag_path(teh_run: Path, participant_id: int) -> Path:
    return teh_run / f"participant_{participant_id}"


def _pearson_correlation(
    xs: Sequence[float], ys: Sequence[float]
) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _analyze_truncation_correlations(
    participants: Sequence[_ParticipantRow],
) -> _TruncationCorrelation:
    with_diag = [p for p in participants if p.has_prompt_diagnostics]
    out = _TruncationCorrelation(n_with_diagnostics=len(with_diag))
    if not with_diag:
        return out

    losing = [p for p in with_diag if p.failure_category]
    winning = [p for p in with_diag if p.teh_win]

    def _mean_trunc(group: Sequence[_ParticipantRow]) -> Optional[float]:
        vals = [p.truncation_ratio for p in group if p.truncation_ratio is not None]
        return statistics.mean(vals) if vals else None

    out.losing_mean_truncation = _mean_trunc(losing)
    out.winning_mean_truncation = _mean_trunc(winning)

    conv_lose = [p for p in losing if p.converged_but_losing]
    other_lose = [p for p in losing if not p.converged_but_losing]
    out.conv_losing_mean_truncation = _mean_trunc(conv_lose)
    inc_vals = [p.included_fraction for p in conv_lose if p.included_fraction is not None]
    out.conv_losing_mean_included = statistics.mean(inc_vals) if inc_vals else None
    out.other_losing_mean_truncation = _mean_trunc(other_lose)

    xs: List[float] = []
    ys: List[float] = []
    for p in with_diag:
        if p.included_fraction is None or p.teh_test_loglik is None:
            continue
        xs.append(float(p.included_fraction))
        ys.append(float(p.teh_test_loglik))
    out.corr_included_fraction_test_loglik = _pearson_correlation(xs, ys)

    xs2: List[float] = []
    ys2: List[float] = []
    for p in with_diag:
        if p.truncation_ratio is None or p.gap_to_best_baseline is None:
            continue
        xs2.append(float(p.truncation_ratio))
        ys2.append(float(p.gap_to_best_baseline))
    out.corr_truncation_ratio_gap = _pearson_correlation(xs2, ys2)

    f8 = [p for p in with_diag if p.dataset == "8flesch2018comparing"]
    out.f8_avg_truncation = _mean_trunc(f8)
    out.global_avg_truncation = _mean_trunc(with_diag)
    return out


def _per_dataset_included_fraction_correlations(
    participants: Sequence[_ParticipantRow],
) -> List[Tuple[str, Optional[float], int]]:
    """(dataset, pearson r, n) for datasets with varying included_fraction."""
    by_ds: Dict[str, List[_ParticipantRow]] = defaultdict(list)
    for p in participants:
        if p.has_prompt_diagnostics and p.included_fraction is not None:
            by_ds[p.dataset].append(p)
    out: List[Tuple[str, Optional[float], int]] = []
    for ds in sorted(by_ds):
        group = by_ds[ds]
        fracs = [float(p.included_fraction) for p in group if p.included_fraction is not None]
        if len(set(round(x, 4) for x in fracs)) < 2:
            continue
        xs = [float(p.included_fraction) for p in group if p.included_fraction is not None and p.teh_test_loglik is not None]
        ys = [float(p.teh_test_loglik) for p in group if p.included_fraction is not None and p.teh_test_loglik is not None]
        out.append((ds, _pearson_correlation(xs, ys), len(xs)))
    return out


def _best_baseline_for_participant(
    baseline_maps: Mapping[str, Dict[int, float]], participant_id: int
) -> Tuple[str, Optional[float]]:
    best_method = ""
    best_score: Optional[float] = None
    for method in _BASELINE_METHODS:
        scores = baseline_maps.get(method, {})
        if participant_id not in scores:
            continue
        score = scores[participant_id]
        if not math.isfinite(score):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_method = method
    return best_method, best_score


def _dominant_baseline_per_dataset(
    rows: Sequence[_ParticipantRow],
) -> Tuple[str, int]:
    """Baseline that is best for the most participants (ties split credit)."""
    counts: Counter[str] = Counter()
    for r in rows:
        if r.best_baseline_method:
            counts[r.best_baseline_method] += 1
    if not counts:
        return "", 0
    method, count = counts.most_common(1)[0]
    return method, count


def _classify_participant(
    row: _ParticipantRow,
    *,
    loss_margin: float,
) -> None:
    """Set win/loss flags and failure categories on row in place."""
    teh_test = row.teh_test_loglik
    best = row.best_baseline_score

    if teh_test is not None and best is not None:
        row.gap_to_best_baseline = best - teh_test
        row.teh_is_best = teh_test >= best
        row.teh_within_margin = (
            row.teh_is_best or abs(teh_test - best) <= loss_margin
        )
        row.teh_win = row.teh_is_best or row.teh_within_margin
    else:
        row.gap_to_best_baseline = None
        row.teh_is_best = False
        row.teh_within_margin = False
        row.teh_win = False

    train = row.teh_train_loglik
    if train is None:
        train = row.final_train_loglik
    if train is not None and teh_test is not None:
        row.possible_overfit = (train - teh_test) > _OVERFIT_TRAIN_TEST_GAP
    if row.final_train_loglik is not None:
        row.weak_train_fit = row.final_train_loglik <= _WEAK_TRAIN_LOGLIK

    losing = (
        row.gap_to_best_baseline is not None
        and row.gap_to_best_baseline > loss_margin
    )
    if losing:
        low_conv = (not row.probably_enough) or (
            row.tail_converged_steps < _MIN_TAIL_CONVERGED
        )
        high_conv = row.probably_enough and (
            row.tail_converged_steps >= _MIN_TAIL_CONVERGED
        )
        row.search_budget_failure = low_conv
        row.converged_but_losing = high_conv
        if row.search_budget_failure:
            row.failure_category = "search_budget_failure"
        elif row.converged_but_losing:
            row.failure_category = "converged_but_losing"
        else:
            row.failure_category = "losing_other"
    else:
        row.search_budget_failure = False
        row.converged_but_losing = False
        row.failure_category = ""


def _finite_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return statistics.mean(vals)


def _rate(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return count / total


def _analyze_dataset(
    repo: Path,
    dataset: str,
    *,
    config_data: Mapping[str, Any],
    convergence: Mapping[Tuple[str, int], Dict[str, Any]],
    loss_margin: float,
    quiet: bool,
) -> Tuple[List[_ParticipantRow], _DatasetSummary]:
    alias = _normalize_dataset(dataset)
    psych_split = _PSYCH_SPLIT if not is_mixed_gambles_dataset(alias) else _PSYCH_SPLIT

    teh_run = cmp._auto_discover_teh_run(
        repo, dataset=alias, psych_dataset_split=psych_split
    )
    if teh_run is None:
        root = cmp._teh_search_root(repo, alias, psych_split)
        return [], _DatasetSummary(
            dataset=alias,
            error=f"no TEH run found under {root.relative_to(repo)}",
        )

    if not quiet:
        print(
            f"Auto-selected TEH for {alias}: {teh_run.relative_to(repo)}",
            flush=True,
        )

    teh_csv = cmp._resolve_loglik_csv(teh_run)
    run_name = teh_run.name if teh_run.is_dir() else teh_run.parent.name

    teh_train = cmp._read_loglik_csv(teh_csv, "train_loglik", required=False)
    teh_val = cmp._read_loglik_csv(teh_csv, "val_loglik", required=False)
    teh_test = cmp._read_loglik_csv(teh_csv, _TEST_LOGLIK, required=False)
    teh_gated = cmp._read_loglik_csv(teh_csv, _GATED_LOGLIK, required=False)

    baseline_paths = cmp._resolve_baseline_run_paths(
        config_data, repo, alias, psych_split, quiet=quiet
    )
    baseline_maps: Dict[str, Dict[int, float]] = {}
    for method in _BASELINE_METHODS:
        if method not in baseline_paths:
            baseline_maps[method] = {}
            continue
        try:
            baseline_maps[method] = cmp._load_scores_from_run(
                baseline_paths[method], _TEST_LOGLIK, required=False
            )
        except (OSError, ValueError) as exc:
            if not quiet:
                print(
                    f"Warning: could not load {method} for {alias}: {exc}",
                    file=sys.stderr,
                )
            baseline_maps[method] = {}

    participant_ids = sorted(set(teh_test) | set(teh_train))
    if not participant_ids and baseline_maps:
        for scores in baseline_maps.values():
            participant_ids = sorted(set(scores))
            if participant_ids:
                break

    if not participant_ids:
        return [], _DatasetSummary(
            dataset=alias,
            run_name=run_name,
            error=f"no participants with TEH scores in {teh_csv.relative_to(repo)}",
        )

    rows: List[_ParticipantRow] = []
    for pid in participant_ids:
        conv = convergence.get((alias, pid), {})
        row = _ParticipantRow(
            dataset=alias,
            run_name=run_name,
            participant_id=pid,
            teh_train_loglik=teh_train.get(pid),
            teh_val_loglik=teh_val.get(pid),
            teh_test_loglik=teh_test.get(pid),
            teh_gated_test_loglik=teh_gated.get(pid),
            tail_converged_steps=int(conv.get("tail_converged_steps", 0)),
            probably_enough=bool(conv.get("probably_enough", False)),
            n_points=int(conv.get("n_points", 0)),
            final_train_loglik=conv.get("final_train_loglik"),
            first_train_loglik=conv.get("first_train_loglik"),
            final_improvement=conv.get("final_improvement"),
            baseline_scores={
                m: baseline_maps[m][pid]
                for m in _BASELINE_METHODS
                if pid in baseline_maps.get(m, {})
            },
        )
        method, score = _best_baseline_for_participant(baseline_maps, pid)
        row.best_baseline_method = method
        row.best_baseline_score = score
        _classify_participant(row, loss_margin=loss_margin)
        pdir = _participant_diag_path(teh_run, pid)
        if pdir.is_dir():
            _apply_prompt_diagnostics(row, _summarize_prompt_diagnostics(pdir))
        rows.append(row)

    comparable = [r for r in rows if r.gap_to_best_baseline is not None]
    diag_rows = [r for r in rows if r.has_prompt_diagnostics]
    dom_method, dom_count = _dominant_baseline_per_dataset(rows)
    n = len(rows)
    summary = _DatasetSummary(
        dataset=alias,
        run_name=run_name,
        n_participants=n,
        avg_teh_test=_finite_mean([r.teh_test_loglik for r in rows]),
        avg_teh_gated_test=_finite_mean([r.teh_gated_test_loglik for r in rows]),
        avg_best_baseline=_finite_mean([r.best_baseline_score for r in comparable]),
        avg_gap_to_best=_finite_mean([r.gap_to_best_baseline for r in comparable]),
        teh_win_count=sum(1 for r in rows if r.teh_win),
        teh_win_rate=_rate(sum(1 for r in rows if r.teh_win), n),
        within_margin_count=sum(1 for r in rows if r.teh_within_margin),
        within_margin_rate=_rate(sum(1 for r in rows if r.teh_within_margin), n),
        search_budget_failure_count=sum(1 for r in rows if r.search_budget_failure),
        converged_but_losing_count=sum(1 for r in rows if r.converged_but_losing),
        possible_overfit_count=sum(1 for r in rows if r.possible_overfit),
        weak_train_fit_count=sum(1 for r in rows if r.weak_train_fit),
        avg_tail_converged_steps=_finite_mean(
            [float(r.tail_converged_steps) for r in rows]
        ),
        probably_enough_rate=_rate(sum(1 for r in rows if r.probably_enough), n),
        dominant_baseline_method=dom_method,
        dominant_baseline_count=dom_count,
        avg_truncation_ratio=_finite_mean(
            [r.truncation_ratio for r in diag_rows]
        ),
        avg_included_fraction=_finite_mean(
            [r.included_fraction for r in diag_rows]
        ),
        severe_truncation_count=sum(1 for r in diag_rows if r.severe_truncation),
        n_with_prompt_diagnostics=len(diag_rows),
    )
    return rows, summary


def _participant_to_dict(row: _ParticipantRow) -> Dict[str, str]:
    def fmt(v: Optional[float], ndigits: int = 6) -> str:
        if v is None or not math.isfinite(v):
            return ""
        return f"{v:.{ndigits}f}"

    return {
        "dataset": row.dataset,
        "run_name": row.run_name,
        "participant_id": str(row.participant_id),
        "teh_train_loglik": fmt(row.teh_train_loglik, 4),
        "teh_val_loglik": fmt(row.teh_val_loglik, 4),
        "teh_test_loglik": fmt(row.teh_test_loglik, 4),
        "teh_gated_test_loglik": fmt(row.teh_gated_test_loglik, 4),
        "best_baseline_method": row.best_baseline_method,
        "best_baseline_score": fmt(row.best_baseline_score, 4),
        "gap_to_best_baseline": fmt(row.gap_to_best_baseline, 4),
        "teh_is_best": str(row.teh_is_best),
        "teh_within_margin": str(row.teh_within_margin),
        "teh_win": str(row.teh_win),
        "tail_converged_steps": str(row.tail_converged_steps),
        "probably_enough": str(row.probably_enough),
        "n_points": str(row.n_points),
        "final_train_loglik": fmt(row.final_train_loglik, 6),
        "first_train_loglik": fmt(row.first_train_loglik, 6),
        "final_improvement": fmt(row.final_improvement, 6),
        "search_budget_failure": str(row.search_budget_failure),
        "converged_but_losing": str(row.converged_but_losing),
        "possible_overfit": str(row.possible_overfit),
        "weak_train_fit": str(row.weak_train_fit),
        "failure_category": row.failure_category,
        "total_train_trials": (
            str(row.total_train_trials) if row.total_train_trials is not None else ""
        ),
        "included_train_trials": (
            str(row.included_train_trials)
            if row.included_train_trials is not None
            else ""
        ),
        "excluded_train_trials": (
            str(row.excluded_train_trials)
            if row.excluded_train_trials is not None
            else ""
        ),
        "truncation_ratio": fmt(row.truncation_ratio, 4),
        "included_fraction": fmt(row.included_fraction, 4),
        "prompt_char_count": (
            str(row.prompt_char_count) if row.prompt_char_count is not None else ""
        ),
        "prompt_token_estimate": (
            str(row.prompt_token_estimate)
            if row.prompt_token_estimate is not None
            else ""
        ),
        "unique_problems_included": (
            str(row.unique_problems_included)
            if row.unique_problems_included is not None
            else ""
        ),
        "truncation_occurred": str(row.truncation_occurred),
        "per_problem_cap_clipping": str(row.per_problem_cap_clipping),
        "prompt_near_limit": str(row.prompt_near_limit),
        "severe_truncation": str(row.severe_truncation),
        "has_prompt_diagnostics": str(row.has_prompt_diagnostics),
    }


def _dataset_summary_to_dict(s: _DatasetSummary) -> Dict[str, str]:
    def fmt(v: Optional[float], ndigits: int = 4) -> str:
        if v is None or not math.isfinite(v):
            return ""
        return f"{v:.{ndigits}f}"

    def fmt_rate(v: Optional[float]) -> str:
        if v is None:
            return ""
        return f"{v:.4f}"

    return {
        "dataset": s.dataset,
        "run_name": s.run_name,
        "n_participants": str(s.n_participants),
        "avg_teh_test": fmt(s.avg_teh_test),
        "avg_teh_gated_test": fmt(s.avg_teh_gated_test),
        "avg_best_baseline": fmt(s.avg_best_baseline),
        "avg_gap_to_best": fmt(s.avg_gap_to_best),
        "teh_win_count": str(s.teh_win_count),
        "teh_win_rate": fmt_rate(s.teh_win_rate),
        "within_margin_count": str(s.within_margin_count),
        "within_margin_rate": fmt_rate(s.within_margin_rate),
        "search_budget_failure_count": str(s.search_budget_failure_count),
        "converged_but_losing_count": str(s.converged_but_losing_count),
        "possible_overfit_count": str(s.possible_overfit_count),
        "weak_train_fit_count": str(s.weak_train_fit_count),
        "avg_tail_converged_steps": fmt(s.avg_tail_converged_steps, 2),
        "probably_enough_rate": fmt_rate(s.probably_enough_rate),
        "dominant_baseline_method": s.dominant_baseline_method,
        "dominant_baseline_count": str(s.dominant_baseline_count),
        "avg_truncation_ratio": fmt(s.avg_truncation_ratio, 4),
        "avg_included_fraction": fmt(s.avg_included_fraction, 4),
        "severe_truncation_count": str(s.severe_truncation_count),
        "n_with_prompt_diagnostics": str(s.n_with_prompt_diagnostics),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _failure_rows(participants: Sequence[_ParticipantRow]) -> List[Dict[str, str]]:
    losing = [
        r
        for r in participants
        if r.gap_to_best_baseline is not None and not r.teh_win
    ]
    losing.sort(
        key=lambda r: (
            -(r.gap_to_best_baseline or 0),
            r.dataset,
            r.participant_id,
        )
    )
    out: List[Dict[str, str]] = []
    for r in losing:
        gap = r.gap_to_best_baseline or 0.0
        out.append(
            {
                "dataset": r.dataset,
                "run_name": r.run_name,
                "participant_id": str(r.participant_id),
                "teh_test_loglik": f"{r.teh_test_loglik:.4f}"
                if r.teh_test_loglik is not None
                else "",
                "best_baseline_method": r.best_baseline_method,
                "best_baseline_score": f"{r.best_baseline_score:.4f}"
                if r.best_baseline_score is not None
                else "",
                "gap_to_best_baseline": f"{gap:.4f}",
                "tail_converged_steps": str(r.tail_converged_steps),
                "probably_enough": str(r.probably_enough),
                "final_train_loglik": f"{r.final_train_loglik:.6f}"
                if r.final_train_loglik is not None
                else "",
                "failure_category": r.failure_category,
                "possible_overfit": str(r.possible_overfit),
                "weak_train_fit": str(r.weak_train_fit),
                "truncation_ratio": f"{r.truncation_ratio:.4f}"
                if r.truncation_ratio is not None
                else "",
                "included_fraction": f"{r.included_fraction:.4f}"
                if r.included_fraction is not None
                else "",
                "severe_truncation": str(r.severe_truncation),
                "prompt_near_limit": str(r.prompt_near_limit),
            }
        )
    return out


def _pct(n: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * n / total:.1f}%"


def _append_truncation_recommendations(
    lines: List[str],
    participants: Sequence[_ParticipantRow],
    summaries: Sequence[_DatasetSummary],
    trunc: _TruncationCorrelation,
) -> None:
    """Prompt truncation section; only recommend changes when stats support them."""
    lines.append("## Prompt truncation / information loss")
    lines.append(
        f"Heuristics: severe_truncation if trial omission ratio > {_SEVERE_TRUNCATION_RATIO}; "
        f"prompt_near_limit if tokens >= {_PROMPT_NEAR_LIMIT_FRAC:.0%} of hard_prompt_token_cap. "
        "Trial omission is measured via train_trials_before vs train_trials_after in "
        "prompt_diagnostics.jsonl (includes max_prompt_train_trials capping, not only "
        "token-level truncated=true)."
    )
    lines.append("")

    if trunc.n_with_diagnostics == 0:
        lines.append("- No prompt_diagnostics.jsonl files found; skipping truncation analysis.")
        lines.append("")
        return

    lines.append(f"- Participants with diagnostics: {trunc.n_with_diagnostics}")
    if trunc.losing_mean_truncation is not None and trunc.winning_mean_truncation is not None:
        delta = trunc.losing_mean_truncation - trunc.winning_mean_truncation
        lines.append(
            f"- Mean truncation_ratio: losing={trunc.losing_mean_truncation:.3f}, "
            f"winning={trunc.winning_mean_truncation:.3f}, delta={delta:+.3f}"
        )
        if delta >= _TRUNCATION_CORRELATION_MIN:
            lines.append(
                "  **Losing participants are more truncated** (trial omission), supporting "
                "prompt-budget experiments on affected datasets."
            )
        elif delta <= -_TRUNCATION_CORRELATION_MIN:
            lines.append(
                "  Losing participants are *less* truncated on average — losses are unlikely "
                "driven by trial omission globally."
            )
        else:
            lines.append(
                "  No meaningful gap in truncation between losing and winning participants globally."
            )

    if trunc.conv_losing_mean_truncation is not None:
        lines.append(
            f"- Converged-but-losing mean truncation_ratio: "
            f"{trunc.conv_losing_mean_truncation:.3f}"
            + (
                f", mean included_fraction: {trunc.conv_losing_mean_included:.3f}"
                if trunc.conv_losing_mean_included is not None
                else ""
            )
        )
        if (
            trunc.conv_losing_mean_truncation is not None
            and trunc.conv_losing_mean_truncation <= _TRUNCATION_CORRELATION_MIN
        ):
            lines.append(
                "  Converged-but-losing cases are **not** highly truncated on average — "
                "iteration increases are unlikely to fix these via more training data in prompts."
            )
        elif trunc.conv_losing_mean_truncation > _SEVERE_TRUNCATION_RATIO:
            lines.append(
                "  Some converged-but-losing mass coincides with heavy trial omission; "
                "consider prompt budget before longer search."
            )

    if trunc.corr_included_fraction_test_loglik is not None:
        lines.append(
            f"- Pearson r(included_fraction, TEH test_loglik): "
            f"{trunc.corr_included_fraction_test_loglik:.3f}"
        )
    if trunc.corr_truncation_ratio_gap is not None:
        lines.append(
            f"- Pearson r(truncation_ratio, gap_to_best_baseline): "
            f"{trunc.corr_truncation_ratio_gap:.3f} (positive => more omission, worse vs baseline)"
        )

    per_ds_corr = _per_dataset_included_fraction_correlations(participants)
    if per_ds_corr:
        lines.append("- Per-dataset r(included_fraction, TEH test_loglik) where fraction varies:")
        for ds, r, n in per_ds_corr:
            if r is None:
                continue
            trend = (
                "higher inclusion ↔ better test (prompt budget may help)"
                if r >= 0.2
                else (
                    "higher inclusion ↔ worse test (omission may act as regularization)"
                    if r <= -0.2
                    else "weak/no linear link within dataset"
                )
            )
            lines.append(f"  - {ds}: r={r:.3f} (n={n}) — {trend}")

    ok_summaries = [s for s in summaries if not s.error]
    by_trunc = sorted(
        [s for s in ok_summaries if s.avg_truncation_ratio is not None],
        key=lambda s: s.avg_truncation_ratio or 0,
        reverse=True,
    )
    if by_trunc:
        lines.append("- Datasets by avg trial omission (truncation_ratio):")
        for s in by_trunc[:4]:
            lines.append(
                f"  - {s.dataset}: avg_truncation={s.avg_truncation_ratio:.3f}, "
                f"severe_truncation={s.severe_truncation_count}/{s.n_with_prompt_diagnostics}, "
                f"avg_included_fraction={s.avg_included_fraction:.3f}"
                if s.avg_included_fraction is not None
                else f"  - {s.dataset}: avg_truncation={s.avg_truncation_ratio:.3f}"
            )

    f8 = next((s for s in ok_summaries if s.dataset == "8flesch2018comparing"), None)
    if f8 and f8.avg_truncation_ratio is not None:
        global_avg = trunc.global_avg_truncation or 0.0
        lines.append("")
        lines.append("### 8flesch2018comparing truncation")
        lines.append(
            f"- avg_truncation_ratio={f8.avg_truncation_ratio:.3f} "
            f"(global avg {global_avg:.3f}); "
            f"{f8.severe_truncation_count}/{f8.n_participants} severe_truncation."
        )
        if f8.avg_truncation_ratio >= _SEVERE_TRUNCATION_RATIO:
            lines.append(
                "- Structural disadvantage: most train trials are omitted from prompts "
                "(360 train trials capped to 60 via max_prompt_train_trials). "
                "Token-level truncated=false in logs because capping happens before budget enforcement."
            )
        f8_losing = [
            p
            for p in participants
            if p.dataset == "8flesch2018comparing"
            and p.has_prompt_diagnostics
            and p.failure_category
        ]
        f8_win = [
            p
            for p in participants
            if p.dataset == "8flesch2018comparing"
            and p.has_prompt_diagnostics
            and p.teh_win
        ]
        if f8_losing and f8_win:
            lt = statistics.mean(
                [p.truncation_ratio for p in f8_losing if p.truncation_ratio is not None]
            )
            wt = statistics.mean(
                [p.truncation_ratio for p in f8_win if p.truncation_ratio is not None]
            )
            lines.append(
                f"- Within 8flesch: losing avg truncation={lt:.3f}, winning={wt:.3f} "
                f"(delta {lt - wt:+.3f}) — "
                + (
                    "losses align with truncation."
                    if lt - wt >= _TRUNCATION_CORRELATION_MIN
                    else "all participants share similar omission; losses are not selective by truncation."
                )
            )

    lines.append("")
    lines.append("### Prompt-budget recommendations (evidence-gated)")

    losing_more_truncated = (
        trunc.losing_mean_truncation is not None
        and trunc.winning_mean_truncation is not None
        and (trunc.losing_mean_truncation - trunc.winning_mean_truncation)
        >= _TRUNCATION_CORRELATION_MIN
    )
    positive_gap_corr = (
        trunc.corr_truncation_ratio_gap is not None
        and trunc.corr_truncation_ratio_gap >= 0.15
    )
    positive_inc_corr = (
        trunc.corr_included_fraction_test_loglik is not None
        and trunc.corr_included_fraction_test_loglik >= 0.15
    )

    if losing_more_truncated or positive_gap_corr or positive_inc_corr:
        high_trunc_ds = [
            s
            for s in ok_summaries
            if (s.avg_truncation_ratio or 0) >= _SEVERE_TRUNCATION_RATIO
            and s.severe_truncation_count >= 5
        ]
        names = ", ".join(s.dataset for s in high_trunc_ds) if high_trunc_ds else "(see above)"
        lines.append(
            f"- **Prompt budget vs iterations**: On {names}, increasing "
            "max_prompt_train_trials / hard_prompt_token_cap may help more than "
            "n_iterations when trial omission is high. Evidence: losing participants "
            "show higher truncation or positive truncation–gap correlation."
        )
    else:
        lines.append(
            "- **Prompt budget vs iterations**: No global evidence that losing participants "
            "are more truncated; prefer iteration/modeling levers unless dataset-specific "
            "stats above show high omission."
        )

    near_limit_n = sum(1 for p in participants if p.prompt_near_limit)
    if near_limit_n >= 10:
        lines.append(
            f"- **max_parent_chars (3500)**: {near_limit_n} participants hit "
            f">={_PROMPT_NEAR_LIMIT_FRAC:.0%} of hard_prompt_token_cap — parent code may "
            "crowd out trials when combined with large trial corpora. Consider ablation "
            "reducing parent context on high-omission datasets only."
        )
    else:
        lines.append(
            "- **max_parent_chars (3500)**: Few prompts near token cap — parent crowding is "
            "not strongly supported as a global issue (trial caps dominate on 8flesch)."
        )

    if losing_more_truncated:
        lines.append(
            "- **explore_candidates / fresh_n_candidates**: Indirect risk — larger parent "
            "pools can shrink trial space under a fixed cap. Only experiment if truncation "
            "correlates with loss on that dataset; otherwise keep current values."
        )
    else:
        lines.append(
            "- **explore_candidates / fresh_n_candidates**: No evidence that current "
            "values worsen truncation relative to wins; do not reduce solely for truncation."
        )

    structural = [
        s
        for s in ok_summaries
        if (s.avg_truncation_ratio or 0) >= _SEVERE_TRUNCATION_RATIO
        and (s.avg_included_fraction or 1) <= 0.5
    ]
    if structural:
        lines.append(
            "- **Prioritize train trials over parent diversity** on: "
            + ", ".join(s.dataset for s in structural)
            + " — many trials/problems omitted structurally (low included_fraction)."
        )
    lines.append("")


def _build_recommendations(
    participants: Sequence[_ParticipantRow],
    summaries: Sequence[_DatasetSummary],
    *,
    loss_margin: float,
    trunc: _TruncationCorrelation,
) -> str:
    """Evidence-based text recommendations; no hyperparameter claims without stats."""
    lines: List[str] = []
    lines.append("Psych-101 TEH grand analysis — evidence-based recommendations")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        f"Settings: loss_margin={loss_margin}, overfit gap>{_OVERFIT_TRAIN_TEST_GAP}, "
        f"weak train<={_WEAK_TRAIN_LOGLIK}, tail_converged>={_MIN_TAIL_CONVERGED}."
    )
    lines.append(
        "Well-tested hyperparameters (n_candidates, sample_parents, sample_size, "
        "max_prompt_*, refinement_val_threshold) are not discussed unless strong "
        "counter-evidence appears."
    )
    lines.append("")

    ok_summaries = [s for s in summaries if not s.error]
    all_losing = [p for p in participants if p.failure_category]
    n_losing = len(all_losing)
    n_search = sum(1 for p in all_losing if p.search_budget_failure)
    n_conv_lose = sum(1 for p in all_losing if p.converged_but_losing)
    n_overfit = sum(1 for p in participants if p.possible_overfit)
    n_weak = sum(1 for p in participants if p.weak_train_fit)

    lines.append("## Global failure mix (participants with TEH losing vs best baseline)")
    lines.append(
        f"- Total losing participants: {n_losing}"
    )
    if n_losing:
        lines.append(
            f"- search_budget_failure: {n_search} ({_pct(n_search, n_losing)}) "
            f"(low convergence: not probably_enough or tail<{_MIN_TAIL_CONVERGED})"
        )
        lines.append(
            f"- converged_but_losing: {n_conv_lose} ({_pct(n_conv_lose, n_losing)})"
        )
    lines.append(f"- possible_overfit (all participants): {n_overfit}")
    lines.append(f"- weak_train_fit (all participants): {n_weak}")
    lines.append("")

    if n_losing:
        if n_search > n_conv_lose:
            lines.append(
                "**Overall:** Failures skew toward insufficient search/convergence "
                f"({n_search} vs {n_conv_lose} converged-but-losing). "
                "More iterations/refinement may help datasets with many search_budget failures."
            )
        elif n_conv_lose > n_search:
            lines.append(
                "**Overall:** Failures skew toward converged-but-still-losing "
                f"({n_conv_lose} vs {n_search} search-budget). "
                "Raising n_iterations alone is unlikely to fix most losses; focus on "
                "modeling, prompts, and program representation."
            )
        else:
            lines.append(
                "**Overall:** Search-budget and converged-but-losing failures are "
                "similar in count; tailor per dataset (see below)."
            )
    lines.append("")

    # Per-dataset: more iterations may help vs won't help
    lines.append("## Per-dataset: iteration / refinement outlook")
    for s in sorted(ok_summaries, key=lambda x: x.dataset):
        n = s.n_participants or 1
        lose_n = s.search_budget_failure_count + s.converged_but_losing_count
        if lose_n == 0:
            lines.append(
                f"- **{s.dataset}**: TEH wins or within margin for all comparable "
                f"participants (win rate { _pct(s.teh_win_count, n) })."
            )
            continue
        sb_rate = s.search_budget_failure_count / lose_n if lose_n else 0
        cb_rate = s.converged_but_losing_count / lose_n if lose_n else 0
        prob = s.probably_enough_rate
        prob_s = f"{prob:.1%}" if prob is not None else "n/a"
        if sb_rate >= 0.5 and (s.probably_enough_rate or 0) < 0.7:
            lines.append(
                f"- **{s.dataset}**: MORE search/refinement may help — "
                f"{s.search_budget_failure_count}/{lose_n} losing participants are "
                f"search_budget failures; probably_enough rate {prob_s}; "
                f"avg tail_converged_steps={s.avg_tail_converged_steps:.2f}"
                if s.avg_tail_converged_steps is not None
                else f"search_budget failures; probably_enough {prob_s}."
            )
        elif cb_rate >= 0.5 and (s.probably_enough_rate or 0) >= 0.7:
            lines.append(
                f"- **{s.dataset}**: More iterations likely will NOT help most losses — "
                f"{s.converged_but_losing_count}/{lose_n} converged_but_losing; "
                f"probably_enough {prob_s}; dominant baseline {s.dominant_baseline_method} "
                f"({s.dominant_baseline_count}/{n} best-participant wins)."
            )
        else:
            lines.append(
                f"- **{s.dataset}**: Mixed — search_budget {s.search_budget_failure_count}, "
                f"converged_but_losing {s.converged_but_losing_count} "
                f"(of {lose_n} losing); probably_enough {prob_s}."
            )
    lines.append("")

    # 8flesch and 7hilbig deep dives
    f8 = next((s for s in ok_summaries if s.dataset == "8flesch2018comparing"), None)
    h7 = next((s for s in ok_summaries if s.dataset == "7hilbig2014generalized"), None)
    lines.append("## Dataset deep dives")
    if f8:
        lose = f8.search_budget_failure_count + f8.converged_but_losing_count
        lines.append(f"### 8flesch2018comparing")
        lines.append(
            f"- TEH avg test {f8.avg_teh_test:.4f}, avg gap to best {f8.avg_gap_to_best:.4f}, "
            f"win rate {_pct(f8.teh_win_count, f8.n_participants)}."
            if f8.avg_teh_test is not None and f8.avg_gap_to_best is not None
            else f"- TEH metrics available in dataset summary CSV."
        )
        conv_f8 = (
            f"- Convergence: probably_enough rate {f8.probably_enough_rate:.1%}"
            if f8.probably_enough_rate is not None
            else "- Convergence: probably_enough n/a"
        )
        if f8.avg_tail_converged_steps is not None:
            conv_f8 += f", avg tail_converged_steps {f8.avg_tail_converged_steps:.2f}."
        lines.append(conv_f8)
        lines.append(
            f"- Losing breakdown: search_budget {f8.search_budget_failure_count}, "
            f"converged_but_losing {f8.converged_but_losing_count} (of {lose} losing)."
        )
        lines.append(
            f"- Dominant baseline: {f8.dominant_baseline_method} "
            f"({f8.dominant_baseline_count}/{f8.n_participants} participants)."
        )
        if f8.probably_enough_rate is not None and f8.probably_enough_rate < 0.5:
            verdict = (
                "Primarily a **search-budget** issue (low probably_enough), with an "
                "additional **modeling** gap vs Centaur."
            )
        elif f8.search_budget_failure_count > f8.converged_but_losing_count:
            verdict = (
                "**Both**: search-budget failures dominate, but Centaur also leads — "
                "improve exploration and representation."
            )
        else:
            verdict = (
                "**Both**: substantial converged-but-losing mass suggests modeling limits "
                "even where training plateaued; low avg tail steps still indicate many "
                "runs did not fully plateau."
            )
        lines.append(f"- Assessment: {verdict}")
        if f8.avg_truncation_ratio is not None and f8.avg_truncation_ratio >= _SEVERE_TRUNCATION_RATIO:
            lines.append(
                f"- Truncation: avg trial omission {f8.avg_truncation_ratio:.1%} — "
                "prompt trial cap likely limits learning signal (see truncation section)."
            )
    if h7:
        lose = h7.search_budget_failure_count + h7.converged_but_losing_count
        lines.append("")
        lines.append(f"### 7hilbig2014generalized")
        lines.append(
            f"- TEH avg test {h7.avg_teh_test:.4f}, gap {h7.avg_gap_to_best:.4f}, "
            f"win rate {_pct(h7.teh_win_count, h7.n_participants)}; "
            f"dominant baseline {h7.dominant_baseline_method} "
            f"({h7.dominant_baseline_count}/{h7.n_participants})."
            if h7.avg_teh_test is not None
            else f"- See dataset summary."
        )
        prob_h = (
            f"{h7.probably_enough_rate:.1%}"
            if h7.probably_enough_rate is not None
            else "n/a"
        )
        lines.append(
            f"- Convergence: probably_enough {prob_h}, "
            f"converged_but_losing {h7.converged_but_losing_count}/{lose} losing."
        )
        if (h7.probably_enough_rate or 0) >= 0.9 and h7.converged_but_losing_count >= h7.search_budget_failure_count:
            lines.append(
                "- Assessment: Looks like a **modeling / baseline-strength** issue despite "
                "convergence — TEH training plateaus but Centaur wins most participants."
            )
        else:
            lines.append(
                "- Assessment: Mixed convergence and modeling factors; see per-participant CSV."
            )
    lines.append("")

    # Baseline dominance (dataset-level weakness)
    lines.append("## Dataset-level baseline dominance")
    for s in sorted(ok_summaries, key=lambda x: -x.dominant_baseline_count):
        if not s.dominant_baseline_method:
            continue
        share = s.dominant_baseline_count / (s.n_participants or 1)
        if share >= 0.5:
            lines.append(
                f"- **{s.dataset}**: {s.dominant_baseline_method} is best for "
                f"{s.dominant_baseline_count}/{s.n_participants} participants "
                f"({share:.0%}) — TEH may be capped by task–baseline fit, not search alone."
            )
    lines.append("")

    # Cautious hyperparameter notes (only with evidence)
    lines.append("## Cautious hyperparameter notes (evidence-gated)")
    lines.append(
        "Suggestions below reference only failure/convergence statistics from this run. "
        "No change recommended unless the stated condition holds."
    )

    low_conv_datasets = [
        s
        for s in ok_summaries
        if s.probably_enough_rate is not None
        and s.probably_enough_rate < 0.6
        and s.search_budget_failure_count >= 5
    ]
    high_conv_lose_datasets = [
        s
        for s in ok_summaries
        if s.probably_enough_rate is not None
        and s.probably_enough_rate >= 0.85
        and s.converged_but_losing_count >= 5
    ]

    if low_conv_datasets:
        names = ", ".join(s.dataset for s in low_conv_datasets)
        lines.append(
            f"- **--explore_candidates / --elite_pool_size / --fresh_n_candidates**: "
            f"Consider a modest increase only for datasets with low probably_enough and "
            f"many search_budget failures: {names}. Evidence: probably_enough < 60% "
            f"and >=5 search_budget failures."
        )
    else:
        lines.append(
            "- **--explore_candidates / --elite_pool_size / --fresh_n_candidates**: "
            "No strong evidence to increase globally; most datasets show adequate convergence."
        )

    if high_conv_lose_datasets:
        names = ", ".join(s.dataset for s in high_conv_lose_datasets)
        lines.append(
            f"- **--refinement_iters / --n_iterations**: Unlikely to help on {names} — "
            f"high probably_enough with >=5 converged_but_losing cases. Prefer prompt/program "
            f"changes over longer runs."
        )
    else:
        lines.append(
            "- **--refinement_iters / --n_iterations**: No dataset shows both high convergence "
            "and heavy converged-but-losing load at the threshold used; iteration bumps are "
            "not broadly supported."
        )

    early_stop_candidates = [
        s
        for s in ok_summaries
        if s.avg_tail_converged_steps is not None
        and s.avg_tail_converged_steps >= 10
        and s.probably_enough_rate is not None
        and s.probably_enough_rate >= 0.9
        and s.converged_but_losing_count >= 3
    ]
    if early_stop_candidates:
        names = ", ".join(s.dataset for s in early_stop_candidates)
        lines.append(
            f"- **--early_stop_iters**: On {names}, training often plateaus early "
            f"(high tail_converged_steps and probably_enough) yet TEH still loses — "
            "reducing early_stop_iters is NOT supported (would not address modeling gap). "
            "If anything, early stopping is already triggering late; do not shorten without "
            "new underfitting evidence."
        )
    else:
        lines.append(
            "- **--early_stop_iters**: Insufficient pattern of early plateau + losses to "
            "recommend changing early_stop_iters."
        )

    max_parent_note = False
    for s in ok_summaries:
        if s.weak_train_fit_count >= 5 and (s.probably_enough_rate or 0) < 0.7:
            max_parent_note = True
            break
    if max_parent_note:
        lines.append(
            "- **--max_parent_chars**: Some datasets show weak_train_fit with low "
            "convergence — if programs are truncated, consider monitoring program length; "
            "statistical link only, not confirmed from this CSV alone."
        )

    _append_truncation_recommendations(lines, participants, summaries, trunc)

    lines.append("## Suggested next experiments")
    worst = sorted(
        [s for s in ok_summaries if s.avg_gap_to_best is not None],
        key=lambda s: s.avg_gap_to_best,
        reverse=True,
    )[:3]
    for s in worst:
        action = (
            "increase search (explore/elite/fresh) and verify convergence"
            if s.search_budget_failure_count > s.converged_but_losing_count
            else "prompt/modeling ablations vs "
            f"{s.dominant_baseline_method}"
        )
        lines.append(
            f"- **{s.dataset}** (avg gap {s.avg_gap_to_best:.4f}): {action}."
        )
    best = sorted(
        [s for s in ok_summaries if s.avg_gap_to_best is not None],
        key=lambda s: s.avg_gap_to_best,
    )[:2]
    if best:
        lines.append(
            "- Strong TEH datasets (small avg gap): "
            + ", ".join(f"{s.dataset} ({s.avg_gap_to_best:.4f})" for s in best)
            + " — use as regression checks when tuning."
        )

    return "\n".join(lines) + "\n"


def _print_terminal_summary(
    participants: Sequence[_ParticipantRow],
    summaries: Sequence[_DatasetSummary],
    *,
    loss_margin: float,
    trunc: _TruncationCorrelation,
) -> None:
    ok = [s for s in summaries if not s.error and s.avg_gap_to_best is not None]
    print("\n=== grand_analysis.py summary ===")
    print(f"loss_margin={loss_margin}")
    if ok:
        by_gap = sorted(ok, key=lambda s: s.avg_gap_to_best or 0)
        print("\nBest datasets for TEH (smallest avg gap to best baseline):")
        for s in by_gap[:3]:
            print(
                f"  {s.dataset}: gap={s.avg_gap_to_best:.4f}, "
                f"win_rate={s.teh_win_rate:.1%}"
                if s.teh_win_rate is not None
                else f"  {s.dataset}: gap={s.avg_gap_to_best:.4f}"
            )
        print("\nWorst datasets (largest avg gap):")
        for s in by_gap[-3:][::-1]:
            print(
                f"  {s.dataset}: gap={s.avg_gap_to_best:.4f}, "
                f"search_budget={s.search_budget_failure_count}, "
                f"conv_losing={s.converged_but_losing_count}"
            )

    n_search = sum(1 for p in participants if p.search_budget_failure)
    n_conv = sum(1 for p in participants if p.converged_but_losing)
    n_over = sum(1 for p in participants if p.possible_overfit)
    n_weak = sum(1 for p in participants if p.weak_train_fit)
    n_win = sum(1 for p in participants if p.teh_win)
    print("\nFailure category counts (participants):")
    print(f"  teh_win: {n_win}")
    print(f"  search_budget_failure: {n_search}")
    print(f"  converged_but_losing: {n_conv}")
    print(f"  possible_overfit: {n_over}")
    print(f"  weak_train_fit: {n_weak}")

    if trunc.n_with_diagnostics:
        print("\nPrompt truncation (trial omission from diagnostics):")
        if (
            trunc.losing_mean_truncation is not None
            and trunc.winning_mean_truncation is not None
        ):
            print(
                f"  losing avg truncation_ratio: {trunc.losing_mean_truncation:.3f}"
            )
            print(
                f"  winning avg truncation_ratio: {trunc.winning_mean_truncation:.3f}"
            )
        if trunc.corr_included_fraction_test_loglik is not None:
            print(
                f"  r(included_fraction, test_loglik): "
                f"{trunc.corr_included_fraction_test_loglik:.3f}"
            )
        severe = sum(1 for p in participants if p.severe_truncation)
        print(f"  severe_truncation participants: {severe}")

    print("\nSuggested next experiments (see grand_analysis_recommendations.txt for detail):")
    losing_ds = sorted(
        ok,
        key=lambda s: (s.avg_gap_to_best or 0),
        reverse=True,
    )[:2]
    for s in losing_ds:
        if s.search_budget_failure_count > s.converged_but_losing_count:
            print(
                f"  {s.dataset}: prioritize search budget "
                f"({s.search_budget_failure_count} search failures)"
            )
        else:
            print(
                f"  {s.dataset}: prioritize modeling vs {s.dominant_baseline_method} "
                f"({s.converged_but_losing_count} converged-but-losing)"
            )

    err = [s for s in summaries if s.error]
    for s in err:
        print(f"\nERROR {s.dataset}: {s.error}")


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description="Grand TEH vs baseline analysis for psych-101 train + mixed_gambles."
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset alias. Required unless --all_in.",
    )
    p.add_argument(
        "--all_in",
        action="store_true",
        help="Analyze all train Psych-101 datasets plus mixed_gambles.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(_DEFAULT_OUT_DIR),
        help=f"Output directory (default: {_DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--convergence_csv",
        type=Path,
        default=Path(_DEFAULT_CONVERGENCE_CSV),
        help=f"iteration_convergence.csv from iter.py (default: {_DEFAULT_CONVERGENCE_CSV}).",
    )
    p.add_argument(
        "--baseline_config",
        type=Path,
        default=Path(_DEFAULT_BASELINE_CONFIG),
        help="Baseline paths YAML (same as compare.py).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Convergence threshold (informational; iter.py CSV already computed).",
    )
    p.add_argument(
        "--loss_margin",
        type=float,
        default=0.02,
        help="TEH counts as 'win' if within this loglik of best baseline.",
    )
    args = p.parse_args()

    if not args.all_in and args.dataset is None:
        raise SystemExit("Provide --dataset or use --all_in.")

    out_dir = Path(args.out_dir).expanduser()
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (repo / out_dir).resolve()

    conv_path = Path(args.convergence_csv).expanduser()
    conv_path = (
        conv_path.resolve()
        if conv_path.is_absolute()
        else (repo / conv_path).resolve()
    )
    convergence = _load_convergence_csv(conv_path)
    if not convergence:
        print(
            f"Warning: no convergence data at {conv_path}. "
            "Run: python analysis/code/psych-101/iter.py --all_in",
            file=sys.stderr,
        )

    config_path = Path(args.baseline_config).expanduser()
    config_path = (
        config_path.resolve()
        if config_path.is_absolute()
        else (repo / config_path).resolve()
    )
    config_data = cmp._load_baseline_config_file(config_path)

    datasets = list(_ALL_IN_DATASETS) if args.all_in else [_normalize_dataset(args.dataset)]

    all_participants: List[_ParticipantRow] = []
    all_summaries: List[_DatasetSummary] = []

    for ds in datasets:
        alias = _normalize_dataset(ds)
        if args.all_in:
            print(f"[--all_in] {alias} ...", flush=True)
        try:
            rows, summary = _analyze_dataset(
                repo,
                alias,
                config_data=config_data,
                convergence=convergence,
                loss_margin=float(args.loss_margin),
                quiet=args.all_in,
            )
            all_participants.extend(rows)
            all_summaries.append(summary)
        except (OSError, ValueError) as exc:
            msg = str(exc) or type(exc).__name__
            print(f"ERROR {alias}: {msg}", file=sys.stderr)
            all_summaries.append(_DatasetSummary(dataset=alias, error=msg))

    part_csv = out_dir / "grand_analysis_participants.csv"
    ds_csv = out_dir / "grand_analysis_dataset_summary.csv"
    fail_csv = out_dir / "grand_analysis_failure_cases.csv"
    rec_path = out_dir / "grand_analysis_recommendations.txt"

    _write_csv(
        part_csv,
        _PARTICIPANT_FIELDS,
        [_participant_to_dict(r) for r in all_participants],
    )
    _write_csv(
        ds_csv,
        _DATASET_SUMMARY_FIELDS,
        [_dataset_summary_to_dict(s) for s in all_summaries],
    )
    _write_csv(fail_csv, _FAILURE_FIELDS, _failure_rows(all_participants))

    trunc_corr = _analyze_truncation_correlations(all_participants)

    rec_text = _build_recommendations(
        all_participants,
        all_summaries,
        loss_margin=float(args.loss_margin),
        trunc=trunc_corr,
    )
    rec_path.write_text(rec_text, encoding="utf-8")

    print(f"\nWrote {len(all_participants)} participant rows -> {part_csv}")
    print(f"Wrote {len(all_summaries)} dataset summaries -> {ds_csv}")
    print(f"Wrote failure cases -> {fail_csv}")
    print(f"Wrote recommendations -> {rec_path}")

    _print_terminal_summary(
        all_participants,
        all_summaries,
        loss_margin=float(args.loss_margin),
        trunc=trunc_corr,
    )


if __name__ == "__main__":
    main()
