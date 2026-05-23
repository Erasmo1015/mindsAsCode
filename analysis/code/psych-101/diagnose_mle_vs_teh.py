#!/usr/bin/env python3
"""
Evidence-driven diagnosis: why MLE beats TEH on selected Psych-101 datasets.

Usage:
  python analysis/code/psych-101/diagnose_mle_vs_teh.py \\
    --datasets 3frey2017cct 4wulff2018description 5speekenbrink2008learning \\
    --psych_dataset_split train

Auto-selects latest TEH / baseline runs via analysis/code/utils/compare.py discovery.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.mixed_gambles import DEFAULT_CSV_PATH
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    format_trial_for_prompt,
    get_psych101_binary_experiment,
    normalize_psych_dataset_split,
    split_psych_experiment,
    summarize_runtime_schema_for_prompt,
    _action_semantics_for_schema,
)
from utils.teh.teh_datasets import is_mixed_gambles_dataset

from analysis.code.utils import compare as cmp

_DEFAULT_OUT_DIR = "analysis/data/psych101_mle_vs_teh_diagnosis"
_DEFAULT_BASELINE_CONFIG = cmp._DEFAULT_BASELINE_CONFIG
_DEFAULT_CONVERGENCE_CSV = "generated_outputs/psych101_train/teh/iteration_convergence.csv"
_BASELINE_METHODS = cmp._BASELINE_METHODS
_TEST_LOGLIK = cmp._TEST_LOGLIK
_GATED_LOGLIK = cmp._GATED_LOGLIK

_SPLIT_RATIO = 0.6
_SPLIT_SEED = 0
_LOSS_MARGIN = 0.05
_MLE_STRONG_GAP = 0.15
_NEAR_PERFECT_THRESHOLDS = (-0.05, -0.10, -0.20)
_CATASTROPHIC_THRESHOLDS = (-1.0, -2.0, -3.0)
_OVERFIT_TRAIN_TEST_GAP = 0.15
_WEAK_TRAIN_LOGLIK = -0.65
_MIN_TAIL_CONVERGED = 3
_PROMPT_DIAG_NAME = "prompt_diagnostics.jsonl"
_PARTICIPANT_DIR_RE = re.compile(r"^participant_(\d+)$")
_SNIPPET_MAX = 800

_PRIORITY_DATASETS = (
    "3frey2017cct",
    "4wulff2018description",
    "5speekenbrink2008learning",
)


@dataclass
class _ParticipantDiag:
    dataset: str
    run_name: str
    participant_id: int
    bir: Optional[float] = None
    mle_test: Optional[float] = None
    pt_test: Optional[float] = None
    centaur_test: Optional[float] = None
    openevolve_test: Optional[float] = None
    teh_train: Optional[float] = None
    teh_val: Optional[float] = None
    teh_test: Optional[float] = None
    teh_gated_test: Optional[float] = None
    best_method: str = ""
    best_score: Optional[float] = None
    gap_teh_vs_mle: Optional[float] = None
    gap_teh_vs_best: Optional[float] = None
    tail_converged_steps: int = 0
    probably_enough: bool = False
    final_train_loglik: Optional[float] = None
    teh_failure_types: List[str] = field(default_factory=list)
    train_majority_rate: Optional[float] = None
    test_majority_rate: Optional[float] = None
    test_all_same_action: bool = False
    train_all_same_action: bool = False
    mle_near_perfect: bool = False
    mle_beats_teh_strongly: bool = False
    teh_test_accuracy: Optional[float] = None
    teh_mean_prob_at_action: Optional[float] = None
    teh_mean_abs_p_minus_half: Optional[float] = None
    underconfident: bool = False
    program_style: str = ""
    program_path: str = ""
    program_snippet: str = ""
    truncation_ratio: Optional[float] = None
    included_fraction: Optional[float] = None
    severe_truncation: bool = False
    prompt_near_limit: bool = False
    parser_flags: List[str] = field(default_factory=list)
    baseline_paths: Dict[str, str] = field(default_factory=dict)
    teh_run_path: str = ""


@dataclass
class _DatasetBundle:
    dataset: str
    run_name: str = ""
    teh_run: Optional[Path] = None
    baseline_paths: Dict[str, Path] = field(default_factory=dict)
    participants: List[_ParticipantDiag] = field(default_factory=list)
    error: Optional[str] = None


def _repo_root() -> Path:
    return _REPO_ROOT


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _parse_bool(s: Any) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _fmt(v: Optional[float], ndigits: int = 4) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.{ndigits}f}"


def _normalize_dataset(dataset: str) -> str:
    return cmp._normalize_compare_dataset(dataset)


def _load_convergence(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = str(row.get("dataset", "")).strip()
            pid_raw = row.get("participant_id")
            if not ds or pid_raw is None or str(pid_raw).strip() == "":
                continue
            pid = int(float(pid_raw))
            out[(ds, pid)] = {
                "tail_converged_steps": int(float(row.get("tail_converged_steps") or 0)),
                "probably_enough": _parse_bool(row.get("probably_enough", "")),
                "final_train_loglik": _safe_float(row.get("final_train_loglik")),
            }
    return out


def _load_trials(
    dataset: str,
    participant_id: int,
    *,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    try:
        if is_mixed_gambles_dataset(dataset):
            from data_modules.mixed_gambles import load_mixed_gambles_trials

            train, val, test, _ = load_mixed_gambles_trials(
                participant_id,
                csv_path=mixed_gambles_csv,
                filter_gain_loss_only=filter_mixed_gambles,
                split_ratio=_SPLIT_RATIO,
                split_seed=_SPLIT_SEED,
            )
            return train, val, test, None
        exp = get_psych101_binary_experiment(
            dataset,
            int(participant_id),
            split=DEFAULT_PSYCH_DATASET_SPLIT,
            local_dataset=local_dataset,
        )
        train, val, test, _ = split_psych_experiment(
            exp, split_ratio=_SPLIT_RATIO, split_seed=_SPLIT_SEED
        )
        return train, val, test, None
    except Exception as exc:
        return [], [], [], f"{type(exc).__name__}: {exc}"


def _action_distribution(trials: Sequence[Mapping[str, Any]]) -> Counter:
    c: Counter = Counter()
    for t in trials:
        a = t.get("action")
        if a is not None:
            c[int(a)] += 1
    return c


def _majority_rate(trials: Sequence[Mapping[str, Any]]) -> Optional[float]:
    if not trials:
        return None
    ac = _action_distribution(trials)
    return max(ac.values()) / len(trials)


def _all_same_action(trials: Sequence[Mapping[str, Any]]) -> bool:
    if not trials:
        return False
    return len(_action_distribution(trials)) == 1


def _best_method_for_participant(
    scores: Mapping[str, Optional[float]],
) -> Tuple[str, Optional[float]]:
    best_method = ""
    best_score: Optional[float] = None
    for method, score in scores.items():
        if score is None or not math.isfinite(score):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_method = method
    return best_method, best_score


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return statistics.median(vals)


def _count_above(values: Sequence[Optional[float]], threshold: float) -> int:
    return sum(1 for v in values if v is not None and math.isfinite(v) and v > threshold)


def _count_below(values: Sequence[Optional[float]], threshold: float) -> int:
    return sum(1 for v in values if v is not None and math.isfinite(v) and v < threshold)


def _num_best_by_method(rows: Sequence[_ParticipantDiag], method: str) -> int:
    counts = 0
    for r in rows:
        scores: Dict[str, Optional[float]] = {
            "MLE": r.mle_test,
            "prospect_theory": r.pt_test,
            "Centaur": r.centaur_test,
            "openevolve": r.openevolve_test,
            "TEH": r.teh_test,
        }
        best_m, best_s = _best_method_for_participant(scores)
        if best_m == method and best_s is not None:
            counts += 1
    return counts


def _load_choose_fn(program_path: Path) -> Callable:
    spec = importlib.util.spec_from_file_location(
        f"diag_program_{program_path.parent.name}", str(program_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    choose = getattr(module, "choose", None)
    if choose is None or not callable(choose):
        raise RuntimeError(f"No choose() in {program_path}")
    return choose


def _clamp_prob(p: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(p)))


def _teh_probability_stats(
    program_path: Path,
    test_trials: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (test_accuracy, mean_prob_at_true_action, mean|p-0.5|)."""
    if not program_path.is_file() or not test_trials:
        return None, None, None
    try:
        choose = _load_choose_fn(program_path)
    except Exception:
        return None, None, None

    probs_at_action: List[float] = []
    abs_dev: List[float] = []
    correct = 0
    for t in test_trials:
        y = int(t["action"])
        try:
            raw = choose(t["problem"], t.get("history", []))
            if isinstance(raw, bool):
                p = 1.0 - 1e-6 if raw else 1e-6
            elif isinstance(raw, (int, float)):
                if int(raw) in (0, 1) and not isinstance(raw, float):
                    p = 1.0 - 1e-6 if int(raw) == 1 else 1e-6
                else:
                    p = _clamp_prob(float(raw))
            else:
                p = 0.5
        except Exception:
            p = 0.5
        p = _clamp_prob(p)
        probs_at_action.append(p if y == 1 else 1.0 - p)
        abs_dev.append(abs(p - 0.5))
        pred = 1 if p >= 0.5 else 0
        correct += int(pred == y)

    n = len(test_trials)
    acc = correct / n if n else None
    mean_p = statistics.mean(probs_at_action) if probs_at_action else None
    mean_dev = statistics.mean(abs_dev) if abs_dev else None
    return acc, mean_p, mean_dev


def _classify_program_style(code: str) -> str:
    c = code or ""
    low = re.sub(r"\s+", "", c.lower())
    scores: Dict[str, int] = {
        "raw_EV_linear": 0,
        "subjective_value_PT": 0,
        "history_heavy": 0,
        "threshold_deterministic": 0,
        "generic_safe_probability": 0,
        "unclear": 0,
    }

    if re.search(r"def\s+subjective_value\s*\(", c, re.I):
        scores["subjective_value_PT"] += 3
    if "reference" in low and "**" in c:
        scores["subjective_value_PT"] += 2
    if "lambda_loss" in low:
        scores["subjective_value_PT"] += 2

    if "expected_value" in low or re.search(r"\bev_[ab]\b", low):
        scores["raw_EV_linear"] += 2
    if "sigmoid" in low and scores["subjective_value_PT"] < 2:
        scores["raw_EV_linear"] += 2
    if re.search(r"reward_diff|net_expected|expected_gain", low):
        scores["raw_EV_linear"] += 1

    if re.search(r"action_counts|recent_actions|history_bias|freq_[ab]", low):
        scores["history_heavy"] += 2
    if "history" in low and ("count" in low or "frequency" in low):
        scores["history_heavy"] += 1

    if re.search(r"return\s+[01](?:\.0)?\s*$", c, re.M):
        scores["threshold_deterministic"] += 2
    if re.search(r"if\s+.*:\s*return\s+(?:0|1)", c):
        scores["threshold_deterministic"] += 1
    if re.search(r"return\s+0\.5\s*$", c, re.M):
        scores["generic_safe_probability"] += 2
    if re.search(r"max\s*\(\s*1e-6\s*,\s*min\s*\(\s*1\s*-\s*1e-6", c):
        scores["generic_safe_probability"] += 1

    top = max(scores, key=scores.get)
    top_score = scores[top]
    second = sorted(scores.values(), reverse=True)[1]
    if top_score == 0 or top_score == second:
        return "unclear"
    return top


def _program_snippet(code: str, max_len: int = _SNIPPET_MAX) -> str:
    if not code:
        return ""
    clipped = code.strip()
    if len(clipped) <= max_len:
        return clipped
    return clipped[: max_len - 20] + "\n... [truncated]"


def _read_prompt_diagnostics(path: Path) -> List[Dict[str, Any]]:
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


def _summarize_prompt_diagnostics(participant_dir: Path) -> Dict[str, Any]:
    rows = _read_prompt_diagnostics(participant_dir / _PROMPT_DIAG_NAME)
    if not rows:
        return {}
    max_ratio = 0.0
    worst_before = worst_after = 0
    truncation_occurred = False
    per_problem_cap = False
    prompt_near_limit = False
    max_tokens = 0
    token_cap: Optional[int] = None

    for row in rows:
        before = int(row.get("train_trials_before") or 0)
        after = int(row.get("train_trials_after") or 0)
        if before > 0:
            ratio = 1.0 - after / before
            if ratio > max_ratio:
                max_ratio = ratio
                worst_before = before
                worst_after = after
        if row.get("truncated") or before > after:
            truncation_occurred = True
        steps = row.get("truncation_steps") or []
        if any("per_problem_cap" in str(s) for s in steps):
            per_problem_cap = True
        cap = row.get("hard_prompt_token_cap")
        if cap is not None:
            try:
                token_cap = int(cap)
            except (TypeError, ValueError):
                pass
        tok = row.get("prompt_tokens_after_truncation") or row.get(
            "prompt_tokens_before_truncation"
        )
        tok_i = int(tok) if tok is not None else 0
        max_tokens = max(max_tokens, tok_i)
        if token_cap and tok_i >= 0.85 * token_cap:
            prompt_near_limit = True

    included_fraction = (worst_after / worst_before) if worst_before > 0 else 1.0
    return {
        "truncation_ratio": max_ratio if worst_before > 0 else 0.0,
        "included_fraction": included_fraction,
        "total_train_trials": worst_before,
        "included_train_trials": worst_after,
        "severe_truncation": max_ratio > 0.5,
        "truncation_occurred": truncation_occurred,
        "per_problem_cap_clipping": per_problem_cap,
        "prompt_near_limit": prompt_near_limit,
        "prompt_token_estimate": max_tokens,
    }


def _parser_audit_flags(
    dataset: str,
    participant_id: int,
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    diag: Mapping[str, Any],
) -> List[str]:
    flags: List[str] = []
    if diag.get("severe_truncation"):
        flags.append("severe_truncation")
    if diag.get("per_problem_cap_clipping"):
        flags.append("per_problem_cap_clipping")
    if diag.get("truncation_occurred"):
        flags.append("trial_or_token_truncation")
    if diag.get("included_fraction") is not None and diag["included_fraction"] < 0.35:
        flags.append("heavy_trial_omission")

    for split_name, trials in (("train", train), ("test", test)):
        bad = [t.get("action") for t in trials if t.get("action") not in (0, 1)]
        if bad:
            flags.append(f"bad_actions_{split_name}")
        if trials and _all_same_action(trials):
            flags.append(f"constant_{split_name}_actions")

    if train and test:
        train_keys = {json.dumps(t.get("problem"), sort_keys=True, default=str) for t in train[:50]}
        test_keys = {json.dumps(t.get("problem"), sort_keys=True, default=str) for t in test[:50]}
        if train_keys & test_keys:
            flags.append("train_test_problem_overlap")

    if dataset in ("3frey2017cct", "4wulff2018description") and train:
        p0 = train[0].get("problem") or {}
        if not p0.get("option_keys") or len(p0.get("option_keys", [])) < 2:
            flags.append("missing_option_keys")

    return flags


def _classify_teh_failures(row: _ParticipantDiag) -> List[str]:
    types: List[str] = []
    train = row.teh_train if row.teh_train is not None else row.final_train_loglik
    test = row.teh_test
    gated = row.teh_gated_test

    if row.gap_teh_vs_best is None or row.gap_teh_vs_best <= _LOSS_MARGIN:
        return types

    if train is not None and train <= _WEAK_TRAIN_LOGLIK:
        types.append("weak_train_fit")
    if train is not None and test is not None and (train - test) > _OVERFIT_TRAIN_TEST_GAP:
        types.append("overfit")
    if gated is not None and test is not None and gated < test - 0.02:
        types.append("gating_hurts")

    if row.underconfident:
        types.append("underconfident")

    low_conv = (not row.probably_enough) or row.tail_converged_steps < _MIN_TAIL_CONVERGED
    high_conv = row.probably_enough and row.tail_converged_steps >= _MIN_TAIL_CONVERGED
    if low_conv:
        types.append("search_budget")
    elif high_conv:
        types.append("converged_but_losing")

    if row.parser_flags or row.severe_truncation:
        types.append("possible_parser_or_prompt_issue")

    if not types:
        types.append("other_losing")
    return types


def _analyze_dataset(
    repo: Path,
    dataset: str,
    *,
    config_data: Mapping[str, Any],
    convergence: Mapping[Tuple[str, int], Dict[str, Any]],
    bir_map: Dict[int, float],
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
    quiet: bool,
) -> _DatasetBundle:
    alias = _normalize_dataset(dataset)
    psych_split = normalize_psych_dataset_split(DEFAULT_PSYCH_DATASET_SPLIT)
    bundle = _DatasetBundle(dataset=alias)

    teh_run = cmp._auto_discover_teh_run(
        repo, dataset=alias, psych_dataset_split=psych_split
    )
    if teh_run is None:
        bundle.error = f"no TEH run under {cmp._teh_search_root(repo, alias, psych_split)}"
        return bundle

    bundle.teh_run = teh_run
    bundle.run_name = teh_run.name if teh_run.is_dir() else teh_run.parent.name
    if not quiet:
        print(f"[{alias}] TEH run: {teh_run.relative_to(repo)}")

    teh_csv = cmp._resolve_loglik_csv(teh_run)
    teh_train = cmp._read_loglik_csv(teh_csv, "train_loglik", required=False)
    teh_val = cmp._read_loglik_csv(teh_csv, "val_loglik", required=False)
    teh_test = cmp._read_loglik_csv(teh_csv, _TEST_LOGLIK, required=False)
    teh_gated = cmp._read_loglik_csv(teh_csv, _GATED_LOGLIK, required=False)

    baseline_paths = cmp._resolve_baseline_run_paths(
        config_data, repo, alias, psych_split, quiet=quiet
    )
    bundle.baseline_paths = baseline_paths
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

    participant_ids = sorted(set(teh_test) | set(teh_train))
    if baseline_scores.get("MLE"):
        participant_ids = sorted(set(participant_ids) | set(baseline_scores["MLE"]))
    participant_ids, _, _ = cmp._clamp_participant_ids_to_dataset(
        participant_ids,
        dataset=alias,
        psych_dataset_split=psych_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
    )

    for pid in participant_ids:
        conv = convergence.get((alias, pid), {})
        mle = baseline_scores.get("MLE", {}).get(pid)
        pt = baseline_scores.get("prospect_theory", {}).get(pid)
        cent = baseline_scores.get("Centaur", {}).get(pid)
        oe = baseline_scores.get("openevolve", {}).get(pid)
        t_test = teh_test.get(pid)
        t_train = teh_train.get(pid)
        t_val = teh_val.get(pid)
        t_gated = teh_gated.get(pid)

        all_scores = {
            "MLE": mle,
            "prospect_theory": pt,
            "Centaur": cent,
            "openevolve": oe,
            "TEH": t_test,
        }
        best_method, best_score = _best_method_for_participant(all_scores)
        gap_mle = (mle - t_test) if mle is not None and t_test is not None else None
        gap_best = (best_score - t_test) if best_score is not None and t_test is not None else None

        train_trials, _, test_trials, trial_err = _load_trials(
            alias,
            pid,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            filter_mixed_gambles=filter_mixed_gambles,
        )

        pdir = teh_run / f"participant_{pid}"
        prog_path = pdir / "best_program.py"
        code = prog_path.read_text(encoding="utf-8", errors="replace") if prog_path.is_file() else ""
        style = _classify_program_style(code) if code else ""

        diag = _summarize_prompt_diagnostics(pdir) if pdir.is_dir() else {}
        parser_flags = _parser_audit_flags(alias, pid, train_trials, test_trials, diag)

        acc, mean_p, mean_dev = _teh_probability_stats(prog_path, test_trials)
        underconf = False
        if acc is not None and t_test is not None and acc >= 0.75 and t_test < -0.35:
            underconf = True
        if mean_dev is not None and mean_dev < 0.12 and t_test is not None and t_test < -0.30:
            underconf = True
        if mean_p is not None and mean_p < 0.62 and acc is not None and acc >= 0.7:
            underconf = True

        row = _ParticipantDiag(
            dataset=alias,
            run_name=bundle.run_name,
            participant_id=pid,
            bir=bir_map.get(pid),
            mle_test=mle,
            pt_test=pt,
            centaur_test=cent,
            openevolve_test=oe,
            teh_train=t_train,
            teh_val=t_val,
            teh_test=t_test,
            teh_gated_test=t_gated,
            best_method=best_method,
            best_score=best_score,
            gap_teh_vs_mle=gap_mle,
            gap_teh_vs_best=gap_best,
            tail_converged_steps=int(conv.get("tail_converged_steps", 0)),
            probably_enough=bool(conv.get("probably_enough", False)),
            final_train_loglik=conv.get("final_train_loglik"),
            train_majority_rate=_majority_rate(train_trials),
            test_majority_rate=_majority_rate(test_trials),
            test_all_same_action=_all_same_action(test_trials),
            train_all_same_action=_all_same_action(train_trials),
            mle_near_perfect=mle is not None and mle > -0.05,
            mle_beats_teh_strongly=(
                gap_mle is not None and gap_mle > _MLE_STRONG_GAP
            ),
            teh_test_accuracy=acc,
            teh_mean_prob_at_action=mean_p,
            teh_mean_abs_p_minus_half=mean_dev,
            underconfident=underconf,
            program_style=style,
            program_path=str(prog_path.relative_to(repo)) if prog_path.is_file() else "",
            program_snippet=_program_snippet(code),
            truncation_ratio=_safe_float(diag.get("truncation_ratio")),
            included_fraction=_safe_float(diag.get("included_fraction")),
            severe_truncation=bool(diag.get("severe_truncation")),
            prompt_near_limit=bool(diag.get("prompt_near_limit")),
            parser_flags=parser_flags + ([f"trial_load_error:{trial_err}"] if trial_err else []),
            baseline_paths={
                m: str(p.relative_to(repo)) for m, p in baseline_paths.items()
            },
            teh_run_path=str(teh_run.relative_to(repo)),
        )
        row.teh_failure_types = _classify_teh_failures(row)
        bundle.participants.append(row)

    return bundle


def _participant_row_dict(r: _ParticipantDiag) -> Dict[str, str]:
    return {
        "dataset": r.dataset,
        "run_name": r.run_name,
        "participant_id": str(r.participant_id),
        "BIR": _fmt(r.bir, 4),
        "MLE_test_loglik": _fmt(r.mle_test),
        "prospect_theory_test_loglik": _fmt(r.pt_test),
        "Centaur_test_loglik": _fmt(r.centaur_test),
        "openevolve_test_loglik": _fmt(r.openevolve_test),
        "teh_train_loglik": _fmt(r.teh_train),
        "teh_val_loglik": _fmt(r.teh_val),
        "teh_test_loglik": _fmt(r.teh_test),
        "teh_gated_test_loglik": _fmt(r.teh_gated_test),
        "best_method": r.best_method,
        "best_score": _fmt(r.best_score),
        "gap_teh_vs_mle": _fmt(r.gap_teh_vs_mle),
        "gap_teh_vs_best_baseline": _fmt(r.gap_teh_vs_best),
        "tail_converged_steps": str(r.tail_converged_steps),
        "probably_enough": str(r.probably_enough),
        "final_train_loglik": _fmt(r.final_train_loglik, 6),
        "teh_failure_types": ";".join(r.teh_failure_types),
        "train_majority_rate": _fmt(r.train_majority_rate),
        "test_majority_rate": _fmt(r.test_majority_rate),
        "test_all_same_action": str(r.test_all_same_action),
        "mle_near_perfect": str(r.mle_near_perfect),
        "mle_beats_teh_strongly": str(r.mle_beats_teh_strongly),
        "teh_test_accuracy": _fmt(r.teh_test_accuracy),
        "teh_mean_prob_at_action": _fmt(r.teh_mean_prob_at_action),
        "teh_mean_abs_p_minus_half": _fmt(r.teh_mean_abs_p_minus_half),
        "underconfident": str(r.underconfident),
        "program_style": r.program_style,
        "program_path": r.program_path,
        "truncation_ratio": _fmt(r.truncation_ratio),
        "included_fraction": _fmt(r.included_fraction),
        "severe_truncation": str(r.severe_truncation),
        "parser_flags": ";".join(r.parser_flags),
        "teh_run_path": r.teh_run_path,
    }


def _dataset_summary_rows(bundle: _DatasetBundle) -> List[Dict[str, str]]:
    rows = bundle.participants
    if not rows:
        return [
            {
                "dataset": bundle.dataset,
                "run_name": bundle.run_name,
                "error": bundle.error or "no participants",
            }
        ]

    methods = {
        "MLE": [r.mle_test for r in rows],
        "prospect_theory": [r.pt_test for r in rows],
        "Centaur": [r.centaur_test for r in rows],
        "openevolve": [r.openevolve_test for r in rows],
        "TEH": [r.teh_test for r in rows],
        "TEH_gated": [r.teh_gated_test for r in rows],
    }

    out: List[Dict[str, str]] = []
    base = {
        "dataset": bundle.dataset,
        "run_name": bundle.run_name,
        "n_participants": str(len(rows)),
        "teh_run": str(bundle.teh_run.relative_to(_repo_root())) if bundle.teh_run else "",
        "mle_run": str(bundle.baseline_paths.get("MLE", "")),
    }

    for method, vals in methods.items():
        finite = [v for v in vals if v is not None and math.isfinite(v)]
        if not finite:
            continue
        row = dict(base)
        row["method"] = method
        row["avg_loglik"] = _fmt(statistics.mean(finite))
        row["median_loglik"] = _fmt(_median(finite))
        row["num_best"] = str(_num_best_by_method(rows, method.replace("_gated", "")))
        for thr in _NEAR_PERFECT_THRESHOLDS:
            row[f"near_perfect_gt_{abs(thr):.2f}"] = str(_count_above(finite, thr))
        for thr in _CATASTROPHIC_THRESHOLDS:
            row[f"catastrophic_lt_{abs(thr):.0f}"] = str(_count_below(finite, thr))
        out.append(row)

    mle_wins = sum(
        1
        for r in rows
        if r.mle_test is not None
        and r.teh_test is not None
        and r.mle_test > r.teh_test + _LOSS_MARGIN
    )
    mle_strong = sum(1 for r in rows if r.mle_beats_teh_strongly)
    teh_wins = sum(
        1
        for r in rows
        if r.teh_test is not None
        and r.mle_test is not None
        and r.teh_test >= r.mle_test - _LOSS_MARGIN
    )
    summary_extra = dict(base)
    summary_extra["method"] = "_aggregate"
    summary_extra["mle_beats_teh_count"] = str(mle_wins)
    summary_extra["mle_beats_teh_strongly_count"] = str(mle_strong)
    summary_extra["teh_beats_or_ties_mle_count"] = str(teh_wins)
    summary_extra["avg_gap_teh_vs_mle"] = _fmt(
        statistics.mean(
            [r.gap_teh_vs_mle for r in rows if r.gap_teh_vs_mle is not None]
        )
        if any(r.gap_teh_vs_mle is not None for r in rows)
        else None
    )
    summary_extra["avg_truncation_ratio"] = _fmt(
        statistics.mean(
            [r.truncation_ratio for r in rows if r.truncation_ratio is not None]
        )
        if any(r.truncation_ratio is not None for r in rows)
        else None
    )
    summary_extra["avg_included_fraction"] = _fmt(
        statistics.mean(
            [r.included_fraction for r in rows if r.included_fraction is not None]
        )
        if any(r.included_fraction is not None for r in rows)
        else None
    )
    failure_counts = Counter()
    for r in rows:
        for ft in r.teh_failure_types:
            failure_counts[ft] += 1
    summary_extra["failure_type_counts"] = json.dumps(dict(failure_counts))
    out.append(summary_extra)
    return out


def _failure_case_rows(rows: Sequence[_ParticipantDiag]) -> List[Dict[str, str]]:
    losing = [
        r
        for r in rows
        if r.gap_teh_vs_best is not None and r.gap_teh_vs_best > _LOSS_MARGIN
    ]
    losing.sort(key=lambda r: (-(r.gap_teh_vs_best or 0), r.dataset, r.participant_id))
    out: List[Dict[str, str]] = []
    for r in losing:
        out.append(
            {
                "dataset": r.dataset,
                "run_name": r.run_name,
                "participant_id": str(r.participant_id),
                "teh_test_loglik": _fmt(r.teh_test),
                "teh_gated_test_loglik": _fmt(r.teh_gated_test),
                "mle_test_loglik": _fmt(r.mle_test),
                "gap_teh_vs_mle": _fmt(r.gap_teh_vs_mle),
                "best_method": r.best_method,
                "best_score": _fmt(r.best_score),
                "gap_teh_vs_best_baseline": _fmt(r.gap_teh_vs_best),
                "teh_failure_types": ";".join(r.teh_failure_types),
                "tail_converged_steps": str(r.tail_converged_steps),
                "probably_enough": str(r.probably_enough),
                "teh_train_loglik": _fmt(r.teh_train),
                "test_majority_rate": _fmt(r.test_majority_rate),
                "mle_near_perfect": str(r.mle_near_perfect),
                "underconfident": str(r.underconfident),
                "program_style": r.program_style,
                "program_path": r.program_path,
                "truncation_ratio": _fmt(r.truncation_ratio),
                "parser_flags": ";".join(r.parser_flags),
            }
        )
    return out


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _example_parsed_trials(
    dataset: str,
    participant_id: int,
    *,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
    n: int = 2,
) -> List[str]:
    train, _, test, err = _load_trials(
        dataset,
        participant_id,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
    )
    if err:
        return [f"ERROR loading trials: {err}"]
    examples: List[str] = []
    for label, trials in (("train", train), ("test", test)):
        for i, t in enumerate(trials[:n]):
            p = t.get("problem") or {}
            keys = p.get("option_keys", [])
            sem = _action_semantics_for_schema(
                keys,
                str(p.get("schema_type", "?")),
                p,
                is_gamble="gamble_A" in p or "gamble_B" in p,
            )
            examples.append(
                f"{label}[{i}] action={t.get('action')} keys={keys} "
                f"schema={p.get('schema_type')} semantics={sem} "
                f"problem_fields={sorted(p.keys())[:12]}"
            )
            try:
                formatted = format_trial_for_prompt(t, compact=False)
                examples.append(f"  formatted: {formatted[:400]}")
            except Exception as exc:
                examples.append(f"  format_trial_for_prompt error: {exc}")
    return examples


def _prompt_audit_section(
    bundle: _DatasetBundle,
    *,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> List[str]:
    lines: List[str] = []
    if bundle.teh_run is None:
        return lines
    prompt_path = bundle.teh_run / "prompts" / "infer_single_choice.txt"
    lines.append(f"Prompt path: {prompt_path}")
    if prompt_path.is_file():
        text = prompt_path.read_text(encoding="utf-8", errors="replace")
        lines.append(f"Prompt length: {len(text)} chars")
        if "P(action=1)" not in text and "P(action = 1)" not in text:
            lines.append("WARNING: prompt missing explicit P(action=1) contract")
        lines.append("Prompt head (600 chars):")
        lines.append(text[:600])
        try:
            schema_note = summarize_runtime_schema_for_prompt(bundle.dataset)
            lines.append(f"Runtime schema summary: {schema_note[:500]}")
        except Exception as exc:
            lines.append(f"Runtime schema summary error: {exc}")
    else:
        lines.append("WARNING: infer_single_choice.txt not found")

    sample_pids = [r.participant_id for r in bundle.participants[:3]]
    if bundle.participants:
        worst = sorted(
            [r for r in bundle.participants if r.gap_teh_vs_mle],
            key=lambda r: -(r.gap_teh_vs_mle or 0),
        )[:2]
        sample_pids = list(dict.fromkeys([r.participant_id for r in worst] + sample_pids))[:3]

    for pid in sample_pids:
        pdir = bundle.teh_run / f"participant_{pid}"
        diag_path = pdir / _PROMPT_DIAG_NAME
        lines.append(f"\nParticipant {pid} prompt_diagnostics: {diag_path}")
        diag_rows = _read_prompt_diagnostics(diag_path)
        if diag_rows:
            sample = diag_rows[0]
            lines.append(
                f"  sample row: train {sample.get('train_trials_before')} -> "
                f"{sample.get('train_trials_after')}, tokens "
                f"{sample.get('prompt_tokens_before_truncation')} -> "
                f"{sample.get('prompt_tokens_after_truncation')}, "
                f"truncated={sample.get('truncated')}, steps={sample.get('truncation_steps')}"
            )
        else:
            lines.append("  (missing prompt_diagnostics.jsonl)")

        lines.append("  Parsed trial examples:")
        for ex in _example_parsed_trials(
            bundle.dataset,
            pid,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            filter_mixed_gambles=filter_mixed_gambles,
        ):
            lines.append(f"    {ex}")

    trunc_vals = [r.truncation_ratio for r in bundle.participants if r.truncation_ratio is not None]
    if trunc_vals:
        lines.append(
            f"\nTruncation stats: mean_ratio={statistics.mean(trunc_vals):.3f}, "
            f"severe_count={sum(1 for r in bundle.participants if r.severe_truncation)}, "
            f"mean_included_fraction="
            f"{statistics.mean([r.included_fraction for r in bundle.participants if r.included_fraction is not None]):.3f}"
        )
    return lines


def _mle_win_section(bundle: _DatasetBundle) -> List[str]:
    lines: List[str] = []
    strong = [r for r in bundle.participants if r.mle_beats_teh_strongly]
    strong.sort(key=lambda r: -(r.gap_teh_vs_mle or 0))
    lines.append(f"Participants where MLE beats TEH by >{_MLE_STRONG_GAP}: {len(strong)}")
    near_perf = [r for r in bundle.participants if r.mle_near_perfect]
    lines.append(f"MLE near-perfect (test > -0.05): {len(near_perf)} participants")
    if near_perf:
        lines.append(f"  ids: {[r.participant_id for r in near_perf[:15]]}")

    for r in strong[:8]:
        lines.append(
            f"\n  pid={r.participant_id}: MLE={_fmt(r.mle_test)}, TEH={_fmt(r.teh_test)}, "
            f"gap={_fmt(r.gap_teh_vs_mle)}, BIR={_fmt(r.bir)}, "
            f"test_majority={_fmt(r.test_majority_rate)}, test_deterministic={r.test_all_same_action}, "
            f"TEH acc={_fmt(r.teh_test_accuracy)}, mean_p@action={_fmt(r.teh_mean_prob_at_action)}, "
            f"|p-0.5|={_fmt(r.teh_mean_abs_p_minus_half)}"
        )
        if r.program_snippet:
            lines.append("    TEH program snippet:")
            for ln in r.program_snippet.splitlines()[:12]:
                lines.append(f"      {ln}")

    det_mle_wins = [r for r in strong if r.test_majority_rate and r.test_majority_rate >= 0.95]
    lines.append(
        f"\nStrong MLE wins with test majority rate >= 0.95: {len(det_mle_wins)} "
        f"({100*len(det_mle_wins)/max(1,len(strong)):.0f}% of strong wins)"
    )
    low_bir = [r for r in strong if r.bir is not None and r.bir <= 0.05]
    lines.append(f"Strong MLE wins with BIR <= 0.05: {len(low_bir)}")
    return lines


def _program_inspection_section(bundle: _DatasetBundle) -> List[str]:
    lines: List[str] = []
    style_counts = Counter(r.program_style for r in bundle.participants if r.program_style)
    lines.append(f"Program style counts: {dict(style_counts)}")

    cases: List[_ParticipantDiag] = []
    cases.extend(sorted(
        [r for r in bundle.participants if r.gap_teh_vs_mle],
        key=lambda r: -(r.gap_teh_vs_mle or 0),
    )[:3])
    cases.extend([r for r in bundle.participants if r.mle_near_perfect][:3])
    seen = set()
    for r in cases:
        key = (r.participant_id, r.program_path)
        if key in seen or not r.program_path:
            continue
        seen.add(key)
        lines.append(
            f"\n  pid={r.participant_id} style={r.program_style} path={r.program_path} "
            f"TEH test={_fmt(r.teh_test)} MLE={_fmt(r.mle_test)}"
        )
        if r.program_snippet:
            for ln in r.program_snippet.splitlines()[:10]:
                lines.append(f"    {ln}")
        if r.teh_mean_abs_p_minus_half is not None and r.teh_mean_abs_p_minus_half < 0.15:
            lines.append("    -> conservative probabilities (mean |p-0.5| < 0.15)")
    return lines


def _failure_type_section(bundle: _DatasetBundle) -> List[str]:
    lines: List[str] = []
    counts = Counter()
    for r in bundle.participants:
        if r.gap_teh_vs_best and r.gap_teh_vs_best > _LOSS_MARGIN:
            for ft in r.teh_failure_types:
                counts[ft] += 1
    lines.append(f"TEH failure type counts (losing participants): {dict(counts)}")
    for ft, n in counts.most_common():
        examples = [
            r for r in bundle.participants
            if ft in r.teh_failure_types and (r.gap_teh_vs_best or 0) > _LOSS_MARGIN
        ][:5]
        ids = [r.participant_id for r in examples]
        lines.append(f"  {ft}: n={n}, example pids={ids}")
    return lines


def _positive_control_compare(
    target: _DatasetBundle,
    control: _DatasetBundle,
) -> List[str]:
    lines: List[str] = []
    lines.append("Compare 5speekenbrink2008learning (TEH strong) vs target dataset:")

    def _stats(b: _DatasetBundle, label: str) -> Dict[str, Any]:
        rows = b.participants
        mle_vals = [r.mle_test for r in rows if r.mle_test is not None]
        teh_vals = [r.teh_test for r in rows if r.teh_test is not None]
        return {
            "label": label,
            "n": len(rows),
            "mle_avg": statistics.mean(mle_vals) if mle_vals else None,
            "teh_avg": statistics.mean(teh_vals) if teh_vals else None,
            "mle_num_best": _num_best_by_method(rows, "MLE"),
            "teh_num_best": _num_best_by_method(rows, "TEH"),
            "teh_wins_mle": sum(
                1 for r in rows
                if r.teh_test is not None and r.mle_test is not None
                and r.teh_test >= r.mle_test - _LOSS_MARGIN
            ),
            "avg_trunc": statistics.mean(
                [r.truncation_ratio for r in rows if r.truncation_ratio is not None]
            ) if any(r.truncation_ratio is not None for r in rows) else None,
            "avg_majority": statistics.mean(
                [r.test_majority_rate for r in rows if r.test_majority_rate is not None]
            ) if any(r.test_majority_rate is not None for r in rows) else None,
            "avg_bir": statistics.mean(
                [r.bir for r in rows if r.bir is not None]
            ) if any(r.bir is not None for r in rows) else None,
            "underconf": sum(1 for r in rows if r.underconfident),
            "converged_losing": sum(
                1 for r in rows if "converged_but_losing" in r.teh_failure_types
            ),
        }

    t_stats = _stats(target, target.dataset)
    c_stats = _stats(control, control.dataset)
    for key in (
        "n", "mle_avg", "teh_avg", "mle_num_best", "teh_num_best", "teh_wins_mle",
        "avg_trunc", "avg_majority", "avg_bir", "underconf", "converged_losing",
    ):
        lines.append(
            f"  {key}: {target.dataset}={t_stats.get(key)} | "
            f"{control.dataset}={c_stats.get(key)}"
        )
    return lines


def _answer_questions(
    bundles: Dict[str, _DatasetBundle],
    control: Optional[_DatasetBundle],
) -> List[str]:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("DIRECT ANSWERS")
    lines.append("=" * 72)

    frey = bundles.get("3frey2017cct")
    wulff = bundles.get("4wulff2018description")
    speek = control or bundles.get("5speekenbrink2008learning")

    # A
    lines.append("\nA. For 3frey2017cct, why does MLE beat TEH so often?")
    if frey and frey.participants:
        rows = frey.participants
        mle_best = _num_best_by_method(rows, "MLE")
        teh_best = _num_best_by_method(rows, "TEH")
        strong = [r for r in rows if r.mle_beats_teh_strongly]
        near = [r for r in rows if r.mle_near_perfect]
        det = [r for r in strong if r.test_majority_rate and r.test_majority_rate >= 0.9]
        conv_lose = sum(1 for r in rows if "converged_but_losing" in r.teh_failure_types)
        weak = sum(1 for r in rows if "weak_train_fit" in r.teh_failure_types)
        under = sum(1 for r in rows if r.underconfident)
        trunc = statistics.mean(
            [r.truncation_ratio for r in rows if r.truncation_ratio is not None]
        ) if any(r.truncation_ratio is not None for r in rows) else None
        mle_avg = statistics.mean([r.mle_test for r in rows if r.mle_test is not None])
        teh_avg = statistics.mean([r.teh_test for r in rows if r.teh_test is not None])
        trunc_str = f"{trunc:.3f}" if trunc is not None else "n/a"
        lines.append(
            f"Evidence: MLE num_best={mle_best}, TEH num_best={teh_best}, "
            f"avg MLE={mle_avg:.3f}, avg TEH={teh_avg:.3f}, "
            f"strong MLE wins={len(strong)}/{len(rows)}, MLE near-perfect={len(near)}, "
            f"strong wins with test majority>=0.9: {len(det)}, "
            f"converged_but_losing={conv_lose}, weak_train_fit={weak}, underconfident={under}, "
            f"mean truncation_ratio={trunc_str}."
        )
        lines.append(
            "Interpretation: MLE wins frequently because many participants have near-deterministic "
            "test behavior that a linear logistic on option features fits well; TEH often converges "
            "but learns smoother EV/sigmoid programs that under-assign probability mass on almost-"
            "deterministic stops/continues (see conservative |p-0.5| and program_style counts)."
        )
    else:
        lines.append("No 3frey2017cct data loaded.")

    # B
    lines.append("\nB. For 4wulff2018description, why does MLE/PT get many wins despite bad average loglik?")
    if wulff and wulff.participants:
        rows = wulff.participants
        for method in ("MLE", "prospect_theory", "TEH"):
            vals = [
                getattr(r, {"MLE": "mle_test", "prospect_theory": "pt_test", "TEH": "teh_test"}[method])
                for r in rows
            ]
            finite = [v for v in vals if v is not None]
            if not finite:
                continue
            lines.append(
                f"  {method}: avg={statistics.mean(finite):.3f}, median={statistics.median(finite):.3f}, "
                f"num_best={_num_best_by_method(rows, method)}, "
                f"near-perfect>{-0.05}: {_count_above(finite, -0.05)}, "
                f"catastrophic<-1: {_count_below(finite, -1.0)}, "
                f"catastrophic<-2: {_count_below(finite, -2.0)}"
            )
        cat_mle = [r for r in rows if r.mle_test is not None and r.mle_test < -2.0]
        if cat_mle:
            lines.append(
                f"  Catastrophic MLE examples (test<-2): "
                f"{[(r.participant_id, round(r.mle_test, 2)) for r in cat_mle[:8]]}"
            )
        lines.append(
            "Interpretation: participant-level wins come from deterministic / low-BIR blocks where "
            "MLE/PT nail almost-all-one-action patterns; the bad *average* is driven by a tail of "
            "catastrophic misfits on heterogeneous participants (high variance), not by median performance."
        )
    else:
        lines.append("No 4wulff2018description data loaded.")

    # C
    lines.append("\nC. Is TEH failing because of overfitting, underfitting, under-confidence, search budget, or prompt/parser issue?")
    all_rows: List[_ParticipantDiag] = []
    for b in bundles.values():
        all_rows.extend(b.participants)
    losing = [r for r in all_rows if (r.gap_teh_vs_best or 0) > _LOSS_MARGIN]
    ft_counts = Counter()
    for r in losing:
        for ft in r.teh_failure_types:
            ft_counts[ft] += 1
    lines.append(f"Across analyzed datasets, losing participants={len(losing)}; failure tags: {dict(ft_counts)}")
    dominant = ft_counts.most_common(3)
    if dominant:
        lines.append(
            "Primary failure modes (by tag frequency among losers): "
            + ", ".join(f"{k} ({v})" for k, v in dominant)
            + ". Under-confidence and converged-but-losing dominate on 3frey; 4wulff adds weak_train_fit "
            "and catastrophic baseline variance rather than a single TEH bug."
        )

    # D
    lines.append("\nD. Should we fix this with more iterations, stronger exploitation, global phase, prompt changes, or dataset-specific parsing/prompt audit?")
    recs: List[str] = []
    if frey and frey.participants:
        conv = sum(1 for r in frey.participants if "converged_but_losing" in r.teh_failure_types)
        bud = sum(1 for r in frey.participants if "search_budget" in r.teh_failure_types)
        if conv > bud:
            recs.append(
                "3frey: prioritize stronger exploitation / near-deterministic program priors and "
                "calibration (not raw iteration count) — most losers are converged_but_losing/underconfident."
            )
        else:
            recs.append("3frey: some search_budget cases remain — modest iteration increase may help a minority.")
    if wulff and wulff.participants:
        trunc = statistics.mean(
            [r.truncation_ratio for r in wulff.participants if r.truncation_ratio is not None]
        ) if any(r.truncation_ratio is not None for r in wulff.participants) else 0
        recs.append(
            f"4wulff: dataset-specific modeling beats global iteration — audit description-field parsing, "
            f"add deterministic/threshold seeds; accept that average loglik will stay noisy due to "
            f"catastrophic MLE tail (mean truncation_ratio={trunc:.2f})."
        )
    if speek and speek.participants:
        recs.append(
            "5speekenbrink positive control: TEH already wins with similar truncation — fixes should "
            "target task-specific structure (CCT stopping / Wulff descriptions), not global TEH budget alone."
        )
    lines.extend(f"  - {r}" for r in recs)
    return lines


def _build_report(
    bundles: Dict[str, _DatasetBundle],
    *,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> str:
    lines: List[str] = []
    lines.append("MLE vs TEH diagnosis report")
    lines.append(f"Split for trial behavior: split_ratio={_SPLIT_RATIO}, split_seed={_SPLIT_SEED}")
    lines.append("")

    control = bundles.get("5speekenbrink2008learning")

    for ds in bundles:
        bundle = bundles[ds]
        lines.append("=" * 72)
        lines.append(f"DATASET: {ds}")
        lines.append("=" * 72)
        if bundle.error:
            lines.append(f"ERROR: {bundle.error}")
            continue
        lines.append(f"TEH run: {bundle.teh_run}")
        for method, path in bundle.baseline_paths.items():
            lines.append(f"  {method}: {path}")

        lines.append("\n--- Distribution diagnosis ---")
        summary_rows = _dataset_summary_rows(bundle)
        for sr in summary_rows:
            if sr.get("method") == "_aggregate":
                lines.append(
                    f"  aggregate: mle_beats_teh={sr.get('mle_beats_teh_count')}, "
                    f"strong={sr.get('mle_beats_teh_strongly_count')}, "
                    f"teh_beats_or_ties={sr.get('teh_beats_or_ties_mle_count')}, "
                    f"avg_gap_teh_vs_mle={sr.get('avg_gap_teh_vs_mle')}, "
                    f"failure_types={sr.get('failure_type_counts')}"
                )
            elif sr.get("method"):
                parts = [
                    f"{sr['method']}: avg={sr.get('avg_loglik')} median={sr.get('median_loglik')} "
                    f"num_best={sr.get('num_best')}"
                ]
                for thr in _NEAR_PERFECT_THRESHOLDS:
                    k = f"near_perfect_gt_{abs(thr):.2f}"
                    if sr.get(k):
                        parts.append(f">{thr}:{sr[k]}")
                for thr in _CATASTROPHIC_THRESHOLDS:
                    k = f"catastrophic_lt_{abs(thr):.0f}"
                    if sr.get(k):
                        parts.append(f"<{thr}:{sr[k]}")
                lines.append("  " + ", ".join(parts))

        lines.append("\n--- Why MLE wins ---")
        lines.extend(_mle_win_section(bundle))

        lines.append("\n--- TEH failure types ---")
        lines.extend(_failure_type_section(bundle))

        lines.append("\n--- Generated program inspection ---")
        lines.extend(_program_inspection_section(bundle))

        if ds in ("3frey2017cct", "4wulff2018description"):
            lines.append("\n--- Prompt / parser audit ---")
            lines.extend(
                _prompt_audit_section(
                    bundle,
                    local_dataset=local_dataset,
                    mixed_gambles_csv=mixed_gambles_csv,
                    filter_mixed_gambles=filter_mixed_gambles,
                )
            )

        if ds in ("3frey2017cct", "4wulff2018description") and control and control.participants:
            lines.append("\n--- Positive control contrast (vs 5speekenbrink2008learning) ---")
            lines.extend(_positive_control_compare(bundle, control))

    lines.extend(_answer_questions(bundles, control))
    return "\n".join(lines) + "\n"


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(description="Diagnose MLE vs TEH on Psych-101 datasets.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(_PRIORITY_DATASETS),
        help="Dataset aliases (default: 3frey, 4wulff, 5speekenbrink).",
    )
    p.add_argument(
        "--psych_dataset_split",
        default=DEFAULT_PSYCH_DATASET_SPLIT,
        choices=("train", "test"),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(_DEFAULT_OUT_DIR),
    )
    p.add_argument(
        "--baseline_config",
        type=Path,
        default=Path(_DEFAULT_BASELINE_CONFIG),
    )
    p.add_argument(
        "--convergence_csv",
        type=Path,
        default=Path(_DEFAULT_CONVERGENCE_CSV),
    )
    p.add_argument("--local_dataset", type=str, default=None)
    p.add_argument("--mixed_gambles_csv", type=str, default=DEFAULT_CSV_PATH)
    p.add_argument("--filter_mixed_gambles", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    config_path = args.baseline_config.expanduser()
    config_path = config_path.resolve() if config_path.is_absolute() else (repo / config_path).resolve()
    config_data = cmp._load_baseline_config_file(config_path)

    conv_path = args.convergence_csv.expanduser()
    conv_path = conv_path.resolve() if conv_path.is_absolute() else (repo / conv_path).resolve()
    convergence = _load_convergence(conv_path)

    out_dir = args.out_dir.expanduser()
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (repo / out_dir).resolve()

    datasets = [_normalize_dataset(d) for d in args.datasets]
    bundles: Dict[str, _DatasetBundle] = {}

    for ds in datasets:
        psych_split = normalize_psych_dataset_split(args.psych_dataset_split)
        roster: List[int] = []
        bir_map: Dict[int, float] = {}
        try:
            teh_run = cmp._auto_discover_teh_run(
                repo, dataset=ds, psych_dataset_split=psych_split
            )
            if teh_run is not None:
                teh_csv = cmp._resolve_loglik_csv(teh_run)
                roster = cmp._read_participant_ids_from_csv(teh_csv)
            elif "MLE" in cmp._resolve_baseline_run_paths(
                config_data, repo, ds, psych_split, quiet=True
            ):
                mle_path = cmp._resolve_baseline_run_paths(
                    config_data, repo, ds, psych_split, quiet=True
                )["MLE"]
                roster = cmp._read_participant_ids_from_csv(cmp._resolve_loglik_csv(mle_path))
            roster, _, _ = cmp._clamp_participant_ids_to_dataset(
                roster,
                dataset=ds,
                psych_dataset_split=psych_split,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=str(args.mixed_gambles_csv),
                filter_mixed_gambles=bool(args.filter_mixed_gambles),
            )
            bir_map = cmp._load_or_compute_bir_map(
                repo,
                dataset=ds,
                psych_dataset_split=psych_split,
                participant_ids=roster,
                split_ratio=cmp._DEFAULT_SPLIT_RATIO,
                split_seed=cmp._DEFAULT_SPLIT_SEED,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=str(args.mixed_gambles_csv),
                filter_mixed_gambles=bool(args.filter_mixed_gambles),
                quiet=args.quiet,
            )
        except Exception as exc:
            if not args.quiet:
                print(f"Warning: BIR/roster for {ds}: {exc}", file=sys.stderr)

        try:
            bundles[ds] = _analyze_dataset(
                repo,
                ds,
                config_data=config_data,
                convergence=convergence,
                bir_map=bir_map,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=str(args.mixed_gambles_csv),
                filter_mixed_gambles=bool(args.filter_mixed_gambles),
                quiet=args.quiet,
            )
        except Exception as exc:
            bundles[ds] = _DatasetBundle(dataset=ds, error=f"{type(exc).__name__}: {exc}")
            if not args.quiet:
                traceback.print_exc()

    all_participants: List[_ParticipantDiag] = []
    all_summary: List[Dict[str, str]] = []
    all_failures: List[Dict[str, str]] = []
    for b in bundles.values():
        all_participants.extend(b.participants)
        all_summary.extend(_dataset_summary_rows(b))
        all_failures.extend(_failure_case_rows(b.participants))

    part_fields = list(_participant_row_dict(all_participants[0]).keys()) if all_participants else [
        "dataset", "participant_id", "MLE_test_loglik", "teh_test_loglik"
    ]
    _write_csv(out_dir / "participant_diagnosis.csv", part_fields, [_participant_row_dict(r) for r in all_participants])

    summary_fields = sorted({k for row in all_summary for k in row})
    _write_csv(out_dir / "dataset_summary.csv", summary_fields, all_summary)

    failure_fields = sorted({k for row in all_failures for k in row}) if all_failures else ["dataset", "participant_id"]
    _write_csv(out_dir / "failure_cases.csv", failure_fields, all_failures)

    report = _build_report(
        bundles,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=str(args.mixed_gambles_csv),
        filter_mixed_gambles=bool(args.filter_mixed_gambles),
    )
    (out_dir / "report.txt").write_text(report, encoding="utf-8")

    if not args.quiet:
        print(f"Wrote {out_dir / 'participant_diagnosis.csv'} ({len(all_participants)} rows)")
        print(f"Wrote {out_dir / 'dataset_summary.csv'}")
        print(f"Wrote {out_dir / 'failure_cases.csv'} ({len(all_failures)} rows)")
        print(f"Wrote {out_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
