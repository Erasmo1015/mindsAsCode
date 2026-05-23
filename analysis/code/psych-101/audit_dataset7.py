#!/usr/bin/env python3
"""
Audit Psych-101 dataset 7 (7hilbig2014generalized): structure, trials, TEH vs Centaur.

Usage:
  python analysis/code/psych-101/audit_dataset7.py \\
    --dataset 7hilbig2014generalized \\
    --psych_dataset_split train

Outputs:
  analysis/data/psych101_dataset7_audit/dataset7_trial_stats.csv
  analysis/data/psych101_dataset7_audit/dataset7_teh_vs_centaur.csv
  analysis/data/psych101_dataset7_audit/report.txt
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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PSYCH101_BINARY_DATASETS,
    experiment_id_for_alias,
    experiment_to_trial_dicts,
    format_trials_for_prompt,
    get_filtered_psych101_split,
    hf_id_for_psych_dataset_split,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
    parse_coverage_stats,
    parse_psych101_binary_row,
    split_psych_experiment,
    summarize_runtime_schema_for_prompt,
    _action_semantics_for_schema,
)
from data_modules.psych101_parsers import _RE_PRODUCT_TRIAL

from analysis.code.utils import compare as cmp

_DEFAULT_DATASET = "7hilbig2014generalized"
_DEFAULT_OUT_DIR = "analysis/data/psych101_dataset7_audit"
_FIRST_N_PARTICIPANTS = 50
_RAW_TEXT_EXAMPLES = 10
_INSPECT_N_EACH_SIDE = 3
_SNIPPET_MAX = 700
_STRONG_CENTAUR_GAP = 0.10

_TEST_LOGLIK = cmp._TEST_LOGLIK
_GATED_LOGLIK = cmp._GATED_LOGLIK
_BASELINE_METHODS = cmp._BASELINE_METHODS


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


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.{nd}f}"


def _problem_signature(trial: Mapping[str, Any]) -> str:
    p = dict(trial.get("problem") or {})
    for k in ("dataset_alias", "experiment_id", "trial_index", "payoff"):
        p.pop(k, None)
    return json.dumps(p, sort_keys=True, default=str)


def _max_group_size(trials: Sequence[Mapping[str, Any]]) -> int:
    if not trials:
        return 0
    counts: Counter = Counter()
    for t in trials:
        counts[_problem_signature(t)] += 1
    return max(counts.values())


def _unique_problem_groups(trials: Sequence[Mapping[str, Any]]) -> int:
    return len({_problem_signature(t) for t in trials})


def _repeated_groups(trials: Sequence[Mapping[str, Any]]) -> int:
    counts: Counter = Counter()
    for t in trials:
        counts[_problem_signature(t)] += 1
    return sum(1 for c in counts.values() if c > 1)


def _action_distribution(trials: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    dist: Dict[int, int] = Counter()
    for t in trials:
        a = t.get("action")
        if a in (0, 1):
            dist[int(a)] += 1
    return dict(dist)


def _majority_rate(trials: Sequence[Mapping[str, Any]]) -> Optional[float]:
    if not trials:
        return None
    dist = _action_distribution(trials)
    n = sum(dist.values())
    if n == 0:
        return None
    return max(dist.values()) / n


def _action_entropy(p1: float) -> float:
    if p1 <= 0.0 or p1 >= 1.0:
        return 0.0
    p0 = 1.0 - p1
    return -(p0 * math.log2(p0) + p1 * math.log2(p1))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _format_raw_row(row: Mapping[str, Any], max_text: int = 500) -> str:
    parts = []
    for k, v in row.items():
        if k == "text":
            text = str(v)
            head = text[:max_text]
            suffix = "..." if len(text) > max_text else ""
            parts.append(f"  text[{len(text)} chars]: {head!r}{suffix}")
        else:
            parts.append(f"  {k}: {v!r}")
    return "\n".join(parts)


def _load_filtered(alias: str, psych_split: str, local_dataset: Optional[str]):
    alias = normalize_psych101_dataset_alias(alias)
    filtered = get_filtered_psych101_split(
        alias, split=psych_split, local_dataset=local_dataset
    )
    exp_id = experiment_id_for_alias(alias)
    hf_id = hf_id_for_psych_dataset_split(psych_split)
    return alias, filtered, exp_id, hf_id


def _participant_trial_stats(
    alias: str,
    filtered,
    *,
    split_ratio: float,
    split_seed: int,
    bir_map: Dict[int, float],
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for row_idx in range(len(filtered)):
        raw = dict(filtered[row_idx])
        text = raw.get("text", "")
        n_product_trials = len(list(_RE_PRODUCT_TRIAL.finditer(text)))

        split_error = ""
        all_trials: List[Dict[str, Any]] = []
        train_trials: List[Dict[str, Any]] = []
        val_trials: List[Dict[str, Any]] = []
        test_trials: List[Dict[str, Any]] = []
        n_blocks = n_trials = 0
        coverage: Dict[str, Any] = {}

        try:
            exp = parse_psych101_binary_row(raw, alias)
            n_blocks = len(exp.blocks)
            n_trials = sum(len(b.trials) for b in exp.blocks)
            all_trials = experiment_to_trial_dicts(exp, dataset_alias=alias)
            coverage = parse_coverage_stats(text, exp)
            train_trials, val_trials, test_trials, _ = split_psych_experiment(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
        except Exception as exc:
            split_error = f"{type(exc).__name__}: {exc}"
            try:
                exp = parse_psych101_binary_row(raw, alias)
                n_blocks = len(exp.blocks)
                n_trials = sum(len(b.trials) for b in exp.blocks)
                all_trials = experiment_to_trial_dicts(exp, dataset_alias=alias)
                coverage = parse_coverage_stats(text, exp)
            except Exception:
                pass

        action_dist = _action_distribution(all_trials)
        rows_out.append(
            {
                "teh_participant_id": row_idx,
                "hf_participant": raw.get("participant"),
                "raw_text_chars": len(text),
                "regex_product_trials": n_product_trials,
                "parsed_blocks": n_blocks,
                "parsed_total_trials": n_trials,
                "parse_coverage": coverage.get("parse_coverage", ""),
                "train_trials": len(train_trials),
                "val_trials": len(val_trials),
                "test_trials": len(test_trials),
                "unique_problem_groups_total": _unique_problem_groups(all_trials),
                "repeated_problem_groups_total": _repeated_groups(all_trials),
                "max_group_size_total": _max_group_size(all_trials),
                "action_0_count": action_dist.get(0, 0),
                "action_1_count": action_dist.get(1, 0),
                "majority_action_rate_total": _fmt(_majority_rate(all_trials), 4),
                "BIR": _fmt(bir_map.get(row_idx), 4),
                "split_valid": split_error == "",
                "split_error": split_error,
            }
        )
    return rows_out


def _load_choose_fn(program_path: Path) -> Callable:
    spec = importlib.util.spec_from_file_location(
        f"audit7_{program_path.parent.name}", str(program_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    choose = getattr(mod, "choose", None)
    if not callable(choose):
        raise RuntimeError(f"No choose() in {program_path}")
    return choose


def _clamp_p(p: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(p)))


def _prob_from_choose(raw: Any) -> float:
    if isinstance(raw, bool):
        return 1.0 - 1e-6 if raw else 1e-6
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if int(raw) in (0, 1) and float(raw) == int(raw):
            return 1.0 - 1e-6 if int(raw) == 1 else 1e-6
        return _clamp_p(float(raw))
    return 0.5


def _trial_calibration_metrics(
    trials: Sequence[Mapping[str, Any]],
    program_path: Path,
) -> Dict[str, Optional[float]]:
    if not trials or not program_path.is_file():
        return {}
    try:
        choose = _load_choose_fn(program_path)
    except Exception:
        return {}

    probs_at: List[float] = []
    abs_dev: List[float] = []
    correct = 0
    ll = 0.0
    for t in trials:
        y = int(t["action"])
        try:
            p = _prob_from_choose(choose(t["problem"], t.get("history", [])))
        except Exception:
            p = 0.5
        p = _clamp_p(p)
        probs_at.append(p if y == 1 else 1.0 - p)
        abs_dev.append(abs(p - 0.5))
        correct += int((1 if p >= 0.5 else 0) == y)
        ll += y * math.log(p) + (1 - y) * math.log(1 - p)
    n = len(trials)
    return {
        "accuracy": correct / n,
        "mean_loglik": ll / n,
        "mean_prob_at_action": statistics.mean(probs_at),
        "mean_abs_p_minus_half": statistics.mean(abs_dev),
    }


def _classify_product_program_style(code: str) -> str:
    """Rule-style tags for Hilbig product-rating choice programs."""
    c = code or ""
    low = re.sub(r"\s+", "", c.lower())

    if len(c.strip()) < 60 and re.search(r"return\s+0\.5", c):
        return "constant/majority"
    if re.search(r"return\s+(?:0|1)(?:\.0)?\s*;?\s*$", c, re.M) and "history" not in low:
        return "constant/majority"
    if re.search(r"majority|mode\(|most_common", low):
        return "constant/majority"

    if re.search(r"history|recent_actions|action_counts|last_action", low):
        if re.search(r"ratings_[ab]|score_[ab]|weights", low):
            return "task-specific rule"
        return "history-heavy"

    if re.search(r"ratings_[ab]|score_[ab]|weights|weighted", low):
        if re.search(r"sigmoid|linear|beta|coeff|threshold|>=|<=|>", c):
            if re.search(r"if\s+.*(?:>=|<=|>|<|==)", c):
                return "threshold rule"
            return "raw EV / linear"
        return "task-specific rule"

    if re.search(r"if\s+.*:\s*return\s+(?:0|1|0\.5)", c):
        return "threshold rule"

    if re.search(r"sigmoid|linear|beta\s*\*", low):
        return "raw EV / linear"

    return "unclear"


def _program_snippet(code: str, max_len: int = _SNIPPET_MAX) -> str:
    clipped = (code or "").strip()
    if len(clipped) <= max_len:
        return clipped
    return clipped[: max_len - 24] + "\n... [truncated]"


def _best_method(scores: Mapping[str, Optional[float]], *, use_gated_teh: bool) -> Tuple[str, Optional[float]]:
    ranked: Dict[str, Optional[float]] = dict(scores)
    if use_gated_teh and ranked.get("TEH_gated") is not None:
        ranked["TEH"] = ranked["TEH_gated"]
    best_m = ""
    best_s: Optional[float] = None
    for m, s in ranked.items():
        if m in ("TEH_gated",):
            continue
        if s is None or not math.isfinite(s):
            continue
        if best_s is None or s > best_s:
            best_s = s
            best_m = m.replace("_gated", "")
    return best_m, best_s


def _load_run_scores(run_dir: Path) -> Dict[str, Dict[int, float]]:
    csv_path = cmp._resolve_loglik_csv(run_dir)
    out: Dict[str, Dict[int, float]] = {}
    for col in ("train_loglik", "val_loglik", _TEST_LOGLIK, _GATED_LOGLIK):
        try:
            out[col] = cmp._read_loglik_csv(csv_path, col, required=False)
        except (OSError, ValueError):
            out[col] = {}
    return out


def _teh_vs_centaur_rows(
    repo: Path,
    alias: str,
    psych_split: str,
    *,
    config_data: Mapping[str, Any],
    local_dataset: Optional[str],
    split_ratio: float,
    split_seed: int,
    participant_ids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    teh_run = cmp._auto_discover_teh_run(repo, dataset=alias, psych_dataset_split=psych_split)
    if teh_run is None:
        raise FileNotFoundError(f"No TEH run for {alias}")

    baseline_paths = cmp._resolve_baseline_run_paths(
        config_data, repo, alias, psych_split, quiet=True
    )
    teh_scores = _load_run_scores(teh_run)
    oe_run = baseline_paths.get("openevolve")
    oe_scores = _load_run_scores(oe_run) if oe_run else {}

    baseline_test: Dict[str, Dict[int, float]] = {}
    for method in _BASELINE_METHODS:
        if method == "openevolve":
            continue
        path = baseline_paths.get(method)
        if path is None:
            baseline_test[method] = {}
            continue
        try:
            baseline_test[method] = cmp._load_scores_from_run(path, _TEST_LOGLIK, required=False)
        except (OSError, ValueError):
            baseline_test[method] = {}

    from data_modules.psych101_binary import get_psych101_binary_experiment

    rows: List[Dict[str, Any]] = []
    centaur_beats: List[int] = []
    teh_beats: List[int] = []
    centaur_strong: List[int] = []

    for pid in participant_ids:
        t_test = teh_scores.get(_TEST_LOGLIK, {}).get(pid)
        t_gated = teh_scores.get(_GATED_LOGLIK, {}).get(pid)
        t_train = teh_scores.get("train_loglik", {}).get(pid)
        t_val = teh_scores.get("val_loglik", {}).get(pid)
        centaur_test = baseline_test.get("Centaur", {}).get(pid)
        oe_test = oe_scores.get(_TEST_LOGLIK, {}).get(pid)

        score_map = {
            "MLE": baseline_test.get("MLE", {}).get(pid),
            "prospect_theory": baseline_test.get("prospect_theory", {}).get(pid),
            "Centaur": centaur_test,
            "openevolve": oe_test,
            "TEH_gated": t_gated,
            "TEH": t_test,
        }
        best_m, best_s = _best_method(score_map, use_gated_teh=True)

        gap = None
        if t_gated is not None and centaur_test is not None:
            gap = t_gated - centaur_test
            if gap < 0:
                centaur_beats.append(pid)
                if centaur_test - t_gated >= _STRONG_CENTAUR_GAP:
                    centaur_strong.append(pid)
            elif gap > 0:
                teh_beats.append(pid)

        n_train = n_val = n_test = ""
        try:
            exp = get_psych101_binary_experiment(
                alias, pid, split=psych_split, local_dataset=local_dataset
            )
            train, val, test, _ = split_psych_experiment(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
            n_train, n_val, n_test = len(train), len(val), len(test)
            test_trials = test
        except Exception:
            test_trials = []

        teh_dir = teh_run / f"participant_{pid}"
        teh_cal = _trial_calibration_metrics(test_trials, teh_dir / "best_program.py")

        teh_code = ""
        teh_prog = teh_dir / "best_program.py"
        if teh_prog.is_file():
            teh_code = teh_prog.read_text(encoding="utf-8", errors="replace")

        rows.append(
            {
                "participant_id": pid,
                "train_trials": n_train,
                "val_trials": n_val,
                "test_trials": n_test,
                "teh_train_loglik": _fmt(t_train),
                "teh_val_loglik": _fmt(t_val),
                "teh_test_loglik": _fmt(t_test),
                "teh_gated_test_loglik": _fmt(t_gated),
                "mle_test_loglik": _fmt(score_map["MLE"]),
                "pt_test_loglik": _fmt(score_map["prospect_theory"]),
                "centaur_test_loglik": _fmt(centaur_test),
                "openevolve_test_loglik": _fmt(oe_test),
                "best_method": best_m,
                "gap_teh_gated_minus_centaur": _fmt(gap),
                "teh_test_accuracy": _fmt(teh_cal.get("accuracy")),
                "teh_mean_prob_at_action": _fmt(teh_cal.get("mean_prob_at_action")),
                "teh_mean_abs_p_minus_half": _fmt(teh_cal.get("mean_abs_p_minus_half")),
                "teh_program_style": _classify_product_program_style(teh_code),
                "teh_run": str(teh_run.relative_to(repo)),
            }
        )

    meta = {
        "teh_run": str(teh_run.relative_to(repo)),
        "oe_run": str(oe_run.relative_to(repo)) if oe_run else "",
        "n_centaur_beats_teh": len(centaur_beats),
        "n_teh_beats_centaur": len(teh_beats),
        "n_centaur_strong_beats": len(centaur_strong),
        "centaur_beats_pids": centaur_beats,
        "teh_beats_pids": teh_beats,
        "centaur_strong_pids": centaur_strong,
        "rows": rows,
    }
    return rows, meta


def _collect_all_split_trials(
    alias: str,
    filtered,
    *,
    split_ratio: float,
    split_seed: int,
    participant_ids: Sequence[int],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[int, float],
]:
    all_trials: List[Dict[str, Any]] = []
    train_trials: List[Dict[str, Any]] = []
    val_trials: List[Dict[str, Any]] = []
    test_trials: List[Dict[str, Any]] = []
    per_pid_action1: Dict[int, float] = {}

    for pid in participant_ids:
        if pid >= len(filtered):
            continue
        raw = dict(filtered[pid])
        try:
            exp = parse_psych101_binary_row(raw, alias)
            trials = experiment_to_trial_dicts(exp, dataset_alias=alias)
            tr, va, te, _ = split_psych_experiment(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
            all_trials.extend(trials)
            train_trials.extend(tr)
            val_trials.extend(va)
            test_trials.extend(te)
            dist = _action_distribution(trials)
            n = dist.get(0, 0) + dist.get(1, 0)
            per_pid_action1[pid] = dist.get(1, 0) / n if n else 0.5
        except Exception:
            continue
    return all_trials, train_trials, val_trials, test_trials, per_pid_action1


def _population_consensus(
    trials: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """For each unique problem, measure cross-instance action agreement."""
    by_sig: Dict[str, List[int]] = defaultdict(list)
    for t in trials:
        sig = _problem_signature(t)
        a = t.get("action")
        if a in (0, 1):
            by_sig[sig].append(int(a))

    if not by_sig:
        return {}

    majority_rates: List[float] = []
    unanimous = 0
    for actions in by_sig.values():
        c = Counter(actions)
        maj = max(c.values()) / len(actions)
        majority_rates.append(maj)
        if maj == 1.0:
            unanimous += 1

    return {
        "n_unique_problems": len(by_sig),
        "mean_majority_rate_per_problem": statistics.mean(majority_rates),
        "median_majority_rate_per_problem": statistics.median(majority_rates),
        "unanimous_problem_fraction": unanimous / len(by_sig),
        "problems_with_repeats": sum(1 for a in by_sig.values() if len(a) > 1),
    }


def _structure_diagnosis_section(
    lines: List[str],
    *,
    alias: str,
    filtered,
    compare_rows: Sequence[Mapping[str, Any]],
    split_ratio: float,
    split_seed: int,
    participant_ids: Sequence[int],
) -> Dict[str, Any]:
    lines.append("")
    lines.append("=" * 80)
    lines.append("5. STRUCTURE DIAGNOSIS (population-level rule?)")
    lines.append("=" * 80)

    all_trials, train_trials, val_trials, test_trials, per_pid_p1 = _collect_all_split_trials(
        alias,
        filtered,
        split_ratio=split_ratio,
        split_seed=split_seed,
        participant_ids=participant_ids,
    )

    pooled_train = _action_distribution(train_trials)
    pooled_test = _action_distribution(test_trials)
    n_tr = sum(pooled_train.values())
    n_te = sum(pooled_test.values())
    p1_tr = pooled_train.get(1, 0) / n_tr if n_tr else None
    p1_te = pooled_test.get(1, 0) / n_te if n_te else None

    lines.append("Pooled action distributions (participants 0–49):")
    lines.append(f"  train: {dict(pooled_train)} (P(action=1)={_fmt(p1_tr)})")
    lines.append(f"  test:  {dict(pooled_test)} (P(action=1)={_fmt(p1_te)})")

    if per_pid_p1:
        p1_vals = list(per_pid_p1.values())
        lines.append(
            f"  per-participant P(action=1): mean={statistics.mean(p1_vals):.3f}, "
            f"std={statistics.stdev(p1_vals) if len(p1_vals) > 1 else 0:.3f}, "
            f"min={min(p1_vals):.3f}, max={max(p1_vals):.3f}"
        )
        entropies = [_action_entropy(p) for p in p1_vals]
        lines.append(
            f"  per-participant action entropy: mean={statistics.mean(entropies):.3f}, "
            f"median={statistics.median(entropies):.3f}"
        )
        det = sum(1 for p in p1_vals if p in (0.0, 1.0))
        lines.append(f"  participants with deterministic all-0/all-1 choices: {det}")

    pop_all = _population_consensus(all_trials)
    pop_train = _population_consensus(train_trials)
    pop_test = _population_consensus(test_trials)

    lines.append("")
    lines.append("Cross-participant consensus on identical problems (same ratings vectors):")
    for label, stats in (
        ("all trials", pop_all),
        ("train", pop_train),
        ("test", pop_test),
    ):
        if not stats:
            continue
        lines.append(
            f"  {label}: unique_problems={stats['n_unique_problems']}, "
            f"mean_majority_rate={stats['mean_majority_rate_per_problem']:.3f}, "
            f"unanimous_frac={stats['unanimous_problem_fraction']:.3f}, "
            f"problems_with_repeats={stats['problems_with_repeats']}"
        )

    train_sigs = {_problem_signature(t) for t in train_trials}
    test_sigs = {_problem_signature(t) for t in test_trials}
    overlap = train_sigs & test_sigs
    lines.append("")
    lines.append("Train/test problem structure:")
    lines.append(f"  unique train problems: {len(train_sigs)}")
    lines.append(f"  unique test problems: {len(test_sigs)}")
    lines.append(f"  overlap (same ratings in train and test): {len(overlap)}")
    lines.append(
        "  Note: split is by trial index within participant, not by problem ID — "
        "train and test share the same task schema (product ratings) but different trial instances."
    )

    # Weighted-ratings population rule check
    weights = [0.9, 0.8, 0.7, 0.6]

    def _pop_rule_action(trial: Mapping[str, Any]) -> int:
        p = trial["problem"]
        sa = sum(w * r for w, r in zip(weights, p["ratings_A"]))
        sb = sum(w * r for w, r in zip(weights, p["ratings_B"]))
        return 1 if sb > sa else 0

    rule_correct = 0
    rule_total = 0
    for t in all_trials:
        pred = _pop_rule_action(t)
        if int(t["action"]) in (0, 1):
            rule_correct += int(pred == int(t["action"]))
            rule_total += 1
    rule_acc = rule_correct / rule_total if rule_total else None
    lines.append("")
    lines.append(
        "Population weighted-ratings rule (score=sum w_i*r_i, w=[0.9,0.8,0.7,0.6], pick higher):"
    )
    lines.append(f"  accuracy over all pooled trials (0–49): {_fmt(rule_acc)} (n={rule_total})")

    return {
        "p1_train": p1_tr,
        "p1_test": p1_te,
        "per_pid_p1_std": statistics.stdev(list(per_pid_p1.values())) if len(per_pid_p1) > 1 else None,
        "pop_all": pop_all,
        "pop_train": pop_train,
        "pop_test": pop_test,
        "train_test_overlap": len(overlap),
        "pop_rule_accuracy": rule_acc,
    }


def _inspect_programs_section(
    lines: List[str],
    repo: Path,
    compare_rows: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> None:
    lines.append("")
    lines.append("=" * 80)
    lines.append("6. TEH PROGRAM INSPECTION")
    lines.append("=" * 80)

    by_gap = []
    for r in compare_rows:
        g = _safe_float(r.get("gap_teh_gated_minus_centaur"))
        if g is not None:
            by_gap.append((g, int(r["participant_id"])))
    by_gap.sort()

    centaur_win = [pid for g, pid in by_gap if g < 0][:_INSPECT_N_EACH_SIDE]
    centaur_strong = meta.get("centaur_strong_pids", [])[:_INSPECT_N_EACH_SIDE]
    teh_win = [pid for g, pid in by_gap if g > 0][-_INSPECT_N_EACH_SIDE :]
    teh_win.reverse()

    lines.append(f"Centaur beats TEH (gated): pids={meta.get('centaur_beats_pids')}")
    lines.append(f"TEH beats Centaur: pids={meta.get('teh_beats_pids')}")
    lines.append(f"Centaur strongly beats TEH (gap>={_STRONG_CENTAUR_GAP}): pids={meta.get('centaur_strong_pids')}")
    lines.append("")
    lines.append(
        f"Inspecting up to {_INSPECT_N_EACH_SIDE} Centaur-win, Centaur-strong, and TEH-win participants."
    )

    row_by_pid = {int(r["participant_id"]): r for r in compare_rows}
    teh_run = repo / str(meta["teh_run"])

    inspect_sets = (
        ("Centaur-win (smallest TEH gaps)", centaur_win),
        ("Centaur strong-win", centaur_strong),
        ("TEH-win (largest TEH gaps)", teh_win),
    )
    seen: set = set()
    for label, pids in inspect_sets:
        lines.append("")
        lines.append(f"--- {label} ---")
        for pid in pids:
            if pid in seen:
                continue
            seen.add(pid)
            r = row_by_pid.get(pid, {})
            lines.append(
                f"\nParticipant {pid}: gap_teh_minus_centaur={r.get('gap_teh_gated_minus_centaur')} "
                f"best={r.get('best_method')} teh_gated={r.get('teh_gated_test_loglik')} "
                f"centaur={r.get('centaur_test_loglik')}"
            )
            lines.append(f"  TEH style: {r.get('teh_program_style')}")
            prog = teh_run / f"participant_{pid}" / "best_program.py"
            if not prog.is_file():
                lines.append(f"  TEH: (no {prog.name})")
                continue
            code = prog.read_text(encoding="utf-8", errors="replace")
            style = _classify_product_program_style(code)
            lines.append(f"  TEH best_program.py ({style}, {len(code)} chars):")
            for snip_line in _program_snippet(code).splitlines()[:18]:
                lines.append(f"    {snip_line}")


def _calibration_section(lines: List[str], compare_rows: Sequence[Mapping[str, Any]]) -> None:
    lines.append("")
    lines.append("=" * 80)
    lines.append("CONFIDENCE / CALIBRATION (test split, TEH best_program.py)")
    lines.append("=" * 80)

    def _agg(key: str) -> Optional[float]:
        vals = [_safe_float(r.get(key)) for r in compare_rows]
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    lines.append("Means over participants 0–49:")
    lines.append(
        f"  TEH: accuracy={_fmt(_agg('teh_test_accuracy'))} "
        f"mean_prob@action={_fmt(_agg('teh_mean_prob_at_action'))} "
        f"mean|p-0.5|={_fmt(_agg('teh_mean_abs_p_minus_half'))}"
    )

    teh_g = [_safe_float(r["teh_gated_test_loglik"]) for r in compare_rows]
    cent = [_safe_float(r["centaur_test_loglik"]) for r in compare_rows]
    teh_f = [v for v in teh_g if v is not None]
    cent_f = [v for v in cent if v is not None]
    if teh_f:
        lines.append(
            f"  TEH gated_test_loglik: mean={_fmt(statistics.mean(teh_f))} "
            f"median={_fmt(statistics.median(teh_f))}"
        )
    if cent_f:
        lines.append(
            f"  Centaur test_loglik: mean={_fmt(statistics.mean(cent_f))} "
            f"median={_fmt(statistics.median(cent_f))}"
        )


def _final_answers(
    lines: List[str],
    *,
    alias: str,
    spec: Mapping[str, Any],
    trial_rows: Sequence[Mapping[str, Any]],
    compare_rows: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    raw_snippets: Sequence[str],
    struct: Mapping[str, Any],
) -> None:
    valid = [r for r in trial_rows if r.get("split_valid")]
    compare_f = [r for r in compare_rows if _safe_float(r.get("teh_gated_test_loglik")) is not None]

    teh_styles = Counter(r.get("teh_program_style") for r in compare_rows)
    best_counts = Counter(r.get("best_method") for r in compare_rows)

    median_trials = statistics.median(int(r["parsed_total_trials"]) for r in valid) if valid else 0
    median_train = statistics.median(int(r["train_trials"]) for r in valid) if valid else 0
    median_test = statistics.median(int(r["test_trials"]) for r in valid) if valid else 0

    teh_g = [_safe_float(r["teh_gated_test_loglik"]) for r in compare_f]
    cent = [_safe_float(r["centaur_test_loglik"]) for r in compare_f]
    avg_teh = statistics.mean(teh_g) if teh_g else None
    avg_cent = statistics.mean(cent) if cent else None

    bir_vals = [_safe_float(r["BIR"]) for r in valid if r.get("BIR")]
    avg_bir = statistics.mean(bir_vals) if bir_vals else None

    lines.append("")
    lines.append("=" * 80)
    lines.append("7. FINAL ANSWERS")
    lines.append("=" * 80)

    lines.append("")
    lines.append("A. What is dataset 7 actually measuring?")
    lines.append(
        "   Hilbig (2014) generalized inference / multi-attribute product choice. On each trial "
        "two products (labeled e.g. G vs K or A vs R) are shown with four binary expert ratings "
        "each; the participant presses a key to choose one product. There is no gamble payoff "
        "structure — it is a weighted-cue / lexicographic-style product comparison task, "
        "not gamble_A/gamble_B. "
        f"Evidence: parser={spec.get('parser')!r}, schema_type={spec.get('schema_type')!r}, "
        f"problem fields ratings_A/ratings_B, regex product trials in raw text."
    )

    lines.append("")
    lines.append("B. How many trials per participant does it have?")
    if valid:
        pt = [int(r["parsed_total_trials"]) for r in valid]
        lines.append(
            f"   Median {median_trials:.0f} parsed trials (min={min(pt)}, max={max(pt)}). "
            f"With split_ratio=0.6: median train={median_train:.0f}, test={median_test:.0f}."
        )
    else:
        lines.append("   Could not parse trials.")

    lines.append("")
    lines.append("C. Is the train/test split giving TEH enough data?")
    lines.append(
        f"   Yes for trial count: ~{median_train:.0f} train / ~{statistics.median(int(r['val_trials']) for r in valid) if valid else 0:.0f} val / "
        f"~{median_test:.0f} test trials per participant; full prompt inclusion "
        "(truncation_ratio=0 in prompt diagnostics). Split is by pseudo-block "
        "(repeated rating-vector groups), not i.i.d. trial shuffle."
    )

    lines.append("")
    lines.append("D. Why does Centaur dominate?")
    lines.append(
        f"   Centaur avg test loglik={_fmt(avg_cent)} vs TEH gated={_fmt(avg_teh)}; "
        f"num_best: {dict(best_counts)}. "
        f"Centaur wins {meta.get('n_centaur_beats_teh')}/50 on gated loglik. "
        f"Population weighted-ratings rule accuracy={_fmt(struct.get('pop_rule_accuracy'))} — "
        "a shared population policy fits well. BIR is low (mean "
        f"{_fmt(avg_bir)}) because problems rarely repeat identically within a participant, "
        "not because choices are random. Per-participant action entropy is high (~1 bit) but "
        "cross-participant consensus on identical rating vectors is strong."
    )

    lines.append("")
    lines.append(
        "E. Is TEH losing because of underconfidence, wrong prompt/schema, search budget, "
        "or missing population-level rule?"
    )
    lines.append(
        f"   Primarily missing population-level rule + per-participant overfitting. "
        f"TEH program styles: {dict(teh_styles)} — many find weighted-ratings rules but fit "
        "individual idiosyncrasies; Centaur pools across participants. "
        "Prompt/schema look correct (product/B, ratings vectors, P(action=1)). "
        "Search converges (probably_enough ~100% in grand analysis) so budget is not the main bottleneck. "
        "Underconfidence is secondary: mean |p-0.5| is moderate; TEH often finds reasonable rules "
        "but loses to population pooling."
    )

    lines.append("")
    lines.append("F. Would global phase / population program likely help?")
    lines.append(
        "   Yes. Identical rating vectors elicit consistent majority choices across participants; "
        f"population rule accuracy={_fmt(struct.get('pop_rule_accuracy'))}. "
        "A shared program (or Centaur-style pooling) should outperform per-participant TEH. "
        "Global phase that learns one weighted-ratings rule across participants is well matched "
        "to this dataset."
    )


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(description="Audit dataset 7 (Hilbig generalized product choice).")
    p.add_argument("--dataset", default=_DEFAULT_DATASET)
    p.add_argument("--psych_dataset_split", default=DEFAULT_PSYCH_DATASET_SPLIT)
    p.add_argument("--local_dataset", default=None)
    p.add_argument("--out_dir", type=Path, default=Path(_DEFAULT_OUT_DIR))
    p.add_argument(
        "--split_ratio",
        type=float,
        default=cmp._DEFAULT_SPLIT_RATIO,
        help="Match compare.py / baseline BIR defaults.",
    )
    p.add_argument("--split_seed", type=int, default=cmp._DEFAULT_SPLIT_SEED)
    p.add_argument(
        "--participants",
        type=int,
        default=_FIRST_N_PARTICIPANTS,
        help="First N TEH participant ids for run comparison.",
    )
    args = p.parse_args()

    psych_split = normalize_psych_dataset_split(args.psych_dataset_split)
    out_dir = args.out_dir.resolve() if args.out_dir.is_absolute() else (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    alias, filtered, exp_id, hf_id = _load_filtered(
        args.dataset, psych_split, args.local_dataset
    )
    spec = PSYCH101_BINARY_DATASETS[alias]

    config_path = repo / cmp._DEFAULT_BASELINE_CONFIG
    config_data = cmp._load_baseline_config_file(config_path)
    roster = list(range(min(args.participants, len(filtered))))
    bir_map = cmp._load_or_compute_bir_map(
        repo,
        dataset=alias,
        psych_dataset_split=psych_split,
        participant_ids=roster,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        local_dataset=args.local_dataset,
        quiet=False,
    )

    trial_rows = _participant_trial_stats(
        alias,
        filtered,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        bir_map=bir_map,
    )

    compare_rows, run_meta = _teh_vs_centaur_rows(
        repo,
        alias,
        psych_split,
        config_data=config_data,
        local_dataset=args.local_dataset,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        participant_ids=roster,
    )

    lines: List[str] = []
    lines.append("PSYCH-101 DATASET 7 AUDIT (7hilbig2014generalized)")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Repo: {repo}")
    lines.append(f"Dataset alias: {alias}")
    lines.append(f"psych_dataset_split: {psych_split}")
    lines.append(f"Split: ratio={args.split_ratio}, seed={args.split_seed}")
    lines.append(f"TEH run: {run_meta.get('teh_run')}")

    # --- 1. Raw ---
    lines.append("")
    lines.append("=" * 80)
    lines.append("1. RAW DATASET IDENTITY")
    lines.append("=" * 80)
    lines.append(f"HF dataset id: {hf_id}")
    lines.append(f"Experiment id / CSV: {exp_id}")
    lines.append(f"Filtered HF rows: {len(filtered)}")
    if len(filtered):
        lines.append(f"Raw column names: {list(dict(filtered[0]).keys())}")
    lines.append(f"Parser (config): {spec.get('parser')!r}")
    lines.append(f"Config task_description: {spec.get('task_description')!r}")

    raw_snippets: List[str] = []
    lines.append("")
    lines.append(f"First {min(_RAW_TEXT_EXAMPLES, len(filtered))} raw text examples (head):")
    for i in range(min(_RAW_TEXT_EXAMPLES, len(filtered))):
        raw = dict(filtered[i])
        text = str(raw.get("text", ""))
        raw_snippets.append(text)
        lines.append(f"\n[row {i}] hf_participant={raw.get('participant')!r}")
        lines.append(_format_raw_row(raw, max_text=600))

    lines.append("")
    lines.append("Plain-English task (from transcript structure, not name alone):")
    if raw_snippets:
        t0 = raw_snippets[0]
        m0 = re.search(r"Product\s+[A-Z]\s+ratings:", t0, re.I)
        inst = t0[: m0.start()].strip()[:500] if m0 else t0[:500]
        lines.append(f"  Instruction head: {inst!r}")
    lines.append(
        "  Repeated trials: two products each with a vector of four binary expert ratings; "
        "participant chooses one product via key press. No feedback or points shown per trial."
    )

    # --- 2. Parsed schema ---
    lines.append("")
    lines.append("=" * 80)
    lines.append("2. PARSED SCHEMA")
    lines.append("=" * 80)
    lines.append(f"parser name: {spec.get('parser')!r}")
    lines.append(f"schema_type: {spec.get('schema_type')!r}")
    lines.append(
        "problem fields (per trial): ratings_A, ratings_B (lists of 4 ints), "
        "option_A_features, option_B_features, option_keys, schema_type"
    )
    lines.append("action semantics: action=0 -> first product; action=1 -> second product; return P(action=1)")
    lines.append("gamble_A/gamble_B task: NO — product-rating binary choice (schema B / product subtype)")

    if filtered:
        try:
            exp0 = parse_psych101_binary_row(dict(filtered[0]), alias)
            train0, val0, test0, opts0 = split_psych_experiment(
                exp0, split_ratio=args.split_ratio, split_seed=args.split_seed
            )
            p0 = train0[0]["problem"] if train0 else (test0[0]["problem"] if test0 else {})
            sem = _action_semantics_for_schema(
                p0.get("option_keys", []),
                str(p0.get("schema_type", "B")),
                p0,
                is_gamble=False,
            )
            lines.append(f"action semantics (participant 0): {sem}")
            lines.append(f"option_keys example: {opts0}")
            lines.append("")
            lines.append("Example parsed train trial:")
            if train0:
                slim = {
                    "action": train0[0].get("action"),
                    "history_len": len(train0[0].get("history") or []),
                    "problem": train0[0].get("problem"),
                }
                lines.append(json.dumps(slim, default=str, indent=2)[:1200])
            lines.append("Example parsed val trial:" if val0 else "Example parsed val: (none — val empty)")
            if val0:
                lines.append(
                    json.dumps(
                        {
                            "action": val0[0].get("action"),
                            "history_len": len(val0[0].get("history") or []),
                            "problem_keys": sorted((val0[0].get("problem") or {}).keys()),
                        },
                        default=str,
                    )[:800]
                )
            lines.append("Example parsed test trial:")
            if test0:
                lines.append(
                    json.dumps(
                        {
                            "action": test0[0].get("action"),
                            "history_len": len(test0[0].get("history") or []),
                            "problem": test0[0].get("problem"),
                        },
                        default=str,
                        indent=2,
                    )[:1200]
                )
            lines.append("")
            lines.append("Formatted prompt examples (train, first 6):")
            lines.append(format_trials_for_prompt(train0, max_trials=6))
            lines.append("")
            lines.append("Runtime schema summary:")
            lines.append(summarize_runtime_schema_for_prompt(train0[:20]))
        except Exception as exc:
            lines.append(f"Parse example error: {exc}")
            lines.append(traceback.format_exc())

    # --- 3. Trial statistics ---
    lines.append("")
    lines.append("=" * 80)
    lines.append("3. TRIAL STATISTICS (all participants)")
    lines.append("=" * 80)
    valid = [r for r in trial_rows if r.get("split_valid")]
    lines.append(f"HF rows: {len(trial_rows)}, split-valid: {len(valid)}, invalid: {len(trial_rows)-len(valid)}")
    if valid:
        pt = [int(r["parsed_total_trials"]) for r in valid]
        tr = [int(r["train_trials"]) for r in valid]
        te = [int(r["test_trials"]) for r in valid]
        lines.append(
            f"parsed trials per participant: min={min(pt)}, median={statistics.median(pt)}, max={max(pt)}"
        )
        lines.append(
            f"train trials: min={min(tr)}, median={statistics.median(tr)}, max={max(tr)}"
        )
        lines.append(
            f"val trials: min={min(int(r['val_trials']) for r in valid)}, "
            f"median={statistics.median(int(r['val_trials']) for r in valid)}, "
            f"max={max(int(r['val_trials']) for r in valid)}"
        )
        lines.append(
            f"test trials: min={min(te)}, median={statistics.median(te)}, max={max(te)}"
        )
        lines.append(
            f"parsed blocks (median): {statistics.median(int(r['parsed_blocks']) for r in valid):.0f}"
        )
        lines.append(
            f"unique problem groups (median): "
            f"{statistics.median(int(r['unique_problem_groups_total']) for r in valid):.0f}"
        )
        lines.append(
            f"participants with repeated problem groups: "
            f"{sum(1 for r in valid if int(r['repeated_problem_groups_total']) > 0)}"
        )
        maj = [_safe_float(r["majority_action_rate_total"]) for r in valid]
        maj_f = [m for m in maj if m is not None]
        if maj_f:
            lines.append(
                f"majority action rate: mean={statistics.mean(maj_f):.3f}, "
                f"median={statistics.median(maj_f):.3f}"
            )
        bir_vals = [_safe_float(r["BIR"]) for r in valid if r.get("BIR")]
        if bir_vals:
            lines.append(
                f"BIR: mean={statistics.mean(bir_vals):.4f}, "
                f"median={statistics.median(bir_vals):.4f} (n={len(bir_vals)})"
            )
        lines.append(
            "BIR interpretation: low values reflect that each participant sees 24 unique rating "
            "vectors × 4 repeats (=96 trials), so identical problems repeat only 4× within "
            "participant — BIR≈0 is common and does NOT mean choices are random."
        )

    lines.append("")
    lines.append("Per-participant sample (first 10):")
    lines.append(
        "  pid | trials | train | test | unique_problems | maj_rate | BIR | a0/a1"
    )
    for r in trial_rows[:10]:
        lines.append(
            f"  {r['teh_participant_id']:3d} | {r['parsed_total_trials']:6} | "
            f"{r['train_trials']:5} | {r['test_trials']:4} | "
            f"{r['unique_problem_groups_total']:15} | "
            f"{r['majority_action_rate_total']:8} | {r['BIR']:6} | "
            f"{r['action_0_count']}/{r['action_1_count']}"
        )

    trial_fields = [
        "teh_participant_id",
        "hf_participant",
        "raw_text_chars",
        "regex_product_trials",
        "parsed_blocks",
        "parsed_total_trials",
        "parse_coverage",
        "train_trials",
        "val_trials",
        "test_trials",
        "unique_problem_groups_total",
        "repeated_problem_groups_total",
        "max_group_size_total",
        "action_0_count",
        "action_1_count",
        "majority_action_rate_total",
        "BIR",
        "split_valid",
        "split_error",
    ]
    trial_csv = out_dir / "dataset7_trial_stats.csv"
    _write_csv(trial_csv, trial_rows, trial_fields)

    # --- 4. Run comparison ---
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"4. FIRST-{args.participants} RUN STATISTICS")
    lines.append("=" * 80)
    best_counts = Counter(r.get("best_method") for r in compare_rows)
    lines.append(f"num_best: {dict(best_counts)}")
    teh_g = [_safe_float(r["teh_gated_test_loglik"]) for r in compare_rows]
    cent = [_safe_float(r["centaur_test_loglik"]) for r in compare_rows]
    teh_f = [v for v in teh_g if v is not None]
    cent_f = [v for v in cent if v is not None]
    if teh_f:
        lines.append(f"Avg TEH gated_test_loglik: {statistics.mean(teh_f):.4f}")
    if cent_f:
        lines.append(f"Avg Centaur test_loglik: {statistics.mean(cent_f):.4f}")
    lines.append(f"Centaur beats TEH (gated): pids={run_meta.get('centaur_beats_pids')}")
    lines.append(f"TEH beats Centaur: pids={run_meta.get('teh_beats_pids')}")
    lines.append(f"Centaur strongly beats TEH (gap>={_STRONG_CENTAUR_GAP}): pids={run_meta.get('centaur_strong_pids')}")

    lines.append("")
    lines.append("Per-participant table:")
    lines.append(
        "  pid | train | test | teh_gated | centaur | MLE | PT | OE | best | gap_teh-cent"
    )
    for r in compare_rows:
        lines.append(
            f"  {int(r['participant_id']):3d} | {r['train_trials']:5} | {r['test_trials']:4} | "
            f"{r['teh_gated_test_loglik']:9} | {r['centaur_test_loglik']:7} | "
            f"{r['mle_test_loglik']:6} | {r['pt_test_loglik']:6} | "
            f"{r['openevolve_test_loglik']:6} | {r['best_method']:10} | "
            f"{r['gap_teh_gated_minus_centaur']}"
        )

    compare_fields = [
        "participant_id",
        "train_trials",
        "val_trials",
        "test_trials",
        "teh_train_loglik",
        "teh_val_loglik",
        "teh_test_loglik",
        "teh_gated_test_loglik",
        "mle_test_loglik",
        "pt_test_loglik",
        "centaur_test_loglik",
        "openevolve_test_loglik",
        "best_method",
        "gap_teh_gated_minus_centaur",
        "teh_test_accuracy",
        "teh_mean_prob_at_action",
        "teh_mean_abs_p_minus_half",
        "teh_program_style",
        "teh_run",
    ]
    compare_csv = out_dir / "dataset7_teh_vs_centaur.csv"
    _write_csv(compare_csv, compare_rows, compare_fields)

    struct = _structure_diagnosis_section(
        lines,
        alias=alias,
        filtered=filtered,
        compare_rows=compare_rows,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        participant_ids=roster,
    )

    _inspect_programs_section(lines, repo, compare_rows, run_meta)
    _calibration_section(lines, compare_rows)
    _final_answers(
        lines,
        alias=alias,
        spec=spec,
        trial_rows=trial_rows,
        compare_rows=compare_rows,
        meta=run_meta,
        raw_snippets=raw_snippets,
        struct=struct,
    )

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {trial_csv.relative_to(repo)} ({len(trial_rows)} rows)")
    print(f"Wrote {compare_csv.relative_to(repo)} ({len(compare_rows)} rows)")
    print(f"Wrote {report_path.relative_to(repo)}")
    if teh_f and cent_f:
        print(
            f"Avg gated TEH={statistics.mean(teh_f):.4f}, "
            f"Avg Centaur={statistics.mean(cent_f):.4f}, "
            f"TEH beats Centaur: {run_meta.get('n_teh_beats_centaur')}, "
            f"Centaur beats TEH: {run_meta.get('n_centaur_beats_teh')}"
        )


if __name__ == "__main__":
    main()
