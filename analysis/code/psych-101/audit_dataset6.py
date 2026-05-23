#!/usr/bin/env python3
"""
Audit Psych-101 dataset 6 (6sadeghiyeh2020temporal): structure, trials, TEH vs OpenEvolve.

Usage:
  python analysis/code/psych-101/audit_dataset6.py \\
    --dataset 6sadeghiyeh2020temporal \\
    --psych_dataset_split train

Outputs:
  analysis/data/psych101_dataset6_audit/dataset6_trial_stats.csv
  analysis/data/psych101_dataset6_audit/dataset6_teh_vs_openevolve.csv
  analysis/data/psych101_dataset6_audit/report.txt
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
    format_trial_for_prompt,
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
from data_modules.psych101_parsers import _RE_GAME_HEADER, _RE_INSTRUCTED, _RE_SLOT_PRESS

from analysis.code.utils import compare as cmp

_DEFAULT_DATASET = "6sadeghiyeh2020temporal"
_DEFAULT_OUT_DIR = "analysis/data/psych101_dataset6_audit"
_FIRST_N_PARTICIPANTS = 50
_RAW_TEXT_EXAMPLES = 10
_INSPECT_N_EACH_SIDE = 3
_SNIPPET_MAX = 700

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
        n_games = len(list(_RE_GAME_HEADER.finditer(text)))
        n_instructed = len(list(_RE_INSTRUCTED.finditer(text)))
        n_free_press = len(list(_RE_SLOT_PRESS.finditer(text)))

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
                "regex_game_headers": n_games,
                "regex_instructed": n_instructed,
                "regex_free_press": n_free_press,
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
        f"audit6_{program_path.parent.name}", str(program_path)
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


def _classify_bandit_program_style(code: str) -> str:
    """Rule-style tags for bandit / temporal slot-machine programs."""
    c = code or ""
    low = re.sub(r"\s+", "", c.lower())

    if len(c.strip()) < 60 and re.search(r"return\s+0\.5", c):
        return "simple_constant_majority"
    if re.search(r"return\s+(?:0|1)(?:\.0)?\s*;?\s*$", c, re.M) and "history" not in low:
        return "simple_constant_majority"
    if re.search(r"majority|mode\(|most_common", low):
        return "simple_constant_majority"

    if re.search(
        r"delay|temporaldiscount|intertemporal|discountfactor|"
        r"smaller.?sooner|larger.?later|discount_rate",
        low,
    ):
        return "temporal_discounting_delay"

    if re.search(
        r"history|action_counts|avg_reward|count_[ab]|bandit|exploration|"
        r"total_reward|freq_|recent_actions",
        low,
    ):
        return "history_heavy"

    if re.search(r"payoff|trial_index|game_id|n_trials_game", low) and "history" not in low:
        return "raw_feature_linear"

    if re.search(r"return\s+[01](?:\.0)?\s*$", c, re.M) or re.search(
        r"if\s+.*:\s*return\s+(?:0|1)", c
    ):
        return "threshold_rule"

    if re.search(r"sigmoid|linear|beta\s*\*|coeff", low):
        return "raw_feature_linear"

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


def _teh_vs_oe_rows(
    repo: Path,
    alias: str,
    psych_split: str,
    *,
    config_data: Mapping[str, Any],
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
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
    oe_beats: List[int] = []
    teh_beats: List[int] = []

    for pid in participant_ids:
        t_test = teh_scores.get(_TEST_LOGLIK, {}).get(pid)
        t_gated = teh_scores.get(_GATED_LOGLIK, {}).get(pid)
        t_train = teh_scores.get("train_loglik", {}).get(pid)
        t_val = teh_scores.get("val_loglik", {}).get(pid)
        oe_test = oe_scores.get(_TEST_LOGLIK, {}).get(pid)

        score_map = {
            "MLE": baseline_test.get("MLE", {}).get(pid),
            "prospect_theory": baseline_test.get("prospect_theory", {}).get(pid),
            "Centaur": baseline_test.get("Centaur", {}).get(pid),
            "openevolve": oe_test,
            "TEH_gated": t_gated,
            "TEH": t_test,
        }
        best_m, best_s = _best_method(score_map, use_gated_teh=True)

        gap = None
        if t_gated is not None and oe_test is not None:
            gap = t_gated - oe_test
            if gap < 0:
                oe_beats.append(pid)
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
        oe_dir = (oe_run / f"participant_{pid}") if oe_run else None
        teh_cal = _trial_calibration_metrics(test_trials, teh_dir / "best_program.py")
        oe_cal = (
            _trial_calibration_metrics(test_trials, oe_dir / "best_program.py")
            if oe_dir
            else {}
        )

        teh_code = ""
        oe_code = ""
        teh_prog = teh_dir / "best_program.py"
        oe_prog = oe_dir / "best_program.py" if oe_dir else None
        if teh_prog.is_file():
            teh_code = teh_prog.read_text(encoding="utf-8", errors="replace")
        if oe_prog and oe_prog.is_file():
            oe_code = oe_prog.read_text(encoding="utf-8", errors="replace")

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
                "openevolve_test_loglik": _fmt(oe_test),
                "mle_test_loglik": _fmt(score_map["MLE"]),
                "pt_test_loglik": _fmt(score_map["prospect_theory"]),
                "centaur_test_loglik": _fmt(score_map["Centaur"]),
                "best_method": best_m,
                "gap_teh_gated_minus_oe": _fmt(gap),
                "teh_test_accuracy": _fmt(teh_cal.get("accuracy")),
                "teh_mean_prob_at_action": _fmt(teh_cal.get("mean_prob_at_action")),
                "teh_mean_abs_p_minus_half": _fmt(teh_cal.get("mean_abs_p_minus_half")),
                "oe_test_accuracy": _fmt(oe_cal.get("accuracy")),
                "oe_mean_prob_at_action": _fmt(oe_cal.get("mean_prob_at_action")),
                "oe_mean_abs_p_minus_half": _fmt(oe_cal.get("mean_abs_p_minus_half")),
                "teh_program_style": _classify_bandit_program_style(teh_code),
                "oe_program_style": _classify_bandit_program_style(oe_code),
                "teh_run": str(teh_run.relative_to(repo)),
                "oe_run": str(oe_run.relative_to(repo)) if oe_run else "",
            }
        )

    meta = {
        "teh_run": str(teh_run.relative_to(repo)),
        "oe_run": str(oe_run.relative_to(repo)) if oe_run else "",
        "n_oe_beats_teh": len(oe_beats),
        "n_teh_beats_oe": len(teh_beats),
        "oe_beats_pids": oe_beats,
        "teh_beats_pids": teh_beats,
        "rows": rows,
    }
    return rows, meta


def _inspect_programs_section(
    lines: List[str],
    repo: Path,
    compare_rows: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> None:
    lines.append("")
    lines.append("=" * 80)
    lines.append("5. PROGRAM / RULE INSPECTION")
    lines.append("=" * 80)

    by_gap = []
    for r in compare_rows:
        g = _safe_float(r.get("gap_teh_gated_minus_oe"))
        if g is not None:
            by_gap.append((g, int(r["participant_id"])))
    by_gap.sort()

    oe_win = [pid for g, pid in by_gap if g < 0][: _INSPECT_N_EACH_SIDE]
    teh_win = [pid for g, pid in by_gap if g > 0][-_INSPECT_N_EACH_SIDE :]
    teh_win.reverse()

    lines.append(f"OpenEvolve beats TEH (gated): {meta.get('oe_beats_pids')}")
    lines.append(f"TEH beats OpenEvolve: {meta.get('teh_beats_pids')}")
    lines.append("")
    lines.append(f"Inspecting up to {_INSPECT_N_EACH_SIDE} TEH-win and {_INSPECT_N_EACH_SIDE} OE-win participants.")

    row_by_pid = {int(r["participant_id"]): r for r in compare_rows}
    teh_run = repo / str(meta["teh_run"])
    oe_run = repo / str(meta["oe_run"]) if meta.get("oe_run") else None

    for label, pids in (("TEH-win (largest gaps)", teh_win), ("OpenEvolve-win (largest gaps)", oe_win)):
        lines.append("")
        lines.append(f"--- {label} ---")
        for pid in pids:
            r = row_by_pid.get(pid, {})
            lines.append(
                f"\nParticipant {pid}: gap_teh_gated_minus_oe={r.get('gap_teh_gated_minus_oe')} "
                f"best={r.get('best_method')} teh_gated={r.get('teh_gated_test_loglik')} "
                f"oe_test={r.get('openevolve_test_loglik')}"
            )
            lines.append(f"  TEH style: {r.get('teh_program_style')} | OE style: {r.get('oe_program_style')}")
            for tag, run in (("TEH", teh_run), ("OpenEvolve", oe_run)):
                if run is None:
                    continue
                prog = run / f"participant_{pid}" / "best_program.py"
                if not prog.is_file():
                    lines.append(f"  {tag}: (no {prog.name})")
                    continue
                code = prog.read_text(encoding="utf-8", errors="replace")
                style = _classify_bandit_program_style(code)
                lines.append(f"  {tag} best_program.py ({style}, {len(code)} chars):")
                for snip_line in _program_snippet(code).splitlines()[:18]:
                    lines.append(f"    {snip_line}")


def _calibration_section(lines: List[str], compare_rows: Sequence[Mapping[str, Any]]) -> None:
    lines.append("")
    lines.append("=" * 80)
    lines.append("6. CONFIDENCE / CALIBRATION DIAGNOSIS (test split)")
    lines.append("=" * 80)

    def _agg(key: str) -> Optional[float]:
        vals = [_safe_float(r.get(key)) for r in compare_rows]
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    lines.append("Means over participants 0–49 (recomputed from best_program.py on test trials):")
    for prefix, label in (("teh_", "TEH"), ("oe_", "OpenEvolve")):
        lines.append(
            f"  {label}: accuracy={_fmt(_agg(f'{prefix}test_accuracy'))} "
            f"mean_prob@action={_fmt(_agg(f'{prefix}mean_prob_at_action'))} "
            f"mean|p-0.5|={_fmt(_agg(f'{prefix}mean_abs_p_minus_half'))}"
        )

    lines.append("")
    lines.append("Per-participant gated/test loglik (from run CSVs):")
    teh_g = [_safe_float(r["teh_gated_test_loglik"]) for r in compare_rows]
    oe_t = [_safe_float(r["openevolve_test_loglik"]) for r in compare_rows]
    teh_g_f = [v for v in teh_g if v is not None]
    oe_t_f = [v for v in oe_t if v is not None]
    if teh_g_f:
        lines.append(
            f"  TEH gated_test_loglik: mean={_fmt(statistics.mean(teh_g_f))} "
            f"median={_fmt(statistics.median(teh_g_f))}"
        )
    if oe_t_f:
        lines.append(
            f"  OpenEvolve test_loglik: mean={_fmt(statistics.mean(oe_t_f))} "
            f"median={_fmt(statistics.median(oe_t_f))}"
        )

    conf_oe_wins = 0
    safe_teh_wins = 0
    for r in compare_rows:
        g = _safe_float(r.get("gap_teh_gated_minus_oe"))
        teh_p = _safe_float(r.get("teh_mean_abs_p_minus_half"))
        oe_p = _safe_float(r.get("oe_mean_abs_p_minus_half"))
        if g is None or teh_p is None or oe_p is None:
            continue
        if g < 0 and oe_p > teh_p + 0.03:
            conf_oe_wins += 1
        if g > 0 and teh_p <= oe_p + 0.02:
            safe_teh_wins += 1

    lines.append("")
    lines.append(
        f"Participants where OE beats TEH on loglik AND OE is more confident "
        f"(mean|p-0.5| higher by >0.03): {conf_oe_wins}"
    )
    lines.append(
        f"Participants where TEH beats OE on loglik AND TEH is not less confident "
        f"(teh |p-0.5| <= oe + 0.02): {safe_teh_wins}"
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
) -> None:
    valid = [r for r in trial_rows if r.get("split_valid")]
    compare_f = [r for r in compare_rows if _safe_float(r.get("teh_gated_test_loglik")) is not None]

    teh_styles = Counter(r.get("teh_program_style") for r in compare_rows)
    oe_styles = Counter(r.get("oe_program_style") for r in compare_rows)
    best_counts = Counter(r.get("best_method") for r in compare_rows)

    median_trials = statistics.median(int(r["parsed_total_trials"]) for r in valid) if valid else 0
    median_blocks = statistics.median(int(r["parsed_blocks"]) for r in valid) if valid else 0
    phases_in_text = any("instructed" in s.lower() for s in raw_snippets)
    games_in_text = any(re.search(r"Game\s+\d+", s, re.I) for s in raw_snippets)

    teh_g = [_safe_float(r["teh_gated_test_loglik"]) for r in compare_f]
    oe_t = [_safe_float(r["openevolve_test_loglik"]) for r in compare_f]
    avg_teh = statistics.mean(teh_g) if teh_g else None
    avg_oe = statistics.mean(oe_t) if oe_t else None

    lines.append("")
    lines.append("=" * 80)
    lines.append("7. FINAL ANSWERS")
    lines.append("=" * 80)

    lines.append("")
    lines.append("A. What is dataset 6 actually measuring?")
    lines.append(
        "   A multi-game two-armed bandit (slot machine) task from Sadeghiyeh et al. (2020), "
        "parsed from Psych-101 NL transcripts. Each participant plays several 'Game' blocks; "
        "each game has instructed trials (forced key presses with stated payoffs) followed by "
        "free-choice trials ('You press <<KEY>> and get N points'). "
        f"Evidence: parser={spec.get('parser')!r}, schema_type={spec.get('schema_type')!r}, "
        f"median parsed blocks={median_blocks:.0f}, median trials={median_trials:.0f}, "
        f"regex game headers present={games_in_text}, instructed phase in text={phases_in_text}."
    )

    lines.append("")
    lines.append("B. Is it a temporal discounting / delayed reward task?")
    lines.append(
        "   NOT primarily intertemporal choice (smaller-sooner vs larger-later). "
        "The alias references 'temporal' from the source paper, but the parsed structure is "
        "bandit learning with immediate point feedback per pull, plus an instructed learning phase. "
        "No delay amounts or future dates appear in parsed problem fields "
        "(game_id, phase, trial_index, payoff, machine_options)."
    )

    lines.append("")
    lines.append("C. Why is OpenEvolve competitive here?")
    lines.append(
        f"   Test loglik averages are close (TEH gated mean={_fmt(avg_teh)}, OE test mean={_fmt(avg_oe)}). "
        f"num_best among MLE/PT/Centaur/OE/TEH(gated): {dict(best_counts)}. "
        f"OE wins {meta.get('n_oe_beats_teh')} participants vs TEH wins {meta.get('n_teh_beats_oe')} on gated loglik. "
        f"OE program styles (n=50): {dict(oe_styles)} — often history-heavy bandit rules. "
        f"TEH styles: {dict(teh_styles)} — includes many constant/unclear programs when search underfits."
    )

    lines.append("")
    lines.append("D. Is TEH losing to OpenEvolve because of search, confidence, prompt structure, or inductive bias?")
    lines.append(
        "   Mixed. Bandit tasks reward history-based exploitation (OE inductive bias). "
        "Several TEH programs are trivial (return 0.5) or unclear, suggesting search/prompt limits. "
        "Gating sometimes helps TEH (gated vs raw test). Confidence differences are modest; "
        "see section 6 counts for OE winning via higher |p-0.5|."
    )

    lines.append("")
    lines.append("E. Should dataset 6 still be counted as a TEH win, a tie, or a competitive case?")
    n_teh_best = best_counts.get("TEH", 0)
    n_oe_best = best_counts.get("openevolve", 0)
    if avg_teh is not None and avg_oe is not None and abs(avg_teh - avg_oe) < 0.02 and abs(n_teh_best - n_oe_best) <= 3:
        verdict = "competitive / statistical tie — not a clear TEH win"
    elif n_teh_best > n_oe_best + 2:
        verdict = "modest TEH win on num_best, but averages nearly tied"
    else:
        verdict = "competitive case"
    lines.append(f"   {verdict} (TEH num_best={n_teh_best}, OE num_best={n_oe_best}).")

    lines.append("")
    lines.append("F. What prompt/seed change would likely help TEH on this dataset?")
    lines.append(
        "   Emphasize per-game history aggregation (running payoffs per arm), instructed vs free phases, "
        "and block-level split (history resets each game). Increase evolution budget / discourage 0.5 constants. "
        "Ensure train prompts include enough free-phase trials per game; consider seed programs with "
        "epsilon-greedy or Thompson-style arm means."
    )


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(description="Audit dataset 6 (Sadeghiyeh temporal bandit).")
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

    compare_rows, run_meta = _teh_vs_oe_rows(
        repo,
        alias,
        psych_split,
        config_data=config_data,
        local_dataset=args.local_dataset,
        mixed_gambles_csv="",
        filter_mixed_gambles=False,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        participant_ids=roster,
    )

    lines: List[str] = []
    lines.append("PSYCH-101 DATASET 6 AUDIT (6sadeghiyeh2020temporal)")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Repo: {repo}")
    lines.append(f"Dataset alias: {alias}")
    lines.append(f"psych_dataset_split: {psych_split}")
    lines.append(f"Split: ratio={args.split_ratio}, seed={args.split_seed}")
    lines.append(f"TEH run: {run_meta.get('teh_run')}")
    lines.append(f"OpenEvolve run: {run_meta.get('oe_run')}")

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
        inst = t0.split("Game")[0][:500] if "Game" in t0 else t0[:500]
        lines.append(f"  Instruction head: {inst!r}")
    lines.append(
        "  Participants choose between two labeled slot machines across multiple games; "
        "each game includes instructed pulls then voluntary pulls with point feedback."
    )

    # --- 2. Parsed schema ---
    lines.append("")
    lines.append("=" * 80)
    lines.append("2. PARSED SCHEMA")
    lines.append("=" * 80)
    lines.append(f"parser name: {spec.get('parser')!r}")
    lines.append(f"schema_type: {spec.get('schema_type')!r}")
    lines.append(
        "problem fields (per trial): trial_index, phase (instructed|free), "
        "machine_options, payoff, plus block-static game_id, n_trials_game, option_keys"
    )

    if filtered:
        try:
            exp0 = parse_psych101_binary_row(dict(filtered[0]), alias)
            train0, val0, test0, opts0 = split_psych_experiment(
                exp0, split_ratio=args.split_ratio, split_seed=args.split_seed
            )
            p0 = train0[0]["problem"] if train0 else (test0[0]["problem"] if test0 else {})
            sem = _action_semantics_for_schema(
                p0.get("option_keys", []),
                str(p0.get("schema_type", "C")),
                p0,
                is_gamble=False,
            )
            lines.append(f"action semantics: {sem}")
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
            lines.append("Example parsed val trial:" if val0 else "Example parsed val: (none)")
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
        lines.append(
            f"parsed trials: min={min(pt)}, median={statistics.median(pt)}, max={max(pt)}"
        )
        lines.append(
            f"train/val/test medians: "
            f"{statistics.median(int(r['train_trials']) for r in valid):.0f} / "
            f"{statistics.median(int(r['val_trials']) for r in valid):.0f} / "
            f"{statistics.median(int(r['test_trials']) for r in valid):.0f}"
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

    lines.append("")
    lines.append("Per-participant sample (first 10):")
    lines.append(
        "  pid | trials | blocks | train | test | maj_rate | BIR | a0/a1"
    )
    for r in trial_rows[:10]:
        lines.append(
            f"  {r['teh_participant_id']:3d} | {r['parsed_total_trials']:6} | "
            f"{r['parsed_blocks']:6} | {r['train_trials']:5} | {r['test_trials']:4} | "
            f"{r['majority_action_rate_total']:8} | {r['BIR']:6} | "
            f"{r['action_0_count']}/{r['action_1_count']}"
        )

    trial_fields = [
        "teh_participant_id",
        "hf_participant",
        "raw_text_chars",
        "regex_game_headers",
        "regex_instructed",
        "regex_free_press",
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
    trial_csv = out_dir / "dataset6_trial_stats.csv"
    _write_csv(trial_csv, trial_rows, trial_fields)

    # --- 4. Run comparison ---
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"4. FIRST-{args.participants} RUN STATISTICS")
    lines.append("=" * 80)
    best_counts = Counter(r.get("best_method") for r in compare_rows)
    lines.append(f"num_best: {dict(best_counts)}")
    teh_g = [_safe_float(r["teh_gated_test_loglik"]) for r in compare_rows]
    oe_t = [_safe_float(r["openevolve_test_loglik"]) for r in compare_rows]
    teh_f = [v for v in teh_g if v is not None]
    oe_f = [v for v in oe_t if v is not None]
    if teh_f:
        lines.append(f"Avg TEH gated_test_loglik: {statistics.mean(teh_f):.4f}")
    if oe_f:
        lines.append(f"Avg OpenEvolve test_loglik: {statistics.mean(oe_f):.4f}")
    lines.append(f"OpenEvolve beats TEH (gated): pids={run_meta.get('oe_beats_pids')}")
    lines.append(f"TEH beats OpenEvolve: pids={run_meta.get('teh_beats_pids')}")

    lines.append("")
    lines.append("Per-participant table (abbrev):")
    lines.append(
        "  pid | train | test | teh_gated | oe_test | MLE | best | gap"
    )
    for r in compare_rows:
        lines.append(
            f"  {int(r['participant_id']):3d} | {r['train_trials']:5} | {r['test_trials']:4} | "
            f"{r['teh_gated_test_loglik']:9} | {r['openevolve_test_loglik']:7} | "
            f"{r['mle_test_loglik']:6} | {r['best_method']:10} | {r['gap_teh_gated_minus_oe']}"
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
        "openevolve_test_loglik",
        "mle_test_loglik",
        "pt_test_loglik",
        "centaur_test_loglik",
        "best_method",
        "gap_teh_gated_minus_oe",
        "teh_test_accuracy",
        "teh_mean_prob_at_action",
        "teh_mean_abs_p_minus_half",
        "oe_test_accuracy",
        "oe_mean_prob_at_action",
        "oe_mean_abs_p_minus_half",
        "teh_program_style",
        "oe_program_style",
        "teh_run",
        "oe_run",
    ]
    compare_csv = out_dir / "dataset6_teh_vs_openevolve.csv"
    _write_csv(compare_csv, compare_rows, compare_fields)

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
    )

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {trial_csv.relative_to(repo)} ({len(trial_rows)} rows)")
    print(f"Wrote {compare_csv.relative_to(repo)} ({len(compare_rows)} rows)")
    print(f"Wrote {report_path.relative_to(repo)}")
    if teh_f and oe_f:
        print(
            f"Avg gated TEH={statistics.mean(teh_f):.4f}, "
            f"Avg OE test={statistics.mean(oe_f):.4f}, "
            f"TEH beats OE: {run_meta.get('n_teh_beats_oe')}, "
            f"OE beats TEH: {run_meta.get('n_oe_beats_teh')}"
        )


if __name__ == "__main__":
    main()
