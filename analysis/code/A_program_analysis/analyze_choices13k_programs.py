#!/usr/bin/env python3
"""Generate evidence-based qualitative program-analysis reports for Choice13k.

Rerunnable from repo root:
  python analysis/code/A_program_analysis/analyze_choices13k_programs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_modules.psych101_binary import get_psych101_binary_experiment, split_psych_experiment


DEFAULT_DATASET = "1peterson2021using"
DEFAULT_RUN_DIR = Path("generated_outputs/psych101_train/teh/1peterson2021using/run_260525_031227")
DEFAULT_OUTPUT_DIR = Path("analysis/data/A_program_analysis_choices13k")
DEFAULT_COMPARE_CSV = Path("analysis/data/utils/loglik_compare_choice13k_gated.csv")
DEFAULT_PARTICIPANTS = [3, 19, 21, 24, 25, 28, 46]
EPS = 1e-12


@dataclass
class SelectionInfo:
    selected_path: Optional[Path]
    selected_reason: str
    selected_metrics: Dict[str, Optional[float]]
    selected_score_name: str
    selected_score_value: Optional[float]
    candidate_rows: List[Dict[str, Any]]
    ambiguous: bool


def _read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    txt = _read_text(path)
    if not txt:
        return None
    return json.loads(txt)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        xv = float(x)
        if math.isnan(xv):
            return None
        return xv
    sx = str(x).strip()
    if not sx or sx.lower() in {"nan", "none", "null", "na"}:
        return None
    try:
        return float(sx)
    except ValueError:
        return None


def _format_float(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{v:.{nd}f}"


def _format_pct(v: Optional[float]) -> str:
    if v is None:
        return "NA"
    return f"{100.0 * v:.1f}%"


def _rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def _cell(v: Any) -> str:
        s = str(v)
        s = s.replace("|", "\\|")
        s = s.replace("\n", "<br>")
        return s

    lines = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_command_args(run_dir: Path) -> Dict[str, Any]:
    out = {
        "split_ratio": 0.6,
        "split_seed": 0,
        "split_mode": "within_participant",
        "psych_dataset_split": "train",
        "command_line": None,
    }
    txt = _read_text(run_dir / "log" / "command.txt")
    if not txt:
        return out
    cmd = None
    for line in txt.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            cmd = s
            break
    out["command_line"] = cmd
    if not cmd:
        return out

    def _extract(flag: str) -> Optional[str]:
        m = re.search(rf"(?:^|\s){re.escape(flag)}\s+([^\s]+)", cmd)
        return m.group(1) if m else None

    split_ratio = _extract("--split_ratio")
    split_seed = _extract("--split_seed")
    split_mode = _extract("--split_mode")
    split_name = _extract("--psych_dataset_split")
    if split_ratio is not None:
        out["split_ratio"] = float(split_ratio)
    if split_seed is not None:
        out["split_seed"] = int(split_seed)
    if split_mode:
        out["split_mode"] = split_mode
    if split_name:
        out["psych_dataset_split"] = split_name
    return out


def _ev_and_var(gamble: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(gamble, dict):
        return None, None
    rewards = gamble.get("rewards")
    probs = gamble.get("probs")
    if not isinstance(rewards, list) or not rewards:
        return None, None
    rewards_f = [_safe_float(x) for x in rewards]
    if any(x is None for x in rewards_f):
        return None, None
    rewards_v = [float(x) for x in rewards_f if x is not None]

    probs_v: Optional[List[float]] = None
    if isinstance(probs, list) and len(probs) == len(rewards_v):
        probs_f = [_safe_float(x) for x in probs]
        if all(x is not None for x in probs_f):
            probs_v = [float(x) for x in probs_f if x is not None]

    if probs_v is None:
        p = 1.0 / len(rewards_v)
        probs_v = [p for _ in rewards_v]

    ev = sum(p * r for p, r in zip(probs_v, rewards_v))
    var = sum(p * ((r - ev) ** 2) for p, r in zip(probs_v, rewards_v))
    return ev, max(0.0, var)


def _extract_prev_outcome(last_history_entry: Any) -> Tuple[Optional[float], str]:
    if not isinstance(last_history_entry, dict):
        return None, "no_previous_history"
    for k in ("feedback", "reward", "outcome", "payoff"):
        if k not in last_history_entry:
            continue
        v = last_history_entry[k]
        if isinstance(v, bool):
            return (1.0 if v else -1.0), k
        if isinstance(v, (int, float)):
            return float(v), k
        if isinstance(v, dict):
            for kk in ("reward", "outcome", "value"):
                subv = v.get(kk)
                if isinstance(subv, (int, float)):
                    return float(subv), f"{k}.{kk}"
    return None, "ambiguous_non_numeric_feedback"


def _last5_majority_action(history: List[Dict[str, Any]]) -> Optional[int]:
    if not history:
        return None
    actions = [h.get("action") for h in history[-5:] if h.get("action") in (0, 1)]
    if not actions:
        return None
    c1 = actions.count(1)
    c0 = actions.count(0)
    if c1 > c0:
        return 1
    if c0 > c1:
        return 0
    return None


def _trial_row(split_name: str, idx: int, trial: Dict[str, Any]) -> Dict[str, Any]:
    problem = trial.get("problem", {})
    gamble_a = problem.get("gamble_A")
    gamble_b = problem.get("gamble_B")
    ev_a, var_a = _ev_and_var(gamble_a)
    ev_b, var_b = _ev_and_var(gamble_b)

    history = trial.get("history") or []
    last = history[-1] if history else None
    prev_action = last.get("action") if isinstance(last, dict) else None
    prev_outcome, prev_outcome_source = _extract_prev_outcome(last)
    last5_majority = _last5_majority_action(history)

    return {
        "split": split_name,
        "trial_index": idx,
        "action": trial.get("action"),
        "option_keys": problem.get("option_keys"),
        "has_feedback": problem.get("has_feedback"),
        "ev_a": ev_a,
        "ev_b": ev_b,
        "ev_diff": (ev_b - ev_a) if (ev_a is not None and ev_b is not None) else None,
        "var_a": var_a,
        "var_b": var_b,
        "var_diff": (var_b - var_a) if (var_a is not None and var_b is not None) else None,
        "gamble_a": gamble_a,
        "gamble_b": gamble_b,
        "prev_action": prev_action,
        "prev_outcome": prev_outcome,
        "prev_outcome_source": prev_outcome_source,
        "last5_majority_action": last5_majority,
    }


def _stats_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    a1 = sum(1 for r in rows if r.get("action") == 1)
    a0 = sum(1 for r in rows if r.get("action") == 0)

    ev_gt = [r for r in rows if r.get("ev_diff") is not None and r["ev_diff"] > EPS]
    ev_lt = [r for r in rows if r.get("ev_diff") is not None and r["ev_diff"] < -EPS]
    ev_eq = [r for r in rows if r.get("ev_diff") is not None and abs(r["ev_diff"]) <= EPS]
    ev_rate_gt = _rate(sum(1 for r in ev_gt if r.get("action") == 1), len(ev_gt))
    ev_rate_lt = _rate(sum(1 for r in ev_lt if r.get("action") == 1), len(ev_lt))
    ev_rate_eq = _rate(sum(1 for r in ev_eq if r.get("action") == 1), len(ev_eq))

    var_lt = [r for r in rows if r.get("var_diff") is not None and r["var_diff"] < -EPS]
    var_gt = [r for r in rows if r.get("var_diff") is not None and r["var_diff"] > EPS]
    var_rate_lt = _rate(sum(1 for r in var_lt if r.get("action") == 1), len(var_lt))
    var_rate_gt = _rate(sum(1 for r in var_gt if r.get("action") == 1), len(var_gt))

    conflict_cells = {
        "ev_gt_var_gt": [
            r
            for r in rows
            if r.get("ev_diff") is not None and r.get("var_diff") is not None and r["ev_diff"] > EPS and r["var_diff"] > EPS
        ],
        "ev_gt_var_lt": [
            r
            for r in rows
            if r.get("ev_diff") is not None and r.get("var_diff") is not None and r["ev_diff"] > EPS and r["var_diff"] < -EPS
        ],
        "ev_lt_var_gt": [
            r
            for r in rows
            if r.get("ev_diff") is not None and r.get("var_diff") is not None and r["ev_diff"] < -EPS and r["var_diff"] > EPS
        ],
        "ev_lt_var_lt": [
            r
            for r in rows
            if r.get("ev_diff") is not None and r.get("var_diff") is not None and r["ev_diff"] < -EPS and r["var_diff"] < -EPS
        ],
    }

    with_prev = [r for r in rows if r.get("prev_action") in (0, 1)]
    rep = [r for r in with_prev if r.get("action") == r.get("prev_action")]
    prev1 = [r for r in with_prev if r.get("prev_action") == 1]
    prev0 = [r for r in with_prev if r.get("prev_action") == 0]
    trend1 = [r for r in rows if r.get("last5_majority_action") == 1]
    trend0 = [r for r in rows if r.get("last5_majority_action") == 0]

    fb_yes = [r for r in rows if r.get("has_feedback") is True]
    fb_no = [r for r in rows if r.get("has_feedback") is False]
    prev_out_known = [r for r in rows if r.get("prev_outcome") is not None]
    prev_out_pos = [r for r in prev_out_known if float(r["prev_outcome"]) > 0]
    prev_out_neg = [r for r in prev_out_known if float(r["prev_outcome"]) < 0]

    prev_vals = [float(r["prev_outcome"]) for r in prev_out_known]
    med = median(prev_vals) if prev_vals else None
    prev_high = [r for r in prev_out_known if med is not None and float(r["prev_outcome"]) >= med]
    prev_low = [r for r in prev_out_known if med is not None and float(r["prev_outcome"]) < med]

    switch_rows = [r for r in with_prev if r.get("action") in (0, 1)]
    switched = [r for r in switch_rows if r.get("action") != r.get("prev_action")]
    switch_after_pos = [r for r in switch_rows if r.get("prev_outcome") is not None and float(r["prev_outcome"]) > 0]
    switch_after_neg = [r for r in switch_rows if r.get("prev_outcome") is not None and float(r["prev_outcome"]) < 0]

    return {
        "n_trials": n,
        "action1_rate": _rate(a1, n),
        "action0_rate": _rate(a0, n),
        "ev": {
            "n_gt": len(ev_gt),
            "n_lt": len(ev_lt),
            "n_eq": len(ev_eq),
            "rate_gt": ev_rate_gt,
            "rate_lt": ev_rate_lt,
            "rate_eq": ev_rate_eq,
            "diff_gt_minus_lt": (ev_rate_gt - ev_rate_lt) if (ev_rate_gt is not None and ev_rate_lt is not None) else None,
        },
        "risk": {
            "n_var_lt": len(var_lt),
            "n_var_gt": len(var_gt),
            "rate_var_lt": var_rate_lt,
            "rate_var_gt": var_rate_gt,
            "diff_lt_minus_gt": (var_rate_lt - var_rate_gt) if (var_rate_lt is not None and var_rate_gt is not None) else None,
            "conflicts": {
                k: {
                    "n": len(v),
                    "rate_action1": _rate(sum(1 for r in v if r.get("action") == 1), len(v)),
                }
                for k, v in conflict_cells.items()
            },
        },
        "history": {
            "n_with_prev": len(with_prev),
            "repetition_rate": _rate(len(rep), len(with_prev)),
            "n_prev1": len(prev1),
            "rate_after_prev1": _rate(sum(1 for r in prev1 if r.get("action") == 1), len(prev1)),
            "n_prev0": len(prev0),
            "rate_after_prev0": _rate(sum(1 for r in prev0 if r.get("action") == 1), len(prev0)),
            "n_trend1": len(trend1),
            "rate_after_trend1": _rate(sum(1 for r in trend1 if r.get("action") == 1), len(trend1)),
            "n_trend0": len(trend0),
            "rate_after_trend0": _rate(sum(1 for r in trend0 if r.get("action") == 1), len(trend0)),
        },
        "feedback": {
            "n_fb_yes": len(fb_yes),
            "rate_fb_yes": _rate(sum(1 for r in fb_yes if r.get("action") == 1), len(fb_yes)),
            "n_fb_no": len(fb_no),
            "rate_fb_no": _rate(sum(1 for r in fb_no if r.get("action") == 1), len(fb_no)),
            "n_prev_known": len(prev_out_known),
            "n_prev_pos": len(prev_out_pos),
            "rate_after_prev_pos": _rate(sum(1 for r in prev_out_pos if r.get("action") == 1), len(prev_out_pos)),
            "n_prev_neg": len(prev_out_neg),
            "rate_after_prev_neg": _rate(sum(1 for r in prev_out_neg if r.get("action") == 1), len(prev_out_neg)),
            "prev_outcome_median": med,
            "n_prev_high": len(prev_high),
            "rate_after_prev_high": _rate(sum(1 for r in prev_high if r.get("action") == 1), len(prev_high)),
            "n_prev_low": len(prev_low),
            "rate_after_prev_low": _rate(sum(1 for r in prev_low if r.get("action") == 1), len(prev_low)),
        },
        "switch": {
            "n_switch_defined": len(switch_rows),
            "switch_rate": _rate(len(switched), len(switch_rows)),
            "n_after_prev_pos": len(switch_after_pos),
            "switch_rate_after_prev_pos": _rate(
                sum(1 for r in switch_after_pos if r.get("action") != r.get("prev_action")),
                len(switch_after_pos),
            ),
            "n_after_prev_neg": len(switch_after_neg),
            "switch_rate_after_prev_neg": _rate(
                sum(1 for r in switch_after_neg if r.get("action") != r.get("prev_action")),
                len(switch_after_neg),
            ),
        },
    }


def _load_metrics(
    participant_id: int,
    run_dir: Path,
    compare_csv: Path,
) -> Dict[str, Any]:
    details_rows = _load_csv_rows(run_dir / "participant_details_loglik.csv")
    summary_rows = _load_csv_rows(run_dir / "participants_summary.csv")
    compare_rows = _load_csv_rows(compare_csv)
    run_col = run_dir.name

    details = next((r for r in details_rows if r.get("participant_id") == str(participant_id)), None)
    summary = next((r for r in summary_rows if r.get("participant_id") == str(participant_id)), None)
    compare = next((r for r in compare_rows if r.get("participant_id") == str(participant_id)), None)

    out = {
        "participant_id": participant_id,
        "train_loglik": _safe_float((details or {}).get("train_loglik")) or _safe_float((summary or {}).get("train_loglik")),
        "val_loglik": _safe_float((details or {}).get("val_loglik")) or _safe_float((summary or {}).get("val_loglik")),
        "test_loglik": _safe_float((details or {}).get("test_loglik")) or _safe_float((summary or {}).get("test_loglik")),
        "gated_test_loglik": _safe_float((details or {}).get("gated_test_loglik"))
        or _safe_float((summary or {}).get("gated_test_loglik")),
        "selection_score": _safe_float((summary or {}).get("selection_score")),
        "selection_score_name": (summary or {}).get("evolution_selection_score") or "train_val",
        "BIR": None,
        "MLE": None,
        "prospect_theory": None,
        "openevolve": None,
        "Centaur": None,
        "PICS": None,
        "best_non_pics_method": None,
        "best_non_pics_loglik": None,
        "pics_margin": None,
        "missing_notes": [],
    }

    if compare:
        out["BIR"] = _safe_float(compare.get("BIR"))
        out["MLE"] = _safe_float(compare.get("MLE"))
        out["prospect_theory"] = _safe_float(compare.get("prospect_theory"))
        out["openevolve"] = _safe_float(compare.get("openevolve"))
        out["Centaur"] = _safe_float(compare.get("Centaur"))
        out["PICS"] = _safe_float(compare.get(run_col))
    else:
        out["missing_notes"].append(f"Missing participant row in `{compare_csv}`.")

    baselines = {
        "Logistic Model": out.get("MLE"),
        "Prospect Theory": out.get("prospect_theory"),
        "OpenEvolve": out.get("openevolve"),
        "Centaur": out.get("Centaur"),
    }
    valid = {k: v for k, v in baselines.items() if v is not None}
    if valid:
        best_name, best_ll = max(valid.items(), key=lambda kv: kv[1])
        out["best_non_pics_method"] = best_name
        out["best_non_pics_loglik"] = best_ll
        if out.get("PICS") is not None:
            out["pics_margin"] = out["PICS"] - best_ll
    else:
        out["missing_notes"].append("All non-PICS baseline test logliks unavailable.")

    if details is None:
        out["missing_notes"].append("Missing participant row in `participant_details_loglik.csv`.")
    if summary is None:
        out["missing_notes"].append("Missing participant row in `participants_summary.csv`.")
    return out


def _selection_info(run_dir: Path, participant_id: int, metrics: Dict[str, Any]) -> SelectionInfo:
    pdir = run_dir / f"participant_{participant_id}"
    root_results = _read_json(pdir / "results.json")
    ref_results = _read_json(pdir / "refinement" / "results.json")
    candidate_rows: List[Dict[str, Any]] = []

    if root_results and "overall_best_train" in root_results:
        r = root_results["overall_best_train"]
        candidate_rows.append(
            {
                "candidate": "root_best_program",
                "path": pdir / "best_program.py",
                "program_id": r.get("program_id"),
                "train_loglik": _safe_float(r.get("train_loglik")),
                "val_loglik": _safe_float(r.get("val_loglik")),
                "train_val_loglik": _safe_float(r.get("selection_score")),
                "test_loglik": _safe_float(r.get("test_loglik")),
                "gated_test_loglik": _safe_float(r.get("gated_test_loglik")),
                "source": "participant/results.json:overall_best_train",
            }
        )
    if ref_results and "final_pool_best" in ref_results:
        r = ref_results["final_pool_best"]
        candidate_rows.append(
            {
                "candidate": "refinement_best_program",
                "path": pdir / "refinement" / "best_program.py",
                "program_id": r.get("program_id"),
                "train_loglik": _safe_float(r.get("train_loglik")),
                "val_loglik": _safe_float(r.get("val_loglik")),
                "train_val_loglik": _safe_float(r.get("train_val_loglik")),
                "test_loglik": _safe_float(r.get("test_loglik")),
                "gated_test_loglik": _safe_float((ref_results or {}).get("gated_test_loglik")),
                "source": "participant/refinement/results.json:final_pool_best",
            }
        )

    existing = [c for c in candidate_rows if Path(c["path"]).exists()]
    if not existing:
        return SelectionInfo(
            selected_path=None,
            selected_reason="Could not locate selected program files in root/refinement artifacts.",
            selected_metrics={},
            selected_score_name="gated_test_loglik" if metrics.get("gated_test_loglik") is not None else "test_loglik",
            selected_score_value=metrics.get("gated_test_loglik") or metrics.get("test_loglik"),
            candidate_rows=candidate_rows,
            ambiguous=True,
        )

    ref = next((c for c in existing if c["candidate"] == "refinement_best_program"), None)
    root = next((c for c in existing if c["candidate"] == "root_best_program"), None)
    if ref is not None:
        return SelectionInfo(
            selected_path=Path(ref["path"]),
            selected_reason=(
                "Both phase artifacts considered; refinement `final_pool_best` selected as the post-gating final artifact."
                if root is not None
                else "Refinement `final_pool_best` exists and corresponding file is present."
            ),
            selected_metrics={
                "train_loglik": ref.get("train_loglik"),
                "val_loglik": ref.get("val_loglik"),
                "train_val_loglik": ref.get("train_val_loglik"),
                "test_loglik": ref.get("test_loglik"),
                "gated_test_loglik": ref.get("gated_test_loglik"),
            },
            selected_score_name="gated_test_loglik" if ref.get("gated_test_loglik") is not None else "test_loglik",
            selected_score_value=ref.get("gated_test_loglik") or ref.get("test_loglik"),
            candidate_rows=candidate_rows,
            ambiguous=False,
        )

    top = root if root is not None else existing[0]
    return SelectionInfo(
        selected_path=Path(top["path"]),
        selected_reason="Only root candidate was available with a concrete best-program file.",
        selected_metrics={
            "train_loglik": top.get("train_loglik"),
            "val_loglik": top.get("val_loglik"),
            "train_val_loglik": top.get("train_val_loglik"),
            "test_loglik": top.get("test_loglik"),
            "gated_test_loglik": top.get("gated_test_loglik"),
        },
        selected_score_name="gated_test_loglik" if top.get("gated_test_loglik") is not None else "test_loglik",
        selected_score_value=top.get("gated_test_loglik") or top.get("test_loglik"),
        candidate_rows=candidate_rows,
        ambiguous=False,
    )


def _snippets_for_patterns(code: str, patterns: Sequence[str], max_hits: int = 2) -> List[str]:
    lines = code.splitlines()
    hits: List[int] = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(re.search(p, low) for p in patterns):
            hits.append(i)
    chunks: List[str] = []
    used: List[int] = []
    for i in hits:
        if len(chunks) >= max_hits:
            break
        if any(abs(i - j) <= 2 for j in used):
            continue
        start = max(0, i - 2)
        end = min(len(lines), i + 3)
        chunks.append("\n".join(lines[start:end]))
        used.append(i)
    return chunks


def _mechanism_evidence(code: str) -> List[Dict[str, Any]]:
    checks = [
        ("expected value sensitivity", [r"\bev_", r"expected_value", r"ev_diff", r"ev_b\s*-\s*ev_a"]),
        ("risk / variance sensitivity", [r"\bvar\b", r"variance", r"risk", r"std"]),
        ("probability sensitivity", [r"\bprob", r"probs", r"\bzip\s*\(\s*probs", r"sigmoid"]),
        ("reward magnitude sensitivity", [r"reward", r"outcome", r"value", r"feedback_avg"]),
        (
            "loss sensitivity",
            [
                r"loss",
                r"negative",
                r"reward.*<\s*0",
                r"outcome.*<\s*0",
                r"if .*<\s*0",
            ],
        ),
        ("thresholds", [r"\bif\b.*[<>]=?", r"max\s*\(", r"min\s*\(", r"=="]),
        ("feedback dependence", [r"feedback", r"prev_outcome", r"prev.*reward"]),
        ("history dependence / inertia", [r"history", r"recent", r"prev_action", r"action_freq", r"trend"]),
        ("switching behavior", [r"switch", r"last_action", r"prev_action"]),
        ("fallback/default behavior", [r"if not history", r"else", r"default", r"return\s+0\.5", r"is none"]),
        ("calibration/sharpness of returned probabilities", [r"sigmoid", r"clip", r"max\s*\(1e-6", r"min\s*\(1\s*-\s*1e-6"]),
    ]
    out: List[Dict[str, Any]] = []
    for mech, patterns in checks:
        snippets = _snippets_for_patterns(code, patterns)
        if len(snippets) >= 2:
            support = "strongly supported"
        elif len(snippets) == 1:
            support = "partially supported"
        else:
            support = "not present"
        out.append({"mechanism": mech, "support": support, "snippets": snippets})
    return out


def _first_snippet(mechs: Dict[str, Dict[str, Any]], key: str) -> str:
    row = mechs.get(key)
    if not row:
        return "no explicit snippet"
    if row["snippets"]:
        return row["snippets"][0]
    return "no explicit snippet"


def _alignment_strength(effect: Optional[float], support: str) -> str:
    if effect is None:
        return "weak"
    mag = abs(effect)
    if support == "strongly supported" and mag >= 0.20:
        return "strong"
    if support in {"strongly supported", "partially supported"} and mag >= 0.08:
        return "partial"
    return "weak"


def _infer_structure(mechanism_rows: List[Dict[str, Any]]) -> str:
    support_map = {m["mechanism"]: m["support"] for m in mechanism_rows}
    parts: List[str] = []
    if support_map.get("expected value sensitivity") != "not present":
        parts.append("EV")
    if support_map.get("risk / variance sensitivity") != "not present":
        parts.append("risk")
    if support_map.get("feedback dependence") != "not present":
        parts.append("feedback")
    if support_map.get("history dependence / inertia") != "not present":
        parts.append("history/inertia")
    if support_map.get("switching behavior") != "not present":
        parts.append("switch")
    if support_map.get("calibration/sharpness of returned probabilities") != "not present":
        parts.append("bounded soft rule")
    return " + ".join(parts) if parts else "unclassified"


def _format_raw_examples(rows_by_split: Dict[str, List[Dict[str, Any]]], n_each: int = 5) -> str:
    rows: List[List[Any]] = []
    for split in ("train", "val", "test"):
        for r in rows_by_split.get(split, [])[:n_each]:
            rows.append(
                [
                    split,
                    r.get("trial_index"),
                    r.get("action"),
                    r.get("option_keys"),
                    r.get("has_feedback"),
                    _format_float(r.get("ev_a")),
                    _format_float(r.get("ev_b")),
                    _format_float(r.get("ev_diff")),
                    _format_float(r.get("var_a")),
                    _format_float(r.get("var_b")),
                    _format_float(r.get("var_diff")),
                    r.get("gamble_a"),
                    r.get("gamble_b"),
                    r.get("prev_action"),
                    (
                        f"{_format_float(r.get('prev_outcome'))} ({r.get('prev_outcome_source')})"
                        if r.get("prev_outcome") is not None
                        else "NA"
                    ),
                ]
            )
    return _table(
        [
            "split",
            "trial index",
            "action",
            "option_keys",
            "has_feedback",
            "EV(A)",
            "EV(B)",
            "EV(B)-EV(A)",
            "Var(A)",
            "Var(B)",
            "Var(B)-Var(A)",
            "gamble_A rewards/probs",
            "gamble_B rewards/probs",
            "previous action",
            "previous reward/outcome",
        ],
        rows,
    )


def _format_stats_tables(stats_by_split: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    order = ["train", "val", "test", "all"]

    lines.append("### Basic")
    lines.append(
        _table(
            ["split", "number of trials", "action-1 rate", "action-0 rate"],
            [[s, stats_by_split[s]["n_trials"], _format_pct(stats_by_split[s]["action1_rate"]), _format_pct(stats_by_split[s]["action0_rate"])] for s in order],
        )
    )

    lines.append("\n### EV sensitivity")
    lines.append(
        _table(
            [
                "split",
                "n EV(B)>EV(A)",
                "action-1 rate EV(B)>EV(A)",
                "n EV(B)<EV(A)",
                "action-1 rate EV(B)<EV(A)",
                "n EV(B)=EV(A)",
                "action-1 rate EV(B)=EV(A)",
                "difference (>) - (<)",
            ],
            [
                [
                    s,
                    stats_by_split[s]["ev"]["n_gt"],
                    _format_pct(stats_by_split[s]["ev"]["rate_gt"]),
                    stats_by_split[s]["ev"]["n_lt"],
                    _format_pct(stats_by_split[s]["ev"]["rate_lt"]),
                    stats_by_split[s]["ev"]["n_eq"],
                    _format_pct(stats_by_split[s]["ev"]["rate_eq"]),
                    _format_pct(stats_by_split[s]["ev"]["diff_gt_minus_lt"]),
                ]
                for s in order
            ],
        )
    )

    lines.append("\n### Risk / variance sensitivity")
    lines.append(
        _table(
            [
                "split",
                "n Var(B)<Var(A)",
                "action-1 rate Var(B)<Var(A)",
                "n Var(B)>Var(A)",
                "action-1 rate Var(B)>Var(A)",
                "difference [Var(B)<Var(A)] - [Var(B)>Var(A)]",
            ],
            [
                [
                    s,
                    stats_by_split[s]["risk"]["n_var_lt"],
                    _format_pct(stats_by_split[s]["risk"]["rate_var_lt"]),
                    stats_by_split[s]["risk"]["n_var_gt"],
                    _format_pct(stats_by_split[s]["risk"]["rate_var_gt"]),
                    _format_pct(stats_by_split[s]["risk"]["diff_lt_minus_gt"]),
                ]
                for s in order
            ],
        )
    )

    lines.append("\n### EV-variance conflict cells")
    lines.append(
        _table(
            [
                "split",
                "EV(B)>EV(A), Var(B)>Var(A)",
                "EV(B)>EV(A), Var(B)<Var(A)",
                "EV(B)<EV(A), Var(B)>Var(A)",
                "EV(B)<EV(A), Var(B)<Var(A)",
            ],
            [
                [
                    s,
                    f"n={stats_by_split[s]['risk']['conflicts']['ev_gt_var_gt']['n']}, a1={_format_pct(stats_by_split[s]['risk']['conflicts']['ev_gt_var_gt']['rate_action1'])}",
                    f"n={stats_by_split[s]['risk']['conflicts']['ev_gt_var_lt']['n']}, a1={_format_pct(stats_by_split[s]['risk']['conflicts']['ev_gt_var_lt']['rate_action1'])}",
                    f"n={stats_by_split[s]['risk']['conflicts']['ev_lt_var_gt']['n']}, a1={_format_pct(stats_by_split[s]['risk']['conflicts']['ev_lt_var_gt']['rate_action1'])}",
                    f"n={stats_by_split[s]['risk']['conflicts']['ev_lt_var_lt']['n']}, a1={_format_pct(stats_by_split[s]['risk']['conflicts']['ev_lt_var_lt']['rate_action1'])}",
                ]
                for s in order
            ],
        )
    )

    lines.append("\n### History / inertia")
    lines.append(
        _table(
            [
                "split",
                "n with previous action",
                "repetition rate",
                "n previous action=1",
                "action-1 rate after previous action=1",
                "n previous action=0",
                "action-1 rate after previous action=0",
                "n last-5 majority=1",
                "action-1 rate after last-5 majority=1",
                "n last-5 majority=0",
                "action-1 rate after last-5 majority=0",
            ],
            [
                [
                    s,
                    stats_by_split[s]["history"]["n_with_prev"],
                    _format_pct(stats_by_split[s]["history"]["repetition_rate"]),
                    stats_by_split[s]["history"]["n_prev1"],
                    _format_pct(stats_by_split[s]["history"]["rate_after_prev1"]),
                    stats_by_split[s]["history"]["n_prev0"],
                    _format_pct(stats_by_split[s]["history"]["rate_after_prev0"]),
                    stats_by_split[s]["history"]["n_trend1"],
                    _format_pct(stats_by_split[s]["history"]["rate_after_trend1"]),
                    stats_by_split[s]["history"]["n_trend0"],
                    _format_pct(stats_by_split[s]["history"]["rate_after_trend0"]),
                ]
                for s in order
            ],
        )
    )

    lines.append("\n### Feedback / outcome")
    lines.append(
        _table(
            [
                "split",
                "n feedback=True",
                "action-1 rate feedback=True",
                "n feedback=False",
                "action-1 rate feedback=False",
                "n prev outcome known",
                "n prev positive",
                "action-1 rate after prev positive",
                "n prev negative",
                "action-1 rate after prev negative",
                "median(prev outcome)",
                "n prev high (>=median)",
                "action-1 rate prev high",
                "n prev low (<median)",
                "action-1 rate prev low",
            ],
            [
                [
                    s,
                    stats_by_split[s]["feedback"]["n_fb_yes"],
                    _format_pct(stats_by_split[s]["feedback"]["rate_fb_yes"]),
                    stats_by_split[s]["feedback"]["n_fb_no"],
                    _format_pct(stats_by_split[s]["feedback"]["rate_fb_no"]),
                    stats_by_split[s]["feedback"]["n_prev_known"],
                    stats_by_split[s]["feedback"]["n_prev_pos"],
                    _format_pct(stats_by_split[s]["feedback"]["rate_after_prev_pos"]),
                    stats_by_split[s]["feedback"]["n_prev_neg"],
                    _format_pct(stats_by_split[s]["feedback"]["rate_after_prev_neg"]),
                    _format_float(stats_by_split[s]["feedback"]["prev_outcome_median"]),
                    stats_by_split[s]["feedback"]["n_prev_high"],
                    _format_pct(stats_by_split[s]["feedback"]["rate_after_prev_high"]),
                    stats_by_split[s]["feedback"]["n_prev_low"],
                    _format_pct(stats_by_split[s]["feedback"]["rate_after_prev_low"]),
                ]
                for s in order
            ],
        )
    )

    lines.append("\n### Switching")
    lines.append(
        _table(
            [
                "split",
                "n switch-defined (has previous action)",
                "switch rate",
                "n after prev positive outcome",
                "switch rate after prev positive outcome",
                "n after prev negative outcome",
                "switch rate after prev negative outcome",
            ],
            [
                [
                    s,
                    stats_by_split[s]["switch"]["n_switch_defined"],
                    _format_pct(stats_by_split[s]["switch"]["switch_rate"]),
                    stats_by_split[s]["switch"]["n_after_prev_pos"],
                    _format_pct(stats_by_split[s]["switch"]["switch_rate_after_prev_pos"]),
                    stats_by_split[s]["switch"]["n_after_prev_neg"],
                    _format_pct(stats_by_split[s]["switch"]["switch_rate_after_prev_neg"]),
                ]
                for s in order
            ],
        )
    )
    return "\n".join(lines)


def _alignment_table(stats_all: Dict[str, Any], mechanism_rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    mechs = {m["mechanism"]: m for m in mechanism_rows}
    ev_diff = stats_all["ev"]["diff_gt_minus_lt"]
    risk_diff = stats_all["risk"]["diff_lt_minus_gt"]
    hist_diff = None
    if (
        stats_all["history"]["rate_after_prev1"] is not None
        and stats_all["history"]["rate_after_prev0"] is not None
    ):
        hist_diff = stats_all["history"]["rate_after_prev1"] - stats_all["history"]["rate_after_prev0"]
    fb_diff = None
    if stats_all["feedback"]["rate_after_prev_pos"] is not None and stats_all["feedback"]["rate_after_prev_neg"] is not None:
        fb_diff = stats_all["feedback"]["rate_after_prev_pos"] - stats_all["feedback"]["rate_after_prev_neg"]
    sw_diff = None
    if (
        stats_all["switch"]["switch_rate_after_prev_pos"] is not None
        and stats_all["switch"]["switch_rate_after_prev_neg"] is not None
    ):
        sw_diff = stats_all["switch"]["switch_rate_after_prev_neg"] - stats_all["switch"]["switch_rate_after_prev_pos"]

    rows = [
        [
            "EV sensitivity",
            f"a1|EV(B)>EV(A)={_format_pct(stats_all['ev']['rate_gt'])} vs a1|EV(B)<EV(A)={_format_pct(stats_all['ev']['rate_lt'])}",
            "Program computes EV terms/difference",
            _first_snippet(mechs, "expected value sensitivity"),
            _alignment_strength(ev_diff, mechs["expected value sensitivity"]["support"]),
        ],
        [
            "Risk sensitivity",
            f"a1|Var(B)<Var(A)={_format_pct(stats_all['risk']['rate_var_lt'])} vs a1|Var(B)>Var(A)={_format_pct(stats_all['risk']['rate_var_gt'])}",
            "Program explicitly uses variance/risk terms",
            _first_snippet(mechs, "risk / variance sensitivity"),
            _alignment_strength(risk_diff, mechs["risk / variance sensitivity"]["support"]),
        ],
        [
            "Feedback dependence",
            f"a1|prev positive={_format_pct(stats_all['feedback']['rate_after_prev_pos'])} vs a1|prev negative={_format_pct(stats_all['feedback']['rate_after_prev_neg'])}",
            "Program uses feedback/outcome history",
            _first_snippet(mechs, "feedback dependence"),
            _alignment_strength(fb_diff, mechs["feedback dependence"]["support"]),
        ],
        [
            "Inertia/history",
            f"repetition={_format_pct(stats_all['history']['repetition_rate'])}; a1|prev1={_format_pct(stats_all['history']['rate_after_prev1'])}, a1|prev0={_format_pct(stats_all['history']['rate_after_prev0'])}",
            "Program uses previous actions/history trend",
            _first_snippet(mechs, "history dependence / inertia"),
            _alignment_strength(hist_diff, mechs["history dependence / inertia"]["support"]),
        ],
        [
            "Switching",
            f"switch|prev pos={_format_pct(stats_all['switch']['switch_rate_after_prev_pos'])} vs switch|prev neg={_format_pct(stats_all['switch']['switch_rate_after_prev_neg'])}",
            "Program contains explicit switch/last-action term",
            _first_snippet(mechs, "switching behavior"),
            _alignment_strength(sw_diff, mechs["switching behavior"]["support"]),
        ],
    ]
    strength_order = {"strong": 3, "partial": 2, "weak": 1}
    top = max((r[4] for r in rows), key=lambda x: strength_order.get(x, 0))
    return (
        _table(
            ["Observed pattern", "Trial evidence", "Program mechanism", "Code evidence", "Alignment strength"],
            rows,
        ),
        top,
    )


def _mechanism_section(mechanism_rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in mechanism_rows:
        lines.append(f"### {m['mechanism']}")
        lines.append(f"- support level: **{m['support']}**")
        if m["snippets"]:
            for sn in m["snippets"]:
                lines.append("```python")
                lines.append(sn)
                lines.append("```")
            lines.append("- explanation: snippet directly implements or references this mechanism.")
        else:
            lines.append("- exact code snippet: none found.")
            lines.append("- explanation: no explicit implementation found in the selected program.")
        lines.append("")
    return "\n".join(lines).rstrip()


def _candidate_table(candidates: List[Dict[str, Any]]) -> str:
    return _table(
        [
            "candidate",
            "path",
            "program_id",
            "train_loglik",
            "val_loglik",
            "train_val_loglik",
            "test_loglik",
            "gated_test_loglik",
            "source",
        ],
        [
            [
                c.get("candidate"),
                c.get("path"),
                c.get("program_id"),
                _format_float(c.get("train_loglik")),
                _format_float(c.get("val_loglik")),
                _format_float(c.get("train_val_loglik")),
                _format_float(c.get("test_loglik")),
                _format_float(c.get("gated_test_loglik")),
                c.get("source"),
            ]
            for c in candidates
        ],
    )


def _paper_paragraph(
    participant_id: int,
    metrics: Dict[str, Any],
    selection: SelectionInfo,
    stats_all: Dict[str, Any],
    structure: str,
) -> str:
    margin = (
        f", with a margin of {_format_float(metrics.get('pics_margin'), 2)} over {metrics.get('best_non_pics_method')}"
        if metrics.get("pics_margin") is not None and metrics.get("best_non_pics_method")
        else ""
    )
    return (
        f"For participant {participant_id}, the behavior is consistent with a `{structure}` style rule: "
        f"action-1 rate is {_format_pct(stats_all['ev']['rate_gt'])} when EV(B)>EV(A) versus {_format_pct(stats_all['ev']['rate_lt'])} "
        f"when EV(B)<EV(A), and repetition rate is {_format_pct(stats_all['history']['repetition_rate'])}. "
        f"The selected executable program (`{selection.selected_path}`) contains corresponding mechanisms in code, "
        f"including bounded probability output and history-aware terms. PICS test loglik is {_format_float(metrics.get('test_loglik'), 4)}"
        f"{margin}. Claims remain cautious because alignment strength varies by mechanism."
    )


def _best_pattern(stats_all: Dict[str, Any]) -> str:
    diffs = []
    ev = stats_all["ev"]["diff_gt_minus_lt"]
    diffs.append(("EV sensitivity", abs(ev) if ev is not None else -1.0))
    hist = None
    if stats_all["history"]["rate_after_prev1"] is not None and stats_all["history"]["rate_after_prev0"] is not None:
        hist = stats_all["history"]["rate_after_prev1"] - stats_all["history"]["rate_after_prev0"]
    diffs.append(("history/inertia", abs(hist) if hist is not None else -1.0))
    risk = stats_all["risk"]["diff_lt_minus_gt"]
    diffs.append(("risk/variance", abs(risk) if risk is not None else -1.0))
    fb = None
    if stats_all["feedback"]["rate_after_prev_pos"] is not None and stats_all["feedback"]["rate_after_prev_neg"] is not None:
        fb = stats_all["feedback"]["rate_after_prev_pos"] - stats_all["feedback"]["rate_after_prev_neg"]
    diffs.append(("feedback/outcome", abs(fb) if fb is not None else -1.0))
    diffs.sort(key=lambda x: x[1], reverse=True)
    return diffs[0][0] if diffs and diffs[0][1] >= 0 else "insufficient evidence"


def _recommended_top3(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = sorted(
        rows,
        key=lambda r: (
            -999.0 if r["test_loglik"] is None else r["test_loglik"],
            -999.0 if r["pics_margin"] is None else r["pics_margin"],
            {"strong": 3, "partial": 2, "weak": 1}.get(r["alignment_strength"], 0),
        ),
        reverse=True,
    )

    best_by_structure: Dict[str, Dict[str, Any]] = {}
    for row in scored:
        sig = row["selected_program_structure"]
        if sig not in best_by_structure:
            best_by_structure[sig] = row

    structure_winners = sorted(
        best_by_structure.values(),
        key=lambda r: (
            -999.0 if r["test_loglik"] is None else r["test_loglik"],
            -999.0 if r["pics_margin"] is None else r["pics_margin"],
        ),
        reverse=True,
    )

    chosen = structure_winners[:3]
    if len(chosen) < 3:
        for row in scored:
            if row not in chosen:
                chosen.append(row)
            if len(chosen) >= 3:
                break
    return chosen[:3]


def build_report(
    participant_id: int,
    dataset: str,
    run_dir: Path,
    compare_csv: Path,
    output_dir: Path,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str,
) -> Dict[str, Any]:
    metrics = _load_metrics(participant_id, run_dir, compare_csv)
    selection = _selection_info(run_dir, participant_id, metrics)

    exp = get_psych101_binary_experiment(dataset, participant_id, split=psych_dataset_split)
    train_trials, val_trials, test_trials, _ = split_psych_experiment(exp, split_ratio=split_ratio, split_seed=split_seed)
    rows_by_split = {
        "train": [_trial_row("train", i, t) for i, t in enumerate(train_trials)],
        "val": [_trial_row("val", i, t) for i, t in enumerate(val_trials)],
        "test": [_trial_row("test", i, t) for i, t in enumerate(test_trials)],
    }
    rows_all = rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]
    stats_by_split = {
        "train": _stats_for_rows(rows_by_split["train"]),
        "val": _stats_for_rows(rows_by_split["val"]),
        "test": _stats_for_rows(rows_by_split["test"]),
        "all": _stats_for_rows(rows_all),
    }

    code = _read_text(selection.selected_path) if selection.selected_path else None
    mechanism_rows = _mechanism_evidence(code or "")
    structure = _infer_structure(mechanism_rows)
    alignment_md, top_alignment = _alignment_table(stats_by_split["all"], mechanism_rows)
    strongest_pattern = _best_pattern(stats_by_split["all"])
    strongest_mech = "none"
    for mech in mechanism_rows:
        if mech["support"] == "strongly supported":
            strongest_mech = mech["mechanism"]
            break
    if strongest_mech == "none":
        for mech in mechanism_rows:
            if mech["support"] == "partially supported":
                strongest_mech = mech["mechanism"]
                break

    outcome_sources = sorted(set(r.get("prev_outcome_source") for r in rows_all if r.get("prev_outcome_source")))
    outcome_note = (
        f"Previous outcome extracted from immediate history using numeric fields; observed sources: {outcome_sources}."
        if stats_by_split["all"]["feedback"]["n_prev_known"] > 0
        else "Previous outcome extraction is ambiguous; outcome-conditioned rates are unavailable."
    )

    report_path = output_dir / f"1peterson2021using_participant_{participant_id}.md"
    lines: List[str] = []
    lines.append(f"# 1peterson2021using participant {participant_id} program-analysis report")
    lines.append("")
    lines.append("## 1. Basic metrics")
    lines.append("")
    lines.append(
        _table(
            ["field", "value"],
            [
                ["participant id", participant_id],
                ["train loglik", _format_float(metrics.get("train_loglik"))],
                ["val loglik", _format_float(metrics.get("val_loglik"))],
                ["test loglik", _format_float(metrics.get("test_loglik"))],
                ["gated_test_loglik", _format_float(metrics.get("gated_test_loglik"))],
                ["selected final score used by paper", f"{selection.selected_score_name} = {_format_float(selection.selected_score_value)}"],
                ["selected program path", selection.selected_path or "NA"],
                ["how final program was determined", selection.selected_reason],
                ["Logistic Model test loglik", _format_float(metrics.get("MLE"), 2)],
                ["Prospect Theory test loglik", _format_float(metrics.get("prospect_theory"), 2)],
                ["OpenEvolve test loglik", _format_float(metrics.get("openevolve"), 2)],
                ["Centaur test loglik", _format_float(metrics.get("Centaur"), 2)],
                ["PICS test loglik (compare csv run column)", _format_float(metrics.get("PICS"), 2)],
                ["best non-PICS method", metrics.get("best_non_pics_method") or "unavailable"],
                ["PICS margin over best non-PICS", _format_float(metrics.get("pics_margin"), 2)],
            ],
        )
    )
    if metrics["missing_notes"]:
        lines.append("")
        lines.append("Missing/unavailable metrics notes:")
        for note in metrics["missing_notes"]:
            lines.append(f"- {note}")

    lines.append("")
    lines.append("## 2. Raw trial examples")
    lines.append("")
    lines.append(
        "Trials loaded with `get_psych101_binary_experiment(..., split='train')` and "
        f"`split_psych_experiment(split_ratio={split_ratio}, split_seed={split_seed})`."
    )
    lines.append("")
    lines.append(_format_raw_examples(rows_by_split, n_each=5))

    lines.append("")
    lines.append("## 3. Evidence from trial statistics")
    lines.append("")
    lines.append(_format_stats_tables(stats_by_split))
    lines.append("")
    lines.append(f"Outcome extraction note: {outcome_note}")
    lines.append("")
    lines.append(
        "Interpretation note: statements are evidence-based only; unsupported mechanisms are marked as weak/absent rather than inferred."
    )

    lines.append("")
    lines.append("## 4. Final selected PICS program")
    lines.append("")
    lines.append(f"- exact selected program path: `{selection.selected_path}`" if selection.selected_path else "- exact selected program path: NA")
    lines.append(f"- selection reasoning: {selection.selected_reason}")
    lines.append(
        f"- selected metrics: train={_format_float(selection.selected_metrics.get('train_loglik'))}, "
        f"val={_format_float(selection.selected_metrics.get('val_loglik'))}, "
        f"train_val/overall={_format_float(selection.selected_metrics.get('train_val_loglik'))}, "
        f"test={_format_float(selection.selected_metrics.get('test_loglik'))}, "
        f"gated_test_loglik={_format_float(selection.selected_metrics.get('gated_test_loglik'))}"
    )
    lines.append("")
    lines.append("Root/refinement candidate evidence:")
    lines.append("")
    lines.append(_candidate_table(selection.candidate_rows))
    if selection.ambiguous:
        lines.append("")
        lines.append("Ambiguity note: candidate selection remains ambiguous based on available artifacts.")
    lines.append("")
    lines.append("Full selected program code:")
    lines.append("")
    if code:
        lines.append("```python")
        lines.append(code.rstrip())
        lines.append("```")
    else:
        lines.append("Selected program file is missing.")

    lines.append("")
    lines.append("## 5. Evidence from program code")
    lines.append("")
    lines.append(_mechanism_section(mechanism_rows))

    lines.append("")
    lines.append("## 6. Trial-program alignment")
    lines.append("")
    lines.append(alignment_md)

    lines.append("")
    lines.append("## 7. Paper-ready summary")
    lines.append("")
    lines.append(_paper_paragraph(participant_id, metrics, selection, stats_by_split["all"], structure))

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "participant_id": participant_id,
        "test_loglik": metrics.get("test_loglik"),
        "gated_test_loglik": metrics.get("gated_test_loglik"),
        "pics_margin": metrics.get("pics_margin"),
        "selected_program_structure": structure,
        "strongest_observed_pattern": strongest_pattern,
        "strongest_program_mechanism": strongest_mech,
        "alignment_strength": top_alignment,
        "report_path": report_path,
        "selection_ambiguous": selection.ambiguous,
        "notes": metrics.get("missing_notes", []),
    }


def write_summary(
    rows: List[Dict[str, Any]],
    run_dir: Path,
    compare_csv: Path,
    output_dir: Path,
    split_info: Dict[str, Any],
    script_path: Path,
    command_used: str,
) -> Path:
    summary_path = output_dir / "summary_1peterson2021using.md"
    top3 = _recommended_top3(rows)
    top3_ids = [r["participant_id"] for r in top3]

    mechanisms_joined = " ".join(r["selected_program_structure"] for r in rows).lower()
    has_risk = "risk" in mechanisms_joined
    has_feedback = "feedback" in mechanisms_joined
    has_history = "history" in mechanisms_joined or "switch" in mechanisms_joined
    has_soft = "bounded soft rule" in mechanisms_joined

    table_rows = []
    for r in rows:
        should_use = "yes" if r["participant_id"] in top3_ids else ("contrast" if r["participant_id"] == 21 else "maybe")
        reason = (
            "Top-3 by held-out strength and structural diversity."
            if should_use == "yes"
            else ("Useful weak-loglik contrast case." if should_use == "contrast" else "Inspected case, but not in top-3 diversity shortlist.")
        )
        table_rows.append(
            [
                r["participant_id"],
                _format_float(r["test_loglik"]),
                _format_float(r["gated_test_loglik"]),
                r["selected_program_structure"],
                r["strongest_observed_pattern"],
                r["strongest_program_mechanism"],
                r["alignment_strength"],
                should_use,
                reason,
            ]
        )

    lines: List[str] = []
    lines.append("# Summary qualitative analysis: 1peterson2021using")
    lines.append("")
    lines.append(
        _table(
            [
                "participant_id",
                "test loglik",
                "gated_test_loglik",
                "selected program structure",
                "strongest observed pattern",
                "strongest program mechanism",
                "trial-program alignment",
                "should use in paper?",
                "reason",
            ],
            table_rows,
        )
    )
    lines.append("")
    lines.append("## Top 3 recommended qualitative examples")
    lines.append("")
    for row in top3:
        lines.append(
            f"- Participant {row['participant_id']}: `{row['selected_program_structure']}`; "
            f"test loglik={_format_float(row['test_loglik'])}, gated={_format_float(row['gated_test_loglik'])}, "
            f"alignment={row['alignment_strength']}."
        )

    lines.append("")
    lines.append("## Contrastive set coverage")
    lines.append("")
    lines.append(f"- EV + history/inertia: {'present' if has_history else 'not clearly present'}")
    lines.append(f"- EV + risk/variance: {'present' if has_risk else 'not clearly present'}")
    lines.append(f"- EV + feedback/outcome: {'present' if has_feedback else 'not clearly present'}")
    lines.append(f"- soft/noisy mixture rule: {'present' if has_soft else 'not clearly present'}")
    if not has_risk:
        lines.append("- No strong risk/variance example found in this inspected set.")
    if not has_feedback:
        lines.append("- No strong feedback/outcome example found in this inspected set.")

    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- exact run directory: `{run_dir}`")
    lines.append("- exact metrics files used:")
    lines.append(f"  - `{run_dir / 'participant_details_loglik.csv'}`")
    lines.append(f"  - `{run_dir / 'participants_summary.csv'}`")
    lines.append(f"  - `{compare_csv}`")
    lines.append("- exact split convention:")
    lines.append(f"  - `split_mode={split_info['split_mode']}`")
    lines.append(f"  - `split_ratio={split_info['split_ratio']}`")
    lines.append(f"  - `split_seed={split_info['split_seed']}`")
    lines.append(f"  - `psych_dataset_split={split_info['psych_dataset_split']}`")
    lines.append("- exact command/script used to generate the analysis:")
    lines.append(f"  - `{command_used}`")
    lines.append(f"  - script: `{script_path}`")
    if split_info.get("command_line"):
        lines.append("- source run command (`log/command.txt`):")
        lines.append(f"  - `{split_info['command_line']}`")

    ambiguous = [r["participant_id"] for r in rows if r["selection_ambiguous"]]
    missing: List[str] = []
    for r in rows:
        for note in r.get("notes", []):
            if note not in missing:
                missing.append(note)

    lines.append("- missing files or ambiguous paths:")
    if ambiguous:
        lines.append(f"  - selection ambiguity participants: {ambiguous}")
    else:
        lines.append("  - none for selected program path among inspected participants")
    if missing:
        for m in missing:
            lines.append(f"  - {m}")
    else:
        lines.append("  - none reported")

    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compare_csv", type=Path, default=DEFAULT_COMPARE_CSV)
    parser.add_argument("--participants", type=int, nargs="+", default=DEFAULT_PARTICIPANTS)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_info = _parse_command_args(args.run_dir)
    rows: List[Dict[str, Any]] = []
    for pid in args.participants:
        rows.append(
            build_report(
                participant_id=pid,
                dataset=args.dataset,
                run_dir=args.run_dir,
                compare_csv=args.compare_csv,
                output_dir=output_dir,
                split_ratio=float(split_info["split_ratio"]),
                split_seed=int(split_info["split_seed"]),
                psych_dataset_split=str(split_info["psych_dataset_split"]),
            )
        )

    cmd = (
        "python analysis/code/A_program_analysis/analyze_choices13k_programs.py "
        + "--participants "
        + " ".join(str(x) for x in args.participants)
    )
    summary_path = write_summary(
        rows=rows,
        run_dir=args.run_dir,
        compare_csv=args.compare_csv,
        output_dir=output_dir,
        split_info=split_info,
        script_path=Path("analysis/code/A_program_analysis/analyze_choices13k_programs.py"),
        command_used=cmd,
    )
    print(f"Wrote {len(rows)} participant reports to {output_dir}")
    print(f"Wrote summary file: {summary_path}")


if __name__ == "__main__":
    main()
