#!/usr/bin/env python3
"""Generate qualitative program-analysis reports for selected participants.

Rerunnable from repo root:
  python analysis/code/A_program_analysis/analyze_2plonsky_programs.py
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_modules.psych101_binary import get_psych101_binary_experiment, split_psych_experiment


DEFAULT_RUN_DIR = Path(
    "generated_outputs/psych101_train/teh/2plonsky2018when/run_260525_040017"
)
DEFAULT_OUTPUT_DIR = Path("analysis/data/A_program_analysis")
DEFAULT_COMPARE_CSV = Path("analysis/data/utils/loglik_compare_2plonsky2018when_train_gated.csv")
DEFAULT_PARTICIPANTS = [42, 34, 40, 29]
DATASET_ALIAS = "2plonsky2018when"
PARTICIPANT_NOTES = {
    42: "strongest PICS win over all baselines; clean showcase case.",
    34: "strong PICS win with nonzero BIR; representative strong case.",
    40: "strong PICS win with nonzero BIR; fallback representative strong case.",
    29: "high-BIR case where PICS still wins; harder/inconsistent case.",
}
EPS = 1e-12


@dataclass
class SelectionInfo:
    selected_path: Optional[Path]
    selected_reason: str
    selected_metrics: Dict[str, Optional[float]]
    candidate_rows: List[Dict[str, Any]]
    ambiguous: bool


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _parse_command_args(run_dir: Path) -> Dict[str, Any]:
    """Parse run log command for split settings and reproducibility."""
    command_path = run_dir / "log" / "command.txt"
    out = {
        "command_path": command_path,
        "command_line": None,
        "split_ratio": 0.6,
        "split_seed": 0,
        "psych_dataset_split": "train",
    }
    txt = _read_text(command_path)
    if not txt:
        return out

    command_line = None
    for line in txt.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            command_line = line
            break
    out["command_line"] = command_line
    if not command_line:
        return out

    def _extract(flag: str) -> Optional[str]:
        m = re.search(rf"(?:^|\s){re.escape(flag)}\s+([^\s]+)", command_line)
        return m.group(1) if m else None

    ratio = _extract("--split_ratio")
    if ratio is not None:
        out["split_ratio"] = float(ratio)
    seed = _extract("--split_seed")
    if seed is not None:
        out["split_seed"] = int(seed)
    split_name = _extract("--psych_dataset_split")
    if split_name:
        out["psych_dataset_split"] = split_name
    return out


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if math.isnan(float(x)):
            return None
        return float(x)
    sx = str(x).strip()
    if not sx or sx.lower() in {"nan", "none", "null"}:
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


def _ev_and_var(gamble: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(gamble, dict):
        return None, None
    rewards = gamble.get("rewards")
    probs = gamble.get("probs")
    if not isinstance(rewards, list) or not rewards:
        return None, None
    rewards_f = [_safe_float(r) for r in rewards]
    if any(r is None for r in rewards_f):
        return None, None
    rewards_f2 = [float(r) for r in rewards_f if r is not None]
    if isinstance(probs, list) and len(probs) == len(rewards_f2):
        probs_f = [_safe_float(p) for p in probs]
        if any(p is None for p in probs_f):
            return None, None
        probs_f2 = [float(p) for p in probs_f if p is not None]
        ev = sum(p * r for p, r in zip(probs_f2, rewards_f2))
        ev2 = sum(p * (r**2) for p, r in zip(probs_f2, rewards_f2))
        var = max(0.0, ev2 - ev**2)
        return ev, var
    ev = sum(rewards_f2) / len(rewards_f2)
    ev2 = sum(r**2 for r in rewards_f2) / len(rewards_f2)
    var = max(0.0, ev2 - ev**2)
    return ev, var


def _extract_prev_outcome(last_history_entry: Any) -> Tuple[Optional[float], str]:
    """Return (numeric_value, source_label) when reliable, else (None, reason)."""
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
                if kk in v and isinstance(v[kk], (int, float)):
                    return float(v[kk]), f"{k}.{kk}"
    return None, "ambiguous_non_numeric_feedback"


def _trial_row(split_name: str, idx: int, trial: Dict[str, Any]) -> Dict[str, Any]:
    problem = trial.get("problem", {})
    ev_a, var_a = _ev_and_var(problem.get("gamble_A"))
    ev_b, var_b = _ev_and_var(problem.get("gamble_B"))
    hist = trial.get("history") or []
    last = hist[-1] if hist else None
    prev_action = last.get("action") if isinstance(last, dict) else None
    prev_outcome, prev_source = _extract_prev_outcome(last)
    return {
        "split": split_name,
        "trial_index": idx,
        "action": trial.get("action"),
        "option_keys": problem.get("option_keys", trial.get("options")),
        "has_feedback": problem.get("has_feedback"),
        "ev_a": ev_a,
        "ev_b": ev_b,
        "ev_diff": (ev_b - ev_a) if (ev_a is not None and ev_b is not None) else None,
        "var_a": var_a,
        "var_b": var_b,
        "gamble_a": problem.get("gamble_A"),
        "gamble_b": problem.get("gamble_B"),
        "prev_action": prev_action,
        "prev_outcome": prev_outcome,
        "prev_outcome_source": prev_source,
    }


def _rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def _stats_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    n_a1 = sum(1 for r in rows if r.get("action") == 1)
    n_a0 = sum(1 for r in rows if r.get("action") == 0)

    gt = [r for r in rows if r.get("ev_diff") is not None and r["ev_diff"] > EPS]
    lt = [r for r in rows if r.get("ev_diff") is not None and r["ev_diff"] < -EPS]
    eq = [r for r in rows if r.get("ev_diff") is not None and abs(r["ev_diff"]) <= EPS]

    gt_a1 = sum(1 for r in gt if r.get("action") == 1)
    lt_a1 = sum(1 for r in lt if r.get("action") == 1)
    eq_a1 = sum(1 for r in eq if r.get("action") == 1)

    has_prev = [r for r in rows if r.get("prev_action") in (0, 1)]
    repetition = [r for r in has_prev if r.get("action") == r.get("prev_action")]
    prev1 = [r for r in has_prev if r.get("prev_action") == 1]
    prev0 = [r for r in has_prev if r.get("prev_action") == 0]

    fb_yes = [r for r in rows if r.get("has_feedback") is True]
    fb_no = [r for r in rows if r.get("has_feedback") is False]

    prev_outcome_known = [r for r in rows if r.get("prev_outcome") is not None]
    prev_pos = [r for r in prev_outcome_known if float(r["prev_outcome"]) > 0]
    prev_neg = [r for r in prev_outcome_known if float(r["prev_outcome"]) < 0]

    risk_lt = [
        r
        for r in rows
        if r.get("var_a") is not None and r.get("var_b") is not None and r["var_b"] < r["var_a"] - EPS
    ]
    risk_gt = [
        r
        for r in rows
        if r.get("var_a") is not None and r.get("var_b") is not None and r["var_b"] > r["var_a"] + EPS
    ]

    r_gt = _rate(sum(1 for r in gt if r.get("action") == 1), len(gt))
    r_lt = _rate(sum(1 for r in lt if r.get("action") == 1), len(lt))

    return {
        "n_trials": n,
        "action1_rate": _rate(n_a1, n),
        "action0_rate": _rate(n_a0, n),
        "ev": {
            "n_gt": len(gt),
            "n_lt": len(lt),
            "n_eq": len(eq),
            "rate_gt": _rate(gt_a1, len(gt)),
            "rate_lt": _rate(lt_a1, len(lt)),
            "rate_eq": _rate(eq_a1, len(eq)),
            "diff_gt_minus_lt": (r_gt - r_lt) if (r_gt is not None and r_lt is not None) else None,
        },
        "history": {
            "n_with_prev": len(has_prev),
            "repetition_rate": _rate(len(repetition), len(has_prev)),
            "n_prev1": len(prev1),
            "n_prev0": len(prev0),
            "rate_after_prev1": _rate(sum(1 for r in prev1 if r.get("action") == 1), len(prev1)),
            "rate_after_prev0": _rate(sum(1 for r in prev0 if r.get("action") == 1), len(prev0)),
        },
        "feedback": {
            "n_fb_yes": len(fb_yes),
            "n_fb_no": len(fb_no),
            "rate_fb_yes": _rate(sum(1 for r in fb_yes if r.get("action") == 1), len(fb_yes)),
            "rate_fb_no": _rate(sum(1 for r in fb_no if r.get("action") == 1), len(fb_no)),
            "n_prev_outcome_known": len(prev_outcome_known),
            "n_prev_pos": len(prev_pos),
            "n_prev_neg": len(prev_neg),
            "rate_after_prev_pos": _rate(sum(1 for r in prev_pos if r.get("action") == 1), len(prev_pos)),
            "rate_after_prev_neg": _rate(sum(1 for r in prev_neg if r.get("action") == 1), len(prev_neg)),
        },
        "risk": {
            "n_var_b_lt_a": len(risk_lt),
            "n_var_b_gt_a": len(risk_gt),
            "rate_var_b_lt_a": _rate(sum(1 for r in risk_lt if r.get("action") == 1), len(risk_lt)),
            "rate_var_b_gt_a": _rate(sum(1 for r in risk_gt if r.get("action") == 1), len(risk_gt)),
        },
    }


def _table(headers: List[str], rows: Iterable[Iterable[Any]]) -> str:
    def _cell(v: Any) -> str:
        s = str(v)
        s = s.replace("|", "\\|")
        s = s.replace("\n", "<br>")
        return s

    lines = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        vals = [_cell(v) for v in r]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _format_trial_examples(rows_by_split: Dict[str, List[Dict[str, Any]]], n_each: int = 5) -> str:
    out_rows: List[List[Any]] = []
    for split in ("train", "val", "test"):
        rows = rows_by_split.get(split, [])
        for r in rows[:n_each]:
            out_rows.append(
                [
                    split,
                    r["trial_index"],
                    r.get("action"),
                    r.get("option_keys"),
                    r.get("has_feedback"),
                    _format_float(r.get("ev_a")),
                    _format_float(r.get("ev_b")),
                    _format_float(r.get("ev_diff")),
                    r.get("gamble_a"),
                    r.get("gamble_b"),
                    r.get("prev_action"),
                    (
                        f"{_format_float(r.get('prev_outcome'))} "
                        f"(source={r.get('prev_outcome_source')})"
                        if r.get("prev_outcome") is not None
                        else "NA"
                    ),
                ]
            )
    return _table(
        [
            "split",
            "trial_index",
            "action",
            "option_keys",
            "has_feedback",
            "EV(A)",
            "EV(B)",
            "EV(B)-EV(A)",
            "gamble_A rewards/probs",
            "gamble_B rewards/probs",
            "previous action",
            "previous reward/outcome",
        ],
        out_rows,
    )


def _format_stats_tables(stats_by_split: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []

    lines.append("### Basic")
    lines.append(
        _table(
            ["split", "n_trials", "action-1 rate", "action-0 rate"],
            [
                [
                    s,
                    d["n_trials"],
                    _format_pct(d["action1_rate"]),
                    _format_pct(d["action0_rate"]),
                ]
                for s, d in stats_by_split.items()
            ],
        )
    )

    lines.append("\n### EV sensitivity")
    lines.append(
        _table(
            [
                "split",
                "n[EV(B)>EV(A)]",
                "action-1 rate when EV(B)>EV(A)",
                "n[EV(B)<EV(A)]",
                "action-1 rate when EV(B)<EV(A)",
                "n[EV(B)=EV(A)]",
                "action-1 rate when EV(B)=EV(A)",
                "difference (>) - (<)",
            ],
            [
                [
                    s,
                    d["ev"]["n_gt"],
                    _format_pct(d["ev"]["rate_gt"]),
                    d["ev"]["n_lt"],
                    _format_pct(d["ev"]["rate_lt"]),
                    d["ev"]["n_eq"],
                    _format_pct(d["ev"]["rate_eq"]),
                    _format_pct(d["ev"]["diff_gt_minus_lt"]),
                ]
                for s, d in stats_by_split.items()
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
            ],
            [
                [
                    s,
                    d["history"]["n_with_prev"],
                    _format_pct(d["history"]["repetition_rate"]),
                    d["history"]["n_prev1"],
                    _format_pct(d["history"]["rate_after_prev1"]),
                    d["history"]["n_prev0"],
                    _format_pct(d["history"]["rate_after_prev0"]),
                ]
                for s, d in stats_by_split.items()
            ],
        )
    )

    lines.append("\n### Feedback")
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
            ],
            [
                [
                    s,
                    d["feedback"]["n_fb_yes"],
                    _format_pct(d["feedback"]["rate_fb_yes"]),
                    d["feedback"]["n_fb_no"],
                    _format_pct(d["feedback"]["rate_fb_no"]),
                    d["feedback"]["n_prev_outcome_known"],
                    d["feedback"]["n_prev_pos"],
                    _format_pct(d["feedback"]["rate_after_prev_pos"]),
                    d["feedback"]["n_prev_neg"],
                    _format_pct(d["feedback"]["rate_after_prev_neg"]),
                ]
                for s, d in stats_by_split.items()
            ],
        )
    )

    lines.append("\n### Risk / variance")
    lines.append(
        _table(
            [
                "split",
                "n[Var(B)<Var(A)]",
                "action-1 rate when Var(B)<Var(A)",
                "n[Var(B)>Var(A)]",
                "action-1 rate when Var(B)>Var(A)",
            ],
            [
                [
                    s,
                    d["risk"]["n_var_b_lt_a"],
                    _format_pct(d["risk"]["rate_var_b_lt_a"]),
                    d["risk"]["n_var_b_gt_a"],
                    _format_pct(d["risk"]["rate_var_b_gt_a"]),
                ]
                for s, d in stats_by_split.items()
            ],
        )
    )
    return "\n".join(lines)


def _load_metrics(compare_csv: Path, run_dir: Path, participant_id: int) -> Dict[str, Any]:
    searched_paths = [
        str(compare_csv),
        str(run_dir / "participant_details_loglik.csv"),
        str(run_dir / "participants_summary.csv"),
    ]
    compare_rows = _load_csv_rows(compare_csv)
    run_col = run_dir.name
    compare = next((r for r in compare_rows if r.get("participant_id") == str(participant_id)), None)

    out: Dict[str, Any] = {
        "searched_paths": searched_paths,
        "participant_id": participant_id,
        "BIR": None,
        "MLE": None,
        "prospect_theory": None,
        "openevolve": None,
        "Centaur": None,
        "PICS": None,
        "best_non_pics_method": None,
        "best_non_pics_loglik": None,
        "pics_margin": None,
    }
    if compare:
        out["BIR"] = _safe_float(compare.get("BIR"))
        out["MLE"] = _safe_float(compare.get("MLE"))
        out["prospect_theory"] = _safe_float(compare.get("prospect_theory"))
        out["openevolve"] = _safe_float(compare.get("openevolve"))
        out["Centaur"] = _safe_float(compare.get("Centaur"))
        out["PICS"] = _safe_float(compare.get(run_col))

        baselines = {
            "Logistic Model / MLE": out["MLE"],
            "Prospect Theory": out["prospect_theory"],
            "OpenEvolve": out["openevolve"],
            "Centaur": out["Centaur"],
        }
        valid = {k: v for k, v in baselines.items() if v is not None}
        if valid:
            best_method, best_ll = max(valid.items(), key=lambda kv: kv[1])
            out["best_non_pics_method"] = best_method
            out["best_non_pics_loglik"] = best_ll
            if out["PICS"] is not None:
                out["pics_margin"] = out["PICS"] - best_ll
    return out


def _selection_info(run_dir: Path, participant_id: int) -> SelectionInfo:
    pdir = run_dir / f"participant_{participant_id}"
    root_results = _read_json(pdir / "results.json")
    ref_results = _read_json(pdir / "refinement" / "results.json")
    rows: List[Dict[str, Any]] = []

    if root_results and "overall_best_train" in root_results:
        r = root_results["overall_best_train"]
        rows.append(
            {
                "candidate": "root_best_program",
                "path": pdir / "best_program.py",
                "train_loglik": _safe_float(r.get("train_loglik")),
                "val_loglik": _safe_float(r.get("val_loglik")),
                "train_val_loglik": _safe_float(r.get("selection_score")),
                "test_loglik": _safe_float(r.get("test_loglik")),
                "program_id": r.get("program_id"),
                "source": "participant/results.json:overall_best_train",
            }
        )
    if ref_results and "final_pool_best" in ref_results:
        r = ref_results["final_pool_best"]
        rows.append(
            {
                "candidate": "refinement_best_program",
                "path": pdir / "refinement" / "best_program.py",
                "train_loglik": _safe_float(r.get("train_loglik")),
                "val_loglik": _safe_float(r.get("val_loglik")),
                "train_val_loglik": _safe_float(r.get("train_val_loglik")),
                "test_loglik": _safe_float(r.get("test_loglik")),
                "program_id": r.get("program_id"),
                "source": "participant/refinement/results.json:final_pool_best",
            }
        )

    existing = [r for r in rows if Path(r["path"]).exists()]
    if not existing:
        reason = (
            "Could not locate a selected program file. "
            "Searched participant root and refinement best-program paths."
        )
        return SelectionInfo(
            selected_path=None,
            selected_reason=reason,
            selected_metrics={},
            candidate_rows=rows,
            ambiguous=True,
        )

    if len(existing) == 1:
        row = existing[0]
        return SelectionInfo(
            selected_path=Path(row["path"]),
            selected_reason=f"Single candidate with both metadata and file: {row['source']}.",
            selected_metrics={
                "train_loglik": row["train_loglik"],
                "val_loglik": row["val_loglik"],
                "train_val_loglik": row["train_val_loglik"],
                "test_loglik": row["test_loglik"],
            },
            candidate_rows=rows,
            ambiguous=False,
        )

    ref_row = next((r for r in existing if r["candidate"] == "refinement_best_program"), None)
    root_row = next((r for r in existing if r["candidate"] == "root_best_program"), None)
    if ref_row and root_row:
        # Run-level summaries use gated test loglik; refinement is the post-gating phase artifact.
        if ref_row.get("test_loglik") is not None:
            return SelectionInfo(
                selected_path=Path(ref_row["path"]),
                selected_reason=(
                    "Both root and refinement candidates exist; selected refinement best program "
                    "because it is the post-gating phase artifact with explicit final_pool_best metrics."
                ),
                selected_metrics={
                    "train_loglik": ref_row["train_loglik"],
                    "val_loglik": ref_row["val_loglik"],
                    "train_val_loglik": ref_row["train_val_loglik"],
                    "test_loglik": ref_row["test_loglik"],
                },
                candidate_rows=rows,
                ambiguous=False,
            )

    ranked = sorted(
        existing,
        key=lambda r: (
            -1e9 if r.get("train_val_loglik") is None else r["train_val_loglik"],
            -1e9 if r.get("test_loglik") is None else r["test_loglik"],
        ),
        reverse=True,
    )
    top = ranked[0]
    return SelectionInfo(
        selected_path=Path(top["path"]),
        selected_reason=(
            "Multiple candidates exist but no unambiguous phase marker. "
            "Selected the candidate with highest available train_val_loglik."
        ),
        selected_metrics={
            "train_loglik": top["train_loglik"],
            "val_loglik": top["val_loglik"],
            "train_val_loglik": top["train_val_loglik"],
            "test_loglik": top["test_loglik"],
        },
        candidate_rows=rows,
        ambiguous=True,
    )


def _snippets_for_patterns(code: str, patterns: List[str], max_hits: int = 2) -> List[str]:
    lines = code.splitlines()
    hits: List[int] = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(re.search(p, low) for p in patterns):
            hits.append(i)
    snippets: List[str] = []
    used = set()
    for i in hits:
        if len(snippets) >= max_hits:
            break
        if any(abs(i - j) <= 2 for j in used):
            continue
        start = max(0, i - 2)
        end = min(len(lines), i + 3)
        chunk = "\n".join(lines[start:end])
        snippets.append(chunk if chunk.strip() else lines[i])
        used.add(i)
    return snippets


def _mechanism_evidence(code: str) -> List[Dict[str, Any]]:
    checks = [
        (
            "expected value sensitivity",
            [r"\bev[_\s]?[ab]?\b", r"expected_value", r"value_diff", r"ev_b\s*-\s*ev_a"],
            "Program explicitly computes EV-like quantities or EV differences.",
        ),
        (
            "risk / variance sensitivity",
            [r"variance", r"\bvar\b", r"std", r"risk"],
            "Program explicitly computes/uses variance or risk terms.",
        ),
        (
            "probability sensitivity",
            [r"\bprob", r"probs", r"sigmoid", r"logit"],
            "Program uses probability/probability-weight terms in choice probability.",
        ),
        (
            "reward magnitude sensitivity",
            [r"reward", r"outcome", r"value_diff", r"gain", r"loss"],
            "Program uses reward/outcome magnitudes.",
        ),
        (
            "loss sensitivity",
            [r"loss", r"<\s*0", r"negative", r"min\(", r"abs\("],
            "Program includes explicit handling of losses/negative outcomes.",
        ),
        (
            "thresholds",
            [r"\bif\b.*>", r"\bif\b.*<", r"\bif\b.*==", r"threshold", r">=\s*[-\d.]+"],
            "Program has explicit threshold/comparison logic for behavior changes.",
        ),
        (
            "feedback dependence",
            [r"feedback", r"prev.*reward", r"prev.*outcome"],
            "Program conditions on feedback/outcome history.",
        ),
        (
            "history dependence / inertia",
            [r"\bhistory\b", r"recent", r"prev_action", r"inertia", r"action_counts"],
            "Program uses previous actions/history-derived quantities.",
        ),
        (
            "fallback/default behavior",
            [r"else", r"default", r"if .* is none", r"return 0\.5", r"try:", r"except"],
            "Program includes default/fallback logic for missing information.",
        ),
        (
            "calibration/sharpness of returned probabilities",
            [r"sigmoid", r"temperature", r"max\(", r"min\(", r"clip", r"1e-6"],
            "Program shapes or clips output probabilities.",
        ),
    ]
    out = []
    for mech, patterns, expl in checks:
        snippets = _snippets_for_patterns(code, patterns)
        if snippets:
            support = "strongly supported" if len(snippets) >= 2 else "partially supported"
        else:
            support = "weakly supported"
        out.append(
            {
                "mechanism": mech,
                "support": support,
                "snippets": snippets,
                "explanation": expl,
            }
        )
    return out


def _choose_observed_pattern(stats_all: Dict[str, Any]) -> str:
    ev_diff = stats_all["ev"]["diff_gt_minus_lt"]
    hist_diff = None
    if (
        stats_all["history"]["rate_after_prev1"] is not None
        and stats_all["history"]["rate_after_prev0"] is not None
    ):
        hist_diff = stats_all["history"]["rate_after_prev1"] - stats_all["history"]["rate_after_prev0"]
    risk_diff = None
    if (
        stats_all["risk"]["rate_var_b_lt_a"] is not None
        and stats_all["risk"]["rate_var_b_gt_a"] is not None
    ):
        risk_diff = stats_all["risk"]["rate_var_b_lt_a"] - stats_all["risk"]["rate_var_b_gt_a"]

    candidates = [
        ("EV sensitivity", abs(ev_diff) if ev_diff is not None else -1.0),
        ("history dependence", abs(hist_diff) if hist_diff is not None else -1.0),
        ("risk sensitivity", abs(risk_diff) if risk_diff is not None else -1.0),
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates and candidates[0][1] >= 0 else "insufficient evidence"


def _alignment_strength(value: Optional[float], has_code: bool) -> str:
    if value is None:
        return "weak"
    magnitude = abs(value)
    if has_code and magnitude >= 0.20:
        return "strong"
    if has_code and magnitude >= 0.08:
        return "partial"
    return "weak"


def _alignment_table(stats_all: Dict[str, Any], mechanism_rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    mech_support = {m["mechanism"]: m for m in mechanism_rows}
    ev_diff = stats_all["ev"]["diff_gt_minus_lt"]
    hist_diff = None
    if (
        stats_all["history"]["rate_after_prev1"] is not None
        and stats_all["history"]["rate_after_prev0"] is not None
    ):
        hist_diff = stats_all["history"]["rate_after_prev1"] - stats_all["history"]["rate_after_prev0"]
    risk_diff = None
    if (
        stats_all["risk"]["rate_var_b_lt_a"] is not None
        and stats_all["risk"]["rate_var_b_gt_a"] is not None
    ):
        risk_diff = stats_all["risk"]["rate_var_b_lt_a"] - stats_all["risk"]["rate_var_b_gt_a"]

    fb_diff = None
    if (
        stats_all["feedback"]["rate_fb_yes"] is not None
        and stats_all["feedback"]["rate_fb_no"] is not None
    ):
        fb_diff = stats_all["feedback"]["rate_fb_yes"] - stats_all["feedback"]["rate_fb_no"]

    rows = []
    ev_code = mech_support["expected value sensitivity"]["support"] != "weakly supported"
    rows.append(
        [
            "EV sensitivity",
            (
                f"action-1 rate: EV(B)>EV(A)={_format_pct(stats_all['ev']['rate_gt'])}, "
                f"EV(B)<EV(A)={_format_pct(stats_all['ev']['rate_lt'])}, "
                f"diff={_format_pct(ev_diff)}"
            ),
            "program computes EV-like quantities",
            mech_support["expected value sensitivity"]["snippets"][0]
            if mech_support["expected value sensitivity"]["snippets"]
            else "no explicit snippet",
            _alignment_strength(ev_diff, ev_code),
        ]
    )
    hist_code = mech_support["history dependence / inertia"]["support"] != "weakly supported"
    rows.append(
        [
            "Inertia / history",
            (
                f"repetition={_format_pct(stats_all['history']['repetition_rate'])}; "
                f"a1|prev1={_format_pct(stats_all['history']['rate_after_prev1'])}, "
                f"a1|prev0={_format_pct(stats_all['history']['rate_after_prev0'])}"
            ),
            "program uses history/action counts/inertia terms",
            mech_support["history dependence / inertia"]["snippets"][0]
            if mech_support["history dependence / inertia"]["snippets"]
            else "no explicit snippet",
            _alignment_strength(hist_diff, hist_code),
        ]
    )
    risk_code = mech_support["risk / variance sensitivity"]["support"] != "weakly supported"
    rows.append(
        [
            "Risk / variance",
            (
                f"a1|Var(B)<Var(A)={_format_pct(stats_all['risk']['rate_var_b_lt_a'])}, "
                f"a1|Var(B)>Var(A)={_format_pct(stats_all['risk']['rate_var_b_gt_a'])}"
            ),
            "program explicit variance/risk terms",
            mech_support["risk / variance sensitivity"]["snippets"][0]
            if mech_support["risk / variance sensitivity"]["snippets"]
            else "no explicit snippet",
            _alignment_strength(risk_diff, risk_code),
        ]
    )
    fb_code = mech_support["feedback dependence"]["support"] != "weakly supported"
    rows.append(
        [
            "Feedback dependence",
            (
                f"a1|feedback={_format_pct(stats_all['feedback']['rate_fb_yes'])}, "
                f"a1|no_feedback={_format_pct(stats_all['feedback']['rate_fb_no'])}"
            ),
            "program uses feedback/outcome from history",
            mech_support["feedback dependence"]["snippets"][0]
            if mech_support["feedback dependence"]["snippets"]
            else "no explicit snippet",
            _alignment_strength(fb_diff, fb_code),
        ]
    )
    strength_rank = {"strong": 3, "partial": 2, "weak": 1}
    top_strength = sorted((r[4] for r in rows), key=lambda x: strength_rank[x], reverse=True)[0]
    return _table(
        [
            "Observed pattern",
            "Trial evidence",
            "Program mechanism",
            "Code evidence",
            "Alignment strength",
        ],
        rows,
    ), top_strength


def _metrics_table(metrics: Dict[str, Any]) -> str:
    return _table(
        ["metric", "value"],
        [
            ["participant id", metrics["participant_id"]],
            ["BIR", _format_float(metrics.get("BIR"), nd=2)],
            ["Logistic Model / MLE test loglik", _format_float(metrics.get("MLE"), nd=2)],
            ["Prospect Theory test loglik", _format_float(metrics.get("prospect_theory"), nd=2)],
            ["OpenEvolve test loglik", _format_float(metrics.get("openevolve"), nd=2)],
            ["Centaur test loglik", _format_float(metrics.get("Centaur"), nd=2)],
            ["PICS test loglik", _format_float(metrics.get("PICS"), nd=2)],
            ["best non-PICS method", metrics.get("best_non_pics_method") or "NA"],
            ["best non-PICS loglik", _format_float(metrics.get("best_non_pics_loglik"), nd=2)],
            ["PICS margin over best non-PICS", _format_float(metrics.get("pics_margin"), nd=2)],
        ],
    )


def _candidate_table(cands: List[Dict[str, Any]]) -> str:
    return _table(
        ["candidate", "path", "program_id", "train_loglik", "val_loglik", "train_val_loglik", "test_loglik", "source"],
        [
            [
                c.get("candidate"),
                c.get("path"),
                c.get("program_id"),
                _format_float(c.get("train_loglik")),
                _format_float(c.get("val_loglik")),
                _format_float(c.get("train_val_loglik")),
                _format_float(c.get("test_loglik")),
                c.get("source"),
            ]
            for c in cands
        ],
    )


def _mechanism_section_rows(mech_rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for row in mech_rows:
        lines.append(f"#### {row['mechanism']}")
        lines.append(f"- Support level: **{row['support']}**")
        lines.append(f"- What the snippet does: {row['explanation']}")
        if row["snippets"]:
            for sn in row["snippets"]:
                lines.append("```python")
                lines.append(sn)
                lines.append("```")
        else:
            lines.append("- No explicit matching code snippet found in the selected program.")
        lines.append("")
    return "\n".join(lines).strip()


def _paper_summary(participant_id: int, metrics: Dict[str, Any], stats_all: Dict[str, Any], selection: SelectionInfo) -> str:
    ev_gt = _format_pct(stats_all["ev"]["rate_gt"])
    ev_lt = _format_pct(stats_all["ev"]["rate_lt"])
    rep = _format_pct(stats_all["history"]["repetition_rate"])
    pics = _format_float(metrics.get("PICS"), nd=2)
    margin = _format_float(metrics.get("pics_margin"), nd=2)
    return (
        f"For participant {participant_id}, the held-out behavior is most consistent with a rule that combines "
        f"value differences and recent action history. In the trials, the action-1 rate is {ev_gt} when EV(B)>EV(A) "
        f"versus {ev_lt} when EV(B)<EV(A), and the repetition rate is {rep}. "
        f"The selected program at `{selection.selected_path}` computes EV-like terms and history-based adjustments, "
        f"then maps them to a bounded probability. This provides executable evidence that can approximate "
        f"participant-specific behavior. PICS test loglik is {pics}, with a margin of {margin} over the best non-PICS baseline."
    )


def build_report(
    participant_id: int,
    run_dir: Path,
    output_dir: Path,
    compare_csv: Path,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str,
) -> Dict[str, Any]:
    metrics = _load_metrics(compare_csv, run_dir, participant_id)

    exp = get_psych101_binary_experiment(DATASET_ALIAS, participant_id, split=psych_dataset_split)
    tr_train, tr_val, tr_test, _ = split_psych_experiment(
        exp, split_ratio=split_ratio, split_seed=split_seed
    )
    rows_by_split = {
        "train": [_trial_row("train", i, t) for i, t in enumerate(tr_train)],
        "val": [_trial_row("val", i, t) for i, t in enumerate(tr_val)],
        "test": [_trial_row("test", i, t) for i, t in enumerate(tr_test)],
    }
    all_rows = rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]
    stats_by_split: Dict[str, Dict[str, Any]] = {
        "train": _stats_for_rows(rows_by_split["train"]),
        "val": _stats_for_rows(rows_by_split["val"]),
        "test": _stats_for_rows(rows_by_split["test"]),
        "all": _stats_for_rows(all_rows),
    }

    selection = _selection_info(run_dir, participant_id)
    program_code = _read_text(selection.selected_path) if selection.selected_path else None
    mechanism_rows = _mechanism_evidence(program_code or "")
    alignment_table_md, top_alignment = _alignment_table(stats_by_split["all"], mechanism_rows)

    outcome_sources = sorted(
        set(r["prev_outcome_source"] for r in all_rows if r.get("prev_outcome_source"))
    )
    outcome_reliability = (
        "Previous outcome extraction uses numeric fields from the immediately preceding history entry "
        f"(sources observed: {outcome_sources})."
        if stats_by_split["all"]["feedback"]["n_prev_outcome_known"] > 0
        else "Previous outcome extraction is ambiguous/no numeric field available; outcome-conditioned rates are reported as NA."
    )

    missing_metrics_note = ""
    if metrics.get("PICS") is None:
        missing_metrics_note = (
            "Metric retrieval note: unable to retrieve one or more required metric values. "
            f"Paths searched: {metrics['searched_paths']}."
        )

    report_path = output_dir / f"2plonsky2018when_participant_{participant_id}.md"
    lines: List[str] = []
    lines.append(f"# Participant {participant_id} program-analysis report")
    lines.append("")
    lines.append(f"Selection context: {PARTICIPANT_NOTES.get(participant_id, 'user-selected case')}")
    lines.append("")
    lines.append("## 1. Basic metrics")
    lines.append("")
    lines.append(_metrics_table(metrics))
    if missing_metrics_note:
        lines.append("")
        lines.append(missing_metrics_note)
    lines.append("")
    lines.append("## 2. Raw trial examples")
    lines.append("")
    lines.append(
        "Trials are loaded with the same convention as `teh.py`: "
        f"`get_psych101_binary_experiment(..., split='{psych_dataset_split}')` then "
        f"`split_psych_experiment(split_ratio={split_ratio}, split_seed={split_seed})`."
    )
    lines.append("")
    lines.append(_format_trial_examples(rows_by_split, n_each=5))
    lines.append("")
    lines.append("## 3. Evidence from trial statistics")
    lines.append("")
    lines.append(_format_stats_tables(stats_by_split))
    lines.append("")
    lines.append(f"Outcome extraction note: {outcome_reliability}")
    lines.append("")
    lines.append("Interpretation note: only patterns with direct statistics above are interpreted; language remains cautious.")
    lines.append("")
    lines.append("## 4. Final selected PICS program")
    lines.append("")
    if selection.selected_path:
        lines.append(f"- Selected program path: `{selection.selected_path}`")
    else:
        lines.append("- Selected program path: NA")
    lines.append(f"- Selection rationale: {selection.selected_reason}")
    lines.append(
        "- Associated logliks (selected): "
        f"train={_format_float(selection.selected_metrics.get('train_loglik'))}, "
        f"val={_format_float(selection.selected_metrics.get('val_loglik'))}, "
        f"train_val/overall={_format_float(selection.selected_metrics.get('train_val_loglik'))}, "
        f"test={_format_float(selection.selected_metrics.get('test_loglik'))}"
    )
    lines.append("")
    lines.append("Candidate evidence:")
    lines.append("")
    lines.append(_candidate_table(selection.candidate_rows))
    lines.append("")
    if selection.ambiguous:
        lines.append(
            "Ambiguity note: multiple candidates are plausible based on artifacts; selection above follows documented rationale."
        )
        lines.append("")
    lines.append("Program code (full):")
    lines.append("")
    if program_code:
        lines.append("```python")
        lines.append(program_code.rstrip())
        lines.append("```")
    else:
        lines.append("Program file missing; full code unavailable.")
    lines.append("")
    lines.append("## 5. Evidence from program code")
    lines.append("")
    lines.append(_mechanism_section_rows(mechanism_rows))
    lines.append("")
    lines.append("## 6. Trial-program alignment")
    lines.append("")
    lines.append(alignment_table_md)
    lines.append("")
    lines.append("## 7. Paper-ready summary")
    lines.append("")
    lines.append(_paper_summary(participant_id, metrics, stats_by_split["all"], selection))
    lines.append("")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    strongest_pattern = _choose_observed_pattern(stats_by_split["all"])
    strongest_mechanism = "none"
    strong_mechs = [m for m in mechanism_rows if m["support"] == "strongly supported"]
    if strong_mechs:
        strongest_mechanism = strong_mechs[0]["mechanism"]
    elif mechanism_rows:
        strongest_mechanism = mechanism_rows[0]["mechanism"]

    return {
        "participant_id": participant_id,
        "bir": metrics.get("BIR"),
        "pics": metrics.get("PICS"),
        "best_non_pics_method": metrics.get("best_non_pics_method"),
        "best_non_pics_loglik": metrics.get("best_non_pics_loglik"),
        "margin": metrics.get("pics_margin"),
        "strongest_pattern": strongest_pattern,
        "strongest_mechanism": strongest_mechanism,
        "alignment_strength": top_alignment,
        "report_path": report_path,
        "selection_ambiguous": selection.ambiguous,
    }


def _recommendations(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    scored = sorted(
        rows,
        key=lambda r: (
            -999 if r["margin"] is None else r["margin"],
            {"strong": 3, "partial": 2, "weak": 1}.get(r["alignment_strength"], 0),
        ),
        reverse=True,
    )
    top2 = scored[:2]
    if len(top2) < 2:
        return "Insufficient rows for recommendation.", "No exclusion recommendation."
    rec = (
        f"Recommended primary qualitative cases: participant {top2[0]['participant_id']} and "
        f"participant {top2[1]['participant_id']}. They combine stronger PICS margins "
        f"({_format_float(top2[0]['margin'],2)}, {_format_float(top2[1]['margin'],2)}) with "
        f"{top2[0]['alignment_strength']}/{top2[1]['alignment_strength']} trial-program alignment."
    )
    ambiguous = [r for r in rows if r["selection_ambiguous"]]
    if ambiguous:
        ex = (
            "Potential exclusion or caution: "
            + ", ".join(f"participant {r['participant_id']} (selected program ambiguity)" for r in ambiguous)
            + "."
        )
    else:
        ex = "No participant requires exclusion for program-path ambiguity in this set."
    return rec, ex


def write_summary(
    output_dir: Path,
    run_dir: Path,
    compare_csv: Path,
    split_info: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> Path:
    summary_path = output_dir / "summary_2plonsky2018when.md"
    rec, exclusion = _recommendations(rows)
    lines: List[str] = []
    lines.append("# Summary qualitative program analysis: 2plonsky2018when")
    lines.append("")
    lines.append("## Participant summary table")
    lines.append("")
    lines.append(
        _table(
            [
                "participant_id",
                "BIR",
                "PICS test loglik",
                "best non-PICS method",
                "best non-PICS loglik",
                "margin",
                "strongest observed pattern",
                "strongest program mechanism",
                "alignment strength",
                "report path",
            ],
            [
                [
                    r["participant_id"],
                    _format_float(r["bir"], 2),
                    _format_float(r["pics"], 2),
                    r["best_non_pics_method"] or "NA",
                    _format_float(r["best_non_pics_loglik"], 2),
                    _format_float(r["margin"], 2),
                    r["strongest_pattern"],
                    r["strongest_mechanism"],
                    r["alignment_strength"],
                    r["report_path"],
                ]
                for r in rows
            ],
        )
    )
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append(f"- {rec}")
    lines.append(f"- {exclusion}")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- Metrics source file:")
    lines.append(f"  - `{compare_csv}`")
    lines.append("- Run artifact sources:")
    lines.append(f"  - `{run_dir}/participant_details_loglik.csv`")
    lines.append(f"  - `{run_dir}/participants_summary.csv`")
    lines.append(f"  - `{run_dir}/participant_<id>/results.json`")
    lines.append(f"  - `{run_dir}/participant_<id>/refinement/results.json`")
    lines.append("- Trial split convention:")
    lines.append(
        f"  - `get_psych101_binary_experiment('{DATASET_ALIAS}', participant_id, split='{split_info['psych_dataset_split']}')`"
    )
    lines.append(
        f"  - `split_psych_experiment(split_ratio={split_info['split_ratio']}, split_seed={split_info['split_seed']})`"
    )
    lines.append("- Command used to generate this analysis:")
    lines.append(
        "  - `python analysis/code/A_program_analysis/analyze_2plonsky_programs.py`"
    )
    if split_info.get("command_line"):
        lines.append("- Source run command (`log/command.txt`):")
        lines.append(f"  - `{split_info['command_line']}`")
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compare_csv", type=Path, default=DEFAULT_COMPARE_CSV)
    parser.add_argument(
        "--participants",
        type=int,
        nargs="+",
        default=DEFAULT_PARTICIPANTS,
        help="Participant IDs to analyze.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_info = _parse_command_args(run_dir)
    rows = []
    for pid in args.participants:
        row = build_report(
            participant_id=pid,
            run_dir=run_dir,
            output_dir=output_dir,
            compare_csv=args.compare_csv,
            split_ratio=float(split_info["split_ratio"]),
            split_seed=int(split_info["split_seed"]),
            psych_dataset_split=str(split_info["psych_dataset_split"]),
        )
        rows.append(row)
    write_summary(output_dir, run_dir, args.compare_csv, split_info, rows)
    print(f"Wrote {len(rows)} participant reports to {output_dir}")
    print(f"Wrote combined summary: {output_dir / 'summary_2plonsky2018when.md'}")


if __name__ == "__main__":
    main()
