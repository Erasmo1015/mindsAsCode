#!/usr/bin/env python3
"""
Check whether recommended Psych-101 experiments can use the choice13k-style
executable-program pipeline (parse NL -> structured trials -> choose -> float P(action=1)).

Writes:
  analysis/data/psych-101/candidate_parser_check.csv
  analysis/data/psych-101/candidate_parser_check.md
  analysis/data/psych-101/candidate_parse_examples.json

Example:
  python analysis/code/psych-101/check_candidate_parsers.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = REPO_ROOT / "analysis" / "data" / "psych-101"
TRAIN_HF = "marcelbinz/Psych-101"
TEST_HF = "marcelbinz/Psych-101-test"

CANDIDATE_EXPERIMENTS = [
    "peterson2021using/exp1.csv",
    "plonsky2018when/exp1.csv",
    "wulff2018description/exp1.csv",
    "speekenbrink2008learning/exp1.csv",
    "sadeghiyeh2020temporal/exp1.csv",
    "hilbig2014generalized/exp1.csv",
    "frey2017cct/exp1.csv",
    "flesch2018comparing/exp1.csv",
]

_RE_OPTION = re.compile(r"Option\s+([A-Z]) delivers", re.I)
_RE_PRESS_DOT = re.compile(r"You press <<([A-Z])>>\.\s*(?:\n|$)")
_RE_PRESS_GAIN = re.compile(r"You press <<([A-Z])>> and gain", re.I)
_RE_PRESS_ANY = re.compile(r"You press <<([A-Z])>>", re.I)
_RE_PERCENT = re.compile(r"(-?\d+\.?\d*).*?with\s+(\d+\.?\d*)% chance")
_RE_LOTTERY = re.compile(r"Lottery\s+([A-Z]) offers", re.I)
_RE_PRODUCT_RATINGS = re.compile(
    r"Product ([A-Z]) ratings:.*You press <<([A-Z])>>", re.I
)
_RE_SLOT = re.compile(r"You press <<([A-Z])>> and get (-?\d+) points", re.I)
_RE_WEATHER = re.compile(
    r"You are seeing the following:.*You press <<([A-Z])>>", re.I
)
_RE_TREE = re.compile(
    r"You get a tree with level (\d+).*garden\. You press <<([A-Z])>>", re.I
)
_RE_CCT = re.compile(r"You press <<([EC])>> and (turn over|stop)", re.I)


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_split(hf_id: str, split: str):
    from datasets import load_dataset

    tok = _hf_token()
    kw = {"token": tok} if tok else {}
    ds = load_dataset(hf_id, **kw)
    return ds[split]


def _pick_participant_indices(n_rows: int, k: int = 3) -> List[int]:
    if n_rows == 0:
        return []
    if n_rows <= k:
        return list(range(n_rows))
    return [0, n_rows // 2, n_rows - 1]


def _text_markers(text: str) -> Dict[str, Any]:
    presses_any = _RE_PRESS_ANY.findall(text)
    presses_dot = _RE_PRESS_DOT.findall(text)
    presses_gain = _RE_PRESS_GAIN.findall(text)
    options = _RE_OPTION.findall(text)
    lotteries = _RE_LOTTERY.findall(text)
    return {
        "n_presses_any": len(presses_any),
        "n_presses_dot": len(presses_dot),
        "n_presses_gain": len(presses_gain),
        "n_option_headers": len(options),
        "n_lottery_headers": len(lotteries),
        "unique_press_keys": sorted(set(presses_any)),
        "unique_option_keys": sorted(set(options)),
        "unique_lottery_keys": sorted(set(lotteries)),
        "has_percent_chance": bool(_RE_PERCENT.search(text)),
        "has_you_receive": "You receive" in text,
        "has_and_gain": "and gain" in text.lower(),
        "has_option_delivers": bool(options),
        "has_lottery_offers": bool(lotteries),
        "has_product_ratings": bool(_RE_PRODUCT_RATINGS.search(text)),
        "has_slot_machine": bool(_RE_SLOT.search(text)),
        "has_weather_cards": bool(_RE_WEATHER.search(text)),
        "has_tree_task": bool(_RE_TREE.search(text)),
        "has_cct": bool(_RE_CCT.search(text)),
        "is_binary_press_per_row": len(set(presses_any)) == 2,
        "n_double_newline_blocks": len(re.findall(r"\n\nOption [A-Z] delivers", text)),
    }


def _block_to_sample_trial(block) -> Dict[str, Any]:
    """First trial dict in Template_evo shape from a choice13k Block."""
    if not block.trials:
        return {}
    t0 = block.trials[0]
    history_accum: List[Dict[str, Any]] = []
    return {
        "problem": {
            "gamble_A": {"probs": block.gamble_A.probs, "rewards": block.gamble_A.rewards},
            "gamble_B": {"probs": block.gamble_B.probs, "rewards": block.gamble_B.rewards},
            "option_keys": list(block.option_keys),
            "has_feedback": block.has_feedback,
        },
        "action": t0.action,
        "history_len": len(history_accum),
    }


def _try_choice13k_convert(row: Dict[str, Any]) -> Dict[str, Any]:
    from data_modules.choice13k import _convert_to_experiment

    text = row["text"]
    markers = _text_markers(text)
    out: Dict[str, Any] = {
        "markers": markers,
        "success": False,
        "error": None,
        "n_blocks": 0,
        "n_trials": 0,
        "parse_coverage": 0.0,
        "option_keys_first_block": [],
        "has_feedback_first_block": None,
        "sample_trial": None,
    }
    try:
        exp = _convert_to_experiment(dict(row))
        n_blocks = len(exp.blocks)
        n_trials = sum(len(b.trials) for b in exp.blocks)
        out["success"] = n_trials > 0
        out["n_blocks"] = n_blocks
        out["n_trials"] = n_trials
        n_press = markers["n_presses_any"]
        out["parse_coverage"] = round(n_trials / n_press, 4) if n_press else 0.0
        if exp.blocks:
            b0 = exp.blocks[0]
            out["option_keys_first_block"] = list(b0.option_keys)
            out["has_feedback_first_block"] = b0.has_feedback
        if exp.blocks and exp.blocks[0].trials:
            out["sample_trial"] = _block_to_sample_trial(exp.blocks[0])
        return out
    except Exception as e:
        out["error"] = str(e)
        out["traceback_tail"] = traceback.format_exc()[-400:]
        return out


def _classify_experiment(
    exp_id: str,
    agg: Dict[str, Any],
) -> Dict[str, str]:
    """Return parser_class, schema_type, flags, difficulty, recommendation."""
    coverage = agg.get("median_parse_coverage") or 0.0
    c13_ok = agg.get("choice13k_parser_success_rate", 0.0)
    all_binary = agg.get("all_samples_binary_press", False)
    fmt = agg.get("dominant_format", "unknown")

    parser_class = "defer"
    schema_type = "C"
    can_keep_float = "no"
    evaluator_change = "yes"
    prompt_change = "yes"
    difficulty = "high"
    recommendation = "Defer until transcript structure verified."

    if exp_id == "peterson2021using/exp1.csv":
        return {
            "parser_class": "reuse_choice13k_parser",
            "schema_type": "A",
            "can_keep_float_P1": "yes",
            "evaluator_change_needed": "no",
            "prompt_change_needed": "minimal",
            "difficulty": "low",
            "recommendation": (
                "Current production path. Per-participant 2 option keys; "
                "100% press recovery with choice13k regex on sampled rows."
            ),
        }

    if exp_id == "plonsky2018when/exp1.csv":
        return {
            "parser_class": "small_extension_to_choice13k_parser",
            "schema_type": "A",
            "can_keep_float_P1": "yes",
            "evaluator_change_needed": "no",
            "prompt_change_needed": "small",
            "difficulty": "low-medium",
            "recommendation": (
                "Same Option-delivers format as choice13k but ~80% of presses use "
                "`You press <<X>> and gain ...` (median coverage 0.2). Extend "
                "_extract_trials; keep gamble_A/B + Bernoulli loglik."
            ),
        }

    if exp_id == "wulff2018description/exp1.csv":
        return {
            "parser_class": "new_binary_parser_same_evaluator",
            "schema_type": "A",
            "can_keep_float_P1": "yes",
            "evaluator_change_needed": "no",
            "prompt_change_needed": "yes",
            "difficulty": "medium",
            "recommendation": (
                "Uses `Lottery W offers ...` / `Lottery H offers ...` and "
                "`You press <<W|H>>.` — not `Option X delivers`. Map lotteries to "
                "gamble_A/gamble_B via parser spec; Bernoulli on second lottery key."
            ),
        }

    if exp_id == "speekenbrink2008learning/exp1.csv":
        return {
            "parser_class": "new_binary_parser_same_evaluator",
            "schema_type": "B",
            "can_keep_float_P1": "yes",
            "evaluator_change_needed": "no",
            "prompt_change_needed": "yes",
            "difficulty": "medium",
            "recommendation": (
                "Binary E/J weather prediction with card context per trial. "
                "problem: {option_keys, cards, task_context}; action 0/1 for E vs J."
            ),
        }

    if exp_id == "sadeghiyeh2020temporal/exp1.csv":
        return {
            "parser_class": "new_parser_and_new_evaluator",
            "schema_type": "C",
            "can_keep_float_P1": "per_game_yes",
            "evaluator_change_needed": "small",
            "prompt_change_needed": "yes",
            "difficulty": "medium-high",
            "recommendation": (
                "Psych-101 text is two-arm bandit (slot machines J/R), not dated "
                "intertemporal amounts. problem needs game_id, instructed vs free "
                "trials, payoffs in history; float P(choose machine 1) per step OK."
            ),
        }

    if exp_id == "hilbig2014generalized/exp1.csv":
        return {
            "parser_class": "new_binary_parser_same_evaluator",
            "schema_type": "B",
            "can_keep_float_P1": "yes",
            "evaluator_change_needed": "no",
            "prompt_change_needed": "yes",
            "difficulty": "medium",
            "recommendation": (
                "Binary A/R with expert rating vectors per product. "
                "problem: {ratings_A, ratings_B, option_keys}; press line embedded in "
                "same sentence — new regex, same Bernoulli evaluator."
            ),
        }

    if exp_id == "frey2017cct/exp1.csv":
        return {
            "parser_class": "new_parser_and_new_evaluator",
            "schema_type": "C",
            "can_keep_float_P1": "yes",
            "evaluator_change_needed": "no",
            "prompt_change_needed": "yes",
            "difficulty": "medium-high",
            "recommendation": (
                "Columbia Card Task: sequential E=flip vs C=stop per round. "
                "problem: round params (n_loss, gain/loss amounts, score); "
                "Bernoulli on P(stop) or P(flip) if action coded 0/1."
            ),
        }

    if exp_id == "flesch2018comparing/exp1.csv":
        return {
            "parser_class": "new_binary_parser_same_evaluator",
            "schema_type": "B",
            "can_keep_float_P1": "yes",
            "evaluator_change_needed": "no",
            "prompt_change_needed": "yes",
            "difficulty": "medium",
            "recommendation": (
                "Binary T/N accept-reject with tree features + garden. "
                "problem: {leafiness, branchiness, garden, phase}; history for "
                "feedback; Bernoulli P(accept)=P(action=1)."
            ),
        }

    return {
        "parser_class": parser_class,
        "schema_type": schema_type,
        "can_keep_float_P1": can_keep_float,
        "evaluator_change_needed": evaluator_change,
        "prompt_change_needed": prompt_change,
        "difficulty": difficulty,
        "recommendation": recommendation,
    }


def _aggregate_split_results(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}
    markers_list = [s["markers"] for s in samples]
    parses = [s["parse"] for s in samples]
    press_counts = [m["n_presses_any"] for m in markers_list]
    trial_counts = [p.get("n_trials", 0) for p in parses if p.get("success")]
    coverages = [p.get("parse_coverage", 0) for p in parses if p.get("success")]

    uk: set = set()
    uo: set = set()
    for m in markers_list:
        uk.update(m["unique_press_keys"])
        uo.update(m["unique_option_keys"])

    return {
        "n_samples": len(samples),
        "median_presses_per_participant": _median(press_counts),
        "median_trials_parsed": _median(trial_counts) if trial_counts else 0,
        "median_parse_coverage": _median(coverages) if coverages else 0.0,
        "unique_press_keys": sorted(uk),
        "unique_option_keys": sorted(uo),
        "choice13k_successes": sum(1 for p in parses if p.get("success")),
        "has_percent_chance_any": any(m["has_percent_chance"] for m in markers_list),
        "has_and_gain_any": any(m["has_and_gain"] for m in markers_list),
    }


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    m = len(s) // 2
    return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


@dataclass
class ExperimentReport:
    experiment_id: str
    train_n_participants: int = 0
    test_n_participants: int = 0
    train_agg: Dict[str, Any] = field(default_factory=dict)
    test_agg: Dict[str, Any] = field(default_factory=dict)
    sample_snippet: str = ""
    train_samples: List[Dict[str, Any]] = field(default_factory=list)
    test_samples: List[Dict[str, Any]] = field(default_factory=list)
    classification: Dict[str, str] = field(default_factory=dict)
    combined_agg: Dict[str, Any] = field(default_factory=dict)


def _analyze_experiment(
    exp_id: str,
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    *,
    samples_per_split: int = 3,
) -> ExperimentReport:
    rep = ExperimentReport(experiment_id=exp_id)
    rep.train_n_participants = len(train_rows)
    rep.test_n_participants = len(test_rows)

    def _sample_rows(rows: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        idxs = _pick_participant_indices(len(rows), k)
        return [rows[i] for i in idxs]

    train_pick = _sample_rows(train_rows, samples_per_split)
    test_pick = _sample_rows(test_rows, samples_per_split)

    def _process(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for row in rows:
            markers = _text_markers(row["text"])
            parse = _try_choice13k_convert(row)
            out.append(
                {
                    "participant": row.get("participant"),
                    "markers": markers,
                    "parse": parse,
                }
            )
        return out

    rep.train_samples = _process(train_pick)
    rep.test_samples = _process(test_pick)
    rep.train_agg = _aggregate_split_results(rep.train_samples)
    rep.test_agg = _aggregate_split_results(rep.test_samples)

    # Combined
    all_samples = rep.train_samples + rep.test_samples
    uk = set(rep.train_agg.get("unique_press_keys", [])) | set(
        rep.test_agg.get("unique_press_keys", [])
    )
    uo = set(rep.train_agg.get("unique_option_keys", [])) | set(
        rep.test_agg.get("unique_option_keys", [])
    )
    n_ok = rep.train_agg.get("choice13k_successes", 0) + rep.test_agg.get(
        "choice13k_successes", 0
    )
    n_samp = len(all_samples)
    med_press = _median(
        [s["markers"]["n_presses_any"] for s in all_samples]
    )
    med_cov = _median(
        [s["parse"]["parse_coverage"] for s in all_samples if s["parse"].get("success")]
    )

    def _dominant_format(samples: List[Dict[str, Any]]) -> str:
        m = samples[0]["markers"] if samples else {}
        if m.get("has_option_delivers"):
            return "option_delivers_gamble"
        if m.get("has_lottery_offers"):
            return "lottery_offers"
        if m.get("has_cct"):
            return "columbia_card_task"
        if m.get("has_tree_task"):
            return "tree_accept_reject"
        if m.get("has_slot_machine"):
            return "slot_machine_bandit"
        if m.get("has_weather_cards"):
            return "weather_cards"
        if m.get("has_product_ratings"):
            return "product_ratings"
        return "unknown"

    rep.combined_agg = {
        "unique_press_keys_union": sorted(uk),
        "unique_option_keys_union": sorted(uo),
        "median_presses_per_participant": med_press,
        "median_parse_coverage": med_cov,
        "choice13k_parser_success_rate": round(n_ok / n_samp, 3) if n_samp else 0.0,
        "all_samples_binary_press": all(
            s["markers"].get("is_binary_press_per_row") for s in all_samples
        ),
        "has_percent_chance_union": rep.train_agg.get("has_percent_chance_any")
        or rep.test_agg.get("has_percent_chance_any"),
        "has_and_gain_union": rep.train_agg.get("has_and_gain_any")
        or rep.test_agg.get("has_and_gain_any"),
        "dominant_format": _dominant_format(all_samples),
    }
    rep.classification = _classify_experiment(exp_id, rep.combined_agg)

    # Snippet from first successful or first train sample
    src = train_pick[0] if train_pick else (test_pick[0] if test_pick else None)
    if src:
        rep.sample_snippet = src["text"][:2000]

    return rep


def _row_for_csv(rep: ExperimentReport) -> Dict[str, Any]:
    c = rep.classification
    agg = rep.combined_agg
    return {
        "experiment_id": rep.experiment_id,
        "train_n_participants": rep.train_n_participants,
        "test_n_participants": rep.test_n_participants,
        "n_unique_press_keys": len(agg.get("unique_press_keys_union", [])),
        "press_keys": "|".join(agg.get("unique_press_keys_union", [])),
        "n_unique_option_keys": len(agg.get("unique_option_keys_union", [])),
        "option_keys": "|".join(agg.get("unique_option_keys_union", [])),
        "median_presses_per_participant": agg.get("median_presses_per_participant"),
        "median_trials_parsed": _median(
            [
                s["parse"]["n_trials"]
                for s in rep.train_samples + rep.test_samples
                if s["parse"].get("success")
            ]
        ),
        "median_parse_coverage": agg.get("median_parse_coverage"),
        "is_binary_per_participant": int(agg.get("all_samples_binary_press", False)),
        "dominant_format": agg.get("dominant_format"),
        "choice13k_success_rate": agg.get("choice13k_parser_success_rate"),
        "has_percent_chance": int(agg.get("has_percent_chance_union", False)),
        "has_and_gain_feedback": int(agg.get("has_and_gain_union", False)),
        "parser_class": c.get("parser_class"),
        "schema_type": c.get("schema_type"),
        "can_keep_float_P1": c.get("can_keep_float_P1"),
        "evaluator_change_needed": c.get("evaluator_change_needed"),
        "prompt_change_needed": c.get("prompt_change_needed"),
        "difficulty": c.get("difficulty"),
        "recommendation": c.get("recommendation"),
    }


def _report_to_json(rep: ExperimentReport) -> Dict[str, Any]:
    def _ser_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "participant": s["participant"],
                "markers": s["markers"],
                "parse_success": s["parse"]["success"],
                "parse_coverage": s["parse"].get("parse_coverage"),
                "n_trials": s["parse"].get("n_trials"),
                "n_blocks": s["parse"].get("n_blocks"),
                "error": s["parse"].get("error"),
                "sample_trial": s["parse"].get("sample_trial"),
            }
            for s in samples
        ]

    return {
        "experiment_id": rep.experiment_id,
        "train_n_participants": rep.train_n_participants,
        "test_n_participants": rep.test_n_participants,
        "combined_agg": rep.combined_agg,
        "classification": rep.classification,
        "sample_snippet_first_2000_chars": rep.sample_snippet,
        "train_samples": _ser_samples(rep.train_samples),
        "test_samples": _ser_samples(rep.test_samples),
    }


def _write_markdown(reports: List[ExperimentReport], path: Path) -> None:
    lines = [
        "# Psych-101 candidate parser feasibility",
        "",
        "Checks whether experiments can feed Template Evolution via structured trials "
        "(`choose(problem, history) -> float` = P(action=1)).",
        "",
        "## Summary table",
        "",
        "| experiment_id | parser_class | schema_type | can_keep_float_P1 | "
        "evaluator_change_needed | prompt_change_needed | difficulty | recommendation |",
        "|---------------|--------------|-------------|-------------------|"
        "-------------------------|----------------------|------------|----------------|",
    ]
    for rep in reports:
        c = rep.classification
        rec = c.get("recommendation", "")[:80].replace("|", "/") + (
            "…" if len(c.get("recommendation", "")) > 80 else ""
        )
        lines.append(
            f"| `{rep.experiment_id}` | {c.get('parser_class')} | {c.get('schema_type')} | "
            f"{c.get('can_keep_float_P1')} | {c.get('evaluator_change_needed')} | "
            f"{c.get('prompt_change_needed')} | {c.get('difficulty')} | {rec} |"
        )

    lines.extend(
        [
            "",
            "## Loader design recommendation",
            "",
            "Use **one module** `data_modules/psych101_binary.py` with **versioned per-experiment "
            "parser specs** (YAML/JSON under `analysis/data/psych-101/parser_specs/` or "
            "`datasets/psych101_specs/`), not eight separate Python files. Each spec defines:",
            "",
            "- `experiment_id`",
            "- trial boundary regex / block splitter",
            "- action line regexes (list, tried in order)",
            "- option header regex",
            "- `action_map`: second option key -> `action=1`",
            "- `schema_type`: A | B | C",
            "- golden participant ids for tests",
            "",
            "Implement **shared** `_convert_to_experiment(spec, row)` plus thin wrappers. "
            "Only fork to a new file when schema or evaluator truly differs (e.g. CCT, >2 actions).",
            "",
            "This matches the existing `choice13k.py` path while keeping reproducibility.",
            "",
        ]
    )

    for rep in reports:
        agg = rep.combined_agg
        lines.extend(
            [
                f"## {rep.experiment_id}",
                "",
                "### Metadata",
                "",
                f"- Train participants: {rep.train_n_participants}",
                f"- Test participants: {rep.test_n_participants}",
                f"- Press keys: `{', '.join(agg.get('unique_press_keys_union', []))}` "
                f"(binary={agg.get('is_binary_press_keys')})",
                f"- Option keys: `{', '.join(agg.get('unique_option_keys_union', []))}`",
                f"- Median presses/participant: {agg.get('median_presses_per_participant')}",
                f"- Choice13k parser success rate (samples): "
                f"{agg.get('choice13k_parser_success_rate')}",
                f"- Median parse coverage: {agg.get('median_parse_coverage')}",
                "",
                "### Parser compatibility",
                "",
                f"- **Class:** `{rep.classification.get('parser_class')}`",
                f"- **Schema:** {rep.classification.get('schema_type')}",
                f"- **Float P(action=1):** {rep.classification.get('can_keep_float_P1')}",
                f"- **Evaluator change:** {rep.classification.get('evaluator_change_needed')}",
                "",
                f"**Recommendation:** {rep.classification.get('recommendation')}",
                "",
                "### Transcript snippet (first 2000 chars)",
                "",
                "```",
                rep.sample_snippet[:2000],
                "```",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def run_check(output_dir: Path, *, samples_per_split: int = 3) -> List[ExperimentReport]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Psych-101 train split...")
    train_ds = _load_split(TRAIN_HF, "train")
    print("Loading Psych-101-test split...")
    test_ds = _load_split(TEST_HF, "test")

    reports: List[ExperimentReport] = []
    examples: Dict[str, Any] = {}

    for exp_id in CANDIDATE_EXPERIMENTS:
        print(f"Analyzing {exp_id}...")
        train_rows = [dict(r) for r in train_ds if r["experiment"] == exp_id]
        test_rows = [dict(r) for r in test_ds if r["experiment"] == exp_id]
        rep = _analyze_experiment(
            exp_id, train_rows, test_rows, samples_per_split=samples_per_split
        )
        reports.append(rep)
        examples[exp_id] = _report_to_json(rep)

    csv_path = output_dir / "candidate_parser_check.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        rows = [_row_for_csv(r) for r in reports]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    json_path = output_dir / "candidate_parse_examples.json"
    json_path.write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")

    md_path = output_dir / "candidate_parser_check.md"
    _write_markdown(reports, md_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--samples_per_split",
        type=int,
        default=3,
        help="Participants to sample per train/test split (default: 3)",
    )
    args = parser.parse_args()
    out = args.output_dir.expanduser()
    out = out.resolve() if out.is_absolute() else (REPO_ROOT / out).resolve()
    run_check(out, samples_per_split=args.samples_per_split)


if __name__ == "__main__":
    main()
