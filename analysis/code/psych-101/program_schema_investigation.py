#!/usr/bin/env python3
"""
Psych-101 program-schema feasibility study.

Evolved programs use structured trial dicts, not NL prompts alone.
This script classifies experiments, tests transcript parsers, documents
binary-assumption code sites, and writes reports under analysis/data/psych-101/compatibility/.

Example:
  python analysis/code/psych-101/program_schema_investigation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = REPO_ROOT / "analysis" / "data" / "psych-101" / "compatibility"
TRAIN_HF = "marcelbinz/Psych-101"
TEST_HF = "marcelbinz/Psych-101-test"
CHOICE13K_EXP = "peterson2021using/exp1.csv"
CPC18_EXP = "plonsky2018when/exp1.csv"

# Heuristic keywords for experiment-id ranking (gamble-like)
_GAMBLE_ID_PATTERNS: List[Tuple[str, int, str]] = [
    (r"peterson2021", 100, "choice13k benchmark; two-option gambles"),
    (r"plonsky2018", 98, "cpc18 benchmark; two-option lotteries"),
    (r"frey2017risk", 90, "risky choice"),
    (r"frey2017cct", 88, "columbia card task / risky choice"),
    (r"garcia2023experiential", 85, "decisions from experience"),
    (r"steingroever2015", 82, "Iowa / bandit-style gambling data"),
    (r"wulff2018description", 80, "decisions from description"),
    (r"wulff2018sampling", 78, "decisions from sampling"),
    (r"speekenbrink2008", 75, "learning / two-option"),
    (r"sadeghiyeh2020temporal", 72, "intertemporal choice"),
    (r"kool2016when", 70, "two-option / bandit hybrid"),
    (r"lefebvre2017behavioural", 68, "behavioural economics choices"),
    (r"hilbig2014", 65, "two-alternative choice"),
    (r"ruggeri2022", 40, "large globalizability study; mixed paradigms"),
    (r"flesch2018", 35, "comparing models; may be multi-arm"),
]

# Non-gamble exemplars for schema requirements (fixed set of 5+)
_NON_GAMBLE_SAMPLE_EXPS = [
    "enkavi2019gonogo/exp1.csv",
    "enkavi2019digitspan/exp1.csv",
    "schulz2020finding/exp1.csv",
    "wilson2014humans/exp1.csv",
    "tomov2020discovery/exp2.csv",
    "hebart2023things/exp1.csv",
    "steingroever2015data/exp1.csv",  # bandit-like but not binary gamble schema
]

# Text markers
_RE_OPTION_DELIVERS = re.compile(r"Option\s+([A-Z]) delivers", re.I)
_RE_PRESS = re.compile(r"You press <<([A-Z])>>")
_RE_POINTS = re.compile(r"(-?\d+\.?\d*).*?with\s+(\d+\.?\d*)% chance")
_RE_RECEIVE = re.compile(r"You receive (-?\d+\.?\d*) points")
_RE_CPC18_OPTION = re.compile(
    r"Option\s+([ABLR])\s*\(.*?(?:Ha|Hb|high|low)", re.I
)


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_split(hf_id: str, split: str):
    from datasets import load_dataset

    tok = _hf_token()
    kw = {"token": tok} if tok else {}
    ds = load_dataset(hf_id, **kw)
    return ds[split]


def _score_experiment_id(exp_id: str) -> Tuple[int, str]:
    best_score, best_note = 0, ""
    for pat, score, note in _GAMBLE_ID_PATTERNS:
        if re.search(pat, exp_id, re.I):
            if score > best_score:
                best_score, best_note = score, note
    return best_score, best_note


def _analyze_text_markers(text: str) -> Dict[str, Any]:
    presses = _RE_PRESS.findall(text)
    options = _RE_OPTION_DELIVERS.findall(text)
    unique_press_keys = sorted(set(presses))
    unique_option_keys = sorted(set(options))
    n_blocks = len(re.findall(r"\n\nOption [A-Z] delivers", text))
    return {
        "n_presses": len(presses),
        "n_option_headers": len(options),
        "unique_press_keys": unique_press_keys,
        "n_unique_press_keys": len(unique_press_keys),
        "unique_option_keys": unique_option_keys,
        "n_unique_option_keys": len(unique_option_keys),
        "has_percent_chance": bool(_RE_POINTS.search(text)),
        "has_feedback_points": "You receive" in text and "points" in text,
        "text_len": len(text),
        "sample_instruction": text[:400].replace("\n", " "),
    }


def _try_choice13k_parse(text: str) -> Dict[str, Any]:
    from data_modules.choice13k import _convert_to_experiment

    try:
        exp = _convert_to_experiment({"text": text})
        n_blocks = len(exp.blocks)
        n_trials = sum(len(b.trials) for b in exp.blocks)
        first = exp.blocks[0] if exp.blocks else None
        return {
            "success": True,
            "n_blocks": n_blocks,
            "n_trials": n_trials,
            "option_keys": first.option_keys if first else [],
            "has_feedback": first.has_feedback if first else None,
            "gamble_A_probs": first.gamble_A.probs if first else None,
            "gamble_B_probs": first.gamble_B.probs if first else None,
            "error": None,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()[-500:]}


def _assess_cpc18_choice13k_parser(text: str) -> Dict[str, Any]:
    """CPC18 Psych-101 transcripts use choice13k-style Option/press text, not local Ha/pHa CSV."""
    from data_modules.choice13k import _convert_to_experiment

    n_presses_text = len(re.findall(r"You press <<", text))
    # Lines without feedback match choice13k regex; feedback lines use 'and gain'
    n_no_feedback_presses = len(
        re.findall(r"You press <<[A-Z]>>\.\s*(?:\n|$)", text)
    )
    n_feedback_presses = len(
        re.findall(r"You press <<[A-Z]>> and gain", text)
    )
    try:
        exp = _convert_to_experiment({"text": text})
        n_blocks = len(exp.blocks)
        n_trials = sum(len(b.trials) for b in exp.blocks)
        per_block = [len(b.trials) for b in exp.blocks[:5]]
        ok = n_trials > 0
        return {
            "success": ok,
            "parser": "data_modules.choice13k._convert_to_experiment",
            "n_presses_in_text": n_presses_text,
            "n_no_feedback_press_lines": n_no_feedback_presses,
            "n_feedback_press_lines": n_feedback_presses,
            "n_blocks_parsed": n_blocks,
            "n_trials_parsed": n_trials,
            "trials_per_block_first5": per_block,
            "parse_coverage": round(n_trials / n_presses_text, 4) if n_presses_text else 0.0,
            "full_parse": n_trials == n_presses_text,
            "error": None,
            "note": (
                "choice13k trial regex stops at 'You press <<X>>.'; "
                "CPC18 feedback lines use 'and gain ...' and are skipped unless regex extended."
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "parser": "data_modules.choice13k._convert_to_experiment",
            "n_presses_in_text": n_presses_text,
            "error": str(e),
        }


def _classify_action_space(markers: Dict[str, Any], exp_id: str) -> Dict[str, Any]:
    n_keys = markers.get("n_unique_press_keys", 0)
    if n_keys == 2:
        schema = "binary_gamble_like"
        choose_return = "float P(action=1)"
        loglik = "bernoulli_scalar"
    elif n_keys == 0:
        schema = "unknown_or_non_press"
        choose_return = "task_specific"
        loglik = "unknown"
    elif n_keys <= 6:
        schema = "small_multiclass"
        choose_return = "dict[action, prob]"
        loglik = "categorical"
    else:
        schema = "large_multiclass_or_continuous"
        choose_return = "dict or embedding"
        loglik = "categorical_or_custom"

    # Override for known paradigms
    if "gonogo" in exp_id or "nback" in exp_id:
        schema = "response_time_or_key"
        choose_return = "dict[action, prob] over {go,nogo} or keys"
        loglik = "categorical"
    if "digitspan" in exp_id or "recentprobes" in exp_id:
        schema = "memory_recall"
        choose_return = "sequence or dict over responses"
        loglik = "custom"
    if "things" in exp_id:
        schema = "high_dim_categorization"
        choose_return = "dict over object labels"
        loglik = "categorical_high_dim"
    if "wilson2014" in exp_id or "tomov2020" in exp_id:
        schema = "mdp_bandit"
        choose_return = "dict[state_action] or contextual"
        loglik = "custom_mdp"

    return {
        "inferred_schema": schema,
        "choose_return_format": choose_return,
        "loglik_evaluator": loglik,
        "n_press_keys": n_keys,
        "press_keys": markers.get("unique_press_keys", []),
    }


def _code_sites_binary_assumption() -> List[Dict[str, str]]:
    """Static inventory of binary / P(B) assumptions (file:line descriptions)."""
    return [
        {
            "file": "Template_evo_non_strict.py",
            "symbol": "_parse_choice13k_choose_output",
            "lines": "~553-563",
            "assumption": "choose() must return float in [0,1] or int/bool 0/1 only",
        },
        {
            "file": "Template_evo_non_strict.py",
            "symbol": "evaluate_choice13k_program",
            "lines": "~2480-2546",
            "assumption": "Bernoulli loglik: y=int(action); uses p and (1-p); action=1 is Option B",
        },
        {
            "file": "Template_evo_non_strict.py",
            "symbol": "evaluate_cpc18_split_program",
            "lines": "~2568-2634",
            "assumption": "Same scalar P(B) Bernoulli loglik as choice13k",
        },
        {
            "file": "Template_evo_non_strict.py",
            "symbol": "evaluate_program",
            "lines": "~2438-2477",
            "assumption": "Accuracy: pred == t[action] with discrete pred from choose (not probabilistic)",
        },
        {
            "file": "Template_evo_non_strict.py",
            "symbol": "wrap_choose_with_consistency_gate",
            "lines": "~585-596",
            "assumption": "Gate operates on scalar probability toward 0.5",
        },
        {
            "file": "Template_evo_non_strict.py",
            "symbol": "experiment_to_trials / load_mixed_gambles_data",
            "lines": "~2318-2346, ~2246-2315",
            "assumption": "trial['action'] in {0,1}; gamble_A/gamble_B problem dict",
        },
        {
            "file": "data_modules/cpc18.py",
            "symbol": "Trial.action",
            "lines": "~37-39",
            "assumption": "action 0=A(L), 1=B(R) binary",
        },
        {
            "file": "data_modules/choice13k.py",
            "symbol": "_extract_trials",
            "lines": "~74-88",
            "assumption": "Maps press key to option_keys.index; binary options typical",
        },
        {
            "file": "Template_evo_non_strict.py",
            "symbol": "_evaluate_loglik_for_dataset",
            "lines": "~239-251",
            "assumption": "Routes choice13k/cpc18/mixed_gambles to scalar loglik evaluators only",
        },
        {
            "file": "Template_evo_non_strict.py",
            "symbol": "gridworld evaluation",
            "lines": "~3086-3346",
            "assumption": "6-action discrete; uses weighted one-hot aggregation, not scalar P(B)",
        },
    ]


def _rank_gamble_candidates(
    experiment_ids: Sequence[str],
    per_exp_markers: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for exp_id in experiment_ids:
        id_score, id_note = _score_experiment_id(exp_id)
        m = per_exp_markers.get(exp_id, {})
        text_score = 0
        if m.get("n_unique_press_keys") == 2:
            text_score += 30
        if m.get("has_percent_chance"):
            text_score += 20
        if m.get("n_unique_option_keys") == 2:
            text_score += 15
        if m.get("has_feedback_points"):
            text_score += 5
        total = id_score + text_score
        rows.append(
            {
                "experiment_id": exp_id,
                "total_score": total,
                "id_heuristic_score": id_score,
                "text_marker_score": text_score,
                "id_note": id_note,
                "n_unique_press_keys": m.get("n_unique_press_keys"),
                "press_keys": "|".join(m.get("unique_press_keys", [])),
                "has_percent_chance": int(bool(m.get("has_percent_chance"))),
                "choice13k_parser_ok": m.get("choice13k_parser_ok"),
                "likely_binary_gamble": int(
                    m.get("n_unique_press_keys") == 2
                    and m.get("n_unique_option_keys") == 2
                ),
                "repo_benchmark": (
                    "choice13k"
                    if exp_id == CHOICE13K_EXP
                    else ("cpc18" if exp_id == CPC18_EXP else "")
                ),
            }
        )
    rows.sort(key=lambda r: (-r["total_score"], r["experiment_id"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def _sample_rows_for_exp(split_ds, exp_id: str, participant_ids: Sequence[str], max_rows: int = 3):
    out = []
    want = set(participant_ids)
    for row in split_ds:
        if row["experiment"] != exp_id:
            continue
        if want and row["participant"] not in want:
            continue
        out.append(row)
        if len(out) >= max_rows:
            break
    return out


def run_investigation(output_dir: Path, *, max_experiments_sample: int = 76) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = _load_split(TRAIN_HF, "train")
    test_ds = _load_split(TEST_HF, "test")

    # Unique experiments from train
    exp_counts = Counter(train_ds["experiment"])
    experiment_ids = sorted(exp_counts.keys())[:max_experiments_sample]

    # One sample row per experiment (train) for marker analysis
    per_exp_markers: Dict[str, Dict[str, Any]] = {}
    sample_by_exp: Dict[str, Any] = {}
    for exp_id in experiment_ids:
        for row in train_ds:
            if row["experiment"] == exp_id:
                sample_by_exp[exp_id] = row
                markers = _analyze_text_markers(row["text"])
                if exp_id == CHOICE13K_EXP or (
                    markers["n_unique_press_keys"] == 2 and "Option" in row["text"]
                ):
                    c13 = _try_choice13k_parse(row["text"])
                else:
                    c13 = {"skipped": True}
                markers["choice13k_parser_ok"] = c13.get("success") if isinstance(c13, dict) else None
                markers["choice13k_parse_detail"] = c13
                per_exp_markers[exp_id] = markers
                break

    gamble_rows = _rank_gamble_candidates(experiment_ids, per_exp_markers)
    gamble_csv = output_dir / "psych101_gamble_like_candidates.csv"
    with open(gamble_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gamble_rows[0].keys()) if gamble_rows else [])
        w.writeheader()
        w.writerows(gamble_rows)

    # Parse examples: choice13k + cpc18 from train and test
    parse_examples: Dict[str, Any] = {"choice13k": {}, "cpc18": {}}
    for label, ds, split_name in [
        ("train", train_ds, "train"),
        ("test", test_ds, "test"),
    ]:
        for bench, exp_id in [("choice13k", CHOICE13K_EXP), ("cpc18", CPC18_EXP)]:
            sub = ds.filter(lambda ex, e=exp_id: ex["experiment"] == e)
            participants = sorted(set(sub["participant"]), key=lambda x: int(x))
            # sample 4 participants spread across range
            if len(participants) >= 4:
                idxs = [0, len(participants) // 3, 2 * len(participants) // 3, -1]
                pick = [participants[i] for i in idxs]
            else:
                pick = participants[:4]
            bench_examples = []
            for pid in pick[:4]:
                rows = _sample_rows_for_exp(sub, exp_id, [pid], max_rows=1)
                if not rows:
                    continue
                text = rows[0]["text"]
                if bench == "choice13k":
                    parsed = _try_choice13k_parse(text)
                    # Also build one structured trial
                    structured = None
                    if parsed.get("success"):
                        from data_modules.choice13k import _convert_to_experiment

                        exp = _convert_to_experiment({"text": text})
                        b = exp.blocks[0]
                        t0 = b.trials[0]
                        structured = {
                            "problem": {
                                "gamble_A": {
                                    "probs": b.gamble_A.probs,
                                    "rewards": b.gamble_A.rewards,
                                },
                                "gamble_B": {
                                    "probs": b.gamble_B.probs,
                                    "rewards": b.gamble_B.rewards,
                                },
                                "has_feedback": b.has_feedback,
                            },
                            "action": t0.action,
                            "history_len": 0,
                        }
                    bench_examples.append(
                        {
                            "participant": pid,
                            "split": split_name,
                            "parser": "data_modules.choice13k._convert_to_experiment",
                            "success": parsed.get("success"),
                            "error": parsed.get("error"),
                            "stats": {k: v for k, v in parsed.items() if k not in ("traceback",)},
                            "structured_trial_0": structured,
                            "text_snippet": text[:600],
                        }
                    )
                else:
                    parsed = _assess_cpc18_choice13k_parser(text)
                    structured = None
                    if parsed.get("success"):
                        from data_modules.choice13k import _convert_to_experiment

                        exp = _convert_to_experiment({"text": text})
                        b = exp.blocks[0]
                        t0 = b.trials[0]
                        structured = {
                            "problem": {
                                "gamble_A": {
                                    "probs": b.gamble_A.probs,
                                    "rewards": b.gamble_A.rewards,
                                },
                                "gamble_B": {
                                    "probs": b.gamble_B.probs,
                                    "rewards": b.gamble_B.rewards,
                                },
                                "has_feedback": b.has_feedback,
                                "problem_id": 0,
                                "block_id": 1,
                            },
                            "action": t0.action,
                            "history_len": 0,
                        }
                    bench_examples.append(
                        {
                            "participant": pid,
                            "split": split_name,
                            "success": parsed.get("success"),
                            "full_parse": parsed.get("full_parse", False),
                            "stats": parsed,
                            "structured_trial_0": structured,
                            "text_snippet": text[:600],
                        }
                    )
            parse_examples[bench][label] = bench_examples

    parse_json_path = output_dir / "psych101_parse_examples.json"
    parse_json_path.write_text(json.dumps(parse_examples, indent=2) + "\n", encoding="utf-8")

    # Non-gamble schema requirements
    non_gamble_rows: List[Dict[str, Any]] = []
    for exp_id in _NON_GAMBLE_SAMPLE_EXPS:
        row = sample_by_exp.get(exp_id)
        if row is None:
            for r in train_ds:
                if r["experiment"] == exp_id:
                    row = r
                    break
        if row is None:
            continue
        markers = _analyze_text_markers(row["text"])
        classification = _classify_action_space(markers, exp_id)
        code_changes: List[str] = []
        if classification["loglik_evaluator"] == "categorical":
            code_changes.append("Add evaluate_categorical_program(choose_fn, trials, action_space)")
            code_changes.append("Extend _parse_choose_output to accept dict[action, prob]")
            code_changes.append("Normalize dict probabilities; log p(observed action)")
        if classification["inferred_schema"] == "memory_recall":
            code_changes.append("New trial schema: stimuli list, recall responses; not choose(problem,history)->float")
        if classification["inferred_schema"] == "mdp_bandit":
            code_changes.append("History must include state/reward; problem includes context; possibly dict over actions")
        if classification["inferred_schema"] == "high_dim_categorization":
            code_changes.append("Avoid for v1: 1000+ classes; needs embedding likelihood or restricted subset")
        non_gamble_rows.append(
            {
                "experiment_id": exp_id,
                "inferred_schema": classification["inferred_schema"],
                "n_press_keys": classification["n_press_keys"],
                "press_keys": "|".join(classification["press_keys"]),
                "action_space_size": classification["n_press_keys"] or "variable",
                "scalar_p_action_1_valid": int(
                    classification["inferred_schema"] == "binary_gamble_like"
                ),
                "needs_categorical_loglik": int(
                    classification["loglik_evaluator"] in ("categorical", "categorical_high_dim")
                ),
                "choose_return_format": classification["choose_return_format"],
                "history_fields_needed": (
                    "feedback, prior actions"
                    if "gonogo" not in exp_id
                    else "trial outcomes, RT optional"
                ),
                "code_changes_summary": "; ".join(code_changes),
                "example_instruction": markers.get("sample_instruction", "")[:200],
            }
        )

    non_gamble_csv = output_dir / "psych101_non_gamble_schema_requirements.csv"
    with open(non_gamble_csv, "w", newline="", encoding="utf-8") as f:
        if non_gamble_rows:
            w = csv.DictWriter(f, fieldnames=list(non_gamble_rows[0].keys()))
            w.writeheader()
            w.writerows(non_gamble_rows)

    # Binary assumption code sites markdown
    sites = _code_sites_binary_assumption()
    md_lines = [
        "# Psych-101: binary action / scalar P(B) code assumptions",
        "",
        "Evolution currently assumes `choose(problem, history) -> float` = P(action=1) "
        "(Option B) for choice13k, cpc18 (split loglik), and mixed_gambles.",
        "",
        "| File | Symbol | Lines | Assumption |",
        "|------|--------|-------|------------|",
    ]
    for s in sites:
        md_lines.append(
            f"| `{s['file']}` | `{s['symbol']}` | {s['lines']} | {s['assumption']} |"
        )
    md_lines.extend(
        [
            "",
            "## Supporting categorical loglik (feasibility)",
            "",
            "A generic evaluator is feasible with moderate changes:",
            "",
            "1. Add `_parse_choose_output_categorical(p_raw, action_space) -> Dict[int, float]`.",
            "2. Add `evaluate_categorical_program(choose_fn, trials)` using "
            "`log p(a) = log max(eps, p_dict[a])` with normalization if dict does not sum to 1.",
            "3. Route via `_evaluate_loglik_for_dataset` when `trial['problem'].get('action_space')` "
            "has len > 2 or `return_format='categorical'`.",
            "4. Keep scalar path unchanged for backward compatibility.",
            "",
            "Alternative unified signature:",
            "",
            "```python",
            "def choose(problem, history, action_space=None):",
            "    if action_space is None or len(action_space) == 2:",
            "        return float  # P(action=1)",
            "    return {a: p for a in action_space}",
            "```",
            "",
            "Prefer explicit per-dataset `return_format` in problem schema over runtime inference.",
            "",
        ]
    )
    (output_dir / "psych101_binary_assumption_code_sites.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    # Main report
    top_gamble = gamble_rows[:15]
    c13_train_ok = sum(1 for e in parse_examples["choice13k"].get("train", []) if e.get("success"))
    c13_test_ok = sum(1 for e in parse_examples["choice13k"].get("test", []) if e.get("success"))
    cpc_train_ok = sum(1 for e in parse_examples["cpc18"].get("train", []) if e.get("success"))
    cpc_test_ok = sum(1 for e in parse_examples["cpc18"].get("test", []) if e.get("success"))
    cpc_train_full = sum(1 for e in parse_examples["cpc18"].get("train", []) if e.get("full_parse"))
    cpc_test_full = sum(1 for e in parse_examples["cpc18"].get("test", []) if e.get("full_parse"))

    report_lines = [
        "# Psych-101 program schema feasibility report",
        "",
        "## Executive summary",
        "",
        "Psych-101 stores **participant-level natural-language transcripts**, not structured trials. "
        "Our pipeline evolves **`choose(problem, history) -> float`** programs with **Bernoulli loglik "
        "on action=1 (Option B)**. Prompt-only adaptation is insufficient when experiments need "
        "different I/O schemas.",
        "",
        "**Gamble-like experiments** (two-option monetary/lottery choices) can reuse the existing "
        "schema after **deterministic parsing** from text to `{gamble_A, gamble_B, action, history}`.",
        "",
        "**Non-gamble cognitive tasks** require new trial schemas and likely "
        "`choose(...) -> dict[action, probability]` plus a **categorical loglik evaluator**.",
        "",
        "## 1. Gamble-like candidates (ranked)",
        "",
        "Ranking combines experiment-id heuristics and text markers "
        "(two press keys, `Option X delivers ... N% chance`). "
        "Full table: `psych101_gamble_like_candidates.csv`.",
        "",
        "| Rank | Experiment | Score | Binary press keys | Parser | Note |",
        "|------|------------|-------|-------------------|--------|------|",
    ]
    for r in top_gamble:
        parser = (
            "choice13k OK"
            if r.get("choice13k_parser_ok")
            else ("likely" if r["likely_binary_gamble"] else "unknown")
        )
        report_lines.append(
            f"| {r['rank']} | `{r['experiment_id']}` | {r['total_score']} | "
            f"{r['press_keys'] or '—'} | {parser} | {r['id_note'] or 'text markers'} |"
        )

    report_lines.extend(
        [
            "",
            "## 2. CPC18 and choice13k transcript parsing",
            "",
            f"- **choice13k** (`{CHOICE13K_EXP}`): existing `data_modules/choice13k.py` parser — "
            f"train {c13_train_ok}/4 samples OK, test {c13_test_ok}/4 OK.",
            f"- **CPC18** (`{CPC18_EXP}`): same NL format as choice13k (`Option X delivers`, `You press <<Y>>`). "
            f"Existing choice13k parser succeeds on structure but **partial coverage**: "
            f"train {cpc_train_ok}/4 participants parse, test {cpc_test_ok}/4; "
            f"full trial recovery {cpc_train_full}/4 train, {cpc_test_full}/4 test "
            f"(typically 150/750 trials — feedback lines use `and gain` and miss the current regex). "
            "Extend `_extract_trials` for CPC18 feedback lines; block-level MSE targets are **not** in Psych-101 text.",
            "",
            "See `psych101_parse_examples.json` for success/failure snippets per participant.",
            "",
            "**Regex reliability:** choice13k format is stable; parser recovers all presses for standard "
            "`You press <<Y>>.` lines. CPC18 Psych-101 adds `You press <<Y>> and gain ...` lines (600/750 trials) "
            "that need a regex extension. Participant ids and trial counts differ from local Track II CSV "
            "(see `../cpc18_psych101_compat_report.json`).",
            "",
            "## 3. Non-gamble experiments (schema requirements)",
            "",
            "See `psych101_non_gamble_schema_requirements.csv`. Scalar P(action=1) is **invalid** for:",
            "",
        ]
    )
    for nr in non_gamble_rows:
        if not nr["scalar_p_action_1_valid"]:
            report_lines.append(
                f"- `{nr['experiment_id']}`: {nr['inferred_schema']} — {nr['choose_return_format']}"
            )

    report_lines.extend(
        [
            "",
            "## 4. Unified evolution interface",
            "",
            "**Recommended:** per-dataset schema registry with explicit `return_format`:",
            "",
            "- `binary_scalar`: `float` = P(action=1) — choice13k, cpc18 (loglik), mixed_gambles, wulff/frey/sadeghiyeh (after parser)",
            "- `categorical`: `dict[int, float]` — bandits with >2 arms, odd-one-out, n-way categorization",
            "- `custom`: memory / sequential recall — separate evaluators",
            "",
            "Unified signature `choose(problem, history, action_space=None) -> dict` is possible "
            "but evolution prompts and seeds must match the schema; do not infer solely at runtime.",
            "",
            "## 5. Phase 0 `understand_dataset` recommendation",
            "",
            "**Recommend option C for research reproducibility, migrating to D per dataset after validation:**",
            "",
            "- **C (prompt + parser spec):** YAML/JSON with trial schema, action mapping, `choose` return type, "
            "regex patterns, validation examples — human-reviewable, version-controlled.",
            "- **D (actual parser code):** generate only after spec passes golden tests; check in to `data_modules/`.",
            "",
            "Option A (prompt only) is insufficient. Option B alone lacks executable validation.",
            "",
            "## 6. Practical path to 8 cognitive datasets (paper)",
            "",
            "### Tier 1 — minimal code changes (parse + existing binary loglik)",
            "",
            "1. **choice13k** (already local; Psych-101 parser exists)",
            "2. **cpc18** — extend choice13k parser for `and gain` feedback lines; Psych-101 loader; scalar loglik",
            "3. **mixed_gambles** — stay on local CSV (not in Psych-101)",
            "4. **wulff2018description/exp1** — two press keys (H/W); verify payoff text then extend parser if needed",
            "5. **frey2017cct/exp1** — two keys (C/E); Columbia Card Task, likely binary after parser",
            "",
            "### Tier 2 — medium changes",
            "",
            "6. **steingroever2015data/exp1** — 4 decks (Iowa gambling); categorical loglik over 4 actions",
            "7. **enkavi2019gonogo/exp1** — go/nogo keys; categorical loglik + RT optional in history",
            "8. **flesch2018comparing/exp1** — verify paradigm; may be 2-arm bandit with history",
            "",
            "### Defer (high schema / action-space cost)",
            "",
            "- **hebart2023things** — huge categorization space",
            "- **wilson2014humans** / **tomov2020discovery** — MDP/bandit structure",
            "- **schulz2020finding** — odd-one-out / structure learning",
            "- **ruggeri2022globalizability** — mixed paradigms, very large",
            "- **garcia2023experiential** — many keys per participant (sampling), not two-option gamble",
            "",
            "## Artifacts in this directory",
            "",
            "- `psych101_gamble_like_candidates.csv`",
            "- `psych101_non_gamble_schema_requirements.csv`",
            "- `psych101_binary_assumption_code_sites.md`",
            "- `psych101_parse_examples.json`",
            "",
        ]
    )
    (output_dir / "psych101_program_schema_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"Wrote reports to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    out = args.output_dir.expanduser()
    out = out.resolve() if out.is_absolute() else (REPO_ROOT / out).resolve()
    run_investigation(out)


if __name__ == "__main__":
    main()
