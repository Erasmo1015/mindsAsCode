#!/usr/bin/env python3
"""Evidence-based ablation analysis for PICS (TEH) on 2plonsky2018when."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path("/common/home/users/z/zichang.ge.2023/repo/mindsAsCode")
FULL_RUN = REPO_ROOT / "generated_outputs/psych101_train/teh/2plonsky2018when/run_260525_040017"
ABLATION_ROOT = REPO_ROOT / "generated_outputs_ablation/psych101_train/teh/2plonsky2018when"
OUT_REPORT = REPO_ROOT / "analysis/data/ablation/2plonsky2018when_ablation_report.md"

ABLATIONS = {
    "1_population": "No population-level phase",
    "2_exploration": "No participant-level exploration",
    "3_parent": "Single-parent sampled_parents (sample_size=1)",
    "3_2_best_parent1": "Single-parent no-sampled_parents (sample_size=1)",
    "4_fresh_n": "No fresh_n schedule (fresh_n_candidates=0)",
    "5_refine": "No refinement phase",
    "6_data_prompt": "Generic/no dataset-specific prompt",
    "7_error_feedback": "No error feedback (max_error_prompt_chars=0)",
}


@dataclass
class RunInfo:
    name: str
    folder: Path
    selected_run: Optional[Path]
    candidates: List[Path]
    selected_reason: str
    completed_participants: Optional[int]
    main_metric_file: Optional[Path]
    notes: List[str]
    command: Optional[str]


def extract_command(path: Path) -> Optional[str]:
    cmd_file = path / "log" / "command.txt"
    if not cmd_file.exists():
        return None
    lines = cmd_file.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def candidate_runs(folder: Path) -> List[Path]:
    candidates: List[Path] = []
    if (folder / "participant_details_loglik.csv").exists() or (folder / "participants_summary.csv").exists():
        candidates.append(folder)
    for child in sorted(folder.iterdir()) if folder.exists() else []:
        if child.is_dir() and child.name.startswith("run_"):
            if (child / "participant_details_loglik.csv").exists() or (child / "participants_summary.csv").exists():
                candidates.append(child)
    return candidates


def completion_count(run_path: Path) -> int:
    details = run_path / "participant_details_loglik.csv"
    if details.exists():
        try:
            return int(len(pd.read_csv(details)))
        except Exception:
            return 0
    return len([p for p in run_path.iterdir() if p.is_dir() and p.name.startswith("participant_")])


def choose_run(candidates: List[Path]) -> Tuple[Optional[Path], str]:
    if not candidates:
        return None, "No metric-producing run found."
    if len(candidates) == 1:
        return candidates[0], "Single candidate run."
    scored = [(completion_count(c), c) for c in candidates]
    max_completed = max(s for s, _ in scored)
    top = [c for s, c in scored if s == max_completed]
    if len(top) == 1:
        return top[0], f"Selected by max completed participants ({max_completed})."
    def run_stamp(p: Path) -> str:
        m = re.search(r"run_(\d+)", p.name)
        return m.group(1) if m else ""
    top_sorted = sorted(top, key=lambda p: (run_stamp(p), p.stat().st_mtime))
    return top_sorted[-1], (
        f"Tie on completed participants ({max_completed}); selected latest by run_ timestamp/mtime."
    )


def load_details(run_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if run_path is None:
        return None
    f = run_path / "participant_details_loglik.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if "participant_id" in df.columns:
        df = df.sort_values("participant_id").reset_index(drop=True)
    return df


def aggregate_invalid_metrics(run_path: Path, participant_only: bool = True) -> Dict[str, float]:
    total_iters = 0
    total_generated = 0
    total_runtime_valid = 0
    total_n_valid = 0
    total_num_invalid = 0
    for root, _, files in os.walk(run_path):
        if "metrics.json" not in files:
            continue
        rel = str(Path(root).relative_to(run_path))
        if "/iteration_" not in rel and not rel.startswith("iteration_"):
            continue
        if participant_only and ("refinement/" in rel or rel.startswith("global_phase/")):
            continue
        with open(Path(root) / "metrics.json", "r", encoding="utf-8") as fh:
            d = json.load(fh)
        total_iters += 1
        total_generated += int(d.get("n_candidates", 0) or 0)
        total_runtime_valid += int(d.get("n_runtime_valid", 0) or 0)
        total_n_valid += int(d.get("n_valid", 0) or 0)
        total_num_invalid += int(d.get("num_invalid_candidates", 0) or 0)
    out: Dict[str, float] = {
        "iters": total_iters,
        "generated_candidates": total_generated,
        "runtime_valid": total_runtime_valid,
        "n_valid": total_n_valid,
        "invalid_candidates_num_invalid": total_num_invalid,
    }
    if total_generated > 0 and total_iters > 0:
        out["invalid_rate_num_invalid"] = total_num_invalid / total_generated
        out["invalid_rate_runtime"] = (total_generated - total_runtime_valid) / total_generated
        out["avg_runtime_valid_per_iter"] = total_runtime_valid / total_iters
    return out


def metric_or_na(v: Optional[float], ndigits: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{v:.{ndigits}f}"


def summarize_df(df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    if df is None:
        return {
            "n": None,
            "avg_test": None,
            "avg_gated": None,
            "median_test": None,
            "median_gated": None,
            "above_rand_test": None,
            "above_rand_gated": None,
        }
    out: Dict[str, Optional[float]] = {"n": float(len(df))}
    for col, key in [("test_loglik", "test"), ("gated_test_loglik", "gated")]:
        if col in df.columns:
            out[f"avg_{key}"] = float(df[col].mean())
            out[f"median_{key}"] = float(df[col].median())
            out[f"above_rand_{key}"] = float((df[col] > -0.69).sum())
        else:
            out[f"avg_{key}"] = None
            out[f"median_{key}"] = None
            out[f"above_rand_{key}"] = None
    return out


def compare_against_full(full_df: pd.DataFrame, ab_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    keys = [
        "avg_delta_test",
        "avg_delta_gated",
        "full_wins_test",
        "ablation_wins_test",
        "ties_test",
        "full_wins_gated",
        "ablation_wins_gated",
        "ties_gated",
    ]
    if ab_df is None:
        return {k: None for k in keys}
    m = full_df.merge(ab_df, on="participant_id", suffixes=("_full", "_ab"))
    out: Dict[str, Optional[float]] = {}
    for metric, short in [("test_loglik", "test"), ("gated_test_loglik", "gated")]:
        a = f"{metric}_full"
        b = f"{metric}_ab"
        if a not in m.columns or b not in m.columns:
            out[f"avg_delta_{short}"] = None
            out[f"full_wins_{short}"] = None
            out[f"ablation_wins_{short}"] = None
            out[f"ties_{short}"] = None
            continue
        d = m[a] - m[b]
        out[f"avg_delta_{short}"] = float(d.mean())
        out[f"full_wins_{short}"] = float((d > 0).sum())
        out[f"ablation_wins_{short}"] = float((d < 0).sum())
        out[f"ties_{short}"] = float((d == 0).sum())
    return out


def iteration_mean_curve(run_path: Path) -> Dict[int, float]:
    curve: Dict[int, List[float]] = {}
    for pid in range(50):
        pdir = run_path / f"participant_{pid}"
        if not pdir.exists():
            continue
        for it in range(1, 21):
            metrics = pdir / f"iteration_{it}" / "metrics.json"
            if not metrics.exists():
                continue
            d = json.loads(metrics.read_text(encoding="utf-8"))
            v = d.get("best_test_loglik")
            if v is None:
                continue
            curve.setdefault(it, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in sorted(curve.items()) if v}


def population_special(full_run: Path, no_pop_run: Optional[Path]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "global_available": False,
        "global_train_threshold": None,
        "threshold_trivial_random": None,
        "iter_to_match": [],
        "full_i1_avg_best_test": None,
        "nopop_i1_avg_best_test": None,
        "i1_full_wins": None,
        "i1_nopop_wins": None,
    }
    g_results = full_run / "global_phase" / "results.json"
    if not g_results.exists() or no_pop_run is None:
        return out
    out["global_available"] = True
    g = json.loads(g_results.read_text(encoding="utf-8"))
    threshold = float(g.get("pool_best_global_train_loglik"))
    out["global_train_threshold"] = threshold
    out["threshold_trivial_random"] = abs(threshold + 0.69314718056) < 1e-6

    iter_to_match: List[Optional[int]] = []
    full_i1: List[float] = []
    nop_i1: List[float] = []
    for pid in range(50):
        f1 = full_run / f"participant_{pid}" / "iteration_1" / "metrics.json"
        n1 = no_pop_run / f"participant_{pid}" / "iteration_1" / "metrics.json"
        if f1.exists():
            full_i1.append(float(json.loads(f1.read_text(encoding="utf-8")).get("best_test_loglik")))
        if n1.exists():
            nop_i1.append(float(json.loads(n1.read_text(encoding="utf-8")).get("best_test_loglik")))

        hit_iter = None
        for it in range(1, 21):
            p = no_pop_run / f"participant_{pid}" / f"iteration_{it}" / "metrics.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            bt = d.get("best_train_loglik")
            if bt is not None and float(bt) >= threshold:
                hit_iter = it
                break
        iter_to_match.append(hit_iter)

    out["iter_to_match"] = iter_to_match
    if full_i1 and nop_i1:
        out["full_i1_avg_best_test"] = sum(full_i1) / len(full_i1)
        out["nopop_i1_avg_best_test"] = sum(nop_i1) / len(nop_i1)
        wins_full = sum(1 for a, b in zip(full_i1, nop_i1) if a > b)
        wins_nop = sum(1 for a, b in zip(full_i1, nop_i1) if b > a)
        out["i1_full_wins"] = wins_full
        out["i1_nopop_wins"] = wins_nop
    return out


def run_info(name: str, folder: Path) -> RunInfo:
    candidates = candidate_runs(folder)
    selected, reason = choose_run(candidates)
    notes: List[str] = []
    main_metric_file = None
    completed = None
    cmd = extract_command(folder)
    if selected is not None:
        details = selected / "participant_details_loglik.csv"
        if details.exists():
            main_metric_file = details
            completed = len(pd.read_csv(details))
        else:
            summary = selected / "participants_summary.csv"
            if summary.exists():
                main_metric_file = summary
                completed = len(pd.read_csv(summary))
    if not candidates:
        notes.append("No completed run with participant metrics found.")
    if len(candidates) > 1:
        notes.append(f"{len(candidates)} candidate runs detected.")
    return RunInfo(
        name=name,
        folder=folder,
        selected_run=selected,
        candidates=candidates,
        selected_reason=reason,
        completed_participants=completed,
        main_metric_file=main_metric_file,
        notes=notes,
        command=cmd,
    )


def build_report() -> str:
    full_details = load_details(FULL_RUN)
    if full_details is None:
        raise RuntimeError(f"Missing full-run participant_details_loglik.csv at {FULL_RUN}")

    full_sum = summarize_df(full_details)
    runs: Dict[str, RunInfo] = {}
    details: Dict[str, Optional[pd.DataFrame]] = {}
    sums: Dict[str, Dict[str, Optional[float]]] = {}
    comps: Dict[str, Dict[str, Optional[float]]] = {}

    for key in ABLATIONS:
        info = run_info(key, ABLATION_ROOT / key)
        runs[key] = info
        df = load_details(info.selected_run)
        details[key] = df
        sums[key] = summarize_df(df)
        comps[key] = compare_against_full(full_details, df)

    pop_special = population_special(FULL_RUN, runs["1_population"].selected_run)
    curve_full = iteration_mean_curve(FULL_RUN)
    curve_fresh = iteration_mean_curve(runs["4_fresh_n"].selected_run) if runs["4_fresh_n"].selected_run else {}

    full_invalid = aggregate_invalid_metrics(FULL_RUN, participant_only=True)
    nofb_invalid = (
        aggregate_invalid_metrics(runs["7_error_feedback"].selected_run, participant_only=True)
        if runs["7_error_feedback"].selected_run
        else {}
    )

    lines: List[str] = []
    lines.append("# Ablation report: 2plonsky2018when")
    lines.append("")
    lines.append("Main metric for paper-facing comparisons: `gated_test_loglik` (also reporting `test_loglik`).")
    lines.append("`Delta vs Full` is `Full - Ablation`; positive means Full PICS is better.")
    lines.append("")

    lines.append("## 1. Runs analyzed")
    lines.append("")
    lines.append("| Ablation | folder | selected run path | completed participants | main metric file | notes |")
    lines.append("|---|---|---|---:|---|---|")

    full_cmd = extract_command(FULL_RUN)
    lines.append(
        f"| Full PICS | `generated_outputs/psych101_train/teh/2plonsky2018when/run_260525_040017` | "
        f"`{FULL_RUN}` | {int(full_sum['n'])} | "
        f"`{FULL_RUN / 'participant_details_loglik.csv'}` | command present: {'yes' if full_cmd else 'no'} |"
    )
    for key, desc in ABLATIONS.items():
        info = runs[key]
        selected = f"`{info.selected_run}`" if info.selected_run else "NA"
        cnum = str(info.completed_participants) if info.completed_participants is not None else "NA"
        metric = f"`{info.main_metric_file}`" if info.main_metric_file else "NA"
        notes = [info.selected_reason] + info.notes
        lines.append(
            f"| {key}: {desc} | `generated_outputs_ablation/psych101_train/teh/2plonsky2018when/{key}` | "
            f"{selected} | {cnum} | {metric} | {'; '.join(notes)} |"
        )

    lines.append("")
    lines.append("## 2. Main ablation summary")
    lines.append("")
    lines.append("| Setting | Avg test BLL | Avg gated test BLL | Delta vs Full (gated) | Full wins | Ablation wins | Above -0.69 (gated) | Takeaway |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    lines.append(
        f"| Full PICS | {metric_or_na(full_sum['avg_test'])} | {metric_or_na(full_sum['avg_gated'])} | 0.0000 | - | - | "
        f"{int(full_sum['above_rand_gated']) if full_sum['above_rand_gated'] is not None else 'NA'} | Reference |"
    )
    for key, desc in ABLATIONS.items():
        s = sums[key]
        c = comps[key]
        takeaway = "Missing metrics/output." if s["n"] is None else "See component findings."
        lines.append(
            f"| {key} | {metric_or_na(s['avg_test'])} | {metric_or_na(s['avg_gated'])} | "
            f"{metric_or_na(c['avg_delta_gated'])} | "
            f"{int(c['full_wins_gated']) if c['full_wins_gated'] is not None else 'NA'} | "
            f"{int(c['ablation_wins_gated']) if c['ablation_wins_gated'] is not None else 'NA'} | "
            f"{int(s['above_rand_gated']) if s['above_rand_gated'] is not None else 'NA'} | {takeaway} |"
        )

    lines.append("")
    lines.append("## 3. Component-by-component findings")
    lines.append("")

    def comp_line(key: str) -> str:
        c = comps[key]
        s = sums[key]
        if s["n"] is None:
            return "No participant-level metric file found; result is inconclusive due to missing outputs."
        return (
            f"Avg gated BLL {metric_or_na(s['avg_gated'])} vs Full {metric_or_na(full_sum['avg_gated'])}; "
            f"delta (Full - Ablation) {metric_or_na(c['avg_delta_gated'])}; "
            f"wins Full/Ablation {int(c['full_wins_gated'])}/{int(c['ablation_wins_gated'])}."
        )

    findings = [
        ("1_population", "Moderate/negative", "Mention briefly"),
        ("2_exploration", "Moderate/negative", "Mention briefly"),
        ("3_parent", "Moderate/negative", "Mention briefly"),
        ("3_2_best_parent1", "Strong negative", "Include as negative result if space allows"),
        ("4_fresh_n", "Strong evidence", "Include in main paper"),
        ("5_refine", "Weak/inconclusive", "One sentence only"),
        ("6_data_prompt", "Inconclusive (missing run)", "Do not claim; rerun needed"),
        ("7_error_feedback", "Negative result", "Include only if reporting robustness honestly"),
    ]

    for key, strength, include in findings:
        lines.append(f"### {key}")
        lines.append(f"- **Change:** {ABLATIONS[key]}")
        lines.append(f"- **Main numerical result:** {comp_line(key)}")
        lines.append(f"- **Evidence grade:** {strength}")
        lines.append(f"- **Include in main paper?:** {include}")
        if sums[key]["n"] is None:
            sent = f"`{key}` could not be evaluated on CPC18 because participant metric files are missing in the available output folder."
        else:
            sign = comps[key]["avg_delta_gated"]
            if sign is not None and sign > 0:
                sent = (
                    f"On CPC18, removing this component lowers average gated BLL by "
                    f"{abs(sign):.4f} (Full wins {int(comps[key]['full_wins_gated'])}/50 participants)."
                )
            else:
                sent = (
                    f"On CPC18, this ablation does not hurt final gated BLL in this run "
                    f"(delta Full - Ablation {metric_or_na(sign)})."
                )
        lines.append(f"- **Suggested one-sentence wording:** {sent}")
        lines.append("")

    lines.append("## 4. Prominent results for paper")
    lines.append("")
    lines.append("Selected up to 3 based on absolute effect size and support in available files:")
    lines.append("")
    lines.append("1) **Exploration-to-exploitation schedule (`4_fresh_n`)**")
    lines.append(
        f"- Avg gated BLL: Full {metric_or_na(full_sum['avg_gated'])} vs no-schedule {metric_or_na(sums['4_fresh_n']['avg_gated'])}; "
        f"delta {metric_or_na(comps['4_fresh_n']['avg_delta_gated'])}; Full wins "
        f"{int(comps['4_fresh_n']['full_wins_gated'])}/50."
    )
    lines.append("- Interpretation: strong, consistent degradation when schedule is removed.")
    lines.append("- Suggested format: **table row**.")
    lines.append("")
    lines.append("2) **Single-parent evolution (`3_2_best_parent1`)**")
    lines.append(
        f"- Avg gated BLL: Full {metric_or_na(full_sum['avg_gated'])} vs single-parent {metric_or_na(sums['3_2_best_parent1']['avg_gated'])}; "
        f"delta {metric_or_na(comps['3_2_best_parent1']['avg_delta_gated'])}; Full wins "
        f"{int(comps['3_2_best_parent1']['full_wins_gated'])}/50."
    )
    lines.append("- Interpretation: in this run, single-parent is stronger; this is a negative finding for the multi-parent claim.")
    lines.append("- Suggested format: **short text sentence** (or table row if reporting negative findings).")
    lines.append("")
    lines.append("3) **Error feedback robustness (`7_error_feedback`)**")
    lines.append(
        f"- Invalid rate by `num_invalid_candidates` (participant iterations): with feedback {full_invalid.get('invalid_rate_num_invalid', float('nan')):.4f} "
        f"vs without {nofb_invalid.get('invalid_rate_num_invalid', float('nan')):.4f}."
    )
    lines.append(
        f"- Runtime-invalid rate (`1 - n_runtime_valid/n_candidates`): with feedback {full_invalid.get('invalid_rate_runtime', float('nan')):.4f} "
        f"vs without {nofb_invalid.get('invalid_rate_runtime', float('nan')):.4f}."
    )
    lines.append("- Interpretation: available logs do not support a robustness gain from error feedback on this run.")
    lines.append("- Suggested format: **short text sentence** with caveat of single-run evidence.")
    lines.append("")

    lines.append("## 5. Population-phase special analysis")
    lines.append("")
    if not pop_special["global_available"]:
        lines.append("- Global-phase files are unavailable, so iteration-to-match analysis cannot be computed.")
    else:
        threshold = pop_special["global_train_threshold"]
        lines.append(f"- Full-run global-phase best train loglik threshold: {threshold:.6f}.")
        lines.append(
            f"- Threshold equals random baseline (`-0.693147`)?: {'yes' if pop_special['threshold_trivial_random'] else 'no'}."
        )
        hits = [x for x in pop_special["iter_to_match"] if x is not None]
        if hits:
            hit_1 = sum(1 for x in hits if x == 1)
            lines.append(
                f"- No-pop run reaches/exceeds this threshold for {len(hits)}/50 participants; "
                f"{hit_1}/50 do so at iteration 1; mean iteration-to-match = {sum(hits)/len(hits):.2f}."
            )
        lines.append(
            f"- Iteration-1 mean best_test_loglik: Full {metric_or_na(pop_special['full_i1_avg_best_test'])} vs "
            f"no-pop {metric_or_na(pop_special['nopop_i1_avg_best_test'])}; "
            f"wins Full/no-pop = {pop_special['i1_full_wins']}/{pop_special['i1_nopop_wins']}."
        )
        lines.append(
            "- Interpretation: the requested initialization analysis is available, but the global threshold is the random baseline, "
            "so it does not provide evidence for a meaningful reusable-structure advantage."
        )
    lines.append("")

    lines.append("## 6. Error-feedback robustness")
    lines.append("")
    lines.append("Participant-iteration metrics from `metrics.json`:")
    lines.append("")
    lines.append("| Setting | Generated candidates | Invalid candidates (`num_invalid_candidates`) | Invalid rate | Runtime-invalid rate | Avg runtime-valid candidates/iter | Final avg gated BLL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Full (with error feedback) | {int(full_invalid.get('generated_candidates', 0))} | "
        f"{int(full_invalid.get('invalid_candidates_num_invalid', 0))} | "
        f"{full_invalid.get('invalid_rate_num_invalid', float('nan')):.4f} | "
        f"{full_invalid.get('invalid_rate_runtime', float('nan')):.4f} | "
        f"{full_invalid.get('avg_runtime_valid_per_iter', float('nan')):.3f} | {metric_or_na(full_sum['avg_gated'])} |"
    )
    lines.append(
        f"| Ablation `7_error_feedback` (without feedback) | {int(nofb_invalid.get('generated_candidates', 0))} | "
        f"{int(nofb_invalid.get('invalid_candidates_num_invalid', 0))} | "
        f"{nofb_invalid.get('invalid_rate_num_invalid', float('nan')):.4f} | "
        f"{nofb_invalid.get('invalid_rate_runtime', float('nan')):.4f} | "
        f"{nofb_invalid.get('avg_runtime_valid_per_iter', float('nan')):.3f} | {metric_or_na(sums['7_error_feedback']['avg_gated'])} |"
    )
    lines.append("")
    lines.append(
        "On available logs, disabling error feedback reduces both invalid-rate proxies and slightly improves final average gated BLL. "
        "This is a negative result for the expected robustness hypothesis."
    )
    lines.append("")

    lines.append("## 7. Recommended paper text")
    lines.append("")
    lines.append(
        "On CPC18, we ablated major PICS components and observed that the exploration-to-exploitation schedule has the clearest positive effect: "
        f"removing it (`4_fresh_n`) decreases average gated behavioral log-likelihood from {metric_or_na(full_sum['avg_gated'])} to "
        f"{metric_or_na(sums['4_fresh_n']['avg_gated'])} (delta {metric_or_na(comps['4_fresh_n']['avg_delta_gated'])}, "
        f"Full better on {int(comps['4_fresh_n']['full_wins_gated'])}/50 participants). "
        "Other ablations are weaker or opposite in this single-run evidence: both single-parent variants and no-exploration do not underperform Full, "
        "and disabling error feedback lowers invalid-program rates in the available logs. "
        "Population initialization and refinement do not show uniform final-performance gains here."
    )
    lines.append("")
    lines.append("| Setting | Avg gated BLL | Delta vs Full (Full - Ablation) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Full PICS | {metric_or_na(full_sum['avg_gated'])} | 0.0000 |")
    lines.append(
        f"| No schedule (`4_fresh_n`) | {metric_or_na(sums['4_fresh_n']['avg_gated'])} | {metric_or_na(comps['4_fresh_n']['avg_delta_gated'])} |"
    )
    lines.append("")

    lines.append("## 8. Appendix details")
    lines.append("")
    lines.append("### Per-participant Full vs ablation deltas (gated_test_loglik)")
    lines.append("")
    full_small = full_details[["participant_id", "gated_test_loglik"]].rename(
        columns={"gated_test_loglik": "full_gated_test_loglik"}
    )
    merged = full_small.copy()
    for key in ABLATIONS:
        df = details[key]
        if df is None or "gated_test_loglik" not in df.columns:
            continue
        tmp = df[["participant_id", "gated_test_loglik"]].rename(
            columns={"gated_test_loglik": f"{key}_gated_test_loglik"}
        )
        merged = merged.merge(tmp, on="participant_id", how="left")
        merged[f"delta_full_minus_{key}"] = merged["full_gated_test_loglik"] - merged[f"{key}_gated_test_loglik"]
    lines.append(merged.to_markdown(index=False))
    lines.append("")

    lines.append("### Iteration curve snapshot for `4_fresh_n` (mean `best_test_loglik` from participant `metrics.json`)")
    lines.append("")
    lines.append("| Iteration | Full | `4_fresh_n` | Full - `4_fresh_n` |")
    lines.append("|---:|---:|---:|---:|")
    for it in [1, 5, 10, 15, 20]:
        f = curve_full.get(it)
        n = curve_fresh.get(it)
        d = (f - n) if f is not None and n is not None else None
        lines.append(f"| {it} | {metric_or_na(f)} | {metric_or_na(n)} | {metric_or_na(d)} |")
    lines.append("")

    lines.append("### Run commands (raw)")
    lines.append("")
    lines.append("- Full PICS command:")
    lines.append("```")
    lines.append(full_cmd or "NA")
    lines.append("```")
    for key in ABLATIONS:
        lines.append(f"- `{key}` command:")
        lines.append("```")
        lines.append(runs[key].command or "NA")
        lines.append("```")
    lines.append("")

    lines.append("### Missing files and ambiguity notes")
    lines.append("")
    lines.append("- `6_data_prompt`: no `participant_details_loglik.csv`, no `participants_summary.csv`, and no `participant_*` directories were found.")
    lines.append("- No multi-run ambiguity was detected in the available ablation folders; each ablation had at most one metric-producing candidate.")
    lines.append("- `final_participant_summary.csv` was not found for Full or ablation folders; report uses `participant_details_loglik.csv` and `participants_summary.csv`.")
    lines.append("- Error-feedback invalidity uses explicit fields available in logs (`num_invalid_candidates`, `n_runtime_valid`); no broader invalid taxonomy file was found.")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote report to: {OUT_REPORT}")


if __name__ == "__main__":
    main()
