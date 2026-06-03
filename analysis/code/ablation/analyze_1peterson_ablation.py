#!/usr/bin/env python3
"""Evidence-based ablation report for 1peterson2021using (choices13k)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


REPO_ROOT = Path("/common/home/users/z/zichang.ge.2023/repo/mindsAsCode")
FULL_ROOT = REPO_ROOT / "generated_outputs/psych101_train/teh/1peterson2021using"
ABL_ROOT = REPO_ROOT / "generated_outputs_ablation/psych101_train/teh/1peterson2021using"

SELECTED_FULL = FULL_ROOT / "run_260525_031227"
ABLATIONS = {
    "population": "No population-level phase",
    "2_exploration": "No participant-level exploration",
}

OUT_REPORT = REPO_ROOT / "analysis/data/ablation/1peterson2021using_ablation_report.md"


def first_command(run_path: Path) -> Optional[str]:
    cmd = run_path / "log/command.txt"
    if not cmd.exists():
        return None
    for line in cmd.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "n": float(len(df)),
        "avg_test": float(df["test_loglik"].mean()),
        "avg_gated": float(df["gated_test_loglik"].mean()),
        "median_test": float(df["test_loglik"].median()),
        "median_gated": float(df["gated_test_loglik"].median()),
        "above_rand_test": float((df["test_loglik"] > -0.69).sum()),
        "above_rand_gated": float((df["gated_test_loglik"] > -0.69).sum()),
    }


def compare(full_df: pd.DataFrame, ab_df: pd.DataFrame) -> Dict[str, float]:
    m = full_df.merge(ab_df, on="participant_id", suffixes=("_full", "_ab"))
    d_test = m["test_loglik_full"] - m["test_loglik_ab"]
    d_gated = m["gated_test_loglik_full"] - m["gated_test_loglik_ab"]
    return {
        "avg_delta_test": float(d_test.mean()),
        "avg_delta_gated": float(d_gated.mean()),
        "full_wins_test": float((d_test > 0).sum()),
        "ab_wins_test": float((d_test < 0).sum()),
        "full_wins_gated": float((d_gated > 0).sum()),
        "ab_wins_gated": float((d_gated < 0).sum()),
    }


def fmt(v: Optional[float], n: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{v:.{n}f}"


def main() -> None:
    if not SELECTED_FULL.exists():
        raise RuntimeError(f"Missing selected full run: {SELECTED_FULL}")

    # Gather complete full-run candidates to document ambiguity.
    full_candidates: List[Path] = []
    for child in sorted(FULL_ROOT.iterdir()):
        details = child / "participant_details_loglik.csv"
        if child.is_dir() and child.name.startswith("run_") and details.exists():
            try:
                n = len(pd.read_csv(details))
            except Exception:
                n = -1
            if n == 50:
                full_candidates.append(child)

    full_df = pd.read_csv(SELECTED_FULL / "participant_details_loglik.csv")
    full_sum = summarize(full_df)

    ab_data = {}
    for key, desc in ABLATIONS.items():
        run_path = ABL_ROOT / key
        details_path = run_path / "participant_details_loglik.csv"
        if details_path.exists():
            df = pd.read_csv(details_path)
            ab_data[key] = {
                "desc": desc,
                "path": run_path,
                "df": df,
                "sum": summarize(df),
                "cmp": compare(full_df, df),
                "cmd": first_command(run_path),
            }
        else:
            ab_data[key] = {
                "desc": desc,
                "path": run_path,
                "df": None,
                "sum": None,
                "cmp": None,
                "cmd": first_command(run_path),
            }

    lines: List[str] = []
    lines.append("# Ablation report: 1peterson2021using")
    lines.append("")
    lines.append("Main metric for paper-facing comparison: `gated_test_loglik` (higher is better).")
    lines.append("`Delta vs Full` is `Full - Ablation`; positive means Full PICS is better.")
    lines.append("")
    lines.append("## 1. Runs analyzed")
    lines.append("")
    lines.append("| Ablation | folder | selected run path | completed participants | main metric file | notes |")
    lines.append("|---|---|---|---:|---|---|")
    lines.append(
        f"| Full PICS | `generated_outputs/psych101_train/teh/1peterson2021using` | `{SELECTED_FULL}` | "
        f"{int(full_sum['n'])} | `{SELECTED_FULL / 'participant_details_loglik.csv'}` | "
        f"{len(full_candidates)} complete full runs found; selected latest complete run by timestamp (`run_260525_031227`). |"
    )
    for key, item in ab_data.items():
        if item["sum"] is None:
            lines.append(
                f"| {key}: {item['desc']} | `generated_outputs_ablation/psych101_train/teh/1peterson2021using/{key}` | "
                f"`{item['path']}` | NA | NA | missing participant_details_loglik.csv |"
            )
        else:
            lines.append(
                f"| {key}: {item['desc']} | `generated_outputs_ablation/psych101_train/teh/1peterson2021using/{key}` | "
                f"`{item['path']}` | {int(item['sum']['n'])} | `{item['path'] / 'participant_details_loglik.csv'}` | "
                f"completed ablation run |"
            )

    lines.append("")
    lines.append("## 2. Main ablation summary")
    lines.append("")
    lines.append("| Setting | Avg test BLL | Avg gated test BLL | Delta vs Full (gated) | Full wins | Ablation wins | Above -0.69 (gated) | Takeaway |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    lines.append(
        f"| Full PICS (`run_260525_031227`) | {fmt(full_sum['avg_test'])} | {fmt(full_sum['avg_gated'])} | 0.0000 | - | - | "
        f"{int(full_sum['above_rand_gated'])} | Reference |"
    )
    for key, item in ab_data.items():
        if item["sum"] is None:
            lines.append(f"| {key} | NA | NA | NA | NA | NA | NA | Missing output |")
            continue
        s = item["sum"]
        c = item["cmp"]
        lines.append(
            f"| {key} | {fmt(s['avg_test'])} | {fmt(s['avg_gated'])} | {fmt(c['avg_delta_gated'])} | "
            f"{int(c['full_wins_gated'])} | {int(c['ab_wins_gated'])} | {int(s['above_rand_gated'])} | "
            "Ablation outperforms Full in this run. |"
        )

    lines.append("")
    lines.append("## 3. Component-by-component findings")
    lines.append("")
    for key, item in ab_data.items():
        lines.append(f"### {key}")
        lines.append(f"- **Change:** {item['desc']}")
        if item["sum"] is None:
            lines.append("- **Main numerical result:** Missing participant-level metrics; inconclusive.")
            lines.append("- **Evidence grade:** Inconclusive")
            lines.append("- **Include in main paper?:** No (rerun needed)")
            lines.append(
                f"- **Suggested one-sentence wording:** `{key}` for CPC18 could not be evaluated from available outputs."
            )
            lines.append("")
            continue
        s = item["sum"]
        c = item["cmp"]
        lines.append(
            f"- **Main numerical result:** Avg gated BLL {fmt(s['avg_gated'])} vs Full {fmt(full_sum['avg_gated'])}; "
            f"delta {fmt(c['avg_delta_gated'])}; wins Full/Ablation {int(c['full_wins_gated'])}/{int(c['ab_wins_gated'])}."
        )
        lines.append("- **Evidence grade:** Strong negative for the expected Full-PICS advantage")
        lines.append("- **Include in main paper?:** Mention briefly as negative/inconclusive against hypothesis")
        lines.append(
            f"- **Suggested one-sentence wording:** On 1peterson2021using, removing this component does not reduce final gated BLL in this run (delta Full - Ablation {fmt(c['avg_delta_gated'])})."
        )
        lines.append("")

    lines.append("## 4. Prominent results for paper")
    lines.append("")
    lines.append("Only two completed ablations are available; both are negative results relative to the selected Full run:")
    for key, item in ab_data.items():
        if item["sum"] is None:
            continue
        c = item["cmp"]
        s = item["sum"]
        lines.append(
            f"- `{key}`: Full gated BLL {fmt(full_sum['avg_gated'])} vs ablation {fmt(s['avg_gated'])}; "
            f"delta {fmt(c['avg_delta_gated'])}; participant wins Full/Ablation "
            f"{int(c['full_wins_gated'])}/{int(c['ab_wins_gated'])}."
        )
    lines.append("")

    lines.append("## 5. Recommended paper text")
    lines.append("")
    lines.append(
        "On 1peterson2021using (choices13k), only two ablations have complete outputs in the current logs (no population phase and no participant exploration). "
        f"Against the selected Full PICS reference run (`run_260525_031227`), both ablations achieve higher average gated behavioral log-likelihood "
        f"(population: {fmt(ab_data['population']['sum']['avg_gated'])}, exploration: {fmt(ab_data['2_exploration']['sum']['avg_gated'])}, "
        f"Full: {fmt(full_sum['avg_gated'])}). "
        "Therefore, this dataset currently does not provide supportive ablation evidence for these two components; we recommend reporting this as inconclusive/negative single-run evidence rather than as a main positive claim."
    )
    lines.append("")

    lines.append("## 6. Appendix details")
    lines.append("")
    lines.append("### Per-participant Full vs ablation deltas (gated_test_loglik)")
    lines.append("")
    merged = full_df[["participant_id", "gated_test_loglik"]].rename(
        columns={"gated_test_loglik": "full_gated_test_loglik"}
    )
    for key, item in ab_data.items():
        if item["df"] is None:
            continue
        tmp = item["df"][["participant_id", "gated_test_loglik"]].rename(
            columns={"gated_test_loglik": f"{key}_gated_test_loglik"}
        )
        merged = merged.merge(tmp, on="participant_id", how="left")
        merged[f"delta_full_minus_{key}"] = merged["full_gated_test_loglik"] - merged[f"{key}_gated_test_loglik"]
    lines.append(merged.to_markdown(index=False))
    lines.append("")

    lines.append("### Run commands")
    lines.append("")
    lines.append("- Full command:")
    lines.append("```")
    lines.append(first_command(SELECTED_FULL) or "NA")
    lines.append("```")
    for key, item in ab_data.items():
        lines.append(f"- `{key}` command:")
        lines.append("```")
        lines.append(item["cmd"] or "NA")
        lines.append("```")
    lines.append("")

    lines.append("### Missing files / ambiguity notes")
    lines.append("")
    lines.append(
        f"- Full-run ambiguity: {len(full_candidates)} complete full runs were found under `generated_outputs/psych101_train/teh/1peterson2021using`."
    )
    lines.append("- Selection rule used here: latest complete run by timestamp (`run_260525_031227`).")
    lines.append("- Only `population` and `2_exploration` were analyzed per user request.")
    lines.append("- `final_participant_summary.csv` not found in analyzed runs; metrics are from `participant_details_loglik.csv`.")
    lines.append("")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to: {OUT_REPORT}")


if __name__ == "__main__":
    main()
