#!/usr/bin/env python3
"""Build proposal-ready evidence table for Choices13k participants 2 and 4."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_trials(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _problem_key(trial: Dict[str, Any]) -> str:
    return json.dumps(trial["problem"], sort_keys=True)


def split_blocks(trials: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for t in trials:
        k = _problem_key(t)
        if k not in grouped:
            grouped[k] = []
            order.append(k)
        grouped[k].append(t)
    return [grouped[k] for k in order]


def safe_rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom > 0 else float("nan")


def compute_stats(trials: List[Dict[str, Any]]) -> Dict[str, float]:
    blocks = split_blocks(trials)
    repeats = 0
    switches = 0
    total_pairs = 0
    neg_fb_repeats = 0
    neg_fb_total = 0
    no_fb_repeats = 0
    no_fb_pairs = 0

    for block in blocks:
        if len(block) < 2:
            continue
        has_feedback = bool(block[0]["problem"].get("has_feedback", False))
        for i in range(1, len(block)):
            prev = block[i - 1]
            cur = block[i]
            prev_a = int(prev["action"])
            cur_a = int(cur["action"])
            total_pairs += 1
            if cur_a == prev_a:
                repeats += 1
            else:
                switches += 1

            if not has_feedback:
                no_fb_pairs += 1
                if cur_a == prev_a:
                    no_fb_repeats += 1

            prev_hist = cur.get("history", [])
            if prev_hist:
                prev_feedback = prev_hist[-1].get("feedback")
                if prev_feedback is not None and float(prev_feedback) < 0:
                    neg_fb_total += 1
                    if cur_a == prev_a:
                        neg_fb_repeats += 1

    return {
        "within_problem_repeat_rate": safe_rate(repeats, total_pairs),
        "switch_count": float(switches),
        "after_negative_feedback_repeat_rate": safe_rate(neg_fb_repeats, neg_fb_total),
        "no_feedback_repeat_rate": safe_rate(no_fb_repeats, no_fb_pairs),
    }


def fmt_rate(v: float) -> str:
    return "N/A" if v != v else f"{v:.2f}"


def build_observed_text(tr: Dict[str, float], te: Dict[str, float]) -> str:
    text = (
        f"train repeat={fmt_rate(tr['within_problem_repeat_rate'])}, "
        f"test repeat={fmt_rate(te['within_problem_repeat_rate'])}; "
        f"test switches={int(te['switch_count'])}; "
        f"repeat after neg fb (test)={fmt_rate(te['after_negative_feedback_repeat_rate'])}; "
        f"no-fb repeat (test)={fmt_rate(te['no_feedback_repeat_rate'])}"
    )
    return textwrap.fill(text, width=44)


def wrap(text: str, width: int) -> str:
    return textwrap.fill(text, width=width)


def draw_table(rows: List[Dict[str, str]], out_png: Path, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")
    ax.set_facecolor("white")

    headers = [
        "Participant",
        "Held-out gain over Centaur",
        "Observed behavior evidence",
        "Synthesized program mechanism",
        "Interpretation",
    ]
    col_x = [0.01, 0.13, 0.30, 0.58, 0.80]
    col_w = [0.11, 0.16, 0.27, 0.21, 0.19]
    y_top = 0.86
    row_h = 0.26

    ax.text(
        0.01,
        0.97,
        "Choices13k: evidence for participant-specific synthesized programs",
        fontsize=14,
        weight="bold",
        va="top",
    )

    for i, h in enumerate(headers):
        ax.add_patch(Rectangle((col_x[i], y_top), col_w[i], 0.09, facecolor="#eeeeee", edgecolor="#cccccc", lw=0.8))
        ax.text(col_x[i] + 0.006, y_top + 0.045, h, fontsize=10.5, weight="bold", va="center")

    for r, row in enumerate(rows):
        y = y_top - (r + 1) * row_h
        face = "#f9f9f9" if r % 2 == 0 else "#ffffff"
        for i in range(len(headers)):
            ax.add_patch(Rectangle((col_x[i], y), col_w[i], row_h, facecolor=face, edgecolor="#dddddd", lw=0.6))
        ax.text(col_x[0] + 0.006, y + row_h - 0.03, row["participant"], fontsize=10.5, weight="bold", va="top")
        ax.text(col_x[1] + 0.006, y + row_h - 0.03, row["gain"], fontsize=12, weight="bold", va="top")
        ax.text(col_x[2] + 0.006, y + row_h - 0.03, row["observed"], fontsize=9.5, va="top")
        ax.text(col_x[3] + 0.006, y + row_h - 0.03, row["mechanism"], fontsize=9.5, va="top")
        ax.text(col_x[4] + 0.006, y + row_h - 0.03, row["interpretation"], fontsize=9.5, va="top")

    ax.text(
        0.01,
        0.02,
        "Repeat statistics are computed within repeated trials of the same problem.",
        fontsize=9,
        color="#444444",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Choices13k participant evidence table figure.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/analysis_plot/proposal"),
    )
    parser.add_argument("--save_pdf", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    candidates = [
        root / "analysis" / "code" / "choices13k",
        root / "analysis" / "data" / "choices13k",
    ]

    def _pick(name: str) -> Path:
        for d in candidates:
            p = d / name
            if p.exists():
                return p
        raise FileNotFoundError(f"Missing required input JSON: {name}")

    p2_train = load_trials(_pick("participant_2_train_trials.json"))
    p2_test = load_trials(_pick("participant_2_test_trials.json"))
    p4_train = load_trials(_pick("participant_4_train_trials.json"))
    p4_test = load_trials(_pick("participant_4_test_trials.json"))

    p2_tr_stats = compute_stats(p2_train)
    p2_te_stats = compute_stats(p2_test)
    p4_tr_stats = compute_stats(p4_train)
    p4_te_stats = compute_stats(p4_test)

    print("[Participant 2 stats]")
    print(json.dumps({"train": p2_tr_stats, "test": p2_te_stats}, indent=2))
    print("[Participant 4 stats]")
    print(json.dumps({"train": p4_tr_stats, "test": p4_te_stats}, indent=2))

    rows = [
        {
            "participant": "Participant 2",
            "gain": "+0.70",
            "observed": build_observed_text(p2_tr_stats, p2_te_stats),
            "mechanism": wrap(
                "Recent-action terms: last-5 majority ±0.3; last action ±0.2; feedback only ±0.15",
                34,
            ),
            "interpretation": wrap(
                "Strong within-problem persistence; recent actions can dominate EV.",
                30,
            ),
        },
        {
            "participant": "Participant 4",
            "gain": "+0.14",
            "observed": build_observed_text(p4_tr_stats, p4_te_stats),
            "mechanism": wrap(
                "Uses past action counts twice; no feedback variable used.",
                34,
            ),
            "interpretation": wrap(
                "Self-reinforcing choice habit; repeats choices even after bad outcomes.",
                30,
            ),
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.output_dir / "choices13k_program_evidence_table.png"
    draw_table(rows=rows, out_png=out_png, save_pdf=args.save_pdf)
    print(f"Saved: {out_png}")
    if args.save_pdf:
        print(f"Saved: {out_png.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
