#!/usr/bin/env python3
"""Create proposal-ready Choices13k comparison figures (ours vs Centaur)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OURS_DIR = "generated_outputs/choice13k/non_strict/run_260430_013702"
DEFAULT_CENTAUR_DIR = "generated_outputs/choice13k/centaur/run_260416_000815"
DEFAULT_OUT_DIR = "analysis/analysis_plot/proposal"


def _to_int(value: str) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_float(value: str) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _score_csv_path(path: Path) -> int:
    name = path.name.lower()
    score = 0
    if "participant_details_loglik.csv" in name:
        score += 100
    if "participants_summary.csv" in name:
        score += 90
    if "summary.csv" == name:
        score += 70
    if "participant" in name:
        score += 20
    if "loglik" in name:
        score += 20
    return score


def _discover_candidate_csvs(run_dir: Path) -> List[Path]:
    csv_paths = [p for p in run_dir.rglob("*.csv") if p.is_file()]
    csv_paths.sort(key=lambda p: (_score_csv_path(p), str(p)), reverse=True)
    return csv_paths


def _load_participant_loglik_from_csv(path: Path) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    rows: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows
        fields = {field.strip(): field for field in reader.fieldnames if field}
        pid_col = fields.get("participant_id")
        train_col = fields.get("train_loglik")
        test_col = fields.get("test_loglik")
        if pid_col is None or train_col is None or test_col is None:
            return rows
        for row in reader:
            pid = _to_int(row.get(pid_col, ""))
            if pid is None:
                continue
            rows[pid] = (_to_float(row.get(train_col, "")), _to_float(row.get(test_col, "")))
    return rows


def load_participant_loglik_data(run_dir: Path, method_name: str) -> Tuple[Dict[int, Tuple[Optional[float], Optional[float]]], Path]:
    if not run_dir.exists():
        raise FileNotFoundError(f"{method_name}: run directory does not exist: {run_dir}")
    candidates = _discover_candidate_csvs(run_dir)
    if not candidates:
        raise FileNotFoundError(f"{method_name}: no CSV files found under {run_dir}")
    for csv_path in candidates:
        rows = _load_participant_loglik_from_csv(csv_path)
        if rows:
            return rows, csv_path
    raise ValueError(
        f"{method_name}: could not find a CSV with required columns "
        f"'participant_id', 'train_loglik', 'test_loglik' under {run_dir}"
    )


def _save_fig(fig: plt.Figure, out_base: Path, save_pdf: bool) -> None:
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_delta_from_centaur_plot(
    participant_ids: np.ndarray,
    improvements: np.ndarray,
    n: int,
    ours_better_count: int,
    mean_improvement: float,
    median_improvement: float,
    out_dir: Path,
    save_pdf: bool,
) -> None:
    order = np.argsort(-improvements)
    pids = participant_ids[order].astype(str)
    deltas = improvements[order]
    colors = np.where(deltas > 0, "#4caf50", "#e57373")

    y = np.arange(len(deltas))
    fig, ax = plt.subplots(figsize=(11.0, max(5.5, 0.42 * n + 2.0)))
    ax.barh(y, deltas, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(pids)
    ax.invert_yaxis()
    ax.set_xlabel("Δ held-out log-likelihood (Ours − Centaur)")
    ax.set_ylabel("participant_id")
    ax.set_title("Choices13k: improvement over Centaur by participant")
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    ax.grid(axis="x", alpha=0.25)

    x_range = max(abs(float(np.min(deltas))), abs(float(np.max(deltas))), 0.05)
    x_pad = 0.03 * x_range
    for i, delta in enumerate(deltas):
        x_text = delta + x_pad if delta >= 0 else delta - x_pad
        ha = "left" if delta >= 0 else "right"
        ax.text(x_text, y[i], f"{delta:+.2f}", va="center", ha=ha, fontsize=9, alpha=0.9)

    ax.set_xlim(float(np.min(deltas)) - 0.12 * x_range, float(np.max(deltas)) + 0.18 * x_range)

    _save_fig(fig, out_dir / "choices13k_delta_from_centaur", save_pdf=save_pdf)


def make_grouped_score_bar_plot(
    participant_ids: np.ndarray,
    ours_test: np.ndarray,
    centaur_test: np.ndarray,
    improvements: np.ndarray,
    n: int,
    ours_better_count: int,
    mean_improvement: float,
    median_improvement: float,
    out_dir: Path,
    save_pdf: bool,
) -> None:
    random_baseline = -0.6931
    order = np.argsort(-improvements)
    pids = participant_ids[order].astype(str)
    ours_vals = ours_test[order] - random_baseline
    centaur_vals = centaur_test[order] - random_baseline
    deltas = improvements[order]

    x = np.arange(len(deltas))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.bar(x - width / 2, ours_vals, width=width, color="#4C78A8", alpha=0.9, label="Ours")
    ax.bar(x + width / 2, centaur_vals, width=width, color="#9e9e9e", alpha=0.9, label="Centaur")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pids, rotation=0)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Test log-likelihood relative to random baseline")
    ax.set_title("Choices13k: held-out performance relative to random baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    y_min = float(min(np.min(ours_vals), np.min(centaur_vals)))
    y_max = float(max(np.max(ours_vals), np.max(centaur_vals)))
    y_span = max(y_max - y_min, 0.05)
    for i, delta in enumerate(deltas):
        top = max(ours_vals[i], centaur_vals[i])
        ax.text(x[i], top + 0.035 * y_span, f"{delta:+.2f}", ha="center", va="bottom", fontsize=8, alpha=0.9)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.15 * y_span)
    _save_fig(fig, out_dir / "choices13k_grouped_scores", save_pdf=save_pdf)


def write_summary_csv(
    out_path: Path,
    participant_ids: Iterable[int],
    ours_train: np.ndarray,
    ours_test: np.ndarray,
    centaur_train: np.ndarray,
    centaur_test: np.ndarray,
    improvements: np.ndarray,
    ours_better: np.ndarray,
) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "participant_id",
                "ours_train_loglik",
                "ours_test_loglik",
                "centaur_train_loglik",
                "centaur_test_loglik",
                "improvement",
                "ours_better",
            ]
        )
        for i, pid in enumerate(participant_ids):
            writer.writerow(
                [
                    pid,
                    f"{ours_train[i]:.6f}",
                    f"{ours_test[i]:.6f}",
                    f"{centaur_train[i]:.6f}",
                    f"{centaur_test[i]:.6f}",
                    f"{improvements[i]:.6f}",
                    bool(ours_better[i]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create proposal graphs for Choices13k: Ours vs Centaur.")
    parser.add_argument("--ours_dir", type=Path, default=Path(DEFAULT_OURS_DIR), help="Our run directory")
    parser.add_argument("--centaur_dir", type=Path, default=Path(DEFAULT_CENTAUR_DIR), help="Centaur run directory")
    parser.add_argument("--out_dir", type=Path, default=Path(DEFAULT_OUT_DIR), help="Output directory for plots/csv")
    parser.add_argument("--save_pdf", action="store_true", help="Also save PDF versions of figures")
    args = parser.parse_args()

    ours_rows, ours_source = load_participant_loglik_data(args.ours_dir, "Ours")
    centaur_rows, centaur_source = load_participant_loglik_data(args.centaur_dir, "Centaur")

    aligned_ids = sorted(set(ours_rows.keys()) & set(centaur_rows.keys()))
    if not aligned_ids:
        raise ValueError(
            "No aligned participant_id between methods. "
            f"Ours source={ours_source}, Centaur source={centaur_source}"
        )

    ours_train_vals: List[float] = []
    ours_test_vals: List[float] = []
    centaur_train_vals: List[float] = []
    centaur_test_vals: List[float] = []
    aligned_valid_ids: List[int] = []

    for pid in aligned_ids:
        ours_train, ours_test = ours_rows[pid]
        centaur_train, centaur_test = centaur_rows[pid]
        if None in (ours_train, ours_test, centaur_train, centaur_test):
            continue
        aligned_valid_ids.append(pid)
        ours_train_vals.append(float(ours_train))
        ours_test_vals.append(float(ours_test))
        centaur_train_vals.append(float(centaur_train))
        centaur_test_vals.append(float(centaur_test))

    if not aligned_valid_ids:
        raise ValueError(
            "Aligned participants were found, but no rows had complete "
            "train/test loglik values for both methods."
        )

    participant_ids = np.array(aligned_valid_ids, dtype=int)
    ours_train = np.array(ours_train_vals, dtype=float)
    ours_test = np.array(ours_test_vals, dtype=float)
    centaur_train = np.array(centaur_train_vals, dtype=float)
    centaur_test = np.array(centaur_test_vals, dtype=float)
    improvements = ours_test - centaur_test
    ours_better = improvements > 0

    n = len(participant_ids)
    ours_better_count = int(np.sum(ours_better))
    ours_better_pct = 100.0 * ours_better_count / n
    mean_improvement = float(np.mean(improvements))
    median_improvement = float(np.median(improvements))
    if n < 20:
        print(
            f"WARNING: only {n} aligned participants found. "
            "Check whether this run contains the expected participant subset."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_delta_from_centaur_plot(
        participant_ids,
        improvements,
        n,
        ours_better_count,
        mean_improvement,
        median_improvement,
        args.out_dir,
        save_pdf=args.save_pdf,
    )
    make_grouped_score_bar_plot(
        participant_ids,
        ours_test,
        centaur_test,
        improvements,
        n,
        ours_better_count,
        mean_improvement,
        median_improvement,
        args.out_dir,
        save_pdf=args.save_pdf,
    )
    summary_csv = args.out_dir / "choices13k_proposal_summary.csv"
    write_summary_csv(
        summary_csv,
        participant_ids,
        ours_train,
        ours_test,
        centaur_train,
        centaur_test,
        improvements,
        ours_better,
    )

    order = np.argsort(-improvements)
    top_k = min(5, n)
    bot_order = np.argsort(improvements)
    print(f"[INFO] Ours source CSV: {ours_source}")
    print(f"[INFO] Centaur source CSV: {centaur_source}")
    print(f"[INFO] Output directory: {args.out_dir}")
    print(f"[INFO] Number of aligned participants: {n}")
    print(f"[INFO] Number ours better: {ours_better_count}")
    print(f"[INFO] Percentage ours better: {ours_better_pct:.2f}%")
    print(f"[INFO] Mean improvement: {mean_improvement:.6f}")
    print(f"[INFO] Median improvement: {median_improvement:.6f}")
    print("[INFO] Best 5 participants by improvement:")
    for idx in order[:top_k]:
        print(f"  participant {participant_ids[idx]}: improvement={improvements[idx]:.6f}")
    print("[INFO] Worst 5 participants by improvement:")
    for idx in bot_order[:top_k]:
        print(f"  participant {participant_ids[idx]}: improvement={improvements[idx]:.6f}")


if __name__ == "__main__":
    main()
