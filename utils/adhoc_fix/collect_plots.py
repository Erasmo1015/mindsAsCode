#!/usr/bin/env python3
"""Rebuild per-participant loglik CSV/plots from old te_aggregate run folders.

Expected input:
- run directory containing participant folders, e.g.
  generated_outputs/choice13k/te_aggregate/run_260430_184849

Per participant, this script extracts evolution trajectory for the best program
in the pool at each iteration:
- train_loglik
- test_loglik

Data source priority:
1) participant_*/wandb_metrics.jsonl
2) participant_*/iteration_*/metrics.json (+ participant_*/results.json baseline)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isfinite(v):
        return v
    return None


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _collect_from_wandb_jsonl(participant_dir: Path, pid: int) -> List[dict]:
    jsonl_path = participant_dir / "wandb_metrics.jsonl"
    rows = _read_jsonl(jsonl_path)
    if not rows:
        return []

    out: List[dict] = []
    train_k = f"p{pid}_train_loglik"
    test_k = f"p{pid}_test_loglik"
    for r in rows:
        iteration = r.get("iteration")
        if iteration is None:
            continue
        out.append(
            {
                "iteration": int(iteration),
                "step": int(r.get("step", iteration + 1)),
                "train_loglik": _safe_float(r.get(train_k)),
                "test_loglik": _safe_float(r.get(test_k)),
                "source": "wandb_metrics.jsonl",
            }
        )
    out.sort(key=lambda d: d["iteration"])
    return out


def _collect_from_iteration_metrics(participant_dir: Path) -> List[dict]:
    out: List[dict] = []

    # Baseline from results.json if available
    results_path = participant_dir / "results.json"
    if results_path.exists():
        try:
            res = json.loads(results_path.read_text(encoding="utf-8"))
            base = res.get("baseline", {})
            out.append(
                {
                    "iteration": -1,
                    "step": 0,
                    "train_loglik": _safe_float(base.get("train_loglik")),
                    "test_loglik": _safe_float(base.get("test_loglik")),
                    "source": "results.json+iteration_metrics",
                }
            )
        except json.JSONDecodeError:
            pass

    iter_metrics = sorted(
        participant_dir.glob("iteration_*/metrics.json"),
        key=lambda p: int(re.search(r"iteration_(\d+)", str(p)).group(1)),
    )
    for mpath in iter_metrics:
        try:
            payload = json.loads(mpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        iteration = int(payload.get("iteration", -999))
        out.append(
            {
                "iteration": iteration,
                "step": iteration + 1,
                "train_loglik": _safe_float(payload.get("best_train_loglik")),
                "test_loglik": _safe_float(payload.get("best_test_loglik")),
                "source": "results.json+iteration_metrics",
            }
        )
    out.sort(key=lambda d: d["iteration"])
    return out


def _write_csv(rows: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["iteration", "train_loglik", "test_loglik"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = dict(r)
            row.pop("source", None)
            row.pop("step", None)
            if row.get("train_loglik") is not None:
                row["train_loglik"] = round(float(row["train_loglik"]), 2)
            if row.get("test_loglik") is not None:
                row["test_loglik"] = round(float(row["test_loglik"]), 2)
            w.writerow(row)


def _plot_rows(rows: List[dict], png_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it in your env."
        ) from exc

    xs = [r["iteration"] for r in rows]
    ys_train = [r["train_loglik"] for r in rows]
    ys_test = [r["test_loglik"] for r in rows]

    plt.figure(figsize=(8, 4.8))
    plt.plot(xs, ys_train, marker="o", linewidth=1.6, label="train_loglik")
    plt.plot(xs, ys_test, marker="o", linewidth=1.6, label="test_loglik")
    plt.axvline(x=0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
    plt.title(title)
    plt.xlabel("iteration (-1 is baseline)")
    plt.ylabel("log-likelihood")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=160)
    plt.close()


def _participant_id_from_name(name: str) -> Optional[int]:
    m = re.match(r"participant_(\d+)$", name)
    if not m:
        return None
    return int(m.group(1))


def _collect_aggregate_rows(run_dir: Path) -> List[dict]:
    aggregate_dir = run_dir / "aggregate"
    if not aggregate_dir.exists():
        return []
    out: List[dict] = []
    iter_metrics = sorted(
        aggregate_dir.glob("iteration_*/metrics.json"),
        key=lambda p: int(re.search(r"iteration_(\d+)", str(p)).group(1)),
    )
    for mpath in iter_metrics:
        try:
            payload = json.loads(mpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        iteration = int(payload.get("iteration", -999))
        out.append(
            {
                "iteration": iteration,
                "train_loglik": _safe_float(payload.get("best_aggregate_train_loglik")),
                "test_loglik": _safe_float(payload.get("best_aggregate_test_loglik")),
            }
        )
    out.sort(key=lambda d: d["iteration"])
    return out


def collect_for_run(run_dir: Path) -> Tuple[int, int]:
    participant_dirs = sorted(
        [p for p in run_dir.glob("participant_*") if p.is_dir()],
        key=lambda p: _participant_id_from_name(p.name) or -1,
    )
    csv_dir = run_dir / "csvs"
    plot_dir = run_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    n_found = 0
    n_written = 0
    for pdir in participant_dirs:
        pid = _participant_id_from_name(pdir.name)
        if pid is None:
            continue
        n_found += 1
        rows = _collect_from_wandb_jsonl(pdir, pid)
        if not rows:
            rows = _collect_from_iteration_metrics(pdir)
        if not rows:
            print(f"[WARN] skip {pdir.name}: no usable loglik trajectory found")
            continue

        csv_path = csv_dir / f"{pdir.name}_evolution_loglik.csv"
        png_path = plot_dir / f"{pdir.name}_evolution_loglik.png"
        _write_csv(rows, csv_path)
        _plot_rows(rows, png_path, title=f"{run_dir.name} / {pdir.name}")
        n_written += 1
        print(f"[OK] {pdir.name}: wrote {csv_path} and {png_path}")

    aggregate_rows = _collect_aggregate_rows(run_dir)
    if aggregate_rows:
        agg_csv = csv_dir / "aggregate_evolution_loglik.csv"
        agg_png = plot_dir / "aggregate_evolution_loglik.png"
        _write_csv(aggregate_rows, agg_csv)
        _plot_rows(aggregate_rows, agg_png, title=f"{run_dir.name} / aggregate")
        print(f"[OK] aggregate: wrote {agg_csv} and {agg_png}")
    return n_found, n_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect participant train/test loglik curves from te_aggregate logs."
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to te_aggregate run directory (contains participant_* folders).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run dir is not a directory: {run_dir}")

    n_found, n_written = collect_for_run(run_dir)
    if n_found == 0:
        print(
            "[INFO] no participant_* folders found. "
            "This run may only contain aggregate phase outputs."
        )
    print(f"[DONE] participants found={n_found}, outputs written={n_written}")


if __name__ == "__main__":
    main()
