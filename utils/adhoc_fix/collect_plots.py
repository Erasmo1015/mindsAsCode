#!/usr/bin/env python3
"""Rebuild per-participant loglik CSV/plots from TE run folders.

Expected input:
- evolution run directory, e.g.
  generated_outputs/choice13k/te_aggregate/run_260430_184849
- refinement (refine-only or post-evolution) run directory, e.g.
  generated_outputs/choice13k/non_strict/run_260517_181057

Per participant, extracts trajectory for the elite pool-best program at each
iteration (refinement) or best program in the pool (evolution):
- train_loglik, val_loglik, test_loglik
- train_val_loglik (refinement: 0.5*train + 0.5*val combined fitness)

Refinement iteration metrics (current format) use pool_best_* keys; older runs
may only have best_* / iter_best_* fallbacks.

Data source priority:
1) participant_*/wandb_metrics.jsonl (or participant_*/refinement/wandb_metrics.jsonl)
2) participant_*/iteration_*/metrics.json (+ results.json baseline)
3) participant_*/refinement/iteration_*/metrics.json (+ refinement/results.json baseline)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# metric_name -> JSON keys tried in order (first finite float wins)
_REFINEMENT_ITER_METRIC_KEYS: Mapping[str, Sequence[str]] = {
    "train_loglik": (
        "pool_best_train_loglik",
        "iter_best_train_loglik",
        "best_train_loglik",
    ),
    "val_loglik": (
        "pool_best_val_loglik",
        "iter_best_val_loglik",
        "best_val_loglik",
    ),
    "test_loglik": ("pool_best_test_loglik", "best_test_loglik"),
    "train_val_loglik": (
        "pool_best_train_val_loglik",
        "iter_best_train_val_loglik",
        "train_val_loglik",
    ),
}
_EVOLUTION_ITER_METRIC_KEYS: Mapping[str, Sequence[str]] = {
    "train_loglik": ("best_train_loglik",),
    "val_loglik": ("best_val_loglik",),
    "test_loglik": ("best_test_loglik",),
}
_CSV_METRIC_FIELDS = ("train_loglik", "val_loglik", "test_loglik", "train_val_loglik")


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


def _first_metric(payload: dict, *keys: str) -> Optional[float]:
    for key in keys:
        v = _safe_float(payload.get(key))
        if v is not None:
            return v
    return None


def _metrics_from_payload(
    payload: dict, key_map: Mapping[str, Sequence[str]]
) -> Dict[str, Optional[float]]:
    return {name: _first_metric(payload, *keys) for name, keys in key_map.items()}


def _row_has_any_metric(row: dict, fields: Sequence[str] = _CSV_METRIC_FIELDS) -> bool:
    return any(row.get(f) is not None for f in fields)


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


def _collect_from_wandb_jsonl(
    participant_dir: Path, pid: int, *, refinement: bool = False
) -> List[dict]:
    candidates = [participant_dir / "wandb_metrics.jsonl"]
    if refinement:
        candidates.insert(0, participant_dir / "refinement" / "wandb_metrics.jsonl")
    rows: List[dict] = []
    for jsonl_path in candidates:
        rows = _read_jsonl(jsonl_path)
        if rows:
            break
    if not rows:
        return []

    out: List[dict] = []
    train_k = f"p{pid}_train_loglik"
    val_k = f"p{pid}_val_loglik"
    test_k = f"p{pid}_test_loglik"
    train_val_k = f"p{pid}_train_val_loglik"
    train_fitness_k = f"p{pid}_train_fitness"  # legacy evolution runs
    for r in rows:
        iteration = r.get("iteration")
        if iteration is None:
            continue
        row = {
            "iteration": int(iteration),
            "step": int(r.get("step", iteration + 1)),
            "train_loglik": _safe_float(r.get(train_k)),
            "val_loglik": _safe_float(r.get(val_k)),
            "test_loglik": _safe_float(r.get(test_k)),
            "train_val_loglik": _first_metric(
                r, train_val_k, train_fitness_k, "train_val_loglik"
            ),
            "source": "wandb_metrics.jsonl",
        }
        if _row_has_any_metric(row):
            out.append(row)
    out.sort(key=lambda d: d["iteration"])
    return out


def _collect_from_iteration_metrics(
    participant_dir: Path, *, refinement: bool = False
) -> List[dict]:
    out: List[dict] = []

    if refinement:
        refinement_dir = participant_dir / "refinement"
        results_path = refinement_dir / "results.json"
        iter_glob = refinement_dir.glob("iteration_*/metrics.json")
        baseline_from_results = _refinement_baseline_from_results
        iter_metric_keys = _REFINEMENT_ITER_METRIC_KEYS
    else:
        results_path = participant_dir / "results.json"
        iter_glob = participant_dir.glob("iteration_*/metrics.json")
        baseline_from_results = _evolution_baseline_from_results
        iter_metric_keys = _EVOLUTION_ITER_METRIC_KEYS

    if results_path.exists():
        try:
            res = json.loads(results_path.read_text(encoding="utf-8"))
            baseline = baseline_from_results(res)
            if baseline is not None:
                out.append(baseline)
        except json.JSONDecodeError:
            pass

    iter_metrics = sorted(
        iter_glob,
        key=lambda p: int(re.search(r"iteration_(\d+)", str(p)).group(1)),
    )
    for mpath in iter_metrics:
        try:
            payload = json.loads(mpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        iteration = int(payload.get("iteration", -999))
        row = {
            "iteration": iteration,
            "step": iteration + 1,
            "source": "results.json+iteration_metrics",
            **_metrics_from_payload(payload, iter_metric_keys),
        }
        if _row_has_any_metric(row):
            out.append(row)
    out.sort(key=lambda d: d["iteration"])
    return out


def _evolution_baseline_from_results(res: dict) -> Optional[dict]:
    base = res.get("baseline", {})
    train = _safe_float(base.get("train_loglik"))
    test = _safe_float(base.get("test_loglik"))
    val = _safe_float(base.get("val_loglik"))
    if train is None and test is None and val is None:
        return None
    return {
        "iteration": -1,
        "step": 0,
        "train_loglik": train,
        "val_loglik": val,
        "test_loglik": test,
        "source": "results.json+iteration_metrics",
    }


def _refinement_baseline_from_results(res: dict) -> Optional[dict]:
    if res.get("refinement_skipped"):
        source = res.get("checkpoint", {})
    else:
        source = res.get("refinement_seed", {})
    row = {
        "iteration": -1,
        "step": 0,
        "train_loglik": _safe_float(source.get("train_loglik")),
        "val_loglik": _safe_float(source.get("val_loglik")),
        "test_loglik": _safe_float(source.get("test_loglik")),
        "train_val_loglik": _safe_float(source.get("train_val_loglik")),
        "source": "refinement/results.json+iteration_metrics",
    }
    if not _row_has_any_metric(row):
        return None
    return row


def _has_refinement_data(participant_dir: Path) -> bool:
    refinement_dir = participant_dir / "refinement"
    if not refinement_dir.is_dir():
        return False
    if any(refinement_dir.glob("iteration_*/metrics.json")):
        return True
    results_path = refinement_dir / "results.json"
    return results_path.is_file()


def _has_evolution_data(participant_dir: Path) -> bool:
    if (participant_dir / "wandb_metrics.jsonl").exists():
        return True
    if any(participant_dir.glob("iteration_*/metrics.json")):
        return True
    results_path = participant_dir / "results.json"
    if not results_path.exists():
        return False
    try:
        res = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return "baseline" in res


def _csv_fields(rows: List[dict]) -> List[str]:
    fields = ["iteration"]
    for name in _CSV_METRIC_FIELDS:
        if any(r.get(name) is not None for r in rows):
            fields.append(name)
    return fields


def _write_csv(rows: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = _csv_fields(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fields}
            for key in fields:
                if key == "iteration":
                    continue
                if row.get(key) is not None:
                    row[key] = round(float(row[key]), 2)
            w.writerow(row)


def _plot_rows(rows: List[dict], png_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it in your env."
        ) from exc

    xs = [r["iteration"] for r in rows]
    series = [
        ("train_loglik", "train_loglik"),
        ("val_loglik", "val_loglik"),
        ("test_loglik", "test_loglik"),
        ("train_val_loglik", "train_val_loglik"),
    ]

    plt.figure(figsize=(8, 4.8))
    for key, label in series:
        ys = [r.get(key) for r in rows]
        if any(v is not None for v in ys):
            plt.plot(xs, ys, marker="o", linewidth=1.6, label=label)
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


def _collect_participant_rows(
    pdir: Path, pid: int, *, phase: str
) -> List[dict]:
    refinement = phase == "refinement"
    rows = _collect_from_wandb_jsonl(pdir, pid, refinement=refinement)
    if not rows:
        rows = _collect_from_iteration_metrics(pdir, refinement=refinement)
    return rows


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
                "val_loglik": _safe_float(payload.get("best_aggregate_val_loglik")),
                "test_loglik": _safe_float(payload.get("best_aggregate_test_loglik")),
            }
        )
    out.sort(key=lambda d: d["iteration"])
    return out


def _output_paths(
    run_dir: Path, pdir_name: str, *, phase: str
) -> Tuple[Path, Path, str]:
    if phase == "refinement":
        csv_dir = run_dir / "csvs" / "refinement"
        plot_dir = run_dir / "plots" / "refinement"
        stem = f"{pdir_name}_refinement_evolution_loglik"
        title_suffix = " / refinement"
    else:
        csv_dir = run_dir / "csvs"
        plot_dir = run_dir / "plots"
        stem = f"{pdir_name}_evolution_loglik"
        title_suffix = ""
    return csv_dir / f"{stem}.csv", plot_dir / f"{stem}.png", title_suffix


def collect_for_run(run_dir: Path, *, phase: str = "auto") -> Tuple[int, int]:
    participant_dirs = sorted(
        [p for p in run_dir.glob("participant_*") if p.is_dir()],
        key=lambda p: _participant_id_from_name(p.name) or -1,
    )
    n_found = 0
    n_written = 0
    for pdir in participant_dirs:
        pid = _participant_id_from_name(pdir.name)
        if pid is None:
            continue
        n_found += 1

        phases_to_collect: List[str] = []
        if phase == "auto":
            if _has_refinement_data(pdir):
                phases_to_collect.append("refinement")
            if _has_evolution_data(pdir):
                phases_to_collect.append("evolution")
        elif phase == "refinement":
            if _has_refinement_data(pdir):
                phases_to_collect.append("refinement")
        else:
            phases_to_collect.append("evolution")

        if not phases_to_collect:
            print(f"[WARN] skip {pdir.name}: no usable loglik trajectory found")
            continue

        for collect_phase in phases_to_collect:
            rows = _collect_participant_rows(pdir, pid, phase=collect_phase)
            if not rows:
                print(
                    f"[WARN] skip {pdir.name} ({collect_phase}): "
                    "no usable loglik trajectory found"
                )
                continue

            csv_path, png_path, title_suffix = _output_paths(
                run_dir, pdir.name, phase=collect_phase
            )
            title = f"{run_dir.name} / {pdir.name}{title_suffix}"
            _write_csv(rows, csv_path)
            _plot_rows(rows, png_path, title=title)
            n_written += 1
            print(f"[OK] {pdir.name} ({collect_phase}): wrote {csv_path} and {png_path}")

    if phase in ("auto", "evolution"):
        aggregate_rows = _collect_aggregate_rows(run_dir)
        if aggregate_rows:
            csv_dir = run_dir / "csvs"
            plot_dir = run_dir / "plots"
            agg_csv = csv_dir / "aggregate_evolution_loglik.csv"
            agg_png = plot_dir / "aggregate_evolution_loglik.png"
            _write_csv(aggregate_rows, agg_csv)
            _plot_rows(aggregate_rows, agg_png, title=f"{run_dir.name} / aggregate")
            print(f"[OK] aggregate: wrote {agg_csv} and {agg_png}")
    return n_found, n_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect participant train/val/test loglik curves from TE run logs "
            "(evolution and/or refinement)."
        )
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to run directory (contains participant_* folders).",
    )
    parser.add_argument(
        "--phase",
        choices=("auto", "evolution", "refinement"),
        default="auto",
        help=(
            "Which phase to collect: auto detects refinement/ and iteration_*/ "
            "folders per participant (default: auto)."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run dir is not a directory: {run_dir}")

    n_found, n_written = collect_for_run(run_dir, phase=args.phase)
    if n_found == 0:
        print(
            "[INFO] no participant_* folders found. "
            "This run may only contain aggregate phase outputs."
        )
    print(f"[DONE] participants found={n_found}, outputs written={n_written}")


if __name__ == "__main__":
    main()
