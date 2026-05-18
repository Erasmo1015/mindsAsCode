#!/usr/bin/env python3
"""
Compare per-participant test log-likelihood from one or more runs against a Centaur baseline CSV.

Writes a wide table under analysis/data/utils/ with footer rows Avg, Better, Similar, Worse.

Usage (choice13k; default dataset):
  python analysis/code/utils/compare.py \\
    --experiment_paths generated_outputs/choice13k/te_dr/run_260514_231815 ...

Usage (cpc18; Centaur + output defaults for that dataset):
  python analysis/code/utils/compare.py --dataset cpc18 \\
    --experiment_paths generated_outputs/cpc18/non_strict/run_260517_211601

Usage (mixed_gambles):
  python analysis/code/utils/compare.py --dataset mixed_gambles \\
    --experiment_paths generated_outputs/mixed_gambles/non_strict/run_260518_100539
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_DATASETS = ("choice13k", "cpc18", "mixed_gambles")


@dataclass(frozen=True)
class _DatasetDefaults:
    centaur_csv: Path
    output_csv: Path
    bir_csv: Optional[Path]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dataset_defaults(repo: Path, dataset: str) -> _DatasetDefaults:
    if dataset not in _DATASETS:
        raise ValueError(f"dataset must be one of {_DATASETS}, got {dataset!r}")
    gen = repo / "generated_outputs"
    if dataset == "choice13k":
        return _DatasetDefaults(
            centaur_csv=gen
            / "choice13k"
            / "centaur"
            / "run_260517_190700"
            / "participant_details_loglik.csv",
            output_csv=repo / "analysis" / "data" / "utils" / "loglik_compare_choice13k.csv",
            bir_csv=gen
            / "choice13k"
            / "te_aggregate"
            / "run_260513_234734"
            / "analysis"
            / "behavioral_inconsistency_rate.csv",
        )
    if dataset == "cpc18":
        return _DatasetDefaults(
            centaur_csv=gen
            / "cpc18"
            / "centaur"
            / "run_260517_190927"
            / "participant_details_loglik.csv",
            output_csv=repo / "analysis" / "data" / "utils" / "loglik_compare_cpc18.csv",
            bir_csv=None,
        )
    return _DatasetDefaults(
        centaur_csv=gen
        / "mixed_gambles"
        / "centaur"
        / "run_260517_190705"
        / "participant_details_loglik.csv",
        output_csv=repo / "analysis" / "data" / "utils" / "loglik_compare_mixed_gambles.csv",
        bir_csv=None,
    )


def _resolve_loglik_csv(path: Path) -> Path:
    """Accept either a run directory or a path to participant_details_loglik.csv."""
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidate = path / "participant_details_loglik.csv"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Expected a CSV file or a run directory containing participant_details_loglik.csv; got {path}"
    )


def _run_column_name(path: Path) -> str:
    if path.is_file():
        return path.parent.name if path.parent.name else path.stem
    return path.name


def _read_test_loglik_csv(csv_path: Path) -> Dict[int, float]:
    out: Dict[int, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        id_key = "participant_id" if "participant_id" in reader.fieldnames else None
        if id_key is None:
            raise ValueError(f"{csv_path}: missing participant_id column (got {reader.fieldnames})")
        if "test_loglik" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: missing test_loglik column (got {reader.fieldnames})")
        for row in reader:
            raw = row.get(id_key)
            if raw is None or str(raw).strip() == "":
                continue
            pid = int(float(raw))
            tl = row.get("test_loglik")
            if tl is None or str(tl).strip() == "":
                continue
            out[pid] = float(tl)
    return out


def _read_bir_csv_file(bir_csv: Path) -> Dict[int, float]:
    """Load BIR keyed by participant_ordinal from behavioral_inconsistency_rate.csv."""
    if not bir_csv.is_file():
        return {}
    out: Dict[int, float] = {}
    with open(bir_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        if "participant_ordinal" not in reader.fieldnames:
            return out
        bir_field = "BIR" if "BIR" in reader.fieldnames else (
            "behavioral_inconsistency_rate" if "behavioral_inconsistency_rate" in reader.fieldnames else None
        )
        if bir_field is None:
            return out
        for row in reader:
            o = row.get("participant_ordinal")
            if o is None or str(o).strip() == "":
                continue
            b = row.get(bir_field)
            if b is None or str(b).strip() == "":
                continue
            out[int(float(o))] = float(b)
    return out


def _resolve_bir_csv(path: Path) -> Path:
    """Path to behavioral_inconsistency_rate.csv: file as-is, or run dir / analysis / csv."""
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidate = path / "analysis" / "behavioral_inconsistency_rate.csv"
    return candidate


def _read_bir_map(run_or_csv: Path) -> Dict[int, float]:
    """Load BIR from analysis/behavioral_inconsistency_rate.csv under run (or next to loglik csv)."""
    base = run_or_csv if run_or_csv.is_dir() else run_or_csv.parent
    return _read_bir_csv_file(base / "analysis" / "behavioral_inconsistency_rate.csv")


def _merge_bir_from_summaries(run_paths: Sequence[Path]) -> Dict[int, float]:
    """Fallback: participants_summary.csv behavioral_inconsistency_rate."""
    for base in run_paths:
        root = base if base.is_dir() else base.parent
        summary = root / "participants_summary.csv"
        if not summary.is_file():
            continue
        out: Dict[int, float] = {}
        with open(summary, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            if "participant_id" not in reader.fieldnames or "behavioral_inconsistency_rate" not in reader.fieldnames:
                continue
            for row in reader:
                pid = row.get("participant_id")
                br = row.get("behavioral_inconsistency_rate")
                if pid is None or br is None or str(pid).strip() == "" or str(br).strip() == "":
                    continue
                out[int(float(pid))] = float(br)
        if out:
            return out
    return {}


def _finite_mean(values: Iterable[float], ndigits: int = 2) -> str:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return ""
    return f"{statistics.mean(vals):.{ndigits}f}"


def _classify_vs_centaur(
    centaur: Dict[int, float],
    experiment: Dict[int, float],
    threshold: float,
) -> Tuple[int, int, int]:
    """Counts (better, similar, worse) among participants with both values."""
    better = similar = worse = 0
    for pid, c_ll in centaur.items():
        if pid not in experiment:
            continue
        e_ll = experiment[pid]
        if not (math.isfinite(c_ll) and math.isfinite(e_ll)):
            continue
        if e_ll > c_ll:
            better += 1
        elif abs(e_ll - c_ll) <= threshold:
            similar += 1
        else:
            worse += 1
    return better, similar, worse

def main() -> None:
    repo = _repo_root()

    p = argparse.ArgumentParser(description="Compare experiment test_loglik to Centaur baseline.")
    p.add_argument(
        "--dataset",
        choices=list(_DATASETS),
        default="choice13k",
        help=(
            "Dataset family: sets default Centaur CSV and output path "
            "(choice13k | cpc18 | mixed_gambles)."
        ),
    )
    p.add_argument(
        "--experiment_paths",
        nargs="+",
        type=Path,
        required=True,
        help="Run directories (or participant_details_loglik.csv paths), e.g. .../run_260514_153610",
    )
    p.add_argument(
        "--centaur_csv",
        type=Path,
        default=None,
        help=(
            "Centaur participant_details_loglik.csv (default depends on --dataset: "
            "choice13k run_260517_190700, cpc18 run_260517_190927, mixed_gambles run_260517_190705)."
        ),
    )
    p.add_argument(
        "--similar_threshold",
        type=float,
        default=0.05,
        help="|exp - centaur| <= this counts as Similar when not strictly better than Centaur.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: analysis/data/utils/loglik_compare_<dataset>.csv).",
    )
    p.add_argument(
        "--bir_csv",
        type=Path,
        default=None,
        help=(
            "behavioral_inconsistency_rate.csv (or its run directory). "
            "Default for choice13k: te_aggregate run_260513_234734; cpc18/mixed_gambles: none. "
            "If missing, BIR is taken from experiment paths / summaries when available."
        ),
    )
    args = p.parse_args()

    ds_defaults = _dataset_defaults(repo, args.dataset)
    centaur_arg = Path(args.centaur_csv).expanduser() if args.centaur_csv is not None else ds_defaults.centaur_csv
    output_arg = Path(args.output).expanduser() if args.output is not None else ds_defaults.output_csv
    bir_arg: Optional[Path]
    if args.bir_csv is not None:
        bir_arg = Path(args.bir_csv).expanduser()
    else:
        bir_arg = ds_defaults.bir_csv

    centaur_path = _resolve_loglik_csv(centaur_arg)
    centaur = _read_test_loglik_csv(centaur_path)

    exp_resolved = [_resolve_loglik_csv(Path(ep).expanduser()) for ep in args.experiment_paths]
    run_labels = [_run_column_name(p) for p in exp_resolved]
    if len(set(run_labels)) != len(run_labels):
        raise ValueError(f"Duplicate run column names after resolution: {run_labels}")

    experiments: List[Tuple[str, Dict[int, float]]] = []
    for label, csv_path in zip(run_labels, exp_resolved):
        experiments.append((label, _read_test_loglik_csv(csv_path)))

    bir: Dict[int, float] = {}
    if bir_arg is not None:
        bir_path = _resolve_bir_csv(bir_arg)
        if bir_path.is_file():
            bir = _read_bir_csv_file(bir_path)
    if not bir:
        for ep in args.experiment_paths:
            bir = _read_bir_map(Path(ep).expanduser().resolve())
            if bir:
                break
    if not bir:
        bir = _merge_bir_from_summaries([p.parent if p.is_file() else p for p in exp_resolved])

    pids = set(centaur.keys())
    for _, m in experiments:
        pids |= set(m.keys())
    ordered = sorted(pids)

    fieldnames = ["participant_id", "BIR", "Centaur"] + run_labels
    out_path = output_arg
    out_path = out_path.resolve() if out_path.is_absolute() else (repo / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for pid in ordered:
        row: Dict[str, str] = {
            "participant_id": str(pid),
            "BIR": f"{bir[pid]:.2f}" if pid in bir else "",
            "Centaur": f"{centaur[pid]:.2f}" if pid in centaur else "",
        }
        for label, m in experiments:
            row[label] = f"{m[pid]:.2f}" if pid in m else ""
        rows.append(row)

    # Footer: Avg
    avg_row: Dict[str, str] = {"participant_id": "Avg", "BIR": "", "Centaur": ""}
    if bir:
        avg_row["BIR"] = _finite_mean([bir[pid] for pid in ordered if pid in bir])
    avg_row["Centaur"] = _finite_mean([centaur[pid] for pid in ordered if pid in centaur])
    for label, m in experiments:
        avg_row[label] = _finite_mean([m[pid] for pid in ordered if pid in m])
    rows.append(avg_row)

    th = float(args.similar_threshold)
    counts_by_label = {label: _classify_vs_centaur(centaur, m, th) for label, m in experiments}
    for footer, idx in (("Better", 0), ("Similar", 1), ("Worse", 2)):
        r = {fn: "" for fn in fieldnames}
        r["participant_id"] = footer
        r["BIR"] = ""
        r["Centaur"] = ""
        for label, _ in experiments:
            r[label] = str(counts_by_label[label][idx])
        rows.append(r)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(
        f"Wrote {out_path} (dataset={args.dataset}, {len(ordered)} participants, "
        f"{len(experiments)} experiments, centaur={centaur_path})."
    )


if __name__ == "__main__":
    main()
