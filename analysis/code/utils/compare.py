#!/usr/bin/env python3
"""
Compare per-participant log-likelihood from experiment runs against a Centaur baseline CSV.

Participant rows follow the Centaur ``participant_details_loglik.csv`` (every
``participant_id`` in that file, in file order). Experiment columns are blank when
a run has no score for that id.

Always writes a table for test_loglik. If gated_test_loglik appears in any
experiment participant_details_loglik.csv, also writes a separate *_gated.csv
with the same layout (Avg / Better / Similar / Worse footers). In that gated
table, experiment columns use gated_test_loglik; the Centaur column uses
gated_test_loglik when present, otherwise test_loglik (Centaur runs typically
have no gated scores).

Usage (choice13k; default dataset):
  python analysis/code/utils/compare.py --experiment_paths generated_outputs/choice13k/te_dr/run_260514_231815 ...

Usage (cpc18):
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
_TEST_LOGLIK = "test_loglik"
_GATED_LOGLIK = "gated_test_loglik"


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


def _csv_fieldnames(csv_path: Path) -> Optional[List[str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames) if reader.fieldnames else None


def _csv_has_column(csv_path: Path, column: str) -> bool:
    fields = _csv_fieldnames(csv_path)
    return fields is not None and column in fields


def _read_centaur_participant_ids(centaur_path: Path) -> List[int]:
    """All participant_id values from the Centaur CSV, in file order (deduped)."""
    ids: List[int] = []
    seen: set[int] = set()
    with open(centaur_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{centaur_path}: empty CSV")
        if "participant_id" not in reader.fieldnames:
            raise ValueError(
                f"{centaur_path}: missing participant_id column (got {reader.fieldnames})"
            )
        for row in reader:
            raw = row.get("participant_id")
            if raw is None or str(raw).strip() == "":
                continue
            pid = int(float(raw))
            if pid in seen:
                continue
            seen.add(pid)
            ids.append(pid)
    if not ids:
        raise ValueError(f"{centaur_path}: no participant_id rows found")
    return ids


def _read_loglik_csv(csv_path: Path, column: str, *, required: bool) -> Dict[int, float]:
    """Read one log-likelihood column keyed by participant_id."""
    out: Dict[int, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        if "participant_id" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: missing participant_id column (got {reader.fieldnames})")
        if column not in reader.fieldnames:
            if required:
                raise ValueError(f"{csv_path}: missing {column} column (got {reader.fieldnames})")
            return out
        for row in reader:
            raw = row.get("participant_id")
            if raw is None or str(raw).strip() == "":
                continue
            pid = int(float(raw))
            val = row.get(column)
            if val is None or str(val).strip() == "":
                continue
            out[pid] = float(val)
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
    return path / "analysis" / "behavioral_inconsistency_rate.csv"


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


def _gated_output_path(test_output: Path) -> Path:
    """Derive gated comparison path from test output (e.g. loglik_compare_cpc18_gated.csv)."""
    if test_output.suffix:
        return test_output.with_name(f"{test_output.stem}_gated{test_output.suffix}")
    return test_output.parent / f"{test_output.name}_gated"


def _any_csv_has_gated(centaur_path: Path, experiment_csvs: Sequence[Path]) -> bool:
    if _csv_has_column(centaur_path, _GATED_LOGLIK):
        return True
    return any(_csv_has_column(p, _GATED_LOGLIK) for p in experiment_csvs)


def _centaur_scores_for_gated_table(
    centaur_test: Dict[int, float],
    centaur_gated: Dict[int, float],
) -> Dict[int, float]:
    """Centaur baseline for gated comparison: gated when available, else test_loglik."""
    merged = dict(centaur_test)
    merged.update(centaur_gated)
    return merged


def _write_comparison_csv(
    *,
    out_path: Path,
    participant_ids: Sequence[int],
    centaur: Dict[int, float],
    experiments: Sequence[Tuple[str, Dict[int, float]]],
    bir: Dict[int, float],
    similar_threshold: float,
) -> int:
    """Write comparison table; return number of participant rows."""
    ordered = list(participant_ids)

    fieldnames = ["participant_id", "BIR", "Centaur"] + [label for label, _ in experiments]
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

    avg_row: Dict[str, str] = {"participant_id": "Avg", "BIR": "", "Centaur": ""}
    if bir:
        avg_row["BIR"] = _finite_mean([bir[pid] for pid in ordered if pid in bir])
    avg_row["Centaur"] = _finite_mean([centaur[pid] for pid in ordered if pid in centaur])
    for label, m in experiments:
        avg_row[label] = _finite_mean([m[pid] for pid in ordered if pid in m])
    rows.append(avg_row)

    th = float(similar_threshold)
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

    return len(ordered)


def main() -> None:
    repo = _repo_root()

    p = argparse.ArgumentParser(
        description="Compare experiment log-likelihood to Centaur (test_loglik; optional gated file)."
    )
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
            "Centaur participant_details_loglik.csv; defines which participant_id rows "
            "appear in the output (default depends on --dataset: choice13k run_260517_190700, "
            "cpc18 run_260517_190927, mixed_gambles run_260517_190705)."
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
        help="Output CSV for test_loglik (default: analysis/data/utils/loglik_compare_<dataset>.csv).",
    )
    p.add_argument(
        "--output_gated",
        type=Path,
        default=None,
        help=(
            "Output CSV for gated_test_loglik (default: <test output stem>_gated.csv). "
            "Written only if gated_test_loglik column exists in Centaur or any experiment CSV."
        ),
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
    output_arg = output_arg.resolve() if output_arg.is_absolute() else (repo / output_arg).resolve()
    bir_arg: Optional[Path]
    if args.bir_csv is not None:
        bir_arg = Path(args.bir_csv).expanduser()
    else:
        bir_arg = ds_defaults.bir_csv

    centaur_path = _resolve_loglik_csv(centaur_arg)
    centaur_participant_ids = _read_centaur_participant_ids(centaur_path)
    exp_resolved = [_resolve_loglik_csv(Path(ep).expanduser()) for ep in args.experiment_paths]
    run_labels = [_run_column_name(p) for p in exp_resolved]
    if len(set(run_labels)) != len(run_labels):
        raise ValueError(f"Duplicate run column names after resolution: {run_labels}")

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

    th = float(args.similar_threshold)

    centaur_test = _read_loglik_csv(centaur_path, _TEST_LOGLIK, required=True)
    experiments_test: List[Tuple[str, Dict[int, float]]] = []
    for label, csv_path in zip(run_labels, exp_resolved):
        experiments_test.append(
            (label, _read_loglik_csv(csv_path, _TEST_LOGLIK, required=True))
        )

    n_test = _write_comparison_csv(
        out_path=output_arg,
        participant_ids=centaur_participant_ids,
        centaur=centaur_test,
        experiments=experiments_test,
        bir=bir,
        similar_threshold=th,
    )
    print(
        f"Wrote {output_arg} ({_TEST_LOGLIK}, dataset={args.dataset}, "
        f"{n_test} participants (from Centaur CSV), {len(experiments_test)} experiments, "
        f"centaur={centaur_path})."
    )

    if not _any_csv_has_gated(centaur_path, exp_resolved):
        return

    gated_out = (
        Path(args.output_gated).expanduser()
        if args.output_gated is not None
        else _gated_output_path(output_arg)
    )
    gated_out = gated_out.resolve() if gated_out.is_absolute() else (repo / gated_out).resolve()

    centaur_gated = _read_loglik_csv(centaur_path, _GATED_LOGLIK, required=False)
    centaur_for_gated = _centaur_scores_for_gated_table(centaur_test, centaur_gated)
    experiments_gated: List[Tuple[str, Dict[int, float]]] = []
    for label, csv_path in zip(run_labels, exp_resolved):
        experiments_gated.append(
            (label, _read_loglik_csv(csv_path, _GATED_LOGLIK, required=False))
        )

    # Same participant roster as test_loglik table (all Centaur CSV ids, not union of
    # experiment rows that happen to have non-empty gated_test_loglik).
    n_gated = _write_comparison_csv(
        out_path=gated_out,
        participant_ids=centaur_participant_ids,
        centaur=centaur_for_gated,
        experiments=experiments_gated,
        bir=bir,
        similar_threshold=th,
    )
    if centaur_gated:
        centaur_note = f"centaur uses {_GATED_LOGLIK} where present, else {_TEST_LOGLIK}"
    else:
        centaur_note = f"centaur uses {_TEST_LOGLIK} (no {_GATED_LOGLIK} in Centaur CSV)"
    print(
        f"Wrote {gated_out} ({_GATED_LOGLIK}, dataset={args.dataset}, "
        f"{n_gated} participants (from Centaur CSV), {len(experiments_gated)} experiments; "
        f"{centaur_note})."
    )


if __name__ == "__main__":
    main()
