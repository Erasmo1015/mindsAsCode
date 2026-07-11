#!/usr/bin/env python3
"""
Compare avg gated test log-likelihood across TEH ablation runs.

Usage:
  python analysis/code/utils/compare_ablation.py
  python analysis/code/utils/compare_ablation.py --dataset 1peterson2021using
  python analysis/code/utils/compare_ablation.py --all_in

Rows are datasets; columns are base (full TEH from baseline config) then each
ablation from analysis/config/ablation/config.yaml. Missing runs show N/A.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.teh.teh_datasets import is_mixed_gambles_dataset, normalize_psych101_dataset_alias
_DEFAULT_DATASET = "2plonsky2018when"
_DEFAULT_BASELINE_CONFIG = "analysis/data/baseline_methods/config_EMNLP_rerun1.yaml"
_DEFAULT_ABLATION_CONFIG = "analysis/config/ablation/config.yaml"
_GENERATED_OUTPUTS_DIR = "generated_outputs"
_ABLATION_OUTPUTS_DIR = "generated_outputs_ablation"
_LOGLIK_CSV_NAME = "participant_details_loglik.csv"
_GATED_LOGLIK = "gated_test_loglik"
_LOGLIK_NDIGITS = 2
_NA = "N/A"
_BASE_COL = "base"

_ALL_IN_DATASETS: Tuple[str, ...] = (
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "5speekenbrink2008learning",
    "7hilbig2014generalized",
    "10frey2017risk",
    "11enkavi2019recentprobes",
    "12badham2017deficits",
    "mixed_gambles",
)


def _resolve_repo_path(repo: Path, raw: str) -> Path:
    p = Path(str(raw)).expanduser()
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_ablation_names(config_path: Path) -> List[str]:
    data = _load_yaml(config_path)
    raw = data.get("ablations")
    if not isinstance(raw, list):
        raise ValueError(f"{config_path}: expected top-level 'ablations' list")
    names: List[str] = []
    for item in raw:
        name = str(item).strip()
        if not name:
            raise ValueError(f"{config_path}: ablation names must be non-empty strings")
        names.append(name)
    if not names:
        raise ValueError(f"{config_path}: 'ablations' list is empty")
    return names


def _config_dataset_key(dataset: str) -> str:
    alias = normalize_psych101_dataset_alias(dataset)
    if is_mixed_gambles_dataset(alias):
        return "mixed_gambles"
    return alias


def _base_teh_path(
    baseline_config: Mapping[str, Any],
    dataset: str,
) -> Optional[str]:
    datasets = baseline_config.get("datasets")
    if not isinstance(datasets, dict):
        return None
    entry = datasets.get(_config_dataset_key(dataset))
    if not isinstance(entry, dict):
        return None
    raw = entry.get("TEH")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _ablation_run_dir(repo: Path, dataset: str, ablation: str) -> Path:
    alias = normalize_psych101_dataset_alias(dataset)
    root = repo / _ABLATION_OUTPUTS_DIR
    if is_mixed_gambles_dataset(alias):
        return root / "mixed_gambles" / "teh" / ablation
    return root / "psych101_train" / "teh" / alias / ablation


def _resolve_loglik_csv(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidate = path / _LOGLIK_CSV_NAME
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Expected a CSV file or a run directory containing {_LOGLIK_CSV_NAME}; got {path}"
    )


def _read_gated_loglik(csv_path: Path) -> Dict[int, float]:
    out: Dict[int, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        if "participant_id" not in reader.fieldnames or _GATED_LOGLIK not in reader.fieldnames:
            return out
        for row in reader:
            raw_pid = row.get("participant_id")
            raw_val = row.get(_GATED_LOGLIK)
            if raw_pid is None or str(raw_pid).strip() == "":
                continue
            if raw_val is None or str(raw_val).strip() == "":
                continue
            out[int(float(raw_pid))] = float(raw_val)
    return out


def _avg_gated_from_path(repo: Path, raw_path: Optional[str]) -> str:
    if not raw_path:
        return _NA
    try:
        csv_path = _resolve_loglik_csv(_resolve_repo_path(repo, raw_path))
    except (FileNotFoundError, OSError):
        return _NA
    scores = _read_gated_loglik(csv_path)
    if not scores:
        return _NA
    return f"{statistics.mean(scores.values()):.{_LOGLIK_NDIGITS}f}"


def _avg_gated_from_ablation(repo: Path, dataset: str, ablation: str) -> str:
    run_dir = _ablation_run_dir(repo, dataset, ablation)
    if not run_dir.is_dir():
        return _NA
    try:
        csv_path = _resolve_loglik_csv(run_dir)
    except FileNotFoundError:
        return _NA
    scores = _read_gated_loglik(csv_path)
    if not scores:
        return _NA
    return f"{statistics.mean(scores.values()):.{_LOGLIK_NDIGITS}f}"


def _build_row(
    repo: Path,
    dataset: str,
    baseline_config: Mapping[str, Any],
    ablation_names: Sequence[str],
) -> Dict[str, str]:
    row = {"dataset": dataset}
    row[_BASE_COL] = _avg_gated_from_path(repo, _base_teh_path(baseline_config, dataset))
    for name in ablation_names:
        row[name] = _avg_gated_from_ablation(repo, dataset, name)
    return row


def _format_table(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    if not rows:
        return "avg gated test log-likelihood\n  (no datasets)\n"

    ds_col_w = max(len("dataset"), max(len(str(r["dataset"])) for r in rows))
    col_widths = {col: max(len(col), 6) for col in columns}
    for row in rows:
        for col in columns:
            cell = str(row.get(col, _NA))
            col_widths[col] = max(col_widths[col], len(cell))

    header = f"{'dataset':<{ds_col_w}}"
    for col in columns:
        header += f"  {col:>{col_widths[col]}}"
    lines = ["avg gated test log-likelihood", header]

    for row in rows:
        line = f"{row['dataset']:<{ds_col_w}}"
        for col in columns:
            cell = str(row.get(col, _NA))
            line += f"  {cell:>{col_widths[col]}}"
        lines.append(line)

    avg_line = f"{'Avg':<{ds_col_w}}"
    for col in columns:
        vals: List[float] = []
        for row in rows:
            raw = row.get(col, _NA)
            if raw in (_NA, "", None):
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            vals.append(val)
        cell = _NA if not vals else f"{statistics.mean(vals):.{_LOGLIK_NDIGITS}f}"
        avg_line += f"  {cell:>{col_widths[col]}}"
    lines.append(avg_line)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TEH ablation avg gated test log-likelihood.")
    parser.add_argument(
        "--dataset",
        default=_DEFAULT_DATASET,
        help=f"Psych-101 dataset alias (default: {_DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--all_in",
        action="store_true",
        help=f"Compare all {_len_all_in()} EMNLP datasets (overrides --dataset).",
    )
    parser.add_argument(
        "--baseline_config",
        type=Path,
        default=Path(_DEFAULT_BASELINE_CONFIG),
        help=f"Baseline methods config with TEH paths (default: {_DEFAULT_BASELINE_CONFIG}).",
    )
    parser.add_argument(
        "--ablation_config",
        type=Path,
        default=Path(_DEFAULT_ABLATION_CONFIG),
        help=f"Ablation name list (default: {_DEFAULT_ABLATION_CONFIG}).",
    )
    args = parser.parse_args()

    repo = _REPO_ROOT
    baseline_path = (
        args.baseline_config.resolve()
        if args.baseline_config.is_absolute()
        else (repo / args.baseline_config).resolve()
    )
    ablation_path = (
        args.ablation_config.resolve()
        if args.ablation_config.is_absolute()
        else (repo / args.ablation_config).resolve()
    )

    baseline_config = _load_yaml(baseline_path)
    ablation_names = _load_ablation_names(ablation_path)
    columns = [_BASE_COL, *ablation_names]

    datasets = list(_ALL_IN_DATASETS) if args.all_in else [args.dataset.strip()]
    rows = [_build_row(repo, ds, baseline_config, ablation_names) for ds in datasets]
    print(_format_table(rows, columns))


def _len_all_in() -> int:
    return len(_ALL_IN_DATASETS)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
