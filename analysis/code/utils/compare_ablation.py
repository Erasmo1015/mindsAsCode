#!/usr/bin/env python3
"""
Compare avg gated test log-likelihood across TEH (PICS) ablation runs.

Usage:
  python analysis/code/utils/compare_ablation.py
  python analysis/code/utils/compare_ablation.py --dataset 1peterson2021using
  python analysis/code/utils/compare_ablation.py --all_in
  python analysis/code/utils/compare_ablation.py --all_in --config_path analysis/config/ablation/config.yaml

Rows are datasets; columns are base (full TEH) then each ablation listed under
``ablations:`` in the config. Paths come from ``datasets:`` in the same file
(PICS / TEH only). Missing runs show N/A.

Config shape (see analysis/config/ablation/config.yaml):

  ablations:
    - population
    - 2_exploration
    ...
  datasets:
    1peterson2021using:
      base: generated_outputs/psych101_train/teh/.../run_*
      population: generated_outputs_ablation/psych101_train/teh/.../population
      ...
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
_DEFAULT_ABLATION_CONFIG = "analysis/config/ablation/config.yaml"
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


def _load_ablation_names(config: Mapping[str, Any], config_path: Path) -> List[str]:
    raw = config.get("ablations")
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


def _dataset_entry(
    ablation_config: Mapping[str, Any],
    dataset: str,
) -> Dict[str, Any]:
    datasets = ablation_config.get("datasets")
    if not isinstance(datasets, dict):
        return {}
    entry = datasets.get(_config_dataset_key(dataset))
    return entry if isinstance(entry, dict) else {}


def _path_from_entry(entry: Mapping[str, Any], key: str) -> Optional[str]:
    raw = entry.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _default_ablation_run_dir(repo: Path, dataset: str, ablation: str) -> Path:
    """Conventional layout used when config omits a path."""
    alias = normalize_psych101_dataset_alias(dataset)
    root = repo / _ABLATION_OUTPUTS_DIR
    if is_mixed_gambles_dataset(alias):
        return root / "mixed_gambles" / "teh" / ablation
    return root / "psych101_train" / "teh" / alias / ablation


def _ablation_path(
    repo: Path,
    ablation_config: Mapping[str, Any],
    dataset: str,
    ablation: str,
) -> Optional[str]:
    entry = _dataset_entry(ablation_config, dataset)
    configured = _path_from_entry(entry, ablation)
    if configured:
        return configured
    run_dir = _default_ablation_run_dir(repo, dataset, ablation)
    if run_dir.is_dir():
        try:
            return str(run_dir.relative_to(repo))
        except ValueError:
            return str(run_dir)
    return None


def _base_path(
    ablation_config: Mapping[str, Any],
    dataset: str,
) -> Optional[str]:
    entry = _dataset_entry(ablation_config, dataset)
    for key in (_BASE_COL, "TEH", "PICS"):
        configured = _path_from_entry(entry, key)
        if configured:
            return configured
    return None


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


def _build_row(
    repo: Path,
    dataset: str,
    ablation_config: Mapping[str, Any],
    ablation_names: Sequence[str],
) -> Dict[str, str]:
    row = {"dataset": dataset}
    row[_BASE_COL] = _avg_gated_from_path(repo, _base_path(ablation_config, dataset))
    for name in ablation_names:
        row[name] = _avg_gated_from_path(
            repo, _ablation_path(repo, ablation_config, dataset, name)
        )
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
        "--config_path",
        "--ablation_config",
        type=Path,
        default=Path(_DEFAULT_ABLATION_CONFIG),
        dest="config_path",
        help=(
            "YAML with ablations list and datasets -> base / ablation TEH paths "
            f"(default: {_DEFAULT_ABLATION_CONFIG})."
        ),
    )
    args = parser.parse_args()

    repo = _REPO_ROOT
    config_path = (
        args.config_path.resolve()
        if args.config_path.is_absolute()
        else (repo / args.config_path).resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Ablation config not found: {config_path}")

    ablation_config = _load_yaml(config_path)
    ablation_names = _load_ablation_names(ablation_config, config_path)
    columns = [_BASE_COL, *ablation_names]

    datasets = list(_ALL_IN_DATASETS) if args.all_in else [args.dataset.strip()]
    rows = [_build_row(repo, ds, ablation_config, ablation_names) for ds in datasets]
    print(_format_table(rows, columns))


def _len_all_in() -> int:
    return len(_ALL_IN_DATASETS)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
