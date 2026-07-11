#!/usr/bin/env python3
"""
Before running, set up the baseline methods config file (default:
analysis/data/baseline_methods/config.yaml; override with --config_path).

Usage:
  python analysis/code/utils/compare.py --all_in --config_path analysis/data/baseline_methods/config_EMNLP_bef_review.yaml
  python analysis/code/utils/compare.py --dataset 2plonsky2018when --psych_dataset_split train
  python analysis/code/utils/compare.py --config_path analysis/data/baseline_methods/config_old.yaml

Compare per-participant test_loglik across baseline methods, optional Centaur, and TEH runs.

When ``--experiment_paths`` is omitted, the newest TEH ``run_*`` under
``generated_outputs/psych101_{train|test}/teh/<dataset>/`` (or
``generated_outputs/mixed_gambles/teh/``) is selected automatically.

``--all_in`` runs the default dataset subset in ``_ALL_IN_DATASETS`` and prints
a cross-dataset summary (avg test_loglik, avg gated, num_best per method, plus
PT-best/Ours-second and MLE-best/Ours-second counts with average gaps).

``--accuracy`` compares test accuracy instead of test_loglik (works with ``--all_in``).
Accuracy is loaded from ``participants_summary.csv`` when present, else
``participant_*/results.json`` (``test_accuracy`` / ``test_acc``), else Centaur
``log/predictions_vs_actual.csv``, else a cache under the run directory
(``participant_details_test_acc.csv``), else recomputed from
``participant_*/best_program.py`` and written back to that run cache.

Participant ids from CSVs are clamped to each dataset's supported ordinal range
(0..N-1 HF rows for Psych-101; 0..max subject for mixed_gambles).

Baseline paths: config.yaml only supplies optional manual overrides per method.
For each method, an explicit config path is used when set; otherwise the newest
``run_*`` under the standard ``generated_outputs/`` layout is auto-discovered
(see ``_auto_discover_baseline_run``). A dataset block in config is not required.
Config dataset keys (when overriding paths):

  datasets:
    1peterson2021using:          # psych_dataset_split train
      MLE: <run_dir_or_csv>
      prospect_theory: <run_dir_or_csv>
      openevolve: <run_dir_or_csv>
      Centaur: <run_dir_or_csv>
      TEH: <run_dir_or_csv>
    1peterson2021using_test:     # psych_dataset_split test
      ...
    2plonsky2018when:            # train
    2plonsky2018when_test:       # test
    mixed_gambles:               # split ignored

If neither ``{alias}_test`` nor ``mixed_gambles`` applies, split defaults to train
(``{alias}`` key). Legacy CLI names choice13k / cpc18 map to 1peterson2021using /
2plonsky2018when.

Output columns (when baselines are loaded from config):
  participant_id, BIR, MLE, prospect_theory, openevolve, Centaur, <teh_run_1>, ...
Footer rows: Avg (per-column mean), num_best (per-column count of tied-best test_loglik).

BIR is loaded from ``analysis/data/baseline_methods/bir/{config_key}.csv`` when
present (metadata must match ``--split_ratio`` / ``--split_seed``). Otherwise it is
computed on train trials with a progress bar and saved for the next run. Psych-101
train and test use separate cache files (e.g. ``1peterson2021using`` vs
``1peterson2021using_test``).

Baseline columns always use ``test_loglik`` from each method's config path. The gated
output CSV uses ``gated_test_loglik`` for TEH runs (and Centaur when present); other
baselines keep ``test_loglik`` because those runs have no gated column. Log-likelihood
Log-likelihood values are written with 2 decimal places.


"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
        return iterable

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PETERSON2021USING_ALIAS,
    PSYCH101_LEGACY_ALIASES,
    get_filtered_psych101_split,
    get_psych101_binary_experiment,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
    split_psych_experiment,
)
from analysis.code.choices13k.bir import compute_bir
from utils.teh.teh_datasets import PARTICIPANT_DATASETS, is_mixed_gambles_dataset

# Legacy compare.py dataset names -> TEH Psych-101 aliases (train split).
_COMPARE_LEGACY_DATASETS = {
    "choice13k": PETERSON2021USING_ALIAS,
    "cpc18": "2plonsky2018when",
}

_COMPARE_DATASET_CHOICES = sorted(
    PARTICIPANT_DATASETS | set(PSYCH101_LEGACY_ALIASES) | set(_COMPARE_LEGACY_DATASETS)
)

_TEST_LOGLIK = "test_loglik"
_GATED_LOGLIK = "gated_test_loglik"
_LOGLIK_NDIGITS = 2
_LOGLIK_WIN_THRESHOLD = -0.69
_TEST_ACC = "test_acc"
_TRAIN_ACC = "train_acc"
_ACC_NDIGITS = 4
_PARTICIPANTS_SUMMARY_CSV = "participants_summary.csv"
_ACC_CACHE_CSV_NAME = "participant_details_test_acc.csv"
_ACC_CACHE_META_SUFFIX = ".meta.json"
_ACC_CACHE_CSV_FIELDS = ("participant_id", "test_acc")
_BASELINE_METHODS = ("MLE", "prospect_theory", "openevolve", "Centaur")
_SUMMARY_METHOD_LABELS = _BASELINE_METHODS + ("TEH",)
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
# Temporary: hide these datasets from --all_in printed wide tables only.
_ALL_IN_PRINT_EXCLUDE: Tuple[str, ...] = ("8flesch2018comparing",)
_DEFAULT_ALL_IN_SUMMARY_CSV = "analysis/data/utils/loglik_compare_all_in_summary.csv"
_DEFAULT_ALL_IN_ACC_SUMMARY_CSV = "analysis/data/utils/acc_compare_all_in_summary.csv"
_DEFAULT_BASELINE_CONFIG = "analysis/data/baseline_methods/config.yaml"
_GENERATED_OUTPUTS_DIR = "generated_outputs"
_LOGLIK_CSV_NAME = "participant_details_loglik.csv"
_BIR_CACHE_SUBDIR = "bir"
_BIR_CACHE_CSV_FIELDS = (
    "participant_id",
    "BIR",
    "num_problem_groups",
    "num_inconsistent_problem_groups",
)


@dataclass(frozen=True)
class _DatasetDefaults:
    output_csv: Path


# Default output for 1peterson2021using --psych_dataset_split train only.
_PETERSON_TRAIN_OUTPUT_CSV = "analysis/data/utils/loglik_compare_choice13k.csv"

# Match Psych-101 baseline defaults (MLE / prospect_theory / openevolve / Centaur).
_DEFAULT_SPLIT_RATIO = 0.6
_DEFAULT_SPLIT_SEED = 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_compare_dataset(dataset: str) -> str:
    """TEH alias; accepts legacy choice13k/cpc18 and unprefixed Psych-101 names."""
    key = str(dataset).strip()
    if key in _COMPARE_LEGACY_DATASETS:
        return _COMPARE_LEGACY_DATASETS[key]
    return normalize_psych101_dataset_alias(key)


def _effective_psych_dataset_split(dataset: str, psych_dataset_split: str) -> str:
    if is_mixed_gambles_dataset(dataset):
        return DEFAULT_PSYCH_DATASET_SPLIT
    return normalize_psych_dataset_split(psych_dataset_split)


def _is_peterson_train(dataset: str, psych_dataset_split: str) -> bool:
    return (
        normalize_psych101_dataset_alias(dataset) == PETERSON2021USING_ALIAS
        and normalize_psych_dataset_split(psych_dataset_split) == "train"
    )


def _default_output_csv(
    repo: Path, dataset: str, psych_dataset_split: str, *, accuracy: bool = False
) -> Path:
    if accuracy:
        utils_dir = repo / "analysis" / "data" / "utils"
        if is_mixed_gambles_dataset(dataset):
            return utils_dir / "acc_compare_mixed_gambles.csv"
        split = normalize_psych_dataset_split(psych_dataset_split)
        alias = normalize_psych101_dataset_alias(dataset)
        return utils_dir / f"acc_compare_{alias}_{split}.csv"
    if _is_peterson_train(dataset, psych_dataset_split):
        return repo / _PETERSON_TRAIN_OUTPUT_CSV
    utils_dir = repo / "analysis" / "data" / "utils"
    if is_mixed_gambles_dataset(dataset):
        return utils_dir / "loglik_compare_mixed_gambles.csv"
    split = normalize_psych_dataset_split(psych_dataset_split)
    alias = normalize_psych101_dataset_alias(dataset)
    return utils_dir / f"loglik_compare_{alias}_{split}.csv"


def _config_dataset_key(dataset: str, psych_dataset_split: str) -> str:
    """YAML key under ``datasets:`` (mixed_gambles, {alias}_test, or {alias})."""
    alias = normalize_psych101_dataset_alias(dataset)
    if is_mixed_gambles_dataset(alias):
        return "mixed_gambles"
    if normalize_psych_dataset_split(psych_dataset_split) == "test":
        return f"{alias}_test"
    return alias


def _resolve_repo_path(repo: Path, raw: str) -> Path:
    p = Path(str(raw)).expanduser()
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def _load_baseline_config_file(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        return {}
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _method_output_folder(method: str) -> str:
    """Filesystem folder name under generated_outputs (Centaur -> centaur)."""
    if method == "Centaur":
        return "centaur"
    return method


def _psych101_outputs_split(psych_dataset_split: str) -> str:
    return f"psych101_{normalize_psych_dataset_split(psych_dataset_split)}"


def _config_dataset_entry(
    config_data: Mapping[str, Any],
    dataset: str,
    psych_dataset_split: str,
) -> Optional[Dict[str, Any]]:
    datasets = config_data.get("datasets")
    if not isinstance(datasets, dict):
        return None
    key = _config_dataset_key(dataset, psych_dataset_split)
    entry = datasets.get(key)
    return entry if isinstance(entry, dict) else None


def _baseline_search_roots(
    repo: Path,
    *,
    method: str,
    dataset: str,
    psych_dataset_split: str,
) -> List[Path]:
    """Candidate parent directories to scan for newest run_*."""
    alias = normalize_psych101_dataset_alias(dataset)
    method_dir = _method_output_folder(method)
    roots: List[Path] = []
    gen = repo / _GENERATED_OUTPUTS_DIR

    if is_mixed_gambles_dataset(alias):
        if method == "openevolve":
            roots.append(gen / "psych101_train" / "openevolve" / "mixed_gambles")
        roots.append(gen / "mixed_gambles" / method_dir)
        return roots

    split_dir = _psych101_outputs_split(psych_dataset_split)
    roots.append(gen / split_dir / method_dir / alias)
    return roots


def _is_valid_baseline_run_path(path: Path) -> bool:
    if path.is_file():
        return path.name == _LOGLIK_CSV_NAME
    if path.is_dir():
        return (path / _LOGLIK_CSV_NAME).is_file()
    return False


def _run_sort_key(path: Path) -> Tuple[str, float]:
    """Prefer newest run_* by name, then by mtime."""
    if path.is_file():
        run_name = path.parent.name
        mtime = path.stat().st_mtime
    else:
        run_name = path.name
        mtime = path.stat().st_mtime
    return run_name, mtime


def _collect_run_candidates(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    candidates: List[Path] = []
    for child in root.iterdir():
        if child.name.startswith("run_") and _is_valid_baseline_run_path(child):
            candidates.append(child)
    direct_csv = root / _LOGLIK_CSV_NAME
    if direct_csv.is_file():
        candidates.append(direct_csv)
    return candidates


def _latest_baseline_run_under_roots(roots: Sequence[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    for root in roots:
        candidates.extend(_collect_run_candidates(root))
    if not candidates:
        return None
    return max(candidates, key=_run_sort_key)


def _auto_discover_baseline_run(
    repo: Path,
    *,
    method: str,
    dataset: str,
    psych_dataset_split: str,
) -> Optional[Path]:
    roots = _baseline_search_roots(
        repo, method=method, dataset=dataset, psych_dataset_split=psych_dataset_split
    )
    return _latest_baseline_run_under_roots(roots)


def _teh_search_root(
    repo: Path,
    dataset: str,
    psych_dataset_split: str,
) -> Path:
    alias = normalize_psych101_dataset_alias(dataset)
    gen = repo / _GENERATED_OUTPUTS_DIR
    if is_mixed_gambles_dataset(alias):
        return gen / "mixed_gambles" / "teh"
    split_dir = _psych101_outputs_split(psych_dataset_split)
    return gen / split_dir / "teh" / alias


def _auto_discover_teh_run(
    repo: Path,
    *,
    dataset: str,
    psych_dataset_split: str,
) -> Optional[Path]:
    """Newest TEH run_* with participant_details_loglik.csv (by run name, then mtime)."""
    root = _teh_search_root(repo, dataset, psych_dataset_split)
    return _latest_baseline_run_under_roots([root])


def _resolve_teh_run_path(
    config_data: Mapping[str, Any],
    repo: Path,
    dataset: str,
    psych_dataset_split: str,
    *,
    quiet: bool = False,
) -> Optional[Path]:
    """Config TEH path overrides auto-discovery when set."""
    entry = _config_dataset_entry(config_data, dataset, psych_dataset_split)
    raw = entry.get("TEH") if entry is not None else None
    config_key = _config_dataset_key(dataset, psych_dataset_split)
    if raw is not None and str(raw).strip() != "":
        path = _resolve_repo_path(repo, str(raw))
        if not quiet:
            print(
                f"Using TEH from config for {config_key}: {path.relative_to(repo)}"
            )
        return path
    discovered = _auto_discover_teh_run(
        repo, dataset=dataset, psych_dataset_split=psych_dataset_split
    )
    if discovered is not None:
        teh_root = _teh_search_root(repo, dataset, psych_dataset_split)
        if not quiet:
            print(
                f"Auto-selected TEH for {config_key}: "
                f"{discovered.relative_to(repo)} "
                f"(newest run_* in {teh_root.relative_to(repo)})"
            )
        return discovered
    teh_root = _teh_search_root(repo, dataset, psych_dataset_split)
    print(
        f"Warning: no TEH run found for {config_key}; "
        f"searched {teh_root.relative_to(repo)}; continuing without TEH.",
        file=sys.stderr,
    )
    return None


def _resolve_baseline_run_paths(
    config_data: Mapping[str, Any],
    repo: Path,
    dataset: str,
    psych_dataset_split: str,
    *,
    quiet: bool = False,
) -> Dict[str, Path]:
    """
    Method name -> run dir or CSV.

    Config paths override auto-discovery when set; otherwise always auto-discovers.
    """
    entry = _config_dataset_entry(config_data, dataset, psych_dataset_split)
    out: Dict[str, Path] = {}
    config_key = _config_dataset_key(dataset, psych_dataset_split)
    for method in _BASELINE_METHODS:
        raw = entry.get(method) if entry is not None else None
        if raw is not None and str(raw).strip() != "":
            out[method] = _resolve_repo_path(repo, str(raw))
            continue
        discovered = _auto_discover_baseline_run(
            repo, method=method, dataset=dataset, psych_dataset_split=psych_dataset_split
        )
        if discovered is not None:
            out[method] = discovered
            if not quiet:
                print(
                    f"Auto-selected {method} for {config_key}: "
                    f"{discovered.relative_to(repo)}"
                )
        else:
            roots = _baseline_search_roots(
                repo,
                method=method,
                dataset=dataset,
                psych_dataset_split=psych_dataset_split,
            )
            searched = ", ".join(str(r.relative_to(repo)) for r in roots)
            print(
                f"Warning: no run found for {method} ({config_key}); "
                f"searched [{searched}]; skipping.",
                file=sys.stderr,
            )
    return out


def _baseline_run_paths_from_config(
    config_data: Mapping[str, Any],
    repo: Path,
    dataset: str,
    psych_dataset_split: str,
) -> Dict[str, Path]:
    """Backward-compatible alias."""
    return _resolve_baseline_run_paths(
        config_data, repo, dataset, psych_dataset_split
    )


def _dataset_defaults(
    repo: Path, dataset: str, psych_dataset_split: str, *, accuracy: bool = False
) -> _DatasetDefaults:
    alias = _normalize_compare_dataset(dataset)
    if alias not in PARTICIPANT_DATASETS:
        raise ValueError(
            f"dataset must be a TEH participant dataset {sorted(PARTICIPANT_DATASETS)} "
            f"or legacy alias in {sorted(_COMPARE_LEGACY_DATASETS)}, got {dataset!r}"
        )
    split = _effective_psych_dataset_split(alias, psych_dataset_split)
    return _DatasetDefaults(
        output_csv=_default_output_csv(repo, alias, split, accuracy=accuracy)
    )


def _resolve_loglik_csv(path: Path) -> Path:
    """Accept either a run directory or a path to participant_details_loglik.csv."""
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidate = path / _LOGLIK_CSV_NAME
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Expected a CSV file or a run directory containing {_LOGLIK_CSV_NAME}; got {path}"
    )


def _run_dir_from_path(run_or_csv: Path) -> Path:
    csv_path = _resolve_loglik_csv(run_or_csv)
    return csv_path.parent


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


def _format_loglik(value: float) -> str:
    return f"{value:.{_LOGLIK_NDIGITS}f}"


def _format_acc(value: float) -> str:
    return f"{value:.{_ACC_NDIGITS}f}"


def _read_participant_ids_from_csv(csv_path: Path) -> List[int]:
    """All participant_id values from a loglik CSV, in file order (deduped)."""
    ids: List[int] = []
    seen: set[int] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path}: empty CSV")
        if "participant_id" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: missing participant_id column (got {reader.fieldnames})"
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
        raise ValueError(f"{csv_path}: no participant_id rows found")
    return ids


def _read_centaur_participant_ids(centaur_path: Path) -> List[int]:
    """Backward-compatible alias for participant roster from Centaur CSV."""
    return _read_participant_ids_from_csv(centaur_path)


def _participant_ids_from_experiment_csvs(csv_paths: Sequence[Path]) -> List[int]:
    """Participant ids in file order from experiment CSVs (first file, then extras)."""
    ids: List[int] = []
    seen: set[int] = set()
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "participant_id" not in reader.fieldnames:
                continue
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
        raise ValueError("No participant_id rows found in experiment CSVs")
    return ids


def _mixed_gambles_max_participant_index(
    mixed_gambles_csv: str,
    *,
    filter_gain_loss_only: bool,
) -> int:
    """Largest subject id in the mixed_gambles CSV (inclusive upper bound for ordinals)."""
    max_id = -1
    with open(mixed_gambles_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "subject" not in reader.fieldnames:
            raise ValueError(
                f"{mixed_gambles_csv}: missing subject column (got {reader.fieldnames})"
            )
        for row in reader:
            if filter_gain_loss_only and row.get("gamble_type") != "gain_loss":
                continue
            sid = int(float(row["subject"]))
            if sid > max_id:
                max_id = sid
    if max_id < 0:
        raise ValueError(f"{mixed_gambles_csv}: no subject rows found")
    return max_id


def _dataset_participant_ordinal_bounds(
    dataset: str,
    *,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> Tuple[int, int]:
    """
    Inclusive ordinal range supported by the dataset corpus (0-based row/subject ids).

    Psych-101: filtered HF rows for the experiment. mixed_gambles: max subject in CSV.
    """
    if is_mixed_gambles_dataset(dataset):
        return 0, _mixed_gambles_max_participant_index(
            mixed_gambles_csv, filter_gain_loss_only=filter_mixed_gambles
        )
    alias = normalize_psych101_dataset_alias(dataset)
    filtered = get_filtered_psych101_split(
        alias, split=psych_dataset_split, local_dataset=local_dataset
    )
    n_rows = len(filtered)
    if n_rows == 0:
        raise ValueError(
            f"No HF rows for {alias!r} psych_dataset_split={psych_dataset_split!r}"
        )
    return 0, n_rows - 1


def _clamp_participant_ids_to_dataset(
    participant_ids: Sequence[int],
    *,
    dataset: str,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
    ordinal_bounds: Optional[Tuple[int, int]] = None,
) -> Tuple[List[int], int, int]:
    """
    Drop participant ids outside the dataset's supported ordinal range.

    Prevents IndexError / empty BIR when CSVs list ids from a larger default range
    (e.g. 0--49) than the dataset has (e.g. 5speekenbrink2008learning: 0--22).

    Returns (kept_ids, ord_min, ord_max).
    """
    if ordinal_bounds is None:
        ord_min, ord_max = _dataset_participant_ordinal_bounds(
            dataset,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            filter_mixed_gambles=filter_mixed_gambles,
        )
    else:
        ord_min, ord_max = ordinal_bounds
    kept_set: set[int] = set()
    kept: List[int] = []
    for pid in participant_ids:
        ipid = int(pid)
        if ord_min <= ipid <= ord_max and ipid not in kept_set:
            kept_set.add(ipid)
            kept.append(ipid)
    dropped = [int(pid) for pid in participant_ids if int(pid) not in kept_set]
    if dropped:
        label = (
            dataset
            if is_mixed_gambles_dataset(dataset)
            else f"{dataset} ({psych_dataset_split})"
        )
        print(
            f"Warning: dropped {len(dropped)} participant_id(s) outside ordinal range "
            f"[{ord_min}, {ord_max}] for {label}: "
            f"{dropped[:10]}{'...' if len(dropped) > 10 else ''}",
            file=sys.stderr,
        )
    if not kept:
        raise SystemExit(
            f"No participant_id values in supported ordinal range [{ord_min}, {ord_max}] "
            f"for {dataset!r}."
        )
    return kept, ord_min, ord_max


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


def _read_numeric_csv_column(csv_path: Path, column: str, *, required: bool) -> Dict[int, float]:
    """Read one numeric column keyed by participant_id."""
    return _read_loglik_csv(csv_path, column, required=required)


def _load_accuracy_from_participant_results(
    run_dir: Path,
    *,
    json_keys: Sequence[str],
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for pdir in sorted(run_dir.glob("participant_*")):
        if not pdir.is_dir():
            continue
        try:
            pid = int(pdir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        results_path = pdir / "results.json"
        if not results_path.is_file():
            continue
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for key in json_keys:
            raw = data.get(key)
            if raw is None or str(raw).strip() == "":
                continue
            out[pid] = float(raw)
            break
    return out


def _load_centaur_test_accuracy(run_dir: Path) -> Dict[int, float]:
    path = run_dir / "log" / "predictions_vs_actual.csv"
    if not path.is_file():
        return {}
    totals: Dict[int, int] = {}
    correct: Dict[int, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        for row in reader:
            if row.get("split") != "test":
                continue
            raw_pid = row.get("participant_id")
            if raw_pid is None or str(raw_pid).strip() == "":
                continue
            pid = int(float(raw_pid))
            try:
                pred = int(float(row.get("pred_action", "")))
                actual = int(float(row.get("actual_action", "")))
            except (TypeError, ValueError):
                continue
            totals[pid] = totals.get(pid, 0) + 1
            if pred == actual:
                correct[pid] = correct.get(pid, 0) + 1
    return {
        pid: correct[pid] / totals[pid]
        for pid in totals
        if totals[pid] > 0 and pid in correct
    }


def _accuracy_cache_csv_path(run_dir: Path) -> Path:
    return run_dir / _ACC_CACHE_CSV_NAME


def _accuracy_cache_meta_path(cache_csv: Path) -> Path:
    return cache_csv.with_name(cache_csv.stem + _ACC_CACHE_META_SUFFIX)


def _accuracy_cache_meta(
    *,
    dataset: str,
    psych_dataset_split: str,
    split_ratio: float,
    split_seed: int,
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
    local_dataset: Optional[str],
) -> Dict[str, Any]:
    alias = (
        normalize_psych101_dataset_alias(dataset)
        if not is_mixed_gambles_dataset(dataset)
        else dataset
    )
    return {
        "dataset": alias,
        "psych_dataset_split": _effective_psych_dataset_split(dataset, psych_dataset_split),
        "config_key": _config_dataset_key(dataset, psych_dataset_split),
        "split_ratio": float(split_ratio),
        "split_seed": int(split_seed),
        "mixed_gambles_csv": str(mixed_gambles_csv),
        "filter_mixed_gambles": bool(filter_mixed_gambles),
        "local_dataset": local_dataset,
        "source": "compare.py --accuracy recompute from participant_*/best_program.py",
    }


def _accuracy_meta_matches(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = (
        "dataset",
        "psych_dataset_split",
        "config_key",
        "split_ratio",
        "split_seed",
        "mixed_gambles_csv",
        "filter_mixed_gambles",
        "local_dataset",
    )
    for key in keys:
        if saved.get(key) != expected.get(key):
            return False
    return True


def _read_accuracy_cache_rows(cache_csv: Path) -> Dict[int, float]:
    if not cache_csv.is_file():
        return {}
    return _read_numeric_csv_column(cache_csv, _TEST_ACC, required=False)


def _write_accuracy_cache(
    cache_csv: Path,
    meta_path: Path,
    *,
    scores_by_pid: Mapping[int, float],
    meta: Mapping[str, Any],
) -> None:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"participant_id": str(int(pid)), "test_acc": _format_acc(float(acc))}
        for pid, acc in sorted(scores_by_pid.items())
    ]
    with cache_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_ACC_CACHE_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    meta_path.write_text(json.dumps(dict(meta), indent=2), encoding="utf-8")


def _recompute_test_accuracy_for_participant(
    run_dir: Path,
    participant_id: int,
    *,
    dataset: str,
    psych_dataset_split: str,
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> Optional[float]:
    from teh import compile_program, evaluate_choice13k_program

    prog_path = run_dir / f"participant_{participant_id}" / "best_program.py"
    if not prog_path.is_file():
        return None
    choose_fn = compile_program(prog_path.read_text(encoding="utf-8"))
    if choose_fn is None:
        return None
    if is_mixed_gambles_dataset(dataset):
        _, _, test_trials, _ = load_mixed_gambles_trials(
            participant_id,
            csv_path=mixed_gambles_csv,
            filter_gain_loss_only=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
    else:
        alias = normalize_psych101_dataset_alias(dataset)
        exp = get_psych101_binary_experiment(
            alias,
            int(participant_id),
            split=psych_dataset_split,
            local_dataset=local_dataset,
        )
        _, _, test_trials, _ = split_psych_experiment(
            exp, split_ratio=split_ratio, split_seed=split_seed
        )
    if not test_trials:
        return None
    eval_result = evaluate_choice13k_program(choose_fn, test_trials, n_seeds=1)
    return float(eval_result["accuracy"])


def _load_or_recompute_test_accuracy_from_best_programs(
    run_dir: Path,
    *,
    dataset: str,
    psych_dataset_split: str,
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
    participant_ids: Optional[Sequence[int]] = None,
    recompute: bool = False,
    quiet: bool = False,
) -> Dict[int, float]:
    cache_csv = _accuracy_cache_csv_path(run_dir)
    meta_path = _accuracy_cache_meta_path(cache_csv)
    meta = _accuracy_cache_meta(
        dataset=dataset,
        psych_dataset_split=psych_dataset_split,
        split_ratio=split_ratio,
        split_seed=split_seed,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
        local_dataset=local_dataset,
    )

    scores: Dict[int, float] = {}
    if not recompute and cache_csv.is_file() and meta_path.is_file():
        try:
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved_meta = {}
        if _accuracy_meta_matches(saved_meta, meta):
            scores = _read_accuracy_cache_rows(cache_csv)
            if scores and not quiet:
                print(
                    f"Loaded test accuracy cache ({len(scores)} participants) from {cache_csv}"
                )

    roster: Optional[List[int]] = None
    if participant_ids is not None:
        roster = sorted({int(pid) for pid in participant_ids})

    if roster is not None:
        missing = [pid for pid in roster if pid not in scores]
    else:
        missing = []
        for pdir in sorted(run_dir.glob("participant_*")):
            if not pdir.is_dir():
                continue
            try:
                pid = int(pdir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if pid not in scores:
                missing.append(pid)

    if recompute:
        if roster is not None:
            missing = roster
        else:
            missing = []
            for pdir in sorted(run_dir.glob("participant_*")):
                if not pdir.is_dir():
                    continue
                try:
                    pid = int(pdir.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                missing.append(pid)
        scores = {}

    if missing:
        label = meta["config_key"]
        if not quiet:
            print(
                f"Computing test accuracy for {len(missing)} participant(s) "
                f"({label}, split_ratio={split_ratio}, split_seed={split_seed})..."
            )
        for pid in tqdm(
            missing, desc=f"test_acc {run_dir.name}", unit="participant", disable=quiet
        ):
            acc = _recompute_test_accuracy_for_participant(
                run_dir,
                int(pid),
                dataset=dataset,
                psych_dataset_split=psych_dataset_split,
                split_ratio=split_ratio,
                split_seed=split_seed,
                local_dataset=local_dataset,
                mixed_gambles_csv=mixed_gambles_csv,
                filter_mixed_gambles=filter_mixed_gambles,
            )
            if acc is not None:
                scores[int(pid)] = acc
        _write_accuracy_cache(cache_csv, meta_path, scores_by_pid=scores, meta=meta)
        if not quiet:
            print(f"Wrote test accuracy cache ({len(scores)} participants) -> {cache_csv}")

    if roster is not None:
        return {pid: scores[pid] for pid in roster if pid in scores}
    return scores


def _load_accuracy_scores_from_run(
    run_or_csv: Path,
    *,
    method: str,
    dataset: str,
    psych_dataset_split: str,
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
    participant_ids: Optional[Sequence[int]] = None,
    recompute: bool = False,
    quiet: bool = False,
) -> Dict[int, float]:
    """Load per-participant test accuracy from run artifacts (summary, results, cache, or recompute)."""
    run_dir = _run_dir_from_path(run_or_csv)
    summary_path = run_dir / _PARTICIPANTS_SUMMARY_CSV
    if summary_path.is_file():
        for column in (_TEST_ACC, "test_accuracy"):
            scores = _read_numeric_csv_column(summary_path, column, required=False)
            if scores:
                return scores

    json_keys = (_TEST_ACC, "test_accuracy")
    scores = _load_accuracy_from_participant_results(run_dir, json_keys=json_keys)
    if scores:
        return scores

    if method == "Centaur":
        scores = _load_centaur_test_accuracy(run_dir)
        if scores:
            return scores

    return _load_or_recompute_test_accuracy_from_best_programs(
        run_dir,
        dataset=dataset,
        psych_dataset_split=psych_dataset_split,
        split_ratio=split_ratio,
        split_seed=split_seed,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
        participant_ids=participant_ids,
        recompute=recompute,
        quiet=quiet,
    )


def _train_trials_for_participant(
    dataset: str,
    participant_id: int,
    *,
    psych_dataset_split: str,
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
) -> List[Dict[str, Any]]:
    """Train trials for one participant (same split logic as TEH / Psych-101 baselines)."""
    if is_mixed_gambles_dataset(dataset):
        train_trials, _, _, _ = load_mixed_gambles_trials(
            participant_id,
            csv_path=mixed_gambles_csv,
            filter_gain_loss_only=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        return train_trials
    alias = normalize_psych101_dataset_alias(dataset)
    exp = get_psych101_binary_experiment(
        alias,
        int(participant_id),
        split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    train_trials, _, _, _ = split_psych_experiment(
        exp, split_ratio=split_ratio, split_seed=split_seed
    )
    return train_trials


def _bir_cache_dir(repo: Path) -> Path:
    return repo / "analysis" / "data" / "baseline_methods" / _BIR_CACHE_SUBDIR


def _bir_cache_csv_path(repo: Path, dataset: str, psych_dataset_split: str) -> Path:
    key = _config_dataset_key(dataset, psych_dataset_split)
    return _bir_cache_dir(repo) / f"{key}.csv"


def _bir_cache_meta_path(cache_csv: Path) -> Path:
    return cache_csv.with_suffix(".meta.json")


def _bir_cache_meta(
    *,
    dataset: str,
    psych_dataset_split: str,
    split_ratio: float,
    split_seed: int,
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
    local_dataset: Optional[str],
) -> Dict[str, Any]:
    alias = normalize_psych101_dataset_alias(dataset) if not is_mixed_gambles_dataset(dataset) else dataset
    return {
        "dataset": alias,
        "psych_dataset_split": _effective_psych_dataset_split(dataset, psych_dataset_split),
        "config_key": _config_dataset_key(dataset, psych_dataset_split),
        "split_ratio": float(split_ratio),
        "split_seed": int(split_seed),
        "mixed_gambles_csv": str(mixed_gambles_csv),
        "filter_mixed_gambles": bool(filter_mixed_gambles),
        "local_dataset": local_dataset,
    }


def _bir_meta_matches(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = (
        "dataset",
        "psych_dataset_split",
        "config_key",
        "split_ratio",
        "split_seed",
        "mixed_gambles_csv",
        "filter_mixed_gambles",
        "local_dataset",
    )
    for k in keys:
        if saved.get(k) != expected.get(k):
            return False
    return True


def _read_bir_cache_rows(cache_csv: Path) -> Dict[int, Dict[str, Any]]:
    if not cache_csv.is_file():
        return {}
    rows: Dict[int, Dict[str, Any]] = {}
    with open(cache_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "participant_id" not in reader.fieldnames:
            return {}
        for row in reader:
            pid_raw = row.get("participant_id")
            bir_raw = row.get("BIR")
            if pid_raw is None or bir_raw is None or str(pid_raw).strip() == "" or str(bir_raw).strip() == "":
                continue
            pid = int(float(pid_raw))
            rows[pid] = {
                "participant_id": pid,
                "BIR": float(bir_raw),
                "num_problem_groups": int(float(row.get("num_problem_groups") or 0)),
                "num_inconsistent_problem_groups": int(
                    float(row.get("num_inconsistent_problem_groups") or 0)
                ),
            }
    return rows


def _write_bir_cache(
    cache_csv: Path,
    meta_path: Path,
    *,
    rows_by_pid: Dict[int, Dict[str, Any]],
    meta: Dict[str, Any],
) -> None:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, Any]] = []
    for pid in sorted(rows_by_pid):
        r = rows_by_pid[pid]
        out_rows.append(
            {
                "participant_id": str(int(r["participant_id"])),
                "BIR": f"{float(r['BIR']):.4f}",
                "num_problem_groups": str(int(r.get("num_problem_groups", 0))),
                "num_inconsistent_problem_groups": str(
                    int(r.get("num_inconsistent_problem_groups", 0))
                ),
            }
        )
    with open(cache_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_BIR_CACHE_CSV_FIELDS))
        w.writeheader()
        w.writerows(out_rows)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _bir_for_participant(
    dataset: str,
    participant_id: int,
    *,
    psych_dataset_split: str,
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> Optional[Dict[str, Any]]:
    try:
        train_trials = _train_trials_for_participant(
            dataset,
            int(participant_id),
            psych_dataset_split=psych_dataset_split,
            split_ratio=split_ratio,
            split_seed=split_seed,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            filter_mixed_gambles=filter_mixed_gambles,
        )
        bir_val, n_groups, n_incon = compute_bir(train_trials)
        return {
            "participant_id": int(participant_id),
            "BIR": float(bir_val),
            "num_problem_groups": int(n_groups),
            "num_inconsistent_problem_groups": int(n_incon),
        }
    except Exception:
        return None


def _load_or_compute_bir_map(
    repo: Path,
    *,
    dataset: str,
    psych_dataset_split: str,
    participant_ids: Sequence[int],
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    filter_mixed_gambles: bool = False,
    recompute: bool = False,
    quiet: bool = False,
) -> Dict[int, float]:
    """
    Load cached BIR for this dataset config key, or compute missing participants.

    Cache lives under analysis/data/baseline_methods/bir/{config_key}.csv with a
    sidecar .meta.json (split_ratio, split_seed, psych_dataset_split, etc.).
    """
    cache_csv = _bir_cache_csv_path(repo, dataset, psych_dataset_split)
    meta_path = _bir_cache_meta_path(cache_csv)
    meta = _bir_cache_meta(
        dataset=dataset,
        psych_dataset_split=psych_dataset_split,
        split_ratio=split_ratio,
        split_seed=split_seed,
        mixed_gambles_csv=mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
        local_dataset=local_dataset,
    )

    rows_by_pid: Dict[int, Dict[str, Any]] = {}
    if not recompute and cache_csv.is_file() and meta_path.is_file():
        try:
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved_meta = {}
        if _bir_meta_matches(saved_meta, meta):
            rows_by_pid = _read_bir_cache_rows(cache_csv)

    roster = sorted({int(p) for p in participant_ids})
    missing = [pid for pid in roster if pid not in rows_by_pid]

    if missing:
        label = meta["config_key"]
        if not quiet:
            print(
                f"Computing BIR for {len(missing)} participant(s) "
                f"({label}, split_ratio={split_ratio}, split_seed={split_seed})..."
            )
        for pid in tqdm(
            missing, desc=f"BIR {label}", unit="participant", disable=quiet
        ):
            row = _bir_for_participant(
                dataset,
                pid,
                psych_dataset_split=psych_dataset_split,
                split_ratio=split_ratio,
                split_seed=split_seed,
                local_dataset=local_dataset,
                mixed_gambles_csv=mixed_gambles_csv,
                filter_mixed_gambles=filter_mixed_gambles,
            )
            if row is not None:
                rows_by_pid[pid] = row
        _write_bir_cache(cache_csv, meta_path, rows_by_pid=rows_by_pid, meta=meta)
        if not quiet:
            print(f"Wrote BIR cache ({len(rows_by_pid)} participants) -> {cache_csv}")
    elif rows_by_pid and not quiet:
        print(f"Loaded BIR cache ({len(rows_by_pid)} participants) from {cache_csv}")

    return {pid: float(rows_by_pid[pid]["BIR"]) for pid in roster if pid in rows_by_pid}


def _finite_mean(values: Iterable[float], ndigits: int = 2) -> str:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return ""
    return f"{statistics.mean(vals):.{ndigits}f}"


_NA = "NA"


def _comparison_metric_cell(
    row_map: Mapping[str, Any],
    column: str,
    *,
    found_methods: set[str],
) -> str:
    if column not in found_methods:
        return _NA
    value = row_map.get(column)
    if value is None or value == "":
        return _NA
    return str(value)


def _format_metrics_table(
    *,
    title: str,
    columns: Sequence[str],
    metric_rows: Sequence[Tuple[str, Mapping[str, Any]]],
    cell_fn: Any,
) -> str:
    if not columns:
        return f"{title}\n  (no methods)\n"
    metric_col_w = max(len("metric"), *(len(label) for label, _ in metric_rows))
    col_widths = {column: max(len(column), 6) for column in columns}
    header = f"{'metric':<{metric_col_w}}"
    for column in columns:
        header += f"  {column:>{col_widths[column]}}"
    lines = [title, header]
    for row_label, row_map in metric_rows:
        line = f"{row_label:<{metric_col_w}}"
        for column in columns:
            cell = cell_fn(row_map, column)
            line += f"  {cell:>{col_widths[column]}}"
        lines.append(line)
    return "\n".join(lines) + "\n"


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


def _any_csv_has_gated(
    centaur_path: Optional[Path], experiment_csvs: Sequence[Path]
) -> bool:
    if centaur_path is not None and _csv_has_column(centaur_path, _GATED_LOGLIK):
        return True
    return any(_csv_has_column(p, _GATED_LOGLIK) for p in experiment_csvs)


def _scores_for_gated_table(
    test_scores: Dict[int, float],
    gated_scores: Dict[int, float],
) -> Dict[int, float]:
    """Baseline/TEH scores for gated table: gated when available, else test_loglik."""
    merged = dict(test_scores)
    merged.update(gated_scores)
    return merged


def _load_scores_from_run(run_or_csv: Path, column: str, *, required: bool) -> Dict[int, float]:
    csv_path = _resolve_loglik_csv(run_or_csv)
    return _read_loglik_csv(csv_path, column, required=required)


def _num_best_counts(
    participant_ids: Sequence[int],
    method_columns: Sequence[Tuple[str, Dict[int, float]]],
) -> Dict[str, int]:
    """
    Per-column count of participants where this method tied for best test_loglik.

    Ties: every method at the maximum receives +1 (not a unique winner).
    """
    counts = {label: 0 for label, _ in method_columns}
    for pid in participant_ids:
        vals = [
            (label, m[pid])
            for label, m in method_columns
            if pid in m and math.isfinite(m[pid])
        ]
        if not vals:
            continue
        best = max(v for _, v in vals)
        for label, v in vals:
            if v >= best:
                counts[label] += 1
    return counts


def _count_above_threshold(
    participant_ids: Sequence[int],
    method_columns: Sequence[Tuple[str, Dict[int, float]]],
    *,
    threshold: float,
) -> Dict[str, int]:
    """Per-method participant count with finite score > threshold."""
    counts = {label: 0 for label, _ in method_columns}
    for pid in participant_ids:
        for label, scores in method_columns:
            if pid not in scores:
                continue
            value = scores[pid]
            if math.isfinite(value) and value > threshold:
                counts[label] += 1
    return counts


def _best_method_ours_second_stats(
    participant_ids: Sequence[int],
    method_columns: Sequence[Tuple[str, Dict[int, float]]],
    *,
    best_label: str,
    ours_label: str,
) -> Tuple[int, str]:
    """
    Count participants where best_label is best and Ours (TEH) is second.

    "Best" allows ties for best_label at the maximum score.
    "Second" allows ties at the second-highest score among non-best methods.
    """
    method_map = {label: scores for label, scores in method_columns}
    best_scores = method_map.get(best_label, {})
    ours_scores = method_map.get(ours_label, {})
    count = 0
    gaps: List[float] = []
    for pid in participant_ids:
        if pid not in best_scores or pid not in ours_scores:
            continue
        best_val = best_scores[pid]
        ours_val = ours_scores[pid]
        if not (math.isfinite(best_val) and math.isfinite(ours_val)):
            continue
        vals = {
            label: scores[pid]
            for label, scores in method_columns
            if pid in scores and math.isfinite(scores[pid])
        }
        if best_label not in vals or ours_label not in vals:
            continue
        top_val = max(vals.values())
        if vals[best_label] < top_val:
            continue
        if vals[ours_label] >= vals[best_label]:
            continue
        non_best = [v for label, v in vals.items() if label != best_label]
        if not non_best:
            continue
        second_val = max(non_best)
        if vals[ours_label] == second_val:
            count += 1
            gaps.append(vals[best_label] - vals[ours_label])
    return count, _finite_mean(gaps, ndigits=_LOGLIK_NDIGITS)


def _best_method_avg_gap(
    participant_ids: Sequence[int],
    method_columns: Sequence[Tuple[str, Dict[int, float]]],
    *,
    best_label: str,
    ours_label: str,
) -> str:
    """Average (best_label - Ours) over participants where best_label is best."""
    method_map = {label: scores for label, scores in method_columns}
    best_scores = method_map.get(best_label, {})
    ours_scores = method_map.get(ours_label, {})
    gaps: List[float] = []
    for pid in participant_ids:
        if pid not in best_scores or pid not in ours_scores:
            continue
        best_val = best_scores[pid]
        ours_val = ours_scores[pid]
        if not (math.isfinite(best_val) and math.isfinite(ours_val)):
            continue
        vals = {
            label: scores[pid]
            for label, scores in method_columns
            if pid in scores and math.isfinite(scores[pid])
        }
        if best_label not in vals or ours_label not in vals:
            continue
        top_val = max(vals.values())
        if vals[best_label] < top_val:
            continue
        gaps.append(vals[best_label] - vals[ours_label])
    return _finite_mean(gaps, ndigits=_LOGLIK_NDIGITS)


def _write_baseline_comparison_csv(
    *,
    out_path: Path,
    participant_ids: Sequence[int],
    baselines: Sequence[Tuple[str, Dict[int, float]]],
    teh_runs: Sequence[Tuple[str, Dict[int, float]]],
    bir: Dict[int, float],
    format_score: Any = _format_loglik,
    score_ndigits: int = _LOGLIK_NDIGITS,
) -> Tuple[int, Dict[str, str], Dict[str, int]]:
    """Write MLE / prospect_theory / openevolve / Centaur / TEH table with Avg and num_best rows."""
    ordered = list(participant_ids)
    score_columns = list(baselines) + list(teh_runs)
    fieldnames = ["participant_id", "BIR"] + [label for label, _ in score_columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for pid in ordered:
        row: Dict[str, str] = {
            "participant_id": str(pid),
            "BIR": f"{bir[pid]:.2f}" if pid in bir else "",
        }
        for label, m in score_columns:
            row[label] = format_score(m[pid]) if pid in m else ""
        rows.append(row)

    avg_row: Dict[str, str] = {"participant_id": "Avg", "BIR": ""}
    if bir:
        avg_row["BIR"] = _finite_mean([bir[pid] for pid in ordered if pid in bir])
    for label, m in score_columns:
        avg_row[label] = _finite_mean(
            [m[pid] for pid in ordered if pid in m], ndigits=score_ndigits
        )
    rows.append(avg_row)

    best_counts = _num_best_counts(ordered, score_columns)
    best_row: Dict[str, str] = {fn: "" for fn in fieldnames}
    best_row["participant_id"] = "num_best"
    for label, _ in score_columns:
        best_row[label] = str(best_counts[label])
    rows.append(best_row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return len(ordered), avg_row, best_counts


def _write_legacy_comparison_csv(
    *,
    out_path: Path,
    participant_ids: Sequence[int],
    centaur: Dict[int, float],
    experiments: Sequence[Tuple[str, Dict[int, float]]],
    bir: Dict[int, float],
    similar_threshold: float,
) -> int:
    """Legacy Centaur + TEH table with Avg and Better / Similar / Worse footers."""
    ordered = list(participant_ids)

    fieldnames = ["participant_id", "BIR", "Centaur"] + [label for label, _ in experiments]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for pid in ordered:
        row: Dict[str, str] = {
            "participant_id": str(pid),
            "BIR": f"{bir[pid]:.2f}" if pid in bir else "",
            "Centaur": _format_loglik(centaur[pid]) if pid in centaur else "",
        }
        for label, m in experiments:
            row[label] = _format_loglik(m[pid]) if pid in m else ""
        rows.append(row)

    avg_row: Dict[str, str] = {"participant_id": "Avg", "BIR": "", "Centaur": ""}
    if bir:
        avg_row["BIR"] = _finite_mean([bir[pid] for pid in ordered if pid in bir])
    avg_row["Centaur"] = _finite_mean(
        [centaur[pid] for pid in ordered if pid in centaur], ndigits=_LOGLIK_NDIGITS
    )
    for label, m in experiments:
        avg_row[label] = _finite_mean(
            [m[pid] for pid in ordered if pid in m], ndigits=_LOGLIK_NDIGITS
        )
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


def _print_comparison_metrics_table(
    *,
    test_csv: Path,
    test_avg_row: Mapping[str, str],
    test_counts: Mapping[str, int],
    method_column_order: Sequence[str],
    found_methods: set[str],
    gated_csv: Optional[Path] = None,
    gated_avg_row: Optional[Mapping[str, str]] = None,
    gated_counts: Optional[Mapping[str, int]] = None,
) -> None:
    columns = list(method_column_order)

    def cell_fn(row_map: Mapping[str, Any], column: str) -> str:
        return _comparison_metric_cell(
            row_map, column, found_methods=found_methods
        )
    metric_rows: List[Tuple[str, Mapping[str, Any]]] = []
    if gated_avg_row is not None and gated_counts is not None:
        metric_rows.extend(
            [
                ("gated_avg", gated_avg_row),
                ("gated_num_best", gated_counts),
            ]
        )
    metric_rows.extend(
        [
            ("avg", test_avg_row),
            ("num_best", test_counts),
        ]
    )
    title = f"=== comparison summary ({test_csv}) ==="
    if gated_csv is not None:
        title = f"=== comparison summary ({test_csv}, gated: {gated_csv}) ==="
    print(
        _format_metrics_table(
            title=title,
            columns=columns,
            metric_rows=metric_rows,
            cell_fn=cell_fn,
        ).rstrip()
    )


@dataclass
class _DatasetCompareSummary:
    dataset: str
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT
    n_participants: int = 0
    teh_run: str = ""
    avg_test: Dict[str, str] = field(default_factory=dict)
    avg_gated: Dict[str, str] = field(default_factory=dict)
    num_best_test: Dict[str, int] = field(default_factory=dict)
    num_best_gated: Dict[str, int] = field(default_factory=dict)
    num_test_loglik_gt_threshold: Dict[str, int] = field(default_factory=dict)
    found_methods: frozenset[str] = field(default_factory=frozenset)
    output_csv: Optional[Path] = None
    gated_csv: Optional[Path] = None
    error: Optional[str] = None
    compare_accuracy: bool = False
    pt_best_ours_second_count: int = 0
    pt_best_ours_second_avg_gap: str = ""
    mle_best_ours_second_count: int = 0
    mle_best_avg_gap: str = ""


def _summary_found_methods(
    baseline_paths: Mapping[str, Path],
    run_labels: Sequence[str],
) -> set[str]:
    """Method keys used in --all_in tables (_BASELINE_METHODS names + TEH)."""
    found = set(baseline_paths.keys())
    if run_labels:
        found.add("TEH")
    return found


def _comparison_found_methods(
    baseline_paths: Mapping[str, Path],
    run_labels: Sequence[str],
) -> set[str]:
    """Method keys used in single-dataset printed tables (run_* labels for TEH)."""
    return set(baseline_paths.keys()) | set(run_labels)


def _collapse_metric_row(
    row: Mapping[str, Any],
    *,
    teh_labels: Sequence[str],
    found_methods: set[str],
    value_type: type = str,
) -> Dict[str, Any]:
    """Map avg_row or num_best row to fixed method keys (TEH <- run_* column)."""
    out: Dict[str, Any] = {}
    for method in _BASELINE_METHODS:
        if method not in found_methods:
            continue
        if method in row and row[method] not in ("", None):
            out[method] = row[method]
    if "TEH" in found_methods:
        for tl in teh_labels:
            if tl in row and row[tl] not in ("", None):
                out["TEH"] = row[tl]
                break
    if value_type is int:
        return {k: int(v) for k, v in out.items()}
    return {k: str(v) for k, v in out.items()}


def _build_dataset_summary(
    *,
    dataset: str,
    psych_dataset_split: str,
    n_participants: int,
    teh_labels: Sequence[str],
    found_methods: set[str],
    avg_row: Mapping[str, str],
    best_test: Mapping[str, int],
    output_csv: Path,
    compare_accuracy: bool = False,
    gated_csv: Optional[Path] = None,
    avg_gated: Optional[Mapping[str, str]] = None,
    best_gated: Optional[Mapping[str, int]] = None,
    num_test_loglik_gt_threshold: Optional[Mapping[str, int]] = None,
    pt_best_ours_second_count: int = 0,
    pt_best_ours_second_avg_gap: str = "",
    mle_best_ours_second_count: int = 0,
    mle_best_avg_gap: str = "",
) -> _DatasetCompareSummary:
    return _DatasetCompareSummary(
        dataset=dataset,
        psych_dataset_split=psych_dataset_split,
        n_participants=n_participants,
        teh_run=teh_labels[0] if teh_labels else "",
        avg_test=_collapse_metric_row(
            avg_row, teh_labels=teh_labels, found_methods=found_methods, value_type=str
        ),
        avg_gated=_collapse_metric_row(
            avg_gated or {}, teh_labels=teh_labels, found_methods=found_methods, value_type=str
        ),
        num_best_test=_collapse_metric_row(
            best_test, teh_labels=teh_labels, found_methods=found_methods, value_type=int
        ),
        num_best_gated=_collapse_metric_row(
            best_gated or {}, teh_labels=teh_labels, found_methods=found_methods, value_type=int
        ),
        num_test_loglik_gt_threshold=_collapse_metric_row(
            num_test_loglik_gt_threshold or {},
            teh_labels=teh_labels,
            found_methods=found_methods,
            value_type=int,
        ),
        found_methods=frozenset(found_methods),
        output_csv=output_csv,
        gated_csv=gated_csv,
        compare_accuracy=compare_accuracy,
        pt_best_ours_second_count=int(pt_best_ours_second_count),
        pt_best_ours_second_avg_gap=str(pt_best_ours_second_avg_gap),
        mle_best_ours_second_count=int(mle_best_ours_second_count),
        mle_best_avg_gap=str(mle_best_avg_gap),
    )


def _write_all_in_summary_csv(
    path: Path, summaries: Sequence[_DatasetCompareSummary]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compare_accuracy = any(s.compare_accuracy for s in summaries)
    if compare_accuracy:
        fieldnames = [
            "dataset",
            "psych_dataset_split",
            "n_participants",
            "teh_run",
            "method",
            "avg_test_acc",
            "num_best_test",
            "pt_best_ours_second_count",
            "pt_best_ours_second_avg_gap",
            "mle_best_ours_second_count",
            "mle_best_avg_gap",
            "num_test_loglik_gt_threshold",
            "output_csv",
            "error",
        ]
    else:
        fieldnames = [
            "dataset",
            "psych_dataset_split",
            "n_participants",
            "teh_run",
            "method",
            "avg_test_loglik",
            "avg_gated_test_loglik",
            "num_best_test",
            "num_best_gated",
            "pt_best_ours_second_count",
            "pt_best_ours_second_avg_gap",
            "mle_best_ours_second_count",
            "mle_best_avg_gap",
            "num_test_loglik_gt_threshold",
            "output_csv",
            "gated_csv",
            "error",
        ]
    rows: List[Dict[str, str]] = []
    for s in summaries:
        if s.error:
            err_row = {
                "dataset": s.dataset,
                "psych_dataset_split": s.psych_dataset_split,
                "n_participants": str(s.n_participants),
                "teh_run": s.teh_run,
                "method": "",
                "num_best_test": "",
                "pt_best_ours_second_count": str(s.pt_best_ours_second_count),
                "pt_best_ours_second_avg_gap": s.pt_best_ours_second_avg_gap,
                "mle_best_ours_second_count": str(s.mle_best_ours_second_count),
                "mle_best_avg_gap": s.mle_best_avg_gap,
                "num_test_loglik_gt_threshold": "",
                "output_csv": str(s.output_csv or ""),
                "error": s.error,
            }
            if compare_accuracy:
                err_row["avg_test_acc"] = ""
            else:
                err_row["avg_test_loglik"] = ""
                err_row["avg_gated_test_loglik"] = ""
                err_row["num_best_gated"] = ""
                err_row["gated_csv"] = str(s.gated_csv or "")
            rows.append(err_row)
            continue
        for method in _SUMMARY_METHOD_LABELS:
            row = {
                "dataset": s.dataset,
                "psych_dataset_split": s.psych_dataset_split,
                "n_participants": str(s.n_participants),
                "teh_run": s.teh_run,
                "method": method,
                "num_best_test": str(s.num_best_test.get(method, "")),
                "pt_best_ours_second_count": str(s.pt_best_ours_second_count),
                "pt_best_ours_second_avg_gap": s.pt_best_ours_second_avg_gap,
                "mle_best_ours_second_count": str(s.mle_best_ours_second_count),
                "mle_best_avg_gap": s.mle_best_avg_gap,
                "num_test_loglik_gt_threshold": str(
                    s.num_test_loglik_gt_threshold.get(method, "")
                ),
                "output_csv": str(s.output_csv or ""),
                "error": "",
            }
            if compare_accuracy:
                row["avg_test_acc"] = s.avg_test.get(method, "")
            else:
                row["avg_test_loglik"] = s.avg_test.get(method, "")
                row["avg_gated_test_loglik"] = s.avg_gated.get(method, "")
                row["num_best_gated"] = str(s.num_best_gated.get(method, ""))
                row["gated_csv"] = str(s.gated_csv or "")
            rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _format_all_in_best_vs_ours_second_table(
    summaries: Sequence[_DatasetCompareSummary],
    *,
    title: str,
    count_attr: str,
    gap_attr: str,
    gap_col: str,
) -> str:
    ok = [s for s in summaries if not s.error]
    if not ok:
        return f"{title}\n  (no successful datasets)\n"
    ds_col_w = max(max(len(s.dataset) for s in ok), len("dataset"))
    count_col = count_attr
    count_w = max(len(count_col), 8)
    gap_w = max(len(gap_col), 8)
    header = (
        f"{'dataset':<{ds_col_w}}  "
        f"{count_col:>{count_w}}  "
        f"{gap_col:>{gap_w}}"
    )
    lines = [title, header]
    for s in ok:
        count = int(getattr(s, count_attr))
        gap = str(getattr(s, gap_attr) or _NA)
        lines.append(
            f"{s.dataset:<{ds_col_w}}  "
            f"{count:>{count_w}}  "
            f"{gap:>{gap_w}}"
        )
    err = [s for s in summaries if s.error]
    for s in err:
        lines.append(f"{s.dataset:<{ds_col_w}}  ERROR: {s.error}")
    return "\n".join(lines) + "\n"


def _format_all_in_wide_table(
    summaries: Sequence[_DatasetCompareSummary],
    *,
    title: str,
    value_attr: str,
    include_avg_last_row: bool = False,
    avg_ndigits: int = 2,
) -> str:
    """Render dataset x method table for one metric family."""
    ok = [s for s in summaries if not s.error]
    if not ok:
        return f"{title}\n  (no successful datasets)\n"
    ds_col_w = max(len(s.dataset) for s in ok)
    ds_col_w = max(ds_col_w, len("dataset"))
    header = f"{'dataset':<{ds_col_w}}"
    for method in _SUMMARY_METHOD_LABELS:
        header += f"  {method:>18}"
    lines = [title, header]
    for s in ok:
        row_map: Mapping[str, Any] = getattr(s, value_attr)
        line = f"{s.dataset:<{ds_col_w}}"
        for method in _SUMMARY_METHOD_LABELS:
            if method not in s.found_methods:
                cell = _NA
            else:
                val = row_map.get(method, "")
                cell = _NA if val is None or val == "" else str(val)
            line += f"  {cell:>18}"
        lines.append(line)
    err = [s for s in summaries if s.error]
    for s in err:
        lines.append(f"{s.dataset:<{ds_col_w}}  ERROR: {s.error}")
    if include_avg_last_row:
        avg_line = f"{'Avg':<{ds_col_w}}"
        for method in _SUMMARY_METHOD_LABELS:
            vals: List[float] = []
            for s in ok:
                if method not in s.found_methods:
                    continue
                row_map = getattr(s, value_attr)
                raw = row_map.get(method, "")
                if raw in ("", None):
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    vals.append(v)
            cell = _NA if not vals else f"{statistics.mean(vals):.{avg_ndigits}f}"
            avg_line += f"  {cell:>18}"
        lines.append(avg_line)
    return "\n".join(lines) + "\n"


def _format_all_in_threshold_percentage_table(
    summaries: Sequence[_DatasetCompareSummary],
    *,
    title: str,
) -> str:
    """Render dataset x method table for num/total and percentage above threshold."""
    ok = [s for s in summaries if not s.error]
    if not ok:
        return f"{title}\n  (no successful datasets)\n"
    ds_col_w = max(len(s.dataset) for s in ok)
    ds_col_w = max(ds_col_w, len("dataset"))
    header = f"{'dataset':<{ds_col_w}}"
    for method in _SUMMARY_METHOD_LABELS:
        header += f"  {method:>18}"
    lines = [title, header]

    total_by_method: Dict[str, int] = {m: 0 for m in _SUMMARY_METHOD_LABELS}
    total_participants_by_method: Dict[str, int] = {m: 0 for m in _SUMMARY_METHOD_LABELS}

    for s in ok:
        line = f"{s.dataset:<{ds_col_w}}"
        total = int(s.n_participants)
        for method in _SUMMARY_METHOD_LABELS:
            if method not in s.found_methods:
                cell = _NA
            else:
                count = int(s.num_test_loglik_gt_threshold.get(method, 0))
                if total <= 0:
                    cell = _NA
                else:
                    pct = 100.0 * count / total
                    cell = f"{pct:.0f}% ({count}/{total})"
                    total_by_method[method] += count
                    total_participants_by_method[method] += total
            line += f"  {cell:>18}"
        lines.append(line)

    avg_line = f"{'Avg':<{ds_col_w}}"
    for method in _SUMMARY_METHOD_LABELS:
        denom = total_participants_by_method[method]
        if denom <= 0:
            cell = _NA
        else:
            num = total_by_method[method]
            pct = 100.0 * num / denom
            cell = f"{pct:.0f}% ({num}/{denom})"
        avg_line += f"  {cell:>18}"
    lines.append(avg_line)

    err = [s for s in summaries if s.error]
    for s in err:
        lines.append(f"{s.dataset:<{ds_col_w}}  ERROR: {s.error}")
    return "\n".join(lines) + "\n"


def _print_all_in_summary(
    summaries: Sequence[_DatasetCompareSummary],
    summary_csv: Path,
) -> None:
    _write_all_in_summary_csv(summary_csv, summaries)
    printable = [s for s in summaries if s.dataset not in _ALL_IN_PRINT_EXCLUDE]
    compare_accuracy = any(s.compare_accuracy for s in summaries)
    label = "compare.py --all_in --accuracy summary" if compare_accuracy else "compare.py --all_in summary"
    print(f"\n=== {label} ===")
    print(f"Wrote {summary_csv}\n")
    test_metric = _TEST_ACC if compare_accuracy else _TEST_LOGLIK
    print(
        _format_all_in_wide_table(
            printable,
            title=f"Avg {test_metric}",
            value_attr="avg_test",
            include_avg_last_row=True,
        )
    )
    if not compare_accuracy:
        print(
            _format_all_in_wide_table(
                printable,
                title=f"Avg {_GATED_LOGLIK}",
                value_attr="avg_gated",
                include_avg_last_row=True,
            )
        )
    print(
        _format_all_in_wide_table(
            printable,
            title=f"num_best ({test_metric})",
            value_attr="num_best_test",
            include_avg_last_row=True,
        )
    )
    if not compare_accuracy:
        print(
            _format_all_in_wide_table(
                printable,
                title=f"num_best ({_GATED_LOGLIK})",
                value_attr="num_best_gated",
                include_avg_last_row=True,
            )
        )
        print(
            _format_all_in_wide_table(
                printable,
                title=f"num ({_TEST_LOGLIK} > {_LOGLIK_WIN_THRESHOLD})",
                value_attr="num_test_loglik_gt_threshold",
                include_avg_last_row=True,
            )
        )
        print(
            _format_all_in_threshold_percentage_table(
                printable,
                title=f"% num ({_TEST_LOGLIK} > {_LOGLIK_WIN_THRESHOLD})",
            )
        )
        print(
            _format_all_in_best_vs_ours_second_table(
                printable,
                title="PT-best with Ours-second",
                count_attr="pt_best_ours_second_count",
                gap_attr="pt_best_ours_second_avg_gap",
                gap_col="avg_pt_minus_ours",
            )
        )
        print(
            _format_all_in_best_vs_ours_second_table(
                printable,
                title="MLE-best with Ours-second",
                count_attr="mle_best_ours_second_count",
                gap_attr="mle_best_avg_gap",
                gap_col="avg_mle_minus_ours_when_mle_best",
            )
        )


def _run_compare_dataset(
    repo: Path,
    args: argparse.Namespace,
    *,
    dataset: str,
    psych_dataset_split: str,
    quiet: bool,
) -> _DatasetCompareSummary:
    """Run one dataset comparison; write per-dataset CSVs and return summary metrics."""
    psych_split = _effective_psych_dataset_split(dataset, psych_dataset_split)
    config_path = Path(args.config_path).expanduser()
    config_path = (
        config_path.resolve()
        if config_path.is_absolute()
        else (repo / config_path).resolve()
    )
    config_data = _load_baseline_config_file(config_path)
    config_key = _config_dataset_key(dataset, psych_split)

    baseline_paths = _resolve_baseline_run_paths(
        config_data, repo, dataset, psych_split, quiet=quiet
    )

    if args.centaur_csv is not None:
        centaur_override = Path(args.centaur_csv).expanduser()
        baseline_paths["Centaur"] = (
            centaur_override.parent
            if centaur_override.suffix.lower() == ".csv"
            else centaur_override
        )

    ds_defaults = _dataset_defaults(
        repo, dataset, psych_split, accuracy=bool(args.accuracy)
    )
    output_arg = (
        Path(args.output).expanduser()
        if args.output is not None
        else ds_defaults.output_csv
    )
    output_arg = (
        output_arg.resolve()
        if output_arg.is_absolute()
        else (repo / output_arg).resolve()
    )

    experiment_paths = args.experiment_paths
    if experiment_paths is None:
        teh_path = _resolve_teh_run_path(
            config_data, repo, dataset, psych_split, quiet=quiet
        )
        teh_inputs = [teh_path] if teh_path is not None else []
    else:
        teh_inputs = list(experiment_paths)

    exp_resolved: List[Path] = []
    if teh_inputs:
        exp_resolved = [_resolve_loglik_csv(Path(ep).expanduser()) for ep in teh_inputs]
    run_labels = [_run_column_name(p) for p in exp_resolved]
    if len(set(run_labels)) != len(run_labels):
        raise ValueError(f"Duplicate TEH run column names: {run_labels}")

    if not exp_resolved and not baseline_paths:
        raise SystemExit(
            f"{dataset}: no baseline or TEH runs found "
            "(check generated_outputs/ or pass --experiment_paths)."
        )

    compare_accuracy = bool(args.accuracy)
    score_kind = _TEST_ACC if compare_accuracy else _TEST_LOGLIK
    format_score = _format_acc if compare_accuracy else _format_loglik
    score_ndigits = _ACC_NDIGITS if compare_accuracy else _LOGLIK_NDIGITS

    all_loglik_csvs: List[Path] = list(exp_resolved)
    baseline_columns: Dict[str, str] = {}
    for method in _BASELINE_METHODS:
        if method in baseline_paths:
            csv_path = _resolve_loglik_csv(baseline_paths[method])
            all_loglik_csvs.append(csv_path)
            baseline_columns[method] = _TEST_ACC if compare_accuracy else _TEST_LOGLIK

    centaur_path: Optional[Path] = None
    if "Centaur" in baseline_paths:
        centaur_path = _resolve_loglik_csv(baseline_paths["Centaur"])

    roster_csvs = list(all_loglik_csvs)
    if "MLE" in baseline_paths:
        participant_ids = _read_participant_ids_from_csv(
            _resolve_loglik_csv(baseline_paths["MLE"])
        )
    elif centaur_path is not None:
        participant_ids = _read_participant_ids_from_csv(centaur_path)
    elif roster_csvs:
        participant_ids = _participant_ids_from_experiment_csvs(roster_csvs)
    else:
        raise SystemExit(f"{dataset}: no participant_id rows found.")

    n_before = len(participant_ids)
    participant_ids, ord_min, ord_max = _clamp_participant_ids_to_dataset(
        participant_ids,
        dataset=dataset,
        psych_dataset_split=psych_split,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=str(args.mixed_gambles_csv),
        filter_mixed_gambles=bool(args.filter_mixed_gambles),
    )
    if len(participant_ids) != n_before and not quiet:
        ds_roster_label = (
            dataset
            if is_mixed_gambles_dataset(dataset)
            else f"{dataset}, psych_dataset_split={psych_split}"
        )
        print(
            f"Participant roster clamped to ordinals [{ord_min}, {ord_max}] "
            f"for {ds_roster_label}: {len(participant_ids)} participant(s)."
        )

    bir = _load_or_compute_bir_map(
        repo,
        dataset=dataset,
        psych_dataset_split=psych_split,
        participant_ids=participant_ids,
        split_ratio=float(args.split_ratio),
        split_seed=int(args.split_seed),
        local_dataset=args.local_dataset,
        mixed_gambles_csv=str(args.mixed_gambles_csv),
        filter_mixed_gambles=bool(args.filter_mixed_gambles),
        recompute=bool(args.recompute_bir),
        quiet=quiet,
    )

    ds_label = (
        f"{dataset}"
        if is_mixed_gambles_dataset(dataset)
        else f"{dataset}, psych_dataset_split={psych_split}"
    )

    accuracy_load_kwargs = {
        "dataset": dataset,
        "psych_dataset_split": psych_split,
        "split_ratio": float(args.split_ratio),
        "split_seed": int(args.split_seed),
        "local_dataset": args.local_dataset,
        "mixed_gambles_csv": str(args.mixed_gambles_csv),
        "filter_mixed_gambles": bool(args.filter_mixed_gambles),
        "participant_ids": participant_ids,
        "recompute": bool(args.recompute_accuracy),
        "quiet": quiet,
    }

    baseline_scores: List[Tuple[str, Dict[int, float]]] = []
    for method in _BASELINE_METHODS:
        scores: Dict[int, float] = {}
        if method in baseline_paths:
            run_path = baseline_paths[method]
            if compare_accuracy:
                scores = _load_accuracy_scores_from_run(
                    run_path,
                    method=method,
                    **accuracy_load_kwargs,
                )
            else:
                scores = _load_scores_from_run(run_path, _TEST_LOGLIK, required=False)
        baseline_scores.append((method, scores))

    teh_test: List[Tuple[str, Dict[int, float]]] = []
    for label, csv_path in zip(run_labels, exp_resolved):
        run_path = csv_path.parent if csv_path.is_file() else csv_path
        if compare_accuracy:
            teh_scores = _load_accuracy_scores_from_run(
                run_path,
                method="TEH",
                **accuracy_load_kwargs,
            )
        else:
            teh_scores = _read_loglik_csv(csv_path, _TEST_LOGLIK, required=True)
        teh_test.append((label, teh_scores))

    pt_best_ours_second_count = 0
    pt_best_ours_second_avg_gap = ""
    mle_best_ours_second_count = 0
    mle_best_avg_gap = ""
    num_test_loglik_gt_threshold: Dict[str, int] = {}
    combined_method_scores = baseline_scores + teh_test
    if not compare_accuracy:
        num_test_loglik_gt_threshold = _count_above_threshold(
            participant_ids,
            combined_method_scores,
            threshold=_LOGLIK_WIN_THRESHOLD,
        )
    if (
        not compare_accuracy
        and "prospect_theory" in baseline_paths
        and len(teh_test) > 0
    ):
        pt_best_ours_second_count, pt_best_ours_second_avg_gap = _best_method_ours_second_stats(
            participant_ids,
            combined_method_scores,
            best_label="prospect_theory",
            ours_label=teh_test[0][0],
        )
    if (
        not compare_accuracy
        and "MLE" in baseline_paths
        and len(teh_test) > 0
    ):
        mle_best_ours_second_count, _ = _best_method_ours_second_stats(
            participant_ids,
            combined_method_scores,
            best_label="MLE",
            ours_label=teh_test[0][0],
        )
        mle_best_avg_gap = _best_method_avg_gap(
            participant_ids,
            combined_method_scores,
            best_label="MLE",
            ours_label=teh_test[0][0],
        )

    method_column_order = list(_BASELINE_METHODS) + (run_labels if run_labels else ["TEH"])
    summary_found_methods = _summary_found_methods(baseline_paths, run_labels)
    found_methods = _comparison_found_methods(baseline_paths, run_labels)
    teh_columns_test = {
        label: (_TEST_ACC if compare_accuracy else _TEST_LOGLIK) for label, _ in teh_test
    }
    n_test, avg_row, best_counts = _write_baseline_comparison_csv(
        out_path=output_arg,
        participant_ids=participant_ids,
        baselines=baseline_scores,
        teh_runs=teh_test,
        bir=bir,
        format_score=format_score,
        score_ndigits=score_ndigits,
    )
    if args.verbose:
        _print_run_summary(
            dataset_label=ds_label,
            config_key=config_key,
            baseline_paths=baseline_paths,
            baseline_columns=baseline_columns,
            teh_paths=[p.parent if p.is_file() else p for p in exp_resolved],
            teh_columns=teh_columns_test,
            n_participants=n_test,
            avg_row=avg_row,
            out_path=output_arg,
            score_kind=score_kind,
        )
    elif not quiet:
        print(f"Wrote {output_arg} ({n_test} participants)")

    if compare_accuracy:
        if not quiet:
            _print_comparison_metrics_table(
                test_csv=output_arg,
                test_avg_row=avg_row,
                test_counts=best_counts,
                method_column_order=method_column_order,
                found_methods=found_methods,
            )
        return _build_dataset_summary(
            dataset=dataset,
            psych_dataset_split=psych_split,
            n_participants=n_test,
            teh_labels=run_labels,
            found_methods=summary_found_methods,
            avg_row=avg_row,
            best_test=best_counts,
            output_csv=output_arg,
            compare_accuracy=True,
            num_test_loglik_gt_threshold=num_test_loglik_gt_threshold,
            pt_best_ours_second_count=pt_best_ours_second_count,
            pt_best_ours_second_avg_gap=pt_best_ours_second_avg_gap,
            mle_best_ours_second_count=mle_best_ours_second_count,
            mle_best_avg_gap=mle_best_avg_gap,
        )

    gated_csvs = list(exp_resolved)
    if centaur_path is not None:
        gated_csvs.insert(0, centaur_path)
    if not _any_csv_has_gated(centaur_path, gated_csvs):
        if not quiet:
            _print_comparison_metrics_table(
                test_csv=output_arg,
                test_avg_row=avg_row,
                test_counts=best_counts,
                method_column_order=method_column_order,
                found_methods=found_methods,
            )
        return _build_dataset_summary(
            dataset=dataset,
            psych_dataset_split=psych_split,
            n_participants=n_test,
            teh_labels=run_labels,
            found_methods=summary_found_methods,
            avg_row=avg_row,
            best_test=best_counts,
            output_csv=output_arg,
            num_test_loglik_gt_threshold=num_test_loglik_gt_threshold,
            pt_best_ours_second_count=pt_best_ours_second_count,
            pt_best_ours_second_avg_gap=pt_best_ours_second_avg_gap,
            mle_best_ours_second_count=mle_best_ours_second_count,
            mle_best_avg_gap=mle_best_avg_gap,
        )

    gated_out = (
        Path(args.output_gated).expanduser()
        if args.output_gated is not None
        else _gated_output_path(output_arg)
    )
    gated_out = (
        gated_out.resolve()
        if gated_out.is_absolute()
        else (repo / gated_out).resolve()
    )

    test_by_method = dict(baseline_scores)
    gated_baselines: List[Tuple[str, Dict[int, float]]] = []
    for method in _BASELINE_METHODS:
        scores_g: Dict[int, float] = {}
        if method in baseline_paths:
            test_scores = test_by_method.get(method, {})
            if method == "Centaur" and centaur_path is not None:
                gated = _read_loglik_csv(centaur_path, _GATED_LOGLIK, required=False)
            else:
                gated = _load_scores_from_run(
                    baseline_paths[method], _GATED_LOGLIK, required=False
                )
            scores_g = _scores_for_gated_table(test_scores, gated)
        gated_baselines.append((method, scores_g))

    teh_test_by_label = dict(teh_test)
    teh_gated: List[Tuple[str, Dict[int, float]]] = []
    for label, csv_path in zip(run_labels, exp_resolved):
        test_scores = teh_test_by_label.get(label, {})
        gated = _read_loglik_csv(csv_path, _GATED_LOGLIK, required=False)
        teh_gated.append((label, _scores_for_gated_table(test_scores, gated)))

    n_gated, avg_g, best_g = _write_baseline_comparison_csv(
        out_path=gated_out,
        participant_ids=participant_ids,
        baselines=gated_baselines,
        teh_runs=teh_gated,
        bir=bir,
    )
    if args.verbose:
        print(f"--- gated ({_GATED_LOGLIK}) ---")
        _print_run_summary(
            dataset_label=ds_label,
            config_key=config_key,
            baseline_paths=baseline_paths,
            baseline_columns=baseline_columns,
            teh_paths=[p.parent if p.is_file() else p for p in exp_resolved],
            teh_columns={label: _GATED_LOGLIK for label, _ in teh_gated},
            n_participants=n_gated,
            avg_row=avg_g,
            out_path=gated_out,
            score_kind=_GATED_LOGLIK,
        )
    elif not quiet:
        print(f"Wrote {gated_out} ({n_gated} participants)")

    if not quiet:
        _print_comparison_metrics_table(
            test_csv=output_arg,
            test_avg_row=avg_row,
            test_counts=best_counts,
            method_column_order=method_column_order,
            found_methods=found_methods,
            gated_csv=gated_out,
            gated_avg_row=avg_g,
            gated_counts=best_g,
        )

    return _build_dataset_summary(
        dataset=dataset,
        psych_dataset_split=psych_split,
        n_participants=n_test,
        teh_labels=run_labels,
        found_methods=summary_found_methods,
        avg_row=avg_row,
        best_test=best_counts,
        output_csv=output_arg,
        gated_csv=gated_out,
        avg_gated=avg_g,
        best_gated=best_g,
        num_test_loglik_gt_threshold=num_test_loglik_gt_threshold,
        pt_best_ours_second_count=pt_best_ours_second_count,
        pt_best_ours_second_avg_gap=pt_best_ours_second_avg_gap,
        mle_best_ours_second_count=mle_best_ours_second_count,
        mle_best_avg_gap=mle_best_avg_gap,
    )


def _print_run_summary(
    *,
    dataset_label: str,
    config_key: str,
    baseline_paths: Dict[str, Path],
    baseline_columns: Dict[str, str],
    teh_paths: Sequence[Path],
    teh_columns: Dict[str, str],
    n_participants: int,
    avg_row: Dict[str, str],
    out_path: Path,
    score_kind: str,
) -> None:
    print(f"dataset={dataset_label} (config key: {config_key})")
    if baseline_paths:
        print(f"Loaded baselines (column: {score_kind}; MLE/PT/openevolve fall back to "
              f"{_TEST_LOGLIK} in gated output when {_GATED_LOGLIK} is absent):")
        for method in _BASELINE_METHODS:
            if method in baseline_paths:
                col = baseline_columns.get(method, score_kind)
                csv_path = _resolve_loglik_csv(baseline_paths[method])
                print(f"  {method}: {csv_path} [{col}]")
    else:
        print("Loaded baselines: (none from config)")
    if teh_paths:
        print(f"Loaded TEH runs (column: {score_kind}):")
        for p in teh_paths:
            label = p.name if p.is_dir() else p.parent.name
            col = teh_columns.get(label, score_kind)
            csv_path = _resolve_loglik_csv(p)
            print(f"  {label}: {csv_path} [{col}]")
    else:
        print("Loaded TEH runs: (none)")
    print(f"Participants compared: {n_participants}")
    print(f"Wrote {out_path}")


def main() -> None:
    repo = _repo_root()

    p = argparse.ArgumentParser(
        description=(
            "Compare test_loglik for baseline methods (from config) and optional TEH runs."
        )
    )
    p.add_argument(
        "--dataset",
        choices=_COMPARE_DATASET_CHOICES,
        default=PETERSON2021USING_ALIAS,
        help=(
            "Psych-101 binary alias or mixed_gambles. Legacy choice13k / cpc18 accepted. "
            "Selects config key and default output filename."
        ),
    )
    p.add_argument(
        "--psych_dataset_split",
        type=str,
        default=DEFAULT_PSYCH_DATASET_SPLIT,
        choices=sorted({"train", "test"}),
        help="Psych-101 HF corpus (train | test). Ignored for mixed_gambles.",
    )
    p.add_argument(
        "--config_path",
        "--baseline_config",
        type=Path,
        default=Path(_DEFAULT_BASELINE_CONFIG),
        dest="config_path",
        help=(
            f"YAML with datasets -> MLE / prospect_theory / openevolve / Centaur / TEH paths "
            f"(default: {_DEFAULT_BASELINE_CONFIG})."
        ),
    )
    p.add_argument(
        "--experiment_paths",
        nargs="*",
        default=None,
        type=Path,
        help=(
            "TEH run directories or participant_details_loglik.csv paths (zero or more). "
            "If omitted, auto-select the newest run_* under generated_outputs/.../teh/."
        ),
    )
    p.add_argument(
        "--centaur_csv",
        type=Path,
        default=None,
        help="Override Centaur baseline run directory or participant_details_loglik.csv (else from config).",
    )
    p.add_argument(
        "--similar_threshold",
        type=float,
        default=0.05,
        help="Legacy mode only: |teh - centaur| <= threshold counts as Similar.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default includes dataset and split in filename).",
    )
    p.add_argument(
        "--output_gated",
        type=Path,
        default=None,
        help="Gated output CSV (default: <output stem>_gated.csv). Legacy layout only.",
    )
    p.add_argument(
        "--split_ratio",
        type=float,
        default=_DEFAULT_SPLIT_RATIO,
        help="Within-participant train fraction for BIR (default matches Psych-101 baselines).",
    )
    p.add_argument(
        "--split_seed",
        type=int,
        default=_DEFAULT_SPLIT_SEED,
        help="RNG seed for within-participant train/val/test split used in BIR.",
    )
    p.add_argument(
        "--local_dataset",
        type=str,
        default=None,
        help="Optional local HuggingFace dataset path for Psych-101 loading.",
    )
    p.add_argument(
        "--mixed_gambles_csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="CSV path for mixed_gambles BIR (default: data_modules.mixed_gambles.DEFAULT_CSV_PATH).",
    )
    p.add_argument(
        "--filter_mixed_gambles",
        action="store_true",
        help="For mixed_gambles only: restrict BIR to gain_loss trials.",
    )
    p.add_argument(
        "--recompute_bir",
        action="store_true",
        help="Ignore cached BIR and recompute all participants in the comparison roster.",
    )
    p.add_argument(
        "--accuracy",
        action="store_true",
        help=(
            "Compare test accuracy instead of test_loglik (works with --all_in). "
            "Loads participants_summary.csv or participant results.json when available; "
            "otherwise uses participant_details_test_acc.csv in the run directory, "
            "or recomputes from best_program.py and writes that cache there."
        ),
    )
    p.add_argument(
        "--recompute_accuracy",
        action="store_true",
        help=(
            "With --accuracy: ignore run-level participant_details_test_acc.csv caches "
            "and recompute test accuracy from best_program.py."
        ),
    )
    p.add_argument(
        "--all_in",
        action="store_true",
        help=(
            "Run the default dataset subset for --all_in (see _ALL_IN_DATASETS) "
            "and print a cross-dataset summary table."
        ),
    )
    p.add_argument(
        "--summary_output",
        type=Path,
        default=None,
        help=(
            f"CSV path for --all_in summary (default: {_DEFAULT_ALL_IN_SUMMARY_CSV}). "
            "Per-dataset comparison CSVs still use default names under analysis/data/utils/."
        ),
    )
    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Print full comparison log (paths, averages). Default: compact output with "
            "num_best summary only (single dataset) or final --all_in tables only."
        ),
    )
    args = p.parse_args()

    if not (0.0 < args.split_ratio < 1.0):
        raise SystemExit(f"--split_ratio must be in (0, 1), got {args.split_ratio}.")

    if args.all_in:
        if args.output is not None or args.output_gated is not None:
            raise SystemExit(
                "--all_in writes per-dataset CSVs to default paths; "
                "use --summary_output for the cross-dataset summary CSV."
            )
        summary_path = (
            Path(args.summary_output).expanduser()
            if args.summary_output is not None
            else Path(
                _DEFAULT_ALL_IN_ACC_SUMMARY_CSV
                if args.accuracy
                else _DEFAULT_ALL_IN_SUMMARY_CSV
            )
        )
        summary_path = (
            summary_path.resolve()
            if summary_path.is_absolute()
            else (repo / summary_path).resolve()
        )
        quiet = not args.verbose
        summaries: List[_DatasetCompareSummary] = []
        for ds in _ALL_IN_DATASETS:
            dataset = _normalize_compare_dataset(ds)
            psych_split = (
                DEFAULT_PSYCH_DATASET_SPLIT
                if is_mixed_gambles_dataset(dataset)
                else "train"
            )
            if not quiet:
                print(f"\n========== {dataset} (psych_dataset_split={psych_split}) ==========")
            try:
                summaries.append(
                    _run_compare_dataset(
                        repo,
                        args,
                        dataset=dataset,
                        psych_dataset_split=psych_split,
                        quiet=quiet,
                    )
                )
            except (SystemExit, ValueError, FileNotFoundError, OSError) as exc:
                msg = str(exc) or type(exc).__name__
                print(f"ERROR {dataset}: {msg}", file=sys.stderr)
                summaries.append(
                    _DatasetCompareSummary(
                        dataset=dataset,
                        psych_dataset_split=psych_split,
                        error=msg,
                    )
                )
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                print(f"ERROR {dataset}: {msg}", file=sys.stderr)
                summaries.append(
                    _DatasetCompareSummary(
                        dataset=dataset,
                        psych_dataset_split=psych_split,
                        error=msg,
                    )
                )
        _print_all_in_summary(summaries, summary_path)
        return

    dataset = _normalize_compare_dataset(args.dataset)
    psych_split = _effective_psych_dataset_split(dataset, args.psych_dataset_split)
    if dataset not in PARTICIPANT_DATASETS:
        raise SystemExit(
            f"Unknown dataset {args.dataset!r} (normalized {dataset!r}). "
            f"Choose from {sorted(PARTICIPANT_DATASETS)} or legacy {sorted(_COMPARE_LEGACY_DATASETS)}."
        )

    _run_compare_dataset(
        repo,
        args,
        dataset=dataset,
        psych_dataset_split=psych_split,
        quiet=False,
    )


if __name__ == "__main__":
    main()
