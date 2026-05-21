#!/usr/bin/env python3
"""
Before running, set up the baseline methods config file:
analysis/data/baseline_methods/config.yaml

Usage:
  python analysis/code/utils/compare.py --dataset 1peterson2021using --psych_dataset_split train --experiment_paths 
   \\ generated_outputs/choice13k/te_dr/run_260514_231815

Compare per-participant test_loglik across baseline methods, optional Centaur, and TEH runs.

Baseline paths come from ``analysis/data/baseline_methods/config.yaml`` (see file
comments for key naming). Config dataset keys:

  datasets:
    1peterson2021using:          # psych_dataset_split train
      MLE: <run_dir_or_csv>
      prospect_theory: <run_dir_or_csv>
      openevolve: <run_dir_or_csv>
      Centaur: <run_dir_or_csv>
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

Without config entries, legacy layout is participant_id, BIR, Centaur, <teh_run>, ...
with Avg and Better / Similar / Worse vs Centaur footers.


"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
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
_BASELINE_METHODS = ("MLE", "prospect_theory", "openevolve", "Centaur")
_DEFAULT_BASELINE_CONFIG = "analysis/data/baseline_methods/config.yaml"
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


def _default_output_csv(repo: Path, dataset: str, psych_dataset_split: str) -> Path:
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


def _baseline_run_paths_from_config(
    config_data: Mapping[str, Any],
    repo: Path,
    dataset: str,
    psych_dataset_split: str,
) -> Dict[str, Path]:
    """Method name -> run dir or CSV (only methods present in config)."""
    datasets = config_data.get("datasets")
    if not isinstance(datasets, dict):
        return {}
    key = _config_dataset_key(dataset, psych_dataset_split)
    entry = datasets.get(key)
    if not isinstance(entry, dict):
        return {}
    out: Dict[str, Path] = {}
    for method in _BASELINE_METHODS:
        raw = entry.get(method)
        if raw is None or str(raw).strip() == "":
            continue
        out[method] = _resolve_repo_path(repo, str(raw))
    return out


def _dataset_defaults(
    repo: Path, dataset: str, psych_dataset_split: str
) -> _DatasetDefaults:
    alias = _normalize_compare_dataset(dataset)
    if alias not in PARTICIPANT_DATASETS:
        raise ValueError(
            f"dataset must be a TEH participant dataset {sorted(PARTICIPANT_DATASETS)} "
            f"or legacy alias in {sorted(_COMPARE_LEGACY_DATASETS)}, got {dataset!r}"
        )
    split = _effective_psych_dataset_split(alias, psych_dataset_split)
    return _DatasetDefaults(output_csv=_default_output_csv(repo, alias, split))


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


def _format_loglik(value: float) -> str:
    return f"{value:.{_LOGLIK_NDIGITS}f}"


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
        print(
            f"Computing BIR for {len(missing)} participant(s) "
            f"({label}, split_ratio={split_ratio}, split_seed={split_seed})..."
        )
        for pid in tqdm(missing, desc=f"BIR {label}", unit="participant"):
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
        print(f"Wrote BIR cache ({len(rows_by_pid)} participants) -> {cache_csv}")
    elif rows_by_pid:
        print(f"Loaded BIR cache ({len(rows_by_pid)} participants) from {cache_csv}")

    return {pid: float(rows_by_pid[pid]["BIR"]) for pid in roster if pid in rows_by_pid}


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


def _write_baseline_comparison_csv(
    *,
    out_path: Path,
    participant_ids: Sequence[int],
    baselines: Sequence[Tuple[str, Dict[int, float]]],
    teh_runs: Sequence[Tuple[str, Dict[int, float]]],
    bir: Dict[int, float],
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
            row[label] = _format_loglik(m[pid]) if pid in m else ""
        rows.append(row)

    avg_row: Dict[str, str] = {"participant_id": "Avg", "BIR": ""}
    if bir:
        avg_row["BIR"] = _finite_mean([bir[pid] for pid in ordered if pid in bir])
    for label, m in score_columns:
        avg_row[label] = _finite_mean(
            [m[pid] for pid in ordered if pid in m], ndigits=_LOGLIK_NDIGITS
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
    best_counts: Dict[str, int],
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
    print(f"Avg ({score_kind}):")
    for k, v in avg_row.items():
        if k not in ("participant_id", "BIR") and v != "":
            print(f"  {k}: {v}")
    print("num_best (ties count toward each tied best):")
    for k, v in best_counts.items():
        print(f"  {k}: {v}")
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
        "--baseline_config",
        type=Path,
        default=Path(_DEFAULT_BASELINE_CONFIG),
        help=(
            f"YAML with datasets -> MLE / prospect_theory / openevolve / Centaur paths "
            f"(default: {_DEFAULT_BASELINE_CONFIG})."
        ),
    )
    p.add_argument(
        "--experiment_paths",
        nargs="*",
        default=[],
        type=Path,
        help="TEH run directories or participant_details_loglik.csv paths (zero or more).",
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
    args = p.parse_args()

    dataset = _normalize_compare_dataset(args.dataset)
    psych_split = _effective_psych_dataset_split(dataset, args.psych_dataset_split)
    if dataset not in PARTICIPANT_DATASETS:
        raise SystemExit(
            f"Unknown dataset {args.dataset!r} (normalized {dataset!r}). "
            f"Choose from {sorted(PARTICIPANT_DATASETS)} or legacy {sorted(_COMPARE_LEGACY_DATASETS)}."
        )

    config_path = Path(args.baseline_config).expanduser()
    config_path = config_path.resolve() if config_path.is_absolute() else (repo / config_path).resolve()
    config_data = _load_baseline_config_file(config_path)
    config_key = _config_dataset_key(dataset, psych_split)

    config_baselines = _baseline_run_paths_from_config(
        config_data, repo, dataset, psych_split
    )
    baseline_paths = dict(config_baselines)

    if args.centaur_csv is not None:
        centaur_override = Path(args.centaur_csv).expanduser()
        baseline_paths["Centaur"] = (
            centaur_override.parent
            if centaur_override.suffix.lower() == ".csv"
            else centaur_override
        )

    ds_defaults = _dataset_defaults(repo, dataset, psych_split)
    output_arg = Path(args.output).expanduser() if args.output is not None else ds_defaults.output_csv
    output_arg = output_arg.resolve() if output_arg.is_absolute() else (repo / output_arg).resolve()

    if not (0.0 < args.split_ratio < 1.0):
        raise SystemExit(f"--split_ratio must be in (0, 1), got {args.split_ratio}.")

    teh_inputs = list(args.experiment_paths)
    exp_resolved: List[Path] = []
    if teh_inputs:
        exp_resolved = [_resolve_loglik_csv(Path(ep).expanduser()) for ep in teh_inputs]
    run_labels = [_run_column_name(p) for p in exp_resolved]
    if len(set(run_labels)) != len(run_labels):
        raise ValueError(f"Duplicate TEH run column names: {run_labels}")

    use_baseline_layout = bool(config_baselines)
    if not use_baseline_layout and not exp_resolved and not baseline_paths:
        raise SystemExit("Provide --experiment_paths and/or baseline entries in config.")

    all_loglik_csvs: List[Path] = list(exp_resolved)
    baseline_scores: List[Tuple[str, Dict[int, float]]] = []
    baseline_columns: Dict[str, str] = {}
    for method in _BASELINE_METHODS:
        scores: Dict[int, float] = {}
        if method in baseline_paths:
            run_path = baseline_paths[method]
            csv_path = _resolve_loglik_csv(run_path)
            all_loglik_csvs.append(csv_path)
            scores = _load_scores_from_run(run_path, _TEST_LOGLIK, required=False)
            baseline_columns[method] = _TEST_LOGLIK
        baseline_scores.append((method, scores))

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
        raise SystemExit("No participant_id rows found (add baselines or TEH runs).")

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
    )

    ds_label = (
        f"{dataset}"
        if is_mixed_gambles_dataset(dataset)
        else f"{dataset}, psych_dataset_split={psych_split}"
    )

    if use_baseline_layout:
        teh_test: List[Tuple[str, Dict[int, float]]] = []
        for label, csv_path in zip(run_labels, exp_resolved):
            teh_test.append(
                (label, _read_loglik_csv(csv_path, _TEST_LOGLIK, required=True))
            )

        teh_columns_test = {label: _TEST_LOGLIK for label, _ in teh_test}
        n_test, avg_row, best_counts = _write_baseline_comparison_csv(
            out_path=output_arg,
            participant_ids=participant_ids,
            baselines=baseline_scores,
            teh_runs=teh_test,
            bir=bir,
        )
        _print_run_summary(
            dataset_label=ds_label,
            config_key=config_key,
            baseline_paths=baseline_paths,
            baseline_columns=baseline_columns,
            teh_paths=[p.parent if p.is_file() else p for p in exp_resolved],
            teh_columns=teh_columns_test,
            n_participants=n_test,
            avg_row=avg_row,
            best_counts=best_counts,
            out_path=output_arg,
            score_kind=_TEST_LOGLIK,
        )

        gated_csvs = list(exp_resolved)
        if centaur_path is not None:
            gated_csvs.insert(0, centaur_path)
        if not _any_csv_has_gated(centaur_path, gated_csvs):
            return

        gated_out = (
            Path(args.output_gated).expanduser()
            if args.output_gated is not None
            else _gated_output_path(output_arg)
        )
        gated_out = gated_out.resolve() if gated_out.is_absolute() else (repo / gated_out).resolve()

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
        teh_columns_gated: Dict[str, str] = {}
        for label, csv_path in zip(run_labels, exp_resolved):
            test_scores = teh_test_by_label.get(label, {})
            gated = _read_loglik_csv(csv_path, _GATED_LOGLIK, required=False)
            teh_gated.append((label, _scores_for_gated_table(test_scores, gated)))
            if gated:
                teh_columns_gated[label] = _GATED_LOGLIK
            elif test_scores:
                teh_columns_gated[label] = f"{_GATED_LOGLIK} (fallback {_TEST_LOGLIK})"
            else:
                teh_columns_gated[label] = _GATED_LOGLIK

        gated_baseline_columns = dict(baseline_columns)
        for method in _BASELINE_METHODS:
            if method not in baseline_paths:
                continue
            if method == "Centaur" and centaur_path is not None:
                has_gated = _csv_has_column(centaur_path, _GATED_LOGLIK)
            else:
                has_gated = _csv_has_column(
                    _resolve_loglik_csv(baseline_paths[method]), _GATED_LOGLIK
                )
            gated_baseline_columns[method] = (
                _GATED_LOGLIK if has_gated else f"{_TEST_LOGLIK} (no {_GATED_LOGLIK} in run)"
            )

        n_gated, avg_g, best_g = _write_baseline_comparison_csv(
            out_path=gated_out,
            participant_ids=participant_ids,
            baselines=gated_baselines,
            teh_runs=teh_gated,
            bir=bir,
        )
        print(f"--- gated ({_GATED_LOGLIK}) ---")
        _print_run_summary(
            dataset_label=ds_label,
            config_key=config_key,
            baseline_paths=baseline_paths,
            baseline_columns=gated_baseline_columns,
            teh_paths=[p.parent if p.is_file() else p for p in exp_resolved],
            teh_columns=teh_columns_gated,
            n_participants=n_gated,
            avg_row=avg_g,
            best_counts=best_g,
            out_path=gated_out,
            score_kind=_GATED_LOGLIK,
        )
        return

    # Legacy: no baseline config for this dataset key — Centaur column + TEH only.
    centaur_scores = dict(baseline_scores[-1][1]) if baseline_scores else {}

    th = float(args.similar_threshold)
    experiments_test: List[Tuple[str, Dict[int, float]]] = []
    for label, csv_path in zip(run_labels, exp_resolved):
        experiments_test.append(
            (label, _read_loglik_csv(csv_path, _TEST_LOGLIK, required=True))
        )

    n_test = _write_legacy_comparison_csv(
        out_path=output_arg,
        participant_ids=participant_ids,
        centaur=centaur_scores,
        experiments=experiments_test,
        bir=bir,
        similar_threshold=th,
    )
    centaur_note = str(centaur_path) if centaur_path is not None else "(blank)"
    print(
        f"Wrote {output_arg} (legacy layout, {_TEST_LOGLIK}, dataset={ds_label}, "
        f"{n_test} participants, {len(experiments_test)} TEH runs, centaur={centaur_note})."
    )

    if not _any_csv_has_gated(centaur_path, exp_resolved):
        return

    gated_out = (
        Path(args.output_gated).expanduser()
        if args.output_gated is not None
        else _gated_output_path(output_arg)
    )
    gated_out = gated_out.resolve() if gated_out.is_absolute() else (repo / gated_out).resolve()

    centaur_gated: Dict[int, float] = {}
    if centaur_path is not None:
        centaur_gated = _read_loglik_csv(centaur_path, _GATED_LOGLIK, required=False)
    centaur_for_gated = _scores_for_gated_table(centaur_scores, centaur_gated)
    experiments_gated: List[Tuple[str, Dict[int, float]]] = []
    for label, csv_path in zip(run_labels, exp_resolved):
        experiments_gated.append(
            (label, _read_loglik_csv(csv_path, _GATED_LOGLIK, required=False))
        )

    n_gated = _write_legacy_comparison_csv(
        out_path=gated_out,
        participant_ids=participant_ids,
        centaur=centaur_for_gated,
        experiments=experiments_gated,
        bir=bir,
        similar_threshold=th,
    )
    print(f"Wrote {gated_out} (legacy gated, {n_gated} participants).")


if __name__ == "__main__":
    main()
