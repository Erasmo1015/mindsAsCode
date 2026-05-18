#!/usr/bin/env python3
"""
Backfill val_loglik and gated_test_loglik in experiment-level loglik CSVs (May 18, 2026 fixes).

For each experiment path (per participant, in participant_id order):
1. Fill missing ``val_loglik`` from participant artifacts, then last-iteration metrics,
   then re-evaluate ``best_program.py`` on the val split (CPC18) when still missing.
2. Fill missing ``gated_test_loglik`` from refinement/results artifacts, or use ``test_loglik``.

Updates only ``val_loglik`` / ``gated_test_loglik`` in participant_details_loglik.csv and
``avg_val_loglik`` / ``avg_gated_test_loglik`` in summary_loglik.csv. Other columns and files
under the run directory are left unchanged.

Usage:
  python utils/adhoc_fix/adhoc_fix_report_May18.py \\
    generated_outputs/choice13k/non_strict/run_260518_000357 \\
    generated_outputs/cpc18/non_strict/run_260517_211601

  python utils/adhoc_fix/adhoc_fix_report_May18.py --dry-run <paths...>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_PARTICIPANT_DIR_RE = re.compile(r"^participant_(\d+)$")
_ITERATION_DIR_RE = re.compile(r"^iteration_(\d+)$")
_LOG_CSV_NDIGITS = 4
_LOGLIK_CLAMP_EPS = 1e-9


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_loglik(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{round(float(value), _LOG_CSV_NDIGITS):.{_LOG_CSV_NDIGITS}f}"


def _participant_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_dir():
            continue
        match = _PARTICIPANT_DIR_RE.match(path.name)
        if match:
            out.append((int(match.group(1)), path))
    return out


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _infer_dataset(run_dir: Path) -> Optional[str]:
    parts = run_dir.resolve().parts
    for dataset in ("cpc18", "choice13k", "mixed_gambles"):
        if dataset in parts:
            return dataset
    return None


def _parse_run_config(run_dir: Path, participant_dir: Path) -> Dict[str, Any]:
    """Parse split/eval args from saved command.txt (run log or participant dir)."""
    config: Dict[str, Any] = {
        "split_ratio": 0.8,
        "split_seed": 0,
        "n_eval_seeds": 1,
    }
    cmd_paths = [
        run_dir / "log" / "command.txt",
        participant_dir / "command.txt",
    ]
    text = ""
    for path in cmd_paths:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            break
    if not text:
        return config
    for key in ("split_ratio", "split_seed", "n_eval_seeds"):
        match = re.search(rf"--{key}\s+(\S+)", text)
        if match:
            val = _safe_float(match.group(1))
            if val is not None:
                config[key] = int(val) if key.endswith("_seed") or key == "n_eval_seeds" else val
    return config


def _val_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    direct = _safe_float(payload.get("val_loglik"))
    if direct is not None:
        return direct
    for block_key in ("overall_best_train", "overall_best_test"):
        block = payload.get(block_key)
        if isinstance(block, dict):
            nested = _safe_float(block.get("val_loglik"))
            if nested is not None:
                return nested
    final_pool = payload.get("final_pool_best")
    if isinstance(final_pool, dict):
        nested = _safe_float(final_pool.get("val_loglik"))
        if nested is not None:
            return nested
    return None


def _gated_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    direct = _safe_float(payload.get("gated_test_loglik"))
    if direct is not None:
        return direct
    overall_test = payload.get("overall_best_test")
    if isinstance(overall_test, dict):
        nested = _safe_float(overall_test.get("gated_test_loglik"))
        if nested is not None:
            return nested
    final_pool = payload.get("final_pool_best")
    if isinstance(final_pool, dict):
        nested = _safe_float(final_pool.get("test_loglik"))
        if nested is not None:
            return nested
    return None


def _test_loglik_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    overall_test = payload.get("overall_best_test")
    if isinstance(overall_test, dict):
        test_ll = _safe_float(overall_test.get("test_loglik"))
        if test_ll is not None:
            return test_ll
    final_pool = payload.get("final_pool_best")
    if isinstance(final_pool, dict):
        test_ll = _safe_float(final_pool.get("test_loglik"))
        if test_ll is not None:
            return test_ll
    return _safe_float(payload.get("test_loglik"))


def _loglik_from_summary_csv(path: Path, column: str) -> Optional[float]:
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    if not rows:
        return None
    return _safe_float(rows[0].get(column))


def _val_from_last_iteration_metrics(participant_dir: Path) -> Optional[float]:
    """Use best_val_loglik from the highest-numbered iteration_*/metrics.json."""
    best_iter = -1
    best_val: Optional[float] = None
    for path in participant_dir.iterdir():
        if not path.is_dir():
            continue
        match = _ITERATION_DIR_RE.match(path.name)
        if not match:
            continue
        metrics = _load_json(path / "metrics.json")
        if metrics is None:
            continue
        iteration = int(match.group(1))
        val_ll = _safe_float(metrics.get("best_val_loglik"))
        if val_ll is not None and iteration >= best_iter:
            best_iter = iteration
            best_val = val_ll
    return best_val


def extract_val_loglik_from_folder(participant_dir: Path) -> Optional[float]:
    """Read val_loglik from saved participant artifacts (no re-evaluation)."""
    json_sources = [
        participant_dir / "results.json",
        participant_dir / "refinement" / "results.json",
    ]
    for path in json_sources:
        payload = _load_json(path)
        if payload is None:
            continue
        val_ll = _val_from_payload(payload)
        if val_ll is not None:
            return val_ll
    for path in (
        participant_dir / "summary.csv",
        participant_dir / "refinement" / "summary_loglik.csv",
    ):
        val_ll = _loglik_from_summary_csv(path, "val_loglik")
        if val_ll is not None:
            return val_ll
    return _val_from_last_iteration_metrics(participant_dir)


def extract_gated_test_loglik(participant_dir: Path) -> Optional[float]:
    """Prefer refinement / results artifacts; None if not found in folder."""
    sources = [
        participant_dir / "refinement" / "results.json",
        participant_dir / "results.json",
        participant_dir / "refinement" / "summary_loglik.csv",
        participant_dir / "summary.csv",
    ]
    for path in sources:
        if path.suffix == ".json":
            payload = _load_json(path)
            if payload is None:
                continue
            gated = _gated_from_payload(payload)
            if gated is not None:
                return gated
        else:
            gated = _loglik_from_summary_csv(path, "gated_test_loglik")
            if gated is not None:
                return gated
    return None


def _evaluate_cpc18_split_avg_loglik(
    choose_fn: Callable,
    trials: List[Dict[str, Any]],
    *,
    n_seeds: int = 1,
) -> Optional[float]:
    """Bernoulli log-lik on CPC18 held-out trials (matches Template evaluate_cpc18_split_program)."""
    import numpy as np

    total = len(trials)
    if total == 0:
        return None
    seed_avg_logliks: List[float] = []

    def _one_pass() -> float:
        loglik_acc = 0.0
        for trial in trials:
            y = int(trial["action"])
            try:
                p_raw = choose_fn(trial["problem"], trial["history"])
            except Exception:
                p = 0.5
                p_clamped = min(max(p, _LOGLIK_CLAMP_EPS), 1.0 - _LOGLIK_CLAMP_EPS)
                loglik_acc += y * np.log(p_clamped) + (1 - y) * np.log(1.0 - p_clamped)
                continue
            if isinstance(p_raw, bool) or (
                isinstance(p_raw, (int, np.integer)) and int(p_raw) in (0, 1)
            ):
                p_use = 1.0 if int(p_raw) == 1 else 0.0
            elif isinstance(p_raw, float):
                p_use = p_raw
            else:
                return float("-inf")
            if not (0.0 <= p_use <= 1.0):
                return float("-inf")
            p = min(max(p_use, _LOGLIK_CLAMP_EPS), 1.0 - _LOGLIK_CLAMP_EPS)
            loglik_acc += y * np.log(p) + (1 - y) * np.log(1.0 - p)
        return loglik_acc / total

    for _ in range(max(1, int(n_eval_seeds))):
        seed_avg_logliks.append(_one_pass())
    return float(sum(seed_avg_logliks) / len(seed_avg_logliks))


def recompute_val_loglik_cpc18(
    participant_dir: Path,
    participant_id: int,
    *,
    split_ratio: float,
    split_seed: int,
    n_eval_seeds: int,
) -> Optional[float]:
    """Evaluate best_program.py on the CPC18 val split."""
    best_path = participant_dir / "best_program.py"
    if not best_path.exists():
        return None
    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from baselines.choice13k.program_executor import compile_program
        from data_modules.cpc18 import load_cpc18_track2_data, split_cpc18_trials_three_way
    except ImportError as exc:
        print(f"  [WARN] participant_{participant_id}: CPC18 recompute import failed: {exc}")
        return None

    choose_fn = compile_program(best_path.read_text(encoding="utf-8"))
    if choose_fn is None:
        print(f"  [WARN] participant_{participant_id}: best_program.py failed to compile")
        return None
    try:
        participant_data = load_cpc18_track2_data(
            data_path=str(repo / "datasets" / "cpc18"),
            participant_id=int(participant_id),
        )
        _train, val_trials, _test = split_cpc18_trials_three_way(
            participant_data,
            split_ratio=float(split_ratio),
            split_seed=int(split_seed),
        )
    except Exception as exc:
        print(f"  [WARN] participant_{participant_id}: CPC18 val trials load failed: {exc}")
        return None
    if not val_trials:
        return None
    return _evaluate_cpc18_split_avg_loglik(
        choose_fn, val_trials, n_seeds=int(n_eval_seeds)
    )


def _loglik_triplet_from_folder(participant_dir: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """train, val, test loglik from results.json (for missing CSV rows only)."""
    payload = _load_json(participant_dir / "results.json")
    if payload is None:
        return None, None, None
    train_block = payload.get("overall_best_train")
    test_block = payload.get("overall_best_test")
    train_ll = val_ll = test_ll = None
    if isinstance(train_block, dict):
        train_ll = _safe_float(train_block.get("train_loglik"))
        val_ll = _safe_float(train_block.get("val_loglik"))
    if isinstance(test_block, dict):
        test_ll = _safe_float(test_block.get("test_loglik"))
        if val_ll is None:
            val_ll = _safe_float(test_block.get("val_loglik"))
    return train_ll, val_ll, test_ll


def extract_test_loglik_fallback(
    participant_dir: Path,
    csv_test_loglik: Optional[float],
) -> Optional[float]:
    if csv_test_loglik is not None:
        return csv_test_loglik
    payload = _load_json(participant_dir / "results.json")
    if payload is not None:
        test_ll = _test_loglik_from_payload(payload)
        if test_ll is not None:
            return test_ll
    refine_payload = _load_json(participant_dir / "refinement" / "results.json")
    if refine_payload is not None:
        return _test_loglik_from_payload(refine_payload)
    return None


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ensure_val_column(fieldnames: List[str]) -> List[str]:
    if "val_loglik" in fieldnames:
        return fieldnames
    if "test_loglik" in fieldnames:
        idx = fieldnames.index("test_loglik")
        return fieldnames[:idx] + ["val_loglik"] + fieldnames[idx:]
    return fieldnames + ["val_loglik"]


def _ensure_gated_column(fieldnames: List[str]) -> List[str]:
    if "gated_test_loglik" in fieldnames:
        return fieldnames
    if "test_loglik" in fieldnames:
        idx = fieldnames.index("test_loglik") + 1
        return fieldnames[:idx] + ["gated_test_loglik"] + fieldnames[idx:]
    return fieldnames + ["gated_test_loglik"]


def _row_by_participant(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    indexed: Dict[int, Dict[str, str]] = {}
    for row in rows:
        pid = _safe_float(row.get("participant_id"))
        if pid is not None:
            indexed[int(pid)] = row
    return indexed


def fix_experiment(run_dir: Path, *, dry_run: bool = False) -> None:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {run_dir}")

    details_path = run_dir / "participant_details_loglik.csv"
    summary_path = run_dir / "summary_loglik.csv"
    if not details_path.exists():
        raise FileNotFoundError(f"Missing {details_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    dataset = _infer_dataset(run_dir)
    run_config = _parse_run_config(run_dir, run_dir)

    fieldnames, rows = _read_csv(details_path)
    has_val_column = "val_loglik" in fieldnames
    fieldnames = _ensure_val_column(fieldnames)
    fieldnames = _ensure_gated_column(fieldnames)
    rows_by_pid = _row_by_participant(rows)

    val_updates: List[str] = []
    gated_updates: List[str] = []
    val_values: List[float] = []
    gated_values: List[float] = []

    for participant_id, participant_dir in _participant_dirs(run_dir):
        pconfig = _parse_run_config(run_dir, participant_dir)
        row = rows_by_pid.get(participant_id)
        if row is None:
            train_ll, val_ll, test_ll = _loglik_triplet_from_folder(participant_dir)
            if test_ll is None:
                print(
                    f"  [WARN] participant_{participant_id}: no CSV row and no results.json "
                    "test_loglik; skipping"
                )
                continue
            row = {
                "participant_id": str(participant_id),
                "train_loglik": _format_loglik(train_ll),
                "val_loglik": _format_loglik(val_ll),
                "test_loglik": _format_loglik(test_ll),
                "gated_test_loglik": "",
            }
            rows_by_pid[participant_id] = row
            print(
                f"  [ADD] participant_{participant_id}: appended missing CSV row from results.json"
            )

        # --- val_loglik (before gated) ---
        if has_val_column or "val_loglik" in fieldnames:
            csv_val = _safe_float(row.get("val_loglik"))
            if csv_val is None:
                val_ll = extract_val_loglik_from_folder(participant_dir)
                val_source = "participant artifacts"
                if val_ll is None and dataset == "cpc18":
                    val_ll = recompute_val_loglik_cpc18(
                        participant_dir,
                        participant_id,
                        split_ratio=float(pconfig.get("split_ratio", run_config["split_ratio"])),
                        split_seed=int(pconfig.get("split_seed", run_config["split_seed"])),
                        n_eval_seeds=int(pconfig.get("n_eval_seeds", run_config["n_eval_seeds"])),
                    )
                    if val_ll is not None:
                        val_source = "recomputed on val split (best_program.py)"
                if val_ll is not None:
                    old_val = row.get("val_loglik", "")
                    new_val = _format_loglik(val_ll)
                    row["val_loglik"] = new_val
                    val_values.append(float(val_ll))
                    if old_val != new_val:
                        val_updates.append(
                            f"  participant_{participant_id}: val_loglik "
                            f"{old_val!r} -> {new_val!r} ({val_source})"
                        )
                else:
                    print(
                        f"  [WARN] participant_{participant_id}: could not resolve val_loglik"
                    )
            else:
                val_values.append(float(csv_val))

        # --- gated_test_loglik ---
        csv_test = _safe_float(row.get("test_loglik"))
        gated = extract_gated_test_loglik(participant_dir)
        gated_source = "participant folder"
        if gated is None:
            gated = extract_test_loglik_fallback(participant_dir, csv_test)
            gated_source = "test_loglik fallback"
        if gated is None:
            print(
                f"  [WARN] participant_{participant_id}: no gated or test loglik; "
                "leaving gated empty"
            )
        else:
            old_gated = row.get("gated_test_loglik", "")
            new_gated = _format_loglik(gated)
            row["gated_test_loglik"] = new_gated
            gated_values.append(float(gated))
            if old_gated != new_gated:
                gated_updates.append(
                    f"  participant_{participant_id}: gated_test_loglik "
                    f"{old_gated!r} -> {new_gated!r} ({gated_source})"
                )

    ordered_rows = [rows_by_pid[pid] for pid in sorted(rows_by_pid)]

    summary_fieldnames, summary_rows = _read_csv(summary_path)
    if not summary_rows:
        raise RuntimeError(f"Empty summary CSV: {summary_path}")
    summary_row = dict(summary_rows[0])

    old_avg_val = summary_row.get("avg_val_loglik", "")
    new_avg_val = _format_loglik(
        sum(val_values) / len(val_values) if val_values else None
    )
    if "avg_val_loglik" in summary_fieldnames or has_val_column:
        if "avg_val_loglik" not in summary_fieldnames:
            summary_fieldnames = list(summary_fieldnames) + ["avg_val_loglik"]
        summary_row["avg_val_loglik"] = new_avg_val

    old_avg_gated = summary_row.get("avg_gated_test_loglik", "")
    new_avg_gated = _format_loglik(
        sum(gated_values) / len(gated_values) if gated_values else None
    )
    if "avg_gated_test_loglik" not in summary_fieldnames:
        summary_fieldnames = list(summary_fieldnames) + ["avg_gated_test_loglik"]
    summary_row["avg_gated_test_loglik"] = new_avg_gated

    print(f"\n=== {run_dir} ===")
    if val_updates:
        for line in val_updates:
            print(line)
    else:
        print("  (no participant val_loglik changes)")
    if gated_updates:
        for line in gated_updates:
            print(line)
    else:
        print("  (no participant gated_test_loglik changes)")
    print(
        f"  summary avg_val_loglik: {old_avg_val!r} -> {new_avg_val!r} "
        f"(n={len(val_values)})"
    )
    print(
        f"  summary avg_gated_test_loglik: {old_avg_gated!r} -> {new_avg_gated!r} "
        f"(n={len(gated_values)})"
    )

    if dry_run:
        print("  [dry-run] no files written")
        return

    _write_csv(details_path, fieldnames, ordered_rows)
    _write_csv(summary_path, summary_fieldnames, [summary_row])
    print(f"  [OK] updated {details_path.name} and {summary_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill val_loglik and gated_test_loglik in experiment loglik CSVs "
            "for one or more runs."
        )
    )
    parser.add_argument(
        "experiment_paths",
        nargs="+",
        help="Experiment run directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing files.",
    )
    args = parser.parse_args()

    errors = 0
    for raw_path in args.experiment_paths:
        run_dir = Path(raw_path)
        try:
            fix_experiment(run_dir, dry_run=args.dry_run)
        except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
            print(f"\n[ERROR] {run_dir}: {exc}")
            errors += 1

    if errors:
        raise SystemExit(errors)


if __name__ == "__main__":
    main()
