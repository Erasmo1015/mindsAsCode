"""Reporting utilities for teh_psych prototype runs."""
from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


CSV_COLUMNS = [
    "experiment_id",
    "status",
    "stage_reached",
    "failure_stage",
    "failure_message",
    "adapter_type",
    "parse_plan_status",
    "parse_plan_model",
    "parse_plan_cached",
    "parse_plan_human_review_required",
    "parse_plan_raw_format_type",
    "parse_plan_failure_message",
    "n_rows_total",
    "n_rows_used",
    "n_parse_plan_rows",
    "n_parsed_trials",
    "n_prediction_trials",
    "num_actions_min",
    "num_actions_max",
    "is_variable_k",
    "train_loglik",
    "val_loglik",
    "test_loglik",
    "best_program_path",
    "parse_plan_path",
    "used_existing_parser_fallback",
    "notes",
]


@dataclass
class DatasetResult:
    experiment_id: str
    status: str = "pending"
    stage_reached: str = "discover_experiments"
    failure_stage: str = ""
    failure_message: str = ""
    adapter_type: str = ""
    parse_plan_status: str = ""
    parse_plan_model: str = ""
    parse_plan_cached: bool = False
    parse_plan_human_review_required: bool = False
    parse_plan_raw_format_type: str = ""
    parse_plan_failure_message: str = ""
    n_rows_total: int = 0
    n_rows_used: int = 0
    n_parse_plan_rows: int = 0
    n_parsed_trials: int = 0
    n_prediction_trials: int = 0
    parse_attempt: Optional[int] = None
    num_actions_min: Optional[int] = None
    num_actions_max: Optional[int] = None
    is_variable_k: bool = False
    train_loglik: Optional[float] = None
    val_loglik: Optional[float] = None
    test_loglik: Optional[float] = None
    best_program_path: str = ""
    parse_plan_path: str = ""
    used_existing_parser_fallback: bool = False
    notes: str = ""
    traceback: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_csv_row(self) -> Dict[str, Any]:
        row = {k: getattr(self, k, "") for k in CSV_COLUMNS}
        for key in ("train_loglik", "val_loglik", "test_loglik"):
            if row.get(key) is None:
                row[key] = ""
        row["is_variable_k"] = "true" if self.is_variable_k else "false"
        row["parse_plan_cached"] = "true" if self.parse_plan_cached else "false"
        row["parse_plan_human_review_required"] = (
            "true" if self.parse_plan_human_review_required else "false"
        )
        row["used_existing_parser_fallback"] = (
            "true" if self.used_existing_parser_fallback else "false"
        )
        return row

    def to_json(self) -> Dict[str, Any]:
        d = self.to_csv_row()
        if self.traceback:
            d["traceback"] = self.traceback
        if self.extra:
            d["extra"] = self.extra
        return d


def ensure_summary_dir(run_dir: Path, prototype_summary_dir: Optional[str] = None) -> Path:
    if prototype_summary_dir:
        summary = Path(prototype_summary_dir).expanduser()
        if not summary.is_absolute():
            summary = run_dir / summary
    else:
        summary = run_dir / "prototype_summary"
    summary.mkdir(parents=True, exist_ok=True)
    return summary


def dataset_debug_dir(summary_dir: Path, experiment_id: str) -> Path:
    from utils.teh_psych.dataset_loop import safe_experiment_id_for_path

    d = summary_dir / "by_dataset" / safe_experiment_id_for_path(experiment_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_dataset_result_csv(summary_dir: Path, result: DatasetResult) -> None:
    path = summary_dir / "dataset_results.csv"
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(result.to_csv_row())


def append_dataset_result_jsonl(summary_dir: Path, result: DatasetResult) -> None:
    path = summary_dir / "dataset_results.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_json()) + "\n")


def write_failure_summary_md(summary_dir: Path, results: List[DatasetResult]) -> None:
    failed = [r for r in results if r.status != "success"]
    counts: Dict[str, int] = {}
    for r in failed:
        stage = r.failure_stage or "unknown_error"
        counts[stage] = counts.get(stage, 0) + 1
    lines = ["# Failure summary\n", f"Total failed: {len(failed)}\n\n", "## By failure stage\n\n"]
    for stage, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- **{stage}**: {n}\n")
    lines.append("\n## Failed experiments\n\n")
    for r in failed:
        lines.append(
            f"- `{r.experiment_id}` — stage `{r.failure_stage}`: {r.failure_message}\n"
        )
    (summary_dir / "failure_summary.md").write_text("".join(lines), encoding="utf-8")


def write_success_summary_md(summary_dir: Path, results: List[DatasetResult]) -> None:
    ok = [r for r in results if r.status == "success"]
    lines = ["# Success summary\n", f"Total succeeded: {len(ok)}\n\n"]
    if not ok:
        lines.append("_No successful datasets._\n")
    else:
        lines.append(
            "| experiment_id | adapter | n_prediction_trials | train_loglik | val_loglik | test_loglik |\n"
        )
        lines.append("|---|---|---:|---:|---:|---:|\n")
        for r in ok:
            lines.append(
                f"| `{r.experiment_id}` | {r.adapter_type} | {r.n_prediction_trials} | "
                f"{_fmt(r.train_loglik)} | {_fmt(r.val_loglik)} | {_fmt(r.test_loglik)} |\n"
            )
    (summary_dir / "success_summary.md").write_text("".join(lines), encoding="utf-8")


def _fmt(val: Optional[float]) -> str:
    if val is None:
        return ""
    return f"{val:.4f}"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def record_failure(
    result: DatasetResult,
    stage: str,
    message: str,
    exc: Optional[BaseException] = None,
) -> DatasetResult:
    result.status = "failed"
    result.failure_stage = stage
    result.failure_message = message
    result.stage_reached = stage
    if exc is not None:
        result.traceback = traceback.format_exc()
    return result


def finalize_summaries(summary_dir: Path, results: List[DatasetResult]) -> None:
    write_failure_summary_md(summary_dir, results)
    write_success_summary_md(summary_dir, results)


DEFAULT_PRIORITY_TARGET_CSV = (
    Path(__file__).resolve().parents[2]
    / "analysis"
    / "report"
    / "260715_teh_psych"
    / "psych101_priority_target_experiments.csv"
)

PRIORITY_RESULT_COLUMNS = [
    "experiment_id",
    "dataset_name",
    "task_group",
    "status",
    "parse_attempt",
    "n_participants",
    "n_trials",
    "test_loglik",
]

PRIORITY_GROUP_COLUMNS = [
    "task_group",
    "total",
    "success",
    "failed",
    "incomplete",
    "mean_test_loglik",
]


def load_priority_task_groups(
    priority_csv: Optional[Path] = None,
) -> Dict[str, str]:
    """Map experiment_id -> task_group from the priority target CSV."""
    path = Path(priority_csv) if priority_csv is not None else DEFAULT_PRIORITY_TARGET_CSV
    if not path.is_file():
        return {}
    out: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("experiment_id") or "").strip()
            if not eid:
                continue
            out[eid] = str(row.get("task_group") or "").strip() or "N/A"
    return out


def _priority_status(result: Optional[DatasetResult]) -> str:
    if result is None:
        return "not_started"
    status = str(result.status or "").strip().lower()
    if status == "success":
        return "success"
    if status in ("pending", "running"):
        return "incomplete"
    if status == "failed":
        return "failed"
    if status in ("incomplete", "not_started"):
        return status
    # Treat unexpected non-success terminal states as failed.
    return "failed" if status else "not_started"


def _na_or_int(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return "N/A"
    if iv <= 0:
        return "N/A"
    return str(iv)


def write_priority_results_summaries(
    run_dir: Path,
    results: List[DatasetResult],
    *,
    requested_experiment_ids: Sequence[str],
    priority_csv: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Write experiment-level and group-level priority summary CSVs under ``run_dir``.

    Includes every requested experiment_id (failed/incomplete/not_started rows preserved).
    """
    from collections import defaultdict

    task_groups = load_priority_task_groups(priority_csv)
    by_id = {r.experiment_id: r for r in results}
    requested = [str(x).strip() for x in requested_experiment_ids if str(x).strip()]
    # De-dupe while preserving request order.
    seen: set = set()
    ordered_ids: List[str] = []
    for eid in requested:
        if eid in seen:
            continue
        seen.add(eid)
        ordered_ids.append(eid)

    exp_rows: List[Dict[str, str]] = []
    for eid in ordered_ids:
        result = by_id.get(eid)
        status = _priority_status(result)
        dataset_name = eid.split("/", 1)[0] if "/" in eid else eid
        task_group = task_groups.get(eid, "N/A")
        parse_attempt = _na_or_int(result.parse_attempt if result else None)
        n_participants = _na_or_int(result.n_rows_used if result else None)
        n_trials = _na_or_int(result.n_prediction_trials if result else None)
        # Keep full precision for successful evals; N/A otherwise.
        if result is not None and result.test_loglik is not None and status == "success":
            test_loglik = format(float(result.test_loglik), ".16g")
        else:
            test_loglik = "N/A"
        exp_rows.append(
            {
                "experiment_id": eid,
                "dataset_name": dataset_name,
                "task_group": task_group,
                "status": status,
                "parse_attempt": parse_attempt,
                "n_participants": n_participants,
                "n_trials": n_trials,
                "test_loglik": test_loglik,
            }
        )

    # Also surface any unexpected processed IDs not in the request list.
    for result in results:
        if result.experiment_id in seen:
            continue
        status = _priority_status(result)
        eid = result.experiment_id
        exp_rows.append(
            {
                "experiment_id": eid,
                "dataset_name": eid.split("/", 1)[0] if "/" in eid else eid,
                "task_group": task_groups.get(eid, "N/A"),
                "status": status,
                "parse_attempt": _na_or_int(result.parse_attempt),
                "n_participants": _na_or_int(result.n_rows_used),
                "n_trials": _na_or_int(result.n_prediction_trials),
                "test_loglik": (
                    format(float(result.test_loglik), ".16g")
                    if result.test_loglik is not None and status == "success"
                    else "N/A"
                ),
            }
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    exp_path = run_dir / "priority_results_summary.csv"
    with exp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRIORITY_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(exp_rows)

    group_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "success": 0,
            "failed": 0,
            "incomplete": 0,
            "not_started": 0,
            "logliks": [],
        }
    )
    for row in exp_rows:
        g = row["task_group"] if row["task_group"] and row["task_group"] != "N/A" else "unknown"
        st = group_stats[g]
        st["total"] += 1
        status = row["status"]
        if status == "success":
            st["success"] += 1
            if row["test_loglik"] != "N/A":
                st["logliks"].append(float(row["test_loglik"]))
        elif status == "failed":
            st["failed"] += 1
        elif status == "incomplete":
            st["incomplete"] += 1
        elif status == "not_started":
            # Count as incomplete bucket for group rollup only if never started.
            st["incomplete"] += 1
            st["not_started"] += 1
        else:
            st["failed"] += 1

    group_path = run_dir / "priority_results_by_group.csv"
    with group_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRIORITY_GROUP_COLUMNS)
        writer.writeheader()
        for task_group in sorted(group_stats.keys()):
            st = group_stats[task_group]
            if st["logliks"]:
                mean_ll = sum(st["logliks"]) / len(st["logliks"])
                mean_disp = f"{mean_ll:.2f}"
            else:
                mean_disp = "N/A"
            writer.writerow(
                {
                    "task_group": task_group,
                    "total": st["total"],
                    "success": st["success"],
                    "failed": st["failed"],
                    "incomplete": st["incomplete"],
                    "mean_test_loglik": mean_disp,
                }
            )
    return exp_path, group_path
