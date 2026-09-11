#!/usr/bin/env python3
"""Load / validate MEM annotation queue manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_JOB_KEYS = (
    "dataset",
    "run_id",
    "teh_path",
    "reference_mode",
    "annotation_status",
    "mem_wave",
    "intended_out",
)


def default_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "cluster"
        / "2026Sep_MEM"
        / "manifests"
        / "mem_annotation_queue.json"
    )


def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else default_manifest_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data.get("jobs"), list):
        raise ValueError(f"manifest missing jobs list: {p}")
    return data


def validate_manifest(
    data: Dict[str, Any],
    *,
    repo_root: Optional[Path] = None,
    require_teh_exists: bool = True,
) -> List[str]:
    """Return list of validation errors (empty = ok)."""
    errors: List[str] = []
    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    jobs = data.get("jobs") or []
    seen = set()
    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            errors.append(f"jobs[{i}] not an object")
            continue
        for k in REQUIRED_JOB_KEYS:
            if k not in job or job[k] in (None, ""):
                errors.append(f"jobs[{i}] missing {k}")
        key = (job.get("dataset"), job.get("run_id"), job.get("mem_wave"))
        if key in seen:
            errors.append(f"duplicate job key {key}")
        seen.add(key)
        mode = job.get("reference_mode")
        if mode not in ("pool_best_proxy", "live_mem_trace"):
            errors.append(f"jobs[{i}] bad reference_mode={mode!r}")
        if job.get("skip_reconstruct") and mode != "live_mem_trace":
            errors.append(
                f"jobs[{i}] skip_reconstruct requires reference_mode=live_mem_trace"
            )
        teh = repo / str(job.get("teh_path") or "")
        if require_teh_exists and not teh.is_dir():
            errors.append(f"jobs[{i}] teh_path missing: {teh}")
        if job.get("skip_reconstruct"):
            src = job.get("trace_source_dir") or job.get("teh_path")
            if require_teh_exists and not (repo / str(src)).is_dir():
                errors.append(f"jobs[{i}] trace_source_dir missing: {src}")
    return errors


def queue_jobs(data: Dict[str, Any], *, include_optional: bool = True) -> List[Dict[str, Any]]:
    out = []
    for job in data.get("jobs") or []:
        if not job.get("include_in_submit_queue"):
            continue
        status = str(job.get("annotation_status") or "")
        if status.startswith("done") and status != "done_fix_rerun1_only":
            continue
        if (not include_optional) and status == "pending_optional":
            continue
        out.append(job)
    return out


def write_tsv(data: Dict[str, Any], path: Path) -> None:
    cols = [
        "dataset",
        "dataset_label",
        "run_id",
        "teh_path",
        "reference_mode",
        "annotation_status",
        "mem_wave",
        "intended_out",
        "skip_reconstruct",
        "include_in_submit_queue",
    ]
    lines = ["\t".join(cols)]
    for job in data.get("jobs") or []:
        lines.append(
            "\t".join(str(job.get(c, "")) for c in cols)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="")
    p.add_argument("--write_tsv", default="")
    p.add_argument("--no_require_teh", action="store_true")
    args = p.parse_args()
    path = Path(args.manifest) if args.manifest else default_manifest_path()
    data = load_manifest(path)
    errs = validate_manifest(data, require_teh_exists=not args.no_require_teh)
    if args.write_tsv:
        write_tsv(data, Path(args.write_tsv))
    if errs:
        print("VALIDATION ERRORS:")
        for e in errs:
            print(" -", e)
        raise SystemExit(1)
    q = queue_jobs(data)
    print(f"OK jobs={len(data['jobs'])} queue={len(q)} path={path}")
