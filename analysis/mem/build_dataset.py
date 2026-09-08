#!/usr/bin/env python3
"""Build a schema-v2 MEM analysis CSV from mem_trace.jsonl + annotations_v2.jsonl.

Directional behavioral columns (binary, never averaged):
  {motif}_added / {motif}_removed / {motif}_modified

Structural columns:
  structural_{operation}

Writes:
  --output_csv
  sibling build_exclusions.jsonl (every excluded row + reason)
  sibling build_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mem.schema_v2 import (  # noqa: E402
    BEHAVIORAL_MOTIFS_V2,
    SCHEMA_VERSION,
    STRUCTURAL_OPERATIONS_V2,
    all_directional_behavioral_columns,
    all_structural_columns,
    annotation_resume_key,
    directional_flags_from_annotation,
    is_schema_v2_row,
)
from utils.mem.trace import iter_jsonl_records, record_contains_test_metrics  # noqa: E402


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _load_annotations_v2(path: Path) -> Dict[Tuple[Any, str], Dict[str, Any]]:
    """Index by (participant_id, candidate_id). Ignore non-v2 rows."""
    by_key: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"annotations file not found: {path}")
    n_skipped_v1 = 0
    for rec in iter_jsonl_records([path]):
        if record_contains_test_metrics(rec):
            raise ValueError(
                "Test metric keys must not appear in annotations "
                f"(candidate_id={rec.get('candidate_id')!r})"
            )
        if not is_schema_v2_row(rec):
            n_skipped_v1 += 1
            continue
        cid = rec.get("candidate_id")
        if not isinstance(cid, str):
            continue
        if "participant_id" not in rec:
            continue
        key = annotation_resume_key(rec.get("participant_id"), cid)
        by_key[key] = rec
    if n_skipped_v1:
        print(
            f"[build] Skipped {n_skipped_v1} non-v2 annotation row(s) "
            "(v1 never treated as completed v2).",
            flush=True,
        )
    return by_key


def _discover_trace_files(run_dir: Path) -> List[Path]:
    run_dir = Path(run_dir)
    found: Set[Path] = set()
    direct = run_dir / "mem_trace.jsonl"
    if direct.is_file():
        found.add(direct.resolve())
    for root, _dirs, files in os.walk(run_dir, followlinks=True):
        if "mem_trace.jsonl" in files:
            found.add((Path(root) / "mem_trace.jsonl").resolve())
    return sorted(found)


def _iter_candidate_traces(run_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in _discover_trace_files(run_dir):
        for rec in iter_jsonl_records([path]):
            if rec.get("record_type") == "candidate":
                if record_contains_test_metrics(rec):
                    raise ValueError(
                        "Test metric keys must not appear in mem_trace candidate records"
                    )
                yield rec


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows_v2(
    *,
    run_dir: Path,
    annotations: Dict[Tuple[Any, str], Dict[str, Any]],
    phase: str = "evolution",
    source: str = "normal",
    require_runtime_valid: bool = True,
    require_finite_delta_f: bool = True,
    require_annotation: bool = True,
    exclusions_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Build analysis rows; record every exclusion reason (no silent drops)."""
    rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[Any, str]] = set()
    excl: Counter = Counter()
    dir_cols = all_directional_behavioral_columns()
    struct_cols = all_structural_columns()

    for rec in _iter_candidate_traces(run_dir):
        pid = rec.get("participant_id")
        cid = str(rec.get("candidate_id"))
        key = annotation_resume_key(pid, cid)
        base_excl = {
            "participant_id": pid,
            "candidate_id": cid,
            "iteration": rec.get("iteration"),
            "source": rec.get("source"),
            "phase": rec.get("phase"),
        }

        def _exclude(reason: str) -> None:
            excl[reason] += 1
            if exclusions_path is not None:
                _append_jsonl(exclusions_path, {**base_excl, "reason": reason})

        if phase and rec.get("phase") != phase:
            _exclude("excl_phase")
            continue
        if source and rec.get("source") != source:
            _exclude(f"excl_source_{rec.get('source')}")
            continue
        if require_runtime_valid and not rec.get("runtime_valid"):
            _exclude("excl_not_runtime_valid")
            continue
        if require_finite_delta_f and not _is_finite(rec.get("delta_f")):
            _exclude("excl_delta_f_nonfinite_or_none")
            continue
        ann = annotations.get(key)
        if require_annotation and ann is None:
            _exclude("excl_missing_annotation_v2")
            continue
        if key in seen:
            _exclude("excl_duplicate_participant_candidate")
            continue
        seen.add(key)

        flags = directional_flags_from_annotation(ann) if ann else {
            c: 0 for c in dir_cols + struct_cols
        }
        row: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": rec.get("run_id"),
            "dataset": rec.get("dataset"),
            "participant_id": pid,
            "phase": rec.get("phase"),
            "iteration": rec.get("iteration"),
            "candidate_id": cid,
            "candidate_idx": rec.get("candidate_idx"),
            "source": rec.get("source"),
            "runtime_valid": rec.get("runtime_valid"),
            "train_loglik": rec.get("train_loglik"),
            "val_loglik": rec.get("val_loglik"),
            "selection_score": rec.get("selection_score"),
            "reference_parent_id": rec.get("reference_parent_id"),
            "reference_parent_score": rec.get("reference_parent_score"),
            "reference_kind": rec.get("reference_kind"),
            "delta_f": rec.get("delta_f"),
            "survived_elite_truncation": rec.get("survived_elite_truncation"),
            "evolution_selection_score": rec.get("evolution_selection_score"),
            "no_meaningful_change": int(bool(ann.get("no_meaningful_change"))) if ann else None,
            "confidence": ann.get("confidence") if ann else None,
            "added_motifs": json.dumps(ann.get("added_motifs", []), ensure_ascii=False)
            if ann
            else None,
            "removed_motifs": json.dumps(ann.get("removed_motifs", []), ensure_ascii=False)
            if ann
            else None,
            "modified_motifs": json.dumps(ann.get("modified_motifs", []), ensure_ascii=False)
            if ann
            else None,
            "structural_operations": json.dumps(
                ann.get("structural_operations", []), ensure_ascii=False
            )
            if ann
            else None,
        }
        for c in dir_cols + struct_cols:
            row[c] = int(flags.get(c, 0))
        rows.append(row)
    return rows, excl


def _descriptive_counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts by direction, dataset, participant (never averages add/rem/mod)."""
    by_dataset: Dict[str, Counter] = defaultdict(Counter)
    by_participant: Dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    dir_cols = all_directional_behavioral_columns()
    struct_cols = all_structural_columns()
    for r in rows:
        ds = str(r.get("dataset"))
        pid = str(r.get("participant_id"))
        for c in dir_cols + struct_cols:
            if int(r.get(c) or 0) == 1:
                overall[c] += 1
                by_dataset[ds][c] += 1
                by_participant[pid][c] += 1
        if r.get("no_meaningful_change"):
            overall["no_meaningful_change"] += 1
            by_dataset[ds]["no_meaningful_change"] += 1
            by_participant[pid]["no_meaningful_change"] += 1
    return {
        "overall": dict(overall),
        "by_dataset": {k: dict(v) for k, v in sorted(by_dataset.items())},
        "by_participant": {k: dict(v) for k, v in sorted(by_participant.items())},
        "behavioral_motifs": list(BEHAVIORAL_MOTIFS_V2),
        "structural_operations": list(STRUCTURAL_OPERATIONS_V2),
        "n_rows": len(rows),
    }


# Backward-compatible name used by older tests; schema v1 fields removed.
def build_rows(
    *,
    run_dir: Path,
    annotations: Dict[str, Dict[str, Any]],
    phase: str = "evolution",
    source: str = "normal",
    require_runtime_valid: bool = True,
    require_finite_delta_f: bool = True,
    require_annotation: bool = True,
) -> List[Dict[str, Any]]:
    """Legacy wrapper: annotations keyed by candidate_id only (tests). Prefer build_rows_v2."""
    keyed: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    for cid, ann in annotations.items():
        # Infer participant from annotation if present; else allow None key match via scan
        pid = ann.get("participant_id", 0)
        a = dict(ann)
        a.setdefault("schema_version", SCHEMA_VERSION)
        keyed[annotation_resume_key(pid, cid)] = a
    rows, _excl = build_rows_v2(
        run_dir=run_dir,
        annotations=keyed,
        phase=phase,
        source=source,
        require_runtime_valid=require_runtime_valid,
        require_finite_delta_f=require_finite_delta_f,
        require_annotation=require_annotation,
        exclusions_path=None,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="annotations_v2.jsonl path (schema_version=2 only)",
    )
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--phase", type=str, default="evolution")
    parser.add_argument(
        "--source",
        type=str,
        default="normal",
        help="Candidate source filter (default: normal). Use empty string for all.",
    )
    args = parser.parse_args()

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    excl_path = out.with_name(out.stem + "_exclusions.jsonl")
    summary_path = out.with_name(out.stem + "_build_summary.json")
    if excl_path.exists():
        excl_path.unlink()

    annotations = _load_annotations_v2(Path(args.annotations))
    source = args.source if args.source != "" else ""
    rows, excl = build_rows_v2(
        run_dir=Path(args.run_dir),
        annotations=annotations,
        phase=args.phase,
        source=source,
        exclusions_path=excl_path,
    )

    dir_cols = all_directional_behavioral_columns()
    struct_cols = all_structural_columns()
    fieldnames = [
        "schema_version",
        "run_id",
        "dataset",
        "participant_id",
        "phase",
        "iteration",
        "candidate_id",
        "candidate_idx",
        "source",
        "runtime_valid",
        "train_loglik",
        "val_loglik",
        "selection_score",
        "reference_parent_id",
        "reference_parent_score",
        "reference_kind",
        "delta_f",
        "survived_elite_truncation",
        "evolution_selection_score",
        "no_meaningful_change",
        "confidence",
        "added_motifs",
        "removed_motifs",
        "modified_motifs",
        "structural_operations",
        *dir_cols,
        *struct_cols,
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_rows": len(rows),
        "n_annotations_v2_indexed": len(annotations),
        "exclusions_by_reason": dict(sorted(excl.items())),
        "n_exclusions": int(sum(excl.values())),
        "descriptive_counts": _descriptive_counts(rows),
        "output_csv": str(out),
        "exclusions_path": str(excl_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {out}")
    print(f"Wrote exclusions to {excl_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
