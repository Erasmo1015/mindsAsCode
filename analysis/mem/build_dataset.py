#!/usr/bin/env python3
"""Build a MEM analysis CSV from mem_trace.jsonl + annotations.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mem.trace import iter_jsonl_records  # noqa: E402


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _load_annotations(path: Path) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"annotations file not found: {path}")
    for rec in iter_jsonl_records([path]):
        cid = rec.get("candidate_id")
        if not isinstance(cid, str):
            continue
        by_id[cid] = rec
    return by_id


def _iter_candidate_traces(run_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(run_dir.rglob("mem_trace.jsonl")):
        for rec in iter_jsonl_records([path]):
            if rec.get("record_type") == "candidate":
                yield rec


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
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for rec in _iter_candidate_traces(run_dir):
        if phase and rec.get("phase") != phase:
            continue
        if source and rec.get("source") != source:
            continue
        if require_runtime_valid and not rec.get("runtime_valid"):
            continue
        if require_finite_delta_f and not _is_finite(rec.get("delta_f")):
            continue
        cid = str(rec.get("candidate_id"))
        ann = annotations.get(cid)
        if require_annotation and ann is None:
            continue
        if cid in seen:
            continue
        seen.add(cid)
        row = {
            "run_id": rec.get("run_id"),
            "dataset": rec.get("dataset"),
            "participant_id": rec.get("participant_id"),
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
            "primary_edit": ann.get("primary_edit") if ann else None,
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
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True, help="annotations.jsonl path")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--phase", type=str, default="evolution")
    parser.add_argument("--source", type=str, default="normal")
    args = parser.parse_args()

    annotations = _load_annotations(Path(args.annotations))
    rows = build_rows(
        run_dir=Path(args.run_dir),
        annotations=annotations,
        phase=args.phase,
        source=args.source,
    )
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("Warning: no rows after filtering; writing header-only CSV.")
        fieldnames = [
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
            "primary_edit",
            "confidence",
            "added_motifs",
            "removed_motifs",
            "modified_motifs",
        ]
    else:
        fieldnames = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
