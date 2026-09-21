#!/usr/bin/env python3
"""Build MEM analysis CSV from mem_trace.jsonl + annotations_v{2,3,5}.jsonl.

Schema v2: directional behavioral columns only (binary, never averaged).
  {motif}_added / {motif}_removed / {motif}_modified
  structural_{operation}
  No eligibility / state columns (absent, not zero-filled).

Schema v3 (default): same directional columns plus:
  reference_has_*, candidate_has_*, eligible_*_*, retained_unmodified_*
  State lists stored as JSON strings. Motifs = v2 six (incl. risk).

Schema v5: five ICLR constructs (history, value, probability_used, feedback,
  learning); global resume keys; transition_* string columns; eligibility as v3.
  Builds from **corrected** annotation fields (post semantic postprocess).
  Propagates semantic_resolution_status / nmc_adjudication_status /
  exclude_from_construct_effect_fitting so unresolved NMC rows stay in the CSV
  for audit/coverage but are excluded from construct-transition effect fitting.

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
    SCHEMA_VERSION as SCHEMA_VERSION_V2,
    STRUCTURAL_OPERATIONS_V2,
    all_directional_behavioral_columns,
    all_structural_columns,
    annotation_resume_key,
    directional_flags_from_annotation,
    is_schema_v2_row,
)
from utils.mem.schema_v3 import (  # noqa: E402
    SCHEMA_VERSION as SCHEMA_VERSION_V3,
    all_eligibility_columns,
    all_state_columns,
    assert_transition_identities,
    is_schema_v3_row,
    state_and_eligibility_flags,
)
from utils.mem.schema_participant_transition_v5 import (  # noqa: E402
    SCHEMA_VERSION as SCHEMA_VERSION_V5,
    all_directional_behavioral_columns as all_directional_behavioral_columns_v5,
    all_eligibility_columns as all_eligibility_columns_v5,
    all_state_columns as all_state_columns_v5,
    all_structural_columns as all_structural_columns_v5,
    all_transition_type_columns as all_transition_type_columns_v5,
    annotation_resume_key as annotation_resume_key_v5,
    annotation_resume_key_legacy_no_phase,
    assert_transition_identities as assert_transition_identities_v5,
    global_candidate_id,
    is_schema_v5_row,
    state_and_eligibility_flags as state_and_eligibility_flags_v5,
    verify_delta_f_consistency,
)
from utils.mem.trace import iter_jsonl_records, record_contains_test_metrics  # noqa: E402

# Backward-compat alias for older imports/tests.
SCHEMA_VERSION = SCHEMA_VERSION_V2


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _load_annotations(
    path: Path,
    *,
    schema_version: int,
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    """Index annotations by the schema-appropriate resume key."""
    by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"annotations file not found: {path}")
    n_skipped = 0
    for rec in iter_jsonl_records([path]):
        if record_contains_test_metrics(rec):
            raise ValueError(
                "Test metric keys must not appear in annotations "
                f"(candidate_id={rec.get('candidate_id')!r})"
            )
        if schema_version == 5:
            if not is_schema_v5_row(rec):
                n_skipped += 1
                continue
        elif schema_version >= 3:
            if not is_schema_v3_row(rec):
                n_skipped += 1
                continue
        else:
            if not is_schema_v2_row(rec):
                n_skipped += 1
                continue
        cid = rec.get("candidate_id")
        if not isinstance(cid, str):
            continue
        if "participant_id" not in rec:
            continue
        ref_id = rec.get("reference_id") or rec.get("reference_parent_id")
        ref_type = rec.get("reference_type") or rec.get("reference_kind") or ""
        # Legacy v2 annotations lacked reference_type; they used pool-best pairing.
        if not ref_type and ref_id and schema_version < 5:
            ref_type = "pool_best_proxy"
        if schema_version == 5:
            key = annotation_resume_key_v5(
                rec.get("dataset"),
                rec.get("run_id"),
                rec.get("participant_id"),
                rec.get("iteration"),
                cid,
                reference_id=ref_id,
                reference_type=ref_type,
                phase=rec.get("phase"),
            )
            # Also index legacy pilot keys (no phase) for evolution rows.
            legacy = annotation_resume_key_legacy_no_phase(
                rec.get("dataset"),
                rec.get("run_id"),
                rec.get("participant_id"),
                rec.get("iteration"),
                cid,
                reference_id=ref_id,
                reference_type=ref_type,
            )
            by_key[legacy] = rec
        else:
            key = annotation_resume_key(
                rec.get("participant_id"),
                cid,
                reference_id=ref_id,
                reference_type=ref_type,
            )
        by_key[key] = rec
    if n_skipped:
        print(
            f"[build] Skipped {n_skipped} non-v{schema_version} annotation row(s).",
            flush=True,
        )
    return by_key


def _load_annotations_v2(path: Path) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    return _load_annotations(path, schema_version=2)


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


def _base_row_from_trace(
    rec: Dict[str, Any],
    *,
    pid: Any,
    cid: str,
    ref_id: Any,
    ref_type: Any,
    ann: Optional[Dict[str, Any]],
    schema_version: int,
) -> Dict[str, Any]:
    return {
        "schema_version": schema_version,
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
        "reference_id": ref_id,
        "reference_parent_score": rec.get("reference_parent_score"),
        "reference_score": rec.get("reference_score") or rec.get("reference_parent_score"),
        "reference_kind": rec.get("reference_kind"),
        "reference_type": ref_type,
        "reference_is_exact": rec.get("reference_is_exact"),
        "reference_is_proxy": rec.get("reference_is_proxy"),
        "delta_f": rec.get("delta_f"),
        "delta_f_vs_baseline": rec.get("delta_f_vs_baseline"),
        "delta_f_vs_population_program": rec.get("delta_f_vs_population_program"),
        "delta_f_vs_best_prompted_parent": rec.get("delta_f_vs_best_prompted_parent"),
        "delta_f_vs_pool_best": rec.get("delta_f_vs_pool_best"),
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


def build_rows_v2(
    *,
    run_dir: Path,
    annotations: Dict[Tuple[Any, ...], Dict[str, Any]],
    phase: str = "evolution",
    source: str = "normal",
    require_runtime_valid: bool = True,
    require_finite_delta_f: bool = True,
    require_annotation: bool = True,
    exclusions_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Build analysis rows; record every exclusion reason (no silent drops).

    Legacy path: no eligibility/state columns (absent, not zero-filled).
    """
    rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[Any, ...]] = set()
    excl: Counter = Counter()
    dir_cols = all_directional_behavioral_columns()
    struct_cols = all_structural_columns()

    for rec in _iter_candidate_traces(run_dir):
        pid = rec.get("participant_id")
        cid = str(rec.get("candidate_id"))
        ref_id = rec.get("reference_id") or rec.get("reference_parent_id")
        ref_type = rec.get("reference_type") or rec.get("reference_kind") or ""
        key = annotation_resume_key(
            pid, cid, reference_id=ref_id, reference_type=ref_type
        )
        base_excl = {
            "participant_id": pid,
            "candidate_id": cid,
            "iteration": rec.get("iteration"),
            "source": rec.get("source"),
            "phase": rec.get("phase"),
            "reference_id": ref_id,
            "reference_type": ref_type,
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
            _exclude("excl_duplicate_participant_candidate_reference")
            continue
        seen.add(key)

        flags = directional_flags_from_annotation(ann) if ann else {
            c: 0 for c in dir_cols + struct_cols
        }
        row = _base_row_from_trace(
            rec,
            pid=pid,
            cid=cid,
            ref_id=ref_id,
            ref_type=ref_type,
            ann=ann,
            schema_version=SCHEMA_VERSION_V2,
        )
        for c in dir_cols + struct_cols:
            row[c] = int(flags.get(c, 0))
        rows.append(row)
    return rows, excl


def build_rows_v3(
    *,
    run_dir: Path,
    annotations: Dict[Tuple[Any, ...], Dict[str, Any]],
    phase: str = "evolution",
    source: str = "normal",
    require_runtime_valid: bool = True,
    require_finite_delta_f: bool = True,
    require_annotation: bool = True,
    exclusions_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Build schema-v3 rows with state + eligibility columns.

    Missing annotation state is never coerced to empty sets for eligibility.
    Rows without a v3 annotation are excluded when require_annotation=True.
    """
    rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[Any, ...]] = set()
    excl: Counter = Counter()
    dir_cols = all_directional_behavioral_columns()
    struct_cols = all_structural_columns()
    state_cols = all_state_columns()
    elig_cols = all_eligibility_columns()

    for rec in _iter_candidate_traces(run_dir):
        pid = rec.get("participant_id")
        cid = str(rec.get("candidate_id"))
        ref_id = rec.get("reference_id") or rec.get("reference_parent_id")
        ref_type = rec.get("reference_type") or rec.get("reference_kind") or ""
        key = annotation_resume_key(
            pid, cid, reference_id=ref_id, reference_type=ref_type
        )
        base_excl = {
            "participant_id": pid,
            "candidate_id": cid,
            "iteration": rec.get("iteration"),
            "source": rec.get("source"),
            "phase": rec.get("phase"),
            "reference_id": ref_id,
            "reference_type": ref_type,
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
            _exclude("excl_missing_annotation_v3")
            continue
        if key in seen:
            _exclude("excl_duplicate_participant_candidate_reference")
            continue
        seen.add(key)

        row = _base_row_from_trace(
            rec,
            pid=pid,
            cid=cid,
            ref_id=ref_id,
            ref_type=ref_type,
            ann=ann,
            schema_version=SCHEMA_VERSION_V3,
        )
        if ann is not None:
            if "reference_motif_state" not in ann or "candidate_motif_state" not in ann:
                raise ValueError(
                    f"schema_v3 annotation missing state fields for {cid!r}; "
                    "do not coerce missing state to empty sets"
                )
            assert_transition_identities(ann)
            flags = state_and_eligibility_flags(ann)
            row["prompt_version"] = ann.get("prompt_version")
            row["reference_motif_state"] = json.dumps(
                ann.get("reference_motif_state", []), ensure_ascii=False
            )
            row["candidate_motif_state"] = json.dumps(
                ann.get("candidate_motif_state", []), ensure_ascii=False
            )
            for c in state_cols + dir_cols + elig_cols + struct_cols:
                row[c] = int(flags.get(c, 0))
        else:
            row["prompt_version"] = None
            row["reference_motif_state"] = None
            row["candidate_motif_state"] = None
            for c in state_cols + dir_cols + elig_cols + struct_cols:
                row[c] = None
        rows.append(row)
    return rows, excl


def build_rows_v5(
    *,
    run_dir: Path,
    annotations: Dict[Tuple[Any, ...], Dict[str, Any]],
    phase: str = "evolution",
    source: str = "normal",
    require_runtime_valid: bool = True,
    require_finite_delta_f: bool = True,
    require_annotation: bool = True,
    exclusions_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Build schema-v5 rows with five constructs + state + eligibility + transition_*."""
    rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[Any, ...]] = set()
    excl: Counter = Counter()
    dir_cols = all_directional_behavioral_columns_v5()
    struct_cols = all_structural_columns_v5()
    state_cols = all_state_columns_v5()
    elig_cols = all_eligibility_columns_v5()
    transition_cols = all_transition_type_columns_v5()

    for rec in _iter_candidate_traces(run_dir):
        pid = rec.get("participant_id")
        cid = str(rec.get("candidate_id"))
        ref_id = rec.get("reference_id") or rec.get("reference_parent_id")
        ref_type = rec.get("reference_type") or rec.get("reference_kind") or ""
        key = annotation_resume_key_v5(
            rec.get("dataset"),
            rec.get("run_id"),
            pid,
            rec.get("iteration"),
            cid,
            reference_id=ref_id,
            reference_type=ref_type,
            phase=rec.get("phase"),
        )
        base_excl = {
            "dataset": rec.get("dataset"),
            "run_id": rec.get("run_id"),
            "participant_id": pid,
            "candidate_id": cid,
            "iteration": rec.get("iteration"),
            "source": rec.get("source"),
            "phase": rec.get("phase"),
            "reference_id": ref_id,
            "reference_type": ref_type,
            "global_candidate_id": global_candidate_id(
                rec.get("dataset"),
                rec.get("run_id"),
                pid,
                rec.get("iteration"),
                cid,
                phase=rec.get("phase"),
            ),
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
        ref_score = rec.get("reference_score")
        if not _is_finite(ref_score):
            ref_score = rec.get("reference_parent_score")
        if not _is_finite(ref_score):
            _exclude("excl_reference_score_missing")
            continue
        cand_score = rec.get("selection_score")
        if not _is_finite(cand_score):
            _exclude("excl_candidate_fitness_missing")
            continue
        ok_df, err_df = verify_delta_f_consistency(
            delta_f=rec.get("delta_f"),
            candidate_score=cand_score,
            reference_score=ref_score,
        )
        if not ok_df:
            _exclude(f"excl_{err_df.split(':')[0]}")
            continue
        ann = annotations.get(key)
        if ann is None:
            legacy = annotation_resume_key_legacy_no_phase(
                rec.get("dataset"),
                rec.get("run_id"),
                pid,
                rec.get("iteration"),
                cid,
                reference_id=ref_id,
                reference_type=ref_type,
            )
            ann = annotations.get(legacy)
            if ann is not None:
                key = legacy
        if require_annotation and ann is None:
            _exclude("excl_missing_annotation_v5")
            continue
        if key in seen:
            _exclude("excl_duplicate_global_candidate_reference")
            continue
        seen.add(key)

        row = _base_row_from_trace(
            rec,
            pid=pid,
            cid=cid,
            ref_id=ref_id,
            ref_type=ref_type,
            ann=ann,
            schema_version=SCHEMA_VERSION_V5,
        )
        row["global_candidate_id"] = base_excl["global_candidate_id"]
        if ann is not None:
            if "reference_motif_state" not in ann or "candidate_motif_state" not in ann:
                raise ValueError(
                    f"schema_v5 annotation missing state fields for {cid!r}; "
                    "do not coerce missing state to empty sets"
                )
            assert_transition_identities_v5(ann)
            flags = state_and_eligibility_flags_v5(ann)
            row["prompt_version"] = ann.get("prompt_version")
            row["reference_resolution"] = ann.get("reference_resolution")
            row["reference_motif_state"] = json.dumps(
                ann.get("reference_motif_state", []), ensure_ascii=False
            )
            row["candidate_motif_state"] = json.dumps(
                ann.get("candidate_motif_state", []), ensure_ascii=False
            )
            row["transition_by_construct"] = json.dumps(
                ann.get("transition_by_construct")
                or {c.replace("transition_", ""): flags.get(c) for c in transition_cols},
                ensure_ascii=False,
            )
            row["semantic_postprocess_version"] = ann.get("semantic_postprocess_version")
            row["semantic_resolution_status"] = ann.get(
                "semantic_resolution_status", "resolved"
            )
            row["nmc_adjudication_status"] = ann.get(
                "nmc_adjudication_status", "not_applicable"
            )
            row["exclude_from_construct_effect_fitting"] = int(
                bool(ann.get("exclude_from_construct_effect_fitting"))
            )
            raw_llm = ann.get("raw_llm_annotation")
            row["raw_llm_annotation"] = (
                json.dumps(raw_llm, ensure_ascii=False)
                if isinstance(raw_llm, dict)
                else None
            )
            for c in state_cols + dir_cols + elig_cols + struct_cols:
                row[c] = int(flags.get(c, 0))
            for c in transition_cols:
                row[c] = flags.get(c)
        else:
            row["prompt_version"] = None
            row["reference_resolution"] = None
            row["reference_motif_state"] = None
            row["candidate_motif_state"] = None
            row["transition_by_construct"] = None
            row["semantic_postprocess_version"] = None
            row["semantic_resolution_status"] = None
            row["nmc_adjudication_status"] = None
            row["exclude_from_construct_effect_fitting"] = None
            row["raw_llm_annotation"] = None
            for c in state_cols + dir_cols + elig_cols + struct_cols:
                row[c] = None
            for c in transition_cols:
                row[c] = None
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
        a.setdefault("schema_version", SCHEMA_VERSION_V2)
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


def _fieldnames_for_schema(schema_version: int) -> List[str]:
    if schema_version == 5:
        dir_cols = all_directional_behavioral_columns_v5()
        struct_cols = all_structural_columns_v5()
        state_cols = all_state_columns_v5()
        elig_cols = all_eligibility_columns_v5()
        transition_cols = all_transition_type_columns_v5()
    else:
        dir_cols = all_directional_behavioral_columns()
        struct_cols = all_structural_columns()
        state_cols = all_state_columns()
        elig_cols = all_eligibility_columns()
        transition_cols = []
    base = [
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
        "reference_id",
        "reference_parent_score",
        "reference_score",
        "reference_kind",
        "reference_type",
        "reference_is_exact",
        "reference_is_proxy",
        "delta_f",
        "delta_f_vs_baseline",
        "delta_f_vs_population_program",
        "delta_f_vs_best_prompted_parent",
        "delta_f_vs_pool_best",
        "survived_elite_truncation",
        "evolution_selection_score",
        "no_meaningful_change",
        "confidence",
        "added_motifs",
        "removed_motifs",
        "modified_motifs",
        "structural_operations",
    ]
    if schema_version == 5:
        return [
            *base[:1],
            "prompt_version",
            "global_candidate_id",
            "reference_resolution",
            *base[1:],
            "reference_motif_state",
            "candidate_motif_state",
            "transition_by_construct",
            "semantic_postprocess_version",
            "semantic_resolution_status",
            "nmc_adjudication_status",
            "exclude_from_construct_effect_fitting",
            "raw_llm_annotation",
            *state_cols,
            *dir_cols,
            *elig_cols,
            *transition_cols,
            *struct_cols,
        ]
    if schema_version >= 3:
        return [
            *base[:1],
            "prompt_version",
            *base[1:],
            "reference_motif_state",
            "candidate_motif_state",
            *state_cols,
            *dir_cols,
            *elig_cols,
            *struct_cols,
        ]
    return [*base, *dir_cols, *struct_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="annotations_v2.jsonl, annotations_v3.jsonl, or annotations_v5.jsonl path",
    )
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument(
        "--schema_version",
        type=int,
        default=3,
        choices=[2, 3, 5],
        help="Expected annotation schema (default 3). v5 = participant_transition_v5.",
    )
    parser.add_argument("--phase", type=str, default="evolution")
    parser.add_argument(
        "--source",
        type=str,
        default="normal",
        help="Candidate source filter (default: normal). Use empty string for all.",
    )
    args = parser.parse_args()

    schema_version = int(args.schema_version)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    excl_path = out.with_name(out.stem + "_exclusions.jsonl")
    summary_path = out.with_name(out.stem + "_build_summary.json")
    if excl_path.exists():
        excl_path.unlink()

    annotations = _load_annotations(Path(args.annotations), schema_version=schema_version)
    source = args.source if args.source != "" else ""
    if schema_version == 5:
        rows, excl = build_rows_v5(
            run_dir=Path(args.run_dir),
            annotations=annotations,
            phase=args.phase,
            source=source,
            exclusions_path=excl_path,
        )
    elif schema_version >= 3:
        rows, excl = build_rows_v3(
            run_dir=Path(args.run_dir),
            annotations=annotations,
            phase=args.phase,
            source=source,
            exclusions_path=excl_path,
        )
    else:
        rows, excl = build_rows_v2(
            run_dir=Path(args.run_dir),
            annotations=annotations,
            phase=args.phase,
            source=source,
            exclusions_path=excl_path,
        )

    fieldnames = _fieldnames_for_schema(schema_version)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": schema_version,
        "n_rows": len(rows),
        f"n_annotations_v{schema_version}_indexed": len(annotations),
        "exclusions_by_reason": dict(sorted(excl.items())),
        "n_exclusions": int(sum(excl.values())),
        "descriptive_counts": _descriptive_counts(rows),
        "output_csv": str(out),
        "exclusions_path": str(excl_path),
        "has_eligibility_columns": schema_version >= 3,
        "note_legacy_x0": (
            None
            if schema_version >= 3
            else "Legacy v2 X=0 is not eligibility; eligibility columns are absent."
        ),
        "n_unresolved_nmc_adjudication": (
            sum(
                1
                for r in rows
                if str(r.get("semantic_resolution_status") or "")
                == "nmc_needs_adjudication"
                or str(r.get("nmc_adjudication_status") or "") == "unresolved"
                or int(r.get("exclude_from_construct_effect_fitting") or 0) == 1
            )
            if schema_version == 5
            else 0
        ),
        "semantic_fields_note": (
            "CSV uses corrected annotation fields; raw_llm_annotation retained "
            "for audit. Unresolved NMC rows stay in coverage but "
            "exclude_from_construct_effect_fitting=1."
            if schema_version == 5
            else None
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {out}")
    print(f"Wrote exclusions to {excl_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
