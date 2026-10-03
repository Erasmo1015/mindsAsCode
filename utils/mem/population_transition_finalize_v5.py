#!/usr/bin/env python3
"""Population transition-v5 finalization for verified constant seed baselines.

Uses the participant Schema-v5 seed rule (``SEED_BASELINE_REF_ABSENT`` and
``is_seed_baseline_constant_program``). It does not define a second seed test.
Only a ``seed_baseline`` reference whose file is ``prompts/seed_program.py`` and
whose source hash matches that constant seed is changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from utils.mem.participant_semantic_postprocess_v5 import (
    RULE_SEED_BASELINE_REF_ABSENT,
    is_seed_baseline_constant_program,
)
from utils.mem.schema_participant_transition_v5 import transition_type_for_construct
from utils.mem.schema_population_motif_v5 import (
    BEHAVIORAL_MOTIFS,
    derive_directional_motifs,
    validate_program_motif_response,
)

SEED_REFERENCE_KIND = "seed_baseline"
SEED_PARENT_IDS = frozenset({"global_baseline", "baseline"})
RAW_LLM_KEYS = (
    "reference_motif_state",
    "candidate_motif_state",
    "program_motif_state",
    "added_motifs",
    "removed_motifs",
    "modified_motifs",
)


def parent_source_sha256(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()


def is_verified_canonical_constant_seed(
    row: Mapping[str, Any],
    parent_code: str,
) -> bool:
    """True only for a provenance-checked canonical ``return 0.5`` seed parent."""
    kind = str(row.get("reference_kind") or row.get("reference_type") or "")
    parent = str(row.get("parent") or row.get("parent_id") or "")
    path = str(row.get("parent_code_path") or "").replace("\\", "/")
    recorded = str(row.get("parent_code_sha256") or "")
    if kind != SEED_REFERENCE_KIND or parent not in SEED_PARENT_IDS:
        return False
    if not path.endswith("prompts/seed_program.py"):
        return False
    if not recorded or parent_source_sha256(parent_code) != recorded:
        return False
    return is_seed_baseline_constant_program(parent_code)


def _snapshot(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: list(row.get(key) or []) for key in RAW_LLM_KEYS}


def finalize_constant_seed_reference(
    row: Dict[str, Any],
    parent_code: str,
) -> Optional[Dict[str, Any]]:
    """Clear a verified constant-seed reference and rederive transitions.

    Returns a correction record, or None when the row is out of scope or already
    consistent with an empty reference. The raw LLM labels are kept on the row.
    """
    if not is_verified_canonical_constant_seed(row, parent_code):
        return None
    if "raw_llm_annotation" not in row:
        row["raw_llm_annotation"] = _snapshot(row)
    before = _snapshot(row["raw_llm_annotation"])
    reference: List[str] = []
    candidate = [str(m) for m in (row.get("candidate_motif_state") or [])]
    modified_in = [str(m) for m in (before.get("modified_motifs") or row.get("modified_motifs") or [])]
    added, removed, modified = derive_directional_motifs(reference, candidate, modified_in)
    unchanged = [
        motif
        for motif in BEHAVIORAL_MOTIFS
        if motif in reference and motif in candidate and motif not in modified
    ]
    corrected = {
        "reference_motif_state": reference,
        "added_motifs": list(added),
        "removed_motifs": list(removed),
        "modified_motifs": list(modified),
        "unchanged_motifs": unchanged,
    }
    already = (
        list(row.get("reference_motif_state") or []) == reference
        and list(row.get("added_motifs") or []) == list(added)
        and list(row.get("removed_motifs") or []) == list(removed)
        and list(row.get("modified_motifs") or []) == list(modified)
        and list(row.get("unchanged_motifs") or []) == unchanged
    )
    if already:
        return None
    row["reference_motif_state"] = reference
    row["added_motifs"] = list(added)
    row["removed_motifs"] = list(removed)
    row["modified_motifs"] = list(modified)
    row["unchanged_motifs"] = unchanged
    row["transition_by_construct"] = {
        motif: transition_type_for_construct(
            motif,
            reference_has=False,
            candidate_has=motif in candidate,
            modified=False,
        )
        for motif in BEHAVIORAL_MOTIFS
    }
    correction = {
        "rule_id": RULE_SEED_BASELINE_REF_ABSENT,
        "resume_key": row.get("resume_key"),
        "dataset": row.get("dataset"),
        "parent_code_sha256": row.get("parent_code_sha256"),
        "original_labels": before,
        "corrected_labels": corrected,
        "note": "Canonical constant seed baseline; all five constructs absent on the reference.",
    }
    logged = list(row.get("semantic_corrections") or [])
    logged.append(correction)
    row["semantic_corrections"] = logged
    return correction


def annotation_row_valid(row: Mapping[str, Any]) -> Tuple[bool, str]:
    """Re-validate one stored population row against schema v5."""
    candidate_id = str(row.get("candidate_id") or "")
    payload = [
        {
            "candidate_id": candidate_id,
            "reference_motif_state": list(row.get("reference_motif_state") or []),
            "modified_motifs": list(row.get("modified_motifs") or []),
            "motif_details": row.get("motif_details"),
            "confidence": row.get("confidence"),
        }
    ]
    ok, err, validated = validate_program_motif_response(payload, expected_ids=[candidate_id])
    if not ok:
        return False, err
    derived = validated[0]
    for key in ("added_motifs", "removed_motifs", "modified_motifs", "candidate_motif_state"):
        if list(derived.get(key) or []) != list(row.get(key) or []):
            return False, f"{candidate_id}: {key} does not match schema rederivation"
    return True, ""


def finalize_annotation_jsonl(
    annotation_path: Path,
    repo: Path,
    *,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize one annotation JSONL. Experiment outputs are not read for writing."""
    annotation_path = Path(annotation_path)
    repo = Path(repo)
    original_lines = [
        line for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = [json.loads(line) for line in original_lines]
    corrections: List[Dict[str, Any]] = []
    cache: Dict[str, str] = {}
    written: List[str] = []
    for line, row in zip(original_lines, rows):
        rel = str(row.get("parent_code_path") or "")
        if rel not in cache:
            cache[rel] = (repo / rel).read_text(encoding="utf-8") if rel else ""
        correction = finalize_constant_seed_reference(row, cache[rel])
        if correction is None:
            written.append(line)
            continue
        corrections.append(correction)
        written.append(json.dumps(row, ensure_ascii=False))
    annotation_path.write_text("".join(line + "\n" for line in written), encoding="utf-8")
    destination = Path(log_path) if log_path is not None else annotation_path.with_name(
        "semantic_corrections.jsonl"
    )
    destination.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in corrections),
        encoding="utf-8",
    )
    fresh = [
        row
        for row in rows
        if str(row.get("reference_kind") or "") == SEED_REFERENCE_KIND
        and str(row.get("source") or "") == "fresh"
    ]
    invalid = [annotation_row_valid(row)[1] for row in rows if not annotation_row_valid(row)[0]]
    return {
        "n_rows": len(rows),
        "n_corrected": len(corrections),
        "n_fresh_seed": len(fresh),
        "n_fresh_seed_empty_reference": sum(
            not list(row.get("reference_motif_state") or []) for row in fresh
        ),
        "schema_ok": not invalid,
        "schema_errors": invalid[:5],
        "log_path": str(destination),
    }
