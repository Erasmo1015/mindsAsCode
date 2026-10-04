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
    RULE_UNUSED_HISTORY_ABSENT,
    history_runtime_used,
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


def _labels_now(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "reference_motif_state": list(row.get("reference_motif_state") or []),
        "candidate_motif_state": list(
            row.get("candidate_motif_state") or row.get("program_motif_state") or []
        ),
        "added_motifs": list(row.get("added_motifs") or []),
        "removed_motifs": list(row.get("removed_motifs") or []),
        "modified_motifs": list(row.get("modified_motifs") or []),
        "unchanged_motifs": list(row.get("unchanged_motifs") or []),
    }


def _drop_history_detail(details: Any) -> Any:
    if not isinstance(details, list):
        return details
    updated = []
    for item in details:
        if not isinstance(item, dict) or item.get("motif") != "history":
            updated.append(item)
            continue
        copied = dict(item)
        copied["presence"] = False
        updated.append(copied)
    return updated


def _correction_record(
    row: Mapping[str, Any],
    *,
    rule_id: str,
    original: Mapping[str, Any],
    corrected: Mapping[str, Any],
    evidence: str,
) -> Dict[str, Any]:
    return {
        "rule_id": rule_id,
        "dataset": row.get("dataset"),
        "job": row.get("source_job_id") or row.get("run_id"),
        "iteration": row.get("iteration"),
        "candidate": row.get("candidate_id") or row.get("program_id"),
        "resume_key": row.get("resume_key"),
        "original_labels": dict(original),
        "corrected_labels": dict(corrected),
        "evidence": evidence,
    }


def finalize_population_row(
    row: Dict[str, Any],
    parent_code: str,
    candidate_code: str,
) -> List[Dict[str, Any]]:
    """Apply seed-baseline and unused-history rules, then rederive transitions.

    Raw validated labels are preserved. A second call on the same row is a no-op.
    """
    if "raw_llm_annotation" not in row:
        row["raw_llm_annotation"] = _snapshot(row)
    raw = _snapshot(row["raw_llm_annotation"])
    reference = [str(m) for m in (raw.get("reference_motif_state") or [])]
    candidate = [
        str(m)
        for m in (raw.get("candidate_motif_state") or raw.get("program_motif_state") or [])
    ]
    modified_in = [str(m) for m in (raw.get("modified_motifs") or [])]
    pending: List[Tuple[str, str]] = []
    if is_verified_canonical_constant_seed(row, parent_code):
        if reference:
            pending.append(
                (
                    RULE_SEED_BASELINE_REF_ABSENT,
                    "Canonical return 0.5 seed; history in the signature is not History.",
                )
            )
        reference = []
    if parent_code.strip() and "history" in reference and not history_runtime_used(parent_code):
        reference = [motif for motif in reference if motif != "history"]
        pending.append(
            (
                RULE_UNUSED_HISTORY_ABSENT,
                "Reference history argument is unused.",
            )
        )
    dropped_candidate_history = False
    if candidate_code.strip() and "history" in candidate and not history_runtime_used(candidate_code):
        candidate = [motif for motif in candidate if motif != "history"]
        dropped_candidate_history = True
        pending.append(
            (
                RULE_UNUSED_HISTORY_ABSENT,
                "Candidate history argument is unused.",
            )
        )
    added, removed, modified = derive_directional_motifs(reference, candidate, modified_in)
    unchanged = [
        motif
        for motif in BEHAVIORAL_MOTIFS
        if motif in reference and motif in candidate and motif not in modified
    ]
    corrected = {
        "reference_motif_state": reference,
        "candidate_motif_state": candidate,
        "added_motifs": list(added),
        "removed_motifs": list(removed),
        "modified_motifs": list(modified),
        "unchanged_motifs": unchanged,
    }
    current = _labels_now(row)
    if all(current.get(key) == corrected[key] for key in corrected):
        return []
    original = _snapshot(row["raw_llm_annotation"])
    row["reference_motif_state"] = reference
    row["candidate_motif_state"] = candidate
    row["program_motif_state"] = list(candidate)
    row["added_motifs"] = list(added)
    row["removed_motifs"] = list(removed)
    row["modified_motifs"] = list(modified)
    row["unchanged_motifs"] = unchanged
    if dropped_candidate_history:
        row["motif_details"] = _drop_history_detail(row.get("motif_details"))
    row["transition_by_construct"] = {
        motif: transition_type_for_construct(
            motif,
            reference_has=motif in reference,
            candidate_has=motif in candidate,
            modified=motif in modified,
        )
        for motif in BEHAVIORAL_MOTIFS
    }
    records = [
        _correction_record(
            row,
            rule_id=rule_id,
            original=original,
            corrected=corrected,
            evidence=evidence,
        )
        for rule_id, evidence in pending
    ]
    logged = list(row.get("semantic_corrections") or [])
    logged.extend(records)
    row["semantic_corrections"] = logged
    return records


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


def _read_repo_text(repo: Path, cache: Dict[str, str], rel: str) -> str:
    if rel not in cache:
        path = repo / rel if rel else None
        cache[rel] = path.read_text(encoding="utf-8") if path is not None and path.is_file() else ""
    return cache[rel]


def finalize_annotation_jsonl(
    annotation_path: Path,
    repo: Path,
    *,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize one annotation JSONL. A second call does not rewrite or re-log.

    Raw responses on disk are not touched. Raw labels stay on each row.
    """
    annotation_path = Path(annotation_path)
    repo = Path(repo)
    original_lines = [
        line for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = [json.loads(line) for line in original_lines]
    corrections: List[Dict[str, Any]] = []
    cache: Dict[str, str] = {}
    written: List[str] = []
    changed = False
    for line, row in zip(original_lines, rows):
        parent_code = _read_repo_text(repo, cache, str(row.get("parent_code_path") or ""))
        candidate_code = _read_repo_text(repo, cache, str(row.get("code_path") or ""))
        records = finalize_population_row(row, parent_code, candidate_code)
        if not records:
            written.append(line)
            continue
        changed = True
        corrections.extend(records)
        written.append(json.dumps(row, ensure_ascii=False))
    destination = Path(log_path) if log_path is not None else annotation_path.with_name(
        "semantic_corrections.jsonl"
    )
    if changed:
        annotation_path.write_text("".join(line + "\n" for line in written), encoding="utf-8")
        with destination.open("a", encoding="utf-8") as handle:
            for item in corrections:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
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
        "rewritten": changed,
    }


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--log_path", type=Path, default=None)
    args = parser.parse_args()
    summary = finalize_annotation_jsonl(args.annotations, args.repo, log_path=args.log_path)
    print(json.dumps({k: v for k, v in summary.items() if k != "schema_errors"}))
    if not summary["schema_ok"]:
        print(summary["schema_errors"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
