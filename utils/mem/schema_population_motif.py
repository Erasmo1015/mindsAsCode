#!/usr/bin/env python3
"""Schema helpers for population-program motif *presence* annotation (pilot).

Single-program state inventory only — no reference→candidate transitions,
no fabricated ΔF, no added/removed/modified labels.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from utils.mem.schema_v2 import BEHAVIORAL_MOTIF_DEFINITIONS, BEHAVIORAL_MOTIFS_V2

BEHAVIORAL_MOTIFS = BEHAVIORAL_MOTIFS_V2
_BEHAVIORAL_SET = frozenset(BEHAVIORAL_MOTIFS)

SCHEMA_VERSION = 3
PROMPT_VERSION = "population_state_v3_1"
ANNOTATION_KIND = "population_program_motif_state"

REQUIRED_LLM_FIELDS = (
    "program_id",
    "program_motif_state",
    "evidence",
    "confidence",
)


def resume_key(dataset: str, run_id: str, program_id: str) -> str:
    return f"{dataset}|{run_id}|{program_id}"


def _clean_motifs(cid: str, motifs: Any) -> Tuple[Optional[List[str]], str]:
    if not isinstance(motifs, list):
        return None, f"{cid}: program_motif_state must be a list"
    cleaned: List[str] = []
    seen: Set[str] = set()
    for m in motifs:
        if m not in _BEHAVIORAL_SET:
            return None, f"{cid}: invalid motif {m!r} in program_motif_state"
        if m in seen:
            continue
        seen.add(str(m))
        cleaned.append(str(m))
    return cleaned, ""


def guided_json_schema_for_programs(expected_ids: Sequence[str]) -> Dict[str, Any]:
    motif_items = {"type": "string", "enum": list(BEHAVIORAL_MOTIFS)}
    item = {
        "type": "object",
        "properties": {
            "program_id": {"type": "string", "enum": list(expected_ids)},
            "program_motif_state": {"type": "array", "items": motif_items},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": list(REQUIRED_LLM_FIELDS),
        "additionalProperties": False,
    }
    return {"type": "array", "items": item}


def validate_program_motif_response(
    payload: Any,
    *,
    expected_ids: Sequence[str],
    prompt_version: str = PROMPT_VERSION,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    if not isinstance(payload, list):
        return False, "payload must be a JSON array", []
    expected = [str(x) for x in expected_ids]
    seen: Set[str] = set()
    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            return False, f"item {i}: not an object", []
        for field in REQUIRED_LLM_FIELDS:
            if field not in row:
                return False, f"item {i}: missing field {field}", []
        pid = str(row["program_id"])
        if pid not in expected:
            return False, f"unexpected program_id {pid!r}", []
        if pid in seen:
            return False, f"duplicate program_id {pid!r}", []
        seen.add(pid)
        motifs, err = _clean_motifs(pid, row["program_motif_state"])
        if motifs is None:
            return False, err, []
        conf = row["confidence"]
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            return False, f"{pid}: confidence must be a number", []
        if not (0.0 <= conf_f <= 1.0):
            return False, f"{pid}: confidence out of range", []
        evidence = row["evidence"]
        if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
            return False, f"{pid}: evidence must be a list of strings", []
        # Store presence only; mirror into schema-v3-compatible state fields
        # with empty modified (no transition claimed).
        rows.append(
            {
                "program_id": pid,
                "program_motif_state": motifs,
                "reference_motif_state": list(motifs),
                "candidate_motif_state": list(motifs),
                "modified_motifs": [],
                "evidence": list(evidence),
                "confidence": conf_f,
                "schema_version": SCHEMA_VERSION,
                "prompt_version": prompt_version,
                "annotation_kind": ANNOTATION_KIND,
            }
        )
    missing = [x for x in expected if x not in seen]
    if missing:
        return False, f"missing program_id(s): {missing}", []
    return True, "", rows


def motif_definitions_block() -> str:
    lines = []
    for m in BEHAVIORAL_MOTIFS:
        lines.append(f"- {m}: {BEHAVIORAL_MOTIF_DEFINITIONS[m]}")
    return "\n".join(lines)
