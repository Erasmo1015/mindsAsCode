"""MEM annotation schema v2: directional behavioral motifs + structural ops."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 2

# Legacy flat taxonomy (schema v1). Kept for reading old artifacts only.
MOTIF_TAXONOMY_V1 = (
    "value_or_expected_value",
    "probability_or_risk",
    "history_or_memory",
    "feedback_or_reward",
    "learning_or_adaptation",
    "gating_or_conditional",
    "aggregation_or_combination",
    "nonlinear_transformation",
    "parameter_or_threshold_change",
    "simplification_or_removal",
    "other",
    "no_meaningful_change",
)

BEHAVIORAL_MOTIFS_V2 = (
    "history",
    "value",
    "risk",
    "feedback",
    "learning",
    "other_behavioral",
)

STRUCTURAL_OPERATIONS_V2 = (
    "parameter_change",
    "control_flow_change",
    "aggregation_change",
    "nonlinear_change",
    "simplification",
    "other_structural",
)

_BEHAVIORAL_SET = frozenset(BEHAVIORAL_MOTIFS_V2)
_STRUCTURAL_SET = frozenset(STRUCTURAL_OPERATIONS_V2)

BEHAVIORAL_MOTIF_DEFINITIONS = {
    "history": (
        "changes how past trials, actions, choices, or outcomes influence "
        "the current prediction."
    ),
    "value": (
        "changes how option values or expected values are calculated or compared."
    ),
    "risk": (
        "changes probability processing, risk/loss/sign sensitivity, or "
        "sensitivity to uncertain outcomes."
    ),
    "feedback": (
        "changes how observed rewards, outcomes, or external feedback affect "
        "subsequent predictions."
    ),
    "learning": (
        "changes a mechanism that dynamically updates internal estimates, "
        "parameters, or state from experience."
    ),
    "other_behavioral": "a meaningful behavioral mechanism not covered above.",
}

DIRECTIONAL_SUFFIXES = ("added", "removed", "modified")


def annotation_resume_key(participant_id: Any, candidate_id: str) -> Tuple[Any, str]:
    """Stable resume identity for schema v2."""
    return (participant_id, str(candidate_id))


def is_schema_v2_row(rec: Dict[str, Any]) -> bool:
    try:
        return int(rec.get("schema_version", -1)) == SCHEMA_VERSION
    except (TypeError, ValueError):
        return False


def directional_behavioral_column(motif: str, direction: str) -> str:
    if motif not in _BEHAVIORAL_SET:
        raise ValueError(f"unknown behavioral motif: {motif!r}")
    if direction not in DIRECTIONAL_SUFFIXES:
        raise ValueError(f"unknown direction: {direction!r}")
    return f"{motif}_{direction}"


def structural_column(op: str) -> str:
    if op not in _STRUCTURAL_SET:
        raise ValueError(f"unknown structural operation: {op!r}")
    return f"structural_{op}"


def all_directional_behavioral_columns() -> List[str]:
    return [
        directional_behavioral_column(m, d)
        for m in BEHAVIORAL_MOTIFS_V2
        for d in DIRECTIONAL_SUFFIXES
    ]


def all_structural_columns() -> List[str]:
    return [structural_column(op) for op in STRUCTURAL_OPERATIONS_V2]


def directional_flags_from_annotation(ann: Dict[str, Any]) -> Dict[str, int]:
    """Binary columns: motif_added / motif_removed / motif_modified / structural_*."""
    out: Dict[str, int] = {c: 0 for c in all_directional_behavioral_columns()}
    out.update({c: 0 for c in all_structural_columns()})
    if not is_schema_v2_row(ann) and "schema_version" not in ann:
        # Allow cleaned validator rows that omit schema_version until enrichment.
        pass
    if bool(ann.get("no_meaningful_change")):
        return out
    for motif in ann.get("added_motifs") or []:
        if motif in _BEHAVIORAL_SET:
            out[directional_behavioral_column(motif, "added")] = 1
    for motif in ann.get("removed_motifs") or []:
        if motif in _BEHAVIORAL_SET:
            out[directional_behavioral_column(motif, "removed")] = 1
    for motif in ann.get("modified_motifs") or []:
        if motif in _BEHAVIORAL_SET:
            out[directional_behavioral_column(motif, "modified")] = 1
    for op in ann.get("structural_operations") or []:
        if op in _STRUCTURAL_SET:
            out[structural_column(op)] = 1
    return out


def guided_json_schema_for_batch(expected_ids: Sequence[str]) -> Dict[str, Any]:
    """JSON Schema for a list of per-candidate v2 annotations (vLLM guided_json)."""
    motif_items = {"type": "string", "enum": list(BEHAVIORAL_MOTIFS_V2)}
    struct_items = {"type": "string", "enum": list(STRUCTURAL_OPERATIONS_V2)}
    item = {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "enum": list(expected_ids)},
            "added_motifs": {"type": "array", "items": motif_items},
            "removed_motifs": {"type": "array", "items": motif_items},
            "modified_motifs": {"type": "array", "items": motif_items},
            "structural_operations": {"type": "array", "items": struct_items},
            "no_meaningful_change": {"type": "boolean"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": [
            "candidate_id",
            "added_motifs",
            "removed_motifs",
            "modified_motifs",
            "structural_operations",
            "no_meaningful_change",
            "evidence",
            "confidence",
        ],
        "additionalProperties": False,
    }
    return {"type": "array", "items": item}


def _validate_motif_list(
    cid: str,
    key: str,
    motifs: Any,
    *,
    allowed: frozenset,
) -> Tuple[Optional[List[str]], str]:
    if not isinstance(motifs, list):
        return None, f"{cid}: {key} must be a list"
    cleaned: List[str] = []
    seen: set[str] = set()
    for m in motifs:
        if m not in allowed:
            return None, f"{cid}: invalid motif {m!r} in {key}"
        if m in seen:
            continue
        seen.add(m)
        cleaned.append(str(m))
    return cleaned, ""


def validate_annotation_response_v2(
    payload: Any,
    *,
    expected_ids: Sequence[str],
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Validate schema-v2 annotator JSON.

    Never remaps invalid labels. Returns (ok, error_message, rows).
    """
    expected = list(expected_ids)
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        return False, "expected_ids contains duplicates", []

    if isinstance(payload, dict) and "annotations" in payload:
        rows = payload["annotations"]
    else:
        rows = payload
    if not isinstance(rows, list):
        return False, "response must be a JSON list (or object with 'annotations' list)", []

    seen: set[str] = set()
    cleaned: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            return False, f"annotation[{i}] is not an object", []
        cid = row.get("candidate_id")
        if not isinstance(cid, str) or not cid:
            return False, f"annotation[{i}] missing candidate_id", []
        if cid not in expected_set:
            return False, f"unknown candidate_id {cid!r}", []
        if cid in seen:
            return False, f"duplicate candidate_id {cid!r}", []
        seen.add(cid)

        if "primary_edit" in row:
            return False, f"{cid}: primary_edit is not allowed in schema v2", []

        added, err = _validate_motif_list(
            cid, "added_motifs", row.get("added_motifs"), allowed=_BEHAVIORAL_SET
        )
        if err:
            return False, err, []
        removed, err = _validate_motif_list(
            cid, "removed_motifs", row.get("removed_motifs"), allowed=_BEHAVIORAL_SET
        )
        if err:
            return False, err, []
        modified, err = _validate_motif_list(
            cid, "modified_motifs", row.get("modified_motifs"), allowed=_BEHAVIORAL_SET
        )
        if err:
            return False, err, []
        structural, err = _validate_motif_list(
            cid,
            "structural_operations",
            row.get("structural_operations"),
            allowed=_STRUCTURAL_SET,
        )
        if err:
            # normalize message for structural
            err = err.replace("invalid motif", "invalid structural_operation")
            return False, err, []

        nmc = row.get("no_meaningful_change")
        if not isinstance(nmc, bool):
            return False, f"{cid}: no_meaningful_change must be a boolean", []

        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
            return False, f"{cid}: evidence must be a list of strings", []

        conf = row.get("confidence")
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            return False, f"{cid}: confidence must be a float", []
        if not (0.0 <= conf_f <= 1.0):
            return False, f"{cid}: confidence must be in [0,1]", []

        assert added is not None and removed is not None and modified is not None
        assert structural is not None

        if nmc:
            if added or removed or modified or structural:
                return (
                    False,
                    f"{cid}: no_meaningful_change=true requires empty motif and "
                    "structural lists",
                    [],
                )
        else:
            if not (added or removed or modified or structural):
                return (
                    False,
                    f"{cid}: no_meaningful_change=false requires at least one "
                    "non-empty motif or structural list",
                    [],
                )

        cleaned.append(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": cid,
                "added_motifs": list(added),
                "removed_motifs": list(removed),
                "modified_motifs": list(modified),
                "structural_operations": list(structural),
                "no_meaningful_change": nmc,
                "evidence": list(evidence),
                "confidence": conf_f,
            }
        )

    missing = [cid for cid in expected if cid not in seen]
    if missing:
        return False, f"missing candidate_id(s): {missing}", cleaned
    return True, "", cleaned
