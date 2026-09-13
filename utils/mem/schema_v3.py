"""MEM annotation schema v3: state-aware motif presence + derived directions.

The LLM reports reference/candidate motif *presence* and which shared motifs were
*modified*. Added/removed are derived deterministically:

  added   = candidate_state \\ reference_state
  removed = reference_state \\ candidate_state
  modified ⊆ reference_state ∩ candidate_state

Legacy schema v2 (directional lists from the LLM) remains in schema_v2.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from utils.mem.schema_v2 import (
    BEHAVIORAL_MOTIF_DEFINITIONS,
    BEHAVIORAL_MOTIFS_V2,
    DIRECTIONAL_SUFFIXES,
    STRUCTURAL_OPERATIONS_V2,
    annotation_resume_key,
    directional_behavioral_column,
    structural_column,
    all_directional_behavioral_columns,
    all_structural_columns,
)

SCHEMA_VERSION = 3
PROMPT_VERSION = "state_v3_1"

BEHAVIORAL_MOTIFS_V3 = BEHAVIORAL_MOTIFS_V2
STRUCTURAL_OPERATIONS_V3 = STRUCTURAL_OPERATIONS_V2
_BEHAVIORAL_SET = frozenset(BEHAVIORAL_MOTIFS_V3)
_STRUCTURAL_SET = frozenset(STRUCTURAL_OPERATIONS_V3)

REQUIRED_LLM_FIELDS = (
    "candidate_id",
    "reference_motif_state",
    "candidate_motif_state",
    "modified_motifs",
    "structural_operations",
    "no_meaningful_change",
    "evidence",
    "confidence",
)


def is_schema_v3_row(rec: Dict[str, Any]) -> bool:
    try:
        return int(rec.get("schema_version", -1)) == SCHEMA_VERSION
    except (TypeError, ValueError):
        return False


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
    seen: Set[str] = set()
    for m in motifs:
        if m not in allowed:
            label = "structural_operation" if allowed is _STRUCTURAL_SET else "motif"
            return None, f"{cid}: invalid {label} {m!r} in {key}"
        if m in seen:
            continue
        seen.add(str(m))
        cleaned.append(str(m))
    return cleaned, ""


def derive_directional_motifs(
    reference_state: Sequence[str],
    candidate_state: Sequence[str],
    modified_motifs: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Derive added/removed; keep modified ordered and restricted to intersection."""
    r = set(reference_state)
    c = set(candidate_state)
    inter = r & c
    added = [m for m in candidate_state if m not in r]
    removed = [m for m in reference_state if m not in c]
    modified = [m for m in modified_motifs if m in inter]
    return added, removed, modified


def guided_json_schema_for_batch_v3(expected_ids: Sequence[str]) -> Dict[str, Any]:
    """JSON Schema for schema-v3 LLM responses (no added/removed fields)."""
    motif_items = {"type": "string", "enum": list(BEHAVIORAL_MOTIFS_V3)}
    struct_items = {"type": "string", "enum": list(STRUCTURAL_OPERATIONS_V3)}
    item = {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "enum": list(expected_ids)},
            "reference_motif_state": {"type": "array", "items": motif_items},
            "candidate_motif_state": {"type": "array", "items": motif_items},
            "modified_motifs": {"type": "array", "items": motif_items},
            "structural_operations": {"type": "array", "items": struct_items},
            "no_meaningful_change": {"type": "boolean"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": list(REQUIRED_LLM_FIELDS),
        "additionalProperties": False,
    }
    return {"type": "array", "items": item}


def validate_annotation_response_v3(
    payload: Any,
    *,
    expected_ids: Sequence[str],
    prompt_version: str = PROMPT_VERSION,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Validate schema-v3 JSON; derive added/removed; reject illegal modified."""
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

    seen: Set[str] = set()
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
            return False, f"{cid}: primary_edit is not allowed in schema v3", []

        for key in (
            "reference_motif_state",
            "candidate_motif_state",
            "modified_motifs",
            "structural_operations",
        ):
            if key not in row:
                return False, f"{cid}: missing required field {key}", []

        ref_state, err = _validate_motif_list(
            cid, "reference_motif_state", row["reference_motif_state"], allowed=_BEHAVIORAL_SET
        )
        if err:
            return False, err, []
        cand_state, err = _validate_motif_list(
            cid, "candidate_motif_state", row["candidate_motif_state"], allowed=_BEHAVIORAL_SET
        )
        if err:
            return False, err, []
        modified_raw, err = _validate_motif_list(
            cid, "modified_motifs", row["modified_motifs"], allowed=_BEHAVIORAL_SET
        )
        if err:
            return False, err, []
        structural, err = _validate_motif_list(
            cid,
            "structural_operations",
            row["structural_operations"],
            allowed=_STRUCTURAL_SET,
        )
        if err:
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

        assert ref_state is not None and cand_state is not None and modified_raw is not None
        assert structural is not None

        inter = set(ref_state) & set(cand_state)
        outside = [m for m in modified_raw if m not in inter]
        if outside:
            return (
                False,
                f"{cid}: modified_motifs {outside} not in reference∩candidate "
                f"(intersection={sorted(inter)})",
                [],
            )

        added, removed, modified = derive_directional_motifs(
            ref_state, cand_state, modified_raw
        )

        if nmc:
            if added or removed or modified or structural:
                return (
                    False,
                    f"{cid}: no_meaningful_change=true requires empty derived "
                    "added/removed/modified and structural lists",
                    [],
                )
        else:
            if not (added or removed or modified or structural):
                return (
                    False,
                    f"{cid}: no_meaningful_change=false requires at least one "
                    "non-empty derived direction or structural list",
                    [],
                )

        cleaned.append(
            {
                "schema_version": SCHEMA_VERSION,
                "prompt_version": prompt_version,
                "candidate_id": cid,
                "reference_motif_state": list(ref_state),
                "candidate_motif_state": list(cand_state),
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


def state_and_eligibility_flags(ann: Dict[str, Any]) -> Dict[str, int]:
    """Binary has_*, direction, eligible_*, retained_unmodified_* for a v3 row."""
    if not is_schema_v3_row(ann) and int(ann.get("schema_version", -1) or -1) != 3:
        raise ValueError("state_and_eligibility_flags requires schema_version=3")

    ref = set(ann.get("reference_motif_state") or [])
    cand = set(ann.get("candidate_motif_state") or [])
    added = set(ann.get("added_motifs") or [])
    removed = set(ann.get("removed_motifs") or [])
    modified = set(ann.get("modified_motifs") or [])
    structural = set(ann.get("structural_operations") or [])

    out: Dict[str, int] = {}
    for m in BEHAVIORAL_MOTIFS_V3:
        rh = 1 if m in ref else 0
        ch = 1 if m in cand else 0
        out[f"reference_has_{m}"] = rh
        out[f"candidate_has_{m}"] = ch
        a = 1 if m in added else 0
        rmv = 1 if m in removed else 0
        mod = 1 if m in modified else 0
        if bool(ann.get("no_meaningful_change")):
            a = rmv = mod = 0
        out[directional_behavioral_column(m, "added")] = a
        out[directional_behavioral_column(m, "removed")] = rmv
        out[directional_behavioral_column(m, "modified")] = mod
        out[f"eligible_{m}_added"] = 1 if rh == 0 else 0
        out[f"eligible_{m}_removed"] = 1 if rh == 1 else 0
        out[f"eligible_{m}_modified"] = 1 if (rh == 1 and ch == 1) else 0
        out[f"retained_unmodified_{m}"] = (
            1 if (rh == 1 and ch == 1 and mod == 0) else 0
        )

    for op in STRUCTURAL_OPERATIONS_V3:
        out[structural_column(op)] = (
            0 if bool(ann.get("no_meaningful_change")) else (1 if op in structural else 0)
        )
    return out


def assert_transition_identities(ann: Dict[str, Any]) -> None:
    """Raise AssertionError if stored directions disagree with states."""
    ref = list(ann.get("reference_motif_state") or [])
    cand = list(ann.get("candidate_motif_state") or [])
    added_e, removed_e, _mod = derive_directional_motifs(
        ref, cand, ann.get("modified_motifs") or []
    )
    if set(ann.get("added_motifs") or []) != set(added_e):
        raise AssertionError(
            f"added_motifs {ann.get('added_motifs')} != derived {added_e}"
        )
    if set(ann.get("removed_motifs") or []) != set(removed_e):
        raise AssertionError(
            f"removed_motifs {ann.get('removed_motifs')} != derived {removed_e}"
        )
    inter = set(ref) & set(cand)
    for m in ann.get("modified_motifs") or []:
        if m not in inter:
            raise AssertionError(f"modified {m!r} not in intersection {sorted(inter)}")


def all_state_columns() -> List[str]:
    cols: List[str] = []
    for m in BEHAVIORAL_MOTIFS_V3:
        cols.append(f"reference_has_{m}")
        cols.append(f"candidate_has_{m}")
    return cols


def all_eligibility_columns() -> List[str]:
    cols: List[str] = []
    for m in BEHAVIORAL_MOTIFS_V3:
        for d in DIRECTIONAL_SUFFIXES:
            cols.append(f"eligible_{m}_{d}")
        cols.append(f"retained_unmodified_{m}")
    return cols


def eligible_column(motif: str, direction: str) -> str:
    if motif not in _BEHAVIORAL_SET:
        raise ValueError(f"unknown motif: {motif!r}")
    if direction not in DIRECTIONAL_SUFFIXES:
        raise ValueError(f"unknown direction: {direction!r}")
    return f"eligible_{motif}_{direction}"
