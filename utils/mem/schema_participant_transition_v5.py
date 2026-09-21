"""Participant-program transition schema v5 (ICLR Method; five constructs).

Annotates reference-parent → candidate transitions for person-level MEM.

Final five-construct vocabulary (aligned with population_transition_v5):
  history, value, probability_used, feedback, learning

Removed relative to participant schema v2/v3:
  risk, other_behavioral  (and any explicit_risk synonym)

LLM reports reference/candidate *presence* plus which shared constructs were
*modified*. Added / removed are derived deterministically:

  added   = candidate_state \\ reference_state
  removed = reference_state \\ candidate_state
  modified ⊆ reference_state ∩ candidate_state
  unchanged (retained_unmodified) = intersection \\ modified

Eligibility / risk sets (per construct ``c``):

  eligible_c_added
      Addition risk set: reference lacks ``c``.

  eligible_c_removed
      Removal risk set: reference has ``c``.

  eligible_c_modified
      **Retained-construct modification risk set**:
      ``reference_has_c AND candidate_has_c``.
      Within this set the focal contrast is
      ``c_modified`` vs ``retained_unmodified_c`` (mutually exclusive).
      This is *not* the broader pre-transition modification opportunity.

  modification_opportunity_c
      Broader pre-transition opportunity: ``reference_has_c``
      (construct could be modified *or* removed on the next step).
      Distinct from the retained-construct modification risk set above.

  retained_unmodified_c
      Both have ``c`` and ``c`` was not marked modified.

Resume identity is globally unique across dataset, run, participant, iteration,
candidate, and reference parent (fixes prior collisions when participant_id +
local candidate_id were reused across datasets/runs).

Schema v2/v3 artifacts remain valid; this module is additive.
Paper/professor-facing term: construct (not motif) per ICLR_PAPER_TERMINOLOGY.md.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from utils.mem.schema_population_motif_v5 import (
    BEHAVIORAL_MOTIF_DEFINITIONS_V5,
    BEHAVIORAL_MOTIFS_V5,
)
from utils.mem.schema_v2 import (  # noqa: E402
    DIRECTIONAL_SUFFIXES as _DIRECTIONAL_SUFFIXES_V2,
    STRUCTURAL_OPERATIONS_V2,
)

DIRECTIONAL_SUFFIXES = _DIRECTIONAL_SUFFIXES_V2

BEHAVIORAL_MOTIFS: Tuple[str, ...] = tuple(BEHAVIORAL_MOTIFS_V5)
BEHAVIORAL_MOTIF_DEFINITIONS: Dict[str, str] = dict(BEHAVIORAL_MOTIF_DEFINITIONS_V5)
STRUCTURAL_OPERATIONS: Tuple[str, ...] = tuple(STRUCTURAL_OPERATIONS_V2)

_BEHAVIORAL_SET = frozenset(BEHAVIORAL_MOTIFS)
_STRUCTURAL_SET = frozenset(STRUCTURAL_OPERATIONS)


def directional_behavioral_column(motif: str, direction: str) -> str:
    if motif not in _BEHAVIORAL_SET:
        raise ValueError(f"unknown behavioral construct: {motif!r}")
    if direction not in DIRECTIONAL_SUFFIXES:
        raise ValueError(f"unknown direction: {direction!r}")
    return f"{motif}_{direction}"


def structural_column(op: str) -> str:
    if op not in _STRUCTURAL_SET:
        raise ValueError(f"unknown structural operation: {op!r}")
    return f"structural_{op}"

SCHEMA_VERSION = 5
PROMPT_VERSION = "participant_transition_v5"
ANNOTATION_KIND = "participant_program_motif_transition"

# Primary focal candidates from earlier person-MEM work; reassess support on
# final Schema-v5 runs before locking a model.
PRIMARY_FOCAL_CANDIDATES: Tuple[str, ...] = (
    "history_modified",
    "value_modified",
    "feedback_added",
    "feedback_modified",
)

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

TRANSITION_TYPES = (
    "added",
    "removed",
    "modified",
    "unchanged",
    "absent",
)


def is_schema_v5_row(rec: Dict[str, Any]) -> bool:
    try:
        return int(rec.get("schema_version", -1)) == SCHEMA_VERSION
    except (TypeError, ValueError):
        return False


def annotation_resume_key(
    dataset: Any,
    run_id: Any,
    participant_id: Any,
    iteration: Any,
    candidate_id: str,
    reference_id: Optional[str] = None,
    reference_type: Optional[str] = None,
    phase: Optional[str] = None,
) -> Tuple[Any, ...]:
    """Globally unique resume identity for participant Schema v5.

    Includes dataset + run + participant + **phase** + iteration + candidate +
    reference so evolution and exploration never collide and the same local
    candidate_id cannot collide across jobs or pairings.
    """
    try:
        iter_norm: Any = int(iteration) if iteration is not None else ""
    except (TypeError, ValueError):
        iter_norm = str(iteration) if iteration is not None else ""
    return (
        str(dataset) if dataset is not None else "",
        str(run_id) if run_id is not None else "",
        str(participant_id) if participant_id is not None else "",
        str(phase) if phase is not None else "",
        iter_norm,
        str(candidate_id),
        str(reference_id) if reference_id is not None else "",
        str(reference_type) if reference_type is not None else "",
    )


def annotation_resume_key_legacy_no_phase(
    dataset: Any,
    run_id: Any,
    participant_id: Any,
    iteration: Any,
    candidate_id: str,
    reference_id: Optional[str] = None,
    reference_type: Optional[str] = None,
) -> Tuple[Any, ...]:
    """Pre-phase resume key (pilot rows). Used only for backward-compatible skip."""
    try:
        iter_norm: Any = int(iteration) if iteration is not None else ""
    except (TypeError, ValueError):
        iter_norm = str(iteration) if iteration is not None else ""
    return (
        str(dataset) if dataset is not None else "",
        str(run_id) if run_id is not None else "",
        str(participant_id) if participant_id is not None else "",
        iter_norm,
        str(candidate_id),
        str(reference_id) if reference_id is not None else "",
        str(reference_type) if reference_type is not None else "",
    )


def global_candidate_id(
    dataset: Any,
    run_id: Any,
    participant_id: Any,
    iteration: Any,
    candidate_id: str,
    phase: Optional[str] = None,
) -> str:
    """Stable string ID spanning dataset/run/participant/phase/iteration/candidate."""
    try:
        it = int(iteration)
    except (TypeError, ValueError):
        it = iteration
    ph = str(phase) if phase is not None else ""
    return f"{dataset}|{run_id}|{participant_id}|{ph}|{it}|{candidate_id}"


def normalize_modified_motifs_to_intersection(
    reference_state: Sequence[str],
    candidate_state: Sequence[str],
    modified_motifs: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Keep only modified ∩ (reference ∩ candidate); return (kept, dropped)."""
    inter = set(reference_state) & set(candidate_state)
    kept: List[str] = []
    dropped: List[str] = []
    seen: Set[str] = set()
    for m in modified_motifs:
        if m in seen:
            continue
        seen.add(str(m))
        if m in inter:
            kept.append(str(m))
        else:
            dropped.append(str(m))
    return kept, dropped


def motif_definitions_block() -> str:
    return "\n".join(
        f"- {m}: {BEHAVIORAL_MOTIF_DEFINITIONS[m]}" for m in BEHAVIORAL_MOTIFS
    )


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
            label = "structural_operation" if allowed is _STRUCTURAL_SET else "construct"
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


def transition_type_for_construct(
    motif: str,
    *,
    reference_has: bool,
    candidate_has: bool,
    modified: bool,
    no_meaningful_change: bool = False,
) -> str:
    """Return added|removed|modified|unchanged|absent for one construct."""
    if no_meaningful_change:
        if reference_has and candidate_has:
            return "unchanged"
        return "absent"
    if (not reference_has) and candidate_has:
        return "added"
    if reference_has and (not candidate_has):
        return "removed"
    if reference_has and candidate_has:
        return "modified" if modified else "unchanged"
    return "absent"


def guided_json_schema_for_batch_v5(expected_ids: Sequence[str]) -> Dict[str, Any]:
    """JSON Schema for schema-v5 LLM responses (no added/removed fields).

    ``minItems`` / ``maxItems`` equal ``len(expected_ids)`` so an empty array
    cannot validate for a nonempty batch under guided decoding.
    """
    n = len(list(expected_ids))
    motif_items = {"type": "string", "enum": list(BEHAVIORAL_MOTIFS)}
    struct_items = {"type": "string", "enum": list(STRUCTURAL_OPERATIONS)}
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
            "confidence": {"type": "number"},
        },
        "required": list(REQUIRED_LLM_FIELDS),
        "additionalProperties": False,
    }
    return {
        "type": "array",
        "items": item,
        "minItems": n,
        "maxItems": n,
    }


def validate_annotation_response_v5(
    payload: Any,
    *,
    expected_ids: Sequence[str],
    prompt_version: str = PROMPT_VERSION,
    normalize_modified_outside_intersection: bool = True,
    normalizations_out: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Validate schema-v5 JSON; derive added/removed.

    By default, ``modified_motifs`` entries outside reference∩candidate are
    **deterministically dropped** (adds remain derived from presence sets) and
    each drop is recorded in ``normalizations_out`` when provided.
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

    if len(expected) > 0 and len(rows) == 0:
        return (
            False,
            f"empty annotation array for nonempty batch (expected {len(expected)} "
            f"candidate_id(s): {expected})",
            [],
        )

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
            return False, f"{cid}: primary_edit is not allowed in schema v5", []

        # Reject removed Schema-v2/v3 labels if the model invents them.
        banned = {"risk", "other_behavioral", "explicit_risk", "explicit_risk_mechanism"}
        for key in (
            "reference_motif_state",
            "candidate_motif_state",
            "modified_motifs",
        ):
            if key not in row:
                return False, f"{cid}: missing required field {key}", []
            if isinstance(row[key], list) and any(x in banned for x in row[key]):
                return False, f"{cid}: banned construct in {key} (Schema v5 five only)", []

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
        if "structural_operations" not in row:
            return False, f"{cid}: missing required field structural_operations", []
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
        kept_mod, dropped_mod = normalize_modified_motifs_to_intersection(
            ref_state, cand_state, modified_raw
        )
        if dropped_mod:
            if not normalize_modified_outside_intersection:
                return (
                    False,
                    f"{cid}: modified_motifs {dropped_mod} not in reference∩candidate "
                    f"(intersection={sorted(inter)})",
                    [],
                )
            note = {
                "candidate_id": cid,
                "action": "drop_modified_outside_intersection",
                "dropped_modified_motifs": list(dropped_mod),
                "kept_modified_motifs": list(kept_mod),
                "intersection": sorted(inter),
                "reference_motif_state": list(ref_state),
                "candidate_motif_state": list(cand_state),
            }
            if normalizations_out is not None:
                normalizations_out.append(note)
            modified_raw = kept_mod

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

        transition_by_construct: Dict[str, str] = {}
        for m in BEHAVIORAL_MOTIFS:
            transition_by_construct[m] = transition_type_for_construct(
                m,
                reference_has=m in ref_state,
                candidate_has=m in cand_state,
                modified=m in modified,
                no_meaningful_change=nmc,
            )

        out_row: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": prompt_version,
            "annotation_kind": ANNOTATION_KIND,
            "candidate_id": cid,
            "reference_motif_state": list(ref_state),
            "candidate_motif_state": list(cand_state),
            "added_motifs": list(added),
            "removed_motifs": list(removed),
            "modified_motifs": list(modified),
            "structural_operations": list(structural),
            "transition_by_construct": transition_by_construct,
            "no_meaningful_change": nmc,
            "evidence": list(evidence),
            "confidence": conf_f,
        }
        if dropped_mod and normalize_modified_outside_intersection:
            out_row["modified_motifs_normalization"] = {
                "dropped": list(dropped_mod),
                "kept": list(modified),
            }
        cleaned.append(out_row)

    missing = [cid for cid in expected if cid not in seen]
    if missing:
        return False, f"missing candidate_id(s): {missing}", cleaned
    return True, "", cleaned


def state_and_eligibility_flags(ann: Dict[str, Any]) -> Dict[str, int]:
    """Binary has_*, direction, eligible_*, retained_unmodified_* for a v5 row."""
    if not is_schema_v5_row(ann) and int(ann.get("schema_version", -1) or -1) != 5:
        raise ValueError("state_and_eligibility_flags requires schema_version=5")

    ref = set(ann.get("reference_motif_state") or [])
    cand = set(ann.get("candidate_motif_state") or [])
    added = set(ann.get("added_motifs") or [])
    removed = set(ann.get("removed_motifs") or [])
    modified = set(ann.get("modified_motifs") or [])
    structural = set(ann.get("structural_operations") or [])
    nmc = bool(ann.get("no_meaningful_change"))

    out: Dict[str, Any] = {}
    for m in BEHAVIORAL_MOTIFS:
        rh = 1 if m in ref else 0
        ch = 1 if m in cand else 0
        out[f"reference_has_{m}"] = rh
        out[f"candidate_has_{m}"] = ch
        a = 1 if m in added else 0
        rmv = 1 if m in removed else 0
        mod = 1 if m in modified else 0
        if nmc:
            a = rmv = mod = 0
        out[directional_behavioral_column(m, "added")] = a
        out[directional_behavioral_column(m, "removed")] = rmv
        out[directional_behavioral_column(m, "modified")] = mod
        out[f"eligible_{m}_added"] = 1 if rh == 0 else 0
        out[f"eligible_{m}_removed"] = 1 if rh == 1 else 0
        # Retained-construct modification risk set (NOT reference_has alone).
        out[f"eligible_{m}_modified"] = 1 if (rh == 1 and ch == 1) else 0
        # Broader pre-transition opportunity (modify or remove).
        out[f"modification_opportunity_{m}"] = rh
        out[f"retained_unmodified_{m}"] = (
            1 if (rh == 1 and ch == 1 and mod == 0) else 0
        )
        out[f"transition_{m}"] = transition_type_for_construct(
            m,
            reference_has=bool(rh),
            candidate_has=bool(ch),
            modified=bool(mod),
            no_meaningful_change=nmc,
        )

    for op in STRUCTURAL_OPERATIONS:
        out[structural_column(op)] = 0 if nmc else (1 if op in structural else 0)
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


def all_directional_behavioral_columns() -> List[str]:
    return [
        directional_behavioral_column(m, d)
        for m in BEHAVIORAL_MOTIFS
        for d in DIRECTIONAL_SUFFIXES
    ]


def all_structural_columns() -> List[str]:
    return [structural_column(op) for op in STRUCTURAL_OPERATIONS]


def all_state_columns() -> List[str]:
    cols: List[str] = []
    for m in BEHAVIORAL_MOTIFS:
        cols.append(f"reference_has_{m}")
        cols.append(f"candidate_has_{m}")
    return cols


def all_eligibility_columns() -> List[str]:
    cols: List[str] = []
    for m in BEHAVIORAL_MOTIFS:
        for d in DIRECTIONAL_SUFFIXES:
            cols.append(f"eligible_{m}_{d}")
        cols.append(f"modification_opportunity_{m}")
        cols.append(f"retained_unmodified_{m}")
    return cols


def all_transition_type_columns() -> List[str]:
    return [f"transition_{m}" for m in BEHAVIORAL_MOTIFS]


def eligible_column(motif: str, direction: str) -> str:
    if motif not in _BEHAVIORAL_SET:
        raise ValueError(f"unknown construct: {motif!r}")
    if direction not in DIRECTIONAL_SUFFIXES:
        raise ValueError(f"unknown direction: {direction!r}")
    return f"eligible_{motif}_{direction}"


def risk_set_definition(effect: str) -> Dict[str, str]:
    """Human-readable risk-set definition for a directional effect column."""
    if effect.endswith("_added"):
        c = effect[: -len("_added")]
        return {
            "effect": effect,
            "construct": c,
            "operation": "added",
            "risk_set_name": "addition_risk_set",
            "eligibility_column": f"eligible_{c}_added",
            "rule": "reference_has_c == 0",
            "focal_contrast": f"{c}_added vs not-added within addition risk set",
        }
    if effect.endswith("_removed"):
        c = effect[: -len("_removed")]
        return {
            "effect": effect,
            "construct": c,
            "operation": "removed",
            "risk_set_name": "removal_risk_set",
            "eligibility_column": f"eligible_{c}_removed",
            "rule": "reference_has_c == 1",
            "focal_contrast": f"{c}_removed vs not-removed within removal risk set",
        }
    if effect.endswith("_modified"):
        c = effect[: -len("_modified")]
        return {
            "effect": effect,
            "construct": c,
            "operation": "modified",
            "risk_set_name": "retained_construct_modification_risk_set",
            "eligibility_column": f"eligible_{c}_modified",
            "rule": "reference_has_c == 1 AND candidate_has_c == 1",
            "focal_contrast": (
                f"{c}_modified vs retained_unmodified_{c} "
                "within the retained-construct set"
            ),
            "broader_pre_transition_opportunity": (
                f"modification_opportunity_{c} == reference_has_{c} "
                "(includes rows where c can still be removed)"
            ),
            "note": (
                "eligible_c_modified is NOT the same as reference_has_c; "
                "use retained-construct set for modification focals."
            ),
        }
    raise ValueError(f"not a directional construct effect: {effect!r}")


def verify_delta_f_consistency(
    *,
    delta_f: Any,
    candidate_score: Any,
    reference_score: Any,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tuple[bool, str]:
    """Require finite scores and delta_f ≈ candidate_score - reference_score."""
    try:
        d = float(delta_f)
        c = float(candidate_score)
        r = float(reference_score)
    except (TypeError, ValueError):
        return False, "nonfinite_or_nonnumeric_delta_f_or_scores"
    if not (math.isfinite(d) and math.isfinite(c) and math.isfinite(r)):
        return False, "nonfinite_delta_f_or_scores"
    expected = c - r
    tol = max(float(atol), float(rtol) * max(1.0, abs(c), abs(r), abs(expected)))
    if abs(d - expected) > tol:
        return (
            False,
            f"delta_f_inconsistent: delta_f={d} != candidate({c})-reference({r})={expected}",
        )
    return True, ""