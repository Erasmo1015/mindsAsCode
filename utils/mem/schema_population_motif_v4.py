#!/usr/bin/env python3
"""Population-program motif schema v4 (T-PICS source runs).

Annotates **both**:
  1) candidate motif *presence* with per-motif applicability / confidence /
     rationale / code evidence;
  2) exact parent→candidate transitions (added / removed derived; modified
     reported) for fitness-effect MEM.

Candidate presence inventories are **derived from complete motif_details**
(not trusted from a parallel LLM list). Missing motifs, malformed details, and
truncated/invalid JSON still fail validation.

Frozen motif vocabulary (signature-discovery audit):
  core: history, value, probability_used, feedback, learning
  specialist optional: explicit_risk_mechanism
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

BEHAVIORAL_MOTIFS_V4: Tuple[str, ...] = (
    "history",
    "value",
    "probability_used",
    "feedback",
    "learning",
    "explicit_risk_mechanism",
)

BEHAVIORAL_MOTIF_DEFINITIONS_V4: Dict[str, str] = {
    "history": (
        "Program explicitly uses earlier choices, outcomes, trials, streaks, "
        "recency, counts, or a supplied history/trial buffer to change the "
        "current decision. Unused history parameters do not count."
    ),
    "value": (
        "Program computes or compares option attractiveness, utility, expected "
        "payoff, benefits/costs, or a general option score that drives choice. "
        "Constant/random choice without option scoring does not count."
    ),
    "probability_used": (
        "Program explicitly reads or uses probability/likelihood/odds/uncertainty "
        "fields from the problem, including ordinary linear EV terms such as p*x. "
        "Empirical success rates built only from feedback history are feedback/"
        "learning, not probability_used. Unused probability fields do not count."
    ),
    "feedback": (
        "Program directly uses observed reward, correctness, success/failure, or "
        "outcome feedback to influence a later choice. Static current-trial "
        "payoffs are value, not feedback. Feedback without an update rule is "
        "NOT learning."
    ),
    "learning": (
        "Program updates or reconstructs an internal belief, preference, option "
        "estimate, or decision rule across trials from experience (running means, "
        "Bayesian update, Q-like counts, parameter adaptation). Merely re-reading "
        "the last choice/reward without an update rule is history/feedback, not "
        "learning."
    ),
    "explicit_risk_mechanism": (
        "Beyond merely reading probability, the program explicitly models risk/"
        "uncertainty (nonlinear probability weighting, variance/downside, loss "
        "aversion multipliers, risk penalty/bonus, or other uncertainty-sensitive "
        "transforms). Linear EV alone is probability_used only. A variable named "
        "risk_aversion that only scales a non-probability state (e.g. pump count) "
        "does not count. Dataset subject matter alone does not count."
    ),
}

APPLICABILITY_VALUES: Tuple[str, ...] = (
    "applicable",
    "structural_na",
    "not_applicable",
)

BEHAVIORAL_MOTIFS = BEHAVIORAL_MOTIFS_V4
_BEHAVIORAL_SET = frozenset(BEHAVIORAL_MOTIFS)
_APPLICABILITY_SET = frozenset(APPLICABILITY_VALUES)

SCHEMA_VERSION = 4
PROMPT_VERSION = "population_transition_v4_2"
ANNOTATION_KIND = "population_program_motif_transition"

# candidate_motif_state is derived from motif_details (not required from LLM).
REQUIRED_LLM_FIELDS = (
    "candidate_id",
    "reference_motif_state",
    "modified_motifs",
    "motif_details",
    "confidence",
)


def resume_key(
    dataset: str,
    run_id: str,
    iteration: Any,
    candidate: str,
    parent: str,
) -> str:
    """Resume identity: dataset|run|iteration|candidate|parent."""
    return f"{dataset}|{run_id}|{int(iteration)}|{candidate}|{parent}"


def _clean_motifs(cid: str, key: str, motifs: Any) -> Tuple[Optional[List[str]], str]:
    if not isinstance(motifs, list):
        return None, f"{cid}: {key} must be a list"
    cleaned: List[str] = []
    seen: Set[str] = set()
    for m in motifs:
        if m not in _BEHAVIORAL_SET:
            return None, f"{cid}: invalid motif {m!r} in {key}"
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
    r = set(reference_state)
    c = set(candidate_state)
    inter = r & c
    added = [m for m in candidate_state if m not in r]
    removed = [m for m in reference_state if m not in c]
    modified = [m for m in modified_motifs if m in inter]
    return added, removed, modified


def _validate_motif_details(
    cid: str,
    details: Any,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Require one complete detail row per allowed motif; return ordered details."""
    if not isinstance(details, list):
        return None, f"{cid}: motif_details must be a list"
    by_motif: Dict[str, Dict[str, Any]] = {}
    for i, row in enumerate(details):
        if not isinstance(row, dict):
            return None, f"{cid}: motif_details[{i}] must be an object"
        motif = row.get("motif")
        if motif not in _BEHAVIORAL_SET:
            return None, f"{cid}: motif_details[{i}] invalid motif {motif!r}"
        if motif in by_motif:
            return None, f"{cid}: duplicate motif_details for {motif!r}"
        presence = row.get("presence")
        if not isinstance(presence, bool):
            return None, f"{cid}: motif_details[{motif}].presence must be bool"
        applicability = row.get("applicability")
        if applicability not in _APPLICABILITY_SET:
            return None, (
                f"{cid}: motif_details[{motif}].applicability must be one of "
                f"{list(APPLICABILITY_VALUES)}"
            )
        try:
            conf = float(row.get("confidence"))
        except (TypeError, ValueError):
            return None, f"{cid}: motif_details[{motif}].confidence must be a number"
        if not (0.0 <= conf <= 1.0):
            return None, f"{cid}: motif_details[{motif}].confidence out of range"
        rationale = row.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            return None, f"{cid}: motif_details[{motif}].rationale must be a non-empty string"
        evidence = row.get("code_evidence")
        if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
            return None, f"{cid}: motif_details[{motif}].code_evidence must be list[str]"
        if presence and applicability == "structural_na":
            return None, (
                f"{cid}: motif_details[{motif}] cannot be present with structural_na"
            )
        by_motif[str(motif)] = {
            "motif": str(motif),
            "presence": bool(presence),
            "applicability": str(applicability),
            "confidence": conf,
            "rationale": rationale.strip(),
            "code_evidence": list(evidence),
        }
    missing = [m for m in BEHAVIORAL_MOTIFS if m not in by_motif]
    if missing:
        return None, f"{cid}: motif_details missing motifs: {missing}"
    return [by_motif[m] for m in BEHAVIORAL_MOTIFS], ""


def candidate_state_from_details(details: Sequence[Dict[str, Any]]) -> List[str]:
    """Derive candidate_motif_state from motif_details presence flags."""
    return [str(d["motif"]) for d in details if d.get("presence")]


def guided_json_schema_for_programs(expected_ids: Sequence[str]) -> Dict[str, Any]:
    motif_items = {"type": "string", "enum": list(BEHAVIORAL_MOTIFS)}
    detail_item = {
        "type": "object",
        "properties": {
            "motif": {"type": "string", "enum": list(BEHAVIORAL_MOTIFS)},
            "presence": {"type": "boolean"},
            "applicability": {"type": "string", "enum": list(APPLICABILITY_VALUES)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
            "code_evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "motif",
            "presence",
            "applicability",
            "confidence",
            "rationale",
            "code_evidence",
        ],
        "additionalProperties": False,
    }
    item = {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "enum": list(expected_ids)},
            "reference_motif_state": {"type": "array", "items": motif_items},
            "modified_motifs": {"type": "array", "items": motif_items},
            "motif_details": {"type": "array", "items": detail_item},
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
    """Validate LLM batch; expected_ids are candidate_id / program_id strings."""
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
        cid = str(row["candidate_id"])
        if cid not in expected:
            return False, f"unexpected candidate_id {cid!r}", []
        if cid in seen:
            return False, f"duplicate candidate_id {cid!r}", []
        seen.add(cid)
        ref_state, err = _clean_motifs(cid, "reference_motif_state", row["reference_motif_state"])
        if ref_state is None:
            return False, err, []
        modified_in, err = _clean_motifs(cid, "modified_motifs", row["modified_motifs"])
        if modified_in is None:
            return False, err, []
        details, err = _validate_motif_details(cid, row["motif_details"])
        if details is None:
            return False, err, []
        cand_state = candidate_state_from_details(details)
        added, removed, modified = derive_directional_motifs(
            ref_state, cand_state, modified_in
        )
        try:
            conf_f = float(row["confidence"])
        except (TypeError, ValueError):
            return False, f"{cid}: confidence must be a number", []
        if not (0.0 <= conf_f <= 1.0):
            return False, f"{cid}: confidence out of range", []
        evidence = []
        for d in details:
            for e in d["code_evidence"]:
                evidence.append(f"{d['motif']}: {e}")
        rows.append(
            {
                "candidate_id": cid,
                "program_id": cid,
                "program_motif_state": list(cand_state),
                "reference_motif_state": list(ref_state),
                "candidate_motif_state": list(cand_state),
                "added_motifs": added,
                "removed_motifs": removed,
                "modified_motifs": modified,
                "motif_details": details,
                "presence": {d["motif"]: d["presence"] for d in details},
                "applicability": {d["motif"]: d["applicability"] for d in details},
                "rationale": {d["motif"]: d["rationale"] for d in details},
                "code_evidence": {d["motif"]: d["code_evidence"] for d in details},
                "evidence": evidence,
                "confidence": conf_f,
                "schema_version": SCHEMA_VERSION,
                "prompt_version": prompt_version,
                "annotation_kind": ANNOTATION_KIND,
            }
        )
    missing = [x for x in expected if x not in seen]
    if missing:
        return False, f"missing candidate_id(s): {missing}", []
    return True, "", rows


def motif_definitions_block() -> str:
    lines = []
    for m in BEHAVIORAL_MOTIFS:
        lines.append(f"- {m}: {BEHAVIORAL_MOTIF_DEFINITIONS_V4[m]}")
    return "\n".join(lines)


def applicability_block() -> str:
    return (
        "applicability values:\n"
        "- applicable: motif is meaningful to judge for this program/problem schema\n"
        "- structural_na: problem schema lacks fields needed for this motif "
        "(e.g. no probability fields for probability_used / explicit_risk_mechanism; "
        "no outcome feedback channel for feedback/learning)\n"
        "- not_applicable: other N/A (rare; prefer structural_na when schema-driven)"
    )
