"""
Residual Behavioral Uncertainty (RBU) helpers: structure-score JSON parsing and RBU math.

RBU = clip(BIR - S, 0, 1) where S is the mean of clipped numeric evidence scores in [0, 1].
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union


class StructureScoreParseError(RuntimeError):
    """Raised when participant structure-score JSON cannot be parsed (no silent fallbacks)."""


def extract_first_json_object(text: str) -> str:
    """
    Extract the substring of the first top-level JSON object ``{...}`` from ``text``.

    Ignores leading non-brace noise; tracks string literals and brace depth so nested
    structures inside strings do not confuse the scan.
    """
    if text is None or not str(text).strip():
        raise StructureScoreParseError("empty text; cannot extract JSON object")
    s = str(text)
    start = s.find("{")
    if start == -1:
        raise StructureScoreParseError("no '{' found; expected a JSON object")
    depth = 0
    in_str = False
    esc = False
    quote = ""
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                in_str = False
        else:
            if c in ('"', "'"):
                in_str = True
                quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    raise StructureScoreParseError("unclosed JSON object while scanning for first '}'")


def _is_real_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        v = float(x)
        return math.isfinite(v)
    return False


def _extract_clipped_evidence_components(
    participant_id: int,
    participant_payload: Mapping[str, Any],
) -> Dict[str, float]:
    """
    Extract and validate ``participant_<id>.evidence`` values.

    - Requires ``evidence`` to exist and be an object.
    - Requires at least one evidence entry.
    - Requires each evidence value to be a finite numeric scalar.
    - Clips each value to ``[0, 1]``.
    """
    if "evidence" not in participant_payload:
        raise StructureScoreParseError(
            f"participant_{participant_id}: missing required 'evidence' object; "
            "structure_score must be computed as mean(evidence values)."
        )
    evidence = participant_payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise StructureScoreParseError(
            f"participant_{participant_id}: 'evidence' must be a JSON object, got {type(evidence).__name__}"
        )
    if not evidence:
        raise StructureScoreParseError(
            f"participant_{participant_id}: 'evidence' is empty; require at least one numeric evidence value."
        )

    components: Dict[str, float] = {}
    for key, raw_value in evidence.items():
        if not _is_real_number(raw_value):
            raise StructureScoreParseError(
                f"participant_{participant_id}: evidence['{key}'] must be a finite numeric value, "
                f"got {raw_value!r} ({type(raw_value).__name__})."
            )
        components[str(key)] = clip01(float(raw_value))

    if not components:
        raise StructureScoreParseError(
            f"participant_{participant_id}: no numeric evidence values found."
        )
    return components


_PARTICIPANT_STRUCTURE_KEY_RE = re.compile(r"^participant_(\d+)$")


def count_tokens_approx(text: str) -> int:
    """
    Token count for budgeting: tiktoken ``cl100k_base`` when import succeeds, else ``ceil(len/3.5)``.
    """
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(math.ceil(len(text) / 3.5)))


def parse_all_participant_structure_scores(
    raw_text: str,
    *,
    expected_participant_ids: Optional[Tuple[int, ...]] = None,
) -> Dict[int, Tuple[float, Dict[str, float]]]:
    """
    Parse combined LLM output: one JSON object mapping ``participant_{id}`` -> per-participant payload.

    Each payload must be an object containing ``evidence`` (mapping evidence names -> numeric values)
    and optional ``summary``. ``structure_score`` in JSON, if present, is ignored.

    **S** is computed as:
    ``mean(clip01(evidence_value) for evidence_value in participant.evidence.values())``.
    """
    blob = extract_first_json_object(raw_text)
    try:
        loaded = json.loads(blob)
    except json.JSONDecodeError as exc:
        preview = (raw_text or "")[:800].replace("\n", "\\n")
        raise StructureScoreParseError(
            f"invalid JSON in combined structure score: {exc}; raw preview (800 chars max): {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise StructureScoreParseError(f"JSON root must be an object, got {type(loaded).__name__}")

    out: Dict[int, Tuple[float, Dict[str, float]]] = {}
    for key, v in loaded.items():
        m = _PARTICIPANT_STRUCTURE_KEY_RE.match(str(key))
        if not m:
            continue
        pid = int(m.group(1))
        if not isinstance(v, Mapping):
            raise StructureScoreParseError(
                f"participant_{pid} value must be a JSON object, got {type(v).__name__}"
            )
        comps = _extract_clipped_evidence_components(pid, v)
        values = list(comps.values())
        s_clamped = float(sum(values) / float(len(values)))
        if not math.isfinite(s_clamped):
            raise StructureScoreParseError(
                f"participant_{pid}: non-finite structure_score computed from evidence mean: {s_clamped!r}"
            )
        out[pid] = (s_clamped, comps)

    if not out:
        raise StructureScoreParseError(
            "no participant_* keys found in combined structure score JSON; "
            f"top-level keys: {list(loaded.keys())!r}"
        )
    if expected_participant_ids is not None:
        missing = sorted(set(expected_participant_ids) - set(out.keys()))
        if missing:
            raise StructureScoreParseError(
                f"combined structure score JSON missing participants: {missing}; "
                f"parsed ids: {sorted(out.keys())!r}"
            )
    return out


def parse_structure_score(
    path: Union[str, Path],
    *,
    raw_text: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Parse structure score ``S`` from a single-participant JSON object in ``Structure_score.txt`` (or ``raw_text``).

    Required schema:
    ``{"evidence": {"name": number, ...}, "summary": "..."}``

    ``structure_score`` in JSON is ignored. ``S`` is the mean of clipped evidence values.
    Returns ``(S, components)`` where ``components`` maps evidence names to clipped numeric values.
    """
    p = Path(path)
    if raw_text is None:
        if not p.is_file():
            raise StructureScoreParseError(f"structure score file does not exist: {p}")
        raw_text = p.read_text(encoding="utf-8")
    try:
        blob = extract_first_json_object(raw_text)
    except StructureScoreParseError:
        raise
    except Exception as exc:
        raise StructureScoreParseError(f"failed to extract JSON object from {p}: {exc}") from exc
    try:
        loaded = json.loads(blob)
    except json.JSONDecodeError as exc:
        preview = (raw_text or "")[:800].replace("\n", "\\n")
        raise StructureScoreParseError(
            f"invalid JSON in extracted object from {p}: {exc}; raw preview (800 chars max): {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise StructureScoreParseError(f"JSON root must be an object, got {type(loaded).__name__}")
    components = _extract_clipped_evidence_components(0, loaded)
    values = list(components.values())
    s = float(sum(values) / float(len(values)))
    if not math.isfinite(s):
        raise StructureScoreParseError(
            f"non-finite structure_score computed from evidence mean: {s!r}"
        )
    return s, components


def clip01(x: float) -> float:
    """Clip scalar to [0, 1]."""
    return float(max(0.0, min(1.0, float(x))))


def compute_rbu(bir: float, structure_score: float, *, structure_weight: float = 0.5) -> float:
    """RBU = clip01(BIR - structure_weight * structure_score)."""
    w = float(structure_weight)
    return clip01(float(bir) - w * float(structure_score))
