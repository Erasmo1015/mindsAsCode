"""
Residual Behavioral Uncertainty (RBU) helpers: structure-score JSON parsing and RBU math.

RBU = clip(BIR - S, 0, 1) where S is the sum of numeric JSON component scores in [0, 1].
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


def sum_numeric_components(obj: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Sum all **top-level** numeric fields except ``summary``; ignore nested mappings/lists.

    Returns (raw_sum, component_name -> value).
    """
    if not isinstance(obj, Mapping):
        raise StructureScoreParseError(f"parsed JSON must be an object, got {type(obj).__name__}")
    components: Dict[str, float] = {}
    for k, v in obj.items():
        if k == "summary":
            continue
        if _is_real_number(v):
            components[str(k)] = float(v)
    if not components:
        raise StructureScoreParseError(
            "no top-level numeric component fields found (excluding 'summary'); "
            f"keys present: {list(obj.keys())!r}"
        )
    total = float(sum(components.values()))
    if not math.isfinite(total):
        raise StructureScoreParseError(f"non-finite sum of components: {total!r}")
    return total, components


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

    Each payload may be:
    - ``{"structure_score": float, "evidence_scores": {...}, "summary": ...}`` — **S** is ``structure_score`` (clamped);
      **components** are numeric entries from ``evidence_scores`` (excluding ``summary``), or
      ``{"structure_score": S}`` if no evidence scores.
    - Or a flat component object (legacy): sum numeric fields except ``summary`` as **S** (clamped).
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
        if "structure_score" in v and _is_real_number(v["structure_score"]):
            s_clamped = float(min(1.0, max(0.0, float(v["structure_score"]))))
            comps: Dict[str, float] = {}
            ev = v.get("evidence_scores")
            if isinstance(ev, Mapping):
                for ek, evv in ev.items():
                    if str(ek) == "summary":
                        continue
                    if _is_real_number(evv):
                        comps[str(ek)] = float(evv)
            if not comps:
                comps = {"structure_score": s_clamped}
            out[pid] = (s_clamped, comps)
        elif isinstance(v.get("evidence_scores"), Mapping):
            try:
                raw_sum, comps = sum_numeric_components(v["evidence_scores"])
            except StructureScoreParseError:
                flat = {
                    k: val
                    for k, val in v.items()
                    if k not in ("summary", "evidence_scores") and _is_real_number(val)
                }
                if not flat:
                    raise StructureScoreParseError(
                        f"participant_{pid}: empty or non-numeric evidence_scores and no numeric top-level fields"
                    )
                raw_sum, comps = sum_numeric_components(flat)
            s_clamped = float(min(1.0, max(0.0, raw_sum)))
            out[pid] = (s_clamped, comps)
        else:
            raw_sum, comps = sum_numeric_components(v)
            s_clamped = float(min(1.0, max(0.0, raw_sum)))
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
    Parse structure score ``S`` from ``Structure_score.txt`` (or ``raw_text``).

    - Extracts the first JSON object from the file/text.
    - Sums all top-level numeric values except ``summary``.
    - Clamps ``S`` to ``[0, 1]``.

    Returns ``(S_clamped, components)`` where ``components`` maps field names to numeric values.
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
    raw_sum, components = sum_numeric_components(loaded)
    s_clamped = float(min(1.0, max(0.0, raw_sum)))
    return s_clamped, components


def clip01(x: float) -> float:
    """Clip scalar to [0, 1]."""
    return float(max(0.0, min(1.0, float(x))))


def compute_rbu(bir: float, structure_score: float, *, structure_weight: float = 0.5) -> float:
    """RBU = clip01(BIR - structure_weight * structure_score)."""
    w = float(structure_weight)
    return clip01(float(bir) - w * float(structure_score))
