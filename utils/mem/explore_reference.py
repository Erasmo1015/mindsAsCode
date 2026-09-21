"""Resolve gate-winning target-population rank-1 for explore-phase MEM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils.mem.reference_types import REF_POPULATION_PROGRAM


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def find_gate_record(start: Path) -> Optional[Path]:
    """Walk upward from a run/participant path for ``gate/gate_record.json``."""
    cur = Path(start).resolve()
    for _ in range(8):
        candidate = cur / "gate" / "gate_record.json"
        if candidate.is_file():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def gate_winning_rank1_path(gate_record: Dict[str, Any]) -> Optional[Path]:
    arm = str(gate_record.get("selected_arm") or "").strip().lower()
    if arm == "transfer":
        raw = gate_record.get("transfer_rank1_path")
    else:
        raw = gate_record.get("control_rank1_path")
    if not raw:
        return None
    p = Path(str(raw))
    return p if p.is_file() else None


def resolve_explore_shared_reference(
    *,
    participant_dir: Path,
    ctx: Optional[Dict[str, Any]],
    run_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    """Strict shared reference for explore: gate-winning target-pop rank-1.

    Returns ``(parent_dict, ref_id, resolution_mode)``.

    Preference order:
      1) gate_record winning-arm ``*_rank1_path`` code
      2) match that code to an explore ``selected_parents`` entry (keep its id/score)
      3) else synthetic id ``gate_winning_rank1`` with population_program type
      4) if gate missing: explore context ``best_selected_parent_id`` / first parent
    """
    parents = list((ctx or {}).get("selected_parents") or [])
    search_roots = [Path(participant_dir)]
    if run_dir is not None:
        search_roots.insert(0, Path(run_dir))
    gate_path = None
    for root in search_roots:
        gate_path = find_gate_record(root)
        if gate_path is not None:
            break

    gate_code = ""
    gate_file: Optional[Path] = None
    if gate_path is not None:
        try:
            gate_rec = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            gate_rec = {}
        gate_file = gate_winning_rank1_path(gate_rec) if gate_rec else None
        if gate_file is not None:
            try:
                gate_code = gate_file.read_text(encoding="utf-8")
            except OSError:
                gate_code = ""

    if gate_code.strip():
        gate_sha = _sha256_text(gate_code)
        for p in parents:
            pcode = str(p.get("code") or "")
            if pcode and _sha256_text(pcode) == gate_sha:
                out = dict(p)
                out["code"] = pcode
                pid = str(out.get("program_id") or "gate_winning_rank1")
                return out, pid, "gate_winning_rank1_matched_selected_parent"
        # No hydrated parent match — still use gate code as shared reference.
        score = None
        for p in parents:
            if p.get("selection_score") is not None:
                try:
                    score = float(p["selection_score"])
                except (TypeError, ValueError):
                    score = None
                break
        synth = {
            "program_id": "gate_winning_rank1",
            "code": gate_code,
            "selection_score": score,
            "reference_type": REF_POPULATION_PROGRAM,
            "_gate_rank1_path": str(gate_file) if gate_file else None,
        }
        return synth, "gate_winning_rank1", "gate_winning_rank1_file"

    # Gate unavailable: fall back to explore context shared parent (must exist).
    best_id = (ctx or {}).get("best_selected_parent_id")
    if best_id is not None:
        for p in parents:
            if p.get("program_id") == best_id and str(p.get("code") or "").strip():
                return dict(p), str(best_id), "explore_context_best_selected_parent"
    for p in parents:
        if str(p.get("code") or "").strip():
            pid = str(p.get("program_id") or "explore_shared_parent")
            return dict(p), pid, "explore_context_first_selected_parent"
    return None, None, "unresolved_explore_reference"
