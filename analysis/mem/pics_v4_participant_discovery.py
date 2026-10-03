#!/usr/bin/env python3
"""PICS v4 participant discovery for the final sparse-history rerun.

This ledger is separate from the PICS v3 gated paths and from AAMAS v0.
It names the final jobs and the corrected policy markers, and it refuses to
build participant MEM rows until every final dataset has a completed
participant stage. Exploration, evolution-normal, and evolution-fresh stay
separate. Grouping is ``dataset::run_id::participant_id``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.mem.pics_v4_population_annotation import (  # noqa: E402
    AUTHORITATIVE_JOBS,
    OUTPUT_REL,
    REQUIRED_POLICY_MARKERS,
    PicsV4AnnotationError,
    _job_dir,
    _marker_ok,
    assert_final_job_mapping,
)

PARTICIPANT_LEDGER_REL = "analysis_2026Sep/Sep30_pics_v4/mem/participant_final"
PARTICIPANT_SOURCES = ("exploration", "evolution_normal", "evolution_fresh")
GROUPING_KEY = "dataset::run_id::participant_id"


class PicsV4ParticipantIncomplete(PicsV4AnnotationError):
    """Participant stages are not all complete, so no MEM dataset is built."""


def _participant_marker(job_dir: Path) -> Tuple[bool, str]:
    """Read only the selected-stage completion marker. Traces stay unread."""
    marker = job_dir / "selected" / "STAGE_COMPLETE.json"
    if not marker.is_file():
        return False, "participant STAGE_COMPLETE.json is absent"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"participant STAGE_COMPLETE.json is unreadable: {exc}"
    if payload.get("stage") != "selected":
        return False, "participant stage marker is not selected"
    if not payload.get("participant_ids"):
        return False, "participant stage lists no participants"
    return True, ""


def discover_participant_ledger(
    repo: Path,
    jobs: Sequence[Tuple[str, Sequence[str]]] = AUTHORITATIVE_JOBS,
) -> Dict[str, Any]:
    """Return a path ledger only when every final participant stage is complete.

    Incomplete datasets raise. This function does not write a MEM dataset.
    """
    assert_final_job_mapping(jobs)
    if list(jobs) != list(AUTHORITATIVE_JOBS):
        raise PicsV4ParticipantIncomplete(
            "participant discovery requires the full final job mapping"
        )
    repo = Path(repo)
    ready: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, str]] = []
    for job_id, names in jobs:
        for dataset in names:
            job_dir = _job_dir(repo, dataset, job_id)
            row = {"dataset": dataset, "job_id": str(job_id)}
            if not job_dir.is_dir():
                incomplete.append({**row, "reason": "job directory is absent"})
                continue
            try:
                _marker_ok(job_dir)
            except (PicsV4AnnotationError, RuntimeError, json.JSONDecodeError, OSError) as exc:
                incomplete.append({**row, "reason": str(exc)})
                continue
            ok, reason = _participant_marker(job_dir)
            if not ok:
                incomplete.append({**row, "reason": reason})
                continue
            ready.append(
                {
                    **row,
                    "sources": list(PARTICIPANT_SOURCES),
                    "grouping_key": GROUPING_KEY,
                    "policy_markers": dict(REQUIRED_POLICY_MARKERS),
                }
            )
    if incomplete or len(ready) != sum(len(names) for _job, names in AUTHORITATIVE_JOBS):
        sample = incomplete[0]["dataset"] if incomplete else "unknown"
        raise PicsV4ParticipantIncomplete(
            f"participant stages are incomplete ({len(incomplete)} datasets, "
            f"first={sample}); no participant MEM dataset was built"
        )
    return {
        "ledger_rel": PARTICIPANT_LEDGER_REL,
        "population_annotation_rel": OUTPUT_REL,
        "built": False,
        "n_datasets": len(ready),
        "sources": list(PARTICIPANT_SOURCES),
        "grouping_key": GROUPING_KEY,
        "datasets": ready,
    }
