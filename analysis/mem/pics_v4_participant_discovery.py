#!/usr/bin/env python3
"""PICS v4 participant discovery for the final sparse-history rerun.

This ledger is separate from the PICS v3 gated paths and from AAMAS v0.
It names the final jobs and the corrected policy markers, and it refuses to
build participant MEM rows until every final dataset has a completed
participant stage. Exploration, evolution-normal, and evolution-fresh stay
separate. Grouping is ``dataset::run_id::participant_id``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.mem.pics_v4_uniform_v8_reference import TRANSITION_IDENTITY_POLICY  # noqa: E402

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
SLICE_FILTERS = {
    "exploration": {"phase": "explore", "source": "explore"},
    "evolution_normal": {"phase": "evolution", "source": "normal"},
    "evolution_fresh": {"phase": "evolution", "source": "fresh"},
}
REFERENCE_POLICY = "pics_v4_uniform_v8"


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


def _selected_dir(repo: Path, dataset: str, job_id: str) -> Path:
    return _job_dir(repo, dataset, job_id) / "selected"


def participant_launch_plan(repo: Path) -> Dict[str, Any]:
    """Trace lists and commands for the three slices. Raises until all 15 are complete.

    This does not annotate, build a CSV, or fit. ``discover_participant_ledger``
    is the fail-closed gate. Historical jobs never enter the lists.
    """
    repo = Path(repo)
    ledger = discover_participant_ledger(repo)
    traces: List[Dict[str, str]] = []
    for row in ledger["datasets"]:
        selected = _selected_dir(repo, row["dataset"], row["job_id"])
        marker = json.loads((selected / "STAGE_COMPLETE.json").read_text(encoding="utf-8"))
        for participant_id in marker.get("participant_ids") or []:
            trace = selected / f"participant_{participant_id}" / "mem_trace.jsonl"
            if not trace.is_file():
                raise PicsV4ParticipantIncomplete(
                    f"participant trace missing: {trace}"
                )
            text = str(trace)
            if "/mem/population/" in text or "gated_job_paths_g5e50p30" in text:
                raise PicsV4AnnotationError(f"refusing historical participant path: {trace}")
            traces.append(
                {
                    "dataset": row["dataset"],
                    "job_id": str(row["job_id"]),
                    "participant_id": str(participant_id),
                    "trace": trace.relative_to(repo).as_posix(),
                    "run_dir": selected.relative_to(repo).as_posix(),
                }
            )
    commands = {}
    for source in PARTICIPANT_SOURCES:
        filt = SLICE_FILTERS[source]
        ann = f"{PARTICIPANT_LEDGER_REL}/annotations/{source}"
        csv = f"{PARTICIPANT_LEDGER_REL}/csv/{source}.csv"
        coverage = f"{PARTICIPANT_LEDGER_REL}/coverage/{source}"
        commands[source] = {
            "phase": filt["phase"],
            "source": filt["source"],
            "annotations": ann,
            "csv": csv,
            "coverage": coverage,
            "annotate": (
                "python analysis/mem/annotate_edits.py "
                f"--run_dir <selected-dir> --output_dir {ann} "
                "--schema_version 5 --reference_policy pics_v4_uniform_v8 "
                f"--phases {filt['phase']} "
                + (
                    "--include_explore --no-include_fresh"
                    if source == "exploration"
                    else "--no-include_explore --include_fresh"
                    if source == "evolution_fresh"
                    else "--no-include_explore --no-include_fresh"
                )
            ),
            "build_csv": (
                "python analysis/mem/build_dataset.py "
                f"--run_dir . --official_all --annotations {ann}/annotations_v5.jsonl "
                f"--output_csv {csv} --schema_version 5 "
                "--reference_policy pics_v4_uniform_v8 "
                f"--phase {filt['phase']} --source {filt['source']}"
            ),
            "coverage": (
                "python analysis/mem/coverage_eligibility_v5.py "
                f"--input_csv {csv} --output_json {coverage}/coverage.json"
            ),
        }
    return {
        **ledger,
        "reference_policy": REFERENCE_POLICY,
        "transition_identity_policy": TRANSITION_IDENTITY_POLICY,
        "grouping_key": GROUPING_KEY,
        "n_traces": len(traces),
        "traces": traces,
        "commands": commands,
        "built": False,
    }


def write_participant_launch_plan(repo: Path) -> Dict[str, Any]:
    """Write trace lists and command text after every participant stage is complete."""
    repo = Path(repo)
    plan = participant_launch_plan(repo)
    root = repo / PARTICIPANT_LEDGER_REL
    trace_dir = root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for source in PARTICIPANT_SOURCES:
        filt = SLICE_FILTERS[source]
        lines = [f"# phase={filt['phase']} source={filt['source']} reference_policy={REFERENCE_POLICY}"]
        lines.extend(row["trace"] for row in plan["traces"])
        (trace_dir / f"{source}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    plan["written"] = str(root.relative_to(repo))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=_REPO)
    parser.add_argument(
        "--write-launch",
        action="store_true",
        help="Write trace lists only when all 15 participant stages are complete.",
    )
    args = parser.parse_args()
    if args.write_launch:
        plan = write_participant_launch_plan(args.repo)
    else:
        plan = discover_participant_ledger(args.repo)
    print(
        f"datasets={plan['n_datasets']} built={plan['built']} "
        f"sources={','.join(plan['sources'])}"
    )


if __name__ == "__main__":
    main()
