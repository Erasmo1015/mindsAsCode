"""Uniform-v8 participant reference resolution from the stored mem_trace fields.

Target-only traces already record the parent. This resolver uses that record.
It does not read a gate file and it does not substitute the context best parent
or the maximum-score parent.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from utils.mem.schema_participant_transition_v5 import verify_delta_f_consistency
from utils.mem.trace import record_contains_test_metrics

REFERENCE_POLICY = "pics_v4_uniform_v8"
POPULATION_PROGRAM = "population_program"
SEED_BASELINE = "seed_baseline"
PROMPTED_PARENT = "best_prompted_parent"
SEED_IDS = frozenset({"baseline", "global_baseline"})


class PicsV4ReferenceError(ValueError):
    """The stored uniform-v8 reference cannot be verified."""


def source_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def uniform_v8_resume_key(
    *,
    dataset: Any,
    job_id: Any,
    participant_id: Any,
    phase: Any,
    source: Any,
    iteration: Any,
    candidate_id: Any,
    candidate_sha256: Any,
    reference_type: Any,
    reference_sha256: Any,
    reference_id: Any = "",
    run_id: Any = "",
) -> Tuple[str, ...]:
    """Resume identity for a uniform-v8 participant transition.

    Includes phase, source, reference type, job, participant, iteration,
    candidate id, candidate SHA, and reference SHA. Dataset, run, and
    reference id stay in the key so two official jobs cannot collide.
    """
    try:
        iteration_text = str(int(iteration))
    except (TypeError, ValueError):
        iteration_text = str(iteration if iteration is not None else "")
    parts = (
        str(dataset or ""),
        str(job_id or ""),
        str(run_id or ""),
        str(participant_id if participant_id is not None else ""),
        str(phase or ""),
        str(source or ""),
        iteration_text,
        str(candidate_id or ""),
        str(candidate_sha256 or ""),
        str(reference_id or ""),
        str(reference_type or ""),
        str(reference_sha256 or ""),
    )
    if any(part == "" or "|" in part for part in parts):
        raise PicsV4ReferenceError(f"resume key part is empty or contains '|': {parts}")
    return parts


def find_job_dir(start: Path) -> Path:
    """Walk upward to the official job directory that holds the seed program."""
    cur = Path(start).resolve()
    for _ in range(8):
        if (cur / "prompts" / "seed_program.py").is_file() and (
            cur / "TRIAL_PROMPT_POLICY.json"
        ).is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise PicsV4ReferenceError(f"official job directory not found from {start}")


def _population_candidate_path(job_dir: Path, reference_id: str) -> Path:
    match = re.fullmatch(r"global_iteration_(\d+)_candidate_(\d+)", reference_id)
    if match is None:
        raise PicsV4ReferenceError(
            f"population reference id is not global_iteration_N_candidate_K: {reference_id}"
        )
    iteration, index = int(match.group(1)), int(match.group(2))
    return (
        job_dir
        / "target_population"
        / "control"
        / "global_phase"
        / f"iteration_{iteration}"
        / "candidates"
        / f"candidate_{index}.py"
    )


def _participant_candidate_path(participant_dir: Path, reference_id: str) -> Path:
    explore = re.fullmatch(r"explore_candidate_(\d+)", reference_id)
    if explore is not None:
        return participant_dir / "explore_phase" / "candidates" / f"candidate_{explore.group(1)}.py"
    evolution = re.fullmatch(r"iteration_(\d+)_candidate_(\d+)", reference_id)
    if evolution is not None:
        return (
            participant_dir
            / f"iteration_{int(evolution.group(1))}"
            / "candidates"
            / f"candidate_{int(evolution.group(2))}.py"
        )
    raise PicsV4ReferenceError(f"prompted parent id has no official artifact path: {reference_id}")


def _read_under(root: Path, path: Path) -> str:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise PicsV4ReferenceError(f"reference path escapes the job directory: {path}")
    if not resolved.is_file():
        raise PicsV4ReferenceError(f"reference artifact missing: {path}")
    return resolved.read_text(encoding="utf-8")


def resolve_uniform_v8_reference(
    rec: Mapping[str, Any],
    *,
    participant_dir: Path,
    job_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve one candidate's stored reference. Gate and score fallbacks are refused."""
    if record_contains_test_metrics(dict(rec)):
        raise PicsV4ReferenceError("passive test fields are on the candidate record")
    if rec.get("reference_is_proxy") is True:
        raise PicsV4ReferenceError("proxy reference is refused")

    reference_id = str(rec.get("reference_id") or rec.get("reference_parent_id") or "").strip()
    reference_type = str(rec.get("reference_type") or rec.get("reference_kind") or "").strip()
    if not reference_id or not reference_type:
        raise PicsV4ReferenceError("stored reference_id and reference_type are required")

    phase = str(rec.get("phase") or "")
    source = str(rec.get("source") or "")
    if phase == "explore":
        if reference_type != POPULATION_PROGRAM:
            raise PicsV4ReferenceError(
                f"explore reference_type is {reference_type!r}, not {POPULATION_PROGRAM}"
            )
        mode = "uniform_v8_stored_population_program"
    elif phase == "evolution" and source == "fresh":
        if reference_id not in SEED_IDS or reference_type != SEED_BASELINE:
            raise PicsV4ReferenceError(
                "fresh evolution requires stored baseline/seed_baseline, "
                f"got id={reference_id!r} type={reference_type!r}"
            )
        mode = "uniform_v8_stored_seed_baseline"
    elif phase == "evolution" and source == "normal":
        if reference_type != PROMPTED_PARENT:
            raise PicsV4ReferenceError(
                f"evolution-normal reference_type is {reference_type!r}, not {PROMPTED_PARENT}"
            )
        mode = "uniform_v8_stored_prompted_parent"
    else:
        raise PicsV4ReferenceError(f"unsupported phase/source {phase!r}/{source!r}")

    reference_score = rec.get("reference_score")
    if reference_score is None:
        reference_score = rec.get("reference_parent_score")
    ok, err = verify_delta_f_consistency(
        delta_f=rec.get("delta_f"),
        candidate_score=rec.get("selection_score"),
        reference_score=reference_score,
    )
    if not ok:
        raise PicsV4ReferenceError(err)

    participant_dir = Path(participant_dir)
    job_dir = Path(job_dir) if job_dir is not None else find_job_dir(participant_dir)
    code_rel = str(rec.get("code_path") or "")
    if not code_rel:
        raise PicsV4ReferenceError("candidate code_path is missing")
    candidate_path = (participant_dir / code_rel).resolve()
    if participant_dir.resolve() not in candidate_path.parents:
        raise PicsV4ReferenceError(f"candidate path escapes the participant directory: {code_rel}")
    if not candidate_path.is_file():
        raise PicsV4ReferenceError(f"candidate artifact missing: {candidate_path}")
    candidate_text = candidate_path.read_text(encoding="utf-8")
    candidate_sha = source_sha256(candidate_text)

    if mode == "uniform_v8_stored_seed_baseline":
        reference_path = job_dir / "prompts" / "seed_program.py"
        reference_text = _read_under(job_dir, reference_path)
    elif mode == "uniform_v8_stored_population_program":
        reference_path = _population_candidate_path(job_dir, reference_id)
        reference_text = _read_under(job_dir, reference_path)
        for copy in (participant_dir / "initial_pool_from_global").glob(f"*_{reference_id}.py"):
            copy_text = copy.read_text(encoding="utf-8")
            if source_sha256(copy_text) != source_sha256(reference_text):
                raise PicsV4ReferenceError(
                    f"participant copy SHA disagrees with the population artifact: {copy}"
                )
    else:
        reference_path = _participant_candidate_path(participant_dir, reference_id)
        reference_text = _read_under(job_dir, reference_path)
    reference_sha = source_sha256(reference_text)

    job_token = job_dir.name
    if not job_token.startswith("job_"):
        raise PicsV4ReferenceError(f"job directory is not job_<id>: {job_dir}")
    resume = uniform_v8_resume_key(
        dataset=rec.get("dataset"),
        job_id=job_token,
        run_id=rec.get("run_id"),
        participant_id=rec.get("participant_id"),
        phase=phase,
        source=source,
        iteration=rec.get("iteration"),
        candidate_id=rec.get("candidate_id"),
        candidate_sha256=candidate_sha,
        reference_id=reference_id,
        reference_type=reference_type,
        reference_sha256=reference_sha,
    )
    return {
        "ok": True,
        "resolution_mode": mode,
        "reference_policy": REFERENCE_POLICY,
        "reference_id": reference_id,
        "reference_type": reference_type,
        "reference_code": reference_text,
        "reference_code_path": str(reference_path),
        "reference_sha256": reference_sha,
        "candidate_code": candidate_text,
        "candidate_code_path": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "reference_score": reference_score,
        "job_id": job_token,
        "job_dir": str(job_dir),
        "resume_key": resume,
        "gate_used": False,
        "fallback_used": False,
    }
