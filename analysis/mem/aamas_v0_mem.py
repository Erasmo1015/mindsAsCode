#!/usr/bin/env python3
"""AAMAS v0 discovery and provenance adapters for MEM.

Scientific definitions stay in the existing fitters:
  * population source selection is five-construct Occurrence-EB on iterations 6–10
  * participant MEM is transition-v5 with separate add / modify / remove directions

This module only chooses which artifacts those fitters may see, and it fails
closed when provenance is incomplete or does not match the AAMAS v0 method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.mem.schema_participant_transition_v5 import (  # noqa: E402
    annotation_resume_key,
)
from utils.mem.schema_population_motif_v5 import (  # noqa: E402
    BEHAVIORAL_MOTIFS_V5,
    resume_key as population_resume_key,
)
from utils.mem.trace import (  # noqa: E402
    iter_jsonl_records,
    record_contains_test_metrics,
)
from utils.teh.pics_aamas_v0 import (  # noqa: E402
    CANONICAL_DATASETS,
    EXPLORE_CANDIDATES,
    GLOBAL_ITERS,
    KIND_OFFICIAL_GATE,
    KIND_TARGET_ONLY,
    N_CANDIDATES,
    POPULATION_STAGE,
    PROMPT_POLICY_ID,
    TRACK_OFFICIAL_GATE,
    TRACK_TARGET_ONLY,
    _FORBIDDEN_PATH_MARKERS,
    assert_aamas_population_path,
    assert_aamas_track_output,
    kind_for_track,
)
from utils.teh.t_pics_gated_transfer import program_has_valid_choose  # noqa: E402

CONSTRUCTS: Tuple[str, ...] = tuple(BEHAVIORAL_MOTIFS_V5)
FIT_ITERATION_MIN = 6
FIT_ITERATION_MAX = 10
SOURCE_SELECTOR_NAME = "pics_aamas_v0_source_map"
AAMAS_MEM_ROOT = (
    _REPO / "analysis_2026Sep/Sep27_aamas_v0/others/aamas_v0/mem"
)
SOURCE_MAP_PATH = (
    _REPO
    / "analysis_2026Sep/Sep27_aamas_v0/others/aamas_v0/source_map"
    / "pics_aamas_v0_source_map.yaml"
)
_SCORE_COLUMNS = frozenset(
    {
        "selection_score",
        "delta_f",
        "global_fitness",
        "train_loglik",
        "val_loglik",
        "pooled_train_val_score",
        "reference_score",
        "reference_parent_score",
    }
)


class AamasMemProvenanceError(RuntimeError):
    """Incomplete or mismatched AAMAS v0 MEM provenance."""


def track_provenance(track_mode: str) -> Dict[str, Any]:
    """Isolate a track. Only official_gate is the paper participant MEM."""
    kind = kind_for_track(track_mode)
    paper = track_mode == TRACK_OFFICIAL_GATE
    return {
        "track_mode": str(track_mode),
        "kind": kind,
        "provisional_experimental_track": not paper,
        "paper_mem": paper,
        "official_main_result": paper,
    }


def scoped_run_id(track_mode: str, job_id: str) -> str:
    """Resume run id that cannot collide across tracks or packed datasets.

    On-disk traces store generic run ids (``control``, ``selected``). Those
    collide when several datasets share a Slurm job. The scientific resume
    function is unchanged; this adapter fills its run-id field with a scoped id.
    """
    kind_for_track(track_mode)
    job = str(job_id).strip()
    if not job.startswith("job_"):
        job = f"job_{job}"
    if not job[4:].isdigit():
        raise AamasMemProvenanceError(f"job id is not job_<digits>: {job_id!r}")
    return f"{track_mode}__{job}"


def _posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AamasMemProvenanceError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AamasMemProvenanceError(f"JSON object required: {path}")
    return payload


def _guard_path(check, path: Path) -> None:
    try:
        check(path)
    except RuntimeError as exc:
        raise AamasMemProvenanceError(str(exc)) from exc


def _reject_forbidden(path: Path) -> None:
    text = _posix(Path(path).resolve())
    for marker in _FORBIDDEN_PATH_MARKERS:
        if marker in text:
            raise AamasMemProvenanceError(
                f"refusing historical or superseded path marker {marker!r}: {text}"
            )


def _job_id_from_path(path: Path) -> str:
    for part in Path(path).resolve().parts:
        if part.startswith("job_") and part[4:].isdigit():
            return part
    raise AamasMemProvenanceError(f"no job_<id> in path: {path}")


def _repo_relative(path: Path, repo: Path) -> str:
    """Repo-relative path without resolving symlinks.

    ``generated_outputs`` is a symlink out of the checkout. Annotation loads
    ``repo / code_path``, so the stored path must keep that symlink prefix.
    """
    raw = Path(path)
    if not raw.is_absolute():
        raw = Path(repo) / raw
    try:
        return _posix(raw.relative_to(Path(repo)))
    except ValueError as exc:
        raise AamasMemProvenanceError(
            f"path is outside the repo and cannot be a MEM input: {path}"
        ) from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_target_only_population(arm: Path) -> Dict[str, Any]:
    """Fail closed unless ``arm`` is a completed target-only population."""
    arm = Path(arm)
    _reject_forbidden(arm)
    _guard_path(assert_aamas_population_path, arm)
    text = _posix(arm.resolve())
    if "/selected/" in text or "/participant_" in text:
        raise AamasMemProvenanceError(
            "source selection refuses participant programs"
        )
    if "/target_population/transfer" in text:
        raise AamasMemProvenanceError("source selection refuses transfer populations")
    if f"/{KIND_OFFICIAL_GATE}/" in text or "/pics_aamas_v0_transfer_based_only/" in text:
        raise AamasMemProvenanceError(
            "source selection reads completed target_only populations only"
        )
    prov_path = arm / "POPULATION_PROVENANCE.json"
    stage_path = arm / "STAGE_COMPLETE.json"
    if not prov_path.is_file() or not stage_path.is_file():
        raise AamasMemProvenanceError(f"incomplete population provenance: {arm}")
    prov = _read_json(prov_path)
    marker = _read_json(stage_path)
    if marker.get("stage") != POPULATION_STAGE:
        raise AamasMemProvenanceError(f"bad completion stage: {marker.get('stage')!r}")
    if marker.get("kind") != KIND_TARGET_ONLY or prov.get("kind") != KIND_TARGET_ONLY:
        raise AamasMemProvenanceError("population kind is not target_only")
    if str(prov.get("track_mode") or "") != TRACK_TARGET_ONLY:
        raise AamasMemProvenanceError("population track_mode is not target_only")
    if str(marker.get("track_mode") or "") not in {"", TRACK_TARGET_ONLY}:
        raise AamasMemProvenanceError("completion track_mode is not target_only")
    if (
        marker.get("prompt_policy") != PROMPT_POLICY_ID
        or prov.get("prompt_policy") != PROMPT_POLICY_ID
    ):
        raise AamasMemProvenanceError("population prompt policy mismatch")
    if prov.get("prompt_mode") != "aamas_v0_registered":
        raise AamasMemProvenanceError(
            f"population prompt_mode={prov.get('prompt_mode')!r}"
        )
    dataset = str(prov.get("dataset") or "")
    if not dataset or str(marker.get("dataset") or "") != dataset:
        raise AamasMemProvenanceError("population dataset key mismatch")
    if str(prov.get("limited_data_protocol") or "") != "structure_aware_v3":
        raise AamasMemProvenanceError("population protocol mismatch")
    if int(prov.get("limited_train_val") or 0) != 40:
        raise AamasMemProvenanceError("population SA40 budget mismatch")
    fingerprint = prov.get("sa40_fingerprint")
    if not fingerprint or fingerprint != marker.get("sa40_fingerprint"):
        raise AamasMemProvenanceError("population fingerprint missing or mismatched")
    if int(prov.get("global_iters") or 0) != GLOBAL_ITERS or int(
        marker.get("global_iters") or 0
    ) != GLOBAL_ITERS:
        raise AamasMemProvenanceError("population iteration budget mismatch")
    if int(prov.get("n_candidates") or 0) != N_CANDIDATES or int(
        marker.get("n_candidates") or 0
    ) != N_CANDIDATES:
        raise AamasMemProvenanceError("population candidate budget mismatch")
    if prov.get("runtime_valid") is not True or prov.get("test_used_for_selection") is not False:
        raise AamasMemProvenanceError(
            "population rank-1 is not a completed train/val selection"
        )
    if str(prov.get("evolution_selection_score") or "") != "train_val":
        raise AamasMemProvenanceError("population selection score is not train_val")
    if prov.get("arm_role") not in (None, "target_only"):
        raise AamasMemProvenanceError("refusing a non-target-only arm")
    if prov.get("provisional_experimental_track") is not True:
        raise AamasMemProvenanceError("target_only population must stay provisional")
    rank1 = Path(str(prov.get("rank1_program") or ""))
    if not rank1.is_file():
        raise AamasMemProvenanceError(f"rank-1 missing: {rank1}")
    _reject_forbidden(rank1)
    _guard_path(assert_aamas_population_path, rank1)
    digest = _sha256(rank1)
    if digest != prov.get("rank1_sha256") or digest != marker.get("rank1_sha256"):
        raise AamasMemProvenanceError("rank-1 SHA mismatch")
    valid, reason = program_has_valid_choose(rank1)
    if not valid:
        raise AamasMemProvenanceError(f"rank-1 choose() invalid ({reason}): {rank1}")
    job_dir = arm.parent.parent
    job_id = _job_id_from_path(job_dir)
    if job_dir.name != job_id:
        raise AamasMemProvenanceError(f"arm is not directly under {job_id}: {arm}")
    return {
        "dataset": dataset,
        "job_id": job_id,
        "job_dir": job_dir,
        "arm": arm,
        "scoped_run_id": scoped_run_id(TRACK_TARGET_ONLY, job_id),
        "sa40_fingerprint": fingerprint,
        "track_mode": TRACK_TARGET_ONLY,
        "provisional_experimental_track": True,
    }


def _parent_code_path(
    job_dir: Path,
    global_phase: Path,
    parent_id: str,
    by_candidate_id: Mapping[str, Mapping[str, Any]],
) -> Path:
    if parent_id in {"global_baseline", "baseline"}:
        seed = job_dir / "prompts" / "seed_program.py"
        if not seed.is_file():
            raise AamasMemProvenanceError(f"missing neutral seed: {seed}")
        return seed
    parent = by_candidate_id.get(parent_id)
    if parent is not None:
        code = global_phase / str(parent.get("code_path") or "")
        if code.is_file():
            return code
    elite = global_phase / "global_elite_pool"
    matches = sorted(elite.glob(f"*_{parent_id}.py")) if elite.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    raise AamasMemProvenanceError(f"parent program unresolved: {parent_id}")


def population_candidates(arm: Path, meta: Mapping[str, Any], repo: Path) -> List[Dict[str, Any]]:
    """Every completed target-only candidate. No scores, no participant files."""
    global_phase = Path(arm) / "global_phase"
    trace_path = global_phase / "mem_trace.jsonl"
    if not trace_path.is_file():
        raise AamasMemProvenanceError(f"missing population trace: {trace_path}")
    by_candidate_id: Dict[str, Dict[str, Any]] = {}
    raw: List[Dict[str, Any]] = []
    for rec in iter_jsonl_records([trace_path]):
        if rec.get("record_type") != "candidate":
            continue
        if record_contains_test_metrics(rec):
            raise AamasMemProvenanceError("refusing test metrics in a population trace")
        if rec.get("phase") != "global_evolution":
            raise AamasMemProvenanceError(
                f"population trace phase is not global_evolution: {rec.get('phase')!r}"
            )
        if str(rec.get("participant_id")) != "global":
            raise AamasMemProvenanceError("population trace contains a participant program")
        if str(rec.get("evolution_selection_score") or "") != "train_val":
            raise AamasMemProvenanceError("population candidate was not train_val selected")
        if str(rec.get("dataset") or "") != meta["dataset"]:
            raise AamasMemProvenanceError("population trace dataset mismatch")
        code = global_phase / str(rec.get("code_path") or "")
        if not code.is_file() or global_phase.resolve() not in code.resolve().parents:
            raise AamasMemProvenanceError(f"candidate file missing under global_phase: {code}")
        text = _posix(code.resolve())
        if "/selected/" in text or "/target_population/transfer" in text:
            raise AamasMemProvenanceError(f"candidate path is not a target-only program: {code}")
        raw.append(rec)
        by_candidate_id[str(rec.get("candidate_id"))] = rec

    on_disk = {p.resolve() for p in global_phase.glob("iteration_*/candidates/candidate_*.py")}
    traced = {
        (global_phase / str(rec.get("code_path") or "")).resolve() for rec in raw
    }
    if traced != on_disk:
        raise AamasMemProvenanceError(
            "population candidate files and mem_trace.jsonl do not match "
            f"(files={len(on_disk)} trace={len(traced)})"
        )
    seen: set[str] = set()
    programs: List[Dict[str, Any]] = []
    for rec in raw:
        code = global_phase / str(rec["code_path"])
        iteration = int(rec["iteration"])
        parent = str(rec.get("reference_id") or rec.get("reference_parent_id") or "")
        if not parent:
            raise AamasMemProvenanceError(f"candidate missing reference: {code}")
        parent_path = _parent_code_path(
            Path(meta["job_dir"]), global_phase, parent, by_candidate_id
        )
        parent_text = _posix(parent_path.resolve())
        if "/selected/" in parent_text or "/target_population/transfer" in parent_text:
            raise AamasMemProvenanceError(f"parent path is not a target-only program: {parent_path}")
        stem = code.stem
        key = population_resume_key(
            meta["dataset"], meta["scoped_run_id"], iteration, stem, parent
        )
        if key in seen:
            raise AamasMemProvenanceError(f"population resume-key collision: {key}")
        seen.add(key)
        programs.append(
            {
                "dataset": meta["dataset"],
                "run_id": meta["scoped_run_id"],
                "trace_run_id": rec.get("run_id"),
                "source_job_id": meta["job_id"],
                "track_mode": TRACK_TARGET_ONLY,
                "provisional_experimental_track": True,
                "paper_mem": False,
                "iteration": iteration,
                "candidate_id": stem,
                "program_id": str(rec.get("candidate_id")),
                "candidate_idx": rec.get("candidate_idx"),
                "parent": parent,
                "parent_id": parent,
                "code_path": _repo_relative(code, repo),
                "parent_code_path": _repo_relative(parent_path, repo),
                "resume_key": key,
                "schema_version": 5,
                "constructs": list(CONSTRUCTS),
            }
        )
    iterations = {int(p["iteration"]) for p in programs}
    if iterations != set(range(1, GLOBAL_ITERS + 1)):
        raise AamasMemProvenanceError(
            f"population iterations are {sorted(iterations)}; expected 1–{GLOBAL_ITERS}"
        )
    for program in programs:
        leaked = _SCORE_COLUMNS.intersection(program)
        if leaked:
            raise AamasMemProvenanceError(f"score fields leaked into manifest: {sorted(leaked)}")
    return programs


def discover_target_only_populations(
    repo: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Scan canonical datasets. Errors are returned; nothing is skipped silently."""
    repo = Path(repo)
    root = repo / "generated_outputs" / "psych101_train" / "teh"
    found: List[Dict[str, Any]] = []
    errors: List[str] = []
    for dataset in CANONICAL_DATASETS:
        kind_root = root / dataset / KIND_TARGET_ONLY
        jobs = sorted(p for p in kind_root.glob("job_*") if p.is_dir()) if kind_root.is_dir() else []
        if not jobs:
            errors.append(f"{dataset}: no {KIND_TARGET_ONLY} job")
            continue
        completes: List[Dict[str, Any]] = []
        for job in jobs:
            arm = job / "target_population" / "control"
            try:
                meta = validate_target_only_population(arm)
                if meta["dataset"] != dataset:
                    raise AamasMemProvenanceError(
                        f"path dataset {dataset} != provenance {meta['dataset']}"
                    )
                programs = population_candidates(arm, meta, repo)
                completes.append({**meta, "n_programs": len(programs), "programs": programs})
            except AamasMemProvenanceError as exc:
                errors.append(f"{dataset} {job.name}: {exc}")
        if len(completes) > 1:
            errors.append(
                f"{dataset}: {len(completes)} completed target_only populations; "
                "refusing to guess a job id"
            )
        elif len(completes) == 1:
            found.append(completes[0])
    return found, errors


def population_manifest(found: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    programs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for job in found:
        for program in job["programs"]:
            key = str(program["resume_key"])
            if key in seen:
                raise AamasMemProvenanceError(f"cross-dataset resume-key collision: {key}")
            seen.add(key)
            programs.append(dict(program))
    return {
        "schema": "pics_aamas_v0_population_annotation_manifest_v1",
        "schema_version": 5,
        "constructs": list(CONSTRUCTS),
        "fit_iterations": [FIT_ITERATION_MIN, FIT_ITERATION_MAX],
        "track_mode": TRACK_TARGET_ONLY,
        "uses_test_scores": False,
        "n_datasets": len(found),
        "n_programs": len(programs),
        "programs": programs,
    }


def occurrence_panel_from_annotations(
    records: Iterable[Mapping[str, Any]],
    manifest_programs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Presence rows for Occurrence-EB. Iterations 6–10 only. No scores."""
    by_key = {str(p["resume_key"]): p for p in manifest_programs}
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rec in records:
        if record_contains_test_metrics(dict(rec)):
            raise AamasMemProvenanceError("annotation record contains a test metric")
        key = str(rec.get("resume_key") or "")
        program = by_key.get(key)
        if program is None:
            raise AamasMemProvenanceError(f"annotation resume key is not in the target_only manifest: {key}")
        iteration = int(rec.get("iteration"))
        if iteration != int(program["iteration"]):
            raise AamasMemProvenanceError(f"annotation iteration mismatch for {key}")
        if iteration < FIT_ITERATION_MIN or iteration > FIT_ITERATION_MAX:
            continue
        if key in seen:
            raise AamasMemProvenanceError(f"duplicate occurrence row: {key}")
        seen.add(key)
        state = rec.get("candidate_motif_state") or rec.get("program_motif_state") or []
        if not isinstance(state, list):
            raise AamasMemProvenanceError(f"motif state is not a list: {key}")
        unknown = [m for m in state if m not in CONSTRUCTS]
        if unknown:
            raise AamasMemProvenanceError(f"unexpected constructs {unknown} in {key}")
        row = {
            "dataset": program["dataset"],
            "run_id": program["run_id"],
            "iteration": iteration,
            "program_id": program["program_id"],
            "parent_id": program["parent_id"],
            "resume_key": key,
        }
        for construct in CONSTRUCTS:
            row[construct] = int(construct in state)
        rows.append(row)
    if not rows:
        raise AamasMemProvenanceError(
            f"no occurrence rows in iterations {FIT_ITERATION_MIN}–{FIT_ITERATION_MAX}"
        )
    return rows


def fit_occurrence_source_identities(
    panel_rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str],
    allowlist: Sequence[str],
) -> List[Dict[str, Any]]:
    """Call the existing Occurrence-EB fitter. Does not read scores."""
    import pandas as pd

    from analysis.mem.pop_v4_source_selection import run_source_selection_v4 as rss

    frame = pd.DataFrame(list(panel_rows))
    leaked = [
        col
        for col in frame.columns
        if col in _SCORE_COLUMNS or "test" in str(col).lower()
    ]
    if leaked:
        raise AamasMemProvenanceError(
            f"Occurrence-EB panel contains score columns: {sorted(leaked)}"
        )
    missing = [c for c in CONSTRUCTS if c not in frame.columns]
    if missing:
        raise AamasMemProvenanceError(f"panel missing constructs: {missing}")
    if frame.empty:
        raise AamasMemProvenanceError("empty occurrence panel")
    iterations = {int(v) for v in frame["iteration"].tolist()}
    if iterations - set(range(FIT_ITERATION_MIN, FIT_ITERATION_MAX + 1)):
        raise AamasMemProvenanceError(
            f"panel iterations outside {FIT_ITERATION_MIN}–{FIT_ITERATION_MAX}: {sorted(iterations)}"
        )
    previous = rss.MOTIFS
    rss.MOTIFS = CONSTRUCTS
    try:
        sig_df, _eb, _model, _meta = rss.fit_occurrence_eb(frame)
        vectors = rss.signature_vectors(sig_df, CONSTRUCTS)
        ranks = rss.select_sources(vectors, targets=list(targets), allowlist=list(allowlist))
    finally:
        rss.MOTIFS = previous
    return ranks.to_dict(orient="records")


def source_map_document(
    ranks: Sequence[Mapping[str, Any]],
    job_id_by_dataset: Mapping[str, str],
) -> Dict[str, Any]:
    """Identity map only. No program paths and no scores."""
    sources: Dict[str, str] = {}
    targets: Dict[str, Dict[str, str]] = {}
    for row in ranks:
        target = str(row["target_id"])
        source = str(row["selected_source_id"])
        if source == target:
            raise AamasMemProvenanceError(f"refusing self-source for {target}")
        if target in sources:
            raise AamasMemProvenanceError(f"duplicate source-map row for {target}")
        sources[target] = source
        targets[target] = {"selected_source": source}
    if set(sources) != set(CANONICAL_DATASETS):
        missing = sorted(set(CANONICAL_DATASETS) - set(sources))
        raise AamasMemProvenanceError(f"source map does not cover every dataset: {missing}")
    jobs: Dict[str, str] = {}
    for source in sorted(set(sources.values())):
        job_id = str(job_id_by_dataset.get(source) or "")
        if not job_id.startswith("job_") and not job_id.isdigit():
            raise AamasMemProvenanceError(f"missing target_only job id for source {source}")
        jobs[source] = job_id[4:] if job_id.startswith("job_") else job_id
    return {
        "schema": "pics_aamas_v0_source_job_manifest_v1",
        "selector": {"name": SOURCE_SELECTOR_NAME},
        "sources": sources,
        "targets": targets,
        "jobs": jobs,
    }


def assert_aamas_participant_annotation_root(run_dir: Path, *, track_mode: str) -> None:
    """Participant annotation root must be that track's ``selected`` tree."""
    meta = track_provenance(track_mode)
    run_dir = Path(run_dir)
    _reject_forbidden(run_dir)
    _guard_path(lambda path: assert_aamas_track_output(path, meta["kind"]), run_dir)
    text = _posix(run_dir.resolve())
    if "/target_population/" in text or "/global_phase/" in text:
        raise AamasMemProvenanceError(
            "participant MEM refuses population traces"
        )
    if "/selected/" not in text and not text.rstrip("/").endswith("/selected"):
        raise AamasMemProvenanceError(
            f"participant MEM root is not under selected/: {run_dir}"
        )
    if track_mode == TRACK_OFFICIAL_GATE:
        _require_official_gate_record(run_dir)


def _job_dir_from_selected(path: Path) -> Path:
    resolved = Path(path).resolve()
    parts = list(resolved.parts)
    if "selected" not in parts:
        raise AamasMemProvenanceError(f"no selected/ in {path}")
    return Path(*parts[: parts.index("selected")])


def _require_official_gate_record(path: Path) -> None:
    job_dir = _job_dir_from_selected(path)
    arm_path = job_dir / "selected" / "SELECTED_ARM.txt"
    gate_path = job_dir / "gate" / "gate_record.json"
    if not arm_path.is_file() or not gate_path.is_file():
        raise AamasMemProvenanceError(
            f"official_gate participant MEM missing gate provenance under {job_dir}"
        )
    arm = arm_path.read_text(encoding="utf-8").strip()
    if arm not in {"target_only", "transfer"}:
        raise AamasMemProvenanceError(f"SELECTED_ARM.txt is {arm!r}")
    gate = _read_json(gate_path)
    if gate.get("test_used_for_gate") is not False:
        raise AamasMemProvenanceError("official gate used test labels")


def participant_trace_files(trace_files: Sequence[Path]) -> List[Path]:
    kept: List[Path] = []
    for path in trace_files:
        text = _posix(Path(path).resolve())
        if "/target_population/" in text or "/global_phase/" in text:
            raise AamasMemProvenanceError(f"refusing population trace: {path}")
        if "/selected/participant_" not in text or Path(path).name != "mem_trace.jsonl":
            raise AamasMemProvenanceError(f"not a participant trace: {path}")
        kept.append(Path(path))
    return kept


def assess_participant(
    participant_dir: Path,
    *,
    track_mode: str,
    dataset: str,
    job_id: str,
) -> Dict[str, Any]:
    """One ``selected/participant_*`` tree. Incomplete participants raise."""
    participant_dir = Path(participant_dir)
    meta = track_provenance(track_mode)
    assert_aamas_participant_annotation_root(participant_dir, track_mode=track_mode)
    if not participant_dir.name.startswith("participant_"):
        raise AamasMemProvenanceError(f"not a participant directory: {participant_dir}")
    metrics_path = participant_dir / "explore_phase" / "metrics.json"
    if not metrics_path.is_file():
        raise AamasMemProvenanceError(f"missing explore metrics: {metrics_path}")
    metrics = _read_json(metrics_path)
    if int(metrics.get("explore_candidates_requested") or 0) != EXPLORE_CANDIDATES:
        raise AamasMemProvenanceError(
            f"explore request is {metrics.get('explore_candidates_requested')!r}, "
            f"expected {EXPLORE_CANDIDATES}"
        )
    if str(metrics.get("evolution_selection_score") or "") != "train_val":
        raise AamasMemProvenanceError("participant selection score is not train_val")
    if not (participant_dir / "iteration_10").is_dir():
        raise AamasMemProvenanceError(f"missing iteration_10: {participant_dir}")
    best = participant_dir / "best_program.py"
    valid, reason = program_has_valid_choose(best)
    if not valid:
        raise AamasMemProvenanceError(f"participant choose() invalid ({reason}): {best}")
    trace_path = participant_dir / "mem_trace.jsonl"
    if not trace_path.is_file():
        raise AamasMemProvenanceError(f"missing participant trace: {trace_path}")
    scoped = scoped_run_id(track_mode, job_id)
    counts: Counter = Counter()
    seen: set[Tuple[Any, ...]] = set()
    saw_iter10 = False
    for rec in iter_jsonl_records([trace_path]):
        if rec.get("record_type") != "candidate":
            continue
        if record_contains_test_metrics(rec):
            raise AamasMemProvenanceError("refusing test metrics in a participant trace")
        phase = str(rec.get("phase") or "")
        if phase not in {"explore", "evolution"}:
            raise AamasMemProvenanceError(f"unexpected participant phase {phase!r}")
        if str(rec.get("participant_id")) in {"", "global"}:
            raise AamasMemProvenanceError("participant trace is not person-level")
        if str(rec.get("dataset") or "") != dataset:
            raise AamasMemProvenanceError("participant trace dataset mismatch")
        if phase == "evolution" and int(rec.get("iteration") or 0) == 10:
            saw_iter10 = True
        key = annotation_resume_key(
            dataset,
            scoped,
            rec.get("participant_id"),
            rec.get("iteration"),
            str(rec.get("candidate_id")),
            reference_id=rec.get("reference_id") or rec.get("reference_parent_id"),
            reference_type=rec.get("reference_type") or rec.get("reference_kind"),
            phase=phase,
        )
        if key in seen:
            raise AamasMemProvenanceError(f"participant resume-key collision: {key}")
        seen.add(key)
        counts[(phase, str(rec.get("source") or ""))] += 1
    if counts.get(("explore", "explore"), 0) < 1:
        raise AamasMemProvenanceError(f"no explore candidates in {participant_dir}")
    if not saw_iter10:
        raise AamasMemProvenanceError(f"no iteration-10 evolution candidate in {participant_dir}")
    return {
        "dataset": dataset,
        "participant_dir": str(participant_dir),
        "job_id": job_id if str(job_id).startswith("job_") else f"job_{job_id}",
        "scoped_run_id": scoped,
        "n_candidates": int(sum(counts.values())),
        "n_explore": int(counts.get(("explore", "explore"), 0)),
        "n_evolution_normal": int(counts.get(("evolution", "normal"), 0)),
        "n_evolution_fresh": int(counts.get(("evolution", "fresh"), 0)),
        **meta,
    }


def participant_output_dir(track_mode: str, dataset: str, job_id: str) -> Path:
    job = job_id if str(job_id).startswith("job_") else f"job_{job_id}"
    return AAMAS_MEM_ROOT / "participant" / track_mode / dataset / job


def write_track_marker(directory: Path, track_mode: str) -> None:
    """Refuse to reuse a directory stamped for a different track."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "TRACK_MODE.json"
    payload = track_provenance(track_mode)
    if marker.is_file():
        existing = _read_json(marker)
        if existing.get("track_mode") != payload["track_mode"] or existing.get(
            "paper_mem"
        ) != payload["paper_mem"]:
            raise AamasMemProvenanceError(
                f"output directory track_mode mismatch: {marker}"
            )
        return
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _print_population_report(found: Sequence[Mapping[str, Any]], errors: Sequence[str]) -> None:
    print(
        f"target_only populations complete={len(found)}/{len(CANONICAL_DATASETS)} "
        f"errors={len(errors)}"
    )
    for job in found:
        n_fit = sum(
            1
            for program in job["programs"]
            if FIT_ITERATION_MIN <= int(program["iteration"]) <= FIT_ITERATION_MAX
        )
        print(
            f"  OK {job['dataset']} {job['job_id']} programs={job['n_programs']} "
            f"fit_iters_{FIT_ITERATION_MIN}_{FIT_ITERATION_MAX}={n_fit} "
            f"run_id={job['scoped_run_id']}"
        )
    for err in errors:
        print(f"  MISSING {err}")


def _cmd_population_discover(repo: Path, *, write_manifest: bool) -> int:
    found, errors = discover_target_only_populations(repo)
    _print_population_report(found, errors)
    ready = len(found) == len(CANONICAL_DATASETS) and not errors
    if write_manifest:
        if not ready:
            raise AamasMemProvenanceError(
                "refusing to write a population manifest from an incomplete source bank"
            )
        payload = population_manifest(found)
        out = AAMAS_MEM_ROOT / "population" / "program_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0 if ready else 2


def _cmd_participant_discover(repo: Path, track_mode: str) -> int:
    meta = track_provenance(track_mode)
    root = Path(repo) / "generated_outputs" / "psych101_train" / "teh"
    n_ready = 0
    n_incomplete = 0
    print(
        f"participant track_mode={track_mode} paper_mem={meta['paper_mem']} "
        f"provisional={meta['provisional_experimental_track']}"
    )
    for dataset in CANONICAL_DATASETS:
        kind_root = root / dataset / meta["kind"]
        jobs = sorted(p for p in kind_root.glob("job_*") if p.is_dir()) if kind_root.is_dir() else []
        if not jobs:
            print(f"  MISSING {dataset}: no {meta['kind']} job")
            n_incomplete += 1
            continue
        for job in jobs:
            selected = job / "selected"
            if track_mode == TRACK_OFFICIAL_GATE:
                try:
                    _require_official_gate_record(selected)
                except AamasMemProvenanceError as exc:
                    print(f"  MISSING {dataset} {job.name}: {exc}")
                    n_incomplete += 1
                    continue
            parts = sorted(
                p for p in selected.glob("participant_*") if p.is_dir()
            ) if selected.is_dir() else []
            ok = 0
            for part in parts:
                try:
                    assess_participant(
                        part,
                        track_mode=track_mode,
                        dataset=dataset,
                        job_id=job.name,
                    )
                    ok += 1
                except AamasMemProvenanceError:
                    n_incomplete += 1
            dataset_done = (selected / "STAGE_COMPLETE.json").is_file()
            print(
                f"  {dataset} {job.name} complete_participants={ok} "
                f"dirs={len(parts)} dataset_stage={dataset_done}"
            )
            if dataset_done and ok == len(parts) and ok > 0:
                n_ready += 1
            else:
                n_incomplete += 1
    print(f"participant datasets_ready={n_ready} incomplete_flags={n_incomplete}")
    if track_mode == TRACK_OFFICIAL_GATE and n_ready != len(CANONICAL_DATASETS):
        print("paper participant MEM is not ready: official_gate bank is incomplete")
        return 2
    if track_mode != TRACK_OFFICIAL_GATE:
        print("target-only and transfer participant MEM stay provisional")
    return 0 if n_incomplete == 0 else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=str, default=str(_REPO))
    sub = parser.add_subparsers(dest="cmd", required=True)
    pop = sub.add_parser("population-discover")
    pop.add_argument("--write-manifest", action="store_true")
    person = sub.add_parser("participant-discover")
    person.add_argument("--track-mode", required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "population-discover":
            return _cmd_population_discover(
                Path(args.repo), write_manifest=bool(args.write_manifest)
            )
        return _cmd_participant_discover(Path(args.repo), str(args.track_mode))
    except AamasMemProvenanceError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
