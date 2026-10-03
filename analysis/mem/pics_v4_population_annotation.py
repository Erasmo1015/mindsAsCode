#!/usr/bin/env python3
"""PICS v4 population-program annotation inventory.

Reuses schema-v5 transition annotation (history, value, probability_used,
feedback, learning) via analysis/mem/annotate_population_programs.py.
This module only decides which finished target-only population candidates
that annotator may see. It does not start an LLM.

A dataset is included only when its population stage is complete and its
marker matches the live PICS v4 policy. Partial traces are not read.
Exact-SHA elite deduplication does not drop candidates from the inventory.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.mem.schema_population_motif_v5 import (  # noqa: E402
    BEHAVIORAL_MOTIFS_V5,
    PROMPT_VERSION,
    SCHEMA_VERSION,
)
from utils.mem.trace import iter_jsonl_records, record_contains_test_metrics  # noqa: E402
from utils.teh.elite_sha import elite_policy_provenance, source_sha256  # noqa: E402
from utils.teh.pics_v4 import (  # noqa: E402
    ADDITIONAL_PROMPT_POLICY_ID,
    HARD_PROMPT_TOKEN_CAP,
    KIND_TARGET_ONLY,
    METHOD_VERSION,
    PICS_RUN_SEED_SCOPE,
    PICS_V4_TRIAL_POLICY,
    SEARCH_RNG_POLICY,
    assert_pics_v4_resume,
    uniform_additional_prompt_sha256,
)
from utils.teh.pics_v4_panels import (  # noqa: E402
    CONTINUATION_POLICY_ID,
    PACKING_IMPLEMENTATION_ID,
    PANEL_BANK_POLICY_ID,
    PANEL_POLICY_ID,
    PARENT_ENVELOPE_POLICY_ID,
    SLOT_ASSIGNMENT_POLICY_ID,
)

MANIFEST_SCHEMA = "pics_v4_population_annotation_manifest_v1"
POPULATION_PHASE = "global_evolution"
POPULATION_PARTICIPANT = "global"
DOC_REL = "analysis_2026Sep/Sep30_pics_v4/docs/Documentation_v4.md"
HISTORICAL_OUTPUT_REL = "analysis_2026Sep/Sep30_pics_v4/mem/population"
OUTPUT_REL = "analysis_2026Sep/Sep30_pics_v4/mem/population_final"
ANNOTATION_METHOD = "analysis/mem/annotate_population_programs.py"
REQUIRED_POLICY_MARKERS = {
    "trial_prompt_policy": "structured_snapshot_sparse_history_v1",
    "additional_prompt_policy": "uniform_additional_prompt_v5",
    "panel_policy": "shuffled_block_slot_stable_v4",
    "continuation_policy": "within_block_carry_forward_wrap_fill_v2",
}

# Former pack plus latter pack. A group is submitted only for datasets that are ready.
ANNOTATION_GROUPS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        "pics_v4_popann_a100",
        "gpu:a100:4",
        (
            "guan_2020_stopping",
            "11enkavi2019recentprobes",
            "4wulff2018description",
            "7hilbig2014generalized",
            "12badham2017deficits",
        ),
    ),
    (
        "pics_v4_popann_3090",
        "gpu:3090:4",
        (
            "bergert_nosofsky_2007",
            "5speekenbrink2008learning",
            "14kool2016when",
            "steyvers_2009_bandit",
        ),
    ),
    (
        "pics_v4_popann_a5000",
        "gpu:a5000:4",
        ("3frey2017cct", "10frey2017risk", "1peterson2021using"),
    ),
    (
        "pics_v4_popann_a100b",
        "gpu:a100:4",
        ("2plonsky2018when", "mixed_gambles", "13schulz2020finding"),
    ),
)

# Cancelled sparse-history rerun plus earlier full-history jobs. Both sets are refused.
AUTHORITATIVE_JOBS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("311526", ("guan_2020_stopping", "11enkavi2019recentprobes")),
    ("311528", ("bergert_nosofsky_2007", "5speekenbrink2008learning")),
    ("311472", ("3frey2017cct",)),
    ("311473", ("2plonsky2018when",)),
    ("311529", ("4wulff2018description", "7hilbig2014generalized", "12badham2017deficits")),
    ("311531", ("14kool2016when", "steyvers_2009_bandit")),
    ("311532", ("10frey2017risk", "1peterson2021using")),
    ("311533", ("mixed_gambles", "13schulz2020finding")),
)
HISTORICAL_JOB_IDS = frozenset(
    {
        "310886",
        "310887",
        "310888",
        "310889",
        "310899",
        "310931",
        "310932",
        "310933",
        "311207",
        "311208",
        "311199",
        "311200",
        "311209",
        "311210",
        "311203",
        "311204",
        "311470",
        "311471",
        "311474",
        "311475",
        "311476",
        "311478",
    }
)
FINAL_JOB_IDS = frozenset(job_id for job_id, _names in AUTHORITATIVE_JOBS)

_REQUIRED_DOC_TOKENS = (
    "structured_snapshot_sparse_history_v1",
    "uniform_additional_prompt_v5",
    "shuffled_block_slot_stable_v4",
    "within_block_carry_forward_wrap_fill_v2",
    "conditioning_aware_panel_banks_v2",
    "initial_unique_elite_mean_x_v1",
    "sha256_unique_v1",
    "15,360",
    "1,024",
    "16,384",
    "duplicate_backfill",
    "first unseen",
    "311207",
    "311208",
    "311199",
    "311200",
    "311209",
    "311210",
    "311203",
    "311204",
    "311470",
    "311471",
    "311472",
    "311473",
    "311474",
    "311475",
    "311476",
    "311478",
    "311526",
    "311528",
    "311529",
    "311531",
    "311532",
    "311533",
)


class PicsV4AnnotationError(RuntimeError):
    pass


def assert_final_job_mapping(jobs: Sequence[Tuple[str, Sequence[str]]]) -> None:
    """Accept only final-rerun job ids and their own datasets."""
    allowed = {job_id: set(names) for job_id, names in AUTHORITATIVE_JOBS}
    for job_id, names in jobs:
        jid = str(job_id)
        if jid in HISTORICAL_JOB_IDS or jid not in allowed:
            raise PicsV4AnnotationError(f"refusing non-final population job {jid}")
        unknown = [name for name in names if name not in allowed[jid]]
        if unknown:
            raise PicsV4AnnotationError(
                f"job {jid} does not own datasets {unknown}"
            )


def assert_annotation_output_root(path: Path) -> None:
    """Final annotations stay under population_final. The historical root is refused."""
    text = Path(path).as_posix().rstrip("/")
    historical = HISTORICAL_OUTPUT_REL.rstrip("/")
    final = OUTPUT_REL.rstrip("/")
    if text == historical or text.endswith("/" + historical) or f"/{historical}/" in f"/{text}/":
        raise PicsV4AnnotationError(f"refusing historical population annotation root: {path}")
    if text != final and not text.endswith("/" + final) and f"/{final}/" not in f"/{text}/":
        raise PicsV4AnnotationError(
            f"population annotation output must stay under {OUTPUT_REL}: {path}"
        )


def assert_manifest_job_ids(programs: Sequence[Mapping[str, Any]]) -> None:
    """Refuse programs whose source job is historical or outside the final mapping."""
    ids = {str(row.get("source_job_id") or "") for row in programs}
    historical = sorted(job_id for job_id in ids if job_id in HISTORICAL_JOB_IDS)
    if historical:
        raise PicsV4AnnotationError(f"manifest contains historical jobs: {historical}")
    unknown = sorted(job_id for job_id in ids if job_id not in FINAL_JOB_IDS)
    if unknown:
        raise PicsV4AnnotationError(f"manifest contains non-final jobs: {unknown}")


def assert_manifest_is_final(path: Path) -> None:
    """Refuse a manifest file that names a historical job or a non-final job."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_annotation_output_root(path)
    assert_manifest_job_ids(payload.get("programs") or [])


def assert_manifest_is_final_payload(payload: Mapping[str, Any]) -> None:
    assert_manifest_job_ids(payload.get("programs") or [])


def expected_policy_fields() -> Dict[str, Any]:
    """Live PICS v4 identifiers. Values come from the method modules."""
    fields: Dict[str, Any] = {
        "method_version": METHOD_VERSION,
        "trial_prompt_policy": PICS_V4_TRIAL_POLICY,
        "prompt_policy": PICS_V4_TRIAL_POLICY,
        "additional_prompt_policy": ADDITIONAL_PROMPT_POLICY_ID,
        "additional_prompt_sha256": uniform_additional_prompt_sha256(),
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "packing_implementation": PACKING_IMPLEMENTATION_ID,
        "panel_policy": PANEL_POLICY_ID,
        "continuation_policy": CONTINUATION_POLICY_ID,
        "panel_bank_policy": PANEL_BANK_POLICY_ID,
        "parent_envelope_policy": PARENT_ENVELOPE_POLICY_ID,
        "slot_assignment": SLOT_ASSIGNMENT_POLICY_ID,
        "search_rng": SEARCH_RNG_POLICY,
        "pics_run_seed_scope": PICS_RUN_SEED_SCOPE,
        "split_seed": 0,
        "pics_run_seed": 0,
    }
    fields.update(elite_policy_provenance())
    for key, required in REQUIRED_POLICY_MARKERS.items():
        if fields.get(key) != required:
            raise PicsV4AnnotationError(
                f"live policy {key} is {fields.get(key)!r}, not {required!r}"
            )
    return fields


def documentation_gaps(doc_text: str) -> List[str]:
    """Return missing self-contained policy tokens. Empty means the doc records them."""
    gaps = [token for token in _REQUIRED_DOC_TOKENS if token not in doc_text]
    if "participant_population_unique_cap" not in doc_text or "| 5 |" not in doc_text:
        if "at most four" not in doc_text:
            gaps.append("five-program population handoff")
    prompt_sha = uniform_additional_prompt_sha256()
    if prompt_sha not in doc_text:
        gaps.append(f"uniform prompt sha256 {prompt_sha}")
    return gaps


def annotation_resume_key(
    dataset: str,
    job_id: str,
    phase: str,
    iteration: int,
    candidate_id: str,
    program_sha256: str,
) -> str:
    """Globally unique resume identity.

    dataset, job, phase, iteration, candidate id, and program SHA are all required.
    """
    parts = (
        str(dataset),
        f"job_{int(job_id)}",
        str(phase),
        str(int(iteration)),
        str(candidate_id),
        str(program_sha256),
    )
    if any(part == "" or "|" in part for part in parts):
        raise PicsV4AnnotationError(f"resume key part is empty or contains '|': {parts}")
    return "|".join(parts)


def _repo_relative(path: Path, repo: Path) -> str:
    """Repo-relative path without resolving symlinks."""
    try:
        return path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise PicsV4AnnotationError(f"path is outside the repo: {path}") from exc


def _job_dir(repo: Path, dataset: str, job_id: str) -> Path:
    return (
        repo
        / "generated_outputs"
        / "psych101_train"
        / "teh"
        / dataset
        / KIND_TARGET_ONLY
        / f"job_{job_id}"
    )


def _require_fields(payload: Mapping[str, Any], expected: Mapping[str, Any], where: str) -> None:
    for key, value in expected.items():
        found = payload.get(key)
        if found != value:
            raise PicsV4AnnotationError(
                f"{where} {key} is {found!r}, not {value!r}"
            )


def _marker_ok(job_dir: Path) -> None:
    marker = job_dir / "TRIAL_PROMPT_POLICY.json"
    if not marker.is_file():
        raise PicsV4AnnotationError(f"missing TRIAL_PROMPT_POLICY.json: {job_dir}")
    assert_pics_v4_resume(job_dir)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    _require_fields(payload, expected_policy_fields(), f"marker {marker}")


def _load_candidates(arm: Path, dataset: str, job_id: str, repo: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stage_path = arm / "STAGE_COMPLETE.json"
    prov_path = arm / "POPULATION_PROVENANCE.json"
    global_phase = arm / "global_phase"
    trace_path = global_phase / "mem_trace.jsonl"
    if (arm.parent / "transfer").exists():
        raise PicsV4AnnotationError(f"transfer arm present: {arm.parent / 'transfer'}")
    if not stage_path.is_file() or not prov_path.is_file() or not trace_path.is_file():
        raise PicsV4AnnotationError("population stage, provenance, or trace is missing")
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    if str(stage.get("dataset") or "") != dataset or str(provenance.get("dataset") or "") != dataset:
        raise PicsV4AnnotationError("population dataset does not match the job path")
    if str(stage.get("kind") or provenance.get("kind") or "") != KIND_TARGET_ONLY:
        raise PicsV4AnnotationError(
            f"kind is {stage.get('kind')!r} / {provenance.get('kind')!r}"
        )
    if provenance.get("test_used_for_selection") is not False:
        raise PicsV4AnnotationError("test_used_for_selection is not false")
    if str(provenance.get("evolution_selection_score") or "") != "train_val":
        raise PicsV4AnnotationError("population selection score is not train_val")
    if int(provenance.get("llm_max_tokens") or 0) != 1024:
        raise PicsV4AnnotationError(
            f"llm_max_tokens is {provenance.get('llm_max_tokens')!r}, not 1024"
        )
    if int(provenance.get("vllm_max_model_len") or 0) != 16384:
        raise PicsV4AnnotationError(
            f"vllm_max_model_len is {provenance.get('vllm_max_model_len')!r}, not 16384"
        )
    policy = expected_policy_fields()
    _require_fields(provenance, {k: policy[k] for k in (
        "method_version",
        "additional_prompt_policy",
        "additional_prompt_sha256",
        "hard_prompt_token_cap",
        "search_rng",
        "pics_run_seed_scope",
        "split_seed",
        "pics_run_seed",
        "elite_policy",
        "participant_population_unique_cap",
        "elite_pool_size_is_upper_bound",
        "duplicate_backfill",
    )}, f"provenance {prov_path}")
    n_iters = int(stage.get("global_iters") or 0)
    n_per = int(stage.get("n_candidates") or 0)
    if n_iters != 10 or n_per != 10:
        raise PicsV4AnnotationError(
            f"stage requests {n_iters} iterations x {n_per} candidates, not 10 x 10"
        )

    raw: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for rec in iter_jsonl_records([trace_path]):
        if rec.get("record_type") != "candidate":
            continue
        if record_contains_test_metrics(rec):
            raise PicsV4AnnotationError("population trace contains a test metric")
        if rec.get("phase") != POPULATION_PHASE or str(rec.get("participant_id")) != POPULATION_PARTICIPANT:
            raise PicsV4AnnotationError(
                "population trace is not population-only: "
                f"phase={rec.get('phase')!r} participant={rec.get('participant_id')!r}"
            )
        if str(rec.get("dataset") or "") != dataset:
            raise PicsV4AnnotationError("trace dataset mismatch")
        if str(rec.get("evolution_selection_score") or "") != "train_val":
            raise PicsV4AnnotationError("trace selection score is not train_val")
        cid = str(rec.get("candidate_id") or "")
        if not cid or cid in by_id:
            raise PicsV4AnnotationError(f"missing or duplicate candidate id {cid!r}")
        code = global_phase / str(rec.get("code_path") or "")
        if not code.is_file() or global_phase.resolve() not in code.resolve().parents:
            raise PicsV4AnnotationError(f"candidate file missing: {code}")
        raw.append(rec)
        by_id[cid] = rec

    on_disk = {p.resolve() for p in global_phase.glob("iteration_*/candidates/candidate_*.py")}
    traced = {(global_phase / str(rec.get("code_path") or "")).resolve() for rec in raw}
    if traced != on_disk:
        raise PicsV4AnnotationError(
            f"candidate files and mem_trace differ (files={len(on_disk)} trace={len(traced)})"
        )
    if len(raw) != 100:
        raise PicsV4AnnotationError(f"population trace has {len(raw)} candidates, not 100")
    counts = Counter(int(rec["iteration"]) for rec in raw)
    for iteration in range(1, 11):
        if counts[iteration] != 10:
            raise PicsV4AnnotationError(
                f"iteration {iteration} has {counts[iteration]} candidates, not 10"
            )

    job_token = f"job_{int(job_id)}"
    programs: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    stats: Dict[str, int] = {
        "n": 0,
        "selection_score_present": 0,
        "selection_score_missing": 0,
        "train_loglik_present": 0,
        "val_loglik_present": 0,
        "runtime_valid": 0,
        "runtime_invalid": 0,
        "survived_elite": 0,
        "not_in_elite": 0,
        "duplicate_sha_candidates": 0,
        "sha_resolved": 0,
    }
    sha_counts: Counter[str] = Counter()
    for rec in raw:
        code = global_phase / str(rec["code_path"])
        program_sha = source_sha256(code.read_text(encoding="utf-8"))
        sha_counts[program_sha] += 1
        parent_id = str(rec.get("reference_id") or rec.get("reference_parent_id") or "")
        if not parent_id:
            raise PicsV4AnnotationError(f"candidate missing reference: {rec.get('candidate_id')}")
        parent_path = _parent_path(arm.parent.parent, global_phase, parent_id, by_id)
        parent_sha = source_sha256(parent_path.read_text(encoding="utf-8"))
        trace_cid = str(rec["candidate_id"])
        key = annotation_resume_key(
            dataset, job_id, POPULATION_PHASE, int(rec["iteration"]), trace_cid, program_sha
        )
        if key in seen_keys:
            raise PicsV4AnnotationError(f"resume-key collision: {key}")
        seen_keys.add(key)
        stats["n"] += 1
        if rec.get("selection_score") is not None:
            stats["selection_score_present"] += 1
        else:
            stats["selection_score_missing"] += 1
        if rec.get("train_loglik") is not None:
            stats["train_loglik_present"] += 1
        if rec.get("val_loglik") is not None:
            stats["val_loglik_present"] += 1
        if rec.get("runtime_valid") is True:
            stats["runtime_valid"] += 1
        else:
            stats["runtime_invalid"] += 1
        if rec.get("survived_elite_truncation") is True:
            stats["survived_elite"] += 1
        else:
            stats["not_in_elite"] += 1
        programs.append(
            {
                "dataset": dataset,
                "run_id": job_token,
                "source_job_id": str(job_id),
                "phase": POPULATION_PHASE,
                "iteration": int(rec["iteration"]),
                "candidate_id": f"{dataset}::{job_token}::{trace_cid}",
                "program_id": trace_cid,
                "candidate_idx": rec.get("candidate_idx"),
                "parent": parent_id,
                "parent_id": parent_id,
                "source": rec.get("source"),
                "reference_kind": rec.get("reference_kind"),
                "reference_type": rec.get("reference_type") or rec.get("reference_kind"),
                "code_path": _repo_relative(code, repo),
                "parent_code_path": _repo_relative(parent_path, repo),
                "code_sha256": program_sha,
                "parent_code_sha256": parent_sha,
                "resume_key": key,
                "schema_version": SCHEMA_VERSION,
                "constructs": list(BEHAVIORAL_MOTIFS_V5),
                "runtime_valid": bool(rec.get("runtime_valid")),
                "survived_elite_truncation": bool(rec.get("survived_elite_truncation")),
            }
        )
    stats["duplicate_sha_candidates"] = sum(
        count for count in sha_counts.values() if count > 1
    )
    stats["sha_resolved"] = stats["n"]
    return programs, {k: int(v) for k, v in stats.items()}


def _parent_path(
    job_dir: Path,
    global_phase: Path,
    parent_id: str,
    by_candidate_id: Mapping[str, Mapping[str, Any]],
) -> Path:
    if parent_id in {"global_baseline", "baseline"}:
        seed = job_dir / "prompts" / "seed_program.py"
        if not seed.is_file():
            raise PicsV4AnnotationError(f"missing neutral seed: {seed}")
        return seed
    parent = by_candidate_id.get(parent_id)
    if parent is None:
        raise PicsV4AnnotationError(f"parent candidate is not in the population trace: {parent_id}")
    code = global_phase / str(parent.get("code_path") or "")
    if not code.is_file():
        raise PicsV4AnnotationError(f"parent file missing: {code}")
    return code


def inventory_populations(
    repo: Path,
    jobs: Sequence[Tuple[str, Sequence[str]]] = AUTHORITATIVE_JOBS,
) -> Dict[str, Any]:
    """Resolve each authoritative dataset. Incomplete traces are not parsed."""
    assert_final_job_mapping(jobs)
    repo = Path(repo)
    ready: List[Dict[str, Any]] = []
    datasets: List[Dict[str, Any]] = []
    resume_keys: set[str] = set()
    for job_id, names in jobs:
        for dataset in names:
            job_dir = _job_dir(repo, dataset, job_id)
            arm = job_dir / "target_population" / "control"
            row: Dict[str, Any] = {
                "dataset": dataset,
                "job_id": str(job_id),
                "job_dir": _repo_relative(job_dir, repo) if job_dir.exists() else str(
                    job_dir.relative_to(repo)
                ),
                "ready": False,
            }
            if not job_dir.is_dir():
                row["status"] = "incomplete"
                row["reason"] = "job directory is absent"
                datasets.append(row)
                continue
            try:
                _marker_ok(job_dir)
            except (PicsV4AnnotationError, RuntimeError, json.JSONDecodeError, OSError) as exc:
                row["status"] = "marker_mismatch" if (job_dir / "TRIAL_PROMPT_POLICY.json").is_file() else "incomplete"
                row["reason"] = str(exc)
                datasets.append(row)
                continue
            stage_path = arm / "STAGE_COMPLETE.json"
            trace_path = arm / "global_phase" / "mem_trace.jsonl"
            if not stage_path.is_file():
                row["status"] = "incomplete"
                row["reason"] = "population STAGE_COMPLETE.json is absent"
                row["trace_present_unread"] = trace_path.is_file()
                datasets.append(row)
                continue
            try:
                programs, stats = _load_candidates(arm, dataset, str(job_id), repo)
            except (PicsV4AnnotationError, RuntimeError, json.JSONDecodeError, OSError, UnicodeError) as exc:
                row["status"] = "incompatible"
                row["reason"] = str(exc)
                datasets.append(row)
                continue
            collisions = [p["resume_key"] for p in programs if p["resume_key"] in resume_keys]
            if collisions:
                row["status"] = "incompatible"
                row["reason"] = f"resume key collides with another dataset: {collisions[0]}"
                datasets.append(row)
                continue
            resume_keys.update(p["resume_key"] for p in programs)
            row.update(
                {
                    "ready": True,
                    "status": "ready",
                    "reason": "",
                    "n_programs": len(programs),
                    "stats": stats,
                }
            )
            datasets.append(row)
            ready.extend(programs)
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "constructs": list(BEHAVIORAL_MOTIFS_V5),
        "annotation_method": ANNOTATION_METHOD,
        "population_phase": POPULATION_PHASE,
        "n_programs": len(ready),
        "n_ready_datasets": sum(1 for row in datasets if row["ready"]),
        "n_datasets": len(datasets),
        "datasets": datasets,
        "programs": ready,
    }


def _markdown_report(payload: Mapping[str, Any], doc_gaps: Sequence[str]) -> str:
    lines = [
        "# PICS v4 population annotation inventory",
        "",
        "CPU inventory only. No annotation job was submitted.",
        "",
        f"Documentation gaps: {', '.join(doc_gaps) if doc_gaps else 'none'}.",
        "",
        f"Ready datasets: {payload['n_ready_datasets']} / {payload['n_datasets']}.",
        f"Expected annotations: {payload['n_programs']}.",
        "",
        "| Job | Dataset | Status | Candidates | Selection score | Invalid | Not in elite | Duplicate SHA |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["datasets"]:
        stats = row.get("stats") or {}
        lines.append(
            "| {job} | {dataset} | {status} | {n} | {score} | {invalid} | {elite} | {sha} |".format(
                job=row["job_id"],
                dataset=row["dataset"],
                status=row["status"],
                n=row.get("n_programs", ""),
                score=stats.get("selection_score_present", ""),
                invalid=stats.get("runtime_invalid", ""),
                elite=stats.get("not_in_elite", ""),
                sha=stats.get("duplicate_sha_candidates", ""),
            )
        )
    lines.extend(["", "## Skipped", ""])
    skipped = [row for row in payload["datasets"] if not row["ready"]]
    if not skipped:
        lines.append("None.")
    for row in skipped:
        unread = ""
        if row.get("trace_present_unread"):
            unread = " A population trace exists and was not read."
        lines.append(f"- `{row['dataset']}` job {row['job_id']}: {row['reason']}.{unread}")
    cache = payload.get("sha_label_cache") or {}
    if cache:
        lines.extend(
            [
                "",
                "## SHA label cache",
                "",
                "Schema v5 asks for constructs in the parent and changes in the candidate. "
                "The cache key is the SHA-256 of the rendered system and user prompt. "
                "A label is copied only when that prompt matches exactly.",
                "",
                (
                    f"Batch size {cache['batch_size']}: "
                    f"{cache['llm_items_before']} LLM items / {cache['llm_calls_before']} calls before caching; "
                    f"{cache['llm_items_after']} LLM items / {cache['llm_calls_after']} calls after caching."
                ),
                f"Output rows: {cache['output_rows']}.",
                (
                    f"Parent-mismatched rows: {cache['parent_mismatched_rows']}. "
                    "Each has its own rendered prompt and is annotated separately."
                ),
                "",
                "| Dataset | Rows | Unique prompts | Calls before | Calls after |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in cache["per_dataset"]:
            lines.append(
                f"| {row['dataset']} | {row['rows']} | {row['unique_prompts']} | "
                f"{row['calls_before']} | {row['calls_after']} |"
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Manifest: `{OUTPUT_REL}/program_manifest.json`",
            f"- Inventory: `{OUTPUT_REL}/inventory.json`",
            f"- Annotations, when submitted: `{OUTPUT_REL}/annotations/<job-name>/`",
            "",
            "Resume keys are `dataset|job_<id>|global_evolution|<iteration>|<candidate id>|<program sha256>`.",
            "Elite SHA deduplication does not remove a traced candidate.",
            "",
            "## Future submit",
            "",
            "```bash",
            "DRY_RUN=0 CONFIRM_SUBMIT=1 bash cluster/v0/ours/main/pics_v4/annotation/submit_pop_annot.sh",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def sha_label_cache_summary(
    programs: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 2,
    repo: Path | None = None,
) -> Dict[str, Any]:
    """Expected calls when labels are reused only for identical rendered prompts."""
    from analysis.mem.annotate_population_programs import (
        REPO,
        expected_population_llm_calls,
        plan_sha_label_representatives,
        stamp_annotation_prompt_sha,
    )
    from utils.mem import schema_population_motif_v5 as schema_v5

    root = Path(repo) if repo is not None else REPO
    needs_render = any(not p.get("annotation_prompt_sha256") for p in programs)
    if needs_render:
        stamped = stamp_annotation_prompt_sha(programs, schema_v5, root)
    else:
        stamped = [dict(p) for p in programs]
    planned = plan_sha_label_representatives(stamped)
    totals = expected_population_llm_calls(len(stamped), len(planned), batch_size)
    copied_across_parent = 0
    for rep in planned:
        for row in rep.get("sha_cache_aliases") or []:
            if row.get("parent_code_sha256") != rep.get("parent_code_sha256"):
                copied_across_parent += 1
    first_parent: Dict[Tuple[str, str], str] = {}
    parent_mismatched = 0
    for row in stamped:
        identity = (str(row["dataset"]), str(row["code_sha256"]))
        parent = str(row.get("parent_code_sha256") or "")
        if identity not in first_parent:
            first_parent[identity] = parent
        elif parent != first_parent[identity]:
            parent_mismatched += 1
    per_dataset: List[Dict[str, Any]] = []
    for dataset in sorted({str(p["dataset"]) for p in stamped}):
        rows = [p for p in stamped if p["dataset"] == dataset]
        unique = len({p["annotation_prompt_sha256"] for p in rows})
        calls = expected_population_llm_calls(len(rows), unique, batch_size)
        per_dataset.append(
            {
                "dataset": dataset,
                "rows": len(rows),
                "unique_prompts": unique,
                "calls_before": calls["llm_calls_before"],
                "calls_after": calls["llm_calls_after"],
            }
        )
    return {
        **totals,
        "cache_key": "rendered_annotation_prompt_sha256",
        "output_rows": len(stamped),
        "parent_mismatched_rows": parent_mismatched,
        "labels_copied_across_parent": copied_across_parent,
        "per_dataset": per_dataset,
    }


def submit_plans(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """GPU jobs for ready datasets only. Incomplete datasets are omitted."""
    ready = {row["dataset"] for row in payload["datasets"] if row.get("ready")}
    plans: List[Dict[str, Any]] = []
    for name, gres, datasets in ANNOTATION_GROUPS:
        chosen = [dataset for dataset in datasets if dataset in ready]
        if not chosen:
            continue
        plans.append({"job_name": name, "gres": gres, "datasets": chosen, "vllm_tp": 4})
    return plans


def write_inventory(repo: Path, jobs: Sequence[Tuple[str, Sequence[str]]] = AUTHORITATIVE_JOBS) -> Dict[str, Any]:
    doc_path = repo / DOC_REL
    doc_text = doc_path.read_text(encoding="utf-8") if doc_path.is_file() else ""
    gaps = documentation_gaps(doc_text)
    if gaps:
        raise PicsV4AnnotationError(
            "Documentation_v4.md is missing self-contained policy text: " + ", ".join(gaps)
        )
    payload = inventory_populations(repo, jobs)
    assert_manifest_is_final_payload(payload)
    payload["sha_label_cache"] = sha_label_cache_summary(payload["programs"])
    out_dir = repo / OUTPUT_REL
    assert_annotation_output_root(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": payload["schema"],
        "schema_version": payload["schema_version"],
        "prompt_version": payload["prompt_version"],
        "constructs": payload["constructs"],
        "annotation_method": payload["annotation_method"],
        "population_phase": payload["population_phase"],
        "n_programs": payload["n_programs"],
        "programs": payload["programs"],
    }
    (out_dir / "program_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    public = {k: v for k, v in payload.items() if k != "programs"}
    (out_dir / "inventory.json").write_text(
        json.dumps(public, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "INVENTORY.md").write_text(_markdown_report(payload, gaps), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=_REPO)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    doc_path = args.repo / DOC_REL
    gaps = documentation_gaps(doc_path.read_text(encoding="utf-8"))
    if gaps:
        print("documentation gaps: " + ", ".join(gaps), file=sys.stderr)
        raise SystemExit(2)
    if not args.write:
        payload = inventory_populations(args.repo)
    else:
        payload = write_inventory(args.repo)
    print(
        f"ready_datasets={payload['n_ready_datasets']}/{payload['n_datasets']} "
        f"annotations={payload['n_programs']}"
    )
    for row in payload["datasets"]:
        extra = f" n={row['n_programs']}" if row.get("ready") else f" {row['reason']}"
        print(f"  {row['status']}\t{row['job_id']}\t{row['dataset']}{extra}")


if __name__ == "__main__":
    main()
