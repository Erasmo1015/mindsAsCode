#!/usr/bin/env python3
"""Build the uniform-v8 population transition table. Does not fit a model.

Reads ``population_final`` annotations and the official manifest, then joins
each row to the official population trace. Invalid and no-score candidates stay
in the table and are marked ineligible for a fitness-effect model.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.mem.pics_v4_population_annotation import (  # noqa: E402
    OUTPUT_REL,
    assert_annotation_output_root,
    assert_manifest_is_final,
)
from utils.mem.schema_participant_transition_v5 import verify_delta_f_consistency  # noqa: E402
from utils.mem.trace import record_contains_test_metrics  # noqa: E402
from utils.teh.elite_sha import source_sha256  # noqa: E402

TABLE_DIR_REL = f"{OUTPUT_REL}/analysis_table"
DRY_RUN_DIR_REL = f"{OUTPUT_REL}/table_dry_run"
CSV_NAME = "population_transitions.csv"
EXCLUSIONS_NAME = "exclusions.jsonl"
JOIN_AUDIT_NAME = "join_audit.json"
SHA_AUDIT_NAME = "sha_audit.json"
COVERAGE_NAME = "coverage_summary.json"


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _trace_index(trace_path: Path) -> Dict[str, Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    for rec in _load_jsonl(trace_path):
        if rec.get("record_type") != "candidate":
            continue
        found[str(rec.get("candidate_id"))] = rec
    return found


def _annotation_index(root: Path) -> tuple[Dict[str, Dict[str, Any]], set[str]]:
    found: Dict[str, Dict[str, Any]] = {}
    duplicates: set[str] = set()
    if not root.is_dir():
        return found, duplicates
    for path in sorted(root.rglob("annotations_population_v5.jsonl")):
        for rec in _load_jsonl(path):
            key = str(rec.get("resume_key") or "")
            if not key:
                continue
            if key in found:
                duplicates.add(key)
                continue
            found[key] = rec
    return found, duplicates


def build_population_transition_table(
    repo: Path,
    *,
    output_dir: Path,
    require_annotations: bool = True,
) -> Dict[str, Any]:
    """Join manifest, trace, and annotations. Writes the table and audits."""
    repo = Path(repo)
    manifest_path = repo / OUTPUT_REL / "program_manifest.json"
    assert_manifest_is_final(manifest_path)
    output_dir = Path(output_dir)
    assert_annotation_output_root(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    programs = list(manifest.get("programs") or [])
    annotation_root = repo / OUTPUT_REL / "annotations"
    annotations, duplicate_annotation_keys = _annotation_index(annotation_root)

    seen_keys: Dict[str, int] = Counter(str(row.get("resume_key") or "") for row in programs)
    traces: Dict[str, Dict[str, Dict[str, Any]]] = {}
    kept: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    sha_fail = 0
    sha_ok = 0

    def exclude(reason: str, program: Mapping[str, Any], **extra: Any) -> None:
        exclusions.append(
            {
                "reason": reason,
                "resume_key": program.get("resume_key"),
                "dataset": program.get("dataset"),
                "source_job_id": program.get("source_job_id"),
                "candidate_id": program.get("program_id"),
                **extra,
            }
        )

    for program in programs:
        key = str(program.get("resume_key") or "")
        if not key or seen_keys[key] != 1:
            exclude("duplicate_or_missing_resume_key", program)
            continue
        code_path = repo / str(program.get("code_path") or "")
        parent_path = repo / str(program.get("parent_code_path") or "")
        if not code_path.is_file() or not parent_path.is_file():
            sha_fail += 1
            exclude("missing_source_file", program)
            continue
        code_sha = source_sha256(code_path.read_text(encoding="utf-8"))
        parent_sha = source_sha256(parent_path.read_text(encoding="utf-8"))
        if code_sha != program.get("code_sha256") or parent_sha != program.get("parent_code_sha256"):
            sha_fail += 1
            exclude(
                "sha_mismatch",
                program,
                code_sha256=code_sha,
                parent_code_sha256=parent_sha,
            )
            continue
        trace_path = code_path.parents[2] / "mem_trace.jsonl"
        trace_key = str(trace_path)
        if trace_key not in traces:
            traces[trace_key] = _trace_index(trace_path) if trace_path.is_file() else {}
        trace = traces[trace_key].get(str(program.get("program_id")))
        if trace is None:
            exclude("trace_candidate_missing", program)
            continue
        if record_contains_test_metrics(trace):
            exclude("passive_test_field", program)
            continue
        if str(trace.get("phase")) != str(program.get("phase")):
            exclude("phase_mismatch", program, trace_phase=trace.get("phase"))
            continue
        if int(trace.get("iteration")) != int(program.get("iteration")):
            exclude("iteration_mismatch", program, trace_iteration=trace.get("iteration"))
            continue
        parent_id = str(trace.get("reference_id") or trace.get("reference_parent_id") or "")
        if parent_id != str(program.get("parent_id")):
            exclude("parent_mismatch", program, trace_parent=parent_id)
            continue
        if str(trace.get("dataset") or "") != str(program.get("dataset")):
            exclude("dataset_mismatch", program)
            continue
        sha_ok += 1
        reference_score = trace.get("reference_score")
        if reference_score is None:
            reference_score = trace.get("reference_parent_score")
        stored_delta = trace.get("delta_f")
        consistent, delta_err = verify_delta_f_consistency(
            delta_f=stored_delta,
            candidate_score=trace.get("selection_score"),
            reference_score=reference_score,
        )
        delta_source = "stored"
        delta_value = stored_delta
        if stored_delta is None and trace.get("selection_score") is not None and reference_score is not None:
            consistent, delta_err = verify_delta_f_consistency(
                delta_f=float(trace["selection_score"]) - float(reference_score),
                candidate_score=trace.get("selection_score"),
                reference_score=reference_score,
            )
            delta_value = float(trace["selection_score"]) - float(reference_score)
            delta_source = "computed_from_trace_scores"
        runtime_valid = trace.get("runtime_valid") is True and trace.get("selection_score") is not None
        annotation = annotations.get(key)
        if key in duplicate_annotation_keys:
            exclude("duplicate_annotation_resume_key", program)
            continue
        if require_annotations and annotation is None:
            exclude("annotation_missing", program)
            continue
        score_ok = runtime_valid and consistent and delta_value is not None
        if runtime_valid and not consistent:
            exclude("delta_f_inconsistent", program, detail=delta_err)
            continue
        row = {
            "dataset": program.get("dataset"),
            "source_job_id": program.get("source_job_id"),
            "run_id": program.get("run_id"),
            "phase": program.get("phase"),
            "iteration": program.get("iteration"),
            "candidate_id": program.get("program_id"),
            "resume_key": key,
            "source": program.get("source"),
            "parent_id": program.get("parent_id"),
            "reference_type": program.get("reference_type"),
            "code_sha256": code_sha,
            "parent_code_sha256": parent_sha,
            "runtime_valid": bool(runtime_valid),
            "selection_score": trace.get("selection_score"),
            "reference_score": reference_score,
            "train_loglik": trace.get("train_loglik"),
            "val_loglik": trace.get("val_loglik"),
            "delta_f": delta_value,
            "delta_f_source": delta_source if score_ok or not runtime_valid else delta_source,
            "fitness_effect_eligible": bool(score_ok and annotation is not None),
            "annotation_joined": annotation is not None,
            "survived_elite_truncation": program.get("survived_elite_truncation"),
        }
        if annotation is not None:
            for field in (
                "reference_motif_state",
                "candidate_motif_state",
                "added_motifs",
                "removed_motifs",
                "modified_motifs",
                "unchanged_motifs",
            ):
                row[field] = json.dumps(annotation.get(field) or [])
        kept.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(kept)
    frame.to_csv(output_dir / CSV_NAME, index=False)
    (output_dir / EXCLUSIONS_NAME).write_text(
        "".join(json.dumps(item) + "\n" for item in exclusions),
        encoding="utf-8",
    )
    reason_counts = Counter(item["reason"] for item in exclusions)
    join_audit = {
        "n_manifest_rows": len(programs),
        "n_table_rows": len(kept),
        "n_exclusions": len(exclusions),
        "exclusion_reasons": dict(reason_counts),
        "n_annotation_rows_indexed": len(annotations),
        "require_annotations": require_annotations,
        "passive_test_used": False,
    }
    sha_audit = {
        "n_sha_ok": sha_ok,
        "n_sha_fail": sha_fail,
        "candidate_and_parent_sha_checked": True,
    }
    coverage = {
        "n_rows": len(kept),
        "n_runtime_valid": int(frame["runtime_valid"].sum()) if len(frame) else 0,
        "n_runtime_invalid": int((~frame["runtime_valid"]).sum()) if len(frame) else 0,
        "n_fitness_effect_eligible": int(frame["fitness_effect_eligible"].sum()) if len(frame) else 0,
        "n_annotation_joined": int(frame["annotation_joined"].sum()) if len(frame) else 0,
        "by_dataset": (
            frame.groupby("dataset")
            .agg(n=("resume_key", "size"), n_valid=("runtime_valid", "sum"))
            .reset_index()
            .to_dict(orient="records")
            if len(frame)
            else []
        ),
    }
    (output_dir / JOIN_AUDIT_NAME).write_text(json.dumps(join_audit, indent=2) + "\n", encoding="utf-8")
    (output_dir / SHA_AUDIT_NAME).write_text(json.dumps(sha_audit, indent=2) + "\n", encoding="utf-8")
    (output_dir / COVERAGE_NAME).write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    return {"join_audit": join_audit, "sha_audit": sha_audit, "coverage": coverage, "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=_REPO)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write audits without requiring annotation JSONL. Does not fit.",
    )
    args = parser.parse_args()
    if args.dry_run:
        output = args.output_dir or (args.repo / DRY_RUN_DIR_REL)
        require = False
    else:
        output = args.output_dir or (args.repo / TABLE_DIR_REL)
        require = True
    summary = build_population_transition_table(
        args.repo,
        output_dir=output,
        require_annotations=require,
    )
    audit = summary["join_audit"]
    print(
        f"table_rows={audit['n_table_rows']} exclusions={audit['n_exclusions']} "
        f"output={summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
