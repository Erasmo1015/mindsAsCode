#!/usr/bin/env python3
"""LLM-seed=2 ablation population construct profiles / source selection.

Isolated from the official uniform-v8 freeze:
  - does not call utils.mem.pics_v4_frozen.check_frozen
  - reads only seed-2 population_final annotations/manifest
  - writes only under seed-2 population_source_profiles/

Reuses the same Occurrence-EB + cosine selection helpers as
analysis/mem/pics_v4_source_profiles.py without changing that frozen file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Profile must be set before importing population_annotation bindings used below.
os.environ["PICS_V4_POPANN_PROFILE"] = "seed2"

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.mem.pics_v4_population_annotation import (  # noqa: E402
    AUTHORITATIVE_JOBS,
    OUTPUT_REL,
    SEED2_ANNOTATION_GROUPS,
    apply_popann_profile,
    inventory_populations,
)
from analysis.mem import pics_v4_source_profiles as sp  # noqa: E402
from utils.mem.population_transition_finalize_v5 import (  # noqa: E402
    annotation_row_valid,
    finalize_annotation_jsonl,
)
from utils.mem.trace import record_contains_test_metrics  # noqa: E402

apply_popann_profile("seed2")

SEED2_SHARDS = tuple(name for name, _ in SEED2_ANNOTATION_GROUPS)
DEFAULT_OUTPUT_REL = (
    "analysis_2026Sep/Sep30_pics_v4/mem/rebuttal/diff_seed/seed-2/"
    "population_source_profiles/run_v1"
)
ABLATION_CONFIG_REL = (
    "analysis_2026Sep/Sep30_pics_v4/mem/rebuttal/diff_seed/seed-2/"
    "population_source_profiles/seed2_ablation_config.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def verify_manifest(repo: Path):
    live = inventory_populations(repo)
    if live["n_ready_datasets"] != 15 or len(live["programs"]) != 1500:
        raise ValueError(f"seed2 population integrity failed: {live['datasets']}")
    stored = json.loads((repo / OUTPUT_REL / "program_manifest.json").read_text(encoding="utf-8"))

    def index(programs):
        keyed = {p["resume_key"]: p for p in programs}
        if len(keyed) != 1500 or len(programs) != 1500:
            raise ValueError("Missing/duplicate canonical population identities")
        return keyed

    if index(stored["programs"]) != index(live["programs"]):
        raise ValueError("Manifest differs from SHA-verified seed2 live inventory")
    expected = {d: 100 for _, names in AUTHORITATIVE_JOBS for d in names}
    from collections import Counter

    if Counter(p["dataset"] for p in live["programs"]) != Counter(expected):
        raise ValueError("Not exactly 100 events per dataset")
    return live


def load_join(repo: Path, manifest):
    expected = {p["resume_key"]: p for p in manifest["programs"]}
    annotations, failures, paths = {}, {}, []
    for shard in SEED2_SHARDS:
        root = repo / OUTPUT_REL / "annotations" / shard
        path = root / "annotations_population_v5.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing seed2 annotation shard: {path}")
        paths.append(path)
        for r in sp.jsonl(path):
            key = r.get("resume_key")
            if key not in expected or key in annotations:
                raise ValueError(f"Unmatched/duplicate annotation: {key}")
            p = expected[key]
            for field in (
                "dataset",
                "run_id",
                "iteration",
                "code_sha256",
                "parent_code_sha256",
                "code_path",
                "parent_code_path",
                "parent_id",
                "source",
                "reference_kind",
                "reference_type",
            ):
                if r.get(field) != p.get(field):
                    raise ValueError(f"Annotation provenance mismatch: {key}: {field}")
            if r.get("candidate_id") != p["candidate_id"] or r.get("schema_version") != 5:
                raise ValueError(f"Candidate/schema mismatch: {key}")
            if record_contains_test_metrics(r):
                raise ValueError(f"Passive test fields refused: {key}")
            ok, error = annotation_row_valid(r)
            if not ok:
                raise ValueError(f"Invalid stored annotation {key}: {error}")
            annotations[key] = r
        for r in sp.jsonl(root / "annotation_failures.jsonl"):
            key = r.get("resume_key")
            if key not in expected or key in failures:
                raise ValueError(f"Unmatched/duplicate failure identity: {key}")
            failures[key] = r
    missing = set(expected) - set(annotations) - set(failures)
    if missing:
        raise ValueError(f"Missing joins without a recorded failure: {sorted(missing)[:20]}")
    return annotations, failures, paths


def write_ablation_config(repo: Path) -> dict:
    config = {
        "ablation": "llm_seed_2",
        "profile": "seed2",
        "interpretation": "post_hoc_seed2_ablation_not_official_uniform_v8_freeze",
        "output_rel": OUTPUT_REL,
        "annotation_shards": list(SEED2_SHARDS),
        "official_jobs": {job: list(ds) for job, ds in AUTHORITATIVE_JOBS},
        "constructs": list(sp.CONSTRUCTS),
        "population": {
            "canonical_events": 1500,
            "events_per_dataset": 100,
            "estimator": "existing_fit_occurrence_eb_explicit_five_constructs",
            "selection": "maximum_cosine_self_excluded_exact_tie_lex_canonical_dataset_id",
            "bootstrap": {
                "B": 1000,
                "seed": 20260920,
                "resampling": "within_dataset_iteration_parent_id_clusters",
            },
            "primary_denominator": "runtime_valid_successfully_annotated_semantically_resolved",
            "sensitivity_denominator": "all_generated_successfully_annotated_semantically_resolved",
            "test_and_fitness_scores": "never_used",
        },
        "does_not_modify": [
            "analysis/config/mem/pics_v4_uniform_v8_frozen.json",
            "analysis/config/mem/pics_v4_uniform_v8_inputs_frozen.json",
            "analysis_2026Sep/Sep30_pics_v4/mem/population_final",
            "analysis_2026Sep/Codex/Oct5/others/population_source_profiles/run_v1",
        ],
    }
    path = repo / ABLATION_CONFIG_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, config)
    return config


def run(repo: Path, output: Path, workers: int) -> None:
    apply_popann_profile("seed2")
    if OUTPUT_REL != (
        "analysis_2026Sep/Sep30_pics_v4/mem/rebuttal/diff_seed/seed-2/population_final"
    ):
        raise ValueError(f"refusing unexpected OUTPUT_REL={OUTPUT_REL}")
    if output.exists():
        raise ValueError("Immutable run output exists; use a new versioned output directory")
    # Keep ablation outputs under the seed-2 tree only.
    seed2_root = (repo / OUTPUT_REL).resolve().parent
    if seed2_root.name != "seed-2" or not str(output.resolve()).startswith(str(seed2_root)):
        raise ValueError(f"seed2 source-selection output must stay under {seed2_root}: {output}")

    config = write_ablation_config(repo)
    live = verify_manifest(repo)
    annotations, failures, paths = load_join(repo, live)
    output.mkdir(parents=True)
    dump(output / "ablation_config.json", config)
    before = {str(p.relative_to(repo)): sha(p) for p in paths}
    finalization = {}
    for path in paths:
        first = finalize_annotation_jsonl(path, repo)
        saved = sha(path)
        second = finalize_annotation_jsonl(path, repo)
        if (
            not first["schema_ok"]
            or not second["schema_ok"]
            or second["rewritten"]
            or sha(path) != saved
        ):
            raise ValueError(f"Finalizer validation/idempotence failed: {path}")
        finalization[str(path.relative_to(repo))] = dict(first=first, second=second)
    dump(output / "postprocessing.json", finalization)
    annotations, failures, paths = load_join(repo, live)
    frame = sp.build_master(live, annotations, failures)
    frame.to_csv(output / "population_master.csv", index=False)
    counts = []
    for dataset, g in frame.groupby("dataset", sort=True):
        counts.append(
            dict(
                dataset=dataset,
                canonical_events=len(g),
                runtime_valid=int(g.runtime_valid.sum()),
                runtime_invalid=int((~g.runtime_valid).sum()),
                annotation_failures=int((g.exclusion_reason == "annotation_failure").sum()),
                unresolved_semantic=int((g.exclusion_reason == "unresolved_semantic").sum()),
                primary_denominator=int(g.primary_eligible.sum()),
                sensitivity_denominator=int(g.sensitivity_eligible.sum()),
                duplicate_source_sha_events=len(g) - g.code_sha256.nunique(),
            )
        )
    import pandas as pd

    pd.DataFrame(counts).to_csv(output / "cohort_counts.csv", index=False)
    dump(
        output / "integrity.json",
        dict(
            canonical_events=1500,
            unique_resume_keys=int(frame.resume_key.nunique()),
            unmatched=0,
            missing_unexplained=0,
            sha_mismatches=0,
            duplicate_candidate_identities=0,
            duplicate_source_sha_events=1500 - int(frame.code_sha256.nunique()),
            annotation_failures=len(failures),
            lineage=dict(AUTHORITATIVE_JOBS),
            profile="seed2",
            annotation_shards=list(SEED2_SHARDS),
            resume_executions={},
            official_freeze_untouched=True,
        ),
    )
    maps = {}
    reproducibility = {}
    for name, mask in (
        ("primary_runtime_valid", frame.primary_eligible),
        ("sensitivity_all_generated", frame.sensitivity_eligible),
    ):
        out = output / name
        out.mkdir()
        cols = ["dataset", "iteration", "parent_id", "resume_key", *sp.CONSTRUCTS]
        panel = frame.loc[mask, cols].copy()
        for c in sp.CONSTRUCTS:
            panel[c] = panel[c].astype(int)
        if panel.dataset.nunique() != 15:
            raise ValueError("Every dataset must have a complete profile")
        panel.to_csv(out / "occurrence_panel.csv", index=False)
        profiles, eb, model, meta = sp.fit_profiles(panel)
        repeat, repeat_eb, repeat_model, _ = sp.fit_profiles(panel)
        for a, b in ((profiles, repeat), (eb, repeat_eb), (model, repeat_model)):
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        reproducibility[name] = "two deterministic fits bitwise identical"
        profiles.to_csv(out / "profiles.csv", index=False)
        eb.to_csv(out / "eb_estimates.csv", index=False)
        model.to_csv(out / "model.csv", index=False)
        dump(out / "estimator.json", meta)
        matrix, selected = sp.select_sources(profiles)
        matrix.to_csv(out / "similarity_matrix.csv", index_label="target_id")
        selected.to_csv(out / "selected_sources.csv", index=False)
        maps[name] = selected.set_index("target_id").selected_source_id.to_dict()
        dump(
            out / "source_map.json",
            dict(
                interpretation=(
                    "post hoc seed2 ablation construct-profile/source-selection; "
                    "never used by completed target-only experiments; "
                    "not the official uniform-v8 freeze"
                ),
                profile="seed2",
                sources=maps[name],
            ),
        )
        stable = sp.stability(panel, selected, out, workers)
        sp.figures(profiles, matrix, selected, stable, out)
    differences = {
        d: dict(primary=s, sensitivity=maps["sensitivity_all_generated"][d])
        for d, s in maps["primary_runtime_valid"].items()
        if s != maps["sensitivity_all_generated"][d]
    }
    dump(
        output / "map_comparison.json",
        dict(n_differences=len(differences), differences=differences, primary_policy_unchanged=True),
    )
    dump(output / "reproducibility.json", reproducibility)
    inputs = {str(p.relative_to(repo)): sha(p) for p in paths}
    inputs[str(Path(OUTPUT_REL) / "program_manifest.json")] = sha(
        repo / OUTPUT_REL / "program_manifest.json"
    )
    for shard in SEED2_SHARDS:
        p = repo / OUTPUT_REL / "annotations" / shard / "annotation_failures.jsonl"
        if p.is_file():
            inputs[str(p.relative_to(repo))] = sha(p)
    versions = {}
    from importlib.metadata import version

    for package in ("numpy", "pandas", "matplotlib", "statsmodels", "PyYAML", "scipy"):
        versions[package] = version(package)
    dump(
        output / "run_freeze.json",
        dict(
            date=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
            command=[sys.executable, *sys.argv],
            python=sys.version,
            platform=platform.platform(),
            packages=versions,
            seed=20260920,
            bootstrap_B=1000,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            annotation_hashes_before_finalization=before,
            input_hashes=inputs,
            ablation_config_hash=sha(repo / ABLATION_CONFIG_REL),
            official_frozen_config_untouched=True,
            output_hashes={
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
        ),
    )
    print(
        f"Completed seed2 ablation source selection: {output}; map differences: {len(differences)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=_REPO)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Default: {_REPO / DEFAULT_OUTPUT_REL}",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    repo = args.repo.absolute()
    output = (args.output or (repo / DEFAULT_OUTPUT_REL)).absolute()
    run(repo, output, args.workers)


if __name__ == "__main__":
    main()
