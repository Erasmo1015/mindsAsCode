"""PICS v4 A–E ablations on the frozen uniform-v8 track. Main KINDs stay untouched.

Parent sampling stays ``--sample_parents`` / ``--sample_size`` (default 8) except
in compensation iterations, which use ``--no-sample_parents`` behavior: the top
``sample_size`` elite parents. D disables fresh candidates and keeps sampling.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
GATE_CSV = REPO / "analysis_2026Sep/Sep30_pics_v4/results/official_gate/dataset_results.csv"
OMITTED_UNIFORM_POLICY = "pics_v4_uniform_block_omitted"
PARENT_COUNT_FLAG = "--sample_size"
DETERMINISTIC_PARENT_FLAG = "--no-sample_parents"

SPECS = {
    "A": {
        "id": "A",
        "kind": "pics_v4_ablation_no_transfer",
        "track": "target_only",
        "person_iters": 20,
        "explore": 50,
        "fresh": 10,
        "compensation_from": 11,
        "resume_after": 10,
        "reuse": "target_only_population_and_participants",
    },
    "B": {
        "id": "B",
        "kind": "pics_v4_ablation_no_population",
        "track": "target_only",
        "person_iters": 30,
        "explore": 50,
        "fresh": 10,
        "compensation_from": 11,
        "resume_after": 0,
        "reuse": "none_explore_from_neutral_seed",
        "ablate_population": True,
    },
    "C": {
        "id": "C",
        "kind": "pics_v4_ablation_no_explore",
        "track": "chosen",
        "person_iters": 15,
        "explore": 0,
        "fresh": 10,
        "compensation_from": 11,
        "resume_after": 0,
        "reuse": "chosen_population_elite",
    },
    "D": {
        "id": "D",
        "kind": "pics_v4_ablation_no_fresh",
        "track": "chosen",
        "person_iters": 10,
        "explore": 50,
        "fresh": 0,
        "compensation_from": None,
        "resume_after": 0,
        "reuse": "none_rerun_chosen_branch",
    },
    "E": {
        "id": "E",
        "kind": "pics_v4_ablation_no_uniform_prompt",
        "track": "chosen",
        "person_iters": 10,
        "explore": 50,
        "fresh": 10,
        "compensation_from": None,
        "resume_after": 0,
        "reuse": "none_rerun_chosen_branch",
        "omit_uniform": True,
    },
}

_CURRENT: Optional[dict] = None


def bind(args: Any) -> None:
    global _CURRENT
    label = getattr(args, "pics_v4_ablation", None)
    _CURRENT = None if not label else dict(SPECS[str(label)])
    if _CURRENT and getattr(args, "pics_v4_omit_uniform_block", False):
        _CURRENT["omit_uniform"] = True
    start = getattr(args, "pics_v4_compensation_from_iteration", None)
    if _CURRENT and start:
        _CURRENT["compensation_from"] = int(start)


def current_ablation() -> Optional[dict]:
    return _CURRENT


def omit_uniform_block() -> bool:
    return bool(_CURRENT and _CURRENT.get("omit_uniform"))


def _base_window_fresh(iteration_step: int, fresh_n_max: int, n_candidates: int) -> int:
    """Original 10-iteration fresh decay, independent of a longer compensation horizon."""
    import math

    idx = max(0, int(iteration_step) - 1)
    raw = math.floor(float(fresh_n_max) * (1.0 - idx / 10.0))
    return min(max(1, int(raw)), int(n_candidates))


def iteration_controls(
    iteration_step: int,
    *,
    sample_parents: bool,
    fresh_n: int,
    fresh_n_max: Optional[int] = None,
    n_candidates: Optional[int] = None,
) -> tuple[bool, int]:
    """Original stages keep sampling. Compensation uses top parents and no fresh."""
    spec = _CURRENT
    if spec is None:
        return sample_parents, fresh_n
    start = spec.get("compensation_from")
    if spec["id"] == "D" or (start is not None and int(iteration_step) >= int(start)):
        sampling = False if start is not None and int(iteration_step) >= int(start) else sample_parents
        return sampling, 0
    if start is not None and fresh_n_max is not None and n_candidates is not None:
        return sample_parents, _base_window_fresh(iteration_step, int(fresh_n_max), int(n_candidates))
    return sample_parents, fresh_n


def evolution_resume_index(output_path: Path, n_iterations: int) -> int:
    """A continues after the copied iteration-10 elite. Other ablations start at 0."""
    spec = _CURRENT
    if spec is None or spec["id"] != "A" or int(spec.get("resume_after") or 0) <= 0:
        return 0
    root = Path(output_path)
    after = int(spec["resume_after"])
    pool = root / "evolution_elite_pool" / "pool_manifest.json"
    if not pool.is_file() or not (root / f"iteration_{after}").is_dir():
        raise FileNotFoundError(
            f"Ablation A is missing reusable iteration {after} elite at {root}. "
            "Copy the completed target-only participant directory before launch."
        )
    done = 0
    for step in range(1, int(n_iterations) + 1):
        if (root / f"iteration_{step}").is_dir():
            done = step
        else:
            break
    return done


def load_gate_rows() -> list[dict]:
    with GATE_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plan_dataset(row: dict, ablation: str) -> dict:
    spec = SPECS[ablation]
    chosen = row["chosen_track"]
    status = row["result_status"]
    track = "target_only" if spec["track"] == "target_only" else chosen
    missing = []
    if spec["track"] == "chosen" and chosen == "unresolved":
        missing.append("official_gate_unresolved")
    if ablation == "A" and row["target_only_population_status"] != "complete":
        missing.append("target_only_population")
    if ablation == "A" and row.get("target_status") != "complete":
        missing.append(f"target_participant_stage:{row.get('target_status')}")
    if ablation == "C" and chosen == "unresolved":
        missing.append("chosen_population")
    if ablation == "C" and status == "chosen_participant_stage_incomplete" and chosen != "unresolved":
        pass
    predecessor = row["target_only_output_root"] if track == "target_only" else row.get("transfer_based_only_output_root") or ""
    if ablation == "A":
        predecessor = row["target_only_output_root"]
    if ablation == "C" and not missing:
        pop_status = row["target_only_population_status"] if track == "target_only" else row["transfer_based_only_population_status"]
        if pop_status != "complete":
            missing.append("chosen_population_incomplete")
    return {
        "dataset": row["dataset"],
        "ablation": ablation,
        "kind": spec["kind"],
        "track": track if not missing else None,
        "predecessor": predecessor if not missing else None,
        "missing": missing,
        "person_iters": spec["person_iters"],
        "compensation_from": spec["compensation_from"],
        "parent_count_flag": PARENT_COUNT_FLAG,
        "deterministic_parent_flag": DETERMINISTIC_PARENT_FLAG,
        "deterministic_parents": "compensation_iterations_only",
        "population_iteration_1_parent": "existing_neutral_seed" if ablation == "D" else "unchanged",
    }


def command_for(row: dict, ablation: str) -> Optional[str]:
    item = plan_dataset(row, ablation)
    if item["missing"]:
        return None
    spec = SPECS[ablation]
    dataset = row["dataset"]
    kind = spec["kind"]
    out = REPO / "generated_outputs/psych101_train/teh" / dataset / kind / "job_${SLURM_JOB_ID}"
    flags = [
        f"PICS_V4=1 TRACK_MODE={item['track']} PICS_V4_ABLATION={ablation}",
        f"PICS_V4_ABLATION_KIND={kind}",
        f"AAMAS_PERSON_ITERS={spec['person_iters']}",
        f"AAMAS_EXPLORE_CANDIDATES={spec['explore']}",
        f"AAMAS_FRESH_N={spec['fresh']}",
    ]
    if spec.get("compensation_from"):
        flags.append(f"PICS_V4_COMPENSATION_FROM={spec['compensation_from']}")
    launch = " ".join(flags)
    return (
        f"{launch} OUT_DIR={out} DATASET={dataset} "
        f"PRED={item['predecessor']} "
        "bash cluster/v0/ours/main/pics_v4/ablation/submit_ablations.sh"
    )


GENERATED = Path("/careAIDrive/zichang/misc/mindAsCode/generated_outputs/psych101_train/teh")
LAUNCH_DATASETS = ("bergert_nosofsky_2007", "1peterson2021using", "guan_2020_stopping")
# Refreshed 2026-10-07 from STAGE_COMPLETE on disk. All three chosen tracks are transfer.
LAUNCH_LINEAGE = {
    "bergert_nosofsky_2007": {
        "chosen_track": "transfer_based_only",
        "target_only": GENERATED / "bergert_nosofsky_2007/pics_v4_target_only/job_312023",
        "transfer_population": GENERATED / "bergert_nosofsky_2007/pics_v4_transfer_based_only/job_313159",
    },
    "1peterson2021using": {
        "chosen_track": "transfer_based_only",
        "target_only": GENERATED / "1peterson2021using/pics_v4_target_only/job_311672",
        "transfer_population": GENERATED / "1peterson2021using/pics_v4_transfer_based_only/job_313164",
        "transfer_participants": GENERATED / "1peterson2021using/pics_v4_transfer_based_only/job_314222",
    },
    "guan_2020_stopping": {
        "chosen_track": "transfer_based_only",
        "target_only": GENERATED / "guan_2020_stopping/pics_v4_target_only/job_311666",
        "transfer_population": GENERATED / "guan_2020_stopping/pics_v4_transfer_based_only/job_313158",
    },
}
MAIN_KIND_NAMES = ("pics_v4_target_only", "pics_v4_transfer_based_only", "pics_v4_official_gate")


def ablation_output(dataset: str, ablation: str, job_id: str) -> Path:
    return REPO / "generated_outputs/psych101_train/teh" / dataset / SPECS[ablation]["kind"] / f"job_{job_id}"


def _refuse_main_write(path: Path) -> None:
    resolved = path.resolve()
    if any(f"/{name}/" in f"{resolved}/" or resolved.name == name for name in MAIN_KIND_NAMES):
        raise PermissionError(f"ablation staging refuses a main output path: {resolved}")


def _copy_tree(src: Path, dest: Path) -> None:
    import shutil

    _refuse_main_write(dest)
    if dest.exists():
        raise FileExistsError(dest)
    shutil.copytree(src, dest, symlinks=False, copy_function=shutil.copy2)


def stage_dataset(dataset: str, ablation: str, dest: Path) -> dict:
    """Copy reusable inputs into a new ablation root. Originals are only read."""
    dest = Path(dest)
    if dataset not in LAUNCH_LINEAGE:
        if ablation not in {"D", "E"}:
            raise KeyError(dataset)
        _refuse_main_write(dest)
        provenance_path = dest / "ABLATION_PROVENANCE.json"
        if provenance_path.is_file():
            return json.loads(provenance_path.read_text(encoding="utf-8"))
        dest.mkdir(parents=True, exist_ok=True)
        record = {
            "ablation": ablation,
            "kind": SPECS[ablation]["kind"],
            "dataset": dataset,
            "role": "rerun_source_population",
            "chosen_track": "transfer_based_only",
            "fresh_n_candidates": SPECS[ablation]["fresh"],
            "omit_uniform": bool(SPECS[ablation].get("omit_uniform")),
            "reused": [],
            "predecessor_job": os.environ.get("PICS_V4_PREDECESSOR_JOB") or None,
        }
        provenance_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record
    _refuse_main_write(dest)
    if dest.parent.name != SPECS[ablation]["kind"]:
        raise RuntimeError(f"ablation output parent is not {SPECS[ablation]['kind']}")
    provenance_path = dest / "ABLATION_PROVENANCE.json"
    if provenance_path.is_file():
        return json.loads(provenance_path.read_text(encoding="utf-8"))
    lineage = LAUNCH_LINEAGE[dataset]
    spec = SPECS[ablation]
    record = {
        "ablation": ablation,
        "kind": spec["kind"],
        "dataset": dataset,
        "chosen_track": lineage["chosen_track"],
        "track": "target_only" if spec["track"] == "target_only" else lineage["chosen_track"],
        "person_iters": spec["person_iters"],
        "compensation_from": spec["compensation_from"],
        "sample_size": 8,
        "reused": [],
        "predecessor_job": os.environ.get("PICS_V4_PREDECESSOR_JOB") or None,
    }
    dest.mkdir(parents=True, exist_ok=True)
    if ablation == "A":
        src = lineage["target_only"]
        _copy_tree(src / "target_population", dest / "target_population")
        _copy_tree(src / "selected", dest / "selected")
        if (src / "TRIAL_PROMPT_POLICY.json").is_file():
            import shutil
            shutil.copy2(src / "TRIAL_PROMPT_POLICY.json", dest / "TRIAL_PROMPT_POLICY.json")
        marker = dest / "selected" / "STAGE_COMPLETE.json"
        if marker.exists():
            marker.unlink()
        record["reused"].append(str(src))
    elif ablation == "C":
        src = lineage["transfer_population"]
        (dest / "target_population").mkdir(parents=True, exist_ok=True)
        _copy_tree(src / "target_population" / "transfer", dest / "target_population" / "transfer")
        record["reused"].append(str(src / "target_population" / "transfer"))
    provenance_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def launch_readiness() -> dict:
    rows = []
    ready = True
    for dataset in LAUNCH_DATASETS:
        lineage = LAUNCH_LINEAGE[dataset]
        missing = []
        target = lineage["target_only"]
        transfer = lineage["transfer_population"]
        if not (target / "selected" / "STAGE_COMPLETE.json").is_file():
            missing.append("target_only_participants")
        if not (target / "selected" / "participant_0" / "iteration_10").is_dir() and not (target / "selected" / "participant_1" / "iteration_10").is_dir():
            missing.append("target_iteration_10")
        if not (transfer / "target_population" / "transfer" / "STAGE_COMPLETE.json").is_file():
            missing.append("transfer_population")
        if missing:
            ready = False
        rows.append({
            "dataset": dataset,
            "chosen_track": lineage["chosen_track"],
            "target_only": str(target),
            "transfer_population": str(transfer),
            "missing": missing,
        })
    return {"ready": ready, "jobs": 5, "datasets": rows, "jobs_submitted": 0}


def packed_commands() -> list[str]:
    script = "cluster/v0/ours/main/pics_v4/ablation/job_ablation.sh"
    commands = []
    for ablation, spec in SPECS.items():
        track = "target_only" if spec["track"] == "target_only" else "transfer_based_only"
        extra = ""
        if spec.get("compensation_from"):
            extra = f",PICS_V4_COMPENSATION_FROM={spec['compensation_from']}"
        commands.append(
            "sbatch --job-name=pics_v4_ablation_{ab} --export=ALL,"
            "PICS_V4=1,TRACK_MODE={track},PICS_V4_ABLATION={ab},"
            "PICS_V4_ABLATION_KIND={kind},AAMAS_PERSON_ITERS={iters},"
            "AAMAS_EXPLORE_CANDIDATES={explore},AAMAS_FRESH_N={fresh}{extra},VLLM_TP=1 "
            "{script}".format(
                ab=ablation,
                track=track,
                kind=spec["kind"],
                iters=spec["person_iters"],
                explore=spec["explore"],
                fresh=spec["fresh"],
                extra=extra,
                script=script,
            )
        )
    return commands


def dry_run() -> dict:
    rows = load_gate_rows()
    report = {"jobs_submitted": 0, "parent_count_argument": "--sample_size 8",
              "deterministic_parent_toggle": "--no-sample_parents",
              "compensation_rule": "iterations at or after compensation_from use top sample_size parents and fresh_n=0",
              "d_population_iteration_1": "neutral seed is already the initial elite parent; fresh_n=0 uses that parent",
              "datasets": []}
    for ablation in SPECS:
        for row in rows:
            item = plan_dataset(row, ablation)
            item["command"] = command_for(row, ablation)
            report["datasets"].append(item)
    return report


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="PICS v4 ablation pack dry-run and staging")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--ablation", choices=tuple(SPECS))
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.stage:
        print(json.dumps(stage_dataset(args.dataset, args.ablation, Path(args.output)), indent=2))
        return
    report = launch_readiness()
    report["commands"] = packed_commands()
    if args.check and not report["ready"]:
        print(json.dumps(report, indent=2))
        sys.exit(1)
    if args.check:
        print(json.dumps(report, indent=2))
        return
    print(json.dumps({"readiness": report, "full_catalog": dry_run(), "jobs_submitted": 0}, indent=2))


if __name__ == "__main__":
    main()
