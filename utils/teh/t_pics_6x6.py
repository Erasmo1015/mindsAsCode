"""Prospective 6×6 T-PICS transfer grid (prepare / validate only).

30 directed cross-dataset transfers + 6 target-only controls.
No same-dataset transfer. Sources are schema-v4 G.1 rank-1 programs.
Does not load or print a source-prediction ranking.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from utils.teh.t_pics_sources import load_t_pics_step1_source_pops, t_pics_step1_best_program

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA4_CONFIG = (
    _REPO_ROOT / "analysis/config/misc/Sep17_T-PICS/config_T-PICS_schema4.yaml"
)
DEFAULT_GRID_CONFIG = (
    _REPO_ROOT / "analysis/config/misc/Sep17_T-PICS/config_T-PICS_6x6.yaml"
)

# Family representatives (order is the packed-arm order after control).
REPRESENTATIVES: Tuple[str, ...] = (
    "1peterson2021using",  # Choice13k — gamble choice
    "guan_2020_stopping",  # sequential stopping/risk
    "steyvers_2009_bandit",  # sequential learning
    "bergert_nosofsky_2007",  # cue-based choice
    "11enkavi2019recentprobes",  # recognition/recall
    "12badham2017deficits",  # category learning
)

KIND = "t_pics_6x6"
RUN_TAG = "schema4_g5e50p5"
GLOBAL_ITERS = 5
N_ITERS = 5
EXPLORE_CANDIDATES = 50
MAX_SUBMITTED_JOBS = 16

JOB_NAME = {
    "1peterson2021using": "x6_c13k",
    "guan_2020_stopping": "x6_guan",
    "steyvers_2009_bandit": "x6_stey",
    "bergert_nosofsky_2007": "x6_berg",
    "11enkavi2019recentprobes": "x6_enk",
    "12badham2017deficits": "x6_badh",
}

# Safest split given SA40 Step 2 wall times and H100 NVL being busy:
# H100 5-day: longest / 50-person loops. L40S 5-day researchlong: medium /
# unknown packed loops. L40S 2-day researchshort: historically short.
GPU_PLAN: Dict[str, Dict[str, str]] = {
    "1peterson2021using": {
        "gpu": "h100nvl",
        "partition": "pradeepresearch",
        "qos": "pradeepresearch-priority",
        "gres": "gpu:1",
        "constraint": "h100nvl",
        "wall": "05-00:00:00",
        "mem": "64G",
        "worker": "job_6x6_h100.sh",
    },
    "bergert_nosofsky_2007": {
        "gpu": "h100nvl",
        "partition": "pradeepresearch",
        "qos": "pradeepresearch-priority",
        "gres": "gpu:1",
        "constraint": "h100nvl",
        "wall": "05-00:00:00",
        "mem": "64G",
        "worker": "job_6x6_h100.sh",
    },
    "guan_2020_stopping": {
        "gpu": "l40s_tp2",
        "partition": "researchlong",
        "qos": "research-1-qos",
        "gres": "gpu:l40s:2",
        "constraint": "",
        "wall": "05-00:00:00",
        "mem": "128G",
        "worker": "job_6x6_l40s.sh",
    },
    "steyvers_2009_bandit": {
        "gpu": "l40s_tp2",
        "partition": "researchlong",
        "qos": "research-1-qos",
        "gres": "gpu:l40s:2",
        "constraint": "",
        "wall": "05-00:00:00",
        "mem": "128G",
        "worker": "job_6x6_l40s.sh",
    },
    "11enkavi2019recentprobes": {
        "gpu": "l40s_tp2",
        "partition": "researchshort",
        "qos": "research-1-qos",
        "gres": "gpu:l40s:2",
        "constraint": "",
        "wall": "2-00:00:00",
        "mem": "128G",
        "worker": "job_6x6_l40s.sh",
    },
    "12badham2017deficits": {
        "gpu": "l40s_tp2",
        "partition": "researchshort",
        "qos": "research-1-qos",
        "gres": "gpu:l40s:2",
        "constraint": "",
        "wall": "2-00:00:00",
        "mem": "128G",
        "worker": "job_6x6_l40s.sh",
    },
}


@dataclass(frozen=True)
class GridArm:
    target: str
    arm_id: str
    source: Optional[str]
    role: str  # "control" | "transfer"

    @property
    def is_control(self) -> bool:
        return self.role == "control"


def _check_representatives(datasets: Sequence[str]) -> Tuple[str, ...]:
    out = tuple(str(ds) for ds in datasets)
    if len(out) != 6:
        raise ValueError(f"6×6 grid needs 6 representatives, got {len(out)}: {out}")
    if len(set(out)) != 6:
        raise ValueError(f"duplicate representatives: {out}")
    return out


def arms_for_target(
    target: str,
    representatives: Sequence[str] = REPRESENTATIVES,
) -> List[GridArm]:
    reps = _check_representatives(representatives)
    if target not in reps:
        raise ValueError(f"{target!r} is not a 6×6 representative")
    arms = [
        GridArm(target=target, arm_id="control", source=None, role="control"),
    ]
    for source in reps:
        if source == target:
            continue
        arms.append(
            GridArm(
                target=target,
                arm_id=f"xfer_{source}",
                source=source,
                role="transfer",
            )
        )
    if len(arms) != 6:
        raise RuntimeError(f"expected 6 arms for {target}, got {len(arms)}")
    return arms


def all_arms(representatives: Sequence[str] = REPRESENTATIVES) -> List[GridArm]:
    reps = _check_representatives(representatives)
    out: List[GridArm] = []
    for target in reps:
        out.extend(arms_for_target(target, reps))
    return out


def grid_counts(representatives: Sequence[str] = REPRESENTATIVES) -> Dict[str, int]:
    arms = all_arms(representatives)
    n_control = sum(1 for a in arms if a.is_control)
    n_xfer = sum(1 for a in arms if not a.is_control)
    same = [
        a for a in arms if a.source is not None and a.source == a.target
    ]
    if same:
        raise RuntimeError(f"same-dataset transfer leaked: {same}")
    return {
        "targets": len(_check_representatives(representatives)),
        "jobs": len(_check_representatives(representatives)),
        "arms": len(arms),
        "controls": n_control,
        "transfers": n_xfer,
        "same_dataset": 0,
    }


def arm_output_dir(target: str, arm_id: str, *, repo: Optional[Path] = None) -> Path:
    root = repo if repo is not None else _REPO_ROOT
    return (
        root
        / "generated_outputs"
        / "psych101_train"
        / "teh"
        / target
        / KIND
        / RUN_TAG
        / arm_id
    )


def job_output_root(target: str, *, repo: Optional[Path] = None) -> Path:
    root = repo if repo is not None else _REPO_ROOT
    return root / "generated_outputs" / "psych101_train" / "teh" / target / KIND / RUN_TAG


def arm_is_complete(arm_dir: Path) -> bool:
    return (arm_dir / "summary_loglik.csv").is_file() and (
        arm_dir / "global_phase" / "best_program.py"
    ).is_file()


def required_freeze_flags() -> Tuple[str, ...]:
    return (
        "--evolution_selection_score",
        "train_val",
        "--fresh_n_candidates",
        "10",
        "--prefer_auto_llm_prompt",
        "--max_error_prompt_chars",
        "0",
        "--global_phase",
        "--global_iters",
        "--n_iterations",
        "--explore_candidates",
        "50",
        "--elite_pool_size",
        "50",
        "--no-refinement_phase",
        "--split_mode",
        "within_participant",
        "--split_ratio",
        "0.6",
        "--split_seed",
        "0",
    )


def forbidden_cli_tokens() -> Tuple[str, ...]:
    return (
        "--t_pics",
        "--t_pics_source",
        "--t_pics_source_config",
        "--explore_prompt_source_program",
    )


def source_rank1(
    source: str,
    *,
    config_path: Optional[Path] = None,
) -> Path:
    cfg = config_path if config_path is not None else DEFAULT_SCHEMA4_CONFIG
    return t_pics_step1_best_program(source, cfg)


def validate_schema4_sources(
    representatives: Sequence[str] = REPRESENTATIVES,
    *,
    config_path: Optional[Path] = None,
) -> List[str]:
    cfg = config_path if config_path is not None else DEFAULT_SCHEMA4_CONFIG
    pops = load_t_pics_step1_source_pops(cfg)
    errors: List[str] = []
    for alias in representatives:
        entry = pops.get(alias)
        if entry is None:
            errors.append(f"schema4 missing step1.source_pops[{alias}]")
            continue
        best = Path(entry["best_program"])
        if not best.is_absolute():
            best = _REPO_ROOT / best
        if not best.is_file():
            errors.append(f"missing G.1 rank-1 for {alias}: {best}")
    return errors


def slurm_jobs(
    representatives: Sequence[str] = REPRESENTATIVES,
    targets: Optional[Iterable[str]] = None,
) -> List[Dict[str, object]]:
    reps = _check_representatives(representatives)
    wanted = list(targets) if targets is not None else list(reps)
    out: List[Dict[str, object]] = []
    for target in wanted:
        if target not in GPU_PLAN:
            raise KeyError(f"no GPU plan for {target}")
        plan = dict(GPU_PLAN[target])
        plan["target"] = target
        plan["job_name"] = JOB_NAME[target]
        plan["arms"] = arms_for_target(target, reps)
        out.append(plan)
    if len(out) > MAX_SUBMITTED_JOBS:
        raise ValueError(
            f"6×6 plan has {len(out)} jobs; cap is {MAX_SUBMITTED_JOBS}"
        )
    return out


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="utils.teh.t_pics_6x6")
    parser.add_argument(
        "command",
        choices=("counts", "arms", "plan", "validate", "rank1"),
    )
    parser.add_argument("dataset", nargs="?")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SCHEMA4_CONFIG),
        help="schema-v4 run config with step1.source_pops",
    )
    args = parser.parse_args(argv)
    cfg = Path(args.config)
    if args.command == "counts":
        counts = grid_counts()
        for k, v in counts.items():
            print(f"{k}={v}")
        return 0
    if args.command == "arms":
        if not args.dataset:
            parser.error("arms requires a target dataset")
        for arm in arms_for_target(args.dataset):
            src = arm.source or "NONE"
            print(f"{arm.arm_id}\t{arm.role}\t{src}")
        return 0
    if args.command == "plan":
        for job in slurm_jobs():
            print(
                f"{job['job_name']}\t{job['target']}\t{job['gpu']}\t"
                f"{job['partition']}\t{job['wall']}\t{job['worker']}"
            )
        return 0
    if args.command == "rank1":
        if not args.dataset:
            parser.error("rank1 requires a source dataset")
        print(source_rank1(args.dataset, config_path=cfg))
        return 0
    errors = validate_schema4_sources(config_path=cfg)
    counts = grid_counts()
    if counts["transfers"] != 30 or counts["controls"] != 6:
        errors.append(f"bad grid counts: {counts}")
    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        return 1
    print("OK 6x6 grid: 30 transfers + 6 controls; schema-v4 rank-1s present")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
