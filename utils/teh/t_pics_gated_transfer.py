"""Final default T-PICS transfer pipeline (G.2 dual-arm + train_val gate).

Uses the frozen schema-v4 occurrence-EB cosine map as-is. Does not recompute
source similarity, retune the selector, or inspect transfer outcomes.

The observed-data performance gate (train_val) scores each arm's best
target-generated program on the union of target train+validation trials —
the same ``evolution_selection_score=train_val`` definition used in G.2.
SA40 validation alone is too small for arm selection. Target test is never
used for the gate.

G.3 matches Stage G / old T-PICS: the winner arm's full elite pool is loaded
via ``--initial_pool_dir``, but all ``--explore_candidates`` are generated
from that arm's train_val rank-1 only (``--explore_population_top_k 1``).
Person evolution then uses the full retained target pool plus decaying
seed-parented fresh candidates. Source rank-1 / source examples are not
injected into explore or person prompts.

Generic TEH / original PICS / OpenEvolve / legacy ``--t_pics`` live source
training are unchanged. This module is only the gated T-PICS transfer path.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from utils.teh.sandbox_builtins import compile_choose_with_error
from utils.teh.t_pics_sources import (
    normalize_t_pics_dataset,
    official_t_pics_source,
)
from utils.teh.teh_datasets import PARTICIPANT_DATASETS

_REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG = (
    _REPO_ROOT / "analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml"
)

KIND = "t_pics_gated"
INDEPENDENT_KIND = "t_pics_gated_independent"
RUN_TAG = "g5e50p10_occurrence_eb"
SOURCE_POPULATION_ITERS = 10
DEFAULT_GLOBAL_ITERS = 5
DEFAULT_EXPLORE_CANDIDATES = 50
DEFAULT_N_ITERATIONS = 10
DEFAULT_EXPLORE_POPULATION_TOP_K = 1
GATE_SCORE_FIELD = "mean_train_val_loglik"
GATE_NAME = "train_val"
GATE_LABEL = "observed-data performance gate"
GATE_TIE_TOLERANCE = 1e-12
GATE_RECORD_SCHEMA = "t_pics_gated_transfer_gate_v2"
EVOLUTION_SELECTION_SCORE = "train_val"
DEFAULT_N_CANDIDATES_PER_ITER = 10

# Diagnostic target-test evaluations only. Fitness/ranking/gate never use these.
DIAGNOSTIC_TEST_EVAL_POLICY = {
    "g1_g2_population": "pool_best_after_each_iteration",
    "g3_explore": "selected_best_only",
    "person": "pool_best_after_each_iteration_plus_final_selected",
    "passive": True,
}


def gated_diagnostic_test_eval_budget(
    *,
    n_participants: int,
    n_g2_iters: int = DEFAULT_GLOBAL_ITERS,
    n_person_iters: int = DEFAULT_N_ITERATIONS,
    n_explore_candidates: int = DEFAULT_EXPLORE_CANDIDATES,
    n_candidates_per_iter: int = DEFAULT_N_CANDIDATES_PER_ITER,
    n_g2_arms: int = 2,
) -> Dict[str, Any]:
    """Old vs new diagnostic test-eval counts for one gated target.

    G.1 completed source pops are not rerun. G.2 per-iteration tests score the
    pool-best once on pooled test (not per person). Explore/person counts are
    per person. Assumes every generated candidate was previously test-valid.
    """
    n = int(n_participants)
    g2_iters = int(n_g2_iters)
    person_iters = int(n_person_iters)
    explore = int(n_explore_candidates)
    cands = int(n_candidates_per_iter)
    arms = int(n_g2_arms)
    old = {
        "g2_per_iteration_pool_best": 0,
        "g2_final_summary_per_person": arms * n,
        "g3_explore_all_candidates": explore * n,
        "person_every_valid_candidate": cands * person_iters * n,
        "person_iter_best_reeval": person_iters * n,
        "person_final_selected": n,
        "person_seed_baseline": n,
    }
    new = {
        "g2_per_iteration_pool_best": arms * g2_iters,
        "g2_final_summary_per_person": arms * n,
        "g3_explore_selected_best": n,
        "person_iter_pool_best": person_iters * n,
        "person_final_selected": n,
        "person_seed_baseline": n,
    }
    old_total = int(sum(old.values()))
    new_total = int(sum(new.values()))
    return {
        "n_participants": n,
        "old": old,
        "new": new,
        "old_total": old_total,
        "new_total": new_total,
        "absolute_reduction": old_total - new_total,
        "fraction_remaining": (new_total / old_total) if old_total else None,
        "policy": dict(DIAGNOSTIC_TEST_EVAL_POLICY),
        "excludes_completed_g1": True,
    }


SOURCE_CONDITIONING_FLAGS = (
    "--global_prompt_source_program",
    "--global_prompt_source_dataset",
    "--explore_prompt_source_program",
    "--explore_prompt_source_dataset",
    "--t_pics_source",
)

REASON_TRANSFER_STRICTLY_BETTER = "transfer_strictly_better"
REASON_CONTROL_BETTER = "control_better"
REASON_EXACT_TIE = "exact_tie"
REASON_TOLERANCE_TIE = "tolerance_tie"
REASON_INVALID_SCORE = "invalid_score"
REASON_MISSING_ARM = "missing_arm"
REASON_FAILED_TRANSFER_ARM = "failed_transfer_arm"
REASON_FAILED_TRANSFER_PROGRAM = "failed_transfer_program"


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _read_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            f"PyYAML is required to load T-PICS gated config ({path}). pip install pyyaml"
        ) from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"T-PICS gated config is not a mapping: {path}")
    return payload


def resolve_frozen_config_path(config_path: Optional[Path] = None) -> Path:
    path = Path(config_path) if config_path is not None else FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class SourcePopulationRun:
    dataset: str
    job_id: str
    run_dir: Path
    rank1_program: Path


@dataclass(frozen=True)
class SelectedSourceEntry:
    target: str
    selected_source: str
    selected_source_label: str
    run_dir: Path
    job_id: str
    rank1_program: Path


@dataclass(frozen=True)
class FrozenTransferConfig:
    path: Path
    selector_name: str
    sources: Dict[str, str]
    targets: Dict[str, SelectedSourceEntry]
    source_runs: Dict[str, SourcePopulationRun]


@dataclass(frozen=True)
class GateDecision:
    selected_arm: str
    reason: str
    control_score: Optional[float]
    transfer_score: Optional[float]
    score_difference: Optional[float]
    score_field: str = GATE_SCORE_FIELD
    tie_tolerance: float = GATE_TIE_TOLERANCE

    @property
    def select_transfer(self) -> bool:
        return self.selected_arm == "transfer"


def load_frozen_transfer_config(
    config_path: Optional[Path] = None,
) -> FrozenTransferConfig:
    path = resolve_frozen_config_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen T-PICS transfer source config not found: {path}")
    payload = _read_yaml_mapping(path)
    selector = payload.get("selector") or {}
    selector_name = str(
        selector.get("name") if isinstance(selector, dict) else "occurrence_eb_schema4_iter10"
    )
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError(f"Frozen T-PICS config has no sources mapping: {path}")
    sources = {
        normalize_t_pics_dataset(str(target)): normalize_t_pics_dataset(str(source))
        for target, source in raw_sources.items()
    }
    raw_runs = payload.get("source_population_runs_used_to_fit")
    if not isinstance(raw_runs, dict) or not raw_runs:
        raise ValueError(f"Frozen T-PICS config has no source_population_runs_used_to_fit: {path}")
    source_runs: Dict[str, SourcePopulationRun] = {}
    for alias, entry in raw_runs.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Source run {alias!r} is not a mapping in {path}")
        dataset = normalize_t_pics_dataset(str(alias))
        job_id = str(entry.get("job_id") or "").strip()
        run_dir = _repo_path(str(entry.get("run_dir") or ""))
        rank1 = str(entry.get("rank1_program") or "").strip()
        if not rank1 and run_dir:
            rank1 = str(run_dir / "global_phase" / "best_program.py")
        if not job_id or not rank1:
            raise ValueError(
                f"Source run {dataset!r} needs job_id and rank1_program in {path}"
            )
        source_runs[dataset] = SourcePopulationRun(
            dataset=dataset,
            job_id=job_id,
            run_dir=run_dir,
            rank1_program=_repo_path(rank1),
        )
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise ValueError(f"Frozen T-PICS config has no targets mapping: {path}")
    targets: Dict[str, SelectedSourceEntry] = {}
    for alias, entry in raw_targets.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Target {alias!r} is not a mapping in {path}")
        target = normalize_t_pics_dataset(str(alias))
        selected = normalize_t_pics_dataset(str(entry.get("selected_source") or ""))
        mapped = sources.get(target)
        if mapped is not None and mapped != selected:
            raise ValueError(
                f"Target {target!r}: sources[{target}]={mapped!r} != selected_source={selected!r}"
            )
        run_dir_raw = str(entry.get("selected_source_run_dir") or "").strip()
        rank1_raw = str(entry.get("selected_source_rank1_program") or "").strip()
        job_id = str(entry.get("selected_source_job_id") or "").strip()
        if not rank1_raw and run_dir_raw:
            rank1_raw = str(Path(run_dir_raw) / "global_phase" / "best_program.py")
        if selected in source_runs:
            run = source_runs[selected]
            if not run_dir_raw:
                run_dir_raw = str(run.run_dir)
            if not rank1_raw:
                rank1_raw = str(run.rank1_program)
            if not job_id:
                job_id = run.job_id
        if not selected or not job_id or not rank1_raw:
            raise ValueError(
                f"Target {target!r} needs selected_source, job_id, and rank-1 path in {path}"
            )
        targets[target] = SelectedSourceEntry(
            target=target,
            selected_source=selected,
            selected_source_label=str(entry.get("selected_source_label") or selected),
            run_dir=_repo_path(run_dir_raw),
            job_id=job_id,
            rank1_program=_repo_path(rank1_raw),
        )
    return FrozenTransferConfig(
        path=path,
        selector_name=selector_name,
        sources=sources,
        targets=targets,
        source_runs=source_runs,
    )


def program_has_valid_choose(path: Path) -> Tuple[bool, str]:
    if not path.is_file():
        return False, f"missing file: {path}"
    try:
        code = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"cannot read {path}: {exc}"
    choose_fn, err = compile_choose_with_error(code)
    if choose_fn is None:
        return False, f"invalid choose() in {path}: {err}"
    return True, ""


def validate_frozen_transfer_config(
    config: Optional[FrozenTransferConfig] = None,
    *,
    config_path: Optional[Path] = None,
    require_files: bool = True,
) -> List[str]:
    """Validate every selected-source entry, source run, and rank-1 choose()."""
    cfg = config if config is not None else load_frozen_transfer_config(config_path)
    errors: List[str] = []
    if not cfg.targets:
        errors.append(f"no targets in {cfg.path}")
    if set(cfg.targets) != set(cfg.sources):
        errors.append(
            f"targets keys {sorted(cfg.targets)} != sources keys {sorted(cfg.sources)}"
        )
    for target, source in cfg.sources.items():
        if target == source:
            errors.append(f"self source for {target}")
        if source not in cfg.source_runs:
            errors.append(f"sources[{target}]={source} has no source_population run")
        if target not in PARTICIPANT_DATASETS:
            errors.append(f"unknown target dataset {target}")
        if source not in PARTICIPANT_DATASETS:
            errors.append(f"unknown source dataset {source}")
    for dataset, run in cfg.source_runs.items():
        if not run.job_id:
            errors.append(f"source run {dataset} missing job_id")
        if require_files:
            if not run.run_dir.is_dir():
                errors.append(f"missing source run_dir for {dataset}: {run.run_dir}")
            ok, msg = program_has_valid_choose(run.rank1_program)
            if not ok:
                errors.append(f"source {dataset} rank-1: {msg}")
    for target, entry in cfg.targets.items():
        mapped = cfg.sources.get(target)
        if mapped != entry.selected_source:
            errors.append(
                f"{target}: selected_source {entry.selected_source} != sources map {mapped}"
            )
        run = cfg.source_runs.get(entry.selected_source)
        if run is None:
            errors.append(f"{target}: selected source {entry.selected_source} has no G.1 run")
        elif entry.rank1_program.resolve() != run.rank1_program.resolve():
            errors.append(
                f"{target}: rank-1 path {entry.rank1_program} != source-run path {run.rank1_program}"
            )
        if require_files:
            if not entry.run_dir.is_dir():
                errors.append(f"{target}: missing selected source run_dir {entry.run_dir}")
            ok, msg = program_has_valid_choose(entry.rank1_program)
            if not ok:
                errors.append(f"{target} selected-source rank-1: {msg}")
    return errors


def selected_source_for_target(
    target_dataset: str,
    *,
    config: Optional[FrozenTransferConfig] = None,
    config_path: Optional[Path] = None,
) -> SelectedSourceEntry:
    cfg = config if config is not None else load_frozen_transfer_config(config_path)
    alias = normalize_t_pics_dataset(target_dataset)
    entry = cfg.targets.get(alias)
    if entry is None:
        raise KeyError(
            f"No frozen T-PICS selected source for {alias!r} in {cfg.path} "
            f"(known: {sorted(cfg.targets)})"
        )
    return entry


def resolve_gated_transfer_source_rank1(
    *,
    independent: bool,
    entry: SelectedSourceEntry,
    run_root: Path,
) -> Path:
    """Rank-1 used in the G.2 transfer suffix.

    Default gated T-PICS reuses the frozen YAML G.1 program. Independent mode
    trains a live source population under ``run_root/source_population`` and
    never reads the previous source job's ``best_program.py``.
    """
    if independent:
        return run_layout(Path(run_root))["source_rank1"]
    return Path(entry.rank1_program)


def gated_independent_enabled(args: Any) -> bool:
    return bool(getattr(args, "t_pics_gated_independent", False))


def run_layout(run_dir: Path) -> Dict[str, Path]:
    root = Path(run_dir)
    control = root / "target_population" / "control"
    transfer = root / "target_population" / "transfer"
    gate = root / "gate"
    selected = root / "selected"
    return {
        "root": root,
        "control": control,
        "transfer": transfer,
        "gate": gate,
        "selected": selected,
        "control_global": control / "global_phase",
        "transfer_global": transfer / "global_phase",
        "control_rank1": control / "global_phase" / "best_program.py",
        "transfer_rank1": transfer / "global_phase" / "best_program.py",
        "control_pool": control / "global_phase" / "global_elite_pool",
        "transfer_pool": transfer / "global_phase" / "global_elite_pool",
        "gate_record": gate / "gate_record.json",
        "run_metadata": root / "log" / "run_metadata.json",
        "selected_pool": selected / "retained_global_elite_pool",
        "selected_stage_complete": selected / "STAGE_COMPLETE.json",
        "source_population": root / "source_population",
        "source_rank1": root / "source_population" / "global_phase" / "best_program.py",
        "source_pool": root / "source_population" / "global_phase" / "global_elite_pool",
    }


def population_arm_is_complete(
    arm_dir: Path,
    *,
    expected_global_iters: int = DEFAULT_GLOBAL_ITERS,
) -> bool:
    """True only when G.2 artifacts are genuinely complete (not summary_loglik.csv)."""
    global_dir = Path(arm_dir) / "global_phase"
    best = global_dir / "best_program.py"
    pool_manifest = global_dir / "global_elite_pool" / "pool_manifest.json"
    results_path = global_dir / "results.json"
    if not (best.is_file() and pool_manifest.is_file() and results_path.is_file()):
        return False
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        n_iters = int(payload.get("n_iterations"))
        pool_size = int(payload.get("pool_size") or 0)
    except (TypeError, ValueError):
        return False
    if n_iters != int(expected_global_iters) or pool_size < 1:
        return False
    ok, _ = program_has_valid_choose(best)
    return ok


def participant_run_is_complete(
    participant_dir: Path,
    *,
    expected_n_iterations: int = DEFAULT_N_ITERATIONS,
    expected_explore_candidates: int = DEFAULT_EXPLORE_CANDIDATES,
) -> bool:
    """Skip a person only when explore + person artifacts are genuinely complete."""
    path = Path(participant_dir)
    best = path / "best_program.py"
    results_path = path / "results.json"
    if not (best.is_file() and results_path.is_file()):
        return False
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or "overall_best_train" not in payload:
        return False
    ok, _ = program_has_valid_choose(best)
    if not ok:
        return False
    if int(expected_explore_candidates) > 0:
        explore_metrics = path / "explore" / "metrics.json"
        if not explore_metrics.is_file():
            return False
        try:
            metrics = json.loads(explore_metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        requested = metrics.get("explore_candidates_requested")
        try:
            if int(requested) != int(expected_explore_candidates):
                return False
        except (TypeError, ValueError):
            return False
    iter_dir = path / f"iteration_{int(expected_n_iterations)}"
    iterations_csv = path / "iterations.csv"
    if iter_dir.is_dir():
        return True
    if iterations_csv.is_file():
        text = iterations_csv.read_text(encoding="utf-8")
        return f",{int(expected_n_iterations)}," in text or text.splitlines()[-1].startswith(
            str(int(expected_n_iterations))
        )
    return False


def selected_stage_is_complete(
    selected_dir: Path,
    participant_ids: Sequence[int],
    *,
    expected_n_iterations: int = DEFAULT_N_ITERATIONS,
    expected_explore_candidates: int = DEFAULT_EXPLORE_CANDIDATES,
) -> bool:
    marker = Path(selected_dir) / "STAGE_COMPLETE.json"
    if not marker.is_file():
        ids = [int(p) for p in participant_ids]
        if not ids:
            return False
        return all(
            participant_run_is_complete(
                Path(selected_dir) / f"participant_{pid}",
                expected_n_iterations=expected_n_iterations,
                expected_explore_candidates=expected_explore_candidates,
            )
            for pid in ids
        )
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    done = {int(x) for x in payload.get("participant_ids") or []}
    expected = {int(p) for p in participant_ids}
    return expected.issubset(done) and bool(expected)


def _finite_score(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def decide_gate(
    *,
    control_score: Optional[float],
    transfer_score: Optional[float],
    control_arm_ok: bool = True,
    transfer_arm_ok: bool = True,
    transfer_program_ok: bool = True,
    tie_tolerance: float = GATE_TIE_TOLERANCE,
) -> GateDecision:
    """Select transfer only when mean train_val loglik is strictly greater.

    Exact ties, tolerance ties, missing/non-finite scores, failed transfer
    arms, invalid transfer programs, and missing arms all select control.
    """
    control_ll = _finite_score(control_score)
    transfer_ll = _finite_score(transfer_score)
    diff = None
    if control_ll is not None and transfer_ll is not None:
        diff = float(transfer_ll) - float(control_ll)

    def _control(reason: str) -> GateDecision:
        return GateDecision(
            selected_arm="control",
            reason=reason,
            control_score=control_ll,
            transfer_score=transfer_ll,
            score_difference=diff,
        )

    if not control_arm_ok or not transfer_arm_ok:
        if not transfer_arm_ok and control_arm_ok:
            return _control(REASON_FAILED_TRANSFER_ARM)
        return _control(REASON_MISSING_ARM)
    if not transfer_program_ok:
        return _control(REASON_FAILED_TRANSFER_PROGRAM)
    if control_ll is None or transfer_ll is None:
        return _control(REASON_INVALID_SCORE)
    if transfer_ll == control_ll:
        return _control(REASON_EXACT_TIE)
    if abs(transfer_ll - control_ll) <= float(tie_tolerance):
        return _control(REASON_TOLERANCE_TIE)
    if transfer_ll > control_ll:
        return GateDecision(
            selected_arm="transfer",
            reason=REASON_TRANSFER_STRICTLY_BETTER,
            control_score=control_ll,
            transfer_score=transfer_ll,
            score_difference=diff,
        )
    return _control(REASON_CONTROL_BETTER)


def gate_record_payload(
    *,
    target: str,
    selected_source: str,
    decision: GateDecision,
    control_rank1: Path,
    transfer_rank1: Optional[Path],
    retained_pool: Path,
    config_path: Path,
    selector_name: str,
    selected_source_rank1: Path,
) -> Dict[str, Any]:
    return {
        "schema": GATE_RECORD_SCHEMA,
        "target": target,
        "selected_source": selected_source,
        "selected_source_rank1_program": str(selected_source_rank1),
        "config_path": str(config_path),
        "config_selector": selector_name,
        "score_field": decision.score_field,
        "gate_name": GATE_NAME,
        "gate_label": GATE_LABEL,
        "evolution_selection_score": EVOLUTION_SELECTION_SCORE,
        "participant_weighting": "equal",
        "observed_splits": ["train", "val"],
        "tie_tolerance": decision.tie_tolerance,
        "control_mean_train_val_loglik": decision.control_score,
        "transfer_mean_train_val_loglik": decision.transfer_score,
        GATE_SCORE_FIELD + "_control": decision.control_score,
        GATE_SCORE_FIELD + "_transfer": decision.transfer_score,
        "score_difference": decision.score_difference,
        "selected_arm": decision.selected_arm,
        "reason": decision.reason,
        "control_rank1_path": str(control_rank1),
        "transfer_rank1_path": None if transfer_rank1 is None else str(transfer_rank1),
        "retained_pool_path": str(retained_pool),
        "never_evaluated_raw_source_on_target": True,
        "never_used_target_test_for_gate": True,
    }


def write_gate_record(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    return path


def cli_flag_was_passed(flag: str, argv: Optional[Sequence[str]] = None) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    prefix = flag + "="
    return any(item == flag or item.startswith(prefix) for item in args)


def apply_gated_cli_defaults(args: Any, argv: Optional[Sequence[str]] = None) -> Any:
    """Apply T-PICS gated defaults without changing generic TEH argparse defaults."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args.global_phase = True
    args.refinement_phase = False
    args.explore_from_population_parents = True
    args.explore_population_top_k = DEFAULT_EXPLORE_POPULATION_TOP_K
    if not cli_flag_was_passed("--n_iterations", argv):
        args.n_iterations = DEFAULT_N_ITERATIONS
    if not cli_flag_was_passed("--global_iters", argv):
        args.global_iters = DEFAULT_GLOBAL_ITERS
    if not cli_flag_was_passed("--explore_candidates", argv):
        args.explore_candidates = DEFAULT_EXPLORE_CANDIDATES
    if getattr(args, "t_pics_source_config", None) in (None, ""):
        args.t_pics_source_config = str(FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG)
    return args


def plan_g3_explore_parents(
    elite_parents: Sequence[Tuple[Any, ...]],
    *,
    n_explore: int = DEFAULT_EXPLORE_CANDIDATES,
    explore_seed_candidates: int = 0,
) -> Dict[str, Any]:
    """G.3 explore plan: all candidates from retained train_val rank-1.

    The full winner elite remains available as the person initial pool
    (``retained_pool_ids``). Explore requests use only rank-1. No source
    rank-1 or source-example suffix is part of this plan.
    """
    from utils.teh.explore_handoff import (
        select_explore_handoff_parents,
        split_explore_budget_seed_and_parents,
    )

    pool_ids = [str(parent[3]) for parent in elite_parents]
    programs, pinned = select_explore_handoff_parents(
        elite_parents,
        enabled=True,
        top_k=DEFAULT_EXPLORE_POPULATION_TOP_K,
    )
    explore_ids = [str(pid) for _, pid in (programs or [])]
    n_seed, counts = split_explore_budget_seed_and_parents(
        int(n_explore),
        n_seed=int(explore_seed_candidates),
        n_handoff_parents=len(explore_ids),
    )
    rank1_id = pool_ids[0] if pool_ids else None
    counts_by_id = {
        pid: int(count) for pid, count in zip(explore_ids, counts)
    }
    n_from_rank1 = int(counts_by_id.get(str(rank1_id), 0)) if rank1_id else 0
    n_from_other = int(sum(counts)) - n_from_rank1
    return {
        "retained_pool_ids": pool_ids,
        "explore_parent_ids": list(pinned or []),
        "explore_parent_counts": counts_by_id,
        "n_explore": int(n_explore),
        "n_explore_from_rank1": n_from_rank1,
        "n_explore_from_other_pool_members": n_from_other,
        "n_explore_from_seed": int(n_seed),
        "n_explore_parents": len(explore_ids),
        "source_suffix_in_explore": False,
        "source_examples_in_explore": False,
        "person_fresh_from_seed_only": True,
    }


def gated_run_metadata(
    *,
    config_path: Path,
    target: str,
    selected_source: str,
    selected_source_rank1: Path,
    selector_name: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config_path = Path(config_path)
    config_sha256 = (
        hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.is_file() else None
    )
    payload: Dict[str, Any] = {
        "t_pics_gated_transfer": True,
        "t_pics_source_config": str(config_path),
        "frozen_transfer_source_config": str(config_path),
        "t_pics_source_config_sha256": config_sha256,
        "target": target,
        "selected_source": selected_source,
        "selected_source_rank1_program": str(selected_source_rank1),
        "config_selector": selector_name,
        "source_population_iters": SOURCE_POPULATION_ITERS,
        "target_population_iters": DEFAULT_GLOBAL_ITERS,
        "explore_candidates": DEFAULT_EXPLORE_CANDIDATES,
        "n_iterations": DEFAULT_N_ITERATIONS,
        "explore_population_top_k": DEFAULT_EXPLORE_POPULATION_TOP_K,
        "gate_score_field": GATE_SCORE_FIELD,
        "gate_name": GATE_NAME,
        "gate_label": GATE_LABEL,
        "evolution_selection_score": EVOLUTION_SELECTION_SCORE,
        "gate_tie_tolerance": GATE_TIE_TOLERANCE,
        "kind": KIND,
        "run_tag": RUN_TAG,
        "split_seed": 0,
        "experiment_seed_is_split_seed": True,
        "llm_decoding_seeds_enabled": True,
        "g2_arms_share_target_prompt": True,
        "g2_arms_share_llm_decoding_seed_formula": True,
        "method_decisions": {
            "gate_score_field": GATE_SCORE_FIELD,
            "gate_name": GATE_NAME,
            "evolution_selection_score": EVOLUTION_SELECTION_SCORE,
            "gate_uses_target_test": False,
            "gate_uses_source_raw_eval_on_target": False,
            "require_auto_llm_prompt": True,
            "require_source_examples_on_transfer": True,
            "g3_explore_from_retained_rank1_only": True,
            "g3_explore_population_top_k": DEFAULT_EXPLORE_POPULATION_TOP_K,
            "g3_retains_full_winner_pool_in_person_elite": True,
            "g3_explore_prompt_has_source_suffix": False,
            "g3_explore_prompt_has_source_examples": False,
            "person_fresh_from_seed_only": True,
        },
        "diagnostic_reporting": {
            "target_test_loglik_written": True,
            "target_test_loglik_used_for_prompts": False,
            "target_test_loglik_used_for_fitness": False,
            "target_test_loglik_used_for_ranking": False,
            "target_test_loglik_used_for_elite_pool": False,
            "target_test_loglik_used_for_selection": False,
            "target_test_loglik_used_for_stopping": False,
            "target_test_loglik_used_for_gate": False,
            "target_test_loglik_used_for_resume": False,
            "target_test_loglik_used_for_exploration": False,
            "target_test_loglik_used_for_person_evolution": False,
            "target_test_loglik_used_for_source_selection": False,
            "test_eval_policy": dict(DIAGNOSTIC_TEST_EVAL_POLICY),
            "note": (
                "Target-test loglik is a passive diagnostic record: G.1/G.2 "
                "pool-best after each iteration, G.3 selected explore best, "
                "person pool-best after each iteration plus the final "
                "train_val-selected program. Never used for prompts, fitness, "
                "ranking, parents, pools, gating, stopping, resume, or handoff."
            ),
        },
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _char4_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def audit_gated_transfer_prompt_budgets(
    *,
    config: Optional[FrozenTransferConfig] = None,
    hard_prompt_token_cap: int = 14000,
    max_parent_chars: int = 3500,
    sample_size: int = 8,
    max_prompt_train_trials: int = 60,
    split_seed: int = 0,
    limited_data_protocol: str = "structure_aware",
    limited_train_val: int = 40,
) -> List[Dict[str, Any]]:
    """CPU audit of G.2 transfer instruction+source-suffix token budgets.

    Target instruction is proxied by that dataset's completed G.1
    ``prompts/infer_single_choice.txt`` (same auto-prompt procedure).
    Source suffix is built from the frozen YAML rank-1 program plus
    required source train+val examples. Test trials are never included.
    """
    from utils.teh.explore_source_prompt import build_rank1_explore_prompt_suffix

    cfg = config or load_frozen_transfer_config()
    rows: List[Dict[str, Any]] = []
    for target, entry in sorted(cfg.targets.items()):
        target_run = cfg.source_runs.get(target)
        infer_path = (
            None
            if target_run is None
            else (Path(target_run.run_dir) / "prompts" / "infer_single_choice.txt")
        )
        infer_text = ""
        infer_ok = bool(infer_path is not None and infer_path.is_file())
        if infer_ok:
            infer_text = infer_path.read_text(encoding="utf-8")
        suffix = build_rank1_explore_prompt_suffix(
            source_dataset=entry.selected_source,
            program_path=str(entry.rank1_program),
            split_seed=int(split_seed),
            psych_dataset_split="train",
            split_ratio=0.6,
            limited_data_protocol=limited_data_protocol,
            limited_train_val=limited_train_val,
            filter_mixed_gambles=entry.selected_source == "mixed_gambles",
            require_source_examples=True,
        )
        source_code = Path(entry.rank1_program).read_text(encoding="utf-8")
        combined = infer_text.rstrip() + "\n\n" + suffix.strip() + "\n"
        infer_tokens = _char4_tokens(infer_text)
        suffix_tokens = _char4_tokens(suffix)
        combined_tokens = _char4_tokens(combined)
        parent_budget_tokens = _char4_tokens("x" * (int(max_parent_chars) * int(sample_size)))
        trial_budget_tokens = int(max_prompt_train_trials) * 40
        worst_tokens = combined_tokens + parent_budget_tokens + trial_budget_tokens
        silent_truncation = {
            "source_program_sliced": source_code.strip() not in suffix,
            "source_example_missing": "SOURCE example from source train+validation" not in suffix
            or "(no trials available)" in suffix,
            "source_schema_missing": "Source task / schema" not in suffix,
            "target_instruction_missing": (not infer_ok) or (not infer_text.strip()),
        }
        rows.append(
            {
                "target": target,
                "source": entry.selected_source,
                "infer_path": None if infer_path is None else str(infer_path),
                "infer_ok": infer_ok,
                "infer_tokens": infer_tokens,
                "suffix_tokens": suffix_tokens,
                "combined_instruction_tokens": combined_tokens,
                "hard_prompt_token_cap": int(hard_prompt_token_cap),
                "fits_cap_instruction_plus_suffix": combined_tokens <= int(hard_prompt_token_cap),
                "worst_case_with_parents_trials_tokens": worst_tokens,
                "source_program_chars": len(source_code),
                "source_example_present": not silent_truncation["source_example_missing"],
                "source_schema_present": not silent_truncation["source_schema_missing"],
                "silent_truncation": silent_truncation,
                "suffix_has_source_label": "SOURCE-dataset context only" in suffix,
                "suffix_excludes_test_note": "Source test trials are never included" in suffix,
            }
        )
    return rows


def common_pipeline_flags(
    *,
    dataset: str,
    output_dir: str,
    seed_path: str,
    n_iterations: int,
    global_iters: int,
    explore_candidates: int,
    source_program: Optional[str] = None,
    source_dataset: Optional[str] = None,
    include_gated_flag: bool = False,
    extra: Optional[Sequence[str]] = None,
) -> List[str]:
    """Logical arm argv. Production may run both arms inside one ``--t_pics_gated_transfer`` job."""
    argv = [
        "python",
        "teh.py",
        "--dataset",
        dataset,
        "--seed_path",
        seed_path,
        "--fitness_metric",
        "loglik",
        "--split_mode",
        "within_participant",
        "--split_ratio",
        "0.6",
        "--split_seed",
        "0",
        "--phase",
        "all",
        "--global_phase",
        "--global_iters",
        str(int(global_iters)),
        "--n_iterations",
        str(int(n_iterations)),
        "--n_candidates",
        "10",
        "--fresh_n_candidates",
        "10",
        "--sample_size",
        "8",
        "--sample_parents",
        "--sampled_parents_decay",
        "--elite_pool_size",
        "50",
        "--no-refinement_phase",
        "--prefer_auto_llm_prompt",
        "--evolution_selection_score",
        "train_val",
        "--max_error_prompt_chars",
        "0",
        "--explore_candidates",
        str(int(explore_candidates)),
    ]
    if int(explore_candidates) > 0:
        argv.extend(
            [
                "--explore_from_population_parents",
                "--explore_population_top_k",
                str(DEFAULT_EXPLORE_POPULATION_TOP_K),
            ]
        )
    argv.extend(
        [
            "--mem_trace",
            "--output_dir",
            output_dir,
        ]
    )
    if include_gated_flag:
        argv.append("--t_pics_gated_transfer")
        argv.extend(
            [
                "--t_pics_source_config",
                str(FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG),
            ]
        )
    if source_program:
        argv.extend(
            [
                "--global_prompt_source_program",
                str(source_program),
                "--global_prompt_source_dataset",
                str(source_dataset or ""),
            ]
        )
    if extra:
        argv.extend(list(extra))
    return argv


def build_control_argv(
    *,
    dataset: str,
    output_dir: str,
    seed_path: str,
    extra: Optional[Sequence[str]] = None,
) -> List[str]:
    argv = common_pipeline_flags(
        dataset=dataset,
        output_dir=output_dir,
        seed_path=seed_path,
        n_iterations=0,
        global_iters=DEFAULT_GLOBAL_ITERS,
        explore_candidates=0,
        extra=extra,
    )
    return argv


def build_transfer_argv(
    *,
    dataset: str,
    output_dir: str,
    seed_path: str,
    source_program: str,
    source_dataset: str,
    extra: Optional[Sequence[str]] = None,
) -> List[str]:
    return common_pipeline_flags(
        dataset=dataset,
        output_dir=output_dir,
        seed_path=seed_path,
        n_iterations=0,
        global_iters=DEFAULT_GLOBAL_ITERS,
        explore_candidates=0,
        source_program=source_program,
        source_dataset=source_dataset,
        extra=extra,
    )


def build_selected_argv(
    *,
    dataset: str,
    output_dir: str,
    seed_path: str,
    retained_pool: str,
    extra: Optional[Sequence[str]] = None,
) -> List[str]:
    argv = common_pipeline_flags(
        dataset=dataset,
        output_dir=output_dir,
        seed_path=seed_path,
        n_iterations=DEFAULT_N_ITERATIONS,
        global_iters=DEFAULT_GLOBAL_ITERS,
        explore_candidates=DEFAULT_EXPLORE_CANDIDATES,
        extra=extra,
    )
    # Selected stage must not rerun G.2; load the retained elite pool only.
    # Explore still uses rank-1 only (top_k=1); the rest of the pool is person elite.
    stripped: List[str] = []
    drop_with_value = {"--global_iters"}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--global_phase":
            i += 1
            continue
        if tok in drop_with_value:
            i += 2
            continue
        stripped.append(tok)
        i += 1
    stripped.extend(
        [
            "--no-global_phase",
            "--initial_pool_dir",
            str(retained_pool),
        ]
    )
    return stripped


def argv_source_conditioning_flags(argv: Sequence[str]) -> List[str]:
    found: List[str] = []
    tokens = set(SOURCE_CONDITIONING_FLAGS)
    for item in argv:
        name = item.split("=", 1)[0]
        if name in tokens:
            found.append(name)
    return found


def default_seed_path(dataset: str) -> str:
    alias = normalize_t_pics_dataset(dataset)
    if alias in {"steyvers_2009_bandit", "13schulz2020finding"}:
        return "persona_code_example/teh/categorical_uniform.py"
    return "persona_code_example/te_vanilla/choices13k.py"


def job_output_root(
    target: str,
    *,
    job_tag: str = "manual",
    repo: Optional[Path] = None,
    psych_split: str = "train",
    kind: str = KIND,
) -> Path:
    root = repo if repo is not None else _REPO_ROOT
    return (
        root
        / "generated_outputs"
        / f"psych101_{psych_split}"
        / "teh"
        / target
        / str(kind)
        / f"job_{job_tag}"
    )


def participant_train_val_loglik(
    train_loglik: Optional[float],
    val_loglik: Optional[float],
    n_train: int,
    n_val: int,
) -> Optional[float]:
    """Per-participant train_val score; identical to evolution_selection_score=train_val.

    This is the trial-count-weighted mean of train and val avg_loglik, equal to
    the mean log-likelihood on the union of observed (train+val) trials. Empty
    val falls back to train (all remaining observed trials). Test is not an
    input. Non-finite scores on a non-empty split return None.
    """
    n_tr = max(0, int(n_train))
    n_vl = max(0, int(n_val))
    if n_tr + n_vl <= 0:
        return None
    train_ll = _finite_score(train_loglik) if n_tr > 0 else None
    val_ll = _finite_score(val_loglik) if n_vl > 0 else None
    if n_tr > 0 and train_ll is None:
        return None
    if n_vl > 0 and val_ll is None:
        return None
    if n_tr == 0:
        return val_ll
    from teh import _evolution_selection_score

    return float(
        _evolution_selection_score(
            float(train_ll),
            val_ll,
            n_tr,
            n_vl,
            evolution_selection_score=EVOLUTION_SELECTION_SCORE,
        )
    )


def evaluate_mean_train_val_loglik(
    program_path: Path,
    *,
    dataset: str,
    participant_ids: Sequence[int],
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str = "train",
    filter_mixed_gambles: bool = False,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: Optional[str] = None,
    n_eval_seeds: int = 3,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    max_observed_trials_per_participant: Optional[int] = None,
    speekenbrink_split: str = "chronological",
) -> Optional[float]:
    """Equal-person mean of per-participant train_val loglik. Never evaluates test.

    Each person is scored on the union of their observed train+validation
    trials (SA40: all retained train+val, typically 40) using the same
    ``evolution_selection_score=train_val`` formula as G.2 ranking. People
    are then averaged with equal weight, not pooled-trial weight.
    """
    from teh import (
        _evaluate_loglik_for_dataset,
        _trials_for_loglik_participant,
        compile_program,
    )

    path = _repo_path(program_path)
    ok, _msg = program_has_valid_choose(path)
    if not ok:
        return None
    code = path.read_text(encoding="utf-8")
    choose_fn = compile_program(code)
    if choose_fn is None:
        return None
    scores: List[float] = []
    from data_modules.mixed_gambles import DEFAULT_CSV_PATH

    csv_path = mixed_gambles_csv or DEFAULT_CSV_PATH
    for pid in participant_ids:
        train_trials, val_trials, _test = _trials_for_loglik_participant(
            dataset,
            int(pid),
            split_ratio=float(split_ratio),
            split_seed=int(split_seed),
            filter_mixed_gambles=bool(filter_mixed_gambles),
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=csv_path,
            max_observed_trials_per_participant=max_observed_trials_per_participant,
            limited_data_protocol=limited_data_protocol,
            limited_train_val=limited_train_val,
            speekenbrink_split=speekenbrink_split,
        )
        del _test
        n_train = len(train_trials or [])
        n_val = len(val_trials or [])
        if n_train + n_val <= 0:
            continue
        train_ll: Optional[float] = None
        val_ll: Optional[float] = None
        if n_train > 0:
            train_result = _evaluate_loglik_for_dataset(
                dataset, choose_fn, list(train_trials), n_seeds=int(n_eval_seeds)
            )
            train_ll = _finite_score(train_result.get("avg_loglik"))
            if train_ll is None:
                return None
        if n_val > 0:
            val_result = _evaluate_loglik_for_dataset(
                dataset, choose_fn, list(val_trials), n_seeds=int(n_eval_seeds)
            )
            val_ll = _finite_score(val_result.get("avg_loglik"))
            if val_ll is None:
                return None
        score = participant_train_val_loglik(train_ll, val_ll, n_train, n_val)
        if score is None:
            return None
        scores.append(score)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _print_argv(label: str, argv: Sequence[str]) -> None:
    print(f"===== {label} =====")
    print(" ".join(str(x) for x in argv))


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="utils.teh.t_pics_gated_transfer")
    parser.add_argument(
        "command",
        choices=("validate", "dry-run", "gate-help", "selected-source"),
    )
    parser.add_argument("--dataset", default="1peterson2021using")
    parser.add_argument("--config", default=None)
    parser.add_argument("--job-tag", default="manual")
    parser.add_argument(
        "--independent",
        action="store_true",
        default=False,
        help=(
            "Look up the official selected source dataset only; do not require or "
            "reuse frozen G.1 rank-1 programs (live source population)."
        ),
    )
    args = parser.parse_args(argv)
    cfg_path = Path(args.config) if args.config else None
    if args.command == "gate-help":
        print(f"gate_name={GATE_NAME}")
        print(f"gate_label={GATE_LABEL}")
        print(f"score_field={GATE_SCORE_FIELD}")
        print(f"evolution_selection_score={EVOLUTION_SELECTION_SCORE}")
        print(f"tie_tolerance={GATE_TIE_TOLERANCE}")
        print(f"transfer_iff={GATE_SCORE_FIELD}(transfer) > {GATE_SCORE_FIELD}(control)")
        print("ties_and_invalid=control")
        print("test_never_used_for_gate=true")
        return 0
    cfg = load_frozen_transfer_config(cfg_path)
    independent = bool(getattr(args, "independent", False))
    require_files = not independent
    if args.command == "validate":
        errors = validate_frozen_transfer_config(cfg, require_files=require_files)
        if errors:
            print("[ERROR] frozen T-PICS transfer config failed validation:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print(
            f"[OK] frozen config {cfg.path} "
            f"targets={len(cfg.targets)} source_runs={len(cfg.source_runs)} "
            f"selector={cfg.selector_name}"
            + (" independent_source_lookup_only" if independent else "")
        )
        return 0
    if args.command == "selected-source":
        entry = selected_source_for_target(args.dataset, config=cfg)
        print(entry.selected_source)
        if independent:
            print("LIVE")
            print("live")
        else:
            print(entry.rank1_program)
            print(entry.job_id)
        return 0
    errors = validate_frozen_transfer_config(cfg, require_files=require_files)
    if errors:
        print("[ERROR] frozen T-PICS transfer config failed validation:")
        for err in errors:
            print(f"  - {err}")
        return 1
    entry = selected_source_for_target(args.dataset, config=cfg)
    mapped = official_t_pics_source(args.dataset, config_path=cfg.path)
    if mapped != entry.selected_source:
        print(
            f"[ERROR] sources map {mapped} != targets.selected_source {entry.selected_source}"
        )
        return 1
    out_root = job_output_root(
        args.dataset,
        job_tag=args.job_tag,
        kind=INDEPENDENT_KIND if independent else KIND,
    )
    layout = run_layout(out_root)
    seed = default_seed_path(args.dataset)
    live_rank1 = resolve_gated_transfer_source_rank1(
        independent=independent,
        entry=entry,
        run_root=out_root,
    )
    control_argv = build_control_argv(
        dataset=args.dataset,
        output_dir=str(layout["control"]),
        seed_path=seed,
    )
    transfer_argv = build_transfer_argv(
        dataset=args.dataset,
        output_dir=str(layout["transfer"]),
        seed_path=seed,
        source_program=str(live_rank1),
        source_dataset=entry.selected_source,
    )
    selected_argv = build_selected_argv(
        dataset=args.dataset,
        output_dir=str(layout["selected"]),
        seed_path=seed,
        retained_pool=str(layout["control_pool"]),
    )
    print(f"target={args.dataset}")
    print(f"frozen_config={cfg.path}")
    print(f"selected_source={entry.selected_source}")
    print(f"selected_source_rank1={live_rank1}")
    print(
        "source_job_id="
        + ("live" if independent else entry.job_id)
    )
    print(
        f"kind={INDEPENDENT_KIND if independent else KIND} run_tag={RUN_TAG}"
    )
    print(
        f"defaults: G.1={SOURCE_POPULATION_ITERS} "
        f"({'live in this run' if independent else 'precomputed'}) "
        f"G.2={DEFAULT_GLOBAL_ITERS}x2 arms  explore={DEFAULT_EXPLORE_CANDIDATES} "
        f"from retained rank-1 (top_k={DEFAULT_EXPLORE_POPULATION_TOP_K}) "
        f"person={DEFAULT_N_ITERATIONS}"
    )
    print(
        f"gate: {GATE_LABEL} field={GATE_SCORE_FIELD} "
        f"(evolution_selection_score={EVOLUTION_SELECTION_SCORE}) "
        f"tie_tolerance={GATE_TIE_TOLERANCE}"
    )
    print(f"output_root={out_root}")
    _print_argv("control G.2 argv", control_argv)
    control_src = argv_source_conditioning_flags(control_argv)
    print(f"control_source_flags={control_src or 'none'}")
    _print_argv("transfer G.2 argv", transfer_argv)
    print(f"transfer_source_program={live_rank1}")
    print(f"transfer_source_dataset={entry.selected_source}")
    _print_argv("selected G.3+person argv", selected_argv)
    print(
        "production job uses one `teh.py --t_pics_gated_transfer` process: "
        + (
            "live G.1 source population from the official selected source dataset, "
            if independent
            else ""
        )
        + "sequential matched G.2 arms, shared automatic target prompt, then gate, "
        "then selected-arm explore+person."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
