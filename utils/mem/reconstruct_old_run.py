"""Reconstruct synthetic mem_trace.jsonl from pre-MEM TEH/PICS run artifacts.

Does not rerun teh.py. Uses on-disk iteration/explore metrics + candidate sources,
rescored initial_pool_from_global programs on each participant's train/val split,
and chronological elite-pool updates for proxy references.

Generation-reference policy (participant-level MEM):
  * explore (default TEH): prompted from pure seed/baseline only
      → reference_type=seed_baseline (exact); ΔF vs baseline selection_score
  * explore with handoff population parents (when recoverable): population_program
  * fresh evolution slots: seed/baseline only → seed_baseline (exact)
  * normal evolution: prompted parents not stored in old artifacts
      → reference_type=pool_best_proxy (explicit proxy); ΔF vs pre-iteration pool best

Do not assign pool-best automatically to explore/fresh when baseline is recoverable.
Never reads test metrics into the emitted records.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.mem.reference_types import (
    REF_POOL_BEST_PROXY,
    REF_POPULATION_PROGRAM,
    REF_SEED_BASELINE,
    REFERENCE_KIND_POOL_BEST_PROXY,
)
from utils.mem.rescore_initial_pool import (
    parse_split_hparams_from_command,
    rescore_initial_pool_for_participant,
)
from utils.mem.trace import (
    append_mem_trace_record,
    build_candidate_record,
    build_iteration_context_record,
    compute_delta_f,
    mem_trace_path,
    record_contains_test_metrics,
)

_PARTICIPANT_DIR_RE = re.compile(r"^participant_(\d+)$")
_ITERATION_DIR_RE = re.compile(r"^iteration_(\d+)$")


@dataclass
class PoolProgram:
    program_id: str
    selection_score: float
    code: str
    train_loglik: Optional[float] = None
    val_loglik: Optional[float] = None


@dataclass
class ReconstructStats:
    participants: int = 0
    iterations: int = 0
    candidates_written: int = 0
    runtime_valid_written: int = 0
    fresh_runtime_valid_written: int = 0
    fresh_added_to_pool: int = 0
    explore_runtime_valid_written: int = 0
    skipped_missing_code: int = 0
    initial_pool_rescored: int = 0
    warnings: List[str] = field(default_factory=list)


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_elite_pool_size_from_command(command_txt: Path) -> Optional[int]:
    if not command_txt.is_file():
        return None
    text = command_txt.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"--elite_pool_size(?:\s+|=)(\d+)", text)
    if not m:
        return None
    return int(m.group(1))


def infer_dataset_from_run_dir(run_dir: Path) -> str:
    """Best-effort dataset name from .../<dataset>/<run_id> layout."""
    parent = run_dir.parent.name
    if parent and parent not in {"teh", "TEH"}:
        return parent
    # .../teh/<dataset>/<run_id>
    if run_dir.parent.parent.name.lower() == "teh":
        return run_dir.parent.name
    return parent or "unknown_dataset"


def list_participant_dirs(run_dir: Path) -> List[Path]:
    dirs: List[Tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = _PARTICIPANT_DIR_RE.match(p.name)
        if not m:
            continue
        dirs.append((int(m.group(1)), p))
    dirs.sort(key=lambda t: t[0])
    return [p for _, p in dirs]


def list_iteration_dirs(participant_dir: Path) -> List[Tuple[int, Path]]:
    found: List[Tuple[int, Path]] = []
    for p in participant_dir.iterdir():
        if not p.is_dir():
            continue
        m = _ITERATION_DIR_RE.match(p.name)
        if not m:
            continue
        found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return found


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _load_candidate_code(candidates_dir: Path, idx: int) -> Optional[str]:
    path = candidates_dir / f"candidate_{idx}.py"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _truncate_pool(pool: List[PoolProgram], elite_pool_size: int) -> List[PoolProgram]:
    cap = max(1, int(elite_pool_size))
    ordered = sorted(pool, key=lambda p: p.selection_score, reverse=True)
    return ordered[:cap]


def _pool_best(pool: Sequence[PoolProgram]) -> Optional[PoolProgram]:
    best: Optional[PoolProgram] = None
    best_score = float("-inf")
    for prog in pool:
        if prog.selection_score > best_score:
            best_score = prog.selection_score
            best = prog
    return best


def _parent_record(prog: PoolProgram) -> Dict[str, Any]:
    # Slim mem_trace: ids + scores only; annotate resolves code from run artifacts.
    return {
        "program_id": prog.program_id,
        "selection_score": prog.selection_score,
        "train_loglik": prog.train_loglik,
        "val_loglik": prog.val_loglik,
    }


def _baseline_from_pool_or_results(
    pool: Sequence[PoolProgram],
    participant_dir: Path,
) -> Optional[PoolProgram]:
    """Prefer global_baseline / baseline in pool; else results.json + seed file."""
    for pid in ("global_baseline", "baseline"):
        hit = next((p for p in pool if p.program_id == pid), None)
        if hit is not None:
            return hit
    results_path = participant_dir / "results.json"
    if not results_path.is_file():
        return None
    baseline = _read_json(results_path).get("baseline") or {}
    sel = _safe_float(baseline.get("selection_score"))
    if sel is None:
        return None
    code_path = participant_dir / "initial_pool_from_global" / "000_global_baseline.py"
    code = (
        code_path.read_text(encoding="utf-8")
        if code_path.is_file()
        else "def choose (problem ,history ):\n    return 0.5\n"
    )
    return PoolProgram(
        program_id="global_baseline",
        selection_score=sel,
        code=code,
        train_loglik=_safe_float(baseline.get("train_loglik")),
        val_loglik=_safe_float(baseline.get("val_loglik")),
    )


def seed_pool_from_participant(
    participant_dir: Path,
    *,
    elite_pool_size: int,
    dataset: str,
    split_ratio: float = 0.6,
    split_seed: int = 0,
    n_eval_seeds: int = 3,
    psych_dataset_split: str = "train",
    evolution_selection_score: str = "train_val",
    rescore_cache_path: Optional[Path] = None,
    force_rescore: bool = False,
    stats: Optional[ReconstructStats] = None,
    include_explore: bool = True,
) -> List[PoolProgram]:
    """Build elite pool.

    Includes:
      - all ``initial_pool_from_global`` programs rescored on this participant's
        historical train/val split (cached; never uses global pooled scores)
      - optionally explore_phase runtime-valid candidates (when include_explore)
    """
    pool: List[PoolProgram] = []
    pool_dir = participant_dir / "initial_pool_from_global"
    if pool_dir.is_dir() and any(pool_dir.glob("*.py")):
        try:
            cache = rescore_initial_pool_for_participant(
                participant_dir,
                dataset=dataset,
                split_ratio=split_ratio,
                split_seed=split_seed,
                n_eval_seeds=n_eval_seeds,
                psych_dataset_split=psych_dataset_split,
                evolution_selection_score=evolution_selection_score,
                cache_path=rescore_cache_path,
                force_rescore=force_rescore,
            )
            for prog_meta in cache.get("programs") or []:
                if not isinstance(prog_meta, dict) or not prog_meta.get("ok"):
                    if stats is not None and isinstance(prog_meta, dict):
                        stats.warnings.append(
                            f"{participant_dir.name}: rescore failed for "
                            f"{prog_meta.get('program_id')}: {prog_meta.get('error')}"
                        )
                    continue
                sel = _safe_float(prog_meta.get("selection_score"))
                if sel is None:
                    continue
                filename = str(prog_meta.get("filename") or "")
                code_path = pool_dir / filename
                if not code_path.is_file():
                    continue
                pool.append(
                    PoolProgram(
                        program_id=str(prog_meta.get("program_id")),
                        selection_score=sel,
                        code=code_path.read_text(encoding="utf-8"),
                        train_loglik=_safe_float(prog_meta.get("train_loglik")),
                        val_loglik=_safe_float(prog_meta.get("val_loglik")),
                    )
                )
                if stats is not None:
                    stats.initial_pool_rescored += 1
            # Optional sanity check vs results.json baseline (same split).
            results_path = participant_dir / "results.json"
            if results_path.is_file():
                baseline = _read_json(results_path).get("baseline") or {}
                base_sel = _safe_float(baseline.get("selection_score"))
                gb = next(
                    (p for p in pool if p.program_id in ("global_baseline", "baseline")),
                    None,
                )
                if (
                    base_sel is not None
                    and gb is not None
                    and abs(gb.selection_score - base_sel) > 1e-5
                ):
                    if stats is not None:
                        stats.warnings.append(
                            f"{participant_dir.name}: rescored global_baseline "
                            f"({gb.selection_score:.6f}) != results.json baseline "
                            f"({base_sel:.6f}); check split_seed/ratio"
                        )
        except Exception as exc:
            if stats is not None:
                stats.warnings.append(
                    f"{participant_dir.name}: initial_pool rescore failed "
                    f"({type(exc).__name__}: {exc}); falling back to results.json baseline only"
                )
            pool = []

    if not pool:
        # Fallback when no initial pool / rescore unavailable.
        results_path = participant_dir / "results.json"
        if results_path.is_file():
            results = _read_json(results_path)
            baseline = (
                results.get("baseline") if isinstance(results.get("baseline"), dict) else {}
            )
            sel = _safe_float(baseline.get("selection_score"))
            code_path = participant_dir / "initial_pool_from_global" / "000_global_baseline.py"
            code = (
                code_path.read_text(encoding="utf-8")
                if code_path.is_file()
                else "def choose (problem ,history ):\n    return 0.5\n"
            )
            if sel is not None:
                pool.append(
                    PoolProgram(
                        program_id="global_baseline",
                        selection_score=sel,
                        code=code,
                        train_loglik=_safe_float(baseline.get("train_loglik")),
                        val_loglik=_safe_float(baseline.get("val_loglik")),
                    )
                )

    explore_metrics = participant_dir / "explore_phase" / "metrics.json"
    explore_cand_dir = participant_dir / "explore_phase" / "candidates"
    if include_explore and explore_metrics.is_file():
        metrics = _read_json(explore_metrics)
        for row in metrics.get("candidate_results") or []:
            if not isinstance(row, dict) or not row.get("runtime_valid"):
                continue
            sel = _safe_float(row.get("selection_score"))
            if sel is None:
                sel = _safe_float(row.get("fitness"))
            if sel is None:
                continue
            idx = int(row["idx"])
            code = _load_candidate_code(explore_cand_dir, idx)
            if code is None:
                if stats is not None:
                    stats.skipped_missing_code += 1
                continue
            pool.append(
                PoolProgram(
                    program_id=f"explore_candidate_{idx}",
                    selection_score=sel,
                    code=code,
                    train_loglik=_safe_float(row.get("train_loglik")),
                    val_loglik=_safe_float(row.get("val_loglik")),
                )
            )

    if stats is not None and not pool:
        stats.warnings.append(
            f"{participant_dir.name}: empty seed pool (no baseline/explore RV)"
        )

    return _truncate_pool(pool, elite_pool_size)


def reconstruct_participant_records(
    participant_dir: Path,
    *,
    dataset: str,
    run_id: str,
    split_seed: int = 0,
    split_ratio: float = 0.6,
    n_eval_seeds: int = 3,
    psych_dataset_split: str = "train",
    evolution_selection_score: str = "train_val",
    elite_pool_size: int = 50,
    rescore_cache_path: Optional[Path] = None,
    force_rescore: bool = False,
    stats: Optional[ReconstructStats] = None,
) -> List[Dict[str, Any]]:
    """Return iteration_context + candidate records (explore + evolution)."""
    m = _PARTICIPANT_DIR_RE.match(participant_dir.name)
    if not m:
        raise ValueError(f"Not a participant dir: {participant_dir}")
    participant_id = int(m.group(1))

    # Pre-explore pool (initial_pool / baseline only). Explore is emitted next.
    pool = seed_pool_from_participant(
        participant_dir,
        elite_pool_size=elite_pool_size,
        dataset=dataset,
        split_ratio=split_ratio,
        split_seed=split_seed,
        n_eval_seeds=n_eval_seeds,
        psych_dataset_split=psych_dataset_split,
        evolution_selection_score=evolution_selection_score,
        rescore_cache_path=rescore_cache_path,
        force_rescore=force_rescore,
        stats=stats,
        include_explore=False,
    )
    baseline = _baseline_from_pool_or_results(pool, participant_dir)
    records: List[Dict[str, Any]] = []

    # ---- Explore phase (participant-level MEM start) ----
    explore_metrics_path = participant_dir / "explore_phase" / "metrics.json"
    explore_cand_dir = participant_dir / "explore_phase" / "candidates"
    if explore_metrics_path.is_file() and baseline is not None:
        explore_metrics = _read_json(explore_metrics_path)
        evo_sel = str(
            explore_metrics.get("evolution_selection_score") or evolution_selection_score
        )
        # Default TEH explore: seed/baseline only. Population/handoff parents are
        # not recoverable per-candidate in old metrics (no parent_id per idx).
        # If initial_pool_from_global with handoff explore parents existed, we would
        # need per-candidate parent mapping; absent that, keep seed_baseline.
        handoff_explore = bool(explore_metrics.get("initial_pool_from_global")) and bool(
            explore_metrics.get("explore_used_population_parents")
        )
        if handoff_explore:
            ref_type = REF_POPULATION_PROGRAM
            # Without per-idx parent map, fall back to warning + baseline exactness flag.
            if stats is not None:
                stats.warnings.append(
                    f"{participant_dir.name}: explore marked population parents but "
                    "per-candidate parent ids absent; using baseline score as fallback"
                )
            ref_prog = baseline
            ref_exact = False
        else:
            ref_type = REF_SEED_BASELINE
            ref_prog = baseline
            ref_exact = True

        ctx = build_iteration_context_record(
            dataset=dataset,
            participant_id=participant_id,
            run_id=run_id,
            split_seed=split_seed,
            phase="explore",
            iteration=0,
            evolution_selection_score=evo_sel,
            selected_parents=[_parent_record(ref_prog)],
            best_selected_parent_id=ref_prog.program_id,
        )
        ctx["reference_kind"] = ref_type
        ctx["reference_type"] = ref_type
        if record_contains_test_metrics(ctx):
            raise RuntimeError("explore iteration_context unexpectedly contains test metrics")
        records.append(ctx)

        selected_for_pool: List[PoolProgram] = []
        for row in explore_metrics.get("candidate_results") or []:
            if not isinstance(row, dict) or not row.get("runtime_valid"):
                continue
            idx = int(row["idx"])
            sel = _safe_float(row.get("selection_score"))
            if sel is None:
                sel = _safe_float(row.get("fitness"))
            code = _load_candidate_code(explore_cand_dir, idx)
            if code is None:
                if stats is not None:
                    stats.skipped_missing_code += 1
                continue
            candidate_id = f"explore_candidate_{idx}"
            train_ll = _safe_float(row.get("train_loglik"))
            val_ll = _safe_float(row.get("val_loglik"))
            delta = compute_delta_f(sel, ref_prog.selection_score)
            cand = build_candidate_record(
                dataset=dataset,
                participant_id=participant_id,
                run_id=run_id,
                split_seed=split_seed,
                phase="explore",
                iteration=0,
                candidate_id=candidate_id,
                candidate_idx=idx,
                source="explore",
                code=code,
                runtime_valid=True,
                train_loglik=train_ll,
                val_loglik=val_ll,
                selection_score=sel,
                reference_parent_id=ref_prog.program_id,
                reference_parent_score=ref_prog.selection_score,
                reference_kind=ref_type,
                reference_type=ref_type,
                reference_id=ref_prog.program_id,
                reference_is_exact=ref_exact,
                delta_f=delta,
                survived_elite_truncation=True,  # updated after truncate
                evolution_selection_score=evo_sel,
                prompted_parent_ids=[ref_prog.program_id],
            )
            if sel is not None:
                selected_for_pool.append(
                    PoolProgram(
                        program_id=candidate_id,
                        selection_score=sel,
                        code=code,
                        train_loglik=train_ll,
                        val_loglik=val_ll,
                    )
                )
            records.append(cand)
            if stats is not None:
                stats.candidates_written += 1
                stats.runtime_valid_written += 1
                stats.explore_runtime_valid_written += 1

        selected_for_pool.sort(key=lambda p: p.selection_score, reverse=True)
        pool = _truncate_pool(pool + selected_for_pool, elite_pool_size)
        survived_ids = {p.program_id for p in pool}
        for rec in records:
            if rec.get("phase") == "explore" and rec.get("record_type") == "candidate":
                rec["survived_elite_truncation"] = rec.get("candidate_id") in survived_ids
    elif explore_metrics_path.is_file() and baseline is None:
        if stats is not None:
            stats.warnings.append(
                f"{participant_dir.name}: explore present but baseline unrecovered; "
                "skipping explore MEM emission"
            )

    # ---- Evolution iterations ----
    for iteration, iter_dir in list_iteration_dirs(participant_dir):
        metrics_path = iter_dir / "metrics.json"
        if not metrics_path.is_file():
            if stats is not None:
                stats.warnings.append(
                    f"{participant_dir.name}/iteration_{iteration}: missing metrics.json"
                )
            continue
        metrics = _read_json(metrics_path)
        evo_sel = str(metrics.get("evolution_selection_score") or evolution_selection_score)
        cand_dir = iter_dir / "candidates"
        rows = metrics.get("candidate_results") or []
        if not isinstance(rows, list):
            continue

        pool_ref = _pool_best(pool)
        if pool_ref is None:
            if stats is not None:
                stats.warnings.append(
                    f"{participant_dir.name}/iteration_{iteration}: no pool reference; skipping"
                )
            continue

        # Context lists pool-best for annotation of normal candidates; fresh uses baseline.
        ctx_parents = [_parent_record(pool_ref)]
        if baseline is not None and baseline.program_id != pool_ref.program_id:
            ctx_parents.append(_parent_record(baseline))
        ctx = build_iteration_context_record(
            dataset=dataset,
            participant_id=participant_id,
            run_id=run_id,
            split_seed=split_seed,
            phase="evolution",
            iteration=iteration,
            evolution_selection_score=evo_sel,
            selected_parents=ctx_parents,
            best_selected_parent_id=pool_ref.program_id,
        )
        ctx["reference_kind"] = REF_POOL_BEST_PROXY
        ctx["reference_type"] = REF_POOL_BEST_PROXY
        if record_contains_test_metrics(ctx):
            raise RuntimeError("iteration_context unexpectedly contains test metrics")
        records.append(ctx)

        selected_for_pool = []
        pending_cand_records: List[Dict[str, Any]] = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            if not bool(row.get("runtime_valid")):
                continue
            idx = int(row["idx"])
            sel = _safe_float(row.get("selection_score"))
            if sel is None:
                sel = _safe_float(row.get("fitness"))
            code = _load_candidate_code(cand_dir, idx)
            if code is None:
                if stats is not None:
                    stats.skipped_missing_code += 1
                continue
            source = str(row.get("source") or "normal")
            candidate_id = f"iteration_{iteration}_candidate_{idx}"
            train_ll = _safe_float(row.get("train_loglik"))
            val_ll = _safe_float(row.get("val_loglik"))

            if source == "fresh":
                if baseline is None:
                    if stats is not None:
                        stats.warnings.append(
                            f"{participant_dir.name}/{candidate_id}: fresh but no baseline; "
                            "falling back to pool_best_proxy"
                        )
                    ref_type = REF_POOL_BEST_PROXY
                    ref_prog = pool_ref
                    ref_exact = False
                else:
                    ref_type = REF_SEED_BASELINE
                    ref_prog = baseline
                    ref_exact = True
            else:
                # Exact prompted parents not stored in old metrics/prompt_diagnostics.
                ref_type = REF_POOL_BEST_PROXY
                ref_prog = pool_ref
                ref_exact = False

            delta = compute_delta_f(sel, ref_prog.selection_score)
            pending_cand_records.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_idx": idx,
                    "source": source,
                    "code": code,
                    "runtime_valid": True,
                    "train_loglik": train_ll,
                    "val_loglik": val_ll,
                    "selection_score": sel,
                    "delta_f": delta,
                    "ref_type": ref_type,
                    "ref_prog": ref_prog,
                    "ref_exact": ref_exact,
                }
            )
            if sel is not None:
                selected_for_pool.append(
                    PoolProgram(
                        program_id=candidate_id,
                        selection_score=sel,
                        code=code,
                        train_loglik=train_ll,
                        val_loglik=val_ll,
                    )
                )
                if stats is not None and source == "fresh":
                    stats.fresh_added_to_pool += 1

        selected_for_pool.sort(key=lambda p: p.selection_score, reverse=True)
        pool = _truncate_pool(pool + selected_for_pool, elite_pool_size)
        survived_ids = {p.program_id for p in pool}

        for pending in pending_cand_records:
            ref_prog = pending["ref_prog"]
            ref_type = pending["ref_type"]
            cand = build_candidate_record(
                dataset=dataset,
                participant_id=participant_id,
                run_id=run_id,
                split_seed=split_seed,
                phase="evolution",
                iteration=iteration,
                candidate_id=pending["candidate_id"],
                candidate_idx=pending["candidate_idx"],
                source=pending["source"],
                code=pending["code"],
                runtime_valid=True,
                train_loglik=pending["train_loglik"],
                val_loglik=pending["val_loglik"],
                selection_score=pending["selection_score"],
                reference_parent_id=ref_prog.program_id,
                reference_parent_score=ref_prog.selection_score,
                reference_kind=ref_type,
                reference_type=ref_type,
                reference_id=ref_prog.program_id,
                reference_is_exact=pending["ref_exact"],
                delta_f=pending["delta_f"],
                survived_elite_truncation=pending["candidate_id"] in survived_ids,
                evolution_selection_score=evo_sel,
                prompted_parent_ids=[ref_prog.program_id] if pending["ref_exact"] else None,
            )
            if record_contains_test_metrics(cand):
                raise RuntimeError("candidate record unexpectedly contains test metrics")
            records.append(cand)
            if stats is not None:
                stats.candidates_written += 1
                stats.runtime_valid_written += 1
                if pending["source"] == "fresh":
                    stats.fresh_runtime_valid_written += 1

        if stats is not None:
            stats.iterations += 1

    return records


def write_participant_mem_trace(
    participant_dir: Path,
    records: Sequence[Dict[str, Any]],
    *,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    out = Path(output_path) if output_path is not None else mem_trace_path(participant_dir)
    if out.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing {out} (pass overwrite=True)")
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    for rec in records:
        append_mem_trace_record(out, dict(rec))
    return out


def reconstruct_run(
    run_dir: Path,
    *,
    dataset: Optional[str] = None,
    split_seed: Optional[int] = None,
    split_ratio: Optional[float] = None,
    n_eval_seeds: Optional[int] = None,
    psych_dataset_split: Optional[str] = None,
    evolution_selection_score: Optional[str] = None,
    elite_pool_size: Optional[int] = None,
    participant_ids: Optional[Sequence[int]] = None,
    output_run_dir: Optional[Path] = None,
    rescore_cache_dir: Optional[Path] = None,
    force_rescore: bool = False,
    overwrite: bool = False,
) -> ReconstructStats:
    """Reconstruct mem_trace.jsonl for participants under run_dir.

    If output_run_dir is set, writes
      output_run_dir/participant_K/mem_trace.jsonl
    instead of mutating the source run (recommended on full disks).
    Rescore caches go under rescore_cache_dir or output_run_dir (preferred) /
    participant initial_pool_from_global when writing in-place.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    stats = ReconstructStats()
    ds = dataset or infer_dataset_from_run_dir(run_dir)
    run_id = run_dir.name
    cmd_hparams = parse_split_hparams_from_command(run_dir / "log" / "command.txt")

    split_seed_i = int(cmd_hparams["split_seed"] if split_seed is None else split_seed)
    split_ratio_f = float(cmd_hparams["split_ratio"] if split_ratio is None else split_ratio)
    n_eval_seeds_i = int(cmd_hparams["n_eval_seeds"] if n_eval_seeds is None else n_eval_seeds)
    psych_split = str(
        cmd_hparams["psych_dataset_split"]
        if psych_dataset_split is None
        else psych_dataset_split
    )
    evo_sel = str(
        cmd_hparams["evolution_selection_score"]
        if evolution_selection_score is None
        else evolution_selection_score
    )

    if elite_pool_size is None:
        elite_pool_size = parse_elite_pool_size_from_command(run_dir / "log" / "command.txt")
    if elite_pool_size is None:
        elite_pool_size = 50
        stats.warnings.append("elite_pool_size not found in log/command.txt; defaulting to 50")

    wanted = set(int(x) for x in participant_ids) if participant_ids is not None else None
    for pdir in list_participant_dirs(run_dir):
        pid = int(_PARTICIPANT_DIR_RE.match(pdir.name).group(1))  # type: ignore[union-attr]
        if wanted is not None and pid not in wanted:
            continue
        if rescore_cache_dir is not None:
            cache_path = Path(rescore_cache_dir) / pdir.name / "initial_pool_rescore_cache.json"
        elif output_run_dir is not None:
            cache_path = (
                Path(output_run_dir) / pdir.name / "initial_pool_rescore_cache.json"
            )
        else:
            cache_path = (
                pdir / "initial_pool_from_global" / "participant_selection_scores.json"
            )
        records = reconstruct_participant_records(
            pdir,
            dataset=ds,
            run_id=run_id,
            split_seed=split_seed_i,
            split_ratio=split_ratio_f,
            n_eval_seeds=n_eval_seeds_i,
            psych_dataset_split=psych_split,
            evolution_selection_score=evo_sel,
            elite_pool_size=int(elite_pool_size),
            rescore_cache_path=cache_path,
            force_rescore=force_rescore,
            stats=stats,
        )
        if not records:
            stats.warnings.append(f"{pdir.name}: no records produced")
            continue
        if output_run_dir is not None:
            out_pdir = Path(output_run_dir) / pdir.name
            out_path = mem_trace_path(out_pdir)
        else:
            out_path = None
        write_participant_mem_trace(
            pdir,
            records,
            output_path=out_path,
            overwrite=overwrite,
        )
        stats.participants += 1
    return stats


def validate_artifacts_for_reconstruction(run_dir: Path) -> Dict[str, Any]:
    """Inspect whether an old TEH run has the fields needed for proxy MEM traces."""
    run_dir = Path(run_dir)
    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "ok": True,
        "blocking_issues": [],
        "warnings": [],
        "n_participants": 0,
        "elite_pool_size": parse_elite_pool_size_from_command(run_dir / "log" / "command.txt"),
        "dataset_guess": infer_dataset_from_run_dir(run_dir),
    }
    parts = list_participant_dirs(run_dir)
    report["n_participants"] = len(parts)
    if not parts:
        report["ok"] = False
        report["blocking_issues"].append("no participant_* directories")
        return report

    sample = parts[0]
    need_files = [
        sample / "results.json",
    ]
    for path in need_files:
        if not path.is_file():
            report["ok"] = False
            report["blocking_issues"].append(f"missing {path.relative_to(run_dir)}")

    iters = list_iteration_dirs(sample)
    if not iters:
        report["ok"] = False
        report["blocking_issues"].append(f"{sample.name}: no iteration_* dirs")
    else:
        _, idir = iters[0]
        metrics = idir / "metrics.json"
        cdir = idir / "candidates"
        if not metrics.is_file():
            report["ok"] = False
            report["blocking_issues"].append(f"missing {metrics.relative_to(run_dir)}")
        else:
            obj = _read_json(metrics)
            rows = obj.get("candidate_results") or []
            if not rows:
                report["ok"] = False
                report["blocking_issues"].append("candidate_results empty in first iteration")
            else:
                row0 = rows[0]
                for key in ("runtime_valid", "selection_score", "idx"):
                    if key not in row0:
                        report["ok"] = False
                        report["blocking_issues"].append(
                            f"candidate_results missing key {key!r}"
                        )
                # Ensure we never need test metrics
                report["has_test_metrics_in_artifacts"] = any(
                    k in row0 for k in ("test_loglik", "test_acc", "test_accuracy")
                )
        if not cdir.is_dir():
            report["ok"] = False
            report["blocking_issues"].append(f"missing {cdir.relative_to(run_dir)}")

    init_manifest = sample / "initial_pool_from_global" / "pool_manifest.json"
    if init_manifest.is_file():
        man = _read_json(init_manifest)
        progs = man.get("programs") or []
        has_participant_sel = any(
            isinstance(p, dict) and "selection_score" in p for p in progs
        )
        if not has_participant_sel:
            report["warnings"].append(
                "initial_pool_from_global lacks on-disk participant selection_score; "
                "reconstruction will rescore each program on the participant train/val "
                "split and cache the results"
            )
        report["n_initial_pool_programs"] = len(progs)
    else:
        report["warnings"].append("no initial_pool_from_global (baseline-only seed if no explore)")

    report["split_hparams"] = parse_split_hparams_from_command(run_dir / "log" / "command.txt")

    if not (sample / "explore_phase" / "metrics.json").is_file():
        report["warnings"].append("no explore_phase/metrics.json")

    report["ok"] = report["ok"] and not report["blocking_issues"]
    return report
