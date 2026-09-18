"""Reconstruct parent→candidate lineage for T-PICS global_phase runs.

Live ``--mem_trace`` (default on) writes ``global_phase/mem_trace.jsonl`` from
``run_global_evolution_phase`` and ``participant_*/mem_trace.jsonl`` from
``run_evolution``. Use this module for older runs that never wrote a live
global trace.

Reference pairing recovered from artifacts:

  * ``source=fresh`` → ``global_baseline`` (seed) — always **exact**
    (``reference_type=seed_baseline``).
  * ``source=normal`` → iteration-start pool-best.

Pool-best is the true ``best_prompted_parent`` **only when the generation
path guarantees elite index 0 is among selected parents** (``best_k >= 1`` in
``_select_parent_indices_from_elite_pool``). That is **not** always true:

  * ``sample_parents`` + ``sampled_parents_decay``: ``best_k == 0`` on
    ``iter_idx == 0`` (and whenever the decay leaves zero reserved best slots).
  * ``sample_parents`` without decay: ``best_k == 0`` every iteration.

When the guarantee fails, outputs mark
``reference_type=pool_best_proxy``, ``reference_is_exact=false``,
``reference_is_proxy=true``. Primary fitness-effect analyses must use exact
rows only; proxy rows are sensitivity.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.mem.reference_types import (
    REF_BEST_PROMPTED_PARENT,
    REF_POOL_BEST_PROXY,
    REF_SEED_BASELINE,
)


def candidate_code_path(gp: Path, iteration: int, candidate_idx: int) -> Path:
    return gp / f"iteration_{iteration}" / "candidates" / f"candidate_{candidate_idx}.py"


def parse_program_id(program_id: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.fullmatch(r"global_iteration_(\d+)_candidate_(\d+)", str(program_id))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def resolve_parent_code_path(run_dir: Path, parent_id: str) -> Optional[Path]:
    run_dir = Path(run_dir)
    gp = run_dir / "global_phase"
    if parent_id in ("global_baseline", "baseline"):
        seed = run_dir / "prompts" / "seed_program.py"
        return seed if seed.is_file() else None
    it, idx = parse_program_id(parent_id)
    if it is None or idx is None:
        return None
    p = candidate_code_path(gp, it, idx)
    return p if p.is_file() else None


def _decayed_sampled_parents_k(num_parents: int, iter_idx: int, total_iters: int) -> int:
    num_parents = int(num_parents)
    if num_parents <= 0:
        return 0
    total = max(1, int(total_iters))
    idx = max(0, int(iter_idx))
    raw = math.floor(float(num_parents) * (1.0 - idx / total))
    return max(0, min(int(raw), num_parents))


def best_k_for_iteration(
    *,
    pool_size: int,
    sample_size: int = 8,
    sample_parents: bool = True,
    sampled_parents_decay: bool = True,
    iter_idx: int,
    total_iters: int = 10,
) -> Tuple[int, int, int]:
    """Mirror teh._select_parent_indices_from_elite_pool counts (no RNG)."""
    num_parents = min(int(sample_size), int(pool_size))
    if num_parents <= 0:
        return 0, 0, 0
    if not sample_parents:
        return num_parents, num_parents, 0
    if sampled_parents_decay:
        sampled_k = _decayed_sampled_parents_k(num_parents, iter_idx, total_iters)
    else:
        sampled_k = num_parents
    best_k = num_parents - sampled_k
    return num_parents, best_k, sampled_k


def pool_best_guaranteed_in_prompted(*, best_k: int) -> bool:
    """Elite index 0 is always selected iff best_k >= 1."""
    return int(best_k) >= 1


def reconstruct_iteration_parents(
    gp: Path,
    n_iters: int = 10,
    *,
    sample_size: int = 8,
    elite_pool_size: int = 50,
    sample_parents: bool = True,
    sampled_parents_decay: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Per-iteration parent reference metadata from metrics.json + selection math."""
    gp = Path(gp)
    out: Dict[int, Dict[str, Any]] = {}
    prev_pool_best = "global_baseline"
    pool_size = 1  # global_baseline only at start
    for it in range(1, n_iters + 1):
        metrics_path = gp / f"iteration_{it}" / "metrics.json"
        if not metrics_path.is_file():
            break
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        sources = list(m.get("candidate_sources") or [])
        iter_idx = int(m.get("iter_idx", it - 1))
        total_iters = int(m.get("total_iterations", n_iters))
        num_parents, best_k, sampled_k = best_k_for_iteration(
            pool_size=pool_size,
            sample_size=sample_size,
            sample_parents=sample_parents,
            sampled_parents_decay=sampled_parents_decay,
            iter_idx=iter_idx,
            total_iters=total_iters,
        )
        guaranteed = pool_best_guaranteed_in_prompted(best_k=best_k)
        out[it] = {
            "iteration": it,
            "candidate_sources": sources,
            "pool_best_at_start": prev_pool_best,
            "pool_best_after": m.get("pool_best_program_id"),
            "n_runtime_valid": int(m.get("n_runtime_valid") or 0),
            "fresh_n": int(m.get("fresh_n") or 0),
            "n_normal_candidates": int(m.get("n_normal_candidates") or 0),
            "pool_size_at_start": pool_size,
            "num_parents": num_parents,
            "best_k": best_k,
            "sampled_k": sampled_k,
            "pool_best_guaranteed_in_prompted": guaranteed,
            "iter_idx": iter_idx,
            "total_iterations": total_iters,
        }
        if m.get("pool_best_program_id"):
            prev_pool_best = str(m["pool_best_program_id"])
        # Elite grows by runtime-valid candidates then truncates (same as TEH).
        n_valid = int(m.get("n_runtime_valid") or 0)
        elite_cap = max(int(sample_size), int(elite_pool_size))
        pool_size = min(elite_cap, pool_size + n_valid)
    return out


def parent_for_candidate(
    *,
    iteration: int,
    candidate_idx: int,
    iter_meta: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Return parent id + exact/proxy reference flags for one candidate."""
    meta = iter_meta[int(iteration)]
    sources = meta["candidate_sources"]
    if candidate_idx < 0 or candidate_idx >= len(sources):
        raise KeyError(
            f"candidate_idx={candidate_idx} out of range for iteration {iteration} "
            f"(n_sources={len(sources)})"
        )
    source = str(sources[candidate_idx])
    if source == "fresh":
        return {
            "parent_id": "global_baseline",
            "source": "fresh",
            "reference_kind": REF_SEED_BASELINE,
            "reference_type": REF_SEED_BASELINE,
            "reference_is_exact": True,
            "reference_is_proxy": False,
        }
    if source == "normal":
        parent_id = str(meta["pool_best_at_start"])
        guaranteed = bool(meta.get("pool_best_guaranteed_in_prompted"))
        if guaranteed:
            return {
                "parent_id": parent_id,
                "source": "normal",
                "reference_kind": REF_BEST_PROMPTED_PARENT,
                "reference_type": REF_BEST_PROMPTED_PARENT,
                "reference_is_exact": True,
                "reference_is_proxy": False,
            }
        return {
            "parent_id": parent_id,
            "source": "normal",
            "reference_kind": REF_POOL_BEST_PROXY,
            "reference_type": REF_POOL_BEST_PROXY,
            "reference_is_exact": False,
            "reference_is_proxy": True,
        }
    raise ValueError(f"unknown candidate source {source!r}")


def write_synthetic_mem_trace(
    run_dir: Path,
    *,
    dataset: str,
    run_id: str,
    out_path: Path,
    invalid_ids_by_iter: Optional[Dict[int, set]] = None,
    n_iters: int = 10,
    split_seed: int = 0,
    repo: Optional[Path] = None,
    sample_size: int = 8,
    elite_pool_size: int = 50,
    sample_parents: bool = True,
    sampled_parents_decay: bool = True,
) -> int:
    """Write slim reconstructed mem_trace JSONL (no embedded code / ΔF)."""
    from utils.mem.trace import build_candidate_record, build_iteration_context_record

    run_dir = Path(run_dir)
    gp = run_dir / "global_phase"
    repo = Path(repo) if repo is not None else None
    iter_meta = reconstruct_iteration_parents(
        gp,
        n_iters=n_iters,
        sample_size=sample_size,
        elite_pool_size=elite_pool_size,
        sample_parents=sample_parents,
        sampled_parents_decay=sampled_parents_decay,
    )
    invalid_ids_by_iter = invalid_ids_by_iter or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for it, meta in sorted(iter_meta.items()):
            parent_id = str(meta["pool_best_at_start"])
            ctx = build_iteration_context_record(
                dataset=dataset,
                participant_id="global",
                run_id=run_id,
                split_seed=int(split_seed),
                phase="global_evolution",
                iteration=it,
                evolution_selection_score="train_val",
                selected_parents=[
                    {
                        "program_id": parent_id,
                        "selection_score": None,
                        "train_loglik": None,
                        "val_loglik": None,
                    }
                ],
                best_selected_parent_id=parent_id,
            )
            ctx["lineage_reconstructed"] = True
            ctx["pool_best_guaranteed_in_prompted"] = meta.get(
                "pool_best_guaranteed_in_prompted"
            )
            f.write(json.dumps(ctx, ensure_ascii=False) + "\n")
            n += 1
            bad = invalid_ids_by_iter.get(it, set())
            for idx, source in enumerate(meta["candidate_sources"]):
                cid_short = f"candidate_{idx}"
                if cid_short in bad:
                    continue
                code_path = candidate_code_path(gp, it, idx)
                if not code_path.is_file():
                    continue
                pref = parent_for_candidate(
                    iteration=it, candidate_idx=idx, iter_meta=iter_meta
                )
                rel = (
                    str(code_path.relative_to(repo))
                    if repo is not None
                    else str(code_path)
                )
                rec = build_candidate_record(
                    dataset=dataset,
                    participant_id="global",
                    run_id=run_id,
                    split_seed=int(split_seed),
                    phase="global_evolution",
                    iteration=it,
                    candidate_id=f"global_iteration_{it}_candidate_{idx}",
                    candidate_idx=idx,
                    source=str(source),
                    code="",
                    runtime_valid=True,
                    train_loglik=None,
                    val_loglik=None,
                    selection_score=None,
                    reference_parent_id=pref["parent_id"],
                    reference_parent_score=None,
                    reference_kind=pref["reference_kind"],
                    reference_type=pref["reference_type"],
                    reference_id=pref["parent_id"],
                    reference_is_exact=pref["reference_is_exact"],
                    delta_f=None,
                    survived_elite_truncation=False,
                    evolution_selection_score="train_val",
                    prompted_parent_ids=[pref["parent_id"]],
                    embed_code=False,
                    code_path=rel,
                )
                rec["lineage_reconstructed"] = True
                rec["reference_is_proxy"] = pref["reference_is_proxy"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n
