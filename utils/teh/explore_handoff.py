"""Handoff rules for exploring from population / initial-pool programs."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from utils.teh.sparse_observations import distribute_explore_budget


def resolve_explore_from_handoff_parents(
    *,
    has_initial_pool: bool,
    explore_from_population_parents: bool,
) -> bool:
    """Whether pre-evolution explore should generate from handoff programs.

    ``--initial_pool_programs`` already explores from those files. Live
    ``--global_phase`` stays seed-only unless ``--explore_from_population_parents``.
    """
    return bool(has_initial_pool or explore_from_population_parents)


def select_explore_handoff_parents(
    elite_parents: Sequence[Tuple[Any, ...]],
    *,
    enabled: bool,
    top_k: int = 0,
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[List[str]]]:
    """Return (code, program_id) pairs and ids to pin.

    ``top_k <= 0`` keeps global/handoff order for the full pool. ``top_k == 1``
    is the Stage B pattern: sole parent = rank-1 population program.
    """
    if not enabled or not elite_parents:
        return None, None
    chosen = list(elite_parents)
    k = int(top_k)
    if k > 0:
        chosen = chosen[:k]
    programs = [(str(p[0]), str(p[3])) for p in chosen]
    pinned = [str(p[3]) for p in chosen]
    return programs, pinned


def split_explore_budget_seed_and_parents(
    n_explore: int,
    *,
    n_seed: int,
    n_handoff_parents: int,
) -> Tuple[int, List[int]]:
    """Split explore budget into seed-parent vs handoff-parent counts.

    ``n_seed`` candidates are generated from the vanilla seed. The remainder
    is split across handoff parents (no multiplication). ``n_seed=0`` keeps
    the Stage B/D default: all explore cands from handoff parents when those
    exist.
    """
    n_explore = max(0, int(n_explore))
    n_handoff_parents = max(0, int(n_handoff_parents))
    n_seed = max(0, min(int(n_seed), n_explore))
    n_parent_total = n_explore - n_seed
    if n_handoff_parents <= 0:
        return n_explore, []
    return n_seed, distribute_explore_budget(n_parent_total, n_handoff_parents)


def prefix_elite_program_ids(
    elite_parents: Sequence[Tuple[Any, ...]],
    prefix: str,
) -> List[Tuple[Any, ...]]:
    """Disambiguate ids when concatenating two population pools."""
    tagged: List[Tuple[Any, ...]] = []
    for parent in elite_parents:
        row = list(parent)
        row[3] = f"{prefix}{parent[3]}"
        tagged.append(tuple(row))
    return tagged
