"""Handoff rules for exploring from population / initial-pool programs."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


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
