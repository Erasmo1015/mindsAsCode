"""Exact source-SHA elite rules for the PICS-v4 population and participant pools."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

ELITE_POLICY_SHA256_UNIQUE_V1 = "sha256_unique_v1"
PARTICIPANT_POPULATION_UNIQUE_CAP = 5
ELITE_POOL_SIZE_IS_UPPER_BOUND = True
DUPLICATE_BACKFILL = False
POPULATION_PINNED_PROGRAM_IDS = ("global_baseline",)


def elite_policy_provenance() -> Dict[str, Any]:
    return {
        "elite_policy": ELITE_POLICY_SHA256_UNIQUE_V1,
        "participant_population_unique_cap": PARTICIPANT_POPULATION_UNIQUE_CAP,
        "elite_pool_size_is_upper_bound": ELITE_POOL_SIZE_IS_UPPER_BOUND,
        "duplicate_backfill": DUPLICATE_BACKFILL,
    }


def source_sha256(code: str) -> str:
    """SHA-256 of the program source. No stripping or other normalization."""
    import hashlib

    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()


def _elite_rank_key(parent: Sequence[Any], mdl_lambda: float):
    from utils.teh.mdl_selection import elite_rank_key

    return elite_rank_key(parent, mdl_lambda)


def dedupe_elite_exact_sha_aligned(
    elite_parents: Sequence[Sequence[Any]],
    *,
    elite_cap: int,
    pinned_ids: Sequence[str] = (),
    mdl_lambda: float = 0.0,
    companions: Optional[Sequence[Any]] = None,
) -> Tuple[List[Tuple[Any, ...]], Optional[List[Any]], Dict[str, int]]:
    """Sort, keep one exact source SHA, then cap.

    A pinned program id wins over another copy of the same source. Otherwise
    the better fitness wins, and equal fitness keeps the earlier entry. The
    cap is an upper bound: unused slots stay empty. Companion values follow
    the kept rows.
    """
    parents = [tuple(parent) for parent in elite_parents]
    if companions is None:
        comps: Optional[List[Any]] = None
    else:
        if len(companions) != len(parents):
            raise ValueError("companion length does not match the elite length")
        comps = list(companions)
    order = list(range(len(parents)))
    order.sort(key=lambda index: _elite_rank_key(parents[index], mdl_lambda), reverse=True)
    pinned = {str(item) for item in pinned_ids}
    winners_idx: List[int] = []
    index_by_sha: Dict[str, int] = {}
    for index in order:
        digest = source_sha256(str(parents[index][0] or ""))
        if digest not in index_by_sha:
            index_by_sha[digest] = len(winners_idx)
            winners_idx.append(index)
            continue
        kept_slot = index_by_sha[digest]
        kept_index = winners_idx[kept_slot]
        if str(parents[index][3]) in pinned and str(parents[kept_index][3]) not in pinned:
            winners_idx[kept_slot] = index
    winner_order = list(range(len(winners_idx)))
    winner_order.sort(
        key=lambda slot: _elite_rank_key(parents[winners_idx[slot]], mdl_lambda),
        reverse=True,
    )
    ordered_winners = [winners_idx[slot] for slot in winner_order]
    capped_idx = ordered_winners[: max(0, int(elite_cap))]
    capped = [parents[index] for index in capped_idx]
    capped_comps = [comps[index] for index in capped_idx] if comps is not None else None
    stats = {
        "pool_size": len(capped),
        "n_unique_sha": len({source_sha256(str(parent[0] or "")) for parent in capped}),
        "n_duplicate_removed": len(parents) - len(winners_idx),
        "n_unique_before_cap": len(winners_idx),
    }
    return capped, capped_comps, stats


def dedupe_elite_exact_sha(
    elite_parents: Sequence[Sequence[Any]],
    *,
    elite_cap: int,
    pinned_ids: Sequence[str] = (),
    mdl_lambda: float = 0.0,
) -> Tuple[List[Tuple[Any, ...]], Dict[str, int]]:
    capped, _companions, stats = dedupe_elite_exact_sha_aligned(
        elite_parents,
        elite_cap=elite_cap,
        pinned_ids=pinned_ids,
        mdl_lambda=mdl_lambda,
    )
    return capped, stats


def cap_participant_elite(
    elite_parents: Sequence[Sequence[Any]],
    *,
    elite_cap: int,
    pinned_ids: Sequence[str] = (),
    sha_unique: bool,
    mdl_lambda: float = 0.0,
    companions: Optional[Sequence[Any]] = None,
    sort: bool = True,
) -> Tuple[List[Tuple[Any, ...]], Optional[List[Any]], Optional[Dict[str, int]]]:
    """Cap one participant elite update.

    ``sha_unique`` is the PICS-v4 rule. The historical rule sorts, when asked,
    and slices. It keeps duplicate source SHAs.
    """
    if sha_unique:
        return dedupe_elite_exact_sha_aligned(
            elite_parents,
            elite_cap=elite_cap,
            pinned_ids=pinned_ids,
            mdl_lambda=mdl_lambda,
            companions=companions,
        )
    parents = [tuple(parent) for parent in elite_parents]
    if companions is None:
        comps: Optional[List[Any]] = None
    else:
        if len(companions) != len(parents):
            raise ValueError("companion length does not match the elite length")
        comps = list(companions)
    if sort:
        if comps is None:
            from utils.teh.mdl_selection import sort_elites

            sort_elites(parents, mdl_lambda)
        else:
            from utils.teh.mdl_selection import sort_elite_pairs

            paired = list(zip(parents, comps))
            sort_elite_pairs(paired, mdl_lambda)
            parents = [row[0] for row in paired]
            comps = [row[1] for row in paired]
    limited = max(0, int(elite_cap))
    parents = parents[:limited]
    if comps is not None:
        comps = comps[:limited]
    return parents, comps, None


def build_participant_handoff_pool(
    population: Sequence[Sequence[Any]],
    exploration: Sequence[Sequence[Any]],
    *,
    elite_cap: int = 50,
    population_unique_cap: int = PARTICIPANT_POPULATION_UNIQUE_CAP,
    population_rank1_id: Optional[str] = None,
    mdl_lambda: float = 0.0,
    population_vals: Optional[Sequence[Any]] = None,
    exploration_vals: Optional[Sequence[Any]] = None,
) -> Tuple[List[Tuple[Any, ...]], List[Any], List[str], Dict[str, Any]]:
    """Build the PICS-v4 participant elite after exploration.

    Population programs are already scored on this participant. The shared
    population rank-1 stays pinned. At most four other unique population
    programs are kept, by this participant's score. Unique exploration
    programs fill the remaining slots up to ``elite_cap`` and never replace a
    retained population SHA. Empty slots stay empty.
    """
    pop = [tuple(parent) for parent in population]
    explore = [tuple(parent) for parent in exploration]
    if population_vals is None:
        pop_vals: List[Any] = [None] * len(pop)
    else:
        if len(population_vals) != len(pop):
            raise ValueError("population companion length does not match")
        pop_vals = list(population_vals)
    if exploration_vals is None:
        explore_vals: List[Any] = [None] * len(explore)
    else:
        if len(exploration_vals) != len(explore):
            raise ValueError("exploration companion length does not match")
        explore_vals = list(exploration_vals)
    cap = max(0, int(elite_cap))
    origin_cap = max(0, int(population_unique_cap))
    rank1_id = str(
        population_rank1_id
        if population_rank1_id is not None
        else (pop[0][3] if pop else "")
    )

    def _empty_stats() -> Dict[str, Any]:
        return {
            "pool_size": 0,
            "n_unique_sha": 0,
            "n_population_retained": 0,
            "n_explore_retained": 0,
            "n_duplicate_removed": 0,
            "n_explore_skipped_population_sha": 0,
            "pinned_program_id": rank1_id or None,
        }

    if not pop and not explore:
        return [], [], [], _empty_stats()

    if not pop:
        kept, comps, stats = dedupe_elite_exact_sha_aligned(
            explore,
            elite_cap=cap,
            mdl_lambda=mdl_lambda,
            companions=explore_vals,
        )
        stats = dict(stats)
        stats["n_population_retained"] = 0
        stats["n_explore_retained"] = len(kept)
        stats["n_explore_skipped_population_sha"] = 0
        stats["pinned_program_id"] = None
        return kept, list(comps or []), [], stats

    pop_unique, pop_comps, pop_stats = dedupe_elite_exact_sha_aligned(
        pop,
        elite_cap=max(len(pop), 1),
        pinned_ids=(rank1_id,),
        mdl_lambda=mdl_lambda,
        companions=pop_vals,
    )
    pop_comps = list(pop_comps or [])
    rank1_pair: Optional[Tuple[Tuple[Any, ...], Any]] = None
    others: List[Tuple[Tuple[Any, ...], Any]] = []
    for parent, companion in zip(pop_unique, pop_comps):
        if str(parent[3]) == rank1_id and rank1_pair is None:
            rank1_pair = (parent, companion)
        else:
            others.append((parent, companion))
    if rank1_pair is None:
        raise RuntimeError(f"population rank-1 program {rank1_id!r} was not retained")
    additional_n = max(0, origin_cap - 1)
    population_kept = [rank1_pair, *others[:additional_n]]

    if explore:
        explore_unique, explore_comps, explore_stats = dedupe_elite_exact_sha_aligned(
            explore,
            elite_cap=len(explore),
            mdl_lambda=mdl_lambda,
            companions=explore_vals,
        )
        explore_comps = list(explore_comps or [])
    else:
        explore_unique, explore_comps, explore_stats = [], [], {
            "n_duplicate_removed": 0,
        }
    retained_sha = {source_sha256(str(parent[0] or "")) for parent, _companion in population_kept}
    explore_kept: List[Tuple[Tuple[Any, ...], Any]] = []
    skipped_overlap = 0
    for parent, companion in zip(explore_unique, explore_comps):
        digest = source_sha256(str(parent[0] or ""))
        if digest in retained_sha:
            skipped_overlap += 1
            continue
        if len(population_kept) + len(explore_kept) >= cap:
            break
        explore_kept.append((parent, companion))
        retained_sha.add(digest)
    combined = population_kept + explore_kept
    combined.sort(key=lambda row: _elite_rank_key(row[0], mdl_lambda), reverse=True)
    elites = [row[0] for row in combined]
    vals = [row[1] for row in combined]
    pins = [rank1_id] if any(str(parent[3]) == rank1_id for parent in elites) else []
    stats = {
        "pool_size": len(elites),
        "n_unique_sha": len({source_sha256(str(parent[0] or "")) for parent in elites}),
        "n_population_retained": len(population_kept),
        "n_explore_retained": len(explore_kept),
        "n_duplicate_removed": int(pop_stats["n_duplicate_removed"])
        + int(explore_stats["n_duplicate_removed"]),
        "n_explore_skipped_population_sha": skipped_overlap,
        "pinned_program_id": rank1_id if pins else None,
        "participant_population_unique_cap": origin_cap,
        "elite_pool_size_is_upper_bound": True,
        "duplicate_backfill": False,
    }
    return elites, vals, pins, stats
