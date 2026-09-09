"""Canonical MEM generation-reference kinds (exact vs proxy).

Do not silently mix ΔF definitions across reference types in one regression
without explicit phase/reference factors or separate models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Exact generation references (recoverable from TEH generation path).
REF_SEED_BASELINE = "seed_baseline"
REF_POPULATION_PROGRAM = "population_program"
REF_BEST_PROMPTED_PARENT = "best_prompted_parent"

# Proxy when exact prompted parents cannot be recovered from old artifacts.
REF_POOL_BEST_PROXY = "pool_best_proxy"

# Legacy live-MEM alias (same meaning as best prompted parent among selected parents).
REF_BEST_SELECTED_PARENT = "best_selected_parent"

EXACT_REFERENCE_TYPES = frozenset(
    {
        REF_SEED_BASELINE,
        REF_POPULATION_PROGRAM,
        REF_BEST_PROMPTED_PARENT,
        REF_BEST_SELECTED_PARENT,
    }
)

PROXY_REFERENCE_TYPES = frozenset({REF_POOL_BEST_PROXY})

# Keep pool_best_proxy string for reconstruct / old tests.
REFERENCE_KIND_POOL_BEST_PROXY = REF_POOL_BEST_PROXY


def is_exact_reference(reference_type: Optional[str]) -> bool:
    return str(reference_type or "") in EXACT_REFERENCE_TYPES


def delta_f_field_for_reference(reference_type: str) -> str:
    """Primary named ΔF column for a reference_type (alongside generic ``delta_f``)."""
    mapping = {
        REF_SEED_BASELINE: "delta_f_vs_baseline",
        REF_POPULATION_PROGRAM: "delta_f_vs_population_program",
        REF_BEST_PROMPTED_PARENT: "delta_f_vs_best_prompted_parent",
        REF_BEST_SELECTED_PARENT: "delta_f_vs_best_prompted_parent",
        REF_POOL_BEST_PROXY: "delta_f_vs_pool_best",
    }
    return mapping.get(str(reference_type), "delta_f")


def enrich_candidate_reference_fields(
    rec: Dict[str, Any],
    *,
    reference_type: str,
    reference_id: Optional[str],
    reference_score: Optional[float],
    delta_f: Optional[float],
    reference_is_exact: Optional[bool] = None,
) -> Dict[str, Any]:
    """Attach explicit reference metadata; never overwrite conflicting types silently."""
    out = dict(rec)
    out["reference_type"] = reference_type
    out["reference_kind"] = reference_type  # backward-compatible alias
    out["reference_id"] = reference_id
    out["reference_parent_id"] = reference_id  # annotate/build still read this
    out["reference_parent_score"] = reference_score
    out["reference_score"] = reference_score
    exact = (
        bool(reference_is_exact)
        if reference_is_exact is not None
        else is_exact_reference(reference_type)
    )
    out["reference_is_exact"] = exact
    out["reference_is_proxy"] = not exact
    out["delta_f"] = delta_f
    named = delta_f_field_for_reference(reference_type)
    out[named] = delta_f
    return out
