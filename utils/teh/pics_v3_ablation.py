"""PICS v3 ablation helpers (flags default off; main pics_v3 unchanged)."""
from __future__ import annotations

from typing import Any, Optional

from utils.teh.pics_v3 import (
    ABLATION_KIND_NO_ADAPTIVE_PROMPT,
    ABLATION_KIND_NO_EXPLORE,
    ABLATION_KIND_NO_FRESH,
    ABLATION_KIND_NO_POPULATION,
    ABLATION_KIND_NO_TRANSFER,
    ABLATION_KINDS,
    ABLATION_RUN_TAG,
)

ABLATION_WANDB_GROUP = "t_pics_gated_ablation"
ABLATION_WANDB_TAGS = ("ICLR", "SA40", "gated_t_pics", "ablation")

# Nominal 35-round phase budgets (10 cand/iter; explore 50 ≡ 5 rounds).
ABLATION_PHASE_BUDGETS = {
    "no_transfer": {
        "kind": ABLATION_KIND_NO_TRANSFER,
        "global_iters": 5,
        "n_g2_arms": 1,
        "explore_candidates": 50,
        "n_iterations": 25,
        "fresh_n_candidates": 10,
        "control_only": True,
        "ablate_population": False,
        "ablate_adaptive_prompt": False,
        "live_independent_source": False,
    },
    "no_population": {
        "kind": ABLATION_KIND_NO_POPULATION,
        "global_iters": 0,
        "n_g2_arms": 0,
        "explore_candidates": 50,
        "n_iterations": 30,
        "fresh_n_candidates": 10,
        "control_only": False,
        "ablate_population": True,
        "ablate_adaptive_prompt": False,
        "live_independent_source": False,
    },
    "no_explore": {
        "kind": ABLATION_KIND_NO_EXPLORE,
        "global_iters": 5,
        "n_g2_arms": 2,
        "explore_candidates": 0,
        "n_iterations": 15,
        "fresh_n_candidates": 10,
        "control_only": False,
        "ablate_population": False,
        "ablate_adaptive_prompt": False,
        "live_independent_source": False,
    },
    "no_fresh": {
        "kind": ABLATION_KIND_NO_FRESH,
        "global_iters": 5,
        "n_g2_arms": 2,
        "explore_candidates": 50,
        "n_iterations": 10,
        "fresh_n_candidates": 0,
        "control_only": False,
        "ablate_population": False,
        "ablate_adaptive_prompt": False,
        "live_independent_source": True,
    },
    "no_adaptive_prompt": {
        "kind": ABLATION_KIND_NO_ADAPTIVE_PROMPT,
        "global_iters": 5,
        "n_g2_arms": 2,
        "explore_candidates": 50,
        "n_iterations": 10,
        "fresh_n_candidates": 10,
        "control_only": False,
        "ablate_population": False,
        "ablate_adaptive_prompt": True,
        "live_independent_source": True,
    },
}


def ablation_id_from_args(args: Any) -> Optional[str]:
    """Return canonical ablation id when an ablation flag is set; else None."""
    if bool(getattr(args, "t_pics_ablate_population", False)):
        return "no_population"
    if bool(getattr(args, "t_pics_gated_control_only", False)):
        return "no_transfer"
    if bool(getattr(args, "ablate_dataset_adaptive_prompt", False)):
        return "no_adaptive_prompt"
    explore = int(getattr(args, "explore_candidates", 0) or 0)
    fresh = int(getattr(args, "fresh_n_candidates", 0) or 0)
    n_iters = int(getattr(args, "n_iterations", 0) or 0)
    if bool(getattr(args, "t_pics_gated_transfer", False)) and explore == 0 and n_iters == 15:
        return "no_explore"
    if bool(getattr(args, "t_pics_gated_transfer", False)) and fresh == 0 and n_iters == 10 and explore == 50:
        # Prefer explicit KIND when launcher sets it; else infer no_fresh.
        kind = str(getattr(args, "output_kind", "") or "")
        if kind == ABLATION_KIND_NO_FRESH or kind.endswith("no_fresh"):
            return "no_fresh"
        if "no_fresh" in kind:
            return "no_fresh"
    kind = str(getattr(args, "kind", "") or getattr(args, "output_kind", "") or "")
    for ablation_id, spec in ABLATION_PHASE_BUDGETS.items():
        if kind == spec["kind"]:
            return ablation_id
    return None


def ablation_metadata_payload(args: Any) -> dict:
    """Resolved phase/candidate counts for run config / W&B (ablation runs only)."""
    ablation_id = ablation_id_from_args(args)
    spec = ABLATION_PHASE_BUDGETS.get(ablation_id or "", {})
    return {
        "ablation_id": ablation_id,
        "ablation_kind": spec.get("kind"),
        "ablation_run_tag": ABLATION_RUN_TAG,
        "control_only": bool(getattr(args, "t_pics_gated_control_only", False)),
        "ablate_population": bool(getattr(args, "t_pics_ablate_population", False)),
        "ablate_dataset_adaptive_prompt": bool(
            getattr(args, "ablate_dataset_adaptive_prompt", False)
        ),
        "reuse_gate_pool": str(getattr(args, "t_pics_reuse_gate_pool", "") or "") or None,
        "global_iters": int(getattr(args, "global_iters", 0) or 0),
        "explore_candidates": int(getattr(args, "explore_candidates", 0) or 0),
        "n_iterations": int(getattr(args, "n_iterations", 0) or 0),
        "fresh_n_candidates": int(getattr(args, "fresh_n_candidates", 0) or 0),
        "n_candidates": int(getattr(args, "n_candidates", 10) or 10),
        "known_ablation_kinds": list(ABLATION_KINDS),
    }
