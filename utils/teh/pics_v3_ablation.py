"""PICS v3 ablation helpers (flags default off; main pics_v3 unchanged).

Also hosts budget-allocation F/G/H metadata (separate KINDs from A–E).
"""
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
    ALLOCATION_KIND_F_TRANSFER_INIT,
    ALLOCATION_KIND_G_TARGET_POP_INIT,
    ALLOCATION_KIND_H_DIRECT_PERSON20,
    ALLOCATION_KINDS,
    ALLOCATION_RUN_TAG,
    ALLOCATION_WANDB_GROUP,
)

ABLATION_WANDB_GROUP = "t_pics_gated_ablation"
ABLATION_WANDB_TAGS = ("ICLR", "SA40", "gated_t_pics", "ablation")
ALLOCATION_WANDB_TAGS = ("ICLR", "SA40", "gated_t_pics", "budget_allocation")

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
        # Live independent G.1 + live dual G.2; no explore; no main-pool reuse.
        "live_independent_source": True,
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
        # Also disables HISTORY reminder injection (see setup_teh_run_prompts).
        "ablate_history_reminder": True,
        "live_independent_source": True,
    },
}

# Target-side stage-depth sum = 20 (F:10+10, G:10+10, H:0+20). Not equal compute.
ALLOCATION_PHASE_BUDGETS = {
    "F": {
        "kind": ALLOCATION_KIND_F_TRANSFER_INIT,
        "paper_label": "F",
        "global_iters": 10,
        "n_g2_arms": 1,
        "explore_candidates": 0,
        "n_iterations": 10,
        "fresh_n_candidates": 10,
        "control_only": False,
        "transfer_only": True,
        "ablate_population": False,
        "live_independent_source": False,
        "frozen_source": True,
    },
    "G": {
        "kind": ALLOCATION_KIND_G_TARGET_POP_INIT,
        "paper_label": "G",
        "global_iters": 10,
        "n_g2_arms": 1,
        "explore_candidates": 0,
        "n_iterations": 10,
        "fresh_n_candidates": 10,
        "control_only": True,
        "transfer_only": False,
        "ablate_population": False,
        "live_independent_source": False,
        "frozen_source": False,
    },
    "H": {
        "kind": ALLOCATION_KIND_H_DIRECT_PERSON20,
        "paper_label": "H",
        "global_iters": 0,
        "n_g2_arms": 0,
        "explore_candidates": 0,
        "n_iterations": 20,
        "fresh_n_candidates": 10,
        "control_only": False,
        "transfer_only": False,
        "ablate_population": True,
        "live_independent_source": False,
        "frozen_source": False,
    },
}


def normalize_budget_allocation_label(raw: Any) -> Optional[str]:
    """Return canonical F|G|H or None."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    # Allow verbose aliases from KIND suffixes / fill helpers.
    aliases = {
        "F": "F",
        "TRANSFER_INIT": "F",
        "F_TRANSFER_INIT": "F",
        "G": "G",
        "TARGET_POP_INIT": "G",
        "G_TARGET_POP_INIT": "G",
        "H": "H",
        "DIRECT_PERSON20": "H",
        "H_DIRECT_PERSON20": "H",
    }
    if s in aliases:
        return aliases[s]
    low = s.lower()
    if "allocation_f" in low or low.endswith("_f_transfer_init"):
        return "F"
    if "allocation_g" in low or low.endswith("_g_target_pop_init"):
        return "G"
    if "allocation_h" in low or low.endswith("_h_direct_person20"):
        return "H"
    return None


def budget_allocation_label_from_args(args: Any) -> Optional[str]:
    """Explicit --pics_v3_budget_allocation or KIND-based F/G/H."""
    explicit = normalize_budget_allocation_label(
        getattr(args, "pics_v3_budget_allocation", None)
    )
    if explicit:
        return explicit
    kind = str(getattr(args, "kind", "") or getattr(args, "output_kind", "") or "")
    return normalize_budget_allocation_label(kind)


def ablation_id_from_args(args: Any) -> Optional[str]:
    """Return canonical ablation id when an ablation flag is set; else None.

    Budget-allocation F/G/H takes priority over A–E flag heuristics so G is not
    mis-labeled as ``no_transfer`` and H is not mis-labeled as ``no_population``.
    """
    alloc = budget_allocation_label_from_args(args)
    if alloc is not None:
        return f"allocation_{alloc}"
    if bool(getattr(args, "t_pics_gated_transfer_only", False)):
        return "allocation_F"
    if bool(getattr(args, "t_pics_ablate_population", False)):
        return "no_population"
    if bool(getattr(args, "t_pics_gated_control_only", False)):
        return "no_transfer"
    if bool(getattr(args, "ablate_dataset_adaptive_prompt", False)):
        return "no_adaptive_prompt"
    explore = int(getattr(args, "explore_candidates", 0) or 0)
    fresh = int(getattr(args, "fresh_n_candidates", 0) or 0)
    n_iters = int(getattr(args, "n_iterations", 0) or 0)
    independent = bool(getattr(args, "t_pics_gated_independent", False))
    gated = bool(getattr(args, "t_pics_gated_transfer", False))
    # Live no-explore: independent G.1 + dual G.2, explore=0, person=15.
    if gated and independent and explore == 0 and n_iters == 15:
        return "no_explore"
    # Live no-fresh: independent + fresh_n=0 + person=10 + explore>0.
    if gated and independent and fresh == 0 and n_iters == 10 and explore == 50:
        return "no_fresh"
    kind = str(getattr(args, "kind", "") or getattr(args, "output_kind", "") or "")
    for ablation_id, spec in ABLATION_PHASE_BUDGETS.items():
        if kind == spec["kind"]:
            return ablation_id
    for label, spec in ALLOCATION_PHASE_BUDGETS.items():
        if kind == spec["kind"]:
            return f"allocation_{label}"
    return None


def ablation_metadata_payload(args: Any) -> dict:
    """Resolved phase/candidate counts for run config / W&B (ablation runs only)."""
    ablation_id = ablation_id_from_args(args)
    alloc = budget_allocation_label_from_args(args)
    if alloc is not None:
        spec = ALLOCATION_PHASE_BUDGETS.get(alloc, {})
        return {
            "ablation_id": ablation_id,
            "ablation_kind": spec.get("kind"),
            "ablation_run_tag": ALLOCATION_RUN_TAG,
            "budget_allocation": alloc,
            "control_only": bool(getattr(args, "t_pics_gated_control_only", False)),
            "transfer_only": bool(getattr(args, "t_pics_gated_transfer_only", False)),
            "ablate_population": bool(getattr(args, "t_pics_ablate_population", False)),
            "ablate_dataset_adaptive_prompt": bool(
                getattr(args, "ablate_dataset_adaptive_prompt", False)
            ),
            "ablate_history_reminder": False,
            "live_independent_source": bool(
                getattr(args, "t_pics_gated_independent", False)
            ),
            "reuse_gate_pool": None,
            "global_iters": int(getattr(args, "global_iters", 0) or 0),
            "explore_candidates": int(getattr(args, "explore_candidates", 0) or 0),
            "n_iterations": int(getattr(args, "n_iterations", 0) or 0),
            "fresh_n_candidates": int(getattr(args, "fresh_n_candidates", 0) or 0),
            "n_candidates": int(getattr(args, "n_candidates", 10) or 10),
            "known_ablation_kinds": list(ABLATION_KINDS),
            "known_allocation_kinds": list(ALLOCATION_KINDS),
            "wandb_group": ALLOCATION_WANDB_GROUP,
        }
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
        "ablate_history_reminder": bool(
            getattr(args, "ablate_dataset_adaptive_prompt", False)
        ),
        "live_independent_source": bool(
            getattr(args, "t_pics_gated_independent", False)
        ),
        # Reuse of main/prior gate pools is forbidden for final ablations.
        "reuse_gate_pool": None,
        "global_iters": int(getattr(args, "global_iters", 0) or 0),
        "explore_candidates": int(getattr(args, "explore_candidates", 0) or 0),
        "n_iterations": int(getattr(args, "n_iterations", 0) or 0),
        "fresh_n_candidates": int(getattr(args, "fresh_n_candidates", 0) or 0),
        "n_candidates": int(getattr(args, "n_candidates", 10) or 10),
        "known_ablation_kinds": list(ABLATION_KINDS),
    }
