"""CPU tests for PICS v3 I/J hybrid grounded experiments."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]

from utils.teh.pics_v3 import (  # noqa: E402
    ABLATION_KIND_I_HYBRID_TRANSFER,
    ABLATION_KIND_J_HYBRID_TARGET,
    ABLATION_KINDS,
    ALLOCATION_KINDS,
    IJ_HYBRID_KINDS,
    KIND,
)
from utils.teh.pics_v3_ablation import (  # noqa: E402
    IJ_HYBRID_PHASE_BUDGETS,
    ablation_id_from_args,
    ablation_metadata_payload,
    ij_hybrid_label_from_args,
)


def test_ij_kinds_isolated():
    assert KIND not in IJ_HYBRID_KINDS
    assert not (set(IJ_HYBRID_KINDS) & set(ALLOCATION_KINDS))
    assert ABLATION_KIND_I_HYBRID_TRANSFER in ABLATION_KINDS
    assert ABLATION_KIND_J_HYBRID_TARGET in ABLATION_KINDS
    assert ABLATION_KIND_I_HYBRID_TRANSFER == "pics_v3_ablation_i_hybrid_transfer"
    assert ABLATION_KIND_J_HYBRID_TARGET == "pics_v3_ablation_j_hybrid_target"


def test_ij_phase_budgets():
    i = IJ_HYBRID_PHASE_BUDGETS["I"]
    j = IJ_HYBRID_PHASE_BUDGETS["J"]
    assert i["global_iters"] == j["global_iters"] == 10
    assert i["n_iterations"] == j["n_iterations"] == 10
    assert i["explore_candidates"] == j["explore_candidates"] == 50
    assert i["fresh_n_candidates"] == j["fresh_n_candidates"] == 10
    assert i["transfer_only"] is True
    assert j["control_only"] is True
    assert i["hybrid_grounded_prompt"] is True
    assert j["hybrid_grounded_prompt"] is True
    # Target-side cand budget for 30 people: 100 + 50 + 3000 = 3150
    n_people = 30
    for spec in (i, j):
        total = (
            spec["global_iters"] * 10
            + spec["explore_candidates"]
            + n_people * spec["n_iterations"] * 10
        )
        assert total == 3150


def test_ij_label_and_ablation_id():
    assert (
        ij_hybrid_label_from_args(
            SimpleNamespace(
                pics_v3_ij_hybrid="I",
                pics_v3_hybrid_grounded_prompt=True,
                t_pics_gated_transfer_only=True,
                kind=ABLATION_KIND_I_HYBRID_TRANSFER,
            )
        )
        == "I"
    )
    assert (
        ablation_id_from_args(
            SimpleNamespace(
                pics_v3_ij_hybrid="I",
                pics_v3_hybrid_grounded_prompt=True,
                t_pics_gated_transfer_only=True,
                t_pics_gated_control_only=False,
                t_pics_ablate_population=False,
                ablate_dataset_adaptive_prompt=False,
                pics_v3_budget_allocation=None,
                kind=ABLATION_KIND_I_HYBRID_TRANSFER,
            )
        )
        == "ij_I"
    )
    assert (
        ablation_id_from_args(
            SimpleNamespace(
                pics_v3_ij_hybrid="J",
                pics_v3_hybrid_grounded_prompt=True,
                t_pics_gated_control_only=True,
                t_pics_gated_transfer_only=False,
                t_pics_ablate_population=False,
                ablate_dataset_adaptive_prompt=False,
                pics_v3_budget_allocation=None,
                kind=ABLATION_KIND_J_HYBRID_TARGET,
            )
        )
        == "ij_J"
    )
    # I/J must not be mis-labeled as allocation F/G.
    meta = ablation_metadata_payload(
        SimpleNamespace(
            pics_v3_ij_hybrid="I",
            pics_v3_hybrid_grounded_prompt=True,
            t_pics_gated_transfer_only=True,
            t_pics_gated_control_only=False,
            t_pics_ablate_population=False,
            ablate_dataset_adaptive_prompt=False,
            t_pics_gated_independent=False,
            pics_v3_budget_allocation=None,
            global_iters=10,
            explore_candidates=50,
            n_iterations=10,
            fresh_n_candidates=10,
            n_candidates=10,
            kind=ABLATION_KIND_I_HYBRID_TRANSFER,
        )
    )
    assert meta["ij_hybrid"] == "I"
    assert meta["explore_candidates"] == 50
    assert meta["pics_v3_hybrid_grounded_prompt"] is True
    assert meta["ablate_history_reminder"] is False


def test_common_sh_has_ij_fill_not_main_mutation():
    common = (REPO / "cluster/v2/ours/Qwen/_common.sh").read_text(encoding="utf-8")
    assert "t_pics_v3_fill_ij_hybrid_args()" in common
    assert "pics_v3_hybrid_grounded_prompt" in common
    assert "pics_v3_ablation_i_hybrid_transfer" in common
    # Main fill still has prefer_auto (unchanged default path).
    assert "--prefer_auto_llm_prompt" in common
    # I/J fill must not add prefer_auto in its own block: spot-check nearby markers.
    ij_idx = common.index("t_pics_v3_fill_ij_hybrid_args()")
    ij_block = common[ij_idx : ij_idx + 4500]
    assert "Never pass --prefer_auto_llm_prompt" in ij_block
    assert "--pics_v3_hybrid_grounded_prompt" in ij_block


def test_launchers_exist_and_four_jobs():
    submit = (
        REPO / "cluster/v3/ours/ablation_ij/submit_ij_hybrid.sh"
    ).read_text(encoding="utf-8")
    job = (
        REPO / "cluster/v3/ours/ablation_ij/job_ij_hybrid_h100.sh"
    ).read_text(encoding="utf-8")
    assert "t_pics_v3_fill_ij_hybrid_args" in job
    assert "#SBATCH --requeue" in job
    assert "h100nvl" in job
    assert "I_pack1" not in submit or "pv3ijI_pack1" in submit or "pv3ij${condition}_${pack}" in submit
    assert "pack1" in submit and "pack2" in submit
    assert "CONDITIONS" in submit
    assert "2plonsky2018when" in submit
    assert "14kool2016when" in submit
    assert "bergert_nosofsky_2007" in submit
    assert "steyvers_2009_bandit" in submit
