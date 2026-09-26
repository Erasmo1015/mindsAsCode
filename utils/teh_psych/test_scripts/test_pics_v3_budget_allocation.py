"""CPU tests for PICS v3 budget-allocation F/G/H (default off)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]

from utils.teh.pics_v3 import (  # noqa: E402
    ABLATION_KINDS,
    ALLOCATION_KIND_F_TRANSFER_INIT,
    ALLOCATION_KIND_G_TARGET_POP_INIT,
    ALLOCATION_KIND_H_DIRECT_PERSON20,
    ALLOCATION_KINDS,
    KIND,
)
from utils.teh.pics_v3_ablation import (  # noqa: E402
    ALLOCATION_PHASE_BUDGETS,
    ablation_id_from_args,
    ablation_metadata_payload,
    budget_allocation_label_from_args,
)
from utils.teh.t_pics_gated_transfer import gated_transfer_only_enabled  # noqa: E402


def test_allocation_kinds_distinct_from_main_and_ae():
    assert KIND not in ALLOCATION_KINDS
    assert not (set(ALLOCATION_KINDS) & set(ABLATION_KINDS))
    assert ALLOCATION_KIND_F_TRANSFER_INIT == "pics_v3_allocation_f_transfer_init"
    assert ALLOCATION_KIND_G_TARGET_POP_INIT == "pics_v3_allocation_g_target_pop_init"
    assert ALLOCATION_KIND_H_DIRECT_PERSON20 == "pics_v3_allocation_h_direct_person20"


def test_allocation_phase_budgets_depth_sum_20():
    f = ALLOCATION_PHASE_BUDGETS["F"]
    g = ALLOCATION_PHASE_BUDGETS["G"]
    h = ALLOCATION_PHASE_BUDGETS["H"]
    assert f["global_iters"] + f["n_iterations"] == 20
    assert g["global_iters"] + g["n_iterations"] == 20
    assert h["global_iters"] + h["n_iterations"] == 20
    assert f["explore_candidates"] == g["explore_candidates"] == h["explore_candidates"] == 0
    assert f["transfer_only"] is True
    assert g["control_only"] is True
    assert h["ablate_population"] is True


def test_ablation_id_prefers_allocation_over_ae_heuristics():
    # G looks like A (control_only) without allocation label → still A.
    assert (
        ablation_id_from_args(
            SimpleNamespace(
                t_pics_gated_control_only=True,
                t_pics_ablate_population=False,
                pics_v3_budget_allocation=None,
                kind="",
            )
        )
        == "no_transfer"
    )
    # With allocation G label → allocation_G.
    assert (
        ablation_id_from_args(
            SimpleNamespace(
                t_pics_gated_control_only=True,
                t_pics_ablate_population=False,
                pics_v3_budget_allocation="G",
                kind=ALLOCATION_KIND_G_TARGET_POP_INIT,
            )
        )
        == "allocation_G"
    )
    assert (
        ablation_id_from_args(
            SimpleNamespace(
                t_pics_gated_transfer_only=True,
                pics_v3_budget_allocation="F",
                kind=ALLOCATION_KIND_F_TRANSFER_INIT,
                t_pics_ablate_population=False,
                t_pics_gated_control_only=False,
            )
        )
        == "allocation_F"
    )
    assert (
        ablation_id_from_args(
            SimpleNamespace(
                t_pics_ablate_population=True,
                pics_v3_budget_allocation="H",
                kind=ALLOCATION_KIND_H_DIRECT_PERSON20,
                explore_candidates=0,
                n_iterations=20,
            )
        )
        == "allocation_H"
    )


def test_metadata_payload_allocation():
    meta = ablation_metadata_payload(
        SimpleNamespace(
            pics_v3_budget_allocation="F",
            t_pics_gated_transfer_only=True,
            t_pics_gated_control_only=False,
            t_pics_ablate_population=False,
            t_pics_gated_independent=False,
            ablate_dataset_adaptive_prompt=False,
            global_iters=10,
            explore_candidates=0,
            n_iterations=10,
            fresh_n_candidates=10,
            n_candidates=10,
            kind=ALLOCATION_KIND_F_TRANSFER_INIT,
        )
    )
    assert meta["budget_allocation"] == "F"
    assert meta["ablation_kind"] == ALLOCATION_KIND_F_TRANSFER_INIT
    assert meta["transfer_only"] is True


def test_gated_transfer_only_helper():
    assert gated_transfer_only_enabled(SimpleNamespace(t_pics_gated_transfer_only=True))
    assert not gated_transfer_only_enabled(
        SimpleNamespace(t_pics_gated_transfer_only=False)
    )


def test_cli_flags_present_and_ae_fill_untouched():
    teh_src = (REPO / "teh.py").read_text(encoding="utf-8")
    assert "--t_pics_gated_transfer_only" in teh_src
    assert "--pics_v3_budget_allocation" in teh_src
    assert "gated_transfer_only_enabled" in teh_src
    assert "transfer_only_allocation_f" in teh_src

    common = (REPO / "cluster/v2/ours/Qwen/_common.sh").read_text(encoding="utf-8")
    assert "t_pics_v3_fill_budget_allocation_args()" in common
    assert "--t_pics_gated_transfer_only" in common
    # A–E fill must not gain allocation flags.
    ae = common.split("t_pics_v3_fill_ablation_args() {")[1].split(
        "# Fill PICS_ARGS for family-prompt v3"
    )[0]
    assert "--t_pics_gated_transfer_only" not in ae
    assert "--pics_v3_budget_allocation" not in ae
    gated = common.split("t_pics_v3_fill_gated_args()")[1].split(
        "t_pics_v3_fill_ablation_args"
    )[0]
    assert "--t_pics_gated_transfer_only" not in gated
    assert "--pics_v3_budget_allocation" not in gated


def test_submit_scripts_six_datasets():
    submit = (
        REPO
        / "cluster/v3/ours/ablation/budget_allocation/submit_budget_allocation.sh"
    ).read_text(encoding="utf-8")
    job = (
        REPO
        / "cluster/v3/ours/ablation/budget_allocation/job_budget_allocation_h100.sh"
    ).read_text(encoding="utf-8")
    for ds in (
        "2plonsky2018when",
        "3frey2017cct",
        "bergert_nosofsky_2007",
        "11enkavi2019recentprobes",
        "5speekenbrink2008learning",
        "guan_2020_stopping",
    ):
        assert ds in submit
        assert ds in job
    assert "steyvers_2009_bandit" not in submit
    assert "14kool2016when" not in submit
    assert "pics_v3_allocation_f_transfer_init" in submit
    assert "t_pics_v3_fill_budget_allocation_args" in job
    assert "--requeue" in job
    assert "SEQUENTIAL_RL_REMINDER_V3=0" in job
    assert "SEQUENTIAL_RL_REMINDER_V4=0" in job


def test_label_from_kind():
    assert (
        budget_allocation_label_from_args(
            SimpleNamespace(
                pics_v3_budget_allocation=None,
                kind=ALLOCATION_KIND_H_DIRECT_PERSON20,
            )
        )
        == "H"
    )
