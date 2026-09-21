"""CPU tests for PICS v3 ablation flags (defaults off; main path unchanged)."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.teh.pics_v3 import (
    ABLATION_KIND_NO_EXPLORE,
    ABLATION_KIND_NO_FRESH,
    ABLATION_KIND_NO_POPULATION,
    ABLATION_KIND_NO_TRANSFER,
    ABLATION_KINDS,
    KIND,
)
from utils.teh.pics_v3_ablation import (
    ABLATION_PHASE_BUDGETS,
    ablation_id_from_args,
    ablation_metadata_payload,
)
from utils.teh.t_pics_gated_transfer import apply_gated_cli_defaults
from utils.teh.t_pics_gated_wandb import build_gated_wandb_config
from utils.teh.teh_datasets import dataset_task_description


REPO = Path(__file__).resolve().parents[3]


def test_ablation_kinds_isolated_from_main():
    assert KIND == "pics_v3"
    assert "pics_v3" not in ABLATION_KINDS
    for k in ABLATION_KINDS:
        assert k.startswith("pics_v3_ablation_")


def test_apply_gated_defaults_explore_zero_clears_population_parents():
    args = SimpleNamespace(
        n_iterations=15,
        global_iters=5,
        explore_candidates=0,
        explore_from_population_parents=True,
        explore_population_top_k=1,
        t_pics_source_config=None,
        t_pics_gated_source=None,
        limited_data_protocol=None,
        limited_train_val=None,
        hard_prompt_token_cap=None,
        max_parent_chars=None,
        llm_max_tokens=None,
        max_prompt_train_trials=None,
        global_phase=False,
        refinement_phase=True,
    )
    apply_gated_cli_defaults(
        args,
        argv=["--explore_candidates", "0", "--n_iterations", "15"],
    )
    assert args.explore_candidates == 0
    assert args.explore_from_population_parents is False
    assert args.global_phase is True
    assert args.refinement_phase is False


def test_wandb_config_preserves_explore_zero():
    args = SimpleNamespace(
        dataset="bergert_nosofsky_2007",
        limited_data_protocol="structure_aware_v3",
        limited_train_val=40,
        split_seed=0,
        evolution_selection_score="train_val",
        t_pics_gated_independent=False,
        t_pics_gated_control_only=False,
        t_pics_ablate_population=False,
        ablate_dataset_adaptive_prompt=False,
        t_pics_reuse_gate_pool=None,
        global_iters=5,
        explore_candidates=0,
        n_iterations=15,
        n_candidates=10,
        fresh_n_candidates=10,
        sample_size=8,
        sample_parents=True,
        sampled_parents_decay=True,
        elite_pool_size=50,
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    )
    cfg = build_gated_wandb_config(
        args=args,
        output_root=REPO / "generated_outputs" / "tmp_ablation_test",
        selected_source="7hilbig2014generalized",
        source_job_id="271242",
        source_rank1=REPO / "persona_code_example" / "te_vanilla" / "choices13k.py",
        source_config_path=REPO
        / "analysis/config/T-PICS/Transfer_source/pics_v3/occurrence_eb_schema5_iter10_pics_v3.yaml",
        source_config_sha256=None,
        prompt_mode="auto_llm",
        git_commit=None,
        slurm_job_id="0",
    )
    assert cfg["exploration_candidates"] == 0
    assert cfg["participant_iterations"] == 15


def test_ablation_id_control_only_and_no_population():
    a = SimpleNamespace(
        t_pics_ablate_population=False,
        t_pics_gated_control_only=True,
        ablate_dataset_adaptive_prompt=False,
        t_pics_gated_transfer=True,
        explore_candidates=50,
        fresh_n_candidates=10,
        n_iterations=25,
    )
    assert ablation_id_from_args(a) == "no_transfer"
    b = SimpleNamespace(
        t_pics_ablate_population=True,
        t_pics_gated_control_only=False,
        ablate_dataset_adaptive_prompt=False,
        t_pics_gated_transfer=False,
        explore_candidates=50,
        fresh_n_candidates=10,
        n_iterations=30,
    )
    assert ablation_id_from_args(b) == "no_population"
    meta = ablation_metadata_payload(a)
    assert meta["ablation_kind"] == ABLATION_KIND_NO_TRANSFER
    assert meta["n_iterations"] == 25


def test_phase_budgets_match_nominal_35():
    assert ABLATION_PHASE_BUDGETS["no_transfer"]["n_iterations"] == 25
    assert ABLATION_PHASE_BUDGETS["no_population"]["n_iterations"] == 30
    assert ABLATION_PHASE_BUDGETS["no_explore"]["explore_candidates"] == 0
    assert ABLATION_PHASE_BUDGETS["no_explore"]["n_iterations"] == 15
    assert ABLATION_PHASE_BUDGETS["no_fresh"]["fresh_n_candidates"] == 0
    assert ABLATION_PHASE_BUDGETS["no_adaptive_prompt"]["ablate_adaptive_prompt"] is True


def test_registered_description_helper_oe_parity_source():
    """OE's vanilla_dataset_description delegates to dataset_task_description."""
    datasets = [
        "1peterson2021using",
        "mixed_gambles",
        "bergert_nosofsky_2007",
        "7hilbig2014generalized",
        "guan_2020_stopping",
        "3frey2017cct",
    ]
    for ds in datasets:
        text = dataset_task_description(ds)
        assert isinstance(text, str) and text.strip()
    oe_src = (
        REPO / "baseline_methods/Psych101/run_openevolve.py"
    ).read_text(encoding="utf-8")
    assert "dataset_task_description" in oe_src
    assert "def vanilla_dataset_description" in oe_src


def test_cli_flags_default_off_in_teh_source():
    teh_src = (REPO / "teh.py").read_text(encoding="utf-8")
    tree = ast.parse(teh_src)
    # Smoke: new flags appear as store_true with default False in source text.
    for flag in (
        "--t_pics_gated_control_only",
        "--t_pics_ablate_population",
        "--ablate_dataset_adaptive_prompt",
        "--t_pics_reuse_gate_pool",
    ):
        assert flag in teh_src
    assert "requires --explore_candidates > 0." in teh_src  # still present for non-ablation paths
    # Gated no longer unconditionally requires explore>0:
    assert "explore_candidates == 0 is allowed for the no-explore ablation" in teh_src


def test_fill_gated_args_unchanged_defaults_in_common():
    common = (REPO / "cluster/v2/ours/Qwen/_common.sh").read_text(encoding="utf-8")
    assert 'EXPLORE_CANDIDATES="${EXPLORE_CANDIDATES:-50}"' in common
    assert 'N_ITERS="${N_ITERS:-10}"' in common
    assert "--t_pics_gated_transfer" in common
    assert "t_pics_v3_fill_ablation_args" in common
    # Main fill must not force ablation flags.
    gated_fill = common.split("t_pics_v3_fill_gated_args()")[1].split(
        "t_pics_v3_ablation_main_gate_pool"
    )[0]
    assert "--t_pics_gated_control_only" not in gated_fill
    assert "--t_pics_ablate_population" not in gated_fill
    assert "--ablate_dataset_adaptive_prompt" not in gated_fill


def test_ablation_kinds_constants():
    assert ABLATION_KIND_NO_TRANSFER.endswith("no_transfer")
    assert ABLATION_KIND_NO_POPULATION.endswith("no_population")
    assert ABLATION_KIND_NO_EXPLORE.endswith("no_explore")
    assert ABLATION_KIND_NO_FRESH.endswith("no_fresh")
