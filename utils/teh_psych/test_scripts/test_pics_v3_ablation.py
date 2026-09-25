"""CPU tests for PICS v3 ablation flags (defaults off; main path unchanged)."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
from utils.teh.pics_v3_prompt_robustness import (
    DATASET_KEYED_POST_ADAPTIVE_POLICY_ID,
    HISTORY_ROBUSTNESS_MARKER,
    resolve_history_robustness_policy_id,
)
from utils.teh.t_pics_gated_transfer import apply_gated_cli_defaults
from utils.teh.t_pics_gated_wandb import build_gated_wandb_config
from utils.teh.teh_datasets import dataset_task_description
from utils.teh.teh_runtime import setup_teh_run_prompts


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
        t_pics_gated_transfer=True,
        t_pics_gated_independent=True,
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
        source_job_id="live",
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
    assert cfg["ablation"]["ablation_id"] == "no_explore"
    assert cfg["ablation"]["reuse_gate_pool"] is None


def test_ablation_id_control_only_and_no_population():
    a = SimpleNamespace(
        t_pics_ablate_population=False,
        t_pics_gated_control_only=True,
        ablate_dataset_adaptive_prompt=False,
        t_pics_gated_transfer=True,
        t_pics_gated_independent=False,
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
        t_pics_gated_independent=False,
        explore_candidates=50,
        fresh_n_candidates=10,
        n_iterations=30,
    )
    assert ablation_id_from_args(b) == "no_population"
    meta = ablation_metadata_payload(a)
    assert meta["ablation_kind"] == ABLATION_KIND_NO_TRANSFER
    assert meta["n_iterations"] == 25


def test_ablation_id_live_no_explore_and_no_fresh():
    c = SimpleNamespace(
        t_pics_ablate_population=False,
        t_pics_gated_control_only=False,
        ablate_dataset_adaptive_prompt=False,
        t_pics_gated_transfer=True,
        t_pics_gated_independent=True,
        explore_candidates=0,
        fresh_n_candidates=10,
        n_iterations=15,
    )
    assert ablation_id_from_args(c) == "no_explore"
    d = SimpleNamespace(
        t_pics_ablate_population=False,
        t_pics_gated_control_only=False,
        ablate_dataset_adaptive_prompt=False,
        t_pics_gated_transfer=True,
        t_pics_gated_independent=True,
        explore_candidates=50,
        fresh_n_candidates=0,
        n_iterations=10,
    )
    assert ablation_id_from_args(d) == "no_fresh"


def test_phase_budgets_match_nominal_35_live_independent():
    assert ABLATION_PHASE_BUDGETS["no_transfer"]["n_iterations"] == 25
    assert ABLATION_PHASE_BUDGETS["no_population"]["n_iterations"] == 30
    assert ABLATION_PHASE_BUDGETS["no_explore"]["explore_candidates"] == 0
    assert ABLATION_PHASE_BUDGETS["no_explore"]["n_iterations"] == 15
    assert ABLATION_PHASE_BUDGETS["no_explore"]["live_independent_source"] is True
    assert ABLATION_PHASE_BUDGETS["no_fresh"]["fresh_n_candidates"] == 0
    assert ABLATION_PHASE_BUDGETS["no_fresh"]["live_independent_source"] is True
    assert ABLATION_PHASE_BUDGETS["no_adaptive_prompt"]["ablate_adaptive_prompt"] is True
    assert ABLATION_PHASE_BUDGETS["no_adaptive_prompt"]["ablate_history_reminder"] is True


def test_registered_description_helper_oe_parity_source():
    """OE's vanilla_dataset_description delegates to dataset_task_description."""
    datasets = [
        "1peterson2021using",
        "mixed_gambles",
        "bergert_nosofsky_2007",
        "7hilbig2014generalized",
        "guan_2020_stopping",
        "3frey2017cct",
        "2plonsky2018when",
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
    for flag in (
        "--t_pics_gated_control_only",
        "--t_pics_ablate_population",
        "--ablate_dataset_adaptive_prompt",
        "--t_pics_reuse_gate_pool",
    ):
        assert flag in teh_src
    assert "requires --explore_candidates > 0." in teh_src
    assert "explore_candidates == 0 is allowed for the no-explore ablation" in teh_src
    # Live G.1 respects ablate (no forced require_auto when ablating).
    assert "require_auto_llm_prompt=not ablate_adaptive" in teh_src
    assert "ablate_dataset_adaptive_prompt=ablate_adaptive" in teh_src


def test_fill_gated_args_unchanged_and_ablation_live_only():
    common = (REPO / "cluster/v2/ours/Qwen/_common.sh").read_text(encoding="utf-8")
    assert 'EXPLORE_CANDIDATES="${EXPLORE_CANDIDATES:-50}"' in common
    assert 'N_ITERS="${N_ITERS:-10}"' in common
    assert "--t_pics_gated_transfer" in common
    assert "t_pics_v3_fill_ablation_args" in common
    assert "t_pics_v3_ablation_main_gate_pool" not in common
    gated_fill = common.split("t_pics_v3_fill_gated_args()")[1].split(
        "t_pics_v3_fill_ablation_args"
    )[0]
    assert "--t_pics_gated_control_only" not in gated_fill
    assert "--t_pics_ablate_population" not in gated_fill
    assert "--ablate_dataset_adaptive_prompt" not in gated_fill
    ablation_fill = common.split("t_pics_v3_fill_ablation_args()")[1].split(
        "t_pics_v3_fill_family_prompt_v3_args"
    )[0]
    assert "--t_pics_reuse_gate_pool" not in ablation_fill
    # C/D/E live independent.
    assert ablation_fill.count("--t_pics_gated_independent") >= 3


def test_submit_packed_five_jobs_three_targets():
    submit = (
        REPO / "cluster/v3/ours/ablation/submit_ablations.sh"
    ).read_text(encoding="utf-8")
    job = (REPO / "cluster/v3/ours/ablation/job_ablation_h100.sh").read_text(
        encoding="utf-8"
    )
    assert "2plonsky2018when" in submit
    assert "3frey2017cct" in submit
    assert "bergert_nosofsky_2007" in submit
    assert "1peterson2021using" not in submit
    assert "guan_2020_stopping" not in submit
    assert "pv3a${condition}_${PACK_TAG}" in submit or "pv3a${condition}_packed" in submit
    assert "ABLATION_TARGETS" in submit
    assert "run_one_dataset" in job
    assert "REUSE_GATE_POOL=FORBIDDEN" in job
    assert "--requeue" in job
    assert "job_${JOB_TAG}" in job or 'job_${JOB_TAG}' in job
    assert "dataset_is_complete" in job



def test_ablation_kinds_constants():
    assert ABLATION_KIND_NO_TRANSFER.endswith("no_transfer")
    assert ABLATION_KIND_NO_POPULATION.endswith("no_population")
    assert ABLATION_KIND_NO_EXPLORE.endswith("no_explore")
    assert ABLATION_KIND_NO_FRESH.endswith("no_fresh")
    assert KIND not in ABLATION_KINDS
    assert all(k.startswith("pics_v3_ablation_") for k in ABLATION_KINDS)
    assert "live_reference" not in "".join(ABLATION_KINDS)



def test_main_and_non_e_ablations_keep_reminder_v2_policy_id():
    """Keyed datasets keep dataset_keyed_post_adaptive_v2; injection still on for non-E."""
    from utils.teh.pics_v3_prompt_robustness import configure_pics_v3_legacy_generic_reminder

    configure_pics_v3_legacy_generic_reminder(False)
    assert (
        resolve_history_robustness_policy_id("guan_2020_stopping")
        == DATASET_KEYED_POST_ADAPTIVE_POLICY_ID
    )
    assert (
        resolve_history_robustness_policy_id("14kool2016when")
        == DATASET_KEYED_POST_ADAPTIVE_POLICY_ID
    )
    # Non-keyed targets still receive the HISTORY marker body under SA40 (generic
    # fallback of the v2 policy); only E disables injection entirely.
    assert HISTORY_ROBUSTNESS_MARKER in (
        __import__(
            "utils.teh.pics_v3_prompt_robustness",
            fromlist=["resolve_history_robustness_block"],
        ).resolve_history_robustness_block("2plonsky2018when")
    )


def test_ablate_adaptive_skips_reminder_keeps_registered_desc(tmp_path):
    """E: registered OE-parity description + runtime contract; no HISTORY marker."""
    seed = tmp_path / "seed.py"
    seed.write_text(
        "def choose(problem, history):\n    return 0.5\n", encoding="utf-8"
    )
    run_dir = tmp_path / "run_e"
    ds = "bergert_nosofsky_2007"
    expected_desc = dataset_task_description(ds)
    with mock.patch(
        "utils.teh.teh_runtime._prompt_sample_pid_and_instruction",
        return_value=(0, "unused_instruction"),
    ), mock.patch(
        "utils.teh.teh_runtime._load_prompt_observation_trials",
        return_value=(
            [
                {
                    "problem": {"option_keys": [0, 1], "dataset_alias": ds},
                    "action": 0,
                    "history": [],
                    "split": "train",
                }
            ],
            [],
            None,
        ),
    ), mock.patch(
        "utils.teh.teh_runtime.build_deterministic_runtime_contract",
        return_value="### RUNTIME_CONTRACT\nreturn float in [0,1]",
    ), mock.patch(
        "utils.teh.teh_runtime.attach_runtime_contract_to_prompt",
        side_effect=lambda body, contract: body + "\n" + contract,
    ), mock.patch(
        "utils.teh.teh_runtime.assert_no_generation_oracle_leak",
        return_value=None,
    ), mock.patch(
        "utils.teh.teh_runtime.resolve_base_loglik_prompt_path",
        return_value=tmp_path / "base.txt",
    ):
        (tmp_path / "base.txt").write_text("BASE_TEMPLATE\n", encoding="utf-8")
        with mock.patch(
            "utils.teh.teh_runtime._merge_prompt_fallback",
            return_value=f"TASK\n{expected_desc}\n",
        ) as merge_mock:
            prompts = setup_teh_run_prompts(
                run_dir,
                ds,
                seed,
                client=None,
                use_llm=False,
                limited_data_protocol="structure_aware_v3",
                limited_train_val=40,
                ablate_dataset_adaptive_prompt=True,
                prefer_auto_llm_prompt=False,
                require_auto_llm_prompt=False,
            )
            assert merge_mock.call_args.kwargs.get(
                "force_registered_task_description"
            ) is True
    infer = (prompts / "infer_single_choice.txt").read_text(encoding="utf-8")
    assert HISTORY_ROBUSTNESS_MARKER not in infer
    assert "RUNTIME_CONTRACT" in infer
    meta = json.loads((prompts / "prompt_meta.json").read_text(encoding="utf-8"))
    assert meta["ablate_dataset_adaptive_prompt"] is True
    assert meta["prompt_mode"] == "registered_description_ablation"
    assert meta.get("pics_v3_reminder_policy") == "ablated_no_history_reminder"


def test_sa40_without_ablate_still_injects_reminder(tmp_path):
    """Main / A–D path: SA40 still appends HISTORY reminder (unchanged)."""
    from utils.teh.pics_v3_prompt_robustness import configure_pics_v3_legacy_generic_reminder

    configure_pics_v3_legacy_generic_reminder(False)
    seed = tmp_path / "seed.py"
    seed.write_text(
        "def choose(problem, history):\n    return 0.5\n", encoding="utf-8"
    )
    run_dir = tmp_path / "run_main"
    # Use a keyed dataset so recorded policy id is dataset_keyed_post_adaptive_v2.
    ds = "guan_2020_stopping"
    with mock.patch(
        "utils.teh.teh_runtime._prompt_sample_pid_and_instruction",
        return_value=(0, "unused"),
    ), mock.patch(
        "utils.teh.teh_runtime._load_prompt_observation_trials",
        return_value=(
            [
                {
                    "problem": {"option_keys": [0, 1], "dataset_alias": ds},
                    "action": 0,
                    "history": [],
                    "split": "train",
                }
            ],
            [],
            None,
        ),
    ), mock.patch(
        "utils.teh.teh_runtime.build_deterministic_runtime_contract",
        return_value="### RUNTIME_CONTRACT\nok",
    ), mock.patch(
        "utils.teh.teh_runtime.attach_runtime_contract_to_prompt",
        side_effect=lambda body, contract: body + "\n" + contract,
    ), mock.patch(
        "utils.teh.teh_runtime.assert_no_generation_oracle_leak",
        return_value=None,
    ), mock.patch(
        "utils.teh.teh_runtime.resolve_base_loglik_prompt_path",
        return_value=tmp_path / "base.txt",
    ), mock.patch(
        "utils.teh.teh_runtime._merge_prompt_fallback",
        return_value="TASK BODY\n",
    ):
        (tmp_path / "base.txt").write_text("BASE\n", encoding="utf-8")
        prompts = setup_teh_run_prompts(
            run_dir,
            ds,
            seed,
            client=None,
            use_llm=False,
            limited_data_protocol="structure_aware_v3",
            limited_train_val=40,
            ablate_dataset_adaptive_prompt=False,
            prefer_auto_llm_prompt=False,
            require_auto_llm_prompt=False,
        )
    infer = (prompts / "infer_single_choice.txt").read_text(encoding="utf-8")
    assert HISTORY_ROBUSTNESS_MARKER in infer
    meta = json.loads((prompts / "prompt_meta.json").read_text(encoding="utf-8"))
    assert meta["pics_v3_reminder_policy"] == DATASET_KEYED_POST_ADAPTIVE_POLICY_ID
    assert meta["ablate_dataset_adaptive_prompt"] is False
