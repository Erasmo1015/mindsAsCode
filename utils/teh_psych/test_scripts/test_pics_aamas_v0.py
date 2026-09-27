"""CPU tests for the isolated PICS AAMAS v0 pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

from utils.teh.pics_aamas_v0 import (  # noqa: E402
    BATCH1_TARGET_ONLY_DATASETS,
    BATCH1_TARGET_ONLY_PACKS,
    BATCH2_TARGET_ONLY_PACKS,
    BATCH5_TRANSFER_PACKS,
    BEHAVIOR_OBJECTIVE_MARKER,
    CANONICAL_DATASETS,
    EXPLORE_CANDIDATES,
    GLOBAL_ITERS,
    KIND_OFFICIAL_GATE,
    KIND_TARGET_ONLY,
    KIND_TRANSFER_BASED_ONLY,
    N_CANDIDATES,
    PERSON_ITERS,
    PROMPT_POLICY_ID,
    ReminderDecision,
    apply_aamas_v0_prompt_body,
    assert_aamas_population_path,
    assert_packs_partition,
    configure_pics_aamas_v0_prompt,
    decide_aamas_v0_gate,
    materialize_official_gate,
    prompt_budget_report,
    reminder_decision,
    resolve_aamas_v0_source_program,
    source_identity_for_target,
    using_pics_aamas_v0_prompt,
)
from utils.teh.pics_v3_prompt_robustness import (  # noqa: E402
    HISTORY_ROBUSTNESS_BLOCK,
    dataset_keyed_reminder_body,
    sequential_rl_reminder_v4_block,
)
from utils.teh.prompt_context import RUNTIME_CONTRACT_HEADER  # noqa: E402

_HISTORICAL_SHA256 = {
    "cluster/Old/2026Sep/v3/ours/ablation_ij/submit_ij_hybrid.sh":
        "904af3f8d0a175a584d9bf5c810ca31b914ea5e54a44d6be1e1e0ce04ae1d5d1",
    "cluster/Old/2026Sep/v3/ours/ablation_ij/job_ij_hybrid_h100.sh":
        "3e26c68761ce180daff7f87fc016ede6ef4ccb69dc2c3bb09340841e2b6f7c7b",
    "cluster/Old/2026Sep/Qwen/_common.sh":
        "92eaf4e75191b120e623083cead92d653fb40c585e1afd652f57d8e31741ed7e",
    "utils/teh/pics_v3_prompt_robustness.py":
        "55b7cae766d6e5285dfdd26a1bd03b931f4207f9e4358f0bc90f9d08fb1cc31b",
}
_BODY_SHA256 = {
    "steyvers_v4": "7e901de247df32ea159b955a9090911bcc1c49114497ba6ad6c43e4f68dd9d8f",
    "schulz_v4": "55736156c62494acb82741853ed18e3853035ba2aa8430c0ed436e67db4f72d9",
    "kool_v4": "816fb1c5c5c9625c9cdddbe9aa7be5c0dd333f625b7bd2eee13ca38bf00594d3",
    "kool_v1": "9d743139a561a06dcf03fae1876acf196af6e15757fc1b9d8c7805da717e5e5e",
    "badham_v2": "259a1aa3cb2aa2573da92ed546377ce1d273b6e522a4a02e7531327363278aef",
    "guan": "9833963185b9113feef83d39de4ed690e0962bf246857bb0f61a8da1d7b98b26",
    "generic": "add06561bf19f05a093a946dd77825c6253052a602b1ca3c6ff9ab9d7551157b",
}
_GENERIC_SENTENCE = (
    "Never assume optional fields such as `feedback`, `reward`, or"
)
_SPEEK_REMINDER_SENTENCE = (
    "Prefer empirical association counts over fixed hard-coded card-set lookup tables"
)
_NO_REMINDER = {
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "5speekenbrink2008learning",
    "7hilbig2014generalized",
    "10frey2017risk",
    "11enkavi2019recentprobes",
    "mixed_gambles",
    "bergert_nosofsky_2007",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_historical_prompt_and_launcher_hashes_unchanged():
    for rel, digest in _HISTORICAL_SHA256.items():
        raw = (REPO / rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, rel
    assert _sha(sequential_rl_reminder_v4_block("steyvers_2009_bandit")) == _BODY_SHA256["steyvers_v4"]
    assert _sha(sequential_rl_reminder_v4_block("13schulz2020finding")) == _BODY_SHA256["schulz_v4"]
    assert _sha(sequential_rl_reminder_v4_block("14kool2016when")) == _BODY_SHA256["kool_v4"]
    assert _sha(dataset_keyed_reminder_body("14kool2016when")) == _BODY_SHA256["kool_v1"]
    assert _sha(dataset_keyed_reminder_body("12badham2017deficits")) == _BODY_SHA256["badham_v2"]
    assert _sha(dataset_keyed_reminder_body("guan_2020_stopping")) == _BODY_SHA256["guan"]
    assert _sha(HISTORY_ROBUSTNESS_BLOCK) == _BODY_SHA256["generic"]


def test_packs_cover_fifteen_once():
    assert_packs_partition(BATCH1_TARGET_ONLY_PACKS, BATCH1_TARGET_ONLY_DATASETS)
    batch2 = [ds for pack in BATCH2_TARGET_ONLY_PACKS.values() for ds in pack]
    assert_packs_partition(BATCH2_TARGET_ONLY_PACKS, batch2)
    assert sorted(list(BATCH1_TARGET_ONLY_DATASETS) + batch2) == sorted(CANONICAL_DATASETS)
    assert len(set(BATCH1_TARGET_ONLY_DATASETS) & set(batch2)) == 0
    assert_packs_partition(BATCH5_TRANSFER_PACKS, CANONICAL_DATASETS)
    assert [len(BATCH1_TARGET_ONLY_PACKS[k]) for k in ("pack1", "pack2", "pack3", "pack4")] == [2, 2, 1, 1]


def test_budgets_and_kind_isolation():
    assert GLOBAL_ITERS == 10
    assert N_CANDIDATES == 10
    assert EXPLORE_CANDIDATES == 50
    assert PERSON_ITERS == 10
    assert KIND_TARGET_ONLY == "pics_aamas_v0_target_only"
    assert KIND_TRANSFER_BASED_ONLY == "pics_aamas_v0_transfer_based_only"
    assert KIND_OFFICIAL_GATE == "pics_aamas_v0_official_gate"
    assert "pics_v3" not in KIND_TARGET_ONLY
    assert PROMPT_POLICY_ID == "pics_aamas_v0"


def test_official_gate_uses_train_val_only():
    tie = decide_aamas_v0_gate(target_only_score=-1.25, transfer_score=-1.25)
    assert tie["selected_arm"] == "target_only"
    assert tie["reason"] == "exact_tie_select_target_only"
    assert tie["test_used_for_gate"] is False
    better = decide_aamas_v0_gate(target_only_score=-1.25, transfer_score=-1.10)
    assert better["selected_arm"] == "transfer"
    worse = decide_aamas_v0_gate(target_only_score=-1.10, transfer_score=-1.25)
    assert worse["selected_arm"] == "target_only"
    with pytest.raises(RuntimeError):
        decide_aamas_v0_gate(target_only_score=float("nan"), transfer_score=-1.0)


def test_reminder_routing_matches_frozen_policy():
    expected = {
        "steyvers_2009_bandit": "sequential_rl_reminder_v4_steyvers",
        "13schulz2020finding": "sequential_rl_reminder_v4_schulz",
        "14kool2016when": "dataset_keyed_kool_v1",
        "12badham2017deficits": "dataset_keyed_badham_v2",
        "guan_2020_stopping": "dataset_keyed_guan",
    }
    for ds, reminder_id in expected.items():
        decision = reminder_decision(ds)
        assert decision.reminder_id == reminder_id
        assert decision.block
        assert decision.block.count(decision.block.split("]")[0] + "]") == 1
    for ds in _NO_REMINDER:
        decision = reminder_decision(ds)
        assert decision == ReminderDecision(ds, "none", None)
    kool = reminder_decision("14kool2016when").block
    assert "SEQUENTIAL_RL_BEHAVIOR_REMINDER_V4_KOOL" not in kool
    assert dataset_keyed_reminder_body("14kool2016when") in kool
    stey = reminder_decision("steyvers_2009_bandit").block
    assert stey == sequential_rl_reminder_v4_block("steyvers_2009_bandit")
    assert _GENERIC_SENTENCE not in stey


def test_source_map_is_identity_only():
    cfg = REPO / "analysis/config/T-PICS/Transfer_source/pics_v3/occurrence_eb_schema5_iter10_pics_v3.yaml"
    assert source_identity_for_target("2plonsky2018when", config_path=cfg) == "1peterson2021using"
    assert source_identity_for_target("steyvers_2009_bandit", config_path=cfg) == "3frey2017cct"
    assert source_identity_for_target("14kool2016when", config_path=cfg) == "3frey2017cct"
    assert source_identity_for_target("1peterson2021using", config_path=cfg) == "mixed_gambles"
    with pytest.raises(ValueError):
        source_identity_for_target("not_a_dataset", config_path=cfg)


def test_downstream_loader_fail_closed(tmp_path: Path):
    cfg = REPO / "analysis/config/T-PICS/Transfer_source/pics_v3/occurrence_eb_schema5_iter10_pics_v3.yaml"
    with pytest.raises(FileNotFoundError):
        resolve_aamas_v0_source_program(
            target_dataset="2plonsky2018when",
            source_dataset="1peterson2021using",
            repo_root=tmp_path,
            config_path=cfg,
        )
    with pytest.raises(RuntimeError):
        resolve_aamas_v0_source_program(
            target_dataset="2plonsky2018when",
            source_dataset="3frey2017cct",
            repo_root=tmp_path,
            config_path=cfg,
        )
    iclr = tmp_path / "generated_outputs/psych101_train/teh/1peterson2021using/pics_v3_g1/job_1"
    iclr.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        assert_aamas_population_path(iclr)

    run = (
        tmp_path
        / "generated_outputs/psych101_train/teh/1peterson2021using/pics_aamas_v0_target_only/job_9/target_population/control"
    )
    (run / "global_phase").mkdir(parents=True)
    program = run / "global_phase" / "best_program.py"
    program.write_text("def choose(problem, history):\n    return {0: 0.5, 1: 0.5}\n", encoding="utf-8")
    digest = hashlib.sha256(program.read_bytes()).hexdigest()
    marker = {
        "stage": "pics_aamas_v0_target_population",
        "dataset": "1peterson2021using",
        "kind": "pics_aamas_v0_target_only",
        "track_mode": "target_only",
        "arm_role": "target_only",
        "prompt_policy": "pics_aamas_v0",
        "test_used_for_selection": False,
        "sa40_fingerprint": "abc",
        "global_iters": 10,
        "n_candidates": 10,
        "rank1_program": str(program.resolve()),
        "rank1_sha256": digest,
        "runtime_valid": True,
    }
    prov = {
        **marker,
        "limited_data_protocol": "structure_aware_v3",
        "limited_train_val": 40,
        "runtime_valid": True,
        "reminder_id": "none",
    }
    (run / "STAGE_COMPLETE.json").write_text(json.dumps(marker), encoding="utf-8")
    (run / "POPULATION_PROVENANCE.json").write_text(json.dumps(prov), encoding="utf-8")
    resolved = resolve_aamas_v0_source_program(
        target_dataset="2plonsky2018when",
        source_dataset="1peterson2021using",
        repo_root=tmp_path,
        config_path=cfg,
    )
    assert resolved == program.resolve()
    bad = json.loads((run / "POPULATION_PROVENANCE.json").read_text(encoding="utf-8"))
    bad["prompt_policy"] = "pics_v3"
    (run / "POPULATION_PROVENANCE.json").write_text(json.dumps(bad), encoding="utf-8")
    marker["prompt_policy"] = "pics_v3"
    (run / "STAGE_COMPLETE.json").write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(RuntimeError):
        resolve_aamas_v0_source_program(
            target_dataset="2plonsky2018when",
            source_dataset="1peterson2021using",
            repo_root=tmp_path,
            config_path=cfg,
        )


def test_generation_does_not_stack_generic_reminder():
    import teh

    configure_pics_aamas_v0_prompt(True)
    try:
        assert using_pics_aamas_v0_prompt()
        text = teh._build_psych_prompt_text(
            base_prompt="Registered description.\n",
            state_text="",
            extra_state_text="",
            parent_context="",
            code_template_suffix="",
            candidate_output_rules="",
            runtime_contract="",
            dataset="5speekenbrink2008learning",
        )
    finally:
        configure_pics_aamas_v0_prompt(False)
    assert "HISTORY_ROBUSTNESS_BLOCK_V3" not in text
    assert _GENERIC_SENTENCE not in text
    assert "SEQUENTIAL_RL_BEHAVIOR_REMINDER" not in text


def test_rendered_prompts_for_all_fifteen(tmp_path: Path):
    from utils.teh.teh_runtime import setup_teh_run_prompts

    seed_uniform = REPO / "persona_code_example/teh/categorical_uniform.py"
    seed_choices = REPO / "persona_code_example/te_vanilla/choices13k.py"
    rows = []
    for dataset in CANONICAL_DATASETS:
        seed = seed_uniform if dataset in ("steyvers_2009_bandit", "13schulz2020finding") else seed_choices
        prompts = setup_teh_run_prompts(
            tmp_path / dataset,
            dataset,
            seed,
            client=None,
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            use_llm=False,
            prefer_auto_llm_prompt=False,
            require_auto_llm_prompt=False,
            split_ratio=0.6,
            split_seed=0,
            limited_data_protocol="structure_aware_v3",
            limited_train_val=40,
            pics_aamas_v0_prompt=True,
        )
        infer = (prompts / "infer_single_choice.txt").read_text(encoding="utf-8")
        meta = json.loads((prompts / "prompt_meta.json").read_text(encoding="utf-8"))
        decision = reminder_decision(dataset)
        assert meta["prompt_mode"] == "aamas_v0_registered"
        assert meta["llm_generated"] is False
        assert meta["pics_v3_reminder_policy"] == "pics_aamas_v0"
        assert meta["pics_aamas_v0_reminder_id"] == decision.reminder_id
        assert meta["prompt_examples_exclude_test"] is True
        assert "auto_llm" not in infer
        assert BEHAVIOR_OBJECTIVE_MARKER in infer
        assert infer.count(BEHAVIOR_OBJECTIVE_MARKER) == 2
        assert "maximize behavioral log-likelihood" in infer
        assert "optimal reward policy" in infer
        assert RUNTIME_CONTRACT_HEADER in infer
        assert _GENERIC_SENTENCE not in infer
        assert _SPEEK_REMINDER_SENTENCE not in infer
        if dataset in ("steyvers_2009_bandit", "13schulz2020finding"):
            assert "SEQUENTIAL_RL_BEHAVIOR_REMINDER_V4_" in infer
            assert "HISTORY_ROBUSTNESS_BLOCK_V3" not in infer
            assert decision.block in infer
        elif dataset == "14kool2016when":
            assert "SEQUENTIAL_RL_BEHAVIOR_REMINDER_V4" not in infer
            assert infer.count("HISTORY_ROBUSTNESS_BLOCK_V3") == 2
            assert dataset_keyed_reminder_body(dataset) in infer
        elif dataset in ("12badham2017deficits", "guan_2020_stopping"):
            assert "SEQUENTIAL_RL_BEHAVIOR_REMINDER_V4" not in infer
            assert infer.count("HISTORY_ROBUSTNESS_BLOCK_V3") == 2
            assert dataset_keyed_reminder_body(dataset) in infer
        else:
            assert "HISTORY_ROBUSTNESS_BLOCK_V3" not in infer
            assert "SEQUENTIAL_RL_BEHAVIOR_REMINDER" not in infer
        budget = prompt_budget_report(infer)
        assert budget["base_under_cap"], (dataset, budget)
        rows.append({"dataset": dataset, "reminder_id": decision.reminder_id, **budget, "sha256": meta["infer_prompt_sha256"]})
    out = REPO / "analysis_2026Sep/Sep27_aamas_v0/others/aamas_v0/prompt_render_budget.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def test_launchers_are_isolated():
    superseded = (REPO / "cluster/v0/ours/main/aamas_v0/source_population/submit_source_population.sh").read_text(encoding="utf-8")
    assert "superseded" in superseded
    assert "target_only" in superseded
    job = (REPO / "cluster/v0/ours/main/aamas_v0/job_track_h100.sh").read_text(encoding="utf-8")
    batch1 = (REPO / "cluster/v0/ours/main/aamas_v0/target_only/submit_batch1_target_only.sh").read_text(encoding="utf-8")
    assert "#SBATCH --requeue" in job
    assert "h100nvl" in job
    assert "aamas_v0_fill_track_args" in job
    assert "selected/STAGE_COMPLETE.json" in job
    assert "TEMPORARY" in batch1 or "temporary" in batch1
    assert "2plonsky2018when" in batch1 and "guan_2020_stopping" in batch1
    assert "5speekenbrink2008learning" in batch1
    fill = (REPO / "cluster/v0/ours/main/aamas_v0/_fill.sh").read_text(encoding="utf-8")
    assert "aamas_v0_fill_track_args" in fill
    assert "--pics_aamas_v0_track_mode" in fill
    common = (REPO / "cluster/Old/2026Sep/Qwen/_common.sh").read_text(encoding="utf-8")
    assert "t_pics_v3_fill_gated_args()" in common
    gated = common[common.index("t_pics_v3_fill_gated_args()"): common.index("t_pics_v3_fill_ij_hybrid_args()")]
    assert "pics_aamas_v0" not in gated


def test_materialize_gate_matches_provenance(tmp_path: Path):
    def _arm(root: Path, kind: str, role: str, score_program: str, source: str | None = None):
        arm = root / "target_population" / ("control" if role == "target_only" else "transfer")
        (arm / "global_phase" / "global_elite_pool").mkdir(parents=True)
        program = arm / "global_phase" / "best_program.py"
        program.write_text(score_program, encoding="utf-8")
        (arm / "global_phase" / "global_elite_pool" / "pool_manifest.json").write_text("{}", encoding="utf-8")
        digest = hashlib.sha256(program.read_bytes()).hexdigest()
        prov = {
            "dataset": "2plonsky2018when",
            "kind": kind,
            "track_mode": "target_only" if role == "target_only" else "transfer_based_only",
            "prompt_policy": "pics_aamas_v0",
            "prompt_mode": "aamas_v0_registered",
            "rendered_prompt_sha256": "abc",
            "sa40_fingerprint": "fp",
            "limited_data_protocol": "structure_aware_v3",
            "limited_train_val": 40,
            "global_iters": 10,
            "n_candidates": 10,
            "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "hard_prompt_token_cap": 14000,
            "llm_max_tokens": 1024,
            "max_parent_chars": 5000,
            "vllm_max_model_len": 16384,
            "range_start_ordinal": 0,
            "range_end_ordinal": 29,
            "participant_ids": [1, 2],
            "test_used_for_selection": False,
            "rank1_program": str(program),
            "rank1_sha256": digest,
            "arm_role": role if role == "transfer" else "target_only",
            "source_dataset": source,
            "source_rank1_sha256": "src" if source else None,
        }
        (arm / "POPULATION_PROVENANCE.json").write_text(json.dumps(prov), encoding="utf-8")
        return root

    target_root = tmp_path / "generated_outputs/psych101_train/teh/2plonsky2018when/pics_aamas_v0_target_only/job_1"
    transfer_root = tmp_path / "generated_outputs/psych101_train/teh/2plonsky2018when/pics_aamas_v0_transfer_based_only/job_2"
    out = tmp_path / "generated_outputs/psych101_train/teh/2plonsky2018when/pics_aamas_v0_official_gate/job_3"
    out.mkdir(parents=True)
    _arm(target_root, "pics_aamas_v0_target_only", "target_only", "def choose(problem, history):\n    return {0: 0.5, 1: 0.5}\n")
    _arm(
        transfer_root,
        "pics_aamas_v0_transfer_based_only",
        "transfer",
        "def choose(problem, history):\n    return {0: 0.4, 1: 0.6}\n",
        source="1peterson2021using",
    )
    seen = {}

    def _score(program, prov):
        assert prov["test_used_for_selection"] is False
        text = Path(program).read_text(encoding="utf-8")
        seen["test"] = "test" in text
        return -0.5 if "0.4" in text else -0.8

    record = materialize_official_gate(
        target_only_run=target_root,
        transfer_run=transfer_root,
        output_dir=out,
        target_dataset="2plonsky2018when",
        evaluate_score=_score,
    )
    assert record["selected_arm"] == "transfer"
    assert record["test_used_for_gate"] is False
    assert (out / "selected" / "SELECTED_ARM.txt").read_text(encoding="utf-8").strip() == "transfer"
