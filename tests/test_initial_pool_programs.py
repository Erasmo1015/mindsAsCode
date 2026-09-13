"""Tests for participant initial-pool loading and sparse subset agreement."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
OLD_TRANSFER_RUN = (
    REPO
    / "generated_outputs_transfer"
    / "teh_transfer"
    / "run_260616_235649"
)
OLD_TARGET_BEST = OLD_TRANSFER_RUN / "1peterson2021using" / "global" / "best_program.py"
OLD_TRANSFER_BEST = (
    OLD_TRANSFER_RUN
    / "1peterson2021using"
    / "transfer"
    / "source=2plonsky2018when"
    / "best_program.py"
)
OLD_TARGET_POOL = (
    OLD_TRANSFER_RUN / "1peterson2021using" / "global" / "global_elite_pool"
)

DUMMY_CHOOSE = "def choose(problem, history):\n    return 0.5\n"


def _trials(n: int, prefix: str) -> List[Dict[str, Any]]:
    return [{"id": f"{prefix}{i}", "i": i} for i in range(n)]


def test_mle_trials_for_participant_applies_sparse_helper(monkeypatch):
    from baseline_methods import MLE

    train, val, test = _trials(12, "t"), _trials(4, "v"), _trials(5, "x")

    def fake_split(*_args, **_kwargs):
        return train, val, test, {}

    monkeypatch.setattr(MLE, "is_mixed_gambles_dataset", lambda _d: False)
    monkeypatch.setattr(MLE, "is_psych101_dataset", lambda _d: True)
    monkeypatch.setattr(MLE, "get_psych101_binary_experiment", lambda *a, **k: object())
    monkeypatch.setattr(MLE, "split_psych_experiment", fake_split)

    out_train, out_val, out_test, audit = MLE.trials_for_participant(
        "1peterson2021using",
        9,
        split_ratio=0.6,
        split_seed=0,
        filter_mixed_gambles=False,
        psych_dataset_split="train",
        local_dataset=None,
        mixed_gambles_csv="",
        max_observed_trials_per_participant=6,
        return_audit=True,
    )
    assert out_test == test
    assert len(out_train) + len(out_val) == 6
    assert audit.subset_fingerprint

    from utils.teh.sparse_observations import apply_max_observed_trials

    expected = apply_max_observed_trials(
        train,
        val,
        test,
        max_observed_trials_per_participant=6,
        dataset="1peterson2021using",
        participant_id=9,
        split_seed=0,
    )
    assert audit.subset_fingerprint == expected[3].subset_fingerprint
    assert out_train == expected[0]
    assert out_val == expected[1]


def test_prospect_theory_uses_same_sparse_subset(monkeypatch):
    from baseline_methods import prospect_theory as pt
    from utils.teh.sparse_observations import apply_max_observed_trials

    train, val, test = _trials(12, "t"), _trials(4, "v"), _trials(5, "x")

    def fake_split(*_args, **_kwargs):
        return train, val, test, {}

    monkeypatch.setattr(pt, "is_mixed_gambles_dataset", lambda _d: False)
    monkeypatch.setattr(pt, "is_psych101_dataset", lambda _d: True)
    monkeypatch.setattr(pt, "get_psych101_binary_experiment", lambda *a, **k: object())
    monkeypatch.setattr(pt, "split_psych_experiment", fake_split)

    out = pt.trials_for_participant(
        "1peterson2021using",
        9,
        split_ratio=0.6,
        split_seed=0,
        filter_mixed_gambles=False,
        psych_dataset_split="train",
        local_dataset=None,
        mixed_gambles_csv="",
        max_observed_trials_per_participant=6,
        return_audit=True,
    )
    expected = apply_max_observed_trials(
        train,
        val,
        test,
        max_observed_trials_per_participant=6,
        dataset="1peterson2021using",
        participant_id=9,
        split_seed=0,
    )
    assert out[3].subset_fingerprint == expected[3].subset_fingerprint


def test_load_one_and_two_initial_programs(tmp_path: Path):
    import teh

    p1 = tmp_path / "target_pop_best.py"
    p2 = tmp_path / "transfer_pop_best.py"
    p1.write_text(DUMMY_CHOOSE, encoding="utf-8")
    p2.write_text(
        "def choose(problem, history):\n    return 0.7\n",
        encoding="utf-8",
    )
    one = teh._load_initial_pool_program_files([str(p1)])
    assert len(one) == 1
    assert one[0][3] == "global_target_pop_best"
    two = teh._load_initial_pool_program_files([str(p1), str(p2)])
    assert len(two) == 2
    assert two[0][3] != two[1][3]
    assert teh.compile_program(two[0][0]) is not None
    assert teh.compile_program(two[1][0]) is not None


def test_load_old_transfer_best_programs_if_present():
    import teh

    if not OLD_TARGET_BEST.is_file() or not OLD_TRANSFER_BEST.is_file():
        pytest.skip("old 2plonsky→1peterson transfer artifacts not present")
    loaded = teh._load_initial_pool_program_files(
        [str(OLD_TARGET_BEST), str(OLD_TRANSFER_BEST)]
    )
    assert len(loaded) == 2
    assert loaded[0][3] != loaded[1][3]
    assert teh.compile_program(loaded[0][0]) is not None
    assert teh.compile_program(loaded[1][0]) is not None
    if (OLD_TARGET_POOL / "pool_manifest.json").is_file():
        pool = teh._load_global_elite_pool(OLD_TARGET_POOL)
        assert len(pool) >= 1
        assert teh.compile_program(pool[0][0]) is not None


def test_explore_uses_loaded_initial_programs(monkeypatch, tmp_path: Path):
    import teh

    seen_parents: List[str] = []

    def fake_gen(**kwargs):
        seen_parents.append(kwargs["parent_programs"][0])
        n = int(kwargs["n_variants"])
        return [
            f"def choose(problem, history):\n    return {0.11 + 0.01 * i}\n"
            for i in range(n)
        ]

    monkeypatch.setattr(teh, "generate_program_variants", fake_gen)
    monkeypatch.setattr(
        teh,
        "_evaluate_loglik_for_dataset",
        lambda *a, **k: {"avg_loglik": -0.5, "accuracy": 0.6},
    )
    monkeypatch.setattr(
        teh, "compile_program", lambda code: (lambda problem, history: 0.5)
    )
    monkeypatch.setattr(teh, "is_binary_loglik_dataset", lambda _d: True)

    parent_a = "def choose(problem, history):\n    return 0.2\n"
    parent_b = "def choose(problem, history):\n    return 0.8\n"
    elite = [
        (parent_a, -0.4, 0.5, "global_target", None, None, -0.4),
        (parent_b, -0.45, 0.5, "global_transfer", None, None, -0.45),
    ]
    val_ll = [-0.4, -0.45]
    teh._run_pre_evolution_explore_phase(
        explore_candidates=4,
        client=MagicMock(),
        model_name="dummy",
        seed_code=DUMMY_CHOOSE,
        dataset="1peterson2021using",
        participant_id=0,
        train_trials=[{"problem": {}, "action": 0}],
        test_trials=[{"problem": {}, "action": 1}],
        val_trials=[{"problem": {}, "action": 0}],
        fitness_metric="loglik",
        n_eval_seeds=1,
        elite_parents=elite,
        elite_val_logliks=val_ll,
        track_elite_val_loglik=True,
        sample_size=8,
        elite_pool_size=20,
        baseline_train_eval={"avg_loglik": -0.7, "accuracy": 0.5},
        output_path=tmp_path,
        save_artifacts=False,
        max_prompt_train_trials=10,
        max_prompt_trials_per_problem=5,
        llm_max_tokens=64,
        max_workers=1,
        split_seed=0,
        run_prompts_dir=None,
        max_parent_chars=0,
        warn_parent_truncation_ratio=0.5,
        sample_size_for_warning=2,
        hard_prompt_token_cap=14000,
        strict_prompt_budget=False,
        prompt_token_estimator="chars",
        initial_pool_from_global=True,
        initial_pool_size_before_explore=2,
        evolution_selection_score="train_val",
        explore_parent_programs=[
            (parent_a, "global_target"),
            (parent_b, "global_transfer"),
        ],
        pin_program_ids=["global_target", "global_transfer"],
    )
    assert seen_parents == [parent_a, parent_b]
    ids = {str(p[3]) for p in elite}
    assert "global_target" in ids
    assert "global_transfer" in ids


def test_cap_preserves_pinned_initial_programs():
    import teh

    elite = [
        ("code_a", 0.1, 0.0, "global_target", None, None, 0.1),
        ("code_b", 0.9, 0.0, "explore_candidate_0", None, None, 0.9),
        ("code_c", 0.8, 0.0, "explore_candidate_1", None, None, 0.8),
    ]
    val = [0.1, 0.9, 0.8]
    teh._cap_elite_preserving_program_ids(
        elite,
        val,
        elite_cap=2,
        pinned_ids=["global_target"],
        track_elite_val_loglik=True,
    )
    ids = [str(p[3]) for p in elite]
    assert "global_target" in ids
    assert len(elite) == 2


def test_cli_rejects_initial_pool_with_live_global_phase():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "teh.py"),
            "--dataset",
            "1peterson2021using",
            "--fitness_metric",
            "loglik",
            "--no_log",
            "--initial_pool_programs",
            str(REPO / "teh.py"),
            "--global_phase",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "cannot be combined" in combined


def test_cli_help_exposes_new_flags_and_old_help_still_works():
    teh_help = subprocess.run(
        [sys.executable, str(REPO / "teh.py"), "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    assert "--max_observed_trials_per_participant" in teh_help
    assert "--initial_pool_programs" in teh_help
    assert "--initial_pool_dir" in teh_help
    assert "--explore_from_population_parents" in teh_help
    assert "--explore_population_top_k" in teh_help
    transfer_help = subprocess.run(
        [sys.executable, str(REPO / "teh_transfer.py"), "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    assert "--max_observed_trials_per_participant" in transfer_help
    assert "--max_observed_trials_datasets" in transfer_help
    assert "--rerun_global_datasets" in transfer_help
    assert "--transfer_source_keys" in transfer_help


def test_filter_transfer_jobs_keeps_plonsky_to_peterson_only():
    from utils.teh_transfer.transfer_jobs import TransferJob, filter_transfer_jobs

    jobs = [
        TransferJob(
            target_key="1peterson2021using",
            source_keys=("2plonsky2018when",),
            transfer_mode="single",
        ),
        TransferJob(
            target_key="2plonsky2018when",
            source_keys=("1peterson2021using",),
            transfer_mode="single",
        ),
    ]
    kept = filter_transfer_jobs(
        jobs,
        source_keys=["2plonsky2018when"],
        target_keys=["1peterson2021using"],
    )
    assert len(kept) == 1
    assert kept[0].target_key == "1peterson2021using"
    assert kept[0].source_keys == ("2plonsky2018when",)
