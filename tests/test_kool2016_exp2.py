"""CPU-only tests for Kool 2016 Daw two-step (Psych-101 Bernoulli, dual-stage)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from data_modules.psych101_binary import (
    PSYCH101_BINARY_DATASETS,
    experiment_to_trial_dicts,
    get_filtered_psych101_split,
    get_psych101_binary_experiment,
    parse_psych101_binary_row,
    split_psych_experiment,
)
from data_modules.psych101_extensions.kool2016_exp2 import (
    DATASET_ALIAS,
    DEFAULT_VALIDATION_MAT,
    N_PRESENTED_DAYS,
    assert_no_stage_leakage,
    assert_psych_matches_groupdata,
    assert_split_keeps_days_together,
    build_kool_decision_trials,
    raw_stage1_press_to_action,
    split_kool2016_exp2_experiment,
)
from utils.teh.teh_datasets import (
    PARTICIPANT_DATASETS,
    dataset_output_type,
    is_bernoulli_output_dataset,
    is_binary_loglik_dataset,
    is_categorical_output_dataset,
    is_psych101_dataset,
    teh_output_base_dir,
    valid_participant_ids_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PSYCH = REPO_ROOT / "datasets" / "downloaded" / "Psych-101"
VALIDATION_MAT = REPO_ROOT / DEFAULT_VALIDATION_MAT
SPLIT_RATIO = 0.8
SPLIT_SEED = 42
LOG_HALF = math.log(0.5)


@pytest.fixture(scope="module")
def require_psych():
    if not LOCAL_PSYCH.is_dir():
        pytest.skip(f"Local Psych-101 missing at {LOCAL_PSYCH}")


@pytest.fixture(scope="module")
def require_validation_mat():
    if not VALIDATION_MAT.is_file():
        pytest.skip(
            f"Kool validation MAT missing at {VALIDATION_MAT}; "
            "run scripts/setup_external_kool2016_exp2.py"
        )


@pytest.fixture(scope="module")
def filtered(require_psych):
    return get_filtered_psych101_split(
        DATASET_ALIAS, split="train", local_dataset=str(LOCAL_PSYCH)
    )


@pytest.fixture(scope="module")
def row0(filtered):
    return dict(filtered[0])


@pytest.fixture(scope="module")
def exp0(row0):
    return parse_psych101_binary_row(row0, DATASET_ALIAS)


def test_registry_and_output_type() -> None:
    assert DATASET_ALIAS in PARTICIPANT_DATASETS
    assert is_psych101_dataset(DATASET_ALIAS)
    assert is_binary_loglik_dataset(DATASET_ALIAS)
    assert is_bernoulli_output_dataset(DATASET_ALIAS)
    assert not is_categorical_output_dataset(DATASET_ALIAS)
    assert dataset_output_type(DATASET_ALIAS) == "bernoulli"
    assert PSYCH101_BINARY_DATASETS[DATASET_ALIAS]["experiment_id"] == (
        "kool2016when/exp2.csv"
    )
    assert is_categorical_output_dataset("steyvers_2009_bandit")
    assert is_categorical_output_dataset("13schulz2020finding")
    assert is_bernoulli_output_dataset("bergert_nosofsky_2007")
    path = valid_participant_ids_path(DATASET_ALIAS, REPO_ROOT)
    assert path == (
        REPO_ROOT / "datasets/psych101_train/14kool2016when/valid_participant_ids.json"
    )
    assert teh_output_base_dir(DATASET_ALIAS, "ts") == (
        "generated_outputs/psych101_train/teh/14kool2016when/run_ts"
    )


def test_raw_vs_internal_action_coding() -> None:
    assert raw_stage1_press_to_action("B", ["B", "L"]) == 0
    assert raw_stage1_press_to_action("L", ["B", "L"]) == 1
    assert raw_stage1_press_to_action("L", ["L", "B"]) == 0
    with pytest.raises(ValueError):
        raw_stage1_press_to_action("X", ["B", "L"])


def test_counts_and_continuous_history(exp0, row0) -> None:
    trials = build_kool_decision_trials(exp0)
    # Complete subject 0: 125 complete days × 2 = 250 decisions
    assert len(trials) == 250
    assert {t["problem"]["stage"] for t in trials} == {1, 2}
    assert_no_stage_leakage(trials)
    # History grows across days (not reset).
    assert trials[0]["history"] == []
    assert len(trials[2]["history"]) == 2  # after day1 both stages
    assert trials[1]["problem"]["stage"] == 2
    assert trials[1]["problem"]["planet"]
    assert "reward" not in trials[1]["problem"]
    assert "planet" not in trials[0]["problem"]


def test_split_contiguous_keeps_days(exp0) -> None:
    train, val, test, _ = split_kool2016_exp2_experiment(
        exp0, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    assert_split_keeps_days_together(train, val, test)
    assert len(train) + len(val) + len(test) == 250
    # Contiguous cut on 125 usable days @ 0.8 → 100 / 13 / 12 days × 2
    assert len(train) == 200
    assert len(val) == 26
    assert len(test) == 24


def test_split_psych_experiment_routes_kool(exp0) -> None:
    train, val, test, _ = split_psych_experiment(
        exp0, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    assert_split_keeps_days_together(train, val, test)


def test_experiment_to_trial_dicts_continuous(exp0) -> None:
    trials = experiment_to_trial_dicts(exp0)
    assert_no_stage_leakage(trials)
    assert len(trials[0]["history"]) == 0
    assert len(trials[-1]["history"]) >= 2


def test_uniform_bernoulli_loglik(exp0) -> None:
    from teh import _evaluate_loglik_for_dataset, evaluate_choice13k_program

    train, val, test, _ = split_kool2016_exp2_experiment(
        exp0, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    trials = train + val + test

    def choose(_p, _h):
        return 0.5

    via_dispatch = _evaluate_loglik_for_dataset(DATASET_ALIAS, choose, trials, n_seeds=1)
    via_binary = evaluate_choice13k_program(choose, trials, n_seeds=1)
    assert abs(via_dispatch["avg_loglik"] - LOG_HALF) < 1e-9
    assert abs(via_dispatch["avg_loglik"] - via_binary["avg_loglik"]) < 1e-12


def test_psych_matches_groupdata_samples(filtered, require_validation_mat) -> None:
    for idx in (0, 2, 50, 100, 187):
        text = dict(filtered[idx])["text"]
        assert_psych_matches_groupdata(text, mat_path=VALIDATION_MAT)


def test_stage2_timeout_emits_stage1_only(filtered) -> None:
    # Row 2 has a stage1-then-timeout day.
    row = dict(filtered[2])
    exp = parse_psych101_binary_row(row, DATASET_ALIAS)
    trials = build_kool_decision_trials(exp)
    assert_no_stage_leakage(trials)
    # Fewer than 250 decisions due to timeouts.
    assert len(trials) < 250
    assert len(trials) % 1 == 0
    lone = [
        t
        for t in trials
        if int(t["problem"]["stage"]) == 1
        and not any(
            u
            for u in trials
            if int(u["problem"]["presented_day"]) == int(t["problem"]["presented_day"])
            and int(u["problem"]["stage"]) == 2
        )
    ]
    assert len(lone) >= 1


def test_filtered_row_count(filtered) -> None:
    assert len(filtered) == 188


def test_author_vs_psych_n_documented(require_validation_mat) -> None:
    from data_modules.psych101_extensions.kool2016_exp2 import load_groupdata_subjects

    assert len(load_groupdata_subjects(VALIDATION_MAT)) == 206
    # Psych-101 train is a subset (188); practice already removed (125 days).


def test_static_prompt_documents_stages_and_history() -> None:
    prompt = (REPO_ROOT / "prompts/external/kool2016_exp2.txt").read_text(
        encoding="utf-8"
    )
    low = prompt.lower()
    assert 'problem["stage"]' in prompt or "problem[\"stage\"]" in prompt
    assert "same" in low and "stage" in low
    assert "session_continuous" in prompt or "across days" in low
    assert "p(action=1)" in low or "P(action=1)" in prompt
    assert "treasure" in low
    assert "must not" in low


def test_get_experiment_loader(require_psych) -> None:
    exp = get_psych101_binary_experiment(
        DATASET_ALIAS, 0, split="train", local_dataset=str(LOCAL_PSYCH)
    )
    assert exp.schema_type == "kool_twostep"
    assert len(exp.blocks) == N_PRESENTED_DAYS  # subject 0: all complete
