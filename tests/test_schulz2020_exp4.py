"""CPU-only tests for Schulz 2020 Exp4 (Psych-101 categorical K=8)."""
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
from data_modules.psych101_extensions.schulz2020_exp4 import (
    DATASET_ALIAS,
    DEFAULT_VALIDATION_CSV,
    N_ACTIONS,
    assert_psych_matches_dynamicdata,
    match_psych_row_to_dynamicdata_id,
    raw_press_to_action,
    split_schulz2020_exp4_experiment,
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
VALIDATION_CSV = REPO_ROOT / DEFAULT_VALIDATION_CSV
SPLIT_RATIO = 0.8
SPLIT_SEED = 42
LOG_EIGHTH = math.log(1.0 / 8.0)


@pytest.fixture(scope="module")
def require_psych():
    if not LOCAL_PSYCH.is_dir():
        pytest.skip(f"Local Psych-101 missing at {LOCAL_PSYCH}")


@pytest.fixture(scope="module")
def require_validation_csv():
    if not VALIDATION_CSV.is_file():
        pytest.skip(
            f"Schulz validation CSV missing at {VALIDATION_CSV}; "
            "run scripts/setup_external_schulz2020_exp4.py"
        )


@pytest.fixture(scope="module")
def filtered(require_psych):
    return get_filtered_psych101_split(
        DATASET_ALIAS, split="train", local_dataset=str(LOCAL_PSYCH)
    )


@pytest.fixture(scope="module")
def exp0(filtered):
    return parse_psych101_binary_row(dict(filtered[0]), DATASET_ALIAS)


def test_registry_and_output_type() -> None:
    assert DATASET_ALIAS in PARTICIPANT_DATASETS
    assert is_psych101_dataset(DATASET_ALIAS)
    assert is_binary_loglik_dataset(DATASET_ALIAS)
    assert is_categorical_output_dataset(DATASET_ALIAS)
    assert not is_bernoulli_output_dataset(DATASET_ALIAS)
    assert dataset_output_type(DATASET_ALIAS) == "categorical"
    assert PSYCH101_BINARY_DATASETS[DATASET_ALIAS]["n_actions"] == 8
    assert PSYCH101_BINARY_DATASETS[DATASET_ALIAS]["experiment_id"] == (
        "schulz2020finding/exp4.csv"
    )
    # Bernoulli / Steyvers unchanged
    assert is_bernoulli_output_dataset("bergert_nosofsky_2007")
    assert is_bernoulli_output_dataset("guan_2020_stopping")
    assert is_categorical_output_dataset("steyvers_2009_bandit")
    path = valid_participant_ids_path(DATASET_ALIAS, REPO_ROOT)
    assert path == (
        REPO_ROOT / "datasets/psych101_train/13schulz2020finding/valid_participant_ids.json"
    )
    assert teh_output_base_dir(DATASET_ALIAS, "ts") == (
        "generated_outputs/psych101_train/teh/13schulz2020finding/run_ts"
    )


def test_raw_vs_internal_action_coding() -> None:
    assert raw_press_to_action(1) == 0
    assert raw_press_to_action(8) == 7
    with pytest.raises(ValueError):
        raw_press_to_action(0)
    with pytest.raises(ValueError):
        raw_press_to_action(9)


def test_complete_participant_counts(exp0) -> None:
    assert len(exp0.blocks) == 30
    assert all(len(b.trials) == 10 for b in exp0.blocks)
    assert sum(len(b.trials) for b in exp0.blocks) == 300
    actions = {t.action for b in exp0.blocks for t in b.trials}
    assert actions <= set(range(N_ACTIONS))


def test_history_reset_and_no_structure_leakage(exp0) -> None:
    trials = experiment_to_trial_dicts(exp0)
    assert len(trials) == 300
    for t in trials:
        assert len(t["history"]) == int(t["problem"]["trial"]) - 1
        if int(t["problem"]["trial"]) == 1:
            assert t["history"] == []
        for h in t["history"]:
            assert set(h.keys()) == {"action", "reward"}
            assert h["action"] in range(N_ACTIONS)
            assert isinstance(h["reward"], float)
        prob = t["problem"]
        blob = (str(prob) + str(t["history"])).lower()
        for banned in (
            "cond",
            "rcond",
            "structure-random",
            "srs",
            "rsr",
            "pos",
            "neg",
            "ran",
        ):
            # avoid matching substrings inside unrelated words: check keys / tokens
            assert banned not in {k.lower() for k in prob.keys()}
            if banned in ("srs", "rsr", "pos", "neg", "ran"):
                continue  # soft skip token search for short codes
            assert banned not in blob
        assert [o["action"] for o in prob["options"]] == list(range(8))
        assert prob["raw_press_coding"] == "1-8"
        assert prob["internal_action_coding"] == "0-7"


def test_split_by_whole_rounds(exp0) -> None:
    train, val, test, options = split_schulz2020_exp4_experiment(
        exp0, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    assert options == list(range(8))
    assert len(train) + len(val) + len(test) == 300

    def rounds(trials):
        return {t["problem"]["round"] for t in trials}

    tr, vr, sr = rounds(train), rounds(val), rounds(test)
    assert not (tr & vr)
    assert not (tr & sr)
    assert not (vr & sr)
    assert tr | vr | sr == set(range(1, 31))
    for trials, rset in ((train, tr), (val, vr), (test, sr)):
        for r in rset:
            n = sum(1 for t in trials if t["problem"]["round"] == r)
            assert n == 10


def test_split_psych_experiment_also_finalizes(exp0) -> None:
    train, _, _, options = split_psych_experiment(
        exp0, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    assert options == list(range(8))
    assert "options" in train[0]["problem"]
    assert isinstance(train[0]["problem"]["options"][0], dict)
    nonempty = next(t for t in train if t["history"])
    assert set(nonempty["history"][0].keys()) == {"action", "reward"}


def test_uniform_categorical_loglik(exp0) -> None:
    from teh import _evaluate_loglik_for_dataset

    train, val, test, _ = split_schulz2020_exp4_experiment(
        exp0, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    trials = train + val + test

    def choose(problem, history):
        options = problem["options"]
        p = 1.0 / len(options)
        return {option["action"]: p for option in options}

    result = _evaluate_loglik_for_dataset(DATASET_ALIAS, choose, trials, n_seeds=1)
    assert abs(result["avg_loglik"] - LOG_EIGHTH) < 1e-9
    assert abs(result["avg_loglik"] - (-2.0794415416798357)) < 1e-9


def test_psych_matches_dynamicdata_sample(exp0, require_validation_csv, filtered) -> None:
    sid = assert_psych_matches_dynamicdata(exp0, csv_path=VALIDATION_CSV)
    assert sid == match_psych_row_to_dynamicdata_id(exp0, csv_path=VALIDATION_CSV)

    # Spot-check a few more Psych rows against CSV
    for idx in (1, 10, min(50, len(filtered) - 1)):
        exp = parse_psych101_binary_row(dict(filtered[idx]), DATASET_ALIAS)
        assert_psych_matches_dynamicdata(exp, csv_path=VALIDATION_CSV)


def test_filtered_row_count(filtered) -> None:
    assert len(filtered) == 99


def test_get_experiment_loader(require_psych) -> None:
    exp = get_psych101_binary_experiment(
        DATASET_ALIAS, 0, split="train", local_dataset=str(LOCAL_PSYCH)
    )
    assert len(exp.blocks) == 30


def test_static_prompt_has_no_hidden_structure() -> None:
    prompt = (REPO_ROOT / "prompts/external/schulz2020_exp4.txt").read_text(
        encoding="utf-8"
    )
    low = prompt.lower()
    assert "structure-random" not in low
    assert "rcond" not in low
    assert "cond" not in low or "condition names" in low  # banlist wording OK
    assert "action = raw_press - 1" in prompt
    assert "0..7" in prompt or "0–7" in prompt
    assert "reset" in low


def test_steyvers_still_categorical_unchanged() -> None:
    assert is_categorical_output_dataset("steyvers_2009_bandit")
    assert dataset_output_type("steyvers_2009_bandit") == "categorical"
    assert is_bernoulli_output_dataset("1peterson2021using")
