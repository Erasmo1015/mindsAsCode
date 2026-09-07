"""CPU-only tests for Bergert & Nosofsky (2007) TEH adapter."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from data_modules.external.bergert_nosofsky_2007 import (
    DATASET_ALIAS,
    DEFAULT_DATA_DIR,
    list_participant_ids,
    load_bergert_nosofsky_2007_trials,
    load_bergert_tables,
    load_participant_raw_trials,
)
from utils.teh.teh_datasets import (
    PARTICIPANT_DATASETS,
    is_binary_loglik_dataset,
    is_bergert_nosofsky_2007_dataset,
    is_mixed_gambles_dataset,
    teh_output_base_dir,
    valid_participant_ids_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / DEFAULT_DATA_DIR
SPLIT_RATIO = 0.8
SPLIT_SEED = 42


@pytest.fixture(scope="module")
def tables():
    if not DATA_DIR.is_dir():
        pytest.skip(f"Bergert data missing at {DATA_DIR}")
    return load_bergert_tables(DATA_DIR)


def test_registry_wires_bergert() -> None:
    assert DATASET_ALIAS in PARTICIPANT_DATASETS
    assert is_bergert_nosofsky_2007_dataset(DATASET_ALIAS)
    assert is_binary_loglik_dataset(DATASET_ALIAS)
    assert not is_mixed_gambles_dataset(DATASET_ALIAS)
    assert "mixed_gambles" in PARTICIPANT_DATASETS
    path = valid_participant_ids_path(DATASET_ALIAS, REPO_ROOT)
    assert path == REPO_ROOT / "datasets/external/bergert_nosofsky_2007/valid_participant_ids.json"
    out = teh_output_base_dir(DATASET_ALIAS, "ts")
    assert out == "generated_outputs/external/bergert_nosofsky_2007/teh/run_ts"


def test_counts_and_joins(tables) -> None:
    ids = list_participant_ids(DATA_DIR)
    assert len(ids) == 61
    assert ids[0] == 1 and ids[-1] == 61
    assert len(tables["problems"]) == 40
    assert len(tables["alternatives"]) == 8
    assert len(tables.get("cue_validities_meta") or []) == 6
    assert set(tables["alternatives"]) == set(range(1, 9))

    for pid in ids:
        trials = load_participant_raw_trials(pid, data_dir=DATA_DIR)
        assert len(trials) == 40
        actions = {t["action"] for t in trials}
        assert actions <= {0, 1}
        for t in trials:
            prob = t["problem"]
            assert set(prob["option_A"]["cues"]) == {
                "cue1",
                "cue2",
                "cue3",
                "cue4",
                "cue5",
                "cue6",
            }
            assert set(prob["option_B"]["cues"]) == set(prob["option_A"]["cues"])
            assert all(v in (0, 1) for v in prob["option_A"]["cues"].values())
            assert all(v in (0, 1) for v in prob["option_B"]["cues"].values())
            assert t["history"] == []
            # No choice leakage / no experimenter cue-validity leakage into problem.
            assert "choice" not in prob
            assert "action" not in prob
            assert "cue_validities" not in prob
            assert "log_odds_weight" not in str(prob)


def test_split_atomicity_no_problem_overlap() -> None:
    train, val, test, options = load_bergert_nosofsky_2007_trials(
        1, data_dir=DATA_DIR, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    assert options == [0, 1]
    assert len(train) + len(val) + len(test) == 40
    assert train and val and test
    train_ids = {t["problem"]["problem_id"] for t in train}
    val_ids = {t["problem"]["problem_id"] for t in val}
    test_ids = {t["problem"]["problem_id"] for t in test}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert train_ids | val_ids | test_ids == set(range(1, 41))


def test_dummy_choose_loglik_is_log_half() -> None:
    from teh import evaluate_choice13k_program

    train, val, test, _ = load_bergert_nosofsky_2007_trials(
        7, data_dir=DATA_DIR, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    trials = train + val + test

    def choose(_problem, _history):
        return 0.5

    result = evaluate_choice13k_program(choose, trials, n_seeds=1)
    assert abs(result["avg_loglik"] - math.log(0.5)) < 1e-9


def test_emnlp_aliases_still_registered() -> None:
    # Smoke: core EMNLP-era aliases remain in the participant registry.
    for alias in (
        "1peterson2021using",
        "2plonsky2018when",
        "mixed_gambles",
    ):
        assert alias in PARTICIPANT_DATASETS
        assert is_binary_loglik_dataset(alias)
