"""CPU-only tests for Steyvers 2009 bandit TEH adapter (first categorical dataset)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from data_modules.external.steyvers_2009_bandit import (
    DATASET_ALIAS,
    DEFAULT_DATA_DIR,
    list_participant_ids,
    load_participant_raw_trials,
    load_steyvers_2009_bandit_trials,
    raw_choice_to_action,
)
from utils.teh.teh_datasets import (
    PARTICIPANT_DATASETS,
    dataset_output_type,
    is_bernoulli_output_dataset,
    is_binary_loglik_dataset,
    is_categorical_output_dataset,
    is_steyvers_2009_bandit_dataset,
    teh_output_base_dir,
    valid_participant_ids_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / DEFAULT_DATA_DIR
SPLIT_RATIO = 0.8
SPLIT_SEED = 42
LOG_QUARTER = math.log(0.25)


@pytest.fixture(scope="module")
def require_data():
    if not (DATA_DIR / "trials.csv").is_file():
        pytest.skip(f"Steyvers trials missing at {DATA_DIR}")


def test_registry_and_output_type(require_data) -> None:
    assert DATASET_ALIAS in PARTICIPANT_DATASETS
    assert is_steyvers_2009_bandit_dataset(DATASET_ALIAS)
    assert is_binary_loglik_dataset(DATASET_ALIAS)  # historical eligibility predicate
    assert is_categorical_output_dataset(DATASET_ALIAS)
    assert not is_bernoulli_output_dataset(DATASET_ALIAS)
    assert dataset_output_type(DATASET_ALIAS) == "categorical"
    # Bernoulli externals unchanged
    assert is_bernoulli_output_dataset("bergert_nosofsky_2007")
    assert is_bernoulli_output_dataset("guan_2020_stopping")
    assert is_bernoulli_output_dataset("mixed_gambles")
    path = valid_participant_ids_path(DATASET_ALIAS, REPO_ROOT)
    assert path == REPO_ROOT / "datasets/external/steyvers_2009_bandit/valid_participant_ids.json"
    assert teh_output_base_dir(DATASET_ALIAS, "ts") == (
        "generated_outputs/external/steyvers_2009_bandit/teh/run_ts"
    )


def test_raw_vs_internal_action_coding() -> None:
    assert raw_choice_to_action(1) == 0
    assert raw_choice_to_action(4) == 3
    with pytest.raises(ValueError):
        raw_choice_to_action(0)


def test_counts_history_reset_and_leakage(require_data) -> None:
    ids = list_participant_ids(DATA_DIR)
    assert len(ids) == 451
    assert ids[0] == 1 and ids[-1] == 451

    trials = load_participant_raw_trials(1, data_dir=DATA_DIR)
    assert len(trials) == 300
    games = {t["problem"]["game"] for t in trials}
    assert games == set(range(1, 21))
    assert {t["action"] for t in trials} <= {0, 1, 2, 3}

    # History resets per game; length == trial-1 within game.
    for t in trials:
        assert len(t["history"]) == int(t["problem"]["trial"]) - 1
        if int(t["problem"]["trial"]) == 1:
            assert t["history"] == []
        for h in t["history"]:
            assert set(h.keys()) == {"action", "reward"}
            assert h["action"] in (0, 1, 2, 3)
            assert h["reward"] in (0, 1)

        prob = t["problem"]
        assert "rewardRateChosen" not in prob
        assert "gameRewardRates" not in prob
        assert "reward_rate" not in str(prob).lower()
        blob = str(prob) + str(t["history"])
        assert "rewardRate" not in blob
        assert "0.5613" not in blob  # sample latent rate from game 1 arm1

        opts = prob["options"]
        assert [o["action"] for o in opts] == [0, 1, 2, 3]


def test_split_by_whole_games(require_data) -> None:
    train, val, test, options = load_steyvers_2009_bandit_trials(
        1, data_dir=DATA_DIR, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    assert options == [0, 1, 2, 3]
    assert len(train) + len(val) + len(test) == 300

    def games(trials):
        return {t["problem"]["game"] for t in trials}

    tg, vg, sg = games(train), games(val), games(test)
    assert not (tg & vg)
    assert not (tg & sg)
    assert not (vg & sg)
    assert tg | vg | sg == set(range(1, 21))
    # Every game contributes exactly 15 trials in its split.
    for trials, gset in ((train, tg), (val, vg), (test, sg)):
        for g in gset:
            n = sum(1 for t in trials if t["problem"]["game"] == g)
            assert n == 15


def test_uniform_categorical_loglik(require_data) -> None:
    from teh import _evaluate_loglik_for_dataset

    train, val, test, _ = load_steyvers_2009_bandit_trials(
        5, data_dir=DATA_DIR, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    trials = train + val + test

    def choose(problem, history):
        options = problem["options"]
        p = 1.0 / len(options)
        return {option["action"]: p for option in options}

    result = _evaluate_loglik_for_dataset(DATASET_ALIAS, choose, trials, n_seeds=1)
    assert abs(result["avg_loglik"] - LOG_QUARTER) < 1e-9
    assert math.isfinite(result["avg_loglik"])


def test_bernoulli_evaluator_path_untouched_for_bergert(require_data) -> None:
    """Bernoulli datasets still use float P(action=1) evaluator (~log 0.5)."""
    from data_modules.external.bergert_nosofsky_2007 import load_bergert_nosofsky_2007_trials
    from teh import _evaluate_loglik_for_dataset, evaluate_choice13k_program

    train, val, test, _ = load_bergert_nosofsky_2007_trials(
        1,
        data_dir=REPO_ROOT / "datasets/external/bergert_nosofsky_2007",
        split_ratio=SPLIT_RATIO,
        split_seed=SPLIT_SEED,
    )
    trials = train + val + test

    def choose(_p, _h):
        return 0.5

    via_dispatch = _evaluate_loglik_for_dataset(
        "bergert_nosofsky_2007", choose, trials, n_seeds=1
    )
    via_binary = evaluate_choice13k_program(choose, trials, n_seeds=1)
    assert abs(via_dispatch["avg_loglik"] - math.log(0.5)) < 1e-9
    assert abs(via_dispatch["avg_loglik"] - via_binary["avg_loglik"]) < 1e-12
