"""CPU-only tests for Guan et al. (2020) optimal-stopping TEH adapter."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from data_modules.external.guan_2020_stopping import (
    CONDITION_NAMES,
    DATASET_ALIAS,
    DEFAULT_DATA_DIR,
    expand_stopping_problem,
    list_participant_ids,
    load_guan_2020_stopping_trials,
    load_guan_raw,
    load_participant_raw_trials,
)
from utils.teh.teh_datasets import (
    PARTICIPANT_DATASETS,
    is_binary_loglik_dataset,
    is_guan_2020_stopping_dataset,
    is_mixed_gambles_dataset,
    teh_output_base_dir,
    valid_participant_ids_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / DEFAULT_DATA_DIR
SPLIT_RATIO = 0.8
SPLIT_SEED = 42


@pytest.fixture(scope="module")
def raw():
    if not (DATA_DIR / "OptimalStopping.mat").is_file():
        pytest.skip(f"Guan MAT missing at {DATA_DIR}")
    return load_guan_raw(DATA_DIR)


def test_registry_wires_guan() -> None:
    assert DATASET_ALIAS in PARTICIPANT_DATASETS
    assert is_guan_2020_stopping_dataset(DATASET_ALIAS)
    assert is_binary_loglik_dataset(DATASET_ALIAS)
    assert not is_mixed_gambles_dataset(DATASET_ALIAS)
    path = valid_participant_ids_path(DATASET_ALIAS, REPO_ROOT)
    assert path == REPO_ROOT / "datasets/external/guan_2020_stopping/valid_participant_ids.json"
    assert teh_output_base_dir(DATASET_ALIAS, "ts") == (
        "generated_outputs/external/guan_2020_stopping/teh/run_ts"
    )


def test_counts_and_expansion(raw) -> None:
    ids = list_participant_ids(DATA_DIR)
    assert len(ids) == 56
    assert ids == list(range(1, 57))
    assert tuple(int(x) for x in raw["nstim"]) == (4, 4, 8, 8)
    assert CONDITION_NAMES == (
        "length4_neutral",
        "length4_plentiful",
        "length8_neutral",
        "length8_plentiful",
    )

    trials = load_participant_raw_trials(1, data_dir=DATA_DIR)
    units = {t["problem_signature"] for t in trials}
    assert len(units) == 160

    # Per-unit expansion: S-1 continues + 1 stop; values_observed length == position.
    by_unit: dict = {}
    for t in trials:
        by_unit.setdefault(t["problem_signature"], []).append(t)
    for unit, utrials in by_unit.items():
        actions = [t["action"] for t in utrials]
        assert actions[-1] == 1
        assert actions[:-1] == [0] * (len(actions) - 1)
        for t in utrials:
            pos = int(t["problem"]["position"])
            observed = t["problem"]["values_observed"]
            assert len(observed) == pos
            assert t["problem"]["sequence_length"] in (4, 8)
            assert set(t["problem"]["values_observed"])  # non-empty
            # No future values in problem.
            assert "full_values" not in t["problem"]
            assert max(range(1, pos + 1)) == pos
            for h in t["history"]:
                assert h["action"] == 0
                assert h["position"] < pos


def test_no_future_values_invariant() -> None:
    full = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    expanded = expand_stopping_problem(
        participant_id=1,
        condition_index=2,
        condition="length8_neutral",
        problem_id=1,
        sequence_length=8,
        stop_position=5,
        full_values=full,
    )
    assert len(expanded) == 5
    for i, t in enumerate(expanded, start=1):
        assert t["problem"]["values_observed"] == full[:i]
        assert 80.0 not in t["problem"]["values_observed"] or i == 8
        assert all(v in full[:i] for v in t["problem"]["values_observed"])
        # Explicit: nothing beyond i
        assert t["problem"]["values_observed"][-1] == full[i - 1]
        assert len(t["problem"]["values_observed"]) == i


def test_split_keeps_stopping_problems_atomic() -> None:
    train, val, test, options = load_guan_2020_stopping_trials(
        1, data_dir=DATA_DIR, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    assert options == [0, 1]
    assert train and val and test

    def _units(trials):
        return {
            (t["problem"]["condition_index"], t["problem"]["problem_id"])
            for t in trials
        }

    train_u, val_u, test_u = _units(train), _units(val), _units(test)
    assert not (train_u & val_u)
    assert not (train_u & test_u)
    assert not (val_u & test_u)
    assert len(train_u | val_u | test_u) == 160

    # All expanded decisions for a unit stay in one split.
    for trials, units in ((train, train_u), (val, val_u), (test, test_u)):
        for t in trials:
            key = (t["problem"]["condition_index"], t["problem"]["problem_id"])
            assert key in units


def test_dummy_choose_loglik_is_log_half() -> None:
    from teh import evaluate_choice13k_program

    train, val, test, _ = load_guan_2020_stopping_trials(
        3, data_dir=DATA_DIR, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )
    trials = train + val + test

    def choose(_problem, _history):
        return 0.5

    result = evaluate_choice13k_program(choose, trials, n_seeds=1)
    assert abs(result["avg_loglik"] - math.log(0.5)) < 1e-9


def test_mat_matches_michael_choices_and_first_values() -> None:
    """Author MAT aligns with Michael CSV on choices + value1..4 (CSV incomplete for L=8)."""
    import pandas as pd

    michael = Path(
        "/data/zichang/other_repos/behavioralDataRepository/datasets/"
        "sequential-choice/guan-et-al-2020-optimal-stopping/trials.csv"
    )
    if not michael.is_file():
        pytest.skip("Michael CSV not available for cross-check")
    g = pd.read_csv(michael)
    raw = load_guan_raw(DATA_DIR)
    s = int(raw["participants"][0]["mat_subject_index"])  # participant 1
    dec = raw["decisions"]
    stim = raw["stimuli"]
    for c, cond in enumerate(CONDITION_NAMES):
        sub = g[(g.participant == 1) & (g.condition == cond)].sort_values("problem")
        assert len(sub) == 40
        L = int(raw["nstim"][c])
        for p in range(40):
            assert int(dec[s, p, c]) == int(sub.iloc[p].choicePosition)
            for i in range(min(4, L)):
                assert abs(float(stim[s, p, c, i]) - float(sub.iloc[p][f"value{i+1}"])) < 1e-6
            if L == 8:
                # Author MAT has remaining values; Michael CSV does not.
                assert np.isfinite(stim[s, p, c, 4:8]).all()


def test_emnlp_aliases_still_registered() -> None:
    for alias in ("1peterson2021using", "mixed_gambles", "bergert_nosofsky_2007"):
        assert alias in PARTICIPANT_DATASETS
        assert is_binary_loglik_dataset(alias)
