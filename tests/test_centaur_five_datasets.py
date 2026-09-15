"""Smoke tests for Centaur five-dataset prompts, keys, leakage, and sparse wiring."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baseline_methods" / "Psych101"))

from centaur_prompts import (  # noqa: E402
    build_bergert_prefix,
    build_guan_prefix,
    build_kool_prefix,
    build_schulz_prefix,
    build_steyvers_prefix,
    centaur_display_keys,
    try_build_extended_centaur_prefix,
)

def test_display_keys_schulz_and_steyvers_are_1_indexed():
    schulz = {"dataset_alias": "13schulz2020finding", "schema_type": "categorical_bandit", "n_arms": 8}
    stey = {
        "dataset_alias": "steyvers_2009_bandit",
        "schema_type": "categorical_bandit",
        "n_arms": 4,
    }
    assert centaur_display_keys(schulz) == [str(i) for i in range(1, 9)]
    assert centaur_display_keys(stey) == ["1", "2", "3", "4"]


def test_bergert_prefix_empty_history_and_no_validity_leak():
    trial = {
        "problem": {
            "dataset_alias": "bergert_nosofsky_2007",
            "schema_type": "bergert_pairwise",
            "option_A": {"alternative_id": 1, "cues": {"cue1": 1, "cue2": 0}},
            "option_B": {"alternative_id": 2, "cues": {"cue1": 0, "cue2": 1}},
            "option_keys": [0, 1],
        },
        "history": [],
        "action": 1,
    }
    prefix = build_bergert_prefix([trial], 0, instruction="Bergert task.")
    assert "validity" not in prefix.lower()
    assert "log_odds" not in prefix.lower()
    assert prefix.rstrip().endswith("You press")
    assert "<<0>>" in prefix and "<<1>>" in prefix


def test_guan_rejects_mismatched_values_observed():
    trial = {
        "problem": {
            "dataset_alias": "guan_2020_stopping",
            "schema_type": "guan_stopping",
            "environment": "neutral",
            "sequence_length": 4,
            "position": 2,
            "values_observed": [0.1],  # leak / mismatch
            "option_keys": [0, 1],
        },
        "history": [],
        "action": 0,
    }
    with pytest.raises(ValueError, match="leakage"):
        build_guan_prefix([trial], 0)


def test_schulz_rejects_cond_field():
    trial = {
        "problem": {
            "dataset_alias": "13schulz2020finding",
            "schema_type": "categorical_bandit",
            "round": 1,
            "trial": 1,
            "n_arms": 8,
            "cond": "SRS",
            "option_keys": list(range(8)),
        },
        "history": [],
        "action": 0,
    }
    with pytest.raises(ValueError, match="condition"):
        build_schulz_prefix([trial], 0)


def test_kool_stage_prefix_uses_letter_keys():
    t0 = {
        "problem": {
            "dataset_alias": "14kool2016when",
            "schema_type": "kool_twostep",
            "stage": 1,
            "option_keys": ["R", "U"],
        },
        "history": [],
        "action": 0,
    }
    t1 = {
        "problem": {
            "dataset_alias": "14kool2016when",
            "schema_type": "kool_twostep",
            "stage": 2,
            "planet": "J",
            "option_keys": ["W", "K"],
        },
        "history": [
            {
                "stage": 1,
                "action": 0,
                "option_keys": ["R", "U"],
                "spaceship": "R",
                "planet": "J",
            }
        ],
        "action": 1,
    }
    p0 = build_kool_prefix([t0, t1], 0)
    p1 = build_kool_prefix([t0, t1], 1)
    assert "spaceships R and U" in p0
    assert p0.rstrip().endswith("You press")
    assert "aliens W and K" in p1
    assert "<<R>>" in p1


def test_steyvers_history_uses_1_indexed_keys():
    trial = {
        "problem": {
            "dataset_alias": "steyvers_2009_bandit",
            "schema_type": "categorical_bandit",
            "game": 1,
            "trial": 2,
            "n_arms": 4,
            "option_keys": [0, 1, 2, 3],
        },
        "history": [{"action": 0, "reward": 1}],
        "action": 2,
    }
    prefix = build_steyvers_prefix([trial], 0)
    assert "<<1>>" in prefix  # action 0 -> key 1
    assert try_build_extended_centaur_prefix([trial], 0) is not None


def test_bernoulli_ll_equiv_to_categorical_for_k2():
    """log p[y] matches Bernoulli LL when K=2."""
    p1 = 0.7
    probs = [1.0 - p1, p1]
    for y in (0, 1):
        bern = y * math.log(p1) + (1 - y) * math.log(1.0 - p1)
        cat = math.log(probs[y])
        assert abs(bern - cat) < 1e-12


def test_centaur_cli_lists_focus_datasets():
    import importlib.util

    path = REPO_ROOT / "baseline_methods" / "Psych101" / "Centaur.py"
    spec = importlib.util.spec_from_file_location("centaur_cli_mod", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for alias in mod.CENTAUR_FOCUS_DATASETS:
        assert alias in mod.PSYCH101_CENTAUR_DATASETS
