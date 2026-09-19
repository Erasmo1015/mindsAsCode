"""ICLR Centaur/OpenEvolve guards: no empty generic prefix, no target leak, SA40 isolation."""
from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baseline_methods" / "Psych101"))

_oe = types.ModuleType("openevolve")
_oe.OpenEvolve = object
sys.modules.setdefault("openevolve", _oe)
_oe_cfg = types.ModuleType("openevolve.config")
_oe_cfg.Config = object
sys.modules.setdefault("openevolve.config", _oe_cfg)
_oe_pp = types.ModuleType("openevolve.process_parallel")
_oe_pp.ProcessParallelController = object
sys.modules.setdefault("openevolve.process_parallel", _oe_pp)

from Centaur import (  # noqa: E402
    _build_generic_prefix,
    _prepare_mixed_gambles_centaur_trials,
    build_centaur_prompt_prefix_indexed,
    centaur_display_keys,
)
from run_openevolve import (  # noqa: E402
    _render_evaluator_py,
    _write_evolution_split_json,
    _write_posthoc_test_json,
    cap_and_subsample_prompt_trials,
    format_trial_compact,
    run_participant,
    sanitize_trial_for_program,
    trials_for_participant,
)
from utils.teh.teh_datasets import PARTICIPANT_DATASETS  # noqa: E402

ICLR_15 = (
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "5speekenbrink2008learning",
    "7hilbig2014generalized",
    "10frey2017risk",
    "11enkavi2019recentprobes",
    "12badham2017deficits",
    "mixed_gambles",
    "bergert_nosofsky_2007",
    "guan_2020_stopping",
    "steyvers_2009_bandit",
    "13schulz2020finding",
    "14kool2016when",
)


def test_iclr15_are_openevolve_participant_datasets():
    missing = [a for a in ICLR_15 if a not in PARTICIPANT_DATASETS]
    assert missing == []


def test_supported_dataset_refuses_empty_generic_prefix():
    trial = {
        "problem": {
            "dataset_alias": "mixed_gambles",
            "option_keys": [0, 1],
        },
        "history": [],
        "action": 0,
    }
    with pytest.raises(ValueError, match="refused empty generic prefix"):
        build_centaur_prompt_prefix_indexed([trial], 0, instruction="")


def test_unknown_dataset_may_use_generic_prefix():
    trial = {
        "problem": {"dataset_alias": "not_a_teh_dataset", "option_keys": [0, 1]},
        "history": [],
        "action": 0,
    }
    prefix = _build_generic_prefix([trial], 0, instruction="")
    assert prefix.strip() == "You press"
    assert build_centaur_prompt_prefix_indexed([trial], 0, instruction="").strip() == "You press"


def test_mixed_gambles_numeric_keys_fail_fast_even_with_gamble_fields():
    trial = {
        "problem": {
            "dataset_alias": "mixed_gambles",
            "option_keys": [0, 1],
            "gamble_A": {"rewards": [4.0, -1.0], "probs": [0.5, 0.5]},
            "gamble_B": {"rewards": [0.0], "probs": [1.0]},
        },
        "history": [],
        "action": 0,
    }
    with pytest.raises(ValueError, match="display keys"):
        build_centaur_prompt_prefix_indexed([trial], 0, instruction="")


def test_mixed_gambles_prepared_prefix_is_ab_gamble_text():
    raw = {
        "problem": {
            "dataset_alias": "mixed_gambles",
            "option_keys": [0, 1],
            "gamble_A": {"rewards": [4.0, -1.0], "probs": [0.5, 0.5]},
            "gamble_B": {"rewards": [0.0], "probs": [1.0]},
        },
        "history": [],
        "action": 0,
    }
    prepared = _prepare_mixed_gambles_centaur_trials([raw])[0]
    assert centaur_display_keys(prepared["problem"]) == ["A", "B"]
    prefix = build_centaur_prompt_prefix_indexed([prepared], 0)
    assert "Option A delivers" in prefix
    assert "Option B delivers" in prefix


def test_enkavi_probe_in_set_is_oracle_and_stripped_from_program_input():
    trial = {
        "problem": {
            "dataset_alias": "11enkavi2019recentprobes",
            "schema_type": "B",
            "memory_set_letters": ["A", "C", "F"],
            "probe_letter": "C",
            "probe_in_set": True,
            "option_keys": ["L", "N"],
        },
        "history": [],
        "action": 1,
    }
    compact = format_trial_compact(trial, "train")
    assert "memory_set=" in compact
    assert "probe=C" in compact
    assert "in_set" not in compact
    assert "probe_in_set" not in compact
    assert "y=1" in compact
    cleaned = sanitize_trial_for_program(trial)
    assert "probe_in_set" not in cleaned["problem"]
    assert cleaned["problem"]["memory_set_letters"] == ["A", "C", "F"]
    assert cleaned["problem"]["probe_letter"] == "C"
    assert trial["problem"]["probe_in_set"] is True


def test_kool_stage1_drops_unobserved_planet_s1_reward():
    trial = {
        "problem": {
            "dataset_alias": "14kool2016when",
            "schema_type": "kool_twostep",
            "stage": 1,
            "presented_day": 3,
            "option_keys": ["C", "D"],
            "spaceship_options": ["C", "D"],
            "planet": "J",
            "alien_options": ["P", "Q"],
            "stage1_action": 0,
            "spaceship": "C",
            "reward": 1,
            "treasure": 1,
        },
        "history": [],
        "action": 0,
    }
    cleaned = sanitize_trial_for_program(trial)
    p = cleaned["problem"]
    assert p["stage"] == 1
    assert p["spaceship_options"] == ["C", "D"]
    for key in ("planet", "alien_options", "stage1_action", "spaceship", "reward", "treasure"):
        assert key not in p
    compact = format_trial_compact(trial, "train")
    assert "planet=J" not in compact
    assert "s1=" not in compact
    assert "aliens=" not in compact


def test_kool_stage2_keeps_observed_planet_s1_not_reward():
    trial = {
        "problem": {
            "dataset_alias": "14kool2016when",
            "schema_type": "kool_twostep",
            "stage": 2,
            "presented_day": 3,
            "option_keys": ["P", "Q"],
            "planet": "J",
            "alien_options": ["P", "Q"],
            "stage1_action": 0,
            "spaceship": "C",
            "reward": 1,
            "treasure": 1,
        },
        "history": [
            {
                "stage": 1,
                "action": 0,
                "option_keys": ["C", "D"],
                "spaceship": "C",
                "planet": "J",
            }
        ],
        "action": 1,
    }
    cleaned = sanitize_trial_for_program(trial)
    p = cleaned["problem"]
    assert p["planet"] == "J"
    assert p["stage1_action"] == 0
    assert p["spaceship"] == "C"
    assert "reward" not in p
    assert "treasure" not in p
    assert cleaned["history"][0]["planet"] == "J"
    compact = format_trial_compact(trial, "val")
    assert "planet=J" in compact
    assert "s1=0" in compact
    assert "reward=1" not in compact


def test_evolution_json_and_evaluator_never_load_test(tmp_path: Path):
    train = [{"problem": {"x": 1}, "history": [], "action": 0}]
    val = [{"problem": {"x": 2}, "history": [], "action": 1}]
    test = [{"problem": {"LEAK": True}, "history": [], "action": 0}]
    evo = tmp_path / "trials_evolution_split.json"
    post = tmp_path / "trials_posthoc_test.json"
    _write_evolution_split_json(evo, train, val)
    _write_posthoc_test_json(post, test)
    evo_blob = json.loads(evo.read_text(encoding="utf-8"))
    assert "test" not in evo_blob
    assert "LEAK" not in json.dumps(evo_blob)
    src = _render_evaluator_py(evo, split_ratio=0.6, categorical=False)
    assert 'data["test"]' not in src
    assert "return data[\"train\"], data[\"val\"]" in src
    assert "_pooled_observed_loglik" in src


def test_prompt_subsample_and_run_participant_use_train_val_only():
    src_cap = inspect.getsource(cap_and_subsample_prompt_trials)
    assert "list(train_trials) + list(val_trials)" in src_cap
    assert "test_trials" not in src_cap
    src_run = inspect.getsource(run_participant)
    assert 'participant_ctx = {' in src_run
    assert '"train_trials": train_trials' in src_run
    assert '"val_trials": val_trials' in src_run
    assert '"test_trials"' not in src_run.split("participant_ctx = {", 1)[1].split("}", 1)[0]
    assert "Held-out test is scored once, after program selection" in src_run
    src_load = inspect.getsource(trials_for_participant)
    assert "sanitize_trial_for_program" in src_load
    assert "load_participant_limited_splits" in src_load


def test_all_15_share_the_same_observed_only_evaluator_contract(tmp_path: Path):
    """Isolation is per-run, not per-dataset: every alias gets the same evaluator template."""
    evo = tmp_path / "trials_evolution_split.json"
    _write_evolution_split_json(
        evo,
        [{"problem": {}, "history": [], "action": 0}],
        [{"problem": {}, "history": [], "action": 1}],
    )
    bernoulli = _render_evaluator_py(evo, split_ratio=0.6, categorical=False)
    categorical = _render_evaluator_py(evo, split_ratio=0.6, categorical=True)
    for src in (bernoulli, categorical):
        assert 'return data["train"], data["val"]' in src
        assert 'data["test"]' not in src
    assert len(ICLR_15) == 15
