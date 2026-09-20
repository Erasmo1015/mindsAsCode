"""Tests for frozen parse-plan loading shared by teh_psych and baselines."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from utils.teh_psych.load_trials import (
    CachedParsePlanError,
    fingerprint_trials,
    load_and_validate_cached_plan,
    load_experiment_trials_from_parse_plan,
    trial_fingerprint,
)
from utils.teh_psych.parser_plan import (
    execute_parser_plan_on_rows,
    save_parse_plan_cache,
)
from utils.teh_psych.action_id_normalization import normalize_categorical_trials_action_ids
from utils.teh_psych.trial_validation import partition_pooled_trials
from baseline_methods.teh_psych.features import (
    experiment_prospect_support,
    extract_explicit_binary_gambles,
    option_feature_vector,
    prospect_theory_support_reason,
)

_REPO = Path(__file__).resolve().parents[2]
_RUN = _REPO / "generated_outputs_teh_psych" / "run_260722_004944"
_CACHE = _RUN / "parse_plan_cache"
_BY = _RUN / "prototype_summary" / "by_dataset"
_HAS_CACHE = _CACHE.is_dir() and _BY.is_dir()


def _minimal_press_plan(experiment_id: str = "fake/exp.csv") -> dict:
    return {
        "schema_version": "psych101_parser_plan_v1",
        "experiment_id": experiment_id,
        "task_type": "categorical_choice",
        "line_classifier": {
            "line_types": [
                {
                    "type_id": "action_press",
                    "detection": {"regex": r"You press <<([A-Z])>>", "flags": "i"},
                }
            ]
        },
        "block_boundary_rule": {"strategy": "single_block"},
        "trial_extraction": {
            "boundary_strategy": "one_press_per_trial",
            "action_rule": {
                "source_line_type": "action_press",
                "capture": {"regex": r"<<([A-Z])>>", "group": 1},
            },
            "feedback_rule": {
                "regex": r"get ([-]?\d+) points",
                "value_type": "points",
            },
        },
        "option_normalization": {
            "strategy": "fixed_keys_from_instruction",
            "instruction_regex": r"press\s+([A-Z])\s+and\s+([A-Z])",
            "action_id_order": "instruction_order",
            "raw_response_to_action_id_mapping": {"A": 0, "B": 1},
        },
        "history_rule": {"scope": "block", "fields": ["action", "feedback"]},
        "state_machine": {"enabled": False},
        "validation_expectations": {"min_options": 2},
        "human_review_required": False,
    }


def _synthetic_row(participant: int = 1) -> dict:
    text = "\n".join(
        [
            "You press <<A>> and get 2 points.",
            "You press <<B>> and get -1 points.",
            "You press <<A>> and get 3 points.",
            "You press <<B>> and get 0 points.",
            "You press <<A>> and get 1 points.",
            "You press <<B>> and get 4 points.",
        ]
    )
    return {
        "text": text,
        "participant": participant,
        "experiment": "fake/exp.csv",
        "instruction": "Press A and B.",
    }


def test_synthetic_loader_matches_direct_execute(tmp_path: Path):
    """Binary synthetic task: shared loader ≡ execute→normalize→partition."""
    eid = "fake/exp.csv"
    plan = _minimal_press_plan(eid)
    save_parse_plan_cache(tmp_path, eid, plan)
    rows = [_synthetic_row(1), _synthetic_row(2), _synthetic_row(3)]
    indices = [0, 1, 2]
    bundle = load_experiment_trials_from_parse_plan(
        eid,
        tmp_path,
        require_cached=True,
        rows=rows,
        row_indices=indices,
        do_split=True,
        split_ratio=0.5,
        split_seed=0,
        min_pooled_prediction_trials=1,
    )
    assert bundle.plan_sha256
    plan2, _, _ = load_and_validate_cached_plan(tmp_path, eid)
    trials2, _ = execute_parser_plan_on_rows(plan2, rows, row_indices=indices)
    trials2 = normalize_categorical_trials_action_ids(trials2)
    pred2, _ = partition_pooled_trials(trials2)
    assert fingerprint_trials(bundle.prediction_trials) == fingerprint_trials(pred2)
    assert {len(t["problem"]["options"]) for t in bundle.prediction_trials} == {2}
    fp0 = trial_fingerprint(bundle.prediction_trials[0])
    for key in ("participant", "target_action", "options", "stimulus", "history", "feedback"):
        assert key in fp0
    # sequential history within block
    assert any(t.get("history") for t in bundle.prediction_trials)


def test_synthetic_variable_k_features(tmp_path: Path):
    """K>=2 feature vectors; use digit keys to mimic variable action alphabets."""
    eid = "fake/vark.csv"
    plan = _minimal_press_plan(eid)
    plan["trial_extraction"]["action_rule"]["capture"] = {
        "regex": r"<<(\d+)>>",
        "group": 1,
    }
    plan["option_normalization"] = {
        "strategy": "per_trial_available_keys",
        "instruction_regex": "",
        "action_id_order": "first_seen_in_transcript",
        "raw_response_to_action_id_mapping": {},
    }
    plan["validation_expectations"] = {"min_options": 2}
    save_parse_plan_cache(tmp_path, eid, plan)
    row = {
        "text": (
            "You press <<1>> and get 2 points.\n"
            "You press <<2>> and get 1 points.\n"
            "You press <<3>> and get 0 points.\n"
            "You press <<1>> and get 4 points.\n"
        ),
        "participant": 1,
        "experiment": eid,
        "instruction": "Choose 1, 2, or 3.",
    }
    bundle = load_experiment_trials_from_parse_plan(
        eid,
        tmp_path,
        require_cached=True,
        rows=[row],
        row_indices=[0],
        do_split=False,
        min_pooled_prediction_trials=1,
    )
    assert len(bundle.prediction_trials) >= 1
    # Across trials K can vary with per_trial_available_keys
    ks = {len(t["problem"]["options"]) for t in bundle.prediction_trials}
    assert max(ks) >= 2
    t_last = bundle.prediction_trials[-1]
    feats = option_feature_vector(t_last["problem"], t_last.get("history"))
    assert len(feats) == len(t_last["problem"]["options"])


def test_require_cached_plan_failure(tmp_path: Path):
    empty = tmp_path / "empty_cache"
    empty.mkdir()
    with pytest.raises(CachedParsePlanError, match="missing"):
        load_experiment_trials_from_parse_plan(
            "wilson2014humans/exp1.csv",
            empty,
            require_cached=True,
            rows=[{"text": "x", "participant": 1, "experiment": "wilson2014humans/exp1.csv"}],
            row_indices=[0],
            do_split=False,
            min_pooled_prediction_trials=0,
        )


def test_invalid_cached_plan_failure(tmp_path: Path):
    eid = "fake/exp.csv"
    bad = {"schema_version": "psych101_parser_plan_v1", "experiment_id": eid}
    save_parse_plan_cache(tmp_path, eid, bad)
    with pytest.raises(CachedParsePlanError):
        load_and_validate_cached_plan(tmp_path, eid, require_cached=True)


def test_split_deterministic_same_seed(tmp_path: Path):
    eid = "fake/exp.csv"
    save_parse_plan_cache(tmp_path, eid, _minimal_press_plan(eid))
    rows = [_synthetic_row(i) for i in range(4)]
    kwargs = dict(
        experiment_id=eid,
        cache_dir=tmp_path,
        require_cached=True,
        rows=rows,
        row_indices=list(range(4)),
        do_split=True,
        split_ratio=0.5,
        split_seed=7,
        min_pooled_prediction_trials=1,
    )
    b1 = load_experiment_trials_from_parse_plan(**kwargs)
    b2 = load_experiment_trials_from_parse_plan(**kwargs)
    assert fingerprint_trials(b1.train_trials) == fingerprint_trials(b2.train_trials)
    assert fingerprint_trials(b1.test_trials) == fingerprint_trials(b2.test_trials)


def test_explicit_gamble_extraction_for_pt():
    problem = {
        "options": [{"action": 0, "label": "A"}, {"action": 1, "label": "B"}],
        "stimulus": {
            "gamble_A": {"rewards": [10.0, -5.0], "probs": [0.5, 0.5]},
            "gamble_B": {"rewards": [1.0], "probs": [1.0]},
        },
    }
    assert extract_explicit_binary_gambles(problem) is not None
    ok, _ = prospect_theory_support_reason(problem)
    assert ok is True


@pytest.mark.skipif(not _HAS_CACHE, reason="frozen teh_psych parse_plan_cache not present")
def test_real_cache_frey_cct_one_row_and_pt_unsupported():
    """Frey CCT from frozen cache (1 sampled row) + PT must not invent gambles."""
    eid = "frey2017cct/exp1.csv"
    rows = json.loads((_BY / eid.replace("/", "_") / "sampled_rows.json").read_text())[:1]
    bundle = load_experiment_trials_from_parse_plan(
        eid,
        _CACHE,
        require_cached=True,
        rows=rows,
        row_indices=[0],
        do_split=False,
        min_pooled_prediction_trials=1,
    )
    assert bundle.action_summary.get("num_actions_min") == 2
    stim = bundle.prediction_trials[0]["problem"].get("stimulus") or {}
    assert any(k in stim for k in ("gain_amount", "loss_amount", "loss_cards_count"))
    ok, reason, _ = experiment_prospect_support(bundle.prediction_trials)
    assert ok is False
    assert "explicit" in reason.lower() or "invent" in reason.lower()


@pytest.mark.skipif(not _HAS_CACHE, reason="frozen teh_psych parse_plan_cache not present")
def test_real_cache_binary_wilson_one_row_fingerprint():
    eid = "wilson2014humans/exp1.csv"
    rows = json.loads((_BY / eid.replace("/", "_") / "sampled_rows.json").read_text())[:1]
    bundle = load_experiment_trials_from_parse_plan(
        eid,
        _CACHE,
        require_cached=True,
        rows=rows,
        row_indices=[0],
        do_split=False,
        min_pooled_prediction_trials=1,
    )
    plan2, _, _ = load_and_validate_cached_plan(_CACHE, eid)
    trials2, _ = execute_parser_plan_on_rows(copy.deepcopy(plan2), rows, row_indices=[0])
    trials2 = normalize_categorical_trials_action_ids(trials2)
    pred2, _ = partition_pooled_trials(trials2)
    assert fingerprint_trials(bundle.prediction_trials) == fingerprint_trials(pred2)
