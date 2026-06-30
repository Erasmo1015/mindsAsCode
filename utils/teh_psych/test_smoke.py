"""Lightweight smoke checks for teh_psych parser-plan + categorical pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.teh_psych.action_id_normalization import (
    ActionIdNormalizationError,
    normalize_categorical_trials_action_ids,
    normalize_trial_action_ids,
)
from utils.teh_psych.adapters import binary_trial_to_categorical
from utils.teh_psych.categorical_eval import coerce_choose_output, evaluate_categorical_program
from utils.teh_psych.parser_plan import (
    CACHE_MISS_CLIENT_MSG,
    StateMachineNotImplementedError,
    execute_parser_plan_on_row,
    run_parse_plan_pipeline,
    save_parse_plan_cache,
    validate_parser_plan,
)
from utils.teh_psych.trial_split import split_pooled_categorical_trials


def _trial_k2(target: int = 1):
    return {
        "problem": {"options": [{"action": 0}, {"action": 1}]},
        "history": [],
        "action": target,
        "target_action": target,
        "is_prediction_target": True,
        "_meta": {"row_index": 0, "block_id": 0},
    }


def _trial_k3(target: int = 2):
    return {
        "problem": {"options": [{"action": 0}, {"action": 1}, {"action": 2}]},
        "history": [],
        "action": target,
        "target_action": target,
        "is_prediction_target": True,
        "_meta": {"row_index": 0, "block_id": 0},
    }


def _minimal_plan() -> dict:
    return {
        "schema_version": "psych101_parser_plan_v1",
        "experiment_id": "fake/exp.csv",
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
        },
        "option_normalization": {
            "strategy": "fixed_keys_from_instruction",
            "instruction_regex": r"press\s+([A-Z])\s+and\s+([A-Z])",
            "action_id_order": "instruction_order",
            "raw_response_to_action_id_mapping": {"F": 0, "J": 1},
        },
        "history_rule": {"scope": "block", "fields": ["action", "feedback"]},
        "state_machine": {"enabled": False},
        "validation_expectations": {"min_options": 2},
    }


def test_validate_parser_plan_minimal():
    errors = validate_parser_plan(_minimal_plan())
    assert errors == []


def test_validate_parser_plan_rejects_bad_schema():
    plan = _minimal_plan()
    plan["schema_version"] = "bad"
    assert validate_parser_plan(plan)


def test_parser_engine_two_action_transcript():
    plan = _minimal_plan()
    row = {
        "text": "You press <<F>>.\nYou press <<J>>.\n",
        "instruction": "Press F and J.",
    }
    trials = execute_parser_plan_on_row(plan, row, row_index=0)
    assert len(trials) == 2
    assert trials[0]["target_action"] in (0, 1)
    assert len(trials[0]["problem"]["options"]) == 2


def test_state_machine_not_implemented():
    plan = _minimal_plan()
    plan["state_machine"] = {"enabled": True}
    with pytest.raises(StateMachineNotImplementedError):
        execute_parser_plan_on_row(plan, {"text": "You press <<F>>."}, row_index=0)


def test_coerce_k2_and_k3():
    p2, _ = coerce_choose_output({0: 0.4, 1: 0.6}, [0, 1])
    assert abs(sum(p2.values()) - 1.0) < 1e-9
    p3, _ = coerce_choose_output({0: 0.2, 1: 0.5, 2: 0.3}, [0, 1, 2])
    assert len(p3) == 3


def test_legacy_float_k2():
    probs, _ = coerce_choose_output(0.7, [0, 1])
    assert probs[1] == pytest.approx(0.7)


def test_evaluate_k2_and_k3():
    def choose2(problem, history):
        return {0: 0.4, 1: 0.6}

    def choose3(problem, history):
        return {0: 0.1, 1: 0.2, 2: 0.7}

    assert evaluate_categorical_program(choose2, [_trial_k2(1)])["avg_accuracy"] == 1.0
    assert evaluate_categorical_program(choose3, [_trial_k3(2)])["avg_accuracy"] == 1.0


def test_invalid_return_fallback():
    probs, warnings = coerce_choose_output("bad", [0, 1, 2])
    assert warnings
    assert len(probs) == 3


def test_internal_split_requires_groups():
    trials = [_trial_k2(0), _trial_k2(1)]
    with pytest.raises(ValueError):
        split_pooled_categorical_trials(trials, split_ratio=0.8, split_seed=1)


def test_main_does_not_load_test_split():
    import prototype.teh_psych as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert 'load_psych101_split("test"' not in source
    assert "hf_test_loglik" not in source


def test_internal_split_excludes_context_only_trials():
    from utils.teh_psych.trial_validation import partition_pooled_trials

    def _pred(block_id: int, *, context_only: bool = False):
        return {
            "is_prediction_target": not context_only,
            "target_action": 0,
            "problem": {"options": [{"action": 0}, {"action": 1}]},
            "_meta": {"row_index": 0, "block_id": block_id},
        }

    all_trials = [
        _pred(0, context_only=True),
        _pred(0),
        _pred(1),
        _pred(2),
    ]
    prediction_trials, context_only = partition_pooled_trials(all_trials)
    assert len(context_only) == 1
    assert len(prediction_trials) == 3

    train, val, test = split_pooled_categorical_trials(
        prediction_trials, split_ratio=0.8, split_seed=42
    )
    split_all = train + val + test
    assert len(split_all) == 3
    assert all(t.get("is_prediction_target", True) for t in split_all)
    assert not any(t.get("is_prediction_target") is False for t in split_all)


def test_parse_plan_cache_miss_requires_client(tmp_path):
    result = run_parse_plan_pipeline(
        client=None,
        experiment_id="fake/exp.csv",
        rows=[{"text": "You press <<F>>."}],
        row_indices=[0],
        debug_dir=tmp_path / "debug",
        template_path=_REPO_ROOT / "prompts" / "teh_psych" / "utils" / "parse_plan.txt",
        model_name="test-model",
        reuse_cache=True,
        cache_dir=tmp_path / "cache",
    )
    assert result.status == "generate_failed"
    assert result.failure_stage == "generate_parse_plan"
    assert result.failure_message == CACHE_MISS_CLIENT_MSG


def test_parse_plan_cache_hit_without_client(tmp_path):
    plan = _minimal_plan()
    cache_dir = tmp_path / "cache"
    save_parse_plan_cache(cache_dir, "fake/exp.csv", plan)
    result = run_parse_plan_pipeline(
        client=None,
        experiment_id="fake/exp.csv",
        rows=[{"text": "You press <<F>>.\nYou press <<J>>.", "instruction": "Press F and J."}],
        row_indices=[0],
        debug_dir=tmp_path / "debug2",
        template_path=_REPO_ROOT / "prompts" / "teh_psych" / "utils" / "parse_plan.txt",
        model_name="test-model",
        reuse_cache=True,
        cache_dir=cache_dir,
    )
    assert result.cached is True
    assert result.status == "executed"
    assert len(result.trials) >= 1


def test_eval_only_still_calls_parse_plan_pipeline():
    import prototype.teh_psych as mod

    captured = {}

    def fake_run(*, client, experiment_id, rows, row_indices, debug_dir, **kwargs):
        captured["called"] = True
        from utils.teh_psych.parser_plan import ParsePlanRunResult

        return ParsePlanRunResult(
            experiment_id=experiment_id,
            status="executed",
            trials=[
                {
                    "problem": {"options": [{"action": 0}, {"action": 1}]},
                    "history": [],
                    "target_action": 0,
                    "is_prediction_target": True,
                    "_meta": {"row_index": 0, "block_id": i},
                }
                for i in range(6)
            ],
        )

    with mock.patch.object(mod, "run_parse_plan_pipeline", side_effect=fake_run):
        with mock.patch.object(mod, "split_pooled_categorical_trials") as split_mock:
            split_mock.return_value = ([], [], [])
            with mock.patch.object(mod, "validate_categorical_trials", return_value=([], 6)):
                with mock.patch.object(mod, "summarize_trial_action_space", return_value={
                    "num_actions_min": 2,
                    "num_actions_max": 2,
                    "is_variable_k": False,
                }):
                    args = mod.build_argparser().parse_args([
                        "--eval_only",
                        "--min_pooled_prediction_trials",
                        "1",
                        "--reuse_parse_plan_cache",
                    ])
                    result = mod.DatasetResult(experiment_id="fake/exp.csv")
                    mod._trials_via_parse_plan(
                        client=None,
                        experiment_id="fake/exp.csv",
                        rows=[{"text": "x"}],
                        row_indices=[0],
                        debug_dir=Path("/tmp/teh_psych_test_debug"),
                        args=args,
                        cache_dir=Path("/tmp/teh_psych_test_cache"),
                        result=result,
                    )
    assert captured.get("called") is True


def test_binary_to_categorical_adapter():
    binary = {
        "problem": {"option_keys": ["P", "U"], "schema_type": "A"},
        "history": [],
        "options": ["P", "U"],
        "action": 1,
    }
    cat = binary_trial_to_categorical(binary)
    assert cat["problem"]["options"][1]["label"] == "U"


def test_normalize_action_ids_non_consecutive():
    trial = {
        "problem": {
            "options": [
                {"action": 18, "label": "D"},
                {"action": 19, "label": "N"},
            ]
        },
        "history": [],
        "action": 19,
        "target_action": 19,
        "is_prediction_target": True,
        "_meta": {"raw_key": "N"},
    }
    out = normalize_trial_action_ids(trial)
    assert [o["action"] for o in out["problem"]["options"]] == [0, 1]
    assert out["target_action"] == 1
    assert out["problem"]["options"][1]["raw_action"] == 19


def test_normalize_action_ids_offset_start():
    trial = {
        "problem": {
            "options": [
                {"action": 2, "label": "a"},
                {"action": 3, "label": "b"},
            ]
        },
        "target_action": 2,
        "history": [{"action": 2}],
    }
    out = normalize_trial_action_ids(trial)
    assert [o["action"] for o in out["problem"]["options"]] == [0, 1]
    assert out["target_action"] == 0
    assert out["history"][0]["action"] == 0


def test_normalize_action_ids_unmappable_target_raises():
    trial = {
        "problem": {"options": [{"action": 18, "label": "D"}, {"action": 19, "label": "N"}]},
        "target_action": 99,
    }
    with pytest.raises(ActionIdNormalizationError):
        normalize_trial_action_ids(trial)


def test_option_source_fallback_per_trial_keys():
    plan = {
        "schema_version": "psych101_parser_plan_v1",
        "experiment_id": "fake/exp.csv",
        "task_type": "categorical_choice",
        "line_classifier": {
            "line_types": [
                {
                    "type_id": "instruction",
                    "detection": {"regex": r"Press F for left and J for right", "flags": "i"},
                },
                {
                    "type_id": "action_press",
                    "detection": {"regex": r"You press <<([A-Z])>>", "flags": "i"},
                },
            ]
        },
        "block_boundary_rule": {"strategy": "single_block"},
        "trial_extraction": {
            "boundary_strategy": "one_press_per_trial",
            "action_rule": {
                "source_line_type": "action_press",
                "capture": {"regex": r"<<([A-Z])>>", "group": 1},
            },
            "context_only_trial_rule": {"line_type": "instruction"},
        },
        "option_normalization": {
            "strategy": "per_trial_available_keys",
            "instruction_regex": r"Press ([A-Z]) for left and ([A-Z]) for right",
            "action_id_order": "instruction_order",
        },
        "history_rule": {"scope": "block", "fields": ["action"]},
        "state_machine": {"enabled": False},
        "validation_expectations": {"min_options": 2},
    }
    row = {
        "text": (
            "Press F for left and J for right.\n"
            "Stimulus appears.\n"
            "You press <<F>>.\n"
            "Feedback shown.\n"
        )
    }
    trials = execute_parser_plan_on_row(plan, row, row_index=0)
    assert len(trials) == 1
    assert trials[0]["is_prediction_target"] is True
    labels = [o.get("raw_key", o.get("label")) for o in trials[0]["problem"]["options"]]
    assert "F" in labels and "J" in labels
    assert trials[0]["target_action"] == 0


def test_multiline_transcript_produces_prediction_trial():
    plan = _minimal_plan()
    plan["option_normalization"] = {
        "strategy": "fixed_keys_from_instruction",
        "instruction_regex": r"press\s+([A-Z])\s+and\s+([A-Z])",
        "action_id_order": "instruction_order",
    }
    row = {
        "text": (
            "Instruction line.\n"
            "Stimulus line.\n"
            "You press <<F>>.\n"
            "Feedback line.\n"
        ),
        "instruction": "Press F and J.",
    }
    trials = execute_parser_plan_on_row(plan, row, row_index=0)
    assert len(trials) >= 1
    assert trials[0]["is_prediction_target"] is True


def test_evolution_coerces_string_first_error():
    from utils.teh_psych import evolution as evo_mod

    entry = evo_mod._coerce_eval_error_entry("choose() returned invalid probs")
    assert isinstance(entry, dict)
    assert entry.get("error_message") == "choose() returned invalid probs"
    assert entry.get("normalized_key")
