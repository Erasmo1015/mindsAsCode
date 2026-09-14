"""Tests for budgeted full-JSON prompt-generation context."""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.teh.prompt_context import (
    history_keys_note_from_schema,
    infer_recursive_runtime_schema,
    serialize_train_trials_for_prompt_generation,
)
from utils.teh.teh_runtime import (
    _build_schema_neutral_base_prompt,
    build_prompt_generation_llm_user_content,
    setup_teh_run_prompts,
)


def _bergert_trial() -> dict:
    return {
        "problem": {
            "schema_type": "bergert",
            "problem_id": 1,
            "option_A": {
                "cues": {
                    "cue1": 1,
                    "cue2": 0,
                    "cue3": 1,
                    "cue4": 0,
                    "cue5": 1,
                    "cue6": 0,
                }
            },
            "option_B": {
                "cues": {
                    "cue1": 0,
                    "cue2": 1,
                    "cue3": 0,
                    "cue4": 1,
                    "cue5": 0,
                    "cue6": 1,
                }
            },
            "option_keys": ["A", "B"],
        },
        "action": 0,
        "history": [],
    }


def _kool_stage1() -> dict:
    return {
        "problem": {
            "schema_type": "kool",
            "stage": 1,
            "option_keys": ["B", "L"],
            "spaceship_options": ["B", "L"],
        },
        "action": 0,
        "history": [],
    }


def _kool_stage2() -> dict:
    return {
        "problem": {
            "schema_type": "kool",
            "stage": 2,
            "option_keys": ["G", "R"],
            "planet": "red",
            "stage1_action": 0,
            "stage1_option_keys": ["B", "L"],
        },
        "action": 1,
        "history": [
            {"stage": 1, "action": 0, "option_keys": ["B", "L"], "feedback": None},
        ],
    }


def _choice13k_trial() -> dict:
    return {
        "problem": {
            "schema_type": "choice13k",
            "gamble_A": {"probs": [0.5, 0.5], "rewards": [10.0, 0.0]},
            "gamble_B": {"probs": [1.0], "rewards": [5.0]},
            "option_keys": ["A", "B"],
            "has_feedback": False,
        },
        "action": 1,
        "history": [],
    }


def test_bergert_nested_cues_visible_in_serialized_json() -> None:
    text = serialize_train_trials_for_prompt_generation(
        [_bergert_trial()], char_budget=5000, history_max_entries=8, max_examples=2
    )
    assert "cues" in text
    assert "cue1" in text
    # Nested path must survive (not flattened to option_A['cue1']).
    assert '"option_A"' in text
    parsed_blocks = [
        json.loads(block)
        for block in text.split("\n\n")
        if block.strip().startswith("{")
    ]
    assert parsed_blocks
    cues = parsed_blocks[0]["problem"]["option_A"]["cues"]
    assert cues["cue1"] == 1


def test_kool_stage_conditional_schema() -> None:
    schema = infer_recursive_runtime_schema([_kool_stage1(), _kool_stage2()])
    assert "stage=1" in schema
    assert "stage=2" in schema
    assert "planet" in schema
    # planet only on stage 2
    assert "sometimes" in schema or "stage=2" in schema
    stage2_section = schema.split("stage=2")[1].split("stage=")[0]
    assert "planet" in stage2_section
    stage1_section = schema.split("stage=1")[1].split("stage=2")[0]
    assert "planet" not in stage1_section or "sometimes" in stage1_section


def test_budget_respects_limit_and_valid_json() -> None:
    trials = []
    for i in range(20):
        t = _bergert_trial()
        t = dict(t)
        t["problem"] = dict(t["problem"])
        t["problem"]["problem_id"] = i
        t["history"] = [{"action": j % 2, "feedback": float(j)} for j in range(30)]
        trials.append(t)
    budget = 2500
    text = serialize_train_trials_for_prompt_generation(
        trials, char_budget=budget, history_max_entries=4, max_examples=8
    )
    # Body after header must fit budget.
    parts = text.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else text
    assert len(body) <= budget
    blocks = [b for b in body.split("\n\n") if b.strip().startswith("{")]
    assert blocks
    for block in blocks:
        obj = json.loads(block)
        assert "problem" in obj and "action" in obj and "history" in obj


def test_history_presence_counted_per_trial_not_per_entry() -> None:
    """Presence denominators use n_trials; numerators must not exceed n_trials."""
    n_trials = 4
    entries_per_trial = 30
    trials = []
    for i in range(n_trials):
        trials.append(
            {
                "problem": {"schema_type": "toy", "x": i, "option_keys": ["A", "B"]},
                "action": i % 2,
                "history": [
                    {"action": j % 2, "feedback": float(j)}
                    for j in range(entries_per_trial)
                ],
            }
        )
    schema = infer_recursive_runtime_schema(trials)
    # Old bug: action present counted as 4*30=120; fixed: <= 4.
    matches = re.findall(
        r"- action: \w+ \((?:always|sometimes) present, (\d+)/(\d+)\)", schema
    )
    assert matches, schema
    for present_s, total_s in matches:
        present, total = int(present_s), int(total_s)
        assert present <= n_trials
        assert total == n_trials
        assert present <= total


def test_base_prompt_uses_concrete_history_keys_not_placeholder() -> None:
    trials = [
        {
            "problem": {"schema_type": "toy", "rating": 3, "option_keys": ["A", "B"]},
            "action": 1,
            "history": [{"action": 0, "feedback": 1.0, "rating": 2}],
        }
    ]
    schema = infer_recursive_runtime_schema(trials)
    prompt = _build_schema_neutral_base_prompt(schema, trials)
    assert "see Runtime schema summary" not in prompt
    assert "action" in history_keys_note_from_schema(schema)
    assert "feedback" in prompt or "rating" in prompt
    assert "history: list of dicts (keys observed: " in prompt
    assert "rating" in prompt  # problem key from schema


def test_choice13k_serializes_gambles() -> None:
    text = serialize_train_trials_for_prompt_generation(
        [_choice13k_trial()], char_budget=4000, max_examples=2
    )
    assert "gamble_A" in text
    assert "gamble_B" in text
    schema = infer_recursive_runtime_schema([_choice13k_trial()])
    assert "is_gamble_A/B_task: True" in schema


def test_prompt_generation_user_content_uses_json_not_one_liners() -> None:
    content = build_prompt_generation_llm_user_content(
        "bergert_nosofsky_2007",
        "classify by cues",
        [_bergert_trial()],
        example_char_budget=4000,
        history_max_entries=4,
        max_examples=2,
    )
    assert '"option_A"' in content or "option_A" in content
    assert "cues" in content
    assert "Runtime schema summary" in content
    # One-liner style "option_A cues={...}" should not be the primary format.
    assert "option_A cues={" not in content
    assert "Retain the **Task description**" in content


def test_default_oneshot_meta_no_evolution(tmp_path: Path) -> None:
    seed = tmp_path / "seed.py"
    seed.write_text(
        "def choose(problem, history):\n    return 0.5\n", encoding="utf-8"
    )
    base_prompt = tmp_path / "base.txt"
    base_prompt.write_text(
        "You must implement def choose(problem, history) returning float P(action=1).\n"
        "Requirements:\n- Pure Python, no imports, deterministic.\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    with patch(
        "utils.teh.teh_runtime.resolve_dataset_reference_prompt_path",
        return_value=None,
    ), patch(
        "utils.teh.teh_runtime.resolve_base_loglik_prompt_path",
        return_value=base_prompt,
    ), patch(
        "utils.teh.teh_runtime.is_bergert_nosofsky_2007_dataset",
        return_value=True,
    ), patch(
        "utils.teh.teh_runtime.is_mixed_gambles_dataset",
        return_value=False,
    ), patch(
        "utils.teh.teh_runtime.is_guan_2020_stopping_dataset",
        return_value=False,
    ), patch(
        "utils.teh.teh_runtime.is_steyvers_2009_bandit_dataset",
        return_value=False,
    ), patch(
        "utils.teh.participant_ids.load_valid_participant_ids",
        return_value=[1],
    ), patch(
        "utils.teh.teh_runtime.load_external_loglik_trials",
        return_value=([_bergert_trial()], [], [], None),
    ), patch(
        "utils.teh.teh_runtime.BASE_REFINE_PROMPT", base_prompt
    ), patch(
        "utils.teh.teh_runtime.shutil.copy2"
    ), patch(
        "utils.teh.teh_runtime.external_default_data_dir",
        return_value="datasets/external/bergert_nosofsky_2007",
    ), patch(
        "utils.teh.teh_runtime.is_external_dataset",
        return_value=True,
    ):
        prompts_dir = setup_teh_run_prompts(
            run_dir,
            "bergert_nosofsky_2007",
            seed,
            use_llm=False,
            client=None,
        )
    meta = json.loads((prompts_dir / "prompt_meta.json").read_text(encoding="utf-8"))
    assert meta.get("dataset_prompt_evolved") is False
    assert meta.get("evolution_iterations") == 0
    assert not (prompts_dir / "dataset_prompt_evolution").exists()
    # Default non-evolution path still writes an infer prompt.
    assert (prompts_dir / "infer_single_choice.txt").is_file()
