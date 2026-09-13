"""Unit tests for optional dataset-prompt evolution."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.teh.dataset_prompt_evolution import (
    FROZEN_BEGIN,
    FROZEN_END,
    EVOLVABLE_BEGIN,
    EVOLVABLE_END,
    PromptCandidate,
    frozen_hash,
    maybe_run_dataset_prompt_evolution,
    run_dataset_prompt_evolution,
    select_beam,
    split_frozen_evolvable,
    validate_child_prompt,
    wrap_prompt_with_contract,
)


BASELINE = """\
You are given observations of human choices.
def choose(problem, history):
    return: float, P(action=1)

Requirements:
- Pure Python, no imports, deterministic.
- Do not sample or use randomness.

Behavioral requirements:
- Prefer cue-based models.
- Encourage diversity across candidates.
"""


def test_frozen_violation_rejected() -> None:
    wrapped = wrap_prompt_with_contract(BASELINE)
    frozen, _ = split_frozen_evolvable(wrapped)
    parent_h = frozen_hash(frozen)
    # Mutate frozen content.
    bad = wrapped.replace("Pure Python, no imports, deterministic.", "You may import numpy.")
    ok, reason = validate_child_prompt(bad, parent_frozen_hash=parent_h)
    assert ok is False
    assert reason in {
        "frozen_hash_mismatch",
        "sandbox_weakened",
        "sandbox_requirements_missing",
    }


def test_leakage_language_rejected() -> None:
    wrapped = wrap_prompt_with_contract(BASELINE)
    frozen, evolvable = split_frozen_evolvable(wrapped)
    parent_h = frozen_hash(frozen)
    leaky = (
        f"{FROZEN_BEGIN}\n{frozen}\n{FROZEN_END}\n\n"
        f"{EVOLVABLE_BEGIN}\n{evolvable}\nUse the test set for tuning.\n{EVOLVABLE_END}\n"
    )
    ok, reason = validate_child_prompt(leaky, parent_frozen_hash=parent_h)
    assert ok is False
    assert reason == "test_leakage_language"


def test_beam_keeps_best() -> None:
    cands = [
        PromptCandidate("a", "x" * 100, mean_val_fitness=-1.0, n_chars=100),
        PromptCandidate("b", "y" * 50, mean_val_fitness=-0.5, n_chars=50),
        PromptCandidate("c", "z" * 80, mean_val_fitness=-2.0, n_chars=80),
        PromptCandidate("d", "w", mean_val_fitness=-0.4, n_chars=1, rejected=True),
    ]
    beam = select_beam(cands, beam_size=2)
    ids = [c.prompt_id for c in beam]
    assert ids[0] == "b"
    assert "d" not in ids
    assert len(beam) == 2


def test_iterations_zero_noop(tmp_path: Path) -> None:
    result = maybe_run_dataset_prompt_evolution(
        iterations=0,
        prompts_dir=tmp_path,
        baseline_prompt=BASELINE,
        population=3,
        children_per_round=3,
        dev_participant_ids=[0, 1],
        eval_candidates=5,
        seed_program="def choose(problem, history):\n    return 0.5\n",
        mutate_fn=lambda *a, **k: BASELINE,
        score_fn=lambda _t: {
            "mean_train_fitness": -1.0,
            "mean_val_fitness": -1.0,
            "executable_rate": 1.0,
        },
    )
    assert result is None
    assert not (tmp_path / "dataset_prompt_evolution").exists()


def test_evolution_round_persists_best(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "infer_single_choice.txt").write_text(BASELINE, encoding="utf-8")
    (prompts_dir / "prompt_meta.json").write_text(
        '{"dataset_prompt_evolved": false, "evolution_iterations": 0}\n',
        encoding="utf-8",
    )

    scores = {"n": 0}

    def score_fn(text: str):
        scores["n"] += 1
        # Later prompts get higher fitness.
        return {
            "mean_train_fitness": -1.0 + 0.01 * scores["n"],
            "mean_val_fitness": -1.0 + 0.02 * scores["n"],
            "executable_rate": 1.0,
            "error_signatures": [],
            "strong_snippets": ["def choose(problem, history):\n    return 0.6"],
        }

    def mutate_fn(parent: str, role: str, feedback: str) -> str:
        frozen, evolvable = split_frozen_evolvable(wrap_prompt_with_contract(parent))
        return (
            f"{FROZEN_BEGIN}\n{frozen}\n{FROZEN_END}\n\n"
            f"{EVOLVABLE_BEGIN}\n{evolvable}\n# mutated {role}\n{EVOLVABLE_END}\n"
        )

    result = run_dataset_prompt_evolution(
        prompts_dir=prompts_dir,
        baseline_prompt=BASELINE,
        iterations=1,
        population=3,
        children_per_round=3,
        dev_participant_ids=[1, 2],
        eval_candidates=5,
        seed_program="def choose(problem, history):\n    return 0.5\n",
        mutate_fn=mutate_fn,
        score_fn=score_fn,
    )
    assert result["dataset_prompt_evolved"] is True
    assert result["best_prompt_id"]
    assert (prompts_dir / "dataset_prompt_evolution" / "summary.json").is_file()
    meta = (prompts_dir / "prompt_meta.json").read_text(encoding="utf-8")
    assert "dataset_prompt_evolved" in meta
    assert '"evolution_iterations": 1' in meta
    assert (prompts_dir / "infer_single_choice.txt").read_text(encoding="utf-8")
