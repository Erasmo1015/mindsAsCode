"""Unit tests for optional dataset-prompt evolution."""
from __future__ import annotations

from pathlib import Path

import pytest

from utils.teh.dataset_prompt_evolution import (
    FROZEN_BEGIN,
    FROZEN_END,
    EVOLVABLE_BEGIN,
    EVOLVABLE_END,
    TASK_KNOWLEDGE_HEADER,
    PromptCandidate,
    aggregate_pics_style_prompt_scores,
    ensure_task_knowledge_in_prompt,
    frozen_hash,
    maybe_run_dataset_prompt_evolution,
    render_prompt_for_generation,
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


def test_domain_guidance_is_evolvable_not_frozen() -> None:
    wrapped = wrap_prompt_with_contract(BASELINE)
    frozen, evolvable = split_frozen_evolvable(wrapped)
    assert "def choose" in frozen
    assert "Pure Python" in frozen or "deterministic" in frozen
    assert "Behavioral requirements" in evolvable
    assert "Prefer cue-based models" in evolvable
    # Intro / task narrative should not be locked in frozen.
    assert "You are given observations of human choices." in evolvable
    assert "You are given observations of human choices." not in frozen


def test_task_knowledge_retained_in_evolvable() -> None:
    thin = (
        "def choose(problem, history):\n"
        "    return: float, P(action=1)\n\n"
        "Requirements:\n"
        "- Pure Python, no imports, deterministic.\n\n"
        "Behavioral requirements:\n"
        "- Prefer simple models.\n"
    )
    task = (
        "Participants rate cue validity as 90/80/70/60 and often use take-the-best."
    )
    instr = "In this experiment you will see four cues with expert validity weights."
    out = ensure_task_knowledge_in_prompt(
        thin, task_description=task, instruction_excerpt=instr
    )
    # Unmarked input stays marker-free for one-shot PICS path.
    assert FROZEN_BEGIN not in out
    assert "90/80/70/60" in out
    assert TASK_KNOWLEDGE_HEADER in out
    wrapped = wrap_prompt_with_contract(out)
    frozen, evolvable = split_frozen_evolvable(wrapped)
    assert "90/80/70/60" in evolvable
    assert "take-the-best" in evolvable
    assert "90/80/70/60" not in frozen


def test_task_knowledge_keeps_markers_when_already_present() -> None:
    marked = wrap_prompt_with_contract(
        "def choose(problem, history):\n"
        "    return: float, P(action=1)\n\n"
        "Requirements:\n"
        "- Pure Python, no imports, deterministic.\n\n"
        "Behavioral requirements:\n"
        "- Prefer simple models.\n"
    )
    out = ensure_task_knowledge_in_prompt(
        marked,
        task_description="Domain fact ALPHA_VALIDITY_0.9 must remain evolvable.",
        instruction_excerpt="",
    )
    assert FROZEN_BEGIN in out and EVOLVABLE_BEGIN in out
    _, evolvable = split_frozen_evolvable(out)
    assert "ALPHA_VALIDITY_0.9" in evolvable


def test_pics_style_scoring_uses_best_per_participant_not_mean_all() -> None:
    # Participant A: seed -0.7, gens -1.2 and -0.5 -> best -0.5
    # Participant B: seed -0.6, gens -0.9 -> best -0.6 (seed)
    # Mean of bests = (-0.5 + -0.6) / 2 = -0.55
    # Mean over all gens would be different and worse.
    per = [
        [(-0.69, -0.70), (-1.1, -1.2), (-0.4, -0.5)],
        [(-0.65, -0.60), (-0.8, -0.90)],
    ]
    agg = aggregate_pics_style_prompt_scores(per)
    assert agg["mean_val_fitness"] == pytest.approx(-0.55)
    assert agg["mean_train_fitness"] == pytest.approx((-0.4 + -0.65) / 2)


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
