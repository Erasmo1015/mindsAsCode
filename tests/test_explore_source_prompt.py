"""Stage E explore suffix from a source rank-1 program file."""
from __future__ import annotations

from pathlib import Path

from utils.teh.explore_source_prompt import build_rank1_explore_prompt_suffix


def test_rank1_suffix_includes_source_program(tmp_path: Path):
    prog = tmp_path / "best_program.py"
    prog.write_text("def choose(problem, history):\n    return 0.5\n", encoding="utf-8")
    suffix = build_rank1_explore_prompt_suffix(
        source_dataset="11enkavi2019recentprobes",
        program_path=str(prog),
        split_seed=0,
    )
    assert "Cross-task transfer context" in suffix
    assert "11enkavi2019recentprobes" in suffix
    assert "def choose(" in suffix


def test_rank1_suffix_accepts_spaced_choose_signature(tmp_path: Path):
    prog = tmp_path / "best_program.py"
    prog.write_text(
        "def choose (problem ,history ):\n    return 0.5\n",
        encoding="utf-8",
    )
    suffix = build_rank1_explore_prompt_suffix(
        source_dataset="11enkavi2019recentprobes",
        program_path=str(prog),
        split_seed=0,
    )
    assert "def choose (problem ,history ):" in suffix


def test_rank1_suffix_rejects_code_without_choose(tmp_path: Path):
    prog = tmp_path / "best_program.py"
    prog.write_text("def other(x):\n    return 0.5\n", encoding="utf-8")
    try:
        build_rank1_explore_prompt_suffix(
            source_dataset="11enkavi2019recentprobes",
            program_path=str(prog),
            split_seed=0,
        )
    except ValueError as exc:
        assert "callable choose()" in str(exc)
    else:
        raise AssertionError("expected ValueError")
