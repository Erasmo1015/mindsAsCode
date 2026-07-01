"""Dry-run checks for explain-mode response parsing."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import teh
from utils.teh_transfer.explain_parse import (
    extract_program_from_block,
    parse_explain_response,
)


def _sample_program() -> str:
    return (
        "def choose(problem, history):\n"
        "    return 0.5\n"
    )


def test_code_only_output_still_sanitizes_when_explain_false() -> None:
    """Legacy path: unstructured code-only LLM output."""
    raw = _sample_program()
    parsed = parse_explain_response(raw)
    assert not parsed.ok
    assert parsed.error == "missing_source_rationale_tags"

    cleaned = teh._sanitize_llm_python_candidate(raw, required_markers=("def choose(",))
    assert cleaned
    assert "def choose" in cleaned


def test_structured_fenced_program_parses() -> None:
    raw = (
        "<source_rationale>\n"
        "Target dataset:\n"
        "- 1peterson2021using_test\n"
        "\n"
        "Selected source datasets:\n"
        "- 2plonsky2018when: shared gamble EV logic\n"
        "</source_rationale>\n"
        "\n"
        "<program>\n"
        "```python\n"
        + _sample_program()
        + "```\n"
        "</program>\n"
    )
    parsed = parse_explain_response(raw)
    assert parsed.ok
    assert "2plonsky2018when" in parsed.rationale
    assert "def choose" in parsed.program_text
    assert "```" not in parsed.program_text

    cleaned = teh._sanitize_llm_python_candidate(
        parsed.program_text, required_markers=("def choose(",)
    )
    assert cleaned


def test_structured_raw_program_parses() -> None:
    raw = (
        "<source_rationale>Used peterson history blend.</source_rationale>\n"
        "<program>\n"
        + _sample_program()
        + "</program>"
    )
    parsed = parse_explain_response(raw)
    assert parsed.ok
    assert parsed.rationale == "Used peterson history blend."
    assert extract_program_from_block(parsed.program_text) == parsed.program_text


def test_malformed_output_fails_clearly() -> None:
    raw = "<source_rationale>no program here</source_rationale>"
    parsed = parse_explain_response(raw)
    assert not parsed.ok
    assert parsed.error == "missing_program_tags"
    assert parsed.rationale == "no program here"

    cleaned = teh._sanitize_llm_python_candidate(raw, required_markers=("def choose(",))
    assert not cleaned


if __name__ == "__main__":
    test_code_only_output_still_sanitizes_when_explain_false()
    test_structured_fenced_program_parses()
    test_structured_raw_program_parses()
    test_malformed_output_fails_clearly()
    print("explain_parse checks passed")
