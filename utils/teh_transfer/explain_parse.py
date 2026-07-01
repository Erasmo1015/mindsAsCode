"""Parse structured explain-mode LLM responses for teh_transfer transfer phase."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPLAIN_SUFFIX_PATH = _REPO_ROOT / "prompts" / "teh_transfer" / "explain_suffix.txt"

_SOURCE_RATIONALE_RE = re.compile(
    r"<source_rationale>(.*?)</source_rationale>",
    re.DOTALL | re.IGNORECASE,
)
_PROGRAM_RE = re.compile(
    r"<program>(.*?)</program>",
    re.DOTALL | re.IGNORECASE,
)
_FENCED_PYTHON_RE = re.compile(
    r"```(?:python)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplainParseResult:
    ok: bool
    rationale: str
    program_text: str
    error: str


def load_explain_suffix(path: Optional[Path | str] = None) -> str:
    resolved = Path(path) if path is not None else DEFAULT_EXPLAIN_SUFFIX_PATH
    if not resolved.is_file():
        raise FileNotFoundError(f"Explain suffix prompt not found: {resolved}")
    return resolved.read_text(encoding="utf-8")


def extract_program_from_block(block: str) -> str:
    """Return raw Python from a <program> block (fenced or plain)."""
    text = (block or "").strip()
    if not text:
        return ""
    match = _FENCED_PYTHON_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def parse_explain_response(raw: str) -> ExplainParseResult:
    """Extract rationale and program from structured explain-mode LLM output."""
    if not (raw or "").strip():
        return ExplainParseResult(
            ok=False,
            rationale="",
            program_text="",
            error="empty_response",
        )

    rationale_match = _SOURCE_RATIONALE_RE.search(raw)
    program_match = _PROGRAM_RE.search(raw)

    if rationale_match is None:
        return ExplainParseResult(
            ok=False,
            rationale="",
            program_text="",
            error="missing_source_rationale_tags",
        )

    rationale = rationale_match.group(1).strip()
    if program_match is None:
        return ExplainParseResult(
            ok=False,
            rationale=rationale,
            program_text="",
            error="missing_program_tags",
        )

    program_text = extract_program_from_block(program_match.group(1))
    if not program_text:
        return ExplainParseResult(
            ok=False,
            rationale=rationale,
            program_text="",
            error="empty_program_block",
        )

    return ExplainParseResult(
        ok=True,
        rationale=rationale,
        program_text=program_text,
        error="ok",
    )
