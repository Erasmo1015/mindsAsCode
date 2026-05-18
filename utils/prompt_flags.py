"""Shared flags for prompt assembly across evolution entrypoints."""

from __future__ import annotations

from pathlib import Path

# When False, single_code_template.txt is not loaded or appended to LLM prompts.
USE_SINGLE_CODE_TEMPLATE_IN_PROMPTS = False


def load_single_code_template(path: str | Path) -> str:
    """Load single_code_template.txt when enabled; otherwise return empty string."""
    if not USE_SINGLE_CODE_TEMPLATE_IN_PROMPTS:
        return ""
    return Path(path).read_text(encoding="utf-8")


def single_code_template_prompt_suffix(template: str) -> str:
    """Return newline-wrapped template for prompt suffix, or empty when disabled."""
    if not USE_SINGLE_CODE_TEMPLATE_IN_PROMPTS:
        return ""
    text = (template or "").strip()
    if not text:
        return ""
    return f"\n{text}\n"


def is_single_code_template_path(path: str | Path) -> bool:
    return Path(path).name == "single_code_template.txt"


def read_code_template_for_prompt(path: str | Path) -> str:
    """Load a code template file; respects disable flag only for single_code_template.txt."""
    path = Path(path)
    if is_single_code_template_path(path):
        return load_single_code_template(path)
    return path.read_text(encoding="utf-8")
