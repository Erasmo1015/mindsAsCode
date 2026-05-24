"""
Sanitize LLM outputs for TEH evolution prompts vs candidate programs.
"""
from __future__ import annotations

import io
import re
import tokenize
from typing import List, Optional, Tuple

_IMPORT_LINE_RE = re.compile(r"(?m)^\s*(?:import|from)\s+\S")
_CHOOSE_DEF_RE = re.compile(r"(?m)^\s*def choose\s*\(")
_TOP_LEVEL_CHOOSE_DEF_RE = re.compile(r"(?m)^def choose\s*\(")
_TRAILING_FENCE_RE = re.compile(r"\n```(?:python)?\s*\n[\s\S]*?```\s*$", re.IGNORECASE)
_FAKE_SIGMOID_MARKER = "1/(1+1/(1+"
_FORBIDDEN_DYNAMIC_RE = re.compile(
    r"__import__\s*\(|\bimportlib\b|\beval\s*\(|\bexec\s*\(|\bglobals\s*\(|\blocals\s*\(",
    re.IGNORECASE,
)
_PROBS_GET_WITH_DEFAULT_RE = re.compile(
    r"""
    ^
    (?P<indent>[ \t]*)
    (?P<var>[A-Za-z_][A-Za-z0-9_]*)
    \s*=\s*
    (?P<obj>[A-Za-z_][A-Za-z0-9_]*)
    \s*\.\s*get\s*\(\s*
    (?P<quote>["'])probs(?P=quote)\s*,\s*
    (?P<default>.+?)
    \s*\)\s*$
    """,
    re.VERBOSE,
)


def strip_embedded_choose_from_evolution_prompt(text: str) -> str:
    """
    Keep evolution instructions only; remove trailing choose() implementations.

    The template may document `def choose(...)` once in an API docstring. A second
    top-level `def choose(` is treated as an appended solution and removed.
    """
    if not text:
        return ""
    out = _TRAILING_FENCE_RE.sub("", text).strip()
    # For prompt stripping, only consider top-level choose() definitions.
    matches = list(_TOP_LEVEL_CHOOSE_DEF_RE.finditer(out))
    if len(matches) > 1:
        out = out[: matches[-1].start()].rstrip()
    elif len(matches) == 1:
        # Conservative extra case: remove a single choose() only when it looks like
        # a trailing appended implementation (not API documentation).
        match = matches[0]
        choose_start = out.find("def choose", match.start(), match.end())
        if choose_start < 0:
            choose_start = match.start()
        prefix = out[:choose_start]
        suffix = out[choose_start:]
        if _should_strip_single_trailing_choose(prefix, suffix, len(out), choose_start):
            out = prefix.rstrip()
    return out


def _strip_full_line_comments(code: str) -> str:
    lines: List[str] = []
    for line in code.splitlines():
        if _IMPORT_LINE_RE.match(line):
            continue
        if line.strip().startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _strip_python_comments(code: str) -> str:
    """Remove # comments (full-line and inline) via tokenize; fall back if needed."""
    try:
        tokens = [
            (tok.type, tok.string)
            for tok in tokenize.tokenize(io.BytesIO(code.encode("utf-8")).readline)
            if tok.type not in (tokenize.COMMENT, tokenize.ENCODING)
        ]
        out = tokenize.untokenize(tokens)
        if isinstance(out, bytes):
            out = out.decode("utf-8")
    except (tokenize.TokenError, SyntaxError):
        return _strip_full_line_comments(code)
    lines = [ln.rstrip() for ln in out.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _has_fake_sigmoid(code: str) -> bool:
    compact = re.sub(r"\s+", "", code)
    return _FAKE_SIGMOID_MARKER in compact


def _count_choose_definitions(code: str) -> int:
    return len(_CHOOSE_DEF_RE.findall(code))


def _extract_fenced_blocks(text: str) -> List[str]:
    # Single pass with optional language tag avoids duplicate python extraction.
    fence_re = re.compile(
        r"```(?:[ \t]*[a-zA-Z0-9_+-]+[^\n]*)?\n([\s\S]*?)```",
        re.IGNORECASE,
    )
    blocks: List[str] = []
    for match in fence_re.finditer(text):
        block = (match.group(1) or "").strip()
        if block:
            blocks.append(block)
    return blocks


def _looks_like_instruction_prefix(text: str) -> bool:
    prefix = text.strip().lower()
    if not prefix:
        return False
    marker_count = sum(
        marker in prefix
        for marker in (
            "requirements",
            "instruction",
            "history",
            "action semantics",
            "p(action=1)",
            "return",
            "dataset",
            "prompt",
        )
    )
    if len(prefix) >= 280 and prefix.count("\n") >= 6:
        return True
    return len(prefix) >= 120 and prefix.count("\n") >= 3 and marker_count >= 2


def _looks_like_executable_choose_block(choose_block: str) -> bool:
    lines = choose_block.splitlines()
    if not lines:
        return False
    in_docstring = False
    body_lines = 0
    executable_hits = 0
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            if stripped.count(quote) >= 2:
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if not line.startswith((" ", "\t")):
            break
        body_lines += 1
        if stripped.startswith(("return ", "if ", "for ", "while ", "try:", "except ")):
            executable_hits += 1
        if "=" in stripped and "==" not in stripped and "!=" not in stripped:
            executable_hits += 1
    return body_lines >= 2 and executable_hits >= 1


def _should_strip_single_trailing_choose(
    prefix: str,
    choose_block: str,
    total_len: int,
    choose_start: int,
) -> bool:
    near_end = choose_start >= int(total_len * 0.55)
    return (
        near_end
        and _looks_like_instruction_prefix(prefix)
        and _looks_like_executable_choose_block(choose_block)
    )


def _passes_python_syntax(candidate: str) -> bool:
    try:
        compile(candidate, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False


def _marker_present(code: str, marker: str) -> bool:
    """Match required markers; ``def choose(`` allows optional whitespace before ``(``."""
    if marker == "def choose(":
        return _CHOOSE_DEF_RE.search(code) is not None
    return marker in code


def _slice_from_marker(code: str, marker: str) -> Optional[str]:
    if marker == "def choose(":
        match = _CHOOSE_DEF_RE.search(code)
        if match is None:
            return None
        return code[match.start() :].strip()
    idx = code.find(marker)
    if idx < 0:
        return None
    return code[idx:].strip()


def _normalize_probs_none_fallback(code: str) -> str:
    """
    Repair a common unsafe pattern from LLM outputs:
      probs = gamble.get("probs", <default>)
    This default is not used when probs exists but is None.
    """
    out_lines: List[str] = []
    for line in code.splitlines():
        m = _PROBS_GET_WITH_DEFAULT_RE.match(line)
        if m is None:
            out_lines.append(line)
            continue
        indent = m.group("indent")
        var = m.group("var")
        obj = m.group("obj")
        default_expr = m.group("default").strip()
        out_lines.append(f'{indent}{var} = {obj}.get("probs")')
        out_lines.append(f"{indent}if {var} is None:")
        out_lines.append(f"{indent}    {var} = {default_expr}")
    return "\n".join(out_lines)


def _extract_python_from_llm_reply(
    text: str,
    *,
    required_markers: Optional[Tuple[str, ...]] = ("def choose(",),
) -> str:
    """Extract first syntax-valid Python block containing required markers."""
    if not text:
        return ""

    candidates: List[str] = []
    candidates.extend(_extract_fenced_blocks(text))
    candidates.append(text.strip())

    expanded: List[str] = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        expanded.append(c)
        if required_markers:
            for marker in required_markers:
                sliced = _slice_from_marker(c, marker)
                if sliced is not None and sliced != c:
                    expanded.append(sliced)

    seen = set()
    ordered: List[str] = []
    for c in expanded:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for c in ordered:
        if required_markers and not all(_marker_present(c, m) for m in required_markers):
            continue
        if _passes_python_syntax(c):
            return c
    return ""


def describe_sanitize_failure(
    text: str,
    *,
    required_markers: Optional[Tuple[str, ...]] = ("def choose(",),
) -> str:
    """Human-readable reason ``sanitize_evolution_candidate_code`` would return empty."""
    if not text or not text.strip():
        return "empty_llm_content"
    extracted = _extract_python_from_llm_reply(text, required_markers=required_markers)
    if not extracted:
        if required_markers and not all(
            _marker_present(text, m) for m in required_markers
        ):
            return f"missing_markers:{required_markers}"
        return "no_syntax_valid_block_with_markers"
    if _IMPORT_LINE_RE.search(extracted):
        return "import_statements_present"
    n_choose = _count_choose_definitions(extracted)
    if n_choose != 1:
        return f"choose_definition_count={n_choose}"
    cleaned = _strip_python_comments(extracted)
    if not cleaned or _count_choose_definitions(cleaned) != 1:
        return "comments_stripped_away_choose_or_empty"
    if _has_fake_sigmoid(cleaned):
        return "fake_sigmoid_pattern"
    if _FORBIDDEN_DYNAMIC_RE.search(cleaned):
        return "forbidden_dynamic_code_present"
    if not _passes_python_syntax(cleaned):
        return "syntax_error_after_cleaning"
    return "ok"


def sanitize_evolution_candidate_code(
    text: str,
    *,
    required_markers: Optional[Tuple[str, ...]] = ("def choose(",),
) -> str:
    """
    Extract a single choose() implementation: no imports, one def, comments stripped.
    """
    if not text or not text.strip():
        return ""

    extracted = _extract_python_from_llm_reply(text, required_markers=required_markers)
    if not extracted:
        return ""

    if _IMPORT_LINE_RE.search(extracted):
        return ""

    if _count_choose_definitions(extracted) != 1:
        return ""

    cleaned = _strip_python_comments(extracted)
    if not cleaned or _count_choose_definitions(cleaned) != 1:
        return ""

    cleaned = _normalize_probs_none_fallback(cleaned)

    if _has_fake_sigmoid(cleaned):
        return ""

    if _FORBIDDEN_DYNAMIC_RE.search(cleaned):
        return ""

    if not _passes_python_syntax(cleaned):
        return ""

    return cleaned


CANDIDATE_OUTPUT_RULES = """
Candidate output rules (strict):
1. Output ONLY raw Python code — no prose, no explanations, no preamble or postamble.
2. Output ONLY ONE complete: def choose(problem, history):
3. No markdown code fences (no ```).
4. No # comments (inline or full-line).
5. No import statements.
6. No example usage or __main__ blocks.
7. No helper functions outside choose(); nest helpers inside choose() if needed.
8. Use only the provided problem and history.
9. Keep the function concise, but do not simplify away useful behavioral structure.
10. If gamble probs can be None, use explicit None checks (not only dict.get default fallback).
"""