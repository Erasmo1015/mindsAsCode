"""Immutable history-robustness prompt block for PICS v3 candidate generation."""
from __future__ import annotations

import threading
from typing import Optional

HISTORY_ROBUSTNESS_MARKER = "HISTORY_ROBUSTNESS_BLOCK_V3"

HISTORY_ROBUSTNESS_BLOCK = (
    f"[{HISTORY_ROBUSTNESS_MARKER}]\n"
    "`history` may be empty, and different history entries may contain different "
    "fields. Never assume optional fields such as `feedback`, `reward`, or "
    "outcome fields exist or are non-null. Check for a key or use `.get(...)` "
    "before reading it. Only fields explicitly required by the current "
    "task/API contract may be accessed directly.\n"
    f"[/{HISTORY_ROBUSTNESS_MARKER}]"
)

_tls = threading.local()


def using_pics_v3_prompt_robustness() -> bool:
    return bool(getattr(_tls, "enabled", False))


class pics_v3_prompt_robustness_scope:
    """Thread-local: inject immutable history robustness into candidate prompts."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._prev = False

    def __enter__(self) -> "pics_v3_prompt_robustness_scope":
        self._prev = using_pics_v3_prompt_robustness()
        _tls.enabled = self.enabled
        return self

    def __exit__(self, *exc: object) -> None:
        _tls.enabled = self._prev


def ensure_history_robustness_block(text: str) -> str:
    """Insert the immutable block once; do not invent trial fields."""
    body = text or ""
    if HISTORY_ROBUSTNESS_MARKER in body:
        return body
    return body.rstrip() + "\n\n" + HISTORY_ROBUSTNESS_BLOCK + "\n"


def maybe_attach_history_robustness_after_task_description(base_prompt: str) -> str:
    """Attach after dataset-adaptive task text when the v3 prompt scope is active."""
    if not using_pics_v3_prompt_robustness():
        return base_prompt
    return ensure_history_robustness_block(base_prompt)
