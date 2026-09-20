"""Harmony / gpt-oss reasoning-effort helpers for OpenAI-compatible clients."""

from __future__ import annotations

from typing import Any, Dict, Optional

_ALLOWED = frozenset({"low", "medium", "high"})


def normalize_llm_reasoning_effort(value: Optional[str]) -> Optional[str]:
    """Return ``low``/``medium``/``high``, or ``None`` to leave server defaults."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "off", "disable", "disabled", "default"}:
        return None
    if text not in _ALLOWED:
        raise ValueError(
            f"llm_reasoning_effort must be one of {sorted(_ALLOWED)} "
            f"(or none/off); got {value!r}"
        )
    return text


def apply_reasoning_effort_to_create_kwargs(
    kwargs: Dict[str, Any],
    effort: Optional[str],
) -> Dict[str, Any]:
    """Inject ``extra_body.reasoning_effort`` unless already set by the caller."""
    effort = normalize_llm_reasoning_effort(effort)
    if effort is None:
        return kwargs
    out = dict(kwargs)
    extra = dict(out.get("extra_body") or {})
    if "reasoning_effort" not in extra:
        extra["reasoning_effort"] = effort
    out["extra_body"] = extra
    return out


class _CompletionsProxy:
    def __init__(self, completions: Any, effort: str) -> None:
        self._completions = completions
        self._effort = effort

    def create(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = apply_reasoning_effort_to_create_kwargs(kwargs, self._effort)
        return self._completions.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _ChatProxy:
    def __init__(self, chat: Any, effort: str) -> None:
        self._chat = chat
        self.completions = _CompletionsProxy(chat.completions, effort)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class ReasoningEffortClient:
    """Proxy OpenAI client that stamps Harmony ``reasoning_effort`` on every create."""

    def __init__(self, client: Any, effort: str) -> None:
        self._client = client
        self._effort = effort
        self.chat = _ChatProxy(client.chat, effort)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def maybe_wrap_openai_client(client: Any, effort: Optional[str]) -> Any:
    """Wrap ``client`` when ``effort`` is low/medium/high; otherwise return as-is."""
    normalized = normalize_llm_reasoning_effort(effort)
    if normalized is None:
        return client
    return ReasoningEffortClient(client, normalized)
