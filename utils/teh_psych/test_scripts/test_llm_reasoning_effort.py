"""Tests for Harmony reasoning_effort client wrapping."""

from __future__ import annotations

import pytest

from utils.teh.llm_reasoning import (
    ReasoningEffortClient,
    apply_reasoning_effort_to_create_kwargs,
    maybe_wrap_openai_client,
    normalize_llm_reasoning_effort,
)


def test_normalize_llm_reasoning_effort():
    assert normalize_llm_reasoning_effort(None) is None
    assert normalize_llm_reasoning_effort("none") is None
    assert normalize_llm_reasoning_effort("OFF") is None
    assert normalize_llm_reasoning_effort("low") == "low"
    assert normalize_llm_reasoning_effort("HIGH") == "high"
    with pytest.raises(ValueError):
        normalize_llm_reasoning_effort("smart")


def test_apply_reasoning_effort_preserves_existing():
    out = apply_reasoning_effort_to_create_kwargs(
        {"max_tokens": 1024, "extra_body": {"reasoning_effort": "high", "x": 1}},
        "low",
    )
    assert out["extra_body"]["reasoning_effort"] == "high"
    assert out["extra_body"]["x"] == 1


def test_apply_reasoning_effort_injects():
    out = apply_reasoning_effort_to_create_kwargs({"max_tokens": 1024}, "low")
    assert out["extra_body"] == {"reasoning_effort": "low"}


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return {"ok": True}


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()
        self.marker = "raw"


def test_maybe_wrap_injects_on_create():
    raw = _FakeClient()
    wrapped = maybe_wrap_openai_client(raw, "low")
    assert isinstance(wrapped, ReasoningEffortClient)
    assert wrapped.marker == "raw"
    wrapped.chat.completions.create(model="openai/gpt-oss-120b", max_tokens=1024)
    assert raw.chat.completions.last_kwargs["extra_body"]["reasoning_effort"] == "low"


def test_maybe_wrap_noop_when_unset():
    raw = _FakeClient()
    assert maybe_wrap_openai_client(raw, None) is raw
    assert maybe_wrap_openai_client(raw, "off") is raw
