"""Qwen tokenizer helpers for MEM annotation token budgets."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

# Default snapshot used by pics_v3 / participant annot jobs.
_DEFAULT_TOKENIZER_JSON = (
    Path(os.environ.get("HF_HUB_CACHE", "/careAIDrive/zichang/cache/huggingface/hub"))
    / "models--Qwen--Qwen2.5-Coder-32B-Instruct"
    / "snapshots"
    / "381fc969f78efac66bc87ff7ddeadb7e73c218a7"
    / "tokenizer.json"
)


@lru_cache(maxsize=2)
def _load_tokenizer(tokenizer_json: str):
    from tokenizers import Tokenizer  # lazy: optional on login nodes

    return Tokenizer.from_file(tokenizer_json)


def resolve_qwen_tokenizer_json(
    explicit: Optional[str | Path] = None,
) -> Optional[Path]:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    env = os.environ.get("QWEN_TOKENIZER_JSON")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    if _DEFAULT_TOKENIZER_JSON.is_file():
        return _DEFAULT_TOKENIZER_JSON
    # Any snapshot under the model hub dir.
    hub = Path(os.environ.get("HF_HUB_CACHE", "/careAIDrive/zichang/cache/huggingface/hub"))
    matches = sorted(
        hub.glob(
            "models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/*/tokenizer.json"
        )
    )
    return matches[-1] if matches else None


def make_qwen_token_counter(
    tokenizer_json: Optional[str | Path] = None,
) -> Callable[[str], int]:
    """Return ``count(text) -> n_tokens`` using the Qwen tokenizer.json BPE."""
    path = resolve_qwen_tokenizer_json(tokenizer_json)
    if path is None:
        raise FileNotFoundError(
            "Qwen tokenizer.json not found; set QWEN_TOKENIZER_JSON or HF_HUB_CACHE"
        )
    tok = _load_tokenizer(str(path.resolve()))

    def _count(text: str) -> int:
        return len(tok.encode(text or "").ids)

    return _count


def count_qwen_tokens(text: str, *, tokenizer_json: Optional[str | Path] = None) -> int:
    return make_qwen_token_counter(tokenizer_json)(text)
