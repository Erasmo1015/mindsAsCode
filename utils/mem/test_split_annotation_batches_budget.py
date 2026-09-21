#!/usr/bin/env python3
"""Unit tests for Qwen-aware annotation batch splitting and 16k defaults."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mem.annotation_context import (  # noqa: E402
    ANNOTATION_SAFETY_MARGIN_TOKENS,
    ANNOTATION_VLLM_MAX_MODEL_LEN,
    PARTICIPANT_DEFAULT_MAX_CANDIDATES_PER_BATCH,
    PARTICIPANT_DEFAULT_MAX_TOKENS,
    POPULATION_DEFAULT_BATCH_SIZE,
    POPULATION_DEFAULT_MAX_TOKENS,
)
from utils.mem.trace import (  # noqa: E402
    allowed_annotation_input_tokens,
    pack_items_under_chat_budget,
    split_annotation_batches,
)


class TestCanonicalDefaults(unittest.TestCase):
    def test_annotation_context_aligned_with_synthesis_16k(self) -> None:
        self.assertEqual(ANNOTATION_VLLM_MAX_MODEL_LEN, 16384)
        self.assertEqual(POPULATION_DEFAULT_MAX_TOKENS, 8192)
        self.assertEqual(PARTICIPANT_DEFAULT_MAX_TOKENS, 2048)
        self.assertEqual(POPULATION_DEFAULT_BATCH_SIZE, 2)
        self.assertEqual(PARTICIPANT_DEFAULT_MAX_CANDIDATES_PER_BATCH, 5)
        self.assertEqual(ANNOTATION_SAFETY_MARGIN_TOKENS, 512)

    def test_population_audit_max_fits_under_16k(self) -> None:
        # Audited pics_v3_g1_schema_v5 reconstructed input max (batch≤2).
        max_input = 3517
        total = (
            max_input
            + POPULATION_DEFAULT_MAX_TOKENS
            + ANNOTATION_SAFETY_MARGIN_TOKENS
        )
        self.assertLessEqual(total, ANNOTATION_VLLM_MAX_MODEL_LEN)
        # Headroom under 16k with current settings.
        self.assertEqual(total, 3517 + 8192 + 512)
        self.assertEqual(ANNOTATION_VLLM_MAX_MODEL_LEN - total, 4163)

    def test_participant_audit_max_fits_under_16k(self) -> None:
        # Audited gated eligible packs (batch≤5) Qwen chat max.
        max_input = 6305
        total = (
            max_input
            + PARTICIPANT_DEFAULT_MAX_TOKENS
            + ANNOTATION_SAFETY_MARGIN_TOKENS
        )
        self.assertLessEqual(total, ANNOTATION_VLLM_MAX_MODEL_LEN)
        self.assertEqual(total, 6305 + 2048 + 512)
        self.assertEqual(ANNOTATION_VLLM_MAX_MODEL_LEN - total, 7519)


class TestSplitBudget(unittest.TestCase):
    def test_exact_budget_accounts_for_reserved_output(self) -> None:
        # Fake tokenizer: 1 token per character.
        def counter(text: str) -> int:
            return len(text)

        def build_user(ref: str, batch):
            codes = "".join(c["code"] for c in batch)
            return f"U{ref}{codes}"

        cands = [
            {"candidate_id": "a", "code": "aa"},
            {"candidate_id": "b", "code": "bb"},
            {"candidate_id": "c", "code": "cc"},
        ]
        batches = split_annotation_batches(
            cands,
            reference_code="rr",
            system_prompt="SS",
            build_user_prompt=build_user,
            token_counter=counter,
            max_model_len=20,
            reserved_output_tokens=5,
            safety_margin_tokens=3,
            max_candidates_per_batch=5,
        )
        sizes = [len(b) for b in batches]
        self.assertTrue(all(s <= 3 for s in sizes), sizes)
        self.assertEqual(sum(sizes), 3)

    def test_legacy_char4_still_works(self) -> None:
        cands = [{"candidate_id": "a", "code": "x" * 20}]
        batches = split_annotation_batches(
            cands,
            reference_code="y" * 20,
            base_prompt_chars=100,
            max_input_tokens=500,
            max_candidates_per_batch=5,
        )
        self.assertEqual(len(batches), 1)

    def test_pack_reduces_batch_never_truncates(self) -> None:
        def est(batch):
            # 100 tokens base + 50 per item
            return 100 + 50 * len(batch)

        items = list(range(5))
        # allowed = 16384 - 8192 - 512 = 7680; all items fit easily under soft cap 2
        batches, oversized = pack_items_under_chat_budget(
            items,
            estimate_chat_tokens=est,
            max_items_per_batch=2,
            max_model_len=ANNOTATION_VLLM_MAX_MODEL_LEN,
            reserved_output_tokens=POPULATION_DEFAULT_MAX_TOKENS,
            safety_margin_tokens=ANNOTATION_SAFETY_MARGIN_TOKENS,
        )
        self.assertEqual(oversized, [])
        self.assertTrue(all(len(b) <= 2 for b in batches))
        self.assertEqual(sum(len(b) for b in batches), 5)

    def test_oversized_singleton_logged_not_truncated(self) -> None:
        def est(batch):
            return 9000  # always exceeds allowed under pop budget (7680)

        batches, oversized = pack_items_under_chat_budget(
            [{"id": "huge"}],
            estimate_chat_tokens=est,
            max_items_per_batch=2,
            max_model_len=ANNOTATION_VLLM_MAX_MODEL_LEN,
            reserved_output_tokens=POPULATION_DEFAULT_MAX_TOKENS,
            safety_margin_tokens=ANNOTATION_SAFETY_MARGIN_TOKENS,
        )
        self.assertEqual(batches, [])
        self.assertEqual(len(oversized), 1)
        self.assertEqual(oversized[0][0]["id"], "huge")
        self.assertEqual(oversized[0][1], 9000)

    def test_allowed_input_tokens_inequality(self) -> None:
        allowed = allowed_annotation_input_tokens(
            max_model_len=ANNOTATION_VLLM_MAX_MODEL_LEN,
            reserved_output_tokens=POPULATION_DEFAULT_MAX_TOKENS,
            safety_margin_tokens=ANNOTATION_SAFETY_MARGIN_TOKENS,
        )
        self.assertEqual(allowed, 16384 - 8192 - 512)
        with self.assertRaises(ValueError):
            allowed_annotation_input_tokens(
                max_model_len=100,
                reserved_output_tokens=80,
                safety_margin_tokens=30,
            )


if __name__ == "__main__":
    unittest.main()
