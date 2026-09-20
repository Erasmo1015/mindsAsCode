"""Regression: G.1 mem_trace records rejected non-elite candidates (legacy fields only)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.mem.trace import (
    append_mem_trace_record,
    build_candidate_record,
    iter_jsonl_records,
)


class TestMemTraceRejectedNonElite(unittest.TestCase):
    def test_rejected_non_elite_candidate_record_roundtrip(self) -> None:
        """Runtime-valid non-elite keeps selection_score; invalids are also traced."""
        code = "def choose(problem, history):\n    return 0.5\n"
        rec = build_candidate_record(
            dataset="1peterson2021using",
            participant_id="global",
            run_id="job_test",
            split_seed=0,
            phase="global_evolution",
            iteration=3,
            candidate_id="global_iteration_3_candidate_7",
            candidate_idx=7,
            source="fresh",
            code=code,
            runtime_valid=True,
            train_loglik=-0.9,
            val_loglik=-1.1,
            selection_score=-0.95,
            reference_parent_id="global_baseline",
            reference_parent_score=-0.693,
            reference_kind="seed_baseline",
            reference_type="seed_baseline",
            reference_id="global_baseline",
            reference_is_exact=True,
            delta_f=-0.95 - (-0.693),
            survived_elite_truncation=False,
            evolution_selection_score="train_val",
            prompted_parent_ids=["global_baseline"],
        )
        self.assertFalse(rec["survived_elite_truncation"])
        self.assertTrue(rec["runtime_valid"])
        self.assertEqual(rec["selection_score"], -0.95)
        # Slim legacy schema: no per-seed / sha / n_train extras.
        for banned in (
            "code",
            "program_sha256",
            "n_train",
            "n_eval_seeds",
            "selection_score_per_seed",
            "train_loglik_per_seed",
            "validity_status",
            "is_elite",
        ):
            self.assertNotIn(banned, rec)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem_trace.jsonl"
            append_mem_trace_record(path, rec)
            bad = build_candidate_record(
                dataset="1peterson2021using",
                participant_id="global",
                run_id="job_test",
                split_seed=0,
                phase="global_evolution",
                iteration=3,
                candidate_id="global_iteration_3_candidate_8",
                candidate_idx=8,
                source="normal",
                code="",
                runtime_valid=False,
                train_loglik=None,
                val_loglik=None,
                selection_score=None,
                reference_parent_id="global_iteration_1_candidate_0",
                reference_parent_score=-0.5,
                reference_kind="best_prompted_parent",
                delta_f=None,
                survived_elite_truncation=False,
                evolution_selection_score="train_val",
            )
            append_mem_trace_record(path, bad)
            rows = list(iter_jsonl_records([path]))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["candidate_id"], "global_iteration_3_candidate_7")
            self.assertFalse(rows[0]["survived_elite_truncation"])
            self.assertFalse(rows[1]["runtime_valid"])
            self.assertIsNone(rows[1]["selection_score"])
            line_bytes = len(
                json.dumps(rows[0], ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            self.assertLess(line_bytes, 1200)


if __name__ == "__main__":
    unittest.main()
