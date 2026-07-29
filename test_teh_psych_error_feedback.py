"""Focused tests for teh_psych error-feedback parity with teh.py."""

from __future__ import annotations

import unittest

import teh
from utils.teh_psych import evolution as evo
from utils.teh_psych.categorical_eval import evaluate_categorical_program


def _trial(target: int = 0) -> dict:
    return {
        "problem": {"options": [{"action": 0}, {"action": 1}]},
        "history": [],
        "target_action": target,
    }


class CategoricalEvalErrorEntryTests(unittest.TestCase):
    def test_first_error_carries_type_and_source_line(self) -> None:
        code = "def choose(problem, history):\n    return history[-1]['feedback']\n"
        choose_fn, err = teh.compile_program_with_error(code)
        self.assertIsNone(err)

        result = evaluate_categorical_program(choose_fn, [_trial()], n_seeds=1)
        self.assertNotEqual(result["errors"], 0)

        entry = result["first_error"]
        self.assertIsInstance(entry, dict)
        self.assertEqual(entry["error_type"], "IndexError")
        self.assertIn("history[-1]['feedback']", entry["invalid_line"])
        self.assertTrue(entry["normalized_key"])

    def test_falls_back_to_message_without_source(self) -> None:
        def choose(problem, history):
            raise ValueError("boom")

        result = evaluate_categorical_program(choose, [_trial()], n_seeds=1)
        self.assertEqual(result["first_error"], "boom")


class TehPsychFeedbackWiringTests(unittest.TestCase):
    def test_distinct_source_lines_group_separately(self) -> None:
        store: list = []
        for cid, code in (
            ("candidate_0", "def choose(problem, history):\n    return history[-1]['feedback']\n"),
            ("candidate_1", "def choose(problem, history):\n    return problem['missing_key']\n"),
        ):
            fn, _ = teh.compile_program_with_error(code)
            res = evaluate_categorical_program(fn, [_trial()], n_seeds=1)
            teh._record_invalid_program_error_summary(
                store,
                evo._coerce_eval_error_entry(res["first_error"]),
                iteration=2,
                participant_id=None,
                candidate_id=cid,
                history_path=None,
                quality_score=-0.5,
                eval_split="train",
                n_candidates_in_iteration=10,
            )
        groups = teh._group_errors_from_previous_iteration(
            store, iteration=3, previous_n_candidates=10
        )
        self.assertEqual(len(groups), 2)

    def test_same_failure_groups_and_counts(self) -> None:
        code = "def choose(problem, history):\n    return history[-1]['feedback']\n"
        fn, _ = teh.compile_program_with_error(code)
        store: list = []
        for cid in ("candidate_0", "candidate_3", "candidate_7"):
            res = evaluate_categorical_program(fn, [_trial()], n_seeds=1)
            teh._record_invalid_program_error_summary(
                store,
                evo._coerce_eval_error_entry(res["first_error"]),
                iteration=4,
                participant_id=None,
                candidate_id=cid,
                history_path=None,
                quality_score=-1.0,
                eval_split="train",
                n_candidates_in_iteration=10,
            )
        prompt = teh._build_past_error_prompt_section(
            store,
            iteration=5,
            max_error_prompt_chars=1200,
            previous_n_candidates=10,
        )
        self.assertIn("3/10 candidates", prompt)
        self.assertIn("IndexError", prompt)
        self.assertIn("contextual only", prompt.lower())

    def test_val_errors_recorded_and_test_still_excluded(self) -> None:
        entry = evo._coerce_eval_error_entry("choose() returned invalid probs")
        store: list = []
        teh._record_invalid_program_error_summary(
            store,
            entry,
            iteration=1,
            participant_id=None,
            candidate_id="candidate_0",
            history_path=None,
            quality_score=-0.2,
            eval_split="val",
            n_candidates_in_iteration=10,
        )
        self.assertEqual(len(store), 1)
        self.assertEqual(store[0]["eval_split"], "val")

        teh._record_invalid_program_error_summary(
            store,
            entry,
            iteration=1,
            participant_id=None,
            candidate_id="candidate_1",
            history_path=None,
            eval_split="test",
            n_candidates_in_iteration=10,
        )
        self.assertEqual(len(store), 1, "test-split errors must never be stored")

        prompt = teh._build_past_error_prompt_section(
            store, iteration=2, max_error_prompt_chars=1200, previous_n_candidates=10
        )
        self.assertIn("1/10 candidates", prompt)

    def test_evolution_passes_new_kwargs(self) -> None:
        import inspect

        src = inspect.getsource(evo.run_population_evolution)
        self.assertIn("previous_n_candidates=n_candidates_per_iteration", src)
        self.assertIn('eval_split="compile"', src)
        self.assertIn('eval_split="train"', src)
        self.assertIn('eval_split="val"', src)
        self.assertEqual(src.count("n_candidates_in_iteration=n_candidates_per_iteration"), 3)


if __name__ == "__main__":
    unittest.main()
