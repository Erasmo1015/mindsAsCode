"""Guards for TEH sandbox builtins and auto-prompt schema notes (not the base prompt file)."""

from __future__ import annotations

import unittest

import teh
from utils.teh.prompt_context import infer_recursive_runtime_schema
from utils.teh_transfer.prompts import SourceTransferContext, build_transfer_source_suffix


class SafeBuiltinTests(unittest.TestCase):
    def test_set_sorted_any_all_ord_compile(self) -> None:
        code = """
def choose(problem, history):
    letters = set(problem.get("option_keys") or [])
    xs = sorted(letters)
    ok = any(len(h) > 0 for h in (history or [])) and all(True for _ in xs)
    return 0.5 if ok or ord("A") else 0.5
"""
        fn, err = teh.compile_program_with_error(code)
        self.assertIsNone(err, msg=str(err))
        self.assertTrue(callable(fn))
        self.assertEqual(
            fn({"option_keys": ["E", "J"]}, [{"action": 0}]),
            0.5,
        )


class SchemaNoteTests(unittest.TestCase):
    def test_option_keys_note_says_integer_action(self) -> None:
        trials = [
            {
                "problem": {
                    "cards": [1, 2],
                    "option_keys": ["E", "J"],
                    "schema_type": "B",
                },
                "history": [{"action": 0, "cards": [1]}],
            }
        ]
        schema = infer_recursive_runtime_schema(trials)
        self.assertIn("integer action index", schema)
        self.assertIn("0, …, 1", schema)
        self.assertIn("option_keys.index", schema)
        self.assertIn("history vs problem", schema)
        self.assertIn("Guard every division", schema)
        self.assertNotIn("is int 0/1 (or a categorical", schema)

    def test_stage_note_when_stage_varies(self) -> None:
        trials = [
            {
                "problem": {"stage": 1, "option_keys": ["A", "B"], "spaceship": "x"},
                "history": [{"stage": 1, "action": 0, "spaceship": "x"}],
            },
            {
                "problem": {"stage": 2, "option_keys": ["A", "B"], "planet": "y"},
                "history": [{"stage": 2, "action": 1, "planet": "y"}],
            },
        ]
        schema = infer_recursive_runtime_schema(trials)
        self.assertIn("stage-conditional keys", schema)


class TransferSuffixTests(unittest.TestCase):
    def test_rewrite_target_schema_instruction(self) -> None:
        text = build_transfer_source_suffix(
            [
                SourceTransferContext(
                    dataset_alias="11enkavi2019recentprobes",
                    display_name="Enkavi",
                    task_description="probe task",
                    example_trial_text="(example)",
                    best_program_code="def choose(problem, history):\n    return 0.5\n",
                    best_loglik=-0.3,
                )
            ]
        )
        self.assertIn("Do not copy source problem keys", text)
        self.assertIn("integer action id", text)


if __name__ == "__main__":
    unittest.main()
