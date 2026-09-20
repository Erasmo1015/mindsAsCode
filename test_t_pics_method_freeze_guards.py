"""Lightweight T-PICS method-freeze guards (no job launch, no test-set tuning)."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import teh
from utils.mem.rescore_initial_pool import compile_choose
from utils.teh.prompt_context import (
    RUNTIME_CONTRACT_HEADER,
    append_runtime_contract_if_present,
    attach_runtime_contract_to_prompt,
    build_deterministic_runtime_contract,
    infer_action_cardinality,
    infer_recursive_runtime_schema,
    observed_trials_excluding_test,
    trial_identity_key,
)
from utils.teh.sandbox_builtins import (
    TEH_SAFE_BUILTIN_CALLABLES,
    TEH_SAFE_BUILTIN_NAMES,
    TEH_UNSAFE_BUILTIN_NAMES,
    compile_choose_with_error,
    teh_program_namespace,
)


def _binary_trials():
    return [
        {
            "problem": {"cards": [1, 2], "option_keys": ["E", "J"], "schema_type": "B"},
            "history": [{"action": 0, "cards": [1]}],
            "action": 1,
        }
    ]


def _schulz_trials():
    return [
        {
            "problem": {
                "n_arms": 8,
                "option_keys": list(range(8)),
                "round": 1,
                "schema_type": "categorical_bandit",
            },
            "history": [{"action": 3, "reward": 1.0}],
            "action": 4,
            "options": list(range(8)),
        }
    ]


def _steyvers_trials():
    return [
        {
            "problem": {
                "n_arms": 4,
                "option_keys": [0, 1, 2, 3],
                "game": 1,
                "schema_type": "categorical_bandit",
            },
            "history": [{"action": 0, "reward": 1}],
            "action": 2,
        }
    ]


class SafeBuiltinParityTests(unittest.TestCase):
    def test_new_helpers_compile_in_teh_and_rescore(self) -> None:
        code = """
def choose(problem, history):
    letters = set(problem.get("option_keys") or [])
    xs = sorted(letters)
    ok = any(True for _ in xs) and all(True for _ in xs)
    n = ord("A") + chr(65).__len__()
    return 0.5 if ok and n else 0.5
"""
        fn, err = teh.compile_program_with_error(code)
        self.assertIsNone(err, msg=str(err))
        self.assertEqual(fn({"option_keys": ["E", "J"]}, []), 0.5)
        fn2 = compile_choose(code)
        self.assertTrue(callable(fn2))
        self.assertEqual(fn2({"option_keys": ["E", "J"]}, []), 0.5)

    def test_no_unsafe_io_builtins(self) -> None:
        ns = teh_program_namespace()
        builtins = ns["__builtins__"]
        for name in TEH_UNSAFE_BUILTIN_NAMES:
            self.assertNotIn(name, builtins)
        self.assertEqual(set(TEH_SAFE_BUILTIN_CALLABLES), set(TEH_SAFE_BUILTIN_NAMES))

    def test_open_is_not_available(self) -> None:
        code = "def choose(problem, history):\n    open('x','w')\n    return 0.5\n"
        fn, err = compile_choose_with_error(code)
        self.assertIsNone(err)
        with self.assertRaises(NameError):
            fn({}, [])


class ActionCardinalityTests(unittest.TestCase):
    def test_binary_and_multi_action(self) -> None:
        self.assertEqual(infer_action_cardinality(_binary_trials()), 2)
        self.assertEqual(infer_action_cardinality(_schulz_trials()), 8)
        self.assertEqual(infer_action_cardinality(_steyvers_trials()), 4)
        schema8 = infer_recursive_runtime_schema(_schulz_trials())
        self.assertIn("0, …, 7", schema8)
        self.assertNotIn("is int 0/1 (or a categorical", schema8)


class PromptContractTests(unittest.TestCase):
    def test_contract_covers_runtime_requirements(self) -> None:
        contract = build_deterministic_runtime_contract(_binary_trials())
        self.assertIn(RUNTIME_CONTRACT_HEADER, contract)
        self.assertIn("0, …, 1", contract)
        self.assertIn("option_keys[j]", contract)
        self.assertIn("string key labels", contract)
        self.assertIn("Allowed builtins", contract)
        self.assertIn("overrides any transferred source program", contract)
        wrapped = attach_runtime_contract_to_prompt("SOURCE PROGRAM HERE", contract)
        self.assertTrue(wrapped.rstrip().endswith(contract.strip()))

    def test_generation_append_is_last_and_legacy_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompts = Path(tmp)
            contract = build_deterministic_runtime_contract(_steyvers_trials())
            (prompts / "runtime_contract.txt").write_text(contract, encoding="utf-8")
            prompt = "base\n\n## source program\ndef choose():\n    return 0.5\n"
            out = append_runtime_contract_if_present(prompt, prompts)
            self.assertGreater(out.rfind(RUNTIME_CONTRACT_HEADER), out.find("source program"))
            self.assertIn("0, …, 3", out)
        self.assertEqual(
            append_runtime_contract_if_present("legacy", None),
            "legacy",
        )

    def test_train_val_examples_exclude_test_list(self) -> None:
        train = [{"problem": {"block_index": 0}, "action": 0, "history": []}]
        val = [{"problem": {"block_index": 2}, "action": 1, "history": []}]
        test = [{"problem": {"block_index": 1}, "action": 0, "history": []}]
        observed = observed_trials_excluding_test(train, val, test)
        ids = {trial_identity_key(t) for t in observed}
        self.assertEqual(ids, {"block_index:0", "block_index:2"})
        self.assertNotIn("block_index:1", ids)


class SplitIdentityTests(unittest.TestCase):
    def test_current_action_not_in_problem_or_history_prefix(self) -> None:
        trial = {
            "problem": {"cards": [2, 4], "option_keys": ["E", "J"]},
            "history": [{"action": 0, "feedback": 1.0}],
            "action": 1,
        }
        self.assertNotIn("action", trial["problem"])
        self.assertNotEqual(trial["history"][-1]["action"], trial["action"])

    def test_causal_history_appends_after_prediction_inputs(self) -> None:
        history: list = []
        rows = []
        for action in (0, 1, 0):
            rows.append({"history": list(history), "action": action})
            history.append({"action": action})
        self.assertEqual(rows[0]["history"], [])
        self.assertEqual(rows[1]["history"], [{"action": 0}])
        self.assertEqual(rows[2]["history"], [{"action": 0}, {"action": 1}])


class LegacyArgparseTests(unittest.TestCase):
    def test_max_error_prompt_chars_default_is_1200(self) -> None:
        tree = ast.parse(Path("teh.py").read_text(encoding="utf-8"))
        found = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "--max_error_prompt_chars":
                for kw in node.keywords:
                    if kw.arg == "default":
                        found = ast.literal_eval(kw.value)
        self.assertEqual(found, 1200)

    def test_t_pics_scripts_pass_max_error_zero(self) -> None:
        scripts = [
            Path("cluster/2026Sep17_T_PICS") / name
            for name in (
                "job_t_pics_target.sh",
                "job_t_pics_target_other_gpu.sh",
                "job_t_pics_source_pop.sh",
                "job_t_pics_source_pop_other_gpu.sh",
            )
        ] + [
            Path("cluster/2026Sep18_T_PICS_6x6/_6x6_common.sh"),
            Path("cluster/2026Sep18_T_PICS_gated/job_gated_l40s.sh"),
        ]
        for path in scripts:
            text = path.read_text(encoding="utf-8")
            self.assertIn("--max_error_prompt_chars 0", text, msg=str(path))

    def test_t_pics_scripts_pass_train_val_and_fresh_ten(self) -> None:
        paths = [
            Path("cluster/2026Sep17_T_PICS") / name
            for name in (
                "job_t_pics_target.sh",
                "job_t_pics_target_other_gpu.sh",
                "job_t_pics_source_pop.sh",
                "job_t_pics_source_pop_other_gpu.sh",
            )
        ] + [
            Path("cluster/2026Sep18_T_PICS_6x6/_6x6_common.sh"),
            Path("cluster/2026Sep18_T_PICS_6x6/job_6x6_h100.sh"),
            Path("cluster/2026Sep18_T_PICS_6x6/job_6x6_l40s.sh"),
            Path("cluster/2026Sep18_T_PICS_gated/job_gated_l40s.sh"),
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            if path.name != "job_6x6_h100.sh" and path.name != "job_6x6_l40s.sh":
                self.assertIn("--evolution_selection_score train_val", text, msg=str(path))
                self.assertIn("--fresh_n_candidates 10", text, msg=str(path))
                self.assertIn("--prefer_auto_llm_prompt", text, msg=str(path))
            if path.name != "_6x6_common.sh":
                self.assertIn(
                    'SEED_PATH="${SEED_PATH:-$(t_pics_seed_path "${DATASET}")}"',
                    text,
                    msg=str(path),
                )
            self.assertNotIn(
                "--seed_path persona_code_example/te_vanilla/choices13k.py",
                text,
                msg=str(path),
            )

    def test_t_pics_seed_path_helper(self) -> None:
        import subprocess

        script = r"""
set -euo pipefail
source cluster/2026Sep17_T_PICS/_t_pics_common.sh
t_pics_seed_path 13schulz2020finding
t_pics_seed_path steyvers_2009_bandit
t_pics_seed_path 1peterson2021using
t_pics_seed_path 14kool2016when
"""
        out = subprocess.check_output(["bash", "-c", script], text=True)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        self.assertEqual(
            lines,
            [
                "persona_code_example/teh/categorical_uniform.py",
                "persona_code_example/teh/categorical_uniform.py",
                "persona_code_example/te_vanilla/choices13k.py",
                "persona_code_example/te_vanilla/choices13k.py",
            ],
        )

    def test_t_pics_main_passes_limited_data_into_prompt_setup(self) -> None:
        text = Path("teh.py").read_text(encoding="utf-8")
        self.assertIn("limited_data_protocol=str(args.limited_data_protocol)", text)
        self.assertIn("--prefer_auto_llm_prompt", text)
        runtime = Path("utils/teh/teh_runtime.py").read_text(encoding="utf-8")
        self.assertIn("load_participant_limited_splits", runtime)
        self.assertIn("prompt_examples_from_retained_observed", runtime)


class MemTraceDefaultOnTests(unittest.TestCase):
    def test_argparse_mem_trace_default_true(self) -> None:
        tree = ast.parse(Path("teh.py").read_text(encoding="utf-8"))
        found = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "--mem_trace":
                for kw in node.keywords:
                    if kw.arg == "default":
                        found = ast.literal_eval(kw.value)
        self.assertIs(found, True)

    def test_run_evolution_and_global_phase_default_true(self) -> None:
        import inspect

        self.assertIs(
            inspect.signature(teh.run_evolution).parameters["mem_trace"].default,
            True,
        )
        self.assertIs(
            inspect.signature(teh.run_global_evolution_phase)
            .parameters["mem_trace"]
            .default,
            True,
        )

    def test_source_and_target_global_calls_pass_args_mem_trace(self) -> None:
        text = Path("teh.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("mem_trace=args.mem_trace"), 6)
        source_fn = text[
            text.find("def _run_t_pics_source_population") : text.find(
                "def _refinement_pool_best_metrics"
            )
        ]
        self.assertIn("mem_trace=args.mem_trace", source_fn)
        self.assertNotIn("mem_trace=False", source_fn)
        self.assertNotRegex(
            text,
            r"if\s+[^\n]*dataset[^\n]*:\s*\n\s*.*mem_trace\s*=\s*False",
        )


class GatedSourceExamplePidTests(unittest.TestCase):
    def test_suffix_example_uses_emnlp_range_first_person_not_valid_zero(self) -> None:
        from utils.teh.teh_datasets import emnlp_ordinal_range

        text = Path("utils/teh/explore_source_prompt.py").read_text(encoding="utf-8")
        self.assertIn("emnlp_ordinal_range(source_dataset)", text)
        self.assertIn("pid = int(pids[start])", text)
        self.assertNotIn("pid = int(pids[0])", text)
        self.assertEqual(emnlp_ordinal_range("4wulff2018description"), (1290, 1319))
        self.assertEqual(emnlp_ordinal_range("11enkavi2019recentprobes"), (0, 29))


if __name__ == "__main__":
    unittest.main()
