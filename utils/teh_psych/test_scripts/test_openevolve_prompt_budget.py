"""CPU tests: ICLR OpenEvolve prompts fit the Qwen 30000 input ceiling."""
from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path("/home/zichang/repo/mindsAsCode")
if not (REPO_ROOT / "baseline_methods/Psych101/run_openevolve.py").is_file():
    REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baseline_methods" / "Psych101"))

_oe = types.ModuleType("openevolve")
_oe.OpenEvolve = object
sys.modules.setdefault("openevolve", _oe)
_oe_cfg = types.ModuleType("openevolve.config")
_oe_cfg.Config = object
sys.modules.setdefault("openevolve.config", _oe_cfg)
_oe_pp = types.ModuleType("openevolve.process_parallel")
_oe_pp.ProcessParallelController = object
sys.modules.setdefault("openevolve.process_parallel", _oe_pp)

from run_openevolve import (  # noqa: E402
    ICLR_FROZEN_INCLUDE_ARTIFACTS,
    ICLR_FROZEN_INPUT_TOKEN_CEILING,
    ICLR_FROZEN_LLM_MAX_TOKENS,
    ICLR_FROZEN_MAX_PROGRAM_CHARS,
    ICLR_FROZEN_MAX_PROMPT_TRAIN_TRIALS,
    ICLR_FROZEN_NUM_DIVERSE_PROGRAMS,
    ICLR_FROZEN_NUM_TOP_PROGRAMS,
    ICLR_FROZEN_VLLM_MAX_MODEL_LEN,
    RequiredPromptOverflowError,
    _drop_inspiration_section,
    build_arg_parser,
    chat_input_token_count,
    choose_api_text,
    dataset_n_actions,
    format_trial_compact,
    get_qwen_tokenizer,
    resolve_openevolve_seed_and_prompt,
    split_official_optional_programs,
    truncate_vanilla_messages,
    vanilla_dataset_description,
)
from utils.teh.teh_datasets import is_categorical_output_dataset  # noqa: E402
from utils.teh_psych.test_scripts.test_iclr_baseline_guards import ICLR_15  # noqa: E402

pytest.importorskip("transformers")


def _require_qwen():
    tok = get_qwen_tokenizer()
    if tok is None:
        pytest.skip("Qwen tokenizer not in local HuggingFace cache")
    return tok


def _max_code(n_chars: int = ICLR_FROZEN_MAX_PROGRAM_CHARS) -> str:
    lines = ["def choose(problem, history):", "    x = 0"]
    i = 0
    while sum(len(ln) + 1 for ln in lines) < n_chars - 40:
        lines.append(f"    x = x + {i}  # {i:#034x}")
        i += 1
    lines.append("    return 0.5")
    text = "\n".join(lines) + "\n"
    return text[:n_chars] if len(text) > n_chars else text


def _prog(pid: str, code: str, score: float) -> dict:
    return {"id": pid, "code": code, "metrics": {"combined_score": score}}


def _examples(alias: str, n: int = 60) -> str:
    lines = []
    for i in range(n):
        trial = {
            "problem": {
                "dataset_alias": alias,
                "option_keys": [0, 1, 2, 3] if alias == "steyvers_2009_bandit" else [0, 1],
                "options": [{"action": a} for a in range(8)]
                if alias == "13schulz2020finding"
                else [{"action": a} for a in range(4)],
                "n_arms": 8 if alias == "13schulz2020finding" else 4,
                "gamble_A": {"probs": [0.5, 0.5], "rewards": [1.0, -1.0]},
                "gamble_B": {"probs": [1.0], "rewards": [0.0]},
            },
            "history": [],
            "action": i % 2,
        }
        lines.append(format_trial_compact(trial, "train" if i % 2 else "val"))
    return "\n".join(lines)


def _pack(
    alias: str,
    parent: str,
    *,
    top=None,
    diverse=None,
    inspirations=None,
    previous_programs=None,
    artifacts=None,
    n_examples: int = 60,
    max_prompt_tokens: int = ICLR_FROZEN_INPUT_TOKEN_CEILING,
):
    return truncate_vanilla_messages(
        task_text=vanilla_dataset_description(alias),
        program_code=parent,
        metrics={"combined_score": -0.6931},
        trials_compact=_examples(alias, n=n_examples),
        max_prompt_tokens=max_prompt_tokens,
        reserved_completion_tokens=ICLR_FROZEN_LLM_MAX_TOKENS,
        model_context_len=ICLR_FROZEN_VLLM_MAX_MODEL_LEN,
        categorical=is_categorical_output_dataset(alias),
        n_actions=dataset_n_actions(alias),
        top_programs=list(top or []),
        diverse_programs=list(diverse or []),
        inspirations=list(inspirations or []),
        previous_programs=list(previous_programs or []),
        artifacts=artifacts,
    )


def _assert_required(alias: str, prompt: dict, state, *, n_examples: int = 60) -> int:
    n = chat_input_token_count(prompt["system"], prompt["user"])
    assert n == state.estimated_tokens
    assert n <= ICLR_FROZEN_INPUT_TOKEN_CEILING
    assert n + ICLR_FROZEN_LLM_MAX_TOKENS <= ICLR_FROZEN_VLLM_MAX_MODEL_LEN
    assert state.examples_available == n_examples
    assert state.examples_included == n_examples
    assert state.prompt_trial_count == n_examples
    assert not any(s.startswith("cap_prompt_trials_") for s in state.steps)
    assert "# API" in prompt["user"]
    assert "def choose(problem, history)" in prompt["user"]
    task_section = prompt["user"].split("# API", 1)[0]
    assert "Only output the function" not in task_section
    assert "Pure Python, no imports" not in task_section
    assert "remain constant within a problem" not in task_section
    assert "Do not read a current-trial action/label field." not in prompt["user"]
    assert "no current response" not in prompt["user"]
    api = choose_api_text(
        categorical=is_categorical_output_dataset(alias),
        n_actions=dataset_n_actions(alias),
    )
    assert api.splitlines()[0] in prompt["user"]
    if is_categorical_output_dataset(alias):
        assert "dict[int, float]" in prompt["user"]
        assert "P(action=1) for the second option_keys entry" not in prompt["user"]
        assert f"K={dataset_n_actions(alias)}" in prompt["user"]
    return n


def test_official_inspiration_default_is_two_not_zero():
    args = build_arg_parser().parse_args([])
    assert args.num_diverse_programs == ICLR_FROZEN_NUM_DIVERSE_PROGRAMS == 2
    assert args.num_top_programs == ICLR_FROZEN_NUM_TOP_PROGRAMS == 3
    assert args.max_prompt_train_trials == ICLR_FROZEN_MAX_PROMPT_TRAIN_TRIALS == 60
    assert args.hard_prompt_token_cap == ICLR_FROZEN_INPUT_TOKEN_CEILING == 30000
    assert args.llm_max_tokens == ICLR_FROZEN_LLM_MAX_TOKENS == 1024
    assert args.include_artifacts is ICLR_FROZEN_INCLUDE_ARTIFACTS is True


def test_all_15_early_small_official_optional_blocks_fit():
    _require_qwen()
    seed_parent = resolve_openevolve_seed_and_prompt("1peterson2021using")[0].read_text(
        encoding="utf-8"
    )
    cat_parent = resolve_openevolve_seed_and_prompt("steyvers_2009_bandit")[0].read_text(
        encoding="utf-8"
    )
    for alias in ICLR_15:
        parent = cat_parent if is_categorical_output_dataset(alias) else seed_parent
        worker_top = [_prog(f"t{i}", parent, -0.1 * i) for i in range(5)]
        top, diverse, _insp = split_official_optional_programs(
            worker_top, [], num_top=3, num_diverse=2, rng=random.Random(0)
        )
        insp = [_prog(f"i{i}", parent, -0.7 - 0.1 * i) for i in range(2)]
        previous = [
            {
                "id": f"p{i}",
                "changes_description": f"attempt {i}",
                "metrics": {"combined_score": -0.2 * i},
                "metadata": {"changes": f"attempt {i}"},
            }
            for i in range(3)
        ]
        prompt, state = _pack(
            alias,
            parent,
            top=top,
            diverse=diverse,
            inspirations=insp,
            previous_programs=previous,
            artifacts={"stdout": "ok"},
        )
        _assert_required(alias, prompt, state)
        assert state.top_programs_kept == 3
        assert state.diverse_programs_kept == 2
        assert state.inspirations_kept == 2
        assert state.previous_attempts_kept == 1
        assert state.artifacts_kept == 1
        assert "## Top Performing Programs" in prompt["user"]
        assert "## Diverse Programs" in prompt["user"]
        assert "## Inspiration Programs" in prompt["user"]
        assert "## Previous Attempts" in prompt["user"]
        assert "## Last Execution Output" in prompt["user"]


def test_all_15_mixed_size_keeps_examples_and_some_optional():
    _require_qwen()
    seed_parent = resolve_openevolve_seed_and_prompt("1peterson2021using")[0].read_text(
        encoding="utf-8"
    )
    cat_parent = resolve_openevolve_seed_and_prompt("steyvers_2009_bandit")[0].read_text(
        encoding="utf-8"
    )
    huge = _max_code(ICLR_FROZEN_MAX_PROGRAM_CHARS)
    for alias in ICLR_15:
        parent = cat_parent if is_categorical_output_dataset(alias) else seed_parent
        top = [_prog(f"t{i}", parent, -0.1 * i) for i in range(3)]
        diverse = [_prog(f"d{i}", huge, -0.5 - 0.1 * i) for i in range(2)]
        insp = [_prog(f"i{i}", parent, -0.8 - 0.1 * i) for i in range(2)]
        prompt, state = _pack(alias, parent, top=top, diverse=diverse, inspirations=insp)
        _assert_required(alias, prompt, state)
        kept = state.top_programs_kept + state.diverse_programs_kept + state.inspirations_kept
        # Under the 30000 ceiling, mixed-size optional blocks may all fit; never
        # trim the 60 examples before optional programs are considered.
        assert state.top_programs_kept == 3
        assert kept <= 7
        assert state.examples_included == 60
        stripped = _drop_inspiration_section(prompt["user"])
        assert stripped.count("split=") == 60
        assert "# API" in stripped


def test_all_15_max_10k_programs_drop_optional_keep_examples():
    _require_qwen()
    parent = _max_code(ICLR_FROZEN_MAX_PROGRAM_CHARS)
    huge = _max_code(ICLR_FROZEN_MAX_PROGRAM_CHARS)
    for alias in ICLR_15:
        top = [_prog(f"t{i}", huge, -0.1 * i) for i in range(3)]
        diverse = [_prog(f"d{i}", huge, -0.5 - 0.1 * i) for i in range(2)]
        insp = [_prog(f"i{i}", huge, -0.8 - 0.1 * i) for i in range(2)]
        prompt, state = _pack(alias, parent, top=top, diverse=diverse, inspirations=insp)
        n = _assert_required(alias, prompt, state)
        assert n + ICLR_FROZEN_LLM_MAX_TOKENS <= ICLR_FROZEN_VLLM_MAX_MODEL_LEN
        kept = state.top_programs_kept + state.diverse_programs_kept + state.inspirations_kept
        assert kept < 7
        assert any(s.startswith("drop_") for s in state.steps)
        stripped = _drop_inspiration_section(prompt["user"])
        assert stripped.count("split=") == 60
        assert "# API" in stripped
        if is_categorical_output_dataset(alias):
            assert "dict[int, float]" in stripped
            assert f"K={dataset_n_actions(alias)}" in stripped


def test_safety_guard_drops_optional_not_the_contract():
    _require_qwen()
    from run_openevolve import _minimal_shrink_user_message

    alias = "13schulz2020finding"
    parent = resolve_openevolve_seed_and_prompt(alias)[0].read_text(encoding="utf-8")
    prompt, _state = _pack(
        alias,
        parent,
        top=[_prog("t0", parent, -0.1)],
        diverse=[_prog("d0", parent, -0.2)],
        inspirations=[_prog("i0", parent, -0.3)],
        previous_programs=[{"id": "p0", "metrics": {"combined_score": -0.1}, "metadata": {"changes": "x"}}],
        artifacts={"stdout": "ok"},
    )
    shrunk, steps = _minimal_shrink_user_message(
        prompt["user"], cap=1, system_message=prompt["system"], reserved=1024
    )
    assert any(s.startswith("guard_drop_") for s in steps)
    assert "# API" in shrunk
    assert "dict[int, float]" in shrunk
    assert "K=8" in shrunk
    assert "def choose(problem, history)" in shrunk


def test_sixty_examples_are_kept_when_they_fit():
    _require_qwen()
    parent = resolve_openevolve_seed_and_prompt("1peterson2021using")[0].read_text(
        encoding="utf-8"
    )
    prompt, state = _pack("1peterson2021using", parent)
    assert state.prompt_trial_count == 60
    assert state.examples_included == 60
    assert not any(s.startswith("cap_prompt_trials_") for s in state.steps)
    n_ex = prompt["user"].count("split=")
    assert n_ex == 60


def test_optional_blocks_dropped_before_reducing_sixty_examples():
    _require_qwen()
    huge = _max_code(ICLR_FROZEN_MAX_PROGRAM_CHARS)
    parent = _max_code(ICLR_FROZEN_MAX_PROGRAM_CHARS)
    for alias in (
        "1peterson2021using",
        "bergert_nosofsky_2007",
        "steyvers_2009_bandit",
        "13schulz2020finding",
    ):
        top = [_prog(f"t{i}", huge, -0.1 * i) for i in range(3)]
        diverse = [_prog(f"d{i}", huge, -0.5 - 0.1 * i) for i in range(2)]
        insp = [_prog(f"i{i}", huge, -0.8 - 0.1 * i) for i in range(2)]
        prompt, state = _pack(alias, parent, top=top, diverse=diverse, inspirations=insp, n_examples=60)
        n = chat_input_token_count(prompt["system"], prompt["user"])
        assert n <= ICLR_FROZEN_INPUT_TOKEN_CEILING
        assert state.prompt_trial_count == 60
        assert not any(s.startswith("cap_prompt_trials_") for s in state.steps)
        assert any(s.startswith("drop_") for s in state.steps)
        assert "# API" in prompt["user"]
        stripped = _drop_inspiration_section(prompt["user"])
        assert stripped.count("split=") == 60


def test_required_content_overflow_fails_fast():
    _require_qwen()
    huge = _max_code(ICLR_FROZEN_MAX_PROGRAM_CHARS)
    with pytest.raises(RequiredPromptOverflowError, match="required system/task/interface/parent"):
        _pack(
            "1peterson2021using",
            huge,
            n_examples=0,
            max_prompt_tokens=80,
        )


def test_kool_overshoot_41_examples_are_not_hard_capped_to_40():
    _require_qwen()
    parent = resolve_openevolve_seed_and_prompt("14kool2016when")[0].read_text(
        encoding="utf-8"
    )
    prompt, state = _pack("14kool2016when", parent, n_examples=41)
    assert state.prompt_trial_count == 41
    assert not any(s.startswith("cap_prompt_trials_") for s in state.steps)
    assert prompt["user"].count("split=") == 41
