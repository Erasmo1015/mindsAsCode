"""ICLR PICS v3: structure_aware_v3, 30k/5k context, parent cap, Enkavi oracle."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "baseline_methods" / "Psych101"))

from utils.teh.limited_data_protocol import (
    load_participant_limited_splits,
    load_raw_participant_splits,
)
from utils.teh.limited_data_registry import (
    CONTINUOUS_SESSION,
    INDEPENDENT_TRIAL,
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2,
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3,
    LIMITED_DATA_REGISTRY,
    uses_training_only_sa40,
)
from utils.teh.pics_v3 import (
    G1_KIND,
    HARD_PROMPT_TOKEN_CAP,
    INDEPENDENT_KIND,
    KIND,
    LIMITED_DATA_PROTOCOL,
    MAX_PARENT_CHARS,
    PRELIMINARY_V2_HARD_PROMPT_TOKEN_CAP,
    PRELIMINARY_V2_MAX_PARENT_CHARS,
    SOURCE_YAML,
    VLLM_MAX_MODEL_LEN,
)
from utils.teh.prompt_snapshots import (
    current_or_future_leak_paths,
    sanitize_problem_for_choose,
)
from utils.teh.t_pics_gated_transfer import (
    FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG,
    FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG_V1,
    FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG_V2,
    KIND as GATED_KIND,
    apply_gated_cli_defaults,
)
from utils.teh.t_pics_v2 import KIND_V2, V1_FROZEN_SOURCE_YAML, V2_SOURCE_YAML

SPLIT = dict(split_ratio=0.6, split_seed=0, psych_dataset_split="train")


def _hist(t):
    return json.dumps(t.get("history") or [], sort_keys=True, default=str)


def _one_pid(alias: str) -> int:
    if alias == "mixed_gambles":
        path = REPO / "datasets/mixed_gambles/valid_participant_ids.json"
        return int(json.loads(path.read_text())["valid_participant_ids"][0])
    if alias in {"bergert_nosofsky_2007", "guan_2020_stopping", "steyvers_2009_bandit"}:
        path = REPO / "datasets/external" / alias / "valid_participant_ids.json"
        return int(json.loads(path.read_text())["valid_participant_ids"][0])
    return 0


def test_pics_v3_constants_and_defaults():
    from utils.teh.limited_data_protocol import add_limited_data_cli_arguments

    assert KIND == "pics_v3"
    assert G1_KIND == "pics_v3_g1"
    assert INDEPENDENT_KIND == "pics_v3_independent"
    assert LIMITED_DATA_PROTOCOL == "structure_aware_v3"
    assert HARD_PROMPT_TOKEN_CAP == 30_000
    assert MAX_PARENT_CHARS == 5_000
    assert VLLM_MAX_MODEL_LEN == 32_768
    assert PRELIMINARY_V2_HARD_PROMPT_TOKEN_CAP == 14_000
    assert PRELIMINARY_V2_MAX_PARENT_CHARS == 3_500
    assert GATED_KIND == KIND
    assert FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG == SOURCE_YAML
    assert FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG_V2 == V2_SOURCE_YAML
    assert FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG_V1 == V1_FROZEN_SOURCE_YAML
    assert KIND_V2 != KIND
    assert V1_FROZEN_SOURCE_YAML.is_file()
    assert V2_SOURCE_YAML.is_file()

    parser = argparse.ArgumentParser()
    add_limited_data_cli_arguments(parser)
    args = parser.parse_args([])
    assert args.limited_data_protocol == LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3
    ns = argparse.Namespace(
        t_pics_source_config=None,
        limited_data_protocol="off",
        limited_train_val=None,
        hard_prompt_token_cap=14000,
        max_parent_chars=3500,
        llm_max_tokens=800,
        max_prompt_train_trials=40,
        n_iterations=1,
        global_iters=1,
        explore_candidates=1,
    )
    apply_gated_cli_defaults(ns, argv=["--t_pics_gated_transfer"])
    assert ns.limited_data_protocol == LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3
    assert ns.hard_prompt_token_cap == 30_000
    assert ns.max_parent_chars == 5_000
    assert ns.llm_max_tokens == 1024
    assert ns.max_prompt_train_trials == 60
    if SOURCE_YAML.is_file():
        assert ns.t_pics_source_config == str(SOURCE_YAML)
    else:
        assert ns.t_pics_source_config == str(V2_SOURCE_YAML)


def test_teh_cli_pics_v3_token_defaults():
    from utils.teh.limited_data_protocol import add_limited_data_cli_arguments

    p = argparse.ArgumentParser()
    add_limited_data_cli_arguments(p)
    p.add_argument("--hard_prompt_token_cap", type=int, default=HARD_PROMPT_TOKEN_CAP)
    p.add_argument("--max_parent_chars", type=int, default=MAX_PARENT_CHARS)
    p.add_argument("--llm_max_tokens", type=int, default=1024)
    p.add_argument("--max_prompt_train_trials", type=int, default=60)
    args = p.parse_args([])
    assert args.limited_data_protocol == "structure_aware_v3"
    assert args.hard_prompt_token_cap == 30_000
    assert args.max_parent_chars == 5_000
    assert args.llm_max_tokens == 1024
    assert args.max_prompt_train_trials == 60


def test_v1_and_preliminary_v2_protocols_still_selectable():
    assert uses_training_only_sa40("structure_aware_v2")
    assert uses_training_only_sa40("structure_aware_v3")
    assert not uses_training_only_sa40("structure_aware")
    pid = _one_pid("11enkavi2019recentprobes")
    train, val, test, _a, man = load_participant_limited_splits(
        "11enkavi2019recentprobes",
        pid,
        limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2,
        limited_train_val=40,
        max_observed_trials_per_participant=40,
        **SPLIT,
    )
    assert man.protocol == LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2
    assert len(train) + len(val) <= 40
    train3, val3, test3, _a3, man3 = load_participant_limited_splits(
        "11enkavi2019recentprobes",
        pid,
        limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3,
        limited_train_val=40,
        max_observed_trials_per_participant=40,
        **SPLIT,
    )
    assert man3.protocol == LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3
    assert len(train3) + len(val3) == len(train) + len(val)
    assert [t["action"] for t in test3] == [t["action"] for t in test]


@pytest.mark.parametrize("alias", sorted(LIMITED_DATA_REGISTRY))
def test_structure_aware_v3_train_val_cap_and_test_isolation(alias):
    spec = LIMITED_DATA_REGISTRY[alias]
    ordinal = _one_pid(alias)
    train, val, test, _audit, man = load_participant_limited_splits(
        alias,
        ordinal,
        limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3,
        limited_train_val=40,
        max_observed_trials_per_participant=40,
        **SPLIT,
    )
    assert man.protocol == LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3
    assert len(train) + len(val) <= 40
    _raw_train, _raw_val, raw_test, _ = load_raw_participant_splits(
        alias, ordinal, **SPLIT
    )
    assert [t.get("action") for t in test] == [t.get("action") for t in raw_test]
    if spec.category == INDEPENDENT_TRIAL:
        for split in (train, val, test):
            for t in split:
                assert (t.get("history") or []) == []
    if spec.category == CONTINUOUS_SESSION:
        assert [_hist(a) for a in test] == [_hist(b) for b in raw_test]


def test_kool_exact_40_under_v3():
    for ordinal in (41, 45, 0):
        train, val, _test, _a, man = load_participant_limited_splits(
            "14kool2016when",
            ordinal,
            limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V3,
            limited_train_val=40,
            max_observed_trials_per_participant=40,
            **SPLIT,
        )
        assert len(train) + len(val) == 40
        assert "kool_include_matching_stage1" not in (man.fallback_reason or "")


def test_parent_char_cap_preserves_kool_sized_and_compacts_oversized():
    from teh import _truncate_parent_program_for_prompt

    line = "    x = 1  # pad\n"
    header = "def choose(problem, history):\n"
    n = 0
    while len(header) + n * len(line) < 3900:
        n += 1
    koolish = header + line * n
    assert 3800 < len(koolish) < 5000
    kept, clipped = _truncate_parent_program_for_prompt(koolish, MAX_PARENT_CHARS)
    assert clipped is False
    assert kept == koolish

    oversized = header + ("    y = 2  # pad\n" * 800)

    assert len(oversized) > 5000
    compact, clipped = _truncate_parent_program_for_prompt(oversized, MAX_PARENT_CHARS)
    assert clipped is True
    assert len(compact) <= MAX_PARENT_CHARS
    assert "# truncated; keep concise" in compact
    assert len(oversized) > MAX_PARENT_CHARS


def test_preliminary_v2_parent_cap_unchanged():
    from teh import _truncate_parent_program_for_prompt

    code = "a" * 4000
    _, clipped_v2 = _truncate_parent_program_for_prompt(
        code, PRELIMINARY_V2_MAX_PARENT_CHARS
    )
    assert clipped_v2 is True
    _, clipped_v3 = _truncate_parent_program_for_prompt(code, MAX_PARENT_CHARS)
    assert clipped_v3 is False


def test_enkavi_no_probe_in_set_in_prompts_and_sanitize():
    path = REPO / "prompts/teh/cursor_designed/11enkavi2019recentprobes.txt"
    text = path.read_text(encoding="utf-8")
    assert "probe_in_set" not in text
    problem = {
        "memory_set_letters": ["A", "B"],
        "probe_letter": "A",
        "probe_in_set": True,
        "option_keys": ["n", "y"],
    }
    cleaned = sanitize_problem_for_choose(problem)
    assert "probe_in_set" not in cleaned
    assert current_or_future_leak_paths(
        {"problem": cleaned, "history": [], "action": 1}
    ) == []


def test_openevolve_shares_v3_context_not_pics_kind():
    from run_openevolve import (
        ICLR_DEFAULT_LIMITED_DATA_PROTOCOL,
        ICLR_FROZEN_INPUT_TOKEN_CEILING,
        ICLR_FROZEN_LLM_MAX_TOKENS,
        ICLR_V2_LIMITED_DATA_PROTOCOL,
        build_arg_parser,
    )

    args = build_arg_parser().parse_args([])
    assert ICLR_DEFAULT_LIMITED_DATA_PROTOCOL == "structure_aware_v3"
    assert ICLR_V2_LIMITED_DATA_PROTOCOL == "structure_aware_v2"
    assert args.limited_data_protocol == "structure_aware_v3"
    assert args.hard_prompt_token_cap == 30_000 == ICLR_FROZEN_INPUT_TOKEN_CEILING
    assert args.llm_max_tokens == ICLR_FROZEN_LLM_MAX_TOKENS == 1024
    from run_openevolve import ICLR_FROZEN_VLLM_MAX_MODEL_LEN

    assert args.max_model_len == 32_768 == ICLR_FROZEN_VLLM_MAX_MODEL_LEN
    assert KIND == "pics_v3"
    src = inspect.getsource(build_arg_parser)
    assert "pics_v3" not in src
