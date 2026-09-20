"""ICLR T-PICS v2: training-only SA40, original test histories, snapshots, packing."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "baseline_methods" / "Psych101"))

from data_modules.psych101_binary import format_trial_for_prompt
from utils.teh.limited_data_protocol import (
    apply_structure_aware_protocol,
    load_participant_limited_splits,
    load_raw_participant_splits,
    _contiguous_tv_suffix,
)
from utils.teh.limited_data_registry import (
    CONTINUOUS_SESSION,
    INDEPENDENT_TRIAL,
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2,
    LIMITED_DATA_REGISTRY,
    RESETTING_UNIT,
)
from utils.teh.prompt_context import _trial_to_example_dict, DEFAULT_HISTORY_MAX_ENTRIES
from utils.teh.prompt_snapshots import (
    current_or_future_leak_paths,
    format_snapshot_example,
    prompt_contract_scope,
    sanitize_problem_for_choose,
    stamp_prompt_participant_id,
)
from utils.teh.prompt_units import (
    largest_balanced_prefix_that_fits,
    select_structure_aware_prompt_examples,
)
from utils.teh.t_pics_gated_transfer import EVOLUTION_SELECTION_SCORE, GATE_SCORE_FIELD, KIND
from utils.teh.t_pics_v2 import KIND_V2, V1_FROZEN_SOURCE_YAML, V1_G1_JOBS, V2_SOURCE_YAML

SPLIT = dict(split_ratio=0.6, split_seed=0, psych_dataset_split="train")


def _hist(t):
    return json.dumps(t.get("history") or [], sort_keys=True, default=str)


def test_v1_kool_suffix_still_retains_41():
    tv = []
    for i in range(80):
        stage = 2 if i % 2 == 0 else 1
        tv.append({"problem": {"stage": stage, "presented_day": i // 20}, "history": [], "action": 0})
    kept, reason = _contiguous_tv_suffix(tv, 40, kool=True)
    assert len(kept) == 41
    assert reason == "kool_include_matching_stage1"


def test_v2_kool_suffix_is_exact_40():
    tv = []
    for i in range(80):
        stage = 2 if i % 2 == 0 else 1
        tv.append({"problem": {"stage": stage, "presented_day": i // 20}, "history": [], "action": 0})
    kept, reason = _contiguous_tv_suffix(tv, 40, kool=True, exact_40=True)
    assert len(kept) == 40
    assert reason == ""
    assert int(kept[0]["problem"]["stage"]) == 2


def test_v2_kool_protocol_never_exceeds_40():
    tv = []
    for i in range(80):
        stage = 2 if i % 2 == 0 else 1
        tv.append(
            {
                "problem": {
                    "dataset_alias": "14kool2016when",
                    "schema_type": "kool_twostep",
                    "stage": stage,
                    "presented_day": i // 20,
                    "option_keys": [0, 1],
                },
                "history": [],
                "action": 0,
            }
        )
    train, val, test = tv[:48], tv[48:70], tv[70:]
    new_train, new_val, new_test, _a, man = apply_structure_aware_protocol(
        train,
        val,
        test,
        dataset="14kool2016when",
        participant_id=0,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_chronological_days",
        revision="v2",
    )
    assert len(new_train) + len(new_val) == 40
    assert "kool_include_matching_stage1" not in (man.fallback_reason or "")
    assert man.protocol == LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2
    assert len(new_test) == len(test)


def test_v1_artifacts_untouched():
    assert V1_FROZEN_SOURCE_YAML.is_file()
    assert KIND_V2 != "t_pics_gated"
    assert 257174 in V1_G1_JOBS and 257188 in V1_G1_JOBS


def test_cli_and_gated_defaults_are_v2():
    """Preliminary v2 kinds/YAML remain frozen; CLI default is now PICS v3."""
    import argparse

    from utils.teh.limited_data_protocol import add_limited_data_cli_arguments
    from utils.teh.pics_v3 import KIND as KIND_PICS_V3, SOURCE_YAML as PICS_V3_SOURCE_YAML
    from utils.teh.t_pics_gated_transfer import (
        FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG,
        FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG_V1,
        FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG_V2,
        apply_gated_cli_defaults,
    )

    parser = argparse.ArgumentParser()
    add_limited_data_cli_arguments(parser)
    args = parser.parse_args([])
    assert args.limited_data_protocol == "structure_aware_v3"
    assert args.limited_train_val == 40
    assert KIND == KIND_PICS_V3
    assert KIND_V2 == "t_pics_gated_sa40_v2"
    assert FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG == PICS_V3_SOURCE_YAML
    assert FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG_V2 == V2_SOURCE_YAML
    assert FROZEN_T_PICS_TRANSFER_SOURCE_CONFIG_V1 == V1_FROZEN_SOURCE_YAML
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
    assert ns.limited_data_protocol == "structure_aware_v3"
    assert ns.limited_train_val == 40
    assert ns.t_pics_source_config in {
        str(PICS_V3_SOURCE_YAML),
        str(V2_SOURCE_YAML),
    }
    if PICS_V3_SOURCE_YAML.is_file():
        assert ns.t_pics_source_config == str(PICS_V3_SOURCE_YAML)
    else:
        assert ns.t_pics_source_config == str(V2_SOURCE_YAML)
    # Preliminary v2 map remains frozen on disk.
    assert V2_SOURCE_YAML.is_file()
    assert V1_FROZEN_SOURCE_YAML.is_file()
    assert V1_FROZEN_SOURCE_YAML != V2_SOURCE_YAML
    assert KIND_V2 != KIND


def test_gate_independent_of_test():
    assert EVOLUTION_SELECTION_SCORE == "train_val"
    assert GATE_SCORE_FIELD == "mean_train_val_loglik"


def test_badham_frey_v2_formatter():
    badham = {
        "problem": {
            "schema_type": "B",
            "rule_block_id": 3,
            "stimulus_features": {"f1": 1},
            "option_keys": ["A", "B"],
            "correct_category": "A",
        },
        "action": 0,
        "history": [],
    }
    v1 = format_trial_for_prompt(badham, 1, contract="v1")
    v2 = format_trial_for_prompt(badham, 1, contract="v2")
    assert "ratings_A" in v1
    assert "stimulus_features" in v2
    assert "correct_category" not in v2
    balloon = {
        "problem": {
            "schema_type": "D",
            "balloon_id": 7,
            "step_index": 2,
            "pump_count_before": 2,
            "accumulated_points_before": 10,
            "pump_key": "P",
            "stop_key": "S",
            "option_keys": ["P", "S"],
        },
        "action": 0,
        "history": [],
    }
    v1b = format_trial_for_prompt(balloon, 1, contract="v1")
    v2b = format_trial_for_prompt(balloon, 1, contract="v2")
    assert "round_id" in v1b
    assert "balloon_id=7" in v2b
    assert "0=pump" in v2b


def test_snapshot_and_auto_prompt_sanitizer_parity():
    trial = {
        "problem": {
            "dataset_alias": "12badham2017deficits",
            "schema_type": "B",
            "stimulus_features": {"shape": 1},
            "correct_category": "X",
            "response_key": "r",
            "option_keys": ["A", "B"],
        },
        "action": 1,
        "history": [{"action": 0, "feedback": 1}] * 12,
    }
    snap_prob = sanitize_problem_for_choose(trial["problem"])
    auto = _trial_to_example_dict(trial, 1, history_max_entries=DEFAULT_HISTORY_MAX_ENTRIES)
    assert "correct_category" not in snap_prob
    assert "correct_category" not in auto["problem"]
    assert auto["history_truncated"] is True
    assert auto["history_original_len"] == 12
    assert len(auto["history"]) == 8
    text = format_snapshot_example(trial, 1)
    assert "observed_action_label=1" in text
    assert current_or_future_leak_paths({"problem": snap_prob, "action": 1}) == []


def test_balanced_prefix_does_not_drop_later_participants_only():
    trials = []
    for pid in (0, 1, 2):
        for i in range(4):
            trials.append(
                stamp_prompt_participant_id(
                    {
                        "problem": {"dataset_alias": "7hilbig2014generalized", "option_keys": [0, 1]},
                        "action": 0,
                        "history": [],
                    },
                    pid,
                )
            )
    selected, _ = select_structure_aware_prompt_examples(
        trials,
        dataset="7hilbig2014generalized",
        max_trials=6,
        subsample_seed=0,
        pooled=True,
    )
    pids = [t["_prompt_participant_id"] for t in selected]
    assert set(pids) == {0, 1, 2}

    def tok(text: str) -> int:
        return text.count("### example")

    kept, diag = largest_balanced_prefix_that_fits(
        selected,
        required_prompt="REQ",
        token_count=tok,
        token_ceiling=3,
    )
    assert diag["n_retained"] == 3
    assert diag["n_participants_retained"] == 3
    assert set(t["_prompt_participant_id"] for t in kept) == {0, 1, 2}


def test_centaur_v2_timeline_unscored_context():
    from Centaur import _centaur_prompt_timeline_v2

    raw_tv = [{"problem": {"x": i}, "history": [], "action": 0} for i in range(10)]
    test = [{"problem": {"x": 99}, "history": [{}] * 10, "action": 1}]
    prompt, scores = _centaur_prompt_timeline_v2(raw_tv[:8], raw_tv[8:], test)
    assert scores == [10]
    assert len(prompt) == 11
    assert prompt[scores[0]]["problem"]["x"] == 99


def _one_pid(alias: str) -> int:
    if alias == "mixed_gambles":
        path = REPO / "datasets/mixed_gambles/valid_participant_ids.json"
        return int(json.loads(path.read_text())["valid_participant_ids"][0])
    if alias in LIMITED_DATA_REGISTRY and alias.startswith(
        ("bergert", "guan", "steyvers")
    ) or alias in {"bergert_nosofsky_2007", "guan_2020_stopping", "steyvers_2009_bandit"}:
        path = REPO / "datasets/external" / alias / "valid_participant_ids.json"
        return int(json.loads(path.read_text())["valid_participant_ids"][0])
    return 0


@pytest.mark.parametrize("alias", sorted(LIMITED_DATA_REGISTRY))
def test_v2_all_15_structure_categories_one_person(alias):
    spec = LIMITED_DATA_REGISTRY[alias]
    pid = _one_pid(alias)
    raw_tr, raw_va, raw_te, _ = load_raw_participant_splits(alias, pid, **SPLIT)
    v2_tr, v2_va, v2_te, _a, man = load_participant_limited_splits(
        alias,
        pid,
        **SPLIT,
        limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2,
        limited_train_val=40,
        max_observed_trials_per_participant=40,
    )
    assert man.protocol == LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2
    assert len(v2_tr) + len(v2_va) <= 40
    assert len(v2_te) == len(raw_te)
    assert [t.get("action") for t in v2_te] == [t.get("action") for t in raw_te]
    if spec.category == INDEPENDENT_TRIAL:
        for split_rows in (v2_tr, v2_va, v2_te):
            assert all(not (t.get("history") or []) for t in split_rows)
        assert man.extra.get("test_history_policy") == "independent_empty_all_splits"
    else:
        assert [_hist(a) for a in v2_te] == [_hist(b) for b in raw_te]
    for t in v2_te:
        assert current_or_future_leak_paths(t) == []
        san = sanitize_problem_for_choose(t.get("problem") or {})
        assert "probe_in_set" not in san
    assert spec.category in {INDEPENDENT_TRIAL, RESETTING_UNIT, CONTINUOUS_SESSION}


def test_v2_kool_41_and_45_exact_40_and_original_test():
    for pid in (41, 45):
        raw_tr, raw_va, raw_te, _ = load_raw_participant_splits("14kool2016when", pid, **SPLIT)
        v1_tr, v1_va, v1_te, _a1, m1 = load_participant_limited_splits(
            "14kool2016when",
            pid,
            **SPLIT,
            limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
            limited_train_val=40,
            max_observed_trials_per_participant=40,
        )
        v2_tr, v2_va, v2_te, _a2, m2 = load_participant_limited_splits(
            "14kool2016when",
            pid,
            **SPLIT,
            limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2,
            limited_train_val=40,
            max_observed_trials_per_participant=40,
        )
        assert len(v1_tr) + len(v1_va) == 41
        assert len(v2_tr) + len(v2_va) == 40
        assert int((v2_tr + v2_va)[0]["problem"]["stage"]) == 2
        assert [_hist(a) for a in v2_te] == [_hist(b) for b in raw_te]
        assert [_hist(a) for a in v1_te] != [_hist(b) for b in raw_te]
        last_v1 = (v1_tr + v1_va)[-1]
        last_v2 = (v2_tr + v2_va)[-1]
        assert last_v1["problem"]["presented_day"] == last_v2["problem"]["presented_day"]
        assert last_v1["problem"]["stage"] == last_v2["problem"]["stage"]


def test_v2_continuous_keeps_raw_test_history_independents_clear():
    # Continuous: restore original pre-choice test histories.
    raw_tr, raw_va, raw_te, _ = load_raw_participant_splits(
        "5speekenbrink2008learning", 0, **SPLIT
    )
    v1_tr, v1_va, v1_te, *_ = load_participant_limited_splits(
        "5speekenbrink2008learning",
        0,
        **SPLIT,
        limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
        limited_train_val=40,
        max_observed_trials_per_participant=40,
    )
    v2_tr, v2_va, v2_te, *_ = load_participant_limited_splits(
        "5speekenbrink2008learning",
        0,
        **SPLIT,
        limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2,
        limited_train_val=40,
        max_observed_trials_per_participant=40,
    )
    assert [_hist(a) for a in v2_te] == [_hist(b) for b in raw_te]
    assert len(raw_te[0].get("history") or []) > len(v1_te[0].get("history") or [])

    # Independents with loader-accumulated raw history: cleared under v1 and v2.
    for alias in ("7hilbig2014generalized", "11enkavi2019recentprobes"):
        raw_tr, raw_va, raw_te, _ = load_raw_participant_splits(alias, 0, **SPLIT)
        assert any(raw_te[i].get("history") for i in range(len(raw_te)))
        for proto in (
            LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
            LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE_V2,
        ):
            _tr, _va, te, *_rest = load_participant_limited_splits(
                alias,
                0,
                **SPLIT,
                limited_data_protocol=proto,
                limited_train_val=40,
                max_observed_trials_per_participant=40,
            )
            assert all(not (t.get("history") or []) for t in te)
            assert [t.get("action") for t in te] == [t.get("action") for t in raw_te]


def test_probe_in_set_never_reaches_choose_evaluator():
    """Shared sanitizer must strip Enkavi oracle from fitness/diagnostic choose()."""
    import teh

    seen = {"had_probe_in_set": False, "calls": 0}

    def choose(problem, history):
        seen["calls"] += 1
        if "probe_in_set" in problem:
            seen["had_probe_in_set"] = True
        mem = problem.get("memory_set_letters") or []
        probe = problem.get("probe_letter")
        return 0.9 if probe in mem else 0.1

    trials = [
        {
            "problem": {
                "dataset_alias": "11enkavi2019recentprobes",
                "memory_set_letters": ["A", "B"],
                "probe_letter": "A",
                "probe_in_set": True,
                "option_keys": [0, 1],
            },
            "history": [{"action": 1, "probe_in_set": False}],
            "action": 1,
        }
    ]
    out = teh.evaluate_choice13k_program(choose, trials, n_seeds=1)
    assert seen["calls"] == 1
    assert seen["had_probe_in_set"] is False
    assert out["avg_loglik"] > -0.2
    # Prompt sanitizer matches evaluator contract.
    assert "probe_in_set" not in sanitize_problem_for_choose(trials[0]["problem"])


def test_v1_independent_history_policy_unchanged():
    """v1 still empties independent test history; continuous v1 still rebuilds."""
    for alias in ("11enkavi2019recentprobes", "7hilbig2014generalized", "4wulff2018description"):
        _tr, _va, te, _a, man = load_participant_limited_splits(
            alias,
            0 if not alias.startswith("bergert") else _one_pid(alias),
            **SPLIT,
            limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
            limited_train_val=40,
            max_observed_trials_per_participant=40,
        )
        assert all(not (t.get("history") or []) for t in te)
        assert man.protocol == LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE
    v1_tr, v1_va, v1_te, *_ = load_participant_limited_splits(
        "5speekenbrink2008learning",
        0,
        **SPLIT,
        limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
        limited_train_val=40,
        max_observed_trials_per_participant=40,
    )
    raw_tr, raw_va, raw_te, _ = load_raw_participant_splits(
        "5speekenbrink2008learning", 0, **SPLIT
    )
    assert [_hist(a) for a in v1_te] != [_hist(b) for b in raw_te]
    assert any(v1_te[i].get("history") for i in range(len(v1_te)))


def test_oe_combined_score_excludes_test():
    from run_openevolve import _find_best_program_by_observed_loglik, _render_evaluator_py

    src = inspect.getsource(_render_evaluator_py)
    assert "combined_score" in src
    assert "observed-union objective; no test access" in src
    assert "never test" in inspect.getdoc(_find_best_program_by_observed_loglik)


def test_lm_fit_excludes_test():
    from baseline_methods.MLE import _fit_and_evaluate_participant

    src = inspect.getsource(_fit_and_evaluate_participant)
    assert "fit_trials = train_trials + val_trials" in src
    assert "test_trials" in src
    assert "loglik_fn(test_trials)" in src


def test_v2_estimate_tokens_uses_qwen_not_char4(monkeypatch):
    import teh

    monkeypatch.setattr(
        "utils.teh.prompt_units.qwen_user_prompt_token_count",
        lambda user: 12345,
    )
    with prompt_contract_scope(True):
        assert teh.estimate_tokens("abcd" * 100) == 12345
    with prompt_contract_scope(False):
        assert teh.estimate_tokens("abcd" * 100) == 100


def test_v2_truncation_drops_trial_count_keeps_snapshots(monkeypatch):
    import teh

    def fake_estimate(text, *, estimator="char4"):
        n = text.count("### example")
        return 20000 if n > 30 else 1000

    monkeypatch.setattr(teh, "estimate_tokens", fake_estimate)
    trials = []
    for i in range(80):
        trials.append(
            stamp_prompt_participant_id(
                {
                    "problem": {
                        "dataset_alias": "7hilbig2014generalized",
                        "option_keys": [0, 1],
                    },
                    "action": 0,
                    "history": [{"action": 0, "reward": 1}],
                    "split": "train",
                },
                i % 10,
            )
        )
    with prompt_contract_scope(True):
        prompt, diag, steps = teh._truncate_psych_prompt_to_budget(
            base_prompt="TASK",
            train_trials=trials[:60],
            train_trials_source=trials,
            val_trials=None,
            val_trials_source=None,
            extra_prompt_trials_label="validation",
            parent_programs=["def choose(problem, history):\n    return 0.5\n"],
            parent_context_builder=lambda prompt_parent_programs, **_: "PARENT",
            parent_context_kwargs={},
            code_template_suffix="TEMPLATE",
            candidate_output_rules="RULES",
            dataset="7hilbig2014generalized",
            dataset_type="7hilbig2014generalized",
            hard_prompt_token_cap=14000,
            prompt_token_estimator="char4",
            max_prompt_train_trials=60,
            max_prompt_trials_per_problem=5,
            prompt_train_trials_seed=0,
            max_parent_chars=3500,
            refinement_val_observations=False,
            pre_capped_train=False,
            pre_capped_val=False,
        )
    assert diag["train_trials_after"] == 30
    assert "compact_trial_serialization" not in steps
    assert diag.get("compact_serialization") is False
    assert "### example" in prompt
    assert "observed_action_label" in prompt
    assert "train_trials_cap_30" in steps


def test_qwen_budget_count_is_chat_templated_system_plus_user(monkeypatch):
    captured: dict = {}

    class _FakeTok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            captured["messages"] = messages
            captured["add_generation_prompt"] = add_generation_prompt
            return "TEMPLATED"

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3, 4]

    from utils.teh import prompt_units as pu

    monkeypatch.setattr(pu, "_ensure_qwen_tokenizer", lambda: _FakeTok())
    n = pu.qwen_user_prompt_token_count("hello")
    assert n == 4
    assert captured["add_generation_prompt"] is True
    assert captured["messages"][0] == {
        "role": "system",
        "content": pu.QWEN_DEFAULT_SYSTEM,
    }
    assert captured["messages"][1] == {"role": "user", "content": "hello"}


def test_enkavi_schema_and_examples_omit_probe_in_set():
    """Generation schema/examples must not teach the membership oracle field name."""
    from utils.teh.prompt_context import (
        assert_no_generation_oracle_leak,
        build_deterministic_runtime_contract,
        infer_recursive_runtime_schema,
        serialize_train_trials_for_prompt_generation,
    )
    from utils.teh.teh_runtime import (
        _merge_prompt_fallback,
        _prompt_trial_fingerprint,
        build_prompt_generation_llm_user_content,
    )

    trials = [
        {
            "problem": {
                "dataset_alias": "11enkavi2019recentprobes",
                "schema_type": "B",
                "memory_set_letters": ["A", "B", "C"],
                "probe_letter": "A",
                "probe_in_set": True,
                "option_keys": ["Q", "C"],
                "yes_key": "C",
                "no_key": "Q",
                "key_mapping": {"no": "Q", "yes": "C"},
                "task": "recent_probes_memory",
            },
            "history": [],
            "action": 1,
        },
        {
            "problem": {
                "dataset_alias": "11enkavi2019recentprobes",
                "schema_type": "B",
                "memory_set_letters": ["X", "Y", "Z"],
                "probe_letter": "W",
                "probe_in_set": False,
                "option_keys": ["Q", "C"],
                "yes_key": "C",
                "no_key": "Q",
                "key_mapping": {"no": "Q", "yes": "C"},
                "task": "recent_probes_memory",
            },
            "history": [],
            "action": 0,
        },
    ]
    schema = infer_recursive_runtime_schema(trials)
    assert "probe_in_set" not in schema
    examples = serialize_train_trials_for_prompt_generation(trials)
    assert "probe_in_set" not in examples
    assert "observed_action_label" in examples
    assert '"probe_in_set"' not in examples
    # choose() input key must not appear; label is observed_action_label only.
    assert '"action":' not in examples
    contract = build_deterministic_runtime_contract(trials)
    assert "probe_in_set" not in contract
    fp = _prompt_trial_fingerprint(trials[0])
    assert "probe_in_set" not in fp
    llm_user = build_prompt_generation_llm_user_content(
        "11enkavi2019recentprobes",
        "Remember letters; say if probe was among them.",
        trials,
    )
    assert_no_generation_oracle_leak(llm_user, context="llm_user_test")
    merged = _merge_prompt_fallback(
        "11enkavi2019recentprobes",
        "Remember letters; say if probe was among them.",
        trials,
    )
    assert_no_generation_oracle_leak(merged, context="merge_fallback_test")
    ex0 = _trial_to_example_dict(trials[0], 1, history_max_entries=DEFAULT_HISTORY_MAX_ENTRIES)
    assert "probe_in_set" not in (ex0.get("problem") or {})
    assert "action" not in ex0
    assert ex0.get("observed_action_label") == 1


def test_probe_in_set_keyerror_is_invalid_not_chance_fitness():
    """Direct indexing of missing oracle must invalidate; no usable chance LL."""
    import teh

    def choose(problem, history):
        if problem["probe_in_set"]:
            return 0.9
        return 0.1

    trials = [
        {
            "problem": {
                "dataset_alias": "11enkavi2019recentprobes",
                "memory_set_letters": ["A"],
                "probe_letter": "A",
                "probe_in_set": True,
                "option_keys": [0, 1],
            },
            "history": [],
            "action": 1,
        }
    ]
    out = teh.evaluate_choice13k_program(choose, trials, n_seeds=1)
    assert out["errors"] >= 1
    assert out["avg_loglik"] == float("-inf")
    # first_error is populated when source is attached; fitness invalidation is the contract.


def test_cursor_designed_enkavi_prompt_has_no_probe_in_set():
    path = REPO / "prompts/teh/cursor_designed/11enkavi2019recentprobes.txt"
    text = path.read_text(encoding="utf-8")
    assert "probe_in_set" not in text
    assert "memory_set_letters" in text
    assert "probe_letter" in text
