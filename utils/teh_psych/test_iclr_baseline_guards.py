"""ICLR Centaur/OpenEvolve guards: no empty generic prefix, no target leak, SA40 isolation."""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
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

from Centaur import (  # noqa: E402
    _build_generic_prefix,
    _centaur_prompt_timeline_v2,
    _ensure_mixed_gambles_centaur_contract,
    _prepare_mixed_gambles_centaur_trials,
    build_centaur_prompt_prefix_indexed,
    centaur_display_keys,
)
from run_openevolve import (  # noqa: E402
    CHOOSE_API_BERNOULLI,
    CHOOSE_API_CATEGORICAL,
    DEFAULT_CATEGORICAL_SEED_PATH,
    DEFAULT_SEED_PATH,
    EXPECTED_OPENEVOLVE_GIT_SHA,
    FAILED_COMBINED_SCORE,
    ICLR_FROZEN_LIMITED_DATA_PROTOCOL,
    ICLR_DEFAULT_LIMITED_DATA_PROTOCOL,
    ICLR_FROZEN_LIMITED_TRAIN_VAL,
    ICLR_FROZEN_LLM_MAX_TOKENS,
    ICLR_FROZEN_MAX_PROMPT_TRAIN_TRIALS,
    ICLR_FROZEN_MODEL,
    ICLR_FROZEN_N_ITERATIONS,
    ICLR_FROZEN_NUM_TOP_PROGRAMS,
    ICLR_FROZEN_INPUT_TOKEN_CEILING,
    ICLR_FROZEN_PARALLEL_EVALUATIONS,
    ICLR_FROZEN_PARALLEL_PARTICIPANTS,
    ICLR_FROZEN_SPLIT_RATIO,
    ICLR_FROZEN_SPLIT_SEED,
    VanillaProcessParallelController,
    _LEGACY_VANILLA_TASK_FILES,
    _patched_build_prompt,
    _adapt_official_failure_metrics_for_loglik,
    _render_evaluator_py,
    _write_evolution_split_json,
    _write_posthoc_test_json,
    apply_iclr_frozen_range_ordinals,
    build_arg_parser,
    cap_and_subsample_prompt_trials,
    choose_api_text,
    dataset_n_actions,
    format_trial_compact,
    iclr_frozen_argv,
    openevolve_checkout_sha,
    require_openevolve_checkout,
    resolve_openevolve_seed_and_prompt,
    run_participant,
    sanitize_trial_for_program,
    split_official_optional_programs,
    trials_for_participant,
    vanilla_dataset_description,
    vanilla_llm_user_prefix,
)
from utils.teh.limited_data_protocol import (  # noqa: E402
    apply_structure_aware_protocol,
    select_complete_then_prefix,
    _contiguous_tv_suffix,
)
from utils.teh.limited_data_registry import (  # noqa: E402
    CONTINUOUS_SESSION,
    INDEPENDENT_TRIAL,
    LIMITED_DATA_REGISTRY,
    RESETTING_UNIT,
)
from utils.teh.teh_datasets import (  # noqa: E402
    PARTICIPANT_DATASETS,
    dataset_task_description,
    emnlp_ordinal_range,
    is_categorical_output_dataset,
)
from data_modules.mixed_gambles import TASK_DESCRIPTION as MIXED_GAMBLES_TASK_DESCRIPTION  # noqa: E402
from data_modules.psych101_binary import PSYCH101_BINARY_DATASETS  # noqa: E402

ICLR_15 = (
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "5speekenbrink2008learning",
    "7hilbig2014generalized",
    "10frey2017risk",
    "11enkavi2019recentprobes",
    "12badham2017deficits",
    "mixed_gambles",
    "bergert_nosofsky_2007",
    "guan_2020_stopping",
    "steyvers_2009_bandit",
    "13schulz2020finding",
    "14kool2016when",
)


def test_iclr15_are_openevolve_participant_datasets():
    missing = [a for a in ICLR_15 if a not in PARTICIPANT_DATASETS]
    assert missing == []


def test_supported_dataset_refuses_empty_generic_prefix():
    trial = {
        "problem": {
            "dataset_alias": "mixed_gambles",
            "option_keys": [0, 1],
        },
        "history": [],
        "action": 0,
    }
    with pytest.raises(ValueError, match="refused empty generic prefix"):
        build_centaur_prompt_prefix_indexed([trial], 0, instruction="")


def test_unknown_dataset_may_use_generic_prefix():
    trial = {
        "problem": {"dataset_alias": "not_a_teh_dataset", "option_keys": [0, 1]},
        "history": [],
        "action": 0,
    }
    prefix = _build_generic_prefix([trial], 0, instruction="")
    assert prefix.strip() == "You press"
    assert build_centaur_prompt_prefix_indexed([trial], 0, instruction="").strip() == "You press"


def test_mixed_gambles_numeric_keys_fail_fast_even_with_gamble_fields():
    trial = {
        "problem": {
            "dataset_alias": "mixed_gambles",
            "option_keys": [0, 1],
            "gamble_A": {"rewards": [4.0, -1.0], "probs": [0.5, 0.5]},
            "gamble_B": {"rewards": [0.0], "probs": [1.0]},
        },
        "history": [],
        "action": 0,
    }
    with pytest.raises(ValueError, match="display keys"):
        build_centaur_prompt_prefix_indexed([trial], 0, instruction="")


def test_mixed_gambles_prepared_prefix_is_ab_gamble_text():
    raw = {
        "problem": {
            "dataset_alias": "mixed_gambles",
            "option_keys": [0, 1],
            "gamble_A": {"rewards": [4.0, -1.0], "probs": [0.5, 0.5]},
            "gamble_B": {"rewards": [0.0], "probs": [1.0]},
        },
        "history": [],
        "action": 0,
    }
    prepared = _prepare_mixed_gambles_centaur_trials([raw])[0]
    assert centaur_display_keys(prepared["problem"]) == ["A", "B"]
    prefix = build_centaur_prompt_prefix_indexed([prepared], 0)
    assert "Option A delivers" in prefix
    assert "Option B delivers" in prefix


def test_mixed_gambles_v2_raw_prefix_is_remapped_to_ab():
    raw = {
        "problem": {
            "dataset_alias": "mixed_gambles",
            "option_keys": [0, 1],
            "gamble_A": {"rewards": [4.0, -1.0], "probs": [0.5, 0.5]},
            "gamble_B": {"rewards": [0.0], "probs": [1.0]},
        },
        "history": [],
        "action": 0,
    }
    test = _prepare_mixed_gambles_centaur_trials([dict(raw)])
    prompt, scores = _centaur_prompt_timeline_v2([raw] * 3, [raw] * 2, test)
    prompt = _ensure_mixed_gambles_centaur_contract("mixed_gambles", prompt)
    assert scores == [5]
    assert all(centaur_display_keys(t["problem"]) == ["A", "B"] for t in prompt)
    prefix = build_centaur_prompt_prefix_indexed(prompt, scores[0])
    assert "Option A delivers" in prefix
    assert "<<0>>" not in prefix


def test_enkavi_probe_in_set_is_oracle_and_stripped_from_program_input():
    trial = {
        "problem": {
            "dataset_alias": "11enkavi2019recentprobes",
            "schema_type": "B",
            "memory_set_letters": ["A", "C", "F"],
            "probe_letter": "C",
            "probe_in_set": True,
            "option_keys": ["L", "N"],
        },
        "history": [],
        "action": 1,
    }
    compact = format_trial_compact(trial, "train")
    assert "memory_set=" in compact
    assert "probe=C" in compact
    assert "in_set" not in compact
    assert "probe_in_set" not in compact
    assert "y=1" in compact
    cleaned = sanitize_trial_for_program(trial)
    assert "probe_in_set" not in cleaned["problem"]
    assert cleaned["problem"]["memory_set_letters"] == ["A", "C", "F"]
    assert cleaned["problem"]["probe_letter"] == "C"
    assert trial["problem"]["probe_in_set"] is True


def test_kool_stage1_drops_unobserved_planet_s1_reward():
    trial = {
        "problem": {
            "dataset_alias": "14kool2016when",
            "schema_type": "kool_twostep",
            "stage": 1,
            "presented_day": 3,
            "option_keys": ["C", "D"],
            "spaceship_options": ["C", "D"],
            "planet": "J",
            "alien_options": ["P", "Q"],
            "stage1_action": 0,
            "spaceship": "C",
            "reward": 1,
            "treasure": 1,
        },
        "history": [],
        "action": 0,
    }
    cleaned = sanitize_trial_for_program(trial)
    p = cleaned["problem"]
    assert p["stage"] == 1
    assert p["spaceship_options"] == ["C", "D"]
    for key in ("planet", "alien_options", "stage1_action", "spaceship", "reward", "treasure"):
        assert key not in p
    compact = format_trial_compact(trial, "train")
    assert "planet=J" not in compact
    assert "s1=" not in compact
    assert "aliens=" not in compact


def test_kool_stage2_keeps_observed_planet_s1_not_reward():
    trial = {
        "problem": {
            "dataset_alias": "14kool2016when",
            "schema_type": "kool_twostep",
            "stage": 2,
            "presented_day": 3,
            "option_keys": ["P", "Q"],
            "planet": "J",
            "alien_options": ["P", "Q"],
            "stage1_action": 0,
            "spaceship": "C",
            "reward": 1,
            "treasure": 1,
        },
        "history": [
            {
                "stage": 1,
                "action": 0,
                "option_keys": ["C", "D"],
                "spaceship": "C",
                "planet": "J",
            }
        ],
        "action": 1,
    }
    cleaned = sanitize_trial_for_program(trial)
    p = cleaned["problem"]
    assert p["planet"] == "J"
    assert p["stage1_action"] == 0
    assert p["spaceship"] == "C"
    assert "reward" not in p
    assert "treasure" not in p
    assert cleaned["history"][0]["planet"] == "J"
    compact = format_trial_compact(trial, "val")
    assert "planet=J" in compact
    assert "s1=0" in compact
    assert "reward=1" not in compact


def test_evolution_json_and_evaluator_never_load_test(tmp_path: Path):
    train = [{"problem": {"x": 1}, "history": [], "action": 0}]
    val = [{"problem": {"x": 2}, "history": [], "action": 1}]
    test = [{"problem": {"LEAK": True}, "history": [], "action": 0}]
    evo = tmp_path / "trials_evolution_split.json"
    post = tmp_path / "trials_posthoc_test.json"
    _write_evolution_split_json(evo, train, val)
    _write_posthoc_test_json(post, test)
    evo_blob = json.loads(evo.read_text(encoding="utf-8"))
    assert "test" not in evo_blob
    assert "LEAK" not in json.dumps(evo_blob)
    src = _render_evaluator_py(evo, split_ratio=0.6, categorical=False)
    assert 'data["test"]' not in src
    assert "return data[\"train\"], data[\"val\"]" in src
    assert "_pooled_observed_loglik" in src
    assert "Prompt example caps do not apply here" in src


def test_prompt_subsample_and_run_participant_use_train_val_only():
    src_cap = inspect.getsource(cap_and_subsample_prompt_trials)
    assert "list(train_trials) + list(val_trials)" in src_cap
    assert "test_trials" not in src_cap
    src_run = inspect.getsource(run_participant)
    assert 'participant_ctx = {' in src_run
    assert '"train_trials": train_trials' in src_run
    assert '"val_trials": val_trials' in src_run
    assert '"test_trials"' not in src_run.split("participant_ctx = {", 1)[1].split("}", 1)[0]
    assert "Held-out test is scored once, after program selection" in src_run
    src_load = inspect.getsource(trials_for_participant)
    assert "sanitize_trial_for_program" in src_load
    assert "load_participant_limited_splits" in src_load


def _tv_trials(n: int, *, start: int = 0):
    return [
        {"problem": {"i": start + i}, "history": [], "action": i % 2}
        for i in range(n)
    ]


def test_prompt_cap_40_vs_60_identical_when_sa40_has_at_most_40():
    train, val = _tv_trials(24), _tv_trials(16, start=24)
    a, *_ = cap_and_subsample_prompt_trials(
        train, val, max_trials=40, max_trials_per_problem=5, subsample_seed=0
    )
    b, *_ = cap_and_subsample_prompt_trials(
        train, val, max_trials=60, max_trials_per_problem=5, subsample_seed=0
    )
    assert len(a) == len(b) == 40
    assert [t["problem"]["i"] for t in a] == [t["problem"]["i"] for t in b]


def test_prompt_cap_60_keeps_kool_overshoot_that_40_would_drop():
    train, val = _tv_trials(30), _tv_trials(11, start=30)
    assert len(train) + len(val) == 41
    selected40, *_ = cap_and_subsample_prompt_trials(
        train, val, max_trials=40, max_trials_per_problem=5, subsample_seed=0
    )
    selected60, *_ = cap_and_subsample_prompt_trials(
        train, val, max_trials=60, max_trials_per_problem=5, subsample_seed=0
    )
    assert len(selected40) == 40
    assert len(selected60) == 41
    assert [t["problem"]["i"] for t in selected60] == list(range(41))


def test_prompt_cap_does_not_shrink_evaluator_json(tmp_path: Path):
    train, val = _tv_trials(30), _tv_trials(11, start=30)
    evo = tmp_path / "trials_evolution_split.json"
    _write_evolution_split_json(evo, train, val)
    blob = json.loads(evo.read_text(encoding="utf-8"))
    assert len(blob["train"]) + len(blob["val"]) == 41
    src = _render_evaluator_py(evo, split_ratio=0.6, categorical=False)
    assert "Prompt example caps do not apply here" in src
    assert "return data[\"train\"], data[\"val\"]" in src


def test_sa40_structure_preserving_categories_cannot_exceed_40_except_kool():
    assert set(ICLR_15) == set(LIMITED_DATA_REGISTRY)
    for alias in ICLR_15:
        spec = LIMITED_DATA_REGISTRY[alias]
        assert spec.prefix_valid is True
        if alias == "14kool2016when":
            assert spec.category == CONTINUOUS_SESSION
            continue
        if spec.category == INDEPENDENT_TRIAL:
            continue
        if spec.category == RESETTING_UNIT:
            units = [["t"] * 15, ["t"] * 15, ["t"] * 15]
            retained, _partial, _reason = select_complete_then_prefix(
                units, 40, prefix_valid=True
            )
            assert len(retained) == 40
            continue
        assert spec.category == CONTINUOUS_SESSION
        tv = [{"problem": {"stage": 1}, "history": [], "action": 0} for _ in range(80)]
        kept, reason = _contiguous_tv_suffix(tv, 40, kool=False)
        assert len(kept) == 40
        assert reason == ""


def test_sa40_kool_stage2_cut_retains_41():
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
    kept, reason = _contiguous_tv_suffix(tv, 40, kool=True)
    assert len(kept) == 41
    assert reason == "kool_include_matching_stage1"
    train, val, test = tv[:48], tv[48:80], tv[-5:]
    new_train, new_val, new_test, _audit, manifest = apply_structure_aware_protocol(
        train,
        val,
        test,
        dataset="14kool2016when",
        participant_id=0,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_chronological_days",
    )
    n_tv = len(new_train) + len(new_val)
    assert n_tv == 41
    assert n_tv <= ICLR_FROZEN_MAX_PROMPT_TRAIN_TRIALS
    assert len(new_test) == len(test)
    assert "kool_include_matching_stage1" in (manifest.fallback_reason or "")


def test_sa40_independent_and_resetting_protocol_never_exceed_budget():
    independent = [
        {
            "problem": {"dataset_alias": "7hilbig2014generalized", "option_keys": [0, 1]},
            "history": [],
            "action": i % 2,
        }
        for i in range(100)
    ]
    nt, nv, nte, _audit, _manifest = apply_structure_aware_protocol(
        independent[:60],
        independent[60:80],
        independent[80:],
        dataset="7hilbig2014generalized",
        participant_id=0,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_unit_shuffle",
    )
    assert len(nt) + len(nv) == 40
    assert len(nte) == 20

    resetting = []
    for game in range(8):
        for t in range(15):
            resetting.append(
                {
                    "problem": {
                        "dataset_alias": "steyvers_2009_bandit",
                        "game": game,
                        "trial": t,
                        "n_arms": 4,
                        "option_keys": [0, 1, 2, 3],
                    },
                    "history": [] if t == 0 else [{"action": 0}],
                    "action": 0,
                }
            )
    nt, nv, nte, _audit, _manifest = apply_structure_aware_protocol(
        resetting[:60],
        resetting[60:90],
        resetting[90:],
        dataset="steyvers_2009_bandit",
        participant_id=0,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_unit_shuffle",
    )
    assert len(nt) + len(nv) == 40
    assert len(nte) == len(resetting[90:])


def test_all_15_share_the_same_observed_only_evaluator_contract(tmp_path: Path):
    """Isolation is per-run, not per-dataset: every alias gets the same evaluator template."""
    evo = tmp_path / "trials_evolution_split.json"
    _write_evolution_split_json(
        evo,
        [{"problem": {}, "history": [], "action": 0}],
        [{"problem": {}, "history": [], "action": 1}],
    )
    bernoulli = _render_evaluator_py(evo, split_ratio=0.6, categorical=False)
    categorical = _render_evaluator_py(evo, split_ratio=0.6, categorical=True)
    for src in (bernoulli, categorical):
        assert 'return data["train"], data["val"]' in src
        assert 'data["test"]' not in src
        assert 'choose_fn(t["problem"], t["history"])' in src or "choose_fn(problem, t.get(\"history\")" in src
        assert '"open"' not in src.split("safe_builtins", 1)[1].split("}", 1)[0]
        assert '"import"' not in src.split("safe_builtins", 1)[1].split("}", 1)[0]
    assert len(ICLR_15) == 15


def _load_generated_evaluator(tmp_path: Path, *, train, val, categorical: bool):
    evo = tmp_path / "trials_evolution_split.json"
    _write_evolution_split_json(evo, train, val)
    src = _render_evaluator_py(evo, split_ratio=0.6, categorical=categorical)
    path = tmp_path / "evaluator.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("oe_generated_evaluator", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def _write_program(tmp_path: Path, code: str) -> Path:
    p = tmp_path / "program.py"
    p.write_text(code, encoding="utf-8")
    return p


def test_official_failure_metrics_cannot_outrank_negative_loglik():
    valid = {"combined_score": -0.6931, "train_loglik": -0.6931}
    timeout = {"error": 0.0, "timeout": True}
    raised = {"error": 0.0}
    adapted_timeout = _adapt_official_failure_metrics_for_loglik(timeout)
    adapted_raised = _adapt_official_failure_metrics_for_loglik(raised)
    assert adapted_timeout["combined_score"] == FAILED_COMBINED_SCORE
    assert adapted_raised["combined_score"] == FAILED_COMBINED_SCORE
    assert math.isfinite(FAILED_COMBINED_SCORE)
    assert FAILED_COMBINED_SCORE < math.log(1e-9)
    assert adapted_timeout["error"] == 0.0
    assert adapted_timeout["timeout"] is True
    adapted_valid = _adapt_official_failure_metrics_for_loglik(dict(valid))
    assert adapted_valid["combined_score"] == valid["combined_score"]
    assert adapted_timeout["combined_score"] < valid["combined_score"]
    assert adapted_raised["combined_score"] < valid["combined_score"]
    json.dumps(adapted_timeout, allow_nan=False)
    json.dumps(adapted_raised, allow_nan=False)


def test_malformed_categorical_raises_explicit_uniform_is_valid(tmp_path: Path):
    problem = {
        "options": [{"action": 0}, {"action": 1}, {"action": 2}, {"action": 3}],
        "option_keys": [0, 1, 2, 3],
    }
    train = [{"problem": problem, "history": [], "action": 0}]
    val = [{"problem": problem, "history": [], "action": 1}]
    mod, _ = _load_generated_evaluator(tmp_path, train=train, val=val, categorical=True)
    src = (tmp_path / "evaluator.py").read_text(encoding="utf-8")
    assert "malformed categorical choose()" in src
    assert "1.0 / K" not in src

    uniform = _write_program(
        tmp_path,
        "def choose(problem, history):\n"
        "    ids = [opt['action'] for opt in problem['options']]\n"
        "    u = 1.0 / len(ids)\n"
        "    return {i: u for i in ids}\n",
    )
    metrics = mod.evaluate(str(uniform))
    assert math.isfinite(metrics["combined_score"])
    assert metrics["combined_score"] == pytest.approx(math.log(0.25), abs=1e-6)

    malformed = _write_program(tmp_path, "def choose(problem, history):\n    return 'not a dict'\n")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        mod.evaluate(str(malformed))

    empty_mass = _write_program(
        tmp_path,
        "def choose(problem, history):\n    return {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}\n",
    )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        mod.evaluate(str(empty_mass))


def test_fail_on_train_or_val_raises_like_official_evaluator_failure(tmp_path: Path):
    train = [{"problem": {"split": "train"}, "history": [], "action": 0}]
    val = [{"problem": {"split": "val"}, "history": [], "action": 1}]
    mod, _ = _load_generated_evaluator(tmp_path, train=train, val=val, categorical=False)

    fail_val = _write_program(
        tmp_path,
        "def choose(problem, history):\n"
        "    if problem.get('split') == 'val':\n"
        "        return 2.0\n"
        "    return 0.5\n",
    )
    with pytest.raises((TypeError, ValueError)):
        mod.evaluate(str(fail_val))

    fail_train = _write_program(
        tmp_path,
        "def choose(problem, history):\n"
        "    if problem.get('split') == 'train':\n"
        "        return 2.0\n"
        "    return 0.5\n",
    )
    with pytest.raises((TypeError, ValueError)):
        mod.evaluate(str(fail_train))

    ok = _write_program(tmp_path, "def choose(problem, history):\n    return 0.5\n")
    metrics = mod.evaluate(str(ok))
    assert math.isfinite(metrics["combined_score"])
    assert metrics["combined_score"] == pytest.approx(math.log(0.5), abs=1e-6)


_CHOICE13K_MARKERS = ("gamble_A", "gamble_B", "Option B")
_BINARY_ONLY_API = (
    "P(action=1) for the second option_keys entry",
    "returning a float in [0,1]: P(action=1)",
    "Return a single float",
    "probability of choosing option 1 (Option B)",
)
_Gamble_SCHEMA_OK = frozenset({"1peterson2021using", "mixed_gambles", "2plonsky2018when"})


def test_choose_api_is_in_runner_not_disk_templates():
    leftover = REPO_ROOT / "prompts" / "openevolve_vanilla"
    assert not (leftover / "interface_bernoulli.txt").exists()
    assert not (leftover / "interface_categorical.txt").exists()
    src = inspect.getsource(choose_api_text)
    assert "read_text" not in src
    assert "CHOOSE_API_BERNOULLI" in src
    bern = CHOOSE_API_BERNOULLI
    cat = CHOOSE_API_CATEGORICAL
    assert "P(action=1)" in bern
    assert "dict[int, float]" in cat
    assert "Laplace" not in bern and "Laplace" not in cat
    assert "Behavioral notes" not in bern and "Behavioral notes" not in cat
    assert "Do not read a current-trial action/label field." not in bern
    assert "Do not read a current-trial action/label field." not in cat
    assert "no current response" not in bern
    assert "no current response" not in cat
    assert "current-trial task features" in bern
    assert "previously realized trials" in bern
    assert "current-trial task features" in cat
    assert "previously realized trials" in cat
    assert choose_api_text(categorical=False) == bern.strip()
    assert "K=4" in choose_api_text(categorical=True, n_actions=4)
    assert "K=8" in choose_api_text(categorical=True, n_actions=8)


_TASK_BLOCK_BANNED = (
    "Only output the function",
    "Pure Python, no imports",
    "remain constant within a problem",
    "probabilities are unknown",
    "Write Python:",
    "def choose(problem, history):",
    "return: float, probability of choosing option 1",
)


def test_all_15_use_registered_task_descriptions_not_vanilla_files():
    src = inspect.getsource(vanilla_dataset_description)
    assert "dataset_task_description" in src
    assert "MIXED_GAMBLES_VANILLA_PROMPT" not in src
    assert "DEFAULT_BASE_PROMPT" not in src
    leftover_choices13k = (
        REPO_ROOT / "prompts" / "openevolve_vanilla" / "choices13k" / "infer_single_choice.txt"
    )
    leftover_mixed = (
        REPO_ROOT / "prompts" / "openevolve_vanilla" / "mixed_gambles" / "infer_single_choice.txt"
    )
    assert leftover_choices13k.resolve() in _LEGACY_VANILLA_TASK_FILES
    assert leftover_mixed.resolve() in _LEGACY_VANILLA_TASK_FILES

    peterson = PSYCH101_BINARY_DATASETS["1peterson2021using"]["task_description"].strip()
    assert vanilla_dataset_description("1peterson2021using") == peterson
    assert "explicit outcome probabilities" in peterson
    mixed = vanilla_dataset_description("mixed_gambles")
    assert mixed == MIXED_GAMBLES_TASK_DESCRIPTION.strip()
    assert mixed == dataset_task_description("mixed_gambles")
    assert "certain outcome" in mixed
    assert "history is empty" in mixed

    if leftover_choices13k.is_file():
        file_text = leftover_choices13k.read_text(encoding="utf-8")
        assert vanilla_dataset_description("1peterson2021using") != file_text
        assert vanilla_dataset_description(
            "1peterson2021using", base_prompt=str(leftover_choices13k)
        ) == peterson
    if leftover_mixed.is_file():
        file_text = leftover_mixed.read_text(encoding="utf-8")
        assert vanilla_dataset_description("mixed_gambles") != file_text
        assert vanilla_dataset_description(
            "mixed_gambles", base_prompt=str(leftover_mixed)
        ) == MIXED_GAMBLES_TASK_DESCRIPTION.strip()

    assert len(ICLR_15) == 15
    for alias in ICLR_15:
        registered = dataset_task_description(alias)
        task = vanilla_dataset_description(alias)
        prefix = vanilla_llm_user_prefix(alias)
        assert registered.strip()
        assert task == registered
        task_section = prefix.split("# API", 1)[0]
        for needle in _TASK_BLOCK_BANNED:
            assert needle not in task_section, f"{alias} task block still has {needle!r}"
        assert "# API" in prefix
        api = choose_api_text(
            categorical=is_categorical_output_dataset(alias),
            n_actions=dataset_n_actions(alias),
        )
        assert api.splitlines()[0] in prefix
        if is_categorical_output_dataset(alias):
            assert "dict[int, float]" in prefix
        else:
            assert "P(action=1)" in prefix


def test_prompt_api_contract_for_all_15_iclr_datasets():
    assert len(ICLR_15) == 15
    for alias in ICLR_15:
        categorical = is_categorical_output_dataset(alias)
        n_act = dataset_n_actions(alias)
        desc = vanilla_dataset_description(alias)
        api = choose_api_text(categorical=categorical, n_actions=n_act)
        prefix = vanilla_llm_user_prefix(alias)
        seed, _prompt = resolve_openevolve_seed_and_prompt(alias)
        assert desc.strip()
        assert "# API" in prefix
        assert "def choose(problem, history)" in prefix
        assert "Behavioral notes" not in prefix
        assert "prompts/external/" not in prefix
        if categorical:
            assert alias in ("13schulz2020finding", "steyvers_2009_bandit")
            assert n_act in (4, 8)
            assert f"K={n_act}" in api
            assert "dict[int, float]" in api
            assert "Do not return a single Bernoulli float" in api
            for needle in _BINARY_ONLY_API:
                assert needle not in prefix
            for needle in _CHOICE13K_MARKERS:
                assert needle not in prefix
            assert seed.resolve() == DEFAULT_CATEGORICAL_SEED_PATH.resolve()
        else:
            assert "P(action=1)" in api
            assert "dict[int, float]" not in api
            assert seed.resolve() == DEFAULT_SEED_PATH.resolve()
            if alias not in _Gamble_SCHEMA_OK:
                for needle in _CHOICE13K_MARKERS:
                    assert needle not in desc, f"{alias} description still assumes Choice13k: {desc[:200]}"
        assert "Do not read a current-trial action/label field." not in api
        assert "no current response" not in api


def test_categorical_seeds_return_dict_over_k_actions():
    for alias, k in (("steyvers_2009_bandit", 4), ("13schulz2020finding", 8)):
        seed, _ = resolve_openevolve_seed_and_prompt(alias)
        ns: dict = {}
        exec(seed.read_text(encoding="utf-8"), ns)
        problem = {
            "options": [{"action": i} for i in range(k)],
            "option_keys": list(range(k)),
        }
        out = ns["choose"](problem, [])
        assert isinstance(out, dict)
        assert set(out) == set(range(k))
        assert all(isinstance(v, float) for v in out.values())
        assert pytest.approx(sum(out.values()), abs=1e-9) == 1.0


def test_bernoulli_seed_returns_float():
    ns: dict = {}
    exec(DEFAULT_SEED_PATH.read_text(encoding="utf-8"), ns)
    out = ns["choose"]({"option_keys": ["L", "N"]}, [])
    assert isinstance(out, float)
    assert out == 0.5


def test_observed_example_formatter_never_labels_test_split():
    src = inspect.getsource(format_trial_compact)
    assert 'split="test"' not in src
    compact = format_trial_compact(
        {"problem": {"option_keys": [0, 1]}, "history": [], "action": 0},
        "train",
    )
    assert "split=test" not in compact
    assert "y=0" in compact


def test_frozen_iclr_openevolve_cli_defaults():
    args = build_arg_parser().parse_args([])
    assert args.n_iterations == ICLR_FROZEN_N_ITERATIONS == 350
    assert args.parallel_participants == ICLR_FROZEN_PARALLEL_PARTICIPANTS == 1
    assert args.parallel_evaluations == ICLR_FROZEN_PARALLEL_EVALUATIONS == 4
    assert args.limited_data_protocol == ICLR_DEFAULT_LIMITED_DATA_PROTOCOL == "structure_aware_v2"
    assert ICLR_FROZEN_LIMITED_DATA_PROTOCOL == "structure_aware"
    assert args.limited_train_val == ICLR_FROZEN_LIMITED_TRAIN_VAL == 40
    assert args.split_ratio == ICLR_FROZEN_SPLIT_RATIO == 0.6
    assert args.split_seed == ICLR_FROZEN_SPLIT_SEED == 0
    assert args.llm_max_tokens == ICLR_FROZEN_LLM_MAX_TOKENS == 1024
    assert args.model == ICLR_FROZEN_MODEL
    assert args.num_diverse_programs == 2
    assert args.num_top_programs == ICLR_FROZEN_NUM_TOP_PROGRAMS == 3
    assert args.max_prompt_train_trials == ICLR_FROZEN_MAX_PROMPT_TRAIN_TRIALS == 60
    assert args.hard_prompt_token_cap == ICLR_FROZEN_INPUT_TOKEN_CEILING == 14000
    assert args.include_artifacts is True
    assert args.base_prompt is None
    assert args.n_iterations != 600


def test_frozen_iclr_argv_and_yaml_ordinals_for_all_15():
    expected = {
        "4wulff2018description": (1290, 1339),
        "5speekenbrink2008learning": (0, 22),
        "12badham2017deficits": (0, 9),
    }
    for alias in ICLR_15:
        argv = iclr_frozen_argv(alias)
        parsed = build_arg_parser().parse_args(argv)
        start, end = emnlp_ordinal_range(alias)
        assert parsed.n_iterations == 350
        assert parsed.parallel_participants == 1
        assert parsed.parallel_evaluations == 4
        assert parsed.limited_data_protocol == "structure_aware_v2"
        assert parsed.limited_train_val == 40
        assert parsed.max_prompt_train_trials == 60
        assert parsed.num_diverse_programs == 2
        assert parsed.num_top_programs == 3
        assert argv[argv.index("--num_top_programs") + 1] == "3"
        assert parsed.hard_prompt_token_cap == ICLR_FROZEN_INPUT_TOKEN_CEILING == 14000
        assert argv[argv.index("--hard_prompt_token_cap") + 1] == "14000"
        assert "--max_prompt_train_trials" in argv
        assert argv[argv.index("--max_prompt_train_trials") + 1] == "60"
        assert parsed.range_start_ordinal == start
        assert parsed.range_end_ordinal == end
        ns = types.SimpleNamespace(
            dataset=alias,
            participant_scope="range",
            range_start_ordinal=0,
            range_end_ordinal=49,
            ordinals=None,
        )
        apply_iclr_frozen_range_ordinals(ns)
        assert (ns.range_start_ordinal, ns.range_end_ordinal) == (start, end)
        if alias in expected:
            assert (start, end) == expected[alias]
        assert "--max_workers" not in argv
        assert "600" not in argv


def test_official_optional_split_uses_seeded_random_sample_not_score_prefix():
    import random as _random

    tops = [
        {"id": f"t{i}", "code": f"def choose(problem, history):\n    return {i}\n", "metrics": {"combined_score": -0.1 * i}}
        for i in range(7)
    ]
    insp = [
        {"id": "t0", "code": "def choose(problem, history):\n    return 0\n", "metrics": {"combined_score": 0.0}},
        {"id": "i1", "code": "def choose(problem, history):\n    return 0.2\n", "metrics": {"combined_score": -0.2}},
        {"id": "i2", "code": "def choose(problem, history):\n    return 0.3\n", "metrics": {"combined_score": -0.3}},
    ]
    remaining = tops[3:]
    expected_diverse = _random.Random(12345).sample(remaining, 2)
    top, diverse, inspirations = split_official_optional_programs(
        tops, insp, num_top=3, num_diverse=2, rng=_random.Random(12345)
    )
    assert [p["id"] for p in top] == ["t0", "t1", "t2"]
    assert [p["id"] for p in diverse] == [p["id"] for p in expected_diverse]
    src = inspect.getsource(split_official_optional_programs)
    assert "random.sample" in src or "sample_fn" in src
    assert [p["id"] for p in inspirations] == ["i1", "i2"]


def test_require_openevolve_checkout_fails_without_clone(tmp_path: Path):
    with pytest.raises(SystemExit, match="checkout missing"):
        require_openevolve_checkout(root=tmp_path / "absent")


def test_require_openevolve_checkout_rejects_wrong_sha():
    audit = REPO_ROOT / "reference_repos" / "openevolve_official_audit"
    if not (audit / "openevolve").is_dir():
        pytest.skip("official audit clone absent")
    with pytest.raises(SystemExit, match="expected"):
        require_openevolve_checkout(root=audit, expected_sha="0" * 40)


def test_expected_sha_constant_and_audit_clone_if_present():
    assert EXPECTED_OPENEVOLVE_GIT_SHA == "411fb59c886c18704caaffb611e17cf9e7d824d2"
    audit = REPO_ROOT / "reference_repos" / "openevolve_official_audit"
    if not (audit / "openevolve").is_dir():
        pytest.skip("official audit clone absent")
    assert openevolve_checkout_sha(audit) == EXPECTED_OPENEVOLVE_GIT_SHA


def test_official_child_has_one_formal_parent_id():
    audit = REPO_ROOT / "reference_repos" / "openevolve_official_audit" / "openevolve" / "process_parallel.py"
    if not audit.is_file():
        pytest.skip("official audit clone absent")
    src = audit.read_text(encoding="utf-8")
    assert "parent_id=parent.id" in src
    our = inspect.getsource(VanillaProcessParallelController._submit_iteration)
    assert "super()._submit_iteration" in our
    build_src = inspect.getsource(_patched_build_prompt)
    assert "current_program" in build_src
    assert "parent_ids" not in build_src


def test_worker_snapshot_carries_tracked_interface_contract():
    src = inspect.getsource(VanillaProcessParallelController._create_database_snapshot)
    assert '"interface_text"' in src
    assert '"n_actions"' in src
    assert "read_text" not in inspect.getsource(choose_api_text)




