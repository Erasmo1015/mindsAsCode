"""CPU tests for the optional Centaur one_call readout. multi_call scoring stays put."""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest

try:
    import openai  # noqa: F401
except Exception:
    _openai = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    _openai.OpenAI = _OpenAI
    sys.modules["openai"] = _openai

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "baseline_methods" / "Psych101"))

from Centaur import (  # noqa: E402
    CENTAUR_EVAL_MODE_MULTI,
    CENTAUR_EVAL_MODE_ONE,
    CentaurChooser,
    assert_centaur_output_mode,
    build_centaur_prompt_prefix_indexed,
    clip_action_probs,
    one_call_context_from_prefix,
    resolve_centaur_mode_output_dir,
    softmax_valid_action_logits,
    verify_one_call_action_tokens,
    write_centaur_eval_provenance,
)
from centaur_prompts import centaur_display_keys  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from utils.teh.limited_data_protocol import (  # noqa: E402
    load_participant_limited_splits,
    load_raw_participant_splits,
)
from utils.teh.limited_data_registry import limited_data_spec  # noqa: E402
from Centaur import (  # noqa: E402
    _centaur_prompt_timeline_sa40_fair,
    _ensure_mixed_gambles_centaur_contract,
    _task_instruction_for_participant,
)

TOKENIZER = (
    Path.home()
    / ".cache/huggingface/hub/models--unsloth--Llama-3.1-8B/snapshots/"
    "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814/tokenizer.json"
)


def _llama_encode(text: str):
    tok = Tokenizer.from_file(str(TOKENIZER))
    return tok.encode(text, add_special_tokens=False).ids


def _gamble_trials():
    problem = {
        "dataset_alias": "2plonsky2018when",
        "schema_type": "A",
        "option_keys": ["F", "X"],
        "gamble_A": {"probs": [1.0], "rewards": [1.0]},
        "gamble_B": {"probs": [1.0], "rewards": [2.0]},
        "has_feedback": True,
        "block_index": 0,
    }
    return [
        {
            "problem": problem,
            "history": [],
            "action": 1,
            "options": ["F", "X"],
        }
    ]


def _prompt_for_dataset(alias: str, participant_id: int):
    kw = dict(
        split_ratio=0.6,
        split_seed=0,
        psych_dataset_split="train",
        limited_data_protocol="structure_aware_v3",
        limited_train_val=40,
        speekenbrink_split="chronological",
    )
    train, val, test, _audit, _manifest = load_participant_limited_splits(
        alias, participant_id, **kw
    )
    raw_train, raw_val, _raw_test, _kind = load_raw_participant_splits(
        alias,
        participant_id,
        split_ratio=0.6,
        split_seed=0,
        psych_dataset_split="train",
        speekenbrink_split="chronological",
    )
    raw_train = _ensure_mixed_gambles_centaur_contract(alias, raw_train)
    raw_val = _ensure_mixed_gambles_centaur_contract(alias, raw_val)
    test = _ensure_mixed_gambles_centaur_contract(alias, test)
    prompt, score = _centaur_prompt_timeline_sa40_fair(
        train,
        val,
        test,
        category=limited_data_spec(alias).category,
        raw_train=raw_train,
        raw_val=raw_val,
    )
    instruction = _task_instruction_for_participant(
        alias, participant_id, psych_dataset_split="train", local_dataset=None
    )
    index = score[min(3, len(score) - 1)]
    prefix = build_centaur_prompt_prefix_indexed(prompt, index, instruction=instruction)
    return prompt, index, prefix, instruction


def _chooser_with_logits(logits_by_id):
    chooser = CentaurChooser("unused-for-cpu-test")
    chooser.eval_mode = CENTAUR_EVAL_MODE_ONE
    calls = []

    def encode(text: str):
        return _llama_encode(text)

    def logits(text: str):
        calls.append(text)
        chooser.n_model_forwards += 1

        class _Row(dict):
            pass

        return logits_by_id

    chooser._encode_ids = encode
    chooser._next_token_logits = logits
    return chooser, calls


def test_multi_call_still_softmaxes_full_suffixes():
    trials = _gamble_trials()
    chooser = CentaurChooser("unused-for-cpu-test")
    chooser.eval_mode = CENTAUR_EVAL_MODE_MULTI
    seen = []

    def suffix(prefix, suf):
        seen.append(suf)
        return {"<<F>>.": -0.2, "<<X>>.": -1.2}[suf], None

    chooser._suffix_logprob_detailed = suffix
    chooser.action_probs_one_call = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("multi_call must not use one_call")
    )
    probs = chooser.score_action_probs(trials, 0)
    direct = chooser.action_probs_from_suffixes(trials, 0)
    assert seen == ["<<F>>.", "<<X>>.", "<<F>>.", "<<X>>."]
    expected = clip_action_probs(softmax_valid_action_logits([-0.2, -1.2]))
    assert probs == pytest.approx(expected)
    assert direct == pytest.approx(expected)
    assert abs(sum(softmax_valid_action_logits([-0.2, -1.2])) - 1.0) < 1e-12


def test_one_call_binary_synthetic_one_forward_and_no_suffix():
    trials = _gamble_trials()
    chooser = CentaurChooser("unused-for-cpu-test")
    chooser.task_instruction = "instr"
    prefix = build_centaur_prompt_prefix_indexed(trials, 0, instruction="instr")
    context = one_call_context_from_prefix(prefix)
    ids = verify_one_call_action_tokens(_llama_encode, context, ["F", "X"])
    chooser, calls = _chooser_with_logits({ids[0]: 0.0, ids[1]: 1.0})
    chooser.task_instruction = "instr"
    probs = chooser.action_probs_one_call(trials, 0)
    again = chooser.action_probs_one_call(trials, 0)
    assert calls == [context]
    assert chooser.n_model_forwards == 1
    assert again == pytest.approx(probs)
    assert abs(sum(chooser.last_prob_debug["probs_unclipped"]) - 1.0) < 1e-12
    assert calls[0].endswith("You press <<")
    assert not calls[0].endswith(">>")
    assert ">>." not in calls[0][calls[0].rfind("You press <<") :]
    assert trials[0]["action"] == 1
    assert not calls[0].endswith("<<" + "X")
    assert probs[1] > probs[0]


@pytest.mark.parametrize(
    "alias,participant_id,expected_k",
    [
        ("14kool2016when", 0, 2),
        ("steyvers_2009_bandit", 1, 4),
        ("13schulz2020finding", 0, 8),
    ],
)
def test_real_dataset_one_call_token_alignment(alias, participant_id, expected_k):
    if not TOKENIZER.is_file():
        pytest.skip("Llama 3.1 tokenizer.json is not cached")
    _prompt, index, prefix, instruction = _prompt_for_dataset(alias, participant_id)
    assert prefix.endswith("You press ")
    context = one_call_context_from_prefix(prefix)
    assert context == prefix + "<<"
    labels = [str(k) for k in centaur_display_keys(_prompt[index]["problem"])]
    assert len(labels) == expected_k
    token_ids = verify_one_call_action_tokens(_llama_encode, context, labels)
    assert len(token_ids) == expected_k
    assert len(set(token_ids)) == expected_k
    action = int(_prompt[index]["action"])
    observed = labels[action]
    tail = context[context.rfind("You press <<") :]
    assert tail == "You press <<"
    assert observed not in tail
    assert not context.endswith(">>.")
    chooser, calls = _chooser_with_logits({tid: float(i) for i, tid in enumerate(token_ids)})
    chooser.task_instruction = instruction
    probs = chooser.action_probs_one_call(_prompt, index)
    assert len(calls) == 1
    assert chooser.n_model_forwards == 1
    assert calls[0] == context
    unclipped = chooser.last_prob_debug["probs_unclipped"]
    assert abs(sum(unclipped) - 1.0) < 1e-12
    assert len(chooser.last_prob_debug["raw_logits"]) == expected_k
    assert math.isfinite(math.log(max(probs[action], 1e-12)))


def test_modes_cannot_share_or_resume_one_directory(tmp_path: Path):
    multi = tmp_path / "job_multi"
    one = tmp_path / "job_one" / "one_call"
    write_centaur_eval_provenance(multi, CENTAUR_EVAL_MODE_MULTI)
    write_centaur_eval_provenance(one, CENTAUR_EVAL_MODE_ONE)
    with pytest.raises(ValueError, match="cannot share or resume"):
        assert_centaur_output_mode(multi, CENTAUR_EVAL_MODE_ONE)
    with pytest.raises(ValueError, match="cannot share"):
        assert_centaur_output_mode(one, CENTAUR_EVAL_MODE_MULTI)
    with pytest.raises(ValueError, match="named"):
        resolve_centaur_mode_output_dir(str(one), CENTAUR_EVAL_MODE_MULTI)
    legacy = tmp_path / "legacy_multi"
    legacy.mkdir()
    (legacy / "participant_details_loglik.csv").write_text("participant_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unmarked"):
        resolve_centaur_mode_output_dir(str(legacy), CENTAUR_EVAL_MODE_ONE)
    with pytest.raises(ValueError, match="cannot share or resume"):
        resolve_centaur_mode_output_dir(str(multi), CENTAUR_EVAL_MODE_ONE)
    nested = resolve_centaur_mode_output_dir(str(tmp_path / "fresh"), CENTAUR_EVAL_MODE_ONE)
    assert nested.name == "one_call"
    assert nested != multi
