"""CPU tests for greedy seeded trial packing."""
from __future__ import annotations

import hashlib
from typing import List

import pytest

from utils.teh.aamas_v0_lossless_trials import (
    PHASE_EXPLORATION,
    PHASE_PARTICIPANT_EVOLUTION,
    PHASE_POPULATION,
    ScheduleState,
    SelectionRequest,
    TrialWindowDoesNotFitError,
    _Cursor,
    chat_tokens,
    pack_from_state,
    render_for_request,
    selection_digest,
)
import teh


def _fake_tokens(text: str) -> int:
    trial_lines = sum(
        1 for line in text.splitlines() if line.startswith("{") and "observed_action" in line
    )
    return 100 + 10 * trial_lines + text.count("P")


def _trial(block_id: str, position: int, size: int, history: list) -> dict:
    return {
        "problem": {"left_option": 1},
        "action": position % 2,
        "history": list(history),
        "_prompt_participant_id": 0,
        "_block_id": block_id,
        "_block_pos": position,
        "_block_len": size,
        "_ldp": {
            "unit_id": block_id,
            "chrono": position,
            "session_index": position,
            "origin_split": "train",
            "origin_index": position,
        },
    }


def _cursor(block_id: str, size: int) -> _Cursor:
    history: list = []
    trials = [_trial(block_id, index, size, history) for index in range(size)]
    for index, trial in enumerate(trials):
        if index + 1 < size:
            history.append({"action": index % 2, "reward": index})
            trials[index + 1]["history"] = list(history)
        trial["_real_next"] = trials[index + 1] if index + 1 < size else None
    return _Cursor(participant_id=0, block_id=block_id, trials=trials, index=0)


def _state(sizes: List[int]) -> ScheduleState:
    names = [f"block-{index}" for index in range(len(sizes))]
    blocks = [_cursor(name, size) for name, size in zip(names, sizes)]
    return ScheduleState(
        participant_order=[0],
        queues={0: list(blocks)},
        blocks=list(blocks),
        block_index=0,
    )


def _pack(state: ScheduleState, cap: int):
    return pack_from_state(
        state,
        wrap=lambda body: f"HEAD\n{body}",
        cap=cap,
        dataset="2plonsky2018when",
    )


def test_plonsky_tail_continues_into_later_blocks(monkeypatch):
    monkeypatch.setattr(
        "utils.teh.aamas_v0_lossless_trials.chat_tokens",
        _fake_tokens,
    )
    # 10 + 25 + first 2 of the size-5 block. The next prompt must take the
    # remaining 3 and continue into later blocks.
    cap = 100 + 10 * 37
    first_text, first_state, first = _pack(_state([10, 25, 5, 10, 8]), cap)
    assert [trial["trial_id"] for trial in first["trials"][:10]] == [
        f"block-0#{index}" for index in range(10)
    ]
    assert [trial["trial_id"] for trial in first["trials"][10:35]] == [
        f"block-1#{index}" for index in range(25)
    ]
    assert [trial["trial_id"] for trial in first["trials"][35:]] == [
        "block-2#0",
        "block-2#1",
    ]
    assert first["cursor_after"]["next_unused_trial_offset"] == 2
    assert first["next_trial_would_exceed"] is True
    assert first["full_retained_fits"] is False
    assert sum(1 for line in first_text.splitlines() if line.startswith("{")) == 37

    _second_text, _second_state, second = _pack(first_state, cap)
    ids = [trial["trial_id"] for trial in second["trials"]]
    assert ids[:3] == ["block-2#2", "block-2#3", "block-2#4"]
    assert "block-3#0" in ids
    assert len(ids) > 3
    assert len(ids) == len(set(ids))
    assert second["prompt_tokens"] <= cap
    assert second["next_trial_would_exceed"] is True


def test_no_repeat_before_pass_and_deterministic_wrap(monkeypatch):
    monkeypatch.setattr(
        "utils.teh.aamas_v0_lossless_trials.chat_tokens",
        _fake_tokens,
    )
    cap = 100 + 10 * 2
    state = _state([3])
    text0, state1, step0 = _pack(state, cap)
    text1, state2, step1 = _pack(state1, cap)
    ids0 = [trial["trial_id"] for trial in step0["trials"]]
    ids1 = [trial["trial_id"] for trial in step1["trials"]]
    assert ids0 == ["block-0#0", "block-0#1"]
    assert ids1[0] == "block-0#2"
    assert len(ids0) == len(set(ids0))
    assert len(ids1) == len(set(ids1))
    assert step1["cursor_before"]["completed_pass_count"] == 0
    # The third trial completes the pass inside this prompt, then the wrap
    # may add the start again only after that pass.
    assert ids1[0] == "block-0#2"
    if len(ids1) > 1:
        assert ids1[1] == "block-0#0"
        assert step1["cursor_after"]["completed_pass_count"] >= 1
    again, _state_again, stats_again = _pack(state1, cap)
    assert again == text1
    assert stats_again["trials"] == step1["trials"]
    assert stats_again["cursor_after"] == step1["cursor_after"]
    del text0, state2


def test_complete_retained_set_is_included_once(monkeypatch):
    monkeypatch.setattr(
        "utils.teh.aamas_v0_lossless_trials.chat_tokens",
        _fake_tokens,
    )
    cap = 100 + 10 * 4
    _text, state, stats = _pack(_state([2, 1]), cap)
    ids = [trial["trial_id"] for trial in stats["trials"]]
    assert ids == ["block-0#0", "block-0#1", "block-1#0"]
    assert len(ids) == len(set(ids))
    assert stats["full_retained_fits"] is True
    assert stats["next_trial_would_exceed"] is False
    assert stats["cursor_after"]["current_block_index"] == 0
    assert stats["cursor_after"]["next_unused_trial_offset"] == 0
    assert stats["cursor_after"]["completed_pass_count"] == 1
    _again, _advanced, again = _pack(state, cap)
    assert [trial["trial_id"] for trial in again["trials"]] == ids


def test_single_trial_that_cannot_fit_raises(monkeypatch):
    monkeypatch.setattr(
        "utils.teh.aamas_v0_lossless_trials.chat_tokens",
        lambda _text: 500,
    )
    with pytest.raises(TrialWindowDoesNotFitError, match="not truncated"):
        _pack(_state([1]), cap=100)


def test_parent_length_changes_capacity_without_corrupting_cursor(monkeypatch):
    monkeypatch.setattr(
        "utils.teh.aamas_v0_lossless_trials.chat_tokens",
        _fake_tokens,
    )
    trials = []
    for block_index, size in enumerate((4, 4)):
        cursor = _cursor(f"block-{block_index}", size)
        trials.extend(cursor.trials)

    def _render(extra: str, step: int):
        request = SelectionRequest(
            dataset="2plonsky2018when",
            phase=PHASE_EXPLORATION,
            step=step,
            master_seed=0,
            participant_id=0,
        )
        wrap = lambda body: f"HEAD\n{extra}{body}"
        text, stats = render_for_request(
            trials,
            dataset="2plonsky2018when",
            request=request,
            actual_wrap=wrap,
            canonical_wrap=wrap,
            cap=100 + 30 + 10 * 2,
        )
        return text, stats

    short0_text, short0 = _render("", 0)
    long0_text, long0 = _render("P" * 30, 0)
    _long1_text, long1 = _render("P" * 30, 1)
    assert long0["n_supervised"] < short0["n_supervised"]
    assert long0["n_supervised"] == 2
    assert long0["prompt_tokens"] <= 100 + 30 + 10 * 2
    assert short0["prompt_tokens"] <= 100 + 30 + 10 * 2
    assert long1["cursor_before"] == long0["cursor_after"]
    assert long1["trials"][0]["trial_id"] != long0["trials"][-1]["trial_id"]
    assert short0_text.startswith("participant ")
    assert long0_text.startswith("participant ")


def test_exact_qwen_count_respects_cap():
    trials = _cursor("only", 4).trials
    request = SelectionRequest(
        dataset="5speekenbrink2008learning",
        phase=PHASE_POPULATION,
        step=0,
        master_seed=0,
        participant_id=0,
    )
    wrap = lambda body: f"Task.\n{body}"
    text, stats = render_for_request(
        trials,
        dataset="5speekenbrink2008learning",
        request=request,
        actual_wrap=wrap,
        canonical_wrap=wrap,
        cap=14000,
    )
    assert stats["prompt_tokens"] == chat_tokens(wrap(text))
    assert stats["prompt_tokens"] <= 14000
    assert stats["full_retained_fits"] is True
    assert stats["n_supervised"] == 4


def test_worker_order_does_not_change_windows(monkeypatch):
    monkeypatch.setattr(
        "utils.teh.aamas_v0_lossless_trials.chat_tokens",
        _fake_tokens,
    )
    trials = _cursor("only", 6).trials

    def _at(step: int):
        request = SelectionRequest(
            dataset="3frey2017cct",
            phase=PHASE_PARTICIPANT_EVOLUTION,
            step=step,
            master_seed=0,
            participant_id=0,
        )
        wrap = lambda body: f"HEAD\n{body}"
        _text, stats = render_for_request(
            trials,
            dataset="3frey2017cct",
            request=request,
            actual_wrap=wrap,
            canonical_wrap=wrap,
            cap=100 + 10 * 2,
        )
        return stats

    forward = [_at(step) for step in (0, 1, 2)]
    backward = [_at(step) for step in (2, 1, 0)]
    for left, right in zip(forward, reversed(backward)):
        assert left["trials"] == right["trials"]
        assert left["cursor_before"] == right["cursor_before"]
        assert left["cursor_after"] == right["cursor_after"]
        assert left["assigned_text_sha256"] == right["assigned_text_sha256"]
        assert left["prompt_tokens"] == right["prompt_tokens"]


def test_per_candidate_diagnostic_keeps_its_own_trials():
    shared = {
        "aamas_selection": {"trials": [{"trial_id": "copied-last"}]},
        "selection_step": 49,
        "train_trials_before": 40,
        "train_trials_after": 4,
        "val_trials_before": 0,
        "val_trials_after": 0,
    }
    own = {
        "aamas_selection": {
            "trials": [
                {"block_id": "unit:0:round:7", "trial_position": 0, "trial_id": "unit:0:round:7#0"}
            ]
        },
        "selection_step": 0,
        "train_trials_before": 40,
        "train_trials_after": 36,
        "val_trials_before": 0,
        "val_trials_after": 0,
        "truncated": True,
    }
    merged = teh._candidate_prompt_diagnostic(shared, own)
    assert merged["selection_step"] == 0
    assert merged["aamas_selection"]["trials"][0]["trial_id"] == "unit:0:round:7#0"
    assert merged["train_trials_after"] == 36
    assert shared["aamas_selection"]["trials"][0]["trial_id"] == "copied-last"


def test_native_participant_id_is_not_replaced_by_ordinal_zero():
    trials = []
    for index in range(3):
        trials.append(
            {
                "problem": {"left_option": 1},
                "action": 0,
                "history": [],
                "_ldp": {
                    "unit_id": f"trial:train:{index}",
                    "chrono": 0,
                    "session_index": index,
                    "origin_split": "train",
                    "origin_index": index,
                },
            }
        )
    request = SelectionRequest(
        dataset="bergert_nosofsky_2007",
        phase=PHASE_EXPLORATION,
        step=0,
        master_seed=0,
        participant_id=1,
    )
    wrap = lambda body: f"HEAD\n{body}"
    text, stats = render_for_request(
        trials,
        dataset="bergert_nosofsky_2007",
        request=request,
        actual_wrap=wrap,
        canonical_wrap=wrap,
        cap=14000,
    )
    assert stats["participant_ids"] == [1]
    assert all(trial["block_id"].startswith("indep:1:") for trial in stats["trials"])
    assert "participant 1" in text
    assert "indep:0:" not in text
    digest_native = selection_digest(
        master_seed=0,
        dataset="bergert_nosofsky_2007",
        phase=PHASE_EXPLORATION,
        participant_id=1,
    )
    digest_ordinal = selection_digest(
        master_seed=0,
        dataset="bergert_nosofsky_2007",
        phase=PHASE_EXPLORATION,
        participant_id=0,
    )
    assert digest_native != digest_ordinal
    assert stats["digest"] == digest_native


def test_selection_map_entries_match_rendered_body_and_hash(monkeypatch):
    monkeypatch.setattr(
        "utils.teh.aamas_v0_lossless_trials.chat_tokens",
        _fake_tokens,
    )
    trials = []
    for cursor in _state([2, 2]).blocks:
        trials.extend(cursor.trials)
    request = SelectionRequest(
        dataset="2plonsky2018when",
        phase=PHASE_EXPLORATION,
        step=0,
        master_seed=0,
        participant_id=0,
    )
    wrap = lambda body: f"HEAD\n{body}"
    text, stats = render_for_request(
        trials,
        dataset="2plonsky2018when",
        request=request,
        actual_wrap=wrap,
        canonical_wrap=wrap,
        cap=100 + 10 * 3,
    )
    row = teh._aamas_explore_map_row(step=0, participant_id=0, selection=stats)
    assert row["trials"] == stats["trials"]
    assert len(row["assigned_block_ids"]) == len(row["trials"])
    assert len(row["assigned_positions"]) == len(row["trials"])
    body_lines = [line for line in text.splitlines() if line.startswith("{")]
    assert len(body_lines) == len(row["trials"])
    assert [trial["trial_id"] for trial in row["trials"]] == [
        trial["trial_id"] for trial in stats["trials"]
    ]
    assert row["assigned_text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_llm_phases_share_the_packer():
    source = teh.__file__
    text = open(source, encoding="utf-8").read()
    assert text.count("def _pack_aamas_lossless_prompt") == 1
    assert "render_for_request(" in text
    assert "def generate_program_variants(" in text
