"""AAMAS-v0 lossless trial prompts.

Process-wide for an AAMAS run. Historical PICS-v3 generation does not import
this module's renderer unless the AAMAS prompt flag is on.
"""
from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from utils.teh.limited_data_registry import (
    CONTINUOUS_SESSION,
    INDEPENDENT_TRIAL,
    limited_data_spec,
)
from utils.teh.prompt_snapshots import sanitize_problem_for_choose
from utils.teh.prompt_units import qwen_user_prompt_token_count, trial_unit_id

TRIAL_PROMPT_POLICY_ID = "compact_faithful_trial_v2"
LEGACY_TRIAL_PROMPT_POLICY_ID = "pics_aamas_v0"
PREVIOUS_COMPACT_POLICY_ID = "compact_faithful_trial_v1"
# Greedy seeded fill. Pre-fix lossless runs omit this field and must not be resumed.
PACKING_IMPLEMENTATION_ID = "aamas_v0_greedy_pack_v1"
SELECTION_NAMESPACE = "trial_selection"
PHASE_POPULATION = "population"
PHASE_EXPLORATION = "exploration"
PHASE_PARTICIPANT_EVOLUTION = "participant_evolution"
CAP = 14000


class TrialWindowDoesNotFitError(RuntimeError):
    """A single complete trial does not fit in an otherwise empty trial section."""

# Method randomness. split_seed stays the data-split seed.
_PICS_RUN_SEED: Optional[int] = None
_PICS_SPLIT_SEED: Optional[int] = None

_SELECTION: ContextVar[Optional["SelectionRequest"]] = ContextVar(
    "aamas_v0_lossless_selection", default=None
)

def set_pics_run_seeds(*, run_seed: int, split_seed: int) -> None:
    """Record the method seed and the data-split seed. They are not interchangeable."""
    global _PICS_RUN_SEED, _PICS_SPLIT_SEED
    _PICS_RUN_SEED = int(run_seed)
    _PICS_SPLIT_SEED = int(split_seed)


def pics_run_seed() -> Optional[int]:
    return _PICS_RUN_SEED


def pics_recorded_split_seed() -> Optional[int]:
    return _PICS_SPLIT_SEED


def pics_lossless_subseed(namespace: str, **coords: Any) -> int:
    """SHA-256 sub-seed. Python hash() is not used. Does not read split_seed."""
    run_seed = pics_run_seed()
    if run_seed is None:
        raise RuntimeError("PICS run seed is not set")
    payload: Dict[str, Any] = {
        "namespace": str(namespace),
        "pics_run_seed": int(run_seed),
    }
    for key in sorted(coords):
        value = coords[key]
        if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
            payload[str(key)] = value
        else:
            payload[str(key)] = str(value)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def pics_lossless_llm_request_seed(
    *,
    dataset: str,
    phase: str,
    participant: Optional[int],
    iteration: Optional[int],
    role: str,
    candidate: int,
) -> int:
    return pics_lossless_subseed(
        "llm_request",
        dataset=str(dataset),
        phase=str(phase),
        participant=None if participant is None else int(participant),
        iteration=None if iteration is None else int(iteration),
        role=str(role),
        candidate=int(candidate),
    )


HISTORY_INTERPRETATION = """History interpretation:

- `problem` and `history` are the inputs available before a choice.
- `observed_action` is the supervised prediction target and is not an input.
- The one explicitly displayed `history` list is the complete real history received by that displayed trial.
- For each following trial in the same block, its history is that displayed history plus `history_entry_after_this_choice` from every earlier displayed trial in chronological order.
- The complete history is not printed repeatedly only to avoid duplicating the same information. It must not be interpreted as empty or truncated.
- At a marked reset boundary, discard the preceding history and begin the new block.
- `history` is a list and may be empty.
- Fields inside a history dictionary may be absent or None. Generated code must check keys/use `.get(...)` and handle None before numerical operations.
- All displayed histories contain retained SA40 observations only; omitted train/validation observations are not restored.
"""

CONTINUOUS_NOTE = (
    "This dataset is one continuous session per participant. History does not "
    "reset at day boundaries. History resets when the participant changes.\n"
)
INDEPENDENT_NOTE = (
    "All trials are independent. Each trial has history = [].\n"
)


@dataclass
class SelectionRequest:
    dataset: str
    phase: str
    step: int
    master_seed: int
    participant_id: Optional[int] = None
    cursor_parent: str = ""
    vary_exhausted_pass: bool = False
    panel_bank: str = ""
    actual_parent_tokens: int = 0


def bind_selection(request: Optional[SelectionRequest]) -> None:
    _SELECTION.set(request)


def current_selection() -> Optional[SelectionRequest]:
    return _SELECTION.get()


@contextmanager
def selection_scope(request: SelectionRequest) -> Iterator[SelectionRequest]:
    token = _SELECTION.set(request)
    try:
        yield request
    finally:
        _SELECTION.reset(token)


def chat_tokens(text: str) -> int:
    if not text:
        return 0
    return int(qwen_user_prompt_token_count(text))


def selection_digest(
    *,
    master_seed: int,
    dataset: str,
    phase: str,
    participant_id: Optional[int],
) -> str:
    payload = json.dumps(
        {
            "master_seed": int(master_seed),
            "dataset": str(dataset),
            "phase": str(phase),
            "participant": None if participant_id is None else int(participant_id),
            "namespace": SELECTION_NAMESPACE,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rng(digest: str) -> np.random.Generator:
    words = [int(digest[i : i + 8], 16) for i in range(0, 64, 8)]
    return np.random.default_rng(np.random.SeedSequence(words))


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _pid(trial: Dict[str, Any]) -> int:
    """Native participant identity. Missing ids are an error, not ordinal 0."""
    raw = trial.get("_prompt_participant_id")
    if raw is None:
        raw = (trial.get("problem") or {}).get("participant_id")
    if raw is None:
        raise ValueError(
            "Trial has no participant identity. Seeding and block ids use the native "
            "participant id; processing ordinal 0 is not a substitute."
        )
    return int(raw)


def _stamp_missing_participant(
    trials: Sequence[Dict[str, Any]],
    participant_id: Optional[int],
) -> List[Dict[str, Any]]:
    """Fill a missing prompt id from the request's native participant id."""
    stamped: List[Dict[str, Any]] = []
    for trial in trials:
        row = dict(trial)
        if participant_id is not None and row.get("_prompt_participant_id") is None:
            row["_prompt_participant_id"] = int(participant_id)
        stamped.append(row)
    return stamped


def _scope_id(trial: Dict[str, Any], dataset: str) -> str:
    spec = limited_data_spec(dataset)
    pid = _pid(trial)
    if spec.category == CONTINUOUS_SESSION:
        return f"session:{pid}"
    if spec.category == INDEPENDENT_TRIAL:
        ldp = trial.get("_ldp") or {}
        marker = (
            f"{ldp.get('origin_split', 'na')}:"
            f"{ldp.get('origin_index', 'na')}:"
            f"{trial.get('_pool_order', 0)}"
        )
        return f"indep:{pid}:{marker}"
    unit = trial_unit_id(trial, dataset=dataset, fallback=int(trial.get("_pool_order", 0)))
    return f"unit:{pid}:{unit}"


def _chrono(trial: Dict[str, Any]) -> Tuple[int, int]:
    ldp = trial.get("_ldp") or {}
    if ldp.get("session_index") is not None:
        return int(ldp["session_index"]), int(trial.get("_pool_order", 0))
    chrono = ldp.get("chrono")
    if chrono is None:
        chrono = 10**9
    return int(chrono), int(trial.get("_pool_order", 0))


def _ensure_tags(trials: Sequence[Dict[str, Any]], dataset: str) -> List[Dict[str, Any]]:
    if trials and all(isinstance(trial.get("_ldp"), dict) for trial in trials):
        return [dict(trial) for trial in trials]
    from utils.teh.limited_data_protocol import tag_split_trials

    by_pid: Dict[int, List[Dict[str, Any]]] = {}
    pid_order: List[int] = []
    for trial in trials:
        pid = _pid(trial)
        if pid not in by_pid:
            pid_order.append(pid)
            by_pid[pid] = []
        by_pid[pid].append(trial)
    tagged: List[Dict[str, Any]] = []
    for pid in pid_order:
        train, _val, _test = tag_split_trials(dataset, by_pid[pid], [], [])
        for trial in train:
            trial["_prompt_participant_id"] = pid
        tagged.extend(train)
    return tagged


def group_blocks(trials: Sequence[Dict[str, Any]], dataset: str) -> List[List[Dict[str, Any]]]:
    tagged = _ensure_tags(trials, dataset)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for index, trial in enumerate(tagged):
        stamped = dict(trial)
        stamped["_pool_order"] = int(trial.get("_pool_order", index))
        key = _scope_id(stamped, dataset)
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(stamped)
    blocks: List[List[Dict[str, Any]]] = []
    for key in order:
        group = sorted(buckets[key], key=_chrono)
        for position, trial in enumerate(group):
            trial["_block_id"] = key
            trial["_block_pos"] = position
            trial["_block_len"] = len(group)
        blocks.append(group)
    return blocks


def history_delta(prev: Dict[str, Any], nxt: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Entries that appear in the next retained history. None when SA40 has no later trial."""
    if nxt is None or not isinstance(nxt.get("history"), list):
        return None
    prev_hist = prev.get("history") if isinstance(prev.get("history"), list) else []
    nxt_hist = nxt["history"]
    if len(nxt_hist) <= len(prev_hist):
        return None
    return [copy.deepcopy(item) for item in nxt_hist[len(prev_hist) :]]


def anchor_index(trials: Sequence[Dict[str, Any]]) -> int:
    """Which selected trial prints the complete history list."""
    if not trials:
        return 0
    first_hist = trials[0].get("history") if isinstance(trials[0].get("history"), list) else []
    if first_hist:
        return 0
    for index, trial in enumerate(trials):
        hist = trial.get("history") if isinstance(trial.get("history"), list) else []
        if hist and index < len(trials) - 1:
            return index
    for index, trial in enumerate(trials):
        hist = trial.get("history") if isinstance(trial.get("history"), list) else []
        if hist:
            return index
    return 0


def render_window(trials: Sequence[Dict[str, Any]], *, trial_offset: int = 0) -> str:
    if not trials:
        return ""
    anchor = anchor_index(trials)
    lines: List[str] = []
    for offset, trial in enumerate(trials):
        nxt = trials[offset + 1] if offset + 1 < len(trials) else None
        # The delta must come from the real following retained trial, which is
        # the next trial in the block even when it is outside this window.
        real_next = trial.get("_real_next")
        if real_next is None and nxt is not None and nxt.get("_block_id") == trial.get("_block_id"):
            real_next = nxt
        delta = history_delta(trial, real_next if isinstance(real_next, dict) else None)
        lines.append(f"Trial {trial_offset + offset + 1}")
        lines.append(f"participant_id = { _pid(trial) }")
        lines.append(f"block_id = {trial.get('_block_id')}")
        lines.append(f"position = {int(trial.get('_block_pos', offset))}")
        resets = offset == 0 and int(trial.get("_block_pos", 0)) == 0
        lines.append(f"reset_before = {str(resets)}")
        problem = sanitize_problem_for_choose(trial.get("problem") or {})
        lines.append("problem = " + _dumps(problem))
        if offset == anchor:
            history = list(trial.get("history") or [])
            lines.append("history = " + _dumps(history))
        lines.append("observed_action = " + _dumps(trial.get("action")))
        if delta is not None:
            rendered = delta[0] if len(delta) == 1 else delta
            lines.append("history_entry_after_this_choice = " + _dumps(rendered))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _one_line_schedule(windows: Sequence[Sequence[Dict[str, Any]]], dataset: str) -> str:
    """Historical participant one-line text. The cursor chooses the trials."""
    from data_modules.psych101_binary import format_trial_for_prompt

    lines: List[str] = []
    number = 0
    for window in windows:
        for offset, trial in enumerate(window):
            number += 1
            line = format_trial_for_prompt(trial, number)
            if str(dataset) == "14kool2016when":
                nxt = trial.get("_real_next")
                if (
                    nxt is None
                    and offset + 1 < len(window)
                    and window[offset + 1].get("_block_id") == trial.get("_block_id")
                ):
                    nxt = window[offset + 1]
                delta = history_delta(trial, nxt if isinstance(nxt, dict) else None)
                if delta:
                    rendered = delta[0] if len(delta) == 1 else delta
                    line += "; history_update=" + _dumps(rendered)
            lines.append(line)
    return "\n".join(lines)


_PRECHOICE_FORBIDDEN = (
    "weather_outcome",
    "was_correct",
    "exploded",
    "outcome_marker",
    "probe_in_set",
    "correct_category",
    "response_key",
    "observed_action",
    "history_before",
)


def _assert_one_line_prechoice(text: str) -> None:
    """Current outcomes stay after the observed action. Hidden answers stay out."""
    if "probe_in_set" in text or '"observed_action"' in text:
        raise RuntimeError("PICS v4 one-line prompt contains a wrapper or hidden-answer field.")
    for line in text.splitlines():
        if "action=" not in line:
            continue
        prefix = line.split("action=", 1)[0]
        for key in _PRECHOICE_FORBIDDEN:
            if key in prefix:
                raise RuntimeError(f"PICS v4 pre-choice text contains {key}.")
        if "[kool/stage1]" in line and "; planet=" in prefix:
            raise RuntimeError("PICS v4 stage-1 line contains the current planet.")


def render_schedule_text(windows: Sequence[Sequence[Dict[str, Any]]], dataset: str) -> str:
    """Trial text for the seeded cursor.

    PICS v4 uses the historical one-line formatter. Other AAMAS callers keep
    the compact records already shipped by those runs.
    """
    from utils.teh.pics_v4 import using_pics_v4

    if using_pics_v4():
        return _one_line_schedule(windows, dataset)
    from utils.teh.compact_faithful_trials import render_compact_schedule

    return render_compact_schedule(windows)


def reconstruct_histories(window: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    """Rebuild each selected trial's history from the one displayed list plus entries."""
    if not window:
        return []
    anchor = anchor_index(window)
    displayed = list(window[anchor].get("history") or [])
    deltas: List[List[Any]] = []
    for offset, trial in enumerate(window):
        real_next = trial.get("_real_next")
        if real_next is None and offset + 1 < len(window):
            real_next = window[offset + 1]
        delta = history_delta(trial, real_next if isinstance(real_next, dict) else None) or []
        deltas.append(delta)
    rebuilt: List[List[Any]] = []
    if anchor == 0:
        running = list(displayed)
        for offset, trial in enumerate(window):
            if offset == 0:
                rebuilt.append(list(displayed))
            else:
                rebuilt.append(list(running))
            running = list(running) + list(deltas[offset])
        return rebuilt
    # Trials before the anchor are the prefix of the displayed history.
    before = window[:anchor]
    consumed: List[Any] = []
    prefix_lists: List[List[Any]] = []
    for offset, _trial in enumerate(before):
        prefix_lists.append(list(consumed))
        consumed.extend(deltas[offset])
    if _dumps(consumed) != _dumps(displayed):
        # Still return the mechanical reconstruction; tests compare to SA40.
        pass
    running = list(displayed)
    rebuilt.extend(prefix_lists)
    rebuilt.append(list(displayed))
    for offset in range(anchor, len(window) - 1):
        running = list(running) + list(deltas[offset])
        rebuilt.append(list(running))
    return rebuilt


def attach_real_next(blocks: Sequence[Sequence[Dict[str, Any]]]) -> None:
    for block in blocks:
        for index, trial in enumerate(block):
            trial["_real_next"] = block[index + 1] if index + 1 < len(block) else None


@dataclass
class _Cursor:
    participant_id: int
    block_id: str
    trials: List[Dict[str, Any]]
    index: int = 0

    @property
    def remaining(self) -> int:
        return len(self.trials) - self.index

    def key(self) -> Tuple[int, str, int]:
        return (self.participant_id, self.block_id, self.index)


@dataclass
class ScheduleState:
    participant_order: List[int]
    queues: Dict[int, List[_Cursor]]
    blocks: List[_Cursor]
    block_index: int = 0
    rr: int = 0
    exhausted_passes: int = 0

    def clone(self) -> "ScheduleState":
        return copy.deepcopy(self)


def _linearize_blocks(
    participant_order: Sequence[int],
    queues: Dict[int, List[_Cursor]],
) -> List[_Cursor]:
    """One seeded block sequence: round-robin, one whole block at a time."""
    if not participant_order:
        return []
    progress = {int(pid): 0 for pid in participant_order}
    order: List[_Cursor] = []
    rr = 0
    n = len(participant_order)
    total = sum(len(queues.get(int(pid)) or []) for pid in participant_order)
    while len(order) < total:
        placed = False
        for attempt in range(n):
            pid = int(participant_order[(rr + attempt) % n])
            queue = queues.get(pid) or []
            index = progress[pid]
            if index < len(queue):
                order.append(queue[index])
                progress[pid] = index + 1
                rr = (list(participant_order).index(pid) + 1) % n
                placed = True
                break
        if not placed:
            break
    return order


def _trial_key(trial: Dict[str, Any]) -> Tuple[str, int]:
    return (str(trial.get("_block_id")), int(trial.get("_block_pos", 0)))


def _trial_entry(trial: Dict[str, Any]) -> Dict[str, Any]:
    block_id = str(trial.get("_block_id"))
    position = int(trial.get("_block_pos", 0))
    return {
        "participant_id": _pid(trial),
        "block_id": block_id,
        "trial_position": position,
        "trial_id": f"{block_id}#{position}",
    }


def _cursor_view(state: ScheduleState) -> Dict[str, int]:
    if not state.blocks or state.block_index >= len(state.blocks):
        index = 0
        offset = 0
    else:
        index = int(state.block_index)
        offset = int(state.blocks[index].index)
    return {
        "current_block_index": index,
        "next_unused_trial_offset": offset,
        "completed_pass_count": int(state.exhausted_passes),
    }


def _rewind_to_pass_start(state: ScheduleState) -> None:
    for cursor in state.blocks:
        cursor.index = 0
    state.block_index = 0
    state.rr = 0


def _pass_has_remaining(state: ScheduleState) -> bool:
    if not state.blocks or state.block_index >= len(state.blocks):
        return False
    return any(cursor.remaining > 0 for cursor in state.blocks[state.block_index :])


def _build_state(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    master_seed: int,
    phase: str,
    participant_id: Optional[int],
) -> ScheduleState:
    tagged = _stamp_missing_participant(trials, participant_id)
    for index, trial in enumerate(tagged):
        trial["_pool_order"] = index
    blocks = group_blocks(tagged, dataset)
    attach_real_next(blocks)
    by_pid: Dict[int, List[List[Dict[str, Any]]]] = {}
    pid_order: List[int] = []
    for block in blocks:
        if not block:
            continue
        pid = _pid(block[0])
        if pid not in by_pid:
            by_pid[pid] = []
            pid_order.append(pid)
        by_pid[pid].append(block)
    digest = selection_digest(
        master_seed=master_seed,
        dataset=dataset,
        phase=phase,
        participant_id=participant_id,
    )
    rng = _rng(digest)
    shuffled_pids = [pid_order[int(i)] for i in rng.permutation(len(pid_order))] if pid_order else []
    queues: Dict[int, List[_Cursor]] = {}
    for pid in shuffled_pids:
        person_blocks = by_pid[pid]
        local = selection_digest(
            master_seed=master_seed,
            dataset=dataset,
            phase=phase + ":blocks",
            participant_id=pid,
        )
        local_rng = _rng(local)
        order = [person_blocks[int(i)] for i in local_rng.permutation(len(person_blocks))]
        # Continuous sessions stay chronological; shuffling one block is a no-op.
        spec = limited_data_spec(dataset)
        if spec.category == CONTINUOUS_SESSION:
            order = person_blocks
        queues[pid] = [
            _Cursor(participant_id=pid, block_id=str(block[0]["_block_id"]), trials=block, index=0)
            for block in order
            if block
        ]
    return ScheduleState(
        participant_order=shuffled_pids,
        queues=queues,
        blocks=_linearize_blocks(shuffled_pids, queues),
        block_index=0,
        rr=0,
    )


def _advance_to_open_trial(state: ScheduleState, included: set) -> Optional[_Cursor]:
    """Move to the next trial in traversal order that this prompt has not used."""
    while state.block_index < len(state.blocks):
        cursor = state.blocks[state.block_index]
        while cursor.index < len(cursor.trials) and _trial_key(cursor.trials[cursor.index]) in included:
            cursor.index += 1
        if cursor.remaining > 0:
            return cursor
        state.block_index += 1
    return None


def pack_from_state(
    state: ScheduleState,
    *,
    wrap,
    cap: int,
    dataset: str,
) -> Tuple[str, ScheduleState, Dict[str, Any]]:
    """Greedy fill in the seeded block order.

    A whole remaining block suffix that fits is included, and packing continues
    into the next block of the same prompt. A suffix that does not fit contributes
    its largest complete-trial prefix, then the prompt stops. Reaching the end of
    the pass with room left wraps to the start without repeating a trial already
    in this prompt. The retained set is never duplicated inside one prompt.
    """
    working = state.clone()
    cursor_before = _cursor_view(working)
    windows: List[List[Dict[str, Any]]] = []
    included: set = set()
    wraps_inside = 0
    stopped_on_limit = False
    retained = {
        _trial_key(trial)
        for cursor in working.blocks
        for trial in cursor.trials
    }
    guard = len(retained) + 2
    while guard > 0:
        guard -= 1
        cursor = _advance_to_open_trial(working, included)
        if cursor is None:
            if included and included >= retained:
                _rewind_to_pass_start(working)
                working.exhausted_passes += 1
                break
            if not working.blocks:
                break
            _rewind_to_pass_start(working)
            working.exhausted_passes += 1
            wraps_inside += 1
            if wraps_inside > 1:
                break
            continue
        suffix = []
        for trial in cursor.trials[cursor.index :]:
            if _trial_key(trial) in included:
                break
            suffix.append(trial)
        if not suffix:
            working.block_index += 1
            continue
        whole_body = render_schedule_text(windows + [suffix], dataset)
        if chat_tokens(wrap(whole_body)) <= cap:
            windows.append(list(suffix))
            for trial in suffix:
                included.add(_trial_key(trial))
            cursor.index += len(suffix)
            working.block_index += 1
            continue
        count = 0
        for size in range(1, len(suffix) + 1):
            body = render_schedule_text(windows + [suffix[:size]], dataset)
            if chat_tokens(wrap(body)) <= cap:
                count = size
            else:
                break
        if count <= 0:
            if not included:
                lone = cursor.trials[cursor.index : cursor.index + 1]
                lone_body = render_schedule_text([lone], dataset)
                lone_tokens = chat_tokens(wrap(lone_body))
                raise TrialWindowDoesNotFitError(
                    "AAMAS v0 trial does not fit under the prompt token limit "
                    f"(cap={cap}, tokens={lone_tokens}, block_id={cursor.block_id}, "
                    f"trial_position={int(cursor.trials[cursor.index].get('_block_pos', cursor.index))}). "
                    "The trial text was not truncated."
                )
            stopped_on_limit = True
            break
        prefix = list(cursor.trials[cursor.index : cursor.index + count])
        windows.append(prefix)
        for trial in prefix:
            included.add(_trial_key(trial))
        cursor.index += count
        stopped_on_limit = True
        break
    text = render_schedule_text(windows, dataset) if windows else ""
    stats = _stats(windows, dataset)
    flat = [trial for window in windows for trial in window]
    from utils.teh.pics_v4 import using_pics_v4

    if using_pics_v4():
        _assert_one_line_prechoice(text)
        mismatches = 0
    else:
        from utils.teh.compact_faithful_trials import round_trip_mismatches

        mismatches = round_trip_mismatches(text, flat) if text else 0
        if mismatches:
            raise RuntimeError(
                f"Compact trial text does not round-trip sanitized problem and history "
                f"({mismatches} trials)."
            )
    stats["history_mismatches"] = mismatches
    stats["prompt_tokens"] = chat_tokens(wrap(text)) if text or windows else chat_tokens(wrap(""))
    stats["new_pass_started_inside_prompt"] = int(wraps_inside)
    stats["trials"] = [_trial_entry(trial) for trial in flat]
    stats["cursor_before"] = cursor_before
    stats["cursor_after"] = _cursor_view(working)
    stats["full_retained_fits"] = bool(retained) and included == retained and len(flat) == len(retained)
    stats["next_trial_would_exceed"] = bool(stopped_on_limit)
    if stats["prompt_tokens"] > int(cap) and flat:
        raise TrialWindowDoesNotFitError(
            f"AAMAS v0 packed prompt is {stats['prompt_tokens']} tokens, over cap {cap}."
        )
    return text, working, stats


def state_at_step(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    master_seed: int,
    phase: str,
    participant_id: Optional[int],
    step: int,
    canonical_wrap,
    cap: int,
    vary_exhausted_pass: bool = False,
) -> ScheduleState:
    state = _build_state(
        trials,
        dataset=dataset,
        master_seed=master_seed,
        phase=phase,
        participant_id=participant_id,
    )
    for _ in range(max(0, int(step))):
        if _fully_exhausted(state):
            state = _restart_pass(state, vary=vary_exhausted_pass)
        _text, state, _stats = pack_from_state(
            state, wrap=canonical_wrap, cap=cap, dataset=dataset
        )
    if _fully_exhausted(state):
        state = _restart_pass(state, vary=vary_exhausted_pass)
    return state


def _fully_exhausted(state: ScheduleState) -> bool:
    return not _pass_has_remaining(state)


def _restart_pass(state: ScheduleState, *, vary: bool = False) -> ScheduleState:
    """Start the next pass at block 0. The seeded order does not rotate."""
    del vary
    restarted = state.clone()
    _rewind_to_pass_start(restarted)
    restarted.exhausted_passes += 1
    return restarted


def _stats(windows: Sequence[Sequence[Dict[str, Any]]], dataset: str) -> Dict[str, Any]:
    trials = [trial for window in windows for trial in window]
    participants = []
    blocks = []
    for trial in trials:
        pid = _pid(trial)
        block = str(trial.get("_block_id"))
        if pid not in participants:
            participants.append(pid)
        if block not in blocks:
            blocks.append(block)
    positions = [int(trial.get("_block_pos", 0)) for trial in trials]
    return {
        "n_supervised": len(trials),
        "n_participants": len(participants),
        "n_blocks": len(blocks),
        "participant_ids": participants,
        "block_ids": blocks,
        "positions": positions,
        "min_position": min(positions) if positions else None,
        "max_position": max(positions) if positions else None,
        "dataset": dataset,
    }


def render_for_request(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    request: SelectionRequest,
    actual_wrap,
    canonical_wrap,
    cap: int = CAP,
) -> Tuple[str, Dict[str, Any]]:
    from utils.teh.pics_v4 import PICS_V4_TRIAL_POLICY, using_pics_v4

    if using_pics_v4():
        from utils.teh.pics_v4_panels import render_slot_panel

        text, stats = render_slot_panel(
            trials,
            dataset=dataset,
            request=request,
            actual_wrap=actual_wrap,
            canonical_wrap=canonical_wrap,
            cap=cap,
            policy_id=PICS_V4_TRIAL_POLICY,
        )
        if int(stats.get("panel_n") or 0) and not int(stats.get("n_supervised") or 0):
            raise TrialWindowDoesNotFitError(
                f"PICS v4 slot panel does not fit a single trial under cap {cap}."
            )
        if int(stats.get("prompt_tokens") or 0) > int(cap):
            raise TrialWindowDoesNotFitError(
                f"PICS v4 prompt is {stats['prompt_tokens']} tokens, over cap {cap}."
            )
        return text, stats
    # Exploration used to rotate block order on each new pass. The cursor now
    # wraps to the same seeded order. vary_exhausted_pass is accepted and ignored.
    vary = False
    state = state_at_step(
        trials,
        dataset=dataset,
        master_seed=request.master_seed,
        phase=request.phase,
        participant_id=request.participant_id,
        step=request.step,
        canonical_wrap=canonical_wrap,
        cap=cap,
        vary_exhausted_pass=vary,
    )
    assigned_text, _assigned_state, assigned_stats = pack_from_state(
        state, wrap=canonical_wrap, cap=cap, dataset=dataset
    )
    text, _advanced, stats = pack_from_state(
        state, wrap=actual_wrap, cap=cap, dataset=dataset
    )
    stats["selection_step"] = int(request.step)
    stats["phase"] = request.phase
    from utils.teh.pics_v4 import PICS_V4_TRIAL_POLICY, using_pics_v4

    stats["policy"] = PICS_V4_TRIAL_POLICY if using_pics_v4() else TRIAL_PROMPT_POLICY_ID
    stats["packing_implementation"] = PACKING_IMPLEMENTATION_ID
    stats["assigned_positions"] = assigned_stats.get("positions")
    stats["assigned_block_ids"] = assigned_stats.get("block_ids")
    stats["assigned_participant_ids"] = assigned_stats.get("participant_ids")
    stats["assigned_supervised"] = assigned_stats.get("n_supervised")
    # Hash the trial body that is actually rendered. Canonical assignment stays
    # on assigned_* so a shared cursor parent does not depend on parent length.
    stats["assigned_text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    stats["canonical_text_sha256"] = hashlib.sha256(assigned_text.encode("utf-8")).hexdigest()
    stats["digest"] = selection_digest(
        master_seed=request.master_seed,
        dataset=dataset,
        phase=request.phase,
        participant_id=request.participant_id,
    )
    return text, stats


def map_generation_phase(phase: str) -> str:
    name = str(phase or "")
    if name in {"global_evolution", "global", "population", "source_population", "target_population"}:
        return PHASE_POPULATION
    if name == "explore":
        return PHASE_EXPLORATION
    if name in {"evolution", "participant_evolution"}:
        return PHASE_PARTICIPANT_EVOLUTION
    return name


def assert_legacy_output_not_resumed(output_dir: Any, *, ignore_dir: Any = None) -> None:
    """Fail closed when a directory already contains the legacy AAMAS prompt policy."""
    from pathlib import Path

    root = Path(output_dir)
    if not root.exists():
        return
    policy_path = root / "TRIAL_PROMPT_POLICY.json"
    if policy_path.is_file():
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        found = str(payload.get("trial_prompt_policy") or payload.get("prompt_policy") or "")
        if str(payload.get("method_version") or "") == "pics_v4":
            raise RuntimeError(
                f"Refusing to resume {root} under {TRIAL_PROMPT_POLICY_ID}; "
                "directory method_version is pics_v4."
            )
        if found != TRIAL_PROMPT_POLICY_ID:
            raise RuntimeError(
                f"Refusing to resume {root} under {TRIAL_PROMPT_POLICY_ID}; "
                f"directory policy is {found!r}."
            )
        recorded_pack = payload.get("packing_implementation")
        if recorded_pack != PACKING_IMPLEMENTATION_ID:
            raise RuntimeError(
                f"Refusing to resume {root}: packing implementation "
                f"{recorded_pack!r} does not match {PACKING_IMPLEMENTATION_ID}."
            )
        _assert_recorded_seeds_match(root, payload)
        return
    legacy_names = ("STAGE_COMPLETE.json", "POPULATION_PROVENANCE.json", "prompt_stats.json")
    ignored = Path(ignore_dir).resolve() if ignore_dir is not None else None
    for path in root.rglob("*"):
        if path.name not in legacy_names or not path.is_file():
            continue
        if ignored is not None and (path.resolve() == ignored or ignored in path.resolve().parents):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        found = str(payload.get("prompt_policy") or "")
        if found in {LEGACY_TRIAL_PROMPT_POLICY_ID, PREVIOUS_COMPACT_POLICY_ID} or (
            found and found != TRIAL_PROMPT_POLICY_ID
        ):
            raise RuntimeError(
                f"Refusing to resume legacy AAMAS output {path} "
                f"(prompt_policy={found!r}) under {TRIAL_PROMPT_POLICY_ID}."
            )
        if "pics_run_seed" in payload or "split_seed" in payload:
            _assert_recorded_seeds_match(root, payload)


def write_trial_policy_marker(output_dir: Any) -> None:
    from pathlib import Path

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "TRIAL_PROMPT_POLICY.json"
    if path.is_file():
        assert_legacy_output_not_resumed(root)
        return
    payload: Dict[str, Any] = {
        "trial_prompt_policy": TRIAL_PROMPT_POLICY_ID,
        "prompt_policy": TRIAL_PROMPT_POLICY_ID,
        "packing_implementation": PACKING_IMPLEMENTATION_ID,
    }
    run_seed = pics_run_seed()
    split_seed = pics_recorded_split_seed()
    if run_seed is not None:
        payload["pics_run_seed"] = int(run_seed)
    if split_seed is not None:
        payload["split_seed"] = int(split_seed)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _assert_recorded_seeds_match(root: Any, payload: Dict[str, Any]) -> None:
    current_run = pics_run_seed()
    current_split = pics_recorded_split_seed()
    if current_run is None and current_split is None:
        return
    recorded_run = payload.get("pics_run_seed")
    recorded_split = payload.get("split_seed")
    if recorded_run is None or recorded_split is None:
        raise RuntimeError(
            f"Refusing to resume {root}: recorded split_seed and pics_run_seed are required."
        )
    if int(recorded_run) != int(current_run):
        raise RuntimeError(
            f"Refusing to resume {root}: pics_run_seed {current_run} "
            f"does not match recorded seed {recorded_run}."
        )
    if int(recorded_split) != int(current_split):
        raise RuntimeError(
            f"Refusing to resume {root}: split_seed {current_split} "
            f"does not match recorded seed {recorded_split}."
        )

