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

TRIAL_PROMPT_POLICY_ID = "aamas_v0_lossless_data_v1"
LEGACY_TRIAL_PROMPT_POLICY_ID = "pics_aamas_v0"
SELECTION_NAMESPACE = "trial_selection"
PHASE_POPULATION = "population"
PHASE_EXPLORATION = "exploration"
PHASE_PARTICIPANT_EVOLUTION = "participant_evolution"
CAP = 14000

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
    raw = trial.get("_prompt_participant_id")
    if raw is None:
        raw = (trial.get("problem") or {}).get("participant_id", 0)
    return int(raw)


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


def render_schedule_text(windows: Sequence[Sequence[Dict[str, Any]]], dataset: str) -> str:
    spec = limited_data_spec(dataset)
    parts = [HISTORY_INTERPRETATION]
    if spec.category == CONTINUOUS_SESSION:
        parts.append(CONTINUOUS_NOTE)
    elif spec.category == INDEPENDENT_TRIAL:
        parts.append(INDEPENDENT_NOTE)
    offset = 0
    for window in windows:
        parts.append(render_window(window, trial_offset=offset))
        offset += len(window)
    return "\n".join(parts)


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
    rr: int = 0
    exhausted_passes: int = 0

    def clone(self) -> "ScheduleState":
        return copy.deepcopy(self)


def _build_state(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    master_seed: int,
    phase: str,
    participant_id: Optional[int],
) -> ScheduleState:
    tagged = [dict(trial) for trial in trials]
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
    return ScheduleState(participant_order=shuffled_pids, queues=queues, rr=0)


def _next_cursor(state: ScheduleState) -> Optional[_Cursor]:
    if not state.participant_order:
        return None
    n = len(state.participant_order)
    for attempt in range(n):
        pid = state.participant_order[(state.rr + attempt) % n]
        queue = state.queues.get(pid) or []
        for cursor in queue:
            if cursor.remaining > 0:
                state.rr = (state.participant_order.index(pid) + 1) % n
                return cursor
    return None


def _take_window(
    cursor: _Cursor,
    existing: Sequence[Sequence[Dict[str, Any]]],
    wrap,
    cap: int,
    dataset: str,
) -> int:
    """How many trials of this cursor fit after the windows already chosen."""
    best = 0
    remaining = cursor.remaining
    for count in range(1, remaining + 1):
        window = cursor.trials[cursor.index : cursor.index + count]
        body = render_schedule_text(list(existing) + [window], dataset)
        if chat_tokens(wrap(body)) <= cap:
            best = count
        else:
            break
    return best


def pack_from_state(
    state: ScheduleState,
    *,
    wrap,
    cap: int,
    dataset: str,
) -> Tuple[str, ScheduleState, Dict[str, Any]]:
    """Pack one prompt. Does not start a new pass inside this prompt."""
    working = state.clone()
    windows: List[List[Dict[str, Any]]] = []
    taken_keys = set()
    while True:
        cursor = _next_cursor(working)
        if cursor is None:
            break
        identity = (cursor.participant_id, cursor.block_id, cursor.index)
        if identity in taken_keys:
            break
        whole = cursor.remaining
        whole_windows = windows + [cursor.trials[cursor.index : cursor.index + whole]]
        whole_body = render_schedule_text(whole_windows, dataset)
        if chat_tokens(wrap(whole_body)) <= cap:
            windows.append(cursor.trials[cursor.index : cursor.index + whole])
            cursor.index += whole
            taken_keys.add(identity)
            continue
        count = _take_window(cursor, windows, wrap, cap, dataset)
        if count <= 0:
            break
        windows.append(cursor.trials[cursor.index : cursor.index + count])
        cursor.index += count
        taken_keys.add(identity)
        break
    text = render_schedule_text(windows, dataset) if windows else ""
    stats = _stats(windows, dataset)
    mismatches = 0
    for window in windows:
        rebuilt = reconstruct_histories(window)
        for trial, history in zip(window, rebuilt):
            if history != list(trial.get("history") or []):
                mismatches += 1
    stats["history_mismatches"] = mismatches
    stats["prompt_tokens"] = chat_tokens(wrap(text)) if text else chat_tokens(wrap(""))
    stats["new_pass_started_inside_prompt"] = 0
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
    return _next_peek_without_moving(state) is None


def _next_peek_without_moving(state: ScheduleState) -> Optional[_Cursor]:
    probe = state.clone()
    return _next_cursor(probe)


def _restart_pass(state: ScheduleState, *, vary: bool = False) -> ScheduleState:
    restarted = state.clone()
    for queue in restarted.queues.values():
        for cursor in queue:
            cursor.index = 0
    restarted.rr = 0
    restarted.exhausted_passes += 1
    if vary:
        pids = restarted.participant_order
        if len(pids) > 1:
            restarted.participant_order = pids[1:] + pids[:1]
        for pid, queue in list(restarted.queues.items()):
            if len(queue) > 1:
                restarted.queues[pid] = queue[1:] + queue[:1]
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
    vary = bool(request.vary_exhausted_pass or request.phase == PHASE_EXPLORATION)
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
    stats["policy"] = TRIAL_PROMPT_POLICY_ID
    stats["assigned_positions"] = assigned_stats.get("positions")
    stats["assigned_block_ids"] = assigned_stats.get("block_ids")
    stats["assigned_participant_ids"] = assigned_stats.get("participant_ids")
    stats["assigned_supervised"] = assigned_stats.get("n_supervised")
    stats["assigned_text_sha256"] = hashlib.sha256(assigned_text.encode("utf-8")).hexdigest()
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


def assert_legacy_output_not_resumed(output_dir: Any) -> None:
    """Fail closed when a directory already contains the legacy AAMAS prompt policy."""
    from pathlib import Path

    root = Path(output_dir)
    if not root.exists():
        return
    policy_path = root / "TRIAL_PROMPT_POLICY.json"
    if policy_path.is_file():
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        found = str(payload.get("trial_prompt_policy") or payload.get("prompt_policy") or "")
        if found != TRIAL_PROMPT_POLICY_ID:
            raise RuntimeError(
                f"Refusing to resume {root} under {TRIAL_PROMPT_POLICY_ID}; "
                f"directory policy is {found!r}."
            )
        _assert_recorded_seeds_match(root, payload)
        return
    legacy_names = ("STAGE_COMPLETE.json", "POPULATION_PROVENANCE.json", "prompt_stats.json")
    for path in root.rglob("*"):
        if path.name not in legacy_names or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        found = str(payload.get("prompt_policy") or "")
        if found == LEGACY_TRIAL_PROMPT_POLICY_ID or (
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

