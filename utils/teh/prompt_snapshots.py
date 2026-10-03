"""ICLR T-PICS v2 candidate-example snapshots and shared sanitizer.

v1 still uses ``format_trial_for_prompt``. v2 candidate prompts serialize
complete sanitized problem dicts with bounded head+tail history.
Automatic dataset-prompt generation shares ``sanitize_problem_for_choose``.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.teh.prompt_context import (
    DEFAULT_HISTORY_MAX_ENTRIES,
    _truncate_history_entries,
)

PROMPT_PARTICIPANT_KEY = "_prompt_participant_id"

_CURRENT_OUTCOME_KEYS = frozenset(
    {
        "reward",
        "treasure",
        "outcome",
        "outcome_marker",
        "exploded",
        "was_correct",
        "weather_outcome",
        "correct_category",
        "response_key",
        "feedback",
        "points_received",
        "probe_in_set",
    }
)
_KOOL_STAGE1_UNOBSERVED = frozenset(
    {
        "planet",
        "alien_options",
        "stage1_action",
        "spaceship",
        "reward",
        "treasure",
    }
)

_tls = threading.local()


def using_v2_prompt_contract() -> bool:
    return bool(getattr(_tls, "v2", False))


class prompt_contract_scope:
    """Thread-local v2 prompt-contract flag for T-PICS serialization."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._prev = False

    def __enter__(self) -> "prompt_contract_scope":
        self._prev = using_v2_prompt_contract()
        _tls.v2 = self.enabled
        return self

    def __exit__(self, *exc: object) -> None:
        _tls.v2 = self._prev


def stamp_prompt_participant_id(trial: Dict[str, Any], participant_id: int) -> Dict[str, Any]:
    out = dict(trial)
    out[PROMPT_PARTICIPANT_KEY] = int(participant_id)
    return out


def prompt_participant_id(trial: Dict[str, Any]) -> Optional[int]:
    raw = trial.get(PROMPT_PARTICIPANT_KEY)
    if raw is None:
        raw = trial.get("participant_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def sanitize_problem_for_choose(problem: Any) -> Dict[str, Any]:
    """Observable choice-time problem dict. History is not modified here."""
    if not isinstance(problem, dict):
        return {}
    out = dict(problem)
    alias = str(out.get("dataset_alias") or "")
    schema = str(out.get("schema_type") or "")
    for key in list(out):
        if key in _CURRENT_OUTCOME_KEYS:
            out.pop(key, None)
        kl = str(key).lower()
        if kl in {"action", "choice", "chosen", "response"}:
            out.pop(key, None)
    if schema == "kool_twostep" or alias == "14kool2016when":
        stage = int(out.get("stage") or 1)
        if stage == 1:
            for key in _KOOL_STAGE1_UNOBSERVED:
                out.pop(key, None)
        else:
            out.pop("reward", None)
            out.pop("treasure", None)
    if alias == "11enkavi2019recentprobes":
        out.pop("probe_in_set", None)
    return out


def sanitize_trial_snapshot(trial: Dict[str, Any]) -> Dict[str, Any]:
    nt = dict(trial)
    nt["problem"] = sanitize_problem_for_choose(trial.get("problem") or {})
    return nt


def current_or_future_leak_paths(trial: Dict[str, Any]) -> List[str]:
    """Paths where the current action/outcome is visible in choose() inputs."""
    leaks: List[str] = []
    action = trial.get("action")
    problem = sanitize_problem_for_choose(trial.get("problem") or {})
    if action is not None:
        for key in ("action", "response_key", "choice", "chosen", "response"):
            if key in problem and problem[key] == action:
                leaks.append(f"problem.{key}")
    for key in _CURRENT_OUTCOME_KEYS:
        if key in problem:
            leaks.append(f"problem.{key}")
    return leaks


def snapshot_example_dict(
    trial: Dict[str, Any],
    index: int,
    *,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
) -> Dict[str, Any]:
    sanitized = sanitize_trial_snapshot(trial)
    problem = sanitized.get("problem") or {}
    hist, was_trunc, orig_len = _truncate_history_entries(
        trial.get("history") or [],
        max_entries=history_max_entries,
    )
    ldp = trial.get("_ldp") or {}
    pid = prompt_participant_id(trial)
    identity = {
        "participant_id": pid,
        "unit_id": ldp.get("unit_id"),
        "chrono": ldp.get("chrono", ldp.get("session_index")),
        "origin_split": ldp.get("origin_split"),
        "origin_index": ldp.get("origin_index"),
        "block_index": problem.get("block_index"),
        "presented_day": problem.get("presented_day"),
        "balloon_id": problem.get("balloon_id"),
        "rule_block_id": problem.get("rule_block_id"),
        "game": problem.get("game"),
        "round": problem.get("round") or problem.get("round_id"),
        "problem_id": problem.get("problem_id"),
        "stage": problem.get("stage"),
    }
    identity = {k: v for k, v in identity.items() if v is not None}
    out: Dict[str, Any] = {
        "index": index,
        "session_boundary": {
            "participant_id": pid,
            "dataset_alias": problem.get("dataset_alias"),
        },
        "identity": identity,
        "problem": problem,
        "history": hist,
        "label": {"action": trial.get("action")},
    }
    if was_trunc:
        out["history_truncated"] = True
        out["history_original_len"] = orig_len
        out["history_entries_retained"] = len(hist) if isinstance(hist, list) else 0
        out["history_truncation_note"] = (
            f"Displayed {len(hist)} of {orig_len} history entries "
            "(earliest + most recent; middle entries omitted); "
            "each shown entry is complete."
        )
    return out


def format_snapshot_example(
    trial: Dict[str, Any],
    index: int,
    *,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
) -> str:
    payload = snapshot_example_dict(
        trial, index, history_max_entries=history_max_entries
    )
    body = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    pid = prompt_participant_id(trial)
    header = f"### example {index}"
    if pid is not None:
        header += f" | participant={pid}"
    label = payload.get("label") or {}
    return (
        f"{header}\n"
        f"{body}\n"
        f"observed_action_label={label.get('action')!r} "
        "(target for this example; not passed into choose())"
    )


SPARSE_HISTORY_POLICY_ID = "structured_snapshot_sparse_history_v1"
PROMPT_ONLY_METADATA_PREFIX = (
    "[PROMPT_ONLY_METADATA — NOT AVAILABLE TO choose(problem, history)]"
)
_PROMPT_ONLY_KEYS = frozenset(
    {
        "PROMPT_ONLY_METADATA",
        "history_entries_omitted",
        "actual_runtime_history_length",
        "displayed_history_entries",
        "history_truncated",
        "history_original_len",
        "history_entries_retained",
        "history_truncation_note",
    }
)
_BLOCK_POS: Dict[int, Tuple[Any, int]] = {}


def remember_trial_block_position(trial: Dict[str, Any], *, unit: Any, offset: int) -> None:
    """Record the chronological offset of this trial object inside its block."""
    _BLOCK_POS[id(trial)] = (unit, int(offset))


def clear_trial_block_positions() -> None:
    _BLOCK_POS.clear()


def _block_position(trial: Dict[str, Any], index: int) -> Tuple[Any, int]:
    found = _BLOCK_POS.get(id(trial))
    if found is not None:
        return found
    return (id(trial), int(index))


def contiguous_block_segment_ends(trials: Sequence[Dict[str, Any]]) -> List[int]:
    """Last index of each run of consecutive within-block offsets."""
    if not trials:
        return []
    ends: List[int] = []
    prev: Optional[Tuple[Any, int]] = None
    for index, trial in enumerate(trials):
        pos = _block_position(trial, index)
        if prev is not None and not (pos[0] == prev[0] and pos[1] == prev[1] + 1):
            ends.append(index - 1)
        prev = pos
    ends.append(len(trials) - 1)
    return ends


def _runtime_history(
    trial: Dict[str, Any],
    *,
    history_max_entries: int,
) -> Tuple[Any, bool, int, str]:
    """Return display history, whether it was capped, the real length, and kind."""
    if "history" not in trial or trial.get("history") is None:
        return None, False, 0, "none"
    raw = trial.get("history")
    if raw == []:
        return [], False, 0, "empty"
    if isinstance(raw, list):
        shown, was_trunc, orig_len = _truncate_history_entries(
            raw, max_entries=history_max_entries
        )
        return shown, bool(was_trunc), int(orig_len), "nonempty"
    length = len(raw) if hasattr(raw, "__len__") else 1
    return raw, False, int(length), "nonempty"


def _assert_runtime_object_has_no_prompt_metadata(payload: Dict[str, Any]) -> None:
    dumped = json.dumps(payload, ensure_ascii=False, default=str)

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key) in _PROMPT_ONLY_KEYS or "PROMPT_ONLY" in str(key):
                    raise RuntimeError(
                        f"prompt-only metadata {key!r} is inside a runtime JSON object"
                    )
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(payload)
    if "PROMPT_ONLY_METADATA" in dumped or "history_entries_omitted" in dumped:
        raise RuntimeError("prompt-only metadata text is inside a runtime JSON object")


def _sparse_payload(
    trial: Dict[str, Any],
    index: int,
    history: Any,
    *,
    include_history: bool,
) -> Dict[str, Any]:
    sanitized = sanitize_trial_snapshot(trial)
    problem = sanitized.get("problem") or {}
    ldp = trial.get("_ldp") or {}
    pid = prompt_participant_id(trial)
    identity = {
        "participant_id": pid,
        "unit_id": ldp.get("unit_id"),
        "chrono": ldp.get("chrono", ldp.get("session_index")),
        "origin_split": ldp.get("origin_split"),
        "origin_index": ldp.get("origin_index"),
        "block_index": problem.get("block_index"),
        "presented_day": problem.get("presented_day"),
        "balloon_id": problem.get("balloon_id"),
        "rule_block_id": problem.get("rule_block_id"),
        "game": problem.get("game"),
        "round": problem.get("round") or problem.get("round_id"),
        "problem_id": problem.get("problem_id"),
        "stage": problem.get("stage"),
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    out: Dict[str, Any] = {
        "index": index,
        "session_boundary": {
            "participant_id": pid,
            "dataset_alias": problem.get("dataset_alias"),
        },
        "identity": identity,
        "problem": problem,
    }
    if include_history:
        out["history"] = history
    out["label"] = {"action": trial.get("action")}
    _assert_runtime_object_has_no_prompt_metadata(out)
    return out


def format_sparse_history_example(
    trial: Dict[str, Any],
    index: int,
    *,
    show_history: bool,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
) -> str:
    """One example. Nonempty history is omitted except on a segment checkpoint."""
    shown, was_trunc, actual_len, kind = _runtime_history(
        trial, history_max_entries=history_max_entries
    )
    include_history = kind != "nonempty" or show_history
    payload = _sparse_payload(
        trial, index, shown, include_history=include_history
    )
    meta = ""
    if kind == "nonempty" and not show_history:
        meta = (
            f"{PROMPT_ONLY_METADATA_PREFIX} history_entries_omitted=true "
            f"actual_runtime_history_length={actual_len}"
        )
    elif show_history and was_trunc:
        displayed = len(shown) if isinstance(shown, list) else 0
        meta = (
            f"{PROMPT_ONLY_METADATA_PREFIX} history_entries_omitted=false "
            f"actual_runtime_history_length={actual_len} "
            f"displayed_history_entries={displayed}"
        )
    body = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    pid = prompt_participant_id(trial)
    header = f"### example {index}"
    if pid is not None:
        header += f" | participant={pid}"
    label = payload.get("label") or {}
    parts = [header]
    if meta:
        parts.append(meta)
    parts.append(body)
    parts.append(
        f"observed_action_label={label.get('action')!r} "
        "(target for this example; not passed into choose())"
    )
    return "\n".join(parts)


def format_sparse_history_examples(
    trials: Sequence[Dict[str, Any]],
    *,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
) -> str:
    """Live PICS v4 serializer. Omission metadata stays outside runtime JSON."""
    if not trials:
        return ""
    checkpoints = set(contiguous_block_segment_ends(trials))
    blocks: List[str] = []
    prev_pid: Optional[int] = object()  # type: ignore[assignment]
    for index, trial in enumerate(trials, start=1):
        pid = prompt_participant_id(trial)
        if pid != prev_pid:
            blocks.append(f"===== participant {pid} =====")
            prev_pid = pid
        blocks.append(
            format_sparse_history_example(
                trial,
                index,
                show_history=(index - 1) in checkpoints,
                history_max_entries=history_max_entries,
            )
        )
    return "\n\n".join(blocks)


def format_snapshot_examples(
    trials: Sequence[Dict[str, Any]],
    *,
    history_max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
) -> str:
    if not trials:
        return ""
    blocks: List[str] = []
    prev_pid: Optional[int] = object()  # type: ignore[assignment]
    for i, trial in enumerate(trials, start=1):
        pid = prompt_participant_id(trial)
        if pid != prev_pid:
            blocks.append(f"===== participant {pid} =====")
            prev_pid = pid
        blocks.append(
            format_snapshot_example(
                trial, i, history_max_entries=history_max_entries
            )
        )
    return "\n\n".join(blocks)


def assert_no_test_trials(trials: Iterable[Dict[str, Any]], *, label: str) -> None:
    for i, trial in enumerate(trials):
        origin = (trial.get("_ldp") or {}).get("origin_split")
        if origin == "test":
            raise AssertionError(f"{label} contains a test trial at index {i}")
