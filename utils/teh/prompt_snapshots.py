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
