"""Compact trial lines that round-trip the sanitized problem and history.

The seeded block cursor lives in ``aamas_v0_lossless_trials``. This module only
renders and parses the trial text. It does not add synthetic trial ids,
positions, or reconstruction instructions.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence

from utils.teh.prompt_snapshots import _CURRENT_OUTCOME_KEYS, sanitize_problem_for_choose

POLICY_ID = "compact_faithful_trial_v1"


def _canon(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canon(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canon(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item) and type(value).__module__ == "numpy":
        return _canon(item())
    return value


def _dumps(value: Any) -> str:
    return json.dumps(_canon(value), separators=(",", ":"), default=str)


def _pid(trial: Dict[str, Any]) -> int:
    raw = trial.get("_prompt_participant_id")
    if raw is None:
        raw = (trial.get("problem") or {}).get("participant_id")
    if raw is None:
        raise ValueError("Trial has no participant identity.")
    return int(raw)


def history_delta(prev: Dict[str, Any], nxt: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Entries that appear in the next retained history. None when there is no later trial."""
    if nxt is None or not isinstance(nxt.get("history"), list):
        return None
    prev_hist = prev.get("history") if isinstance(prev.get("history"), list) else []
    nxt_hist = nxt["history"]
    if len(nxt_hist) <= len(prev_hist):
        return None
    return [copy.deepcopy(item) for item in nxt_hist[len(prev_hist) :]]


def _real_next(trial: Dict[str, Any], nxt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    real_next = trial.get("_real_next")
    if real_next is None and nxt is not None and nxt.get("_block_id") == trial.get("_block_id"):
        real_next = nxt
    return real_next if isinstance(real_next, dict) else None


def render_trial_record(
    trial: Dict[str, Any],
    *,
    nxt: Optional[Dict[str, Any]],
    include_history_before: bool,
) -> Dict[str, Any]:
    problem = _canon(sanitize_problem_for_choose(trial.get("problem") or {}))
    leaked = sorted(set(problem) & set(_CURRENT_OUTCOME_KEYS))
    if leaked:
        raise RuntimeError(
            "Pre-choice problem still contains outcome fields: " + ", ".join(leaked)
        )
    record: Dict[str, Any] = {"problem": problem, "observed_action": trial.get("action")}
    if include_history_before:
        record = {
            "history_before": _canon(list(trial.get("history") or [])),
            **record,
        }
    delta = history_delta(trial, _real_next(trial, nxt))
    if delta:
        record["outcome"] = _canon(delta)
    return record


def render_compact_schedule(windows: Sequence[Sequence[Dict[str, Any]]]) -> str:
    """One JSON object per trial, with participant and reset headers between blocks."""
    lines: List[str] = []
    previous_pid: Optional[int] = None
    for window in windows:
        if not window:
            continue
        pid = _pid(window[0])
        if pid != previous_pid:
            lines.append(f"participant {pid}")
            previous_pid = pid
        history = list(window[0].get("history") or [])
        if not history:
            lines.append("history resets")
        for offset, trial in enumerate(window):
            nxt = window[offset + 1] if offset + 1 < len(window) else None
            record = render_trial_record(
                trial,
                nxt=nxt,
                include_history_before=offset == 0 and bool(history),
            )
            lines.append(json.dumps(record, separators=(",", ":"), default=str))
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def parse_compact_schedule(text: str) -> List[Dict[str, Any]]:
    """Rebuild each displayed trial's problem, history, and action from the text."""
    history: List[Any] = []
    rows: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("participant "):
            history = []
            continue
        if line == "history resets":
            history = []
            continue
        record = json.loads(line)
        if "history_before" in record:
            history = list(record["history_before"])
        rows.append(
            {
                "problem": _canon(record.get("problem") or {}),
                "history": _canon(list(history)),
                "action": record.get("observed_action"),
            }
        )
        outcome = record.get("outcome")
        if isinstance(outcome, list) and outcome:
            history = list(history) + list(outcome)
    return rows


def round_trip_mismatches(text: str, trials: Sequence[Dict[str, Any]]) -> int:
    parsed = parse_compact_schedule(text)
    if len(parsed) != len(trials):
        return max(len(parsed), len(trials))
    mismatches = 0
    for record, trial in zip(parsed, trials):
        problem = _canon(sanitize_problem_for_choose(trial.get("problem") or {}))
        history = _canon(list(trial.get("history") or []))
        if (
            record["problem"] != problem
            or record["history"] != history
            or record["action"] != trial.get("action")
        ):
            mismatches += 1
        if set(record["problem"]) & set(_CURRENT_OUTCOME_KEYS):
            mismatches += 1
    return mismatches
