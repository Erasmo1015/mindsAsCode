"""Convert existing Psych-101 binary parsed trials to categorical format."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from data_modules.psych101_binary import (
    experiment_to_trial_dicts,
    parse_psych101_binary_row,
)


class AdapterError(Exception):
    """Base adapter failure."""


class UnsupportedParserError(AdapterError):
    """No implemented parser/adapter for this experiment."""


class ParseAdapterError(AdapterError):
    """Parser or conversion failed for a row."""


@dataclass
class ManualParserStatus:
    experiment_id: str
    alias: Optional[str]
    n_rows_attempted: int
    n_rows_parsed: int
    parse_errors: List[str]


def _option_dict_for_index(problem: Dict[str, Any], option_keys: List[str], idx: int) -> Dict[str, Any]:
    opt: Dict[str, Any] = {"action": idx}
    if idx < len(option_keys):
        opt["label"] = option_keys[idx]
    if idx == 0 and "gamble_A" in problem:
        opt["gamble"] = problem["gamble_A"]
    if idx == 1 and "gamble_B" in problem:
        opt["gamble"] = problem["gamble_B"]
    return opt


def binary_trial_to_categorical(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one binary TEH trial dict to categorical API format."""
    problem = dict(trial.get("problem") or {})
    option_keys = problem.get("option_keys")
    if not isinstance(option_keys, list) or not option_keys:
        option_keys = trial.get("options")
    if not isinstance(option_keys, list) or not option_keys:
        option_keys = ["0", "1"]
    K = len(option_keys)
    options = [_option_dict_for_index(problem, option_keys, i) for i in range(K)]

    stimulus: Dict[str, Any] = {}
    for key in ("cards", "weather_outcome", "tree_features", "ratings_A", "ratings_B", "features", "stimulus"):
        if key in problem:
            stimulus[key] = problem[key]
    if stimulus:
        problem["stimulus"] = stimulus

    context: Dict[str, Any] = {}
    for key in ("game_id", "round_id", "block_id", "schema_type", "has_feedback", "machine_labels"):
        if key in problem:
            context[key] = problem[key]
    if context:
        problem["context"] = context

    problem["options"] = options
    action = int(trial["action"])
    return {
        "problem": problem,
        "history": list(trial.get("history") or []),
        "action": action,
        "target_action": action,
        "feedback": trial.get("feedback"),
        "is_prediction_target": True,
        "_meta": {"adapter": "manual_fallback"},
    }


def convert_binary_trials_to_categorical(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [binary_trial_to_categorical(t) for t in trials]


def pool_manual_parser_trials_from_rows(
    rows: List[Dict[str, Any]],
    *,
    alias: str,
    experiment_id: str,
    row_indices: Optional[List[int]] = None,
) -> tuple[List[Dict[str, Any]], ManualParserStatus]:
    """Optional fallback: existing manual binary parsers, no per-row HF split."""
    indices = row_indices if row_indices is not None else list(range(len(rows)))
    all_trials: List[Dict[str, Any]] = []
    parse_errors: List[str] = []
    n_parsed = 0
    for row_idx, row in zip(indices, rows):
        try:
            exp = parse_psych101_binary_row(row, alias)
            n_parsed += 1
            binary = experiment_to_trial_dicts(
                exp, dataset_alias=alias, experiment_id=experiment_id
            )
            for t in convert_binary_trials_to_categorical(binary):
                meta = dict(t.get("_meta") or {})
                meta["row_index"] = row_idx
                t["_meta"] = meta
                all_trials.append(t)
        except Exception as exc:
            parse_errors.append(f"row {row_idx}: {type(exc).__name__}: {exc}")

    status = ManualParserStatus(
        experiment_id=experiment_id,
        alias=alias,
        n_rows_attempted=len(rows),
        n_rows_parsed=n_parsed,
        parse_errors=parse_errors,
    )
    return all_trials, status
