"""Convert existing Psych-101 binary parsed trials to categorical format."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from data_modules.psych101_binary import (
    parse_psych101_binary_row,
    split_psych_experiment,
)


class AdapterError(Exception):
    """Base adapter failure."""


class UnsupportedParserError(AdapterError):
    """No implemented parser/adapter for this experiment."""


class ParseAdapterError(AdapterError):
    """Parser or conversion failed for a row."""


@dataclass
class AdapterStatus:
    experiment_id: str
    alias: Optional[str]
    n_rows_attempted: int
    n_rows_parsed: int
    n_rows_split_ok: int
    parse_errors: List[str]


def _option_dict_for_index(problem: Dict[str, Any], option_keys: List[str], idx: int) -> Dict[str, Any]:
    opt: Dict[str, Any] = {"action": idx}
    if idx < len(option_keys):
        opt["label"] = option_keys[idx]
    if idx == 0 and "gamble_A" in problem:
        opt["gamble"] = problem["gamble_A"]
    if idx == 1 and "gamble_B" in problem:
        opt["gamble"] = problem["gamble_B"]
    for key in ("rewards", "probs", "outcomes", "features", "ratings", "description"):
        pk = f"{key}_{option_keys[idx]}" if idx < len(option_keys) else None
        if pk and pk in problem:
            opt[key] = problem[pk]
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
    for key in (
        "cards",
        "weather_outcome",
        "tree_features",
        "ratings_A",
        "ratings_B",
        "features",
        "stimulus",
    ):
        if key in problem:
            stimulus[key] = problem[key]
    if stimulus:
        problem["stimulus"] = stimulus

    context: Dict[str, Any] = {}
    for key in (
        "game_id",
        "round_id",
        "block_id",
        "schema_type",
        "has_feedback",
        "machine_labels",
    ):
        if key in problem:
            context[key] = problem[key]
    if context:
        problem["context"] = context

    problem["options"] = options
    action = int(trial["action"])
    history = list(trial.get("history") or [])
    return {
        "problem": problem,
        "history": history,
        "action": action,
        "target_action": action,
    }


def convert_binary_trials_to_categorical(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [binary_trial_to_categorical(t) for t in trials]


def parse_row_to_binary_trials(row: Dict[str, Any], alias: str):
    return parse_psych101_binary_row(row, alias)


def pool_categorical_trials_from_rows(
    rows: List[Dict[str, Any]],
    *,
    alias: str,
    experiment_id: str,
    split_ratio: float,
    split_seed: int,
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    AdapterStatus,
]:
    train_all: List[Dict[str, Any]] = []
    val_all: List[Dict[str, Any]] = []
    test_all: List[Dict[str, Any]] = []
    parse_errors: List[str] = []
    n_parsed = 0
    n_split_ok = 0

    for row_idx, row in enumerate(rows):
        try:
            exp = parse_row_to_binary_trials(row, alias)
            n_parsed += 1
            train_b, val_b, test_b, _ = split_psych_experiment(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
            n_split_ok += 1
            train_all.extend(convert_binary_trials_to_categorical(train_b))
            val_all.extend(convert_binary_trials_to_categorical(val_b))
            test_all.extend(convert_binary_trials_to_categorical(test_b))
        except Exception as exc:
            parse_errors.append(f"row {row_idx}: {type(exc).__name__}: {exc}")

    status = AdapterStatus(
        experiment_id=experiment_id,
        alias=alias,
        n_rows_attempted=len(rows),
        n_rows_parsed=n_parsed,
        n_rows_split_ok=n_split_ok,
        parse_errors=parse_errors,
    )
    return train_all, val_all, test_all, status
