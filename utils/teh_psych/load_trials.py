"""Deterministic Psych-101 trial loading from a frozen parse-plan cache (no LLM)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.teh_psych.action_id_normalization import normalize_categorical_trials_action_ids
from utils.teh_psych.dataset_loop import (
    filter_rows_for_experiment,
    load_psych101_split,
    sample_row_indices,
)
from utils.teh_psych.parser_plan import (
    ParsePlanError,
    execute_parser_plan_on_rows,
    load_cached_parse_plan,
    parse_plan_cache_path,
    repair_parser_plan,
    unsupported_pipeline_reason,
    validate_parser_plan,
)
from utils.teh_psych.trial_split import split_pooled_categorical_trials
from utils.teh_psych.trial_validation import (
    partition_pooled_trials,
    summarize_trial_action_space,
    validate_categorical_trials,
)


class CachedParsePlanError(ParsePlanError):
    """Missing or invalid cached parse plan when require_cached is set."""


@dataclass
class ExperimentTrialsBundle:
    experiment_id: str
    plan: Dict[str, Any]
    plan_path: Path
    plan_sha256: str
    rows: List[Dict[str, Any]]
    row_indices: List[int]
    all_trials: List[Dict[str, Any]]
    prediction_trials: List[Dict[str, Any]]
    context_only_trials: List[Dict[str, Any]]
    execution_errors: List[str]
    action_summary: Dict[str, Any]
    train_trials: List[Dict[str, Any]] = field(default_factory=list)
    val_trials: List[Dict[str, Any]] = field(default_factory=list)
    test_trials: List[Dict[str, Any]] = field(default_factory=list)
    sampling_note: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trial_fingerprint(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Stable identity record for loader ↔ teh_psych equality checks."""
    meta = trial.get("_meta") or {}
    problem = trial.get("problem") or {}
    options = []
    for opt in problem.get("options") or []:
        if isinstance(opt, dict):
            options.append(
                {
                    "action": opt.get("action"),
                    "label": opt.get("label"),
                    "raw_key": opt.get("raw_key"),
                }
            )
    history = []
    for h in trial.get("history") or []:
        if isinstance(h, dict):
            history.append({"action": h.get("action"), "feedback": h.get("feedback")})
        else:
            history.append(h)
    return {
        "participant": meta.get("participant"),
        "row_index": meta.get("row_index"),
        "block_id": meta.get("block_id"),
        "target_action": trial.get("target_action", trial.get("action")),
        "action": trial.get("action"),
        "options": options,
        "stimulus": problem.get("stimulus"),
        "history": history,
        "feedback": trial.get("feedback"),
        "is_prediction_target": trial.get("is_prediction_target", True),
    }


def fingerprint_trials(trials: Sequence[Dict[str, Any]]) -> str:
    return sha256_json([trial_fingerprint(t) for t in trials])


def load_and_validate_cached_plan(
    cache_dir: Path,
    experiment_id: str,
    *,
    require_cached: bool = True,
) -> Tuple[Dict[str, Any], Path, str]:
    """
    Load parse_plan.json from cache_dir, repair, and validate.

    Never calls an LLM. Raises CachedParsePlanError when require_cached and the
    plan is missing or invalid.
    """
    cache_dir = Path(cache_dir)
    path = parse_plan_cache_path(cache_dir, experiment_id)
    plan = load_cached_parse_plan(cache_dir, experiment_id)
    if plan is None:
        if require_cached:
            raise CachedParsePlanError(
                f"Required cached parse plan missing for {experiment_id!r} at {path}"
            )
        raise CachedParsePlanError(f"No cached parse plan for {experiment_id!r} at {path}")

    repair_parser_plan(plan)
    validation_errors = validate_parser_plan(plan)
    if validation_errors:
        raise CachedParsePlanError(
            f"Cached parse plan invalid for {experiment_id!r}: {validation_errors[0]}"
        )
    unsupported = unsupported_pipeline_reason(plan)
    if unsupported:
        raise CachedParsePlanError(
            f"Cached parse plan unsupported for {experiment_id!r}: {unsupported}"
        )
    return plan, path, sha256_file(path)


def load_experiment_rows(
    experiment_id: str,
    *,
    split_ds=None,
    psych_dataset_split: str = "train",
    local_dataset: Optional[str] = None,
    max_participants: int = 50,
    range_start_ordinal: Optional[int] = None,
    range_end_ordinal: Optional[int] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
    row_indices: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, Any]], List[int], str]:
    """Load / sample Psych-101 rows for one experiment (same sampling as teh_psych)."""
    if rows is not None:
        indices = list(row_indices) if row_indices is not None else list(range(len(rows)))
        if len(indices) != len(rows):
            raise ValueError("row_indices length must match rows")
        return list(rows), indices, "caller_provided_rows"

    if split_ds is None:
        split_ds = load_psych101_split(psych_dataset_split, local_dataset=local_dataset)
    filtered = filter_rows_for_experiment(split_ds, experiment_id)
    n_rows = len(filtered)
    if n_rows == 0:
        raise ValueError(f"No rows for experiment {experiment_id!r}")
    indices, note = sample_row_indices(
        n_rows,
        max_participants=max_participants,
        range_start_ordinal=range_start_ordinal,
        range_end_ordinal=range_end_ordinal,
    )
    if not indices:
        raise ValueError(f"No rows selected for {experiment_id!r}: {note}")
    selected = [dict(filtered[i]) for i in indices]
    return selected, indices, note


def apply_parse_plan_to_rows(
    plan: Dict[str, Any],
    rows: List[Dict[str, Any]],
    row_indices: List[int],
    *,
    min_pooled_prediction_trials: int = 0,
    show_progress: bool = False,
    progress_desc: str = "parse rows",
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[str],
    Dict[str, Any],
]:
    """Execute plan → normalize → partition prediction trials (teh_psych path)."""
    all_trials, exec_errors = execute_parser_plan_on_rows(
        plan,
        rows,
        row_indices=row_indices,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    all_trials = normalize_categorical_trials_action_ids(all_trials)
    prediction_trials, context_only = partition_pooled_trials(all_trials)
    val_errors, _ = validate_categorical_trials(prediction_trials)
    if val_errors:
        raise ParsePlanError(val_errors[0])
    if min_pooled_prediction_trials and len(prediction_trials) < min_pooled_prediction_trials:
        raise ParsePlanError(
            f"Only {len(prediction_trials)} prediction trials; "
            f"need >= {min_pooled_prediction_trials}"
        )
    action_summary = summarize_trial_action_space(prediction_trials)
    return all_trials, prediction_trials, context_only, exec_errors, action_summary


def load_experiment_trials_from_parse_plan(
    experiment_id: str,
    cache_dir: Path,
    *,
    require_cached: bool = True,
    split_ds=None,
    psych_dataset_split: str = "train",
    local_dataset: Optional[str] = None,
    max_participants: int = 50,
    range_start_ordinal: Optional[int] = None,
    range_end_ordinal: Optional[int] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
    row_indices: Optional[List[int]] = None,
    do_split: bool = True,
    split_ratio: float = 0.8,
    split_seed: int = 42,
    min_pooled_prediction_trials: int = 0,
    show_progress: bool = False,
) -> ExperimentTrialsBundle:
    """
    Load raw rows + frozen parse plan and produce the same categorical trials as teh_psych.

    Never calls an LLM. When ``require_cached`` is True (default), a missing or
    invalid plan raises ``CachedParsePlanError``.
    """
    if not require_cached:
        # Baselines and strict teh_psych paths must not silently regenerate plans.
        raise ValueError(
            "load_experiment_trials_from_parse_plan only supports require_cached=True; "
            "use run_parse_plan_pipeline for LLM generation"
        )

    plan, plan_path, plan_sha = load_and_validate_cached_plan(
        cache_dir, experiment_id, require_cached=True
    )
    selected_rows, indices, note = load_experiment_rows(
        experiment_id,
        split_ds=split_ds,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        max_participants=max_participants,
        range_start_ordinal=range_start_ordinal,
        range_end_ordinal=range_end_ordinal,
        rows=rows,
        row_indices=row_indices,
    )
    all_trials, prediction, context_only, exec_errors, action_summary = apply_parse_plan_to_rows(
        plan,
        selected_rows,
        indices,
        min_pooled_prediction_trials=min_pooled_prediction_trials,
        show_progress=show_progress,
        progress_desc=f"parse {experiment_id}",
    )
    bundle = ExperimentTrialsBundle(
        experiment_id=experiment_id,
        plan=plan,
        plan_path=plan_path,
        plan_sha256=plan_sha,
        rows=selected_rows,
        row_indices=indices,
        all_trials=all_trials,
        prediction_trials=prediction,
        context_only_trials=context_only,
        execution_errors=exec_errors,
        action_summary=action_summary,
        sampling_note=note,
    )
    if do_split:
        bundle.train_trials, bundle.val_trials, bundle.test_trials = (
            split_pooled_categorical_trials(
                prediction,
                split_ratio=split_ratio,
                split_seed=split_seed,
            )
        )
    return bundle
