"""Remap categorical trial action ids to consecutive 0..K-1."""
from __future__ import annotations

from typing import Any, Dict, List


class ActionIdNormalizationError(ValueError):
    """Raised when target_action cannot be remapped."""


def _consecutive_zero_based(action_ids: List[int]) -> bool:
    if not action_ids:
        return True
    return sorted(action_ids) == list(range(len(action_ids)))


def normalize_trial_action_ids(trial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap problem option actions and trial target/history to 0..K-1.

    Preserves prior ids in option ``raw_action`` and label as ``raw_key`` when absent.
    Idempotent when actions are already consecutive from 0.
    """
    problem = trial.get("problem")
    if not isinstance(problem, dict):
        return trial
    options = problem.get("options")
    if not isinstance(options, list) or not options:
        return trial

    old_ids: List[int] = []
    for j, opt in enumerate(options):
        if not isinstance(opt, dict) or "action" not in opt:
            raise ActionIdNormalizationError(
                f"trial option {j} missing action id"
            )
        old_ids.append(int(opt["action"]))

    if _consecutive_zero_based(old_ids):
        for opt in options:
            if isinstance(opt, dict):
                if "raw_action" not in opt:
                    opt["raw_action"] = int(opt["action"])
                if opt.get("label") is not None and "raw_key" not in opt:
                    opt["raw_key"] = str(opt["label"])
        return trial

    id_map = {old: new for new, old in enumerate(old_ids)}
    if len(set(old_ids)) != len(old_ids):
        raise ActionIdNormalizationError(
            f"duplicate option action ids: {old_ids}"
        )

    for new_id, opt in enumerate(options):
        old_action = int(opt["action"])
        if "raw_action" not in opt:
            opt["raw_action"] = old_action
        if opt.get("label") is not None and "raw_key" not in opt:
            opt["raw_key"] = str(opt["label"])
        opt["action"] = new_id

    def _remap_action(value: Any, field_name: str) -> int:
        if value is None:
            raise ActionIdNormalizationError(f"missing {field_name} for remap")
        iv = int(value)
        if iv not in id_map:
            raise ActionIdNormalizationError(
                f"{field_name} {iv} not in option action ids {old_ids}"
            )
        return id_map[iv]

    for field in ("action", "target_action"):
        if field in trial:
            trial[field] = _remap_action(trial[field], field)

    for entry in trial.get("history") or []:
        if isinstance(entry, dict) and "action" in entry:
            entry["action"] = _remap_action(entry["action"], "history.action")

    meta = trial.setdefault("_meta", {})
    if isinstance(meta, dict):
        if "raw_key" in meta and meta.get("raw_key") is not None:
            rk = str(meta["raw_key"]).upper()
            for opt in options:
                if str(opt.get("raw_key", opt.get("label", ""))).upper() == rk:
                    meta["raw_action"] = opt.get("raw_action", opt["action"])
                    break

    return trial


def normalize_categorical_trials_action_ids(
    trials: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [normalize_trial_action_ids(t) for t in trials]
