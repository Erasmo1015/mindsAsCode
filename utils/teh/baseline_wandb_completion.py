"""Reporting-only W&B completion contract for Centaur and OpenEvolve.

Matches gated T-PICS ``final/*`` semantics. Does not import wandb, and is not
used by fitness, selection, splits, or local scientific CSV writers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

TEST_KEY = "test_loglik"


def finite_loglik(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def row_is_failed(row: Mapping[str, Any], *, test_key: str = TEST_KEY) -> bool:
    status = str(row.get("status") or "").strip().lower()
    if status == "failed":
        return True
    return finite_loglik(row.get(test_key)) is None


def row_is_scored(row: Mapping[str, Any], *, test_key: str = TEST_KEY) -> bool:
    if str(row.get("status") or "").strip().lower() == "failed":
        return False
    return finite_loglik(row.get(test_key)) is not None


def equal_person_mean_test(
    rows: Sequence[Mapping[str, Any]], *, test_key: str = TEST_KEY
) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        number = finite_loglik(row.get(test_key))
        if number is None:
            return None
        values.append(number)
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def wandb_completion_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_n: int,
    run_finished: bool = False,
    distinguish_failures: bool = False,
    test_key: str = TEST_KEY,
) -> Dict[str, Any]:
    """Summary/history fields for the completion contract.

    ``final/mean_test_loglik`` is present only when every expected person has a
    finite test log-likelihood and none are marked failed.
    """
    expected = max(0, int(expected_n))
    attempted = len(rows)
    n_scored = sum(1 for row in rows if row_is_scored(row, test_key=test_key))
    n_failed = sum(1 for row in rows if row_is_failed(row, test_key=test_key))
    is_complete = (
        expected > 0
        and attempted == expected
        and n_scored == expected
        and n_failed == 0
    )
    if is_complete:
        state = "completed"
    elif not run_finished:
        state = "running"
    elif n_failed > 0:
        state = "failed"
    else:
        state = "interrupted"

    fields: Dict[str, Any] = {
        "progress/expected_participants": expected,
        "progress/completed_participants": n_scored,
        "status/state": state,
        "final/is_complete": bool(is_complete),
    }
    if distinguish_failures:
        fields["progress/attempted_participants"] = attempted
        fields["progress/failed_participants"] = n_failed
        fields["progress/scored_participants"] = n_scored
    if is_complete:
        mean_te = equal_person_mean_test(rows, test_key=test_key)
        if mean_te is not None:
            fields["final/mean_test_loglik"] = mean_te
        else:
            fields["final/is_complete"] = False
            fields["status/state"] = "failed" if run_finished else "running"
    return fields


def merge_wandb_payload(
    diagnostic: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> Dict[str, Any]:
    """History payload: diagnostics (including running ``avg_test_loglik``) plus contract."""
    payload = {k: v for k, v in diagnostic.items() if v is not None}
    payload.update(dict(completion))
    return payload


def apply_wandb_payload(wandb_module: Any, payload: Mapping[str, Any]) -> None:
    """Log history and mirror onto summary. Never writes a missing mean as null."""
    if wandb_module is None or not payload:
        return
    wandb_module.log(dict(payload))
    summary = getattr(wandb_module, "summary", None)
    if summary is None:
        return
    for key, value in payload.items():
        summary[key] = value
