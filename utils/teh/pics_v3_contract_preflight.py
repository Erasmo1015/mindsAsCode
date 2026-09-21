"""PICS v3 interface-robustness preflight (test-independent).

Generated programs must tolerate admissible ``problem`` / ``history`` variants
derived from the declared interface — including histories that omit optional
outcome fields such as ``feedback`` / ``reward``. This check never uses held-out
test trials and never influences selection via test scores.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from utils.teh.pics_v3_observed import OPTIONAL_HISTORY_OUTCOME_FIELDS
from utils.teh.prompt_snapshots import sanitize_problem_for_choose

# History keys that may appear on some trials / stages but are not required.
OPTIONAL_HISTORY_FIELDS = frozenset(OPTIONAL_HISTORY_OUTCOME_FIELDS) | frozenset(
    {
        "planet",
        "alien_options",
        "stage1_action",
        "spaceship",
        "probe_in_set",
    }
)


def strip_optional_history_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only non-optional history fields (typically ``action``)."""
    out: Dict[str, Any] = {}
    for key, value in entry.items():
        if key in OPTIONAL_HISTORY_FIELDS:
            continue
        out[key] = copy.deepcopy(value)
    if "action" not in out and "action" in entry:
        out["action"] = entry["action"]
    return out


def null_optional_history_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve keys but set optional outcome fields to None (genuine null)."""
    out = dict(entry)
    for key in OPTIONAL_HISTORY_OUTCOME_FIELDS:
        if key in out:
            out[key] = None
    return out


def build_admissible_history_variants(
    history: Any,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Synthetic admissible histories — not copies of held-out test trials."""
    variants: List[Tuple[str, List[Dict[str, Any]]]] = [("empty_history", [])]
    if not isinstance(history, list) or not history:
        return variants
    cleaned: List[Dict[str, Any]] = []
    for raw in history:
        if not isinstance(raw, dict):
            continue
        cleaned.append(dict(raw))
    if not cleaned:
        return variants
    variants.append(("as_observed", copy.deepcopy(cleaned)))
    action_only = [strip_optional_history_fields(e) for e in cleaned]
    variants.append(("action_only_no_outcomes", action_only))
    # Single-step action-only (common when delayed feedback has not arrived).
    last = strip_optional_history_fields(cleaned[-1])
    variants.append(("last_action_only", [last]))
    # Explicitly drop feedback/reward even if other optional keys remain.
    no_fb: List[Dict[str, Any]] = []
    for entry in cleaned:
        e = dict(entry)
        for key in ("feedback", "reward"):
            e.pop(key, None)
        no_fb.append(e)
    variants.append(("missing_feedback_and_reward", no_fb))
    # Null optional outcomes (key present, value None) — not placeholder fill-in.
    null_opt = [null_optional_history_fields(e) for e in cleaned]
    variants.append(("null_optional_outcomes", null_opt))
    # Heterogeneous entries within one history: mix action-only + full + null.
    if len(cleaned) >= 1:
        hetero: List[Dict[str, Any]] = []
        for i, entry in enumerate(cleaned):
            if i % 3 == 0:
                hetero.append(strip_optional_history_fields(entry))
            elif i % 3 == 1:
                e = dict(entry)
                e.pop("feedback", None)
                e.pop("reward", None)
                hetero.append(e)
            else:
                hetero.append(null_optional_history_fields(entry))
        if len(hetero) == 1:
            # Force heterogeneity with an extra action-only step.
            hetero = [strip_optional_history_fields(cleaned[0]), dict(cleaned[0])]
            hetero[1].pop("feedback", None)
            hetero[1].pop("reward", None)
        variants.append(("heterogeneous_entries", hetero))
    return variants


def build_admissible_preflight_cases(
    observed_trials: Sequence[Dict[str, Any]],
    *,
    max_problems: int = 4,
) -> List[Dict[str, Any]]:
    """Build (problem, history) cases from observed-union trials only.

    Problems are sanitized (required dataset/stage fields kept; oracle/outcome
    leak keys stripped). Histories are synthetic admissible variants — optional
    fields may be absent or null; required task fields stay on the problem.
    """
    cases: List[Dict[str, Any]] = []
    seen_problems = set()
    for trial in observed_trials:
        problem = sanitize_problem_for_choose(trial.get("problem") or {})
        fingerprint = json_fingerprint(problem)
        if fingerprint in seen_problems:
            continue
        seen_problems.add(fingerprint)
        history = trial.get("history") or []
        for label, hist in build_admissible_history_variants(history):
            cases.append(
                {
                    "label": label,
                    "problem": problem,
                    "history": hist,
                }
            )
        # Required problem fields with empty history (stage/schema preserved).
        cases.append(
            {
                "label": "required_problem_fields_empty_history",
                "problem": problem,
                "history": [],
            }
        )
        if len(seen_problems) >= max_problems:
            break
    if not cases:
        # Minimal binary interface with empty history.
        cases.append(
            {
                "label": "empty_default",
                "problem": {"option_keys": ["A", "B"]},
                "history": [],
            }
        )
        cases.append(
            {
                "label": "action_only_default",
                "problem": {"option_keys": ["A", "B"]},
                "history": [{"action": 0}],
            }
        )
        cases.append(
            {
                "label": "heterogeneous_default",
                "problem": {"option_keys": ["A", "B"]},
                "history": [
                    {"action": 0},
                    {"action": 1, "feedback": None},
                    {"action": 0, "feedback": 1.0},
                ],
            }
        )
    return cases


def json_fingerprint(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def run_interface_contract_preflight(
    choose_fn: Callable,
    observed_trials: Sequence[Dict[str, Any]],
    *,
    max_problems: int = 4,
) -> Dict[str, Any]:
    """Return ``ok=False`` if ``choose`` raises on any admissible synthetic case.

    Does not inspect test scores. Exceptions are recorded for error feedback.
    Does not inject placeholder values into observed trials.
    """
    cases = build_admissible_preflight_cases(
        observed_trials, max_problems=max_problems
    )
    failures: List[Dict[str, Any]] = []
    for case in cases:
        try:
            choose_fn(case["problem"], case["history"])
        except Exception as exc:
            failures.append(
                {
                    "label": case["label"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
            )
    return {
        "ok": len(failures) == 0,
        "n_cases": len(cases),
        "failures": failures,
        "test_trials_used": False,
        "placeholders_injected": False,
    }


def preflight_invalidates_candidate(
    choose_fn: Optional[Callable],
    observed_trials: Sequence[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    """Convenience: ``(is_invalid, report)``."""
    if choose_fn is None:
        return True, {
            "ok": False,
            "n_cases": 0,
            "failures": [
                {
                    "label": "compile",
                    "error_type": "CompileError",
                    "error_message": "none",
                }
            ],
            "test_trials_used": False,
            "placeholders_injected": False,
        }
    report = run_interface_contract_preflight(choose_fn, observed_trials)
    return (not bool(report.get("ok"))), report
