"""Golden tests for legacy/new TEH error-feedback mode isolation."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Set

import teh
from prototype.teh_psych import build_argparser


def _historical_select(
    error_store: List[Dict[str, Any]], *, iteration: Optional[int]
) -> List[Dict[str, Any]]:
    """Verbatim selection logic from teh.py at commit 2daa9da."""
    if not error_store:
        return []
    if iteration is None:
        recency_floor = -10**9
    else:
        recency_floor = int(iteration) - 3 + 1
    recent = [
        item
        for item in error_store
        if int(item.get("last_seen_iteration", -10**9)) >= recency_floor
    ]
    recent.sort(
        key=lambda x: (
            -int(x.get("last_seen_iteration", -10**9)),
            -int(x.get("count", 0)),
        )
    )
    selected: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    for item in recent:
        key = str(item.get("normalized_key") or "")
        if key and key not in seen_keys:
            selected.append(item)
            seen_keys.add(key)
        if len(selected) >= 8:
            return selected
    frequent = sorted(
        error_store,
        key=lambda x: (
            -int(x.get("count", 0)),
            -int(x.get("last_seen_iteration", -10**9)),
        ),
    )
    kept = 0
    for item in frequent:
        key = str(item.get("normalized_key") or "")
        if not key or key in seen_keys:
            continue
        selected.append(item)
        seen_keys.add(key)
        kept += 1
        if kept >= 3 or len(selected) >= 8:
            break
    return selected[:8]


def _historical_prompt(
    error_store: List[Dict[str, Any]],
    *,
    iteration: Optional[int],
    max_error_prompt_chars: int,
) -> str:
    """Verbatim prompt construction from teh.py at commit 2daa9da."""
    if not error_store:
        return ""
    max_chars = int(max_error_prompt_chars)
    if max_chars <= 0:
        return ""
    selected = _historical_select(error_store, iteration=iteration)
    if not selected:
        return ""
    items: List[str] = []
    used = 0
    for entry in selected:
        line = str(entry.get("invalid_line") or "").strip()
        err_type = str(entry.get("error_type") or "").strip()
        err_msg = str(entry.get("error_message") or "").strip()
        if not err_type and not err_msg:
            continue
        err = f"{err_type}: {err_msg}" if err_msg else err_type
        if line:
            item = f"- Line: {line}\n  Error: {err}"
        else:
            item = f"- Error: {err}"
        projected = used + len(item) + (2 if items else 0)
        if projected > max_chars and items:
            break
        if projected > max_chars:
            continue
        items.append(item)
        used = projected
    if not items:
        return ""
    past_error_summary = "\n\n".join(items)
    return (
        "Past invalid-program errors to avoid:\n"
        "The following are previous invalid generated-program mistakes. Do not repeat them. "
        "Each item shows the invalid line, when available, and the error it caused.\n\n"
        f"{past_error_summary}"
    )


def _entry(key: str, *, last: int, count: int, size: int = 1) -> Dict[str, Any]:
    return {
        "normalized_key": key,
        "invalid_line": f"x_{key} = {'x' * size}",
        "error_type": "ValueError",
        "error_message": f"message-{key}",
        "count": count,
        "first_seen_iteration": 1,
        "last_seen_iteration": last,
    }


def test_legacy_selection_exactly_matches_historical_oracle() -> None:
    store = [
        _entry("old-most-frequent", last=2, count=100),
        _entry("recent-low", last=9, count=1),
        _entry("recent-high", last=9, count=7),
        _entry("boundary", last=8, count=3),
        _entry("too-old", last=7, count=50),
        _entry("old-second", last=4, count=80),
        _entry("old-third", last=3, count=70),
        _entry("old-fourth", last=6, count=60),
    ]
    expected = _historical_select(copy.deepcopy(store), iteration=10)
    actual = teh._select_errors_for_prompt(
        copy.deepcopy(store), iteration=10, error_feedback_mode="legacy"
    )
    assert actual == expected
    assert [x["normalized_key"] for x in actual] == [
        "recent-high",
        "recent-low",
        "boundary",
        "old-most-frequent",
        "old-second",
        "old-third",
    ]


def test_legacy_item_limit_and_tie_order_exactly_match_history() -> None:
    store = [_entry(f"k{i}", last=9, count=1) for i in range(12)]
    expected = _historical_select(copy.deepcopy(store), iteration=10)
    actual = teh._select_errors_for_prompt(
        copy.deepcopy(store), iteration=10, error_feedback_mode="legacy"
    )
    assert actual == expected
    assert len(actual) == 8
    assert [x["normalized_key"] for x in actual] == [f"k{i}" for i in range(8)]


def test_legacy_prompt_text_length_and_budget_match_history() -> None:
    store = [_entry("a", last=4, count=3), _entry("b", last=4, count=2)]
    for budget in (1, 25, 60, 120, 1200):
        expected = _historical_prompt(
            copy.deepcopy(store), iteration=5, max_error_prompt_chars=budget
        )
        actual = teh._build_past_error_prompt_section(
            copy.deepcopy(store),
            iteration=5,
            max_error_prompt_chars=budget,
            error_feedback_mode="legacy",
        )
        assert actual == expected
        assert len(actual) == len(expected)
    full = teh._build_past_error_prompt_section(
        store,
        iteration=5,
        max_error_prompt_chars=1200,
        error_feedback_mode="legacy",
    )
    assert "Do not repeat them." in full


def test_legacy_oversized_first_item_is_skipped_not_truncated() -> None:
    store = [
        _entry("oversized", last=5, count=10, size=500),
        _entry("small", last=5, count=1),
    ]
    expected = _historical_prompt(store, iteration=6, max_error_prompt_chars=80)
    actual = teh._build_past_error_prompt_section(
        store,
        iteration=6,
        max_error_prompt_chars=80,
        error_feedback_mode="legacy",
    )
    assert actual == expected
    assert "oversized" not in actual
    assert "message-small" in actual
    assert len(actual) > 80  # Historical budget excludes the header.


def test_legacy_recorder_store_and_jsonl_match_historical_fields() -> None:
    error = {
        "invalid_line": "return missing",
        "error_type": "NameError",
        "error_message": "name 'missing' is not defined",
        "normalized_key": "ignored-current-key",
    }
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "error_history.jsonl"
        store: List[Dict[str, Any]] = teh._ErrorFeedbackStore("legacy")
        for iteration, candidate in ((2, "candidate_0"), (4, "candidate_3")):
            teh._record_invalid_program_error_summary(
                store,
                error,
                iteration=iteration,
                participant_id=7,
                candidate_id=candidate,
                history_path=path,
                eval_split="train",
            )
        assert store == [
            {
                "normalized_key": (
                    "nameerror||name 'missing' is not defined||return missing"
                ),
                "invalid_line": "return missing",
                "error_type": "NameError",
                "error_message": "name 'missing' is not defined",
                "count": 2,
                "first_seen_iteration": 2,
                "last_seen_iteration": 4,
            }
        ]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]["count"] == 1
        assert rows[1]["count"] == 2
        assert rows[1]["last_seen_iteration"] == 4
        assert set(rows[1]) == {
            "iteration",
            "participant_id",
            "candidate_id",
            "invalid_line",
            "error_type",
            "error_message",
            "normalized_key",
            "count",
            "last_seen_iteration",
        }


def test_test_errors_are_excluded_in_both_modes() -> None:
    entry = _entry("test", last=1, count=1)
    for mode in ("legacy", "new"):
        store: List[Dict[str, Any]] = teh._ErrorFeedbackStore(mode)
        teh._record_invalid_program_error_summary(
            store,
            entry,
            iteration=1,
            participant_id=None,
            candidate_id="candidate_0",
            history_path=None,
            eval_split="test",
        )
        assert store == []


def test_new_mode_is_byte_equivalent_to_current_helper_path() -> None:
    store = [
        {
            "normalized_key": "valueerror||bad||return x",
            "invalid_line": "return x",
            "error_type": "ValueError",
            "error_message": "bad",
            "iteration": 4,
            "candidate_id": "candidate_2",
            "quality_score": -0.2,
            "eval_split": "train",
            "n_candidates_in_iteration": 10,
        },
        {
            "normalized_key": "valueerror||bad||return x",
            "invalid_line": "return x",
            "error_type": "ValueError",
            "error_message": "bad",
            "iteration": 4,
            "candidate_id": "candidate_7",
            "quality_score": -0.1,
            "eval_split": "val",
            "n_candidates_in_iteration": 10,
        },
    ]
    expected_selected = teh._select_errors_for_prompt_new(
        copy.deepcopy(store),
        iteration=5,
        previous_n_candidates=10,
    )
    actual_selected = teh._select_errors_for_prompt(
        copy.deepcopy(store),
        iteration=5,
        previous_n_candidates=10,
        error_feedback_mode="new",
    )
    assert actual_selected == expected_selected

    expected_prompt = teh._build_past_error_prompt_section_new(
        copy.deepcopy(store),
        iteration=5,
        max_error_prompt_chars=1200,
        previous_n_candidates=10,
    )
    actual_prompt = teh._build_past_error_prompt_section(
        copy.deepcopy(store),
        iteration=5,
        max_error_prompt_chars=1200,
        previous_n_candidates=10,
        error_feedback_mode="new",
    )
    assert actual_prompt == expected_prompt
    assert actual_prompt.encode() == expected_prompt.encode()


def test_defaults_are_legacy_for_run_apis_and_teh_psych_cli() -> None:
    assert (
        inspect.signature(teh.run_evolution)
        .parameters["error_feedback_mode"]
        .default
        == "legacy"
    )
    assert (
        inspect.signature(teh.run_global_evolution_phase)
        .parameters["error_feedback_mode"]
        .default
        == "legacy"
    )
    args = build_argparser().parse_args([])
    assert args.error_feedback_mode == "legacy"
    assert build_argparser().parse_args(["--error-feedback-mode", "new"]).error_feedback_mode == "new"
