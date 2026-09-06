"""Unit tests for PICS MEM tracing / annotation / dataset helpers."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from utils.mem.trace import (
    MOTIF_TAXONOMY,
    append_mem_trace_record,
    best_reference_parent,
    build_candidate_record,
    build_iteration_context_record,
    compute_delta_f,
    estimate_tokens_char4,
    json_safe_value,
    parent_record_from_elite_tuple,
    record_contains_test_metrics,
    selection_score_from_elite_tuple,
    split_annotation_batches,
    validate_annotation_response,
)
from analysis.mem.build_dataset import build_rows


def test_best_reference_parent_selects_highest_score_first_tie():
    parents = [
        {"program_id": "a", "selection_score": 0.1},
        {"program_id": "b", "selection_score": 0.5},
        {"program_id": "c", "selection_score": 0.5},
        {"program_id": "d", "selection_score": None},
    ]
    best = best_reference_parent(parents)
    assert best is not None
    assert best["program_id"] == "b"


def test_compute_delta_f_uses_selection_scores():
    assert compute_delta_f(-0.3, -0.5) == pytest.approx(0.2)
    assert compute_delta_f(None, -0.5) is None
    assert compute_delta_f(-0.3, None) is None
    assert compute_delta_f(float("nan"), -0.5) is None
    assert compute_delta_f(-0.3, float("-inf")) is None


def test_selection_score_from_elite_tuple():
    tup = ("code", -0.42, 0.5, "iteration_1_candidate_0", None, None, -0.4)
    assert selection_score_from_elite_tuple(tup) == pytest.approx(-0.42)
    assert selection_score_from_elite_tuple(("code", float("nan"), 0, "x")) is None


def test_parent_and_candidate_records_exclude_test_metrics():
    parent = parent_record_from_elite_tuple(
        ("print(1)", -0.2, 0.9, "baseline", None, None, -0.25),
        val_loglik=-0.3,
        train_loglik=-0.25,
    )
    assert not record_contains_test_metrics(parent)
    ctx = build_iteration_context_record(
        dataset="1peterson2021using",
        participant_id=0,
        run_id="run_x",
        split_seed=0,
        phase="evolution",
        iteration=1,
        evolution_selection_score="train_val",
        selected_parents=[parent],
        best_selected_parent_id="baseline",
    )
    assert not record_contains_test_metrics(ctx)
    cand = build_candidate_record(
        dataset="1peterson2021using",
        participant_id=0,
        run_id="run_x",
        split_seed=0,
        phase="evolution",
        iteration=1,
        candidate_id="iteration_1_candidate_0",
        candidate_idx=0,
        source="normal",
        code="def choose(problem, history):\n    return 0.5\n",
        runtime_valid=True,
        train_loglik=-0.2,
        val_loglik=-0.25,
        selection_score=-0.22,
        reference_parent_id="baseline",
        reference_parent_score=-0.2,
        reference_kind="best_selected_parent",
        delta_f=-0.02,
        survived_elite_truncation=True,
        evolution_selection_score="train_val",
    )
    assert not record_contains_test_metrics(cand)
    assert "test_loglik" not in cand
    assert "test_acc" not in cand


def test_mem_logging_helpers_do_not_mutate_search_inputs(tmp_path: Path):
    parents = [
        {"program_id": "p0", "selection_score": -0.1, "code": "a"},
        {"program_id": "p1", "selection_score": -0.2, "code": "b"},
    ]
    parents_before = copy.deepcopy(parents)
    codes = ["c0", "c1"]
    codes_before = list(codes)
    rng_state = np.random.default_rng(0).bit_generator.state

    best = best_reference_parent(parents)
    assert best["program_id"] == "p0"
    path = tmp_path / "mem_trace.jsonl"
    append_mem_trace_record(
        path,
        build_iteration_context_record(
            dataset="1peterson2021using",
            participant_id=1,
            run_id="run",
            split_seed=0,
            phase="evolution",
            iteration=1,
            evolution_selection_score="train_val",
            selected_parents=parents,
            best_selected_parent_id="p0",
        ),
    )
    append_mem_trace_record(
        path,
        build_candidate_record(
            dataset="1peterson2021using",
            participant_id=1,
            run_id="run",
            split_seed=0,
            phase="evolution",
            iteration=1,
            candidate_id="iteration_1_candidate_0",
            candidate_idx=0,
            source="normal",
            code=codes[0],
            runtime_valid=True,
            train_loglik=-0.11,
            val_loglik=-0.12,
            selection_score=-0.115,
            reference_parent_id="p0",
            reference_parent_score=-0.1,
            reference_kind="best_selected_parent",
            delta_f=compute_delta_f(-0.115, -0.1),
            survived_elite_truncation=False,
            evolution_selection_score="train_val",
        ),
    )

    assert parents == parents_before
    assert codes == codes_before
    assert np.random.default_rng(0).bit_generator.state == rng_state
    # JSONL has no test keys
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        assert not record_contains_test_metrics(obj)
        assert "test_loglik" not in json.dumps(obj)


def test_json_safe_nonfinite():
    assert json_safe_value(float("nan")) is None
    assert json_safe_value(float("inf")) is None
    assert json_safe_value({"a": float("-inf")}) == {"a": None}


def test_validate_annotation_response_ok_and_errors():
    expected = ["c1", "c2"]
    payload = [
        {
            "candidate_id": "c1",
            "added_motifs": ["history_or_memory"],
            "removed_motifs": [],
            "modified_motifs": [],
            "primary_edit": "history_or_memory",
            "evidence": ["added history list"],
            "confidence": 0.8,
        },
        {
            "candidate_id": "c2",
            "added_motifs": [],
            "removed_motifs": ["parameter_or_threshold_change"],
            "modified_motifs": [],
            "primary_edit": "no_meaningful_change",
            "evidence": ["same logic"],
            "confidence": 0.5,
        },
    ]
    ok, err, rows = validate_annotation_response(payload, expected_ids=expected)
    assert ok and err == "" and len(rows) == 2

    bad = copy.deepcopy(payload)
    bad[0]["primary_edit"] = "not_a_motif"
    ok, err, _ = validate_annotation_response(bad, expected_ids=expected)
    assert not ok and "primary_edit" in err

    dup = copy.deepcopy(payload)
    dup[1]["candidate_id"] = "c1"
    ok, err, _ = validate_annotation_response(dup, expected_ids=expected)
    assert not ok and "duplicate" in err

    unknown = copy.deepcopy(payload)
    unknown[1]["candidate_id"] = "c99"
    ok, err, _ = validate_annotation_response(unknown, expected_ids=expected)
    assert not ok and "unknown" in err

    missing = [payload[0]]
    ok, err, _ = validate_annotation_response(missing, expected_ids=expected)
    assert not ok and "missing" in err

    assert "history_or_memory" in MOTIF_TAXONOMY


def test_split_annotation_batches_no_code_truncation():
    ref = "REF"
    cands = [
        {"candidate_id": "a", "code": "CODE_A_" + ("a" * 200)},
        {"candidate_id": "b", "code": "CODE_B_" + ("b" * 200)},
        {"candidate_id": "c", "code": "CODE_C_" + ("c" * 200)},
    ]
    # Budget fits at most ~1-2 candidates: force splitting without truncating code.
    one_plus_ref = estimate_tokens_char4(
        "x" * 50
        + json.dumps(
            {
                "reference_program": ref,
                "candidates": [{"candidate_id": "a", "code": cands[0]["code"]}],
            },
            ensure_ascii=False,
        )
    )
    batches = split_annotation_batches(
        cands,
        reference_code=ref,
        base_prompt_chars=50,
        max_input_tokens=one_plus_ref + 20,
        max_candidates_per_batch=10,
    )
    assert len(batches) >= 2
    flat = [c for b in batches for c in b]
    assert [c["candidate_id"] for c in flat] == ["a", "b", "c"]
    for orig, got in zip(cands, flat):
        assert got["code"] == orig["code"]

    huge = {"candidate_id": "z", "code": "y" * 100000}
    with pytest.raises(ValueError, match="refusing to truncate"):
        split_annotation_batches(
            [huge],
            reference_code=ref,
            base_prompt_chars=100,
            max_input_tokens=10,
            max_candidates_per_batch=10,
        )


def test_build_dataset_filters(tmp_path: Path):
    run_dir = tmp_path / "run"
    pdir = run_dir / "participant_0"
    pdir.mkdir(parents=True)
    trace = pdir / "mem_trace.jsonl"
    rows = [
        {
            "record_type": "candidate",
            "run_id": "run",
            "dataset": "1peterson2021using",
            "participant_id": 0,
            "phase": "evolution",
            "iteration": 1,
            "candidate_id": "iteration_1_candidate_0",
            "candidate_idx": 0,
            "source": "normal",
            "runtime_valid": True,
            "delta_f": 0.1,
            "train_loglik": -0.2,
            "val_loglik": -0.3,
            "selection_score": -0.25,
        },
        {
            "record_type": "candidate",
            "run_id": "run",
            "dataset": "1peterson2021using",
            "participant_id": 0,
            "phase": "evolution",
            "iteration": 1,
            "candidate_id": "iteration_1_candidate_1",
            "candidate_idx": 1,
            "source": "fresh",
            "runtime_valid": True,
            "delta_f": 0.2,
        },
        {
            "record_type": "candidate",
            "run_id": "run",
            "dataset": "1peterson2021using",
            "participant_id": 0,
            "phase": "evolution",
            "iteration": 1,
            "candidate_id": "iteration_1_candidate_2",
            "candidate_idx": 2,
            "source": "normal",
            "runtime_valid": False,
            "delta_f": 0.3,
        },
        {
            "record_type": "candidate",
            "run_id": "run",
            "dataset": "1peterson2021using",
            "participant_id": 0,
            "phase": "evolution",
            "iteration": 1,
            "candidate_id": "iteration_1_candidate_3",
            "candidate_idx": 3,
            "source": "normal",
            "runtime_valid": True,
            "delta_f": None,
        },
    ]
    with trace.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    annotations = {
        "iteration_1_candidate_0": {
            "candidate_id": "iteration_1_candidate_0",
            "primary_edit": "history_or_memory",
            "confidence": 0.9,
            "added_motifs": ["history_or_memory"],
            "removed_motifs": [],
            "modified_motifs": [],
        }
    }
    out = build_rows(run_dir=run_dir, annotations=annotations)
    assert len(out) == 1
    assert out[0]["candidate_id"] == "iteration_1_candidate_0"
    assert out[0]["primary_edit"] == "history_or_memory"
