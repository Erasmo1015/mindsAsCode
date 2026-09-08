"""Unit tests for MEM schema v2 annotation / resume / dataset / fit helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

from analysis.mem.annotate_edits import (
    _load_completed_v2_keys,
    annotate_with_splits,
)
from analysis.mem.build_dataset import build_rows_v2
from analysis.mem.fit_mem import build_formula, select_predictors
from utils.mem.schema_v2 import (
    BEHAVIORAL_MOTIFS_V2,
    SCHEMA_VERSION,
    annotation_resume_key,
    directional_flags_from_annotation,
    is_schema_v2_row,
    validate_annotation_response_v2,
)
from utils.mem.trace import record_contains_test_metrics


def _valid_row(cid: str, **overrides: Any) -> Dict[str, Any]:
    row = {
        "candidate_id": cid,
        "added_motifs": ["history"],
        "removed_motifs": [],
        "modified_motifs": [],
        "structural_operations": ["parameter_change"],
        "no_meaningful_change": False,
        "evidence": ["added history[-5:]"],
        "confidence": 0.9,
    }
    row.update(overrides)
    return row


def test_validate_v2_ok_multilabel():
    payload = [
        _valid_row("c1", added_motifs=["history", "risk"], structural_operations=[]),
        _valid_row(
            "c2",
            added_motifs=[],
            removed_motifs=[],
            modified_motifs=[],
            structural_operations=[],
            no_meaningful_change=True,
            evidence=["rename only"],
        ),
    ]
    ok, err, rows = validate_annotation_response_v2(payload, expected_ids=["c1", "c2"])
    assert ok and err == ""
    assert rows[0]["schema_version"] == SCHEMA_VERSION
    assert rows[0]["added_motifs"] == ["history", "risk"]
    assert rows[1]["no_meaningful_change"] is True


def test_validate_v2_rejects_invented_labels():
    payload = [_valid_row("c1", added_motifs=["sigmoid"])]
    ok, err, _ = validate_annotation_response_v2(payload, expected_ids=["c1"])
    assert not ok
    assert "sigmoid" in err


def test_validate_v2_rejects_primary_edit_field():
    row = _valid_row("c1")
    row["primary_edit"] = "history"
    ok, err, _ = validate_annotation_response_v2([row], expected_ids=["c1"])
    assert not ok
    assert "primary_edit" in err


def test_validate_v2_nmc_consistency():
    # nmc true but nonempty lists
    payload = [
        _valid_row(
            "c1",
            no_meaningful_change=True,
            added_motifs=["history"],
        )
    ]
    ok, err, _ = validate_annotation_response_v2(payload, expected_ids=["c1"])
    assert not ok
    assert "no_meaningful_change=true" in err

    # nmc false but all empty
    payload2 = [
        _valid_row(
            "c1",
            added_motifs=[],
            removed_motifs=[],
            modified_motifs=[],
            structural_operations=[],
            no_meaningful_change=False,
        )
    ]
    ok, err, _ = validate_annotation_response_v2(payload2, expected_ids=["c1"])
    assert not ok
    assert "no_meaningful_change=false" in err


def test_participant_aware_resume_ignores_v1(tmp_path: Path):
    path = tmp_path / "annotations_v2.jsonl"
    rows = [
        # v1-like row: must NOT count as completed
        {
            "candidate_id": "iteration_1_candidate_0",
            "participant_id": 0,
            "primary_edit": "history_or_memory",
            "added_motifs": ["history_or_memory"],
        },
        # v2 for participant 0
        {
            "schema_version": 2,
            "candidate_id": "iteration_1_candidate_0",
            "participant_id": 0,
            "added_motifs": ["history"],
            "removed_motifs": [],
            "modified_motifs": [],
            "structural_operations": [],
            "no_meaningful_change": False,
            "evidence": ["x"],
            "confidence": 0.8,
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    done = _load_completed_v2_keys(path)
    assert annotation_resume_key(0, "iteration_1_candidate_0") in done
    # Same candidate_id for participant 1 is NOT done
    assert annotation_resume_key(1, "iteration_1_candidate_0") not in done
    assert len(done) == 1


def test_v1_v2_separation_is_schema_v2_row():
    assert is_schema_v2_row({"schema_version": 2})
    assert not is_schema_v2_row({"schema_version": 1})
    assert not is_schema_v2_row({"primary_edit": "history_or_memory"})


def test_directional_flags_not_averaged():
    ann = {
        "schema_version": 2,
        "added_motifs": ["history"],
        "removed_motifs": [],
        "modified_motifs": ["history"],
        "structural_operations": ["nonlinear_change"],
        "no_meaningful_change": False,
    }
    flags = directional_flags_from_annotation(ann)
    assert flags["history_added"] == 1
    assert flags["history_removed"] == 0
    assert flags["history_modified"] == 1
    assert flags["structural_nonlinear_change"] == 1
    # other motifs zero
    assert flags["risk_added"] == 0


def test_nonfatal_singleton_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    failures = tmp_path / "annotation_failures.jsonl"
    client = MagicMock()

    def _fail(*_a, **_k):
        return [], "{not json", "JSON parse error: Expecting value"

    monkeypatch.setattr(
        "analysis.mem.annotate_edits._annotate_batch",
        _fail,
    )
    out = annotate_with_splits(
        client,
        model_name="dummy",
        reference_code="def choose(problem, history):\n    return 0.5\n",
        batch=[{"candidate_id": "iteration_1_candidate_0", "code": "return 0.5"}],
        base_prompt_chars=100,
        max_input_tokens=12000,
        max_candidates_per_batch=5,
        raw_dir=raw_dir,
        batch_tag="t",
        use_guided_json=False,
        participant_id=7,
        failures_path=failures,
        max_attempts=2,
    )
    assert out == []
    assert failures.is_file()
    fail_row = json.loads(failures.read_text(encoding="utf-8").splitlines()[0])
    assert fail_row["participant_id"] == 7
    assert fail_row["candidate_id"] == "iteration_1_candidate_0"
    assert "JSON parse" in fail_row["error"]
    assert fail_row["attempts"] == 2
    assert len(fail_row["raw_responses"]) == 2


def test_build_rows_v2_directional_and_exclusions(tmp_path: Path):
    run_dir = tmp_path / "run"
    pdir = run_dir / "participant_0"
    pdir.mkdir(parents=True)
    trace = pdir / "mem_trace.jsonl"
    cands = [
        {
            "record_type": "candidate",
            "run_id": "run",
            "dataset": "ds",
            "participant_id": 0,
            "phase": "evolution",
            "iteration": 1,
            "candidate_id": "iteration_1_candidate_0",
            "candidate_idx": 0,
            "source": "normal",
            "runtime_valid": True,
            "delta_f": 0.1,
        },
        {
            "record_type": "candidate",
            "run_id": "run",
            "dataset": "ds",
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
            "dataset": "ds",
            "participant_id": 0,
            "phase": "evolution",
            "iteration": 1,
            "candidate_id": "iteration_1_candidate_2",
            "candidate_idx": 2,
            "source": "normal",
            "runtime_valid": True,
            "delta_f": 0.3,
        },
    ]
    with trace.open("w", encoding="utf-8") as f:
        for r in cands:
            f.write(json.dumps(r) + "\n")

    ann = {
        annotation_resume_key(0, "iteration_1_candidate_0"): {
            "schema_version": 2,
            "candidate_id": "iteration_1_candidate_0",
            "participant_id": 0,
            "added_motifs": ["history"],
            "removed_motifs": [],
            "modified_motifs": ["risk"],
            "structural_operations": ["parameter_change"],
            "no_meaningful_change": False,
            "confidence": 0.9,
        }
    }
    excl_path = tmp_path / "excl.jsonl"
    rows, excl = build_rows_v2(
        run_dir=run_dir,
        annotations=ann,
        exclusions_path=excl_path,
    )
    assert len(rows) == 1
    assert rows[0]["history_added"] == 1
    assert rows[0]["risk_modified"] == 1
    assert rows[0]["history_modified"] == 0
    assert rows[0]["structural_parameter_change"] == 1
    assert "primary_edit" not in rows[0]
    assert excl["excl_source_fresh"] == 1
    assert excl["excl_missing_annotation_v2"] == 1
    # every exclusion recorded
    excl_lines = excl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(excl_lines) == 2
    reasons = {json.loads(x)["reason"] for x in excl_lines}
    assert "excl_source_fresh" in reasons
    assert "excl_missing_annotation_v2" in reasons


def test_fit_select_predictors_reports_constant_no_silent_drop():
    df = pd.DataFrame(
        {
            "delta_f": [0.1, -0.1, 0.0, 0.05],
            "iteration": [1, 2, 3, 4],
            "participant_id": [0, 0, 1, 1],
            "history_added": [1, 0, 1, 0],
            "risk_added": [0, 0, 0, 0],  # constant
            "value_added": [1, 1, 1, 0],
        }
    )
    included_b, included_s, reports = select_predictors(
        df,
        requested_behavioral=["history_added", "risk_added", "not_a_col"],
        requested_structural=[],
        min_count=1,
    )
    assert included_b == ["history_added"]
    assert included_s == []
    by_col = {r["column"]: r for r in reports}
    assert by_col["risk_added"]["included"] is False
    assert by_col["risk_added"]["reason"] == "constant_predictor"
    assert by_col["not_a_col"]["reason"] == "not_a_directional_behavioral_column"
    assert "primary_edit" not in build_formula(included_b, included_s)


def test_no_test_metric_leakage_in_schema_helpers():
    assert "test_loglik" not in BEHAVIORAL_MOTIFS_V2
    rec = {"delta_f": 0.1, "history_added": 1}
    assert not record_contains_test_metrics(rec)
    assert record_contains_test_metrics({"test_loglik": -1.0, "x": 1})


def test_malformed_json_path_via_validate_wrapper():
    # validate is after parse; annotate returns JSON parse error string separately.
    # Here ensure invented structural labels rejected (second check after guided_json).
    payload = [_valid_row("c1", structural_operations=["sigmoid_fn"])]
    ok, err, _ = validate_annotation_response_v2(payload, expected_ids=["c1"])
    assert not ok
    assert "sigmoid_fn" in err
