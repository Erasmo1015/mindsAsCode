"""Unit tests for MEM schema v3 state-aware annotation / CSV / eligibility hooks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from analysis.mem.annotate_edits import _load_completed_keys
from analysis.mem.build_dataset import build_rows_v2, build_rows_v3
from utils.mem.schema_v2 import annotation_resume_key, is_schema_v2_row
from utils.mem.schema_v3 import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    assert_transition_identities,
    derive_directional_motifs,
    state_and_eligibility_flags,
    validate_annotation_response_v3,
)

_REPO = Path(__file__).resolve().parents[1]


def _llm_row(cid: str, **overrides: Any) -> Dict[str, Any]:
    row = {
        "candidate_id": cid,
        "reference_motif_state": ["history"],
        "candidate_motif_state": ["history", "value"],
        "modified_motifs": ["history"],
        "structural_operations": [],
        "no_meaningful_change": False,
        "evidence": ["value newly present; history rewritten"],
        "confidence": 0.9,
    }
    row.update(overrides)
    return row


def test_derive_transitions():
    # 0→0: absent both
    a, r, m = derive_directional_motifs([], [], [])
    assert a == r == m == []
    # 0→1 added
    a, r, m = derive_directional_motifs([], ["value"], [])
    assert a == ["value"] and r == [] and m == []
    # 1→0 removed
    a, r, m = derive_directional_motifs(["value"], [], [])
    assert a == [] and r == ["value"] and m == []
    # 1→1 unmodified
    a, r, m = derive_directional_motifs(["value"], ["value"], [])
    assert a == r == m == []
    # 1→1 modified
    a, r, m = derive_directional_motifs(["value"], ["value"], ["value"])
    assert a == r == [] and m == ["value"]


def test_validate_v3_ok_and_derives():
    ok, err, rows = validate_annotation_response_v3(
        [_llm_row("c1")], expected_ids=["c1"]
    )
    assert ok and err == ""
    assert rows[0]["schema_version"] == SCHEMA_VERSION
    assert rows[0]["prompt_version"] == PROMPT_VERSION
    assert rows[0]["added_motifs"] == ["value"]
    assert rows[0]["removed_motifs"] == []
    assert rows[0]["modified_motifs"] == ["history"]
    assert "added_motifs" not in _llm_row("c1")  # LLM payload has no added field


def test_modified_outside_intersection_rejected():
    ok, err, _ = validate_annotation_response_v3(
        [
            _llm_row(
                "c1",
                reference_motif_state=["history"],
                candidate_motif_state=["value"],
                modified_motifs=["history"],
            )
        ],
        expected_ids=["c1"],
    )
    assert not ok
    assert "not in reference" in err or "intersection" in err


def test_no_conflicting_directions_after_derive():
    ok, err, rows = validate_annotation_response_v3(
        [
            _llm_row(
                "c1",
                reference_motif_state=["history", "risk"],
                candidate_motif_state=["history", "value"],
                modified_motifs=["history"],
            )
        ],
        expected_ids=["c1"],
    )
    assert ok and err == ""
    added, removed, modified = (
        set(rows[0]["added_motifs"]),
        set(rows[0]["removed_motifs"]),
        set(rows[0]["modified_motifs"]),
    )
    assert added.isdisjoint(removed)
    assert added.isdisjoint(modified)
    assert removed.isdisjoint(modified)
    assert_transition_identities(rows[0])


def test_unknown_duplicate_missing_state():
    ok, err, _ = validate_annotation_response_v3(
        [_llm_row("c1", reference_motif_state=["not_a_motif"])],
        expected_ids=["c1"],
    )
    assert not ok and "not_a_motif" in err

    ok, err, rows = validate_annotation_response_v3(
        [_llm_row("c1", reference_motif_state=["history", "history"])],
        expected_ids=["c1"],
    )
    assert ok and rows[0]["reference_motif_state"] == ["history"]

    payload = _llm_row("c1")
    del payload["reference_motif_state"]
    ok, err, _ = validate_annotation_response_v3([payload], expected_ids=["c1"])
    assert not ok and "missing required field" in err


def test_nmc_consistency():
    ok, err, _ = validate_annotation_response_v3(
        [
            _llm_row(
                "c1",
                reference_motif_state=[],
                candidate_motif_state=[],
                modified_motifs=[],
                structural_operations=[],
                no_meaningful_change=True,
                evidence=["noop"],
            )
        ],
        expected_ids=["c1"],
    )
    assert ok and err == ""

    ok, err, _ = validate_annotation_response_v3(
        [
            _llm_row(
                "c1",
                no_meaningful_change=True,
                evidence=["claimed nmc but changes"],
            )
        ],
        expected_ids=["c1"],
    )
    assert not ok


def test_eligibility_flags_transitions():
    ann = {
        "schema_version": 3,
        "reference_motif_state": ["history"],
        "candidate_motif_state": ["history", "value"],
        "added_motifs": ["value"],
        "removed_motifs": [],
        "modified_motifs": ["history"],
        "structural_operations": [],
        "no_meaningful_change": False,
    }
    flags = state_and_eligibility_flags(ann)
    assert flags["reference_has_history"] == 1
    assert flags["candidate_has_value"] == 1
    assert flags["value_added"] == 1
    assert flags["history_modified"] == 1
    assert flags["eligible_value_added"] == 1  # ref absent
    assert flags["eligible_history_added"] == 0
    assert flags["eligible_history_removed"] == 1
    assert flags["eligible_history_modified"] == 1
    assert flags["retained_unmodified_history"] == 0
    # risk 0→0
    assert flags["reference_has_risk"] == 0
    assert flags["candidate_has_risk"] == 0
    assert flags["eligible_risk_added"] == 1
    assert flags["eligible_risk_removed"] == 0
    assert flags["eligible_risk_modified"] == 0


def test_legacy_v2_lacks_eligibility_cols(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "mem_trace.jsonl").parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "record_type": "candidate",
        "participant_id": 0,
        "candidate_id": "c1",
        "phase": "evolution",
        "source": "normal",
        "runtime_valid": True,
        "delta_f": 0.1,
        "reference_id": "r1",
        "reference_type": "pool_best_proxy",
    }
    (run_dir / "mem_trace.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    ann = {
        "schema_version": 2,
        "participant_id": 0,
        "candidate_id": "c1",
        "reference_id": "r1",
        "reference_type": "pool_best_proxy",
        "added_motifs": ["history"],
        "removed_motifs": [],
        "modified_motifs": [],
        "structural_operations": [],
        "no_meaningful_change": False,
        "confidence": 0.5,
    }
    key = annotation_resume_key(0, "c1", reference_id="r1", reference_type="pool_best_proxy")
    rows, _ = build_rows_v2(run_dir=run_dir, annotations={key: ann})
    assert len(rows) == 1
    assert "eligible_history_added" not in rows[0]
    assert rows[0].get("eligible_history_added") is None


def test_v3_csv_columns_and_fixture(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "record_type": "candidate",
        "participant_id": 0,
        "candidate_id": "c1",
        "phase": "evolution",
        "source": "normal",
        "runtime_valid": True,
        "delta_f": 0.2,
        "reference_id": "r1",
        "reference_type": "pool_best_proxy",
    }
    (run_dir / "mem_trace.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    ok, err, cleaned = validate_annotation_response_v3(
        [_llm_row("c1")], expected_ids=["c1"]
    )
    assert ok and err == ""
    ann = dict(cleaned[0])
    ann.update(
        {
            "participant_id": 0,
            "reference_id": "r1",
            "reference_type": "pool_best_proxy",
        }
    )
    key = annotation_resume_key(0, "c1", reference_id="r1", reference_type="pool_best_proxy")
    rows, _ = build_rows_v3(run_dir=run_dir, annotations={key: ann})
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == 3
    assert "eligible_value_added" in row
    assert row["value_added"] == 1
    assert row["eligible_value_added"] == 1
    assert "reference_motif_state" in row
    assert json.loads(row["reference_motif_state"]) == ["history"]


def test_resume_ignores_v2_for_v3(tmp_path: Path):
    p = tmp_path / "mixed.jsonl"
    v2 = {
        "schema_version": 2,
        "participant_id": 0,
        "candidate_id": "c1",
        "reference_id": "r1",
        "reference_type": "pool_best_proxy",
    }
    v3 = {
        "schema_version": 3,
        "participant_id": 0,
        "candidate_id": "c2",
        "reference_id": "r1",
        "reference_type": "pool_best_proxy",
    }
    p.write_text(
        json.dumps(v2) + "\n" + json.dumps(v3) + "\n", encoding="utf-8"
    )
    keys_v3 = _load_completed_keys(p, schema_version=3)
    keys_v2 = _load_completed_keys(p, schema_version=2)
    assert len(keys_v3) == 1
    assert len(keys_v2) == 1
    assert annotation_resume_key(
        0, "c1", reference_id="r1", reference_type="pool_best_proxy"
    ) not in keys_v3


def test_eligibility_fit_refuses_legacy(tmp_path: Path):
    csv_path = tmp_path / "legacy.csv"
    pd.DataFrame(
        {
            "participant_id": [0, 0],
            "delta_f": [0.1, -0.1],
            "iteration": [1, 2],
            "phase": ["evolution", "evolution"],
            "history_added": [1, 0],
            "value_added": [0, 1],
        }
    ).to_csv(csv_path, index=False)
    out = tmp_path / "fit_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO / "analysis/mem/fit_mem_random_slopes.py"),
            "--input_csv",
            str(csv_path),
            "--output_dir",
            str(out),
            "--focal_motifs",
            "history_added",
            "--eligibility_mode",
            "restrict",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "eligibility" in (proc.stderr + proc.stdout).lower()

    proc_j = subprocess.run(
        [
            sys.executable,
            str(_REPO / "analysis/mem/fit_mem_joint_random_slopes.py"),
            "--input_csv",
            str(csv_path),
            "--output_dir",
            str(tmp_path / "joint_out"),
            "--random_slopes",
            "history_added",
            "--fixed_effects",
            "history_added,iteration",
            "--eligibility_mode",
            "fe_adjust",
            "--methods",
            "lbfgs",
        ],
        capture_output=True,
        text=True,
    )
    assert proc_j.returncode != 0
    assert "eligibility" in (proc_j.stderr + proc_j.stdout).lower()


def test_is_schema_v2_row_not_v3():
    assert is_schema_v2_row({"schema_version": 2})
    assert not is_schema_v2_row({"schema_version": 3})
