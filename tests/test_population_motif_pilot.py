#!/usr/bin/env python3
"""Smoke tests for population motif pilot (no LLM)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "analysis_2026Sep/mem/population_motif_pilot_emnlp_r1"


def test_schema_validation() -> None:
    sys.path.insert(0, str(REPO))
    from utils.mem.schema_population_motif import (
        BEHAVIORAL_MOTIFS,
        guided_json_schema_for_programs,
        validate_program_motif_response,
    )

    assert "history" in BEHAVIORAL_MOTIFS
    schema = guided_json_schema_for_programs(["p1"])
    assert schema["type"] == "array"
    ok, err, rows = validate_program_motif_response(
        [
            {
                "program_id": "p1",
                "program_motif_state": ["feedback"],
                "evidence": ["reward term"],
                "confidence": 0.7,
            }
        ],
        expected_ids=["p1"],
    )
    assert ok, err
    assert rows[0]["modified_motifs"] == []
    assert rows[0]["program_motif_state"] == ["feedback"]
    bad, _, _ = validate_program_motif_response(
        [
            {
                "program_id": "p1",
                "program_motif_state": ["history_added"],
                "evidence": [],
                "confidence": 0.5,
            }
        ],
        expected_ids=["p1"],
    )
    assert not bad


def test_manifest_build_and_counts() -> None:
    build = PILOT / "build_manifest.py"
    subprocess.check_call([sys.executable, str(build)], cwd=str(REPO))
    payload = json.loads((PILOT / "program_manifest.json").read_text(encoding="utf-8"))
    assert payload["n_datasets"] == 15
    assert payload["n_programs"] == 149
    assert sum(payload["per_dataset_counts"].values()) == 149
    assert min(payload["per_dataset_counts"].values()) >= 5
    assert max(payload["per_dataset_counts"].values()) <= 11
    # every code path exists
    for prog in payload["programs"]:
        assert (REPO / prog["code_path"]).is_file(), prog["code_path"]
        assert prog["resume_key"].count("|") == 2


def test_annotate_dry_run() -> None:
    out = PILOT / "dry_run_smoke"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO / "analysis/mem/annotate_population_programs.py"),
        "--manifest",
        str(PILOT / "program_manifest.json"),
        "--output_dir",
        str(out),
        "--dry_run",
    ]
    subprocess.check_call(cmd, cwd=str(REPO), env={**dict(**__import__("os").environ), "PYTHONPATH": str(REPO)})
    summary = json.loads((out / "dry_run_summary.json").read_text(encoding="utf-8"))
    assert summary["n_datasets"] == 15
    assert summary["n_programs"] == 149
    assert summary["schema_ok"] is True


if __name__ == "__main__":
    test_schema_validation()
    print("ok schema")
    test_manifest_build_and_counts()
    print("ok manifest")
    test_annotate_dry_run()
    print("ok dry_run")
    print("ALL SMOKE PASSED")
