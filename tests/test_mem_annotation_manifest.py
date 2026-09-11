"""Tests for MEM annotation queue manifest discovery / validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.mem.annotation_manifest import (
    default_manifest_path,
    load_manifest,
    queue_jobs,
    validate_manifest,
    write_tsv,
)


def test_default_manifest_loads_and_validates():
    path = default_manifest_path()
    assert path.is_file()
    data = load_manifest(path)
    errs = validate_manifest(data, require_teh_exists=True)
    assert errs == [], errs
    assert len(data["jobs"]) >= 8
    modes = {j["reference_mode"] for j in data["jobs"]}
    assert "pool_best_proxy" in modes
    assert "live_mem_trace" in modes


def test_queue_jobs_excludes_done_choice13k():
    data = load_manifest()
    q = queue_jobs(data, include_optional=True)
    ids = {(j["dataset"], j["run_id"]) for j in q}
    assert ("1peterson2021using", "run_260706_211034") not in ids
    assert ("1peterson2021using", "run_260709_120406") in ids
    assert any(j["dataset"] == "bergert_nosofsky_2007" for j in q)


def test_bergert_skip_reconstruct_flags():
    data = load_manifest()
    berg = [j for j in data["jobs"] if j["dataset"] == "bergert_nosofsky_2007"]
    assert len(berg) == 1
    assert berg[0]["skip_reconstruct"] is True
    assert berg[0]["reference_mode"] == "live_mem_trace"


def test_write_tsv(tmp_path: Path):
    data = load_manifest()
    out = tmp_path / "q.tsv"
    write_tsv(data, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("dataset\t")
    assert "bergert_nosofsky_2007" in text


def test_validate_rejects_bad_skip(tmp_path: Path):
    data = {
        "jobs": [
            {
                "dataset": "x",
                "run_id": "r",
                "teh_path": "nope",
                "reference_mode": "pool_best_proxy",
                "annotation_status": "pending",
                "mem_wave": "w",
                "intended_out": "o",
                "skip_reconstruct": True,
            }
        ]
    }
    errs = validate_manifest(data, require_teh_exists=False)
    assert any("skip_reconstruct" in e for e in errs)
