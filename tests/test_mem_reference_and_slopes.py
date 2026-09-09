"""Tests for MEM generation-reference selection, support report, and random slopes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.mem.annotate_edits import _nmc_repair_hint
from analysis.mem.predictor_support import motif_support_report
from utils.mem.reference_types import (
    REF_POOL_BEST_PROXY,
    REF_SEED_BASELINE,
    enrich_candidate_reference_fields,
    is_exact_reference,
)
from utils.mem.schema_v2 import annotation_resume_key
from utils.mem.reconstruct_old_run import reconstruct_participant_records
from utils.mem.trace import record_contains_test_metrics


def _write_cand(dir_path: Path, idx: int, body: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"candidate_{idx}.py").write_text(body, encoding="utf-8")


def test_annotation_resume_key_includes_reference_pairing():
    a = annotation_resume_key(0, "c1", reference_id="baseline", reference_type="seed_baseline")
    b = annotation_resume_key(0, "c1", reference_id="pool_x", reference_type="pool_best_proxy")
    assert a != b
    assert a == annotation_resume_key(
        0, "c1", reference_id="baseline", reference_type="seed_baseline"
    )


def test_nmc_repair_hint_mentions_empty_lists():
    hint = _nmc_repair_hint("c0: no_meaningful_change=true requires empty motif")
    assert "empty" in hint.lower()
    assert "no_meaningful_change=false" in hint


def test_enrich_reference_fields_sets_named_delta():
    rec = enrich_candidate_reference_fields(
        {"record_type": "candidate"},
        reference_type=REF_SEED_BASELINE,
        reference_id="global_baseline",
        reference_score=-0.7,
        delta_f=0.1,
    )
    assert rec["reference_is_exact"] is True
    assert rec["delta_f_vs_baseline"] == 0.1
    assert is_exact_reference(REF_SEED_BASELINE)
    assert not is_exact_reference(REF_POOL_BEST_PROXY)


def test_predictor_support_reports_exclusions_not_by_significance():
    df = pd.DataFrame(
        {
            "participant_id": [0, 0, 1, 1, 2, 2],
            "phase": ["evolution"] * 6,
            "dataset": ["ds"] * 6,
            "history_added": [1, 0, 1, 0, 1, 0],
            "value_modified": [1, 1, 1, 1, 0, 0],
            "risk_removed": [0, 0, 0, 0, 0, 0],
            "learning_added": [1, 0, 0, 0, 0, 0],
        }
    )
    from utils.mem.schema_v2 import all_directional_behavioral_columns

    for c in all_directional_behavioral_columns():
        if c not in df.columns:
            df[c] = 0
    report = motif_support_report(df, min_positive_rows=2, min_participants_with_pos=2)
    assert "history_added" in report["supported_columns"]
    assert "value_modified" in report["supported_columns"]
    excluded = {e["column"]: e["reasons"] for e in report["excluded_columns"]}
    assert "risk_removed" in excluded
    assert "constant_predictor" in excluded["risk_removed"]
    assert "learning_added" in excluded


def test_reconstruct_no_future_in_explore_or_fresh(tmp_path: Path):
    pdir = tmp_path / "participant_0"
    pdir.mkdir()
    (pdir / "results.json").write_text(
        json.dumps(
            {
                "baseline": {
                    "selection_score": -1.0,
                    "train_loglik": -1.0,
                    "val_loglik": -1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    init = pdir / "initial_pool_from_global"
    init.mkdir()
    (init / "000_global_baseline.py").write_text("BASE\n", encoding="utf-8")
    (init / "participant_selection_scores.json").write_text(
        json.dumps(
            {
                "cache_version": 1,
                "dataset": "ds",
                "participant_id": 0,
                "split_ratio": 0.6,
                "split_seed": 0,
                "n_eval_seeds": 3,
                "psych_dataset_split": "train",
                "evolution_selection_score": "train_val",
                "programs": [
                    {
                        "program_id": "global_baseline",
                        "filename": "000_global_baseline.py",
                        "code_sha1": __import__("hashlib").sha1(b"BASE\n").hexdigest(),
                        "ok": True,
                        "selection_score": -1.0,
                        "train_loglik": -1.0,
                        "val_loglik": -1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    explore = pdir / "explore_phase"
    _write_cand(explore / "candidates", 0, "E\n")
    (explore / "metrics.json").write_text(
        json.dumps(
            {
                "candidate_results": [
                    {
                        "idx": 0,
                        "runtime_valid": True,
                        "selection_score": -0.4,
                        "train_loglik": -0.4,
                        "val_loglik": -0.4,
                        "fitness": -0.4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    it1 = pdir / "iteration_1"
    _write_cand(it1 / "candidates", 0, "N\n")
    _write_cand(it1 / "candidates", 1, "F\n")
    (it1 / "metrics.json").write_text(
        json.dumps(
            {
                "evolution_selection_score": "train_val",
                "candidate_results": [
                    {
                        "idx": 0,
                        "source": "normal",
                        "runtime_valid": True,
                        "selection_score": -0.2,
                        "train_loglik": -0.2,
                        "val_loglik": -0.2,
                        "fitness": -0.2,
                        "test_loglik": -9.9,
                    },
                    {
                        "idx": 1,
                        "source": "fresh",
                        "runtime_valid": True,
                        "selection_score": -0.8,
                        "train_loglik": -0.8,
                        "val_loglik": -0.8,
                        "fitness": -0.8,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    records = reconstruct_participant_records(
        pdir,
        dataset="ds",
        run_id="run_t",
        elite_pool_size=50,
        rescore_cache_path=init / "participant_selection_scores.json",
    )
    for rec in records:
        assert not record_contains_test_metrics(rec)

    explore_rec = next(r for r in records if r.get("candidate_id") == "explore_candidate_0")
    assert explore_rec["reference_type"] == REF_SEED_BASELINE
    assert explore_rec["reference_parent_score"] == pytest.approx(-1.0)
    assert explore_rec["delta_f"] == pytest.approx(0.6)

    normal = next(r for r in records if r.get("candidate_id") == "iteration_1_candidate_0")
    assert normal["reference_type"] == REF_POOL_BEST_PROXY
    assert normal["reference_parent_id"] == "explore_candidate_0"
    assert normal["delta_f"] == pytest.approx(0.2)

    fresh = next(r for r in records if r.get("candidate_id") == "iteration_1_candidate_1")
    assert fresh["reference_type"] == REF_SEED_BASELINE
    assert fresh["reference_parent_score"] == pytest.approx(-1.0)
    assert fresh["delta_f"] == pytest.approx(0.2)


def test_random_slope_extraction_smoke():
    from analysis.mem.fit_mem_random_slopes import fit_focal_motif
    import numpy as np

    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(12):
        slope = float(rng.normal(0.15, 0.05))
        for it in range(8):
            m = int(rng.integers(0, 2))
            rows.append(
                {
                    "participant_id": pid,
                    "iteration": it,
                    "history_added": m,
                    "value_modified": int(rng.integers(0, 2)),
                    "delta_f": slope * m + float(rng.normal(0, 0.05)),
                    "phase": "evolution",
                    "reference_type": "pool_best_proxy",
                }
            )
    df = pd.DataFrame(rows)
    res = fit_focal_motif(
        df,
        focal="history_added",
        controls=["value_modified"],
        include_phase_effects=False,
        structural_controls=[],
    )
    assert res["status"] in ("ok", "singular_or_boundary", "not_converged")
    assert "formula" in res
    assert "history_added" in res.get("re_formula", "1 + history_added")
