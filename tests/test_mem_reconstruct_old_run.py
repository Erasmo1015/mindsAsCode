"""Tests for old-run MEM trace reconstruction (pool_best_proxy)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.mem.reconstruct_old_run import (
    REFERENCE_KIND_POOL_BEST_PROXY,
    PoolProgram,
    _truncate_pool,
    reconstruct_participant_records,
    seed_pool_from_participant,
    validate_artifacts_for_reconstruction,
    write_participant_mem_trace,
)
from utils.mem.trace import compute_delta_f, iter_jsonl_records, record_contains_test_metrics


def _write_candidate(dir_path: Path, idx: int, body: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"candidate_{idx}.py").write_text(body, encoding="utf-8")


def test_truncate_pool_keeps_highest_scores():
    pool = [
        PoolProgram("a", -0.5, "a"),
        PoolProgram("b", -0.1, "b"),
        PoolProgram("c", -0.3, "c"),
    ]
    kept = _truncate_pool(pool, elite_pool_size=2)
    assert [p.program_id for p in kept] == ["b", "c"]


def test_reconstruct_uses_pre_iteration_pool_best_not_future(tmp_path: Path):
    pdir = tmp_path / "participant_0"
    pdir.mkdir()
    (pdir / "results.json").write_text(
        json.dumps(
            {
                "baseline": {
                    "selection_score": -0.9,
                    "train_loglik": -0.9,
                    "val_loglik": -0.9,
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
                "dataset": "1peterson2021using",
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
                        "selection_score": -0.9,
                        "train_loglik": -0.9,
                        "val_loglik": -0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Explore produces a better program than baseline.
    explore = pdir / "explore_phase"
    _write_candidate(explore / "candidates", 0, "EXPLORE_BEST\n")
    (explore / "metrics.json").write_text(
        json.dumps(
            {
                "candidate_results": [
                    {
                        "idx": 0,
                        "runtime_valid": True,
                        "selection_score": -0.5,
                        "train_loglik": -0.4,
                        "val_loglik": -0.7,
                        "fitness": -0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Iteration 1: candidate better than explore; must still use explore as reference.
    it1 = pdir / "iteration_1"
    _write_candidate(it1 / "candidates", 0, "ITER1_BEST\n")
    _write_candidate(it1 / "candidates", 1, "ITER1_WEAK\n")
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
                        "test_loglik": -0.99,  # must not leak into trace
                    },
                    {
                        "idx": 1,
                        "source": "normal",
                        "runtime_valid": True,
                        "selection_score": -0.8,
                        "train_loglik": -0.8,
                        "val_loglik": -0.8,
                        "fitness": -0.8,
                    },
                    {
                        "idx": 2,
                        "source": "normal",
                        "runtime_valid": False,
                        "selection_score": -0.1,
                        "fitness": -1e9,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # Iteration 2: reference must be iteration_1_candidate_0 (-0.2), not future.
    it2 = pdir / "iteration_2"
    _write_candidate(it2 / "candidates", 0, "ITER2\n")
    (it2 / "metrics.json").write_text(
        json.dumps(
            {
                "evolution_selection_score": "train_val",
                "candidate_results": [
                    {
                        "idx": 0,
                        "source": "fresh",
                        "runtime_valid": True,
                        "selection_score": -0.15,
                        "train_loglik": -0.15,
                        "val_loglik": -0.15,
                        "fitness": -0.15,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    records = reconstruct_participant_records(
        pdir,
        dataset="1peterson2021using",
        run_id="run_test",
        elite_pool_size=50,
        rescore_cache_path=init / "participant_selection_scores.json",
    )
    assert records
    for rec in records:
        assert not record_contains_test_metrics(rec)
        assert "test_loglik" not in rec

    cand1 = [r for r in records if r.get("candidate_id") == "iteration_1_candidate_0"][0]
    assert cand1["reference_kind"] == REFERENCE_KIND_POOL_BEST_PROXY
    assert cand1["reference_type"] == REFERENCE_KIND_POOL_BEST_PROXY
    assert cand1["reference_is_proxy"] is True
    assert cand1["reference_parent_id"] == "explore_candidate_0"
    assert cand1["reference_parent_score"] == pytest.approx(-0.5)
    assert cand1["delta_f"] == pytest.approx(compute_delta_f(-0.2, -0.5))
    assert cand1["delta_f_vs_pool_best"] == pytest.approx(cand1["delta_f"])

    cand1b = [r for r in records if r.get("candidate_id") == "iteration_1_candidate_1"][0]
    assert cand1b["reference_parent_id"] == "explore_candidate_0"
    assert cand1b["delta_f"] == pytest.approx(compute_delta_f(-0.8, -0.5))

    # Non-runtime-valid excluded
    assert not any(r.get("candidate_id") == "iteration_1_candidate_2" for r in records)

    # Fresh uses exact seed baseline, NOT pool-best proxy.
    cand2 = [r for r in records if r.get("candidate_id") == "iteration_2_candidate_0"][0]
    assert cand2["source"] == "fresh"
    assert cand2["reference_type"] == "seed_baseline"
    assert cand2["reference_is_exact"] is True
    assert cand2["reference_parent_id"] == "global_baseline"
    assert cand2["reference_parent_score"] == pytest.approx(-0.9)
    assert cand2["delta_f"] == pytest.approx(compute_delta_f(-0.15, -0.9))
    assert cand2["delta_f_vs_baseline"] == pytest.approx(cand2["delta_f"])

    # Explore phase emitted vs baseline (exact).
    explore = [r for r in records if r.get("candidate_id") == "explore_candidate_0"][0]
    assert explore["phase"] == "explore"
    assert explore["reference_type"] == "seed_baseline"
    assert explore["reference_is_exact"] is True
    assert explore["delta_f"] == pytest.approx(compute_delta_f(-0.5, -0.9))
    # No future leakage into explore reference.
    assert explore["reference_parent_id"] == "global_baseline"

    out = write_participant_mem_trace(pdir, records, overwrite=True)
    loaded = list(iter_jsonl_records([out]))
    assert len(loaded) == len(records)


def test_seed_pool_prefers_explore_over_baseline(tmp_path: Path):
    pdir = tmp_path / "participant_1"
    pdir.mkdir()
    (pdir / "results.json").write_text(
        json.dumps({"baseline": {"selection_score": -0.7, "train_loglik": -0.7}}),
        encoding="utf-8",
    )
    init = pdir / "initial_pool_from_global"
    init.mkdir()
    (init / "000_global_baseline.py").write_text("BASE\n", encoding="utf-8")
    # Pretend rescore cache already exists so unit test does not need HF data.
    cache = {
        "cache_version": 1,
        "dataset": "1peterson2021using",
        "participant_id": 1,
        "split_ratio": 0.6,
        "split_seed": 0,
        "n_eval_seeds": 3,
        "psych_dataset_split": "train",
        "evolution_selection_score": "train_val",
        "programs": [
            {
                "program_id": "global_baseline",
                "filename": "000_global_baseline.py",
                "code_sha1": __import__("hashlib")
                .sha1(b"BASE\n")
                .hexdigest(),
                "ok": True,
                "selection_score": -0.7,
                "train_loglik": -0.7,
                "val_loglik": -0.7,
            }
        ],
    }
    (init / "participant_selection_scores.json").write_text(
        json.dumps(cache), encoding="utf-8"
    )
    explore = pdir / "explore_phase"
    _write_candidate(explore / "candidates", 3, "E\n")
    (explore / "metrics.json").write_text(
        json.dumps(
            {
                "candidate_results": [
                    {
                        "idx": 3,
                        "runtime_valid": True,
                        "selection_score": -0.4,
                        "fitness": -0.4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pool = seed_pool_from_participant(
        pdir,
        elite_pool_size=10,
        dataset="1peterson2021using",
        rescore_cache_path=init / "participant_selection_scores.json",
    )
    assert pool[0].program_id == "explore_candidate_3"
    assert pool[0].selection_score == pytest.approx(-0.4)


def test_fresh_runtime_valid_updates_pool_and_can_become_next_reference(tmp_path: Path):
    """Fresh RV proposals participate in elite updates even if annotate drops them."""
    pdir = tmp_path / "participant_0"
    pdir.mkdir()
    (pdir / "results.json").write_text(
        json.dumps({"baseline": {"selection_score": -0.9}}), encoding="utf-8"
    )
    init = pdir / "initial_pool_from_global"
    init.mkdir()
    (init / "000_global_baseline.py").write_text("BASE\n", encoding="utf-8")
    (init / "participant_selection_scores.json").write_text(
        json.dumps(
            {
                "cache_version": 1,
                "dataset": "toy",
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
                        "selection_score": -0.9,
                        "train_loglik": -0.9,
                        "val_loglik": -0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    it1 = pdir / "iteration_1"
    _write_candidate(it1 / "candidates", 0, "FRESH_BEST\n")
    (it1 / "metrics.json").write_text(
        json.dumps(
            {
                "evolution_selection_score": "train_val",
                "candidate_results": [
                    {
                        "idx": 0,
                        "source": "fresh",
                        "runtime_valid": True,
                        "selection_score": -0.2,
                        "fitness": -0.2,
                        "train_loglik": -0.2,
                        "val_loglik": -0.2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    it2 = pdir / "iteration_2"
    _write_candidate(it2 / "candidates", 0, "NORMAL\n")
    (it2 / "metrics.json").write_text(
        json.dumps(
            {
                "evolution_selection_score": "train_val",
                "candidate_results": [
                    {
                        "idx": 0,
                        "source": "normal",
                        "runtime_valid": True,
                        "selection_score": -0.25,
                        "fitness": -0.25,
                        "train_loglik": -0.25,
                        "val_loglik": -0.25,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    from utils.mem.reconstruct_old_run import ReconstructStats

    stats = ReconstructStats()
    records = reconstruct_participant_records(
        pdir,
        dataset="toy",
        run_id="run",
        elite_pool_size=50,
        rescore_cache_path=init / "participant_selection_scores.json",
        stats=stats,
    )
    assert stats.fresh_added_to_pool == 1
    assert stats.fresh_runtime_valid_written == 1
    cand2 = [r for r in records if r.get("candidate_id") == "iteration_2_candidate_0"][0]
    assert cand2["reference_parent_id"] == "iteration_1_candidate_0"
    assert cand2["reference_parent_score"] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "run_rel",
    [
        "generated_outputs_old/psych101_train/teh/1peterson2021using/run_260706_211034",
    ],
)
def test_real_old_run_artifacts_ready(run_rel: str):
    root = Path(__file__).resolve().parents[1]
    run_dir = root / run_rel
    if not run_dir.is_dir():
        pytest.skip(f"old run not present: {run_dir}")
    report = validate_artifacts_for_reconstruction(run_dir)
    assert report["ok"], report
    assert report["elite_pool_size"] == 50
