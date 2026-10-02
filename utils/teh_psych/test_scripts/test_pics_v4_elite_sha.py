"""CPU tests for PICS-v4 population exact-SHA elite deduplication."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.teh.aamas_v0_lossless_trials import set_pics_run_seeds
from utils.teh.elite_sha import (
    ELITE_POLICY_SHA256_UNIQUE_V1,
    PARTICIPANT_POPULATION_UNIQUE_CAP,
    build_participant_handoff_pool,
    cap_participant_elite,
    dedupe_elite_exact_sha,
    source_sha256,
)
from utils.teh.pics_v4 import using_pics_v4
from utils.teh.pics_v4 import (
    KIND_TARGET_ONLY,
    assert_pics_v4_resume,
    write_pics_v4_marker,
)


def _row(code: str, fitness: float, program_id: str):
    return (code, fitness, None, program_id, None, None, fitness)


def test_top50_duplicates_do_not_block_later_unique_programs():
    duplicate = "def choose(problem, history):\n    return 0.1\n"
    rows = [_row(duplicate, 1.0, f"dup_{i}") for i in range(30)]
    rows.extend(
        _row(f"def choose(problem, history):\n    return {i}\n", 0.5, f"uniq_{i}")
        for i in range(20)
    )
    rows.append(_row("def choose(problem, history):\n    return 99\n", 0.1, "below_50"))
    kept, stats = dedupe_elite_exact_sha(rows, elite_cap=50)
    ids = [parent[3] for parent in kept]
    assert "below_50" in ids
    assert stats["n_duplicate_removed"] == 29
    assert stats["pool_size"] == 22
    assert stats["pool_size"] < 50
    assert stats["n_unique_sha"] == stats["pool_size"]
    assert len({source_sha256(parent[0]) for parent in kept}) == len(kept)


def test_pinned_program_is_the_kept_duplicate():
    code = "def choose(problem, history):\n    return 0.2\n"
    rows = [
        _row(code, -0.2, "better_copy"),
        _row(code, -1.0, "global_baseline"),
    ]
    kept, stats = dedupe_elite_exact_sha(
        rows, elite_cap=50, pinned_ids=("global_baseline",)
    )
    assert [parent[3] for parent in kept] == ["global_baseline"]
    assert stats["n_duplicate_removed"] == 1
    assert stats["pool_size"] == 1


def test_better_fitness_wins_when_neither_copy_is_pinned():
    code = "def choose(problem, history):\n    return 0.3\n"
    rows = [
        _row(code, -1.0, "worse"),
        _row(code, -0.2, "better"),
    ]
    kept, _stats = dedupe_elite_exact_sha(rows, elite_cap=50)
    assert [parent[3] for parent in kept] == ["better"]


def test_equal_fitness_keeps_the_earlier_entry():
    code = "def choose(problem, history):\n    return 0.4\n"
    rows = [
        _row(code, -0.5, "earlier"),
        _row(code, -0.5, "later"),
    ]
    kept, _stats = dedupe_elite_exact_sha(rows, elite_cap=50)
    assert [parent[3] for parent in kept] == ["earlier"]


def test_pool_smaller_than_cap_is_not_refilled():
    rows = [_row(f"def choose(problem, history):\n    return {i}\n", -float(i), f"p{i}") for i in range(3)]
    kept, stats = dedupe_elite_exact_sha(rows, elite_cap=50)
    assert len(kept) == 3
    assert stats["pool_size"] == 3
    assert stats["n_duplicate_removed"] == 0
    assert stats["n_unique_sha"] == 3


def test_parent_sampling_uses_unique_shas_without_replacement():
    import teh

    duplicate = "def choose(problem, history):\n    return 0.5\n"
    rows = [_row(duplicate, 1.0, f"dup_{i}") for i in range(4)]
    rows.extend(
        _row(f"def choose(problem, history):\n    return {i + 10}\n", 0.4, f"u{i}")
        for i in range(2)
    )
    kept, _stats = dedupe_elite_exact_sha(rows, elite_cap=50)
    assert len(kept) == 3
    indices, _best_k, _sampled_k = teh._select_parent_indices_from_elite_pool(
        len(kept),
        sample_size=8,
        sample_parents=True,
        sampled_parents_decay=False,
        iter_idx=1,
        total_iters=10,
        rng=np.random.default_rng(0),
    )
    assert len(indices) == min(8, len(kept))
    assert len(set(indices)) == len(indices)
    shas = [source_sha256(kept[index][0]) for index in indices]
    assert len(set(shas)) == len(shas)


def test_candidate_files_and_traces_remain_after_dedup(tmp_path: Path):
    for index in range(10):
        (tmp_path / f"candidate_{index}.py").write_text(
            f"def choose(problem, history):\n    return {index}\n",
            encoding="utf-8",
        )
    trace = tmp_path / "mem_trace.jsonl"
    trace.write_text(
        "".join(json.dumps({"candidate_id": f"candidate_{i}"}) + "\n" for i in range(10)),
        encoding="utf-8",
    )
    rows = [
        _row((tmp_path / f"candidate_{index}.py").read_text(encoding="utf-8"), 1.0, f"c{index}")
        for index in range(10)
    ]
    dedupe_elite_exact_sha(rows, elite_cap=50)
    assert sorted(path.name for path in tmp_path.glob("candidate_*.py")) == [
        f"candidate_{index}.py" for index in range(10)
    ]
    assert len(trace.read_text(encoding="utf-8").splitlines()) == 10


def test_resume_fails_closed_on_a_different_elite_policy(tmp_path: Path):
    set_pics_run_seeds(run_seed=0, split_seed=0)
    run = tmp_path / KIND_TARGET_ONLY / "job_1"
    run.mkdir(parents=True)
    write_pics_v4_marker(run)
    payload = json.loads((run / "TRIAL_PROMPT_POLICY.json").read_text(encoding="utf-8"))
    assert payload["elite_policy"] == ELITE_POLICY_SHA256_UNIQUE_V1
    assert_pics_v4_resume(run)
    missing = dict(payload)
    missing.pop("elite_policy")
    (run / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(missing) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(run)
    except RuntimeError as exc:
        assert "elite_policy" in str(exc)
    else:
        raise AssertionError("resume accepted a marker without elite_policy")
    other = dict(payload)
    other["elite_policy"] = "keep_duplicates_v0"
    (run / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(other) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(run)
    except RuntimeError as exc:
        assert "keep_duplicates_v0" in str(exc)
    else:
        raise AssertionError("resume accepted a different elite policy")
    (run / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert payload["participant_population_unique_cap"] == PARTICIPANT_POPULATION_UNIQUE_CAP
    assert payload["elite_pool_size_is_upper_bound"] is True
    assert payload["duplicate_backfill"] is False
    missing_cap = dict(payload)
    missing_cap.pop("participant_population_unique_cap")
    (run / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(missing_cap) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(run)
    except RuntimeError as exc:
        assert "participant_population_unique_cap" in str(exc)
    else:
        raise AssertionError("resume accepted a marker without the population cap")
    backfill = dict(payload)
    backfill["duplicate_backfill"] = True
    (run / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(backfill) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(run)
    except RuntimeError as exc:
        assert "duplicate_backfill" in str(exc)
    else:
        raise AssertionError("resume accepted duplicate backfill")


def test_population_rank1_is_kept_when_exploration_has_the_same_sha():
    code = "def choose(problem, history):\n    return 1\n"
    elites, vals, pins, stats = build_participant_handoff_pool(
        [_row(code, -1.0, "global_rank1")],
        [_row(code, -0.1, "explore_candidate_0")],
        elite_cap=50,
        population_vals=[-1.5],
        exploration_vals=[-0.1],
    )
    assert [parent[3] for parent in elites] == ["global_rank1"]
    assert vals == [-1.5]
    assert pins == ["global_rank1"]
    assert stats["n_explore_skipped_population_sha"] == 1
    assert stats["n_population_retained"] == 1


def test_at_most_five_unique_population_shas_enter_participant_evolution():
    population = [
        _row(f"def choose(problem, history):\n    return {index}\n", -float(index), f"global_{index}")
        for index in range(20)
    ]
    explore = [_row("def choose(problem, history):\n    return 100\n", -20.0, "explore_0")]
    elites, _vals, _pins, stats = build_participant_handoff_pool(population, explore, elite_cap=50)
    ids = [parent[3] for parent in elites]
    population_ids = [program_id for program_id in ids if str(program_id).startswith("global_")]
    assert len(population_ids) == 5
    assert stats["n_population_retained"] == 5
    assert "global_5" not in ids
    assert "explore_0" in ids
    assert len(elites) == 6


def test_four_additional_population_programs_use_participant_rescores():
    rank1 = _row("def choose(problem, history):\n    return 'rank'\n", -9.0, "global_rank1")
    population_order = [
        _row(
            f"def choose(problem, history):\n    return 'order{index}'\n",
            -8.0,
            f"pop_order_{index}",
        )
        for index in range(4)
    ]
    participant_best = [
        _row(
            f"def choose(problem, history):\n    return 'best{index}'\n",
            -0.1 * (index + 1),
            f"participant_best_{index}",
        )
        for index in range(4)
    ]
    elites, _vals, pins, _stats = build_participant_handoff_pool(
        [rank1, *population_order, *participant_best],
        [],
        elite_cap=50,
        population_rank1_id="global_rank1",
    )
    assert {parent[3] for parent in elites} == {
        "global_rank1",
        "participant_best_0",
        "participant_best_1",
        "participant_best_2",
        "participant_best_3",
    }
    assert pins == ["global_rank1"]


def test_exploration_candidates_fill_remaining_slots_only():
    population = [
        _row(f"def choose(problem, history):\n    return 'p{index}'\n", -1.0, f"global_{index}")
        for index in range(5)
    ]
    explore = [
        _row(f"def choose(problem, history):\n    return 'e{index}'\n", -0.2, f"explore_{index}")
        for index in range(10)
    ]
    elites, _vals, _pins, stats = build_participant_handoff_pool(
        population, explore, elite_cap=8
    )
    assert stats["n_population_retained"] == 5
    assert stats["n_explore_retained"] == 3
    assert len(elites) == 8


def test_duplicate_exploration_programs_occupy_one_slot():
    code = "def choose(problem, history):\n    return 2\n"
    explore = [_row(code, -0.5 + index * 0.01, f"explore_{index}") for index in range(5)]
    elites, _vals, _pins, stats = build_participant_handoff_pool(
        [_row("def choose(problem, history):\n    return 0\n", -1.0, "global_rank1")],
        explore,
        elite_cap=50,
    )
    assert [parent[3] for parent in elites if str(parent[3]).startswith("explore_")] == ["explore_4"]
    assert stats["n_explore_retained"] == 1
    assert stats["n_duplicate_removed"] == 4
    assert len(elites) == 2


def test_exploration_sha_matching_a_retained_population_sha_is_skipped():
    shared = "def choose(problem, history):\n    return 3\n"
    elites, _vals, _pins, stats = build_participant_handoff_pool(
        [
            _row("def choose(problem, history):\n    return 9\n", -2.0, "global_rank1"),
            _row(shared, -0.4, "global_extra"),
        ],
        [
            _row(shared, -0.1, "explore_same"),
            _row("def choose(problem, history):\n    return 4\n", -0.2, "explore_new"),
        ],
        elite_cap=50,
    )
    ids = [parent[3] for parent in elites]
    assert "global_extra" in ids
    assert "explore_new" in ids
    assert "explore_same" not in ids
    assert stats["n_explore_skipped_population_sha"] == 1


def test_participant_handoff_smaller_than_50_is_not_refilled():
    elites, _vals, _pins, stats = build_participant_handoff_pool(
        [_row("def choose(problem, history):\n    return 1\n", -1.0, "global_rank1")],
        [
            _row(f"def choose(problem, history):\n    return {index + 10}\n", -0.5, f"explore_{index}")
            for index in range(2)
        ],
        elite_cap=50,
    )
    assert len(elites) == 3
    assert stats["pool_size"] == 3
    assert stats["duplicate_backfill"] is False


def test_participant_evolution_update_remains_sha_unique():
    code = "def choose(problem, history):\n    return 7\n"
    start = [
        _row(code, -3.0, "global_rank1"),
        _row("def choose(problem, history):\n    return 8\n", -1.0, "explore_0"),
        _row("def choose(problem, history):\n    return 9\n", -1.2, "explore_1"),
    ]
    updated = [
        *start,
        _row(code, -0.1, "iteration_1_candidate_0"),
        _row("def choose(problem, history):\n    return 10\n", -0.2, "iteration_1_candidate_1"),
    ]
    kept, _vals, stats = cap_participant_elite(
        updated,
        elite_cap=50,
        pinned_ids=("global_rank1",),
        sha_unique=True,
    )
    assert stats is not None
    assert stats["pool_size"] == 4
    assert len({source_sha256(parent[0]) for parent in kept}) == 4
    assert any(parent[3] == "global_rank1" for parent in kept)
    assert all(parent[3] != "iteration_1_candidate_0" for parent in kept)


def test_parent_sampling_with_fewer_than_eight_unique_programs():
    import teh

    elites, _vals, _pins, _stats = build_participant_handoff_pool(
        [_row("def choose(problem, history):\n    return 1\n", -1.0, "global_rank1")],
        [
            _row(f"def choose(problem, history):\n    return {index + 10}\n", -0.4, f"explore_{index}")
            for index in range(2)
        ],
        elite_cap=50,
    )
    assert len(elites) == 3
    indices, _best_k, _sampled_k = teh._select_parent_indices_from_elite_pool(
        len(elites),
        sample_size=8,
        sample_parents=True,
        sampled_parents_decay=False,
        iter_idx=0,
        total_iters=10,
        rng=np.random.default_rng(0),
    )
    assert indices == [0, 1, 2] or len(indices) == 3
    assert len(set(indices)) == 3
    assert len({source_sha256(elites[index][0]) for index in indices}) == 3


def test_handoff_does_not_delete_candidate_or_mem_records(tmp_path: Path):
    for index in range(4):
        (tmp_path / f"candidate_{index}.py").write_text(
            f"def choose(problem, history):\n    return {index}\n",
            encoding="utf-8",
        )
    trace = tmp_path / "mem_trace.jsonl"
    trace.write_text(
        "".join(json.dumps({"candidate_id": f"explore_candidate_{index}"}) + "\n" for index in range(4)),
        encoding="utf-8",
    )
    explore = [
        _row((tmp_path / f"candidate_{index}.py").read_text(encoding="utf-8"), -0.2, f"explore_{index}")
        for index in range(4)
    ]
    build_participant_handoff_pool(
        [_row("def choose(problem, history):\n    return 0\n", -1.0, "global_rank1")],
        explore,
        elite_cap=50,
    )
    cap_participant_elite(explore, elite_cap=50, sha_unique=True)
    assert sorted(path.name for path in tmp_path.glob("candidate_*.py")) == [
        f"candidate_{index}.py" for index in range(4)
    ]
    assert len(trace.read_text(encoding="utf-8").splitlines()) == 4


def test_pics_v3_and_aamas_v0_keep_duplicate_elite_slots():
    assert using_pics_v4() is False
    duplicate = "def choose(problem, history):\n    return 0.1\n"
    rows = [_row(duplicate, 1.0, f"dup_{index}") for index in range(30)]
    rows.extend(
        _row(f"def choose(problem, history):\n    return {index}\n", 0.5, f"uniq_{index}")
        for index in range(20)
    )
    rows.append(_row("def choose(problem, history):\n    return 99\n", 0.1, "below_50"))
    kept, _vals, stats = cap_participant_elite(rows, elite_cap=50, sha_unique=False)
    ids = [parent[3] for parent in kept]
    assert stats is None
    assert len(kept) == 50
    assert "below_50" not in ids
    assert sum(parent[0] == duplicate for parent in kept) == 30
