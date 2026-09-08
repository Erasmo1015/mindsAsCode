"""Lightweight tests for sparse train+val observation budgets."""

from __future__ import annotations

from pathlib import Path

from utils.teh.sparse_observations import (
    allocate_train_val_counts,
    apply_max_observed_trials,
    distribute_explore_budget,
    initial_program_id_from_path,
    normalize_max_observed_trials,
    sparse_budget_for_dataset,
    should_persist_sparse_audit,
)


def _trials(n: int, prefix: str) -> list[dict]:
    return [{"id": f"{prefix}{i}", "i": i} for i in range(n)]


def test_no_cap_is_identity():
    train, val, test = _trials(6, "t"), _trials(3, "v"), _trials(4, "x")
    out_train, out_val, out_test, audit = apply_max_observed_trials(
        train,
        val,
        test,
        max_observed_trials_per_participant=None,
        dataset="1peterson2021using",
        participant_id=7,
        split_seed=0,
    )
    assert out_train == train
    assert out_val == val
    assert out_test == test
    assert audit.applied is False
    assert should_persist_sparse_audit(audit) is False
    assert normalize_max_observed_trials(0) is None
    assert normalize_max_observed_trials(-1) is None


def test_test_set_unchanged_and_budget_respected():
    train, val, test = _trials(60, "t"), _trials(20, "v"), _trials(20, "x")
    out_train, out_val, out_test, audit = apply_max_observed_trials(
        train,
        val,
        test,
        max_observed_trials_per_participant=10,
        dataset="1peterson2021using",
        participant_id=3,
        split_seed=0,
    )
    assert out_test == test
    assert audit.retained_n_test == audit.original_n_test == 20
    assert len(out_train) + len(out_val) == 10
    assert audit.retained_n_train + audit.retained_n_val == 10
    assert audit.applied is True
    assert len(out_train) >= 1 and len(out_val) >= 1
    assert all(row in train for row in out_train)
    assert all(row in val for row in out_val)


def test_keep_all_when_available_leq_budget():
    train, val, test = _trials(4, "t"), _trials(2, "v"), _trials(3, "x")
    out_train, out_val, out_test, audit = apply_max_observed_trials(
        train,
        val,
        test,
        max_observed_trials_per_participant=40,
        dataset="1peterson2021using",
        participant_id=1,
        split_seed=0,
    )
    assert out_train == train
    assert out_val == val
    assert out_test == test
    assert audit.applied is False
    assert should_persist_sparse_audit(audit) is True


def test_selection_is_deterministic():
    train, val, test = _trials(30, "t"), _trials(10, "v"), _trials(10, "x")
    kwargs = dict(
        train_trials=train,
        val_trials=val,
        test_trials=test,
        max_observed_trials_per_participant=10,
        dataset="1peterson2021using",
        participant_id=11,
        split_seed=0,
    )
    a = apply_max_observed_trials(**kwargs)
    b = apply_max_observed_trials(**kwargs)
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert a[3].subset_fingerprint == b[3].subset_fingerprint
    assert a[3].selected_train_indices == b[3].selected_train_indices


def test_independent_of_execution_order_and_dataset():
    train, val, test = _trials(24, "t"), _trials(8, "v"), _trials(8, "x")
    first = apply_max_observed_trials(
        train,
        val,
        test,
        max_observed_trials_per_participant=10,
        dataset="1peterson2021using",
        participant_id=5,
        split_seed=0,
    )
    _other = apply_max_observed_trials(
        train,
        val,
        test,
        max_observed_trials_per_participant=10,
        dataset="2plonsky2018when",
        participant_id=5,
        split_seed=0,
    )
    second = apply_max_observed_trials(
        train,
        val,
        test,
        max_observed_trials_per_participant=10,
        dataset="1peterson2021using",
        participant_id=5,
        split_seed=0,
    )
    assert first[3].subset_fingerprint == second[3].subset_fingerprint
    assert first[3].subset_fingerprint != _other[3].subset_fingerprint


def test_source_full_target_sparse_budget_resolution():
    assert (
        sparse_budget_for_dataset(
            dataset_alias="2plonsky2018when",
            config_key="2plonsky2018when",
            max_observed_trials_per_participant=10,
            max_observed_trials_datasets=["1peterson2021using"],
        )
        is None
    )
    assert (
        sparse_budget_for_dataset(
            dataset_alias="1peterson2021using",
            config_key="1peterson2021using",
            max_observed_trials_per_participant=10,
            max_observed_trials_datasets=["1peterson2021using"],
        )
        == 10
    )
    assert (
        sparse_budget_for_dataset(
            dataset_alias="2plonsky2018when",
            max_observed_trials_per_participant=10,
            max_observed_trials_datasets=None,
        )
        == 10
    )


def test_allocate_train_val_keeps_both_splits():
    t_keep, v_keep = allocate_train_val_counts(60, 20, 10)
    assert t_keep + v_keep == 10
    assert t_keep >= 1 and v_keep >= 1
    t_keep, v_keep = allocate_train_val_counts(60, 20, 1)
    assert t_keep + v_keep == 1


def test_explore_budget_is_split_not_multiplied():
    assert distribute_explore_budget(50, 1) == [50]
    assert sum(distribute_explore_budget(50, 2)) == 50
    assert distribute_explore_budget(5, 2) == [3, 2]
    assert distribute_explore_budget(4, 3) == [2, 1, 1]


def test_initial_program_ids_are_distinct():
    used: list[str] = []
    a = initial_program_id_from_path(Path("global/best_program.py"), used)
    used.append(a)
    b = initial_program_id_from_path(
        Path("transfer/source=2plonsky2018when/best_program.py"), used
    )
    assert a != b
    assert a.startswith("global_")
    assert b.startswith("global_")
