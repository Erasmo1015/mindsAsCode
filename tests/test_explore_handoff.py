"""Handoff-parent explore rules (population phase → participant explore)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_explore_handoff():
    path = REPO / "utils" / "teh" / "explore_handoff.py"
    spec = importlib.util.spec_from_file_location("teh_explore_handoff", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_handoff = _load_explore_handoff()
resolve_explore_from_handoff_parents = _handoff.resolve_explore_from_handoff_parents
select_explore_handoff_parents = _handoff.select_explore_handoff_parents


def _parent(pid: str, fitness: float = 0.0) -> tuple:
    return (f"code_{pid}", fitness, 0.0, pid, None, None, 0.0)


def test_resolve_initial_pool_always_explores_from_handoff():
    assert resolve_explore_from_handoff_parents(
        has_initial_pool=True,
        explore_from_population_parents=False,
    )
    assert resolve_explore_from_handoff_parents(
        has_initial_pool=True,
        explore_from_population_parents=True,
    )


def test_resolve_live_global_stays_seed_only_unless_flag():
    assert not resolve_explore_from_handoff_parents(
        has_initial_pool=False,
        explore_from_population_parents=False,
    )
    assert resolve_explore_from_handoff_parents(
        has_initial_pool=False,
        explore_from_population_parents=True,
    )


def test_select_disabled_or_empty_returns_none():
    programs, pinned = select_explore_handoff_parents(
        [_parent("a")],
        enabled=False,
        top_k=1,
    )
    assert programs is None and pinned is None
    programs, pinned = select_explore_handoff_parents([], enabled=True, top_k=1)
    assert programs is None and pinned is None


def test_select_top_k_zero_keeps_global_order():
    elite = [_parent("rank1"), _parent("rank2"), _parent("rank3")]
    programs, pinned = select_explore_handoff_parents(elite, enabled=True, top_k=0)
    assert pinned == ["rank1", "rank2", "rank3"]
    assert [pid for _, pid in programs] == pinned
    assert programs[0][0] == "code_rank1"


def test_select_top_k_one_is_stage_b_sole_parent():
    elite = [_parent("rank1"), _parent("rank2")]
    programs, pinned = select_explore_handoff_parents(elite, enabled=True, top_k=1)
    assert programs == [("code_rank1", "rank1")]
    assert pinned == ["rank1"]


def test_select_top_k_larger_than_pool_keeps_all():
    elite = [_parent("rank1")]
    programs, pinned = select_explore_handoff_parents(elite, enabled=True, top_k=20)
    assert pinned == ["rank1"]
    assert programs == [("code_rank1", "rank1")]


def test_split_explore_budget_stage_f_half_seed():
    n_seed, parent_counts = _handoff.split_explore_budget_seed_and_parents(
        50, n_seed=25, n_handoff_parents=4
    )
    assert n_seed == 25
    assert sum(parent_counts) == 25
    assert len(parent_counts) == 4


def test_split_explore_budget_zero_seed_all_handoff():
    n_seed, parent_counts = _handoff.split_explore_budget_seed_and_parents(
        50, n_seed=0, n_handoff_parents=2
    )
    assert n_seed == 0
    assert parent_counts == [25, 25]


def test_prefix_elite_program_ids():
    tagged = _handoff.prefix_elite_program_ids([_parent("rank1")], "pool0_")
    assert tagged[0][3] == "pool0_rank1"
    assert tagged[0][0] == "code_rank1"


def _run_teh(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO / "teh.py"),
            "--dataset",
            "1peterson2021using",
            "--fitness_metric",
            "loglik",
            "--no_log",
            *extra,
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_rejects_explore_from_population_without_global_or_pool():
    proc = _run_teh("--explore_from_population_parents", "--explore_candidates", "50")
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "requires --global_phase" in combined


def test_cli_rejects_explore_from_population_without_explore_budget():
    proc = _run_teh("--global_phase", "--explore_from_population_parents")
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "requires --explore_candidates > 0" in combined


def test_cli_rejects_negative_explore_population_top_k():
    proc = _run_teh(
        "--global_phase",
        "--explore_from_population_parents",
        "--explore_candidates",
        "50",
        "--explore_population_top_k",
        "-1",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "explore_population_top_k must be >= 0" in combined


def test_cli_rejects_explore_prompt_source_without_dataset():
    proc = _run_teh(
        "--explore_candidates",
        "50",
        "--explore_prompt_source_program",
        "missing.py",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "must be set together" in combined


def test_cli_help_exposes_stage_e_f_flags():
    proc = subprocess.run(
        [sys.executable, str(REPO / "teh.py"), "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "--explore_seed_candidates" in proc.stdout
    assert "--explore_prompt_source_program" in proc.stdout
    assert "--explore_prompt_source_dataset" in proc.stdout
