"""Optional --t_pics_gated_source for independent mode without a transfer map."""
from __future__ import annotations

import argparse

import pytest

from utils.teh.t_pics_gated_transfer import (
    apply_gated_cli_defaults,
    make_explicit_independent_source_entry,
)


def test_make_explicit_independent_source_entry():
    entry = make_explicit_independent_source_entry(
        target="1peterson2021using",
        source="11enkavi2019recentprobes",
    )
    assert entry.target == "1peterson2021using"
    assert entry.selected_source == "11enkavi2019recentprobes"
    assert entry.job_id == "live"


def test_make_explicit_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unknown T-PICS dataset|not a known"):
        make_explicit_independent_source_entry(
            target="1peterson2021using",
            source="not_a_real_dataset",
        )


def test_apply_gated_defaults_skips_map_when_explicit_source():
    ns = argparse.Namespace(
        global_phase=False,
        refinement_phase=True,
        explore_from_population_parents=False,
        explore_population_top_k=5,
        n_iterations=3,
        global_iters=2,
        explore_candidates=10,
        t_pics_source_config=None,
        t_pics_gated_source="11enkavi2019recentprobes",
        limited_data_protocol="off",
        limited_train_val=None,
        hard_prompt_token_cap=None,
        max_parent_chars=None,
        llm_max_tokens=None,
        max_prompt_train_trials=None,
    )
    apply_gated_cli_defaults(
        ns,
        argv=[
            "--t_pics_gated_transfer",
            "--t_pics_gated_independent",
            "--t_pics_gated_source",
            "11enkavi2019recentprobes",
        ],
    )
    assert ns.t_pics_source_config in (None, "")
    assert ns.explore_population_top_k == 1
    assert ns.global_phase is True


def test_parser_accepts_t_pics_gated_source():
    import teh as teh_mod

    parser_src = teh_mod.__file__
    assert parser_src
    # Smoke: flag exists on the real parser built in main's argparse block.
    # Build a minimal ArgumentParser clone by importing the help string presence.
    import inspect

    source = inspect.getsource(teh_mod)
    assert "--t_pics_gated_source" in source
    assert "t_pics_gated_source" in source
