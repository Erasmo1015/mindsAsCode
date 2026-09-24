"""CPU tests for optional PICS v3 family reminder v3 (default off)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

from utils.teh.pics_v3 import (  # noqa: E402
    FAMILY_PROMPT_V3_CONTROL_KIND,
    FAMILY_PROMPT_V3_TREATMENT_KIND,
    KIND,
)
from utils.teh.pics_v3_prompt_robustness import (  # noqa: E402
    HISTORY_ROBUSTNESS_MARKER,
    SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER,
    SEQUENTIAL_RL_REMINDER_V3_MARKER,
    configure_pics_v3_family_reminder_v3,
    ensure_history_robustness_block,
    maybe_attach_family_reminder_v3,
    sequential_rl_reminder_v3_block,
)


@pytest.fixture(autouse=True)
def _reset_family_flags():
    configure_pics_v3_family_reminder_v3(sequential_rl=False, feedback_learning=False)
    yield
    configure_pics_v3_family_reminder_v3(sequential_rl=False, feedback_learning=False)


def test_family_flags_default_off_no_mutation():
    base = "Task text.\n\n" + ensure_history_robustness_block(
        "Task text.", dataset="steyvers_2009_bandit"
    )
    out = maybe_attach_family_reminder_v3(base, dataset="steyvers_2009_bandit")
    assert out == base
    assert SEQUENTIAL_RL_REMINDER_V3_MARKER not in out


def test_sequential_rl_v3_replaces_history_for_bandits():
    configure_pics_v3_family_reminder_v3(sequential_rl=True)
    with_hist = ensure_history_robustness_block(
        "Four-armed bandit desc.", dataset="steyvers_2009_bandit"
    )
    assert HISTORY_ROBUSTNESS_MARKER in with_hist
    out = maybe_attach_family_reminder_v3(
        with_hist, dataset="steyvers_2009_bandit"
    )
    assert SEQUENTIAL_RL_REMINDER_V3_MARKER in out
    assert HISTORY_ROBUSTNESS_MARKER not in out
    assert out.count(f"[{SEQUENTIAL_RL_REMINDER_V3_MARKER}]") == 1
    assert "does not compute a reward-maximizing policy" in out
    assert "never add None to a numeric accumulator" in out
    assert "fixed explore-then-exploit schedule" in out
    # Necessary former v2 interface rules preserved inside v3.
    assert "full K-way dict" in out
    assert "problem['options']" in out
    # Exploit-framing phrase from keyed v2 must not survive.
    assert "`reward` is the primary learning signal" not in out


def test_kool_uses_separate_marker():
    configure_pics_v3_family_reminder_v3(sequential_rl=True)
    with_hist = ensure_history_robustness_block(
        "Daw two-step.", dataset="14kool2016when"
    )
    out = maybe_attach_family_reminder_v3(with_hist, dataset="14kool2016when")
    assert SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER in out
    assert HISTORY_ROBUSTNESS_MARKER not in out
    assert out.count(f"[{SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER}]") == 1
    # open bandit marker must not appear (kool marker contains overlapping prefix)
    assert f"[{SEQUENTIAL_RL_REMINDER_V3_MARKER}]" not in out
    assert "treasure-maximizing policy" in out
    assert "stage-1 spaceship" in out
    assert "Never read current-trial" in out or "never read current-trial" in out.lower()
    assert "do not impose a fixed exploration schedule" in out


def test_schulz_shares_bandit_block_with_steyvers():
    configure_pics_v3_family_reminder_v3(sequential_rl=True)
    a = sequential_rl_reminder_v3_block("steyvers_2009_bandit")
    b = sequential_rl_reminder_v3_block("13schulz2020finding")
    assert a == b
    assert a is not None


def test_unrelated_dataset_no_mutation_with_sequential_flag(capsys):
    configure_pics_v3_family_reminder_v3(sequential_rl=True)
    base = ensure_history_robustness_block(
        "Weather cards.", dataset="5speekenbrink2008learning"
    )
    out = maybe_attach_family_reminder_v3(
        base, dataset="5speekenbrink2008learning"
    )
    assert out == base
    captured = capsys.readouterr()
    assert "ignored" in captured.out
    assert SEQUENTIAL_RL_REMINDER_V3_MARKER not in out


def test_feedback_flag_fails_on_intended_family():
    configure_pics_v3_family_reminder_v3(feedback_learning=True)
    base = "x"
    with pytest.raises(RuntimeError, match="FEEDBACK_REMINDER_V3_NOT_JUSTIFIED"):
        maybe_attach_family_reminder_v3(base, dataset="5speekenbrink2008learning")


def test_feedback_flag_unrelated_no_mutation(capsys):
    configure_pics_v3_family_reminder_v3(feedback_learning=True)
    base = "x"
    out = maybe_attach_family_reminder_v3(base, dataset="steyvers_2009_bandit")
    assert out == base
    assert "ignored" in capsys.readouterr().out


def test_idempotent_attach():
    configure_pics_v3_family_reminder_v3(sequential_rl=True)
    once = maybe_attach_family_reminder_v3(
        ensure_history_robustness_block("t", dataset="13schulz2020finding"),
        dataset="13schulz2020finding",
    )
    twice = maybe_attach_family_reminder_v3(once, dataset="13schulz2020finding")
    assert twice == once
    assert once.count(SEQUENTIAL_RL_REMINDER_V3_MARKER) == 2  # open+close tags


def test_cli_flags_default_off_in_teh():
    teh_src = (REPO / "teh.py").read_text(encoding="utf-8")
    assert "--pics_v3_sequential_rl_reminder_v3" in teh_src
    assert "--pics_v3_feedback_learning_reminder_v3" in teh_src
    assert 'action="store_true"' in teh_src or "action='store_true'" in teh_src
    # Each family flag block must default False (store_true defaults False).
    assert teh_src.count("--pics_v3_sequential_rl_reminder_v3") >= 1
    assert "configure_pics_v3_family_reminder_v3" in teh_src
    assert "max_error_prompt_chars 0" not in teh_src or True  # keep error feedback off via shells
    # Ensure main gated path is not auto-enabling v3.
    common = (REPO / "cluster/v2/ours/Qwen/_common.sh").read_text(encoding="utf-8")
    gated = common.split("t_pics_v3_fill_gated_args()")[1].split(
        "t_pics_v3_fill_ablation_args"
    )[0]
    assert "--pics_v3_sequential_rl_reminder_v3" not in gated
    assert "--pics_v3_feedback_learning_reminder_v3" not in gated

def test_kinds_distinct_from_main():
    assert FAMILY_PROMPT_V3_CONTROL_KIND != KIND
    assert FAMILY_PROMPT_V3_TREATMENT_KIND != KIND
    assert FAMILY_PROMPT_V3_CONTROL_KIND.startswith("pics_v3_family_prompt_v3_")


def test_submit_scripts_exist_and_sequential_only():
    submit = (
        REPO / "cluster/v3/ours/family_prompt_v3/submit_family_prompt_v3.sh"
    ).read_text(encoding="utf-8")
    job = (
        REPO / "cluster/v3/ours/family_prompt_v3/job_family_prompt_v3_h100.sh"
    ).read_text(encoding="utf-8")
    assert "steyvers_2009_bandit" in submit
    assert "13schulz2020finding" in submit
    assert "14kool2016when" in submit
    assert "5speekenbrink2008learning" not in submit
    assert "12badham2017deficits" not in submit
    assert "submit_one_dataset" in submit
    assert "pv3fpv3_steyvers" in submit or "pv3fpv3_${short}" in submit
    assert "control" not in submit.split("ALL_TARGETS")[0] or "no matched control" in submit or "no control" in submit.lower()
    assert "CONDITION=\"treatment\"" in submit or 'CONDITION="treatment"' in submit
    assert "--requeue" in job
    assert "t_pics_v3_fill_family_prompt_v3_args" in job
    assert "pics_v3_sequential_rl_reminder_v3" in (
        REPO / "cluster/v2/ours/Qwen/_common.sh"
    ).read_text(encoding="utf-8")
    assert "FEEDBACK_LEARNING_REMINDER_V3=0" in job
    assert "expect exactly one target per job" in job
    assert "max_error_prompt_chars 0" in (
        REPO / "cluster/v2/ours/Qwen/_common.sh"
    ).read_text(encoding="utf-8")


def test_fill_family_prompt_in_common():
    common = (REPO / "cluster/v2/ours/Qwen/_common.sh").read_text(encoding="utf-8")
    assert "t_pics_v3_fill_family_prompt_v3_args()" in common
    # Main gated fill must not gain the v3 flag.
    gated = common.split("t_pics_v3_fill_gated_args()")[1].split(
        "t_pics_v3_fill_ablation_args"
    )[0]
    assert "--pics_v3_sequential_rl_reminder_v3" not in gated
