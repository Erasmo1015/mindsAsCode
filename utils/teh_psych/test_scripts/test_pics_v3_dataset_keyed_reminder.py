"""CPU regressions for dataset_keyed_post_adaptive_v1 (six datasets only)."""
from __future__ import annotations

import re

from utils.teh.pics_v3_prompt_robustness import (
    DATASET_KEYED_POST_ADAPTIVE_POLICY_ID,
    DATASET_KEYED_REMINDER_ALIASES,
    HISTORY_ROBUSTNESS_BLOCK,
    HISTORY_ROBUSTNESS_MARKER,
    LEGACY_GENERIC_REMINDER_POLICY_ID,
    all_dataset_keyed_reminder_bodies,
    configure_pics_v3_legacy_generic_reminder,
    ensure_history_robustness_block,
    maybe_attach_history_robustness_after_task_description,
    pics_v3_prompt_robustness_scope,
    resolve_history_robustness_block,
    resolve_history_robustness_policy_id,
)

UNAFFECTED_ALIASES = (
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "7hilbig2014generalized",
    "10frey2017risk",
    "11enkavi2019recentprobes",
    "mixed_gambles",
    "bergert_nosofsky_2007",
)

ORACLE_LEAK_PATTERNS = (
    re.compile(r"\buse(?:s|ing)? current[- ]trial (?:weather|correct|reward|treasure|category)", re.I),
    re.compile(r"problem\[['\"]weather_outcome['\"]\]"),
    re.compile(r"problem\[['\"]correct_category['\"]\]"),
    re.compile(r"problem\[['\"]was_correct['\"]\]"),
    re.compile(r"problem\[['\"]reward['\"]\]"),
    re.compile(r"problem\[['\"]treasure['\"]\]"),
    re.compile(r"read(?:ing)? (?:the )?current correct category from `?problem", re.I),
)


def setup_function() -> None:
    configure_pics_v3_legacy_generic_reminder(False)


def teardown_function() -> None:
    configure_pics_v3_legacy_generic_reminder(False)


def test_keyed_aliases_are_exactly_the_six() -> None:
    assert DATASET_KEYED_REMINDER_ALIASES == {
        "5speekenbrink2008learning",
        "14kool2016when",
        "steyvers_2009_bandit",
        "13schulz2020finding",
        "guan_2020_stopping",
        "12badham2017deficits",
    }
    assert set(all_dataset_keyed_reminder_bodies()) == DATASET_KEYED_REMINDER_ALIASES


def test_unaffected_datasets_byte_identical_to_legacy_block() -> None:
    for alias in UNAFFECTED_ALIASES:
        block = resolve_history_robustness_block(alias)
        assert block == HISTORY_ROBUSTNESS_BLOCK
        assert resolve_history_robustness_policy_id(alias) == LEGACY_GENERIC_REMINDER_POLICY_ID


def test_unspecified_dataset_keeps_legacy() -> None:
    assert resolve_history_robustness_block(None) == HISTORY_ROBUSTNESS_BLOCK
    assert resolve_history_robustness_block("") == HISTORY_ROBUSTNESS_BLOCK
    assert resolve_history_robustness_block("unknown_dataset_xyz") == HISTORY_ROBUSTNESS_BLOCK


def test_keyed_datasets_resolve_deterministically_and_differ_from_legacy() -> None:
    bodies = all_dataset_keyed_reminder_bodies()
    for alias in sorted(DATASET_KEYED_REMINDER_ALIASES):
        block = resolve_history_robustness_block(alias)
        assert HISTORY_ROBUSTNESS_MARKER in block
        assert bodies[alias] in block
        assert block != HISTORY_ROBUSTNESS_BLOCK
        assert resolve_history_robustness_policy_id(alias) == (
            DATASET_KEYED_POST_ADAPTIVE_POLICY_ID
        )
        # Idempotent / deterministic.
        assert resolve_history_robustness_block(alias) == block


def test_placement_immediately_after_adaptive_task_description() -> None:
    task = "## Task description (high-level)\n\nAdaptive text."
    with pics_v3_prompt_robustness_scope(True, dataset="5speekenbrink2008learning"):
        out = maybe_attach_history_robustness_after_task_description(
            task, dataset="5speekenbrink2008learning"
        )
    assert out.startswith(task)
    assert out.index(task) == 0
    assert out.index(HISTORY_ROBUSTNESS_MARKER) > len(task) - 1
    # Marker appears once, before any later contract-like section would.
    assert out.count(HISTORY_ROBUSTNESS_MARKER) == 2  # open + close tags share prefix
    assert "[/" + HISTORY_ROBUSTNESS_MARKER + "]" in out


def test_v3_scope_off_leaves_prompt_unchanged_even_for_keyed_dataset() -> None:
    task = "adaptive"
    with pics_v3_prompt_robustness_scope(False, dataset="14kool2016when"):
        assert (
            maybe_attach_history_robustness_after_task_description(
                task, dataset="14kool2016when"
            )
            == task
        )


def test_legacy_flag_forces_generic_for_keyed_datasets() -> None:
    configure_pics_v3_legacy_generic_reminder(True)
    try:
        for alias in DATASET_KEYED_REMINDER_ALIASES:
            assert resolve_history_robustness_block(alias) == HISTORY_ROBUSTNESS_BLOCK
            assert (
                resolve_history_robustness_policy_id(alias)
                == LEGACY_GENERIC_REMINDER_POLICY_ID
            )
    finally:
        configure_pics_v3_legacy_generic_reminder(False)


def test_speekenbrink_reminder_content_and_no_oracle() -> None:
    body = all_dataset_keyed_reminder_bodies()["5speekenbrink2008learning"]
    assert "problem['cards']" in body
    assert "was_correct" in body
    assert "weather_outcome" in body
    assert "feedback" in body
    assert "~0.5" in body
    for pat in ORACLE_LEAK_PATTERNS:
        assert not pat.search(body), pat.pattern
    assert "Do not read current-trial weather" in body


def test_kool_stage_and_action_contract() -> None:
    body = all_dataset_keyed_reminder_bodies()["14kool2016when"]
    assert "stage==1" in body
    assert "stage==2" in body
    assert "spaceship" in body
    assert "planet" in body
    assert "alien" in body
    assert "option_keys.index" in body
    assert "Never read current-trial `reward`" in body
    for pat in ORACLE_LEAK_PATTERNS:
        assert not pat.search(body), pat.pattern


def test_bandit_reminders_require_kway_and_reward_primary() -> None:
    for alias, k in (
        ("steyvers_2009_bandit", "K=4"),
        ("13schulz2020finding", "K=8"),
    ):
        body = all_dataset_keyed_reminder_bodies()[alias]
        assert "reward" in body
        assert "primary" in body
        assert "K-way" in body or k in body
        assert "uniform" in body
        assert "problem['options']" in body
        assert "usually missing" in body


def test_guan_has_no_feedback_reward_coaching() -> None:
    body = all_dataset_keyed_reminder_bodies()["guan_2020_stopping"]
    assert "No `feedback`/`reward` interface" in body
    assert "0=continue" in body
    assert "1=stop" in body
    assert "values_observed" in body
    assert "position" in body
    assert "primary learning signal" not in body


def test_badham_never_exposes_current_correct_category() -> None:
    body = all_dataset_keyed_reminder_bodies()["12badham2017deficits"]
    assert "stimulus_features" in body
    assert "rule_block_id" in body
    assert "feedback.is_correct" in body or "feedback.correct_category" in body
    assert "never read a current-trial correct category from `problem`" in body
    assert "problem['correct_category']" not in body
    for pat in ORACLE_LEAK_PATTERNS:
        assert not pat.search(body), pat.pattern


def test_ensure_idempotent_and_unaffected_path_stable() -> None:
    base = "Task A"
    out1 = ensure_history_robustness_block(base, dataset="3frey2017cct")
    out2 = ensure_history_robustness_block(out1, dataset="3frey2017cct")
    assert out1 == out2
    assert HISTORY_ROBUSTNESS_BLOCK in out1
    # Keyed dataset upgrades a legacy-embedded prompt to the keyed body.
    out3 = ensure_history_robustness_block(out1, dataset="5speekenbrink2008learning")
    assert "problem['cards']" in out3
    assert HISTORY_ROBUSTNESS_BLOCK not in out3
    out4 = ensure_history_robustness_block(out3, dataset="5speekenbrink2008learning")
    assert out4 == out3


def test_independent_unaffected_do_not_get_keyed_bandit_or_stopping_text() -> None:
    forbidden_snippets = (
        "K-way dict, K=4",
        "K-way dict, K=8",
        "0=continue, 1=stop",
        "rule_block_id",
        "was_correct` / `weather_outcome",
    )
    for alias in UNAFFECTED_ALIASES:
        block = resolve_history_robustness_block(alias)
        for snip in forbidden_snippets:
            assert snip not in block


def test_tls_dataset_used_when_explicit_arg_omitted() -> None:
    task = "adaptive"
    with pics_v3_prompt_robustness_scope(True, dataset="guan_2020_stopping"):
        out = maybe_attach_history_robustness_after_task_description(task)
    assert "0=continue" in out
    assert HISTORY_ROBUSTNESS_BLOCK not in out
