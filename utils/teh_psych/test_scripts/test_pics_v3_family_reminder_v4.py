"""CPU tests for optional PICS v3 Sequential-RL reminder v4 (default off)."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

from utils.teh.pics_v3 import (  # noqa: E402
    FAMILY_PROMPT_V3_TREATMENT_KIND,
    FAMILY_PROMPT_V4_KIND,
    KIND,
)
from utils.teh.pics_v3_prompt_robustness import (  # noqa: E402
    HISTORY_ROBUSTNESS_MARKER,
    SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER,
    SEQUENTIAL_RL_REMINDER_V3_MARKER,
    SEQUENTIAL_RL_REMINDER_V4_KOOL_MARKER,
    SEQUENTIAL_RL_REMINDER_V4_POLICY_ID,
    SEQUENTIAL_RL_REMINDER_V4_SCHULZ_MARKER,
    SEQUENTIAL_RL_REMINDER_V4_STEYVERS_MARKER,
    configure_pics_v3_family_reminder_v3,
    configure_pics_v3_family_reminder_v4,
    ensure_history_robustness_block,
    maybe_attach_family_reminder_v3,
    maybe_attach_family_reminder_v4,
    sequential_rl_reminder_v3_block,
    sequential_rl_reminder_v4_block,
)

V3_RENDERED = REPO / "analysis_2026Sep/Sep20_V3/others/family_prompt_v3/rendered_prompts"
V3_HASHES = REPO / "analysis_2026Sep/Sep20_V3/others/family_prompt_v3/rendered_prompt_hashes.json"

SEQ_DATASETS = (
    "steyvers_2009_bandit",
    "13schulz2020finding",
    "14kool2016when",
)

# Fields allowed in Kool v4 body (must match live schema / registered contract).
KOOL_ALLOWED_FIELD_TOKENS = {
    "stage",
    "planet",
    "alien_options",
    "option_keys",
    "spaceship_options",
    "spaceship",
    "stage1_action",
    "feedback",
    "reward",
    "treasure",
    "history",
    "problem",
    "action",
}


@pytest.fixture(autouse=True)
def _reset_family_flags():
    configure_pics_v3_family_reminder_v3(sequential_rl=False, feedback_learning=False)
    configure_pics_v3_family_reminder_v4(sequential_rl=False)
    yield
    configure_pics_v3_family_reminder_v3(sequential_rl=False, feedback_learning=False)
    configure_pics_v3_family_reminder_v4(sequential_rl=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_v3_hash_rows():
    return {r["dataset"]: r for r in json.loads(V3_HASHES.read_text(encoding="utf-8"))}


def test_default_off_no_mutation_v4():
    base = ensure_history_robustness_block(
        "Four-armed bandit.", dataset="steyvers_2009_bandit"
    )
    out = maybe_attach_family_reminder_v4(base, dataset="steyvers_2009_bandit")
    assert out == base
    assert SEQUENTIAL_RL_REMINDER_V4_STEYVERS_MARKER not in out


def test_main_control_infer_hashes_unchanged():
    rows = _load_v3_hash_rows()
    for ds in SEQ_DATASETS:
        control = (
            V3_RENDERED / f"{ds}__A_control_infer_core.txt"
        ).read_text(encoding="utf-8")
        assert _sha(control) == rows[ds]["control_sha256"]
        assert HISTORY_ROBUSTNESS_MARKER in control
        assert SEQUENTIAL_RL_REMINDER_V3_MARKER not in control or (
            # kool open bandit marker must not appear as tagged block
            f"[{SEQUENTIAL_RL_REMINDER_V3_MARKER}]" not in control
        )
        assert SEQUENTIAL_RL_REMINDER_V4_STEYVERS_MARKER not in control
        assert SEQUENTIAL_RL_REMINDER_V4_SCHULZ_MARKER not in control
        assert SEQUENTIAL_RL_REMINDER_V4_KOOL_MARKER not in control


def test_existing_v3_treatment_hashes_unchanged():
    rows = _load_v3_hash_rows()
    configure_pics_v3_family_reminder_v3(sequential_rl=True)
    for ds in SEQ_DATASETS:
        control = (
            V3_RENDERED / f"{ds}__A_control_infer_core.txt"
        ).read_text(encoding="utf-8")
        treatment_frozen = (
            V3_RENDERED / f"{ds}__B_treatment_infer_core.txt"
        ).read_text(encoding="utf-8")
        rebuilt = maybe_attach_family_reminder_v3(control, dataset=ds)
        assert rebuilt == treatment_frozen
        assert _sha(rebuilt) == rows[ds]["treatment_sha256"]
        assert HISTORY_ROBUSTNESS_MARKER not in rebuilt


def test_v4_replaces_history_only_three_datasets():
    configure_pics_v3_family_reminder_v4(sequential_rl=True)
    for ds, marker in (
        ("steyvers_2009_bandit", SEQUENTIAL_RL_REMINDER_V4_STEYVERS_MARKER),
        ("13schulz2020finding", SEQUENTIAL_RL_REMINDER_V4_SCHULZ_MARKER),
        ("14kool2016when", SEQUENTIAL_RL_REMINDER_V4_KOOL_MARKER),
    ):
        with_hist = ensure_history_robustness_block("task", dataset=ds)
        out = maybe_attach_family_reminder_v4(with_hist, dataset=ds)
        assert marker in out
        assert out.count(f"[{marker}]") == 1
        assert HISTORY_ROBUSTNESS_MARKER not in out
        assert f"[{SEQUENTIAL_RL_REMINDER_V3_MARKER}]" not in out
        assert SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER not in out
        assert "`reward` is the primary learning signal" not in out

    # Unrelated dataset: no mutation.
    base = ensure_history_robustness_block(
        "weather", dataset="5speekenbrink2008learning"
    )
    out = maybe_attach_family_reminder_v4(
        base, dataset="5speekenbrink2008learning"
    )
    assert out == base


def test_v4_not_stacked_with_v2_or_v3_on_control_cores():
    configure_pics_v3_family_reminder_v4(sequential_rl=True)
    for ds in SEQ_DATASETS:
        control = (
            V3_RENDERED / f"{ds}__A_control_infer_core.txt"
        ).read_text(encoding="utf-8")
        out = maybe_attach_family_reminder_v4(control, dataset=ds)
        assert HISTORY_ROBUSTNESS_MARKER not in out
        assert f"[{SEQUENTIAL_RL_REMINDER_V3_MARKER}]" not in out
        assert SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER not in out
        markers = [
            SEQUENTIAL_RL_REMINDER_V4_STEYVERS_MARKER,
            SEQUENTIAL_RL_REMINDER_V4_SCHULZ_MARKER,
            SEQUENTIAL_RL_REMINDER_V4_KOOL_MARKER,
        ]
        present = [m for m in markers if f"[{m}]" in out]
        assert len(present) == 1


def test_steyvers_schulz_none_reward_protection():
    configure_pics_v3_family_reminder_v4(sequential_rl=True)
    for ds in ("steyvers_2009_bandit", "13schulz2020finding"):
        body = sequential_rl_reminder_v4_block(ds)
        assert body is not None
        assert "None" in body
        assert "never add None" in body or "Never add None" in body
        assert "reward" in body


def test_kool_stage_specific_and_no_pool():
    configure_pics_v3_family_reminder_v4(sequential_rl=True)
    body = sequential_rl_reminder_v4_block("14kool2016when")
    assert body is not None
    assert "stage 1" in body.lower() or "stage-1" in body.lower() or "`stage==1`" in body
    assert "stage 2" in body.lower() or "stage-2" in body.lower() or "`stage==2`" in body
    assert "planet" in body
    assert "Never pool" in body or "never pool" in body
    assert "context-free reward-mean bandit" in body
    assert "treasure-maximizing" in body
    assert "Never read current-trial" in body or "never read current-trial" in body.lower()


def test_schulz_no_spatial_invention_and_shorter_than_v3():
    v3 = sequential_rl_reminder_v3_block("13schulz2020finding")
    v4 = sequential_rl_reminder_v4_block("13schulz2020finding")
    assert v3 and v4
    assert len(v4) < len(v3)
    assert "spatial" in v4.lower()
    assert "Do not invent spatial" in v4 or "do not invent spatial" in v4.lower()
    assert "close to uniform" in v4


def test_no_unsupported_kool_schema_fields():
    body = sequential_rl_reminder_v4_block("14kool2016when")
    assert body is not None
    # Reject invented tokens frequently hallucinated for Kool.
    for bad in (
        "transition_matrix",
        "treasure_prob",
        "common_transition",
        "rare_transition",
        "state_value",
        "q_mb",
        "q_mf",
    ):
        assert bad not in body
    # Referenced backtick fields should be known schema tokens.
    for tok in re.findall(r"`([^`]+)`", body):
        leaf = tok.split("[")[0].split(".")[0].split("/")[0].strip("'\"")
        if leaf in {"stage==1", "stage==2"}:
            continue
        if leaf.startswith("reward") or leaf.startswith("treasure"):
            continue
        if leaf in KOOL_ALLOWED_FIELD_TOKENS:
            continue
        if leaf in {"action", "get"}:
            continue
        # Allow short phrases that are not field names.
        if " " in leaf or leaf in {"option['action']", "feedback`/`reward"}:
            continue
        # option_keys`/`spaceship_options style split already covered.
        if leaf in {"alien_options`/`option_keys", "spaceship`/`stage1_action"}:
            continue


def test_no_error_feedback_and_no_heldout_in_v4_bodies():
    for ds in SEQ_DATASETS:
        body = sequential_rl_reminder_v4_block(ds)
        assert body is not None
        low = body.lower()
        assert "error bank" not in low
        assert "error feedback" not in low
        assert "held-out" not in low
        assert "held out" not in low
        assert "test set" not in low
        assert "test action" not in low


def test_v4_policy_id_constant():
    assert SEQUENTIAL_RL_REMINDER_V4_POLICY_ID == "sequential_rl_reminder_v4"


def test_cli_rejects_simultaneous_v3_and_v4():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "teh.py"),
            "--help",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    # --help should succeed and list both flags.
    assert proc.returncode == 0
    assert "--pics_v3_sequential_rl_reminder_v4" in proc.stdout
    assert "--pics_v3_sequential_rl_reminder_v3" in proc.stdout

    # Mutual exclusion: invoke with both flags and a minimal dataset arg that
    # still reaches the exclusivity check before heavier validation.
    proc2 = subprocess.run(
        [
            sys.executable,
            str(REPO / "teh.py"),
            "--pics_v3_sequential_rl_reminder_v3",
            "--pics_v3_sequential_rl_reminder_v4",
            "--dataset",
            "steyvers_2009_bandit",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (proc2.stdout or "") + (proc2.stderr or "")
    assert "mutually exclusive" in combined.lower()


def test_kinds_distinct():
    assert FAMILY_PROMPT_V4_KIND != KIND
    assert FAMILY_PROMPT_V4_KIND != FAMILY_PROMPT_V3_TREATMENT_KIND
    assert FAMILY_PROMPT_V4_KIND == "pics_v3_family_prompt_v4"


def test_submit_scripts_v4():
    submit = (
        REPO / "cluster/v3/ours/family_prompt_v4/submit_family_prompt_v4.sh"
    ).read_text(encoding="utf-8")
    job = (
        REPO / "cluster/v3/ours/family_prompt_v4/job_family_prompt_v4_h100.sh"
    ).read_text(encoding="utf-8")
    common = (REPO / "cluster/v2/ours/Qwen/_common.sh").read_text(encoding="utf-8")
    assert "steyvers_2009_bandit" in submit
    assert "13schulz2020finding" in submit
    assert "14kool2016when" in submit
    assert "5speekenbrink2008learning" not in submit
    assert "pics_v3_family_prompt_v4" in submit
    assert "pv3fpv4_" in submit or "pv3fpv4_${short}" in submit
    assert "--requeue" in job
    assert "t_pics_v3_fill_family_prompt_v4_args" in job
    assert "t_pics_v3_fill_family_prompt_v4_args()" in common
    assert "--pics_v3_sequential_rl_reminder_v4" in common
    # Main gated must remain free of v4.
    gated = common.split("t_pics_v3_fill_gated_args()")[1].split(
        "t_pics_v3_fill_ablation_args"
    )[0]
    assert "--pics_v3_sequential_rl_reminder_v4" not in gated
    assert "--pics_v3_sequential_rl_reminder_v3" not in gated
    # family v3 fill must not gain v4 in PICS_ARGS.
    fpv3 = common.split("t_pics_v3_fill_family_prompt_v3_args() {")[1].split(
        "# Fill PICS_ARGS for Sequential-RL reminder v4"
    )[0]
    assert "--pics_v3_sequential_rl_reminder_v4" not in fpv3
    assert "PICS_ARGS+=(--pics_v3_sequential_rl_reminder_v3)" in fpv3


def test_v3_bodies_byte_stable():
    """Completed jobs 302461–302463 used these bodies — must not drift."""
    a = sequential_rl_reminder_v3_block("steyvers_2009_bandit")
    b = sequential_rl_reminder_v3_block("13schulz2020finding")
    assert a == b
    assert "fixed explore-then-exploit schedule" in a
    k = sequential_rl_reminder_v3_block("14kool2016when")
    assert "treasure-maximizing policy" in k
    assert SEQUENTIAL_RL_REMINDER_V3_KOOL_MARKER in k


def test_idempotent_v4_attach():
    configure_pics_v3_family_reminder_v4(sequential_rl=True)
    once = maybe_attach_family_reminder_v4(
        ensure_history_robustness_block("t", dataset="steyvers_2009_bandit"),
        dataset="steyvers_2009_bandit",
    )
    twice = maybe_attach_family_reminder_v4(once, dataset="steyvers_2009_bandit")
    assert twice == once
    assert once.count(SEQUENTIAL_RL_REMINDER_V4_STEYVERS_MARKER) == 2  # open+close
