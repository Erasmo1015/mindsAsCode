"""Focused PICS v4 checks. Does not replace the compact-trial suite."""
from __future__ import annotations

import json
from pathlib import Path

from utils.teh.aamas_v0_lossless_trials import (
    TRIAL_PROMPT_POLICY_ID,
    assert_legacy_output_not_resumed,
    render_schedule_text,
    set_pics_run_seeds,
)
from utils.teh.pics_aamas_v0 import apply_aamas_v0_prompt_body, reminder_decision
from utils.teh.pics_v3_prompt_robustness import (
    _DATASET_KEYED_REMINDER_BODIES,
    _SEQUENTIAL_RL_KOOL_BODY_V4,
    _SEQUENTIAL_RL_SCHULZ_BODY_V4,
    _SEQUENTIAL_RL_STEYVERS_BODY_V4,
)
from utils.teh.pics_v4 import (
    ADDITIONAL_PROMPT_POLICY_ID,
    CANONICAL_DATASETS,
    KIND_TARGET_ONLY,
    METHOD_VERSION,
    TEXT_PATH,
    apply_pics_v4_prompt_body,
    assert_pics_v4_resume,
    configure_pics_v4,
    dataset_names_in_uniform_block,
    uniform_additional_prompt_sha256,
    uniform_additional_prompt_text,
    using_pics_v4,
    write_pics_v4_marker,
)


def test_uniform_block_matches_saved_file_and_has_no_dataset_names():
    text = uniform_additional_prompt_text()
    assert TEXT_PATH.read_text(encoding="utf-8") == text + "\n" or TEXT_PATH.read_text(
        encoding="utf-8"
    ) == text
    assert not dataset_names_in_uniform_block()
    assert "SEQUENTIAL_RL_BEHAVIOR_REMINDER_V3" not in text
    assert _SEQUENTIAL_RL_KOOL_BODY_V4 not in text
    assert "weather_outcome" not in text
    assert "Do not read current-trial weather" not in text


def test_approved_bodies_appear_once():
    text = uniform_additional_prompt_text()
    assert ADDITIONAL_PROMPT_POLICY_ID == "uniform_additional_prompt_v2"
    assert "apply this section" not in text
    assert "Otherwise ignore it" not in text
    assert text.count(
        "Apply only the section whose task type matches the task described above. "
        "Ignore every section that does not match."
    ) == 1
    for body in (
        _DATASET_KEYED_REMINDER_BODIES["14kool2016when"],
        _DATASET_KEYED_REMINDER_BODIES["12badham2017deficits"],
        _DATASET_KEYED_REMINDER_BODIES["guan_2020_stopping"],
    ):
        assert text.count(body.strip()) == 1
    assert _SEQUENTIAL_RL_STEYVERS_BODY_V4.strip() not in text
    assert _SEQUENTIAL_RL_SCHULZ_BODY_V4.strip() not in text
    for phrase in (
        "does not compute an optimal reward-maximizing policy",
        "`history` may be empty; when present it has `action` and `reward` (use `.get`—reward may be absent or None)",
        "never add None to a numeric accumulator",
        "K=4",
        "K=8",
        "do not impose a fixed schedule such as always explore early then exploit",
        "`.get('reward', 0)` is insufficient when the key exists with value None",
        "non-empty usable history must not collapse to uniform",
        "rather than stacking many weakly supported mechanisms",
        "Do not invent spatial coordinates",
        "may appropriately remain close to uniform",
        "Stage 1 (`stage==1`)",
        "Stage 2 (`stage==2`)",
        "rule_block_id",
        "feedback.is_correct",
        "values_observed",
        "0=continue, 1=stop",
    ):
        assert text.count(phrase) == 1


def test_every_dataset_receives_the_same_bytes():
    digest = uniform_additional_prompt_sha256()
    rendered = [apply_pics_v4_prompt_body("Task for the participant.") for _ in CANONICAL_DATASETS]
    assert len(rendered) == 15
    assert len(set(rendered)) == 1
    block = uniform_additional_prompt_text()
    assert rendered[0].count(block) == 1
    assert digest == uniform_additional_prompt_sha256()


def test_aamas_reminder_routing_is_unchanged():
    configure_pics_v4(False)
    assert using_pics_v4() is False
    assert reminder_decision("14kool2016when").reminder_id == "dataset_keyed_kool_v1"
    assert reminder_decision("steyvers_2009_bandit").reminder_id == "sequential_rl_reminder_v4_steyvers"
    assert reminder_decision("5speekenbrink2008learning").block is None
    kool = apply_aamas_v0_prompt_body("Task.", "14kool2016when")
    guan = apply_aamas_v0_prompt_body("Task.", "guan_2020_stopping")
    assert kool != guan
    assert "PICS_V4_UNIFORM_ADDITIONAL_PROMPT" not in kool


def test_serializer_stays_compact_faithful():
    assert TRIAL_PROMPT_POLICY_ID == "compact_faithful_trial_v1"
    text = render_schedule_text(
        [[
            {
                "problem": {"option_keys": ["A", "B"], "participant_id": 0},
                "action": 0,
                "history": [],
                "_prompt_participant_id": 0,
                "_block_id": "block",
                "_block_pos": 0,
            }
        ]],
        "mixed_gambles",
    )
    assert text.startswith("participant ")
    assert '"observed_action":' in text
    assert "Trial " not in text
    assert "history_prefix" not in text


def test_audited_prompts_stay_within_14000():
    path = (
        Path(__file__).resolve().parents[3]
        / "analysis_2026Sep"
        / "Sep30_pics_v4"
        / "reminder_coverage_v2.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["additional_prompt_policy"] == "uniform_additional_prompt_v2"
    assert payload["uniform_block_sha256"] == uniform_additional_prompt_sha256()
    seen = []
    for row in payload["datasets"]:
        for phase, pair in row["phases"].items():
            after = pair["uniform"]
            assert after["error"] is None
            assert after["over_cap"] is False
            assert int(after["tokens"]) <= 14000
            assert after["contract_copies"] == 1
            assert after["uniform_copies"] == 1
            assert after["parents_intact"] is True
            seen.append((row["dataset"], phase))
    assert len(seen) == 5 * 5


def test_resume_refuses_other_methods(tmp_path: Path):
    set_pics_run_seeds(run_seed=0, split_seed=0)
    v4 = tmp_path / KIND_TARGET_ONLY / "job_1"
    v4.mkdir(parents=True)
    write_pics_v4_marker(v4)
    payload = json.loads((v4 / "TRIAL_PROMPT_POLICY.json").read_text(encoding="utf-8"))
    assert payload["method_version"] == METHOD_VERSION
    assert payload["trial_prompt_policy"] == "compact_faithful_trial_v1"
    assert payload["additional_prompt_policy"] == ADDITIONAL_PROMPT_POLICY_ID
    assert payload["additional_prompt_sha256"] == uniform_additional_prompt_sha256()
    assert payload["split_seed"] == 0
    assert payload["pics_run_seed"] == 0
    assert_pics_v4_resume(v4)
    stale = json.loads((v4 / "TRIAL_PROMPT_POLICY.json").read_text(encoding="utf-8"))
    stale["additional_prompt_policy"] = "uniform_additional_prompt_v1"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(stale) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "uniform_additional_prompt_v1" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted uniform_additional_prompt_v1")
    stale["additional_prompt_policy"] = ADDITIONAL_PROMPT_POLICY_ID
    stale["additional_prompt_sha256"] = "0" * 64
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(stale) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted a different uniform-prompt hash")
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    try:
        assert_legacy_output_not_resumed(v4)
    except RuntimeError as exc:
        assert "pics_v4" in str(exc)
    else:
        raise AssertionError("AAMAS resume accepted a PICS v4 directory")
    aamas = tmp_path / "pics_aamas_v0_target_only" / "job_9"
    aamas.mkdir(parents=True)
    (aamas / "TRIAL_PROMPT_POLICY.json").write_text(
        json.dumps({"trial_prompt_policy": "compact_faithful_trial_v1", "prompt_policy": "compact_faithful_trial_v1"})
        + "\n",
        encoding="utf-8",
    )
    try:
        assert_pics_v4_resume(aamas)
    except RuntimeError as exc:
        assert "pics_v4" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted an AAMAS directory")
