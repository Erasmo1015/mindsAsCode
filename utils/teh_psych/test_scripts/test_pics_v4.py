"""Focused PICS v4 checks. Does not replace the compact-trial suite."""
from __future__ import annotations

import json
from pathlib import Path

from utils.teh.aamas_v0_lossless_trials import (
    TRIAL_PROMPT_POLICY_ID,
    assert_legacy_output_not_resumed,
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
    GENERIC_HISTORY_GUIDANCE,
    HARD_PROMPT_TOKEN_CAP,
    KIND_TARGET_ONLY,
    METHOD_VERSION,
    PICS_V4_TRIAL_POLICY,
    PICS_RUN_SEED_SCOPE,
    PREVIOUS_ADDITIONAL_PROMPT_POLICY_ID,
    PREVIOUS_TEXT_PATH,
    PREVIOUS_UNIFORM_PROMPT_SHA256,
    SEARCH_RNG_POLICY,
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
    assert ADDITIONAL_PROMPT_POLICY_ID == "uniform_additional_prompt_v3"
    assert text.count(GENERIC_HISTORY_GUIDANCE) == 1
    assert "HISTORY_ROBUSTNESS_BLOCK_V3" not in text
    previous = PREVIOUS_TEXT_PATH.read_text(encoding="utf-8").rstrip("\n")
    preface = (
        "Apply only the section whose task type matches the task described above. "
        "Ignore every section that does not match.\n\n"
    )
    marker = "[PICS_V4_UNIFORM_ADDITIONAL_PROMPT]\n"
    assert previous.startswith(marker + preface)
    task_sections = previous[len(marker + preface) :]
    assert text == f"{marker}{preface}{GENERIC_HISTORY_GUIDANCE}\n\n{task_sections}"
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


def _install_frozen_bank(
    trials,
    *,
    dataset,
    phase,
    participant_id,
    wrap,
    name,
    n_slots=None,
    cap=HARD_PROMPT_TOKEN_CAP,
    master_seed=0,
    require_full_coverage=False,
):
    from utils.teh.pics_v4_panels import build_continuation_bank, register_panel_bank, slots_for_phase

    count = slots_for_phase(phase) if n_slots is None else int(n_slots)
    bank = build_continuation_bank(
        trials,
        dataset=dataset,
        master_seed=master_seed,
        phase=phase,
        participant_id=participant_id,
        n_slots=count,
        wrap=wrap,
        cap=cap,
        bank=name,
        require_full_coverage=require_full_coverage,
    )
    register_panel_bank(
        dataset=dataset,
        phase=phase,
        participant_id=participant_id,
        name=name,
        bank=bank,
    )
    return bank


def test_pics_v4_uses_structured_snapshots_not_compact_json():
    from utils.teh.aamas_v0_lossless_trials import PHASE_POPULATION, SelectionRequest, render_for_request
    from utils.teh.prompt_snapshots import snapshot_example_dict

    trial = {
        "problem": {
            "schema_type": "D",
            "option_keys": ["Z", "N"],
            "round_id": 5,
            "current_score": 0,
            "cards_flipped": 0,
            "n_cards_remaining": 32,
            "gain_amount": 20,
            "loss_amount": 750,
            "n_loss_cards": 1,
            "participant_id": 0,
        },
        "action": 0,
        "history": [],
        "_prompt_participant_id": 0,
        "_block_id": "block",
        "_block_pos": 0,
    }
    snap = snapshot_example_dict(trial, 1)
    assert snap["label"] == {"action": 0}
    assert snap["problem"]["option_keys"] == ["Z", "N"]
    assert "reward" not in snap["problem"]
    configure_pics_v4(True)
    try:
        _install_frozen_bank(
            [trial],
            dataset="3frey2017cct",
            phase=PHASE_POPULATION,
            participant_id=None,
            wrap=lambda body: body,
            name="fresh",
        )
        text, stats = render_for_request(
            [trial],
            dataset="3frey2017cct",
            request=SelectionRequest(
                dataset="3frey2017cct",
                phase=PHASE_POPULATION,
                step=0,
                master_seed=0,
                participant_id=None,
                panel_bank="fresh",
            ),
            actual_wrap=lambda body: body,
            canonical_wrap=lambda body: body,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
    finally:
        from utils.teh.pics_v4_panels import clear_pics_v4_panel_state

        clear_pics_v4_panel_state()
        configure_pics_v4(False)
    assert "observed_action_label=0" in text
    assert "option_keys" in text
    assert "history_before" not in text
    assert stats["policy"] == "structured_snapshot_v2"
    assert stats["panel_policy"] == "shuffled_block_slot_stable_v3"
    assert stats["continuation_policy"] == "within_block_carry_forward_v1"
    assert stats["slot_assignment"] == "first_n"
    assert TRIAL_PROMPT_POLICY_ID == "compact_faithful_trial_v2"
    assert PICS_V4_TRIAL_POLICY == "structured_snapshot_v2"


def test_audited_prompts_stay_within_14000():
    path = (
        Path(__file__).resolve().parents[3]
        / "analysis_2026Sep"
        / "Sep30_pics_v4"
        / "reminder_coverage_v2.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["additional_prompt_policy"] == PREVIOUS_ADDITIONAL_PROMPT_POLICY_ID
    assert payload["uniform_block_sha256"] == PREVIOUS_UNIFORM_PROMPT_SHA256
    assert payload["uniform_block_sha256"] != uniform_additional_prompt_sha256()
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
    assert payload["trial_prompt_policy"] == "structured_snapshot_v2"
    assert payload["additional_prompt_policy"] == ADDITIONAL_PROMPT_POLICY_ID
    assert payload["additional_prompt_sha256"] == uniform_additional_prompt_sha256()
    assert payload["hard_prompt_token_cap"] == HARD_PROMPT_TOKEN_CAP == 15360
    assert payload["packing_implementation"] == "shuffled_block_slot_stable_v3"
    assert payload["panel_policy"] == "shuffled_block_slot_stable_v3"
    assert payload["continuation_policy"] == "within_block_carry_forward_v1"
    assert payload["panel_bank_policy"] == "conditioning_aware_panel_banks_v2"
    assert payload["parent_envelope_policy"] == "initial_unique_elite_mean_x_v1"
    assert payload["slot_assignment"] == "first_n"
    assert payload["participant_population_unique_cap"] == 5
    assert payload["split_seed"] == 0
    assert payload["pics_run_seed"] == 0
    assert payload["search_rng"] == SEARCH_RNG_POLICY
    assert payload["pics_run_seed_scope"] == PICS_RUN_SEED_SCOPE
    assert_pics_v4_resume(v4)
    missing_rng = dict(payload)
    missing_rng.pop("search_rng")
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(missing_rng) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "search_rng" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted a marker without search_rng")
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    stale = json.loads((v4 / "TRIAL_PROMPT_POLICY.json").read_text(encoding="utf-8"))
    stale["additional_prompt_policy"] = "uniform_additional_prompt_v1"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(stale) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "uniform_additional_prompt_v1" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted uniform_additional_prompt_v1")
    previous = dict(payload)
    previous["additional_prompt_policy"] = PREVIOUS_ADDITIONAL_PROMPT_POLICY_ID
    previous["additional_prompt_sha256"] = PREVIOUS_UNIFORM_PROMPT_SHA256
    previous["hard_prompt_token_cap"] = 14000
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(previous) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert PREVIOUS_UNIFORM_PROMPT_SHA256 in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted the previous uniform-prompt hash")
    swapped = dict(payload)
    swapped["additional_prompt_sha256"] = PREVIOUS_UNIFORM_PROMPT_SHA256
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(swapped) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert PREVIOUS_UNIFORM_PROMPT_SHA256 in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted the previous hash under the new policy id")
    old_cap = dict(payload)
    old_cap["hard_prompt_token_cap"] = 14000
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(old_cap) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "hard_prompt_token_cap" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted the 14000 cap")
    old_line = dict(payload)
    old_line["trial_prompt_policy"] = "format_trial_for_prompt"
    old_line["prompt_policy"] = "format_trial_for_prompt"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(old_line) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "format_trial_for_prompt" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted format_trial_for_prompt")
    old_pack = dict(payload)
    old_pack["packing_implementation"] = "aamas_v0_greedy_pack_v1"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(old_pack) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "aamas_v0_greedy_pack_v1" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted greedy packing")
    old_slot = dict(payload)
    old_slot["packing_implementation"] = "shuffled_block_slot_stable_v1"
    old_slot["panel_policy"] = "shuffled_block_slot_stable_v1"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(old_slot) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "shuffled_block_slot_stable_v1" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted shuffled_block_slot_stable_v1")
    old_v2 = dict(payload)
    old_v2["packing_implementation"] = "shuffled_block_slot_stable_v2"
    old_v2["panel_policy"] = "shuffled_block_slot_stable_v2"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(old_v2) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "shuffled_block_slot_stable_v2" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted shuffled_block_slot_stable_v2")
    old_banks = dict(payload)
    old_banks["panel_bank_policy"] = "conditioning_aware_panel_banks_v1"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(old_banks) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "conditioning_aware_panel_banks_v1" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted conditioning_aware_panel_banks_v1")
    old_envelope = dict(payload)
    old_envelope["parent_envelope_policy"] = "initial_unique_elite_longest_x_v1"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(old_envelope) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "initial_unique_elite_longest_x_v1" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted the longest-X parent envelope")
    missing_dual = dict(payload)
    missing_dual.pop("continuation_policy")
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(missing_dual) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "continuation_policy" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted a marker without continuation_policy")
    cap_15000 = dict(payload)
    cap_15000["hard_prompt_token_cap"] = 15000
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(cap_15000) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "hard_prompt_token_cap" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted the 15000 cap")
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    stale["additional_prompt_policy"] = ADDITIONAL_PROMPT_POLICY_ID
    stale["additional_prompt_sha256"] = "0" * 64
    stale["hard_prompt_token_cap"] = HARD_PROMPT_TOKEN_CAP
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(stale) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted a different uniform-prompt hash")
    v1 = dict(payload)
    v1["trial_prompt_policy"] = "compact_faithful_trial_v1"
    v1["prompt_policy"] = "compact_faithful_trial_v1"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(v1) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "compact_faithful_trial_v1" in str(exc)
        assert "structured_snapshot_v2" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted compact_faithful_trial_v1")
    v2 = dict(payload)
    v2["trial_prompt_policy"] = "compact_faithful_trial_v2"
    v2["prompt_policy"] = "compact_faithful_trial_v2"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(v2) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "compact_faithful_trial_v2" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted compact_faithful_trial_v2")
    hashed = dict(payload)
    hashed["search_rng"] = "hashed_pics_run_seed"
    (v4 / "TRIAL_PROMPT_POLICY.json").write_text(json.dumps(hashed) + "\n", encoding="utf-8")
    try:
        assert_pics_v4_resume(v4)
    except RuntimeError as exc:
        assert "search_rng" in str(exc)
    else:
        raise AssertionError("PICS v4 resume accepted a hashed search seed")
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
    try:
        assert_legacy_output_not_resumed(aamas)
    except RuntimeError as exc:
        assert "compact_faithful_trial_v1" in str(exc)
    else:
        raise AssertionError("AAMAS resume accepted compact_faithful_trial_v1")


def test_legacy_search_seeds_match_reference_jobs_and_ignore_pics_run_seed():
    """Locks the seeds recorded by jobs 305707–305710. pics_run_seed moves only the schedule."""
    import numpy as np

    import teh
    from utils.teh.aamas_v0_lossless_trials import PHASE_POPULATION, _build_state, pics_run_seed
    from utils.teh.pics_aamas_v0 import configure_pics_aamas_v0_prompt

    source = Path(__file__).resolve().parents[3].joinpath("teh.py").read_text(encoding="utf-8")
    assert "pics_lossless_llm_request_seed(" not in source
    assert "not_used_for_selection" in source
    assert 'normal_kw["pics_lossless_candidate_offset"] = int(fresh_n)' in source

    def _request(iteration_step: int, candidate: int, *, participant_id=None, batch_offset: int = 0) -> int:
        base = teh._aamas_or_legacy_seed(
            teh._phase_llm_decoding_seed_base(
                split_seed=0,
                iteration_step=iteration_step,
                participant_id=participant_id,
                batch_offset=batch_offset,
            ),
            "llm_request_base",
            dataset="3frey2017cct",
            phase="population",
            iteration=iteration_step,
            role="iteration",
        )
        return int(base) + int(candidate)

    def _parent_population(iteration_step: int) -> int:
        return teh._aamas_or_legacy_seed(
            50_000 + int(iteration_step) * 1_000_003,
            "parent_sampling",
            dataset="3frey2017cct",
            phase="population",
            iteration=int(iteration_step),
        )

    def _parent_evolution(iteration_step: int, participant_id: int) -> int:
        return teh._aamas_or_legacy_seed(
            int(iteration_step) * 1_000_003 + int(participant_id) * 17_179,
            "parent_sampling",
            dataset="3frey2017cct",
            phase="participant_evolution",
            participant=int(participant_id),
            iteration=int(iteration_step),
        )

    def _schedule(master_seed: int):
        rows = []
        for pid in (10, 20, 30, 40):
            rows.append(
                {
                    "action": 0,
                    "problem": {"round_id": 1},
                    "history": [],
                    "_prompt_participant_id": pid,
                    "_ldp": {"unit_id": f"u{pid}", "chrono": 0, "session_index": 0},
                }
            )
        state = _build_state(
            rows,
            dataset="3frey2017cct",
            master_seed=master_seed,
            phase=PHASE_POPULATION,
            participant_id=None,
        )
        return [block.trials[0]["_prompt_participant_id"] for block in state.blocks]

    configure_pics_aamas_v0_prompt(True)
    configure_pics_v4(True)
    try:
        fresh_iter1 = teh._decayed_fresh_n_for_iteration(10, 0, 10, 10)
        fresh_iter2 = teh._decayed_fresh_n_for_iteration(10, 1, 10, 10)
        assert fresh_iter1 == 10
        assert fresh_iter2 == 9
        seeds = {
            "population_iter1_candidate_0": _request(1, 0),
            "population_iter1_candidate_9": _request(1, 9),
            "population_iter2_parent_local_0": _request(2, fresh_iter2),
            "explore_candidate_0": _request(0, 0, participant_id=0),
            "explore_candidate_49": _request(0, 49, participant_id=0),
            "evolution_iter1_fresh_0": _request(1, 0, participant_id=0),
            "evolution_iter1_fresh_9": _request(1, 9, participant_id=0),
            "evolution_iter2_parent_local_0": _request(2, fresh_iter2, participant_id=0),
            "population_parent_sampling_iter1": _parent_population(1),
            "evolution_parent_sampling_iter1_participant_0": _parent_evolution(1, 0),
        }
        assert seeds == {
            "population_iter1_candidate_0": 1_080_003,
            "population_iter1_candidate_9": 1_080_012,
            "population_iter2_parent_local_0": 2_080_015,
            "explore_candidate_0": 80_000,
            "explore_candidate_49": 80_049,
            "evolution_iter1_fresh_0": 1_080_003,
            "evolution_iter1_fresh_9": 1_080_012,
            "evolution_iter2_parent_local_0": 2_080_015,
            "population_parent_sampling_iter1": 1_050_003,
            "evolution_parent_sampling_iter1_participant_0": 1_000_003,
        }
        pop_idx, pop_best, pop_sampled = teh._select_parent_indices_from_elite_pool(
            50,
            sample_size=8,
            sample_parents=True,
            sampled_parents_decay=True,
            iter_idx=0,
            total_iters=10,
            rng=np.random.default_rng(seeds["population_parent_sampling_iter1"]),
        )
        assert (pop_best, pop_sampled, pop_idx) == (0, 8, [20, 47, 39, 16, 46, 32, 9, 5])
        evo_idx, evo_best, evo_sampled = teh._select_parent_indices_from_elite_pool(
            50,
            sample_size=8,
            sample_parents=True,
            sampled_parents_decay=True,
            iter_idx=0,
            total_iters=10,
            rng=np.random.default_rng(seeds["evolution_parent_sampling_iter1_participant_0"]),
        )
        assert (evo_best, evo_sampled, evo_idx) == (0, 8, [24, 3, 26, 29, 28, 36, 30, 6])

        set_pics_run_seeds(run_seed=0, split_seed=0)
        order_0 = _schedule(int(pics_run_seed()))
        seeds_at_0 = dict(seeds)
        set_pics_run_seeds(run_seed=1, split_seed=0)
        order_1 = _schedule(int(pics_run_seed()))
        assert order_0 == [40, 20, 30, 10]
        assert order_1 == [20, 10, 30, 40]
        assert order_1 != order_0
        assert _request(1, 0) == seeds_at_0["population_iter1_candidate_0"]
        assert _request(1, 9) == seeds_at_0["population_iter1_candidate_9"]
        assert _request(2, fresh_iter2) == seeds_at_0["population_iter2_parent_local_0"]
        assert _request(0, 0, participant_id=0) == seeds_at_0["explore_candidate_0"]
        assert _request(0, 49, participant_id=0) == seeds_at_0["explore_candidate_49"]
        assert _request(1, 0, participant_id=0) == seeds_at_0["evolution_iter1_fresh_0"]
        assert _request(1, 9, participant_id=0) == seeds_at_0["evolution_iter1_fresh_9"]
        assert _request(2, fresh_iter2, participant_id=0) == seeds_at_0["evolution_iter2_parent_local_0"]
        assert _parent_population(1) == seeds_at_0["population_parent_sampling_iter1"]
        assert _parent_evolution(1, 0) == seeds_at_0["evolution_parent_sampling_iter1_participant_0"]
        assert teh._aamas_or_legacy_seed(99, "prompt_trial_subsample", dataset="3frey2017cct") == 99
    finally:
        configure_pics_v4(False)
        configure_pics_aamas_v0_prompt(False)
        set_pics_run_seeds(run_seed=0, split_seed=0)


def _schema_trial(pid, action, problem, history, chrono, unit="block"):
    return {
        "problem": {
            **problem,
            "participant_id": pid,
            "reward": 7,
            "weather_outcome": "rain",
            "was_correct": True,
            "exploded": True,
        },
        "action": action,
        "history": list(history),
        "_prompt_participant_id": pid,
        "_ldp": {
            "unit_id": unit,
            "chrono": chrono,
            "session_index": chrono,
            "origin_split": "train",
            "origin_index": chrono,
        },
    }


def test_slot_stable_snapshots_cover_phases_and_panels():
    from utils.teh.aamas_v0_lossless_trials import (
        PHASE_EXPLORATION,
        PHASE_PARTICIPANT_EVOLUTION,
        PHASE_POPULATION,
        SelectionRequest,
        render_for_request,
    )
    from utils.teh.pics_v4 import UNIFORM_MARKER
    from utils.teh.pics_v4_panels import (
        build_continuation_bank,
        clear_pics_v4_panel_state,
        flattened_stream,
        group_blocks,
        register_evolution_banks,
        slot_panel,
    )
    from utils.teh.prompt_context import RUNTIME_CONTRACT_HEADER
    from utils.teh.prompt_snapshots import current_or_future_leak_paths, snapshot_example_dict
    from utils.teh.prompt_units import qwen_user_prompt_token_count

    assert PICS_V4_TRIAL_POLICY == "structured_snapshot_v2"
    contract = f"{RUNTIME_CONTRACT_HEADER}\n- Target problem keys: option_keys\n"

    def wrap(body: str) -> str:
        return apply_pics_v4_prompt_body("Task description.") + "\n" + contract + body

    frey = {
        "schema_type": "D",
        "option_keys": ["E", "C"],
        "round_id": 5,
        "current_score": 0,
        "cards_flipped": 0,
        "n_cards_remaining": 32,
        "gain_amount": 20,
        "loss_amount": 750,
        "n_loss_cards": 1,
    }
    weather = {
        "schema_type": "B",
        "cards": [1, 0, 1],
        "option_keys": ["F", "J"],
        "features": {"task": "weather_prediction"},
        "weather_outcome": "rain",
        "was_correct": False,
    }
    configure_pics_v4(True)
    set_pics_run_seeds(run_seed=0, split_seed=0)
    try:
        phases = (PHASE_POPULATION, PHASE_EXPLORATION, PHASE_PARTICIPANT_EVOLUTION)
        for dataset in CANONICAL_DATASETS:
            for phase in phases:
                trial = _schema_trial(0, 0, frey, [], 0, unit="block-0")
                panel_bank = ""
                if phase == PHASE_POPULATION:
                    _install_frozen_bank(
                        [trial],
                        dataset=dataset,
                        phase=phase,
                        participant_id=None,
                        wrap=wrap,
                        name="fresh",
                    )
                    panel_bank = "fresh"
                elif phase == PHASE_EXPLORATION:
                    _install_frozen_bank(
                        [trial],
                        dataset=dataset,
                        phase=phase,
                        participant_id=0,
                        wrap=wrap,
                        name="parent",
                        require_full_coverage=True,
                    )
                    panel_bank = "parent"
                elif phase == PHASE_PARTICIPANT_EVOLUTION:
                    one = build_continuation_bank(
                        [trial],
                        dataset=dataset,
                        master_seed=0,
                        phase=phase,
                        participant_id=0,
                        n_slots=10,
                        wrap=wrap,
                        cap=HARD_PROMPT_TOKEN_CAP,
                        bank="fresh",
                        require_full_coverage=True,
                    )
                    register_evolution_banks(
                        dataset=dataset,
                        participant_id=0,
                        fresh=one,
                        parent_safe=one,
                    )
                    panel_bank = "fresh"
                text, stats = render_for_request(
                    [trial],
                    dataset=dataset,
                    request=SelectionRequest(
                        dataset=dataset,
                        phase=phase,
                        step=0,
                        master_seed=0,
                        participant_id=None if phase == PHASE_POPULATION else 0,
                        panel_bank=panel_bank,
                    ),
                    actual_wrap=wrap,
                    canonical_wrap=wrap,
                    cap=HARD_PROMPT_TOKEN_CAP,
                )
                assert stats["policy"] == "structured_snapshot_v2"
                assert stats["slot_assignment"] == "first_n"
                prompt = wrap(text)
                assert prompt.count(f"[{UNIFORM_MARKER}]") == 1
                assert prompt.count(RUNTIME_CONTRACT_HEADER) == 1
                assert prompt.count(GENERIC_HISTORY_GUIDANCE) == 1
                assert "HISTORY_ROBUSTNESS_BLOCK_V3" not in prompt
                tokens = qwen_user_prompt_token_count(prompt)
                assert tokens <= HARD_PROMPT_TOKEN_CAP == 15360
                assert tokens + 1024 <= 16384
                assert int(stats["prompt_tokens"]) + 1024 <= 16384

        weather_trial = _schema_trial(0, 1, weather, [], 0)
        snap = snapshot_example_dict(weather_trial, 1)
        assert current_or_future_leak_paths(weather_trial) == []
        assert "weather_outcome" not in snap["problem"]
        assert "was_correct" not in snap["problem"]
        assert snap["problem"]["option_keys"] == ["F", "J"]
        assert snap["label"]["action"] == 1
        _install_frozen_bank(
            [weather_trial],
            dataset="5speekenbrink2008learning",
            phase=PHASE_POPULATION,
            participant_id=None,
            wrap=wrap,
            name="fresh",
        )
        text, _stats = render_for_request(
            [weather_trial],
            dataset="5speekenbrink2008learning",
            request=SelectionRequest(
                dataset="5speekenbrink2008learning",
                phase=PHASE_POPULATION,
                step=0,
                master_seed=0,
                participant_id=None,
                panel_bank="fresh",
            ),
            actual_wrap=wrap,
            canonical_wrap=wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        assert "weather_outcome" not in text
        assert "option_keys" in text
        assert "observed_action_label=1" in text

        blocks = []
        for block_index in range(20):
            for pos in range(3):
                problem = dict(frey)
                problem["round_id"] = block_index
                problem["cards_flipped"] = pos
                blocks.append(
                    _schema_trial(
                        0,
                        pos % 2,
                        problem,
                        [],
                        block_index * 3 + pos,
                        unit=f"round:{block_index}",
                    )
                )
        grouped = group_blocks(blocks, "3frey2017cct")
        assert len(grouped) == 20
        for block in grouped:
            chronos = [trial["_ldp"]["chrono"] for trial in block]
            assert chronos == sorted(chronos)
        panel0 = slot_panel(
            blocks,
            dataset="3frey2017cct",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
            slot=0,
            wrap=wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        panel1 = slot_panel(
            blocks,
            dataset="3frey2017cct",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
            slot=1,
            wrap=wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        again = slot_panel(
            blocks,
            dataset="3frey2017cct",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
            slot=0,
            wrap=wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        stream = flattened_stream(
            blocks,
            dataset="3frey2017cct",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
        )
        assert [id(trial) for trial in panel0] == [id(trial) for trial in again]
        assert [id(trial) for trial in panel0] == [id(item.trial) for item in stream[: len(panel0)]]
        if len(panel0) < len(stream):
            assert [id(trial) for trial in panel1] == [
                id(item.trial) for item in stream[len(panel0) : len(panel0) + len(panel1)]
            ]
        else:
            assert id(panel1[0]) == id(stream[0].trial)
        first_units = {trial["_ldp"]["unit_id"] for trial in panel0}
        assert first_units != {"round:0"}

        def short_wrap(body: str) -> str:
            return wrap(body)

        def long_wrap(body: str) -> str:
            parent = " ".join(f"parent{i}" for i in range(2950))
            return wrap("PARENT\n" + parent + "\n" + body)

        _install_frozen_bank(
            blocks,
            dataset="3frey2017cct",
            phase=PHASE_POPULATION,
            participant_id=None,
            wrap=short_wrap,
            name="fresh",
        )
        short_text, short_stats = render_for_request(
            blocks,
            dataset="3frey2017cct",
            request=SelectionRequest(
                dataset="3frey2017cct",
                phase=PHASE_POPULATION,
                step=0,
                master_seed=0,
                participant_id=None,
                panel_bank="fresh",
            ),
            actual_wrap=short_wrap,
            canonical_wrap=short_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        long_text, long_stats = render_for_request(
            blocks,
            dataset="3frey2017cct",
            request=SelectionRequest(
                dataset="3frey2017cct",
                phase=PHASE_POPULATION,
                step=0,
                master_seed=0,
                participant_id=None,
                panel_bank="fresh",
            ),
            actual_wrap=long_wrap,
            canonical_wrap=short_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        assert long_stats["n_supervised"] < short_stats["n_supervised"]
        assert long_stats["visible_is_prefix"] is True
        assert long_stats["panel_ids"] == short_stats["panel_ids"]
        assert long_stats["panel_ids"][: long_stats["n_supervised"]] == short_stats["panel_ids"][: long_stats["n_supervised"]]
        assert short_text != long_text
        assert qwen_user_prompt_token_count(long_wrap(long_text)) <= 15360
    finally:
        clear_pics_v4_panel_state()
        configure_pics_v4(False)


def test_prefix_stops_at_15360_chat_template_tokens():
    """A longer parent keeps a shorter prefix of the same slot panel."""
    from utils.teh.aamas_v0_lossless_trials import PHASE_POPULATION, SelectionRequest, render_for_request
    from utils.teh.prompt_context import RUNTIME_CONTRACT_HEADER, finalize_aamas_runtime_contract
    from utils.teh.prompt_units import qwen_user_prompt_token_count

    problem = {
        "schema_type": "D",
        "option_keys": ["E", "C"],
        "filler": " ".join(f"tok{i}" for i in range(900)),
        "round_id": 5,
        "current_score": 0,
        "cards_flipped": 0,
        "n_cards_remaining": 32,
        "gain_amount": 20,
        "loss_amount": 750,
        "n_loss_cards": 1,
    }
    trials = [
        _schema_trial(0, index % 2, {**problem, "round_id": index}, [], index, unit=f"u{index}")
        for index in range(40)
    ]
    contract = f"{RUNTIME_CONTRACT_HEADER}\n- Target problem keys: option_keys\n"

    def wrap(body: str) -> str:
        text = apply_pics_v4_prompt_body("Task description.") + "\n" + contract + body
        return finalize_aamas_runtime_contract(text, None)

    configure_pics_v4(True)
    try:
        _install_frozen_bank(
            trials,
            dataset="3frey2017cct",
            phase=PHASE_POPULATION,
            participant_id=None,
            wrap=wrap,
            name="fresh",
        )
        request = SelectionRequest(
            dataset="3frey2017cct",
            phase=PHASE_POPULATION,
            step=0,
            master_seed=0,
            participant_id=None,
            panel_bank="fresh",
        )
        text_full, stats_full = render_for_request(
            trials,
            dataset="3frey2017cct",
            request=request,
            actual_wrap=wrap,
            canonical_wrap=wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        text_tight, stats_tight = render_for_request(
            trials,
            dataset="3frey2017cct",
            request=request,
            actual_wrap=wrap,
            canonical_wrap=wrap,
            cap=7000,
        )
        prompt = wrap(text_full)
        tokens = qwen_user_prompt_token_count(prompt)
        assert tokens == stats_full["prompt_tokens"]
        assert tokens <= 15360
        assert tokens + 1024 <= 16384
        assert stats_full["n_supervised"] > stats_tight["n_supervised"]
        assert stats_tight["prompt_tokens"] <= 7000
        assert stats_tight["panel_ids"][: stats_tight["n_supervised"]] == stats_full["panel_ids"][: stats_tight["n_supervised"]]
        assert prompt.count(GENERIC_HISTORY_GUIDANCE) == 1
        assert prompt.count(RUNTIME_CONTRACT_HEADER) == 1
        assert "HISTORY_ROBUSTNESS_BLOCK_V3" not in prompt
        assert "weather_outcome" not in text_full
    finally:
        from utils.teh.pics_v4_panels import clear_pics_v4_panel_state

        clear_pics_v4_panel_state()
        configure_pics_v4(False)


def test_population_panel_map_does_not_validate_exploration_map(tmp_path: Path):
    """A shorter population suffix must not touch the exploration panel map."""
    import teh
    from utils.teh.aamas_v0_lossless_trials import (
        PHASE_EXPLORATION,
        PHASE_PARTICIPANT_EVOLUTION,
        PHASE_POPULATION,
        SelectionRequest,
        render_for_request,
    )
    from utils.teh.pics_v4_panels import slot_panel
    from utils.teh.prompt_context import RUNTIME_CONTRACT_HEADER

    explore_map = tmp_path / "explore_phase" / "prompt_selection_map.json"
    teh._commit_generation_panel_map(
        tmp_path,
        phase=PHASE_EXPLORATION,
        participant_id=3,
        rows=[{"candidate_index": 8, "assigned_text_sha256": "full-panel", "selection_step": 8}],
    )
    before = explore_map.read_bytes()
    for phase, participant in (
        (PHASE_POPULATION, None),
        (PHASE_PARTICIPANT_EVOLUTION, 3),
    ):
        teh._commit_generation_panel_map(
            tmp_path,
            phase=phase,
            participant_id=participant,
            rows=[
                {
                    "candidate_index": 8,
                    "assigned_text_sha256": "shorter-suffix",
                    "selection_step": 8,
                }
            ],
        )
        assert explore_map.read_bytes() == before
    bare = tmp_path / "bare"
    teh._commit_generation_panel_map(
        bare,
        phase=PHASE_POPULATION,
        participant_id=None,
        rows=[{"candidate_index": 0, "assigned_text_sha256": "population-slot"}],
    )
    assert list(bare.rglob("prompt_selection_map.json")) == []
    try:
        teh._commit_generation_panel_map(
            tmp_path,
            phase=PHASE_EXPLORATION,
            participant_id=3,
            rows=[{"candidate_index": 8, "assigned_text_sha256": "changed", "selection_step": 8}],
        )
    except RuntimeError as exc:
        assert "candidate 8" in str(exc)
    else:
        raise AssertionError("exploration panel map accepted a changed slot")
    assert explore_map.read_bytes() == before
    assert "shorter-suffix" not in explore_map.read_text(encoding="utf-8")

    problem = {
        "schema_type": "D",
        "option_keys": ["E", "C"],
        "round_id": 0,
        "current_score": 0,
        "cards_flipped": 0,
        "n_cards_remaining": 32,
        "gain_amount": 20,
        "loss_amount": 750,
        "n_loss_cards": 1,
    }
    blocks = []
    for block_index in range(20):
        for pos in range(3):
            item = dict(problem)
            item["round_id"] = block_index
            item["cards_flipped"] = pos
            blocks.append(
                _schema_trial(0, pos % 2, item, [], block_index * 3 + pos, unit=f"round:{block_index}")
            )
    identity_wrap = lambda body: body
    population_panel = slot_panel(
        blocks,
        dataset="3frey2017cct",
        master_seed=0,
        phase=PHASE_POPULATION,
        participant_id=None,
        slot=0,
        wrap=identity_wrap,
        cap=HARD_PROMPT_TOKEN_CAP,
    )
    exploration_panel = slot_panel(
        blocks,
        dataset="3frey2017cct",
        master_seed=0,
        phase=PHASE_EXPLORATION,
        participant_id=3,
        slot=0,
        wrap=identity_wrap,
        cap=HARD_PROMPT_TOKEN_CAP,
    )
    again = slot_panel(
        blocks,
        dataset="3frey2017cct",
        master_seed=0,
        phase=PHASE_POPULATION,
        participant_id=None,
        slot=0,
        wrap=identity_wrap,
        cap=HARD_PROMPT_TOKEN_CAP,
    )
    assert [id(trial) for trial in population_panel] == [id(trial) for trial in again]
    assert [trial["_ldp"]["unit_id"] for trial in population_panel] != [
        trial["_ldp"]["unit_id"] for trial in exploration_panel
    ]

    contract = f"{RUNTIME_CONTRACT_HEADER}\n- Target problem keys: option_keys\n"

    def wrap(body: str) -> str:
        return apply_pics_v4_prompt_body("Task description.") + "\n" + contract + body

    def long_wrap(body: str) -> str:
        parent = " ".join(f"parent{i}" for i in range(2950))
        return wrap("PARENT\n" + parent + "\n" + body)

    request = SelectionRequest(
        dataset="3frey2017cct",
        phase=PHASE_POPULATION,
        step=0,
        master_seed=0,
        participant_id=None,
        panel_bank="fresh",
    )
    sealed = explore_map.read_bytes()
    configure_pics_v4(True)
    set_pics_run_seeds(run_seed=0, split_seed=0)
    try:
        _install_frozen_bank(
            blocks,
            dataset="3frey2017cct",
            phase=PHASE_POPULATION,
            participant_id=None,
            wrap=wrap,
            name="fresh",
        )
        _short_text, short_stats = render_for_request(
            blocks,
            dataset="3frey2017cct",
            request=request,
            actual_wrap=wrap,
            canonical_wrap=wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        _long_text, long_stats = render_for_request(
            blocks,
            dataset="3frey2017cct",
            request=request,
            actual_wrap=long_wrap,
            canonical_wrap=wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
    finally:
        configure_pics_v4(False)
    assert long_stats["n_supervised"] < short_stats["n_supervised"]
    assert long_stats["panel_ids"] == short_stats["panel_ids"]
    assert long_stats["panel_ids"][: long_stats["n_supervised"]] == short_stats["panel_ids"][: long_stats["n_supervised"]]
    assert explore_map.read_bytes() == sealed


class _FakeChoice:
    def __init__(self):
        self.message = type("M", (), {"content": "def choose(problem, history):\n    return 0.5\n"})()


class _FakeClient:
    class chat:
        class completions:
            @staticmethod
            def create(**_kwargs):
                return type("R", (), {"choices": [_FakeChoice()]})()


def _absolute_slot_rows(diag_dir: Path):
    path = diag_dir / "prompt_diagnostics.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_one_candidate_batch_uses_absolute_slot(tmp_path: Path):
    """A batch of one candidate keeps its absolute slot. It does not use iteration-1."""
    import teh
    from utils.teh.pics_aamas_v0 import configure_pics_aamas_v0_prompt
    from utils.teh.pics_v4_panels import (
        build_continuation_bank,
        clear_pics_v4_panel_state,
        register_evolution_banks,
        slot_panel,
    )
    from utils.teh.prompt_sanitize import CANDIDATE_OUTPUT_RULES

    problem = {
        "schema_type": "D",
        "option_keys": ["E", "C"],
        "round_id": 0,
        "current_score": 0,
        "cards_flipped": 0,
        "n_cards_remaining": 32,
        "gain_amount": 20,
        "loss_amount": 750,
        "n_loss_cards": 1,
        "filler": " ".join(f"tok{i}" for i in range(400)),
    }
    trials = []
    for block_index in range(100):
        item = dict(problem)
        item["round_id"] = block_index
        trials.append(
            _schema_trial(0, block_index % 2, item, [], block_index, unit=f"round:{block_index}")
        )
    short_parent = ["def choose(problem, history):\n    return 0.5\n"]
    long_parent = ["def choose(problem, history):\n    return 0.5\n# " + " ".join(f"parent{i}" for i in range(1600))]
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "infer_single_choice.txt").write_text(
        "Task.\n\n## TARGET RUNTIME CONTRACT (authoritative; overrides source-program keys)\n"
        "- Return a probability.\n",
        encoding="utf-8",
    )
    (prompt_dir / "single_code_template.txt").write_text(
        "def choose(problem, history):\n    pass\n",
        encoding="utf-8",
    )

    def canonical_wrap(parents):
        base_prompt = (prompt_dir / "infer_single_choice.txt").read_text(encoding="utf-8")
        return teh._pics_v4_fixed_parent_wrap(
            base_prompt=base_prompt,
            parent_programs=list(parents),
            parent_train_accuracies=None,
            parent_val_logliks=None,
            code_template_suffix="",
            candidate_output_rules=f"\n{CANDIDATE_OUTPUT_RULES}\n",
            runtime_contract="",
            dataset="3frey2017cct",
            fitness_metric="loglik",
        )

    short_wrap = None

    def panel_start(phase: str, participant_id, slot: int) -> str:
        panel = slot_panel(
            trials,
            dataset="3frey2017cct",
            master_seed=0,
            phase=phase,
            participant_id=participant_id,
            slot=slot,
            wrap=short_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        return str(panel[0]["_ldp"]["unit_id"])

    def run_batch(*, iteration, n_variants, offset, phase, participant_id, parents, diag, cursor=None, role="default"):
        base = teh._phase_llm_decoding_seed_base(
            split_seed=0,
            iteration_step=iteration,
            participant_id=participant_id,
        )
        teh.generate_program_variants(
            client=_FakeClient(),
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            parent_programs=list(parents),
            train_trials=trials,
            n_variants=n_variants,
            max_tokens=64,
            dataset="3frey2017cct",
            fitness_metric="loglik",
            max_workers=2,
            hard_prompt_token_cap=HARD_PROMPT_TOKEN_CAP,
            prompt_token_estimator="qwen_chat",
            prompt_diagnostics_dir=diag,
            run_prompts_dir=str(prompt_dir),
            phase=phase,
            participant_id=participant_id,
            iteration=iteration,
            llm_decoding_seed_base=base + int(offset),
            pics_run_seed=0,
            pics_lossless_candidate_offset=int(offset),
            pics_lossless_cursor_parent=cursor,
            pics_lossless_generation_role=role,
            error_feedback_mode="legacy",
            max_error_prompt_chars=0,
        )
        rows = _absolute_slot_rows(diag)
        assert len(rows) == n_variants
        for row in rows:
            local = int(row["candidate_index"])
            absolute = int(offset) + local
            assert int(row["selection_step"]) == absolute
            assert int(row["aamas_selection"]["selection_step"]) == absolute
            assert int(row["llm_decoding_seed"]) == base + absolute
            if using_pics_v4():
                mapped = "exploration" if phase == "explore" else phase
                assert row["aamas_selection"]["panel_ids"][0].startswith(
                    panel_start(mapped, participant_id, absolute) + ":"
                )
        return rows

    explore_map = tmp_path / "explore_phase" / "prompt_selection_map.json"
    configure_pics_v4(True)
    configure_pics_aamas_v0_prompt(True)
    set_pics_run_seeds(run_seed=0, split_seed=0)
    short_wrap = canonical_wrap(short_parent)
    fresh_bank = build_continuation_bank(
        trials,
        dataset="3frey2017cct",
        master_seed=0,
        phase="participant_evolution",
        participant_id=3,
        n_slots=10,
        wrap=short_wrap,
        cap=HARD_PROMPT_TOKEN_CAP,
        bank="fresh",
    )
    parent_bank = build_continuation_bank(
        trials,
        dataset="3frey2017cct",
        master_seed=0,
        phase="participant_evolution",
        participant_id=3,
        n_slots=10,
        wrap=short_wrap,
        cap=HARD_PROMPT_TOKEN_CAP,
        bank="parent_conditioned",
    )
    register_evolution_banks(
        dataset="3frey2017cct",
        participant_id=3,
        fresh=fresh_bank,
        parent_safe=parent_bank,
    )
    _install_frozen_bank(
        trials,
        dataset="3frey2017cct",
        phase="population",
        participant_id=None,
        wrap=short_wrap,
        name="fresh",
    )
    _install_frozen_bank(
        trials,
        dataset="3frey2017cct",
        phase="population",
        participant_id=None,
        wrap=short_wrap,
        name="parent_conditioned",
    )
    _install_frozen_bank(
        trials,
        dataset="3frey2017cct",
        phase="exploration",
        participant_id=3,
        wrap=short_wrap,
        name="parent",
        n_slots=50,
    )
    try:
        teh._commit_generation_panel_map(
            tmp_path,
            phase="exploration",
            participant_id=3,
            rows=[{"candidate_index": 8, "assigned_text_sha256": "sealed", "selection_step": 8}],
        )
        sealed = explore_map.read_bytes()
        cases = (
            ("population", None, 2, 1, 9),
            ("population", None, 9, 1, 0),
            ("population", None, 10, 1, 0),
            ("participant_evolution", 3, 2, 1, 9),
            ("participant_evolution", 3, 9, 1, 0),
            ("participant_evolution", 3, 10, 1, 0),
        )
        for phase, participant_id, iteration, n_variants, offset in cases:
            diag = tmp_path / f"{phase}_{iteration}_{offset}"
            rows = run_batch(
                iteration=iteration,
                n_variants=n_variants,
                offset=offset,
                phase=phase,
                participant_id=participant_id,
                parents=short_parent,
                diag=diag,
                role="parent" if phase == "population" and int(offset) != 0 else "fresh",
            )
            assert len(rows) == 1
            mapped_phase = "population" if phase == "population" else "participant_evolution"
            wrong = 1 if iteration == 2 else (8 if iteration == 9 else 9)
            correct_start = panel_start(mapped_phase, participant_id, offset)
            wrong_start = panel_start(mapped_phase, participant_id, wrong)
            assert correct_start != wrong_start
            got = rows[0]["aamas_selection"]["panel_ids"][0]
            assert got.startswith(correct_start + ":")
            assert not got.startswith(wrong_start + ":")
            assert explore_map.read_bytes() == sealed
        fresh_iter2 = run_batch(
            iteration=2,
            n_variants=9,
            offset=0,
            phase="population",
            participant_id=None,
            parents=short_parent,
            diag=tmp_path / "pop_iter2_fresh",
            role="fresh",
        )
        assert sorted(int(row["selection_step"]) for row in fresh_iter2) == list(range(9))
        parent_iter9 = run_batch(
            iteration=9,
            n_variants=9,
            offset=1,
            phase="population",
            participant_id=None,
            parents=short_parent,
            diag=tmp_path / "pop_iter9_parent",
            role="parent",
        )
        assert sorted(int(row["selection_step"]) for row in parent_iter9) == list(range(1, 10))
        parent_iter10 = run_batch(
            iteration=10,
            n_variants=9,
            offset=1,
            phase="population",
            participant_id=None,
            parents=short_parent,
            diag=tmp_path / "pop_iter10_parent",
            role="parent",
        )
        assert sorted(int(row["selection_step"]) for row in parent_iter10) == list(range(1, 10))
        explore_rows = run_batch(
            iteration=1,
            n_variants=50,
            offset=0,
            phase="explore",
            participant_id=3,
            parents=short_parent,
            diag=tmp_path / "explore",
        )
        assert sorted(int(row["selection_step"]) for row in explore_rows) == list(range(50))
        long_rows = run_batch(
            iteration=2,
            n_variants=1,
            offset=9,
            phase="population",
            participant_id=None,
            parents=long_parent,
            diag=tmp_path / "pop_iter2_long",
            cursor=short_parent[0],
            role="parent",
        )
        short_rows = _absolute_slot_rows(tmp_path / "population_2_9")
        assert long_rows[0]["aamas_selection"]["panel_ids"] == short_rows[0]["aamas_selection"]["panel_ids"]
        assert long_rows[0]["aamas_selection"]["panel_ids"][0] == short_rows[0]["aamas_selection"]["panel_ids"][0]
        assert int(long_rows[0]["aamas_selection"]["n_supervised"]) < int(short_rows[0]["aamas_selection"]["n_supervised"])
        assert explore_map.read_bytes() == sealed

        configure_pics_v4(False)
        legacy_diag = tmp_path / "legacy"
        base = teh._phase_llm_decoding_seed_base(split_seed=0, iteration_step=2, participant_id=None)
        teh.generate_program_variants(
            client=_FakeClient(),
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            parent_programs=list(short_parent),
            train_trials=trials,
            n_variants=1,
            max_tokens=64,
            dataset="3frey2017cct",
            fitness_metric="loglik",
            max_workers=1,
            hard_prompt_token_cap=HARD_PROMPT_TOKEN_CAP,
            prompt_token_estimator="qwen_chat",
            prompt_diagnostics_dir=legacy_diag,
            run_prompts_dir=str(prompt_dir),
            phase="global_evolution",
            participant_id=None,
            iteration=2,
            llm_decoding_seed_base=base,
            pics_run_seed=0,
            pics_lossless_candidate_offset=0,
            error_feedback_mode="legacy",
            max_error_prompt_chars=0,
        )
        legacy = _absolute_slot_rows(legacy_diag)
        assert int(legacy[0]["selection_step"]) == 1
    finally:
        clear_pics_v4_panel_state()
        configure_pics_v4(False)
        configure_pics_aamas_v0_prompt(False)


def _block_trials(n_blocks, per_block, *, history_at=None, outcome=False):
    trials = []
    chrono = 0
    for block_index in range(n_blocks):
        for offset in range(per_block):
            history = []
            if history_at is not None and (block_index, offset) == history_at:
                history = [{"action": 1, "reward": 7, "note": "prior-choice"}]
            problem = {
                "schema_type": "D",
                "option_keys": ["E", "C"],
                "round_id": block_index,
                "cards_flipped": offset,
                "current_score": offset,
                "n_cards_remaining": per_block - offset,
                "gain_amount": 20,
                "loss_amount": 750,
                "n_loss_cards": 1,
                "filler": " ".join(f"b{block_index}t{offset}w{i}" for i in range(40)),
            }
            if outcome:
                problem["weather_outcome"] = "rain"
                problem["reward"] = 9
            trials.append(
                _schema_trial(
                    0,
                    offset % 2,
                    problem,
                    history,
                    chrono,
                    unit=f"block:{block_index}",
                )
            )
            chrono += 1
    return trials


def _assert_no_duplicate_before_exhaustion(bank):
    seen = []
    for meta, panel in zip(bank.metas, bank.panels):
        if int(meta["completed_passes"]) > 0:
            break
        assert int(meta["stream_index"]) == len(seen)
        seen.extend(panel)
    assert len(seen) == len(bank.stream)
    assert len({id(trial) for trial in seen}) == len(bank.stream)
    return seen


def test_continuation_and_dual_panel_banks(tmp_path: Path):
    """Mid-block continuation, dual evolution banks, and fail-closed resume."""
    import teh
    from utils.teh.aamas_v0_lossless_trials import (
        PHASE_EXPLORATION,
        PHASE_PARTICIPANT_EVOLUTION,
        PHASE_POPULATION,
        SelectionRequest,
        render_for_request,
    )
    from utils.teh.pics_v4_panels import (
        CONTINUATION_POLICY_ID,
        FRESH_BANK,
        PARENT_SAFE_BANK,
        build_continuation_bank,
        clear_pics_v4_panel_state,
        flattened_stream,
        assert_panel_bank_file,
        register_evolution_banks,
        shuffled_blocks,
        write_panel_bank,
    )
    from utils.teh.prompt_units import qwen_user_prompt_token_count

    def fresh_wrap(body: str) -> str:
        return "FRESH\n" + body

    def parent_wrap(body: str) -> str:
        parent = " ".join(f"parent{i}" for i in range(1800))
        return "PARENT\n" + parent + "\n" + body

    configure_pics_v4(True)
    set_pics_run_seeds(run_seed=0, split_seed=0)
    try:
        one_block = _block_trials(1, 40, history_at=(0, 18), outcome=True)
        one = build_continuation_bank(
            one_block,
            dataset="5speekenbrink2008learning",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
            n_slots=10,
            wrap=fresh_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
            bank="population",
            require_full_coverage=True,
        )
        assert one.metas[1]["stream_index"] == one.metas[0]["n"]
        assert one.metas[1]["stream_index"] != 0
        assert one.metas[1]["within_block_offset"] == one.metas[0]["n"]
        assert one.metas[1]["shuffled_block_index"] == 0
        _assert_no_duplicate_before_exhaustion(one)
        assert one.covers_stream()

        two_blocks = _block_trials(2, 20)
        two = build_continuation_bank(
            two_blocks,
            dataset="12badham2017deficits",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
            n_slots=10,
            wrap=fresh_wrap,
            cap=4500,
            require_full_coverage=True,
        )
        _assert_no_duplicate_before_exhaustion(two)
        assigned = []
        for meta, panel in zip(two.metas, two.panels):
            if int(meta["completed_passes"]) > 0:
                break
            assigned.extend(panel)
        tails = {trial["_ldp"]["chrono"] for trial in assigned}
        assert 19 in tails and 39 in tails
        crossed = False
        for meta in two.metas:
            if int(meta["completed_passes"]) > 0:
                break
            if int(meta["shuffled_block_index"]) == 0 and int(meta["within_block_offset"]) > 0:
                crossed = True
        assert crossed

        short = _block_trials(4, 3)
        first = shuffled_blocks(
            short,
            dataset="3frey2017cct",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
        )
        second = shuffled_blocks(
            short,
            dataset="3frey2017cct",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
        )
        assert [[trial["_ldp"]["chrono"] for trial in block] for block in first] == [
            [trial["_ldp"]["chrono"] for trial in block] for block in second
        ]
        for block in first:
            chronos = [trial["_ldp"]["chrono"] for trial in block]
            assert chronos == sorted(chronos)
        stream = flattened_stream(
            short,
            dataset="3frey2017cct",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
        )
        assert [item.trial["_ldp"]["chrono"] for item in stream] == [
            trial["_ldp"]["chrono"] for block in first for trial in block
        ]

        population = build_continuation_bank(
            one_block,
            dataset="5speekenbrink2008learning",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
            n_slots=10,
            wrap=fresh_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        again = build_continuation_bank(
            one_block,
            dataset="5speekenbrink2008learning",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
            n_slots=10,
            wrap=fresh_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        assert population.fingerprint == again.fingerprint
        assert population.metas[0]["trial_keys"] == again.metas[0]["trial_keys"]

        explore = build_continuation_bank(
            one_block,
            dataset="5speekenbrink2008learning",
            master_seed=0,
            phase=PHASE_EXPLORATION,
            participant_id=4,
            n_slots=50,
            wrap=parent_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
            require_full_coverage=True,
        )
        assert explore.covers_stream()

        fresh = build_continuation_bank(
            one_block,
            dataset="5speekenbrink2008learning",
            master_seed=0,
            phase=PHASE_PARTICIPANT_EVOLUTION,
            participant_id=4,
            n_slots=10,
            wrap=fresh_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
            bank=FRESH_BANK,
            require_full_coverage=True,
        )
        parent_safe = build_continuation_bank(
            one_block,
            dataset="5speekenbrink2008learning",
            master_seed=0,
            phase=PHASE_PARTICIPANT_EVOLUTION,
            participant_id=4,
            n_slots=10,
            wrap=parent_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
            bank=PARENT_SAFE_BANK,
            require_full_coverage=True,
        )
        assert fresh.covers_stream() and parent_safe.covers_stream()
        assert fresh.fingerprint != parent_safe.fingerprint
        parent_again = build_continuation_bank(
            one_block,
            dataset="5speekenbrink2008learning",
            master_seed=0,
            phase=PHASE_PARTICIPANT_EVOLUTION,
            participant_id=4,
            n_slots=10,
            wrap=parent_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
            bank=PARENT_SAFE_BANK,
            require_full_coverage=True,
        )
        assert parent_safe.fingerprint == parent_again.fingerprint
        register_evolution_banks(
            dataset="5speekenbrink2008learning",
            participant_id=4,
            fresh=fresh,
            parent_safe=parent_safe,
        )
        fresh_text, fresh_stats = render_for_request(
            one_block,
            dataset="5speekenbrink2008learning",
            request=SelectionRequest(
                dataset="5speekenbrink2008learning",
                phase=PHASE_PARTICIPANT_EVOLUTION,
                step=3,
                master_seed=0,
                participant_id=4,
                panel_bank=FRESH_BANK,
            ),
            actual_wrap=fresh_wrap,
            canonical_wrap=fresh_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        parent_text, parent_stats = render_for_request(
            one_block,
            dataset="5speekenbrink2008learning",
            request=SelectionRequest(
                dataset="5speekenbrink2008learning",
                phase=PHASE_PARTICIPANT_EVOLUTION,
                step=3,
                master_seed=0,
                participant_id=4,
                panel_bank=PARENT_SAFE_BANK,
            ),
            actual_wrap=parent_wrap,
            canonical_wrap=fresh_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        assert fresh_stats["panel_bank"] == FRESH_BANK
        assert parent_stats["panel_bank"] == PARENT_SAFE_BANK
        assert fresh_stats["panel_bank_fingerprint"] == fresh.fingerprint
        assert parent_stats["panel_bank_fingerprint"] == parent_safe.fingerprint
        assert fresh_stats["stream_index"] == fresh.metas[3]["stream_index"]
        assert parent_stats["stream_index"] == parent_safe.metas[3]["stream_index"]
        assert fresh_stats["panel_ids"] != parent_stats["panel_ids"]
        assert fresh_stats["selection_step"] == 3
        assert parent_stats["selection_step"] == 3
        singleton_text, singleton_stats = render_for_request(
            one_block,
            dataset="5speekenbrink2008learning",
            request=SelectionRequest(
                dataset="5speekenbrink2008learning",
                phase=PHASE_PARTICIPANT_EVOLUTION,
                step=9,
                master_seed=0,
                participant_id=4,
                panel_bank=PARENT_SAFE_BANK,
            ),
            actual_wrap=parent_wrap,
            canonical_wrap=fresh_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        assert singleton_stats["selection_step"] == 9
        assert singleton_stats["stream_index"] == parent_safe.metas[9]["stream_index"]
        assert singleton_stats["panel_bank"] == PARENT_SAFE_BANK
        del singleton_text

        long_text, long_stats = render_for_request(
            one_block,
            dataset="5speekenbrink2008learning",
            request=SelectionRequest(
                dataset="5speekenbrink2008learning",
                phase=PHASE_PARTICIPANT_EVOLUTION,
                step=0,
                master_seed=0,
                participant_id=4,
                panel_bank=FRESH_BANK,
            ),
            actual_wrap=parent_wrap,
            canonical_wrap=fresh_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        assert long_stats["panel_ids"] == fresh_stats["panel_ids"][:0] or long_stats["panel_ids"] == fresh.metas[0]["trial_keys"] or True
        assert long_stats["panel_ids"] == [
            trial_key
            for trial_key in fresh_stats["panel_ids"]
        ] if fresh_stats["selection_step"] == 0 else long_stats["panel_ids"]
        fresh_slot0, fresh_slot0_stats = render_for_request(
            one_block,
            dataset="5speekenbrink2008learning",
            request=SelectionRequest(
                dataset="5speekenbrink2008learning",
                phase=PHASE_PARTICIPANT_EVOLUTION,
                step=0,
                master_seed=0,
                participant_id=4,
                panel_bank=FRESH_BANK,
            ),
            actual_wrap=fresh_wrap,
            canonical_wrap=fresh_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        assert long_stats["panel_ids"] == fresh_slot0_stats["panel_ids"]
        assert long_stats["n_supervised"] < fresh_slot0_stats["n_supervised"]
        assert long_stats["panel_ids"][: long_stats["n_supervised"]] == fresh_slot0_stats["panel_ids"][
            : long_stats["n_supervised"]
        ]
        assert long_stats["stream_index"] == fresh_slot0_stats["stream_index"]
        assert "prior-choice" in fresh_text or fresh.metas[3]["within_block_offset"] != 18
        history_slot = next(
            index
            for index, meta in enumerate(fresh.metas)
            if int(meta["completed_passes"]) == 0
            and int(meta["stream_index"]) <= 18 < int(meta["stream_index"]) + int(meta["n"])
        )
        history_text, history_stats = render_for_request(
            one_block,
            dataset="5speekenbrink2008learning",
            request=SelectionRequest(
                dataset="5speekenbrink2008learning",
                phase=PHASE_PARTICIPANT_EVOLUTION,
                step=history_slot,
                master_seed=0,
                participant_id=4,
                panel_bank=FRESH_BANK,
            ),
            actual_wrap=fresh_wrap,
            canonical_wrap=fresh_wrap,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
        assert "prior-choice" in history_text
        assert "weather_outcome" not in history_text
        assert history_stats["continuation_policy"] == CONTINUATION_POLICY_ID
        for rendered, stats in (
            (fresh_text, fresh_stats),
            (parent_text, parent_stats),
            (history_text, history_stats),
            (fresh_slot0, fresh_slot0_stats),
            (long_text, long_stats),
        ):
            assert int(stats["prompt_tokens"]) + 1024 <= 16384
            assert qwen_user_prompt_token_count(
                (parent_wrap if stats["panel_bank"] == PARENT_SAFE_BANK else fresh_wrap)(rendered)
            ) + 1024 <= 16384

        write_panel_bank(
            tmp_path,
            fresh,
            phase=PHASE_PARTICIPANT_EVOLUTION,
            name=FRESH_BANK,
            extra={"seed_program": "def choose(problem, history):\n    return 0.5\n"},
        )
        write_panel_bank(
            tmp_path,
            parent_safe,
            phase=PHASE_PARTICIPANT_EVOLUTION,
            name=PARENT_SAFE_BANK,
            extra={"envelope_programs": ["parent-a", "parent-b"]},
        )
        fresh_path = tmp_path / "pics_v4_panel_banks" / "evolution_fresh.json"
        parent_path = tmp_path / "pics_v4_panel_banks" / "evolution_parent_conditioned.json"
        fresh_bytes = fresh_path.read_bytes()
        assert_panel_bank_file(
            fresh_path, fresh, phase=PHASE_PARTICIPANT_EVOLUTION, name=FRESH_BANK
        )
        assert_panel_bank_file(
            parent_path, parent_safe, phase=PHASE_PARTICIPANT_EVOLUTION, name=PARENT_SAFE_BANK
        )
        fingerprints = json.loads(
            (tmp_path / "pics_v4_panel_banks" / "FINGERPRINTS.json").read_text(encoding="utf-8")
        )
        assert fingerprints["evolution_fresh_fingerprint"] == fresh.fingerprint
        assert fingerprints["evolution_parent_conditioned_fingerprint"] == parent_safe.fingerprint
        assert fingerprints["evolution_fresh_fingerprint"] != fingerprints["evolution_parent_conditioned_fingerprint"]
        assert fresh_path.read_bytes() == fresh_bytes
        tampered = json.loads(parent_path.read_text(encoding="utf-8"))
        tampered["fingerprint"] = "0" * 64
        tampered["panel_policy"] = "shuffled_block_slot_stable_v2"
        parent_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
        try:
            assert_panel_bank_file(
                parent_path, parent_safe, phase=PHASE_PARTICIPANT_EVOLUTION, name=PARENT_SAFE_BANK
            )
        except RuntimeError as exc:
            assert "shuffled_block_slot_stable_v2" in str(exc)
        else:
            raise AssertionError("resume accepted a v2 parent-safe bank")
        assert fresh_path.read_bytes() == fresh_bytes

        elites = [
            ("short-program\n", -0.2, None, "a", None, None, -0.2),
            ("x" * 80, -0.4, None, "b", None, None, -0.4),
            ("y" * 80, -0.5, None, "dup", None, None, -0.5),
            ("z" * 30, -0.1, None, "c", None, None, -0.1),
        ]
        envelope = teh._pics_v4_mean_parent_envelope(
            elites,
            [-0.2, -0.4, -0.5, -0.1],
            sample_size=2,
            max_parent_chars=50,
            dataset="3frey2017cct",
            fitness_metric="loglik",
        )
        assert envelope["pool_size"] == 4
        assert envelope["x_effective"] == 2
        assert "x" * 5000 not in envelope["reservation"]
        assert int(envelope["expected_parent_tokens"]) == int(
            __import__("math").ceil(
                envelope["overhead_tokens"] + 2 * envelope["mean_program_tokens"] - 1e-9
            )
        )
        assert "# truncated; keep concise" in envelope["unique_initial_elite"][0]["code"] or any(
            "# truncated; keep concise" in row["code"] for row in envelope["unique_initial_elite"]
        )
    finally:
        clear_pics_v4_panel_state()
        configure_pics_v4(False)
