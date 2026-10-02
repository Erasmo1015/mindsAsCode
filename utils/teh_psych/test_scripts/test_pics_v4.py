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
        text, stats = render_for_request(
            [trial],
            dataset="3frey2017cct",
            request=SelectionRequest(
                dataset="3frey2017cct",
                phase=PHASE_POPULATION,
                step=0,
                master_seed=0,
                participant_id=None,
            ),
            actual_wrap=lambda body: body,
            canonical_wrap=lambda body: body,
            cap=HARD_PROMPT_TOKEN_CAP,
        )
    finally:
        configure_pics_v4(False)
    assert "observed_action_label=0" in text
    assert "option_keys" in text
    assert "history_before" not in text
    assert stats["policy"] == "structured_snapshot_v2"
    assert stats["panel_policy"] == "shuffled_block_slot_stable_v1"
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
    assert payload["packing_implementation"] == "shuffled_block_slot_stable_v1"
    assert payload["panel_policy"] == "shuffled_block_slot_stable_v1"
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
    from utils.teh.pics_v4_panels import group_blocks, slot_panel
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
                text, stats = render_for_request(
                    [trial],
                    dataset=dataset,
                    request=SelectionRequest(
                        dataset=dataset,
                        phase=phase,
                        step=0,
                        master_seed=0,
                        participant_id=None if phase == PHASE_POPULATION else 0,
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
        text, _stats = render_for_request(
            [weather_trial],
            dataset="5speekenbrink2008learning",
            request=SelectionRequest(
                dataset="5speekenbrink2008learning",
                phase=PHASE_POPULATION,
                step=0,
                master_seed=0,
                participant_id=None,
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
        )
        panel1 = slot_panel(
            blocks,
            dataset="3frey2017cct",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
            slot=1,
        )
        again = slot_panel(
            blocks,
            dataset="3frey2017cct",
            master_seed=0,
            phase=PHASE_POPULATION,
            participant_id=None,
            slot=0,
        )
        assert [id(trial) for trial in panel0] == [id(trial) for trial in again]
        assert set(map(id, panel0)).isdisjoint(set(map(id, panel1)))
        first_units = {trial["_ldp"]["unit_id"] for trial in panel0}
        assert first_units != {"round:0"}

        def short_wrap(body: str) -> str:
            return wrap(body)

        def long_wrap(body: str) -> str:
            parent = " ".join(f"parent{i}" for i in range(2950))
            return wrap("PARENT\n" + parent + "\n" + body)

        short_text, short_stats = render_for_request(
            blocks,
            dataset="3frey2017cct",
            request=SelectionRequest(
                dataset="3frey2017cct",
                phase=PHASE_POPULATION,
                step=0,
                master_seed=0,
                participant_id=None,
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
        request = SelectionRequest(
            dataset="3frey2017cct",
            phase=PHASE_POPULATION,
            step=0,
            master_seed=0,
            participant_id=None,
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
    population_panel = slot_panel(
        blocks,
        dataset="3frey2017cct",
        master_seed=0,
        phase=PHASE_POPULATION,
        participant_id=None,
        slot=0,
    )
    exploration_panel = slot_panel(
        blocks,
        dataset="3frey2017cct",
        master_seed=0,
        phase=PHASE_EXPLORATION,
        participant_id=3,
        slot=0,
    )
    again = slot_panel(
        blocks,
        dataset="3frey2017cct",
        master_seed=0,
        phase=PHASE_POPULATION,
        participant_id=None,
        slot=0,
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
    )
    sealed = explore_map.read_bytes()
    configure_pics_v4(True)
    set_pics_run_seeds(run_seed=0, split_seed=0)
    try:
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
