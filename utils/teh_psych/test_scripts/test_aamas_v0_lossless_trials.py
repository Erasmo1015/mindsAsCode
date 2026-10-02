"""CPU tests for the AAMAS-v0 lossless trial prompt."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from utils.teh.aamas_v0_lossless_trials import (
    CAP,
    PACKING_IMPLEMENTATION_ID,
    PHASE_EXPLORATION,
    PHASE_PARTICIPANT_EVOLUTION,
    PHASE_POPULATION,
    TRIAL_PROMPT_POLICY_ID,
    SelectionRequest,
    assert_legacy_output_not_resumed,
    group_blocks,
    map_generation_phase,
    reconstruct_histories,
    render_for_request,
    render_window,
    pics_run_seed,
    selection_digest,
    set_pics_run_seeds,
    write_trial_policy_marker,
)
from utils.teh.pics_aamas_v0 import (
    configure_pics_aamas_v0_prompt,
    using_pics_aamas_v0_prompt,
)

DATASETS = [
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "mixed_gambles",
    "7hilbig2014generalized",
    "bergert_nosofsky_2007",
    "11enkavi2019recentprobes",
    "5speekenbrink2008learning",
    "12badham2017deficits",
    "10frey2017risk",
    "guan_2020_stopping",
    "steyvers_2009_bandit",
    "13schulz2020finding",
    "14kool2016when",
]


def _trial(pid, action, history, *, unit, session, reward=None, day=None):
    problem = {
        "participant_id": pid,
        "left_option": 1,
        "reward": reward,
        "feedback": reward,
    }
    if day is not None:
        problem["presented_day"] = day
        problem["dataset_alias"] = "14kool2016when"
        problem["schema_type"] = "kool_twostep"
        problem["stage"] = 1
    return {
        "problem": problem,
        "action": action,
        "history": history,
        "_prompt_participant_id": pid,
        "_ldp": {
            "unit_id": unit,
            "chrono": session,
            "session_index": session,
            "origin_split": "train",
            "origin_index": session,
        },
    }


def _chain(pid, n, unit):
    rows = []
    history = []
    for index in range(n):
        rows.append(
            _trial(pid, index % 2, list(history), unit=unit, session=index, reward=1000 + index)
        )
        history = list(history) + [{"action": index % 2, "reward": None if index % 3 == 0 else index}]
    return rows


def _wrap(body, extra=""):
    return f"Task description.\nReminder.\nRuntime contract.\n{extra}{body}"


def _render(trials, dataset, *, step=0, phase=PHASE_POPULATION, seed=0, participant=None, extra="", cap=CAP):
    request = SelectionRequest(
        dataset=dataset,
        phase=phase,
        step=step,
        master_seed=seed,
        participant_id=participant,
    )
    text, stats = render_for_request(
        trials,
        dataset=dataset,
        request=request,
        actual_wrap=lambda body: _wrap(body, extra),
        canonical_wrap=lambda body: _wrap(body, ""),
        cap=cap,
    )
    return _wrap(text, extra), text, stats


def test_all_datasets_share_phase_mapping_and_serializer_name():
    for dataset in DATASETS:
        assert map_generation_phase("global_evolution") == PHASE_POPULATION
        assert map_generation_phase("explore") == PHASE_EXPLORATION
        assert map_generation_phase("evolution") == PHASE_PARTICIPANT_EVOLUTION
        text, _body, stats = _render(_chain(0, 2, "unit:0"), dataset, cap=4000)
        assert '"problem":' in text
        assert '"observed_action":' in text
        assert "history_prefix" not in text
        assert "reset_before" not in text
        assert stats["policy"] == TRIAL_PROMPT_POLICY_ID
        assert stats["history_mismatches"] == 0


def test_roundtrip_and_no_current_outcome_in_problem():
    trials = _chain(3, 4, "problem:1")
    _prompt, body, _stats = _render(trials, "steyvers_2009_bandit", cap=8000)
    blocks = group_blocks(trials, "steyvers_2009_bandit")
    window = blocks[0]
    rendered = render_window(window)
    assert "history_prefix" not in rendered
    problem_line = next(line for line in rendered.splitlines() if line.startswith("problem = "))
    assert "1000" not in problem_line
    rebuilt = reconstruct_histories(window)
    for trial, history in zip(window, rebuilt):
        assert history == list(trial["history"])
    assert '"observed_action":' in body
    assert "history_before" in body or '"outcome":' in body
    for line in body.splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        assert "1000" not in json.dumps(record.get("problem"))


def test_independent_history_empty_and_continuous_session():
    indep = [
        _trial(1, 0, [], unit=f"trial:{i}", session=i)
        for i in range(3)
    ]
    text, body, _stats = _render(indep, "mixed_gambles", cap=8000)
    assert "history resets" in text
    assert '"observed_action":' in body
    assert "reset_before" not in text
    kool = []
    history = []
    for day, action in ((1, 0), (1, 1), (2, 0)):
        kool.append(_trial(4, action, list(history), unit=f"day:{day}", session=len(kool), day=day))
        history = list(history) + [{"action": action, "reward": 1}]
    kool.append(_trial(5, 0, [], unit="day:1", session=0, day=1))
    text, _body, _stats = _render(kool, "14kool2016when", cap=8000)
    assert text.count("history resets") == 2
    assert "participant 4" in text and "participant 5" in text
    assert "reset_before" not in text


def test_later_window_reconstructs_and_kool_late_position_is_reachable():
    rows = _chain(0, 6, "session")
    for trial in rows:
        trial["problem"]["dataset_alias"] = "14kool2016when"
        trial["problem"]["presented_day"] = 1 + trial["_ldp"]["session_index"] // 3
    early, _body, early_stats = _render(rows, "14kool2016when", step=0, cap=140)
    late, _body, late_stats = _render(rows, "14kool2016when", step=1, cap=140)
    assert early_stats["min_position"] == 0
    assert late_stats["min_position"] > early_stats["max_position"]
    reached = []
    for step in range(8):
        _text, _unused, stats = _render(rows, "14kool2016when", step=step, cap=140)
        reached.extend(stats["positions"])
    assert max(reached) == 5
    line = next(line for line in late.splitlines() if "history_before" in line)
    assert line.count("{") > 1


def test_seed_reproducibility_balance_and_shared_population_schedule():
    trials = []
    for pid in range(6):
        trials.extend(_chain(pid, 2, f"unit:{pid}:a"))
        trials.extend(_chain(pid, 2, f"unit:{pid}:b"))
    _a, _b, first = _render(trials, "2plonsky2018when", seed=7, cap=2500)
    _c, _d, again = _render(trials, "2plonsky2018when", seed=7, cap=2500)
    _e, _f, other = _render(trials, "2plonsky2018when", seed=8, cap=2500)
    assert first["assigned_positions"] == again["assigned_positions"]
    assert first["assigned_block_ids"] == again["assigned_block_ids"]
    assert first["assigned_participant_ids"] != other["assigned_participant_ids"]
    assert len(set(first["participant_ids"])) == len(first["participant_ids"])
    control, _b, control_stats = _render(trials, "2plonsky2018when", seed=7, cap=2500, extra="CONTROL")
    transfer, _d, transfer_stats = _render(
        trials, "2plonsky2018when", seed=7, cap=2500, extra="TRANSFER_SOURCE"
    )
    assert control_stats["assigned_block_ids"] == transfer_stats["assigned_block_ids"]
    assert control_stats["positions"][0] == transfer_stats["positions"][0]
    digest_control = selection_digest(
        master_seed=7, dataset="2plonsky2018when", phase=PHASE_POPULATION, participant_id=None
    )
    digest_transfer = selection_digest(
        master_seed=7, dataset="2plonsky2018when", phase=PHASE_POPULATION, participant_id=None
    )
    assert digest_control == digest_transfer
    assert "CONTROL" in control and "TRANSFER_SOURCE" in transfer


def test_population_candidates_share_evidence_and_new_pass_waits():
    trials = _chain(0, 3, "unit:0")
    _p0, _b0, step0 = _render(trials, "12badham2017deficits", step=0, cap=100)
    _p1, _b1, step1 = _render(trials, "12badham2017deficits", step=1, cap=100)
    _p2, _b2, step2 = _render(trials, "12badham2017deficits", step=2, cap=100)
    assert step0["positions"] == [0]
    assert step1["positions"] == [1]
    assert step2["positions"] == [2]
    _p3, _b3, step3 = _render(trials, "12badham2017deficits", step=3, cap=100)
    assert step3["positions"] == [0]
    same_a, _x, stats_a = _render(trials, "12badham2017deficits", step=1, cap=100)
    same_b, _y, stats_b = _render(trials, "12badham2017deficits", step=1, cap=100, extra="PARENT")
    assert stats_a["assigned_positions"] == stats_b["assigned_positions"]
    assert stats_b["positions"][0] == stats_a["positions"][0]


def test_exploration_steps_and_resume_map(tmp_path: Path):
    import teh

    trials = []
    for pid in (0,):
        trials.extend(_chain(pid, 1, "unit:a"))
        trials.extend(_chain(pid, 1, "unit:b"))
        trials.extend(_chain(pid, 1, "unit:c"))
    texts = []
    for step in (0, 1, 49):
        text, _body, stats = _render(
            trials,
            "steyvers_2009_bandit",
            step=step,
            phase=PHASE_EXPLORATION,
            participant=0,
            cap=8000,
        )
        texts.append(text)
        assert stats["selection_step"] == step
        assert '"observed_action":' in text
        assert stats["full_retained_fits"] is True
        assert stats["n_supervised"] == 3
    assert texts[0] == texts[1] == texts[2]
    rows = [
        {"candidate_index": step, "assigned_text_sha256": f"hash-{step}"}
        for step in range(50)
    ]
    teh._commit_aamas_explore_prompt_map(tmp_path, participant_id=0, rows=rows)
    teh._commit_aamas_explore_prompt_map(tmp_path, participant_id=0, rows=rows)
    with pytest.raises(RuntimeError):
        teh._commit_aamas_explore_prompt_map(
            tmp_path,
            participant_id=0,
            rows=[{"candidate_index": 1, "assigned_text_sha256": "changed"}],
        )


def test_evolution_advances_and_parent_text_survives():
    trials = _chain(9, 4, "unit:9")
    parent = "def choose(problem, history):\n    marker = 'PARENT_SURVIVES'\n    return 0.5\n"
    first, _b, first_stats = _render(
        trials,
        "13schulz2020finding",
        step=0,
        phase=PHASE_PARTICIPANT_EVOLUTION,
        participant=9,
        extra=parent,
        cap=5000,
    )
    second, _c, second_stats = _render(
        trials,
        "13schulz2020finding",
        step=1,
        phase=PHASE_PARTICIPANT_EVOLUTION,
        participant=9,
        extra=parent,
        cap=5000,
    )
    assert "PARENT_SURVIVES" in first
    assert first_stats["positions"] != second_stats["positions"] or first_stats["n_supervised"] == len(trials)
    assert first_stats["prompt_tokens"] <= 5000


def test_no_capped_history_and_token_cap():
    history = [{"action": index % 2, "reward": None if index % 2 == 0 else index} for index in range(10)]
    trials = [
        _trial(0, 0, [], unit="unit:long", session=0),
        _trial(0, 1, history, unit="unit:long", session=1),
        _trial(0, 0, history + [{"action": 1, "reward": 1}], unit="unit:long", session=2),
    ]
    text, _body, stats = _render(trials, "10frey2017risk", cap=14000)
    records = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    long_histories = [
        record["outcome"] for record in records if len(record.get("outcome") or []) > 8
    ] + [
        record["history_before"]
        for record in records
        if len(record.get("history_before") or []) > 8
    ]
    assert long_histories
    assert stats["prompt_tokens"] <= 14000
    assert "history_prefix" not in text


def test_pics_v3_serializer_unchanged_and_legacy_resume_fails(tmp_path: Path):
    import teh

    configure_pics_aamas_v0_prompt(False)
    assert using_pics_aamas_v0_prompt() is False
    assert teh.estimate_tokens("abcd") == 1
    text = teh._serialize_trials_for_prompt(
        [{"problem": {"left": 1}, "action": 0, "history": []}],
        dataset="mixed_gambles",
        compact=False,
    )
    assert "history = " not in text
    configure_pics_aamas_v0_prompt(True)
    seen = {}

    def _worker():
        seen["on"] = using_pics_aamas_v0_prompt()

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    assert seen["on"] is True
    with pytest.raises(RuntimeError):
        teh._serialize_trials_for_prompt(
            [{"problem": {}, "action": 0, "history": []}],
            dataset="mixed_gambles",
            compact=False,
        )
    configure_pics_aamas_v0_prompt(False)
    legacy = tmp_path / "job_old"
    legacy.mkdir()
    (legacy / "POPULATION_PROVENANCE.json").write_text(
        json.dumps({"prompt_policy": "pics_aamas_v0"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        assert_legacy_output_not_resumed(legacy)
    fresh = tmp_path / "job_new"
    write_trial_policy_marker(fresh)
    marker = json.loads((fresh / "TRIAL_PROMPT_POLICY.json").read_text(encoding="utf-8"))
    assert marker["trial_prompt_policy"] == TRIAL_PROMPT_POLICY_ID
    assert marker["packing_implementation"] == PACKING_IMPLEMENTATION_ID
    pre_fix = tmp_path / "job_pre_fix"
    pre_fix.mkdir()
    (pre_fix / "TRIAL_PROMPT_POLICY.json").write_text(
        json.dumps({"trial_prompt_policy": TRIAL_PROMPT_POLICY_ID, "pics_run_seed": 0, "split_seed": 0}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="packing implementation"):
        write_trial_policy_marker(pre_fix)
    assert "packing_implementation" not in json.loads(
        (pre_fix / "TRIAL_PROMPT_POLICY.json").read_text(encoding="utf-8")
    )


def test_pics_run_seed_keeps_sa40_and_reruns_method(tmp_path: Path):
    import inspect

    import teh
    from utils.teh.limited_data_protocol import (
        load_participant_limited_splits,
        structure_aware_subset_fingerprint,
    )

    source = Path(__file__).resolve().parents[3].joinpath("teh.py").read_text(encoding="utf-8")
    assert "--pics_run_seed" in source
    assert "--aamas_run_seed" not in source
    assert "--aamas_trial_selection_seed" not in source

    trial = {
        "action": 1,
        "problem": {"x": 1},
        "_ldp": {"unit_id": "u0", "chrono": 0, "session_index": 0},
    }
    fp_args = dict(
        dataset="2plonsky2018when",
        participant_id=1,
        split_seed=0,
        budget=40,
        protocol="structure_aware_v3",
        train_trials=[trial],
        val_trials=[],
        test_trials=[],
    )
    assert "pics_run_seed" not in inspect.signature(structure_aware_subset_fingerprint).parameters
    assert "pics_run_seed" not in inspect.signature(load_participant_limited_splits).parameters
    fp_fixed = structure_aware_subset_fingerprint(**fp_args)
    fp_other_split = structure_aware_subset_fingerprint(**{**fp_args, "split_seed": 1})
    assert fp_fixed != fp_other_split

    def _llm(candidate: int = 0) -> int:
        base = teh._aamas_or_legacy_seed(
            teh._phase_llm_decoding_seed_base(split_seed=0, iteration_step=1),
            "llm_request_base",
            dataset="2plonsky2018when",
            phase="population",
            iteration=1,
            role="iteration",
        )
        return int(base) + int(candidate)

    configure_pics_aamas_v0_prompt(False)
    pics_llm = teh._phase_llm_decoding_seed_base(split_seed=4, iteration_step=2, participant_id=3, batch_offset=5)
    assert pics_llm == 4 + 80_000 + 2 * 1_000_003 + 3 * 17_179 + 5
    assert teh._aamas_or_legacy_seed(12345, "parent_sampling", dataset="d", phase="population", iteration=1) == 12345

    configure_pics_aamas_v0_prompt(True)
    try:
        set_pics_run_seeds(run_seed=0, split_seed=0)
        schedule_0 = selection_digest(
            master_seed=int(pics_run_seed()),
            dataset="2plonsky2018when",
            phase=PHASE_POPULATION,
            participant_id=None,
        )
        parent_0 = teh._aamas_or_legacy_seed(
            99, "parent_sampling", dataset="2plonsky2018when", phase="population", iteration=1
        )
        llm_0 = _llm()
        assert structure_aware_subset_fingerprint(**fp_args) == fp_fixed

        set_pics_run_seeds(run_seed=0, split_seed=7)
        assert structure_aware_subset_fingerprint(**fp_args) == fp_fixed
        assert teh._aamas_or_legacy_seed(
            99, "parent_sampling", dataset="2plonsky2018when", phase="population", iteration=1
        ) == parent_0
        assert _llm() == llm_0

        set_pics_run_seeds(run_seed=1, split_seed=0)
        schedule_1 = selection_digest(
            master_seed=int(pics_run_seed()),
            dataset="2plonsky2018when",
            phase=PHASE_POPULATION,
            participant_id=None,
        )
        assert schedule_1 != schedule_0
        assert teh._aamas_or_legacy_seed(
            99, "parent_sampling", dataset="2plonsky2018when", phase="population", iteration=1
        ) == parent_0
        assert _llm() == llm_0
        assert structure_aware_subset_fingerprint(**fp_args) == fp_fixed
        transfer = selection_digest(
            master_seed=1,
            dataset="2plonsky2018when",
            phase=PHASE_POPULATION,
            participant_id=None,
        )
        assert transfer == schedule_1

        set_pics_run_seeds(run_seed=1, split_seed=0)
        assert _llm() == 80_000 + 1_000_003
        assert _llm(9) == 80_000 + 1_000_003 + 9

        marked = tmp_path / "job"
        write_trial_policy_marker(marked)
        payload = json.loads((marked / "TRIAL_PROMPT_POLICY.json").read_text(encoding="utf-8"))
        assert payload["pics_run_seed"] == 1
        assert payload["split_seed"] == 0
        set_pics_run_seeds(run_seed=2, split_seed=0)
        with pytest.raises(RuntimeError, match="pics_run_seed"):
            write_trial_policy_marker(marked)
        set_pics_run_seeds(run_seed=1, split_seed=3)
        with pytest.raises(RuntimeError, match="split_seed"):
            write_trial_policy_marker(marked)
    finally:
        configure_pics_aamas_v0_prompt(False)
        set_pics_run_seeds(run_seed=0, split_seed=0)

    assert teh._phase_llm_decoding_seed_base(
        split_seed=4, iteration_step=2, participant_id=3, batch_offset=5
    ) == pics_llm


def test_final_prompt_keeps_one_runtime_contract(tmp_path: Path):
    from utils.teh.prompt_context import (
        RUNTIME_CONTRACT_HEADER,
        append_runtime_contract_if_present,
        finalize_aamas_runtime_contract,
    )

    contract = f"{RUNTIME_CONTRACT_HEADER}\n- Target problem keys: x\n"
    infer = f"Task description.\n\n{contract}"
    packed = infer + "\nTrial 1\nproblem = {}\nhistory = []\n"
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "runtime_contract.txt").write_text(contract, encoding="utf-8")
    doubled = append_runtime_contract_if_present(packed, prompts)
    assert doubled.count(RUNTIME_CONTRACT_HEADER) == 2
    final = finalize_aamas_runtime_contract(packed, prompts)
    assert final == packed
    assert final.count(RUNTIME_CONTRACT_HEADER) == 1
    missing = finalize_aamas_runtime_contract("Task only.\n", prompts)
    assert missing.count(RUNTIME_CONTRACT_HEADER) == 1
    with pytest.raises(RuntimeError, match="2 TARGET RUNTIME CONTRACT"):
        finalize_aamas_runtime_contract(doubled, prompts)
