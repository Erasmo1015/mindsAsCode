"""One saved AAMAS prompt per phase, without changing the call payload."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from utils.teh.prompt_examples import (
    EVOLUTION,
    EXPLORATION,
    POPULATION_TARGET,
    POPULATION_TRANSFER,
    configure_aamas_prompt_examples,
    maybe_save_aamas_prompt_example,
    reset_aamas_prompt_examples,
)

USER = "block_id = unit:0:round:1\nposition = 0\nproblem = {}\n"


def _save(tmp: Path, **overrides):
    payload = dict(
        api_messages=[{"role": "user", "content": USER}],
        user_text=USER,
        qwen_input_tokens=11,
        parent_programs=["def choose(problem, history):\n    return 0.5\n"],
        parent_ids=["global_baseline"],
        generation_phase="global_evolution",
        dataset="3frey2017cct",
        participant_id=None,
        iteration=2,
        selection_step=1,
        candidate_index=0,
    )
    payload.update(overrides)
    messages = payload["api_messages"]
    before = json.dumps(messages)
    saved = maybe_save_aamas_prompt_example(**payload)
    assert json.dumps(messages) == before
    assert messages[0]["content"] == USER
    return saved


def test_each_llm_phase_saves_one_complete_prompt(tmp_path: Path):
    configure_aamas_prompt_examples(
        tmp_path,
        track_mode="target_only",
        population_iters=10,
        evolution_iters=10,
        explore_candidates=50,
        representative_participant_id=0,
    )
    try:
        assert _save(tmp_path) is True
        assert _save(tmp_path) is False
        assert _save(tmp_path, iteration=1) is False
        assert _save(
            tmp_path,
            generation_phase="explore",
            participant_id=0,
            iteration=None,
            selection_step=25,
            candidate_index=25,
            parent_ids=["global_iteration_3_candidate_0"],
        ) is True
        assert _save(
            tmp_path,
            generation_phase="explore",
            participant_id=0,
            iteration=None,
            selection_step=0,
            candidate_index=0,
        ) is False
        assert _save(
            tmp_path,
            generation_phase="evolution",
            participant_id=0,
            iteration=2,
            selection_step=1,
            candidate_index=0,
        ) is True
        assert _save(
            tmp_path,
            generation_phase="evolution",
            participant_id=4,
            iteration=2,
            selection_step=1,
            candidate_index=0,
        ) is False
        names = sorted(path.name for path in (tmp_path / "prompt_examples").iterdir())
        assert names == sorted(
            [
                f"{POPULATION_TARGET}.json",
                f"{EXPLORATION}.json",
                f"{EVOLUTION}.json",
            ]
        )
        for name in names:
            body = json.loads((tmp_path / "prompt_examples" / name).read_text(encoding="utf-8"))
            assert body["api_messages"][0]["role"] == "user"
            assert body["api_messages"][0]["content"] == USER
            assert body["qwen_chat_messages"][0]["role"] == "system"
            assert body["qwen_chat_messages"][1]["content"] == USER
            assert body["parent_programs"]
            assert body["packed_trial_ids"] == [{"block_id": "unit:0:round:1", "position": 0}]
            assert body["qwen_input_tokens"] == 11
            assert body["arm"] == name[: -len(".json")]
    finally:
        reset_aamas_prompt_examples()


def test_transfer_population_is_separate_and_gate_is_not_saved(tmp_path: Path):
    configure_aamas_prompt_examples(
        tmp_path,
        track_mode="transfer_based_only",
        population_iters=10,
        evolution_iters=1,
        explore_candidates=1,
        representative_participant_id=7,
    )
    try:
        assert _save(tmp_path, generation_phase="global_evolution", iteration=2) is True
        assert _save(tmp_path, generation_phase="global_evolution", iteration=2) is False
        reset_aamas_prompt_examples()
        configure_aamas_prompt_examples(
            tmp_path,
            track_mode="official_gate",
            population_iters=10,
            evolution_iters=1,
            explore_candidates=1,
            representative_participant_id=7,
        )
        assert _save(tmp_path, generation_phase="global_evolution", iteration=2) is False
        assert not (tmp_path / "prompt_examples" / "official_gate.json").exists()
        saved = sorted(path.name for path in (tmp_path / "prompt_examples").iterdir())
        assert saved == [f"{POPULATION_TRANSFER}.json"]
    finally:
        reset_aamas_prompt_examples()


def test_parallel_workers_keep_one_file(tmp_path: Path):
    configure_aamas_prompt_examples(
        tmp_path,
        track_mode="target_only",
        population_iters=1,
        evolution_iters=1,
        explore_candidates=1,
        representative_participant_id=0,
    )
    try:
        results = []

        def _once():
            results.append(_save(tmp_path, iteration=1, selection_step=0))

        threads = [threading.Thread(target=_once) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results.count(True) == 1
        files = list((tmp_path / "prompt_examples").iterdir())
        assert len(files) == 1
        body = json.loads(files[0].read_text(encoding="utf-8"))
        assert body["qwen_input_tokens"] == 11
        assert body["api_messages"][0]["content"] == USER
    finally:
        reset_aamas_prompt_examples()
