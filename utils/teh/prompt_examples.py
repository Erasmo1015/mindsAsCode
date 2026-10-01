"""Save one final AAMAS prompt per run and LLM phase.

The write happens only for a deterministic representative call. An exclusive
create keeps parallel workers from saving a second copy. The official
observed-data gate has no LLM prompt and is not given a file.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from utils.teh.aamas_v0_lossless_trials import map_generation_phase
from utils.teh.prompt_units import QWEN_DEFAULT_SYSTEM

_LOCK = threading.Lock()
_ID_RE = re.compile(r"^block_id = (.+)\nposition = (\d+)$", re.M)

POPULATION_TARGET = "target_only_population"
POPULATION_TRANSFER = "transfer_based_population"
EXPLORATION = "participant_exploration"
EVOLUTION = "participant_evolution"


@dataclass
class _ExampleConfig:
    run_dir: Path
    track_mode: str
    population_iters: int
    evolution_iters: int
    explore_candidates: int
    representative_participant_id: Optional[int]


_CONFIG: Optional[_ExampleConfig] = None


def configure_aamas_prompt_examples(
    run_dir: Path,
    *,
    track_mode: str,
    population_iters: int,
    evolution_iters: int,
    explore_candidates: int,
    representative_participant_id: Optional[int],
) -> None:
    """Enable saving under this run directory. Safe to call once before workers start."""
    global _CONFIG
    with _LOCK:
        _CONFIG = _ExampleConfig(
            run_dir=Path(run_dir),
            track_mode=str(track_mode),
            population_iters=int(population_iters),
            evolution_iters=int(evolution_iters),
            explore_candidates=int(explore_candidates),
            representative_participant_id=(
                None
                if representative_participant_id is None
                else int(representative_participant_id)
            ),
        )


def reset_aamas_prompt_examples() -> None:
    global _CONFIG
    with _LOCK:
        _CONFIG = None


def example_arm(track_mode: str, generation_phase: str) -> Optional[str]:
    """File arm for a real LLM phase. The observed-data gate returns None."""
    mapped = map_generation_phase(generation_phase)
    track = str(track_mode or "")
    if mapped == "population":
        if track == "target_only":
            return POPULATION_TARGET
        if track == "transfer_based_only":
            return POPULATION_TRANSFER
        return None
    if mapped == "exploration":
        return EXPLORATION
    if mapped == "participant_evolution":
        return EVOLUTION
    return None


def is_representative_call(
    config: _ExampleConfig,
    *,
    generation_phase: str,
    iteration: Optional[int],
    selection_step: Optional[int],
    candidate_index: int,
    participant_id: Optional[int],
) -> bool:
    mapped = map_generation_phase(generation_phase)
    if example_arm(config.track_mode, generation_phase) is None:
        return False
    if mapped == "population":
        preferred = 2 if config.population_iters >= 2 else 1
        return (
            participant_id is None
            and iteration == preferred
            and int(candidate_index) == 0
        )
    if mapped == "participant_evolution":
        preferred = 2 if config.evolution_iters >= 2 else 1
        return (
            _same_participant(config, participant_id)
            and iteration == preferred
            and int(candidate_index) == 0
        )
    if mapped == "exploration":
        middle = config.explore_candidates // 2 if config.explore_candidates >= 2 else 0
        return _same_participant(config, participant_id) and int(selection_step or -1) == middle
    return False


def _same_participant(config: _ExampleConfig, participant_id: Optional[int]) -> bool:
    if config.representative_participant_id is None or participant_id is None:
        return False
    return int(participant_id) == int(config.representative_participant_id)


def maybe_save_aamas_prompt_example(
    *,
    api_messages: Sequence[Dict[str, str]],
    user_text: str,
    qwen_input_tokens: int,
    parent_programs: Sequence[str],
    parent_ids: Sequence[str],
    generation_phase: str,
    dataset: str,
    participant_id: Optional[int],
    iteration: Optional[int],
    selection_step: Optional[int],
    candidate_index: int,
) -> bool:
    """Write the representative prompt once. Does not modify the arguments."""
    with _LOCK:
        config = _CONFIG
    if config is None:
        return False
    if not is_representative_call(
        config,
        generation_phase=generation_phase,
        iteration=iteration,
        selection_step=selection_step,
        candidate_index=candidate_index,
        participant_id=participant_id,
    ):
        return False
    arm = example_arm(config.track_mode, generation_phase)
    if arm is None:
        return False
    messages = [dict(message) for message in api_messages]
    payload = {
        "arm": arm,
        "track_mode": config.track_mode,
        "generation_phase": str(generation_phase),
        "dataset": str(dataset),
        "participant_id": participant_id,
        "iteration": iteration,
        "selection_step": selection_step,
        "candidate_index": int(candidate_index),
        "parent_ids": list(parent_ids),
        "parent_programs": list(parent_programs),
        "api_messages": messages,
        "qwen_chat_messages": [
            {"role": "system", "content": QWEN_DEFAULT_SYSTEM},
            *messages,
        ],
        "system_message_source": "qwen_chat_template_default_when_the_request_has_no_system_message",
        "packed_trial_ids": [
            {"block_id": block, "position": int(pos)} for block, pos in _ID_RE.findall(user_text)
        ],
        "qwen_input_tokens": int(qwen_input_tokens),
    }
    path = config.run_dir / "prompt_examples" / f"{arm}.json"
    return _create_exclusive_json(path, payload)


def _create_exclusive_json(path: Path, payload: Dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return True
