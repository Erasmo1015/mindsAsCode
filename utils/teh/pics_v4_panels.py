"""Slot-stable shuffled-block panels for PICS v4.

Blocks are shuffled with the method seed. Trials stay in their original order
inside each block. A slot keeps one fixed panel across iterations. After the
parent text is inserted, only a prefix of that panel is kept.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.teh.limited_data_registry import INDEPENDENT_TRIAL, limited_data_spec
from utils.teh.prompt_snapshots import current_or_future_leak_paths, format_snapshot_examples
from utils.teh.prompt_units import prompt_participant_id, qwen_user_prompt_token_count, trial_unit_id

PANEL_POLICY_ID = "shuffled_block_slot_stable_v1"
SLOT_ASSIGNMENT_POLICY_ID = "first_n"
PACKING_IMPLEMENTATION_ID = "shuffled_block_slot_stable_v1"
POPULATION_SLOTS = 10
EXPLORATION_SLOTS = 50
EVOLUTION_SLOTS = 10


def slots_for_phase(phase: str) -> int:
    name = str(phase or "")
    if name == "exploration":
        return EXPLORATION_SLOTS
    if name in {"evolution", "participant_evolution"}:
        return EVOLUTION_SLOTS
    return POPULATION_SLOTS


def group_blocks(trials: Sequence[Dict[str, Any]], dataset: str) -> List[List[Dict[str, Any]]]:
    """Whole blocks, chronological inside each block. Independent trials stay singletons."""
    spec = limited_data_spec(dataset)
    blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_key: Optional[Tuple[Any, ...]] = None
    for index, trial in enumerate(trials):
        if spec.category == INDEPENDENT_TRIAL:
            key: Tuple[Any, ...] = ("trial", index)
        else:
            key = (
                prompt_participant_id(trial),
                trial_unit_id(trial, dataset=dataset, fallback=index),
            )
        if current and key != current_key:
            blocks.append(current)
            current = []
        current.append(trial)
        current_key = key
    if current:
        blocks.append(current)
    return blocks


def shuffle_seed(
    *,
    master_seed: int,
    dataset: str,
    phase: str,
    participant_id: Optional[int],
) -> int:
    payload = {
        "namespace": "shuffled_block_panels",
        "pics_run_seed": int(master_seed),
        "dataset": str(dataset),
        "phase": str(phase),
        "participant": None if participant_id is None else int(participant_id),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def shuffled_blocks(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    master_seed: int,
    phase: str,
    participant_id: Optional[int],
) -> List[List[Dict[str, Any]]]:
    blocks = group_blocks(trials, dataset)
    if len(blocks) <= 1:
        return blocks
    rng = np.random.default_rng(
        shuffle_seed(
            master_seed=master_seed,
            dataset=dataset,
            phase=phase,
            participant_id=participant_id,
        )
    )
    order = [int(index) for index in rng.permutation(len(blocks))]
    return [blocks[index] for index in order]


def assign_slot_panels(
    blocks: Sequence[Sequence[Dict[str, Any]]],
    n_slots: int,
) -> List[List[Dict[str, Any]]]:
    """Disjoint contiguous slices when there are enough blocks. Otherwise rotate the start."""
    n_slots = max(1, int(n_slots))
    if not blocks:
        return [[] for _ in range(n_slots)]
    if len(blocks) >= n_slots:
        sizes = [len(blocks) // n_slots] * n_slots
        for index in range(len(blocks) % n_slots):
            sizes[index] += 1
        panels: List[List[Dict[str, Any]]] = []
        cursor = 0
        for size in sizes:
            chunk = list(blocks[cursor : cursor + size])
            cursor += size
            panels.append([trial for block in chunk for trial in block])
        return panels
    panels = []
    for slot in range(n_slots):
        start = slot % len(blocks)
        rotated = list(blocks[start:]) + list(blocks[:start])
        panels.append([trial for block in rotated for trial in block])
    return panels


def slot_panel(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    master_seed: int,
    phase: str,
    participant_id: Optional[int],
    slot: int,
) -> List[Dict[str, Any]]:
    blocks = shuffled_blocks(
        trials,
        dataset=dataset,
        master_seed=master_seed,
        phase=phase,
        participant_id=participant_id,
    )
    panels = assign_slot_panels(blocks, slots_for_phase(phase))
    index = int(slot)
    if index < 0 or index >= len(panels):
        raise RuntimeError(
            f"PICS v4 slot {index} is outside 0..{len(panels) - 1} for phase {phase}"
        )
    return list(panels[index])


def longest_prefix(
    panel: Sequence[Dict[str, Any]],
    wrap,
    cap: int,
) -> List[Dict[str, Any]]:
    """Longest prefix whose wrapped snapshot stays within the Qwen input cap."""
    if not panel:
        return []

    def fits(count: int) -> bool:
        if count <= 0:
            return True
        body = format_snapshot_examples(list(panel[:count]))
        return int(qwen_user_prompt_token_count(wrap(body))) <= int(cap)

    if not fits(1):
        return []
    lo = 1
    hi = len(panel)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return list(panel[:lo])


def render_slot_panel(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    request,
    actual_wrap,
    cap: int,
    policy_id: str,
) -> Tuple[str, Dict[str, Any]]:
    panel = slot_panel(
        trials,
        dataset=dataset,
        master_seed=int(request.master_seed),
        phase=str(request.phase),
        participant_id=request.participant_id,
        slot=int(request.step),
    )
    visible = longest_prefix(panel, actual_wrap, cap)
    for trial in visible:
        leaks = current_or_future_leak_paths(trial)
        if leaks:
            raise RuntimeError(
                "PICS v4 snapshot would put a current outcome in problem: "
                + ", ".join(leaks)
            )
    text = format_snapshot_examples(visible) if visible else ""
    wrapped_tokens = int(qwen_user_prompt_token_count(actual_wrap(text)))
    rows = []
    for pos, trial in enumerate(visible):
        rows.append(
            {
                "trial_position": pos,
                "block_id": trial_unit_id(trial, dataset=dataset, fallback=pos),
                "participant_id": prompt_participant_id(trial),
            }
        )
    panel_ids = [
        trial_unit_id(trial, dataset=dataset, fallback=pos) + f":{pos}"
        for pos, trial in enumerate(panel)
    ]
    stats: Dict[str, Any] = {
        "n_supervised": len(visible),
        "n_participants": len({row["participant_id"] for row in rows}),
        "n_blocks": len({row["block_id"] for row in rows}),
        "participant_ids": [row["participant_id"] for row in rows],
        "block_ids": [row["block_id"] for row in rows],
        "positions": list(range(len(visible))),
        "trials": rows,
        "panel_n": len(panel),
        "panel_ids": panel_ids,
        "visible_is_prefix": True,
        "full_retained_fits": len(visible) == len(panel),
        "assigned_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "canonical_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "policy": policy_id,
        "packing_implementation": PACKING_IMPLEMENTATION_ID,
        "panel_policy": PANEL_POLICY_ID,
        "slot_assignment": SLOT_ASSIGNMENT_POLICY_ID,
        "selection_step": int(request.step),
        "phase": str(request.phase),
        "prompt_tokens": wrapped_tokens,
        "digest": PANEL_POLICY_ID,
    }
    return text, stats
