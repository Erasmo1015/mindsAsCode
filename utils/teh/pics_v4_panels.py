"""Slot-stable shuffled-block panels for PICS v4.

Whole blocks are shuffled with the method seed. Trials stay chronological
inside each block. Panels are cut sequentially from that stream: the next
panel starts at the first snapshot the previous panel did not take. A panel
wraps to the start of the same stream only after every retained snapshot has
been assigned once. After construction, a longer parent may keep only a prefix.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.teh.limited_data_registry import INDEPENDENT_TRIAL, limited_data_spec
from utils.teh.prompt_snapshots import current_or_future_leak_paths, format_snapshot_examples
from utils.teh.prompt_units import prompt_participant_id, qwen_user_prompt_token_count, trial_unit_id

PANEL_POLICY_ID = "shuffled_block_slot_stable_v3"
# v1 selected iteration-1 for a one-candidate batch.
# v2 restarted a long block at its first trial for every slot.
REFUSED_PANEL_POLICY_ID = "shuffled_block_slot_stable_v1"
REFUSED_PANEL_POLICY_IDS = frozenset(
    {
        "shuffled_block_slot_stable_v1",
        "shuffled_block_slot_stable_v2",
    }
)
SLOT_ASSIGNMENT_POLICY_ID = "first_n"
PACKING_IMPLEMENTATION_ID = "shuffled_block_slot_stable_v3"
CONTINUATION_POLICY_ID = "within_block_carry_forward_v1"
PANEL_BANK_POLICY_ID = "conditioning_aware_panel_banks_v2"
PARENT_ENVELOPE_POLICY_ID = "initial_unique_elite_mean_x_v1"
REFUSED_PANEL_BANK_POLICY_IDS = frozenset(
    {
        "conditioning_aware_panel_banks_v1",
        "fresh_parent_dual_bank_v1",
    }
)
REFUSED_PARENT_ENVELOPE_POLICY_IDS = frozenset(
    {
        "initial_unique_elite_longest_x_v1",
    }
)
POPULATION_SLOTS = 10
EXPLORATION_SLOTS = 50
EVOLUTION_SLOTS = 10
FRESH_BANK = "fresh"
PARENT_CONDITIONED_BANK = "parent_conditioned"
PARENT_SAFE_BANK = PARENT_CONDITIONED_BANK
EXPLORATION_PARENT_BANK = "parent"
FINGERPRINT_KEYS = {
    ("population", FRESH_BANK): "population_fresh_fingerprint",
    ("population", PARENT_CONDITIONED_BANK): "population_parent_conditioned_fingerprint",
    ("exploration", EXPLORATION_PARENT_BANK): "exploration_parent_fingerprint",
    ("participant_evolution", FRESH_BANK): "evolution_fresh_fingerprint",
    ("participant_evolution", PARENT_CONDITIONED_BANK): "evolution_parent_conditioned_fingerprint",
}
BANK_FILENAMES = {
    ("population", FRESH_BANK): "population_fresh.json",
    ("population", PARENT_CONDITIONED_BANK): "population_parent_conditioned.json",
    ("exploration", EXPLORATION_PARENT_BANK): "exploration_parent.json",
    ("participant_evolution", FRESH_BANK): "evolution_fresh.json",
    ("participant_evolution", PARENT_CONDITIONED_BANK): "evolution_parent_conditioned.json",
}

Wrap = Callable[[str], str]


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


def _assignment_key(
    trial: Dict[str, Any],
    *,
    dataset: str,
    block_index: int,
    within_block_offset: int,
) -> str:
    unit = trial_unit_id(trial, dataset=dataset, fallback=within_block_offset)
    chrono = (trial.get("_ldp") or {}).get("chrono")
    return f"{int(block_index)}:{int(within_block_offset)}:{unit}:{chrono}"


@dataclass
class StreamItem:
    trial: Dict[str, Any]
    block_index: int
    within_block_offset: int
    key: str


@dataclass
class ContinuationBank:
    panels: List[List[Dict[str, Any]]]
    metas: List[Dict[str, Any]]
    stream: List[StreamItem]
    fingerprint: str
    covered_before_wrap: int = 0
    expected_parent_tokens: int = 0

    def covers_stream(self) -> bool:
        return int(self.covered_before_wrap) == len(self.stream)


def flattened_stream(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    master_seed: int,
    phase: str,
    participant_id: Optional[int],
) -> List[StreamItem]:
    blocks = shuffled_blocks(
        trials,
        dataset=dataset,
        master_seed=master_seed,
        phase=phase,
        participant_id=participant_id,
    )
    items: List[StreamItem] = []
    for block_index, block in enumerate(blocks):
        for offset, trial in enumerate(block):
            items.append(
                StreamItem(
                    trial=trial,
                    block_index=int(block_index),
                    within_block_offset=int(offset),
                    key=_assignment_key(
                        trial,
                        dataset=dataset,
                        block_index=block_index,
                        within_block_offset=offset,
                    ),
                )
            )
    return items


def bank_fingerprint(slot_keys: Sequence[Sequence[str]], *, bank: str) -> str:
    payload = {
        "panel_policy": PANEL_POLICY_ID,
        "continuation_policy": CONTINUATION_POLICY_ID,
        "bank": str(bank),
        "slots": [list(keys) for keys in slot_keys],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def longest_prefix(
    panel: Sequence[Dict[str, Any]],
    wrap: Wrap,
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


def build_continuation_bank(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    master_seed: int,
    phase: str,
    participant_id: Optional[int],
    n_slots: int,
    wrap: Wrap,
    cap: int,
    bank: str = "",
    require_full_coverage: bool = False,
) -> ContinuationBank:
    """Cut panels in order from one shuffled stream. Do not wrap inside a panel."""
    from utils.teh.aamas_v0_lossless_trials import TrialWindowDoesNotFitError

    n_slots = max(1, int(n_slots))
    stream = flattened_stream(
        trials,
        dataset=dataset,
        master_seed=master_seed,
        phase=phase,
        participant_id=participant_id,
    )
    n_stream = len(stream)
    cursor = 0
    passes = 0
    covered = 0
    panels: List[List[Dict[str, Any]]] = []
    metas: List[Dict[str, Any]] = []
    slot_keys: List[List[str]] = []
    for slot in range(n_slots):
        if n_stream and cursor >= n_stream:
            cursor = 0
            passes += 1
        if n_stream:
            item = stream[cursor]
            block_index = item.block_index
            within_offset = item.within_block_offset
            remaining = [entry.trial for entry in stream[cursor:]]
        else:
            block_index = 0
            within_offset = 0
            remaining = []
        visible = longest_prefix(remaining, wrap, cap)
        if remaining and not visible:
            raise TrialWindowDoesNotFitError(
                f"PICS v4 slot {slot} cannot fit one complete snapshot under cap {cap}."
            )
        if passes == 0:
            covered = cursor + len(visible)
        panel = list(visible)
        panels.append(panel)
        keys = [entry.key for entry in stream[cursor : cursor + len(panel)]]
        slot_keys.append(keys)
        metas.append(
            {
                "slot": int(slot),
                "stream_index": int(cursor),
                "shuffled_block_index": int(block_index),
                "within_block_offset": int(within_offset),
                "completed_passes": int(passes),
                "n": len(panel),
                "trial_keys": keys,
            }
        )
        cursor += len(panel)
    fingerprint = bank_fingerprint(slot_keys, bank=bank or str(phase))
    built = ContinuationBank(
        panels=panels,
        metas=metas,
        stream=stream,
        fingerprint=fingerprint,
        covered_before_wrap=int(covered),
    )
    if require_full_coverage and not built.covers_stream():
        raise RuntimeError(
            f"PICS v4 {phase} panels cover {built.covered_before_wrap} of "
            f"{n_stream} retained snapshots before wrapping "
            f"(bank={bank or phase})."
        )
    return built


_BANK_CACHE: Dict[Tuple[Any, ...], ContinuationBank] = {}
_BANKS: Dict[Tuple[str, str, Optional[int]], Dict[str, ContinuationBank]] = {}


def clear_pics_v4_panel_state() -> None:
    _BANK_CACHE.clear()
    _BANKS.clear()


def _wrap_fingerprint(wrap: Wrap) -> str:
    return hashlib.sha256(wrap("").encode("utf-8")).hexdigest()


def cached_continuation_bank(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    master_seed: int,
    phase: str,
    participant_id: Optional[int],
    n_slots: int,
    wrap: Wrap,
    cap: int,
    require_full_coverage: bool = False,
) -> ContinuationBank:
    stream = flattened_stream(
        trials,
        dataset=dataset,
        master_seed=master_seed,
        phase=phase,
        participant_id=participant_id,
    )
    stream_fp = hashlib.sha256(
        "\n".join(item.key for item in stream).encode("utf-8")
    ).hexdigest()
    key = (
        str(dataset),
        str(phase),
        None if participant_id is None else int(participant_id),
        int(master_seed),
        int(n_slots),
        int(cap),
        stream_fp,
        _wrap_fingerprint(wrap),
        bool(require_full_coverage),
    )
    cached = _BANK_CACHE.get(key)
    if cached is not None:
        return cached
    built = build_continuation_bank(
        trials,
        dataset=dataset,
        master_seed=master_seed,
        phase=phase,
        participant_id=participant_id,
        n_slots=n_slots,
        wrap=wrap,
        cap=cap,
        require_full_coverage=require_full_coverage,
    )
    _BANK_CACHE[key] = built
    return built


def slot_panel(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    master_seed: int,
    phase: str,
    participant_id: Optional[int],
    slot: int,
    wrap: Wrap,
    cap: int,
    n_slots: Optional[int] = None,
) -> List[Dict[str, Any]]:
    count = slots_for_phase(phase) if n_slots is None else int(n_slots)
    bank = cached_continuation_bank(
        trials,
        dataset=dataset,
        master_seed=master_seed,
        phase=phase,
        participant_id=participant_id,
        n_slots=count,
        wrap=wrap,
        cap=cap,
    )
    index = int(slot)
    if index < 0 or index >= len(bank.panels):
        raise RuntimeError(
            f"PICS v4 slot {index} is outside 0..{len(bank.panels) - 1} for phase {phase}"
        )
    return list(bank.panels[index])


def panel_registry_key(
    dataset: str,
    phase: str,
    participant_id: Optional[int],
) -> Tuple[str, str, Optional[int]]:
    return (
        str(dataset),
        str(phase),
        None if participant_id is None else int(participant_id),
    )


def register_panel_bank(
    *,
    dataset: str,
    phase: str,
    participant_id: Optional[int],
    name: str,
    bank: ContinuationBank,
) -> None:
    if not bank.fingerprint:
        raise RuntimeError(f"PICS v4 panel bank {name} requires a fingerprint.")
    key = panel_registry_key(dataset, phase, participant_id)
    slot = _BANKS.setdefault(key, {})
    slot[str(name)] = bank


def register_evolution_banks(
    *,
    dataset: str,
    participant_id: Optional[int],
    fresh: ContinuationBank,
    parent_safe: ContinuationBank,
) -> None:
    register_panel_bank(
        dataset=dataset,
        phase="participant_evolution",
        participant_id=participant_id,
        name=FRESH_BANK,
        bank=fresh,
    )
    register_panel_bank(
        dataset=dataset,
        phase="participant_evolution",
        participant_id=participant_id,
        name=PARENT_CONDITIONED_BANK,
        bank=parent_safe,
    )


def registered_panel_bank(
    *,
    dataset: str,
    phase: str,
    participant_id: Optional[int],
    name: str,
) -> ContinuationBank:
    key = panel_registry_key(dataset, phase, participant_id)
    found = _BANKS.get(key) or {}
    bank = found.get(str(name))
    if bank is None:
        raise RuntimeError(
            f"PICS v4 {phase} panel bank {name!r} is not installed for "
            f"{dataset} participant {participant_id}."
        )
    return bank


def evolution_bank_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "pics_v4_panel_banks"


def bank_filename(phase: str, name: str) -> str:
    try:
        return BANK_FILENAMES[(str(phase), str(name))]
    except KeyError as exc:
        raise RuntimeError(f"PICS v4 has no panel-bank file for {phase} {name}.") from exc


def fingerprint_field(phase: str, name: str) -> str:
    try:
        return FINGERPRINT_KEYS[(str(phase), str(name))]
    except KeyError as exc:
        raise RuntimeError(f"PICS v4 has no fingerprint field for {phase} {name}.") from exc


def _bank_payload(bank: ContinuationBank, *, phase: str, name: str) -> Dict[str, Any]:
    return {
        "panel_policy": PANEL_POLICY_ID,
        "packing_implementation": PACKING_IMPLEMENTATION_ID,
        "continuation_policy": CONTINUATION_POLICY_ID,
        "panel_bank_policy": PANEL_BANK_POLICY_ID,
        "parent_envelope_policy": PARENT_ENVELOPE_POLICY_ID,
        "phase": str(phase),
        "bank": str(name),
        "fingerprint": bank.fingerprint,
        "fingerprint_field": fingerprint_field(phase, name),
        "n_slots": len(bank.panels),
        "covered_before_wrap": int(bank.covered_before_wrap),
        "n_stream": len(bank.stream),
        "slots": [list(meta["trial_keys"]) for meta in bank.metas],
        "cursors": [
            {
                "slot": meta["slot"],
                "stream_index": meta["stream_index"],
                "shuffled_block_index": meta["shuffled_block_index"],
                "within_block_offset": meta["within_block_offset"],
                "completed_passes": meta["completed_passes"],
            }
            for meta in bank.metas
        ],
    }


def write_panel_bank_file(
    path: Path,
    bank: ContinuationBank,
    *,
    phase: str,
    name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _bank_payload(bank, phase=phase, name=name)
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def record_panel_fingerprint(
    output_dir: Path,
    *,
    phase: str,
    name: str,
    fingerprint: str,
) -> Path:
    root = evolution_bank_dir(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "FINGERPRINTS.json"
    payload: Dict[str, Any] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    field_name = fingerprint_field(phase, name)
    previous = str(payload.get(field_name) or "")
    if previous and previous != str(fingerprint):
        raise RuntimeError(
            f"Refusing to overwrite {path}: {field_name} {previous} "
            f"does not match {fingerprint}."
        )
    payload["panel_bank_policy"] = PANEL_BANK_POLICY_ID
    payload["continuation_policy"] = CONTINUATION_POLICY_ID
    payload["parent_envelope_policy"] = PARENT_ENVELOPE_POLICY_ID
    payload[field_name] = str(fingerprint)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def assert_panel_bank_file(path: Path, bank: ContinuationBank, *, phase: str, name: str) -> None:
    """Refuse a saved bank whose policy or fingerprint does not match the rebuild."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"PICS v4 panel bank file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    found_panel = str(payload.get("panel_policy") or "")
    found_pack = str(payload.get("packing_implementation") or "")
    if found_panel in REFUSED_PANEL_POLICY_IDS or found_pack in REFUSED_PANEL_POLICY_IDS:
        raise RuntimeError(
            f"Refusing to resume {path}: panel policy {found_panel or found_pack} "
            f"does not implement within-block continuation. "
            f"Required panel policy is {PANEL_POLICY_ID}."
        )
    found_bank_policy = str(payload.get("panel_bank_policy") or "")
    found_envelope = str(payload.get("parent_envelope_policy") or "")
    if found_bank_policy in REFUSED_PANEL_BANK_POLICY_IDS:
        raise RuntimeError(
            f"Refusing to resume {path}: panel bank policy {found_bank_policy} "
            f"is not {PANEL_BANK_POLICY_ID}."
        )
    if found_envelope in REFUSED_PARENT_ENVELOPE_POLICY_IDS:
        raise RuntimeError(
            f"Refusing to resume {path}: parent envelope {found_envelope} "
            "uses the longest programs. "
            f"Required parent envelope is {PARENT_ENVELOPE_POLICY_ID}."
        )
    if found_panel != PANEL_POLICY_ID or found_pack != PACKING_IMPLEMENTATION_ID:
        raise RuntimeError(
            f"Refusing to resume {path}: panel policy {found_panel!r} / {found_pack!r} "
            f"is not {PANEL_POLICY_ID}."
        )
    for field_name, expected in (
        ("continuation_policy", CONTINUATION_POLICY_ID),
        ("panel_bank_policy", PANEL_BANK_POLICY_ID),
        ("parent_envelope_policy", PARENT_ENVELOPE_POLICY_ID),
        ("phase", str(phase)),
        ("bank", str(name)),
    ):
        found = str(payload.get(field_name) or "")
        if found != expected:
            raise RuntimeError(
                f"Refusing to resume {path}: {field_name} is {found!r}, not {expected}."
            )
    found_fp = str(payload.get("fingerprint") or "")
    if found_fp != bank.fingerprint:
        raise RuntimeError(
            f"Refusing to resume {path}: {name} panel-bank fingerprint {found_fp} "
            f"does not match {bank.fingerprint}."
        )
    expected_slots = [list(meta["trial_keys"]) for meta in bank.metas]
    if payload.get("slots") != expected_slots:
        raise RuntimeError(
            f"Refusing to resume {path}: {name} panel-bank slot map does not match "
            "the rebuilt continuation."
        )
    summary_path = path.parent / "FINGERPRINTS.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Refusing to resume {path.parent}: missing panel-bank fingerprints.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    field_name = fingerprint_field(phase, name)
    if str(summary.get(field_name) or "") != bank.fingerprint:
        raise RuntimeError(
            f"Refusing to resume {summary_path}: {field_name} does not match the rebuilt bank."
        )
    if str(summary.get("panel_bank_policy") or "") != PANEL_BANK_POLICY_ID:
        raise RuntimeError(
            f"Refusing to resume {summary_path}: panel_bank_policy is "
            f"{summary.get('panel_bank_policy')!r}."
        )


def write_panel_bank(
    output_dir: Path,
    bank: ContinuationBank,
    *,
    phase: str,
    name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = evolution_bank_dir(output_dir) / bank_filename(phase, name)
    write_panel_bank_file(path, bank, phase=phase, name=name, extra=extra)
    record_panel_fingerprint(
        output_dir,
        phase=phase,
        name=name,
        fingerprint=bank.fingerprint,
    )
    return path


def render_slot_panel(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    request,
    actual_wrap: Wrap,
    canonical_wrap: Wrap,
    cap: int,
    policy_id: str,
) -> Tuple[str, Dict[str, Any]]:
    from utils.teh.aamas_v0_lossless_trials import TrialWindowDoesNotFitError

    del canonical_wrap
    phase = str(request.phase)
    if phase == "evolution":
        phase = "participant_evolution"
    slot = int(request.step)
    bank_name = str(getattr(request, "panel_bank", "") or "")
    if phase == "exploration":
        bank_name = EXPLORATION_PARENT_BANK
    if bank_name not in {FRESH_BANK, PARENT_CONDITIONED_BANK, EXPLORATION_PARENT_BANK}:
        raise RuntimeError(
            "PICS v4 selects a frozen panel bank from fresh versus "
            f"parent-conditioned status, not {bank_name!r}."
        )
    built = registered_panel_bank(
        dataset=dataset,
        phase=phase,
        participant_id=request.participant_id,
        name=bank_name,
    )
    if slot < 0 or slot >= len(built.panels):
        raise RuntimeError(
            f"PICS v4 slot {slot} is outside the {phase} {bank_name} panel bank."
        )
    panel = list(built.panels[slot])
    meta = dict(built.metas[slot])
    visible = longest_prefix(panel, actual_wrap, cap)
    if panel and not visible:
        raise TrialWindowDoesNotFitError(
            f"PICS v4 slot panel does not fit a single trial under cap {cap}."
        )
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
        "continuation_policy": CONTINUATION_POLICY_ID,
        "panel_bank_policy": PANEL_BANK_POLICY_ID,
        "parent_envelope_policy": PARENT_ENVELOPE_POLICY_ID,
        "slot_assignment": SLOT_ASSIGNMENT_POLICY_ID,
        "selection_step": int(request.step),
        "phase": phase,
        "panel_bank": bank_name,
        "panel_bank_fingerprint": built.fingerprint,
        "stream_index": meta.get("stream_index"),
        "shuffled_block_index": meta.get("shuffled_block_index"),
        "within_block_offset": meta.get("within_block_offset"),
        "completed_passes": meta.get("completed_passes"),
        "prompt_tokens": wrapped_tokens,
        "expected_parent_envelope_tokens": int(built.expected_parent_tokens),
        "actual_parent_tokens": int(getattr(request, "actual_parent_tokens", 0) or 0),
        "nominal_panel_trials": len(panel),
        "visible_trial_count": len(visible),
        "suffix_trials_hidden": len(panel) - len(visible),
        "input_token_count": wrapped_tokens,
        "digest": PANEL_POLICY_ID,
    }
    return text, stats


