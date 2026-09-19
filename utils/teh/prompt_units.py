"""Structure-aware T-PICS v2 prompt-example selection and token packing.

Selection and presentation are separate. Fitness still uses the complete
retained train+val union; this module only chooses display examples.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.teh.limited_data_protocol import _unit_id_from_trial
from utils.teh.limited_data_registry import (
    CONTINUOUS_SESSION,
    INDEPENDENT_TRIAL,
    RESETTING_UNIT,
    limited_data_spec,
    normalize_limited_dataset_alias,
)
from utils.teh.prompt_snapshots import (
    format_snapshot_examples,
    prompt_participant_id,
)

PROMPT_DISPLAY_CEILING = 60
QWEN_INPUT_CEILING = 14_000
OUTPUT_RESERVE = 1_024
VLLM_CONTEXT = 16_384
QWEN_TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
# Qwen2.5 chat template inserts this when the first message is not system.
# vLLM OpenAI serving does the same for teh.py's user-only candidate calls.
QWEN_DEFAULT_SYSTEM = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)

_QWEN_TOKENIZER = None
_QWEN_TOKENIZER_FAILED: Optional[str] = None
_QWEN_TOKENIZER_LOCK = threading.Lock()


def infer_prompt_dataset_alias(trials: Sequence[Dict[str, Any]]) -> str:
    for trial in trials:
        problem = trial.get("problem") or {}
        alias = problem.get("dataset_alias")
        if alias:
            return normalize_limited_dataset_alias(str(alias))
    raise ValueError("cannot infer dataset_alias from prompt trials")


def trial_unit_id(trial: Dict[str, Any], *, dataset: str, fallback: int = 0) -> str:
    spec = limited_data_spec(dataset)
    ldp = trial.get("_ldp") or {}
    if ldp.get("unit_id"):
        return str(ldp["unit_id"])
    return _unit_id_from_trial(spec, trial, fallback)


def group_units_chronological(
    trials: Sequence[Dict[str, Any]], *, dataset: str
) -> List[List[Dict[str, Any]]]:
    spec = limited_data_spec(dataset)
    if spec.category == INDEPENDENT_TRIAL:
        return [[t] for t in trials]
    units: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_id: Optional[str] = None
    for i, trial in enumerate(trials):
        uid = trial_unit_id(trial, dataset=dataset, fallback=i)
        if spec.category == CONTINUOUS_SESSION:
            uid = f"session:{prompt_participant_id(trial)}"
        if not current:
            current = [trial]
            current_id = uid
            continue
        if uid != current_id:
            units.append(current)
            current = [trial]
            current_id = uid
        else:
            current.append(trial)
    if current:
        units.append(current)
    return units


def _participant_order(trials: Sequence[Dict[str, Any]]) -> List[Optional[int]]:
    seen: List[Optional[int]] = []
    for trial in trials:
        pid = prompt_participant_id(trial)
        if pid not in seen:
            seen.append(pid)
    return seen


def select_structure_aware_prompt_examples(
    trials: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    max_trials: int,
    subsample_seed: int,
    pooled: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deterministic participant-stratified unit/window selection.

    Returns trials in *selection* order (balanced round-robin). Callers render
    a kept prefix grouped by participant.
    """
    pool = list(trials)
    diag: Dict[str, Any] = {
        "dataset": dataset,
        "max_trials": int(max_trials),
        "n_available": len(pool),
        "pooled": bool(pooled),
        "subsample_seed": int(subsample_seed),
        "mode": "structure_aware_v2",
    }
    if max_trials <= 0 or not pool:
        return [], diag
    spec = limited_data_spec(dataset)
    n_people = len(_participant_order(pool))
    person_mode = (not pooled) or n_people <= 1
    diag["n_participants"] = n_people
    diag["person_mode"] = person_mode

    if person_mode and len(pool) <= max_trials:
        diag["selection"] = "full_observed_chronology"
        return list(pool), diag

    by_pid: Dict[Optional[int], List[List[Dict[str, Any]]]] = defaultdict(list)
    order = _participant_order(pool)
    for pid in order:
        person_trials = [t for t in pool if prompt_participant_id(t) == pid]
        if spec.category == CONTINUOUS_SESSION:
            by_pid[pid] = [person_trials] if person_trials else []
        else:
            by_pid[pid] = group_units_chronological(person_trials, dataset=dataset)

    rng = np.random.default_rng(int(subsample_seed))
    if spec.category == INDEPENDENT_TRIAL:
        for pid in order:
            units = by_pid[pid]
            idx = rng.permutation(len(units))
            by_pid[pid] = [units[int(i)] for i in idx]

    queues = {pid: deque(by_pid[pid]) for pid in order}
    selected_units: List[List[Dict[str, Any]]] = []
    n_kept = 0
    while n_kept < max_trials and any(queues[pid] for pid in order):
        progress = False
        for pid in order:
            if n_kept >= max_trials:
                break
            if not queues[pid]:
                continue
            unit = list(queues[pid].popleft())
            remaining = max_trials - n_kept
            if len(unit) <= remaining:
                selected_units.append(unit)
                n_kept += len(unit)
            elif spec.prefix_valid or spec.category == CONTINUOUS_SESSION:
                selected_units.append(unit[:remaining])
                n_kept += remaining
            progress = True
        if not progress:
            break

    selected: List[Dict[str, Any]] = []
    for unit in selected_units:
        selected.extend(unit)
    diag["n_selected"] = len(selected)
    diag["n_units_selected"] = len(selected_units)
    diag["category"] = spec.category
    return selected, diag


def render_examples_grouped(
    trials: Sequence[Dict[str, Any]],
) -> str:
    """Render kept examples grouped by participant, chronological within person."""
    by_pid: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
    order: List[Optional[int]] = []
    for trial in trials:
        pid = prompt_participant_id(trial)
        if pid not in by_pid:
            order.append(pid)
        by_pid[pid].append(trial)
    grouped: List[Dict[str, Any]] = []
    for pid in order:
        grouped.extend(by_pid[pid])
    return format_snapshot_examples(grouped)


def largest_balanced_prefix_that_fits(
    selected: Sequence[Dict[str, Any]],
    *,
    required_prompt: str,
    token_count: Callable[[str], int],
    token_ceiling: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep the longest prefix of the balanced selection that still fits.

    Does not sort-then-tail-drop (that would drop later participants).
    """
    selected_list = list(selected)
    diag: Dict[str, Any] = {
        "n_requested": len(selected_list),
        "token_ceiling": int(token_ceiling),
    }
    if not selected_list:
        diag.update({"n_retained": 0, "tokens": token_count(required_prompt)})
        return [], diag

    def fits(k: int) -> Tuple[bool, int]:
        text = required_prompt + "\n\n" + render_examples_grouped(selected_list[:k])
        n_tok = token_count(text)
        return n_tok <= token_ceiling, n_tok

    lo, hi = 0, len(selected_list)
    best_k = 0
    best_tokens = token_count(required_prompt)
    # Shrink from the end of the balanced list (complete examples / prefixes).
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, n_tok = fits(mid)
        if ok:
            best_k = mid
            best_tokens = n_tok
            lo = mid + 1
        else:
            hi = mid - 1
    kept = selected_list[:best_k]
    pids = []
    for t in kept:
        pid = prompt_participant_id(t)
        if pid not in pids:
            pids.append(pid)
    diag.update(
        {
            "n_retained": len(kept),
            "n_participants_retained": len(pids),
            "tokens": best_tokens,
            "participants_retained": pids,
        }
    )
    return kept, diag


def _ensure_qwen_tokenizer():
    """Load the Qwen tokenizer once. Fail clearly; never silently fall back to char/4."""
    global _QWEN_TOKENIZER, _QWEN_TOKENIZER_FAILED
    with _QWEN_TOKENIZER_LOCK:
        if _QWEN_TOKENIZER_FAILED:
            raise RuntimeError(_QWEN_TOKENIZER_FAILED)
        if _QWEN_TOKENIZER is None:
            try:
                from transformers import AutoTokenizer  # type: ignore
            except Exception as exc:
                _QWEN_TOKENIZER_FAILED = (
                    f"Qwen tokenizer unavailable ({type(exc).__name__}: {exc}). "
                    "Refusing to silently change T-PICS v2 packing. Install/cache "
                    f"{QWEN_TOKENIZER_NAME} or pass a tokenizer in tests."
                )
                raise RuntimeError(_QWEN_TOKENIZER_FAILED) from exc
            try:
                _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
                    QWEN_TOKENIZER_NAME, trust_remote_code=True
                )
            except Exception as exc:
                _QWEN_TOKENIZER_FAILED = (
                    f"Failed to load {QWEN_TOKENIZER_NAME}: {type(exc).__name__}: {exc}"
                )
                raise RuntimeError(_QWEN_TOKENIZER_FAILED) from exc
        return _QWEN_TOKENIZER


def qwen_chat_token_count(system: str, user: str) -> int:
    """vLLM-matching Qwen chat-templated input tokens (system + user).

    Must apply the chat template with a system turn. Encoding the user string
    alone, or templating user-only, undershoots vLLM's ``tokens in the messages``
    (Qwen injects the default system even when the OpenAI client sent user-only).
    Fail clearly if the tokenizer is missing; never fall back to char/4.
    """
    tokenizer = _ensure_qwen_tokenizer()
    messages = [
        {"role": "system", "content": system or QWEN_DEFAULT_SYSTEM},
        {"role": "user", "content": user or ""},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return int(len(tokenizer.encode(text, add_special_tokens=False)))


def qwen_user_prompt_token_count(user: str) -> int:
    """Candidate-generation prompt count: chat-templated default system + user."""
    return qwen_chat_token_count(QWEN_DEFAULT_SYSTEM, user)
