#!/usr/bin/env python3
# OpenEvolve intentionally keeps vanilla/original prompt behavior independent from TEH prompt-generation.
"""
Usage: 

python baseline_methods/Psych101/run_openevolve.py \
  --dataset 1peterson2021using \
  --psych_dataset_split train \
  --participant_scope range \
  --range_start_ordinal 0 \
  --range_end_ordinal 49 \
  --api_base http://localhost:8000/v1 \
  --n_iterations 600 \
  --parallel_evaluations 4

OpenEvolve baseline for Psych-101 binary datasets (vanilla prompt, full OpenEvolve machinery).

Uses reference_repos/openevolve as a library. Does NOT use TEH prompt engineering;
task text comes from prompts/openevolve_vanilla/choices13k/infer_single_choice.txt only.

Evolution optimizes train log-likelihood (combined_score = train_loglik). Validation
log-likelihood is logged but not optimized. Test log-likelihood is computed only after
evolution on the participant's best-by-train-loglik program.

OpenEvolve still uses islands / MAP-Elites / archive for parent selection, but the
LLM prompt is intentionally vanilla/minimal (not OpenEvolve's rich history prompt):
current program, vanilla task text, sampled train+val trials, and current metrics only.
num_top_programs / num_diverse_programs config args are not injected into the patched prompt.

Default --n_iterations 600 matches TEH candidate budget (40×10 + 20×10).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import shutil
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_OPENVOLVE_ROOT = _REPO_ROOT / "reference_repos" / "openevolve"
if str(_OPENVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENVOLVE_ROOT))

from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PETERSON2021USING_ALIAS,
    PSYCH101_LEGACY_ALIASES,
    get_psych101_binary_experiment,
    hf_id_for_psych_dataset_split,
    is_psych101_dataset,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
    split_psych_experiment,
)
# utils.teh.* here: dataset registry + participant-id paths only (not TEH prompts/runtime).
from utils.psych101_openevolve_pool import WORKER_VANILLA as _WORKER_VANILLA
from utils.teh.participant_ids import load_valid_participant_ids
from utils.teh.teh_datasets import (
    PARTICIPANT_DATASETS,
    is_binary_loglik_dataset,
    is_mixed_gambles_dataset,
)

from openevolve import OpenEvolve
from openevolve.config import Config
from openevolve.process_parallel import ProcessParallelController

WANDB_PROJECT = "openevolve"
CHOICE13K_LOGLIK_EPS = 1e-9
DEFAULT_SEED_PATH = _REPO_ROOT / "persona_code_example" / "openevolve_vanilla" / "choices13k.py"
DEFAULT_BASE_PROMPT = (
    _REPO_ROOT / "prompts" / "openevolve_vanilla" / "choices13k" / "infer_single_choice.txt"
)

_SHARED_CSV_LOCK = threading.Lock()
_WANDB_LOG_LOCK = threading.Lock()

_ORIG_BUILD_PROMPT = None
_ORIG_GENERATE_WITH_CONTEXT = None
_TRUNCATION_WARN_COUNTS: Dict[int, int] = {}

# Parser-only instructions (not TEH behavioral tricks); required for full-rewrite parsing.
_FULL_REWRITE_OUTPUT_FORMAT = """\
Output only one fenced Python code block:
```python
# full program here
```

The program must define:

def choose(problem, history):
    ...
"""


@dataclass
class TruncationState:
    estimated_tokens: int = 0
    prompt_trial_count: int = 0
    prompt_train_trials: int = 0
    prompt_val_trials: int = 0
    steps: List[str] = field(default_factory=list)


def _effective_psych_dataset_split(dataset: str, psych_dataset_split: str) -> str:
    if is_mixed_gambles_dataset(dataset):
        return DEFAULT_PSYCH_DATASET_SPLIT
    return normalize_psych_dataset_split(psych_dataset_split)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round_floats_for_csv_row(row: Dict[str, Any], ndigits: int = 4) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (float, np.floating)):
            x = float(v)
            out[k] = round(x, ndigits) if math.isfinite(x) else x
        else:
            out[k] = v
    return out


def _round_floats_for_csv_rows(rows: List[Dict[str, Any]], ndigits: int = 4) -> List[Dict[str, Any]]:
    return [_round_floats_for_csv_row(r, ndigits) for r in rows]


def load_valid_participant_ids_from_json(
    dataset: str,
    repo_root: Path,
    filter_mixed_gambles: bool = False,
    *,
    split_ratio: float = 0.8,
    split_seed: int = 42,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
) -> List[int]:
    if not is_binary_loglik_dataset(dataset):
        raise ValueError(f"Unsupported dataset for OpenEvolve baseline: {dataset!r}")
    return load_valid_participant_ids(
        dataset,
        repo_root,
        filter_mixed_gambles=filter_mixed_gambles,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=_effective_psych_dataset_split(dataset, psych_dataset_split),
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )


def resolve_participants_for_scope(
    *,
    dataset: str,
    repo_root: Path,
    participant_scope: str,
    single_participant_id: int,
    range_start_ordinal: Optional[int],
    range_end_ordinal: Optional[int],
    all_max_participants: Optional[int],
    participant_ordinals: Optional[List[int]],
    filter_mixed_gambles: bool,
    split_ratio: float = 0.8,
    split_seed: int = 42,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
) -> List[int]:
    valid = load_valid_participant_ids_from_json(
        dataset,
        repo_root,
        filter_mixed_gambles=filter_mixed_gambles,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
    if participant_scope == "single":
        if single_participant_id not in valid:
            raise ValueError(f"participant id {single_participant_id} not in valid list")
        return [single_participant_id]
    if participant_scope == "range":
        if range_start_ordinal is None or range_end_ordinal is None:
            raise ValueError("range scope requires --range_start_ordinal and --range_end_ordinal")
        if range_start_ordinal < 0 or range_end_ordinal >= len(valid) or range_start_ordinal > range_end_ordinal:
            raise ValueError(f"Invalid ordinal range [{range_start_ordinal}, {range_end_ordinal}]")
        return valid[range_start_ordinal : range_end_ordinal + 1]
    if participant_scope == "ordinals":
        if not participant_ordinals:
            raise ValueError("ordinals scope requires --ordinals")
        out: List[int] = []
        seen: set[int] = set()
        for o in participant_ordinals:
            oi = int(o)
            if oi < 0 or oi >= len(valid):
                raise ValueError(f"Ordinal {oi} out of range (0..{len(valid)-1})")
            pid = int(valid[oi])
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out
    if participant_scope == "all":
        if all_max_participants is not None:
            return valid[: max(0, int(all_max_participants))]
        return list(valid)
    raise ValueError(f"Unknown participant_scope: {participant_scope!r}")


def trials_for_participant(
    dataset: str,
    participant_id: int,
    *,
    split_ratio: float,
    split_seed: int,
    filter_mixed_gambles: bool,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if is_mixed_gambles_dataset(dataset):
        train_trials, val_trials, test_trials, _ = load_mixed_gambles_trials(
            participant_id,
            csv_path=mixed_gambles_csv,
            filter_gain_loss_only=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        return train_trials, val_trials, test_trials
    if not is_psych101_dataset(dataset):
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    exp = get_psych101_binary_experiment(
        dataset,
        int(participant_id),
        split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    train_trials, val_trials, test_trials, _ = split_psych_experiment(
        exp, split_ratio=split_ratio, split_seed=split_seed
    )
    return train_trials, val_trials, test_trials


def compile_program(code_str: str) -> Optional[Callable]:
    import builtins

    safe_builtins = {
        "zip": zip,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "sum": sum,
        "abs": abs,
        "min": min,
        "max": max,
        "float": float,
        "int": int,
        "str": str,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "bool": bool,
        "isinstance": isinstance,
        "hasattr": hasattr,
        "getattr": getattr,
    }
    global_ns = {"__builtins__": safe_builtins}
    local_ns: Dict[str, Any] = {}
    try:
        exec(code_str, global_ns, local_ns)
    except Exception:
        return None
    choose_fn = local_ns.get("choose") or global_ns.get("choose")
    return choose_fn if callable(choose_fn) else None


def _parse_choose_output(p_raw: Any) -> float:
    if isinstance(p_raw, bool) or (isinstance(p_raw, (int, np.integer)) and int(p_raw) in (0, 1)):
        return 1.0 if int(p_raw) == 1 else 0.0
    if isinstance(p_raw, float):
        if not (0.0 <= p_raw <= 1.0) or not math.isfinite(p_raw):
            raise ValueError(f"invalid probability: {p_raw!r}")
        return float(p_raw)
    raise TypeError(f"choose must return float or 0/1, got {type(p_raw)}")


def _clamp_probability(p: float) -> float:
    return min(max(float(p), CHOICE13K_LOGLIK_EPS), 1.0 - CHOICE13K_LOGLIK_EPS)


def evaluate_loglik(choose_fn: Callable, trials: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trials:
        return {"avg_loglik": float("-inf"), "total": 0, "errors": 0}
    loglik_acc = 0.0
    errors = 0
    for t in trials:
        y = int(t["action"])
        try:
            p = _clamp_probability(_parse_choose_output(choose_fn(t["problem"], t["history"])))
        except Exception:
            errors += 1
            p = 0.5
            p = _clamp_probability(p)
        loglik_acc += y * math.log(p) + (1 - y) * math.log(1.0 - p)
    return {
        "avg_loglik": float(loglik_acc / len(trials)),
        "total": len(trials),
        "errors": errors,
    }


def _prompt_block_key(trial: Dict[str, Any]) -> Any:
    problem = trial.get("problem") or {}
    if "gamble_A" in problem and "gamble_B" in problem:
        ga = problem.get("gamble_A", {})
        gb = problem.get("gamble_B", {})
        return (
            tuple(ga.get("probs") or []),
            tuple(ga.get("rewards") or []),
            tuple(gb.get("probs") or []),
            tuple(gb.get("rewards") or []),
            bool(problem.get("has_feedback")),
        )
    keys = tuple(problem.get("option_keys") or [])
    meta_keys = tuple(sorted(k for k in problem.keys() if k not in ("dataset_alias", "experiment_id")))
    return (keys, meta_keys)


def _group_trials_by_block(
    trials: List[Dict[str, Any]],
) -> Tuple[List[Any], Dict[Any, List[Dict[str, Any]]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    order: List[Any] = []
    for t in trials:
        key = _prompt_block_key(t)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(t)
    return order, grouped


def cap_and_subsample_prompt_trials(
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    *,
    max_trials: int,
    max_trials_per_problem: int,
    subsample_seed: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Sample from train+val union only (never test). Returns (trials, n_train, n_val)."""
    pool = list(train_trials) + list(val_trials)
    train_ids = {id(t) for t in train_trials}
    if max_trials <= 0 or not pool:
        return [], 0, 0

    rng = np.random.default_rng(subsample_seed)
    orig_index = {id(t): i for i, t in enumerate(pool)}

    def _sort_chronological(selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(selected, key=lambda t: orig_index[id(t)])

    if max_trials_per_problem <= 0:
        if len(pool) <= max_trials:
            selected = pool
        else:
            idx = rng.choice(len(pool), size=max_trials, replace=False)
            selected = [pool[int(i)] for i in sorted(idx)]
        selected = _sort_chronological(selected)
        n_train = sum(1 for t in selected if id(t) in train_ids)
        return selected, n_train, len(selected) - n_train

    per_block = int(max_trials_per_problem)
    n_blocks_target = max_trials // per_block
    n_extra = max_trials % per_block
    block_order, grouped = _group_trials_by_block(pool)
    n_blocks = len(block_order)
    out: List[Dict[str, Any]] = []
    used_ids: set[int] = set()

    if n_blocks_target > 0 and n_blocks > 0:
        n_blocks_sample = min(n_blocks_target, n_blocks)
        block_idxs = rng.choice(n_blocks, size=n_blocks_sample, replace=False)
        for bi in sorted(int(i) for i in block_idxs):
            block_trials = grouped[block_order[bi]]
            if len(block_trials) <= per_block:
                picked = block_trials
            else:
                tidx = rng.choice(len(block_trials), size=per_block, replace=False)
                picked = [block_trials[int(j)] for j in sorted(tidx)]
            for t in picked:
                if id(t) not in used_ids:
                    out.append(t)
                    used_ids.add(id(t))

    if n_extra > 0:
        remaining = [t for t in pool if id(t) not in used_ids]
        if remaining:
            n_pick = min(n_extra, len(remaining))
            ridx = rng.choice(len(remaining), size=n_pick, replace=False)
            for j in sorted(int(i) for i in ridx):
                t = remaining[j]
                out.append(t)
                used_ids.add(id(t))

    out = _sort_chronological(out)
    n_train = sum(1 for t in out if id(t) in train_ids)
    return out, n_train, len(out) - n_train


def _compact_gamble(problem: Dict[str, Any]) -> str:
    ga = problem.get("gamble_A") or {}
    gb = problem.get("gamble_B") or {}
    return f"A=({list(ga.get('probs') or [])},{list(ga.get('rewards') or [])}) B=({list(gb.get('probs') or [])},{list(gb.get('rewards') or [])})"


def _compact_history(history: List[Dict[str, Any]], max_items: int = 6) -> str:
    if not history:
        return "[]"
    tail = history[-max_items:]
    parts = []
    for h in tail:
        fb = h.get("feedback")
        if fb is None:
            parts.append(f"a{int(h.get('action',0))}")
        else:
            parts.append(f"a{int(h.get('action',0))},r{fb}")
    return "[" + ";".join(parts) + "]"


def format_trial_compact(trial: Dict[str, Any], split_label: str) -> str:
    problem = trial.get("problem") or {}
    if "gamble_A" in problem:
        core = _compact_gamble(problem)
    else:
        keys = problem.get("option_keys") or []
        core = f"keys={list(keys)}"
    fb = 1 if problem.get("has_feedback") else 0
    hist = _compact_history(trial.get("history") or [])
    return f"{core} fb={fb} hist={hist} y={int(trial['action'])} split={split_label}"


def format_trials_compact(
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
) -> str:
    id_to_split: Dict[int, str] = {}
    for t in train_trials:
        id_to_split[id(t)] = "train"
    for t in val_trials:
        id_to_split[id(t)] = "val"
    lines = [format_trial_compact(t, id_to_split.get(id(t), "train")) for t in selected]
    return "\n".join(lines) if lines else "(no prompt trials)"


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def truncate_vanilla_messages(
    *,
    task_text: str,
    program_code: str,
    metrics: Dict[str, Any],
    trials_compact: str,
    max_prompt_tokens: int,
    reserved_completion_tokens: int,
    model_context_len: int,
) -> Tuple[Dict[str, str], TruncationState]:
    """
    Deterministically shrink prompt until estimated tokens <= max_prompt_tokens.
    Never includes test trials (caller must pass train+val only).
    """
    state = TruncationState()
    trial_line_caps = [40, 30, 20, 10, 5]
    all_lines = [ln for ln in trials_compact.splitlines() if ln.strip()]

    def _format_metric(k: str, v: Any) -> str:
        if isinstance(v, bool):
            return f"{k}={v}"
        if isinstance(v, (int, float, np.integer, np.floating)):
            x = float(v)
            if math.isfinite(x):
                return f"{k}={x:.4f}"
            return f"{k}={x}"
        return f"{k}={v}"

    metrics_str = ", ".join(
        _format_metric(k, v)
        for k, v in sorted(metrics.items())
        if k not in ("error",)
    )

    def build_user(trials_text: str, include_metrics: bool) -> str:
        parts = [
            "# Task (vanilla — no TEH prompt engineering)",
            task_text.strip(),
            "",
            "# API",
            "Implement `def choose(problem, history)` returning a float in [0,1]: P(action=1) for the second option_keys entry.",
            "",
            "# Output format (required for parser)",
            _FULL_REWRITE_OUTPUT_FORMAT.strip(),
            "",
            "# Current program",
            "```python",
            program_code,
            "```",
        ]
        if include_metrics and metrics_str:
            parts.extend(["", "# Current metrics", metrics_str])
        if trials_text:
            parts.extend(["", "# Example trials (train+val only; compact)", trials_text])
        return "\n".join(parts)

    system = (
        "You improve Python programs for human choice prediction. "
        "Return one fenced ```python code block containing the full revised program."
    )
    trials_text = trials_compact
    include_metrics = True

    for cap in trial_line_caps:
        if len(all_lines) > cap:
            trials_text = "\n".join(all_lines[:cap])
            state.steps.append(f"cap_prompt_trials_{cap}")
        user = build_user(trials_text, include_metrics)
        est = estimate_tokens(system) + estimate_tokens(user) + reserved_completion_tokens
        state.estimated_tokens = est
        state.prompt_trial_count = len([ln for ln in trials_text.splitlines() if ln.strip()])
        if est <= max_prompt_tokens:
            return {"system": system, "user": user}, state
        state.steps.append("drop_metrics")
        include_metrics = False
        user = build_user(trials_text, include_metrics)
        est = estimate_tokens(system) + estimate_tokens(user) + reserved_completion_tokens
        state.estimated_tokens = est
        if est <= max_prompt_tokens:
            return {"system": system, "user": user}, state

    state.steps.append("minimal_trials_5")
    trials_text = "\n".join(all_lines[:5])
    user = build_user(trials_text, include_metrics=False)
    state.estimated_tokens = estimate_tokens(system) + estimate_tokens(user) + reserved_completion_tokens
    state.prompt_trial_count = min(5, len(all_lines))
    return {"system": system, "user": user}, state


def _estimate_total_prompt_tokens(system_message: str, messages: List[Dict[str, str]], reserved: int) -> int:
    user_text = messages[-1].get("content", "") if messages else ""
    return estimate_tokens(system_message or "") + estimate_tokens(user_text) + reserved


def _append_prompt_diagnostic(
    ctx: Dict[str, Any],
    *,
    source: str,
    state: TruncationState,
    status: str,
    safety_guard_passed: Optional[bool] = None,
) -> None:
    diag_path = ctx.get("diagnostics_path")
    if not diag_path:
        return
    row = {
        "participant_id": ctx.get("participant_id"),
        "iteration": ctx.get("iteration"),
        "source": source,
        "estimated_prompt_tokens": state.estimated_tokens,
        "hard_prompt_token_cap": ctx.get("hard_prompt_token_cap"),
        "prompt_trial_count": state.prompt_trial_count,
        "prompt_train_trials": ctx.get("prompt_train_trials"),
        "prompt_val_trials": ctx.get("prompt_val_trials"),
        "truncation_steps": state.steps,
        "status": status,
        "safety_guard_passed": safety_guard_passed,
        "openevolve_rich_prompt_included": False,
    }
    try:
        with open(diag_path, "a", encoding="utf-8") as df:
            df.write(json.dumps(row) + "\n")
    except Exception:
        pass
    if state.steps:
        pid = int(ctx.get("participant_id", -1))
        _TRUNCATION_WARN_COUNTS[pid] = _TRUNCATION_WARN_COUNTS.get(pid, 0) + 1
        n = _TRUNCATION_WARN_COUNTS[pid]
        if n <= 3 or n % 50 == 0:
            print(
                f"[p{pid}] prompt truncated (iter={ctx.get('iteration')}, source={source}): "
                f"{state.steps}"
            )


def _minimal_shrink_user_message(user_text: str, cap: int, system_message: str, reserved: int) -> Tuple[str, List[str]]:
    """Drop lines from the end of user content until estimated tokens fit cap."""
    steps: List[str] = []
    est = estimate_tokens(system_message or "") + estimate_tokens(user_text) + reserved
    if est <= cap:
        return user_text, steps
    lines = user_text.splitlines()
    while lines and est > cap:
        lines = lines[:-1]
        steps.append("guard_drop_user_lines")
        user_text = "\n".join(lines) if lines else "(prompt truncated to fit token cap)"
        est = estimate_tokens(system_message or "") + estimate_tokens(user_text) + reserved
    if est > cap and user_text:
        max_chars = max(200, cap * 3)
        user_text = user_text[:max_chars]
        steps.append("guard_hard_char_cap")
        est = estimate_tokens(system_message or "") + estimate_tokens(user_text) + reserved
    return user_text, steps


def _patched_build_prompt(
    self,
    current_program: str = "",
    parent_program: str = "",
    program_metrics: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> Dict[str, str]:
    """
    Sole builder/truncator for LLM prompts.

    OpenEvolve database/islands still select parents, but top/inspiration/history/artifacts
    from the default OpenEvolve prompt are intentionally omitted here.
    """
    ctx = _WORKER_VANILLA
    task_text = ctx.get("task_text", "")
    program_code = current_program or ""
    metrics = program_metrics or {}
    trials_compact = ctx.get("trials_compact", "")
    max_prompt_tokens = int(ctx.get("hard_prompt_token_cap", 14000))
    reserved = int(ctx.get("llm_max_tokens", 1024)) + 256
    model_len = int(ctx.get("max_model_len", 16384))
    max_prompt_tokens = min(max_prompt_tokens, model_len - reserved)
    prompt, state = truncate_vanilla_messages(
        task_text=task_text,
        program_code=program_code,
        metrics=metrics,
        trials_compact=trials_compact,
        max_prompt_tokens=max_prompt_tokens,
        reserved_completion_tokens=reserved,
        model_context_len=model_len,
    )
    state.prompt_train_trials = int(ctx.get("prompt_train_trials", 0))
    state.prompt_val_trials = int(ctx.get("prompt_val_trials", 0))
    ctx["last_truncation"] = state
    status = "truncated" if state.steps else "ok"
    _append_prompt_diagnostic(ctx, source="build_prompt", state=state, status=status, safety_guard_passed=None)
    return prompt


def _patched_generate_with_context(self, system_message, messages, **kwargs):
    """Final safety guard only — never rebuilds the task prompt from message content."""
    ctx = _WORKER_VANILLA
    if ctx and messages:
        reserved = int(ctx.get("llm_max_tokens", 1024)) + 256
        model_len = int(ctx.get("max_model_len", 16384))
        cap = min(int(ctx.get("hard_prompt_token_cap", 14000)), model_len - reserved)
        total_est = _estimate_total_prompt_tokens(system_message or "", messages, reserved)
        guard_state = TruncationState(estimated_tokens=total_est)
        last_trunc = ctx.get("last_truncation")
        if isinstance(last_trunc, TruncationState):
            guard_state.prompt_trial_count = last_trunc.prompt_trial_count
        else:
            guard_state.prompt_trial_count = int(ctx.get("prompt_trial_count", 0))

        if total_est > cap:
            user_text = messages[-1].get("content", "") or ""
            shrunk_user, guard_steps = _minimal_shrink_user_message(
                user_text, cap, system_message or "", reserved
            )
            messages = [{"role": "user", "content": shrunk_user}]
            total_est = _estimate_total_prompt_tokens(system_message or "", messages, reserved)
            guard_state.steps = list(guard_steps)
            guard_state.estimated_tokens = total_est
            if total_est > cap:
                _append_prompt_diagnostic(
                    ctx,
                    source="generate_guard",
                    state=guard_state,
                    status="skipped_over_cap",
                    safety_guard_passed=False,
                )
                # Skip vLLM call; OpenEvolve will treat unparseable response as failed iteration.
                return "# LLM call skipped: prompt exceeds token budget after safety guard"
            _append_prompt_diagnostic(
                ctx,
                source="generate_guard",
                state=guard_state,
                status="guard_shrink_applied",
                safety_guard_passed=True,
            )
        else:
            _append_prompt_diagnostic(
                ctx,
                source="generate_guard",
                state=guard_state,
                status="ok",
                safety_guard_passed=True,
            )
    return _ORIG_GENERATE_WITH_CONTEXT(self, system_message, messages, **kwargs)


def _install_runtime_patches() -> None:
    global _ORIG_BUILD_PROMPT, _ORIG_GENERATE_WITH_CONTEXT
    import openevolve.llm.openai as oe_openai
    import openevolve.prompt.sampler as oe_sampler

    if _ORIG_BUILD_PROMPT is None:
        _ORIG_BUILD_PROMPT = oe_sampler.PromptSampler.build_prompt
        oe_sampler.PromptSampler.build_prompt = _patched_build_prompt
    if _ORIG_GENERATE_WITH_CONTEXT is None:
        _ORIG_GENERATE_WITH_CONTEXT = oe_openai.OpenAILLM.generate_with_context
        oe_openai.OpenAILLM.generate_with_context = _patched_generate_with_context


_OE_VANILLA_WORKER_CODE = """
def _oe_vanilla_worker_init(config_dict, evaluation_file, parent_env=None):
    from utils.psych101_openevolve_pool import ensure_patches_installed

    ensure_patches_installed()
    return _oe_vanilla_saved_init(config_dict, evaluation_file, parent_env)


def _oe_vanilla_run_iteration_worker(iteration, db_snapshot, parent_id, inspiration_ids):
    from utils.psych101_openevolve_pool import ensure_patches_installed, set_worker_vanilla_ctx

    ensure_patches_installed()
    set_worker_vanilla_ctx(db_snapshot.get("_vanilla_ctx", {}))
    return _oe_vanilla_saved_run_worker(iteration, db_snapshot, parent_id, inspiration_ids)
"""


def _patch_process_parallel_worker() -> None:
    import openevolve.process_parallel as op

    if getattr(op, "_vanilla_worker_patched", False):
        return

    if not hasattr(op, "_oe_vanilla_saved_run_worker"):
        op._oe_vanilla_saved_run_worker = op._run_iteration_worker
        op._oe_vanilla_saved_init = op._worker_init

    exec(_OE_VANILLA_WORKER_CODE, op.__dict__)

    op._worker_init = op._oe_vanilla_worker_init
    op._run_iteration_worker = op._oe_vanilla_run_iteration_worker
    op._vanilla_worker_patched = True


def _hook_openevolve_controller(participant_ctx: Dict[str, Any]) -> None:
    """OpenEvolve.run() constructs ProcessParallelController internally; hook it here."""
    import openevolve.controller as oc

    if not hasattr(oc, "_Orig_ProcessParallelController"):
        oc._Orig_ProcessParallelController = oc.ProcessParallelController

    class _HookedController(VanillaProcessParallelController):
        def __init__(
            self,
            config: Config,
            evaluation_file: str,
            database: Any,
            evolution_tracer=None,
            file_suffix: str = ".py",
        ):
            VanillaProcessParallelController.__init__(
                self,
                config,
                evaluation_file,
                database,
                evolution_tracer,
                file_suffix,
                participant_ctx,
            )

    oc.ProcessParallelController = _HookedController


def _unhook_openevolve_controller() -> None:
    import openevolve.controller as oc

    if hasattr(oc, "_Orig_ProcessParallelController"):
        oc.ProcessParallelController = oc._Orig_ProcessParallelController


class VanillaProcessParallelController(ProcessParallelController):
    """
    Per-iteration train+val prompt trial resampling for the patched vanilla LLM prompt.

    Parent selection still uses OpenEvolve islands/MAP-Elites/archive; only the prompt
    content is minimal (see module docstring).
    """

    def __init__(self, config, evaluation_file, database, evolution_tracer, file_suffix, participant_ctx):
        super().__init__(config, evaluation_file, database, evolution_tracer, file_suffix)
        self.participant_ctx = participant_ctx

    def _create_database_snapshot(self) -> Dict[str, Any]:
        snap = super()._create_database_snapshot()
        ctx = dict(self.participant_ctx)
        train_trials = ctx["train_trials"]
        val_trials = ctx["val_trials"]
        iteration = int(ctx.get("_submit_iteration", 0))
        selected, n_train, n_val = cap_and_subsample_prompt_trials(
            train_trials,
            val_trials,
            max_trials=int(ctx["max_prompt_train_trials"]),
            max_trials_per_problem=int(ctx["max_prompt_trials_per_problem"]),
            subsample_seed=int(ctx["split_seed"]) + iteration * 9973 + int(ctx["participant_id"]) * 100007,
        )
        ctx["prompt_trial_count"] = len(selected)
        ctx["prompt_train_trials"] = n_train
        ctx["prompt_val_trials"] = n_val
        snap["_vanilla_ctx"] = {
            "participant_id": ctx["participant_id"],
            "iteration": iteration,
            "task_text": ctx["task_text"],
            "trials_compact": format_trials_compact(train_trials, val_trials, selected),
            "hard_prompt_token_cap": ctx["hard_prompt_token_cap"],
            "llm_max_tokens": ctx["llm_max_tokens"],
            "max_model_len": ctx["max_model_len"],
            "prompt_train_trials": n_train,
            "prompt_val_trials": n_val,
            "diagnostics_path": ctx.get("diagnostics_path"),
        }
        return snap

    def _submit_iteration(self, iteration: int, island_id: Optional[int] = None):
        self.participant_ctx["_submit_iteration"] = iteration
        return super()._submit_iteration(iteration, island_id)


def openevolve_output_base_dir(
    dataset: str,
    timestamp: str,
    *,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
) -> str:
    split = _effective_psych_dataset_split(dataset, psych_dataset_split)
    alias = normalize_psych101_dataset_alias(dataset)
    return f"generated_outputs/psych101_{split}/openevolve/{alias}/run_{timestamp}"


def wandb_run_name(dataset: str, timestamp: str) -> str:
    """Short wandb run name: ``{dataset}_{timestamp}`` (e.g. ``1peterson2021using_260520_223805``)."""
    return f"{dataset}_{timestamp}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_evolution_split_json(path: Path, train_trials: List[Dict[str, Any]], val_trials: List[Dict[str, Any]]) -> None:
    """Train+val only — written beside evaluator; no test split on disk during evolution."""
    payload = {
        "train": train_trials,
        "val": val_trials,
        "note": "Evolution evaluator reads train+val only. Test is never stored here.",
    }
    path.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")


def _write_posthoc_test_json(path: Path, test_trials: List[Dict[str, Any]]) -> None:
    """Test split for post-hoc evaluation only; outside evaluator directory."""
    payload = {
        "test": test_trials,
        "note": "Post-hoc only. Not read by OpenEvolve evaluator during evolution.",
    }
    path.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")


def _render_evaluator_py(evolution_split_path: Path) -> str:
    split_path = str(evolution_split_path.resolve())
    return f'''"""Auto-generated OpenEvolve evaluator (train objective; no test access)."""
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List

SPLIT_PATH = Path(r\"{split_path}\")
CHOICE13K_LOGLIK_EPS = {CHOICE13K_LOGLIK_EPS}


def _load_splits():
    data = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    return data["train"], data["val"]


def compile_program(code_str: str):
    safe_builtins = {{
        "zip": zip, "len": len, "range": range, "enumerate": enumerate, "sum": sum,
        "abs": abs, "min": min, "max": max, "float": float, "int": int, "str": str,
        "list": list, "dict": dict, "tuple": tuple, "bool": bool, "isinstance": isinstance,
        "hasattr": hasattr, "getattr": getattr,
    }}
    global_ns = {{"__builtins__": safe_builtins}}
    local_ns: Dict[str, Any] = {{}}
    try:
        exec(code_str, global_ns, local_ns)
    except Exception:
        return None
    fn = local_ns.get("choose") or global_ns.get("choose")
    return fn if callable(fn) else None


def _parse_choose_output(p_raw: Any) -> float:
    if isinstance(p_raw, bool) or (isinstance(p_raw, int) and int(p_raw) in (0, 1)):
        return 1.0 if int(p_raw) == 1 else 0.0
    if isinstance(p_raw, float) and math.isfinite(p_raw) and 0.0 <= p_raw <= 1.0:
        return float(p_raw)
    raise ValueError(f"invalid choose output: {{p_raw!r}}")


def _clamp(p: float) -> float:
    return min(max(float(p), CHOICE13K_LOGLIK_EPS), 1.0 - CHOICE13K_LOGLIK_EPS)


def evaluate_trials(choose_fn: Callable, trials: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trials:
        return {{"avg_loglik": float("-inf"), "errors": 1}}
    ll = 0.0
    errors = 0
    for t in trials:
        y = int(t["action"])
        try:
            p = _clamp(_parse_choose_output(choose_fn(t["problem"], t["history"])))
        except Exception:
            errors += 1
            p = 0.5
            p = _clamp(p)
        ll += y * math.log(p) + (1 - y) * math.log(1.0 - p)
    return {{"avg_loglik": float(ll / len(trials)), "errors": errors}}


def evaluate(program_path: str) -> Dict[str, float]:
    path = Path(program_path)
    code = path.read_text(encoding="utf-8")
    choose_fn = compile_program(code)
    if choose_fn is None:
        return {{"combined_score": float("-inf"), "train_loglik": float("-inf"), "val_loglik": float("-inf"), "error": "no choose()"}}
    train_trials, val_trials = _load_splits()
    train_eval = evaluate_trials(choose_fn, train_trials)
    val_eval = evaluate_trials(choose_fn, val_trials)
    if train_eval.get("errors", 0) > 0 and len(train_trials) > 0:
        return {{
            "combined_score": float("-inf"),
            "train_loglik": float("-inf"),
            "val_loglik": float("-inf"),
            "error": "invalid_on_train",
        }}
    train_ll = float(train_eval["avg_loglik"])
    val_ll = float(val_eval["avg_loglik"])
    return {{
        "combined_score": train_ll,
        "train_loglik": train_ll,
        "val_loglik": val_ll,
    }}
'''


def _build_config(args, iterations: int) -> Config:
    cfg = Config()
    cfg.max_iterations = iterations
    cfg.checkpoint_interval = args.checkpoint_interval
    cfg.log_level = args.log_level
    cfg.random_seed = args.random_seed
    cfg.diff_based_evolution = False
    cfg.language = "python"
    cfg.file_suffix = ".py"
    cfg.early_stopping_patience = args.early_stopping_patience
    cfg.convergence_threshold = args.convergence_threshold
    cfg.early_stopping_metric = "combined_score"

    cfg.llm.api_base = args.api_base
    cfg.llm.api_key = args.llm_api_key
    cfg.llm.temperature = args.temperature
    cfg.llm.max_tokens = args.llm_max_tokens
    cfg.llm.timeout = args.llm_timeout
    cfg.llm.retries = args.llm_retries
    cfg.llm.random_seed = args.random_seed
    if args.top_p is not None:
        cfg.llm.top_p = args.top_p
    cfg.llm.primary_model = args.model
    cfg.llm.primary_model_weight = 1.0
    cfg.llm.rebuild_models()

    cfg.prompt.system_message = (
        "Evolve Python code for human binary choice prediction. Return a complete program file."
    )
    cfg.prompt.num_top_programs = args.num_top_programs
    cfg.prompt.num_diverse_programs = args.num_diverse_programs
    cfg.prompt.include_artifacts = args.include_artifacts
    cfg.prompt.use_template_stochasticity = False
    cfg.database.population_size = args.population_size
    cfg.database.archive_size = args.archive_size
    cfg.database.num_islands = args.num_islands
    cfg.database.exploration_ratio = args.exploration_ratio
    cfg.database.exploitation_ratio = args.exploitation_ratio
    cfg.database.elite_selection_ratio = args.elite_selection_ratio
    cfg.database.migration_interval = args.migration_interval
    cfg.database.migration_rate = args.migration_rate
    cfg.database.feature_dimensions = list(args.feature_dimensions)
    cfg.database.feature_bins = args.feature_bins
    cfg.database.log_prompts = True
    cfg.database.random_seed = args.random_seed

    cfg.evaluator.timeout = args.evaluator_timeout
    cfg.evaluator.max_retries = args.evaluator_max_retries
    cfg.evaluator.parallel_evaluations = args.parallel_evaluations
    cfg.evaluator.cascade_evaluation = args.cascade_evaluation
    cfg.evaluator.use_llm_feedback = args.use_llm_feedback
    cfg.evaluator.enable_artifacts = args.enable_artifacts
    return cfg


def _find_best_program_by_train_loglik(checkpoint_dir: Path) -> Tuple[Optional[Path], Optional[float]]:
    programs_dir = checkpoint_dir / "programs"
    if not programs_dir.is_dir():
        return None, None
    best_path: Optional[Path] = None
    best_ll: Optional[float] = None
    for prog_file in programs_dir.glob("*.json"):
        try:
            data = json.loads(prog_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = data.get("metrics") or {}
        train_ll = _safe_float(metrics.get("train_loglik"))
        if train_ll is None:
            train_ll = _safe_float(metrics.get("combined_score"))
        if train_ll is None:
            continue
        if best_ll is None or train_ll > best_ll:
            best_ll = train_ll
            code = data.get("code")
            if code:
                best_path = prog_file
    return best_path, best_ll


def _program_code_from_json(prog_json: Path) -> str:
    data = json.loads(prog_json.read_text(encoding="utf-8"))
    return data.get("code") or ""


def _wandb_log_participant(wandb_module: Any, participant_id: int, metrics: Dict[str, Any], step: int) -> None:
    pid = int(participant_id)
    payload: Dict[str, Any] = {f"p{pid}_step": int(step)}
    for suffix in ("train_loglik", "val_loglik", "test_loglik"):
        if metrics.get(suffix) is not None:
            payload[f"p{pid}_{suffix}"] = metrics[suffix]
            payload[f"p{pid}/{suffix}"] = metrics[suffix]
    with _WANDB_LOG_LOCK:
        wandb_module.log(payload)


def run_participant(
    args,
    participant_id: int,
    participant_ordinal: int,
    run_dir: Path,
    wandb_module: Any = None,
) -> Dict[str, Any]:
    _patch_process_parallel_worker()
    _install_runtime_patches()

    participant_dir = run_dir / f"participant_{participant_id}"
    exp_dir = participant_dir / "openevolve_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_trials, val_trials, test_trials = trials_for_participant(
        args.dataset,
        participant_id,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        filter_mixed_gambles=args.filter_mixed_gambles,
        psych_dataset_split=_effective_psych_dataset_split(args.dataset, args.psych_dataset_split),
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
    )

    evolution_split_path = exp_dir / "trials_evolution_split.json"
    _write_evolution_split_json(evolution_split_path, train_trials, val_trials)
    posthoc_test_path = participant_dir / "trials_test_posthoc.json"
    _write_posthoc_test_json(posthoc_test_path, test_trials)

    task_text = Path(args.base_prompt).read_text(encoding="utf-8")
    (exp_dir / "vanilla_task_prompt.txt").write_text(task_text, encoding="utf-8")

    initial_src = Path(args.seed_path).resolve()
    initial_dst = exp_dir / "initial_program.py"
    shutil.copy2(initial_src, initial_dst)

    evaluator_path = exp_dir / "evaluator.py"
    evaluator_path.write_text(_render_evaluator_py(evolution_split_path), encoding="utf-8")

    cfg = _build_config(args, args.n_iterations)
    cfg.to_yaml(exp_dir / "config.yaml")

    diagnostics_path = participant_dir / "prompt_truncation_diagnostics.jsonl"
    participant_ctx = {
        "participant_id": participant_id,
        "train_trials": train_trials,
        "val_trials": val_trials,
        "task_text": task_text,
        "max_prompt_train_trials": args.max_prompt_train_trials,
        "max_prompt_trials_per_problem": args.max_prompt_trials_per_problem,
        "split_seed": args.split_seed,
        "hard_prompt_token_cap": args.hard_prompt_token_cap,
        "llm_max_tokens": args.llm_max_tokens,
        "max_model_len": args.max_model_len,
        "diagnostics_path": str(diagnostics_path),
    }

    oe_output = participant_dir / "openevolve_output"
    oe = OpenEvolve(
        initial_program_path=str(initial_dst),
        evaluation_file=str(evaluator_path),
        config=cfg,
        output_dir=str(oe_output),
    )
    oe.config.database.novelty_llm = oe.llm_ensemble
    _patch_process_parallel_worker()
    _install_runtime_patches()
    _hook_openevolve_controller(participant_ctx)
    status = "ok"
    error_msg = ""
    n_completed = 0
    best_program_path = participant_dir / "best_program.py"
    train_ll = val_ll = test_ll = None
    best_train_ll: Optional[float] = None

    try:
        try:
            best_program = asyncio.run(oe.run(iterations=args.n_iterations))
        finally:
            _unhook_openevolve_controller()
        n_completed = oe.database.last_iteration if oe.database.last_iteration else args.n_iterations

        ckpt_root = oe_output / "checkpoints"
        ckpt_dirs = sorted(
            [p for p in ckpt_root.glob("checkpoint_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_")[-1]) if "_" in p.name else 0,
        )
        latest_ckpt = ckpt_dirs[-1] if ckpt_dirs else None

        if latest_ckpt is not None:
            prog_json, best_train_ll = _find_best_program_by_train_loglik(latest_ckpt)
            if prog_json is not None:
                code = _program_code_from_json(prog_json)
                best_program_path.write_text(code, encoding="utf-8")
            elif best_program is not None:
                best_program_path.write_text(best_program.code, encoding="utf-8")
        elif best_program is not None:
            best_program_path.write_text(best_program.code, encoding="utf-8")

        oe_best = oe_output / "best" / "best_program.py"
        if oe_best.is_file() and not best_program_path.is_file():
            shutil.copy2(oe_best, best_program_path)

        if best_program_path.is_file():
            code = best_program_path.read_text(encoding="utf-8")
            choose_fn = compile_program(code)
            if choose_fn is None:
                status = "failed"
                error_msg = "best program does not compile or lacks choose()"
            else:
                train_ll = evaluate_loglik(choose_fn, train_trials)["avg_loglik"]
                val_ll = evaluate_loglik(choose_fn, val_trials)["avg_loglik"]
                test_ll = evaluate_loglik(choose_fn, test_trials)["avg_loglik"]
        else:
            status = "failed"
            error_msg = "no best program file produced"

        if diagnostics_path.is_file():
            with diagnostics_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)
                        except Exception:
                            pass
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        traceback.print_exc()

    row = {
        "participant_id": participant_id,
        "participant_ordinal": participant_ordinal,
        "train_loglik": train_ll,
        "val_loglik": val_ll,
        "test_loglik": test_ll,
        "n_iterations_requested": args.n_iterations,
        "n_iterations_completed": n_completed,
        "best_program_path": str(best_program_path) if best_program_path.is_file() else "",
        "status": status,
        "error": error_msg,
        "best_train_loglik_in_pool": best_train_ll,
        "prompt_trials_resampled_per_candidate": True,
        "prompt_trials_source": "train+val_union",
    }
    (participant_dir / "results.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    if wandb_module is not None and train_ll is not None:
        _wandb_log_participant(
            wandb_module,
            participant_id,
            {"train_loglik": train_ll, "val_loglik": val_ll, "test_loglik": test_ll},
            step=max(0, n_completed),
        )
    return row


def _write_experiment_csvs(run_dir: Path, detail_rows: List[Dict[str, Any]]) -> None:
    details_fields = ["participant_id", "participant_ordinal", "train_loglik", "val_loglik", "test_loglik"]
    details_path = run_dir / "participant_details_loglik.csv"
    with details_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_round_floats_for_csv_rows(detail_rows))

    train_vals = [r["train_loglik"] for r in detail_rows if r.get("train_loglik") is not None]
    val_vals = [r["val_loglik"] for r in detail_rows if r.get("val_loglik") is not None]
    test_vals = [r["test_loglik"] for r in detail_rows if r.get("test_loglik") is not None]
    summary_row = {
        "num_of_participants": len(detail_rows),
        "avg_train_loglik": float(np.mean(train_vals)) if train_vals else None,
        "avg_val_loglik": float(np.mean(val_vals)) if val_vals else None,
        "avg_test_loglik": float(np.mean(test_vals)) if test_vals else None,
    }
    summary_fields = ["num_of_participants", "avg_train_loglik", "avg_val_loglik", "avg_test_loglik"]
    summary_path = run_dir / "summary_loglik.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(_round_floats_for_csv_row(summary_row))

    final_path = run_dir / "final_participant_summary.csv"
    final_fields = [
        "participant_id",
        "participant_ordinal",
        "train_loglik",
        "val_loglik",
        "test_loglik",
        "n_iterations_requested",
        "n_iterations_completed",
        "best_program_path",
        "status",
        "error",
    ]
    with final_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=final_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(_round_floats_for_csv_rows(detail_rows))


def build_arg_parser() -> argparse.ArgumentParser:
    psych_choices = sorted(PARTICIPANT_DATASETS | set(PSYCH101_LEGACY_ALIASES))
    p = argparse.ArgumentParser(description="OpenEvolve vanilla baseline for Psych-101 binary datasets")
    p.add_argument("--dataset", type=str, default=PETERSON2021USING_ALIAS, choices=psych_choices)
    p.add_argument("--psych_dataset_split", type=str, default=DEFAULT_PSYCH_DATASET_SPLIT, choices=["train", "test"])
    p.add_argument("--fitness_metric", type=str, default="loglik", choices=["loglik"])
    p.add_argument("--participant_scope", type=str, default="range", choices=["single", "range", "ordinals", "all"])
    p.add_argument("--single_participant_id", type=int, default=0)
    p.add_argument("--range_start_ordinal", type=int, default=0)
    p.add_argument("--range_end_ordinal", type=int, default=49)
    p.add_argument("--ordinals", nargs="+", type=int, default=None)
    p.add_argument("--all_max_participants", type=int, default=None)
    p.add_argument("--filter_mixed_gambles", action="store_true")
    p.add_argument("--split_mode", type=str, default="within_participant", choices=["within_participant", "across_participants"])
    p.add_argument("--split_ratio", type=float, default=0.6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed_path", type=str, default=str(DEFAULT_SEED_PATH))
    p.add_argument("--base_prompt", type=str, default=str(DEFAULT_BASE_PROMPT))
    p.add_argument("--n_iterations", type=int, default=600)
    p.add_argument("--checkpoint_interval", type=int, default=50)
    p.add_argument("--max_prompt_train_trials", type=int, default=40)
    p.add_argument("--max_prompt_trials_per_problem", type=int, default=10)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--api_base", type=str, default=None)
    p.add_argument("--vllm_url", type=str, default=os.environ.get("VLLM_LOCAL_URL", "http://localhost:8000/v1"))
    p.add_argument("--llm_api_key", type=str, default=os.environ.get("VLLM_LOCAL_API_KEY", "EMPTY"))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--llm_max_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--hard_prompt_token_cap", type=int, default=14000)
    p.add_argument("--llm_timeout", type=int, default=300)
    p.add_argument("--llm_retries", type=int, default=3)
    p.add_argument("--parallel_evaluations", type=int, default=4)
    p.add_argument("--evaluator_timeout", type=int, default=120)
    p.add_argument("--evaluator_max_retries", type=int, default=2)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--local_dataset", type=str, default=None)
    p.add_argument("--mixed_gambles_csv", type=str, default=DEFAULT_CSV_PATH)
    p.add_argument(
        "--no_log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable all wandb logging (default: log to wandb project openevolve).",
    )
    p.add_argument("--population_size", type=int, default=1000)
    p.add_argument("--archive_size", type=int, default=100)
    p.add_argument("--num_islands", type=int, default=5)
    p.add_argument("--exploration_ratio", type=float, default=0.2)
    p.add_argument("--exploitation_ratio", type=float, default=0.7)
    p.add_argument("--elite_selection_ratio", type=float, default=0.1)
    p.add_argument("--migration_interval", type=int, default=50)
    p.add_argument("--migration_rate", type=float, default=0.1)
    p.add_argument("--num_top_programs", type=int, default=1)
    p.add_argument("--num_diverse_programs", type=int, default=0)
    p.add_argument("--include_artifacts", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use_llm_feedback", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cascade_evaluation", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enable_artifacts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--feature_dimensions", nargs="+", type=str, default=["complexity", "diversity"])
    p.add_argument("--feature_bins", type=int, default=10)
    p.add_argument("--early_stopping_patience", type=int, default=None)
    p.add_argument("--convergence_threshold", type=float, default=0.001)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.dataset = normalize_psych101_dataset_alias(args.dataset)
    if args.fitness_metric != "loglik":
        raise ValueError("This baseline only supports --fitness_metric loglik")
    if args.split_mode != "within_participant":
        raise ValueError("OpenEvolve baseline requires --split_mode within_participant")
    if args.api_base is None:
        args.api_base = args.vllm_url

    psych_dataset_split = _effective_psych_dataset_split(args.dataset, args.psych_dataset_split)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else Path(
        openevolve_output_base_dir(args.dataset, timestamp, psych_dataset_split=psych_dataset_split)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    valid = load_valid_participant_ids_from_json(
        args.dataset,
        _REPO_ROOT,
        filter_mixed_gambles=args.filter_mixed_gambles,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
    )
    participants = resolve_participants_for_scope(
        dataset=args.dataset,
        repo_root=_REPO_ROOT,
        participant_scope=args.participant_scope,
        single_participant_id=args.single_participant_id,
        range_start_ordinal=args.range_start_ordinal,
        range_end_ordinal=args.range_end_ordinal,
        all_max_participants=args.all_max_participants,
        participant_ordinals=args.ordinals,
        filter_mixed_gambles=args.filter_mixed_gambles,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
    )
    pid_to_ordinal = {pid: i for i, pid in enumerate(valid)}

    wandb_module = None
    if args.no_log:
        os.environ["WANDB_DISABLED"] = "true"
        print("wandb logging disabled (--no_log).")
    else:
        try:
            import wandb as _wandb

            wandb_module = _wandb
            run_name = wandb_run_name(args.dataset, timestamp)
            wandb_module.init(
                project=WANDB_PROJECT,
                name=run_name,
                config=vars(args),
                reinit=False,
            )
            print(f"wandb run: {WANDB_PROJECT}/{run_name}")
            for pid in participants:
                wandb_module.define_metric(f"p{pid}_step")
                wandb_module.define_metric(f"p{pid}/*", step_metric=f"p{pid}_step")
        except Exception as e:
            print(f"wandb disabled (init failed): {e}")

    print(f"OpenEvolve vanilla baseline | dataset={args.dataset} | psych_split={psych_dataset_split}")
    print(f"HF corpus: {hf_id_for_psych_dataset_split(psych_dataset_split)}")
    print(f"Participants: {len(participants)} | n_iterations={args.n_iterations} | output={run_dir}")
    print(
        "Split: split_ratio=0.6 -> 60% train, remainder 50/50 val/test blocks; "
        "evolution uses train_loglik only; prompt trials from train+val union; test post-hoc only."
    )
    print(
        "Prompt: vanilla/minimal (task + program + train/val trials + metrics). "
        "OpenEvolve islands/MAP-Elites/archive still active for parent selection; "
        "top/diverse/history/artifacts are NOT included in LLM prompts."
    )

    detail_rows: List[Dict[str, Any]] = []
    for pid in tqdm(participants, desc="participants"):
        try:
            row = run_participant(args, pid, pid_to_ordinal.get(pid, -1), run_dir, wandb_module)
        except Exception as e:
            row = {
                "participant_id": pid,
                "participant_ordinal": pid_to_ordinal.get(pid, -1),
                "status": "failed",
                "error": str(e),
                "n_iterations_requested": args.n_iterations,
                "n_iterations_completed": 0,
            }
            traceback.print_exc()
        detail_rows.append(row)

    with _SHARED_CSV_LOCK:
        _write_experiment_csvs(run_dir, detail_rows)

    if wandb_module is not None:
        wandb_module.finish()

    print(f"Done. CSVs: {run_dir / 'participant_details_loglik.csv'}, {run_dir / 'summary_loglik.csv'}")


if __name__ == "__main__":
    # ``100 \ --no_log`` (backslash before space) makes bash pass ``' --no_log'``; strip that.
    sys.argv = [a.strip() if a.startswith(" ") else a for a in sys.argv]
    main()
