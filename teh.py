"""
teh.py — TEH (Template Evolution HuggingFace): Psych-101 binary cognitive datasets.

Evolved programs implement choose(problem, history) -> float = P(action=1) on structured trials
parsed from Psych-101 / Psych-101-test natural-language rows (see data_modules/psych101_binary.py).

Psych-101 binary aliases (e.g. peterson2021using) plus local mixed_gambles and
external Bernoulli adapters (e.g. bergert_nosofsky_2007). Legacy choice13k/cpc18/
gridworld entrypoints are not supported here (use dataset aliases instead of choice13k).
"""

import hashlib
import math
import os
import re
import json
import csv
import shlex
import shutil
import socket
import sys
import threading
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Tuple, Set, Sequence, Mapping
from datetime import datetime
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# Gridworld/JAX: loaded on demand via _load_gridworld_stack() (not needed for TEH --help).
_GRIDWORLD_STACK_LOADED = False
jax: Any = None
jnp: Any = None
flax: Any = None
get_all_problem_configs: Any = None
make_dataloader: Any = None
AutomaticityEnv: Any = None
State: Any = None

# Psych-101 binary loaders (TEH)
from data_modules.choice13k import Experiment, Block
from data_modules.mixed_gambles import (
    DEFAULT_CSV_PATH,
    load_mixed_gambles_trials,
)
from data_modules.external import (
    is_external_dataset,
    load_external_loglik_trials,
    external_default_data_dir,
)
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PETERSON2021USING_ALIAS,
    PSYCH101_BINARY_DATASETS,
    PSYCH101_LEGACY_ALIASES,
    experiment_to_trial_dicts,
    format_trials_for_prompt,
    get_psych101_binary_experiment,
    get_psych101_binary_experiments,
    hf_id_for_psych_dataset_split,
    is_psych101_dataset,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
    split_psych_experiment,
)
from utils.teh.teh_datasets import (
    IMPLEMENTED_PSYCH101_ALIASES,
    LOGlik_VAL_SPLIT_DATASETS,
    MIXED_GAMBLES,
    PARTICIPANT_DATASETS,
    dataset_output_type,
    emnlp_ordinal_range,
    is_binary_loglik_dataset,
    is_categorical_output_dataset,
    is_mixed_gambles_dataset,
    uses_train_val_test_loglik_split,
)
from utils.prompt_flags import (
    load_single_code_template,
    single_code_template_prompt_suffix,
)
from utils.teh.teh_runtime import (
    CONCISE_PROGRAM_GUIDANCE,
    DEFAULT_SEED_PROGRAM,
    TEH_WANDB_PROJECT,
    setup_teh_run_prompts,
    teh_output_base_dir,
    teh_wandb_run_name,
    valid_participant_ids_path,
)
from utils.teh.prompt_context import (
    DEFAULT_EXAMPLE_CHAR_BUDGET,
    DEFAULT_HISTORY_MAX_ENTRIES,
    DEFAULT_MAX_EXAMPLES,
    RUNTIME_CONTRACT_FILENAME,
    append_runtime_contract_if_present,
    attach_runtime_contract_to_prompt,
)
from utils.teh.g2_paired_packing import (
    PairedPackingFitError,
    freeze_g2_target_examples,
    load_paired_pack_freeze,
    materialize_frozen_trials,
    prompt_contains_source_suffix,
    stable_target_example_id,
    trim_reason as g2_trim_reason,
    write_paired_pack_freeze,
    fits_input_and_context,
)
from utils.teh.prompt_units import (
    OUTPUT_RESERVE,
    VLLM_CONTEXT,
    effective_hard_prompt_token_cap,
)
from utils.teh.llm_reasoning import (
    maybe_wrap_openai_client,
    normalize_llm_reasoning_effort,
)
from utils.teh.pics_v3 import (
    G2_PAIRED_PACK_FILENAME,
    G2_PAIRED_PACK_VERSION,
    HARD_PROMPT_TOKEN_CAP as PICS_V3_HARD_PROMPT_TOKEN_CAP,
    MAX_PARENT_CHARS as PICS_V3_MAX_PARENT_CHARS,
)
from utils.teh.dataset_prompt_evolution import (
    aggregate_pics_style_prompt_scores,
    build_mutation_user_message,
    maybe_run_dataset_prompt_evolution,
    wrap_prompt_with_contract,
)
from utils.teh.limited_data_protocol import (
    LIMITED_DATA_MANIFEST_CSV_FILENAME,
    LIMITED_DATA_MANIFEST_FILENAME,
    LIMITED_DATA_MANIFEST_JSONL_FILENAME,
    add_limited_data_cli_arguments,
    append_limited_data_manifest_jsonl,
    load_participant_limited_splits,
    rewrite_limited_data_csv_from_jsonl,
    should_persist_limited_data_manifest,
    write_limited_data_manifest,
)
from utils.teh.limited_data_registry import limited_data_protocol_revision, uses_training_only_sa40
from utils.teh.prompt_snapshots import (
    format_snapshot_examples,
    prompt_contract_scope,
    stamp_prompt_participant_id,
    using_v2_prompt_contract,
)
from utils.teh.prompt_units import select_structure_aware_prompt_examples
from utils.teh.sparse_observations import (
    SPARSE_AUDIT_CSV_FILENAME,
    SPARSE_AUDIT_FILENAME,
    SPARSE_AUDIT_JSONL_FILENAME,
    SparseObservationAudit,
    append_sparse_audit_jsonl,
    apply_max_observed_trials,
    distribute_explore_budget,
    initial_program_id_from_path,
    normalize_max_observed_trials,
    rewrite_sparse_audit_csv_from_jsonl,
    should_persist_sparse_audit,
    write_sparse_audit,
    write_sparse_audits_csv,
    write_sparse_audits_payload,
)
from utils.teh.explore_handoff import (
    prefix_elite_program_ids,
    resolve_explore_from_handoff_parents,
    select_explore_handoff_parents,
    split_explore_budget_seed_and_parents,
)
from utils.teh.explore_source_prompt import build_rank1_explore_prompt_suffix
from utils.teh.t_pics_sources import (
    DEFAULT_T_PICS_SOURCE_CONFIG,
    official_t_pics_source,
    has_t_pics_step1_source_pops,
    is_auto_t_pics_source_token,
    normalize_t_pics_dataset,
    resolve_t_pics_reuse_source,
    resolve_t_pics_source_participant_ids,
)
from utils.teh.t_pics_gated_transfer import (
    GATE_NAME,
    GATE_SCORE_FIELD,
    GATE_TIE_TOLERANCE,
    SOURCE_POPULATION_ITERS,
    apply_gated_cli_defaults,
    argv_source_conditioning_flags,
    build_control_argv,
    build_transfer_argv,
    decide_gate,
    default_seed_path,
    evaluate_pooled_train_val_loglik,
    gated_independent_enabled,
    gated_run_metadata,
    gate_record_payload,
    load_frozen_transfer_config,
    participant_run_is_complete,
    population_arm_is_complete,
    program_has_valid_choose,
    resolve_gated_transfer_source_rank1,
    run_layout,
    selected_source_for_target,
    make_explicit_independent_source_entry,
    selected_stage_is_complete,
    validate_frozen_transfer_config,
    write_gate_record,
)
from utils.teh.t_pics_gated_wandb import init_gated_wandb_reporter
from utils.teh.mdl_selection import (
    apply_mdl_to_scored_elite,
    attach_mdl_fields,
    elite_core,
    mdl_enabled,
    normalize_mdl_lambda,
    selection_trial_count,
    selection_used_val,
    sort_candidates,
    sort_elite_pairs,
    sort_elites,
    with_elite_mdl_score,
)
from utils.mem.trace import (
    append_mem_trace_record,
    best_reference_parent,
    build_candidate_record,
    build_iteration_context_record,
    compute_delta_f,
    mem_trace_path,
    parent_record_from_elite_tuple,
)

_REPO_ROOT = Path(__file__).resolve().parent
_PARTICIPANT_DATASETS = PARTICIPANT_DATASETS
BEST_PROGRAM_FILENAME = "best_program.py"

# Serializes writes to experiment-level CSVs (participant_details_loglik, summary_loglik, etc.)
# when --parallel_participants is enabled. Per-participant dirs are not locked (isolated paths).
_SHARED_EXPERIMENT_CSV_LOCK = threading.Lock()
# Serializes wandb.log for participant metrics (parallel workers share one run).
_WANDB_PARTICIPANT_LOG_LOCK = threading.Lock()
_RUN_PHASES = frozenset({"all", "evolution", "refine"})
_EARLY_STOP_MIN_IMPROVEMENT = 0.005
# TEMP: when True, refinement ignores --fresh_n_candidates and samples only from the evolution pool.
_DISABLE_REFINEMENT_FRESH_CANDIDATES = True
MAX_ERROR_MESSAGE_CHARS = 160
MAX_INVALID_LINE_CHARS = 160
ERROR_FEEDBACK_MODES = frozenset({"legacy", "new"})
DEFAULT_ERROR_FEEDBACK_MODE = "legacy"
LEGACY_RECENT_ERROR_WINDOW = 3
LEGACY_KEEP_TOP_FREQUENT = 3
LEGACY_MAX_ERROR_ITEMS = 8
MAX_DISTINCT_ERROR_GROUPS = 3
_ERROR_EVAL_SPLITS_ALLOWED = frozenset({"train", "val", "compile", "unknown"})


def _effective_psych_dataset_split(dataset: str, psych_dataset_split: str) -> str:
    """Psych-101 HF corpus selector; mixed_gambles ignores the argument."""
    if is_mixed_gambles_dataset(dataset):
        return DEFAULT_PSYCH_DATASET_SPLIT
    return normalize_psych_dataset_split(psych_dataset_split)


def _elite_pool_capacity(sample_size: int, elite_pool_size: Optional[int]) -> int:
    """Max programs retained in the elite pool (after sorting by fitness, best first)."""
    if elite_pool_size is None:
        return max(sample_size * 2, 20)
    return max(1, int(elite_pool_size))


def _cap_elite_preserving_program_ids(
    elite_parents: List[Tuple[Any, ...]],
    elite_val_logliks: List[Optional[float]],
    *,
    elite_cap: int,
    pinned_ids: Sequence[str],
    track_elite_val_loglik: bool,
    mdl_lambda: float = 0.0,
) -> None:
    """Sort/cap the elite pool but never drop ``pinned_ids`` (may exceed cap)."""
    pinned_set = {str(x) for x in pinned_ids}
    if track_elite_val_loglik:
        if len(elite_val_logliks) < len(elite_parents):
            elite_val_logliks.extend([None] * (len(elite_parents) - len(elite_val_logliks)))
        paired = list(zip(elite_parents, elite_val_logliks))
        pinned = [row for row in paired if str(row[0][3]) in pinned_set]
        others = [row for row in paired if str(row[0][3]) not in pinned_set]
        sort_elite_pairs(others, mdl_lambda)
        keep_others = max(0, int(elite_cap) - len(pinned))
        kept = pinned + others[:keep_others]
        sort_elite_pairs(kept, mdl_lambda)
        elite_parents[:] = [row[0] for row in kept]
        elite_val_logliks[:] = [row[1] for row in kept]
        return
    pinned = [p for p in elite_parents if str(p[3]) in pinned_set]
    others = [p for p in elite_parents if str(p[3]) not in pinned_set]
    sort_elites(others, mdl_lambda)
    keep_others = max(0, int(elite_cap) - len(pinned))
    kept = pinned + others[:keep_others]
    sort_elites(kept, mdl_lambda)
    elite_parents[:] = kept


def _write_elite_pool_dir(
    pool_dir: Path,
    elite_parents: List[Tuple[Any, ...]],
    *,
    extra_manifest_fields: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Path:
    """Write ``pool_manifest.json`` + ranked ``*.py`` files (MEM initial-pool layout)."""
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    for stale in pool_dir.glob("*.py"):
        stale.unlink()
    manifest_path = pool_dir / "pool_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    manifest: List[Dict[str, Any]] = []
    for rank, parent in enumerate(elite_parents):
        program_id = str(parent[3])
        train_ll = _train_loglik_from_elite_tuple(parent)
        safe_name = re.sub(r"[^\w.\-]+", "_", program_id) or "program"
        filename = f"{rank:03d}_{safe_name}.py"
        (pool_dir / filename).write_text(parent[0] or "", encoding="utf-8")
        entry: Dict[str, Any] = {
            "rank": rank,
            "program_id": program_id,
            "filename": filename,
            "global_train_loglik": train_ll,
            "global_fitness": _safe_float(parent[1]),
        }
        if extra_manifest_fields and rank < len(extra_manifest_fields) and extra_manifest_fields[rank]:
            entry.update(extra_manifest_fields[rank])
        manifest.append(entry)
    manifest_path.write_text(
        json.dumps({"n_programs": len(manifest), "programs": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    return pool_dir


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isfinite(v):
        return v
    return None


def _normalize_early_stop_iters(early_stop_iters: Optional[int]) -> Optional[int]:
    """Normalize CLI/function value: <=0 disables early stopping."""
    if early_stop_iters is None:
        return None
    return int(early_stop_iters) if int(early_stop_iters) > 0 else None


def _participant_metric_id(participant_id: Optional[int]) -> Dict[str, Any]:
    """Fields to include in per-iteration printed/logged metrics for one participant."""
    if participant_id is None:
        return {}
    return {"participant_id": int(participant_id)}


def _truncate_text_for_prompt(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_text(text: Any, max_len: int) -> str:
    if not text:
        return ""
    out = " ".join(str(text).strip().lower().split())
    return out[: max(0, int(max_len))]


def _normalize_error_message_for_grouping(text: Any, max_len: int) -> str:
    """Normalize exception text for grouping; strip noisy ids, keep meaningful values."""
    out = _normalize_text(text, max_len)
    if not out:
        return ""
    # Hex addresses / object ids vary per process and should not split groups.
    out = re.sub(r"0x[0-9a-fA-F]+", "0x?", out)
    out = re.sub(r"<([\w.]+) object at 0x\?>", r"<\1 object>", out)
    return out[: max(0, int(max_len))]


def _normalize_error_feedback_mode(error_feedback_mode: str) -> str:
    mode = str(error_feedback_mode or DEFAULT_ERROR_FEEDBACK_MODE).strip().lower()
    if mode not in ERROR_FEEDBACK_MODES:
        raise ValueError(
            f"error_feedback_mode must be one of {sorted(ERROR_FEEDBACK_MODES)}, got {error_feedback_mode!r}"
        )
    return mode


class _ErrorFeedbackStore(list):
    """In-memory error list carrying its run's selected storage semantics."""

    def __init__(self, error_feedback_mode: str):
        super().__init__()
        self.error_feedback_mode = _normalize_error_feedback_mode(error_feedback_mode)


def _error_feedback_mode_for_store(
    error_store: List[Dict[str, Any]],
    error_feedback_mode: Optional[str],
) -> str:
    if error_feedback_mode is not None:
        return _normalize_error_feedback_mode(error_feedback_mode)
    return _normalize_error_feedback_mode(
        getattr(error_store, "error_feedback_mode", "new")
    )


def _extract_relevant_invalid_source_line(code: str, exc: BaseException) -> str:
    lines = str(code or "").splitlines()
    if isinstance(exc, SyntaxError):
        if exc.text:
            return str(exc.text).strip()
        if exc.lineno and 1 <= int(exc.lineno) <= len(lines):
            return lines[int(exc.lineno) - 1].strip()
    try:
        tb_entries = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    except Exception:
        tb_entries = []
    for frame in reversed(tb_entries):
        if frame.filename == "<string>" and 1 <= int(frame.lineno) <= len(lines):
            return lines[int(frame.lineno) - 1].strip()
    for ln in lines:
        ln_s = ln.strip()
        if ln_s:
            return ln_s
    return ""


def _build_invalid_program_error_entry(
    *,
    code: str,
    exc: BaseException,
) -> Optional[Dict[str, str]]:
    if exc is None:
        return None
    line_raw = _extract_relevant_invalid_source_line(code, exc)
    line = _truncate_text_for_prompt(line_raw, MAX_INVALID_LINE_CHARS) if line_raw else ""
    error_type = _truncate_text_for_prompt(type(exc).__name__, 80)
    error_message = _truncate_text_for_prompt(str(exc), MAX_ERROR_MESSAGE_CHARS)
    norm_type = _normalize_text(error_type, 80)
    norm_msg = _normalize_error_message_for_grouping(error_message, MAX_ERROR_MESSAGE_CHARS)
    norm_line = _normalize_text(line, MAX_INVALID_LINE_CHARS)
    if norm_line:
        dedup_key = (norm_type, norm_msg, norm_line)
    else:
        dedup_key = (norm_type, norm_msg)
    return {
        "invalid_line": line,
        "error_type": error_type,
        "error_message": error_message,
        "normalized_key": "||".join(dedup_key),
    }


def _build_invalid_program_error_entry_legacy(
    *,
    code: str,
    exc: BaseException,
) -> Optional[Dict[str, str]]:
    """Exact pre-f169cb4 error-entry construction from commit 2daa9da."""
    if exc is None:
        return None
    line_raw = _extract_relevant_invalid_source_line(code, exc)
    line = _truncate_text_for_prompt(line_raw, MAX_INVALID_LINE_CHARS) if line_raw else ""
    error_type = _truncate_text_for_prompt(type(exc).__name__, 80)
    error_message = _truncate_text_for_prompt(str(exc), MAX_ERROR_MESSAGE_CHARS)
    norm_type = _normalize_text(error_type, 80)
    norm_msg = _normalize_text(error_message, MAX_ERROR_MESSAGE_CHARS)
    norm_line = _normalize_text(line, MAX_INVALID_LINE_CHARS)
    if norm_line:
        dedup_key = (norm_type, norm_msg, norm_line)
    else:
        dedup_key = (norm_type, norm_msg)
    return {
        "invalid_line": line,
        "error_type": error_type,
        "error_message": error_message,
        "normalized_key": "||".join(dedup_key),
    }


def _append_error_history_jsonl(
    history_path: Optional[Path],
    row: Dict[str, Any],
) -> None:
    if history_path is None:
        return
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalize_error_eval_split(eval_split: Optional[str]) -> Optional[str]:
    split = str(eval_split or "unknown").strip().lower() or "unknown"
    if split == "test":
        return None
    if split not in _ERROR_EVAL_SPLITS_ALLOWED:
        # Unknown non-test labels are kept as contextual train/val/compile-adjacent noise.
        return "unknown"
    return split


def _candidate_id_sort_key(candidate_id: Optional[str]) -> Tuple[int, str]:
    text = str(candidate_id or "")
    match = re.search(r"(\d+)$", text)
    if match:
        return (int(match.group(1)), text)
    return (10**9, text)


def _record_invalid_program_error_summary_new(
    error_store: List[Dict[str, Any]],
    error_entry: Optional[Dict[str, Any]],
    *,
    iteration: int,
    participant_id: Optional[int],
    candidate_id: Optional[str],
    history_path: Optional[Path],
    quality_score: Optional[float] = None,
    eval_split: str = "unknown",
    n_candidates_in_iteration: Optional[int] = None,
) -> None:
    """Append one train/val/compile error event. Test-split errors are ignored."""
    if not error_entry:
        return
    split = _normalize_error_eval_split(eval_split)
    if split is None:
        return
    normalized_key = str(error_entry.get("normalized_key") or "").strip()
    if not normalized_key:
        # Rebuild key if callers only passed display fields.
        err_type = _truncate_text_for_prompt(error_entry.get("error_type", ""), 80)
        err_msg = _truncate_text_for_prompt(
            error_entry.get("error_message", ""), MAX_ERROR_MESSAGE_CHARS
        )
        line = _truncate_text_for_prompt(
            error_entry.get("invalid_line", ""), MAX_INVALID_LINE_CHARS
        )
        norm_type = _normalize_text(err_type, 80)
        norm_msg = _normalize_error_message_for_grouping(err_msg, MAX_ERROR_MESSAGE_CHARS)
        norm_line = _normalize_text(line, MAX_INVALID_LINE_CHARS)
        if not norm_type and not norm_msg:
            return
        normalized_key = (
            "||".join((norm_type, norm_msg, norm_line))
            if norm_line
            else "||".join((norm_type, norm_msg))
        )
    invalid_line = _truncate_text_for_prompt(
        error_entry.get("invalid_line", ""), MAX_INVALID_LINE_CHARS
    )
    error_type = _truncate_text_for_prompt(error_entry.get("error_type", ""), 80)
    error_message = _truncate_text_for_prompt(
        error_entry.get("error_message", ""), MAX_ERROR_MESSAGE_CHARS
    )
    if not error_type and not error_message:
        return
    quality = _safe_float(quality_score)
    n_cand = (
        int(n_candidates_in_iteration)
        if n_candidates_in_iteration is not None and int(n_candidates_in_iteration) > 0
        else None
    )
    event = {
        "normalized_key": normalized_key,
        "invalid_line": invalid_line,
        "error_type": error_type,
        "error_message": error_message,
        "iteration": int(iteration),
        "candidate_id": candidate_id,
        "quality_score": quality,
        "eval_split": split,
        "n_candidates_in_iteration": n_cand,
    }
    error_store.append(event)
    _append_error_history_jsonl(
        history_path,
        {
            "iteration": int(iteration),
            "participant_id": int(participant_id) if participant_id is not None else None,
            "candidate_id": candidate_id,
            "invalid_line": invalid_line or None,
            "error_type": error_type,
            "error_message": error_message,
            "normalized_key": normalized_key,
            "quality_score": quality,
            "eval_split": split,
            "n_candidates_in_iteration": n_cand,
        },
    )


def _record_invalid_program_error_summary_legacy(
    error_store: List[Dict[str, Any]],
    error_entry: Optional[Dict[str, Any]],
    *,
    iteration: int,
    participant_id: Optional[int],
    candidate_id: Optional[str],
    history_path: Optional[Path],
) -> None:
    """Exact pre-f169cb4 cumulative store and JSONL format from commit 2daa9da."""
    if not error_entry:
        return
    normalized_key = str(error_entry.get("normalized_key") or "").strip()
    if not normalized_key:
        return
    invalid_line = _truncate_text_for_prompt(
        error_entry.get("invalid_line", ""), MAX_INVALID_LINE_CHARS
    )
    error_type = _truncate_text_for_prompt(error_entry.get("error_type", ""), 80)
    error_message = _truncate_text_for_prompt(
        error_entry.get("error_message", ""), MAX_ERROR_MESSAGE_CHARS
    )
    if not error_type and not error_message:
        return
    matched: Optional[Dict[str, Any]] = None
    for item in error_store:
        if item.get("normalized_key") == normalized_key:
            matched = item
            break
    if matched is None:
        matched = {
            "normalized_key": normalized_key,
            "invalid_line": invalid_line,
            "error_type": error_type,
            "error_message": error_message,
            "count": 0,
            "first_seen_iteration": int(iteration),
            "last_seen_iteration": int(iteration),
        }
        error_store.append(matched)
    else:
        if invalid_line:
            matched["invalid_line"] = invalid_line
        if error_type:
            matched["error_type"] = error_type
        if error_message:
            matched["error_message"] = error_message
        matched["last_seen_iteration"] = int(iteration)
    matched["count"] = int(matched.get("count", 0)) + 1
    _append_error_history_jsonl(
        history_path,
        {
            "iteration": int(iteration),
            "participant_id": int(participant_id) if participant_id is not None else None,
            "candidate_id": candidate_id,
            "invalid_line": invalid_line or None,
            "error_type": error_type,
            "error_message": error_message,
            "normalized_key": normalized_key,
            "count": int(matched["count"]),
            "last_seen_iteration": int(matched["last_seen_iteration"]),
        },
    )


def _legacy_error_entry_from_display_fields(
    error_entry: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Recreate the historical key for rich evaluator entries produced by current code."""
    if not error_entry:
        return None
    invalid_line = _truncate_text_for_prompt(
        error_entry.get("invalid_line", ""), MAX_INVALID_LINE_CHARS
    )
    error_type = _truncate_text_for_prompt(error_entry.get("error_type", ""), 80)
    error_message = _truncate_text_for_prompt(
        error_entry.get("error_message", ""), MAX_ERROR_MESSAGE_CHARS
    )
    norm_type = _normalize_text(error_type, 80)
    norm_msg = _normalize_text(error_message, MAX_ERROR_MESSAGE_CHARS)
    norm_line = _normalize_text(invalid_line, MAX_INVALID_LINE_CHARS)
    if not norm_type and not norm_msg:
        return None
    parts = (norm_type, norm_msg, norm_line) if norm_line else (norm_type, norm_msg)
    return {
        "invalid_line": invalid_line,
        "error_type": error_type,
        "error_message": error_message,
        "normalized_key": "||".join(parts),
    }


def _record_invalid_program_error_summary(
    error_store: List[Dict[str, Any]],
    error_entry: Optional[Dict[str, Any]],
    *,
    iteration: int,
    participant_id: Optional[int],
    candidate_id: Optional[str],
    history_path: Optional[Path],
    quality_score: Optional[float] = None,
    eval_split: str = "unknown",
    n_candidates_in_iteration: Optional[int] = None,
    error_feedback_mode: Optional[str] = None,
) -> None:
    mode = _error_feedback_mode_for_store(error_store, error_feedback_mode)
    if str(eval_split or "unknown").strip().lower() == "test":
        return
    if mode == "legacy":
        _record_invalid_program_error_summary_legacy(
            error_store,
            _legacy_error_entry_from_display_fields(error_entry),
            iteration=iteration,
            participant_id=participant_id,
            candidate_id=candidate_id,
            history_path=history_path,
        )
        return
    _record_invalid_program_error_summary_new(
        error_store,
        error_entry,
        iteration=iteration,
        participant_id=participant_id,
        candidate_id=candidate_id,
        history_path=history_path,
        quality_score=quality_score,
        eval_split=eval_split,
        n_candidates_in_iteration=n_candidates_in_iteration,
    )


def _record_invalid_program_error(
    error_store: List[Dict[str, Any]],
    *,
    code: str,
    exc: Optional[BaseException],
    iteration: int,
    participant_id: Optional[int],
    candidate_id: Optional[str],
    history_path: Optional[Path],
    quality_score: Optional[float] = None,
    eval_split: str = "unknown",
    n_candidates_in_iteration: Optional[int] = None,
    error_feedback_mode: Optional[str] = None,
) -> None:
    if exc is None:
        return
    mode = _error_feedback_mode_for_store(error_store, error_feedback_mode)
    entry = (
        _build_invalid_program_error_entry_legacy(code=code, exc=exc)
        if mode == "legacy"
        else _build_invalid_program_error_entry(code=code, exc=exc)
    )
    _record_invalid_program_error_summary(
        error_store,
        entry,
        iteration=iteration,
        participant_id=participant_id,
        candidate_id=candidate_id,
        history_path=history_path,
        quality_score=quality_score,
        eval_split=eval_split,
        n_candidates_in_iteration=n_candidates_in_iteration,
        error_feedback_mode=mode,
    )


def _group_errors_from_previous_iteration(
    error_store: List[Dict[str, Any]],
    *,
    iteration: Optional[int],
    previous_n_candidates: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Group train/val/compile errors from the immediately preceding iteration only."""
    if not error_store or iteration is None:
        return []
    prev_iter = int(iteration) - 1
    if prev_iter < 1:
        return []
    events = [
        item
        for item in error_store
        if int(item.get("iteration", -10**9)) == prev_iter
        and str(item.get("eval_split", "unknown")).strip().lower() != "test"
        and str(item.get("normalized_key") or "").strip()
    ]
    if not events:
        return []

    groups: Dict[str, Dict[str, Any]] = {}
    for item in events:
        key = str(item.get("normalized_key") or "").strip()
        group = groups.get(key)
        if group is None:
            group = {
                "normalized_key": key,
                "invalid_line": str(item.get("invalid_line") or ""),
                "error_type": str(item.get("error_type") or ""),
                "error_message": str(item.get("error_message") or ""),
                "candidate_ids": set(),
                "best_quality": None,
                "n_candidates_in_iteration": None,
            }
            groups[key] = group
        cand = item.get("candidate_id")
        if cand is not None and str(cand).strip():
            group["candidate_ids"].add(str(cand))
        q = _safe_float(item.get("quality_score"))
        best = _safe_float(group.get("best_quality"))
        if q is not None and (best is None or q > best):
            group["best_quality"] = q
            # Prefer display fields from the highest-quality failing candidate.
            if item.get("invalid_line"):
                group["invalid_line"] = str(item.get("invalid_line") or "")
            if item.get("error_type"):
                group["error_type"] = str(item.get("error_type") or "")
            if item.get("error_message"):
                group["error_message"] = str(item.get("error_message") or "")
        n_cand = item.get("n_candidates_in_iteration")
        if n_cand is not None:
            try:
                n_cand_i = int(n_cand)
            except (TypeError, ValueError):
                n_cand_i = 0
            if n_cand_i > 0:
                prev = group.get("n_candidates_in_iteration")
                group["n_candidates_in_iteration"] = (
                    max(int(prev), n_cand_i) if prev is not None else n_cand_i
                )

    denom_override = (
        int(previous_n_candidates)
        if previous_n_candidates is not None and int(previous_n_candidates) > 0
        else None
    )
    grouped: List[Dict[str, Any]] = []
    for group in groups.values():
        candidate_ids = sorted(group["candidate_ids"], key=_candidate_id_sort_key)
        n_hit = len(candidate_ids) if candidate_ids else 1
        denom = denom_override or group.get("n_candidates_in_iteration")
        if denom is None:
            # Infer from highest candidate index when budget was not recorded.
            max_idx = None
            for cid in candidate_ids:
                match = re.search(r"(\d+)$", cid)
                if match:
                    idx = int(match.group(1))
                    max_idx = idx if max_idx is None else max(max_idx, idx)
            if max_idx is not None:
                denom = max(n_hit, max_idx + 1)
        grouped.append(
            {
                "normalized_key": group["normalized_key"],
                "invalid_line": group["invalid_line"],
                "error_type": group["error_type"],
                "error_message": group["error_message"],
                "n_candidates": n_hit,
                "n_candidates_total": int(denom) if denom is not None else None,
                "best_quality": group["best_quality"],
                "candidate_ids": candidate_ids,
            }
        )

    def _rank_key(g: Dict[str, Any]) -> Tuple[float, int, str]:
        q = _safe_float(g.get("best_quality"))
        # Finite quality ranks above missing; higher (better) scores first.
        q_rank = q if q is not None else float("-inf")
        return (-q_rank, -int(g.get("n_candidates", 0)), str(g.get("normalized_key") or ""))

    grouped.sort(key=_rank_key)
    return grouped


def _select_errors_for_prompt_new(
    error_store: List[Dict[str, Any]],
    *,
    iteration: Optional[int],
    max_distinct_error_groups: int = MAX_DISTINCT_ERROR_GROUPS,
    previous_n_candidates: Optional[int] = None,
) -> List[Dict[str, Any]]:
    grouped = _group_errors_from_previous_iteration(
        error_store,
        iteration=iteration,
        previous_n_candidates=previous_n_candidates,
    )
    limit = max(0, int(max_distinct_error_groups))
    if limit <= 0:
        return []
    return grouped[:limit]


def _select_errors_for_prompt_legacy(
    error_store: List[Dict[str, Any]],
    *,
    iteration: Optional[int],
) -> List[Dict[str, Any]]:
    """Exact pre-f169cb4 selection and ordering from commit 2daa9da."""
    if not error_store:
        return []
    if iteration is None:
        recency_floor = -10**9
    else:
        recency_floor = int(iteration) - int(LEGACY_RECENT_ERROR_WINDOW) + 1
    recent = [
        item
        for item in error_store
        if int(item.get("last_seen_iteration", -10**9)) >= recency_floor
    ]
    recent.sort(
        key=lambda x: (
            -int(x.get("last_seen_iteration", -10**9)),
            -int(x.get("count", 0)),
        )
    )
    selected: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    for item in recent:
        key = str(item.get("normalized_key") or "")
        if key and key not in seen_keys:
            selected.append(item)
            seen_keys.add(key)
        if len(selected) >= int(LEGACY_MAX_ERROR_ITEMS):
            return selected
    frequent = sorted(
        error_store,
        key=lambda x: (
            -int(x.get("count", 0)),
            -int(x.get("last_seen_iteration", -10**9)),
        ),
    )
    keep_top = int(max(0, LEGACY_KEEP_TOP_FREQUENT))
    kept = 0
    for item in frequent:
        key = str(item.get("normalized_key") or "")
        if not key or key in seen_keys:
            continue
        selected.append(item)
        seen_keys.add(key)
        kept += 1
        if kept >= keep_top or len(selected) >= int(LEGACY_MAX_ERROR_ITEMS):
            break
    return selected[: int(LEGACY_MAX_ERROR_ITEMS)]


def _select_errors_for_prompt(
    error_store: List[Dict[str, Any]],
    *,
    iteration: Optional[int],
    max_distinct_error_groups: int = MAX_DISTINCT_ERROR_GROUPS,
    previous_n_candidates: Optional[int] = None,
    error_feedback_mode: str = "new",
) -> List[Dict[str, Any]]:
    mode = _normalize_error_feedback_mode(error_feedback_mode)
    if mode == "legacy":
        return _select_errors_for_prompt_legacy(error_store, iteration=iteration)
    return _select_errors_for_prompt_new(
        error_store,
        iteration=iteration,
        max_distinct_error_groups=max_distinct_error_groups,
        previous_n_candidates=previous_n_candidates,
    )


def _count_previous_iteration_error_groups(
    error_store: List[Dict[str, Any]],
    *,
    iteration: Optional[int],
    error_feedback_mode: str = "new",
) -> int:
    if _normalize_error_feedback_mode(error_feedback_mode) == "legacy":
        return len(error_store)
    return len(
        _group_errors_from_previous_iteration(error_store, iteration=iteration)
    )


def _build_past_error_prompt_section_new(
    error_store: List[Dict[str, Any]],
    *,
    iteration: Optional[int],
    max_error_prompt_chars: int,
    max_distinct_error_groups: int = MAX_DISTINCT_ERROR_GROUPS,
    previous_n_candidates: Optional[int] = None,
) -> str:
    if not error_store:
        return ""
    max_chars = int(max_error_prompt_chars)
    if max_chars <= 0:
        return ""
    selected = _select_errors_for_prompt(
        error_store,
        iteration=iteration,
        max_distinct_error_groups=max_distinct_error_groups,
        previous_n_candidates=previous_n_candidates,
    )
    if not selected:
        return ""
    header = (
        "Recent train/validation errors from the previous iteration "
        "(contextual only — not every candidate must repair the same issue):\n\n"
    )
    header_len = len(header)
    if header_len >= max_chars:
        return ""
    items: List[str] = []
    used = header_len
    for entry in selected:
        line = str(entry.get("invalid_line") or "").strip()
        err_type = str(entry.get("error_type") or "").strip()
        err_msg = str(entry.get("error_message") or "").strip()
        if not err_type and not err_msg:
            continue
        err = f"{err_type}: {err_msg}" if err_msg else err_type
        n_hit = int(entry.get("n_candidates", 0) or 0)
        n_tot = entry.get("n_candidates_total")
        if n_tot is not None and int(n_tot) > 0:
            frac = f"{n_hit}/{int(n_tot)} candidates"
        else:
            frac = f"{n_hit} candidates"
        if line:
            item = f"- [{frac}] Line: {line}\n  Error: {err}"
        else:
            item = f"- [{frac}] Error: {err}"
        sep = 2 if items else 0
        projected = used + len(item) + sep
        if projected > max_chars:
            if items:
                break
            # Keep a truncated first item when possible so the section is still useful.
            remaining = max_chars - used
            if remaining < 24:
                continue
            item = item[: remaining - 3].rstrip() + "..."
            projected = used + len(item)
            if projected > max_chars:
                continue
        items.append(item)
        used = projected
    if not items:
        return ""
    return header + "\n\n".join(items)


def _build_past_error_prompt_section_legacy(
    error_store: List[Dict[str, Any]],
    *,
    iteration: Optional[int],
    max_error_prompt_chars: int,
) -> str:
    """Exact pre-f169cb4 prompt wording and item packing from commit 2daa9da."""
    if not error_store:
        return ""
    max_chars = int(max_error_prompt_chars)
    if max_chars <= 0:
        return ""
    selected = _select_errors_for_prompt_legacy(error_store, iteration=iteration)
    if not selected:
        return ""
    items: List[str] = []
    used = 0
    for entry in selected:
        line = str(entry.get("invalid_line") or "").strip()
        err_type = str(entry.get("error_type") or "").strip()
        err_msg = str(entry.get("error_message") or "").strip()
        if not err_type and not err_msg:
            continue
        err = f"{err_type}: {err_msg}" if err_msg else err_type
        if line:
            item = f"- Line: {line}\n  Error: {err}"
        else:
            item = f"- Error: {err}"
        projected = used + len(item) + (2 if items else 0)
        if projected > max_chars and items:
            break
        if projected > max_chars:
            continue
        items.append(item)
        used = projected
    if not items:
        return ""
    past_error_summary = "\n\n".join(items)
    return (
        "Past invalid-program errors to avoid:\n"
        "The following are previous invalid generated-program mistakes. Do not repeat them. "
        "Each item shows the invalid line, when available, and the error it caused.\n\n"
        f"{past_error_summary}"
    )


def _build_past_error_prompt_section(
    error_store: List[Dict[str, Any]],
    *,
    iteration: Optional[int],
    max_error_prompt_chars: int,
    max_distinct_error_groups: int = MAX_DISTINCT_ERROR_GROUPS,
    previous_n_candidates: Optional[int] = None,
    error_feedback_mode: str = "new",
) -> str:
    mode = _normalize_error_feedback_mode(error_feedback_mode)
    if mode == "legacy":
        return _build_past_error_prompt_section_legacy(
            error_store,
            iteration=iteration,
            max_error_prompt_chars=max_error_prompt_chars,
        )
    return _build_past_error_prompt_section_new(
        error_store,
        iteration=iteration,
        max_error_prompt_chars=max_error_prompt_chars,
        max_distinct_error_groups=max_distinct_error_groups,
        previous_n_candidates=previous_n_candidates,
    )


def _write_iteration_error_prompt_file(
    iter_dir: Optional[Path],
    error_prompt_section: str,
) -> None:
    if iter_dir is None:
        return
    path = iter_dir / "error_prompt.txt"
    path.write_text(error_prompt_section or "", encoding="utf-8")


def _resolve_best_program_id_for_metrics(metrics: Dict[str, Any]) -> Optional[str]:
    for key in ("best_program_id", "iter_best_program_id", "pool_best_program_id"):
        prog_id = metrics.get(key)
        if prog_id is not None:
            return str(prog_id)
    return None


def _best_from_fresh_candidate(
    program_id: Optional[str],
    iteration_step: int,
    candidate_results: Optional[List[Dict[str, Any]]] = None,
    candidate_sources: Optional[List[str]] = None,
) -> Optional[str]:
    """candidate_<idx> if program_id is this iteration's best and that slot was fresh; else null."""
    if not program_id:
        return None
    match = re.match(
        r"(?:iteration|refinement|global_iteration)_(\d+)_candidate_(\d+)$",
        str(program_id),
    )
    if not match:
        return None
    if int(match.group(1)) != int(iteration_step):
        return None
    cand_idx = int(match.group(2))

    def _source_for_idx(idx: int) -> Optional[str]:
        if candidate_results:
            for row in candidate_results:
                if row.get("idx") == idx:
                    return row.get("source")
        if candidate_sources is not None and 0 <= idx < len(candidate_sources):
            return candidate_sources[idx]
        return None

    if _source_for_idx(cand_idx) == "fresh":
        return f"candidate_{cand_idx}"
    return None


def _parse_final_best_program_origin(program_id: str) -> Dict[str, Any]:
    """Map final program_id to origin_iteration, origin_candidate_idx, and origin_phase."""
    pid = str(program_id)
    match = re.match(r"^iteration_(\d+)_candidate_(\d+)$", pid)
    if match is not None:
        return {
            "origin_iteration": int(match.group(1)),
            "origin_candidate_idx": int(match.group(2)),
            "origin_phase": "evolution",
        }
    match = re.match(r"^global_iteration_(\d+)_candidate_(\d+)$", pid)
    if match is not None:
        return {
            "origin_iteration": int(match.group(1)),
            "origin_candidate_idx": int(match.group(2)),
            "origin_phase": "global",
        }
    match = re.match(r"^explore_candidate_(\d+)$", pid)
    if match is not None:
        return {
            "origin_iteration": None,
            "origin_candidate_idx": int(match.group(1)),
            "origin_phase": "explore",
        }
    if pid == "global_baseline":
        return {
            "origin_iteration": -1,
            "origin_candidate_idx": -1,
            "origin_phase": "global",
        }
    if pid == "baseline":
        return {
            "origin_iteration": -1,
            "origin_candidate_idx": -1,
            "origin_phase": "baseline",
        }
    return {
        "origin_iteration": None,
        "origin_candidate_idx": None,
        "origin_phase": None,
    }


def _resolve_final_best_from_elite_pool(
    elite_parents: List[Tuple[Any, ...]],
    seed_code: str,
) -> Tuple[str, str]:
    """Use elite pool rank-1 when it compiles; otherwise fall back to seed baseline."""
    if elite_parents:
        pool_code = elite_parents[0][0]
        pool_id = str(elite_parents[0][3])
        if pool_code and compile_program(pool_code) is not None:
            return pool_code, pool_id
    print(
        "[WARN] Elite pool empty or rank-1 failed to compile; "
        "using seed baseline for final reporting.",
        flush=True,
    )
    return seed_code, "baseline"


def _build_iteration_metrics_json(
    *,
    participant_id: Optional[int],
    header: Dict[str, Any],
    best: Dict[str, Any],
    candidate_results: Any,
) -> Dict[str, Any]:
    """iteration metrics.json: metadata + pool-best fields, then candidate_results last."""
    out: Dict[str, Any] = {}
    out.update(_participant_metric_id(participant_id))
    out.update(header)
    out.update(best)
    out["candidate_results"] = candidate_results
    iter_step = out.get("iteration")
    if iter_step is not None:
        out["best_from_fresh_candidate"] = _best_from_fresh_candidate(
            _resolve_best_program_id_for_metrics(out),
            int(iter_step),
            list(candidate_results) if candidate_results else None,
        )
    else:
        out["best_from_fresh_candidate"] = None
    return out


_WANDB_JSONL_PARTICIPANT_KEY_SUFFIXES = (
    "train_loglik",
    "val_loglik",
    "test_loglik",
    "gated_test_loglik",
    "selection_score",
    "train_val_loglik",
    "train_fitness",
    "test_fitness",
    "train_acc",
    "test_acc",
    "train_accuracy",
    "test_accuracy",
    "train_mse",
    "test_mse",
    "n_valid",
    "is_baseline",
    "avg_train_accuracy",
    "avg_test_accuracy",
    "avg_train_fitness",
    "avg_train_mse",
    "avg_test_mse",
)


def _wandb_jsonl_log_entry(
    *,
    step: int,
    iteration: int,
    log_dict: Dict[str, Any],
    participant_id: Optional[int] = None,
    agent_id: Optional[int] = None,
) -> Dict[str, Any]:
    """JSONL row: step/iteration first, then p{id}_* (or a/gw) metrics, then other keys."""
    entry: Dict[str, Any] = {"step": step, "iteration": iteration}
    entry.update(_participant_metric_id(participant_id))
    used: set[str] = set()

    def _add_prefixed_keys(prefix: str) -> None:
        for suffix in _WANDB_JSONL_PARTICIPANT_KEY_SUFFIXES:
            key = f"{prefix}{suffix}"
            if key in log_dict:
                entry[key] = log_dict[key]
                used.add(key)
        for key in sorted(log_dict):
            if key.startswith(prefix) and key not in used:
                entry[key] = log_dict[key]
                used.add(key)

    if participant_id is not None:
        pid = int(participant_id)
        _add_prefixed_keys(f"p{pid}_")
        slash_prefix = f"p{pid}/"
        for key in sorted(log_dict):
            if key.startswith(slash_prefix) and key not in used:
                entry[key] = log_dict[key]
                used.add(key)
    if agent_id is not None:
        _add_prefixed_keys(f"a{int(agent_id)}_")
    _add_prefixed_keys("gw_")

    for key in sorted(log_dict):
        if key not in used:
            entry[key] = log_dict[key]
    return entry


_WANDB_PARTICIPANT_CHART_SUFFIXES = (
    "train_loglik",
    "val_loglik",
    "test_loglik",
    "gated_test_loglik",
    "selection_score",
    "train_fitness",
    "test_fitness",
    "train_acc",
    "test_acc",
    "train_accuracy",
    "test_accuracy",
)


def _wandb_participant_chart_dict(
    log_dict: Dict[str, Any],
    participant_id: int,
    step: int,
) -> Dict[str, Any]:
    """
    Build W&B log payload with slash-grouped chart keys (p{pid}/metric), matching te_aggregate.py.

    Underscore keys in log_dict are kept for JSONL; wandb.log should use this dict only.
    """
    pid = int(participant_id)
    out: Dict[str, Any] = {f"p{pid}_step": int(step)}
    for suffix in _WANDB_PARTICIPANT_CHART_SUFFIXES:
        us_key = f"p{pid}_{suffix}"
        if us_key in log_dict and log_dict[us_key] is not None:
            out[f"p{pid}/{suffix}"] = log_dict[us_key]
    for aux in ("n_valid", "is_baseline"):
        aux_key = f"p{pid}_{aux}"
        if aux_key in log_dict:
            out[aux_key] = log_dict[aux_key]
    return out


def _wandb_log_participant_metrics(
    wandb_module: Any,
    log_dict: Dict[str, Any],
    participant_id: int,
    step: int,
) -> None:
    """
    Log participant metrics to W&B with te_aggregate-style slash chart grouping.

    Uses per-participant step via p{pid}_step (see wandb.define_metric); do not pass a global
    wandb.log(step=...) — parallel participants would race on the shared run step axis.

    PICS v3 (``KIND=pics_v3*`` / project ``teh_pics_v3``): do not upload dynamic
    ``p{pid}/*`` Runs-table scalars. Gated reporter remaps to fixed ``participant/*``;
    raw G.1 W&B skips the upload (local CSV/JSON remain authoritative).
    """
    from utils.teh.t_pics_gated_wandb import (
        suppress_dynamic_participant_wandb_scalars,
    )

    payload = _wandb_participant_chart_dict(log_dict, participant_id, step)
    # Gated reporter: always remap via .log (strips p{pid}/* internally).
    if hasattr(wandb_module, "_remap"):
        with _WANDB_PARTICIPANT_LOG_LOCK:
            wandb_module.log(payload)
        return
    if suppress_dynamic_participant_wandb_scalars():
        return
    with _WANDB_PARTICIPANT_LOG_LOCK:
        wandb_module.log(payload)


def _parallel_participant_pool_sizes(
    max_workers: int, n_candidates: int, parallel_participants: bool
) -> Tuple[int, int]:
    """Return (participant_workers, candidate_workers_per_participant)."""
    n_cand = max(1, int(n_candidates))
    if not parallel_participants:
        return 1, max(1, int(max_workers))
    return max(1, int(max_workers) // n_cand), n_cand


_LOGlik_VAL_SPLIT_DATASETS = LOGlik_VAL_SPLIT_DATASETS


def _uses_train_val_test_loglik_split(
    dataset: str, fitness_metric: str, *, cpc18_official_mse: bool = False
) -> bool:
    """True when evolution/refinement use disjoint train/val/test with log-likelihood."""
    return uses_train_val_test_loglik_split(
        dataset, fitness_metric, cpc18_official_mse=cpc18_official_mse
    )


def _supports_loglik_refinement(
    dataset: str,
    fitness_metric: str,
    val_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
    *,
    cpc18_official_mse: bool = False,
) -> bool:
    return _uses_train_val_test_loglik_split(
        dataset, fitness_metric, cpc18_official_mse=cpc18_official_mse
    ) and bool(val_trials and test_trials)


def _val_loglik_below_refinement_threshold(
    val_loglik: Optional[float], refinement_val_threshold: float
) -> bool:
    """True when refinement should run (val loglik strictly below threshold)."""
    if val_loglik is None:
        return False
    return float(val_loglik) < float(refinement_val_threshold)


def _choice13k_val_loglik_below_refinement_threshold(
    val_loglik: Optional[float], refinement_val_threshold: float
) -> bool:
    return _val_loglik_below_refinement_threshold(val_loglik, refinement_val_threshold)


def _gated_loglik_for_participant_summary(participant_summary: Dict[str, Any]) -> Any:
    """CSV/report gated column: refinement output when present, else evolution test loglik."""
    gated = participant_summary.get("gated_test_loglik")
    if gated is not None:
        return gated
    return participant_summary.get("test_loglik")


def _apply_test_loglik_as_gated_when_no_refinement(
    *,
    gated_test_loglik: Optional[float],
    overall_best_train: Dict[str, Any],
    overall_best_test: Dict[str, Any],
    dataset: str,
    fitness_metric: str,
    run_phase: str,
    refinement_phase: bool,
    val_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
    cpc18_official_mse: bool = False,
) -> Optional[float]:
    """
    When refinement did not set gated_test_loglik, copy evolution test loglik for reporting.

    Applies to choice13k / cpc18 loglik / mixed_gambles loglik (--refinement_phase on, --phase all).
    """
    if gated_test_loglik is not None:
        return gated_test_loglik
    if run_phase != "all" or not refinement_phase or fitness_metric != "loglik":
        return None
    if not _supports_loglik_refinement(
        dataset,
        fitness_metric,
        val_trials,
        test_trials,
        cpc18_official_mse=cpc18_official_mse,
    ):
        return None
    test_ll = _safe_float(overall_best_test.get("test_loglik"))
    if test_ll is None:
        return None
    overall_best_train["gated_test_loglik"] = test_ll
    overall_best_test["gated_test_loglik"] = test_ll
    return test_ll


def _evaluate_loglik_for_dataset(
    dataset: str,
    choose_fn: Callable,
    trials: List[Dict[str, Any]],
    *,
    verbose: bool = False,
    n_seeds: int = 1,
) -> Dict[str, float]:
    if is_categorical_output_dataset(dataset):
        from utils.teh_psych.categorical_eval import evaluate_categorical_program

        return evaluate_categorical_program(
            choose_fn, trials, verbose=verbose, n_seeds=n_seeds
        )
    return evaluate_choice13k_program(choose_fn, trials, verbose=verbose, n_seeds=n_seeds)


def _resolve_default_seed_program_path(args: Any, participant_id: int) -> Optional[str]:
    """Default seed path for TEH datasets (Bernoulli 0.5 or categorical uniform)."""
    if args.seed_path is not None:
        return args.seed_path
    dataset = getattr(args, "dataset", None)
    if dataset is not None and is_categorical_output_dataset(str(dataset)):
        return str(_REPO_ROOT / "persona_code_example" / "teh" / "categorical_uniform.py")
    return str(DEFAULT_SEED_PROGRAM)


def _parallel_generate_children(
    n_children: int,
    generate_one: Callable[..., str],
    *,
    max_workers: int = 5,
    desc: str = "Generating candidate programs",
) -> List[str]:
    """Run independent LLM child generations in parallel; preserve child index order."""
    if n_children <= 0:
        return []
    workers = max(1, min(int(max_workers), n_children))
    if workers == 1:
        return [generate_one(i) for i in tqdm(range(n_children), desc=desc)]

    results: List[Optional[str]] = [None] * n_children
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(generate_one, i): i for i in range(n_children)}
        with tqdm(total=n_children, desc=f"{desc} (workers={workers})") as pbar:
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                pbar.update(1)
    return [r if r is not None else "" for r in results]


def _phase_llm_decoding_seed_base(
    *,
    split_seed: int,
    iteration_step: int,
    participant_id: Optional[int] = None,
    batch_offset: int = 0,
) -> int:
    """Deterministic per-request seed base. Matched across G.2 arms (no arm id)."""
    pid_key = int(participant_id) if participant_id is not None else 0
    return (
        int(split_seed)
        + 80_000
        + int(iteration_step) * 1_000_003
        + pid_key * 17_179
        + int(batch_offset)
    )


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
    auto_prepare: bool = True,
) -> List[int]:
    """Load valid participant ids; auto-generate JSON when missing (if auto_prepare)."""
    if not is_binary_loglik_dataset(dataset):
        raise ValueError(f"load_valid_participant_ids_from_json: unsupported TEH dataset {dataset!r}")
    from utils.teh.participant_ids import load_valid_participant_ids

    return load_valid_participant_ids(
        dataset,
        repo_root,
        filter_mixed_gambles=filter_mixed_gambles,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=_effective_psych_dataset_split(dataset, psych_dataset_split),
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        auto_prepare=auto_prepare,
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
    """
    Build the list of raw participant ids to process for choice13k / cpc18 / mixed_gambles.

    - participant_scope=single: one raw id (--single_participant_id).
    - participant_scope=range: inclusive ordinal slice into valid_participant_ids.json.
    - participant_scope=ordinals: raw ids at listed 0-based ordinals (--ordinals), same ordering as range.
    - participant_scope=all: all raw ids from JSON, optionally capped by --all_max_participants (first N valid).
    """
    valid = load_valid_participant_ids_from_json(
        dataset,
        repo_root,
        filter_mixed_gambles,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
    if participant_scope == "single":
        if single_participant_id not in valid:
            raise ValueError(
                f"--single_participant_id={single_participant_id} is not in the precomputed valid list "
                f"({len(valid)} ids). Check datasets/psych101_<train|test>/<dataset>/valid_participant_ids.json."
            )
        return [single_participant_id]
    if participant_scope == "range":
        if range_start_ordinal is None or range_end_ordinal is None:
            raise ValueError(
                "--participant_scope range requires --range_start_ordinal and --range_end_ordinal (inclusive)."
            )
        if range_start_ordinal < 0 or range_end_ordinal >= len(valid) or range_start_ordinal > range_end_ordinal:
            raise ValueError(
                f"Invalid ordinal range [{range_start_ordinal}, {range_end_ordinal}] "
                f"for valid list of length {len(valid)} (0-based inclusive end)."
            )
        return valid[range_start_ordinal : range_end_ordinal + 1]
    if participant_scope == "ordinals":
        if not participant_ordinals:
            raise ValueError(
                "--participant_scope ordinals requires --ordinals with one or more integers "
                "(0-based indices into valid_participant_ids.json), e.g. --ordinals 0 4 9."
            )
        out: List[int] = []
        seen: set[int] = set()
        for o in participant_ordinals:
            oi = int(o)
            if oi < 0 or oi >= len(valid):
                raise ValueError(
                    f"Ordinal {oi} is out of range for valid list of length {len(valid)} "
                    f"(valid indices: 0..{len(valid) - 1})."
                )
            pid = int(valid[oi])
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out
    if participant_scope == "all":
        if all_max_participants is not None:
            n = max(0, int(all_max_participants))
            return valid[:n]
        return list(valid)
    raise ValueError(f"Unknown participant_scope: {participant_scope!r}")


def _psych101_experiment_trial_counts(exp: Experiment) -> Tuple[int, int]:
    n_blocks = len(exp.blocks)
    n_parsed = sum(len(b.trials) for b in exp.blocks)
    return n_blocks, n_parsed


def _print_selected_participants_trial_summary(
    dataset: str,
    participant_ids: List[int],
    *,
    repo_root: Path,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
) -> None:
    """Print parsed/split trial counts for the selected participant ids."""
    if not participant_ids:
        return

    if is_psych101_dataset(dataset):
        valid_list = load_valid_participant_ids_from_json(
            dataset,
            repo_root,
            filter_mixed_gambles=False,
            split_ratio=split_ratio,
            split_seed=split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            auto_prepare=False,
        )
        ordinal_by_pid = {int(pid): idx for idx, pid in enumerate(valid_list)}
        rows: List[Dict[str, Any]] = []
        for pid in participant_ids:
            pid_i = int(pid)
            try:
                exp = get_psych101_binary_experiment(
                    dataset,
                    pid_i,
                    split=psych_dataset_split,
                    local_dataset=local_dataset,
                )
                n_blocks, n_parsed = _psych101_experiment_trial_counts(exp)
                train_trials, val_trials, test_trials, _ = split_psych_experiment(
                    exp, split_ratio=split_ratio, split_seed=split_seed
                )
                rows.append(
                    {
                        "valid_ordinal": ordinal_by_pid.get(pid_i, ""),
                        "raw_id": pid_i,
                        "blocks": n_blocks,
                        "parsed_total": n_parsed,
                        "train": len(train_trials),
                        "val": len(val_trials),
                        "test": len(test_trials),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "valid_ordinal": ordinal_by_pid.get(pid_i, ""),
                        "raw_id": pid_i,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        print("")
        print(
            f"Selected participants trial summary ({len(participant_ids)} ids, "
            f"split_ratio={split_ratio:.3f}, split_seed={split_seed}):"
        )
        print(
            "  valid_ordinal | raw_id | blocks | parsed_total | train | val | test"
        )
        show_rows = rows if len(rows) <= 60 else rows[:30] + rows[-5:]
        for row in show_rows:
            if row.get("error"):
                print(
                    f"  {str(row['valid_ordinal']):>13} | {row['raw_id']:6d} | "
                    f"ERROR: {row['error']}"
                )
                continue
            print(
                f"  {str(row['valid_ordinal']):>13} | {row['raw_id']:6d} | "
                f"{row['blocks']:6d} | {row['parsed_total']:12d} | "
                f"{row['train']:5d} | {row['val']:3d} | {row['test']:4d}"
            )
        if len(rows) > 60:
            print(f"  ... ({len(rows) - 35} rows omitted) ...")

        ok_rows = [r for r in rows if not r.get("error")]
        if ok_rows:
            train_vals = [int(r["train"]) for r in ok_rows]
            parsed_vals = [int(r["parsed_total"]) for r in ok_rows]
            print(
                "  Aggregate: "
                f"train min/mean/max = {min(train_vals)}/{sum(train_vals)/len(train_vals):.2f}/{max(train_vals)}, "
                f"parsed_total min/mean/max = {min(parsed_vals)}/{sum(parsed_vals)/len(parsed_vals):.2f}/{max(parsed_vals)}"
            )
        print(
            "  Note: --range_start_ordinal/--range_end_ordinal index the valid list "
            "(not raw HF row numbers when some rows are excluded)."
        )
        return

    if is_mixed_gambles_dataset(dataset):
        rows = []
        for pid in participant_ids:
            train_trials, val_trials, test_trials, _ = load_mixed_gambles_trials(
                int(pid),
                csv_path=DEFAULT_CSV_PATH,
                filter_gain_loss_only=False,
                split_ratio=split_ratio,
                split_seed=split_seed,
            )
            n_parsed = len(train_trials) + len(val_trials) + len(test_trials)
            rows.append(
                {
                    "raw_id": int(pid),
                    "parsed_total": n_parsed,
                    "train": len(train_trials),
                    "val": len(val_trials),
                    "test": len(test_trials),
                }
            )
        print("")
        print(
            f"Selected participants trial summary ({len(participant_ids)} ids, "
            f"split_ratio={split_ratio:.3f}, split_seed={split_seed}):"
        )
        print("  raw_id | parsed_total | train | val | test")
        for row in rows[:60]:
            print(
                f"  {row['raw_id']:6d} | {row['parsed_total']:12d} | "
                f"{row['train']:5d} | {row['val']:3d} | {row['test']:4d}"
            )
        if len(rows) > 60:
            print(f"  ... ({len(rows) - 60} rows omitted) ...")

    elif is_external_dataset(dataset):
        rows = []
        for pid in participant_ids:
            train_trials, val_trials, test_trials, _ = load_external_loglik_trials(
                dataset,
                int(pid),
                data_dir=str(_REPO_ROOT / external_default_data_dir(dataset)),
                split_ratio=split_ratio,
                split_seed=split_seed,
            )
            n_parsed = len(train_trials) + len(val_trials) + len(test_trials)
            rows.append(
                {
                    "raw_id": int(pid),
                    "parsed_total": n_parsed,
                    "train": len(train_trials),
                    "val": len(val_trials),
                    "test": len(test_trials),
                }
            )
        print("")
        print(
            f"Selected participants trial summary ({len(participant_ids)} ids, "
            f"split_ratio={split_ratio:.3f}, split_seed={split_seed}):"
        )
        print("  raw_id | parsed_total | train | val | test")
        for row in rows[:60]:
            print(
                f"  {row['raw_id']:6d} | {row['parsed_total']:12d} | "
                f"{row['train']:5d} | {row['val']:3d} | {row['test']:4d}"
            )
        if len(rows) > 60:
            print(f"  ... ({len(rows) - 60} rows omitted) ...")


def _load_gridworld_stack() -> None:
    """Import JAX and gridworld modules (only when gridworld code paths run)."""
    global _GRIDWORLD_STACK_LOADED, jax, jnp, flax
    global get_all_problem_configs, make_dataloader, AutomaticityEnv, State
    if _GRIDWORLD_STACK_LOADED:
        return
    import jax as _jax
    import jax.numpy as _jnp
    import flax as _flax
    from plot_and_eval import get_all_problem_configs as _gapc
    from plot_and_eval import make_dataloader as _mdl
    from environment import AutomaticityEnv as _AE
    from environment import State as _S

    jax, jnp, flax = _jax, _jnp, _flax
    get_all_problem_configs, make_dataloader = _gapc, _mdl
    AutomaticityEnv, State = _AE, _S
    _GRIDWORLD_STACK_LOADED = True


def load_seed_program(seed_path: str) -> str:
    """Load the seed program from the specified path.
    Handles markdown code blocks by extracting Python code."""
    with open(seed_path, 'r') as f:
        content = f.read()

    sanitized = _sanitize_llm_python_candidate(content)
    return sanitized if sanitized else content


def _extract_fenced_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    blocks.extend(re.findall(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE))
    blocks.extend(re.findall(r"```(.*?)```", text, re.DOTALL))
    return [b.strip() for b in blocks if b and b.strip()]


def _passes_python_syntax(candidate: str) -> bool:
    try:
        compile(candidate, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False


def _sanitize_llm_python_candidate(
    text: str,
    required_markers: Optional[Tuple[str, ...]] = None,
) -> str:
    """Extract a single choose() implementation from LLM output (no imports/prose)."""
    from utils.teh.prompt_sanitize import sanitize_evolution_candidate_code

    markers = required_markers if required_markers is not None else ("def choose(",)
    return sanitize_evolution_candidate_code(text, required_markers=markers)


def _sanitize_llm_python_candidate_with_reason(
    text: str,
    required_markers: Optional[Tuple[str, ...]] = None,
) -> Tuple[str, str]:
    """Like ``_sanitize_llm_python_candidate`` but also returns a failure reason string."""
    from utils.teh.prompt_sanitize import (
        describe_sanitize_failure,
        sanitize_evolution_candidate_code,
    )

    markers = required_markers if required_markers is not None else ("def choose(",)
    cleaned = sanitize_evolution_candidate_code(text, required_markers=markers)
    reason = describe_sanitize_failure(text, required_markers=markers)
    if cleaned:
        reason = "ok"
    return cleaned, reason


def _save_prompt_debug_bundle(
    debug_dir: Path,
    *,
    phase: str,
    participant_id: Optional[int],
    iteration: Optional[int],
    prompt_text: str,
    trunc_diag: Dict[str, Any],
    captures: List[Dict[str, Any]],
    exit_after_save: bool = True,
) -> None:
    """Write full prompt + raw LLM replies when debugging empty/invalid candidates."""
    pid = participant_id if participant_id is not None else "unknown"
    iter_s = iteration if iteration is not None else "unknown"
    out = debug_dir / f"participant_{pid}" / phase / f"iteration_{iter_s}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (out / "truncation.json").write_text(
        json.dumps(trunc_diag, indent=2) + "\n", encoding="utf-8"
    )
    for cap in captures:
        idx = cap.get("candidate_index", 0)
        raw = cap.get("raw_content") or ""
        (out / f"raw_response_{idx}.txt").write_text(raw, encoding="utf-8")
        meta = {k: v for k, v in cap.items() if k != "raw_content"}
        (out / f"candidate_{idx}_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "phase": phase,
        "participant_id": participant_id,
        "iteration": iteration,
        "n_candidates": len(captures),
        "n_nonempty_raw": sum(1 for c in captures if (c.get("raw_content") or "").strip()),
        "n_sanitized_ok": sum(1 for c in captures if c.get("sanitize_reason") == "ok"),
        "sanitize_reasons": [c.get("sanitize_reason") for c in captures],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[prompt_debug] Wrote debug bundle to {out} "
        f"(raw_ok={summary['n_nonempty_raw']}/{summary['n_candidates']}, "
        f"sanitize_ok={summary['n_sanitized_ok']}/{summary['n_candidates']})"
    )
    if exit_after_save:
        raise SystemExit(
            f"prompt_debug exit: no runtime-valid candidates "
            f"(participant={pid}, phase={phase}, iteration={iter_s}). "
            f"Inspect {out}"
        )


def find_template_program_for_gridworld(num_blocks: int, num_walls: int, agent_id: int) -> Optional[str]:
    """
    Auto-detect template program for gridworld based on problem config and agent_id.
    
    Looks in persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}/
    for a program matching the agent_id.
    
    Args:
        num_blocks: Number of blocks in the problem
        num_walls: Number of walls in the problem
        agent_id: Agent type ID
        
    Returns:
        Path to template program if found, None otherwise
    """
    # Get hand-designed program name mapping
    hand_designed_dir = Path("generated_outputs/hand_designed")
    if not hand_designed_dir.exists():
        return None
    
    files = sorted([f for f in os.listdir(hand_designed_dir) if f.endswith('.txt')])
    if agent_id >= len(files):
        return None
    
    hand_designed_name = files[agent_id].replace('.txt', '')
    
    # Try to find program in the problem-specific folder
    problem_dir = Path(f"persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}")
    
    # Try patterns: hand_designed_name_agent{agent_id}.py or agent_{agent_id}.py
    possible_names = [
        f"{hand_designed_name}_agent{agent_id}.py",
        f"agent_{agent_id}.py",
    ]
    
    for name in possible_names:
        candidate_path = problem_dir / name
        if candidate_path.exists():
            return str(candidate_path)
    
    # If not found, return None
    return None


def _score_dataset_prompt_lightweight(
    *,
    rendered_prompt: str,
    dataset: str,
    client: OpenAI,
    model_name: str,
    seed_program: str,
    prompts_dir: Path,
    dev_participant_ids: Sequence[int],
    eval_candidates: int,
    split_ratio: float,
    split_seed: int,
    filter_mixed_gambles: bool,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    llm_max_tokens: int,
    max_workers: int,
) -> Dict[str, Any]:
    """Generate N programs from seed under a candidate prompt; score PICS-style.

    For each development participant: evaluate the seed and each generated candidate,
    keep the best by val loglik (including seed), then average those bests across
    participants. Do not mean over all generations.
    """
    score_dir = prompts_dir / "dataset_prompt_evolution" / "_score_scratch"
    score_dir.mkdir(parents=True, exist_ok=True)
    # Temporary prompts dir for generate_program_variants (needs infer + template).
    scratch = score_dir / "active"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    (scratch / "infer_single_choice.txt").write_text(rendered_prompt, encoding="utf-8")
    for name in ("refine.txt", "seed_program.py", "single_code_template.txt"):
        src = prompts_dir / name
        if src.is_file():
            shutil.copy2(src, scratch / name)
    # Template may live under repo defaults if not copied into run prompts.
    if not (scratch / "single_code_template.txt").is_file():
        default_tmpl = (
            _REPO_ROOT
            / "prompts"
            / "Template_evo"
            / "choice13k"
            / "non_strict"
            / "loglik"
            / "single_code_template.txt"
        )
        if default_tmpl.is_file():
            shutil.copy2(default_tmpl, scratch / "single_code_template.txt")

    per_participant_scores: List[List[Tuple[float, float]]] = []
    n_exec = 0
    n_total = 0
    error_signatures: List[str] = []
    strong_snippets: List[str] = []

    for pid in dev_participant_ids:
        train_trials, val_trials, _ = _trials_for_loglik_participant(
            dataset,
            int(pid),
            split_ratio=split_ratio,
            split_seed=split_seed,
            filter_mixed_gambles=filter_mixed_gambles,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
        )
        participant_scores: List[Tuple[float, float]] = []

        # Include seed (same as PICS: best candidate vs seed).
        n_total += 1
        seed_fn, seed_err = compile_program_with_error(seed_program or "")
        if seed_fn is None:
            error_signatures.append(
                f"seed_compile:{type(seed_err).__name__ if seed_err else 'unknown'}"
            )
        else:
            try:
                seed_train = _evaluate_loglik_for_dataset(
                    dataset, seed_fn, train_trials, n_seeds=1
                )
                seed_val = _evaluate_loglik_for_dataset(
                    dataset, seed_fn, val_trials, n_seeds=1
                )
                n_exec += 1
                participant_scores.append(
                    (
                        float(seed_train.get("avg_loglik", float("-inf"))),
                        float(seed_val.get("avg_loglik", float("-inf"))),
                    )
                )
            except Exception as exc:
                error_signatures.append(f"seed_eval:{type(exc).__name__}")

        codes = generate_program_variants(
            client=client,
            model_name=model_name,
            parent_programs=[seed_program],
            train_trials=train_trials,
            n_variants=max(1, int(eval_candidates)),
            max_tokens=llm_max_tokens,
            dataset=dataset,
            fitness_metric="loglik",
            run_prompts_dir=str(scratch),
            max_workers=max_workers,
            parent_train_accuracies=None,
            parent_overall_logliks=[None],
        )
        for code in codes:
            n_total += 1
            choose_fn, err = compile_program_with_error(code or "")
            if choose_fn is None:
                error_signatures.append(
                    f"compile:{type(err).__name__ if err else 'unknown'}"
                )
                continue
            try:
                train_eval = _evaluate_loglik_for_dataset(
                    dataset, choose_fn, train_trials, n_seeds=1
                )
                val_eval = _evaluate_loglik_for_dataset(
                    dataset, choose_fn, val_trials, n_seeds=1
                )
            except Exception as exc:
                error_signatures.append(f"eval:{type(exc).__name__}")
                continue
            n_exec += 1
            t_ll = float(train_eval.get("avg_loglik", float("-inf")))
            v_ll = float(val_eval.get("avg_loglik", float("-inf")))
            participant_scores.append((t_ll, v_ll))
            if v_ll > -1.0 and len(strong_snippets) < 3:
                strong_snippets.append((code or "")[:500])

        per_participant_scores.append(participant_scores)

    agg = aggregate_pics_style_prompt_scores(per_participant_scores)
    return {
        "mean_train_fitness": agg["mean_train_fitness"],
        "mean_val_fitness": agg["mean_val_fitness"],
        "executable_rate": (n_exec / n_total) if n_total else 0.0,
        "error_signatures": error_signatures[:20],
        "strong_snippets": strong_snippets,
    }


def _mutate_dataset_prompt_via_llm(
    client: OpenAI,
    model_name: str,
    parent_text: str,
    role: str,
    feedback: str,
    *,
    max_tokens: int = 2048,
    artifacts_dir: Optional[Path] = None,
) -> str:
    user_content = build_mutation_user_message(
        parent_prompt=wrap_prompt_with_contract(parent_text),
        role=role,
        feedback=feedback,
    )
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        safe_role = re.sub(r"[^a-zA-Z0-9_]+", "_", role)
        (artifacts_dir / f"mutate_{safe_role}_in.txt").write_text(
            user_content, encoding="utf-8"
        )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You revise dataset evolution instruction prompts. "
                    "Preserve frozen contract markers exactly; mutate only evolvable text."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    text = (resp.choices[0].message.content or "").strip()
    if artifacts_dir is not None:
        (artifacts_dir / f"mutate_{safe_role}_out.txt").write_text(text, encoding="utf-8")
    if not text:
        return wrap_prompt_with_contract(parent_text)
    return wrap_prompt_with_contract(text)


def compile_program_with_error(code_str: str) -> Tuple[Optional[Callable], Optional[BaseException]]:
    """Safely compile program code; return (choose_fn, compile_error)."""
    from utils.teh.sandbox_builtins import compile_choose_with_error

    return compile_choose_with_error(code_str)


def compile_program(code_str: str) -> Optional[Callable]:
    """Safely compile program code and return choose callable if present."""
    choose_fn, _ = compile_program_with_error(code_str)
    return choose_fn


def _problem_for_choose(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Observable choice-time problem: shared sanitizer used by prompts and evaluators."""
    from utils.teh.prompt_snapshots import sanitize_problem_for_choose

    return sanitize_problem_for_choose(trial.get("problem") or {})


_CHOICE13K_GATE_THRESHOLD = -0.45
_CHOICE13K_LOGLIK_CLAMP_EPS = 1e-9


def _parse_choice13k_choose_output(p_raw: Any) -> float:
    """Coerce choose() return value to a float probability in [0, 1] (before clamping)."""
    if isinstance(p_raw, bool) or (isinstance(p_raw, (int, np.integer)) and int(p_raw) in (0, 1)):
        return 1.0 if int(p_raw) == 1 else 0.0
    if isinstance(p_raw, float):
        p_use = p_raw
    else:
        raise TypeError(f"choose must return float or 0/1, got {type(p_raw)}")
    if not (0.0 <= p_use <= 1.0):
        raise ValueError(f"invalid probability: {p_use!r}")
    return p_use


def _clamp_choice13k_probability(p: float) -> float:
    return min(max(p, _CHOICE13K_LOGLIK_CLAMP_EPS), 1.0 - _CHOICE13K_LOGLIK_CLAMP_EPS)


def apply_consistency_gate_to_probability(
    raw_p: float,
    val_loglik: float,
    *,
    threshold: float = _CHOICE13K_GATE_THRESHOLD,
) -> float:
    """Blend clamped raw_p toward 0.5 using validation log-likelihood consistency."""
    clamped = _clamp_choice13k_probability(raw_p)
    if val_loglik < threshold:
        consistency = max(0.0, 1.0 - (threshold - val_loglik))
    else:
        consistency = 1.0
    return consistency * clamped + (1.0 - consistency) * 0.5


def apply_consistency_gate_to_categorical_probs(
    probs: Dict[int, float],
    val_loglik: float,
    *,
    threshold: float = _CHOICE13K_GATE_THRESHOLD,
) -> Dict[int, float]:
    """Blend categorical probs toward uniform 1/K (categorical analogue of blend-to-0.5)."""
    if not probs:
        return probs
    k = len(probs)
    uniform = 1.0 / float(k)
    if val_loglik < threshold:
        consistency = max(0.0, 1.0 - (threshold - val_loglik))
    else:
        consistency = 1.0
    return {
        int(aid): consistency * float(p) + (1.0 - consistency) * uniform
        for aid, p in probs.items()
    }


def wrap_choose_with_consistency_gate(
    choose_fn: Callable,
    val_loglik: float,
) -> Callable:
    """Return choose(problem, history) that applies the external consistency gate (Bernoulli)."""

    def gated_choose(problem: Any, history: Any) -> float:
        p_raw = choose_fn(problem, history)
        raw_p = _parse_choice13k_choose_output(p_raw)
        return apply_consistency_gate_to_probability(raw_p, val_loglik)

    return gated_choose


def wrap_choose_with_categorical_consistency_gate(
    choose_fn: Callable,
    val_loglik: float,
) -> Callable:
    """Return choose(problem, history) that applies categorical consistency gate."""
    from utils.teh_psych.categorical_eval import (
        coerce_choose_output,
        valid_action_ids_from_problem,
    )

    def gated_choose(problem: Any, history: Any) -> Dict[int, float]:
        valid_ids = valid_action_ids_from_problem(problem or {})
        raw = choose_fn(problem, history)
        probs, _ = coerce_choose_output(raw, valid_ids)
        return apply_consistency_gate_to_categorical_probs(probs, val_loglik)

    return gated_choose


def run_choice13k_gate_phase(
    choose_fn: Callable,
    val_loglik: float,
    test_trials: List[Dict[str, Any]],
    *,
    n_eval_seeds: int = 1,
    dataset: str = "",
) -> Optional[float]:
    """Evaluate test log-likelihood with external consistency gate (source unchanged)."""
    if not test_trials:
        return None
    try:
        if dataset and is_categorical_output_dataset(dataset):
            from utils.teh_psych.categorical_eval import evaluate_categorical_program

            gated_fn = wrap_choose_with_categorical_consistency_gate(
                choose_fn, float(val_loglik)
            )
            gated_eval = evaluate_categorical_program(
                gated_fn, test_trials, n_seeds=n_eval_seeds
            )
        else:
            gated_fn = wrap_choose_with_consistency_gate(choose_fn, float(val_loglik))
            gated_eval = evaluate_choice13k_program(
                gated_fn, test_trials, n_seeds=n_eval_seeds
            )
        return float(gated_eval["avg_loglik"])
    except Exception as e:
        print(f"Warning: gate phase failed: {e}")
        return None


def _refinement_train_val_ratios(split_ratio: float) -> Tuple[float, float]:
    """Train/val block fractions for within-participant three-way split (test is separate)."""
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")
    train_ratio = float(split_ratio)
    val_ratio = (1.0 - train_ratio) / 2.0
    return train_ratio, val_ratio


def _refinement_combined_fitness(
    train_loglik: float,
    val_loglik: float,
    *,
    split_ratio: float,
) -> float:
    """Refinement selection fitness: size-weighted mean of train and validation log-likelihood."""
    train_ratio, val_ratio = _refinement_train_val_ratios(split_ratio)
    denom = train_ratio + val_ratio
    return (train_ratio * float(train_loglik) + val_ratio * float(val_loglik)) / denom


_EVOLUTION_SELECTION_SCORES = frozenset({"train", "train_val"})
_EVOLUTION_SELECTION_FALLBACK_WARNED: Set[str] = set()


def _normalize_evolution_selection_score(mode: str) -> str:
    normalized = str(mode).strip()
    if normalized not in _EVOLUTION_SELECTION_SCORES:
        raise ValueError(
            f"evolution_selection_score must be one of {sorted(_EVOLUTION_SELECTION_SCORES)}, "
            f"got {mode!r}"
        )
    return normalized


def _uses_train_val_evolution_selection(
    evolution_selection_score: str,
    fitness_metric: str,
) -> bool:
    return (
        _normalize_evolution_selection_score(evolution_selection_score) == "train_val"
        and fitness_metric == "loglik"
    )


def _evolution_selection_score(
    train_loglik: float,
    val_loglik: Optional[float],
    n_train: int,
    n_val: int,
    *,
    evolution_selection_score: str = "train_val",
    warn_key: Optional[str] = None,
) -> float:
    """
    Pool ranking score for evolution/explore/global phases.

    train: train_loglik only.
    train_val: trial-count-weighted mean of train+val loglik; falls back to train_loglik
    when val is missing or empty.
    """
    mode = _normalize_evolution_selection_score(evolution_selection_score)
    train_ll = float(train_loglik)
    if mode == "train":
        return train_ll
    val_ll = _safe_float(val_loglik)
    if val_ll is None or int(n_val) <= 0:
        if warn_key is not None and warn_key not in _EVOLUTION_SELECTION_FALLBACK_WARNED:
            _EVOLUTION_SELECTION_FALLBACK_WARNED.add(warn_key)
            print(
                "!Warning: val_loglik missing or val set empty; "
                "falling back to train_loglik for evolution selection score.",
                flush=True,
            )
        return train_ll
    n_tr = max(0, int(n_train))
    n_vl = max(0, int(n_val))
    denom = n_tr + n_vl
    if denom <= 0:
        return train_ll
    return (n_tr * train_ll + n_vl * float(val_ll)) / denom


def _apply_evolution_candidate_selection_fitness(
    *,
    train_loglik: float,
    val_loglik: Optional[float],
    train_acc: float,
    fitness_metric: str,
    n_train: int,
    n_val: int,
    evolution_selection_score: str,
    use_train_val_selection: bool,
    warn_key: str,
    runtime_valid: bool,
) -> Tuple[float, Optional[float]]:
    """Compute pool-ranking fitness and optional selection_score for one candidate."""
    fitness = train_loglik if fitness_metric == "loglik" else train_acc
    selection_score: Optional[float] = None
    if fitness_metric == "loglik":
        selection_score = _evolution_selection_score(
            train_loglik,
            val_loglik,
            n_train,
            n_val,
            evolution_selection_score=evolution_selection_score,
            warn_key=warn_key if use_train_val_selection else None,
        )
        if use_train_val_selection:
            fitness = selection_score
    if not runtime_valid:
        fitness = -1e9 if fitness_metric == "loglik" else float("-inf")
    return fitness, selection_score


def _mdl_n_for_score(
    evolution_selection_score: str,
    n_train: int,
    n_val: int,
    val_loglik: Optional[float],
) -> int:
    return selection_trial_count(
        evolution_selection_score,
        n_train,
        n_val,
        val_used=selection_used_val(evolution_selection_score, n_val, val_loglik),
    )


def _attach_mdl_to_candidate(
    result: Dict[str, Any],
    *,
    mdl_lambda: float,
    n: int,
    selection_score: Optional[float],
) -> None:
    if selection_score is None:
        return
    attach_mdl_fields(
        result,
        code=str(result.get("code") or ""),
        selection_score=float(selection_score),
        n=int(n),
        mdl_lambda=float(mdl_lambda),
        runtime_valid=bool(result.get("runtime_valid", True)),
    )


def _elite_tuple_for_ranking(
    code: Any,
    fitness: float,
    test_acc: Any,
    program_id: str,
    idx6: Any,
    *,
    mdl_lambda: float,
    mdl_score: Optional[float] = None,
) -> Tuple[Any, ...]:
    parent = (code, fitness, test_acc, program_id, None, None, idx6)
    return with_elite_mdl_score(parent, mdl_score, mdl_lambda)


_MDL_LOG_KEYS = (
    "mdl_n",
    "mdl_total_loglik",
    "program_ast_size",
    "mdl_complexity_penalty",
    "mdl_score",
)


def _candidate_mdl_log_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("mdl_score") is None:
        return {}
    return {k: result[k] for k in _MDL_LOG_KEYS if k in result}


def _elite_mdl_manifest_fields(
    parent: Sequence[Any],
    *,
    mdl_lambda: float,
    n: int,
    selection_score: Optional[float] = None,
) -> Dict[str, Any]:
    if not mdl_enabled(mdl_lambda):
        return {}
    sel = float(selection_score) if selection_score is not None else float(parent[1])
    row: Dict[str, Any] = {}
    attach_mdl_fields(
        row,
        code=str(parent[0] or ""),
        selection_score=sel,
        n=int(n),
        mdl_lambda=float(mdl_lambda),
        runtime_valid=True,
    )
    return _candidate_mdl_log_fields(row)


def _refinement_mdl_n(n_train: int, n_val: int) -> int:
    n_tr = max(0, int(n_train))
    n_vl = max(0, int(n_val))
    if n_vl > 0:
        return n_tr + n_vl
    return n_tr


def _apply_mdl_to_refinement_pool(
    elite_parents: List[Tuple[Any, ...]],
    *,
    mdl_lambda: float,
    n_train: int,
    n_val: int,
) -> List[Tuple[Any, ...]]:
    if not mdl_enabled(mdl_lambda):
        return [tuple(p[:7]) for p in elite_parents]
    n = _refinement_mdl_n(n_train, n_val)
    return [
        apply_mdl_to_scored_elite(
            parent,
            selection_score=float(parent[1]),
            n=n,
            mdl_lambda=mdl_lambda,
            runtime_valid=True,
        )
        for parent in elite_parents
    ]


def _train_loglik_from_elite_tuple(
    parent_tuple: Tuple[Any, ...],
    *,
    evolution_selection_score: str = "train",
) -> float:
    """Train loglik from evolution or refinement elite tuple."""
    program_id = str(parent_tuple[3]) if len(parent_tuple) > 3 else ""
    if program_id.startswith("refinement_"):
        if len(parent_tuple) > 6 and parent_tuple[6] is not None:
            return float(parent_tuple[6])
    if (
        _uses_train_val_evolution_selection(evolution_selection_score, "loglik")
        and len(parent_tuple) > 6
        and parent_tuple[6] is not None
    ):
        return float(parent_tuple[6])
    # Evolution pool (train mode): index 1 is train loglik; index 6 is train accuracy.
    return float(parent_tuple[1])


def _evolution_elite_to_refinement_pool(
    elite_parents: List[Tuple[Any, ...]],
    elite_val_logliks: List[Optional[float]],
    *,
    split_ratio: float,
    evolution_selection_score: str = "train",
    mdl_lambda: float = 0.0,
    n_train: int = 0,
    n_val: int = 0,
) -> Tuple[List[Tuple[Any, ...]], List[Optional[float]]]:
    """Copy evolution elite pool into refinement format; preserve evolution order (no sort)."""
    refine_parents: List[Tuple[Any, ...]] = []
    refine_vals: List[Optional[float]] = []
    for parent, val_ll in zip(elite_parents, elite_val_logliks):
        program_id = str(parent[3])
        train_ll = _train_loglik_from_elite_tuple(
            parent, evolution_selection_score=evolution_selection_score
        )
        val_ll_f = _safe_float(val_ll)
        combined = (
            _refinement_combined_fitness(train_ll, val_ll_f, split_ratio=split_ratio)
            if val_ll_f is not None
            else train_ll
        )
        refine_parents.append(
            (parent[0], combined, None, program_id, None, None, train_ll)
        )
        refine_vals.append(val_ll_f)
    refine_parents = _apply_mdl_to_refinement_pool(
        refine_parents,
        mdl_lambda=mdl_lambda,
        n_train=n_train,
        n_val=n_val,
    )
    return refine_parents, refine_vals


def _save_evolution_elite_pool(
    output_path: Path,
    elite_parents: List[Tuple[Any, ...]],
    elite_val_logliks: List[Optional[float]],
    *,
    split_ratio: float,
    n_train: int,
    n_val: int,
    evolution_selection_score: str = "train_val",
    mdl_lambda: float = 0.0,
) -> Path:
    """Persist evolution-phase elite pool programs and manifest (evolution sort order)."""
    pool_dir = output_path / "evolution_elite_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    mode = _normalize_evolution_selection_score(evolution_selection_score)
    mdl_on = mdl_enabled(mdl_lambda)
    for rank, (parent, val_ll) in enumerate(zip(elite_parents, elite_val_logliks)):
        program_id = str(parent[3])
        train_ll = _train_loglik_from_elite_tuple(
            parent, evolution_selection_score=evolution_selection_score
        )
        val_ll_f = _safe_float(val_ll)
        selection_score = _evolution_selection_score(
            train_ll,
            val_ll_f,
            n_train,
            n_val,
            evolution_selection_score=evolution_selection_score,
            warn_key=None,
        )
        combined = (
            _refinement_combined_fitness(train_ll, val_ll_f, split_ratio=split_ratio)
            if val_ll_f is not None
            else train_ll
        )
        safe_name = re.sub(r"[^\w.\-]+", "_", program_id) or "program"
        filename = f"{rank:03d}_{safe_name}.py"
        (pool_dir / filename).write_text(parent[0] or "", encoding="utf-8")
        entry: Dict[str, Any] = {
            "rank": rank,
            "program_id": program_id,
            "filename": filename,
            "train_loglik": train_ll,
            "val_loglik": val_ll_f,
            "selection_score": selection_score,
            "train_val_loglik": combined,
            "evolution_fitness": _safe_float(parent[1]),
            "evolution_selection_score": mode,
        }
        if mdl_on:
            n_sel = _mdl_n_for_score(
                evolution_selection_score, n_train, n_val, val_ll_f
            )
            entry.update(
                _elite_mdl_manifest_fields(
                    parent,
                    mdl_lambda=mdl_lambda,
                    n=n_sel,
                    selection_score=selection_score,
                )
            )
        manifest.append(entry)
    payload: Dict[str, Any] = {
        "n_programs": len(manifest),
        "evolution_selection_score": mode,
        "programs": manifest,
    }
    if mdl_on:
        payload["mdl_lambda"] = float(mdl_lambda)
    (pool_dir / "pool_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return pool_dir


def _load_evolution_elite_pool(
    pool_dir: Path,
    *,
    split_ratio: float,
    evolution_selection_score: str = "train_val",
) -> Tuple[List[Tuple[Any, ...]], List[Optional[float]]]:
    """Load evolution elite pool from saved manifest + program files."""
    manifest_path = pool_dir / "pool_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing evolution elite pool manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    programs = payload.get("programs", payload if isinstance(payload, list) else [])
    pool_mode = str(
        payload.get("evolution_selection_score", evolution_selection_score)
    )
    elite_parents: List[Tuple[Any, ...]] = []
    elite_val_logliks: List[Optional[float]] = []
    for entry in programs:
        filename = entry["filename"]
        code = (pool_dir / filename).read_text(encoding="utf-8")
        train_ll = float(entry["train_loglik"])
        val_ll = _safe_float(entry.get("val_loglik"))
        if entry.get("selection_score") is not None:
            combined = float(entry["selection_score"])
        elif val_ll is not None:
            combined = _refinement_combined_fitness(
                train_ll, val_ll, split_ratio=split_ratio
            )
        else:
            combined = train_ll
        program_id = str(entry.get("program_id", filename))
        idx6 = train_ll if _uses_train_val_evolution_selection(pool_mode, "loglik") else None
        parent: Tuple[Any, ...] = (
            code,
            combined,
            None,
            program_id,
            None,
            None,
            idx6 if idx6 is not None else train_ll,
        )
        mdl_score = entry.get("mdl_score")
        if mdl_score is not None:
            parent = parent + (float(mdl_score),)
        elite_parents.append(parent)
        elite_val_logliks.append(val_ll)
    return elite_parents, elite_val_logliks


def _collect_pooled_split_trials_for_participants(
    dataset: str,
    participant_ids: List[int],
    *,
    split: str,
    split_ratio: float,
    split_seed: int,
    data_path: str = "data",
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    speekenbrink_split: str = "chronological",
    audits_out: Optional[List[SparseObservationAudit]] = None,
) -> List[Dict[str, Any]]:
    """Concatenate per-participant train or val splits (same splits as evolution uses)."""
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    split_idx = 0 if split == "train" else 1
    pooled: List[Dict[str, Any]] = []
    for pid in participant_ids:
        train_trials, val_trials, _, audit = _trials_for_loglik_participant(
            dataset,
            int(pid),
            split_ratio=split_ratio,
            split_seed=split_seed,
            data_path=data_path,
            filter_mixed_gambles=filter_mixed_gambles,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            max_observed_trials_per_participant=max_observed_trials_per_participant,
            limited_data_protocol=limited_data_protocol,
            limited_train_val=limited_train_val,
            speekenbrink_split=speekenbrink_split,
            return_audit=True,
        )
        pooled.extend(
            stamp_prompt_participant_id(t, int(pid))
            for t in (train_trials, val_trials)[split_idx]
        )
        if audits_out is not None:
            audits_out.append(audit)
    return pooled


def _collect_pooled_train_trials_for_participants(
    dataset: str,
    participant_ids: List[int],
    *,
    split_ratio: float,
    split_seed: int,
    data_path: str = "data",
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    speekenbrink_split: str = "chronological",
    audits_out: Optional[List[SparseObservationAudit]] = None,
) -> List[Dict[str, Any]]:
    """Concatenate per-participant train splits (same splits as evolution uses)."""
    return _collect_pooled_split_trials_for_participants(
        dataset,
        participant_ids,
        split="train",
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        speekenbrink_split=speekenbrink_split,
        audits_out=audits_out,
    )


def _save_global_elite_pool(
    global_dir: Path,
    elite_parents: List[Tuple[Any, ...]],
    *,
    mdl_lambda: float = 0.0,
    n: Optional[int] = None,
) -> Path:
    """Persist global-phase elite pool (global sort order)."""
    extra = None
    if mdl_enabled(mdl_lambda) and n is not None:
        extra = [
            _elite_mdl_manifest_fields(parent, mdl_lambda=mdl_lambda, n=int(n)) or None
            for parent in elite_parents
        ]
    pool_dir = _write_elite_pool_dir(
        Path(global_dir) / "global_elite_pool",
        elite_parents,
        extra_manifest_fields=extra,
    )
    if elite_parents:
        (Path(global_dir) / BEST_PROGRAM_FILENAME).write_text(
            elite_parents[0][0] or "", encoding="utf-8"
        )
    return pool_dir


def _load_global_elite_pool(pool_dir: Path) -> List[Tuple[Any, ...]]:
    """Load global elite pool from saved manifest + program files."""
    manifest_path = pool_dir / "pool_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing global elite pool manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    programs = payload.get("programs", payload if isinstance(payload, list) else [])
    elite_parents: List[Tuple[Any, ...]] = []
    for entry in programs:
        filename = entry["filename"]
        code = (pool_dir / filename).read_text(encoding="utf-8")
        train_ll = float(entry.get("global_train_loglik", entry.get("global_fitness", 0.0)))
        program_id = str(entry.get("program_id", filename))
        elite_parents.append(
            (code, train_ll, None, program_id, None, None, train_ll)
        )
    return elite_parents


def _load_initial_pool_program_files(paths: Sequence[str]) -> List[Tuple[Any, ...]]:
    """Load one or more ``choose()`` programs as global-elite tuples."""
    elite_parents: List[Tuple[Any, ...]] = []
    used_ids: List[str] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (_REPO_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Initial pool program not found: {path}")
        code = path.read_text(encoding="utf-8")
        if compile_program(code) is None:
            raise ValueError(f"Failed to compile initial pool program: {path}")
        program_id = initial_program_id_from_path(path, used_ids)
        used_ids.append(program_id)
        elite_parents.append((code, 0.0, None, program_id, None, None, 0.0))
    if not elite_parents:
        raise ValueError("--initial_pool_programs did not load any programs.")
    return elite_parents


def _resolve_global_elite_pool_dir(path: Path) -> Path:
    """Accept a pool dir, global_phase dir, or run dir that contains pool_manifest.json."""
    candidates = (
        path,
        path / "global_elite_pool",
        path / "global_phase" / "global_elite_pool",
    )
    for cand in candidates:
        if (cand / "pool_manifest.json").is_file():
            return cand
    return path


def _load_one_initial_pool_dir(pool_dir: str) -> List[Tuple[Any, ...]]:
    path = Path(pool_dir).expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    if path.is_file() and path.name.endswith(".py"):
        return _load_initial_pool_program_files([str(path)])
    path = _resolve_global_elite_pool_dir(path)
    loaded = _load_global_elite_pool(path)
    if not loaded:
        raise ValueError(f"Empty initial pool directory: {path}")
    for parent in loaded:
        if compile_program(parent[0] or "") is None:
            raise ValueError(
                f"Failed to compile initial pool program {parent[3]!r} from {path}"
            )
    return loaded


def _load_initial_pool_from_cli(
    *,
    program_paths: Optional[Sequence[str]],
    pool_dirs: Optional[Sequence[str]],
) -> List[Tuple[Any, ...]]:
    """Load ``--initial_pool_programs`` and/or one-or-more ``--initial_pool_dir``."""
    elite_parents: List[Tuple[Any, ...]] = []
    if program_paths:
        elite_parents.extend(_load_initial_pool_program_files(program_paths))
    dirs = [d for d in (pool_dirs or []) if d]
    multi = len(dirs) > 1
    for idx, pool_dir in enumerate(dirs):
        loaded = _load_one_initial_pool_dir(pool_dir)
        if multi:
            loaded = prefix_elite_program_ids(loaded, f"pool{idx}_")
        elite_parents.extend(loaded)
    if not elite_parents:
        raise ValueError(
            "No initial-pool programs loaded. Provide --initial_pool_programs and/or --initial_pool_dir."
        )
    return elite_parents


def _global_elite_to_participant_elite(
    global_parents: List[Tuple[Any, ...]],
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    *,
    dataset: str,
    n_eval_seeds: int,
    evolution_selection_score: str = "train",
    selection_warn_key: Optional[str] = None,
    mdl_lambda: float = 0.0,
) -> Tuple[List[Tuple[Any, ...]], List[Optional[float]]]:
    """Map global pool into participant elite tuples; preserve global order (no sort)."""
    elite_parents: List[Tuple[Any, ...]] = []
    elite_val_logliks: List[Optional[float]] = []
    use_train_val = _uses_train_val_evolution_selection(evolution_selection_score, "loglik")
    n_train = len(train_trials)
    n_val = len(val_trials)
    warn_key = selection_warn_key or f"global_handoff_{dataset}"
    mdl_lambda = normalize_mdl_lambda(mdl_lambda)
    for parent in global_parents:
        code = parent[0]
        src_id = str(parent[3])
        program_id = src_id if src_id.startswith("global_") else f"global_{src_id}"
        train_ll = _train_loglik_from_elite_tuple(parent)
        val_ll: Optional[float] = None
        test_acc = 0.0
        train_acc = 0.0
        runtime_valid = False
        choose_fn = compile_program(code)
        if choose_fn is not None:
            train_eval = _evaluate_loglik_for_dataset(
                dataset, choose_fn, train_trials, n_seeds=n_eval_seeds
            )
            train_ll = float(train_eval["avg_loglik"])
            train_acc = float(train_eval["accuracy"])
            test_acc = train_acc
            runtime_valid = int(train_eval.get("errors", 0) or 0) == 0
            if val_trials:
                val_eval = _evaluate_loglik_for_dataset(
                    dataset, choose_fn, val_trials, n_seeds=n_eval_seeds
                )
                val_ll = float(val_eval["avg_loglik"])
                runtime_valid = runtime_valid and int(val_eval.get("errors", 0) or 0) == 0
        pool_fitness = (
            _evolution_selection_score(
                train_ll,
                val_ll,
                n_train,
                n_val,
                evolution_selection_score=evolution_selection_score,
                warn_key=warn_key if use_train_val else None,
            )
            if use_train_val
            else train_ll
        )
        idx6 = train_ll if use_train_val else train_acc
        n_sel = _mdl_n_for_score(evolution_selection_score, n_train, n_val, val_ll)
        elite_parents.append(
            apply_mdl_to_scored_elite(
                (code, pool_fitness, test_acc, program_id, None, None, idx6),
                selection_score=float(pool_fitness),
                n=n_sel,
                mdl_lambda=mdl_lambda,
                runtime_valid=runtime_valid,
            )
        )
        elite_val_logliks.append(val_ll)
    if mdl_enabled(mdl_lambda):
        paired = list(zip(elite_parents, elite_val_logliks))
        sort_elite_pairs(paired, mdl_lambda)
        elite_parents = [p[0] for p in paired]
        elite_val_logliks = [p[1] for p in paired]
    return elite_parents, elite_val_logliks


def run_global_evolution_phase(
    *,
    dataset: str,
    participants: List[int],
    seed_program_path: str,
    n_iterations: int,
    n_candidates_per_iteration: int,
    fresh_n_candidates: int = 0,
    sample_size: int,
    sample_parents: bool,
    sampled_parents_decay: bool = True,
    elite_pool_size: Optional[int],
    model_name: str,
    client: OpenAI,
    split_ratio: float,
    split_seed: int,
    data_path: str = "data",
    filter_mixed_gambles: bool = False,
    max_prompt_train_trials: int = 1_000_000,
    max_prompt_trials_per_problem: int = 0,
    llm_max_tokens: int = 800,
    max_workers: int = 5,
    n_eval_seeds: int = 3,
    output_dir: Path,
    save_artifacts: bool = True,
    wandb_module: Optional[Any] = None,
    run_prompts_dir: Optional[str] = None,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_parent_chars: int = 6000,
    warn_parent_truncation_ratio: float = 0.5,
    early_stop_iters: Optional[int] = None,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    prompt_debug: bool = False,
    prompt_debug_on_no_valid: bool = True,
    prompt_debug_exit: bool = False,
    evolution_selection_score: str = "train_val",
    max_error_prompt_chars: int = 1200,
    error_feedback_mode: str = DEFAULT_ERROR_FEEDBACK_MODE,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    speekenbrink_split: str = "chronological",
    mdl_lambda: float = 0.0,
    mem_trace: bool = True,
    prompt_suffix: Optional[str] = None,
    g2_arm: Optional[str] = None,
    g2_paired_pack_path: Optional[str] = None,
) -> List[Tuple[Any, ...]]:
    """
    Cross-participant evolution on pooled train trials (loglik fitness).

    Runs before per-participant evolution when ``--global_phase`` and ``--phase all``.
    ``prompt_suffix`` (Stage G) injects a live source rank-1 into every global
    generation prompt; evolution still starts from the vanilla seed.
    ``mem_trace`` (default on) writes ``global_phase/mem_trace.jsonl`` for both
    source-population and target global runs. Disable with ``--no-mem_trace``.
    """
    error_feedback_mode = _normalize_error_feedback_mode(error_feedback_mode)
    mdl_lambda = normalize_mdl_lambda(mdl_lambda)
    prompt_contract_scope(
        uses_training_only_sa40(limited_data_protocol)
    ).__enter__()
    participant_ids = [int(p) for p in participants]
    sparse_audits: List[SparseObservationAudit] = []
    pooled_train = _collect_pooled_train_trials_for_participants(
        dataset,
        participant_ids,
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        speekenbrink_split=speekenbrink_split,
        audits_out=sparse_audits,
    )
    pooled_val = _collect_pooled_split_trials_for_participants(
        dataset,
        participant_ids,
        split="val",
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        speekenbrink_split=speekenbrink_split,
    )
    pooled_test = _collect_pooled_test_trials_for_participants_diagnostic(
        dataset,
        participant_ids,
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        speekenbrink_split=speekenbrink_split,
    )
    budget_n = normalize_max_observed_trials(max_observed_trials_per_participant)
    print(f"\n{'='*80}")
    print(
        f"Global phase: {n_iterations} iteration(s), "
        f"{len(participant_ids)} participant(s), "
        f"{len(pooled_train)} pooled train trials, {len(pooled_val)} pooled val trials "
        f"(prompt injects train+val with shared cap; pool ranking uses {evolution_selection_score} score)"
        + (
            f"; sparse train+val cap={budget_n} per participant"
            if budget_n is not None
            else ""
        )
    )
    print(f"{'='*80}")
    if prompt_suffix:
        print("[INFO] Global prompt includes a cross-task source rank-1 suffix (Stage G).")

    global_dir = output_dir / "global_phase"
    if save_artifacts:
        global_dir.mkdir(parents=True, exist_ok=True)
        if sparse_audits and budget_n is not None:
            write_sparse_audits_payload(global_dir / SPARSE_AUDIT_FILENAME, sparse_audits)
            write_sparse_audits_csv(global_dir / SPARSE_AUDIT_CSV_FILENAME, sparse_audits)
            print(
                f"Saved sparse observation audit ({len(sparse_audits)} participant(s)) -> "
                f"{global_dir / SPARSE_AUDIT_FILENAME}"
            )

    seed_code = load_seed_program(seed_program_path)
    seed_fn = compile_program(seed_code)
    if seed_fn is None:
        raise RuntimeError(f"Failed to compile seed program for global phase: {seed_program_path}")
    baseline_eval = _evaluate_loglik_for_dataset(
        dataset, seed_fn, pooled_train, n_seeds=n_eval_seeds
    )
    baseline_ll = float(baseline_eval["avg_loglik"])
    baseline_val_ll: Optional[float] = None
    if pooled_val:
        baseline_val_eval = _evaluate_loglik_for_dataset(
            dataset, seed_fn, pooled_val, n_seeds=n_eval_seeds
        )
        baseline_val_ll = float(baseline_val_eval["avg_loglik"])
    use_train_val = _uses_train_val_evolution_selection(evolution_selection_score, "loglik")
    baseline_fitness = (
        _evolution_selection_score(
            baseline_ll,
            baseline_val_ll,
            len(pooled_train),
            len(pooled_val),
            evolution_selection_score=evolution_selection_score,
            warn_key="global",
        )
        if use_train_val
        else baseline_ll
    )
    elite_parents: List[Tuple[Any, ...]] = [
        apply_mdl_to_scored_elite(
            (
                seed_code,
                baseline_fitness,
                None,
                "global_baseline",
                None,
                None,
                baseline_ll if use_train_val else baseline_ll,
            ),
            selection_score=float(baseline_fitness),
            n=_mdl_n_for_score(
                evolution_selection_score,
                len(pooled_train),
                len(pooled_val),
                baseline_val_ll,
            ),
            mdl_lambda=mdl_lambda,
            runtime_valid=True,
        )
    ]
    print(f"Global baseline train loglik: {baseline_ll:.6f}")
    if baseline_val_ll is not None:
        print(f"Global baseline val loglik: {baseline_val_ll:.6f}")
    if use_train_val:
        print(
            f"Global evolution selection score mode: {evolution_selection_score} "
            f"(baseline selection_score={baseline_fitness:.6f})"
        )
    early_stop_patience = _normalize_early_stop_iters(early_stop_iters)
    last_significant_best = baseline_fitness
    stagnant_iters = 0
    diagnostic_test_by_iteration: List[Dict[str, Any]] = []
    invalid_candidate_errors: List[Dict[str, Any]] = _ErrorFeedbackStore(
        error_feedback_mode
    )
    error_history_path = global_dir / "error_history.jsonl"
    mem_trace_file: Optional[Path] = None
    mem_run_id = ""
    if bool(mem_trace) and save_artifacts:
        if not global_dir.exists():
            global_dir.mkdir(parents=True, exist_ok=True)
        mem_trace_file = mem_trace_path(global_dir)
        mem_run_id = output_dir.name
        print(f"MEM trace logging enabled: {mem_trace_file}")
    if early_stop_patience is not None:
        print(
            f"Global early stop enabled: patience={early_stop_patience}, "
            f"min_improvement={_EARLY_STOP_MIN_IMPROVEMENT:.3f}"
        )

    for iteration in range(n_iterations):
        iteration_step = iteration + 1
        print(f"\n{'='*80}")
        print(f"Global iteration {iteration_step}/{n_iterations}")
        print(f"{'='*80}")

        iter_dir: Optional[Path] = None
        if save_artifacts:
            iter_dir = global_dir / f"iteration_{iteration_step}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "candidates").mkdir(exist_ok=True)

        pool_size = len(elite_parents)
        if sample_parents and pool_size > 0:
            rng = np.random.default_rng(
                int(split_seed) + 50_000 + int(iteration_step) * 1_000_003
            )
            parent_idxs, best_k, sampled_k = _select_parent_indices_from_elite_pool(
                pool_size,
                sample_size=sample_size,
                sample_parents=True,
                sampled_parents_decay=sampled_parents_decay,
                iter_idx=iteration,
                total_iters=n_iterations,
                rng=rng,
            )
            selected_parents = [elite_parents[int(j)] for j in parent_idxs]
            num_parents_to_use = len(selected_parents)
            if sampled_parents_decay:
                print(
                    f"\nUsing {num_parents_to_use} global parent(s) "
                    f"({best_k} best + {sampled_k} sampled, pool size={pool_size}):"
                )
            else:
                print(
                    f"\nUsing {num_parents_to_use} sampled global parent(s) "
                    f"(pool size={pool_size}):"
                )
        else:
            num_parents_to_use = min(sample_size, pool_size)
            selected_parents = elite_parents[:num_parents_to_use]
            parent_idxs = list(range(num_parents_to_use))
            print(f"\nUsing top {num_parents_to_use} global parent(s):")

        for i, parent_tuple in enumerate(selected_parents):
            prog_id = parent_tuple[3]
            train_ll = _train_loglik_from_elite_tuple(parent_tuple)
            print(f"  Parent {i+1}: {prog_id} (global_train_loglik={train_ll:.4f})")

        parent_codes = [p[0] for p in selected_parents]
        parent_train_lls = [_train_loglik_from_elite_tuple(p) for p in selected_parents]

        mem_selected_parent_records: List[Dict[str, Any]] = []
        mem_best_parent_rec: Optional[Dict[str, Any]] = None
        mem_seed_selection_score: Optional[float] = None
        if mem_trace_file is not None:
            for parent_tuple in selected_parents:
                train_ll = _train_loglik_from_elite_tuple(
                    parent_tuple, evolution_selection_score=evolution_selection_score
                )
                mem_selected_parent_records.append(
                    parent_record_from_elite_tuple(
                        parent_tuple,
                        train_loglik=train_ll,
                    )
                )
            mem_best_parent_rec = best_reference_parent(mem_selected_parent_records)
            for ep in elite_parents:
                if str(ep[3]) == "global_baseline":
                    mem_seed_selection_score = float(ep[1]) if ep[1] is not None else None
                    break
            if mem_seed_selection_score is None:
                mem_seed_selection_score = float(baseline_fitness)
            append_mem_trace_record(
                mem_trace_file,
                build_iteration_context_record(
                    dataset=dataset,
                    participant_id="global",
                    run_id=mem_run_id,
                    split_seed=int(split_seed),
                    phase="global_evolution",
                    iteration=iteration_step,
                    evolution_selection_score=evolution_selection_score,
                    selected_parents=mem_selected_parent_records,
                    best_selected_parent_id=(
                        str(mem_best_parent_rec["program_id"])
                        if mem_best_parent_rec is not None
                        else None
                    ),
                ),
            )

        prompt_stats_path = (
            iter_dir / "prompt_stats.json" if iter_dir is not None else None
        )
        capture_gen_debug = bool(prompt_debug or prompt_debug_on_no_valid)
        gen_debug: Dict[str, Any] = {}
        fresh_n = _decayed_fresh_n_for_iteration(
            fresh_n_candidates, iteration, n_iterations, n_candidates_per_iteration
        )
        n_normal = n_candidates_per_iteration - fresh_n
        print(
            f"Fresh candidate schedule (global): iter_idx={iteration}, "
            f"total_iterations={n_iterations}, fresh_n={fresh_n} "
            f"(max fresh_n_candidates={fresh_n_candidates})"
        )
        print(
            f"\nGenerating {n_candidates_per_iteration} global candidates: "
            f"{fresh_n} fresh (seed/baseline only), {n_normal} from sampled parents..."
        )
        error_prompt_section = _build_past_error_prompt_section(
            invalid_candidate_errors,
            iteration=iteration_step,
            max_error_prompt_chars=max_error_prompt_chars,
            previous_n_candidates=n_candidates_per_iteration,
            error_feedback_mode=error_feedback_mode,
        )
        _write_iteration_error_prompt_file(iter_dir, error_prompt_section)
        error_prompt_chars_used = len(error_prompt_section)
        num_unique_errors_available = _count_previous_iteration_error_groups(
            invalid_candidate_errors,
            iteration=iteration_step,
            error_feedback_mode=error_feedback_mode,
        )
        print(
            "Error prompt summary: "
            f"num_unique_errors_available={num_unique_errors_available}, "
            f"error_prompt_chars_used={error_prompt_chars_used}"
        )
        variant_kwargs = {
            "train_trials": pooled_train,
            "extra_prompt_trials": pooled_val if pooled_val else None,
            "max_tokens": llm_max_tokens,
            "dataset": dataset,
            "max_prompt_train_trials": max_prompt_train_trials,
            "max_prompt_trials_per_problem": max_prompt_trials_per_problem,
            "prompt_train_trials_seed": int(split_seed) + 60_000 + iteration_step,
            "fitness_metric": "loglik",
            "max_workers": max_workers,
            "run_prompts_dir": run_prompts_dir,
            "max_parent_chars": max_parent_chars,
            "warn_parent_truncation_ratio": warn_parent_truncation_ratio,
            "sample_size_for_warning": sample_size,
            "prompt_stats_path": prompt_stats_path,
            "hard_prompt_token_cap": hard_prompt_token_cap,
            "strict_prompt_budget": strict_prompt_budget,
            "prompt_token_estimator": prompt_token_estimator,
            "prompt_diagnostics_dir": output_dir,
            "phase": "global_evolution",
            "participant_id": None,
            "iteration": iteration_step,
            "prompt_debug": prompt_debug,
            "prompt_debug_exit": prompt_debug_exit,
            "generation_debug_out": gen_debug if capture_gen_debug else None,
            "past_invalid_program_errors": invalid_candidate_errors,
            "past_error_prompt_section": error_prompt_section,
            "max_error_prompt_chars": max_error_prompt_chars,
            "error_feedback_mode": error_feedback_mode,
            "prompt_suffix": prompt_suffix,
            "g2_arm": g2_arm,
            "g2_paired_pack_path": g2_paired_pack_path,
            "llm_decoding_seed_base": _phase_llm_decoding_seed_base(
                split_seed=int(split_seed),
                iteration_step=int(iteration_step),
            ),
        }
        candidate_codes, candidate_sources = _generate_iteration_candidate_codes(
            client=client,
            model_name=model_name,
            fresh_n_candidates=fresh_n,
            n_candidates=n_candidates_per_iteration,
            fresh_parent_programs=[seed_code],
            normal_parent_programs=parent_codes,
            variant_kwargs=variant_kwargs,
            fresh_parent_train_accuracies=[baseline_ll],
            normal_parent_train_accuracies=parent_train_lls,
            fresh_parent_program_ids=["global_baseline"],
            normal_parent_program_ids=[str(p[3]) for p in selected_parents],
        )

        selected_results: List[Dict[str, Any]] = []
        evaluated_audit: List[Dict[str, Any]] = []
        num_invalid_candidates = 0

        for idx, code in enumerate(candidate_codes):
            if iter_dir is not None:
                (iter_dir / "candidates" / f"candidate_{idx}.py").write_text(code or "")
            code = _sanitize_llm_python_candidate(code, required_markers=("def choose(",))
            program_id = f"global_iteration_{iteration_step}_candidate_{idx}"
            source = (
                candidate_sources[idx]
                if idx < len(candidate_sources)
                else "normal"
            )
            audit_base: Dict[str, Any] = {
                "idx": idx,
                "program_id": program_id,
                "source": source,
                "code": code or "",
                "runtime_valid": False,
                "train_loglik": None,
                "val_loglik": None,
                "selection_score": None,
            }
            if not code:
                evaluated_audit.append(audit_base)
                continue
            choose_fn, compile_error = compile_program_with_error(code)
            if choose_fn is None:
                num_invalid_candidates += 1
                _record_invalid_program_error(
                    invalid_candidate_errors,
                    code=code,
                    exc=compile_error,
                    iteration=iteration_step,
                    participant_id=None,
                    candidate_id=f"candidate_{idx}",
                    eval_split="compile",
                    n_candidates_in_iteration=n_candidates_per_iteration,
                    history_path=error_history_path,
                )
                evaluated_audit.append(audit_base)
                continue
            try:
                train_eval = _evaluate_loglik_for_dataset(
                    dataset, choose_fn, pooled_train, n_seeds=n_eval_seeds
                )
            except (AssertionError, TypeError, ValueError) as exc:
                num_invalid_candidates += 1
                _record_invalid_program_error(
                    invalid_candidate_errors,
                    code=code,
                    exc=exc,
                    iteration=iteration_step,
                    participant_id=None,
                    candidate_id=f"candidate_{idx}",
                    eval_split="train",
                    n_candidates_in_iteration=n_candidates_per_iteration,
                    history_path=error_history_path,
                )
                evaluated_audit.append(audit_base)
                continue
            if train_eval.get("errors", 0) != 0:
                num_invalid_candidates += 1
                _record_invalid_program_error_summary(
                    invalid_candidate_errors,
                    train_eval.get("first_error"),
                    iteration=iteration_step,
                    participant_id=None,
                    candidate_id=f"candidate_{idx}",
                    quality_score=_safe_float(train_eval.get("avg_loglik")),
                    eval_split="train",
                    n_candidates_in_iteration=n_candidates_per_iteration,
                    history_path=error_history_path,
                )
                evaluated_audit.append(audit_base)
                continue
            train_loglik = float(train_eval["avg_loglik"])
            audit_base["train_loglik"] = train_loglik
            val_loglik: Optional[float] = None
            if use_train_val and pooled_val:
                try:
                    val_eval = _evaluate_loglik_for_dataset(
                        dataset, choose_fn, pooled_val, n_seeds=n_eval_seeds
                    )
                except (AssertionError, TypeError, ValueError) as exc:
                    num_invalid_candidates += 1
                    _record_invalid_program_error(
                        invalid_candidate_errors,
                        code=code,
                        exc=exc,
                        iteration=iteration_step,
                        participant_id=None,
                        candidate_id=f"candidate_{idx}",
                        quality_score=_safe_float(train_loglik),
                        eval_split="val",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                    evaluated_audit.append(audit_base)
                    continue
                if val_eval.get("errors", 0) != 0:
                    num_invalid_candidates += 1
                    _record_invalid_program_error_summary(
                        invalid_candidate_errors,
                        val_eval.get("first_error"),
                        iteration=iteration_step,
                        participant_id=None,
                        candidate_id=f"candidate_{idx}",
                        quality_score=_safe_float(train_loglik),
                        eval_split="val",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                    evaluated_audit.append(audit_base)
                    continue
                val_loglik = float(val_eval["avg_loglik"])
            selection_score = _evolution_selection_score(
                train_loglik,
                val_loglik,
                len(pooled_train),
                len(pooled_val),
                evolution_selection_score=evolution_selection_score,
                warn_key="global" if use_train_val else None,
            )
            fitness = selection_score if use_train_val else train_loglik
            row: Dict[str, Any] = {
                "idx": idx,
                "code": code,
                "train_loglik": train_loglik,
                "fitness": fitness,
                "selection_score": selection_score,
                "runtime_valid": True,
            }
            if val_loglik is not None:
                row["val_loglik"] = val_loglik
            _attach_mdl_to_candidate(
                row,
                mdl_lambda=mdl_lambda,
                n=_mdl_n_for_score(
                    evolution_selection_score,
                    len(pooled_train),
                    len(pooled_val),
                    val_loglik,
                ),
                selection_score=selection_score,
            )
            selected_results.append(row)
            audit_base.update(
                {
                    "runtime_valid": True,
                    "val_loglik": val_loglik,
                    "selection_score": selection_score,
                }
            )
            evaluated_audit.append(audit_base)

        print(
            "Iteration invalid summary: "
            f"num_invalid_candidates={num_invalid_candidates}, "
            f"num_unique_errors_available={_count_previous_iteration_error_groups(invalid_candidate_errors, iteration=iteration_step + 1, error_feedback_mode=error_feedback_mode)}, "
            f"error_prompt_chars_used={error_prompt_chars_used}"
        )

        if not selected_results:
            print("Warning: No runtime-valid global candidates; keeping elite pool.")
        else:
            sort_candidates(selected_results, mdl_lambda)
            best = selected_results[0]
            print(
                f"  Best global candidate {best['idx']}: "
                f"train_loglik={best['train_loglik']:.6f}"
                + (
                    f", selection_score={best['selection_score']:.6f}"
                    if use_train_val
                    else ""
                )
            )
            for result in selected_results:
                program_id = f"global_iteration_{iteration_step}_candidate_{result['idx']}"
                elite_parents.append(
                    _elite_tuple_for_ranking(
                        result["code"],
                        result["fitness"],
                        None,
                        program_id,
                        result["train_loglik"],
                        mdl_lambda=mdl_lambda,
                        mdl_score=result.get("mdl_score"),
                    )
                )

        sort_elites(elite_parents, mdl_lambda)
        elite_cap = _elite_pool_capacity(sample_size, elite_pool_size)
        # Prune after all candidates are evaluated (audit rows recorded above).
        elite_parents = elite_parents[:elite_cap]
        pool_best_ll = _train_loglik_from_elite_tuple(
            elite_parents[0], evolution_selection_score=evolution_selection_score
        )
        pool_best_selection = float(elite_parents[0][1])
        print(
            f"\nGlobal elite set updated: {len(elite_parents)} programs "
            f"(cap={elite_cap}, pool_best_train_loglik={pool_best_ll:.6f}"
            + (
                f", pool_best_selection_score={pool_best_selection:.6f})"
                if use_train_val
                else ")"
            )
        )
        pool_best_test_record = _passive_diagnostic_test_of_program(
            dataset=dataset,
            code=elite_parents[0][0],
            test_trials=pooled_test,
            n_eval_seeds=n_eval_seeds,
            program_id=str(elite_parents[0][3]),
            phase="global_evolution",
            iteration=iteration_step,
        )
        diagnostic_test_by_iteration.append(pool_best_test_record)
        if pool_best_test_record.get("test_loglik") is not None:
            print(
                f"  Diagnostic test loglik of pool-best {pool_best_test_record['program_id']}: "
                f"{float(pool_best_test_record['test_loglik']):.6f} "
                "(passive; not used for ranking)"
            )

        if mem_trace_file is not None:
            elite_ids_after = {str(p[3]) for p in elite_parents}
            ref_parent_id = (
                str(mem_best_parent_rec["program_id"])
                if mem_best_parent_rec is not None
                else None
            )
            ref_parent_score = (
                mem_best_parent_rec.get("selection_score")
                if mem_best_parent_rec is not None
                else None
            )
            for audit in evaluated_audit:
                idx = int(audit["idx"])
                program_id = str(audit["program_id"])
                source = str(audit.get("source") or "normal")
                if source == "fresh":
                    reference_kind = "seed_baseline"
                    cand_ref_id = "global_baseline"
                    cand_ref_score = mem_seed_selection_score
                    prompted_ids = ["global_baseline"]
                    ref_exact = True
                else:
                    reference_kind = "best_prompted_parent"
                    cand_ref_id = ref_parent_id
                    cand_ref_score = ref_parent_score
                    prompted_ids = [
                        str(p.get("program_id"))
                        for p in mem_selected_parent_records
                        if p.get("program_id") is not None
                    ]
                    ref_exact = True
                cand_score = audit.get("selection_score")
                if cand_score is None and audit.get("runtime_valid"):
                    cand_score = audit.get("train_loglik")
                append_mem_trace_record(
                    mem_trace_file,
                    build_candidate_record(
                        dataset=dataset,
                        participant_id="global",
                        run_id=mem_run_id,
                        split_seed=int(split_seed),
                        phase="global_evolution",
                        iteration=iteration_step,
                        candidate_id=program_id,
                        candidate_idx=idx,
                        source=str(source),
                        code=audit.get("code") or "",
                        runtime_valid=bool(audit.get("runtime_valid", False)),
                        train_loglik=_safe_float(audit.get("train_loglik")),
                        val_loglik=_safe_float(audit.get("val_loglik")),
                        selection_score=_safe_float(cand_score),
                        reference_parent_id=cand_ref_id,
                        reference_parent_score=_safe_float(cand_ref_score),
                        reference_kind=reference_kind,
                        reference_type=reference_kind,
                        reference_id=cand_ref_id,
                        reference_is_exact=ref_exact,
                        delta_f=compute_delta_f(
                            _safe_float(cand_score), _safe_float(cand_ref_score)
                        ),
                        survived_elite_truncation=program_id in elite_ids_after,
                        evolution_selection_score=evolution_selection_score,
                        prompted_parent_ids=prompted_ids,
                        code_path=(
                            f"iteration_{iteration_step}/candidates/candidate_{idx}.py"
                        ),
                    ),
                )

        if iter_dir is not None:
            metrics = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_runtime_valid": len(selected_results),
                "num_invalid_candidates": num_invalid_candidates,
                "num_unique_errors_available": num_unique_errors_available,
                "error_prompt_chars_used": error_prompt_chars_used,
                "pool_best_program_id": elite_parents[0][3],
                "pool_best_global_train_loglik": pool_best_ll,
                "evolution_selection_score": evolution_selection_score,
                "diagnostic_test": pool_best_test_record,
            }
            if use_train_val:
                metrics["pool_best_selection_score"] = pool_best_selection
            if mdl_enabled(mdl_lambda):
                metrics["mdl_lambda"] = float(mdl_lambda)
                metrics.update(
                    {
                        f"pool_best_{k}": v
                        for k, v in _elite_mdl_manifest_fields(
                            elite_parents[0],
                            mdl_lambda=mdl_lambda,
                            n=_mdl_n_for_score(
                                evolution_selection_score,
                                len(pooled_train),
                                len(pooled_val),
                                0.0 if pooled_val else None,
                            ),
                            selection_score=pool_best_selection,
                        ).items()
                    }
                )
                if selected_results:
                    metrics["iter_candidate_mdl"] = [
                        {
                            "idx": r["idx"],
                            "selection_score": r.get("selection_score"),
                            **_candidate_mdl_log_fields(r),
                        }
                        for r in selected_results
                    ]
            metrics.update(
                _iteration_candidate_source_header(
                    fresh_n_candidates,
                    fresh_n,
                    n_candidates_per_iteration,
                    candidate_sources,
                    iter_idx=iteration,
                    total_iters=n_iterations,
                )
            )
            metrics["best_from_fresh_candidate"] = _best_from_fresh_candidate(
                metrics.get("pool_best_program_id"),
                iteration_step,
                candidate_sources=candidate_sources,
            )
            (iter_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )

        if wandb_module is not None:
            global_log: Dict[str, Any] = {
                "global/train_loglik": pool_best_ll,
                "global/pool_size": len(elite_parents),
                "global/iteration": iteration_step,
                "global/diagnostic_test_loglik": pool_best_test_record.get("test_loglik"),
                "global/diagnostic_test_program_id": pool_best_test_record.get("program_id"),
            }
            if use_train_val:
                global_log["global/selection_score"] = pool_best_selection
            wandb_module.log(global_log, step=iteration_step)

        if early_stop_patience is not None:
            improvement = float(pool_best_selection) - float(last_significant_best)
            if improvement >= _EARLY_STOP_MIN_IMPROVEMENT:
                last_significant_best = float(pool_best_selection)
                stagnant_iters = 0
            else:
                stagnant_iters += 1
                if stagnant_iters >= early_stop_patience:
                    print(
                        f"Early stopping global phase at iteration {iteration_step}: "
                        f"pool best improved by < {_EARLY_STOP_MIN_IMPROVEMENT:.3f} for "
                        f"{stagnant_iters} consecutive iteration(s)."
                    )
                    break

    if save_artifacts:
        pool_dir = _save_global_elite_pool(
            global_dir,
            elite_parents,
            mdl_lambda=mdl_lambda,
            n=_mdl_n_for_score(
                evolution_selection_score,
                len(pooled_train),
                len(pooled_val),
                baseline_val_ll,
            ),
        )
        print(f"Saved global elite pool ({len(elite_parents)} programs) -> {pool_dir}")
        print(f"Saved best program -> {global_dir / BEST_PROGRAM_FILENAME}")
        pool_best_code = elite_parents[0][0]
        pool_best_global_train_ll = _train_loglik_from_elite_tuple(
            elite_parents[0], evolution_selection_score=evolution_selection_score
        )
        pool_best_selection_score = float(elite_parents[0][1])
        _write_global_phase_summary_loglik_csv(
            global_dir,
            dataset=dataset,
            participant_ids=participant_ids,
            pool_best_code=pool_best_code or "",
            split_ratio=split_ratio,
            split_seed=split_seed,
            data_path=data_path,
            filter_mixed_gambles=filter_mixed_gambles,
            n_eval_seeds=n_eval_seeds,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            max_observed_trials_per_participant=max_observed_trials_per_participant,
            limited_data_protocol=limited_data_protocol,
            limited_train_val=limited_train_val,
        )
        global_results: Dict[str, Any] = {
            "phase": "global",
            "dataset": dataset,
            "n_participants": len(participant_ids),
            "participant_ids": participant_ids,
            "n_iterations": int(n_iterations),
            "n_pooled_train_trials": len(pooled_train),
            "n_pooled_val_trials": len(pooled_val),
            "max_observed_trials_per_participant": budget_n,
            "pool_size": len(elite_parents),
            "pool_best_program_id": str(elite_parents[0][3]),
            "pool_best_global_train_loglik": pool_best_global_train_ll,
            "pool_best_selection_score": pool_best_selection_score,
            "evolution_selection_score": evolution_selection_score,
            "baseline_global_train_loglik": baseline_ll,
            "diagnostic_test_by_iteration": diagnostic_test_by_iteration,
        }
        if mdl_enabled(mdl_lambda):
            global_results["mdl_lambda"] = float(mdl_lambda)
            global_results.update(
                _elite_mdl_manifest_fields(
                    elite_parents[0],
                    mdl_lambda=mdl_lambda,
                    n=_mdl_n_for_score(
                        evolution_selection_score,
                        len(pooled_train),
                        len(pooled_val),
                        baseline_val_ll,
                    ),
                    selection_score=pool_best_selection_score,
                )
            )
        summary_csv_path = global_dir / "summary_loglik.csv"
        if summary_csv_path.is_file():
            with summary_csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                summary_row = next(reader, None)
                if summary_row:
                    for key in (
                        "avg_train_loglik",
                        "avg_test_loglik",
                        "avg_val_loglik",
                        "avg_gated_test_loglik",
                    ):
                        if summary_row.get(key) not in (None, ""):
                            global_results[key] = float(summary_row[key])
        (global_dir / "results.json").write_text(
            json.dumps(global_results, indent=2),
            encoding="utf-8",
        )
    return elite_parents


def _cross_task_source_suffix(
    *,
    source_dataset: str,
    program_path: str,
    args: Any,
    psych_dataset_split: str,
    filter_mixed_gambles: bool,
    best_loglik: Optional[float] = None,
    require_source_examples: bool = False,
) -> str:
    """Build the G/E source-program suffix using the same obs protocol as this run."""
    return build_rank1_explore_prompt_suffix(
        source_dataset=source_dataset,
        program_path=program_path,
        split_seed=int(args.split_seed),
        psych_dataset_split=psych_dataset_split,
        best_loglik=best_loglik,
        split_ratio=float(args.split_ratio),
        limited_data_protocol=str(args.limited_data_protocol),
        limited_train_val=args.limited_train_val,
        max_observed_trials_per_participant=args.max_observed_trials_per_participant,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
        filter_mixed_gambles=bool(filter_mixed_gambles)
        or str(source_dataset) == "mixed_gambles",
        require_source_examples=bool(require_source_examples),
    )


def _run_t_pics_source_population(
    *,
    source_dataset: str,
    source_participants: List[int],
    output_dir: Path,
    seed_program_path: str,
    args: Any,
    client: OpenAI,
    wandb_module: Optional[Any],
    psych_dataset_split: str,
    filter_mixed_gambles: bool,
    n_iterations: Optional[int] = None,
    require_auto_llm_prompt: bool = False,
    llm_decoding_seed: Optional[int] = None,
) -> Tuple[Path, Optional[float]]:
    """G.1: live source-dataset population under the same knobs/obs protocol as the target.

    No cross-task prompt. No person evolution. Writes ``output_dir/global_phase/best_program.py``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_iters = (
        int(n_iterations) if n_iterations is not None else int(args.global_iters)
    )
    source_prompts_dir = setup_teh_run_prompts(
        output_dir,
        source_dataset,
        Path(seed_program_path),
        client=client,
        model_name=args.model_name,
        use_llm=not args.no_llm_prompt,
        base_prompt_path=args.base_prompt,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        example_char_budget=int(
            getattr(args, "dataset_prompt_example_char_budget", DEFAULT_EXAMPLE_CHAR_BUDGET)
        ),
        history_max_entries=int(
            getattr(args, "dataset_prompt_history_max_entries", DEFAULT_HISTORY_MAX_ENTRIES)
        ),
        max_examples=int(getattr(args, "dataset_prompt_max_examples", DEFAULT_MAX_EXAMPLES)),
        prefer_auto_llm_prompt=bool(getattr(args, "prefer_auto_llm_prompt", False))
        or int(getattr(args, "dataset_prompt_evolution_iterations", 0) or 0) > 0
        or bool(require_auto_llm_prompt),
        dataset_prompt_file=None,
        split_ratio=float(args.split_ratio),
        split_seed=int(args.split_seed),
        speekenbrink_split=str(getattr(args, "speekenbrink_split", "chronological")),
        data_dir=getattr(args, "data_dir", None),
        limited_data_protocol=str(args.limited_data_protocol),
        limited_train_val=args.limited_train_val,
        max_observed_trials_per_participant=args.max_observed_trials_per_participant,
        require_auto_llm_prompt=bool(require_auto_llm_prompt),
        llm_decoding_seed=llm_decoding_seed,
    )
    source_seed = str(source_prompts_dir / "seed_program.py")
    print(
        f"[T-PICS] Training source population on {source_dataset}: "
        f"{len(source_participants)} participant(s), global_iters={source_iters}, "
        f"protocol={args.limited_data_protocol}, limited_train_val={args.limited_train_val}"
    )
    elite = run_global_evolution_phase(
        dataset=source_dataset,
        participants=[int(p) for p in source_participants],
        seed_program_path=source_seed,
        n_iterations=source_iters,
        n_candidates_per_iteration=args.n_candidates,
        fresh_n_candidates=args.fresh_n_candidates,
        sample_size=args.sample_size,
        sample_parents=args.sample_parents,
        sampled_parents_decay=args.sampled_parents_decay,
        elite_pool_size=args.elite_pool_size,
        model_name=args.model_name,
        client=client,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        data_path=args.data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        max_prompt_train_trials=args.max_prompt_train_trials,
        max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
        llm_max_tokens=args.llm_max_tokens,
        max_workers=args.max_workers,
        n_eval_seeds=args.n_eval_seeds,
        output_dir=output_dir,
        save_artifacts=True,
        wandb_module=None,
        run_prompts_dir=str(source_prompts_dir),
        psych_dataset_split=psych_dataset_split,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
        max_parent_chars=args.max_parent_chars,
        warn_parent_truncation_ratio=args.warn_parent_truncation_ratio,
        early_stop_iters=args.early_stop_iters,
        hard_prompt_token_cap=args.hard_prompt_token_cap,
        strict_prompt_budget=args.strict_prompt_budget,
        prompt_token_estimator=args.prompt_token_estimator,
        prompt_debug=args.prompt_debug,
        prompt_debug_on_no_valid=args.prompt_debug_on_no_valid,
        prompt_debug_exit=args.prompt_debug_exit,
        evolution_selection_score=args.evolution_selection_score,
        max_error_prompt_chars=args.max_error_prompt_chars,
        error_feedback_mode=args.error_feedback_mode,
        max_observed_trials_per_participant=args.max_observed_trials_per_participant,
        limited_data_protocol=args.limited_data_protocol,
            speekenbrink_split=getattr(args, "speekenbrink_split", "chronological"),
        limited_train_val=args.limited_train_val,
        mdl_lambda=args.mdl_lambda,
        mem_trace=args.mem_trace,
        prompt_suffix=None,
    )
    best_path = output_dir / "global_phase" / BEST_PROGRAM_FILENAME
    if not best_path.is_file():
        raise FileNotFoundError(f"T-PICS source population did not write {best_path}")
    best_ll = float(elite[0][1]) if elite else None
    manifest = {
        "phase": "t_pics_source_population",
        "source_dataset": source_dataset,
        "n_participants": len(source_participants),
        "participant_ids": [int(p) for p in source_participants],
        "global_iters": int(source_iters),
        "limited_data_protocol": str(args.limited_data_protocol),
        "limited_train_val": args.limited_train_val,
        "max_observed_trials_per_participant": args.max_observed_trials_per_participant,
        "best_program": str(best_path),
        "pool_best_selection_score": best_ll,
        "elite_pool_size": args.elite_pool_size,
    }
    (output_dir / "SOURCE_POPULATION.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if wandb_module is not None:
        wandb_module.log(
            {
                "t_pics/source_dataset": source_dataset,
                "t_pics/source_n_participants": len(source_participants),
                "t_pics/source_selection_score": best_ll,
            }
        )
    print(f"[T-PICS] Source rank-1 -> {best_path}")
    return best_path, best_ll


def _gated_global_phase_kwargs(
    args: Any,
    *,
    participants: List[int],
    seed_program_path: str,
    client: OpenAI,
    wandb_module: Optional[Any],
    psych_dataset_split: str,
    filter_mixed_gambles: bool,
    run_prompts_dir: str,
    output_dir: Path,
    prompt_suffix: Optional[str],
    g2_arm: Optional[str] = None,
    g2_paired_pack_path: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "dataset": args.dataset,
        "participants": [int(p) for p in participants],
        "seed_program_path": seed_program_path,
        "n_iterations": int(args.global_iters),
        "n_candidates_per_iteration": args.n_candidates,
        "fresh_n_candidates": args.fresh_n_candidates,
        "sample_size": args.sample_size,
        "sample_parents": args.sample_parents,
        "sampled_parents_decay": args.sampled_parents_decay,
        "elite_pool_size": args.elite_pool_size,
        "model_name": args.model_name,
        "client": client,
        "split_ratio": args.split_ratio,
        "split_seed": args.split_seed,
        "data_path": args.data_path,
        "filter_mixed_gambles": filter_mixed_gambles,
        "max_prompt_train_trials": args.max_prompt_train_trials,
        "max_prompt_trials_per_problem": args.max_prompt_trials_per_problem,
        "llm_max_tokens": args.llm_max_tokens,
        "max_workers": args.max_workers,
        "n_eval_seeds": args.n_eval_seeds,
        "output_dir": Path(output_dir),
        "save_artifacts": True,
        "wandb_module": wandb_module,
        "run_prompts_dir": str(run_prompts_dir),
        "psych_dataset_split": psych_dataset_split,
        "local_dataset": args.local_dataset,
        "mixed_gambles_csv": args.mixed_gambles_csv,
        "max_parent_chars": args.max_parent_chars,
        "warn_parent_truncation_ratio": args.warn_parent_truncation_ratio,
        "early_stop_iters": args.early_stop_iters,
        "hard_prompt_token_cap": args.hard_prompt_token_cap,
        "strict_prompt_budget": args.strict_prompt_budget,
        "prompt_token_estimator": args.prompt_token_estimator,
        "prompt_debug": args.prompt_debug,
        "prompt_debug_on_no_valid": args.prompt_debug_on_no_valid,
        "prompt_debug_exit": args.prompt_debug_exit,
        "evolution_selection_score": args.evolution_selection_score,
        "max_error_prompt_chars": args.max_error_prompt_chars,
        "error_feedback_mode": args.error_feedback_mode,
        "max_observed_trials_per_participant": args.max_observed_trials_per_participant,
        "limited_data_protocol": args.limited_data_protocol,
        "speekenbrink_split": getattr(args, "speekenbrink_split", "chronological"),
        "limited_train_val": args.limited_train_val,
        "mdl_lambda": args.mdl_lambda,
        "mem_trace": args.mem_trace,
        "prompt_suffix": prompt_suffix,
        "g2_arm": g2_arm,
        "g2_paired_pack_path": g2_paired_pack_path,
    }


def _write_arm_intended_argv(path: Path, argv: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(str(x) for x in argv) + "\n", encoding="utf-8")


def _load_gated_arm_pool(arm_dir: Path) -> List[Tuple[Any, ...]]:
    return _load_one_initial_pool_dir(str(Path(arm_dir) / "global_phase" / "global_elite_pool"))


def _ensure_gated_independent_source_population(
    *,
    args: Any,
    source_dataset: str,
    output_dir: Path,
    client: OpenAI,
    wandb_module: Optional[Any],
    psych_dataset_split: str,
    filter_mixed_gambles: bool,
) -> Tuple[Path, Optional[float]]:
    """Live G.1 for gated independent runs. Does not read frozen YAML rank-1 programs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_iters = int(SOURCE_POPULATION_ITERS)
    if population_arm_is_complete(output_dir, expected_global_iters=expected_iters):
        best = output_dir / "global_phase" / "best_program.py"
        print(f"[T-PICS gated] skip complete independent G.1 -> {best}")
        return best, None
    source_filter = bool(filter_mixed_gambles) or str(source_dataset) == "mixed_gambles"
    source_start, source_end = emnlp_ordinal_range(str(source_dataset))
    source_pids, clamp_note = resolve_t_pics_source_participant_ids(
        source_dataset=str(source_dataset),
        repo_root=_REPO_ROOT,
        participant_scope="range",
        single_participant_id=args.single_participant_id,
        range_start_ordinal=source_start,
        range_end_ordinal=source_end,
        all_max_participants=args.all_max_participants,
        participant_ordinals=args.ordinals,
        filter_mixed_gambles=source_filter,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
    )
    if clamp_note:
        print(f"[T-PICS gated] independent G.1 {clamp_note}")
    source_seed = default_seed_path(str(source_dataset))
    print(
        f"[T-PICS gated] independent G.1 live source population "
        f"dataset={source_dataset} n={len(source_pids)} "
        f"source_emnlp_range={source_start}-{source_end} iters={expected_iters} "
        f"(not reusing frozen G.1 programs)"
    )
    return _run_t_pics_source_population(
        source_dataset=str(source_dataset),
        source_participants=source_pids,
        output_dir=output_dir,
        seed_program_path=source_seed,
        args=args,
        client=client,
        wandb_module=wandb_module,
        psych_dataset_split=psych_dataset_split,
        filter_mixed_gambles=source_filter,
        n_iterations=expected_iters,
        require_auto_llm_prompt=True,
        llm_decoding_seed=int(args.split_seed) + 90_000,
    )


def _runtime_contract_text(run_prompts_dir: Optional[str]) -> str:
    if not run_prompts_dir:
        return ""
    path = Path(run_prompts_dir) / RUNTIME_CONTRACT_FILENAME
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _g2_paired_pack_example_seed(split_seed: int) -> int:
    """Freeze target examples with the G.2 iteration-0 subsample seed; reuse all five iters."""
    return int(split_seed) + 60_000 + 1


def _ensure_g2_paired_pack_freeze(
    *,
    args: Any,
    participants: List[int],
    seed_program_path: str,
    run_prompts_dir: str,
    psych_dataset_split: str,
    filter_mixed_gambles: bool,
    source_suffix: str,
    freeze_path: Path,
    source_dataset: str,
    source_rank1: Path,
) -> Dict[str, Any]:
    """Freeze shared G.2 target-example IDs under the transfer token condition."""
    freeze_path = Path(freeze_path)
    if freeze_path.is_file():
        payload = load_paired_pack_freeze(freeze_path)
        print(
            f"[T-PICS gated] reuse G.2 paired-pack freeze -> {freeze_path} "
            f"n_examples={payload.get('n_examples_included')} "
            f"version={payload.get('version')}"
        )
        return payload

    from utils.teh.prompt_sanitize import CANDIDATE_OUTPUT_RULES

    infer_path = Path(run_prompts_dir) / "infer_single_choice.txt"
    template_path = Path(run_prompts_dir) / "single_code_template.txt"
    if not infer_path.is_file():
        raise FileNotFoundError(f"G.2 paired packing needs infer prompt: {infer_path}")
    infer_text = infer_path.read_text(encoding="utf-8")
    code_template = (
        load_single_code_template(str(template_path)) if template_path.is_file() else ""
    )
    code_template_suffix = single_code_template_prompt_suffix(code_template)
    runtime_contract = _runtime_contract_text(run_prompts_dir)
    seed_code = Path(seed_program_path).read_text(encoding="utf-8")
    example_seed = _g2_paired_pack_example_seed(int(args.split_seed))
    participant_ids = [int(p) for p in participants]
    source_filter = bool(filter_mixed_gambles) or str(source_dataset) == "mixed_gambles"

    with prompt_contract_scope(
        uses_training_only_sa40(str(args.limited_data_protocol))
    ):
        pooled_train = _collect_pooled_train_trials_for_participants(
            str(args.dataset),
            participant_ids,
            split_ratio=float(args.split_ratio),
            split_seed=int(args.split_seed),
            data_path=str(args.data_path),
            filter_mixed_gambles=bool(filter_mixed_gambles),
            psych_dataset_split=psych_dataset_split,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
            max_observed_trials_per_participant=args.max_observed_trials_per_participant,
            limited_data_protocol=str(args.limited_data_protocol),
            limited_train_val=args.limited_train_val,
            speekenbrink_split=str(getattr(args, "speekenbrink_split", "chronological")),
        )
        pooled_val = _collect_pooled_split_trials_for_participants(
            str(args.dataset),
            participant_ids,
            split="val",
            split_ratio=float(args.split_ratio),
            split_seed=int(args.split_seed),
            data_path=str(args.data_path),
            filter_mixed_gambles=bool(filter_mixed_gambles),
            psych_dataset_split=psych_dataset_split,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
            max_observed_trials_per_participant=args.max_observed_trials_per_participant,
            limited_data_protocol=str(args.limited_data_protocol),
            limited_train_val=args.limited_train_val,
            speekenbrink_split=str(getattr(args, "speekenbrink_split", "chronological")),
        )
        payload = freeze_g2_target_examples(
            dataset=str(args.dataset),
            pooled_train=pooled_train,
            pooled_val=pooled_val,
            infer_text=infer_text,
            source_suffix=source_suffix,
            runtime_contract=runtime_contract,
            seed_code=seed_code,
            code_template_suffix=code_template_suffix,
            candidate_output_rules=f"\n{CANDIDATE_OUTPUT_RULES}\n",
            max_parent_chars=int(args.max_parent_chars),
            max_prompt_train_trials=int(args.max_prompt_train_trials),
            prompt_train_trials_seed=example_seed,
            hard_prompt_token_cap=int(args.hard_prompt_token_cap),
            sample_size=int(getattr(args, "sample_size", 8) or 8),
            output_reserve=int(args.llm_max_tokens),
        )
    payload["source_dataset"] = str(source_dataset)
    payload["source_rank1"] = str(source_rank1)
    payload["source_suffix_sha256"] = hashlib.sha256(
        (source_suffix or "").encode("utf-8")
    ).hexdigest()
    payload["source_filter_mixed_gambles"] = bool(source_filter)
    payload["freeze_seed"] = example_seed
    write_paired_pack_freeze(freeze_path, payload)
    print(
        f"[T-PICS gated] wrote G.2 paired-pack freeze -> {freeze_path} "
        f"n_available={payload.get('n_selected_available')} "
        f"n_included={payload.get('n_examples_included')} "
        f"paired_parent_count={payload.get('paired_parent_count')} "
        f"required_tokens={payload.get('transfer_required_tokens')} "
        f"packed_tokens={payload.get('transfer_packed_tokens_at_freeze')} "
        f"suffix_tokens={payload.get('source_suffix_tokens')}"
    )
    return payload


def _run_t_pics_gated_population_arms(
    *,
    args: Any,
    run_root: Path,
    participants: List[int],
    seed_program_path: str,
    client: OpenAI,
    wandb_module: Optional[Any],
    psych_dataset_split: str,
    filter_mixed_gambles: bool,
    run_prompts_dir: str,
) -> List[Tuple[Any, ...]]:
    """Matched G.2 control vs transfer arms, train_val observed-data gate, retain winner elite pool."""
    independent = gated_independent_enabled(args)
    explicit_source = str(getattr(args, "t_pics_gated_source", "") or "").strip() or None
    if explicit_source:
        if not independent:
            raise RuntimeError(
                "--t_pics_gated_source requires --t_pics_gated_independent"
            )
        entry = make_explicit_independent_source_entry(
            target=str(args.dataset),
            source=explicit_source,
        )
        cfg_path = Path("(none)")
        selector_name = "explicit_t_pics_gated_source"
    else:
        cfg_path = Path(str(args.t_pics_source_config))
        if not cfg_path.is_absolute():
            cfg_path = (_REPO_ROOT / cfg_path).resolve()
        cfg = load_frozen_transfer_config(cfg_path)
        entry = selected_source_for_target(str(args.dataset), config=cfg)
        cfg_path = cfg.path
        selector_name = cfg.selector_name
    layout = run_layout(Path(run_root))
    for key in ("control", "transfer", "gate", "selected"):
        layout[key].mkdir(parents=True, exist_ok=True)
    if independent:
        layout["source_population"].mkdir(parents=True, exist_ok=True)
        live_best, live_ll = _ensure_gated_independent_source_population(
            args=args,
            source_dataset=str(entry.selected_source),
            output_dir=layout["source_population"],
            client=client,
            wandb_module=wandb_module,
            psych_dataset_split=psych_dataset_split,
            filter_mixed_gambles=filter_mixed_gambles,
        )
        source_rank1 = Path(live_best)
        source_best_loglik = live_ll
    else:
        source_rank1 = Path(entry.rank1_program)
        source_best_loglik = None

    seed_path = str(args.seed_path or default_seed_path(str(args.dataset)))
    control_argv = build_control_argv(
        dataset=str(args.dataset),
        output_dir=str(layout["control"]),
        seed_path=seed_path,
    )
    transfer_argv = build_transfer_argv(
        dataset=str(args.dataset),
        output_dir=str(layout["transfer"]),
        seed_path=seed_path,
        source_program=str(source_rank1),
        source_dataset=entry.selected_source,
    )
    _write_arm_intended_argv(layout["control"] / "INTENDED_ARGV.txt", control_argv)
    _write_arm_intended_argv(layout["transfer"] / "INTENDED_ARGV.txt", transfer_argv)
    control_src = argv_source_conditioning_flags(control_argv)
    if control_src:
        raise RuntimeError(f"control argv leaked source-conditioning flags: {control_src}")
    print(
        f"[T-PICS gated] target={args.dataset} selected_source={entry.selected_source} "
        f"rank-1={source_rank1} config={cfg_path} "
        f"g1={'live independent' if independent else 'frozen YAML reuse'}"
        + (
            f" explicit_source={explicit_source}"
            if explicit_source
            else ""
        )
    )
    print(
        f"[T-PICS gated] G.2 sequential matched arms "
        f"(global_iters={args.global_iters}, shared automatic target prompt); "
        f"gate={GATE_NAME} field={GATE_SCORE_FIELD} tolerance={GATE_TIE_TOLERANCE}"
    )

    transfer_suffix = _cross_task_source_suffix(
        source_dataset=str(entry.selected_source),
        program_path=str(source_rank1),
        args=args,
        psych_dataset_split=psych_dataset_split,
        filter_mixed_gambles=filter_mixed_gambles,
        best_loglik=source_best_loglik,
        require_source_examples=True,
    )
    freeze_path = Path(run_root) / G2_PAIRED_PACK_FILENAME
    freeze_payload = _ensure_g2_paired_pack_freeze(
        args=args,
        participants=participants,
        seed_program_path=seed_program_path,
        run_prompts_dir=run_prompts_dir,
        psych_dataset_split=psych_dataset_split,
        filter_mixed_gambles=filter_mixed_gambles,
        source_suffix=transfer_suffix,
        freeze_path=freeze_path,
        source_dataset=str(entry.selected_source),
        source_rank1=source_rank1,
    )
    gate_freeze = layout["gate"] / G2_PAIRED_PACK_FILENAME
    if not gate_freeze.is_file() or gate_freeze.resolve() != freeze_path.resolve():
        write_paired_pack_freeze(gate_freeze, freeze_payload)
    freeze_path_str = str(freeze_path)

    expected_iters = int(args.global_iters)
    if hasattr(wandb_module, "set_g2_arm"):
        wandb_module.set_g2_arm("control")
    control_ok = population_arm_is_complete(
        layout["control"], expected_global_iters=expected_iters
    )
    if control_ok:
        print(f"[T-PICS gated] skip complete control G.2 -> {layout['control']}")
        control_pool = _load_gated_arm_pool(layout["control"])
        if hasattr(wandb_module, "maybe_backfill_g2_arm"):
            wandb_module.maybe_backfill_g2_arm("control", layout["control"])
    else:
        control_pool = run_global_evolution_phase(
            **_gated_global_phase_kwargs(
                args,
                participants=participants,
                seed_program_path=seed_program_path,
                client=client,
                wandb_module=wandb_module,
                psych_dataset_split=psych_dataset_split,
                filter_mixed_gambles=filter_mixed_gambles,
                run_prompts_dir=run_prompts_dir,
                output_dir=layout["control"],
                prompt_suffix=None,
                g2_arm="control",
                g2_paired_pack_path=freeze_path_str,
            )
        )
        control_ok = population_arm_is_complete(
            layout["control"], expected_global_iters=expected_iters
        )
    if not control_ok or not control_pool:
        raise RuntimeError(
            f"T-PICS gated control arm failed to produce a complete G.2 pool: {layout['control']}"
        )

    if hasattr(wandb_module, "set_g2_arm"):
        wandb_module.set_g2_arm("transfer")
    transfer_ok = population_arm_is_complete(
        layout["transfer"], expected_global_iters=expected_iters
    )
    transfer_program_ok = False
    transfer_failed = False
    if transfer_ok:
        print(f"[T-PICS gated] skip complete transfer G.2 -> {layout['transfer']}")
        transfer_program_ok, _ = program_has_valid_choose(layout["transfer_rank1"])
        if hasattr(wandb_module, "maybe_backfill_g2_arm"):
            wandb_module.maybe_backfill_g2_arm("transfer", layout["transfer"])
        try:
            _load_gated_arm_pool(layout["transfer"])
        except (FileNotFoundError, ValueError):
            transfer_ok = False
            transfer_failed = True
    else:
        try:
            print(
                "[T-PICS gated] transfer G.2 prompt includes "
                f"{'live independent' if independent else 'frozen YAML'} rank-1 "
                f"({entry.selected_source}: {source_rank1})"
            )
            run_global_evolution_phase(
                **_gated_global_phase_kwargs(
                    args,
                    participants=participants,
                    seed_program_path=seed_program_path,
                    client=client,
                    wandb_module=wandb_module,
                    psych_dataset_split=psych_dataset_split,
                    filter_mixed_gambles=filter_mixed_gambles,
                    run_prompts_dir=run_prompts_dir,
                    output_dir=layout["transfer"],
                    prompt_suffix=transfer_suffix,
                    g2_arm="transfer",
                    g2_paired_pack_path=freeze_path_str,
                )
            )
            transfer_ok = population_arm_is_complete(
                layout["transfer"], expected_global_iters=expected_iters
            )
            transfer_program_ok, _ = program_has_valid_choose(layout["transfer_rank1"])
        except Exception as exc:
            transfer_failed = True
            transfer_ok = False
            transfer_program_ok = False
            fail_path = layout["transfer"] / "FAILED.json"
            fail_path.write_text(
                json.dumps({"error": str(exc), "type": type(exc).__name__}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            print(f"[T-PICS gated] transfer arm failed; selecting control. {exc}")

    control_score = None
    transfer_score = None
    if control_ok:
        control_score = evaluate_pooled_train_val_loglik(
            layout["control_rank1"],
            dataset=str(args.dataset),
            participant_ids=participants,
            split_ratio=float(args.split_ratio),
            split_seed=int(args.split_seed),
            psych_dataset_split=psych_dataset_split,
            filter_mixed_gambles=filter_mixed_gambles,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
            n_eval_seeds=int(args.n_eval_seeds),
            limited_data_protocol=str(args.limited_data_protocol),
            limited_train_val=args.limited_train_val,
            max_observed_trials_per_participant=args.max_observed_trials_per_participant,
            speekenbrink_split=str(getattr(args, "speekenbrink_split", "chronological")),
        )
    if control_score is None or not math.isfinite(float(control_score)):
        raise RuntimeError(
            f"T-PICS gated control arm did not produce a finite {GATE_SCORE_FIELD}: "
            f"{control_score} ({layout['control_rank1']})"
        )
    if transfer_ok and transfer_program_ok:
        transfer_score = evaluate_pooled_train_val_loglik(
            layout["transfer_rank1"],
            dataset=str(args.dataset),
            participant_ids=participants,
            split_ratio=float(args.split_ratio),
            split_seed=int(args.split_seed),
            psych_dataset_split=psych_dataset_split,
            filter_mixed_gambles=filter_mixed_gambles,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
            n_eval_seeds=int(args.n_eval_seeds),
            limited_data_protocol=str(args.limited_data_protocol),
            limited_train_val=args.limited_train_val,
            max_observed_trials_per_participant=args.max_observed_trials_per_participant,
            speekenbrink_split=str(getattr(args, "speekenbrink_split", "chronological")),
        )

    decision = decide_gate(
        control_score=control_score,
        transfer_score=transfer_score,
        control_arm_ok=control_ok,
        transfer_arm_ok=transfer_ok and not transfer_failed,
        transfer_program_ok=transfer_program_ok,
        tie_tolerance=GATE_TIE_TOLERANCE,
    )
    retained_pool = (
        layout["transfer_pool"] if decision.selected_arm == "transfer" else layout["control_pool"]
    )
    transfer_rank1 = layout["transfer_rank1"] if layout["transfer_rank1"].is_file() else None
    record = gate_record_payload(
        target=str(args.dataset),
        selected_source=entry.selected_source,
        decision=decision,
        control_rank1=layout["control_rank1"],
        transfer_rank1=transfer_rank1,
        retained_pool=retained_pool,
        config_path=cfg_path,
        selector_name=selector_name,
        selected_source_rank1=source_rank1,
    )
    write_gate_record(layout["gate_record"], record)
    if hasattr(wandb_module, "publish_gate_record"):
        wandb_module.publish_gate_record(layout["gate_record"])
    _merge_run_metadata(
        Path(run_root),
        {
            "gate_record_path": str(layout["gate_record"]),
            "gate": record,
            "pooled_train_val_loglik_control": decision.control_score,
            "pooled_train_val_loglik_transfer": decision.transfer_score,
            "gate_reason": decision.reason,
            "selected_arm": decision.selected_arm,
            "retained_pool_path": str(retained_pool),
            "g2_arms_shared_target_prompt": True,
            "g2_paired_pack_path": str(freeze_path),
            "g2_paired_pack_version": G2_PAIRED_PACK_VERSION,
            "g2_paired_n_examples_included": freeze_payload.get("n_examples_included"),
            "g2_paired_example_ids": freeze_payload.get("example_ids"),
        },
    )
    print(
        f"[T-PICS gated] gate selected={decision.selected_arm} reason={decision.reason} "
        f"control_{GATE_SCORE_FIELD}={decision.control_score} "
        f"transfer_{GATE_SCORE_FIELD}={decision.transfer_score} "
        f"diff={decision.score_difference}"
    )

    selected_link = layout["selected_pool"]
    if selected_link.exists() or selected_link.is_symlink():
        if selected_link.is_symlink() or selected_link.is_file():
            selected_link.unlink()
        else:
            shutil.rmtree(selected_link)
    try:
        selected_link.symlink_to(retained_pool.resolve())
    except OSError:
        shutil.copytree(retained_pool, selected_link)
    (layout["selected"] / "SELECTED_ARM.txt").write_text(
        decision.selected_arm + "\n", encoding="utf-8"
    )
    return _load_one_initial_pool_dir(str(retained_pool))


def _refinement_pool_best_metrics(
    elite_parents: List[Tuple[Any, ...]],
    elite_val_logliks: List[Optional[float]],
    test_trials: List[Dict[str, Any]],
    *,
    dataset: str,
    n_eval_seeds: int,
) -> Dict[str, Any]:
    """Metrics for elite pool rank-1 after an iteration (includes held-out test loglik)."""
    pool_best = elite_parents[0]
    program_id = pool_best[3]
    train_loglik = _train_loglik_from_elite_tuple(pool_best)
    val_loglik = (
        float(elite_val_logliks[0]) if elite_val_logliks and elite_val_logliks[0] is not None else None
    )
    fitness = float(pool_best[1])
    test_loglik: Optional[float] = None
    choose_fn = compile_program(pool_best[0])
    if choose_fn is not None:
        test_eval = _evaluate_loglik_for_dataset(
            dataset, choose_fn, test_trials, n_seeds=n_eval_seeds
        )
        test_loglik = float(test_eval["avg_loglik"])
    return {
        "pool_best_program_id": program_id,
        "pool_best_train_val_loglik": fitness,
        "pool_best_train_loglik": train_loglik,
        "pool_best_val_loglik": val_loglik,
        "pool_best_test_loglik": test_loglik,
    }


def _load_refinement_prompt_suffix(
    dataset: str,
    *,
    run_prompts_dir: Optional[str] = None,
) -> str:
    """Suffix appended to the dataset prompt during log-likelihood refinement."""
    if run_prompts_dir:
        path = Path(run_prompts_dir) / "refine.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Refinement prompt not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    # Fallback: legacy choice13k refine prompt (repo template).
    path = (
        _REPO_ROOT
        / "prompts"
        / "Template_evo"
        / "choice13k"
        / "refine"
        / "infer_single_choice.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Refinement prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _prompt_block_key(trial: Dict[str, Any]) -> Any:
    """Stable key for a prompt 'block' (problem / CPC18 block / gamble signature)."""
    if "block_id" in trial and "problem_id" in trial:
        return ("block", trial["problem_id"], trial["block_id"])
    if "problem_id" in trial:
        return ("problem_id", trial["problem_id"])
    if "problem_signature" in trial:
        return ("problem_signature", trial["problem_signature"])
    p = trial.get("problem", {})
    ga = p.get("gamble_A", {})
    gb = p.get("gamble_B", {})
    ga_probs = ga.get("probs", []) or []
    gb_probs = gb.get("probs", []) or []
    return (
        "problem_sig",
        tuple(ga.get("rewards", [])),
        tuple(ga_probs),
        tuple(gb.get("rewards", [])),
        tuple(gb_probs),
    )


def _group_trials_by_prompt_block(
    trials: List[Dict[str, Any]],
) -> Tuple[List[Any], Dict[Any, List[Dict[str, Any]]]]:
    """Group trials by block key, preserving first-seen block order."""
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    order: List[Any] = []
    for t in trials:
        key = _prompt_block_key(t)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(t)
    return order, grouped


def trace_prompt_trial_sampling(
    trials: List[Dict[str, Any]],
    *,
    max_trials: int,
    max_trials_per_problem: int,
    subsample_seed: int,
    label: str = "train",
) -> Dict[str, Any]:
    """Dry-run trace of block/group sampling (same logic as ``_cap_and_subsample_prompt_trials``)."""
    trials = list(trials)
    n_before = len(trials)
    block_order, grouped = _group_trials_by_prompt_block(trials)
    trace: Dict[str, Any] = {
        "label": label,
        "train_trials_before": n_before,
        "max_trials": max_trials,
        "max_trials_per_problem": max_trials_per_problem,
        "subsample_seed": subsample_seed,
        "n_prompt_groups": len(block_order),
        "group_sizes": [len(grouped[k]) for k in block_order],
        "selected_group_indices": [],
        "trials_per_selected_group": [],
        "trials_from_block_sample": 0,
        "n_extra_from_remainder": 0,
        "n_top_up": 0,
        "final_count": n_before if max_trials <= 0 else min(n_before, max_trials),
    }
    if max_trials <= 0 or not trials:
        return trace

    if max_trials_per_problem <= 0:
        trace["mode"] = "flat"
        trace["final_count"] = min(n_before, max_trials)
        return trace

    per_block = int(max_trials_per_problem)
    n_blocks_target = max_trials // per_block
    n_extra = max_trials % per_block
    trace.update(
        {
            "mode": "block_then_top_up",
            "n_blocks_target": n_blocks_target,
            "n_extra_slots": n_extra,
        }
    )

    rng = np.random.default_rng(subsample_seed)
    n_blocks = len(block_order)
    used_ids: Set[int] = set()
    out_n = 0

    if n_blocks_target > 0 and n_blocks > 0:
        n_blocks_sample = min(n_blocks_target, n_blocks)
        block_idxs = rng.choice(n_blocks, size=n_blocks_sample, replace=False)
        trace["selected_group_indices"] = [int(i) for i in sorted(block_idxs)]
        for bi in sorted(int(i) for i in block_idxs):
            block_trials = grouped[block_order[bi]]
            if len(block_trials) <= per_block:
                picked = block_trials
            else:
                tidx = rng.choice(len(block_trials), size=per_block, replace=False)
                picked = [block_trials[int(j)] for j in sorted(tidx)]
            trace["trials_per_selected_group"].append(len(picked))
            for t in picked:
                if id(t) not in used_ids:
                    used_ids.add(id(t))
                    out_n += 1
    trace["trials_from_block_sample"] = out_n

    if n_extra > 0:
        remaining = [t for t in trials if id(t) not in used_ids]
        if remaining:
            n_pick = min(n_extra, len(remaining))
            trace["n_extra_from_remainder"] = n_pick
            out_n += n_pick

    if out_n < max_trials:
        remaining = [t for t in trials if id(t) not in used_ids]
        trace["n_top_up"] = min(max_trials - out_n, len(remaining))
        out_n += trace["n_top_up"]

    trace["final_count"] = min(n_before, out_n)
    return trace


def _print_prompt_sampling_trace(trace: Dict[str, Any]) -> None:
    print(f"[LLM prompt trace] {trace.get('label', 'train')}:")
    print(f"  train_trials_before: {trace.get('train_trials_before')}")
    print(f"  max_trials (CLI --max_prompt_train_trials): {trace.get('max_trials')}")
    print(f"  max_trials_per_problem (CLI): {trace.get('max_trials_per_problem')}")
    print(f"  subsample_seed: {trace.get('subsample_seed')}")
    print(f"  n_prompt_groups (_prompt_block_key): {trace.get('n_prompt_groups')}")
    if trace.get("group_sizes"):
        print(f"  group_sizes: {trace.get('group_sizes')}")
    if trace.get("mode") == "block_then_top_up":
        print(
            f"  block budget: n_blocks_target={trace.get('n_blocks_target')} "
            f"(max_trials // per_problem), n_extra_slots={trace.get('n_extra_slots')}"
        )
        print(f"  selected_group_indices: {trace.get('selected_group_indices')}")
        print(f"  trials_per_selected_group: {trace.get('trials_per_selected_group')}")
        print(f"  trials_from_block_sample: {trace.get('trials_from_block_sample')}")
        print(f"  n_extra_from_remainder: {trace.get('n_extra_from_remainder')}")
        print(f"  n_top_up (fill to max_trials): {trace.get('n_top_up')}")
    print(f"  final_count: {trace.get('final_count')}")


def _cap_and_subsample_prompt_trials(
    trials: List[Dict[str, Any]],
    *,
    max_trials: int,
    max_trials_per_problem: int,
    subsample_seed: int,
    label: str,
    debug_sampling: bool = False,
) -> List[Dict[str, Any]]:
    """Select trials for LLM prompts: sample blocks then optional extra singles.

    When ``max_trials_per_problem`` > 0, sample ``max_trials // max_trials_per_problem`` prompt
    groups (``_prompt_block_key``), up to ``max_trials_per_problem`` trials each, then
    ``max_trials % max_trials_per_problem`` extra trials from the remainder, then **top up**
    from the remainder until ``max_trials`` is reached (if enough trials remain). Groups are
    often smaller than ``max_trials_per_problem`` (e.g. Psych-101 gamble signatures with 5
    trials each), so block-only sampling can under-fill the global cap without top-up.
    When ``max_trials_per_problem`` is 0, sample up to ``max_trials`` trials with no block
    structure. ``max_trials`` <= 0 disables capping.
    """
    trials = list(trials)
    if max_trials <= 0:
        return trials
    if not trials:
        return []

    if using_v2_prompt_contract():
        alias = None
        for t in trials:
            p = t.get("problem") or {}
            if p.get("dataset_alias"):
                alias = str(p["dataset_alias"])
                break
        if alias is None:
            raise ValueError("v2 prompt selection requires dataset_alias on trials")
        pids = {
            t.get("_prompt_participant_id")
            for t in trials
            if t.get("_prompt_participant_id") is not None
        }
        selected, diag = select_structure_aware_prompt_examples(
            trials,
            dataset=alias,
            max_trials=max_trials,
            subsample_seed=subsample_seed,
            pooled=len(pids) > 1,
        )
        print(
            f"[LLM prompt v2] Using {len(selected)} of {len(trials)} {label} trials "
            f"(structure-aware, max={max_trials}, seed={subsample_seed}, "
            f"mode={diag.get('mode')}, people={diag.get('n_participants')})."
        )
        return selected

    rng = np.random.default_rng(subsample_seed)
    orig_index = {id(t): i for i, t in enumerate(trials)}

    def _sort_chronological(selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(selected, key=lambda t: orig_index[id(t)])

    if max_trials_per_problem <= 0:
        if len(trials) <= max_trials:
            return trials
        idx = rng.choice(len(trials), size=max_trials, replace=False)
        out = [trials[int(i)] for i in sorted(idx)]
        print(
            f"[LLM prompt] Using {len(out)} of {len(trials)} {label} trials "
            f"(flat subsample, max={max_trials}, seed={subsample_seed})."
        )
        return out

    per_block = int(max_trials_per_problem)
    n_blocks_target = max_trials // per_block
    n_extra = max_trials % per_block

    block_order, grouped = _group_trials_by_prompt_block(trials)
    n_blocks = len(block_order)
    if n_blocks == 0:
        return []

    out: List[Dict[str, Any]] = []
    used_ids: Set[int] = set()
    n_blocks_sampled = 0

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
            n_blocks_sampled += 1

    if n_extra > 0:
        remaining = [t for t in trials if id(t) not in used_ids]
        if remaining:
            n_pick = min(n_extra, len(remaining))
            ridx = rng.choice(len(remaining), size=n_pick, replace=False)
            for j in sorted(int(i) for i in ridx):
                t = remaining[j]
                out.append(t)
                used_ids.add(id(t))

    top_up_n = 0
    if len(out) < max_trials:
        remaining = [t for t in trials if id(t) not in used_ids]
        if remaining:
            n_pick = min(max_trials - len(out), len(remaining))
            ridx = rng.choice(len(remaining), size=n_pick, replace=False)
            for j in sorted(int(i) for i in ridx):
                t = remaining[j]
                out.append(t)
                used_ids.add(id(t))
            top_up_n = n_pick

    out = _sort_chronological(out)
    extra_note = f", +{n_extra} extra trial(s)" if n_extra else ""
    top_note = f", +{top_up_n} top-up trial(s)" if top_up_n else ""
    print(
        f"[LLM prompt] Using {len(out)} of {len(trials)} {label} trials "
        f"({n_blocks} prompt group(s); {n_blocks_sampled} sampled x up to {per_block}"
        f"{extra_note}{top_note}, max={max_trials}, seed={subsample_seed})."
    )
    if debug_sampling or os.environ.get("TEH_DEBUG_PROMPT_SAMPLING"):
        _print_prompt_sampling_trace(
            trace_prompt_trial_sampling(
                trials,
                max_trials=max_trials,
                max_trials_per_problem=max_trials_per_problem,
                subsample_seed=subsample_seed,
                label=label,
            )
        )
    return out


def _cap_prompt_train_and_val_trials(
    train_trials: List[Dict[str, Any]],
    val_trials: Optional[List[Dict[str, Any]]],
    *,
    max_trials: int,
    max_trials_per_problem: int,
    subsample_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Cap trials for prompts: ``max_trials`` applies to the union when val is present."""
    train_list = list(train_trials)
    val_list = list(val_trials) if val_trials else []
    if not val_list:
        capped_train = _cap_and_subsample_prompt_trials(
            train_list,
            max_trials=max_trials,
            max_trials_per_problem=max_trials_per_problem,
            subsample_seed=subsample_seed,
            label="train",
        )
        return capped_train, []
    train_ids = {id(t) for t in train_list}
    pooled = train_list + val_list
    capped = _cap_and_subsample_prompt_trials(
        pooled,
        max_trials=max_trials,
        max_trials_per_problem=max_trials_per_problem,
        subsample_seed=subsample_seed,
        label="train+validation (union)",
    )
    capped_train = [t for t in capped if id(t) in train_ids]
    capped_val = [t for t in capped if id(t) not in train_ids]
    print(
        f"[LLM prompt] Union cap split: {len(capped)} total "
        f"({len(capped_train)} train, {len(capped_val)} validation) "
        f"from {len(train_list)} train + {len(val_list)} validation "
        f"(max={max_trials}, seed={subsample_seed})."
    )
    return capped_train, capped_val


_PROMPT_DIAGNOSTICS_LOCK = threading.Lock()
_TRAIN_TRIAL_CAP_STEPS = (40, 30, 20, 10, 5)
_VAL_TRIAL_CAP_STEPS = (40, 30, 20, 10, 5)
_PER_PROBLEM_CAP_STEPS = (10, 5, 3, 1)
_MIN_VAL_TRIALS_REFINEMENT = 3
_MIN_TRAIN_TRIALS_FINAL = 5
_MIN_VAL_TRIALS_FINAL = 3


class PromptBudgetExceededError(RuntimeError):
    """Prompt still exceeds hard_prompt_token_cap after structured truncation."""

    def __init__(
        self,
        message: str,
        *,
        tokens: int,
        cap: int,
        overflow_components: Dict[str, int],
        truncation_steps: List[str],
    ):
        super().__init__(message)
        self.tokens = tokens
        self.cap = cap
        self.overflow_components = overflow_components
        self.truncation_steps = truncation_steps


def _v2_prompt_budget_estimator_name() -> str:
    return "qwen_chat" if using_v2_prompt_contract() else "char4"


def estimate_tokens(text: str, *, estimator: str = "char4") -> int:
    """Prompt-budget token count.

    v1 uses char4 (ceil(len/4)). Under the v2 prompt contract, use the real
    Qwen chat tokenizer on chat-templated system+user so the existing
    40→30→20→10→5 trial-count truncation loop matches vLLM's input size.
    Compact one-liners are a separate v1 last resort and are not a v2 format change.
    """
    if not text:
        return 0
    if using_v2_prompt_contract():
        from utils.teh.prompt_units import qwen_user_prompt_token_count

        return qwen_user_prompt_token_count(text)
    if estimator == "char4":
        return math.ceil(len(text) / 4)
    return math.ceil(len(text) / 4)


def _compress_prompt_whitespace(text: str) -> str:
    if not text:
        return ""
    lines: List[str] = []
    prev_blank = False
    for ln in text.splitlines():
        stripped = ln.rstrip()
        blank = not stripped
        if blank and prev_blank:
            continue
        lines.append(stripped)
        prev_blank = blank
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _prompt_example_split_diagnostic_fields(
    *,
    train_before: int,
    train_after: int,
    val_before: int,
    val_after: int,
) -> Dict[str, Any]:
    """Record which observed splits were serialized into an LLM candidate prompt.

    Test is never a prompt-example source; the flag is stored so audits can prove it.
    """
    n_train_b = int(train_before)
    n_train_a = int(train_after)
    n_val_b = int(val_before)
    n_val_a = int(val_after)
    return {
        "train_trials_before": n_train_b,
        "train_trials_after": n_train_a,
        "val_trials_before": n_val_b,
        "val_trials_after": n_val_a,
        "observed_union_trials_before": n_train_b + n_val_b,
        "observed_union_trials_after": n_train_a + n_val_a,
        "test_examples_included": False,
    }


def _append_prompt_diagnostic(record: Dict[str, Any], diagnostics_dir: Optional[Path]) -> None:
    if diagnostics_dir is None:
        return
    path = Path(diagnostics_dir) / "prompt_diagnostics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("timestamp", datetime.now().isoformat())
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _PROMPT_DIAGNOSTICS_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _strip_parent_code_for_prompt(code: str) -> str:
    from utils.teh.prompt_sanitize import _strip_python_comments

    return _strip_python_comments(code)


def format_trials_to_text_compact(
    trials: List[Dict[str, Any]], dataset: str = "choice13k"
) -> str:
    """Compact trial serialization preserving gamble fields, action, feedback, split."""
    if not trials:
        return ""
    lines: List[str] = []
    for idx, t in enumerate(trials):
        prob = t.get("problem", {})
        action = t.get("action")
        split_lbl = t.get("split") or t.get("split_label")
        parts = [str(idx + 1)]
        if "gamble_A" in prob and "gamble_B" in prob:
            ga, gb = prob["gamble_A"], prob["gamble_B"]
            parts.append(f"A:p={ga.get('probs')},r={ga.get('rewards')}")
            parts.append(f"B:p={gb.get('probs')},r={gb.get('rewards')}")
            if prob.get("has_feedback") is not None:
                parts.append(f"fb={prob.get('has_feedback')}")
        elif dataset == "cpc18" or "Ha" in prob:
            parts.append(
                f"A:Ha={prob.get('Ha')},pHa={prob.get('pHa')},La={prob.get('La')};"
                f"B:Hb={prob.get('Hb')},pHb={prob.get('pHb')},Lb={prob.get('Lb')};"
                f"Amb={prob.get('Amb')},Corr={prob.get('Corr')}"
            )
        else:
            psych_alias = prob.get("dataset_alias")
            if is_psych101_dataset(dataset) or (
                psych_alias and is_psych101_dataset(str(psych_alias))
            ) or prob.get("schema_type") in ("A", "B", "C", "D"):
                from data_modules.psych101_binary import format_trial_for_prompt

                line = format_trial_for_prompt(t, idx + 1)
                line = re.sub(r"\s+", " ", line).strip()
                lines.append(line)
                continue
        if action is not None:
            parts.append(f"act={action}")
        hist = t.get("history") or []
        if hist:
            last = hist[-1]
            if last.get("feedback") is not None:
                parts.append(f"last_fb={last.get('feedback')}")
        if split_lbl is not None:
            parts.append(f"split={split_lbl}")
        lines.append("|".join(parts))
    return "\n".join(lines)


def _serialize_trials_for_prompt(
    trials: List[Dict[str, Any]],
    *,
    dataset: str,
    compact: bool,
) -> str:
    if using_v2_prompt_contract():
        # Keep snapshot JSON. v2 over-budget handling is trial-count caps in
        # _truncate_psych_prompt_to_budget, not rewriting examples as one-liners.
        return format_snapshot_examples(trials)
    if compact:
        return format_trials_to_text_compact(trials, dataset=dataset)
    return format_trials_to_text(trials, dataset=dataset)


def _component_token_breakdown(
    parts: Dict[str, str], *, estimator: str
) -> Dict[str, int]:
    return {k: estimate_tokens(v, estimator=estimator) for k, v in parts.items()}


def _enforce_prompt_budget(
    prompt_text: str,
    *,
    hard_prompt_token_cap: int,
    strict_prompt_budget: bool,
    prompt_token_estimator: str,
    overflow_components: Dict[str, int],
    truncation_steps: List[str],
    phase: str,
    participant_id: Optional[int],
    iteration: Optional[int],
    candidate_index: Optional[int],
    diagnostics_dir: Optional[Path],
    diagnostics_base: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Return prompt if within cap; otherwise skip LLM or raise."""
    tokens_after = estimate_tokens(prompt_text, estimator=prompt_token_estimator)
    diag = dict(diagnostics_base)
    diag.update(
        {
            "prompt_tokens_after_truncation": tokens_after,
            "hard_prompt_token_cap": hard_prompt_token_cap,
            "truncation_steps": list(truncation_steps),
        }
    )
    if tokens_after <= hard_prompt_token_cap:
        diag["status"] = "ok"
        _append_prompt_diagnostic(diag, diagnostics_dir)
        return prompt_text, diag

    diag["status"] = "overflow_after_truncation"
    diag["overflow_components"] = overflow_components
    _append_prompt_diagnostic(diag, diagnostics_dir)
    pid = participant_id if participant_id is not None else "?"
    iter_s = iteration if iteration is not None else "?"
    cand_s = candidate_index if candidate_index is not None else "?"
    msg = (
        f"Prompt budget exceeded after truncation: {tokens_after} tokens > cap "
        f"{hard_prompt_token_cap} (phase={phase}, participant={pid}, iteration={iter_s}, "
        f"candidate={cand_s}). Largest components: "
        + ", ".join(
            f"{k}={v}"
            for k, v in sorted(overflow_components.items(), key=lambda x: -x[1])[:6]
        )
    )
    if strict_prompt_budget:
        raise PromptBudgetExceededError(
            msg,
            tokens=tokens_after,
            cap=hard_prompt_token_cap,
            overflow_components=overflow_components,
            truncation_steps=truncation_steps,
        )
    print(f"Warning: {msg}; skipping LLM call.")
    return "", diag


def _warn_prompt_truncation(diag: Dict[str, Any]) -> None:
    if diag.get("truncated"):
        print(
            f"Warning: LLM prompt truncated "
            f"({diag.get('prompt_tokens_before_truncation')} -> "
            f"{diag.get('prompt_tokens_after_truncation')} tokens, "
            f"cap={diag.get('hard_prompt_token_cap')})."
        )
    if diag.get("prompt_tokens_after_truncation", 0) > diag.get("hard_prompt_token_cap", 0):
        print(
            "Warning: prompt still exceeds hard_prompt_token_cap after truncation "
            f"({diag.get('prompt_tokens_after_truncation')} > {diag.get('hard_prompt_token_cap')})."
        )
    if diag.get("train_trials_after", 999) < 10:
        print(
            f"Warning: train/obs trials in prompt reduced to {diag.get('train_trials_after')} (<10)."
        )
    if diag.get("parents_after") == 1 and diag.get("parents_before", 1) > 1:
        print("Warning: parent programs reduced to 1 for LLM prompt.")
    if diag.get("compact_serialization"):
        print("Warning: compact trial serialization used for LLM prompt.")


def _build_psych_prompt_text(
    *,
    base_prompt: str,
    state_text: str,
    extra_state_text: str,
    parent_context: str,
    code_template_suffix: str,
    candidate_output_rules: str,
    runtime_contract: str = "",
) -> str:
    text = (
        f"{base_prompt}\n{state_text}{extra_state_text}\n{parent_context}"
        f"{code_template_suffix}\n{candidate_output_rules}\n"
    )
    if runtime_contract and str(runtime_contract).strip():
        text = attach_runtime_contract_to_prompt(text, str(runtime_contract).strip())
    return text


def _truncate_psych_prompt_to_budget(
    *,
    base_prompt: str,
    train_trials: List[Dict[str, Any]],
    train_trials_source: List[Dict[str, Any]],
    val_trials: Optional[List[Dict[str, Any]]],
    val_trials_source: Optional[List[Dict[str, Any]]],
    extra_prompt_trials_label: str,
    parent_programs: List[str],
    parent_context_builder: Callable[..., str],
    parent_context_kwargs: Dict[str, Any],
    code_template_suffix: str,
    candidate_output_rules: str,
    dataset: str,
    dataset_type: str,
    hard_prompt_token_cap: int,
    prompt_token_estimator: str,
    max_prompt_train_trials: int,
    max_prompt_trials_per_problem: int,
    prompt_train_trials_seed: int,
    max_parent_chars: int,
    refinement_val_observations: bool,
    pre_capped_train: bool,
    pre_capped_val: bool,
    runtime_contract: str = "",
    freeze_examples: bool = False,
    output_reserve: int = OUTPUT_RESERVE,
) -> Tuple[str, Dict[str, Any], List[str]]:
    """
    Structured truncation for Psych/TEH prompts. Returns (prompt, diagnostics, steps).
    """
    steps: List[str] = []
    compact = False
    out_reserve = int(output_reserve)
    use_shared_trial_budget = (
        val_trials_source is not None and not refinement_val_observations
    )
    effective_max_train = max_prompt_train_trials
    effective_max_val = max_prompt_train_trials
    effective_max_total = max_prompt_train_trials
    effective_per_problem = max_prompt_trials_per_problem

    base = base_prompt
    parents = list(parent_programs)
    n_parents_before = len(parents)
    shared_cap_cache: Dict[Tuple[int, int], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}

    def _shared_capped_trials() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        key = (effective_max_total, effective_per_problem)
        if key not in shared_cap_cache:
            shared_cap_cache[key] = _cap_prompt_train_and_val_trials(
                train_trials_source,
                val_trials_source,
                max_trials=effective_max_total,
                max_trials_per_problem=effective_per_problem,
                subsample_seed=prompt_train_trials_seed,
            )
        return shared_cap_cache[key]

    def _cap_train() -> List[Dict[str, Any]]:
        if pre_capped_train:
            return list(train_trials)
        if use_shared_trial_budget:
            return _shared_capped_trials()[0]
        return _cap_and_subsample_prompt_trials(
            train_trials_source,
            max_trials=effective_max_train,
            max_trials_per_problem=effective_per_problem,
            subsample_seed=prompt_train_trials_seed,
            label="train" if not refinement_val_observations else "validation",
        )

    def _cap_val() -> List[Dict[str, Any]]:
        if val_trials_source is None:
            return []
        if pre_capped_val:
            return list(val_trials or [])
        if use_shared_trial_budget:
            return _shared_capped_trials()[1]
        return _cap_and_subsample_prompt_trials(
            val_trials_source,
            max_trials=effective_max_val,
            max_trials_per_problem=effective_per_problem,
            subsample_seed=int(prompt_train_trials_seed) + 424_242,
            label="validation",
        )

    def _assemble() -> Tuple[str, str, str, int, int, int]:
        tr = _cap_train()
        st = _serialize_trials_for_prompt(tr, dataset=dataset_type, compact=compact)
        state_text = st if st else ""
        extra_state_text = ""
        n_val = 0
        if val_trials_source is not None:
            vr = _cap_val()
            n_val = len(vr)
            if vr:
                extra_state_text = (
                    f"\n\n{extra_prompt_trials_label}:\n"
                    f"{_serialize_trials_for_prompt(vr, dataset=dataset_type, compact=compact)}\n"
                )
        pctx = parent_context_builder(
            prompt_parent_programs=parents,
            **parent_context_kwargs,
        )
        prompt = _build_psych_prompt_text(
            base_prompt=base,
            state_text=state_text,
            extra_state_text=extra_state_text,
            parent_context=pctx,
            code_template_suffix=code_template_suffix,
            candidate_output_rules=candidate_output_rules,
            runtime_contract=runtime_contract,
        )
        return prompt, state_text, extra_state_text, len(tr), n_val, len(parents)

    prompt, _, _, n_train, n_val, n_parents = _assemble()
    tokens_before = estimate_tokens(prompt, estimator=prompt_token_estimator)

    def _parts_dict(p: str, st: str, ex: str, pc: str) -> Dict[str, str]:
        return {
            "instruction": base,
            "train_observations": st,
            "val_observations": ex,
            "parents": parent_context_builder(
                prompt_parent_programs=parents, **parent_context_kwargs
            ),
            "template_and_output_rules": code_template_suffix + candidate_output_rules,
        }

    if tokens_before <= hard_prompt_token_cap:
        return prompt, {
            "truncated": False,
            "prompt_tokens_before_truncation": tokens_before,
            "prompt_tokens_after_truncation": tokens_before,
            "prompt_token_estimator": _v2_prompt_budget_estimator_name()
            if using_v2_prompt_contract()
            else prompt_token_estimator,
            "train_trials_before": len(train_trials_source),
            "train_trials_after": n_train,
            "val_trials_before": len(val_trials_source) if val_trials_source is not None else 0,
            "val_trials_after": n_val,
            "parents_before": n_parents_before,
            "parents_after": n_parents,
            "compact_serialization": compact,
            "freeze_examples": bool(freeze_examples),
        }, steps

    # 1) compress whitespace in instruction
    base = _compress_prompt_whitespace(base)
    steps.append("compress_instruction_whitespace")
    prompt, _, _, n_train, n_val, n_parents = _assemble()

    # 2) strip parent comments
    parents = [_strip_parent_code_for_prompt(p) for p in parents]
    steps.append("strip_parent_comments")
    prompt, _, _, n_train, n_val, n_parents = _assemble()

    # 3) whole-trial drops (monotone caps) before dropping parent copies.
    # Frozen G.2 examples skip this ladder so both arms keep the same IDs;
    # freeze already chose the max trial prefix under the transfer budget.
    if not freeze_examples:
        for cap in _TRAIN_TRIAL_CAP_STEPS:
            if estimate_tokens(prompt, estimator=prompt_token_estimator) <= hard_prompt_token_cap:
                break
            if use_shared_trial_budget:
                effective_max_total = (
                    min(effective_max_total, cap) if effective_max_total > 0 else cap
                )
            else:
                effective_max_train = (
                    min(effective_max_train, cap) if effective_max_train > 0 else cap
                )
            steps.append(f"train_trials_cap_{cap}")
            prompt, _, _, n_train, n_val, n_parents = _assemble()

        # 4) per-problem caps (when flat sampling was used, enable block caps under budget pressure)
        if max_prompt_trials_per_problem <= 0:
            for cap in _PER_PROBLEM_CAP_STEPS:
                if estimate_tokens(prompt, estimator=prompt_token_estimator) <= hard_prompt_token_cap:
                    break
                effective_per_problem = cap
                steps.append(f"per_problem_cap_{cap}")
                prompt, _, _, n_train, n_val, n_parents = _assemble()

        # 5) compact serialization (v1 only: rewrite trials as one-liners).
        # v2/v3 keep snapshot JSON and continue dropping trial counts instead.
        if (
            estimate_tokens(prompt, estimator=prompt_token_estimator) > hard_prompt_token_cap
            and not compact
            and not using_v2_prompt_contract()
        ):
            compact = True
            steps.append("compact_trial_serialization")
            prompt, _, _, n_train, n_val, n_parents = _assemble()

        # 6) val-only trial caps when train and val are capped separately
        if not use_shared_trial_budget:
            min_val = _MIN_VAL_TRIALS_REFINEMENT if refinement_val_observations else 1
            for cap in _VAL_TRIAL_CAP_STEPS:
                if val_trials_source is None:
                    break
                if estimate_tokens(prompt, estimator=prompt_token_estimator) <= hard_prompt_token_cap:
                    break
                next_cap = max(cap, min_val) if refinement_val_observations else cap
                effective_max_val = (
                    min(effective_max_val, next_cap) if effective_max_val > 0 else next_cap
                )
                steps.append(f"val_trials_cap_{effective_max_val}")
                prompt, _, _, n_train, n_val, n_parents = _assemble()

    # 7) drop extra parents (keep >=1) — after whole-trial drops when not frozen
    while len(parents) > 1 and estimate_tokens(prompt, estimator=prompt_token_estimator) > hard_prompt_token_cap:
        parents = parents[:-1]
        steps.append("drop_extra_parent")
        prompt, _, _, n_train, n_val, n_parents = _assemble()

    # 8) parent char truncation (comments already stripped)
    if max_parent_chars > 0:
        new_parents: List[str] = []
        changed = False
        for p in parents:
            tc, was = _truncate_parent_program_for_prompt(p, max_parent_chars)
            new_parents.append(tc)
            changed = changed or was
        if changed:
            parents = new_parents
            steps.append("parent_char_truncation")
            prompt, _, _, n_train, n_val, n_parents = _assemble()

    tokens_after = estimate_tokens(prompt, estimator=prompt_token_estimator)
    if tokens_after > hard_prompt_token_cap:
        # 9) final fallback. Frozen G.2 examples are never shrunk here.
        if not freeze_examples:
            if use_shared_trial_budget:
                effective_max_total = _MIN_TRAIN_TRIALS_FINAL
            else:
                effective_max_train = _MIN_TRAIN_TRIALS_FINAL
                if val_trials_source is not None:
                    effective_max_val = (
                        _MIN_VAL_TRIALS_FINAL
                        if refinement_val_observations
                        else _MIN_TRAIN_TRIALS_FINAL
                    )
            effective_per_problem = 1
            if not using_v2_prompt_contract():
                compact = True
        if len(parents) > 1:
            parents = [parents[0]]
        steps.append("final_fallback_minimal")
        prompt, _, _, n_train, n_val, n_parents = _assemble()
        tokens_after = estimate_tokens(prompt, estimator=prompt_token_estimator)

    if tokens_after > hard_prompt_token_cap or tokens_after + out_reserve > VLLM_CONTEXT:
        if freeze_examples:
            raise PairedPackingFitError(
                "G.2 paired packing: frozen target examples plus retained parents "
                f"still overflow after parent-only trim "
                f"(tokens={tokens_after}, cap={hard_prompt_token_cap}, "
                f"vllm={tokens_after}+{out_reserve}>{VLLM_CONTEXT}). "
                "Required instruction + one parent + source suffix + runtime "
                "contract cannot be reduced without changing the frozen example set."
            )

    overflow = _component_token_breakdown(
        _parts_dict(prompt, "", "", ""),
        estimator=prompt_token_estimator,
    )
    return prompt, {
        "truncated": True,
        "prompt_tokens_before_truncation": tokens_before,
        "prompt_tokens_after_truncation": tokens_after,
        "prompt_token_estimator": _v2_prompt_budget_estimator_name()
        if using_v2_prompt_contract()
        else prompt_token_estimator,
        "train_trials_before": len(train_trials_source),
        "train_trials_after": n_train,
        "val_trials_before": len(val_trials_source) if val_trials_source is not None else 0,
        "val_trials_after": n_val,
        "parents_before": n_parents_before,
        "parents_after": n_parents,
        "compact_serialization": compact,
        "freeze_examples": bool(freeze_examples),
        "overflow_components": overflow,
    }, steps


def run_loglik_refinement_phase(
    *,
    dataset: str,
    client: OpenAI,
    model_name: str,
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
    n_iterations: int,
    n_candidates_per_iteration: int,
    fresh_n_candidates: int = 0,
    sample_size: int,
    sample_parents: bool,
    sampled_parents_decay: bool = True,
    elite_pool_size: Optional[int],
    participant_id: int,
    split_ratio: float,
    split_seed: int,
    max_prompt_train_trials: int,
    max_prompt_trials_per_problem: int,
    llm_max_tokens: int,
    max_workers: int,
    n_eval_seeds: int,
    fitness_metric: str = "loglik",
    output_path: Optional[Path] = None,
    save_artifacts: bool = True,
    wandb_module: Optional[Any] = None,
    wandb_step_offset: int = 0,
    evolution_elite_parents: Optional[List[Tuple[Any, ...]]] = None,
    evolution_elite_val_logliks: Optional[List[Optional[float]]] = None,
    initial_code: Optional[str] = None,
    initial_train_loglik: Optional[float] = None,
    initial_val_loglik: Optional[float] = None,
    fresh_parent_code: Optional[str] = None,
    fresh_parent_train_loglik: Optional[float] = None,
    fresh_parent_val_loglik: Optional[float] = None,
    run_prompts_dir: Optional[str] = None,
    max_parent_chars: int = 6000,
    warn_parent_truncation_ratio: float = 0.5,
    early_stop_iters: Optional[int] = None,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    max_error_prompt_chars: int = 1200,
    error_feedback_mode: str = DEFAULT_ERROR_FEEDBACK_MODE,
    mdl_lambda: float = 0.0,
) -> Optional[float]:
    """
    Refinement: val trials in prompt; pool sorted by train_val_loglik only after iteration 1+.

    When ``evolution_elite_parents`` / ``evolution_elite_val_logliks`` are provided, the full
    evolution elite pool is copied in evolution order (no re-sort). Otherwise falls back to a
    single-program pool from ``initial_code``.
    """
    error_feedback_mode = _normalize_error_feedback_mode(error_feedback_mode)
    mdl_lambda = normalize_mdl_lambda(mdl_lambda)
    if not val_trials or not test_trials or n_iterations < 1:
        return None

    if _DISABLE_REFINEMENT_FRESH_CANDIDATES and fresh_n_candidates > 0:
        print(
            "Note: refinement fresh candidates disabled (_DISABLE_REFINEMENT_FRESH_CANDIDATES); "
            f"ignoring fresh_n_candidates={fresh_n_candidates}, using evolution pool only."
        )
        fresh_n_candidates = 0

    refine_suffix = _load_refinement_prompt_suffix(dataset, run_prompts_dir=run_prompts_dir)
    val_for_prompt = _cap_and_subsample_prompt_trials(
        val_trials,
        max_trials=max_prompt_train_trials,
        max_trials_per_problem=max_prompt_trials_per_problem,
        subsample_seed=int(split_seed) + 9_001_001,
        label="validation",
    )

    print(f"\n{'='*80}")
    print(
        f"Refinement phase ({n_iterations} iter(s), val trials in prompt: {len(val_for_prompt)})"
    )
    print(f"{'='*80}")

    pool_from_evolution = bool(evolution_elite_parents)
    if pool_from_evolution:
        elite_parents = [tuple(p) for p in evolution_elite_parents]
        elite_val_logliks = list(evolution_elite_val_logliks or [])
        if len(elite_val_logliks) != len(elite_parents):
            raise ValueError(
                "evolution_elite_val_logliks length must match evolution_elite_parents"
            )
        print(
            f"Refinement initial pool: {len(elite_parents)} program(s) copied from evolution "
            f"(evolution order preserved; sort by train_val_loglik starts after iteration 1)."
        )
        elite_parents = _apply_mdl_to_refinement_pool(
            elite_parents,
            mdl_lambda=mdl_lambda,
            n_train=len(train_trials),
            n_val=len(val_trials),
        )
        seed_test_loglik: Optional[float] = None
        top_fn = compile_program(elite_parents[0][0])
        if top_fn is not None:
            seed_test_eval = _evaluate_loglik_for_dataset(
                dataset, top_fn, test_trials, n_seeds=n_eval_seeds
            )
            seed_test_loglik = float(seed_test_eval["avg_loglik"])
    else:
        if initial_code is None or initial_train_loglik is None or initial_val_loglik is None:
            raise ValueError(
                "Refinement requires evolution_elite_parents or initial_code with loglik metrics."
            )
        seed_combined_fitness = _refinement_combined_fitness(
            initial_train_loglik, initial_val_loglik, split_ratio=split_ratio
        )
        elite_parents = [
            (
                initial_code,
                seed_combined_fitness,
                None,
                "refinement_seed",
                None,
                None,
                float(initial_train_loglik),
            )
        ]
        elite_parents = _apply_mdl_to_refinement_pool(
            elite_parents,
            mdl_lambda=mdl_lambda,
            n_train=len(train_trials),
            n_val=len(val_trials),
        )
        elite_val_logliks = [float(initial_val_loglik)]
        seed_test_loglik = None
        seed_fn = compile_program(initial_code)
        if seed_fn is not None:
            seed_test_eval = _evaluate_loglik_for_dataset(
                dataset, seed_fn, test_trials, n_seeds=n_eval_seeds
            )
            seed_test_loglik = float(seed_test_eval["avg_loglik"])
        print("Refinement initial pool: single checkpoint program (refinement_seed).")

    refinement_dir: Optional[Path] = None
    if save_artifacts and output_path is not None:
        refinement_dir = output_path / "refinement"
        refinement_dir.mkdir(parents=True, exist_ok=True)
        if pool_from_evolution and (output_path / "evolution_elite_pool").is_dir():
            src_pool = output_path / "evolution_elite_pool"
            dst_pool = refinement_dir / "initial_pool_from_evolution"
            if dst_pool.exists():
                shutil.rmtree(dst_pool)
            shutil.copytree(src_pool, dst_pool)
        elif initial_code is not None:
            (refinement_dir / "seed_program.py").write_text(initial_code or "")

    early_stop_patience = _normalize_early_stop_iters(early_stop_iters)
    last_significant_best = float(elite_parents[0][1])
    stagnant_iters = 0
    invalid_candidate_errors: List[Dict[str, Any]] = _ErrorFeedbackStore(
        error_feedback_mode
    )
    error_history_path = (
        (output_path / "error_history.jsonl")
        if output_path is not None
        else None
    )
    if early_stop_patience is not None:
        print(
            f"Refinement early stop enabled: patience={early_stop_patience}, "
            f"min_improvement={_EARLY_STOP_MIN_IMPROVEMENT:.3f}"
        )

    for iteration in range(n_iterations):
        iteration_step = iteration + 1
        print(f"\n{'='*80}")
        print(f"Refinement iteration {iteration_step}/{n_iterations}")
        print(f"{'='*80}")

        iter_dir: Optional[Path] = None
        if refinement_dir is not None:
            iter_dir = refinement_dir / f"iteration_{iteration_step}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "candidates").mkdir(exist_ok=True)

        pool_size = len(elite_parents)
        if sample_parents and pool_size > 0:
            pid_key = int(participant_id) if participant_id is not None else 0
            rng = np.random.default_rng(
                int(split_seed) + 9_000_000 + int(iteration_step) * 1_000_003 + pid_key * 17_179
            )
            parent_idxs, best_k, sampled_k = _select_parent_indices_from_elite_pool(
                pool_size,
                sample_size=sample_size,
                sample_parents=True,
                sampled_parents_decay=sampled_parents_decay,
                iter_idx=iteration,
                total_iters=n_iterations,
                rng=rng,
            )
            selected_parents = [elite_parents[int(j)] for j in parent_idxs]
            selected_val_lls = [elite_val_logliks[int(j)] for j in parent_idxs]
            num_parents_to_use = len(selected_parents)
            if sampled_parents_decay:
                print(
                    f"\nUsing {num_parents_to_use} refinement parent(s) "
                    f"({best_k} best + {sampled_k} sampled, elite pool size={pool_size}):"
                )
            else:
                print(
                    f"\nUsing {num_parents_to_use} sampled refinement parent(s) "
                    f"(elite pool size={pool_size}):"
                )
        else:
            num_parents_to_use = min(sample_size, pool_size)
            selected_parents = elite_parents[:num_parents_to_use]
            selected_val_lls = elite_val_logliks[:num_parents_to_use]
            print(f"\nUsing top {num_parents_to_use} refinement parent(s):")

        for i, parent_tuple in enumerate(selected_parents):
            prog_id = parent_tuple[3]
            train_ll = _train_loglik_from_elite_tuple(parent_tuple)
            val_ll = selected_val_lls[i]
            comb_fitness = float(parent_tuple[1])
            val_str = f"{val_ll:.4f}" if val_ll is not None else "N/A"
            print(
                f"  Parent {i+1}: {prog_id} "
                f"(fitness={comb_fitness:.4f}, train_loglik={train_ll:.4f}, val_loglik={val_str})"
            )

        parent_codes = [p[0] for p in selected_parents]
        parent_train_lls = [_train_loglik_from_elite_tuple(p) for p in selected_parents]
        parent_overall_lls = [float(p[1]) for p in selected_parents]

        print(
            f"[LLM prompt] Refinement uses {len(val_for_prompt)} validation trials only "
            f"(not train+val; cap via max_prompt_train_trials={max_prompt_train_trials})."
        )
        prompt_stats_path = (
            iter_dir / "prompt_stats.json" if iter_dir is not None else None
        )
        fresh_n = _decayed_fresh_n_for_iteration(
            fresh_n_candidates, iteration, n_iterations, n_candidates_per_iteration
        )
        n_normal = n_candidates_per_iteration - fresh_n
        print(
            f"Fresh candidate schedule (refinement): iter_idx={iteration}, "
            f"total_iterations={n_iterations}, fresh_n={fresh_n} "
            f"(max fresh_n_candidates={fresh_n_candidates})"
        )
        print(
            f"\nGenerating {n_candidates_per_iteration} refinement candidates: "
            f"{fresh_n} fresh (seed/baseline only), {n_normal} from sampled parents..."
        )
        error_prompt_section = _build_past_error_prompt_section(
            invalid_candidate_errors,
            iteration=iteration_step,
            max_error_prompt_chars=max_error_prompt_chars,
            previous_n_candidates=n_candidates_per_iteration,
            error_feedback_mode=error_feedback_mode,
        )
        _write_iteration_error_prompt_file(iter_dir, error_prompt_section)
        error_prompt_chars_used = len(error_prompt_section)
        num_unique_errors_available = _count_previous_iteration_error_groups(
            invalid_candidate_errors,
            iteration=iteration_step,
            error_feedback_mode=error_feedback_mode,
        )
        print(
            "Error prompt summary: "
            f"num_unique_errors_available={num_unique_errors_available}, "
            f"error_prompt_chars_used={error_prompt_chars_used}"
        )
        (
            fresh_parent_codes,
            fresh_parent_train_lls,
            fresh_parent_val_lls,
        ) = _resolve_fresh_parent_prompt_context(
            fresh_parent_code=fresh_parent_code,
            fresh_parent_train_loglik=fresh_parent_train_loglik,
            fresh_parent_val_loglik=fresh_parent_val_loglik,
            elite_parents=elite_parents,
        )
        variant_kwargs = {
            "train_trials": train_trials,
            "max_tokens": llm_max_tokens,
            "dataset": dataset,
            "max_prompt_train_trials": max_prompt_train_trials,
            "max_prompt_trials_per_problem": max_prompt_trials_per_problem,
            "prompt_train_trials_seed": split_seed,
            "fitness_metric": fitness_metric,
            "max_workers": max_workers,
            "prompt_suffix": refine_suffix,
            "prompt_observation_trials": val_for_prompt,
            "run_prompts_dir": run_prompts_dir,
            "max_parent_chars": max_parent_chars,
            "warn_parent_truncation_ratio": warn_parent_truncation_ratio,
            "sample_size_for_warning": sample_size,
            "prompt_stats_path": prompt_stats_path,
            "hard_prompt_token_cap": hard_prompt_token_cap,
            "strict_prompt_budget": strict_prompt_budget,
            "prompt_token_estimator": prompt_token_estimator,
            "prompt_diagnostics_dir": output_path,
            "phase": "refinement",
            "participant_id": participant_id,
            "iteration": iteration_step,
            "past_invalid_program_errors": invalid_candidate_errors,
            "past_error_prompt_section": error_prompt_section,
            "max_error_prompt_chars": max_error_prompt_chars,
            "error_feedback_mode": error_feedback_mode,
        }
        candidate_codes, candidate_sources = _generate_iteration_candidate_codes(
            client=client,
            model_name=model_name,
            fresh_n_candidates=fresh_n,
            n_candidates=n_candidates_per_iteration,
            fresh_parent_programs=fresh_parent_codes,
            normal_parent_programs=parent_codes,
            variant_kwargs=variant_kwargs,
            fresh_parent_train_accuracies=fresh_parent_train_lls,
            fresh_parent_val_logliks=fresh_parent_val_lls,
            normal_parent_train_accuracies=parent_train_lls,
            normal_parent_val_logliks=selected_val_lls,
            normal_parent_overall_logliks=parent_overall_lls,
        )

        candidate_results: List[Dict[str, Any]] = []
        num_invalid_candidates = 0
        for idx, code in enumerate(candidate_codes):
            if iter_dir is not None:
                (iter_dir / "candidates" / f"candidate_{idx}.py").write_text(code or "")
            code = _sanitize_llm_python_candidate(code, required_markers=("def choose(",))
            if not code:
                candidate_results.append(
                    {
                        "idx": idx,
                        "code": "",
                        "train_val_loglik": float("-inf"),
                        "train_loglik": float("-inf"),
                        "val_loglik": float("-inf"),
                        "runtime_valid": False,
                    }
                )
                continue
            choose_fn, compile_error = compile_program_with_error(code)
            if choose_fn is None:
                num_invalid_candidates += 1
                _record_invalid_program_error(
                    invalid_candidate_errors,
                    code=code,
                    exc=compile_error,
                    iteration=iteration_step,
                    participant_id=int(participant_id),
                    candidate_id=f"candidate_{idx}",
                    eval_split="compile",
                    n_candidates_in_iteration=n_candidates_per_iteration,
                    history_path=error_history_path,
                )
                candidate_results.append(
                    {
                        "idx": idx,
                        "code": code,
                        "train_val_loglik": float("-inf"),
                        "train_loglik": float("-inf"),
                        "val_loglik": float("-inf"),
                        "runtime_valid": False,
                    }
                )
                continue
            try:
                train_eval = _evaluate_loglik_for_dataset(
                    dataset, choose_fn, train_trials, n_seeds=n_eval_seeds
                )
                val_eval = _evaluate_loglik_for_dataset(
                    dataset, choose_fn, val_trials, n_seeds=n_eval_seeds
                )
            except (AssertionError, TypeError, ValueError) as exc:
                num_invalid_candidates += 1
                _record_invalid_program_error(
                    invalid_candidate_errors,
                    code=code,
                    exc=exc,
                    iteration=iteration_step,
                    participant_id=int(participant_id),
                    candidate_id=f"candidate_{idx}",
                    eval_split="train",
                    n_candidates_in_iteration=n_candidates_per_iteration,
                    history_path=error_history_path,
                )
                candidate_results.append(
                    {
                        "idx": idx,
                        "code": code,
                        "train_val_loglik": float("-inf"),
                        "train_loglik": float("-inf"),
                        "val_loglik": float("-inf"),
                        "runtime_valid": False,
                    }
                )
                continue
            runtime_valid = train_eval.get("errors", 0) == 0
            if train_eval.get("errors", 0) != 0:
                num_invalid_candidates += 1
                _record_invalid_program_error_summary(
                    invalid_candidate_errors,
                    train_eval.get("first_error"),
                    iteration=iteration_step,
                    participant_id=int(participant_id),
                    candidate_id=f"candidate_{idx}",
                    quality_score=_safe_float(train_eval.get("avg_loglik")),
                    eval_split="train",
                    n_candidates_in_iteration=n_candidates_per_iteration,
                    history_path=error_history_path,
                )
            if val_eval.get("errors", 0) != 0:
                num_invalid_candidates += 1
                _record_invalid_program_error_summary(
                    invalid_candidate_errors,
                    val_eval.get("first_error"),
                    iteration=iteration_step,
                    participant_id=int(participant_id),
                    candidate_id=f"candidate_{idx}",
                    quality_score=_safe_float(train_eval.get("avg_loglik")),
                    eval_split="val",
                    n_candidates_in_iteration=n_candidates_per_iteration,
                    history_path=error_history_path,
                )
            train_loglik = float(train_eval["avg_loglik"])
            val_loglik = float(val_eval["avg_loglik"])
            train_val_loglik = (
                _refinement_combined_fitness(
                    train_loglik, val_loglik, split_ratio=split_ratio
                )
                if runtime_valid
                else float("-inf")
            )
            row = {
                "idx": idx,
                "code": code,
                "train_val_loglik": train_val_loglik,
                "train_loglik": train_loglik,
                "val_loglik": val_loglik,
                "runtime_valid": runtime_valid,
            }
            if runtime_valid:
                _attach_mdl_to_candidate(
                    row,
                    mdl_lambda=mdl_lambda,
                    n=_refinement_mdl_n(len(train_trials), len(val_trials)),
                    selection_score=train_val_loglik,
                )
            candidate_results.append(row)

        print(
            "Iteration invalid summary: "
            f"num_invalid_candidates={num_invalid_candidates}, "
            f"num_unique_errors_available={_count_previous_iteration_error_groups(invalid_candidate_errors, iteration=iteration_step + 1, error_feedback_mode=error_feedback_mode)}, "
            f"error_prompt_chars_used={error_prompt_chars_used}"
        )

        selected_results = [r for r in candidate_results if r.get("runtime_valid", False)]
        iter_best_result: Optional[Dict[str, Any]] = None
        if not selected_results:
            print(
                f"Warning: No runtime-valid refinement candidates for participant "
                f"{participant_id} at iteration {iteration_step}; keeping elite pool."
            )
        else:
            sort_candidates(selected_results, mdl_lambda, score_key="train_val_loglik")
            iter_best_result = selected_results[0]
            _tr, _vr = _refinement_train_val_ratios(split_ratio)
            print(
                f"  Best refinement candidate {iter_best_result['idx']}: "
                f"train_val_loglik={iter_best_result['train_val_loglik']:.4f} "
                f"(({_tr:.2f}*train+{_vr:.2f}*val)/{_tr + _vr:.2f}), "
                f"train_loglik={iter_best_result['train_loglik']:.4f}, "
                f"val_loglik={iter_best_result['val_loglik']:.4f}"
            )
            for result in selected_results:
                program_id = f"refinement_{iteration_step}_candidate_{result['idx']}"
                elite_parents.append(
                    _elite_tuple_for_ranking(
                        result["code"],
                        result["train_val_loglik"],
                        None,
                        program_id,
                        result["train_loglik"],
                        mdl_lambda=mdl_lambda,
                        mdl_score=result.get("mdl_score"),
                    )
                )
                elite_val_logliks.append(result.get("val_loglik"))

        paired = list(zip(elite_parents, elite_val_logliks))
        if iteration_step >= 1:
            sort_elite_pairs(paired, mdl_lambda)
        elite_cap = _elite_pool_capacity(sample_size, elite_pool_size)
        paired = paired[:elite_cap]
        elite_parents = [p[0] for p in paired]
        elite_val_logliks = [p[1] for p in paired]

        pool_metrics = _refinement_pool_best_metrics(
            elite_parents,
            elite_val_logliks,
            test_trials,
            dataset=dataset,
            n_eval_seeds=n_eval_seeds,
        )
        pool_test_ll = pool_metrics.get("pool_best_test_loglik")
        pool_id = pool_metrics.get("pool_best_program_id")
        if pool_test_ll is not None:
            print(
                f"  Pool best after iteration {iteration_step}: {pool_id} "
                f"(train_val_loglik={float(pool_metrics['pool_best_train_val_loglik']):.4f}, "
                f"test_loglik={float(pool_test_ll):.6f})"
            )
        else:
            print(
                f"  Pool best after iteration {iteration_step}: {pool_id} "
                f"(test_loglik unavailable)"
            )

        if iter_dir is not None:
            best_fields: Dict[str, Any] = dict(pool_metrics)
            if iter_best_result is not None:
                best_fields["iter_best_program_id"] = (
                    f"refinement_{iteration_step}_candidate_{iter_best_result['idx']}"
                )
                best_fields["iter_best_train_val_loglik"] = iter_best_result["train_val_loglik"]
                best_fields["iter_best_train_loglik"] = iter_best_result["train_loglik"]
                best_fields["iter_best_val_loglik"] = iter_best_result["val_loglik"]
            else:
                best_fields["iter_best_program_id"] = None
                best_fields["iter_best_train_val_loglik"] = None
                best_fields["iter_best_train_loglik"] = None
                best_fields["iter_best_val_loglik"] = None
            refine_header = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_runtime_valid": len(selected_results),
                "num_invalid_candidates": num_invalid_candidates,
                "num_unique_errors_available": num_unique_errors_available,
                "error_prompt_chars_used": error_prompt_chars_used,
            }
            if mdl_enabled(mdl_lambda):
                refine_header["mdl_lambda"] = float(mdl_lambda)
            if iter_best_result is not None:
                best_fields.update(_candidate_mdl_log_fields(iter_best_result))
            refine_header.update(
                _iteration_candidate_source_header(
                    fresh_n_candidates,
                    fresh_n,
                    n_candidates_per_iteration,
                    candidate_sources,
                    iter_idx=iteration,
                    total_iters=n_iterations,
                )
            )
            metrics = _build_iteration_metrics_json(
                participant_id=participant_id,
                header=refine_header,
                best=best_fields,
                candidate_results=_annotate_candidate_results_with_sources(
                    [
                        {
                            "idx": r["idx"],
                            "train_loglik": r.get("train_loglik"),
                            "val_loglik": r.get("val_loglik"),
                            "train_val_loglik": r.get("train_val_loglik"),
                            "runtime_valid": r.get("runtime_valid", False),
                            **_candidate_mdl_log_fields(r),
                        }
                        for r in candidate_results
                    ],
                    candidate_sources,
                ),
            )
            (iter_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )

        if wandb_module is not None:
            refine_iter_log: Dict[str, Any] = {
                f"p{participant_id}_train_val_loglik": pool_metrics.get(
                    "pool_best_train_val_loglik"
                ),
                f"p{participant_id}_train_loglik": pool_metrics.get("pool_best_train_loglik"),
                f"p{participant_id}_val_loglik": pool_metrics.get("pool_best_val_loglik"),
            }
            if pool_metrics.get("pool_best_test_loglik") is not None:
                refine_iter_log[f"p{participant_id}_test_loglik"] = pool_metrics[
                    "pool_best_test_loglik"
                ]
            if iter_best_result is not None:
                refine_iter_log[f"p{participant_id}_iter_best_train_val_loglik"] = (
                    iter_best_result["train_val_loglik"]
                )
            _wandb_log_participant_metrics(
                wandb_module,
                refine_iter_log,
                int(participant_id),
                int(wandb_step_offset) + iteration_step,
            )

        if early_stop_patience is not None:
            pool_best_fitness = float(pool_metrics["pool_best_train_val_loglik"])
            improvement = pool_best_fitness - float(last_significant_best)
            if improvement >= _EARLY_STOP_MIN_IMPROVEMENT:
                last_significant_best = pool_best_fitness
                stagnant_iters = 0
            else:
                stagnant_iters += 1
                if stagnant_iters >= early_stop_patience:
                    print(
                        f"Early stopping refinement at iteration {iteration_step}: "
                        f"pool best train+val loglik improved by < "
                        f"{_EARLY_STOP_MIN_IMPROVEMENT:.3f} for {stagnant_iters} "
                        "consecutive iteration(s)."
                    )
                    break

    best_code = elite_parents[0][0]
    best_fn = compile_program(best_code)
    if best_fn is None:
        print("Warning: Refinement best program failed to compile; skipping gated_test_loglik.")
        return None

    if refinement_dir is not None:
        (refinement_dir / BEST_PROGRAM_FILENAME).write_text(best_code or "")
    if output_path is not None and save_artifacts:
        (output_path / BEST_PROGRAM_FILENAME).write_text(best_code or "")

    test_eval = _evaluate_loglik_for_dataset(
        dataset, best_fn, test_trials, n_seeds=n_eval_seeds
    )
    refinement_test_loglik = float(test_eval["avg_loglik"])
    print(f"Refinement final test avg log-likelihood (gated_test_loglik): {refinement_test_loglik:.6f}")

    if refinement_dir is not None:
        best_val_ll = elite_val_logliks[0] if elite_val_logliks else None
        final_program_id = str(elite_parents[0][3])
        final_is_seed = final_program_id == "refinement_seed"
        refine_results: Dict[str, Any] = {
            "phase": "refinement",
            "dataset": dataset,
            "participant_id": int(participant_id),
            "n_iterations": int(n_iterations),
            "refinement_skipped": False,
            "initial_pool_from_evolution": pool_from_evolution,
            "initial_pool_size": len(elite_parents),
            "final_pool_best": {
                "program_id": final_program_id,
                "train_val_loglik": float(elite_parents[0][1]),
                "train_loglik": _train_loglik_from_elite_tuple(elite_parents[0]),
                "val_loglik": float(best_val_ll) if best_val_ll is not None else None,
                "test_loglik": refinement_test_loglik,
            },
            "final_program_is_seed": final_is_seed,
            "gated_test_loglik": refinement_test_loglik,
        }
        if pool_from_evolution:
            refine_results["evolution_pool_transfer"] = {
                "n_programs": len(evolution_elite_parents or []),
                "sort_by_train_val_loglik_after_iteration": 1,
            }
        else:
            refine_results["refinement_seed"] = {
                "program_id": "refinement_seed",
                "train_loglik": float(initial_train_loglik),
                "val_loglik": float(initial_val_loglik),
                "test_loglik": seed_test_loglik,
                "train_val_loglik": _refinement_combined_fitness(
                    float(initial_train_loglik),
                    float(initial_val_loglik),
                    split_ratio=split_ratio,
                ),
            }
        _write_refinement_results_json(refinement_dir, refine_results)
        _write_refinement_summary_loglik_csv(
            refinement_dir,
            {
                "participant_id": int(participant_id),
                "train_loglik": _train_loglik_from_elite_tuple(elite_parents[0]),
                "val_loglik": float(best_val_ll) if best_val_ll is not None else None,
                "test_loglik": refinement_test_loglik,
                "gated_test_loglik": refinement_test_loglik,
            },
        )
        if final_is_seed and not pool_from_evolution:
            print(
                "Note: Final pool-best is still refinement_seed (no candidate beat seed on "
                "combined fitness); gated_test_loglik equals seed test loglik."
            )
    return refinement_test_loglik


def run_choice13k_refinement_phase(**kwargs: Any) -> Optional[float]:
    """Backward-compatible wrapper for Choice13k refinement."""
    return run_loglik_refinement_phase(**kwargs, dataset="choice13k")


def _load_participant_details_loglik_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"participant_details_loglik.csv not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_participant_details_loglik_csv(
    csv_path: Path,
    rows: List[Dict[str, Any]],
    *,
    dataset: str = "choice13k",
    fitness_metric: str = "loglik",
    cpc18_official_mse: bool = False,
) -> None:
    fieldnames = ["participant_id", "train_loglik"]
    if _uses_train_val_test_loglik_split(
        dataset, fitness_metric, cpc18_official_mse=cpc18_official_mse
    ):
        fieldnames.extend(["val_loglik", "test_loglik", "gated_test_loglik"])
    else:
        fieldnames.append("test_loglik")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_round_floats_for_csv_rows(rows))


def _clear_gated_test_loglik_in_loglik_rows(rows: List[Dict[str, Any]]) -> None:
    """Remove prior-run gated_test_loglik so a new refine experiment starts fresh."""
    for row in rows:
        row["gated_test_loglik"] = None


def _passive_diagnostic_test_eval(
    *,
    dataset: str,
    choose_fn: Any,
    test_trials: Optional[List[Dict[str, Any]]],
    n_eval_seeds: int,
) -> Dict[str, Any]:
    """Held-out test metrics for logging only. Never used for selection."""
    empty = {
        "avg_loglik": None,
        "accuracy": None,
        "errors": 0,
        "correct": 0,
        "total": 0 if not test_trials else len(test_trials),
    }
    if choose_fn is None or not test_trials:
        return empty
    try:
        return _evaluate_loglik_for_dataset(
            dataset, choose_fn, list(test_trials), n_seeds=n_eval_seeds
        )
    except (AssertionError, TypeError, ValueError):
        return {
            "avg_loglik": None,
            "accuracy": None,
            "errors": 1,
            "correct": 0,
            "total": len(test_trials),
        }


def _passive_test_record(
    *,
    program_id: str,
    test_eval: Mapping[str, Any],
    phase: str,
    iteration: Optional[int] = None,
    n_test_trials: Optional[int] = None,
    program_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Metadata tying one diagnostic test score to the exact selected program."""
    return {
        "diagnostic_only": True,
        "not_used_for_selection": True,
        "phase": phase,
        "iteration": None if iteration is None else int(iteration),
        "program_id": str(program_id),
        "program_sha256": program_sha256,
        "test_loglik": _safe_float(test_eval.get("avg_loglik")),
        "test_accuracy": _safe_float(test_eval.get("accuracy")),
        "n_test_trials": (
            int(n_test_trials)
            if n_test_trials is not None
            else int(test_eval.get("total") or 0)
        ),
    }


def _passive_diagnostic_test_of_program(
    *,
    dataset: str,
    code: Optional[str],
    test_trials: Optional[List[Dict[str, Any]]],
    n_eval_seeds: int,
    program_id: str,
    phase: str,
    iteration: Optional[int] = None,
) -> Dict[str, Any]:
    source = code or ""
    choose_fn = compile_program(source)
    test_eval = _passive_diagnostic_test_eval(
        dataset=dataset,
        choose_fn=choose_fn,
        test_trials=test_trials,
        n_eval_seeds=n_eval_seeds,
    )
    return _passive_test_record(
        program_id=program_id,
        test_eval=test_eval,
        phase=phase,
        iteration=iteration,
        n_test_trials=len(test_trials or []),
        program_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest() if source else None,
    )


def _collect_pooled_test_trials_for_participants_diagnostic(
    dataset: str,
    participant_ids: List[int],
    *,
    split_ratio: float,
    split_seed: int,
    data_path: str = "data",
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    speekenbrink_split: str = "chronological",
) -> List[Dict[str, Any]]:
    """Concatenate per-person test splits for passive pool-best diagnostics only."""
    pooled: List[Dict[str, Any]] = []
    for pid in participant_ids:
        _train, _val, test_trials = _trials_for_loglik_participant(
            dataset,
            int(pid),
            split_ratio=split_ratio,
            split_seed=split_seed,
            data_path=data_path,
            filter_mixed_gambles=filter_mixed_gambles,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            max_observed_trials_per_participant=max_observed_trials_per_participant,
            limited_data_protocol=limited_data_protocol,
            limited_train_val=limited_train_val,
            speekenbrink_split=speekenbrink_split,
        )
        del _train, _val
        pooled.extend(list(test_trials or []))
    return pooled


def _write_global_phase_summary_loglik_csv(
    global_dir: Path,
    *,
    dataset: str,
    participant_ids: List[int],
    pool_best_code: str,
    split_ratio: float,
    split_seed: int,
    data_path: str = "data",
    filter_mixed_gambles: bool = False,
    n_eval_seeds: int = 3,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
) -> None:
    """
    Write global_phase/summary_loglik.csv: pool-best program evaluated per participant,
    then averaged (same columns as run-level summary_loglik.csv; gated column left empty).
    """
    choose_fn = compile_program(pool_best_code)
    if choose_fn is None:
        print("Warning: global pool-best failed to compile; skipping summary_loglik.csv.")
        return

    train_vals: List[float] = []
    val_vals: List[float] = []
    test_vals: List[float] = []
    for pid in participant_ids:
        train_trials, val_trials, test_trials = _trials_for_loglik_participant(
            dataset,
            int(pid),
            split_ratio=split_ratio,
            split_seed=split_seed,
            data_path=data_path,
            filter_mixed_gambles=filter_mixed_gambles,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            max_observed_trials_per_participant=max_observed_trials_per_participant,
            limited_data_protocol=limited_data_protocol,
            limited_train_val=limited_train_val,
        )
        train_eval = _evaluate_loglik_for_dataset(
            dataset, choose_fn, train_trials, n_seeds=n_eval_seeds
        )
        train_vals.append(float(train_eval["avg_loglik"]))
        if val_trials:
            val_eval = _evaluate_loglik_for_dataset(
                dataset, choose_fn, val_trials, n_seeds=n_eval_seeds
            )
            val_vals.append(float(val_eval["avg_loglik"]))
        test_eval = _evaluate_loglik_for_dataset(
            dataset, choose_fn, test_trials, n_seeds=n_eval_seeds
        )
        test_vals.append(float(test_eval["avg_loglik"]))

    summary_path = global_dir / "summary_loglik.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_of_participants",
                "avg_train_loglik",
                "avg_test_loglik",
                "avg_val_loglik",
                "avg_gated_test_loglik",
            ],
        )
        writer.writeheader()
        writer.writerow(
            _round_floats_for_csv_row(
                {
                    "num_of_participants": len(participant_ids),
                    "avg_train_loglik": float(np.mean(train_vals)) if train_vals else None,
                    "avg_test_loglik": float(np.mean(test_vals)) if test_vals else None,
                    "avg_val_loglik": float(np.mean(val_vals)) if val_vals else None,
                    "avg_gated_test_loglik": None,
                }
            )
        )
    print(f"Wrote global phase summary: {summary_path}")


def _write_refinement_summary_loglik_csv(refinement_dir: Path, row: Dict[str, Any]) -> None:
    """Single-participant summary_loglik.csv under refinement/ (mirrors run-level columns)."""
    summary_path = refinement_dir / "summary_loglik.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "train_loglik",
                "val_loglik",
                "test_loglik",
                "gated_test_loglik",
            ],
        )
        writer.writeheader()
        writer.writerow(_round_floats_for_csv_row(row))


def _write_refine_summary_loglik_csv(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    """Write experiment-level summary_loglik.csv for refine-only runs."""
    gated_vals = [
        float(r["gated_test_loglik"])
        for r in rows
        if r.get("gated_test_loglik") not in (None, "")
    ]
    val_vals = [float(r["val_loglik"]) for r in rows if r.get("val_loglik") not in (None, "")]
    train_vals = [float(r["train_loglik"]) for r in rows if r.get("train_loglik") not in (None, "")]
    test_vals = [float(r["test_loglik"]) for r in rows if r.get("test_loglik") not in (None, "")]
    summary_loglik_path = output_dir / "summary_loglik.csv"
    with summary_loglik_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_of_participants",
                "avg_train_loglik",
                "avg_test_loglik",
                "avg_val_loglik",
                "avg_gated_test_loglik",
            ],
        )
        writer.writeheader()
        writer.writerow(
            _round_floats_for_csv_row(
                {
                    "num_of_participants": len(rows),
                    "avg_train_loglik": float(np.mean(train_vals)) if train_vals else None,
                    "avg_test_loglik": float(np.mean(test_vals)) if test_vals else None,
                    "avg_val_loglik": float(np.mean(val_vals)) if val_vals else None,
                    "avg_gated_test_loglik": float(np.mean(gated_vals)) if gated_vals else None,
                }
            )
        )


def _write_refinement_results_json(refinement_dir: Path, results: Dict[str, Any]) -> None:
    refinement_dir.mkdir(parents=True, exist_ok=True)
    (refinement_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def _psych101_trials_for_participant(
    dataset_alias: str,
    participant_id: int,
    *,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    exp = get_psych101_binary_experiment(
        dataset_alias,
        int(participant_id),
        split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    train_trials, val_trials, test_trials, _options = split_psych_experiment(
        exp, split_ratio=split_ratio, split_seed=split_seed
    )
    return train_trials, val_trials, test_trials


def _trials_for_loglik_participant(
    dataset: str,
    participant_id: int,
    *,
    split_ratio: float,
    split_seed: int,
    data_path: str = "data",
    filter_mixed_gambles: bool = False,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    speekenbrink_split: str = "chronological",
    return_audit: bool = False,
    return_manifest: bool = False,
):
    del data_path
    train_trials, val_trials, test_trials, audit, manifest = load_participant_limited_splits(
        dataset,
        int(participant_id),
        split_ratio=split_ratio,
        split_seed=split_seed,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        speekenbrink_split=speekenbrink_split,
    )
    if return_manifest:
        train_trials = [stamp_prompt_participant_id(t, int(participant_id)) for t in train_trials]
        val_trials = [stamp_prompt_participant_id(t, int(participant_id)) for t in val_trials]
        test_trials = [stamp_prompt_participant_id(t, int(participant_id)) for t in test_trials]
        return train_trials, val_trials, test_trials, audit, manifest
    if return_audit:
        train_trials = [stamp_prompt_participant_id(t, int(participant_id)) for t in train_trials]
        val_trials = [stamp_prompt_participant_id(t, int(participant_id)) for t in val_trials]
        test_trials = [stamp_prompt_participant_id(t, int(participant_id)) for t in test_trials]
        return train_trials, val_trials, test_trials, audit
    train_trials = [stamp_prompt_participant_id(t, int(participant_id)) for t in train_trials]
    val_trials = [stamp_prompt_participant_id(t, int(participant_id)) for t in val_trials]
    test_trials = [stamp_prompt_participant_id(t, int(participant_id)) for t in test_trials]
    return train_trials, val_trials, test_trials


def run_loglik_refine_participant_from_checkpoint(
    *,
    dataset: str,
    client: OpenAI,
    model_name: str,
    participant_id: int,
    prev_exp_path: Path,
    output_dir: Path,
    prev_loglik_row: Optional[Dict[str, Any]],
    split_ratio: float,
    split_seed: int,
    data_path: str = "data",
    filter_mixed_gambles: bool = False,
    n_iterations: int,
    n_candidates_per_iteration: int,
    fresh_n_candidates: int = 0,
    sample_size: int,
    sample_parents: bool,
    sampled_parents_decay: bool = True,
    elite_pool_size: Optional[int],
    max_prompt_train_trials: int,
    max_prompt_trials_per_problem: int,
    llm_max_tokens: int,
    max_workers: int,
    n_eval_seeds: int,
    save_artifacts: bool = True,
    wandb_module: Optional[Any] = None,
    refinement_val_threshold: float = -1.0,
    run_prompts_dir: Optional[str] = None,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_parent_chars: int = 6000,
    warn_parent_truncation_ratio: float = 0.5,
    early_stop_iters: Optional[int] = None,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    max_error_prompt_chars: int = 1200,
    error_feedback_mode: str = DEFAULT_ERROR_FEEDBACK_MODE,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    mdl_lambda: float = 0.0,
) -> Dict[str, Any]:
    """
    Refinement-only for one participant: load best_program.py from a prior run, refine, return metrics.
    """
    prev_participant_dir = prev_exp_path / f"participant_{participant_id}"
    program_path = prev_participant_dir / BEST_PROGRAM_FILENAME
    if not program_path.exists():
        raise FileNotFoundError(
            f"Missing {BEST_PROGRAM_FILENAME} for participant {participant_id}: {program_path}"
        )

    initial_code = program_path.read_text(encoding="utf-8")
    train_trials, val_trials, test_trials, sparse_audit = _trials_for_loglik_participant(
        dataset,
        participant_id,
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        return_audit=True,
    )
    choose_fn = compile_program(initial_code)
    if choose_fn is None:
        raise RuntimeError(f"Failed to compile checkpoint program: {program_path}")

    train_eval = _evaluate_loglik_for_dataset(
        dataset, choose_fn, train_trials, n_seeds=n_eval_seeds
    )
    val_eval = _evaluate_loglik_for_dataset(
        dataset, choose_fn, val_trials, n_seeds=n_eval_seeds
    )
    train_loglik = float(train_eval["avg_loglik"])
    val_loglik = float(val_eval["avg_loglik"])

    if prev_loglik_row is not None:
        if prev_loglik_row.get("train_loglik") not in (None, ""):
            train_loglik = float(prev_loglik_row["train_loglik"])
        if prev_loglik_row.get("val_loglik") not in (None, ""):
            val_loglik = float(prev_loglik_row["val_loglik"])

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_artifacts:
        (output_dir / BEST_PROGRAM_FILENAME).write_text(initial_code)
        if should_persist_sparse_audit(sparse_audit):
            write_sparse_audit(output_dir / SPARSE_AUDIT_FILENAME, sparse_audit)

    print(
        f"\nRefine-only participant {participant_id}: checkpoint={program_path} "
        f"(train_loglik={train_loglik:.4f}, val_loglik={val_loglik:.4f})"
    )

    test_loglik = prev_loglik_row.get("test_loglik") if prev_loglik_row else None
    if test_loglik not in (None, ""):
        test_loglik = float(test_loglik)
    else:
        test_eval = _evaluate_loglik_for_dataset(
            dataset, choose_fn, test_trials, n_seeds=n_eval_seeds
        )
        test_loglik = float(test_eval["avg_loglik"])

    refinement_dir = output_dir / "refinement"
    if save_artifacts:
        refinement_dir.mkdir(parents=True, exist_ok=True)

    if not _val_loglik_below_refinement_threshold(val_loglik, refinement_val_threshold):
        print(
            f"Refinement skipped: val_loglik={float(val_loglik):.6f} "
            f">= threshold={float(refinement_val_threshold):.6f}"
        )
        if save_artifacts:
            _write_refinement_results_json(
                refinement_dir,
                {
                    "phase": "refinement",
                    "dataset": dataset,
                    "participant_id": int(participant_id),
                    "n_iterations": int(n_iterations),
                    "refinement_skipped": True,
                    "refinement_val_threshold": float(refinement_val_threshold),
                    "checkpoint": {
                        "program_id": BEST_PROGRAM_FILENAME,
                        "train_loglik": train_loglik,
                        "val_loglik": val_loglik,
                        "test_loglik": test_loglik,
                    },
                    "gated_test_loglik": None,
                },
            )
        return {
            "participant_id": participant_id,
            "train_loglik": train_loglik,
            "val_loglik": val_loglik,
            "test_loglik": test_loglik,
            "gated_test_loglik": None,
            "refinement_skipped": True,
        }

    print(
        f"Refinement triggered: val_loglik={float(val_loglik):.6f} "
        f"< threshold={float(refinement_val_threshold):.6f}"
    )
    evolution_pool_dir = prev_participant_dir / "evolution_elite_pool"
    refine_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "client": client,
        "model_name": model_name,
        "train_trials": train_trials,
        "val_trials": val_trials,
        "test_trials": test_trials,
        "n_iterations": n_iterations,
        "n_candidates_per_iteration": n_candidates_per_iteration,
        "fresh_n_candidates": fresh_n_candidates,
        "sample_size": sample_size,
        "sample_parents": sample_parents,
        "sampled_parents_decay": sampled_parents_decay,
        "elite_pool_size": elite_pool_size,
        "participant_id": int(participant_id),
        "split_ratio": float(split_ratio),
        "split_seed": int(split_seed),
        "max_prompt_train_trials": max_prompt_train_trials,
        "max_prompt_trials_per_problem": max_prompt_trials_per_problem,
        "llm_max_tokens": llm_max_tokens,
        "max_workers": max_workers,
        "n_eval_seeds": n_eval_seeds,
        "fitness_metric": "loglik",
        "output_path": output_dir,
        "save_artifacts": save_artifacts,
        "wandb_module": wandb_module,
        "wandb_step_offset": 0,
        "fresh_parent_code": initial_code,
        "fresh_parent_train_loglik": train_loglik,
        "fresh_parent_val_loglik": val_loglik,
        "run_prompts_dir": run_prompts_dir,
        "max_parent_chars": max_parent_chars,
        "warn_parent_truncation_ratio": warn_parent_truncation_ratio,
        "early_stop_iters": early_stop_iters,
        "hard_prompt_token_cap": hard_prompt_token_cap,
        "strict_prompt_budget": strict_prompt_budget,
        "prompt_token_estimator": prompt_token_estimator,
        "max_error_prompt_chars": max_error_prompt_chars,
        "error_feedback_mode": error_feedback_mode,
        "mdl_lambda": mdl_lambda,
    }
    if evolution_pool_dir.is_dir() and (evolution_pool_dir / "pool_manifest.json").exists():
        ref_parents, ref_vals = _load_evolution_elite_pool(
            evolution_pool_dir, split_ratio=split_ratio
        )
        if save_artifacts:
            out_pool = output_dir / "evolution_elite_pool"
            if out_pool.exists():
                shutil.rmtree(out_pool)
            shutil.copytree(evolution_pool_dir, out_pool)
        print(
            f"Loaded evolution elite pool ({len(ref_parents)} programs) from {evolution_pool_dir}"
        )
        refine_kwargs["evolution_elite_parents"] = ref_parents
        refine_kwargs["evolution_elite_val_logliks"] = ref_vals
    else:
        refine_kwargs["initial_code"] = initial_code
        refine_kwargs["initial_train_loglik"] = train_loglik
        refine_kwargs["initial_val_loglik"] = val_loglik
    gated_test_loglik = run_loglik_refinement_phase(**refine_kwargs)

    if gated_test_loglik is not None and wandb_module is not None:
        final_refine_log = {
            f"p{participant_id}_gated_test_loglik": gated_test_loglik,
            f"p{participant_id}_train_loglik": train_loglik,
            f"p{participant_id}_val_loglik": val_loglik,
            f"p{participant_id}_test_loglik": test_loglik,
        }
        _wandb_log_participant_metrics(
            wandb_module,
            final_refine_log,
            int(participant_id),
            int(n_iterations),
        )

    return {
        "participant_id": participant_id,
        "train_loglik": train_loglik,
        "val_loglik": val_loglik,
        "test_loglik": test_loglik,
        "gated_test_loglik": gated_test_loglik,
        "refinement_skipped": False,
    }


def run_choice13k_refine_participant_from_checkpoint(**kwargs: Any) -> Dict[str, Any]:
    return run_loglik_refine_participant_from_checkpoint(**kwargs, dataset="choice13k")


def run_loglik_refine_from_prev_experiment(
    *,
    dataset: str,
    client: OpenAI,
    model_name: str,
    participants: List[int],
    prev_exp_path: Path,
    output_dir: Path,
    split_ratio: float,
    split_seed: int,
    data_path: str = "data",
    filter_mixed_gambles: bool = False,
    n_iterations: int,
    n_candidates_per_iteration: int,
    fresh_n_candidates: int = 0,
    sample_size: int,
    sample_parents: bool,
    sampled_parents_decay: bool = True,
    elite_pool_size: Optional[int],
    max_prompt_train_trials: int,
    max_prompt_trials_per_problem: int,
    llm_max_tokens: int,
    max_workers: int,
    n_eval_seeds: int,
    wandb_module: Optional[Any] = None,
    parallel_participants: bool = False,
    refinement_val_threshold: float = -1.0,
    fitness_metric: str = "loglik",
    cpc18_official_mse: bool = False,
    run_prompts_dir: Optional[str] = None,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_parent_chars: int = 6000,
    warn_parent_truncation_ratio: float = 0.5,
    early_stop_iters: Optional[int] = None,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    max_error_prompt_chars: int = 1200,
    error_feedback_mode: str = DEFAULT_ERROR_FEEDBACK_MODE,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    mdl_lambda: float = 0.0,
) -> None:
    """Refine-only across participants; copy prior loglik CSV and update gated_test_loglik."""
    prev_exp_path = prev_exp_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_workers, candidate_workers_per_participant = _parallel_participant_pool_sizes(
        max_workers, n_candidates_per_iteration, parallel_participants
    )
    if parallel_participants:
        print(
            "[INFO] Parallel participants enabled: "
            f"participant_workers={participant_workers}, "
            f"candidate_workers_per_participant={candidate_workers_per_participant}"
        )

    prev_details_csv = prev_exp_path / "participant_details_loglik.csv"
    loglik_rows = _load_participant_details_loglik_csv(prev_details_csv)
    rows_by_pid: Dict[int, Dict[str, Any]] = {
        int(row["participant_id"]): dict(row) for row in loglik_rows
    }
    for pid in participants:
        pid_int = int(pid)
        if pid_int not in rows_by_pid:
            rows_by_pid[pid_int] = {"participant_id": pid_int}

    out_details_csv = output_dir / "participant_details_loglik.csv"
    shutil.copy2(prev_details_csv, out_details_csv)
    print(f"Copied {prev_details_csv} -> {out_details_csv}")

    _clear_gated_test_loglik_in_loglik_rows(list(rows_by_pid.values()))
    all_pids = sorted(
        {int(r["participant_id"]) for r in loglik_rows} | {int(p) for p in participants}
    )
    cleared_rows = [rows_by_pid[pid] for pid in all_pids]
    _write_participant_details_loglik_csv(
        out_details_csv,
        cleared_rows,
        dataset=dataset,
        fitness_metric=fitness_metric,
        cpc18_official_mse=cpc18_official_mse,
    )
    _write_refine_summary_loglik_csv(output_dir, cleared_rows)
    print("Cleared gated_test_loglik in participant_details_loglik.csv for new refine run.")

    def _commit_participant_refine_metrics(participant_id: int, metrics: Dict[str, Any]) -> None:
        """Update shared experiment CSVs (main thread / lock only; never call from worker threads)."""
        with _SHARED_EXPERIMENT_CSV_LOCK:
            prev_row = rows_by_pid.get(int(participant_id))
            if prev_row is None:
                prev_row = {"participant_id": participant_id}
                rows_by_pid[int(participant_id)] = prev_row
            for key in ("train_loglik", "val_loglik", "test_loglik"):
                if metrics.get(key) is not None:
                    prev_row[key] = metrics[key]
            prev_row["gated_test_loglik"] = metrics.get("gated_test_loglik")
            updated_rows = [rows_by_pid[pid] for pid in all_pids]
            _write_participant_details_loglik_csv(
                out_details_csv,
                updated_rows,
                dataset=dataset,
                fitness_metric=fitness_metric,
                cpc18_official_mse=cpc18_official_mse,
            )
            _write_refine_summary_loglik_csv(output_dir, updated_rows)
        gated_str = (
            f"{float(metrics['gated_test_loglik']):.6f}"
            if metrics.get("gated_test_loglik") is not None
            else "None"
        )
        print(
            f"Updated CSV for participant {participant_id}: "
            f"gated_test_loglik={gated_str}"
        )

    def _refine_one_participant(participant_id: int) -> Tuple[int, Dict[str, Any]]:
        # Snapshot row so parallel workers never read shared dict state mid-update.
        prev_row = rows_by_pid.get(int(participant_id))
        prev_loglik_snapshot = dict(prev_row) if prev_row is not None else None
        participant_output_dir = output_dir / f"participant_{participant_id}"
        metrics = run_loglik_refine_participant_from_checkpoint(
            dataset=dataset,
            client=client,
            model_name=model_name,
            participant_id=int(participant_id),
            prev_exp_path=prev_exp_path,
            output_dir=participant_output_dir,
            prev_loglik_row=prev_loglik_snapshot,
            split_ratio=split_ratio,
            split_seed=split_seed,
            data_path=data_path,
            filter_mixed_gambles=filter_mixed_gambles,
            n_iterations=n_iterations,
            n_candidates_per_iteration=n_candidates_per_iteration,
            fresh_n_candidates=fresh_n_candidates,
            sample_size=sample_size,
            sample_parents=sample_parents,
            sampled_parents_decay=sampled_parents_decay,
            elite_pool_size=elite_pool_size,
            max_prompt_train_trials=max_prompt_train_trials,
            max_prompt_trials_per_problem=max_prompt_trials_per_problem,
            llm_max_tokens=llm_max_tokens,
            max_workers=candidate_workers_per_participant,
            n_eval_seeds=n_eval_seeds,
            save_artifacts=True,
            wandb_module=wandb_module,
            refinement_val_threshold=refinement_val_threshold,
            run_prompts_dir=run_prompts_dir,
            psych_dataset_split=psych_dataset_split,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            max_parent_chars=max_parent_chars,
            warn_parent_truncation_ratio=warn_parent_truncation_ratio,
            early_stop_iters=early_stop_iters,
            hard_prompt_token_cap=hard_prompt_token_cap,
            strict_prompt_budget=strict_prompt_budget,
            prompt_token_estimator=prompt_token_estimator,
            max_error_prompt_chars=max_error_prompt_chars,
            error_feedback_mode=error_feedback_mode,
            max_observed_trials_per_participant=max_observed_trials_per_participant,
            limited_data_protocol=limited_data_protocol,
            limited_train_val=limited_train_val,
            mdl_lambda=mdl_lambda,
        )
        return int(participant_id), metrics

    if parallel_participants:
        with ThreadPoolExecutor(max_workers=participant_workers) as pool:
            futures = {
                pool.submit(_refine_one_participant, int(pid)): int(pid) for pid in participants
            }
            for fut in as_completed(futures):
                participant_id, metrics = fut.result()
                _commit_participant_refine_metrics(participant_id, metrics)
    else:
        for participant_id in participants:
            _pid, metrics = _refine_one_participant(int(participant_id))
            _commit_participant_refine_metrics(_pid, metrics)

    print(f"Wrote participant loglik details -> {out_details_csv}")


def run_choice13k_refine_from_prev_experiment(**kwargs: Any) -> None:
    return run_loglik_refine_from_prev_experiment(**kwargs, dataset="choice13k")


def format_trials_to_text(trials: List[Dict[str, Any]], dataset: str = "choice13k") -> str:
    """Convert trials to numbered text for prompt (Psych-101 schemas + legacy CPC18)."""
    if trials:
        prob0 = trials[0].get("problem", {})
        psych_alias = prob0.get("dataset_alias")
        if is_psych101_dataset(dataset) or (
            psych_alias and is_psych101_dataset(str(psych_alias))
        ) or prob0.get("schema_type") in ("A", "B", "C", "D"):
            return format_trials_for_prompt(trials, max_trials=len(trials))
    lines = []
    for idx, t in enumerate(trials):
        if dataset == "cpc18":
            prob = t["problem"]
            action = t["action"]
            lines.append(
                f"{idx+1}. Problem: Option A (Ha={prob['Ha']}, pHa={prob['pHa']}, La={prob['La']}, "
                f"LotShapeA={prob['LotShapeA']}, LotNumA={prob['LotNumA']}); "
                f"Option B (Hb={prob['Hb']}, pHb={prob['pHb']}, Lb={prob['Lb']}, "
                f"LotShapeB={prob['LotShapeB']}, LotNumB={prob['LotNumB']}); "
                f"Amb={prob['Amb']}, Corr={prob['Corr']}; Observed action: {action}"
            )
        else:
            prob = t["problem"]
            if "gamble_A" in prob and "gamble_B" in prob:
                prob_a = prob["gamble_A"]["probs"]
                rew_a = prob["gamble_A"]["rewards"]
                prob_b = prob["gamble_B"]["probs"]
                rew_b = prob["gamble_B"]["rewards"]
                has_fb = prob.get("has_feedback", False)
                action = t["action"]
                lines.append(
                    f"{idx+1}. Problem: Option A probs {prob_a} rewards {rew_a}; "
                    f"Option B probs {prob_b} rewards {rew_b}; has_feedback={has_fb}; "
                    f"Observed action: {action}"
                )
            else:
                lines.append(format_trials_for_prompt([t], max_trials=1))
    return "\n".join(lines)


def experiment_to_trials(exp: Experiment) -> Tuple[List[Dict[str, Any]], list]:
    """Convert one Choice13k experiment into trial records without splitting."""
    options = exp.blocks[0].option_keys
    all_trials = []
    history_accum = []
    for block in exp.blocks:
        for trial in block.trials:
            history_entry = {"action": trial.action, "feedback": trial.feedback}
            all_trials.append(
                {
                    "problem": {
                        "gamble_A": {
                            "probs": block.gamble_A.probs,
                            "rewards": block.gamble_A.rewards,
                        },
                        "gamble_B": {
                            "probs": block.gamble_B.probs,
                            "rewards": block.gamble_B.rewards,
                        },
                        "option_keys": options,
                        "has_feedback": block.has_feedback,
                    },
                    "history": list(history_accum),
                    "options": options,
                    "action": trial.action,
                }
            )
            history_accum.append(history_entry)
    return all_trials, options


def trials_from_blocks_chronological(
    exp: Experiment, block_indices: set
) -> List[Dict[str, Any]]:
    """Trials from selected blocks in original order; history only within each block (no cross-problem leakage)."""
    out: List[Dict[str, Any]] = []
    for bi, block in enumerate(exp.blocks):
        if bi not in block_indices:
            continue
        options = block.option_keys
        history_accum: List[Dict[str, Any]] = []
        for trial in block.trials:
            out.append(
                {
                    "problem": {
                        "gamble_A": {
                            "probs": block.gamble_A.probs,
                            "rewards": block.gamble_A.rewards,
                        },
                        "gamble_B": {
                            "probs": block.gamble_B.probs,
                            "rewards": block.gamble_B.rewards,
                        },
                        "option_keys": options,
                        "has_feedback": block.has_feedback,
                    },
                    "history": list(history_accum),
                    "options": options,
                    "action": trial.action,
                }
            )
            history_accum.append({"action": trial.action, "feedback": trial.feedback})
    return out


def split_trials(
    exp: Experiment,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], list]:
    """
    Split Choice13k into train / val / test by **problem (block)**.

    ``split_ratio`` is the train **fraction** of blocks. The remainder is split between validation and test
    with sizes differing by at most one block (when odd, validation receives one more block than test).
    """
    n_blocks = len(exp.blocks)
    if n_blocks < 3:
        raise ValueError(
            f"Choice13k train/val/test split requires at least 3 problems (blocks); got {n_blocks}."
        )
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")

    rng = np.random.default_rng(split_seed)
    perm = np.arange(n_blocks)
    rng.shuffle(perm)

    n_train = int(n_blocks * split_ratio)
    n_train = max(1, min(n_train, n_blocks - 2))
    n_rem = n_blocks - n_train
    n_val = (n_rem + 1) // 2
    n_test = n_rem - n_val
    if n_val < 1:
        n_val = 1
        n_test = max(1, n_rem - 1)
        n_train = n_blocks - n_val - n_test
        n_train = max(1, n_train)
        n_rem = n_blocks - n_train
        n_val = n_rem // 2
        n_test = n_rem - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n_rem - 1)
        n_train = n_blocks - n_val - n_test
        n_train = max(1, n_train)

    train_blocks = set(perm[:n_train].tolist())
    val_blocks = set(perm[n_train : n_train + n_val].tolist())
    test_blocks = set(perm[n_train + n_val :].tolist())
    assert len(train_blocks) + len(val_blocks) + len(test_blocks) == n_blocks
    assert train_blocks.isdisjoint(val_blocks) and train_blocks.isdisjoint(test_blocks) and val_blocks.isdisjoint(test_blocks)

    train_trials = trials_from_blocks_chronological(exp, train_blocks)
    val_trials = trials_from_blocks_chronological(exp, val_blocks)
    test_trials = trials_from_blocks_chronological(exp, test_blocks)
    options = exp.blocks[0].option_keys
    return train_trials, val_trials, test_trials, options


def evaluate_program(choose_fn: Callable, trials: List[Dict[str, Any]], verbose: bool = False, n_seeds: int = 1) -> Dict[str, float]:
    """Evaluate a program on trials and return accuracy metrics.
    
    Args:
        choose_fn: The program function to evaluate
        trials: List of trial dictionaries
        verbose: Whether to print verbose output
        n_seeds: Number of evaluation runs to average (default: 1)
    
    Returns:
        Dictionary with averaged accuracy metrics across n_seeds runs
    """
    accuracies = []
    total = len(trials)
    max_errors_per_seed = 0
    first_error: Optional[Dict[str, str]] = None
    source_code = getattr(choose_fn, "__teh_source_code", None)

    for seed in range(n_seeds):
        correct = 0
        errors = 0
        for t in trials:
            try:
                pred = choose_fn(_problem_for_choose(t), t["history"])
                if pred is not None and pred == t["action"]:
                    correct += 1
            except Exception as e:
                errors += 1
                if first_error is None and isinstance(source_code, str) and source_code:
                    first_error = _build_invalid_program_error_entry(code=source_code, exc=e)
                if verbose and errors <= 3 and seed == 0:
                    print(f"  Evaluation error: {e}")
        acc = correct / total if total > 0 else 0.0
        accuracies.append(acc)
        max_errors_per_seed = max(max_errors_per_seed, errors)

    avg_acc = float(np.mean(accuracies)) if accuracies else 0.0
    correct = int(avg_acc * total)
    result = {
        "accuracy": avg_acc,
        "total": total,
        "correct": correct,
        "errors": max_errors_per_seed,
        "first_error": first_error,
    }
    if verbose and max_errors_per_seed > 0:
        print(f"  Total evaluation errors: {max_errors_per_seed}/{total} (max per seed)")
    return result


def evaluate_choice13k_program(
    choose_fn: Callable,
    trials: List[Dict[str, Any]],
    verbose: bool = False,
    n_seeds: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate Choice13k-style programs where choose(problem, history) returns P(action=1).

    Accepts either:
      - float in [0, 1], or
      - int/bool 0/1 (coerced to degenerate Bernoulli probabilities).
    """
    total = len(trials)
    seed_avg_accs: List[float] = []
    seed_avg_logliks: List[float] = []
    total_errors = 0
    max_errors_per_seed = 0
    first_error: Optional[Dict[str, str]] = None
    source_code = getattr(choose_fn, "__teh_source_code", None)

    def _one_pass(seed_idx: int) -> Tuple[float, float, int]:
        nonlocal first_error
        loglik_acc = 0.0
        correct = 0
        errors = 0
        for t in trials:
            y = int(t["action"])
            try:
                p_raw = choose_fn(_problem_for_choose(t), t["history"])
                p_use = _parse_choice13k_choose_output(p_raw)
            except Exception as e:
                errors += 1
                if first_error is None and isinstance(source_code, str) and source_code:
                    first_error = _build_invalid_program_error_entry(code=source_code, exc=e)
                if verbose and errors <= 3 and seed_idx == 0:
                    print(f"  Evaluation error: {e}")
                p = 0.5
                p_clamped = _clamp_choice13k_probability(p)
                loglik_acc += y * np.log(p_clamped) + (1 - y) * np.log(1.0 - p_clamped)
                pred = 1 if p >= 0.5 else 0
                correct += int(pred == y)
                continue

            p = _clamp_choice13k_probability(p_use)
            loglik_acc += y * np.log(p) + (1 - y) * np.log(1.0 - p)
            if isinstance(p_raw, float):
                pred = 1 if p_raw >= 0.5 else 0
            else:
                pred = 1 if int(p_raw) == 1 else 0
            correct += int(pred == y)

        avg_ll = loglik_acc / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_ll, acc, errors

    for seed in range(n_seeds):
        avg_ll, acc, errs = _one_pass(seed)
        seed_avg_logliks.append(avg_ll)
        seed_avg_accs.append(acc)
        total_errors += errs
        max_errors_per_seed = max(max_errors_per_seed, errs)

    avg_acc = float(np.mean(seed_avg_accs)) if seed_avg_accs else 0.0
    # Any per-trial exception (e.g. KeyError on sanitized-away oracle fields)
    # invalidates the program: do not report a chance-filled log-likelihood as
    # usable fitness. Callers already gate on errors==0 / runtime_valid.
    if max_errors_per_seed > 0:
        avg_loglik = float("-inf")
    else:
        avg_loglik = float(np.mean(seed_avg_logliks)) if seed_avg_logliks else float("-inf")
    correct = int(round(avg_acc * total))
    if verbose and max_errors_per_seed > 0:
        print(
            f"  Evaluation errors: {total_errors} total across {n_seeds} seed(s) "
            f"(max {max_errors_per_seed} per seed, {total} trials)"
        )
    return {
        "accuracy": avg_acc,
        "avg_loglik": avg_loglik,
        "total": total,
        "correct": correct,
        "errors": max_errors_per_seed,
        "total_errors": total_errors,
        "first_error": first_error,
    }


def evaluate_cpc18_program(choose_fn: Callable, trials: List[Dict[str, Any]], verbose: bool = False, n_seeds: int = 1) -> Dict[str, float]:
    """
    Evaluate a CPC18 program on trials and return accuracy metrics (trial-level).
    
    This is the auxiliary accuracy metric for CPC18 Track II (not the official MSE metric).
    Same interface as evaluate_program for consistency.
    
    Args:
        choose_fn: The program function to evaluate (takes problem dict and history)
        trials: List of trial dictionaries
        verbose: Whether to print verbose output
        n_seeds: Number of evaluation runs to average (default: 1)
    
    Returns:
        Dictionary with averaged accuracy metrics across n_seeds runs
    """
    return evaluate_program(choose_fn, trials, verbose, n_seeds)


def evaluate_cpc18_split_program(
    choose_fn: Callable, trials: List[Dict[str, Any]], verbose: bool = False, n_seeds: int = 1
) -> Dict[str, Any]:
    """
    CPC18 non-official (held-out) evaluation: P(B) as float in [0,1] (choice13k-style) or
    int 0/1 coerced to degenerate Bernoulli probabilities. Mean Bernoulli log-lik and threshold acc.
    """
    total = len(trials)
    seed_avg_accs: List[float] = []
    seed_avg_logliks: List[float] = []
    total_errors = 0
    max_errors_per_seed = 0
    first_error: Optional[Dict[str, str]] = None
    source_code = getattr(choose_fn, "__teh_source_code", None)

    def _one_pass(seed_idx: int) -> Tuple[float, float, int]:
        nonlocal first_error
        loglik_acc = 0.0
        correct = 0
        errors = 0
        for t in trials:
            y = int(t["action"])
            try:
                p_raw = choose_fn(_problem_for_choose(t), t["history"])
            except Exception as e:
                errors += 1
                if first_error is None and isinstance(source_code, str) and source_code:
                    first_error = _build_invalid_program_error_entry(code=source_code, exc=e)
                if verbose and errors <= 3 and seed_idx == 0:
                    print(f"  Evaluation error: {e}")
                p = 0.5
                p_clamped = min(max(p, 1e-9), 1.0 - 1e-9)
                loglik_acc += y * np.log(p_clamped) + (1 - y) * np.log(1.0 - p_clamped)
                pred = 1 if p >= 0.5 else 0
                correct += int(pred == y)
                continue
            if isinstance(p_raw, bool) or (isinstance(p_raw, (int, np.integer)) and int(p_raw) in (0, 1)):
                p_use = 1.0 if int(p_raw) == 1 else 0.0
            elif isinstance(p_raw, float):
                p_use = p_raw
            else:
                raise TypeError(f"choose must return float or 0/1, got {type(p_raw)}")
            if not (0.0 <= p_use <= 1.0):
                raise ValueError(f"invalid probability: {p_use!r}")
            p = min(max(p_use, 1e-9), 1.0 - 1e-9)
            loglik_acc += y * np.log(p) + (1 - y) * np.log(1.0 - p)
            if isinstance(p_raw, float):
                pred = 1 if p_raw >= 0.5 else 0
            else:
                pred = 1 if int(p_raw) == 1 else 0
            correct += int(pred == y)

        avg_ll = loglik_acc / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_ll, acc, errors

    for seed in range(n_seeds):
        avg_ll, acc, errs = _one_pass(seed)
        seed_avg_logliks.append(avg_ll)
        seed_avg_accs.append(acc)
        total_errors += errs
        max_errors_per_seed = max(max_errors_per_seed, errs)

    avg_acc = float(np.mean(seed_avg_accs)) if seed_avg_accs else 0.0
    if max_errors_per_seed > 0:
        avg_loglik = float("-inf")
    else:
        avg_loglik = float(np.mean(seed_avg_logliks)) if seed_avg_logliks else float("-inf")
    correct = int(round(avg_acc * total))
    if verbose and max_errors_per_seed > 0:
        print(
            f"  Evaluation errors: {total_errors} total across {n_seeds} seed(s) "
            f"(max {max_errors_per_seed} per seed, {total} trials)"
        )
    return {
        "accuracy": avg_acc,
        "avg_loglik": avg_loglik,
        "total": total,
        "correct": correct,
        "errors": max_errors_per_seed,
        "total_errors": total_errors,
        "first_error": first_error,
    }


def evaluate_cpc18_mse(choose_fn: Callable, trials: List[Dict[str, Any]], 
                       observed_blocks: Dict[int, np.ndarray], 
                       verbose: bool = False, n_seeds: int = 1) -> Dict[str, Any]:
    """
    Evaluate CPC18 program and compute block-level MSE (official CPC18 metric).
    
    Computes MSE matching cpc18_baselines formula:
    MSE = 100 * mean((predicted_block_rate - observed_block_rate)^2)
    Averaged over all 5 blocks and all problems.
    
    If the program crashes or produces no valid prediction for any trial in a block,
    the evaluation is marked invalid and MSE is set to Infinity (no silent default to A).
    
    Args:
        choose_fn: The program function to evaluate
        trials: List of trial dictionaries (must include problem_id and block_id)
        observed_blocks: Dict mapping problem_id to observed B-rates (5-element array)
        verbose: Whether to print verbose output
        n_seeds: Number of evaluation runs to average (default: 1)
    
    Returns:
        Dictionary with "mse", "valid", and component metrics. valid=False if any crash or invalid prediction.
    """
    # Group trials by problem_id
    problems_dict = {}
    for trial in trials:
        problem_id = trial["problem_id"]
        if problem_id not in problems_dict:
            problems_dict[problem_id] = []
        problems_dict[problem_id].append(trial)
    
    all_mse_per_problem = []
    all_predicted_blocks = {}
    all_observed_blocks = {}
    evaluation_valid = True
    
    for seed in range(n_seeds):
        predicted_blocks = {}  # problem_id -> array of 5 predicted B-rates
        seed_valid = True
        
        for problem_id, problem_trials in problems_dict.items():
            # Group trials by block_id
            blocks_dict = {}
            for trial in problem_trials:
                block_id = trial["block_id"]
                if block_id not in blocks_dict:
                    blocks_dict[block_id] = []
                blocks_dict[block_id].append(trial)
            
            # Predict for each block
            predicted_rates = np.zeros(5)
            for block_id in range(1, 6):
                block_trials = blocks_dict.get(block_id, [])
                if len(block_trials) > 0:
                    # Run predictions for all trials in this block
                    b_predictions = []
                    for trial in block_trials:
                        try:
                            pred = choose_fn(_problem_for_choose(trial), trial["history"])
                            if pred is not None:
                                b_predictions.append(int(pred == 1))  # 1 if B chosen, 0 if A
                            else:
                                # Invalid prediction (None) -> mark evaluation invalid
                                seed_valid = False
                        except Exception as e:
                            if verbose and seed == 0:
                                print(f"  Prediction error for problem {problem_id}, block {block_id}: {e}")
                            # Do not default to A; mark evaluation invalid
                            seed_valid = False
                    
                    # If no valid predictions for this block, evaluation is invalid
                    if len(b_predictions) == 0 or len(b_predictions) < len(block_trials):
                        seed_valid = False
                    if len(b_predictions) > 0:
                        predicted_rates[block_id - 1] = np.mean(b_predictions)
            
            predicted_blocks[problem_id] = predicted_rates
        
        if not seed_valid:
            evaluation_valid = False
        
        # Compute MSE per problem (matching baseline formula) only when valid
        mse_per_problem = []
        for problem_id in predicted_blocks.keys():
            if problem_id in observed_blocks:
                pred_rates = predicted_blocks[problem_id]
                obs_rates = observed_blocks[problem_id]
                # MSE = 100 * mean((pred - obs)^2) per problem
                mse = 100 * np.mean((pred_rates - obs_rates) ** 2)
                mse_per_problem.append(mse)
        
        all_mse_per_problem.append(mse_per_problem)
        
        if seed == 0:
            all_predicted_blocks = predicted_blocks.copy()
            all_observed_blocks = observed_blocks.copy()
    
    # If invalid: return Infinity MSE and valid=False
    if not evaluation_valid:
        return {
            "mse": float('inf'),
            "mse_per_problem": [],
            "n_problems": len(problems_dict),
            "predicted_blocks": {k: v.tolist() for k, v in all_predicted_blocks.items()},
            "observed_blocks": {k: v.tolist() for k, v in all_observed_blocks.items()},
            "valid": False,
        }
    
    # Average MSE across seeds
    if all_mse_per_problem:
        avg_mse_per_problem = np.mean(all_mse_per_problem, axis=0)
        total_mse = np.mean(avg_mse_per_problem)
    else:
        total_mse = float('inf')
        avg_mse_per_problem = []
    
    return {
        "mse": total_mse,
        "mse_per_problem": avg_mse_per_problem.tolist() if len(avg_mse_per_problem) > 0 else [],
        "n_problems": len(problems_dict),
        "predicted_blocks": {k: v.tolist() for k, v in all_predicted_blocks.items()},
        "observed_blocks": {k: v.tolist() for k, v in all_observed_blocks.items()},
        "valid": True,
    }


def load_gridworld_data(data_path: str, num_blocks: int, num_walls: int, agent_id: int, 
                        num_datapoints: int = 100, start_idx: int = 0):
    """Load gridworld trajectory data for a specific problem config and agent type.
    
    Args:
        data_path: Path to data directory
        num_blocks: Number of blocks in the problem
        num_walls: Number of walls in the problem
        agent_id: Agent type ID (0-indexed)
        num_datapoints: Number of datapoints to load
        start_idx: Starting index for datapoints (for train/test split)
    
    Returns:
        Dictionary with 'states' and 'actions' for evaluation
    """
    _load_gridworld_stack()
    data_folder = f"{data_path}/num_blocks{num_blocks}/num_walls{num_walls}"
    data_file = f"{data_folder}/gt_fsm_traj_data_1agents.msgpack"
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load data structure (similar to plot_and_eval.py)
    with open(data_file, "rb") as f:
        serialized_data = f.read()
    
    # Create target structure
    num_steps = 100
    action_target = jnp.zeros((20, num_datapoints, num_steps, 1))
    agent_id_target = jnp.zeros((20, num_datapoints, num_steps))
    state_target = {
        'agent_id': agent_id_target,
        'agent_locations': jnp.zeros((20, num_datapoints, num_steps, 1, 2)),
        'agent_inventory': jnp.zeros((20, num_datapoints, num_steps, 1)),
        'agent_inventory_colors': jnp.zeros((20, num_datapoints, num_steps, 1, 3)),
        'block_colors': jnp.zeros((20, num_datapoints, num_steps, num_blocks, 3)),
        'block_locations': jnp.zeros((20, num_datapoints, num_steps, num_blocks, 2)),
        'time': jnp.zeros((20, num_datapoints, num_steps)),
        'terminal': jnp.zeros((20, num_datapoints, num_steps)),
        'wall_locations': jnp.zeros((20, num_datapoints, num_steps, num_walls + 2 * (10 * 2 - 1) + 2, 2)),
    }
    target = {
        'states': state_target,
        'actions': action_target,
        'agent_ids': agent_id_target,
    }
    
    loaded_data = flax.serialization.from_bytes(target, serialized_data)
    
    # Extract data for the specific agent type, with start_idx for train/test split
    end_idx = min(start_idx + num_datapoints, loaded_data['states']['agent_locations'].shape[1])
    actual_num = end_idx - start_idx
    
    agent_data = {
        'states': jax.tree.map(lambda x: x[agent_id, start_idx:end_idx, :, ...], loaded_data['states']),
        'actions': loaded_data['actions'][agent_id, start_idx:end_idx, :, :],
    }
    
    return agent_data


def evaluate_gridworld_program(agent_code: str, data_path: str, num_blocks: int, num_walls: int, 
                                agent_id: int, num_datapoints: int = 100, num_steps: int = 20,
                                verbose: bool = False, n_seeds: int = 1, 
                                evaluate_on_observed: bool = False) -> Dict[str, float]:
    """Evaluate a gridworld program on trajectory data using the same logic as ROTE.
    
    Args:
        agent_code: The program code to evaluate
        data_path: Path to gridworld data directory
        num_blocks: Number of blocks in the problem
        num_walls: Number of walls in the problem
        agent_id: Agent type ID (0-indexed)
        num_datapoints: Number of datapoints to evaluate on
        num_steps: Number of steps to evaluate (default: 20)
        verbose: Whether to print verbose output
        n_seeds: Number of evaluation runs to average (default: 1)
        evaluate_on_observed: If True, evaluate on first 20 steps (matching ROTE's training/weighting).
                             If False, evaluate on future steps (matching ROTE's evaluation).
    
    Returns:
        Dictionary with accuracy metrics
    """
    _load_gridworld_stack()
    framework = AgentExecutionFramework()
    
    # Compile the agent
    try:
        agent = framework.compile_agent(agent_code, num_agents=1, num_blocks=num_blocks)
    except Exception as e:
        if verbose:
            print(f"  Compilation error: {e}")
        return {"accuracy": 0.0, "total": 0, "correct": 0, "errors": 1}
    
    # Create a dummy args object for make_dataloader
    class DummyArgs:
        def __init__(self):
            self.data_path = data_path
            self.num_agents = 1
            self.num_datapoints_per_agent = num_datapoints
            self.num_steps = num_steps
            self.group = False
            self.flip_quarter = True  # Data files use _flip_quarter extension
            self.env_size = 10
            self.as_images = False
    
    dummy_args = DummyArgs()
    
    accuracies = []
    total_steps = 0
    correct_steps = 0
    
    for seed in range(n_seeds):
        try:
            # Use make_dataloader to load data (same as ROTE)
            # For train: use first 80 datapoints, for test: use datapoints 80-100
            start_idx = 0 if num_datapoints >= 80 else 80
            num_datapoints_to_load = num_datapoints if num_datapoints >= 80 else 20
            
            # Create dataloader using make_dataloader (same as plot_and_eval.py)
            dataloader = make_dataloader(
                dummy_args,
                num_agents_to_sample=1,
                num_datapoints_per_agent_to_sample=num_datapoints_to_load,
                training=False,
                epoch=0,
                num_blocks=num_blocks,
                num_walls=num_walls,
                agent_indices=[agent_id]
            )
            
            datapoint = next(dataloader)
            
            # Evaluate on datapoints (same structure as eval_fsm_bootstrap)
            seed_correct = 0
            seed_total = 0
            
            # ROTE evaluates on the last datapoint per agent (line 1234: x[a_idx, -1, :20+num_future_steps])
            # Match ROTE exactly: use -1 for datapoint index, iterate through agents
            # But we only have 1 agent, so we'll evaluate on multiple datapoints for better statistics
            for dp_idx in range(num_datapoints_to_load):
                try:
                    # Extract data sample exactly like ROTE (line 1234)
                    # ROTE: data_sample = jax.tree.map(lambda x: x[a_idx, -1, :20+num_future_steps], datapoint)
                    # We use dp_idx instead of -1 to iterate through datapoints
                    # ROTE uses :20+num_future_steps where num_future_steps=20, so :40 steps total
                    data_sample = jax.tree.map(lambda x: x[0, dp_idx, :20+num_steps], datapoint)
                    
                    # Extract initial trajectory (first 20 steps) from data_sample
                    initial_states_traj = jax.tree.map(lambda x: x[:20], data_sample['states'])
                    initial_actions_traj = jax.tree.map(lambda x: x[:20], data_sample['actions'])
                    
                    # Convert JAX arrays to numpy arrays (same as ROTE does implicitly)
                    def to_numpy(x):
                        if isinstance(x, (jnp.ndarray, jax.Array)):
                            return np.array(x)
                        return x
                    
                    if evaluate_on_observed:
                        # TRAIN MODE: Evaluate on first 20 steps (matching ROTE's training/weighting phase)
                        # This matches how ROTE calculates log_prob_hypothesis (baselines/gridROTE.py line 469-496)
                        for timestep in range(initial_actions_traj.shape[0] - 1):  # 0 to 18 (19 steps)
                            try:
                                state = jax.tree.map(lambda x: x[timestep], initial_states_traj)
                                state = jax.tree.map(to_numpy, state)
                                if len(state['agent_locations']) == 1:
                                    state['agent_id'] = 0
                                
                                # Get ground truth action for this timestep
                                gt_action = int(initial_actions_traj[timestep][0])
                                
                                # Get prediction from agent
                                predicted_action = framework.execute_agent(agent, state)
                                
                                # Convert action to int (same as ROTE)
                                action_space = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
                                action_space_2 = ["stay", "right", "left", "down", "up", "interact"]
                                
                                if isinstance(predicted_action, tuple):
                                    predicted_action = list(predicted_action)
                                elif isinstance(predicted_action, str):
                                    predicted_action = predicted_action.lower()
                                    predicted_action = action_space_2.index(predicted_action)
                                else:
                                    predicted_action = int(predicted_action)
                                
                                if predicted_action in action_space:
                                    predicted_action = action_space.index(predicted_action)
                                elif predicted_action in action_space_2:
                                    predicted_action = action_space_2.index(predicted_action)
                                
                                # Compare with ground truth (same as ROTE line 484)
                                if predicted_action == gt_action:
                                    seed_correct += 1
                                seed_total += 1
                                
                            except Exception as e:
                                seed_total += 1  # Count as incorrect
                    else:
                        # TEST MODE: Evaluate on future steps (matching ROTE's evaluation phase)
                        # Get ground truth future actions (from step 19 onwards) - exactly like ROTE (line 1244)
                        gt_future_actions = data_sample['actions'][19:]  # Shape (num_steps, num_env_agents) or (num_steps, 1)
                        
                        # If we don't have enough future actions, use what we have
                        if gt_future_actions.shape[0] < num_steps:
                            actual_num_steps = min(gt_future_actions.shape[0], num_steps)
                        else:
                            actual_num_steps = num_steps
                        
                        # Initialize environment for simulation (same as ROTE)
                        env = AutomaticityEnv(num_agents=1, size=10, max_steps=num_steps, 
                                              num_blocks=num_blocks, num_walls=num_walls)
                        
                        # Extract state at step 19 (end of initial trajectory) exactly like ROTE
                        state_at_t19 = jax.tree.map(lambda x: x[19], initial_states_traj)
                        
                        # Verify state_at_t19 is a dict (should be preserved by jax.tree.map)
                        if not isinstance(state_at_t19, dict):
                            # Try to convert if it's a list or tuple
                            if isinstance(state_at_t19, (list, tuple)) and len(state_at_t19) > 0:
                                # Maybe it's a list of dicts? Take the first one
                                if isinstance(state_at_t19[0], dict):
                                    state_at_t19 = state_at_t19[0]
                                else:
                                    seed_total += actual_num_steps
                                    continue
                            else:
                                seed_total += actual_num_steps
                                continue
                        
                        state_at_t19_np = jax.tree.map(to_numpy, state_at_t19)
                        
                        # Start from state at step 19 (end of initial trajectory) - exactly like ROTE
                        current_sim_state_pytree = state_at_t19_np
                        current_sim_state_pytree = State(
                            wall_locations=current_sim_state_pytree['wall_locations'],
                            agent_locations=current_sim_state_pytree['agent_locations'],
                            block_locations=current_sim_state_pytree['block_locations'],
                            agent_inventory=current_sim_state_pytree['agent_inventory'],
                            agent_inventory_colors=current_sim_state_pytree['agent_inventory_colors'],
                            block_colors=current_sim_state_pytree['block_colors'],
                            time=current_sim_state_pytree['time'],
                            terminal=False,
                            agent_id=0
                        )
                        current_obs = env.get_observation(current_sim_state_pytree)[0]
                        
                        # Simulate future steps (same as ROTE - use ground truth observations from data)
                        for step_idx in range(actual_num_steps):
                            if step_idx >= gt_future_actions.shape[0]:
                                break
                            
                            try:
                                # Get observation from ground truth data (same as ROTE line 1530)
                                # ROTE uses: current_obs = jax.tree.map(lambda x: x[19+step_idx+1], data_sample['states'])
                                current_obs_raw = jax.tree.map(lambda x: x[19+step_idx+1], data_sample['states'])
                                # Convert to numpy and ensure it's a dict
                                current_obs = jax.tree.map(to_numpy, current_obs_raw)
                                current_obs['agent_id'] = 0
                                
                                # Get prediction from agent
                                predicted_action = framework.execute_agent(agent, current_obs)
                                
                                # Extract ground truth action exactly like ROTE (line 1502, 1506)
                                gt_action_this_step = gt_future_actions[step_idx]  # (num_env_agents,) or (1,)
                                # For single agent, use index 0 (same as ROTE line 1506 with aid=0)
                                if hasattr(gt_action_this_step, '__len__') and len(gt_action_this_step) > 0:
                                    gt_action = int(gt_action_this_step[0])
                                else:
                                    gt_action = int(gt_action_this_step)
                                
                                # Convert action to int (same as ROTE)
                                action_space = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
                                action_space_2 = ["stay", "right", "left", "down", "up", "interact"]
                                
                                if isinstance(predicted_action, tuple):
                                    predicted_action = list(predicted_action)
                                elif isinstance(predicted_action, str):
                                    predicted_action = predicted_action.lower()
                                    predicted_action = action_space_2.index(predicted_action)
                                else:
                                    predicted_action = int(predicted_action)
                                
                                if predicted_action in action_space:
                                    predicted_action = action_space.index(predicted_action)
                                elif predicted_action in action_space_2:
                                    predicted_action = action_space_2.index(predicted_action)
                                
                                # Compare with ground truth
                                if predicted_action == gt_action:
                                    seed_correct += 1
                                seed_total += 1
                                
                            except Exception as e:
                                seed_total += 1  # Count as incorrect
                        
                except Exception as e:
                    if verbose:
                        print(f"  Error processing datapoint {dp_idx}: {e}")
                    # Count as incorrect based on mode
                    if evaluate_on_observed:
                        seed_total += 19  # First 20 steps minus 1 (timestep 0 to 18)
                    else:
                        seed_total += num_steps
                    continue
                        
        except Exception as e:
            if verbose:
                print(f"  Data loading error: {e}")
            seed_correct = 0
            seed_total = 1  # Avoid division by zero
        
        acc = seed_correct / seed_total if seed_total > 0 else 0.0
        accuracies.append(acc)
        total_steps = seed_total
        correct_steps = seed_correct
    
    # Average across seeds
    avg_acc = np.mean(accuracies) if accuracies else 0.0
    correct = int(avg_acc * total_steps) if total_steps > 0 else 0
    
    result = {"accuracy": avg_acc, "total": total_steps, "correct": correct, "errors": 0}
    return result


def _normalize_gridworld_action(predicted_action: Any) -> int:
    """Normalize agent output to action index 0-5 (stay, right, left, down, up, interact)."""
    action_space = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    action_space_2 = ["stay", "right", "left", "down", "up", "interact"]
    if isinstance(predicted_action, tuple):
        predicted_action = list(predicted_action)
    elif isinstance(predicted_action, str):
        predicted_action = predicted_action.lower()
        predicted_action = action_space_2.index(predicted_action)
    else:
        predicted_action = int(predicted_action)
    if predicted_action in action_space:
        return action_space.index(predicted_action)
    if predicted_action in action_space_2:
        return action_space_2.index(predicted_action)
    return int(predicted_action)


def _gridworld_correct_counts_first20(
    agent_codes: List[str],
    data_path: str,
    num_blocks: int,
    num_walls: int,
    agent_id: int,
    num_datapoints: int = 80,
    n_seeds: int = 1,
) -> List[float]:
    """Compute number of correct predictions per program on first 20 observed steps (train data).
    Same data/step logic as evaluate_gridworld_program(..., evaluate_on_observed=True).
    No epsilon smoothing: each step is correct (1) or wrong (0). Returns list of K scores.
    When n_seeds > 1, returns mean correct count per program across seeds.
    """
    _load_gridworld_stack()
    framework = AgentExecutionFramework()
    agents = []
    for code in agent_codes:
        try:
            agent = framework.compile_agent(code, num_agents=1, num_blocks=num_blocks)
            agents.append(agent)
        except Exception:
            agents.append(None)
    if not agents or all(a is None for a in agents):
        return [0.0] * len(agent_codes)

    class DummyArgs:
        def __init__(self):
            self.data_path = data_path
            self.num_agents = 1
            self.num_datapoints_per_agent = num_datapoints
            self.num_steps = 20
            self.group = False
            self.flip_quarter = True
            self.env_size = 10
            self.as_images = False

    dummy_args = DummyArgs()

    def to_numpy(x):
        if isinstance(x, (jnp.ndarray, jax.Array)):
            return np.array(x)
        return x

    correct_counts_per_seed = []
    for seed in range(n_seeds):
        try:
            dataloader = make_dataloader(
                dummy_args,
                num_agents_to_sample=1,
                num_datapoints_per_agent_to_sample=num_datapoints,
                training=False,
                epoch=0,
                num_blocks=num_blocks,
                num_walls=num_walls,
                agent_indices=[agent_id],
            )
            datapoint = next(dataloader)
            correct_sum = [1e-6] * len(agents)
            for dp_idx in range(num_datapoints):
                try:
                    data_sample = jax.tree.map(lambda x: x[0, dp_idx, :20 + 20], datapoint)
                    initial_states_traj = jax.tree.map(lambda x: x[:20], data_sample['states'])
                    initial_actions_traj = jax.tree.map(lambda x: x[:20], data_sample['actions'])
                    for timestep in range(initial_actions_traj.shape[0] - 1):
                        state = jax.tree.map(lambda x: x[timestep], initial_states_traj)
                        state = jax.tree.map(to_numpy, state)
                        if len(state['agent_locations']) == 1:
                            state['agent_id'] = 0
                        gt_action = int(initial_actions_traj[timestep][0])
                        for k, agent in enumerate(agents):
                            if agent is None:
                                continue
                            try:
                                pred = framework.execute_agent(agent, state)
                                pred_idx = _normalize_gridworld_action(pred)
                                if pred_idx == gt_action:
                                    correct_sum[k] += 1
                            except Exception:
                                pass
                except Exception:
                    continue
            correct_counts_per_seed.append(correct_sum)
        except Exception:
            correct_counts_per_seed.append([0] * len(agents))
    if not correct_counts_per_seed:
        return [0.0] * len(agent_codes)
    scores = np.mean(correct_counts_per_seed, axis=0)
    return scores.tolist()


def compute_gridworld_ensemble_weights(
    agent_codes: List[str],
    data_path: str,
    num_blocks: int,
    num_walls: int,
    agent_id: int,
    num_datapoints: int = 80,
    n_seeds: int = 1,
) -> List[float]:
    """Compute weights from correct-count scores on first 20 steps (ROTE-aligned).
    score_h = number of correct predictions on first 20 steps (no epsilon).
    scores = scores - max(scores); weights = exp(scores); weights = weights / sum(weights).
    """
    scores = np.array(
        _gridworld_correct_counts_first20(
            agent_codes, data_path, num_blocks, num_walls, agent_id,
            num_datapoints=num_datapoints, n_seeds=n_seeds,
        ),
        dtype=np.float64,
    )
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / weights.sum()
    return weights.tolist()


def evaluate_gridworld_ensemble_test(
    agent_codes: List[str],
    data_path: str,
    num_blocks: int,
    num_walls: int,
    agent_id: int,
    weights: List[float],
    top_k: int = 0,
    num_datapoints: int = 20,
    num_steps: int = 20,
    verbose: bool = False,
    n_seeds: int = 1,
) -> Dict[str, float]:
    """Evaluate an ensemble of gridworld programs on future steps (ROTE bootstrap–aligned).
    
    Hypothesis selection: use first n_hyp = len(agent_codes) (order preserved by fitness).
    If top_k > 0 and top_k < n_hyp: keep top_k programs by weight, renormalize.
    Aggregation: pi[action] += weight for each program's predicted action (weighted one-hot).
    Accuracy: tie-aware — if pi[gt] == max(pi), add 1/num_max where num_max = count of actions at max.
    Uses teacher-forced states: obs = dataset_states[t+1] for each future step.
    """
    _load_gridworld_stack()
    num_actions = 6
    if len(weights) != len(agent_codes):
        raise ValueError("weights must have same length as agent_codes")
    framework = AgentExecutionFramework()
    agents = []
    for code in agent_codes:
        try:
            agent = framework.compile_agent(code, num_agents=1, num_blocks=num_blocks)
            agents.append(agent)
        except Exception as e:
            if verbose:
                print(f"  Ensemble member compile error: {e}")
            agents.append(None)
    if not agents or all(a is None for a in agents):
        return {"accuracy": 0.0, "total": 0, "correct": 0, "errors": 1}

    # top_k: within prefix (first n_hyp), keep top_k by weight and renormalize
    curr_weights = list(weights)
    curr_agents = list(agents)
    n_hyp = len(curr_agents)
    if top_k > 0 and top_k < n_hyp:
        idx_by_weight = np.argsort(curr_weights)[::-1]
        keep_idx = idx_by_weight[:top_k]
        curr_agents = [curr_agents[i] for i in keep_idx]
        w = np.array([curr_weights[i] for i in keep_idx], dtype=np.float64)
        w = w / w.sum()
        curr_weights = w.tolist()

    class DummyArgs:
        def __init__(self):
            self.data_path = data_path
            self.num_agents = 1
            self.num_datapoints_per_agent = num_datapoints
            self.num_steps = num_steps
            self.group = False
            self.flip_quarter = True
            self.env_size = 10
            self.as_images = False

    dummy_args = DummyArgs()
    accuracies = []
    total_steps = 0
    correct_steps = 0

    def to_numpy(x):
        if isinstance(x, (jnp.ndarray, jax.Array)):
            return np.array(x)
        return x

    for seed in range(n_seeds):
        try:
            dataloader = make_dataloader(
                dummy_args,
                num_agents_to_sample=1,
                num_datapoints_per_agent_to_sample=num_datapoints,
                training=False,
                epoch=0,
                num_blocks=num_blocks,
                num_walls=num_walls,
                agent_indices=[agent_id],
            )
            datapoint = next(dataloader)
            seed_correct = 0
            seed_total = 0

            for dp_idx in range(num_datapoints):
                try:
                    data_sample = jax.tree.map(lambda x: x[0, dp_idx, :20 + num_steps], datapoint)
                    initial_states_traj = jax.tree.map(lambda x: x[:20], data_sample['states'])
                    gt_future_actions = data_sample['actions'][19:]
                    if gt_future_actions.shape[0] < num_steps:
                        actual_num_steps = min(gt_future_actions.shape[0], num_steps)
                    else:
                        actual_num_steps = num_steps

                    for step_idx in range(actual_num_steps):
                        if step_idx >= gt_future_actions.shape[0]:
                            break
                        try:
                            current_obs_raw = jax.tree.map(lambda x: x[19 + step_idx + 1], data_sample['states'])
                            current_obs = jax.tree.map(to_numpy, current_obs_raw)
                            current_obs['agent_id'] = 0

                            gt_action_this_step = gt_future_actions[step_idx]
                            if hasattr(gt_action_this_step, '__len__') and len(gt_action_this_step) > 0:
                                gt_action = int(gt_action_this_step[0])
                            else:
                                gt_action = int(gt_action_this_step)

                            # ROTE-aligned: weighted one-hot ensemble distribution pi
                            pi = np.zeros(num_actions, dtype=np.float64)
                            for agent, weight in zip(curr_agents, curr_weights):
                                if agent is None:
                                    continue
                                try:
                                    pred = framework.execute_agent(agent, current_obs)
                                    a = _normalize_gridworld_action(pred)
                                    if 0 <= a < num_actions:
                                        pi[a] += weight
                                except Exception:
                                    pass
                            # Tie-aware accuracy (ROTE): if gt is among max, add 1/num_max
                            max_prob = float(np.max(pi))
                            tol = 1e-9
                            num_max = int(np.sum(np.abs(pi - max_prob) < tol))
                            if num_max > 0 and gt_action < num_actions and abs(pi[gt_action] - max_prob) < tol:
                                seed_correct += 1.0 / num_max
                            seed_total += 1
                        except Exception as e:
                            seed_total += 1
                            if verbose:
                                print(f"  Step error dp={dp_idx} step={step_idx}: {e}")
                except Exception as e:
                    if verbose:
                        print(f"  Error processing datapoint {dp_idx}: {e}")
                    seed_total += num_steps
                    continue
        except Exception as e:
            if verbose:
                print(f"  Data loading error: {e}")
            seed_correct = 0
            seed_total = 1

        acc = seed_correct / seed_total if seed_total > 0 else 0.0
        accuracies.append(acc)
        total_steps = seed_total
        correct_steps = seed_correct

    avg_acc = np.mean(accuracies) if accuracies else 0.0
    correct = avg_acc * total_steps if total_steps > 0 else 0.0  # fractional due to tie-aware scoring
    return {"accuracy": avg_acc, "total": total_steps, "correct": correct, "errors": 0}


# ROTE Gridworld code setting: prefix length and future steps (match plot_and_eval.py)
# Prefix length must always be 20 for Gridworld; do not allow it to vary.
GRIDWORLD_PREFIX_LEN = 20
GRIDWORLD_NUM_FUTURE_STEPS = 20


def _gridworld_state_to_text_single(state: Dict[str, Any]) -> str:
    """Convert a single timestep state dict to ROTE-style text (match gridROTE.convert_state_to_text)."""
    def to_list(x):
        if isinstance(x, (jnp.ndarray, np.ndarray)):
            return np.array(x).tolist()
        return x
    text = ""
    text += f"The agents' inventory is {to_list(state.get('agent_inventory', []))}.\n"
    text += f"The agents' inventory colors are {to_list(state.get('agent_inventory_colors', []))}.\n"
    text += f"The agents' location is {to_list(state.get('agent_locations', []))}.\n"
    text += f"The block colors are {to_list(state.get('block_colors', []))}.\n"
    text += f"The block locations are {to_list(state.get('block_locations', []))}.\n"
    text += f"The wall locations are {to_list(state.get('wall_locations', []))}.\n"
    return text.strip()


def gridworld_prefix_to_text(prefix_states: Dict[str, Any], prefix_actions: Any) -> str:
    """Format exactly the first 20 (state, action) steps as ROTE-style text for prompting.
    Prefix length is always 20 for Gridworld. Deterministic, step-indexed; includes key state
    fields and action name mapping. Injected into initial candidate generation and all evolution prompts.
    """
    _load_gridworld_stack()
    action_names = ["stay", "right", "left", "down", "up", "interact"]
    prefix_len = GRIDWORLD_PREFIX_LEN  # Always 20; do not vary
    lines = []
    for t in range(prefix_len):
        state_t = jax.tree.map(lambda x: x[t] if hasattr(x, '__getitem__') and hasattr(x, 'shape') and len(x.shape) > 0 else x, prefix_states)
        state_t = jax.tree.map(lambda x: np.array(x).tolist() if isinstance(x, (jnp.ndarray, np.ndarray)) else x, state_t)
        text = _gridworld_state_to_text_single(state_t)
        act = prefix_actions[t]
        if hasattr(act, '__len__') and len(act) > 0:
            act = int(act[0])
        else:
            act = int(act)
        action_str = action_names[act] if 0 <= act < 6 else str(act)
        lines.append(f"Step {t+1}. State: {text}. Action: {action_str}")
    return "\n-------\n".join(lines)


def evaluate_gridworld_program_on_prefix(
    agent_code: str,
    prefix_states: Dict[str, Any],
    prefix_actions: Any,
    num_blocks: int,
) -> Dict[str, Any]:
    """Evaluate one program on a single episode's prefix (exactly first 20 steps).
    Returns accuracy and mismatch summary. Used for fitness only; LLM sees only this (train_acc).
    test_acc is never included in prompts or selection.
    """
    _load_gridworld_stack()
    framework = AgentExecutionFramework()
    try:
        agent = framework.compile_agent(agent_code, num_agents=1, num_blocks=num_blocks)
    except Exception:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "mismatch_summary": [], "errors": 1}

    def to_numpy(x):
        if isinstance(x, (jnp.ndarray, jax.Array)):
            return np.array(x)
        return x

    prefix_len = GRIDWORLD_PREFIX_LEN  # Always 20
    correct = 0
    total = 0
    mismatch_summary = []
    # Predict steps 0..prefix_len-2 (19 steps); compare to GT at each step
    for timestep in range(prefix_len - 1):
        try:
            state = jax.tree.map(lambda x: x[timestep] if hasattr(x, '__getitem__') else x, prefix_states)
            state = jax.tree.map(to_numpy, state)
            if isinstance(state, dict) and 'agent_locations' in state and hasattr(state['agent_locations'], 'shape'):
                if state['agent_locations'].ndim >= 1 and state['agent_locations'].shape[0] == 1:
                    state = dict(state)
                    state['agent_id'] = 0
            gt_action = int(prefix_actions[timestep][0]) if hasattr(prefix_actions[timestep], '__len__') else int(prefix_actions[timestep])
            predicted_action = framework.execute_agent(agent, state)
            pred_idx = _normalize_gridworld_action(predicted_action)
            if pred_idx == gt_action:
                correct += 1
            else:
                mismatch_summary.append({"step": timestep + 1, "pred": pred_idx, "gt": gt_action})
            total += 1
        except Exception:
            total += 1
    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc, "correct": correct, "total": total, "mismatch_summary": mismatch_summary, "errors": 0}


def evaluate_gridworld_ensemble_on_future(
    agent_codes: List[str],
    weights: List[float],
    future_states: Dict[str, Any],
    future_actions: Any,
    num_blocks: int,
    num_walls: int,
    num_future_steps: int = GRIDWORLD_NUM_FUTURE_STEPS,
) -> Dict[str, float]:
    """Evaluate ensemble on future steps with teacher-forced GT states (ROTE plot_and_eval multi-step).
    Freeze weights; no reweighting during steps 21..T.
    """
    _load_gridworld_stack()
    num_actions = 6
    if len(weights) != len(agent_codes):
        raise ValueError("weights must have same length as agent_codes")
    framework = AgentExecutionFramework()
    agents = []
    for code in agent_codes:
        try:
            agent = framework.compile_agent(code, num_agents=1, num_blocks=num_blocks)
            agents.append(agent)
        except Exception:
            agents.append(None)
    if not agents or all(a is None for a in agents):
        return {"accuracy": 0.0, "total": 0, "correct": 0.0}

    def to_numpy(x):
        if isinstance(x, (jnp.ndarray, jax.Array)):
            return np.array(x)
        return x

    n_steps = min(num_future_steps, future_actions.shape[0] if hasattr(future_actions, 'shape') else len(future_actions))
    seed_correct = 0.0
    seed_total = 0
    for step_idx in range(n_steps):
        try:
            current_obs = jax.tree.map(lambda x: x[step_idx] if hasattr(x, '__getitem__') else x, future_states)
            current_obs = jax.tree.map(to_numpy, current_obs)
            if isinstance(current_obs, dict):
                current_obs = dict(current_obs)
                current_obs['agent_id'] = 0
            gt_action = int(future_actions[step_idx][0]) if hasattr(future_actions[step_idx], '__len__') else int(future_actions[step_idx])
            pi = np.zeros(num_actions, dtype=np.float64)
            for agent, weight in zip(agents, weights):
                if agent is None:
                    continue
                try:
                    pred = framework.execute_agent(agent, current_obs)
                    a = _normalize_gridworld_action(pred)
                    if 0 <= a < num_actions:
                        pi[a] += weight
                except Exception:
                    pass
            max_prob = float(np.max(pi))
            tol = 1e-9
            num_max = int(np.sum(np.abs(pi - max_prob) < tol))
            if num_max > 0 and gt_action < num_actions and abs(pi[gt_action] - max_prob) < tol:
                seed_correct += 1.0 / num_max
            seed_total += 1
        except Exception:
            seed_total += 1
    acc = seed_correct / seed_total if seed_total > 0 else 0.0
    return {"accuracy": acc, "total": seed_total, "correct": seed_correct}


def _finalize_gridworld_llm_prompt(
    prompt_text: str,
    *,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    phase: str = "gridworld",
    diagnostics_dir: Optional[Path] = None,
    participant_id: Optional[int] = None,
    iteration: Optional[int] = None,
    candidate_index: Optional[int] = None,
) -> str:
    """Compress whitespace and block over-budget gridworld prompts from reaching the LLM."""
    prompt_text = _compress_prompt_whitespace(prompt_text)
    tokens = estimate_tokens(prompt_text, estimator=prompt_token_estimator)
    diag: Dict[str, Any] = {
        "participant_id": participant_id,
        "phase": phase,
        "iteration": iteration,
        "candidate_index": candidate_index,
        "prompt_tokens_before_truncation": tokens,
        "prompt_tokens_after_truncation": tokens,
        "hard_prompt_token_cap": hard_prompt_token_cap,
        "truncated": False,
        "truncation_steps": ["compress_instruction_whitespace"],
    }
    if tokens <= hard_prompt_token_cap:
        diag["status"] = "ok"
        _append_prompt_diagnostic(diag, diagnostics_dir)
        return prompt_text
    overflow = {"full_prompt": tokens}
    return _enforce_prompt_budget(
        prompt_text,
        hard_prompt_token_cap=hard_prompt_token_cap,
        strict_prompt_budget=strict_prompt_budget,
        prompt_token_estimator=prompt_token_estimator,
        overflow_components=overflow,
        truncation_steps=["gridworld_whitespace_only"],
        phase=phase,
        participant_id=participant_id,
        iteration=iteration,
        candidate_index=candidate_index,
        diagnostics_dir=diagnostics_dir,
        diagnostics_base=diag,
    )[0]


def generate_gridworld_initial_candidates(
    client: OpenAI,
    model_name: str,
    template_code: str,
    prefix_text: str,
    n_candidates: int,
    max_tokens: int = 2000,
    max_workers: int = 5,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    prompt_diagnostics_dir: Optional[Path] = None,
    phase: str = "gridworld_initial",
    participant_id: Optional[int] = None,
    iteration: Optional[int] = None,
) -> List[str]:
    """Generate K initial candidate programs for one episode. Prompt injects episode prefix (ROTE-style).
    Used at episode start; no parent code, only environment description + prefix observations + template.
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    prompt_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "gridworld", "non_strict", "infer_single_fsm.txt")
    code_template_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "gridworld", "non_strict", "single_code_template.txt")
    try:
        base_prompt = open(prompt_path).read()
        code_template = load_single_code_template(code_template_path)
    except FileNotFoundError:
        base_prompt = "You are a robot viewing agents acting in an object-centric environment. Model the agent's behavior as FSM code. Experiences (state, action):"
        code_template = ""
    full_prompt = f"""{base_prompt}

Observed trajectory (first 20 steps) for this episode:
{prefix_text}
{single_code_template_prompt_suffix(code_template)}
Output ONLY runnable Python code (no explanations, no markdown fences, no preamble). Generate the variant now:"""

    _gw_call_idx = [0]

    def _generate_one(_cand_idx: int = 0) -> str:
        cand_idx = _gw_call_idx[0]
        _gw_call_idx[0] += 1
        try:
            prompt = _finalize_gridworld_llm_prompt(
                full_prompt,
                hard_prompt_token_cap=hard_prompt_token_cap,
                strict_prompt_budget=strict_prompt_budget,
                prompt_token_estimator=prompt_token_estimator,
                phase=phase,
                diagnostics_dir=prompt_diagnostics_dir,
                participant_id=participant_id,
                iteration=iteration,
                candidate_index=cand_idx,
            )
            if not prompt:
                return template_code
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.95,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            code = _sanitize_llm_python_candidate(
                content, required_markers=("class FSMAgent", "def act(")
            )
            if code and ("class FSMAgent" in code or "def act" in code):
                return code
        except PromptBudgetExceededError:
            raise
        except Exception as e:
            print(f"Warning: Failed to generate gridworld initial candidate: {e}")
        return template_code

    return _parallel_generate_children(
        n_candidates,
        _generate_one,
        max_workers=max_workers,
        desc="Generating gridworld initial candidates",
    )


def generate_gridworld_evolution_variants(
    client: OpenAI,
    model_name: str,
    parent_codes: List[str],
    parent_train_accuracies: List[float],
    parent_prefix_correct_counts: List[int],
    prefix_mismatch_summary: List[Dict[str, Any]],
    prefix_text: str,
    n_variants: int = 10,
    max_tokens: int = 2000,
    max_workers: int = 5,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    prompt_diagnostics_dir: Optional[Path] = None,
    phase: str = "gridworld_evolution",
    participant_id: Optional[int] = None,
    iteration: Optional[int] = None,
) -> List[str]:
    """Generate evolution variants. Prompt MUST include: serialized prefix trajectory, parent code, prefix accuracy (X/20), optional mismatch summary.
    test_acc is NEVER included.
    """
    # Build prompt with prefix trajectory first (required in ALL evolution prompts)
    obs_section = f"""Observed trajectory (first 20 steps):
{prefix_text}

"""
    parent_section = ""
    for i, (code, acc, correct_count) in enumerate(zip(parent_codes, parent_train_accuracies, parent_prefix_correct_counts)):
        parent_section += f"""Current program (parent {i+1}):
```python
{code}
```

Prefix accuracy: {correct_count} / {GRIDWORLD_PREFIX_LEN}

"""
    mismatch_str = "None"
    if prefix_mismatch_summary:
        lines = [f"Step {m['step']}: predicted {m['pred']}, ground truth {m['gt']}" for m in prefix_mismatch_summary[:15]]
        mismatch_str = "\n".join(lines)
    mismatch_section = f"Mismatches (pred vs gt):\n{mismatch_str}\n\n"
    full_prompt = f"""Improve the following agent program. Use only prefix (first 20 steps) performance; do not use any future-step metrics.

{obs_section}{parent_section}{mismatch_section}Generate an improved program variant. Output ONLY runnable Python code (no explanations, no markdown fences, no preamble). Actions: 0=stay, 1=right, 2=left, 3=down, 4=up, 5=interact. Generate now:"""
    fallback = parent_codes[0] if parent_codes else ""
    _gw_call_idx = [0]

    def _generate_one(_cand_idx: int = 0) -> str:
        cand_idx = _gw_call_idx[0]
        _gw_call_idx[0] += 1
        try:
            prompt = _finalize_gridworld_llm_prompt(
                full_prompt,
                hard_prompt_token_cap=hard_prompt_token_cap,
                strict_prompt_budget=strict_prompt_budget,
                prompt_token_estimator=prompt_token_estimator,
                phase=phase,
                diagnostics_dir=prompt_diagnostics_dir,
                participant_id=participant_id,
                iteration=iteration,
                candidate_index=cand_idx,
            )
            if not prompt:
                return fallback
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.95,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            code = _sanitize_llm_python_candidate(
                content, required_markers=("class FSMAgent", "def act(")
            )
            if code and ("class FSMAgent" in code or "def act" in code):
                return code
        except PromptBudgetExceededError:
            raise
        except Exception as e:
            print(f"Warning: Failed to generate gridworld evolution variant: {e}")
        return fallback

    return _parallel_generate_children(
        n_variants,
        _generate_one,
        max_workers=max_workers,
        desc="Generating gridworld evolution variants",
    )


def _make_gridworld_dataloader_args(data_path: str, num_blocks: int, num_walls: int, num_steps: int = 100):
    """Build args object for plot_and_eval.make_dataloader (Gridworld test split)."""
    class Args:
        pass
    args = Args()
    args.data_path = data_path
    args.num_agents = 1
    args.num_datapoints_per_agent = 100
    args.num_steps = num_steps
    args.group = False
    args.flip_quarter = True
    args.env_size = 10
    args.as_images = False
    return args


def get_one_gridworld_episode_from_test(
    data_path: str,
    num_blocks: int,
    num_walls: int,
    agent_id: int,
    episode_idx: int,
    num_steps: int = 100,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any], Any, Dict[str, Any]]:
    """Sample one trajectory from Gridworld TEST split (last 20% datapoints), same as ROTE plot_and_eval.
    Returns (prefix_states, prefix_actions, future_states, future_actions, meta).
    """
    _load_gridworld_stack()
    args = _make_gridworld_dataloader_args(data_path, num_blocks, num_walls, num_steps)
    dataloader = make_dataloader(
        args,
        num_agents_to_sample=1,
        num_datapoints_per_agent_to_sample=1,
        training=False,
        epoch=episode_idx,
        num_blocks=num_blocks,
        num_walls=num_walls,
        agent_indices=[agent_id],
    )
    datapoint = next(dataloader)
    # datapoint['states']: (1, 1, num_steps, ...), 'actions': (1, 1, num_steps, 1)
    data_sample = jax.tree.map(lambda x: x[0, 0, :] if hasattr(x, 'shape') and len(x.shape) >= 3 else x, datapoint)
    # Prefix = exactly first 20 steps (GRIDWORLD_PREFIX_LEN); do not vary
    prefix_states = jax.tree.map(lambda x: x[:GRIDWORLD_PREFIX_LEN] if hasattr(x, '__getitem__') else x, data_sample['states'])
    prefix_actions = data_sample['actions'][:GRIDWORLD_PREFIX_LEN]
    future_len = min(GRIDWORLD_NUM_FUTURE_STEPS, data_sample['actions'].shape[0] - GRIDWORLD_PREFIX_LEN)
    future_states = jax.tree.map(
        lambda x: x[GRIDWORLD_PREFIX_LEN:GRIDWORLD_PREFIX_LEN + future_len] if hasattr(x, '__getitem__') else x,
        data_sample['states'],
    )
    future_actions = data_sample['actions'][GRIDWORLD_PREFIX_LEN:GRIDWORLD_PREFIX_LEN + future_len]
    meta = {
        "num_blocks": num_blocks,
        "num_walls": num_walls,
        "agent_id": agent_id,
        "episode_idx": episode_idx,
        "prefix_len": GRIDWORLD_PREFIX_LEN,
        "num_future_steps": future_len,
    }
    return prefix_states, prefix_actions, future_states, future_actions, meta


def run_evolution_gridworld_rote_episodes(
    seed_program_path: str,
    data_path: str,
    num_blocks: int,
    num_walls: int,
    agent_id: int,
    num_episodes: int,
    K: int,
    N: int,
    n_candidates_per_iteration: int,
    model_name: str,
    client: OpenAI,
    output_dir: str,
    wandb: Optional[Any] = None,
    max_workers: int = 5,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    ROTE-aligned Gridworld: episode loop. For each episode:
    1) Sample one trajectory from test split (make_dataloader training=False, same as ROTE plot_and_eval).
    2) Prefix = exactly first 20 steps; generate K candidates conditioned on this episode's prefix (candidate generation inside episode loop).
    3) Evolve each candidate for N iters; fitness = prefix accuracy only (train_acc); test_acc is never in prompts or parent selection.
    4) Ensemble weights = softmax(prefix_score_i) where prefix_score_i = number of correct predicted actions on first 20 steps; freeze weights; evaluate on future steps (teacher-forced).
    5) Append episode row to episodes_summary.csv.
    Returns (list of episode result dicts, mean episode_test_acc).
    """
    seed_code = load_seed_program(seed_program_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    episode_results = []

    for episode_idx in tqdm(range(num_episodes), desc="Episodes"):
        prefix_states, prefix_actions, future_states, future_actions, meta = get_one_gridworld_episode_from_test(
            data_path, num_blocks, num_walls, agent_id, episode_idx,
        )
        prefix_text = gridworld_prefix_to_text(prefix_states, prefix_actions)
        episode_dir = output_path / f"episode_{episode_idx}"
        episode_dir.mkdir(exist_ok=True)
        with open(episode_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Generate K initial candidates conditioned on this episode's prefix (inside episode loop; not global)
        initial_candidates = generate_gridworld_initial_candidates(
            client, model_name, seed_code, prefix_text, n_candidates=K, max_workers=max_workers,
        )
        final_programs = []
        # Ensemble weights use raw prefix correct counts only (integers 0..19). Do NOT use train_acc or any normalized metric.
        prefix_scores = []  # prefix_score_i = correct_prefix_predictions (integer)

        for cand_idx in range(K):
            cand_dir = episode_dir / f"candidate_{cand_idx}"
            cand_dir.mkdir(exist_ok=True)
            current_code = initial_candidates[cand_idx] if cand_idx < len(initial_candidates) else seed_code
            iter_dir_0 = cand_dir / "iteration_0"
            iter_dir_0.mkdir(exist_ok=True)
            (iter_dir_0 / "candidates").mkdir(exist_ok=True)
            (iter_dir_0 / "candidates" / "candidate_0.py").write_text(current_code)
            eval_0 = evaluate_gridworld_program_on_prefix(current_code, prefix_states, prefix_actions, num_blocks)
            # metrics.json: train_acc only; test_acc is never included (not for LLM or selection)
            with open(iter_dir_0 / "metrics.json", "w") as f:
                json.dump({"train_acc": eval_0["accuracy"]}, f, indent=2)

            for iteration in range(1, N + 1):
                iter_dir = cand_dir / f"iteration_{iteration}"
                iter_dir.mkdir(exist_ok=True)
                (iter_dir / "parents").mkdir(exist_ok=True)
                (iter_dir / "candidates").mkdir(exist_ok=True)
                # Parent naming to avoid collision: parent_{iteration}.py
                (iter_dir / "parents" / f"parent_{iteration}.py").write_text(current_code)
                parent_eval = evaluate_gridworld_program_on_prefix(current_code, prefix_states, prefix_actions, num_blocks)
                parent_train_acc = parent_eval["accuracy"]
                parent_correct_count = parent_eval["correct"]  # raw count for "Prefix accuracy: X / 20"
                parent_mismatch = parent_eval.get("mismatch_summary", [])

                variants = generate_gridworld_evolution_variants(
                    client, model_name,
                    parent_codes=[current_code],
                    parent_train_accuracies=[parent_train_acc],
                    parent_prefix_correct_counts=[parent_correct_count],
                    prefix_mismatch_summary=parent_mismatch,
                    prefix_text=prefix_text,
                    n_variants=n_candidates_per_iteration,
                    max_workers=max_workers,
                )
                best_acc = parent_train_acc
                best_code = current_code
                for m, code in enumerate(variants):
                    (iter_dir / "candidates" / f"candidate_{m}.py").write_text(code)
                    ev = evaluate_gridworld_program_on_prefix(code, prefix_states, prefix_actions, num_blocks)
                    if ev["accuracy"] > best_acc:
                        best_acc = ev["accuracy"]
                        best_code = code
                current_code = best_code
                # Parent selection uses train_acc only; test_acc never in metrics or LLM
                with open(iter_dir / "metrics.json", "w") as f:
                    json.dump({"train_acc": best_acc}, f, indent=2)

                if wandb is not None:
                    wandb.log({f"episode_{episode_idx}_cand_{cand_idx}_train_acc": best_acc, f"episode_{episode_idx}_iteration": iteration}, step=episode_idx * N * K + cand_idx * N + iteration)

            final_dir = cand_dir / "final"
            final_dir.mkdir(exist_ok=True)
            (final_dir / "evolved_program.py").write_text(current_code)
            final_prefix_eval = evaluate_gridworld_program_on_prefix(current_code, prefix_states, prefix_actions, num_blocks)
            correct_prefix_predictions = final_prefix_eval["correct"]  # raw count (integer); used for ensemble weights only
            prefix_scores.append(correct_prefix_predictions)
            final_programs.append(current_code)
            with open(final_dir / "final_metrics.json", "w") as f:
                json.dump({
                    "train_acc": final_prefix_eval["accuracy"],
                    "test_acc": None,
                    "prefix_score": correct_prefix_predictions,
                    "ensemble_weight": None,
                }, f, indent=2)

        # Weights = softmax(prefix_score_i). prefix_score_i = raw correct_prefix_predictions (integer). Do NOT use train_acc.
        score = np.array(prefix_scores, dtype=np.float64)
        weights = np.exp(score - score.max())
        weights = weights / weights.sum()
        weights = weights.tolist()

        for cand_idx, w in enumerate(weights):
            final_metrics_path = episode_dir / f"candidate_{cand_idx}" / "final" / "final_metrics.json"
            with open(final_metrics_path, "r") as f:
                fm = json.load(f)
            fm["ensemble_weight"] = w
            with open(final_metrics_path, "w") as f:
                json.dump(fm, f, indent=2)

        ensemble_eval = evaluate_gridworld_ensemble_on_future(
            final_programs, weights, future_states, future_actions, num_blocks, num_walls,
        )
        episode_train_acc = np.mean([evaluate_gridworld_program_on_prefix(c, prefix_states, prefix_actions, num_blocks)["accuracy"] for c in final_programs])
        episode_test_acc = ensemble_eval["accuracy"]

        ensemble_dir = episode_dir / "ensemble"
        ensemble_dir.mkdir(exist_ok=True)
        with open(ensemble_dir / "weights.json", "w") as f:
            json.dump({"weights": weights}, f, indent=2)
        with open(ensemble_dir / "ensemble_metrics.json", "w") as f:
            json.dump({
                "episode_train_acc": episode_train_acc,
                "episode_test_acc": episode_test_acc,
                "ensemble_test_acc": episode_test_acc,
            }, f, indent=2)

        row = {
            "episode_id": episode_idx,
            "agent_id": agent_id,
            "num_blocks": num_blocks,
            "num_walls": num_walls,
            "K": K,
            "N": N,
            "episode_train_acc": episode_train_acc,
            "episode_test_acc": episode_test_acc,
            "ensemble_test_acc": episode_test_acc,
        }
        summary_rows.append(row)
        episode_results.append(row)

        if wandb is not None:
            wandb.log({
                f"episode_{episode_idx}_train_acc": episode_train_acc,
                f"episode_{episode_idx}_test_acc": episode_test_acc,
                f"episode_{episode_idx}_best_train_acc": max(prefix_scores) / max(1, (GRIDWORLD_PREFIX_LEN - 1)),
            }, step=episode_idx)

    with open(output_path / "episodes_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "agent_id", "num_blocks", "num_walls", "K", "N", "episode_train_acc", "episode_test_acc", "ensemble_test_acc"])
        writer.writeheader()
        writer.writerows(summary_rows)

    mean_test_acc = float(np.mean([r["episode_test_acc"] for r in episode_results])) if episode_results else 0.0
    print(f"\nROTE Gridworld: mean episode_test_acc (ensemble) = {mean_test_acc:.4f} over {num_episodes} episodes")
    return episode_results, mean_test_acc


def generate_gridworld_program_variants(
    client: OpenAI,
    model_name: str,
    template_code: str,
    parent_codes: List[str],
    n_variants: int = 10,
    max_tokens: int = 2000,
    parent_train_accuracies: Optional[List[float]] = None,
    max_workers: int = 5,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    prompt_diagnostics_dir: Optional[Path] = None,
    phase: str = "gridworld_variants",
    participant_id: Optional[int] = None,
    iteration: Optional[int] = None,
) -> List[str]:
    """
    Generate full program code variants for gridworld (non-strict mode).
    The LLM modifies the entire program code, not just parameters.
    
    Args:
        template_code: Original template code
        parent_codes: List of parent program codes (elite programs from previous iterations)
        n_variants: Number of variants to generate
        max_tokens: Maximum tokens for generation
        parent_train_accuracies: List of training accuracies for each parent (for guidance)
    
    Returns a list of program code strings.
    """
    # Load prompts from file
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    prompt_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "gridworld", "non_strict", "infer_single_fsm.txt")
    code_template_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "gridworld", "non_strict", "single_code_template.txt")
    
    try:
        base_prompt_template = open(prompt_path).read()
        code_template = load_single_code_template(code_template_path)
    except FileNotFoundError as e:
        print(f"Warning: Could not load prompt files: {e}")
        print("Falling back to hardcoded prompts.")
        # Fallback to hardcoded prompt
        base_prompt_template = """You are a robot viewing agents acting in an object-centric environment. Your goal is to model the behavior of the agents as a finite state machine (FSM) code in python. You will be provided experiences in the format of (state, action) tuples.

This environment simulates potentially multiple agents interacting in a grid world filled with colored blocks and walls. The world is a square grid (default 7x7) with walls on the perimeter. There are also walls scattered across the interior 6x6 region. Multiple agents, each represented by a distinct color (red, blue, green, etc.), navigate this space alongside colored blocks that can be picked up and transported.

Agents can perform six basic actions: staying in place, moving in any of the four cardinal directions (up, down, left, right), or interacting with blocks. If agent is on a grid cell that a colored block is on and they don't have an item in their inventory, they have to press the 'interact' action to add that block to their inventory. If they press the interact button but have an item in their inventory, they stay in place, but remove the item they had from their inventory.  Importantly, agents can't occupy the same space or swap positions, and they're limited to carrying one block at a time. If they both try to move into the same cell, they will both stay in place. If you don't have an item in your inventory, this is represented by your inventory being equal to -1. If you are holding a block and try walking onto a cell where another block is, you will remain in the same place with the same block in your inventory (equivalent of a stay action).

Each agent receives detailed information about the environment's state, including the positions of all walls, agents, and blocks, as well as information about what blocks are being carried by which agents.

You need to implement the python code to model the logic of the agent's behavior, as seen in the provided experiences. Please follow the template to implement the code. The code needs to be directly runnable on the state and return the action in python as provided in the experiences. Try to keep your code as concise as possible.

You need to implement python code to model the logic of the world as seen in the following experiences:"""
        code_template = ""
    
    # Format multiple parent programs
    num_parents = len(parent_codes)
    parent_programs_text = ""
    if num_parents == 1:
        parent_programs_text = f"Current parent program:\n```python\n{parent_codes[0]}\n```"
    else:
        parent_programs_text = f"Reference parent programs ({num_parents} elite programs):\n"
        for i, (parent_code, acc) in enumerate(zip(parent_codes, parent_train_accuracies or [None] * num_parents)):
            acc_str = f" (train_acc: {acc:.4f})" if acc is not None else ""
            parent_programs_text += f"\nParent {i+1}{acc_str}:\n```python\n{parent_code}\n```\n"
    
    performance_info = ""
    if parent_train_accuracies:
        avg_acc = sum(parent_train_accuracies) / len(parent_train_accuracies)
        max_acc = max(parent_train_accuracies)
        performance_info = f"\nParent performance: Average train accuracy = {avg_acc:.4f}, Best = {max_acc:.4f}\n"
        if avg_acc < 0.5:
            performance_info += "NOTE: Performance is LOW. Consider significant changes to the program logic.\n"
        elif avg_acc > 0.8:
            performance_info += "NOTE: Performance is HIGH. Make refined improvements.\n"
        else:
            performance_info += "NOTE: Performance is MODERATE. Explore different approaches.\n"
        if num_parents > 1:
            performance_info += f"NOTE: You have {num_parents} parent programs to learn from. Combine the best ideas from each.\n"
    
    base_prompt_template_final = f"""{base_prompt_template}

{parent_programs_text}

{performance_info}

Your task: Generate an improved program variant. The variant should:
- Maintain the same class structure (FSMAgent with __init__ and act methods)
- Improve the decision-making logic
- Handle edge cases better
- Be more efficient or accurate
{single_code_template_prompt_suffix(code_template)}
Output format: Provide ONLY runnable Python code (no explanations, no markdown fences, no preamble).
The variant must be a complete, runnable program.

Generate the variant now:"""

    best_parent = parent_codes[0] if parent_codes else ""
    _gw_call_idx = [0]

    def _generate_one(_cand_idx: int = 0) -> str:
        cand_idx = _gw_call_idx[0]
        _gw_call_idx[0] += 1
        try:
            prompt = _finalize_gridworld_llm_prompt(
                base_prompt_template_final,
                hard_prompt_token_cap=hard_prompt_token_cap,
                strict_prompt_budget=strict_prompt_budget,
                prompt_token_estimator=prompt_token_estimator,
                phase=phase,
                diagnostics_dir=prompt_diagnostics_dir,
                participant_id=participant_id,
                iteration=iteration,
                candidate_index=cand_idx,
            )
            if not prompt:
                return best_parent
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.95,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            code = _sanitize_llm_python_candidate(
                content, required_markers=("class FSMAgent", "def act(")
            )
            if code and ("class FSMAgent" in code or "def act" in code):
                return code
        except PromptBudgetExceededError:
            raise
        except Exception as e:
            print(f"Warning: Failed to generate gridworld program variant: {e}")
        return best_parent

    return _parallel_generate_children(
        n_variants,
        _generate_one,
        max_workers=max_workers,
        desc="Generating gridworld variants",
    )[:n_variants]


_PARENT_TRUNCATION_MARKER = "# truncated; keep concise\n"


def _format_parent_loglik_prompt_metrics(
    *,
    train_loglik: float,
    val_loglik: Optional[float] = None,
    overall_loglik: Optional[float] = None,
) -> str:
    """Format parent log-likelihood metrics for LLM prompts."""
    if val_loglik is not None:
        parts = [
            f"train_loglik={train_loglik:.4f}",
            f"val_loglik={val_loglik:.4f}",
        ]
        if overall_loglik is not None:
            parts.append(f"overall_loglik={overall_loglik:.4f}")
        return ", ".join(parts)
    return f"log-likelihood={train_loglik:.4f}"


def _build_parent_context_for_prompt(
    *,
    prompt_parent_programs: List[str],
    num_parents: int,
    dataset: str,
    fitness_metric: str,
    parent_train_accuracies: Optional[List[float]],
    parent_train_mses: Optional[List[float]],
    parent_val_logliks: Optional[List[Optional[float]]],
    parent_overall_logliks: Optional[List[Optional[float]]] = None,
    cpc18_official_mse: bool,
) -> str:
    """Parent program section for LLM prompts (metrics + code blocks)."""
    if num_parents == 1:
        parent_context = (
            f"\n\nReference program (parent):\n```python\n{prompt_parent_programs[0]}\n```\n\n"
        )
        if (
            parent_train_accuracies
            and len(parent_train_accuracies) > 0
            and parent_train_accuracies[0] is not None
            and fitness_metric == "loglik"
            and is_binary_loglik_dataset(dataset)
        ):
            train_ll = parent_train_accuracies[0]
            val_ll = (
                parent_val_logliks[0]
                if parent_val_logliks and len(parent_val_logliks) > 0
                else None
            )
            overall_ll = (
                parent_overall_logliks[0]
                if parent_overall_logliks and len(parent_overall_logliks) > 0
                else None
            )
            parent_context += (
                f"Parent performance: "
                f"{_format_parent_loglik_prompt_metrics(train_loglik=train_ll, val_loglik=val_ll, overall_loglik=overall_ll)}\n\n"
            )
        parent_context += (
            "Generate a variant that improves upon or explores alternatives to the parent program.\n"
        )
        return parent_context

    parent_context = f"\n\nReference parent programs ({num_parents} elite programs):\n"
    for i, parent_program in enumerate(prompt_parent_programs):
        if dataset == "cpc18" and cpc18_official_mse:
            mse = parent_train_mses[i] if (parent_train_mses and i < len(parent_train_mses)) else None
            mse_str = f" (train_block-MSE: {mse:.2f})" if mse is not None else ""
            parent_context += f"\nParent {i+1}{mse_str}:\n```python\n{parent_program}\n```\n"
        else:
            parent_metric = (
                parent_train_accuracies[i]
                if parent_train_accuracies and i < len(parent_train_accuracies)
                else None
            )
            if parent_metric is not None:
                if (
                    fitness_metric == "loglik"
                    and is_binary_loglik_dataset(dataset)
                    and parent_val_logliks
                    and i < len(parent_val_logliks)
                    and parent_val_logliks[i] is not None
                ):
                    overall_ll = (
                        parent_overall_logliks[i]
                        if parent_overall_logliks and i < len(parent_overall_logliks)
                        else None
                    )
                    metric_parts = [
                        f"train_loglik: {parent_metric:.4f}",
                        f"val_loglik: {parent_val_logliks[i]:.4f}",
                    ]
                    if overall_ll is not None:
                        metric_parts.append(f"overall_loglik: {overall_ll:.4f}")
                    metric_str = f" ({', '.join(metric_parts)})"
                elif fitness_metric == "loglik" and is_binary_loglik_dataset(dataset):
                    metric_str = f" (log-likelihood: {parent_metric:.4f})"
                else:
                    metric_str = f" (train_acc: {parent_metric:.4f})"
            else:
                metric_str = ""
            parent_context += f"\nParent {i+1}{metric_str}:\n```python\n{parent_program}\n```\n"

    if dataset == "cpc18" and cpc18_official_mse and parent_train_mses:
        avg_mse = sum(parent_train_mses) / len(parent_train_mses)
        min_mse = min(parent_train_mses)
        parent_context += "\nParent performance on training data:\n"
        parent_context += f"- Average train block-MSE: {avg_mse:.2f}\n"
        parent_context += f"- Best train block-MSE: {min_mse:.2f}\n"
        parent_context += "\nIMPORTANT for CPC18:\n"
        parent_context += "The official CPC18 metric is block-level MSE (lower is better).\n"
        parent_context += "Your goal is to reduce block-level MSE.\n"
        parent_context += f"Current best: train_block-MSE={min_mse:.2f}\n"
        if min_mse > 50:
            parent_context += "\nNOTE: Current MSE is HIGH (>50). Focus on reducing MSE significantly.\n"
        elif min_mse > 30:
            parent_context += "\nNOTE: Current MSE is MODERATE (30-50). Try to reduce MSE further.\n"
        else:
            parent_context += "\nNOTE: Current MSE is LOW (<30). Fine-tune to improve further.\n"

    parent_context += "\nGenerate a variant that combines the best ideas from these parent programs.\n"
    return parent_context


def _truncate_parent_program_for_prompt(code: str, max_parent_chars: int) -> Tuple[str, bool]:
    """Truncate parent code for LLM prompts only; evaluation uses full code."""
    if max_parent_chars <= 0 or len(code) <= max_parent_chars:
        return code, False
    marker = _PARENT_TRUNCATION_MARKER
    join_newline = "\n"
    budget = max_parent_chars - len(marker) - len(join_newline)
    if budget < 2:
        return code[:max_parent_chars], True
    first_len = int(budget * 0.7)
    last_len = budget - first_len
    if first_len + last_len >= len(code):
        return code[:max_parent_chars], True
    head = code[:first_len]
    tail = code[-last_len:] if last_len > 0 else ""
    return f"{head}{join_newline}{marker}{tail}", True


def _decayed_fresh_n_for_iteration(
    fresh_n_max: int,
    iter_idx: int,
    total_iters: int,
    n_candidates: int,
) -> int:
    """Decay fresh count from max toward 1 over a phase's iteration window (0-based iter_idx)."""
    fresh_n_max = int(fresh_n_max)
    if fresh_n_max <= 0:
        return 0
    total = max(1, int(total_iters))
    idx = max(0, int(iter_idx))
    raw = math.floor(float(fresh_n_max) * (1.0 - idx / total))
    fresh_n = max(1, int(raw))
    return min(fresh_n, int(n_candidates))


def _decayed_sampled_parents_k_for_iteration(
    num_parents: int,
    iter_idx: int,
    total_iters: int,
) -> int:
    """Decay sampled parent count from num_parents toward 0 (0-based iter_idx)."""
    num_parents = int(num_parents)
    if num_parents <= 0:
        return 0
    total = max(1, int(total_iters))
    idx = max(0, int(iter_idx))
    raw = math.floor(float(num_parents) * (1.0 - idx / total))
    return max(0, min(int(raw), num_parents))


def _select_parent_indices_from_elite_pool(
    pool_size: int,
    *,
    sample_size: int,
    sample_parents: bool,
    sampled_parents_decay: bool,
    iter_idx: int,
    total_iters: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], int, int]:
    """Return elite indices for parents, plus (best_k, sampled_k) counts."""
    num_parents = min(sample_size, pool_size)
    if num_parents <= 0:
        return [], 0, 0
    if not sample_parents:
        return list(range(num_parents)), num_parents, 0

    if sampled_parents_decay:
        sampled_k = _decayed_sampled_parents_k_for_iteration(
            num_parents, iter_idx, total_iters
        )
    else:
        sampled_k = num_parents
    best_k = num_parents - sampled_k

    indices: List[int] = list(range(best_k))
    if sampled_k > 0 and rng is not None:
        pool_start = best_k
        pool_len = pool_size - pool_start
        if pool_len > 0:
            k = min(sampled_k, pool_len)
            offset_idxs = rng.choice(pool_len, size=k, replace=False)
            indices.extend(int(pool_start + j) for j in offset_idxs)
    return indices, best_k, sampled_k


def _iteration_candidate_source_header(
    fresh_n_max: int,
    fresh_n: int,
    n_candidates: int,
    candidate_sources: List[str],
    *,
    iter_idx: int,
    total_iters: int,
) -> Dict[str, Any]:
    return {
        "fresh_n_candidates": int(fresh_n_max),
        "fresh_n": int(fresh_n),
        "iter_idx": int(iter_idx),
        "total_iterations": int(total_iters),
        "n_normal_candidates": int(n_candidates) - int(fresh_n),
        "candidate_sources": list(candidate_sources),
    }


def _annotate_candidate_results_with_sources(
    candidate_results: List[Dict[str, Any]],
    candidate_sources: List[str],
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for row, source in zip(candidate_results, candidate_sources):
        out = dict(row)
        out["source"] = source
        annotated.append(out)
    return annotated


def _resolve_fresh_parent_prompt_context(
    *,
    fresh_parent_code: Optional[str],
    fresh_parent_train_loglik: Optional[float],
    fresh_parent_val_loglik: Optional[float],
    elite_parents: List[Tuple[Any, ...]],
) -> Tuple[List[str], Optional[List[float]], Optional[List[Optional[float]]]]:
    """Parent list for fresh exploration: explicit seed/baseline, else baseline-like pool entry."""
    if fresh_parent_code:
        codes = [fresh_parent_code]
        train_accs = (
            [float(fresh_parent_train_loglik)]
            if fresh_parent_train_loglik is not None
            else None
        )
        val_lls = (
            [fresh_parent_val_loglik]
            if fresh_parent_val_loglik is not None
            else None
        )
        return codes, train_accs, val_lls
    for parent in elite_parents:
        prog_id = str(parent[3])
        if prog_id in ("baseline", "global_baseline", "refinement_seed"):
            train_acc = _train_loglik_from_elite_tuple(parent)
            return [parent[0]], [train_acc], None
    raise ValueError(
        "fresh_n_candidates > 0 requires fresh_parent_code or a baseline-like parent in the pool"
    )


def _generate_iteration_candidate_codes(
    *,
    client: OpenAI,
    model_name: str,
    fresh_n_candidates: int,
    n_candidates: int,
    fresh_parent_programs: List[str],
    normal_parent_programs: List[str],
    variant_kwargs: Dict[str, Any],
    fresh_parent_train_accuracies: Optional[List[float]] = None,
    fresh_parent_val_logliks: Optional[List[Optional[float]]] = None,
    fresh_parent_overall_logliks: Optional[List[Optional[float]]] = None,
    normal_parent_train_accuracies: Optional[List[float]] = None,
    normal_parent_val_logliks: Optional[List[Optional[float]]] = None,
    normal_parent_overall_logliks: Optional[List[Optional[float]]] = None,
    fresh_parent_program_ids: Optional[List[str]] = None,
    normal_parent_program_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Generate candidates: first fresh_n from seed/baseline only, rest from normal parents."""
    fresh_n = int(fresh_n_candidates)
    n_total = int(n_candidates)
    n_normal = n_total - fresh_n
    codes: List[str] = []
    sources: List[str] = []
    if fresh_n > 0:
        fresh_kw = dict(variant_kwargs)
        fresh_kw.update(
            client=client,
            model_name=model_name,
            parent_programs=list(fresh_parent_programs),
            n_variants=fresh_n,
        )
        if fresh_parent_train_accuracies is not None:
            fresh_kw["parent_train_accuracies"] = fresh_parent_train_accuracies
        if fresh_parent_val_logliks is not None:
            fresh_kw["parent_val_logliks"] = fresh_parent_val_logliks
        if fresh_parent_overall_logliks is not None:
            fresh_kw["parent_overall_logliks"] = fresh_parent_overall_logliks
        if fresh_parent_program_ids is not None:
            fresh_kw["parent_program_ids"] = list(fresh_parent_program_ids)
        fresh_codes = generate_program_variants(**fresh_kw)
        codes.extend(fresh_codes)
        sources.extend(["fresh"] * len(fresh_codes))
    if n_normal > 0:
        normal_kw = dict(variant_kwargs)
        seed_base = normal_kw.get("llm_decoding_seed_base")
        if seed_base is not None:
            normal_kw["llm_decoding_seed_base"] = int(seed_base) + int(fresh_n)
        normal_kw.update(
            client=client,
            model_name=model_name,
            parent_programs=list(normal_parent_programs),
            n_variants=n_normal,
        )
        if normal_parent_train_accuracies is not None:
            normal_kw["parent_train_accuracies"] = normal_parent_train_accuracies
        if normal_parent_val_logliks is not None:
            normal_kw["parent_val_logliks"] = normal_parent_val_logliks
        if normal_parent_overall_logliks is not None:
            normal_kw["parent_overall_logliks"] = normal_parent_overall_logliks
        if normal_parent_program_ids is not None:
            normal_kw["parent_program_ids"] = list(normal_parent_program_ids)
        normal_codes = generate_program_variants(**normal_kw)
        codes.extend(normal_codes)
        sources.extend(["normal"] * len(normal_codes))
    return codes, sources


def generate_program_variants(
    client: OpenAI,
    model_name: str,
    parent_programs: List[str],
    train_trials: List[Dict[str, Any]],
    n_variants: int = 10,
    max_tokens: int = 800,
    parent_train_accuracies: Optional[List[float]] = None,
    parent_train_mses: Optional[List[float]] = None,
    parent_val_logliks: Optional[List[Optional[float]]] = None,
    parent_overall_logliks: Optional[List[Optional[float]]] = None,
    dataset: str = "choice13k",
    max_prompt_train_trials: int = 1_000_000,
    max_prompt_trials_per_problem: int = 0,
    prompt_train_trials_seed: int = 0,
    fitness_metric: str = "accuracy",
    cpc18_official_mse: bool = True,
    max_workers: int = 5,
    prompt_suffix: Optional[str] = None,
    prompt_observation_trials: Optional[List[Dict[str, Any]]] = None,
    extra_prompt_trials: Optional[List[Dict[str, Any]]] = None,
    extra_prompt_trials_label: str = "Validation observations",
    run_prompts_dir: Optional[str] = None,
    max_parent_chars: int = 6000,
    warn_parent_truncation_ratio: float = 0.5,
    sample_size_for_warning: int = 10,
    prompt_stats_path: Optional[Path] = None,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    prompt_diagnostics_dir: Optional[Path] = None,
    phase: str = "evolution",
    participant_id: Optional[int] = None,
    iteration: Optional[int] = None,
    prompt_debug: bool = False,
    prompt_debug_exit: bool = True,
    generation_debug_out: Optional[Dict[str, Any]] = None,
    past_invalid_program_errors: Optional[List[Dict[str, Any]]] = None,
    past_error_prompt_section: Optional[str] = None,
    max_error_prompt_chars: int = 1200,
    error_feedback_mode: str = "new",
    explain_mode: bool = False,
    explain_suffix: Optional[str] = None,
    explain_artifacts_out: Optional[List[Dict[str, Any]]] = None,
    llm_decoding_seed_base: Optional[int] = None,
    g2_arm: Optional[str] = None,
    g2_paired_pack_path: Optional[str] = None,
    parent_program_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate full program variants based on parent program and training trials.
    
    This generates complete choose(problem, history) implementations without
    restrictions on structure or logic - only the function signature is fixed.
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    if run_prompts_dir:
        rp = Path(run_prompts_dir)
        prompt_path = str(rp / "infer_single_choice.txt")
        code_template_path = str(rp / "single_code_template.txt")
    elif fitness_metric == "loglik":
        prompt_path = os.path.join(
            PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "non_strict", "loglik", "infer_single_choice.txt"
        )
        code_template_path = os.path.join(
            PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "non_strict", "loglik", "single_code_template.txt"
        )
    else:
        prompt_path = os.path.join(
            PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "non_strict", "infer_single_choice.txt"
        )
        code_template_path = os.path.join(
            PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "non_strict", "single_code_template.txt"
        )
    
    try:
        base_prompt = open(prompt_path).read()
        code_template = load_single_code_template(code_template_path)
    except FileNotFoundError as e:
        if run_prompts_dir:
            raise FileNotFoundError(
                f"Required run prompt files missing under {run_prompts_dir}: {e}"
            ) from e
        print(f"Warning: Could not load prompt files: {e}")
        print("Falling back to hardcoded prompts.")
        # Fallback to hardcoded prompts
        if fitness_metric == "loglik":
            base_prompt = """You are given observations of human choices in risky-gamble problems.
Each problem presents two gambles: Option A and Option B. A gamble has outcomes and their probabilities (percent).
You will see a short history of previous trials for the same participant and problem, including chosen option and feedback if available.

Write Python code that reproduces the observed behavior. You must generate a program implementing:

def choose(problem, history):
    \"\"\"
    problem: dict with keys
        - gamble_A: {"probs": List[float] or None, "rewards": List[float]}
        - gamble_B: {"probs": List[float] or None, "rewards": List[float]}
        - option_keys: e.g., ["A","B"]
        - has_feedback: bool
    history: list of dicts with keys
        - action: int (0 for A, 1 for B)
        - feedback: float or None
    return: float, probability of choosing option 1 (Option B)
    \"\"\"

Constraints:
- Pure Python, no imports, deterministic.
- Use only the provided problem and history.
- Do not call external APIs.
- Return a single float in [0, 1].
- The returned value must be the probability of choosing option 1 (Option B).
- Higher returned values should mean the participant is more likely to choose Option B.
- Do not sample or use randomness.
- `problem["gamble_A"]["probs"]` or `problem["gamble_B"]["probs"]` may be None.
- Handle missing probabilities safely with an explicit None check.

Provide only the code for choose(...) as a complete function body.

""" + CONCISE_PROGRAM_GUIDANCE + """
"""
            code_template = ""
        else:
            base_prompt = """You are given observations of human choices in risky-gamble problems.
Each problem presents two gambles: Option A and Option B. A gamble has outcomes and their probabilities (percent).
You will see a short history of previous trials for the same participant and problem, including chosen option and feedback if available.

Write Python code that reproduces the observed behavior. You must generate a program implementing:

def choose(problem, history):
    \"\"\"
    problem: dict with keys
        - gamble_A: {"probs": List[float], "rewards": List[float]}
        - gamble_B: {"probs": List[float], "rewards": List[float]}
        - option_keys: e.g., ["A","B"]
        - has_feedback: bool
    history: list of dicts with keys
        - action: int (0 for A, 1 for B)
        - feedback: float or None
    return: int, 0 for Option A or 1 for Option B
    \"\"\"

Constraints:
- Pure Python, no imports, deterministic.
- Use only the provided problem and history.
- Do not call external APIs.

Provide only the code for choose(...) as a complete function body.

""" + CONCISE_PROGRAM_GUIDANCE + """
"""
            code_template = ""
    
    # Trials serialized into the prompt (evaluation still uses full train_trials elsewhere).
    freeze_payload: Optional[Dict[str, Any]] = None
    freeze_examples = False
    runtime_contract = ""
    pre_capped_train = False
    pre_capped_val = False
    refinement_val_observations = prompt_observation_trials is not None
    val_trials_for_budget: Optional[List[Dict[str, Any]]] = None
    val_source: Optional[List[Dict[str, Any]]] = None
    if g2_paired_pack_path:
        freeze_payload = load_paired_pack_freeze(Path(g2_paired_pack_path))
        freeze_examples = True
        runtime_contract = _runtime_contract_text(run_prompts_dir)
        pooled_val_for_freeze = extra_prompt_trials if extra_prompt_trials is not None else []
        frozen_train, frozen_val = materialize_frozen_trials(
            freeze_payload,
            pooled_train=train_trials,
            pooled_val=pooled_val_for_freeze,
        )
        observation_trials_source = frozen_train
        trials_for_prompt = frozen_train
        val_source = frozen_val
        val_trials_for_budget = frozen_val
        refinement_val_observations = False
        pre_capped_train = True
        pre_capped_val = True
    elif refinement_val_observations:
        observation_trials_source = list(prompt_observation_trials)
        trials_for_prompt = list(prompt_observation_trials)
    elif extra_prompt_trials is not None:
        val_source = extra_prompt_trials
        observation_trials_source = train_trials
        trials_for_prompt, val_trials_for_budget = _cap_prompt_train_and_val_trials(
            train_trials,
            extra_prompt_trials,
            max_trials=max_prompt_train_trials,
            max_trials_per_problem=max_prompt_trials_per_problem,
            subsample_seed=prompt_train_trials_seed,
        )
    else:
        observation_trials_source = train_trials
        trials_for_prompt = _cap_and_subsample_prompt_trials(
            train_trials,
            max_trials=max_prompt_train_trials,
            max_trials_per_problem=max_prompt_trials_per_problem,
            subsample_seed=prompt_train_trials_seed,
            label="train",
        )

    if trials_for_prompt and "problem" in trials_for_prompt[0]:
        prob0 = trials_for_prompt[0]["problem"]
        dataset_type = str(prob0.get("dataset_alias") or dataset or "choice13k")
        if "gamble_A" in prob0 and dataset_type not in PSYCH101_BINARY_DATASETS:
            dataset_type = "choice13k"
        elif "Ha" in prob0:
            dataset_type = "cpc18"
    else:
        dataset_type = dataset or "choice13k"

    if prompt_suffix:
        base_prompt = f"{base_prompt.rstrip()}\n\n{prompt_suffix.strip()}\n"
    error_section = (past_error_prompt_section or "").strip()
    if not error_section and past_invalid_program_errors:
        error_section = _build_past_error_prompt_section(
            list(past_invalid_program_errors),
            iteration=iteration,
            max_error_prompt_chars=max_error_prompt_chars,
            error_feedback_mode=error_feedback_mode,
        ).strip()
    if error_section:
        base_prompt = f"{base_prompt.rstrip()}\n\n{error_section}\n"

    from utils.teh.prompt_sanitize import CANDIDATE_OUTPUT_RULES

    code_template_suffix = single_code_template_prompt_suffix(code_template)
    if explain_mode and explain_suffix:
        candidate_output_rules = f"\n{explain_suffix.strip()}\n"
    else:
        candidate_output_rules = f"\n{CANDIDATE_OUTPUT_RULES}\n"
    num_parents = len(parent_programs)
    parent_lengths_before = [len(p) for p in parent_programs]
    prompt_parent_programs = list(parent_programs)
    paired_parent_count = None
    if freeze_payload is not None and freeze_payload.get("paired_parent_count") is not None:
        paired_parent_count = max(1, int(freeze_payload["paired_parent_count"]))
        if len(prompt_parent_programs) > paired_parent_count:
            prompt_parent_programs = prompt_parent_programs[:paired_parent_count]
            if parent_program_ids is not None:
                parent_program_ids = list(parent_program_ids)[:paired_parent_count]
            num_parents = len(prompt_parent_programs)
            parent_lengths_before = [len(p) for p in prompt_parent_programs]

    parent_ctx_kwargs = {
        "dataset": dataset,
        "fitness_metric": fitness_metric,
        "parent_train_accuracies": parent_train_accuracies,
        "parent_train_mses": parent_train_mses,
        "parent_val_logliks": parent_val_logliks,
        "parent_overall_logliks": parent_overall_logliks,
        "cpc18_official_mse": cpc18_official_mse,
    }

    def _parent_ctx_builder(*, prompt_parent_programs: List[str], **_kwargs: Any) -> str:
        return _build_parent_context_for_prompt(
            prompt_parent_programs=prompt_parent_programs,
            num_parents=len(prompt_parent_programs),
            **parent_ctx_kwargs,
        )

    prompt_text, trunc_diag, trunc_steps = _truncate_psych_prompt_to_budget(
        base_prompt=base_prompt,
        train_trials=trials_for_prompt,
        train_trials_source=observation_trials_source,
        val_trials=val_trials_for_budget,
        val_trials_source=val_source,
        extra_prompt_trials_label=extra_prompt_trials_label,
        parent_programs=prompt_parent_programs,
        parent_context_builder=_parent_ctx_builder,
        parent_context_kwargs={},
        code_template_suffix=code_template_suffix,
        candidate_output_rules=candidate_output_rules,
        dataset=dataset,
        dataset_type=dataset_type,
        hard_prompt_token_cap=hard_prompt_token_cap,
        prompt_token_estimator=prompt_token_estimator,
        max_prompt_train_trials=max_prompt_train_trials,
        max_prompt_trials_per_problem=max_prompt_trials_per_problem,
        prompt_train_trials_seed=prompt_train_trials_seed,
        max_parent_chars=max_parent_chars,
        refinement_val_observations=refinement_val_observations,
        pre_capped_train=pre_capped_train,
        pre_capped_val=pre_capped_val,
        runtime_contract=runtime_contract,
        freeze_examples=freeze_examples,
        output_reserve=int(max_tokens),
    )

    parent_lengths_after = [len(p) for p in prompt_parent_programs]
    truncated_count = sum(
        1 for a, b in zip(parent_lengths_before, parent_lengths_after) if a != b
    )
    if (
        max_parent_chars > 0
        and sample_size_for_warning > 0
        and truncated_count / sample_size_for_warning >= warn_parent_truncation_ratio
    ):
        print(
            f"Warning: truncated {truncated_count}/{num_parents} parent program(s) for LLM prompt "
            f"(>= {warn_parent_truncation_ratio:.0%} of sample_size={sample_size_for_warning}); "
            f"max_parent_chars={max_parent_chars}."
        )

    parent_ids_before = (
        list(parent_program_ids)
        if parent_program_ids is not None
        else [f"parent_{i}" for i in range(len(prompt_parent_programs))]
    )
    n_parents_after = int(trunc_diag.get("parents_after") or len(prompt_parent_programs))
    parent_ids_after = parent_ids_before[:n_parents_after]
    parent_retention_differed = n_parents_after != int(
        trunc_diag.get("parents_before") or len(prompt_parent_programs)
    )
    freeze_example_ids = list((freeze_payload or {}).get("example_ids") or [])
    freeze_n_available = int((freeze_payload or {}).get("n_selected_available") or 0)
    freeze_n_included = int((freeze_payload or {}).get("n_examples_included") or 0)
    suffix_tokens = int((freeze_payload or {}).get("source_suffix_tokens") or 0)
    if freeze_examples and g2_arm == "control":
        suffix_tokens = 0

    diag_base: Dict[str, Any] = {
        "participant_id": participant_id,
        "phase": phase,
        "iteration": iteration,
        "hard_prompt_token_cap": hard_prompt_token_cap,
        "prompt_token_estimator": prompt_token_estimator,
        "truncation_steps": trunc_steps,
        **trunc_diag,
        **_prompt_example_split_diagnostic_fields(
            train_before=int(trunc_diag.get("train_trials_before") or 0),
            train_after=int(trunc_diag.get("train_trials_after") or 0),
            val_before=int(trunc_diag.get("val_trials_before") or 0),
            val_after=int(trunc_diag.get("val_trials_after") or 0),
        ),
        "g2_arm": g2_arm,
        "g2_paired_pack_version": (freeze_payload or {}).get("version")
        if freeze_payload
        else None,
        "g2_paired_pack_path": str(g2_paired_pack_path) if g2_paired_pack_path else None,
        "g2_shared_paired_packing_cap": freeze_n_included if freeze_examples else None,
        "paired_parent_count": paired_parent_count if freeze_examples else None,
        "target_examples_available": freeze_n_available if freeze_examples else None,
        "target_examples_included": freeze_n_included if freeze_examples else None,
        "target_example_ids": freeze_example_ids if freeze_examples else None,
        "identical_target_examples_to_freeze": True if freeze_examples else None,
        "parents_before": int(trunc_diag.get("parents_before") or len(prompt_parent_programs)),
        "parents_after": n_parents_after,
        "parent_ids_before": parent_ids_before,
        "parent_ids_after": parent_ids_after,
        "parent_retention_differed": bool(parent_retention_differed) if freeze_examples else None,
        "source_suffix_tokens": suffix_tokens if (g2_arm or prompt_suffix or freeze_examples) else 0,
        "g2_reserved_source_suffix_tokens": int(
            (freeze_payload or {}).get("source_suffix_tokens") or 0
        )
        if freeze_examples
        else None,
        "source_suffix_present": prompt_contains_source_suffix(prompt_text),
        "trim_actions": list(trunc_steps),
        "trim_reason": g2_trim_reason(trunc_steps),
    }
    _warn_prompt_truncation(diag_base)

    if prompt_stats_path is not None:
        prompt_stats: Dict[str, Any] = {
            "n_parents": num_parents,
            "parent_char_lengths_before": parent_lengths_before,
            "parent_char_lengths_after": parent_lengths_after,
            "truncated_parent_count": truncated_count,
            "final_prompt_char_length": len(prompt_text),
            "max_parent_chars": max_parent_chars,
            "prompt_tokens_before_truncation": trunc_diag.get("prompt_tokens_before_truncation"),
            "prompt_tokens_after_truncation": trunc_diag.get("prompt_tokens_after_truncation"),
            "truncation_steps": trunc_steps,
            "g2_arm": g2_arm,
            "g2_paired_pack_version": diag_base.get("g2_paired_pack_version"),
            "target_example_ids": diag_base.get("target_example_ids"),
            "parents_before": diag_base.get("parents_before"),
            "parents_after": diag_base.get("parents_after"),
            "parent_ids_after": parent_ids_after,
            "parent_retention_differed": diag_base.get("parent_retention_differed"),
            "trim_reason": diag_base.get("trim_reason"),
        }
        prompt_stats_path = Path(prompt_stats_path)
        prompt_stats_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_stats_path.write_text(json.dumps(prompt_stats, indent=2) + "\n", encoding="utf-8")

    diagnostics_dir = prompt_diagnostics_dir
    if diagnostics_dir is None and prompt_stats_path is not None:
        diagnostics_dir = prompt_stats_path.parent.parent

    tokens_final = estimate_tokens(prompt_text, estimator=prompt_token_estimator)
    if freeze_examples:
        if not fits_input_and_context(
            tokens_final,
            input_ceiling=hard_prompt_token_cap,
            output_reserve=int(max_tokens),
        ):
            raise PairedPackingFitError(
                "G.2 packed prompt overflowed after truncation with frozen examples "
                f"(tokens={tokens_final}, cap={hard_prompt_token_cap}, "
                f"output_reserve={int(max_tokens)}). "
                "No post-packing re-append or example shrink is allowed."
            )
        # Contract is already inside the packed prompt. Do not re-append.
    else:
        if tokens_final > hard_prompt_token_cap:
            try:
                prompt_text, _ = _enforce_prompt_budget(
                    prompt_text,
                    hard_prompt_token_cap=hard_prompt_token_cap,
                    strict_prompt_budget=strict_prompt_budget,
                    prompt_token_estimator=prompt_token_estimator,
                    overflow_components=trunc_diag.get("overflow_components") or {},
                    truncation_steps=trunc_steps,
                    phase=phase,
                    participant_id=participant_id,
                    iteration=iteration,
                    candidate_index=None,
                    diagnostics_dir=diagnostics_dir,
                    diagnostics_base=diag_base,
                )
            except PromptBudgetExceededError:
                raise
        prompt_text = append_runtime_contract_if_present(prompt_text, run_prompts_dir)
        tokens_final = estimate_tokens(prompt_text, estimator=prompt_token_estimator)

    diag_base["final_chat_template_tokens"] = tokens_final
    diag_base["prompt_tokens_after_truncation"] = tokens_final

    debug_captures: List[Dict[str, Any]] = []
    explain_artifact_by_idx: Dict[int, Dict[str, Any]] = {}

    def _generate_one(cand_idx: int = 0) -> str:
        if not prompt_text:
            return ""
        cand_idx = int(cand_idx)
        request_seed = (
            None
            if llm_decoding_seed_base is None
            else int(llm_decoding_seed_base) + cand_idx
        )
        call_diag = {
            **diag_base,
            "candidate_index": cand_idx,
            "llm_decoding_seed": request_seed,
        }
        tokens = estimate_tokens(prompt_text, estimator=prompt_token_estimator)
        if tokens > hard_prompt_token_cap:
            try:
                _enforce_prompt_budget(
                    prompt_text,
                    hard_prompt_token_cap=hard_prompt_token_cap,
                    strict_prompt_budget=strict_prompt_budget,
                    prompt_token_estimator=prompt_token_estimator,
                    overflow_components=trunc_diag.get("overflow_components") or {},
                    truncation_steps=trunc_steps,
                    phase=phase,
                    participant_id=participant_id,
                    iteration=iteration,
                    candidate_index=cand_idx,
                    diagnostics_dir=diagnostics_dir,
                    diagnostics_base=call_diag,
                )
            except PromptBudgetExceededError:
                return ""
            return ""
        _append_prompt_diagnostic(
            {
                **call_diag,
                "status": "ok",
                "prompt_tokens_before_truncation": trunc_diag.get(
                    "prompt_tokens_before_truncation"
                ),
                "prompt_tokens_after_truncation": tokens,
                "final_chat_template_tokens": tokens,
                "g2_arm": g2_arm,
                "target_examples_available": diag_base.get("target_examples_available"),
                "target_examples_included": diag_base.get("target_examples_included"),
                "target_example_ids": diag_base.get("target_example_ids"),
                "g2_shared_paired_packing_cap": diag_base.get("g2_shared_paired_packing_cap"),
                "g2_paired_pack_version": diag_base.get("g2_paired_pack_version"),
                "parents_before": diag_base.get("parents_before"),
                "parents_after": diag_base.get("parents_after"),
                "parent_ids_before": diag_base.get("parent_ids_before"),
                "parent_ids_after": diag_base.get("parent_ids_after"),
                "source_suffix_tokens": diag_base.get("source_suffix_tokens"),
                "trim_actions": diag_base.get("trim_actions"),
                "trim_reason": diag_base.get("trim_reason"),
            },
            diagnostics_dir,
        )
        raw_content = ""
        sanitize_reason = "llm_not_called"
        try:
            create_kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": max_tokens,
            }
            if request_seed is not None:
                create_kwargs["seed"] = int(request_seed)
            resp = client.chat.completions.create(**create_kwargs)
            raw_content = resp.choices[0].message.content or ""
            if explain_mode:
                from utils.teh_transfer.explain_parse import parse_explain_response

                parsed = parse_explain_response(raw_content)
                artifact: Dict[str, Any] = {
                    "raw_response": raw_content,
                    "rationale": parsed.rationale,
                    "parse_ok": parsed.ok,
                    "parse_error": "" if parsed.ok else parsed.error,
                }
                explain_artifact_by_idx[cand_idx] = artifact
                if not parsed.ok:
                    sanitize_reason = f"explain_parse:{parsed.error}"
                    cleaned = ""
                else:
                    cleaned, sanitize_reason = _sanitize_llm_python_candidate_with_reason(
                        parsed.program_text, required_markers=("def choose(",)
                    )
                    artifact["sanitize_reason"] = sanitize_reason
            else:
                cleaned, sanitize_reason = _sanitize_llm_python_candidate_with_reason(
                    raw_content, required_markers=("def choose(",)
                )
            if prompt_debug or generation_debug_out is not None:
                debug_captures.append(
                    {
                        "candidate_index": cand_idx,
                        "raw_content": raw_content,
                        "sanitize_reason": sanitize_reason,
                        "sanitized_char_length": len(cleaned),
                    }
                )
            return cleaned
        except Exception as e:
            print(f"Warning: Failed to generate program variant: {e}")
            if prompt_debug or generation_debug_out is not None:
                debug_captures.append(
                    {
                        "candidate_index": cand_idx,
                        "raw_content": raw_content,
                        "sanitize_reason": f"llm_exception:{e}",
                        "sanitized_char_length": 0,
                    }
                )
            return ""

    codes = _parallel_generate_children(
        n_variants,
        _generate_one,
        max_workers=max_workers,
        desc="Generating candidate programs",
    )

    if explain_artifacts_out is not None:
        for i in range(n_variants):
            explain_artifacts_out.append(
                explain_artifact_by_idx.get(
                    i,
                    {
                        "raw_response": "",
                        "rationale": "",
                        "parse_ok": False,
                        "parse_error": "missing_artifact",
                    },
                )
            )

    if generation_debug_out is not None:
        generation_debug_out.clear()
        generation_debug_out.update(
            {
                "prompt_text": prompt_text,
                "trunc_diag": trunc_diag,
                "trunc_steps": trunc_steps,
                "captures": list(debug_captures),
                "phase": phase,
                "participant_id": participant_id,
                "iteration": iteration,
            }
        )

    if prompt_debug and diagnostics_dir is not None and not any(c.strip() for c in codes):
        _save_prompt_debug_bundle(
            Path(diagnostics_dir) / "prompt_debug",
            phase=phase,
            participant_id=participant_id,
            iteration=iteration,
            prompt_text=prompt_text,
            trunc_diag=trunc_diag,
            captures=debug_captures,
            exit_after_save=prompt_debug_exit,
        )

    return codes


def _run_pre_evolution_explore_phase(
    *,
    explore_candidates: int,
    client: OpenAI,
    model_name: str,
    seed_code: str,
    dataset: str,
    participant_id: int,
    train_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    fitness_metric: str,
    n_eval_seeds: int,
    elite_parents: List[Tuple[Any, ...]],
    elite_val_logliks: List[Optional[float]],
    track_elite_val_loglik: bool,
    sample_size: int,
    elite_pool_size: Optional[int],
    baseline_train_eval: Dict[str, Any],
    output_path: Optional[Path],
    save_artifacts: bool,
    max_prompt_train_trials: int,
    max_prompt_trials_per_problem: int,
    llm_max_tokens: int,
    max_workers: int,
    split_seed: int,
    run_prompts_dir: Optional[str],
    max_parent_chars: int,
    warn_parent_truncation_ratio: float,
    sample_size_for_warning: int,
    hard_prompt_token_cap: int,
    strict_prompt_budget: bool,
    prompt_token_estimator: str,
    initial_pool_from_global: bool = False,
    initial_pool_size_before_explore: Optional[int] = None,
    evolution_selection_score: str = "train_val",
    explore_parent_programs: Optional[List[Tuple[str, str]]] = None,
    pin_program_ids: Optional[Sequence[str]] = None,
    prompt_suffix: Optional[str] = None,
    explore_seed_candidates: int = 0,
    mem_trace_file: Optional[Path] = None,
    mem_run_id: str = "",
    baseline_selection_score: Optional[float] = None,
    baseline_val_loglik: Optional[float] = None,
    mdl_lambda: float = 0.0,
) -> None:
    """
    One-shot candidate generation before the evolution loop.

    By default generates from the seed program only. When ``explore_parent_programs``
    is set, the fixed ``explore_candidates`` budget is split across those parents.
    Valid runtime candidates are merged into elite_parents (pool is sorted/capped after).
    ``pin_program_ids`` are never dropped by the post-explore cap.
    """
    n_explore = int(explore_candidates)
    if n_explore <= 0:
        return
    mdl_lambda = normalize_mdl_lambda(mdl_lambda)

    use_train_val = _uses_train_val_evolution_selection(evolution_selection_score, fitness_metric)
    n_train = len(train_trials)
    n_val = len(val_trials)
    explore_warn_key = f"explore_p{participant_id}"

    print(f"\n{'='*80}")
    print("PRE-EVOLUTION EXPLORE PHASE")
    print(f"{'='*80}")
    print(f"Requested explore candidates: {n_explore}")
    handoff_parents = list(explore_parent_programs) if explore_parent_programs else []
    n_seed_cands, handoff_counts = split_explore_budget_seed_and_parents(
        n_explore,
        n_seed=int(explore_seed_candidates),
        n_handoff_parents=len(handoff_parents),
    )
    if handoff_parents:
        explore_parents = list(handoff_parents)
        parent_counts = list(handoff_counts)
        if n_seed_cands > 0:
            explore_parents.append((seed_code, "baseline"))
            parent_counts.append(n_seed_cands)
    else:
        explore_parents = [(seed_code, "baseline")]
        parent_counts = [n_explore]
    if initial_pool_from_global:
        if explore_parent_programs:
            alloc = ", ".join(
                f"{pid}={k}" for (_, pid), k in zip(explore_parents, parent_counts)
            )
            print(
                f"Initial elite pool from handoff: "
                f"{initial_pool_size_before_explore or len(elite_parents)} program(s); "
                f"explore budget distributed: {alloc}."
            )
        else:
            print(
                f"Initial elite pool from global phase: "
                f"{initial_pool_size_before_explore or len(elite_parents)} program(s); "
                "explore uses seed program only (not pool parents)."
            )
    if prompt_suffix:
        print("[INFO] Explore prompt includes a cross-task source rank-1 suffix.")

    explore_dir: Optional[Path] = None
    candidates_dir: Optional[Path] = None
    if save_artifacts and output_path is not None:
        explore_dir = output_path / "explore_phase"
        explore_dir.mkdir(parents=True, exist_ok=True)
        candidates_dir = explore_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

    if fitness_metric == "loglik" and is_binary_loglik_dataset(dataset):
        seed_parent_train_accs = [float(baseline_train_eval["avg_loglik"])]
    else:
        seed_parent_train_accs = [float(baseline_train_eval["accuracy"])]

    prompt_stats_path = (
        (explore_dir / "prompt_stats.json") if explore_dir is not None else None
    )
    if val_trials:
        print(
            f"[LLM prompt] Explore phase injects {len(train_trials)} train + "
            f"{len(val_trials)} validation trials "
            f"(shared cap via max_prompt_train_trials={max_prompt_train_trials})."
        )
    else:
        print(
            f"[LLM prompt] Explore phase injects {len(train_trials)} train trials only "
            f"(no validation split available)."
        )
    candidate_codes: List[str] = []
    candidate_prompt_parent_ids: List[str] = []
    for parent_i, ((parent_code, parent_id), n_from_parent) in enumerate(
        zip(explore_parents, parent_counts)
    ):
        if n_from_parent <= 0:
            continue
        if fitness_metric == "loglik" and is_binary_loglik_dataset(dataset):
            parent_accs = [float(baseline_train_eval["avg_loglik"])]
        else:
            parent_accs = [float(baseline_train_eval["accuracy"])]
        variants = generate_program_variants(
            client=client,
            model_name=model_name,
            parent_programs=[parent_code],
            train_trials=train_trials,
            extra_prompt_trials=val_trials if val_trials else None,
            n_variants=n_from_parent,
            max_tokens=llm_max_tokens,
            dataset=dataset,
            parent_train_accuracies=parent_accs,
            max_prompt_train_trials=max_prompt_train_trials,
            max_prompt_trials_per_problem=max_prompt_trials_per_problem,
            prompt_train_trials_seed=int(split_seed) + 70_000 + parent_i * 1_009,
            fitness_metric=fitness_metric,
            cpc18_official_mse=False,
            max_workers=max_workers,
            run_prompts_dir=run_prompts_dir,
            max_parent_chars=max_parent_chars,
            warn_parent_truncation_ratio=warn_parent_truncation_ratio,
            sample_size_for_warning=sample_size_for_warning,
            prompt_stats_path=prompt_stats_path,
            hard_prompt_token_cap=hard_prompt_token_cap,
            strict_prompt_budget=strict_prompt_budget,
            prompt_token_estimator=prompt_token_estimator,
            prompt_diagnostics_dir=output_path,
            phase="explore",
            participant_id=int(participant_id),
            iteration=None,
            prompt_suffix=prompt_suffix,
            llm_decoding_seed_base=_phase_llm_decoding_seed_base(
                split_seed=int(split_seed),
                iteration_step=0,
                participant_id=int(participant_id),
                batch_offset=int(parent_i) * 10_007,
            ),
        )
        candidate_codes.extend(variants)
        candidate_prompt_parent_ids.extend([str(parent_id)] * len(variants))
        print(
            f"Explore generated {len(variants)} candidate(s) from parent {parent_id} "
            f"(requested {n_from_parent})."
        )

    candidate_results: List[Dict[str, Any]] = []
    for idx, code in enumerate(candidate_codes):
        if candidates_dir is not None:
            (candidates_dir / f"candidate_{idx}.py").write_text(code or "")
        code = _sanitize_llm_python_candidate(code, required_markers=("def choose(",))
        if not code:
            row: Dict[str, Any] = {
                "idx": idx,
                "train_loglik": float("-inf"),
                "test_loglik": None,
                "fitness": float("-inf") if fitness_metric == "loglik" else 0.0,
                "runtime_valid": False,
            }
            if val_trials:
                row["val_loglik"] = float("-inf")
            candidate_results.append(row)
            continue
        choose_fn = compile_program(code)
        if choose_fn is None:
            row = {
                "idx": idx,
                "train_loglik": float("-inf"),
                "test_loglik": None,
                "fitness": float("-inf") if fitness_metric == "loglik" else 0.0,
                "runtime_valid": False,
            }
            if val_trials:
                row["val_loglik"] = float("-inf")
            candidate_results.append(row)
            continue
        try:
            train_eval = _evaluate_loglik_for_dataset(
                dataset, choose_fn, train_trials, n_seeds=n_eval_seeds
            )
            val_eval = (
                _evaluate_loglik_for_dataset(
                    dataset, choose_fn, val_trials, n_seeds=n_eval_seeds
                )
                if val_trials
                else None
            )
        except (AssertionError, TypeError, ValueError):
            row = {
                "idx": idx,
                "train_loglik": float("-inf"),
                "test_loglik": None,
                "fitness": float("-inf") if fitness_metric == "loglik" else 0.0,
                "runtime_valid": False,
            }
            if val_trials:
                row["val_loglik"] = float("-inf")
            candidate_results.append(row)
            continue
        train_loglik = float(train_eval["avg_loglik"])
        val_loglik = float(val_eval["avg_loglik"]) if val_eval is not None else None
        runtime_valid = train_eval.get("errors", 0) == 0
        if fitness_metric == "loglik":
            selection_score = _evolution_selection_score(
                train_loglik,
                val_loglik,
                n_train,
                n_val,
                evolution_selection_score=evolution_selection_score,
                warn_key=explore_warn_key if use_train_val else None,
            )
            fitness = selection_score if use_train_val else train_loglik
        else:
            selection_score = None
            fitness = float(train_eval["accuracy"])
        if not runtime_valid:
            fitness = float("-inf") if fitness_metric == "loglik" else 0.0
        row = {
            "idx": idx,
            "code": code,
            "train_acc": float(train_eval["accuracy"]),
            "test_acc": None,
            "train_loglik": train_loglik,
            "test_loglik": None,
            "fitness": fitness,
            "runtime_valid": runtime_valid,
        }
        if selection_score is not None:
            row["selection_score"] = selection_score
        if val_eval is not None:
            row["val_loglik"] = val_loglik
        if fitness_metric == "loglik" and selection_score is not None:
            _attach_mdl_to_candidate(
                row,
                mdl_lambda=mdl_lambda,
                n=_mdl_n_for_score(
                    evolution_selection_score, n_train, n_val, val_loglik
                ),
                selection_score=selection_score,
            )
        candidate_results.append(row)

    selected_results = [r for r in candidate_results if r.get("runtime_valid", False)]
    sort_candidates(selected_results, mdl_lambda)
    best_explore_test_record: Optional[Dict[str, Any]] = None
    if selected_results and fitness_metric == "loglik":
        best_explore_test_record = _passive_diagnostic_test_of_program(
            dataset=dataset,
            code=selected_results[0].get("code"),
            test_trials=test_trials,
            n_eval_seeds=n_eval_seeds,
            program_id=f"explore_candidate_{selected_results[0]['idx']}",
            phase="explore",
            iteration=None,
        )
        selected_results[0]["test_loglik"] = best_explore_test_record.get("test_loglik")
        selected_results[0]["test_acc"] = best_explore_test_record.get("test_accuracy")
        selected_results[0]["diagnostic_test"] = best_explore_test_record
    for result in selected_results:
        program_id = f"explore_candidate_{result['idx']}"
        elite_parents.append(
            _elite_tuple_for_ranking(
                result["code"],
                result["fitness"],
                result.get("test_acc"),
                program_id,
                result["train_loglik"] if use_train_val else result["train_acc"],
                mdl_lambda=mdl_lambda,
                mdl_score=result.get("mdl_score"),
            )
        )
        if track_elite_val_loglik:
            elite_val_logliks.append(_safe_float(result.get("val_loglik")))

    elite_cap = _elite_pool_capacity(sample_size, elite_pool_size)
    if pin_program_ids:
        _cap_elite_preserving_program_ids(
            elite_parents,
            elite_val_logliks,
            elite_cap=elite_cap,
            pinned_ids=pin_program_ids,
            track_elite_val_loglik=track_elite_val_loglik,
            mdl_lambda=mdl_lambda,
        )
    elif track_elite_val_loglik:
        paired = list(zip(elite_parents, elite_val_logliks))
        sort_elite_pairs(paired, mdl_lambda)
        paired = paired[:elite_cap]
        elite_parents[:] = [p[0] for p in paired]
        elite_val_logliks[:] = [p[1] for p in paired]
    else:
        sort_elites(elite_parents, mdl_lambda)
        elite_parents[:] = elite_parents[:elite_cap]

    best_explore_score: Optional[float] = None
    if selected_results:
        best_explore_score = float(selected_results[0]["fitness"])
        if fitness_metric == "loglik":
            print(
                f"Best exploration train log-likelihood: "
                f"{float(selected_results[0]['train_loglik']):.6f}"
            )
            if best_explore_test_record and best_explore_test_record.get("test_loglik") is not None:
                print(
                    f"Best exploration diagnostic test log-likelihood: "
                    f"{float(best_explore_test_record['test_loglik']):.6f} "
                    f"(program {best_explore_test_record['program_id']}; passive)"
                )
        else:
            print(
                f"Best exploration train accuracy: "
                f"{float(selected_results[0]['train_acc']):.4f}"
            )

    print(
        f"Valid explore candidates added to pool: {len(selected_results)} / {n_explore}"
    )
    print(f"Elite pool size after explore phase: {len(elite_parents)} (cap={elite_cap})")

    if explore_dir is not None:
        metrics: Dict[str, Any] = {
            **_participant_metric_id(participant_id),
            "explore_candidates_requested": n_explore,
            "n_runtime_valid": len(selected_results),
            "n_added_to_pool": len(selected_results),
            "elite_pool_size_after": len(elite_parents),
            "best_explore_fitness": best_explore_score,
            "evolution_selection_score": evolution_selection_score,
            "explore_used_population_parents": bool(explore_parent_programs),
            "candidate_results": [
                {
                    "idx": r["idx"],
                    "train_acc": r.get("train_acc"),
                    "test_acc": r.get("test_acc"),
                    "train_loglik": r.get("train_loglik"),
                    "test_loglik": r.get("test_loglik"),
                    "val_loglik": r.get("val_loglik"),
                    "selection_score": r.get("selection_score"),
                    "fitness": r.get("fitness"),
                    "runtime_valid": r.get("runtime_valid", False),
                    "prompt_parent_id": (
                        candidate_prompt_parent_ids[r["idx"]]
                        if r["idx"] < len(candidate_prompt_parent_ids)
                        else None
                    ),
                    **_candidate_mdl_log_fields(r),
                }
                for r in candidate_results
            ],
        }
        if mdl_enabled(mdl_lambda):
            metrics["mdl_lambda"] = float(mdl_lambda)
        if initial_pool_from_global:
            metrics["initial_pool_from_global"] = True
            metrics["initial_pool_size_before_explore"] = (
                initial_pool_size_before_explore
                if initial_pool_size_before_explore is not None
                else len(elite_parents) - len(selected_results)
            )
        if best_explore_score is not None and selected_results:
            metrics["best_explore_idx"] = selected_results[0].get("idx")
            if fitness_metric == "loglik":
                metrics["best_explore_train_loglik"] = selected_results[0].get("train_loglik")
                if selected_results[0].get("selection_score") is not None:
                    metrics["best_explore_selection_score"] = selected_results[0].get(
                        "selection_score"
                    )
                if selected_results[0].get("mdl_score") is not None:
                    metrics["best_explore_mdl_score"] = selected_results[0].get("mdl_score")
            else:
                metrics["best_explore_train_acc"] = selected_results[0].get("train_acc")
        if best_explore_test_record is not None:
            metrics["diagnostic_test"] = best_explore_test_record
        (explore_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    # Passive MEM: explore candidates vs exact generation reference.
    if mem_trace_file is not None:
        from utils.mem.reference_types import (
            REF_POPULATION_PROGRAM,
            REF_SEED_BASELINE,
        )

        use_pop = bool(explore_parent_programs)
        # Build parent id -> selection score from elite pool when available.
        elite_score_by_id = {
            str(p[3]): float(p[1]) for p in elite_parents if p[1] is not None
        }
        seed_sel = baseline_selection_score
        if seed_sel is None and fitness_metric == "loglik":
            seed_sel = float(baseline_train_eval.get("avg_loglik"))
        parent_recs = []
        for parent_code, parent_id in explore_parents:
            pid_s = str(parent_id)
            score = elite_score_by_id.get(pid_s)
            if score is None and pid_s in ("baseline", "global_baseline"):
                score = seed_sel
            parent_recs.append(
                {
                    "program_id": pid_s,
                    "selection_score": score,
                    "train_loglik": score,
                    "val_loglik": baseline_val_loglik,
                }
            )
        append_mem_trace_record(
            mem_trace_file,
            build_iteration_context_record(
                dataset=dataset,
                participant_id=participant_id,
                run_id=mem_run_id,
                split_seed=int(split_seed),
                phase="explore",
                iteration=0,
                evolution_selection_score=evolution_selection_score,
                selected_parents=parent_recs,
                best_selected_parent_id=str(explore_parents[0][1]) if explore_parents else "baseline",
            ),
        )
        elite_ids_after = {str(p[3]) for p in elite_parents}
        for result in candidate_results:
            idx = int(result.get("idx", 0))
            program_id = f"explore_candidate_{idx}"
            prompt_pid = (
                candidate_prompt_parent_ids[idx]
                if idx < len(candidate_prompt_parent_ids)
                else "baseline"
            )
            if use_pop and prompt_pid not in ("baseline", "global_baseline"):
                reference_kind = REF_POPULATION_PROGRAM
            else:
                reference_kind = REF_SEED_BASELINE
            ref_score = elite_score_by_id.get(str(prompt_pid))
            if ref_score is None:
                ref_score = seed_sel
            cand_score = result.get("selection_score")
            if cand_score is None:
                cand_score = result.get("fitness") if result.get("runtime_valid") else None
            append_mem_trace_record(
                mem_trace_file,
                build_candidate_record(
                    dataset=dataset,
                    participant_id=participant_id,
                    run_id=mem_run_id,
                    split_seed=int(split_seed),
                    phase="explore",
                    iteration=0,
                    candidate_id=program_id,
                    candidate_idx=idx,
                    source="explore",
                    code=result.get("code") or "",
                    runtime_valid=bool(result.get("runtime_valid", False)),
                    train_loglik=_safe_float(result.get("train_loglik")),
                    val_loglik=_safe_float(result.get("val_loglik")),
                    selection_score=_safe_float(cand_score),
                    reference_parent_id=str(prompt_pid),
                    reference_parent_score=_safe_float(ref_score),
                    reference_kind=reference_kind,
                    reference_type=reference_kind,
                    reference_id=str(prompt_pid),
                    reference_is_exact=True,
                    delta_f=compute_delta_f(
                        _safe_float(cand_score), _safe_float(ref_score)
                    ),
                    survived_elite_truncation=program_id in elite_ids_after,
                    evolution_selection_score=evolution_selection_score,
                    prompted_parent_ids=[str(prompt_pid)],
                ),
            )


def _person_evolution_extra_prompt_trials(
    *,
    t_pics_gated_transfer: bool,
    val_trials: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Gated person-evolution only: serialize observed validation into candidate prompts.

    Generic TEH / historical PICS omit ``extra_prompt_trials`` (train-only examples).
    Test trials are never accepted; callers must not pass them here.
    """
    if not t_pics_gated_transfer:
        return None
    if not val_trials:
        return None
    return list(val_trials)


def run_evolution(
    seed_program_path: str,
    dataset: str = "choice13k",
    participant_id: int = 0,
    data_path: str = "data",
    num_blocks: Optional[int] = None,
    num_walls: Optional[int] = None,
    agent_id: Optional[int] = None,
    n_iterations: int = 5,
    n_candidates_per_iteration: int = 10,
    fresh_n_candidates: int = 0,
    explore_candidates: int = 0,
    model_name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    client_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    wandb=None,
    n_eval_seeds: int = 3,
    sample_size: int = 10,
    sample_parents: bool = True,
    sampled_parents_decay: bool = True,
    elite_pool_size: Optional[int] = None,
    filter_mixed_gambles: bool = False,
    save_artifacts: bool = True,
    all_data_mode: bool = False,
    choice13k_experiment: Optional[Experiment] = None,
    fitness_metric: str = "accuracy",
    split_ratio: float = 0.8,
    split_seed: int = 42,
    choice13k_train_trials_override: Optional[List[Dict[str, Any]]] = None,
    choice13k_test_trials_override: Optional[List[Dict[str, Any]]] = None,
    choice13k_simple_logging: bool = False,
    max_prompt_train_trials: int = 1_000_000,
    max_prompt_trials_per_problem: int = 0,
    llm_max_tokens: int = 800,
    cpc18_official_mse: bool = False,
    gate_phase: bool = False,
    run_phase: str = "all",
    refinement_phase: bool = True,
    refinement_iters: int = 5,
    refinement_val_threshold: float = -1.0,
    max_workers: int = 5,
    global_elite_parents: Optional[List[Tuple[Any, ...]]] = None,
    global_pool_handoff: bool = False,
    run_prompts_dir: Optional[str] = None,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    ablation: Optional[str] = None,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_parent_chars: int = 6000,
    warn_parent_truncation_ratio: float = 0.5,
    early_stop_iters: Optional[int] = None,
    hard_prompt_token_cap: int = 14000,
    strict_prompt_budget: bool = True,
    prompt_token_estimator: str = "char4",
    prompt_debug: bool = False,
    prompt_debug_on_no_valid: bool = True,
    prompt_debug_exit: bool = False,
    evolution_selection_score: str = "train_val",
    max_error_prompt_chars: int = 1200,
    error_feedback_mode: str = DEFAULT_ERROR_FEEDBACK_MODE,
    mem_trace: bool = True,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    speekenbrink_split: str = "chronological",
    explore_from_handoff_parents: bool = False,
    explore_population_top_k: int = 0,
    mdl_lambda: float = 0.0,
    explore_prompt_suffix: Optional[str] = None,
    explore_seed_candidates: int = 0,
    t_pics_gated_transfer: bool = False,
):
    """
    Run iterative evolution loop over programs (Choice13k, Gridworld, or CPC18 Track II, non-strict mode).
    
    Args:
        seed_program_path: Path to seed program
        dataset: "choice13k", "gridworld", or "cpc18" (Track II)
        participant_id: Which participant's data to use (0-indexed, for choice13k and cpc18)
        data_path: Path to data directory (for gridworld) or CPC18 Track II data directory (for cpc18)
        num_blocks: Number of blocks (for gridworld)
        num_walls: Number of walls (for gridworld)
        agent_id: Agent type ID (for gridworld)
        n_iterations: Number of evolution iterations
        n_candidates_per_iteration: Number of candidate programs per iteration
        model_name: LLM model name for generation
        client_kwargs: Optional OpenAI client kwargs (for local vLLM server)
        output_dir: Optional output directory for saving results
        mem_trace: When True, append passive MEM JSONL under the participant output dir
            (iteration_context + candidate records). Does not affect search behavior.
    """
    error_feedback_mode = _normalize_error_feedback_mode(error_feedback_mode)
    mem_trace_enabled = bool(mem_trace)
    if fitness_metric not in ("accuracy", "loglik"):
        raise ValueError(f"Invalid fitness_metric: {fitness_metric!r} (expected 'accuracy' or 'loglik')")
    if not is_binary_loglik_dataset(dataset):
        raise ValueError(
            f"TEH supports binary-loglik datasets only: {sorted(PARTICIPANT_DATASETS)}; "
            f"got {dataset!r}"
        )
    if fitness_metric == "loglik" and dataset not in _LOGlik_VAL_SPLIT_DATASETS:
        raise ValueError("fitness_metric='loglik' requires a registered TEH binary dataset.")
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")
    evolution_selection_score = _normalize_evolution_selection_score(
        evolution_selection_score
    )
    mdl_lambda = normalize_mdl_lambda(mdl_lambda)
    use_train_val_selection = _uses_train_val_evolution_selection(
        evolution_selection_score, fitness_metric
    )
    selection_warn_key = f"p{participant_id}"
    invalid_candidate_errors: List[Dict[str, Any]] = _ErrorFeedbackStore(
        error_feedback_mode
    )

    val_trials: List[Dict[str, Any]] = []

    # Initialize client
    if client_kwargs is None:
        client_kwargs = {}
    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    
    # Load seed program
    print(f"Loading seed program from {seed_program_path}...")
    seed_code = load_seed_program(seed_program_path)
    
    is_cpc18_mse = False
    is_cpc18_split = False
    test_observed_blocks = None
    if choice13k_train_trials_override is not None and choice13k_test_trials_override is not None:
        train_trials = choice13k_train_trials_override
        test_trials = choice13k_test_trials_override
        if not val_trials:
            val_trials = []
        options = train_trials[0]["options"] if train_trials else ([0, 1] if test_trials else [])
        print(f"Loading {dataset} from across-participants split (precomputed).")
        print(f"[Split] Train trials: {len(train_trials)}, Test trials: {len(test_trials)}")
    elif is_mixed_gambles_dataset(dataset):
        csv_path = mixed_gambles_csv if mixed_gambles_csv else DEFAULT_CSV_PATH
        print(f"Loading mixed_gambles from {csv_path} for subject {participant_id}...")
        train_trials, val_trials, test_trials, options = load_mixed_gambles_trials(
            participant_id,
            csv_path=csv_path,
            filter_gain_loss_only=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        n_parsed = len(train_trials) + len(val_trials) + len(test_trials)
        print(
            f"[Load] Parsed total_trials={n_parsed} "
            f"(train={len(train_trials)}, val={len(val_trials)}, test={len(test_trials)}, "
            f"seed={split_seed}, ratio={split_ratio:.3f})"
        )
    elif is_external_dataset(dataset):
        data_dir = str(_REPO_ROOT / external_default_data_dir(dataset))
        print(f"Loading external dataset {dataset} from {data_dir} for participant {participant_id}...")
        train_trials, val_trials, test_trials, options = load_external_loglik_trials(
            dataset,
            participant_id,
            data_dir=data_dir,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        n_parsed = len(train_trials) + len(val_trials) + len(test_trials)
        print(
            f"[Load] Parsed total_trials={n_parsed} "
            f"(train={len(train_trials)}, val={len(val_trials)}, test={len(test_trials)}, "
            f"seed={split_seed}, ratio={split_ratio:.3f})"
        )
    else:
        print(f"Loading Psych-101 dataset {dataset} for participant {participant_id}...")
        if choice13k_experiment is not None:
            exp = choice13k_experiment
        else:
            exp = get_psych101_binary_experiment(
                dataset,
                participant_id,
                split=psych_dataset_split,
                local_dataset=local_dataset,
            )
        n_blocks, n_parsed = _psych101_experiment_trial_counts(exp)
        from utils.teh.limited_data_protocol import (
            SPEEKENBRINK_ALIAS as _SPEEKEN_ALIAS,
            apply_limited_data_protocol,
            split_continuous_session_chronological,
            use_speekenbrink_chronological_split as _use_speeken_chrono,
        )
        from data_modules.psych101_binary import experiment_to_trial_dicts as _exp_to_trials

        if str(dataset).strip() == _SPEEKEN_ALIAS and _use_speeken_chrono(
            speekenbrink_split
        ):
            all_trials = _exp_to_trials(exp)
            train_trials, val_trials, test_trials = split_continuous_session_chronological(
                all_trials, split_ratio
            )
            options = list(train_trials[0]["options"]) if train_trials else list(
                (test_trials[0]["options"] if test_trials else [])
            )
            _split_kind = "chronological_session"
        else:
            train_trials, val_trials, test_trials, options = split_psych_experiment(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
            _split_kind = (
                "legacy_chronological_days"
                if str(dataset).strip() == "14kool2016when"
                else (
                    "legacy_pseudo_block_shuffle"
                    if str(dataset).strip() == _SPEEKEN_ALIAS
                    else "legacy_unit_shuffle"
                )
            )
        print(
            f"[Load] Parsed blocks={n_blocks}, total_trials={n_parsed}; "
            f"split train={len(train_trials)}, val={len(val_trials)}, test={len(test_trials)} "
            f"(seed={split_seed}, ratio={split_ratio:.3f}, split_kind={_split_kind})"
        )

    sparse_audit: Optional[SparseObservationAudit] = None
    limited_manifest = None
    if choice13k_train_trials_override is None:
        from utils.teh.limited_data_protocol import apply_limited_data_protocol

        train_trials, val_trials, test_trials, sparse_audit, limited_manifest = apply_limited_data_protocol(
            train_trials,
            val_trials,
            test_trials,
            dataset=dataset,
            participant_id=int(participant_id),
            split_seed=int(split_seed),
            split_ratio=float(split_ratio),
            max_observed_trials_per_participant=max_observed_trials_per_participant,
            limited_data_protocol=limited_data_protocol,
            limited_train_val=limited_train_val,
            split_kind=locals().get("_split_kind", "legacy_unit_shuffle"),
        )
        if sparse_audit.applied or (
            limited_manifest is not None and limited_manifest.protocol != "off"
        ):
            print(
                f"[LimitedData protocol={limited_manifest.protocol if limited_manifest else 'off'}] "
                f"train+val cap={sparse_audit.max_observed_trials_per_participant}: "
                f"train {sparse_audit.original_n_train}->{sparse_audit.retained_n_train}, "
                f"val {sparse_audit.original_n_val}->{sparse_audit.retained_n_val}, "
                f"test {sparse_audit.original_n_test}->{sparse_audit.retained_n_test}; "
                f"fingerprint={sparse_audit.subset_fingerprint[:12]}"
            )

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        output_dir = (
            f"{teh_output_base_dir(dataset, timestamp, psych_dataset_split=psych_dataset_split, ablation=ablation)}"
            f"/participant_{participant_id}"
        )
    output_path = Path(output_dir)
    if save_artifacts:
        output_path.mkdir(parents=True, exist_ok=True)
        if sparse_audit is not None and should_persist_sparse_audit(sparse_audit):
            write_sparse_audit(output_path / SPARSE_AUDIT_FILENAME, sparse_audit)
            run_root = (
                output_path.parent
                if output_path.name.startswith("participant_")
                else output_path
            )
            with _SHARED_EXPERIMENT_CSV_LOCK:
                append_sparse_audit_jsonl(run_root / SPARSE_AUDIT_JSONL_FILENAME, sparse_audit)
                rewrite_sparse_audit_csv_from_jsonl(run_root / SPARSE_AUDIT_JSONL_FILENAME)
        if limited_manifest is not None and should_persist_limited_data_manifest(limited_manifest):
            write_limited_data_manifest(
                output_path / LIMITED_DATA_MANIFEST_FILENAME, limited_manifest
            )
            run_root = (
                output_path.parent
                if output_path.name.startswith("participant_")
                else output_path
            )
            with _SHARED_EXPERIMENT_CSV_LOCK:
                append_limited_data_manifest_jsonl(
                    run_root / LIMITED_DATA_MANIFEST_JSONL_FILENAME, limited_manifest
                )
                rewrite_limited_data_csv_from_jsonl(
                    run_root / LIMITED_DATA_MANIFEST_JSONL_FILENAME
                )
    error_history_path = output_path / "error_history.jsonl"
    mem_trace_file: Optional[Path] = None
    mem_run_id = ""
    if mem_trace_enabled and save_artifacts:
        mem_trace_file = mem_trace_path(output_path)
        # Prefer run folder name (parent of participant_*), else participant dir name.
        if output_path.name.startswith("participant_"):
            mem_run_id = output_path.parent.name
        else:
            mem_run_id = output_path.name
        print(f"MEM trace logging enabled: {mem_trace_file}")
    
    # Set up local log file for wandb metrics (if wandb is enabled)
    log_file_path = None
    if wandb is not None and save_artifacts and not (choice13k_simple_logging and is_binary_loglik_dataset(dataset)):
        log_file_path = output_path / "wandb_metrics.jsonl"
    
    # ===== BASELINE EVALUATION =====
    print(f"\n{'='*80}")
    print(f"BASELINE EVALUATION: Evaluating seed program ({seed_program_path})")
    print(f"{'='*80}")

    baseline_val_eval = None
    
    baseline_fn = compile_program(seed_code)
    if baseline_fn is None:
        print("ERROR: Failed to compile baseline program!")
        return None
    baseline_train_mse_eval = None
    baseline_test_mse_eval = None
    if _uses_train_val_test_loglik_split(
        dataset, fitness_metric, cpc18_official_mse=False
    ):
        baseline_train_eval = _evaluate_loglik_for_dataset(
            dataset, baseline_fn, train_trials, verbose=True, n_seeds=n_eval_seeds
        )
        baseline_test_eval = _evaluate_loglik_for_dataset(
            dataset, baseline_fn, test_trials, verbose=True, n_seeds=n_eval_seeds
        )
        if val_trials:
            baseline_val_eval = _evaluate_loglik_for_dataset(
                dataset, baseline_fn, val_trials, verbose=True, n_seeds=n_eval_seeds
            )
    else:
        baseline_train_eval = evaluate_program(
            baseline_fn, train_trials, verbose=True, n_seeds=n_eval_seeds
        )
        baseline_test_eval = evaluate_program(
            baseline_fn, test_trials, verbose=True, n_seeds=n_eval_seeds
        )
    
    print(f"\nBaseline Performance:")
    print(f"  Train accuracy: {baseline_train_eval['accuracy']:.4f} ({baseline_train_eval['correct']}/{baseline_train_eval['total']})")
    print(f"  Test accuracy: {baseline_test_eval['accuracy']:.4f} ({baseline_test_eval['correct']}/{baseline_test_eval['total']})")
    if fitness_metric == "loglik":
        print(
            f"  Train avg log-likelihood: {baseline_train_eval['avg_loglik']:.6f}, "
            f"test: {baseline_test_eval['avg_loglik']:.6f}"
        )
    if baseline_val_eval is not None:
        print(f"  Val avg log-likelihood: {baseline_val_eval['avg_loglik']:.6f}")
    if is_cpc18_mse:
        print(f"  Train MSE: {baseline_train_mse_eval['mse']:.4f}")
        print(f"  Test MSE (official): {baseline_test_mse_eval['mse']:.4f}")
    
    # Store baseline results (will be included in final results.json)
    baseline_results = {
        "train_accuracy": baseline_train_eval['accuracy'],
        "test_accuracy": baseline_test_eval['accuracy'],
        "train_correct": baseline_train_eval['correct'],
        "train_total": baseline_train_eval['total'],
        "test_correct": baseline_test_eval['correct'],
        "test_total": baseline_test_eval['total'],
    }
    if is_cpc18_mse and baseline_train_mse_eval is not None and baseline_test_mse_eval is not None:
        baseline_results["train_mse"] = baseline_train_mse_eval['mse']
        baseline_results["test_mse"] = baseline_test_mse_eval['mse']
    if fitness_metric == "loglik":
        baseline_results["train_loglik"] = baseline_train_eval["avg_loglik"]
        baseline_results["test_loglik"] = baseline_test_eval["avg_loglik"]
    if baseline_val_eval is not None:
        baseline_results["val_loglik"] = baseline_val_eval["avg_loglik"]
    if fitness_metric == "loglik" and is_binary_loglik_dataset(dataset):
        baseline_results["selection_score"] = _evolution_selection_score(
            float(baseline_train_eval["avg_loglik"]),
            _safe_float(baseline_val_eval["avg_loglik"]) if baseline_val_eval else None,
            len(train_trials),
            len(val_trials),
            evolution_selection_score=evolution_selection_score,
            warn_key=selection_warn_key if use_train_val_selection else None,
        )
        baseline_results["evolution_selection_score"] = evolution_selection_score
    
    # Log baseline to wandb at step=0
    if wandb is not None:
        baseline_log_dict = {}
        if dataset == "gridworld":
            # Use agent-specific keys if agent_id is provided
            if agent_id is not None:
                baseline_log_dict = {
                    f"a{agent_id}_train_accuracy": baseline_train_eval["accuracy"],
                    f"a{agent_id}_test_accuracy": baseline_test_eval["accuracy"],
                    f"a{agent_id}_is_baseline": 1,
                }
            else:
                baseline_log_dict = {
                    f"gw_train_accuracy": baseline_train_eval["accuracy"],
                    f"gw_test_accuracy": baseline_test_eval["accuracy"],
                    f"gw_is_baseline": 1,
                }
        elif is_cpc18_mse:
            if all_data_mode:
                baseline_log_dict = {
                    f"p{participant_id}_train_fitness": -baseline_train_mse_eval["mse"],
                    f"p{participant_id}_test_fitness": -baseline_test_mse_eval["mse"],
                }
            else:
                baseline_log_dict = {
                    f"p{participant_id}_train_fitness": -baseline_train_mse_eval["mse"],
                    f"p{participant_id}_train_mse": baseline_train_mse_eval["mse"],
                    f"p{participant_id}_test_mse": baseline_test_mse_eval["mse"],
                    f"p{participant_id}_is_baseline": 1,
                    f"p{participant_id}_train_accuracy": baseline_train_eval["accuracy"],
                    f"p{participant_id}_test_accuracy": baseline_test_eval["accuracy"],
                }
        elif is_cpc18_split:
            baseline_log_dict = {
                f"p{participant_id}_train_fitness": (
                    baseline_train_eval["avg_loglik"]
                    if fitness_metric == "loglik"
                    else baseline_train_eval["accuracy"]
                ),
                f"p{participant_id}_test_fitness": (
                    baseline_test_eval["avg_loglik"]
                    if fitness_metric == "loglik"
                    else baseline_test_eval["accuracy"]
                ),
                f"p{participant_id}_train_mse": None,
                f"p{participant_id}_test_mse": None,
                f"p{participant_id}_is_baseline": 1,
                f"p{participant_id}_train_loglik": baseline_train_eval["avg_loglik"],
                f"p{participant_id}_test_loglik": baseline_test_eval["avg_loglik"],
                f"p{participant_id}_train_acc": baseline_train_eval["accuracy"],
                f"p{participant_id}_test_acc": baseline_test_eval["accuracy"],
            }
            if baseline_val_eval is not None:
                baseline_log_dict[f"p{participant_id}_val_loglik"] = baseline_val_eval["avg_loglik"]
                baseline_log_dict["val_loglik"] = baseline_val_eval["avg_loglik"]
        else:
            if all_data_mode:
                if is_binary_loglik_dataset(dataset) and fitness_metric == "loglik":
                    baseline_log_dict = {
                        f"p{participant_id}_train_fitness": baseline_train_eval["avg_loglik"],
                        f"p{participant_id}_test_fitness": baseline_test_eval["avg_loglik"],
                        f"p{participant_id}_train_acc": baseline_train_eval["accuracy"],
                        f"p{participant_id}_test_acc": baseline_test_eval["accuracy"],
                        f"p{participant_id}_train_loglik": baseline_train_eval["avg_loglik"],
                        f"p{participant_id}_test_loglik": baseline_test_eval["avg_loglik"],
                    }
                    if baseline_val_eval is not None:
                        baseline_log_dict[f"p{participant_id}_val_loglik"] = baseline_val_eval["avg_loglik"]
                        baseline_log_dict["val_loglik"] = baseline_val_eval["avg_loglik"]
                    if baseline_results.get("selection_score") is not None:
                        baseline_log_dict[f"p{participant_id}_selection_score"] = baseline_results[
                            "selection_score"
                        ]
                else:
                    baseline_log_dict = {
                        f"p{participant_id}_train_fitness": baseline_train_eval["accuracy"],
                        f"p{participant_id}_test_fitness": baseline_test_eval["accuracy"],
                    }
                    if is_binary_loglik_dataset(dataset):
                        baseline_log_dict[f"p{participant_id}_train_acc"] = baseline_train_eval["accuracy"]
                        baseline_log_dict[f"p{participant_id}_test_acc"] = baseline_test_eval["accuracy"]
                        baseline_log_dict[f"p{participant_id}_train_loglik"] = baseline_train_eval["avg_loglik"]
                        baseline_log_dict[f"p{participant_id}_test_loglik"] = baseline_test_eval["avg_loglik"]
                    if baseline_val_eval is not None:
                        baseline_log_dict[f"p{participant_id}_val_loglik"] = baseline_val_eval["avg_loglik"]
                        baseline_log_dict["val_loglik"] = baseline_val_eval["avg_loglik"]
            else:
                baseline_log_dict = {
                    f"p{participant_id}_train_accuracy": baseline_train_eval["accuracy"],
                    f"p{participant_id}_test_accuracy": baseline_test_eval["accuracy"],
                    f"p{participant_id}_is_baseline": 1,
                }
                if is_binary_loglik_dataset(dataset):
                    baseline_log_dict[f"p{participant_id}_train_loglik"] = baseline_train_eval["avg_loglik"]
                    baseline_log_dict[f"p{participant_id}_test_loglik"] = baseline_test_eval["avg_loglik"]
                    baseline_log_dict[f"p{participant_id}_train_acc"] = baseline_train_eval["accuracy"]
                    baseline_log_dict[f"p{participant_id}_test_acc"] = baseline_test_eval["accuracy"]
                if baseline_val_eval is not None:
                    baseline_log_dict[f"p{participant_id}_val_loglik"] = baseline_val_eval["avg_loglik"]
                    baseline_log_dict["val_loglik"] = baseline_val_eval["avg_loglik"]
                if fitness_metric == "loglik":
                    baseline_log_dict[f"p{participant_id}_selection_score"] = baseline_results.get(
                        "selection_score"
                    )
        if participant_id is not None:
            _wandb_log_participant_metrics(wandb, baseline_log_dict, int(participant_id), 0)
        else:
            wandb.log(baseline_log_dict, step=0)
        
        # Also save baseline to local JSONL file
        if log_file_path is not None:
            baseline_entry = _wandb_jsonl_log_entry(
                step=0,
                iteration=-1,  # Baseline is before iteration 0
                log_dict=baseline_log_dict,
                participant_id=participant_id,
                agent_id=agent_id,
            )
            with open(log_file_path, "a") as f:
                f.write(json.dumps(baseline_entry) + "\n")
    
    # Initialize best program tracking with baseline
    if is_cpc18_mse and baseline_train_mse_eval is not None:
        best_fitness = -baseline_train_mse_eval['mse']
    elif is_cpc18_split and fitness_metric == "loglik":
        best_fitness = baseline_train_eval["avg_loglik"]
    elif is_cpc18_split:
        best_fitness = baseline_train_eval["accuracy"]
    elif is_binary_loglik_dataset(dataset) and fitness_metric == "loglik":
        best_fitness = baseline_train_eval["avg_loglik"]
    else:
        best_fitness = baseline_train_eval["accuracy"]
    
    # Track overall best across all iterations
    if is_cpc18_mse and baseline_train_mse_eval is not None:
        overall_best_train = {
            "train_fitness": -baseline_train_mse_eval['mse'],
            "train_mse": baseline_train_mse_eval['mse'],
            "test_mse": baseline_test_mse_eval['mse'],
            "train_accuracy": baseline_train_eval['accuracy'],
            "test_accuracy": baseline_test_eval['accuracy'],
            "program_id": "baseline"
        }
        overall_best_test = {
            "train_fitness": -baseline_train_mse_eval['mse'],
            "train_mse": baseline_train_mse_eval['mse'],
            "test_mse": baseline_test_mse_eval['mse'],
            "train_accuracy": baseline_train_eval['accuracy'],
            "test_accuracy": baseline_test_eval['accuracy'],
            "program_id": "baseline"
        }
    elif is_cpc18_split:
        overall_best_train = {
            "train_accuracy": baseline_train_eval['accuracy'],
            "test_accuracy": baseline_test_eval['accuracy'],
            "train_loglik": baseline_train_eval['avg_loglik'],
            "test_loglik": baseline_test_eval['avg_loglik'],
            "program_id": "baseline"
        }
        overall_best_test = {
            "train_accuracy": baseline_train_eval['accuracy'],
            "test_accuracy": baseline_test_eval['accuracy'],
            "train_loglik": baseline_train_eval['avg_loglik'],
            "test_loglik": baseline_test_eval['avg_loglik'],
            "program_id": "baseline"
        }
        if fitness_metric == "loglik" and baseline_val_eval is not None:
            _baseline_val_ll = baseline_val_eval["avg_loglik"]
            overall_best_train["val_loglik"] = _baseline_val_ll
            overall_best_test["val_loglik"] = _baseline_val_ll
    else:
        overall_best_train = {
            "train_accuracy": baseline_train_eval['accuracy'],
            "test_accuracy": baseline_test_eval['accuracy'],
            "program_id": "baseline"
        }
        overall_best_test = {
            "train_accuracy": baseline_train_eval['accuracy'],
            "test_accuracy": baseline_test_eval['accuracy'],
            "program_id": "baseline"
        }
        if is_binary_loglik_dataset(dataset):
            overall_best_train["train_loglik"] = baseline_train_eval["avg_loglik"]
            overall_best_train["test_loglik"] = baseline_test_eval["avg_loglik"]
            overall_best_test["train_loglik"] = baseline_train_eval["avg_loglik"]
            overall_best_test["test_loglik"] = baseline_test_eval["avg_loglik"]
        if baseline_val_eval is not None:
            vl = baseline_val_eval["avg_loglik"]
            overall_best_train["val_loglik"] = vl
            overall_best_test["val_loglik"] = vl
    
    # Track elite parents (top programs across all iterations)
    # Format: list of (code, fitness, test_metric, program_id, train_mse, test_mse) tuples
    # For CPC18: fitness = -train_mse (higher is better), sorted by fitness descending
    # For other datasets: fitness = train_acc, sorted by fitness descending
    if is_cpc18_mse and baseline_train_mse_eval is not None and baseline_test_mse_eval is not None:
        elite_parents = [(
            seed_code,
            -baseline_train_mse_eval['mse'],
            baseline_test_mse_eval['mse'],
            "baseline",
            baseline_train_mse_eval['mse'],
            baseline_test_mse_eval['mse'],
        )]
    elif is_cpc18_split:
        _bfit = (
            baseline_train_eval["avg_loglik"]
            if fitness_metric == "loglik"
            else baseline_train_eval["accuracy"]
        )
        elite_parents = [(
            seed_code,
            _bfit,
            baseline_test_eval["accuracy"],
            "baseline",
            None,
            None,
            baseline_train_eval["accuracy"],
        )]
    else:
        if is_binary_loglik_dataset(dataset):
            train_ll = (
                baseline_train_eval["avg_loglik"]
                if fitness_metric == "loglik"
                else None
            )
            _baseline_fit = (
                baseline_train_eval["avg_loglik"]
                if fitness_metric == "loglik"
                else baseline_train_eval["accuracy"]
            )
            val_ll = (
                _safe_float(baseline_val_eval["avg_loglik"])
                if baseline_val_eval is not None
                else None
            )
            if use_train_val_selection and train_ll is not None:
                _baseline_fit = _evolution_selection_score(
                    train_ll,
                    val_ll,
                    len(train_trials),
                    len(val_trials),
                    evolution_selection_score=evolution_selection_score,
                    warn_key=selection_warn_key,
                )
            idx6 = (
                float(baseline_train_eval["avg_loglik"])
                if use_train_val_selection and fitness_metric == "loglik"
                else baseline_train_eval["accuracy"]
            )
            elite_parents = [apply_mdl_to_scored_elite(
                (
                    seed_code,
                    _baseline_fit,
                    baseline_test_eval["accuracy"],
                    "baseline",
                    None,
                    None,
                    idx6,
                ),
                selection_score=float(_baseline_fit),
                n=_mdl_n_for_score(
                    evolution_selection_score,
                    len(train_trials),
                    len(val_trials),
                    val_ll,
                ),
                mdl_lambda=mdl_lambda,
                runtime_valid=True,
            )]
        else:
            elite_parents = [(
                seed_code,
                baseline_train_eval['accuracy'],  # fitness = accuracy
                baseline_test_eval['accuracy'],  # test_metric = test_acc
                "baseline",
                None,  # train_mse not applicable
                None,  # test_mse not applicable
                baseline_train_eval["accuracy"],
            )]

    runtime_valid_evolved_found = False
    track_elite_val_loglik = bool(
        val_trials and fitness_metric == "loglik"
    )
    elite_val_logliks: List[Optional[float]] = []
    if global_elite_parents:
        if fitness_metric != "loglik":
            raise ValueError("global_elite_parents requires fitness_metric='loglik'")
        elite_parents, elite_val_logliks = _global_elite_to_participant_elite(
            global_elite_parents,
            train_trials,
            val_trials,
            dataset=dataset,
            n_eval_seeds=n_eval_seeds,
            evolution_selection_score=evolution_selection_score,
            selection_warn_key=selection_warn_key,
            mdl_lambda=mdl_lambda,
        )
        global_pool_handoff = True
        print(
            f"\nEvolution initial pool: {len(elite_parents)} program(s) initialized from "
            f"global phase elite pool (per-participant train/val re-evaluated; global order preserved)."
        )
        if save_artifacts and output_path is not None:
            dst_pool = output_path / "initial_pool_from_global"
            if dst_pool.exists():
                shutil.rmtree(dst_pool)
            _write_elite_pool_dir(
                dst_pool,
                elite_parents,
                extra_manifest_fields=(
                    [
                        _elite_mdl_manifest_fields(
                            parent,
                            mdl_lambda=mdl_lambda,
                            n=_mdl_n_for_score(
                                evolution_selection_score,
                                len(train_trials),
                                len(val_trials),
                                elite_val_logliks[i] if i < len(elite_val_logliks) else None,
                            ),
                        )
                        or None
                        for i, parent in enumerate(elite_parents)
                    ]
                    if mdl_enabled(mdl_lambda)
                    else None
                ),
            )
    elif track_elite_val_loglik:
        base_val = (
            _safe_float(baseline_val_eval["avg_loglik"])
            if baseline_val_eval is not None
            else None
        )
        elite_val_logliks = [base_val]

    if run_phase not in _RUN_PHASES:
        raise ValueError(f"run_phase must be one of {sorted(_RUN_PHASES)}, got {run_phase!r}")
    if run_phase == "refine":
        raise ValueError(
            "run_phase='refine' must use run_loglik_refine_from_prev_experiment(), not run_evolution()."
        )
    early_stop_patience = _normalize_early_stop_iters(early_stop_iters)
    if is_binary_loglik_dataset(dataset) and fitness_metric == "loglik":
        best_fitness = float(elite_parents[0][1])
    last_significant_best = float(elite_parents[0][1])
    stagnant_iters = 0
    if early_stop_patience is not None:
        print(
            f"Evolution early stop enabled: patience={early_stop_patience}, "
            f"min_improvement={_EARLY_STOP_MIN_IMPROVEMENT:.3f}"
        )
    if use_train_val_selection:
        print(
            f"Evolution selection score mode: {evolution_selection_score} "
            f"(trial-weighted train+val loglik for pool ranking)"
        )

    if (
        int(explore_candidates) > 0
        and run_phase in ("all", "evolution")
    ):
        initial_pool_size_before_explore = len(elite_parents)
        explore_parent_programs, pin_program_ids = select_explore_handoff_parents(
            elite_parents,
            enabled=explore_from_handoff_parents,
            top_k=explore_population_top_k,
        )
        if explore_from_handoff_parents and not explore_parent_programs:
            print(
                "[WARN] explore_from_handoff_parents is set but the elite pool is empty; "
                "explore uses --seed_path only."
            )
        elif explore_parent_programs:
            print(
                f"Explore from {len(explore_parent_programs)} handoff parent(s) "
                f"(explore_population_top_k={int(explore_population_top_k)}; "
                f"ids={[pid for _, pid in explore_parent_programs]})."
            )
        _run_pre_evolution_explore_phase(
            explore_candidates=int(explore_candidates),
            client=client,
            model_name=model_name,
            seed_code=seed_code,
            dataset=dataset,
            participant_id=int(participant_id),
            train_trials=train_trials,
            test_trials=test_trials,
            val_trials=val_trials,
            fitness_metric=fitness_metric,
            n_eval_seeds=n_eval_seeds,
            elite_parents=elite_parents,
            elite_val_logliks=elite_val_logliks,
            track_elite_val_loglik=track_elite_val_loglik,
            sample_size=sample_size,
            elite_pool_size=elite_pool_size,
            baseline_train_eval=baseline_train_eval,
            output_path=output_path if save_artifacts else None,
            save_artifacts=save_artifacts,
            max_prompt_train_trials=max_prompt_train_trials,
            max_prompt_trials_per_problem=max_prompt_trials_per_problem,
            llm_max_tokens=llm_max_tokens,
            max_workers=max_workers,
            split_seed=int(split_seed),
            run_prompts_dir=run_prompts_dir,
            max_parent_chars=max_parent_chars,
            warn_parent_truncation_ratio=warn_parent_truncation_ratio,
            sample_size_for_warning=sample_size,
            hard_prompt_token_cap=hard_prompt_token_cap,
            strict_prompt_budget=strict_prompt_budget,
            prompt_token_estimator=prompt_token_estimator,
            initial_pool_from_global=global_pool_handoff,
            initial_pool_size_before_explore=initial_pool_size_before_explore,
            evolution_selection_score=evolution_selection_score,
            explore_parent_programs=explore_parent_programs,
            pin_program_ids=pin_program_ids,
            prompt_suffix=explore_prompt_suffix,
            explore_seed_candidates=int(explore_seed_candidates),
            mem_trace_file=mem_trace_file,
            mem_run_id=mem_run_id,
            baseline_selection_score=_safe_float(
                (baseline_results or {}).get("selection_score")
            ),
            baseline_val_loglik=(
                _safe_float(baseline_val_eval.get("avg_loglik"))
                if baseline_val_eval is not None
                else None
            ),
            mdl_lambda=mdl_lambda,
        )
        last_significant_best = float(elite_parents[0][1])

    # Evolution loop (uses elite_parents pool for parent selection, not a single parent_program)
    simple_iterations_rows: List[Dict[str, Any]] = []
    simple_iterations_dir = None
    if choice13k_simple_logging and is_binary_loglik_dataset(dataset) and save_artifacts:
        simple_iterations_dir = output_path / "iterations"
        simple_iterations_dir.mkdir(parents=True, exist_ok=True)
    for iteration in range(n_iterations):
        iteration_step = iteration + 1  # 1-indexed to match wandb (0 = baseline)
        iter_best_selection_score: Optional[float] = None
        print(f"\n{'='*80}")
        print(f"Iteration {iteration_step}/{n_iterations}")
        print(f"{'='*80}")
        
        iter_dir = None
        candidates_dir = None
        if save_artifacts and not (choice13k_simple_logging and is_binary_loglik_dataset(dataset)):
            iter_dir = output_path / f"iteration_{iteration_step}"
            iter_dir.mkdir(exist_ok=True)
            candidates_dir = iter_dir / "candidates"
            candidates_dir.mkdir(exist_ok=True)
        
        # Select sample_size parents: uniform sample from elite pool, or top by fitness
        # (from the trimmed elite pool; see _elite_pool_capacity / elite_pool_size).
        pool_size = len(elite_parents)
        if sample_parents and pool_size > 0:
            pid_key = int(participant_id) if participant_id is not None else 0
            rng = np.random.default_rng(
                int(split_seed) + int(iteration_step) * 1_000_003 + pid_key * 17_179
            )
            parent_idxs, best_k, sampled_k = _select_parent_indices_from_elite_pool(
                pool_size,
                sample_size=sample_size,
                sample_parents=True,
                sampled_parents_decay=sampled_parents_decay,
                iter_idx=iteration,
                total_iters=n_iterations,
                rng=rng,
            )
            selected_parents = [elite_parents[int(j)] for j in parent_idxs]
            num_parents_to_use = len(selected_parents)
            if sampled_parents_decay:
                print(
                    f"\nUsing {num_parents_to_use} parent(s) from elite pool "
                    f"({best_k} best + {sampled_k} sampled, size={pool_size}, "
                    f"sample_size={sample_size}, sample_parents=True):"
                )
            else:
                print(
                    f"\nUsing {num_parents_to_use} uniformly sampled parent(s) from elite pool "
                    f"(size={pool_size}, sample_size={sample_size}, sample_parents=True):"
                )
        else:
            num_parents_to_use = min(sample_size, pool_size)
            parent_idxs = list(range(num_parents_to_use))
            selected_parents = elite_parents[:num_parents_to_use]
            print(
                f"\nUsing {num_parents_to_use} top parent(s) from elite set "
                f"(sample_size={sample_size}, sample_parents=False):"
            )
        parent_codes = [p[0] for p in selected_parents]

        mem_selected_parent_records: List[Dict[str, Any]] = []
        mem_best_parent_rec: Optional[Dict[str, Any]] = None
        mem_seed_selection_score: Optional[float] = None
        if mem_trace_file is not None:
            for local_i, parent_tuple in enumerate(selected_parents):
                elite_j = int(parent_idxs[local_i]) if local_i < len(parent_idxs) else local_i
                val_ll = None
                if track_elite_val_loglik and elite_j < len(elite_val_logliks):
                    val_ll = elite_val_logliks[elite_j]
                train_ll = _train_loglik_from_elite_tuple(
                    parent_tuple, evolution_selection_score=evolution_selection_score
                )
                mem_selected_parent_records.append(
                    parent_record_from_elite_tuple(
                        parent_tuple,
                        val_loglik=val_ll,
                        train_loglik=train_ll,
                    )
                )
            mem_best_parent_rec = best_reference_parent(mem_selected_parent_records)
            # Seed/baseline reference for fresh candidates (same selection metric).
            for ep in elite_parents:
                if str(ep[3]) in ("baseline", "global_baseline", "refinement_seed"):
                    mem_seed_selection_score = float(ep[1]) if ep[1] is not None else None
                    break
            if mem_seed_selection_score is None:
                # Fall back to evaluating stored seed fitness from baseline eval path:
                # elite index-1 for the seed program when present as first pool entry.
                mem_seed_selection_score = (
                    float(elite_parents[0][1])
                    if elite_parents and str(elite_parents[0][3]) == "baseline"
                    else None
                )
                if mem_seed_selection_score is None and fitness_metric == "loglik":
                    # Last resort: recompute from baseline metrics already in memory.
                    if use_train_val_selection:
                        mem_seed_selection_score = _evolution_selection_score(
                            float(baseline_train_eval["avg_loglik"]),
                            _safe_float(baseline_val_eval["avg_loglik"])
                            if baseline_val_eval is not None
                            else None,
                            len(train_trials),
                            len(val_trials),
                            evolution_selection_score=evolution_selection_score,
                            warn_key=None,
                        )
                    else:
                        mem_seed_selection_score = float(baseline_train_eval["avg_loglik"])
            append_mem_trace_record(
                mem_trace_file,
                build_iteration_context_record(
                    dataset=dataset,
                    participant_id=participant_id,
                    run_id=mem_run_id,
                    split_seed=int(split_seed),
                    phase="evolution",
                    iteration=iteration_step,
                    evolution_selection_score=evolution_selection_score,
                    selected_parents=mem_selected_parent_records,
                    best_selected_parent_id=(
                        str(mem_best_parent_rec["program_id"])
                        if mem_best_parent_rec is not None
                        else None
                    ),
                ),
            )

        if is_cpc18_mse:
            for i, parent_tuple in enumerate(selected_parents):
                code, fitness, test_mse, prog_id, train_mse, test_mse = elite_core(parent_tuple)[:6]
                print(f"  Parent {i+1}: {prog_id} (train_mse={train_mse:.2f}, test_mse={test_mse:.2f}, fitness={fitness:.2f})")
        else:
            for i, parent_tuple in enumerate(selected_parents):
                code, fitness, test_acc, prog_id, _, _, train_acc_prompt = elite_core(parent_tuple)
                # Test metrics are often unset until the pool-best diagnostic eval.
                test_acc_s = f"{test_acc:.4f}" if test_acc is not None else "N/A"
                if fitness_metric == "loglik" and is_binary_loglik_dataset(dataset):
                    print(f"  Parent {i+1}: {prog_id} (log-likelihood={fitness:.4f}, test_acc={test_acc_s})")
                else:
                    train_s = (
                        f"{train_acc_prompt:.4f}" if train_acc_prompt is not None else "N/A"
                    )
                    print(f"  Parent {i+1}: {prog_id} (train_acc={train_s}, test_acc={test_acc_s})")
        
        parent_train_accs = None
        parent_train_mses = None
        parent_test_mses = None
        if is_cpc18_mse:
            parent_train_mses = [p[4] for p in selected_parents if p[4] is not None]
            parent_test_mses = [p[5] for p in selected_parents if p[5] is not None]
        else:
            # In loglik mode, feed pool-ranking log-likelihood (elite tuple fitness) into prompts.
            if fitness_metric == "loglik" and is_binary_loglik_dataset(dataset):
                parent_train_accs = [p[1] for p in selected_parents]
            else:
                parent_train_accs = [p[6] for p in selected_parents]
        
        # Generate candidate programs (full code, not just parameters)
        prompt_stats_path: Optional[Path] = None
        if save_artifacts:
            if iter_dir is not None:
                prompt_stats_path = iter_dir / "prompt_stats.json"
            elif simple_iterations_dir is not None:
                prompt_stats_path = (
                    simple_iterations_dir / f"iteration_{iteration_step}" / "prompt_stats.json"
                )
        prompt_diag_dir: Optional[Path] = None
        if output_path is not None:
            prompt_diag_dir = Path(output_path)
        elif output_dir is not None:
            prompt_diag_dir = Path(output_dir)
        capture_gen_debug = bool(prompt_debug or prompt_debug_on_no_valid)
        gen_debug: Dict[str, Any] = {}
        fresh_n = _decayed_fresh_n_for_iteration(
            fresh_n_candidates, iteration, n_iterations, n_candidates_per_iteration
        )
        n_normal = n_candidates_per_iteration - fresh_n
        print(
            f"Fresh candidate schedule (evolution): iter_idx={iteration}, "
            f"total_iterations={n_iterations}, fresh_n={fresh_n} "
            f"(max fresh_n_candidates={fresh_n_candidates})"
        )
        print(
            f"\nGenerating {n_candidates_per_iteration} candidate programs: "
            f"{fresh_n} fresh (seed/baseline only), {n_normal} from sampled parents..."
        )
        error_prompt_section = _build_past_error_prompt_section(
            invalid_candidate_errors,
            iteration=iteration_step,
            max_error_prompt_chars=max_error_prompt_chars,
            previous_n_candidates=n_candidates_per_iteration,
            error_feedback_mode=error_feedback_mode,
        )
        _write_iteration_error_prompt_file(iter_dir, error_prompt_section)
        error_prompt_chars_used = len(error_prompt_section)
        num_unique_errors_available = _count_previous_iteration_error_groups(
            invalid_candidate_errors,
            iteration=iteration_step,
            error_feedback_mode=error_feedback_mode,
        )
        print(
            "Error prompt summary: "
            f"num_unique_errors_available={num_unique_errors_available}, "
            f"error_prompt_chars_used={error_prompt_chars_used}"
        )
        fresh_parent_train_accs = None
        if is_cpc18_mse:
            fresh_parent_train_accs = [
                p[4] for p in elite_parents if p[3] == "baseline" and p[4] is not None
            ] or [baseline_train_mse_eval["mse"] if baseline_train_mse_eval else 0.0]
        elif fitness_metric == "loglik" and is_binary_loglik_dataset(dataset):
            fresh_parent_train_accs = [float(baseline_train_eval["avg_loglik"])]
        else:
            fresh_parent_train_accs = [baseline_train_eval["accuracy"]]
        variant_kwargs = {
            "train_trials": train_trials,
            "max_tokens": llm_max_tokens,
            "dataset": dataset,
            "max_prompt_train_trials": max_prompt_train_trials,
            "max_prompt_trials_per_problem": max_prompt_trials_per_problem,
            "prompt_train_trials_seed": split_seed,
            "fitness_metric": fitness_metric,
            "cpc18_official_mse": False,
            "max_workers": max_workers,
            "run_prompts_dir": run_prompts_dir,
            "max_parent_chars": max_parent_chars,
            "warn_parent_truncation_ratio": warn_parent_truncation_ratio,
            "sample_size_for_warning": sample_size,
            "prompt_stats_path": prompt_stats_path,
            "hard_prompt_token_cap": hard_prompt_token_cap,
            "strict_prompt_budget": strict_prompt_budget,
            "prompt_token_estimator": prompt_token_estimator,
            "prompt_diagnostics_dir": prompt_diag_dir,
            "phase": "evolution",
            "participant_id": int(participant_id) if participant_id is not None else None,
            "iteration": iteration_step,
            "prompt_debug": prompt_debug,
            "prompt_debug_exit": prompt_debug_exit,
            "generation_debug_out": gen_debug if capture_gen_debug else None,
            "past_invalid_program_errors": invalid_candidate_errors,
            "past_error_prompt_section": error_prompt_section,
            "max_error_prompt_chars": max_error_prompt_chars,
            "error_feedback_mode": error_feedback_mode,
            "llm_decoding_seed_base": _phase_llm_decoding_seed_base(
                split_seed=int(split_seed),
                iteration_step=int(iteration_step),
                participant_id=int(participant_id) if participant_id is not None else None,
            ),
        }
        extra_prompt_trials = _person_evolution_extra_prompt_trials(
            t_pics_gated_transfer=bool(t_pics_gated_transfer),
            val_trials=val_trials,
        )
        if extra_prompt_trials is not None:
            variant_kwargs["extra_prompt_trials"] = extra_prompt_trials
            print(
                f"[LLM prompt] Gated person evolution injects {len(train_trials)} train + "
                f"{len(extra_prompt_trials)} validation trials "
                f"(shared cap via max_prompt_train_trials={max_prompt_train_trials})."
            )
        candidate_codes, candidate_sources = _generate_iteration_candidate_codes(
            client=client,
            model_name=model_name,
            fresh_n_candidates=fresh_n,
            n_candidates=n_candidates_per_iteration,
            fresh_parent_programs=[seed_code],
            normal_parent_programs=parent_codes,
            variant_kwargs=variant_kwargs,
            fresh_parent_train_accuracies=fresh_parent_train_accs,
            normal_parent_train_accuracies=parent_train_accs,
        )
        
        # Evaluate candidates
        print(f"\nEvaluating candidates...")
        candidate_results = []
        num_invalid_candidates = 0
        for idx, code in enumerate(tqdm(candidate_codes, desc="Evaluating")):
            if dataset == "gridworld":
                code = _sanitize_llm_python_candidate(
                    code, required_markers=("class FSMAgent", "def act(")
                )
            else:
                code = _sanitize_llm_python_candidate(code, required_markers=("def choose(",))

            # Save candidate code
            if save_artifacts and candidates_dir is not None:
                (candidates_dir / f"candidate_{idx}.py").write_text(code or "")
            
            if not code:
                empty_row: Dict[str, Any] = {
                    "idx": idx,
                    "code": "",
                    "train_acc": 0.0,
                    "test_acc": 0.0,
                    "valid": False,
                }
                if is_binary_loglik_dataset(dataset) or is_cpc18_split:
                    empty_row["train_loglik"] = float("-inf")
                    empty_row["test_loglik"] = float("-inf")
                    empty_row["fitness"] = float("-inf") if fitness_metric == "loglik" else 0.0
                    empty_row["runtime_valid"] = False
                if val_trials:
                    empty_row["val_loglik"] = float("-inf")
                candidate_results.append(empty_row)
                continue
            
            if dataset == "gridworld":
                # Gridworld: evaluate using gridworld evaluation function
                # Train: Evaluate on first 20 steps (matching ROTE's training/weighting phase)
                train_eval = evaluate_gridworld_program(
                    code, data_path, num_blocks, num_walls, agent_id,
                    num_datapoints=80, num_steps=20, n_seeds=n_eval_seeds,
                    evaluate_on_observed=True  # Match ROTE's training: evaluate on first 20 steps
                )
                # Test: Evaluate on future steps (matching ROTE's evaluation phase)
                test_eval = evaluate_gridworld_program(
                    code, data_path, num_blocks, num_walls, agent_id,
                    num_datapoints=20, num_steps=20, n_seeds=n_eval_seeds,
                    evaluate_on_observed=False  # Match ROTE's evaluation: evaluate on future steps
                )
                
                train_acc = train_eval["accuracy"]
                test_acc = test_eval["accuracy"]
                # For gridworld: fitness = train_acc (used for sorting/selection)
                candidate_results.append({
                    "idx": idx,
                    "code": code,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "fitness": train_acc,
                    "train_correct": train_eval["correct"],
                    "test_correct": test_eval["correct"],
                    "train_total": train_eval["total"],
                    "test_total": test_eval["total"],
                    "valid": train_eval["errors"] == 0,
                })
            elif is_cpc18_mse:
                if "problem[" not in code:
                    candidate_results.append({
                        "idx": idx,
                        "code": code,
                        "train_acc": 0.0,
                        "test_acc": 0.0,
                        "train_mse": float('inf'),
                        "test_mse": float('inf'),
                        "fitness": float('-inf'),
                        "valid": False,
                    })
                    continue
                choose_fn = compile_program(code)
                if choose_fn is None:
                    candidate_results.append({
                        "idx": idx,
                        "code": code,
                        "train_acc": 0.0,
                        "test_acc": 0.0,
                        "train_mse": float('inf'),
                        "test_mse": float('inf'),
                        "fitness": float('-inf'),
                        "valid": False,
                    })
                    continue
                train_eval = evaluate_cpc18_program(choose_fn, train_trials, n_seeds=n_eval_seeds)
                test_eval = evaluate_cpc18_program(choose_fn, test_trials, n_seeds=n_eval_seeds)
                train_observed_blocks = test_observed_blocks
                train_mse_eval = evaluate_cpc18_mse(
                    choose_fn, train_trials, train_observed_blocks, n_seeds=n_eval_seeds
                )
                test_mse_eval = evaluate_cpc18_mse(
                    choose_fn, test_trials, test_observed_blocks, n_seeds=n_eval_seeds
                )
                mse_valid = bool(train_mse_eval.get("valid", True))
                if not mse_valid:
                    candidate_results.append({
                        "idx": idx,
                        "code": code,
                        "train_acc": train_eval["accuracy"],
                        "test_acc": test_eval["accuracy"],
                        "train_mse": float('inf'),
                        "test_mse": float(test_mse_eval.get("mse", float("inf"))),
                        "fitness": float('-inf'),
                        "train_correct": train_eval["correct"],
                        "test_correct": test_eval["correct"],
                        "train_total": train_eval["total"],
                        "test_total": test_eval["total"],
                        "valid": False,
                    })
                    continue
                train_acc = train_eval["accuracy"]
                test_acc = test_eval["accuracy"]
                train_mse = train_mse_eval["mse"]
                test_mse = test_mse_eval["mse"]
                fitness = -train_mse
                candidate_results.append({
                    "idx": idx,
                    "code": code,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "fitness": fitness,
                    "train_correct": train_eval["correct"],
                    "test_correct": test_eval["correct"],
                    "train_total": train_eval["total"],
                    "test_total": test_eval["total"],
                    "valid": True,
                })
            elif is_cpc18_split:
                if "problem[" not in code:
                    candidate_results.append({
                        "idx": idx,
                        "code": code,
                        "train_acc": 0.0,
                        "test_acc": 0.0,
                        "train_loglik": float("-inf"),
                        "test_loglik": float("-inf"),
                        "fitness": float("-inf") if fitness_metric == "loglik" else 0.0,
                        "valid": False,
                        "runtime_valid": False,
                    })
                    continue
                choose_fn, compile_error = compile_program_with_error(code)
                _worst = float("-inf") if fitness_metric == "loglik" else 0.0
                if choose_fn is None:
                    num_invalid_candidates += 1
                    _record_invalid_program_error(
                        invalid_candidate_errors,
                        code=code,
                        exc=compile_error,
                        iteration=iteration_step,
                        participant_id=int(participant_id)
                        if participant_id is not None
                        else None,
                        candidate_id=f"candidate_{idx}",
                        eval_split="compile",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                    _fail = {
                        "idx": idx,
                        "code": code,
                        "train_acc": 0.0,
                        "test_acc": 0.0,
                        "train_loglik": float("-inf"),
                        "test_loglik": float("-inf"),
                        "fitness": _worst,
                        "valid": False,
                        "runtime_valid": False,
                    }
                    if val_trials:
                        _fail["val_loglik"] = float("-inf")
                    candidate_results.append(_fail)
                    continue
                try:
                    train_eval = evaluate_cpc18_split_program(choose_fn, train_trials, n_seeds=n_eval_seeds)
                    val_eval = (
                        evaluate_cpc18_split_program(choose_fn, val_trials, n_seeds=n_eval_seeds)
                        if val_trials
                        else None
                    )
                except (TypeError, ValueError, AssertionError) as exc:
                    num_invalid_candidates += 1
                    _record_invalid_program_error(
                        invalid_candidate_errors,
                        code=code,
                        exc=exc,
                        iteration=iteration_step,
                        participant_id=int(participant_id)
                        if participant_id is not None
                        else None,
                        candidate_id=f"candidate_{idx}",
                        eval_split="train",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                    _fail = {
                        "idx": idx,
                        "code": code,
                        "train_acc": 0.0,
                        "test_acc": 0.0,
                        "train_loglik": float("-inf"),
                        "test_loglik": float("-inf"),
                        "fitness": _worst,
                        "valid": False,
                        "runtime_valid": False,
                    }
                    if val_trials:
                        _fail["val_loglik"] = float("-inf")
                    candidate_results.append(_fail)
                    continue
                train_acc = train_eval["accuracy"]
                test_acc = None
                train_loglik = train_eval["avg_loglik"]
                test_loglik = None
                val_loglik = val_eval["avg_loglik"] if val_eval is not None else None
                runtime_valid = train_eval.get("errors", 0) == 0
                if train_eval.get("errors", 0) != 0:
                    num_invalid_candidates += 1
                    _record_invalid_program_error_summary(
                        invalid_candidate_errors,
                        train_eval.get("first_error"),
                        iteration=iteration_step,
                        participant_id=int(participant_id)
                        if participant_id is not None
                        else None,
                        candidate_id=f"candidate_{idx}",
                        quality_score=_safe_float(train_eval.get("avg_loglik")),
                        eval_split="train",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                if val_eval is not None and val_eval.get("errors", 0) != 0:
                    num_invalid_candidates += 1
                    _record_invalid_program_error_summary(
                        invalid_candidate_errors,
                        val_eval.get("first_error"),
                        iteration=iteration_step,
                        participant_id=int(participant_id)
                        if participant_id is not None
                        else None,
                        candidate_id=f"candidate_{idx}",
                        quality_score=_safe_float(train_eval.get("avg_loglik")),
                        eval_split="val",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                fitness, selection_score = _apply_evolution_candidate_selection_fitness(
                    train_loglik=train_loglik,
                    val_loglik=val_loglik,
                    train_acc=train_acc,
                    fitness_metric=fitness_metric,
                    n_train=len(train_trials),
                    n_val=len(val_trials),
                    evolution_selection_score=evolution_selection_score,
                    use_train_val_selection=use_train_val_selection,
                    warn_key=selection_warn_key,
                    runtime_valid=runtime_valid,
                )
                _row = {
                    "idx": idx,
                    "code": code,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_loglik": train_loglik,
                    "test_loglik": test_loglik,
                    "fitness": fitness,
                    "train_correct": train_eval["correct"],
                    "test_correct": None,
                    "train_total": train_eval["total"],
                    "test_total": None,
                    "valid": True,
                    "runtime_valid": runtime_valid,
                }
                if selection_score is not None:
                    _row["selection_score"] = selection_score
                if val_eval is not None:
                    _row["val_loglik"] = val_loglik
                if fitness_metric == "loglik" and selection_score is not None:
                    _attach_mdl_to_candidate(
                        _row,
                        mdl_lambda=mdl_lambda,
                        n=_mdl_n_for_score(
                            evolution_selection_score,
                            len(train_trials),
                            len(val_trials),
                            val_loglik,
                        ),
                        selection_score=selection_score,
                    )
                candidate_results.append(_row)
            elif is_binary_loglik_dataset(dataset):
                choose_fn, compile_error = compile_program_with_error(code)
                _worst = float("-inf") if fitness_metric == "loglik" else 0.0
                if choose_fn is None:
                    num_invalid_candidates += 1
                    _record_invalid_program_error(
                        invalid_candidate_errors,
                        code=code,
                        exc=compile_error,
                        iteration=iteration_step,
                        participant_id=int(participant_id)
                        if participant_id is not None
                        else None,
                        candidate_id=f"candidate_{idx}",
                        eval_split="compile",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                    _fail = {
                        "idx": idx,
                        "code": code,
                        "train_acc": 0.0,
                        "test_acc": 0.0,
                        "train_loglik": float("-inf"),
                        "test_loglik": float("-inf"),
                        "fitness": _worst,
                        "valid": False,
                        "runtime_valid": False,
                    }
                    if val_trials:
                        _fail["val_loglik"] = float("-inf")
                    candidate_results.append(_fail)
                    continue
                try:
                    train_eval = _evaluate_loglik_for_dataset(
                        dataset, choose_fn, train_trials, n_seeds=n_eval_seeds
                    )
                    val_eval = (
                        _evaluate_loglik_for_dataset(
                            dataset, choose_fn, val_trials, n_seeds=n_eval_seeds
                        )
                        if val_trials
                        else None
                    )
                except (AssertionError, TypeError, ValueError) as exc:
                    num_invalid_candidates += 1
                    _record_invalid_program_error(
                        invalid_candidate_errors,
                        code=code,
                        exc=exc,
                        iteration=iteration_step,
                        participant_id=int(participant_id)
                        if participant_id is not None
                        else None,
                        candidate_id=f"candidate_{idx}",
                        eval_split="train",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                    _fail = {
                        "idx": idx,
                        "code": code,
                        "train_acc": 0.0,
                        "test_acc": 0.0,
                        "train_loglik": float("-inf"),
                        "test_loglik": float("-inf"),
                        "fitness": _worst,
                        "valid": False,
                        "runtime_valid": False,
                    }
                    if val_trials:
                        _fail["val_loglik"] = float("-inf")
                    candidate_results.append(_fail)
                    continue
                train_acc = train_eval["accuracy"]
                test_acc = None
                train_loglik = train_eval["avg_loglik"]
                test_loglik = None
                val_loglik = val_eval["avg_loglik"] if val_eval is not None else None
                runtime_valid = train_eval.get("errors", 0) == 0
                if train_eval.get("errors", 0) != 0:
                    num_invalid_candidates += 1
                    _record_invalid_program_error_summary(
                        invalid_candidate_errors,
                        train_eval.get("first_error"),
                        iteration=iteration_step,
                        participant_id=int(participant_id)
                        if participant_id is not None
                        else None,
                        candidate_id=f"candidate_{idx}",
                        quality_score=_safe_float(train_eval.get("avg_loglik")),
                        eval_split="train",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                if val_eval is not None and val_eval.get("errors", 0) != 0:
                    num_invalid_candidates += 1
                    _record_invalid_program_error_summary(
                        invalid_candidate_errors,
                        val_eval.get("first_error"),
                        iteration=iteration_step,
                        participant_id=int(participant_id)
                        if participant_id is not None
                        else None,
                        candidate_id=f"candidate_{idx}",
                        quality_score=_safe_float(train_eval.get("avg_loglik")),
                        eval_split="val",
                        n_candidates_in_iteration=n_candidates_per_iteration,
                        history_path=error_history_path,
                    )
                fitness, selection_score = _apply_evolution_candidate_selection_fitness(
                    train_loglik=train_loglik,
                    val_loglik=val_loglik,
                    train_acc=train_acc,
                    fitness_metric=fitness_metric,
                    n_train=len(train_trials),
                    n_val=len(val_trials),
                    evolution_selection_score=evolution_selection_score,
                    use_train_val_selection=use_train_val_selection,
                    warn_key=selection_warn_key,
                    runtime_valid=runtime_valid,
                )
                _row = {
                    "idx": idx,
                    "code": code,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_loglik": train_loglik,
                    "test_loglik": test_loglik,
                    "fitness": fitness,
                    "train_correct": train_eval["correct"],
                    "test_correct": None,
                    "train_total": train_eval["total"],
                    "test_total": None,
                    "valid": True,
                    "runtime_valid": runtime_valid,
                }
                if selection_score is not None:
                    _row["selection_score"] = selection_score
                if val_eval is not None:
                    _row["val_loglik"] = val_loglik
                if fitness_metric == "loglik" and selection_score is not None:
                    _attach_mdl_to_candidate(
                        _row,
                        mdl_lambda=mdl_lambda,
                        n=_mdl_n_for_score(
                            evolution_selection_score,
                            len(train_trials),
                            len(val_trials),
                            val_loglik,
                        ),
                        selection_score=selection_score,
                    )
                candidate_results.append(_row)

        print(
            "Iteration invalid summary: "
            f"num_invalid_candidates={num_invalid_candidates}, "
            f"num_unique_errors_available={_count_previous_iteration_error_groups(invalid_candidate_errors, iteration=iteration_step + 1, error_feedback_mode=error_feedback_mode)}, "
            f"error_prompt_chars_used={error_prompt_chars_used}"
        )
            
        # Report results
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1} Results:")
        print(f"{'='*80}")
        
        compile_valid_results = [r for r in candidate_results if r.get("valid", False)]
        # For probability-evaluation datasets, selection must use runtime-valid programs only.
        if is_binary_loglik_dataset(dataset) or is_cpc18_split:
            selected_results = [r for r in candidate_results if r.get("runtime_valid", False)]
        else:
            selected_results = list(compile_valid_results)
        person_iter_test_record = None
        if selected_results:
            runtime_valid_evolved_found = True
            # Sort by fitness (for CPC18: -MSE, for others: accuracy)
            sort_candidates(selected_results, mdl_lambda)
            # Keep legacy name for downstream logging blocks.
            valid_results = selected_results
            
            if is_cpc18_mse:
                print(f"\nTop performers (by fitness = -train_MSE, higher is better):")
                for i, result in enumerate(selected_results[:5]):
                    print(
                        f"  {i+1}. Candidate {result['idx']}: "
                        f"train_mse={result['train_mse']:.2f}, "
                        f"test_mse={result['test_mse']:.2f}, "
                        f"fitness={result['fitness']:.2f}"
                    )
            elif is_cpc18_split and fitness_metric == "loglik":
                print(f"\nTop performers (by train avg log-likelihood, higher is better):")
                for i, result in enumerate(selected_results[:5]):
                    _test_ll = (
                        f"{result['test_loglik']:.6f}"
                        if result.get("test_loglik") is not None
                        else "N/A (eval on pool-best only)"
                    )
                    _test_acc = (
                        f"{result['test_acc']:.4f}"
                        if result.get("test_acc") is not None
                        else "N/A"
                    )
                    print(
                        f"  {i+1}. Candidate {result['idx']}: "
                        f"train_loglik={result['train_loglik']:.6f}, "
                        f"test_loglik={_test_ll}, "
                        f"train_acc={result['train_acc']:.4f}, "
                        f"test_acc={_test_acc}"
                    )
            elif is_binary_loglik_dataset(dataset) and fitness_metric == "loglik":
                metric_label = (
                    "selection_score"
                    if use_train_val_selection
                    else "train avg log-likelihood"
                )
                print(f"\nTop performers (by {metric_label}, higher is better):")
                for i, result in enumerate(selected_results[:5]):
                    _test_ll = (
                        f"{result['test_loglik']:.6f}"
                        if result.get("test_loglik") is not None
                        else "N/A (eval on pool-best only)"
                    )
                    _val_ll = (
                        f"{result['val_loglik']:.6f}"
                        if result.get("val_loglik") is not None
                        else "N/A"
                    )
                    _test_acc = (
                        f"{result['test_acc']:.4f}"
                        if result.get("test_acc") is not None
                        else "N/A"
                    )
                    _val_part = f", val_loglik={_val_ll}" if val_trials else ""
                    _sel_part = (
                        f", selection_score={result['selection_score']:.6f}"
                        if use_train_val_selection
                        and result.get("selection_score") is not None
                        else ""
                    )
                    print(
                        f"  {i+1}. Candidate {result['idx']}: "
                        f"train_loglik={result['train_loglik']:.6f}, "
                        f"test_loglik={_test_ll}{_val_part}{_sel_part}, "
                        f"train_acc={result['train_acc']:.4f}, "
                        f"test_acc={_test_acc}"
                    )
            else:
                print(f"\nTop performers (by train accuracy):")
                for i, result in enumerate(selected_results[:5]):
                    print(
                        f"  {i+1}. Candidate {result['idx']}: "
                        f"train_acc={result['train_acc']:.4f}, "
                        f"test_acc={result['test_acc']:.4f}"
                    )
            
            # Best candidate in current generated batch (before elite pool update).
            best_result = selected_results[0]
            best_fitness = best_result["fitness"]
            
            print(f"\nBest candidate in this batch: Candidate {best_result['idx']}")
            if is_cpc18_mse:
                print(f"  Train MSE: {best_result['train_mse']:.2f}")
                print(f"  Test MSE: {best_result['test_mse']:.2f}")
                print(f"  Fitness (-MSE): {best_result['fitness']:.2f}")
            elif is_cpc18_split:
                print(f"  Train accuracy: {best_result['train_acc']:.4f}")
                if best_result["test_acc"] is None:
                    print("  Test accuracy: N/A (eval on pool-best only)")
                    print(
                        f"  Train avg log-likelihood: {best_result['train_loglik']:.6f}, "
                        "test: N/A (eval on pool-best only)"
                    )
                else:
                    print(f"  Test accuracy: {best_result['test_acc']:.4f}")
                    print(
                        f"  Train avg log-likelihood: {best_result['train_loglik']:.6f}, "
                        f"test: {best_result['test_loglik']:.6f}"
                    )
            elif is_binary_loglik_dataset(dataset) and fitness_metric == "loglik":
                print(f"  Train accuracy: {best_result['train_acc']:.4f}")
                if best_result["test_acc"] is None:
                    print("  Test accuracy: N/A (eval on pool-best only)")
                    print(
                        f"  Train avg log-likelihood: {best_result['train_loglik']:.6f}, "
                        "test: N/A (eval on pool-best only)"
                    )
                else:
                    print(f"  Test accuracy: {best_result['test_acc']:.4f}")
                    print(
                        f"  Train avg log-likelihood: {best_result['train_loglik']:.6f}, "
                        f"test: {best_result['test_loglik']:.6f}"
                    )
                if val_trials and best_result.get("val_loglik") is not None:
                    print(f"  Val avg log-likelihood: {best_result['val_loglik']:.6f}")
            else:
                print(f"  Train accuracy: {best_result['train_acc']:.4f}")
                print(f"  Test accuracy: {best_result['test_acc']:.4f}")
            
            # Add only selection-eligible (runtime-valid) candidates to elite set.
            for result in selected_results:
                program_id = f"iteration_{iteration_step}_candidate_{result['idx']}"
                if is_cpc18_mse:
                    elite_parents.append((
                        result["code"],
                        result["fitness"],
                        result.get("test_mse", float('inf')),
                        program_id,
                        result.get("train_mse", None),
                        result.get("test_mse", None),
                    ))
                elif is_cpc18_split:
                    elite_parents.append(
                        _elite_tuple_for_ranking(
                            result["code"],
                            result["fitness"],
                            result["test_acc"],
                            program_id,
                            result["train_loglik"]
                            if use_train_val_selection and fitness_metric == "loglik"
                            else result["train_acc"],
                            mdl_lambda=mdl_lambda,
                            mdl_score=result.get("mdl_score"),
                        )
                    )
                else:
                    elite_parents.append(
                        _elite_tuple_for_ranking(
                            result["code"],
                            result["fitness"],
                            result["test_acc"],
                            program_id,
                            result["train_loglik"]
                            if use_train_val_selection and fitness_metric == "loglik"
                            else result["train_acc"],
                            mdl_lambda=mdl_lambda,
                            mdl_score=result.get("mdl_score"),
                        )
                    )
                if track_elite_val_loglik:
                    elite_val_logliks.append(_safe_float(result.get("val_loglik")))

            # Sort elite set by fitness (descending) and keep top programs.
            # Global handoff: preserve global order through iteration 1; sort from iteration 2+.
            should_sort_elite = (not global_pool_handoff) or (iteration_step >= 1)
            if track_elite_val_loglik:
                paired_elite = list(zip(elite_parents, elite_val_logliks))
                if should_sort_elite:
                    sort_elite_pairs(paired_elite, mdl_lambda)
                elite_cap = _elite_pool_capacity(sample_size, elite_pool_size)
                paired_elite = paired_elite[:elite_cap]
                elite_parents = [p[0] for p in paired_elite]
                elite_val_logliks = [p[1] for p in paired_elite]
            else:
                if should_sort_elite:
                    sort_elites(elite_parents, mdl_lambda)
                elite_cap = _elite_pool_capacity(sample_size, elite_pool_size)
                elite_parents = elite_parents[:elite_cap]

            print(f"\nElite set updated: {len(elite_parents)} programs (elite_pool_cap={elite_cap})")

            # Use the updated elite-pool best for per-iteration reporting.
            # Rank/select on train_val only; diagnostic-test the selected pool-best once.
            iter_best_code, iter_best_fitness, _, iter_best_program_id = elite_parents[0][:4]
            iter_best_selection_score = (
                float(elite_parents[0][1])
                if use_train_val_selection and fitness_metric == "loglik"
                else None
            )
            iter_best_train_acc = best_result["train_acc"]
            iter_best_test_acc = None
            iter_best_train_loglik = best_result.get("train_loglik")
            iter_best_test_loglik = None
            iter_best_val_loglik = best_result.get("val_loglik")
            person_iter_test_record = None
            if fitness_metric == "loglik" and (is_cpc18_split or is_binary_loglik_dataset(dataset)):
                iter_best_fitness = float(elite_parents[0][1])
                if use_train_val_selection:
                    iter_best_selection_score = iter_best_fitness
                iter_best_train_loglik = _train_loglik_from_elite_tuple(
                    elite_parents[0], evolution_selection_score=evolution_selection_score
                )
                if track_elite_val_loglik and elite_val_logliks:
                    iter_best_val_loglik = elite_val_logliks[0]
                pool_pid = str(iter_best_program_id)
                matched = next(
                    (
                        r
                        for r in selected_results
                        if f"iteration_{iteration_step}_candidate_{r['idx']}" == pool_pid
                    ),
                    None,
                )
                if matched is not None:
                    iter_best_train_acc = matched.get("train_acc", iter_best_train_acc)
                    if matched.get("train_loglik") is not None:
                        iter_best_train_loglik = matched.get("train_loglik")
                    if matched.get("val_loglik") is not None:
                        iter_best_val_loglik = matched.get("val_loglik")
                person_iter_test_record = _passive_diagnostic_test_of_program(
                    dataset=dataset,
                    code=iter_best_code,
                    test_trials=test_trials,
                    n_eval_seeds=n_eval_seeds,
                    program_id=pool_pid,
                    phase="participant_evolution",
                    iteration=iteration_step,
                )
                iter_best_test_acc = person_iter_test_record.get("test_accuracy")
                iter_best_test_loglik = person_iter_test_record.get("test_loglik")
                if person_iter_test_record.get("test_loglik") is not None:
                    print(
                        f"  Diagnostic test loglik of pool-best {pool_pid}: "
                        f"{float(person_iter_test_record['test_loglik']):.6f} "
                        "(passive; not used for ranking)"
                    )

            if choice13k_simple_logging and is_binary_loglik_dataset(dataset) and save_artifacts and simple_iterations_dir is not None:
                (simple_iterations_dir / f"iteration_{iteration_step}.py").write_text(iter_best_code or "")
                _simp_row = {
                    **_participant_metric_id(participant_id),
                    "iteration": iteration_step,
                    "train_fitness": iter_best_fitness,
                    "test_fitness": (
                        iter_best_test_loglik
                        if fitness_metric == "loglik"
                        else iter_best_test_acc
                    ),
                    "train_acc": iter_best_train_acc,
                    "test_acc": iter_best_test_acc,
                    "train_loglik": iter_best_train_loglik,
                    "test_loglik": iter_best_test_loglik,
                }
                if val_trials:
                    _simp_row["val_loglik"] = iter_best_val_loglik
                simple_iterations_rows.append(_simp_row)
            best_fitness = iter_best_fitness
            
            # Update overall best tracking
            # For CPC18: compare by fitness (-MSE), for others: compare by accuracy
            if is_cpc18_mse:
                if best_result['fitness'] > overall_best_train["train_fitness"]:
                    overall_best_train = {
                        "train_fitness": best_result['fitness'],
                        "train_mse": best_result['train_mse'],
                        "test_mse": best_result['test_mse'],
                        "train_accuracy": best_result['train_acc'],
                        "test_accuracy": best_result['test_acc'],
                        "program_id": f"iteration_{iteration_step}_candidate_{best_result['idx']}"
                    }
                if best_result['test_mse'] < overall_best_test["test_mse"]:
                    overall_best_test = {
                        "train_fitness": best_result['fitness'],
                        "train_mse": best_result['train_mse'],
                        "test_mse": best_result['test_mse'],
                        "train_accuracy": best_result['train_acc'],
                        "test_accuracy": best_result['test_acc'],
                        "program_id": f"iteration_{iteration_step}_candidate_{best_result['idx']}"
                    }
            elif is_cpc18_split:
                if fitness_metric == "loglik":
                    _train_better2 = (
                        iter_best_train_loglik is not None
                        and iter_best_train_loglik > overall_best_train["train_loglik"]
                    )
                else:
                    _train_better2 = best_result["train_acc"] > overall_best_train["train_accuracy"]
                if _train_better2:
                    overall_best_train = {
                        "train_accuracy": iter_best_train_acc,
                        "test_accuracy": iter_best_test_acc,
                        "train_loglik": iter_best_train_loglik,
                        "test_loglik": iter_best_test_loglik,
                        "program_id": iter_best_program_id,
                    }
                    if val_trials and fitness_metric == "loglik":
                        overall_best_train["val_loglik"] = iter_best_val_loglik
                # Test is a passive diagnostic of the train_val-selected pool-best.
                # Do not rank or overwrite overall_best_test from held-out test.
            elif is_binary_loglik_dataset(dataset):
                if fitness_metric == "loglik":
                    _train_better = (
                        iter_best_train_loglik is not None
                        and iter_best_train_loglik > overall_best_train["train_loglik"]
                    )
                else:
                    _train_better = best_result["train_acc"] > overall_best_train["train_accuracy"]
                if _train_better:
                    overall_best_train = {
                        "train_accuracy": iter_best_train_acc,
                        "test_accuracy": iter_best_test_acc,
                        "train_loglik": iter_best_train_loglik,
                        "test_loglik": iter_best_test_loglik,
                        "program_id": iter_best_program_id,
                    }
                    if val_trials:
                        overall_best_train["val_loglik"] = iter_best_val_loglik
                # Test is a passive diagnostic of the train_val-selected pool-best.
                # Do not rank or overwrite overall_best_test from held-out test.
            else:
                if best_result['train_acc'] > overall_best_train["train_accuracy"]:
                    overall_best_train = {
                        "train_accuracy": best_result['train_acc'],
                        "test_accuracy": best_result['test_acc'],
                        "program_id": f"iteration_{iteration_step}_candidate_{best_result['idx']}"
                    }
                if best_result['test_acc'] > overall_best_test["test_accuracy"]:
                    overall_best_test = {
                        "train_accuracy": best_result['train_acc'],
                        "test_accuracy": best_result['test_acc'],
                        "program_id": f"iteration_{iteration_step}_candidate_{best_result['idx']}"
                    }
        else:
            valid_results = []
            if (is_binary_loglik_dataset(dataset)) or is_cpc18_split:
                print("\nWarning: No runtime-valid programs generated in this iteration!")
            else:
                print("\nWarning: No valid programs generated in this iteration!")
            print("Continuing with elite parents pool from previous iterations...")
            if (
                (prompt_debug or prompt_debug_on_no_valid)
                and gen_debug.get("prompt_text")
                and prompt_diag_dir is not None
            ):
                _save_prompt_debug_bundle(
                    prompt_diag_dir / "prompt_debug",
                    phase="evolution",
                    participant_id=int(participant_id) if participant_id is not None else None,
                    iteration=iteration_step,
                    prompt_text=str(gen_debug["prompt_text"]),
                    trunc_diag=dict(gen_debug.get("trunc_diag") or {}),
                    captures=list(gen_debug.get("captures") or []),
                    exit_after_save=bool(prompt_debug and prompt_debug_exit),
                )
        
        # Passive MEM candidate traces (after elite truncation when candidates were added).
        if mem_trace_file is not None:
            elite_ids_after = {str(p[3]) for p in elite_parents}
            ref_parent_id = (
                str(mem_best_parent_rec["program_id"])
                if mem_best_parent_rec is not None
                else None
            )
            ref_parent_score = (
                mem_best_parent_rec.get("selection_score")
                if mem_best_parent_rec is not None
                else None
            )
            for src_i, result in enumerate(candidate_results):
                idx = int(result.get("idx", src_i))
                program_id = f"iteration_{iteration_step}_candidate_{idx}"
                source = (
                    candidate_sources[src_i]
                    if src_i < len(candidate_sources)
                    else "normal"
                )
                if source == "fresh":
                    reference_kind = "seed_baseline"
                    cand_ref_id = "baseline"
                    cand_ref_score = mem_seed_selection_score
                    prompted_ids = ["baseline"]
                    ref_exact = True
                else:
                    reference_kind = "best_prompted_parent"
                    cand_ref_id = ref_parent_id
                    cand_ref_score = ref_parent_score
                    prompted_ids = [
                        str(p.get("program_id"))
                        for p in mem_selected_parent_records
                        if p.get("program_id") is not None
                    ]
                    ref_exact = True
                cand_score = result.get("selection_score")
                if cand_score is None and fitness_metric == "loglik":
                    # Prefer explicit selection_score; else use pool fitness when valid.
                    cand_score = result.get("fitness") if result.get("runtime_valid") else None
                    if cand_score is not None and not use_train_val_selection:
                        cand_score = result.get("train_loglik", cand_score)
                train_ll = result.get("train_loglik")
                val_ll = result.get("val_loglik") if val_trials else None
                append_mem_trace_record(
                    mem_trace_file,
                    build_candidate_record(
                        dataset=dataset,
                        participant_id=participant_id,
                        run_id=mem_run_id,
                        split_seed=int(split_seed),
                        phase="evolution",
                        iteration=iteration_step,
                        candidate_id=program_id,
                        candidate_idx=idx,
                        source=str(source),
                        code=result.get("code") or "",
                        runtime_valid=bool(result.get("runtime_valid", False)),
                        train_loglik=_safe_float(train_ll),
                        val_loglik=_safe_float(val_ll),
                        selection_score=_safe_float(cand_score),
                        reference_parent_id=cand_ref_id,
                        reference_parent_score=_safe_float(cand_ref_score),
                        reference_kind=reference_kind,
                        reference_type=reference_kind,
                        reference_id=cand_ref_id,
                        reference_is_exact=ref_exact,
                        delta_f=compute_delta_f(
                            _safe_float(cand_score), _safe_float(cand_ref_score)
                        ),
                        survived_elite_truncation=program_id in elite_ids_after,
                        evolution_selection_score=evolution_selection_score,
                        prompted_parent_ids=prompted_ids,
                    ),
                )

        # Save iteration results
        best_program_id = None
        if selected_results:
            best_program_id = iter_best_program_id
        cand_source_header = _iteration_candidate_source_header(
            fresh_n_candidates,
            fresh_n,
            n_candidates_per_iteration,
            candidate_sources,
            iter_idx=iteration,
            total_iters=n_iterations,
        )
        
        if is_cpc18_mse:
            _mse_header = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
                "best_program_id": best_program_id,
            }
            _mse_header.update(cand_source_header)
            metrics = _build_iteration_metrics_json(
                participant_id=participant_id,
                header=_mse_header,
                best={
                    "best_train_fitness": best_fitness if selected_results else None,
                    "best_train_mse": selected_results[0]["train_mse"] if selected_results else None,
                    "best_test_mse": selected_results[0]["test_mse"] if selected_results else None,
                },
                candidate_results=_annotate_candidate_results_with_sources(
                    [
                        {
                            "idx": r["idx"],
                            "train_mse": r.get("train_mse", None),
                            "test_mse": r.get("test_mse", None),
                            "fitness": r.get("fitness", None),
                            "valid": r["valid"],
                            "runtime_valid": r.get("runtime_valid", r["valid"]),
                        }
                        for r in candidate_results
                    ],
                    candidate_sources,
                ),
            )
        elif is_cpc18_split:
            _cpc_cand_rows = []
            for r in candidate_results:
                _cr = {
                    "idx": r["idx"],
                    "train_acc": r["train_acc"],
                    "test_acc": r["test_acc"],
                    "train_loglik": r.get("train_loglik"),
                    "test_loglik": r.get("test_loglik"),
                    "fitness": r.get("fitness"),
                    "valid": r["valid"],
                    "runtime_valid": r.get("runtime_valid", r["valid"]),
                }
                if val_trials:
                    _cr["val_loglik"] = r.get("val_loglik")
                _cpc_cand_rows.append(_cr)
            _cpc_best: Dict[str, Any] = {
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_acc": iter_best_train_acc if selected_results else None,
                "best_test_acc": iter_best_test_acc if selected_results else None,
                "best_train_loglik": iter_best_train_loglik if selected_results else None,
                "best_test_loglik": iter_best_test_loglik if selected_results else None,
            }
            if val_trials:
                _cpc_best["best_val_loglik"] = iter_best_val_loglik if selected_results else None
            _cpc_header = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
                "best_program_id": best_program_id,
            }
            _cpc_header.update(cand_source_header)
            metrics = _build_iteration_metrics_json(
                participant_id=participant_id,
                header=_cpc_header,
                best=_cpc_best,
                candidate_results=_annotate_candidate_results_with_sources(
                    _cpc_cand_rows, candidate_sources
                ),
            )
        elif is_binary_loglik_dataset(dataset):
            _cand_rows = []
            for r in candidate_results:
                _cr = {
                    "idx": r["idx"],
                    "train_acc": r["train_acc"],
                    "test_acc": r["test_acc"],
                    "train_loglik": r.get("train_loglik"),
                    "test_loglik": r.get("test_loglik"),
                    "fitness": r.get("fitness"),
                    "valid": r["valid"],
                    "runtime_valid": r.get("runtime_valid", r["valid"]),
                }
                if val_trials:
                    _cr["val_loglik"] = r.get("val_loglik")
                if r.get("selection_score") is not None:
                    _cr["selection_score"] = r.get("selection_score")
                _cr.update(_candidate_mdl_log_fields(r))
                _cand_rows.append(_cr)
            _loglik_best: Dict[str, Any] = {
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_acc": iter_best_train_acc if selected_results else None,
                "best_test_acc": iter_best_test_acc if selected_results else None,
                "best_train_loglik": iter_best_train_loglik if selected_results else None,
                "best_test_loglik": iter_best_test_loglik if selected_results else None,
            }
            if val_trials:
                _loglik_best["best_val_loglik"] = iter_best_val_loglik if selected_results else None
            if iter_best_selection_score is not None and selected_results:
                _loglik_best["best_selection_score"] = iter_best_selection_score
            _loglik_header = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
                "best_program_id": best_program_id,
                "evolution_selection_score": evolution_selection_score,
            }
            if mdl_enabled(mdl_lambda):
                _loglik_header["mdl_lambda"] = float(mdl_lambda)
                if elite_parents and len(elite_parents[0]) > 7 and elite_parents[0][7] is not None:
                    _loglik_header["pool_best_mdl_score"] = float(elite_parents[0][7])
                if selected_results:
                    _loglik_best.update(_candidate_mdl_log_fields(selected_results[0]))
            _loglik_header.update(cand_source_header)
            metrics = _build_iteration_metrics_json(
                participant_id=participant_id,
                header=_loglik_header,
                best=_loglik_best,
                candidate_results=_annotate_candidate_results_with_sources(
                    _cand_rows, candidate_sources
                ),
            )
        else:
            _acc_header = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
                "best_program_id": best_program_id,
            }
            _acc_header.update(cand_source_header)
            metrics = _build_iteration_metrics_json(
                participant_id=participant_id,
                header=_acc_header,
                best={
                    "best_train_acc": best_fitness if selected_results else None,
                    "best_test_acc": selected_results[0]["test_acc"] if selected_results else None,
                },
                candidate_results=_annotate_candidate_results_with_sources(
                    [
                        {
                            "idx": r["idx"],
                            "train_acc": r["train_acc"],
                            "test_acc": r["test_acc"],
                            "valid": r["valid"],
                            "runtime_valid": r.get("runtime_valid", r["valid"]),
                        }
                        for r in candidate_results
                    ],
                    candidate_sources,
                ),
            )
        metrics["num_invalid_candidates"] = num_invalid_candidates
        metrics["num_unique_errors_available"] = num_unique_errors_available
        metrics["error_prompt_chars_used"] = error_prompt_chars_used
        if person_iter_test_record is not None:
            metrics["diagnostic_test"] = person_iter_test_record
        if save_artifacts and iter_dir is not None:
            (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        
        # Save summary
        if is_cpc18_mse:
            summary = {
                "iteration": iteration_step,
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_mse": selected_results[0]["train_mse"] if selected_results else None,
                "best_test_mse": selected_results[0]["test_mse"] if selected_results else None,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
            }
        elif is_cpc18_split:
            summary = {
                "iteration": iteration_step,
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_acc": iter_best_train_acc if selected_results else None,
                "best_test_acc": iter_best_test_acc if selected_results else None,
                "best_train_loglik": iter_best_train_loglik if selected_results else None,
                "best_test_loglik": iter_best_test_loglik if selected_results else None,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
            }
            if val_trials:
                summary["best_val_loglik"] = iter_best_val_loglik if selected_results else None
            if iter_best_selection_score is not None:
                summary["best_selection_score"] = (
                    iter_best_selection_score if selected_results else None
                )
            summary["evolution_selection_score"] = evolution_selection_score
        elif is_binary_loglik_dataset(dataset):
            summary = {
                "iteration": iteration_step,
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_acc": iter_best_train_acc if selected_results else None,
                "best_test_acc": iter_best_test_acc if selected_results else None,
                "best_train_loglik": iter_best_train_loglik if selected_results else None,
                "best_test_loglik": iter_best_test_loglik if selected_results else None,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
                "evolution_selection_score": evolution_selection_score,
            }
            if val_trials:
                summary["best_val_loglik"] = iter_best_val_loglik if selected_results else None
            if iter_best_selection_score is not None:
                summary["best_selection_score"] = (
                    iter_best_selection_score if selected_results else None
                )
        else:
            summary = {
                "iteration": iteration_step,
                "best_train_acc": best_fitness if selected_results else None,
                "best_test_acc": selected_results[0]["test_acc"] if selected_results else None,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
            }
        summary.update(_participant_metric_id(participant_id))
        print(f"\nSummary: {json.dumps(summary, indent=2)}")
        
        # Log to wandb (use dataset-specific metric names)
        if wandb is not None:
            if dataset == "gridworld":
                # Use agent-specific keys if agent_id is provided
                if agent_id is not None:
                    log_dict = {
                        f"a{agent_id}_n_valid": len(valid_results),
                    }
                    if valid_results:
                        log_dict[f"a{agent_id}_train_accuracy"] = best_fitness
                        log_dict[f"a{agent_id}_test_accuracy"] = valid_results[0]["test_acc"]
                        log_dict[f"a{agent_id}_avg_train_accuracy"] = np.mean([r["train_acc"] for r in valid_results])
                        log_dict[f"a{agent_id}_avg_test_accuracy"] = np.mean([r["test_acc"] for r in valid_results])
                else:
                    log_dict = {
                        f"gw_n_valid": len(valid_results),
                    }
                    if valid_results:
                        log_dict[f"gw_train_accuracy"] = best_fitness
                        log_dict[f"gw_test_accuracy"] = valid_results[0]["test_acc"]
                        log_dict[f"gw_avg_train_accuracy"] = np.mean([r["train_acc"] for r in valid_results])
                        log_dict[f"gw_avg_test_accuracy"] = np.mean([r["test_acc"] for r in valid_results])
            elif is_cpc18_mse:
                if all_data_mode:
                    log_dict = {}
                    if valid_results:
                        log_dict[f"p{participant_id}_train_fitness"] = best_fitness
                        log_dict[f"p{participant_id}_test_fitness"] = -valid_results[0].get("test_mse", float("inf"))
                else:
                    log_dict = {f"p{participant_id}_n_valid": len(valid_results)}
                    if valid_results:
                        log_dict[f"p{participant_id}_train_fitness"] = best_fitness
                        log_dict[f"p{participant_id}_train_mse"] = valid_results[0].get("train_mse", None)
                        log_dict[f"p{participant_id}_test_mse"] = valid_results[0].get("test_mse", None)
                        log_dict[f"p{participant_id}_avg_train_fitness"] = np.mean([r["fitness"] for r in valid_results])
                        log_dict[f"p{participant_id}_avg_train_mse"] = np.mean(
                            [r.get("train_mse", float("inf")) for r in valid_results]
                        )
                        log_dict[f"p{participant_id}_avg_test_mse"] = np.mean(
                            [r.get("test_mse", float("inf")) for r in valid_results]
                        )
                        log_dict[f"p{participant_id}_train_accuracy"] = valid_results[0].get("train_acc", None)
                        log_dict[f"p{participant_id}_test_accuracy"] = valid_results[0].get("test_acc", None)
            elif is_cpc18_split:
                if all_data_mode:
                    log_dict = {}
                    if valid_results:
                        log_dict[f"p{participant_id}_train_fitness"] = best_fitness
                        log_dict[f"p{participant_id}_test_fitness"] = (
                            iter_best_test_loglik
                            if fitness_metric == "loglik"
                            else iter_best_test_acc
                        )
                        log_dict[f"p{participant_id}_train_acc"] = iter_best_train_acc
                        log_dict[f"p{participant_id}_test_acc"] = iter_best_test_acc
                        log_dict[f"p{participant_id}_train_loglik"] = iter_best_train_loglik
                        log_dict[f"p{participant_id}_test_loglik"] = iter_best_test_loglik
                        if val_trials and iter_best_val_loglik is not None:
                            log_dict[f"p{participant_id}_val_loglik"] = iter_best_val_loglik
                            log_dict["val_loglik"] = iter_best_val_loglik
                else:
                    log_dict = {f"p{participant_id}_n_valid": len(valid_results)}
                    if valid_results:
                        log_dict[f"p{participant_id}_train_accuracy"] = iter_best_train_acc
                        log_dict[f"p{participant_id}_test_accuracy"] = iter_best_test_acc
                        log_dict[f"p{participant_id}_train_acc"] = iter_best_train_acc
                        log_dict[f"p{participant_id}_test_acc"] = iter_best_test_acc
                        log_dict[f"p{participant_id}_train_loglik"] = iter_best_train_loglik
                        log_dict[f"p{participant_id}_test_loglik"] = iter_best_test_loglik
                        log_dict[f"p{participant_id}_train_fitness"] = best_fitness
                        log_dict[f"p{participant_id}_test_fitness"] = (
                            iter_best_test_loglik
                            if fitness_metric == "loglik"
                            else iter_best_test_acc
                        )
                        if val_trials and iter_best_val_loglik is not None:
                            log_dict[f"p{participant_id}_val_loglik"] = iter_best_val_loglik
                            log_dict["val_loglik"] = iter_best_val_loglik
                        log_dict[f"p{participant_id}_avg_train_accuracy"] = np.mean(
                            [r["train_acc"] for r in valid_results]
                        )
                        if fitness_metric != "loglik":
                            log_dict[f"p{participant_id}_avg_test_accuracy"] = np.mean(
                                [r["test_acc"] for r in valid_results]
                            )
            elif is_binary_loglik_dataset(dataset):
                if all_data_mode:
                    log_dict = {}
                    if valid_results:
                        if fitness_metric == "loglik":
                            log_dict[f"p{participant_id}_train_fitness"] = best_fitness
                            log_dict[f"p{participant_id}_test_fitness"] = iter_best_test_loglik
                        else:
                            log_dict[f"p{participant_id}_train_fitness"] = best_fitness
                            log_dict[f"p{participant_id}_test_fitness"] = iter_best_test_acc
                        log_dict[f"p{participant_id}_train_acc"] = iter_best_train_acc
                        log_dict[f"p{participant_id}_test_acc"] = iter_best_test_acc
                        log_dict[f"p{participant_id}_train_loglik"] = iter_best_train_loglik
                        log_dict[f"p{participant_id}_test_loglik"] = iter_best_test_loglik
                        if val_trials and iter_best_val_loglik is not None:
                            log_dict[f"p{participant_id}_val_loglik"] = iter_best_val_loglik
                            log_dict["val_loglik"] = iter_best_val_loglik
                        if fitness_metric == "loglik" and iter_best_train_loglik is not None:
                            log_dict[f"p{participant_id}_selection_score"] = (
                                iter_best_selection_score
                                if iter_best_selection_score is not None
                                else iter_best_train_loglik
                            )
                else:
                    log_dict = {
                        f"p{participant_id}_n_valid": len(valid_results),
                    }
                    if valid_results:
                        log_dict[f"p{participant_id}_train_accuracy"] = iter_best_train_acc
                        log_dict[f"p{participant_id}_test_accuracy"] = iter_best_test_acc
                        log_dict[f"p{participant_id}_train_acc"] = iter_best_train_acc
                        log_dict[f"p{participant_id}_test_acc"] = iter_best_test_acc
                        log_dict[f"p{participant_id}_train_loglik"] = iter_best_train_loglik
                        log_dict[f"p{participant_id}_test_loglik"] = iter_best_test_loglik
                        log_dict[f"p{participant_id}_train_fitness"] = best_fitness
                        log_dict[f"p{participant_id}_test_fitness"] = (
                            iter_best_test_loglik
                            if fitness_metric == "loglik"
                            else iter_best_test_acc
                        )
                        if val_trials and iter_best_val_loglik is not None:
                            log_dict[f"p{participant_id}_val_loglik"] = iter_best_val_loglik
                            log_dict["val_loglik"] = iter_best_val_loglik
                        if fitness_metric == "loglik" and iter_best_train_loglik is not None:
                            log_dict[f"p{participant_id}_selection_score"] = (
                                iter_best_selection_score
                                if iter_best_selection_score is not None
                                else iter_best_train_loglik
                            )
                        log_dict[f"p{participant_id}_avg_train_accuracy"] = np.mean([r["train_acc"] for r in valid_results])
                        if fitness_metric != "loglik":
                            log_dict[f"p{participant_id}_avg_test_accuracy"] = np.mean([r["test_acc"] for r in valid_results])
            else:
                if all_data_mode:
                    log_dict = {}
                    if valid_results:
                        log_dict[f"p{participant_id}_train_fitness"] = best_fitness
                        log_dict[f"p{participant_id}_test_fitness"] = valid_results[0]["test_acc"]
                else:
                    log_dict = {
                        f"p{participant_id}_n_valid": len(valid_results),
                    }
                    if valid_results:
                        log_dict[f"p{participant_id}_train_accuracy"] = best_fitness
                        log_dict[f"p{participant_id}_test_accuracy"] = valid_results[0]["test_acc"]
                        log_dict[f"p{participant_id}_avg_train_accuracy"] = np.mean([r["train_acc"] for r in valid_results])
                        log_dict[f"p{participant_id}_avg_test_accuracy"] = np.mean([r["test_acc"] for r in valid_results])
            if participant_id is not None:
                _wandb_log_participant_metrics(wandb, log_dict, int(participant_id), iteration + 1)
            else:
                wandb.log(log_dict, step=iteration + 1)
            
            # Also save to local JSONL file
            if save_artifacts and log_file_path is not None:
                log_dict["best_from_fresh_candidate"] = metrics.get(
                    "best_from_fresh_candidate"
                )
                log_entry = _wandb_jsonl_log_entry(
                    step=iteration + 1,
                    iteration=iteration_step,
                    log_dict=log_dict,
                    participant_id=participant_id,
                    agent_id=agent_id,
                )
                with open(log_file_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

        if early_stop_patience is not None:
            pool_best_fitness = float(elite_parents[0][1])
            improvement = pool_best_fitness - float(last_significant_best)
            if improvement >= _EARLY_STOP_MIN_IMPROVEMENT:
                last_significant_best = pool_best_fitness
                stagnant_iters = 0
            else:
                stagnant_iters += 1
                if stagnant_iters >= early_stop_patience:
                    print(
                        f"Early stopping evolution at iteration {iteration_step}: "
                        f"pool best fitness improved by < {_EARLY_STOP_MIN_IMPROVEMENT:.3f} "
                        f"for {stagnant_iters} consecutive iteration(s)."
                    )
                    break
    
    # Final summary and save comprehensive results.json
    print(f"\n{'='*80}")
    print("Evolution Complete")
    print(f"{'='*80}")

    # Select final best program directly from the final elite pool (already sorted by train fitness).
    # This guarantees final reporting is paired from one candidate.
    final_best_code, final_best_program_id = _resolve_final_best_from_elite_pool(
        elite_parents, seed_code
    )
    origin = _parse_final_best_program_origin(final_best_program_id)
    best_iteration = origin["origin_iteration"]
    best_candidate_idx = origin["origin_candidate_idx"]
    origin_phase = origin["origin_phase"]
    best_program_filename = BEST_PROGRAM_FILENAME

    if save_artifacts:
        (output_path / best_program_filename).write_text(final_best_code or "")

    final_best_fn = compile_program(final_best_code)
    if final_best_fn is None:
        raise RuntimeError(
            f"Final best program failed to compile: {final_best_program_id}"
        )

    if is_cpc18_mse:
        final_train_eval = evaluate_cpc18_program(final_best_fn, train_trials, n_seeds=n_eval_seeds)
        final_test_eval = evaluate_cpc18_program(final_best_fn, test_trials, n_seeds=n_eval_seeds)
        train_observed_blocks = test_observed_blocks
        final_train_mse_eval = evaluate_cpc18_mse(
            final_best_fn, train_trials, train_observed_blocks, n_seeds=n_eval_seeds
        )
        final_test_mse_eval = evaluate_cpc18_mse(
            final_best_fn, test_trials, test_observed_blocks, n_seeds=n_eval_seeds
        )
        overall_best_train = {
            "train_fitness": -final_train_mse_eval["mse"],
            "train_mse": final_train_mse_eval["mse"],
            "test_mse": final_test_mse_eval["mse"],
            "train_accuracy": final_train_eval["accuracy"],
            "test_accuracy": final_test_eval["accuracy"],
            "program_id": final_best_program_id,
            "origin_iteration": best_iteration,
            "origin_candidate_idx": best_candidate_idx,
            "origin_phase": origin_phase,
            "program_file": best_program_filename,
        }
        # Keep paired train/test metrics from the same best-train program.
        overall_best_test = dict(overall_best_train)
    elif is_cpc18_split and fitness_metric != "loglik":
        final_train_eval = evaluate_cpc18_split_program(final_best_fn, train_trials, n_seeds=n_eval_seeds)
        final_test_eval = evaluate_cpc18_split_program(final_best_fn, test_trials, n_seeds=n_eval_seeds)
        overall_best_train = {
            "train_accuracy": final_train_eval["accuracy"],
            "test_accuracy": final_test_eval["accuracy"],
            "train_loglik": final_train_eval["avg_loglik"],
            "test_loglik": final_test_eval["avg_loglik"],
            "program_id": final_best_program_id,
            "origin_iteration": best_iteration,
            "origin_candidate_idx": best_candidate_idx,
            "origin_phase": origin_phase,
            "program_file": best_program_filename,
        }
        overall_best_test = dict(overall_best_train)
    elif _uses_train_val_test_loglik_split(
        dataset, fitness_metric, cpc18_official_mse=is_cpc18_mse
    ) or (dataset == "mixed_gambles" and fitness_metric == "loglik"):
        final_train_eval = _evaluate_loglik_for_dataset(
            dataset, final_best_fn, train_trials, n_seeds=n_eval_seeds
        )
        final_test_eval = _evaluate_loglik_for_dataset(
            dataset, final_best_fn, test_trials, n_seeds=n_eval_seeds
        )
        final_val_eval = (
            _evaluate_loglik_for_dataset(
                dataset, final_best_fn, val_trials, n_seeds=n_eval_seeds
            )
            if val_trials
            else None
        )
        overall_best_train = {
            "train_accuracy": final_train_eval["accuracy"],
            "test_accuracy": final_test_eval["accuracy"],
            "train_loglik": final_train_eval["avg_loglik"],
            "test_loglik": final_test_eval["avg_loglik"],
            "program_id": final_best_program_id,
            "origin_iteration": best_iteration,
            "origin_candidate_idx": best_candidate_idx,
            "origin_phase": origin_phase,
            "program_file": best_program_filename,
        }
        if final_val_eval is not None:
            overall_best_train["val_loglik"] = final_val_eval["avg_loglik"]
        if fitness_metric == "loglik":
            overall_best_train["selection_score"] = _evolution_selection_score(
                float(overall_best_train["train_loglik"]),
                _safe_float(overall_best_train.get("val_loglik")),
                len(train_trials),
                len(val_trials),
                evolution_selection_score=evolution_selection_score,
                warn_key=None,
            )
            overall_best_train["evolution_selection_score"] = evolution_selection_score
        overall_best_test = dict(overall_best_train)
    else:
        final_train_eval = evaluate_program(final_best_fn, train_trials, n_seeds=n_eval_seeds)
        final_test_eval = evaluate_program(final_best_fn, test_trials, n_seeds=n_eval_seeds)
        overall_best_train = {
            "train_accuracy": final_train_eval["accuracy"],
            "test_accuracy": final_test_eval["accuracy"],
            "program_id": final_best_program_id,
            "origin_iteration": best_iteration,
            "origin_candidate_idx": best_candidate_idx,
            "origin_phase": origin_phase,
            "program_file": best_program_filename,
        }
        overall_best_test = dict(overall_best_train)

    if save_artifacts and track_elite_val_loglik and output_path is not None:
        pool_dir = _save_evolution_elite_pool(
            output_path,
            elite_parents,
            elite_val_logliks,
            split_ratio=split_ratio,
            n_train=len(train_trials),
            n_val=len(val_trials),
            evolution_selection_score=evolution_selection_score,
            mdl_lambda=mdl_lambda,
        )
        print(
            f"Saved evolution elite pool ({len(elite_parents)} programs) -> {pool_dir}"
        )

    gated_test_loglik: Optional[float] = None
    refinement_ran = False
    final_val_loglik = overall_best_train.get("val_loglik")
    if (
        run_phase == "all"
        and refinement_phase
        and _supports_loglik_refinement(
            dataset,
            fitness_metric,
            val_trials,
            test_trials,
            cpc18_official_mse=is_cpc18_mse,
        )
        and _val_loglik_below_refinement_threshold(
            final_val_loglik, refinement_val_threshold
        )
    ):
        print(
            f"\nRefinement triggered: val_loglik={float(final_val_loglik):.6f} "
            f"< threshold={float(refinement_val_threshold):.6f}"
        )
        ref_parents, ref_vals = _evolution_elite_to_refinement_pool(
            elite_parents,
            elite_val_logliks,
            split_ratio=split_ratio,
            evolution_selection_score=evolution_selection_score,
            mdl_lambda=mdl_lambda,
            n_train=len(train_trials),
            n_val=len(val_trials),
        )
        _fresh_val_ll = (
            float(baseline_val_eval["avg_loglik"])
            if baseline_val_eval is not None
            else None
        )
        gated_test_loglik = run_loglik_refinement_phase(
            dataset=dataset,
            client=client,
            model_name=model_name,
            train_trials=train_trials,
            val_trials=val_trials,
            test_trials=test_trials,
            n_iterations=refinement_iters,
            n_candidates_per_iteration=n_candidates_per_iteration,
            fresh_n_candidates=fresh_n_candidates,
            sample_size=sample_size,
            sample_parents=sample_parents,
            sampled_parents_decay=sampled_parents_decay,
            elite_pool_size=elite_pool_size,
            participant_id=int(participant_id),
            split_ratio=float(split_ratio),
            split_seed=int(split_seed),
            max_prompt_train_trials=max_prompt_train_trials,
            max_prompt_trials_per_problem=max_prompt_trials_per_problem,
            llm_max_tokens=llm_max_tokens,
            max_workers=max_workers,
            n_eval_seeds=n_eval_seeds,
            fitness_metric=fitness_metric,
            output_path=output_path if save_artifacts else None,
            save_artifacts=save_artifacts,
            wandb_module=wandb,
            wandb_step_offset=int(n_iterations),
            evolution_elite_parents=ref_parents,
            evolution_elite_val_logliks=ref_vals,
            fresh_parent_code=seed_code,
            fresh_parent_train_loglik=float(baseline_train_eval["avg_loglik"]),
            fresh_parent_val_loglik=_fresh_val_ll,
            run_prompts_dir=run_prompts_dir,
            max_parent_chars=max_parent_chars,
            warn_parent_truncation_ratio=warn_parent_truncation_ratio,
            early_stop_iters=early_stop_iters,
            hard_prompt_token_cap=hard_prompt_token_cap,
            strict_prompt_budget=strict_prompt_budget,
            prompt_token_estimator=prompt_token_estimator,
            max_error_prompt_chars=max_error_prompt_chars,
            error_feedback_mode=error_feedback_mode,
            mdl_lambda=mdl_lambda,
        )
        refinement_ran = gated_test_loglik is not None
        if gated_test_loglik is not None:
            overall_best_train["gated_test_loglik"] = gated_test_loglik
            overall_best_test["gated_test_loglik"] = gated_test_loglik
            if wandb is not None and participant_id is not None:
                # Align gated_test_loglik with the last evolution test_loglik step (not
                # n_iterations + refinement_iters, which used the global config size).
                final_evolution_step = int(n_iterations)
                refine_log = {
                    f"p{participant_id}_gated_test_loglik": gated_test_loglik,
                    f"p{participant_id}_train_loglik": overall_best_train.get("train_loglik"),
                    f"p{participant_id}_val_loglik": overall_best_train.get("val_loglik"),
                    f"p{participant_id}_test_loglik": overall_best_train.get("test_loglik"),
                    f"p{participant_id}_train_fitness": overall_best_train.get("train_loglik"),
                    f"p{participant_id}_test_fitness": overall_best_test.get("test_loglik"),
                }
                _wandb_log_participant_metrics(
                    wandb, refine_log, int(participant_id), final_evolution_step
                )
    elif (
        run_phase == "all"
        and refinement_phase
        and _supports_loglik_refinement(
            dataset,
            fitness_metric,
            val_trials,
            test_trials,
            cpc18_official_mse=is_cpc18_mse,
        )
        and final_val_loglik is not None
        and not _val_loglik_below_refinement_threshold(
            final_val_loglik, refinement_val_threshold
        )
    ):
        print(
            f"\nRefinement skipped: val_loglik={float(final_val_loglik):.6f} "
            f">= threshold={float(refinement_val_threshold):.6f}"
        )
    elif (
        run_phase == "all"
        and refinement_phase
        and _supports_loglik_refinement(
            dataset,
            fitness_metric,
            val_trials,
            test_trials,
            cpc18_official_mse=is_cpc18_mse,
        )
        and final_val_loglik is None
    ):
        print(
            "\nRefinement skipped: val_loglik unavailable "
            f"(val_trials={len(val_trials)}, test_trials={len(test_trials)})"
        )

    if (
        gate_phase
        and not refinement_ran
        and is_binary_loglik_dataset(dataset)
        and val_trials
        and test_trials
    ):
        val_ll_for_gate = overall_best_train.get("val_loglik")
        if val_ll_for_gate is not None:
            gated_test_loglik = run_choice13k_gate_phase(
                final_best_fn,
                float(val_ll_for_gate),
                test_trials,
                n_eval_seeds=n_eval_seeds,
                dataset=dataset,
            )
            if gated_test_loglik is not None:
                overall_best_train["gated_test_loglik"] = gated_test_loglik
                overall_best_test["gated_test_loglik"] = gated_test_loglik
                print(f"Gated test avg log-likelihood: {gated_test_loglik:.6f}")
                if wandb is not None and participant_id is not None:
                    gate_log = {
                        f"p{participant_id}_gated_test_loglik": gated_test_loglik,
                        f"p{participant_id}_val_loglik": overall_best_train.get("val_loglik"),
                        f"p{participant_id}_test_loglik": overall_best_test.get("test_loglik"),
                    }
                    _wandb_log_participant_metrics(
                        wandb, gate_log, int(participant_id), int(n_iterations)
                    )

    gated_test_loglik = _apply_test_loglik_as_gated_when_no_refinement(
        gated_test_loglik=gated_test_loglik,
        overall_best_train=overall_best_train,
        overall_best_test=overall_best_test,
        dataset=dataset,
        fitness_metric=fitness_metric,
        run_phase=run_phase,
        refinement_phase=refinement_phase,
        val_trials=val_trials,
        test_trials=test_trials,
        cpc18_official_mse=is_cpc18_mse,
    )

    if is_cpc18_mse or is_cpc18_split:
        results = {
            "baseline": baseline_results,
            "overall_best_train": overall_best_train,
            "overall_best_test": overall_best_test,
        }
    else:
        results = {
            "baseline": baseline_results,
            "overall_best_train": overall_best_train,
            "overall_best_test": overall_best_test,
        }
    if save_artifacts and not choice13k_simple_logging:
        (output_path / "results.json").write_text(json.dumps(results, indent=2))
    if save_artifacts and choice13k_simple_logging and is_binary_loglik_dataset(dataset):
        if simple_iterations_rows:
            with open(output_path / "iterations.csv", "w", newline="") as f:
                _iter_fields = [
                    "participant_id",
                    "iteration",
                    "train_fitness",
                    "test_fitness",
                    "train_acc",
                    "test_acc",
                    "train_loglik",
                    "test_loglik",
                ]
                if val_trials:
                    _iter_fields.append("val_loglik")
                writer = csv.DictWriter(f, fieldnames=_iter_fields)
                writer.writeheader()
                writer.writerows(simple_iterations_rows)
        summary_row = {
            **_participant_metric_id(participant_id),
            "train_fitness": (
                overall_best_train.get("train_loglik")
                if fitness_metric == "loglik"
                else overall_best_train.get("train_accuracy")
            ),
            "test_fitness": (
                overall_best_test.get("test_loglik")
                if fitness_metric == "loglik"
                else overall_best_test.get("test_accuracy")
            ),
            "train_acc": overall_best_train.get("train_accuracy"),
            "test_acc": overall_best_test.get("test_accuracy"),
            "train_loglik": overall_best_train.get("train_loglik"),
            "test_loglik": overall_best_test.get("test_loglik"),
            "fitness_metric": fitness_metric,
        }
        if val_trials:
            summary_row["val_loglik"] = overall_best_train.get("val_loglik")
        if gated_test_loglik is not None:
            summary_row["gated_test_loglik"] = gated_test_loglik
        with open(output_path / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(_round_floats_for_csv_row(summary_row))
    
    if n_iterations > 0:
        if is_cpc18_mse:
            print(f"Final best train MSE: {overall_best_train['train_mse']:.2f} (fitness={overall_best_train['train_fitness']:.2f}) (from {overall_best_train['program_id']})")
            print(f"Final best test MSE: {overall_best_test['test_mse']:.2f} (from {overall_best_test['program_id']})")
            print(f"Baseline train MSE: {baseline_results['train_mse']:.4f}")
            print(f"Baseline test MSE (official): {baseline_results['test_mse']:.4f}")
            print(f"Train MSE improvement: {baseline_results['train_mse'] - overall_best_train['train_mse']:.4f}")
            print(f"Test MSE improvement: {baseline_results['test_mse'] - overall_best_test['test_mse']:.4f}")
        elif is_cpc18_split:
            print(
                f"Final best train accuracy: {overall_best_train['train_accuracy']:.4f} "
                f"(from {overall_best_train['program_id']})"
            )
            print(
                f"Final best test accuracy: {overall_best_test['test_accuracy']:.4f} "
                f"(from {overall_best_test['program_id']})"
            )
            print(
                f"Final best train avg log-likelihood: {overall_best_train['train_loglik']:.6f}, "
                f"test: {overall_best_test['test_loglik']:.6f}"
            )
            print(f"Baseline train accuracy: {baseline_train_eval['accuracy']:.4f}")
            print(
                f"Train accuracy improvement: "
                f"{overall_best_train['train_accuracy'] - baseline_train_eval['accuracy']:.4f}"
            )
            print(
                f"Test accuracy improvement: "
                f"{overall_best_test['test_accuracy'] - baseline_test_eval['accuracy']:.4f}"
            )
            print(
                f"Train avg log-likelihood improvement: "
                f"{overall_best_train['train_loglik'] - baseline_train_eval['avg_loglik']:.6f}"
            )
            print(
                f"Test avg log-likelihood improvement: "
                f"{overall_best_test['test_loglik'] - baseline_test_eval['avg_loglik']:.6f}"
            )
        elif _uses_train_val_test_loglik_split(
            dataset, fitness_metric, cpc18_official_mse=is_cpc18_mse
        ):
            print(f"Final best train accuracy: {overall_best_train['train_accuracy']:.4f} (from {overall_best_train['program_id']})")
            print(f"Final best test accuracy: {overall_best_test['test_accuracy']:.4f} (from {overall_best_test['program_id']})")
            print(
                f"Final best train avg log-likelihood: {overall_best_train['train_loglik']:.6f}, "
                f"test: {overall_best_test['test_loglik']:.6f}"
            )
            if overall_best_train.get("val_loglik") is not None:
                print(f"Final best val avg log-likelihood: {overall_best_train['val_loglik']:.6f}")
            if gated_test_loglik is not None:
                print(f"Final gated test avg log-likelihood: {gated_test_loglik:.6f}")
            print(f"Baseline train accuracy: {baseline_train_eval['accuracy']:.4f}")
            print(f"Train accuracy improvement: {overall_best_train['train_accuracy'] - baseline_train_eval['accuracy']:.4f}")
            print(f"Test accuracy improvement: {overall_best_test['test_accuracy'] - baseline_test_eval['accuracy']:.4f}")
            print(
                f"Train avg log-likelihood improvement: "
                f"{overall_best_train['train_loglik'] - baseline_train_eval['avg_loglik']:.6f}"
            )
            print(
                f"Test avg log-likelihood improvement: "
                f"{overall_best_test['test_loglik'] - baseline_test_eval['avg_loglik']:.6f}"
            )
        else:
            print(f"Final best train accuracy: {overall_best_train['train_accuracy']:.4f} (from {overall_best_train['program_id']})")
            print(f"Final best test accuracy: {overall_best_test['test_accuracy']:.4f} (from {overall_best_test['program_id']})")
            print(f"Baseline train accuracy: {baseline_train_eval['accuracy']:.4f}")
            print(f"Train accuracy improvement: {overall_best_train['train_accuracy'] - baseline_train_eval['accuracy']:.4f}")
            print(f"Test accuracy improvement: {overall_best_test['test_accuracy'] - baseline_test_eval['accuracy']:.4f}")
    if save_artifacts:
        print(f"\nResults saved to: {output_path / 'results.json'}")
    
    if is_cpc18_mse:
        result = {
            "participant_id": participant_id,
            "train_mse": overall_best_train['train_mse'],
            "test_mse": overall_best_test['test_mse'],
            "train_fitness": overall_best_train['train_fitness'],
            "test_fitness": -overall_best_test['test_mse'],
            "seed_program_train_fitness": -baseline_results['train_mse'],
            "seed_program_test_fitness": -baseline_results['test_mse'],
        }
    elif is_cpc18_split and fitness_metric != "loglik":
        result = {
            "participant_id": participant_id,
            "train_acc": overall_best_train["train_accuracy"],
            "test_acc": overall_best_test["test_accuracy"],
            "train_loglik": overall_best_train["train_loglik"],
            "test_loglik": overall_best_test["test_loglik"],
            "train_fitness": overall_best_train["train_accuracy"],
            "test_fitness": overall_best_test["test_accuracy"],
            "seed_program_train_fitness": baseline_train_eval["accuracy"],
            "seed_program_test_fitness": baseline_test_eval["accuracy"],
        }
    elif _uses_train_val_test_loglik_split(
        dataset, fitness_metric, cpc18_official_mse=is_cpc18_mse
    ):
        result = {
            "participant_id": participant_id,
            "train_acc": overall_best_train["train_accuracy"],
            "test_acc": overall_best_test["test_accuracy"],
            "train_loglik": overall_best_train["train_loglik"],
            "test_loglik": overall_best_test["test_loglik"],
            "train_fitness": (
                overall_best_train["train_loglik"]
                if fitness_metric == "loglik"
                else overall_best_train["train_accuracy"]
            ),
            "test_fitness": (
                overall_best_test["test_loglik"]
                if fitness_metric == "loglik"
                else overall_best_test["test_accuracy"]
            ),
            "seed_program_train_fitness": (
                baseline_results["train_loglik"]
                if fitness_metric == "loglik"
                else baseline_results["train_accuracy"]
            ),
            "seed_program_test_fitness": (
                baseline_results["test_loglik"]
                if fitness_metric == "loglik"
                else baseline_results["test_accuracy"]
            ),
        }
        if val_trials:
            result["val_loglik"] = overall_best_train.get("val_loglik")
            result["seed_program_val_fitness"] = baseline_results.get("val_loglik")
        if overall_best_train.get("selection_score") is not None:
            result["selection_score"] = overall_best_train.get("selection_score")
        if overall_best_train.get("evolution_selection_score") is not None:
            result["evolution_selection_score"] = overall_best_train.get(
                "evolution_selection_score"
            )
        if gated_test_loglik is not None:
            result["gated_test_loglik"] = gated_test_loglik
    else:
        result = {
            "participant_id": participant_id if dataset in ["choice13k", "cpc18", "mixed_gambles"] else agent_id,
            "train_acc": overall_best_train['train_accuracy'],
            "test_acc": overall_best_test['test_accuracy'],
            "train_fitness": overall_best_train['train_accuracy'],
            "test_fitness": overall_best_test['test_accuracy'],
            "seed_program_train_fitness": baseline_results['train_accuracy'],
            "seed_program_test_fitness": baseline_results['test_accuracy'],
        }
    return result


def run_evolution_gridworld_ensemble(
    seed_program_path: str,
    participant_id: int = 0,
    data_path: str = "data",
    num_blocks: Optional[int] = None,
    num_walls: Optional[int] = None,
    agent_id: Optional[int] = None,
    n_iterations: int = 5,
    n_candidates_per_iteration: int = 10,
    model_name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    client_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    wandb=None,
    n_eval_seeds: int = 3,
    sample_size: int = 3,
    top_k: int = 0,
    max_workers: int = 5,
    ablation: Optional[str] = None,
):
    """
    Run gridworld evolution with K independent ensemble members; test = ROTE-aligned weighted ensemble.
    Hypothesis selection: first K programs (by fitness). If top_k > 0 and top_k < K, use top_k by weight.
    Weights from first-20-step log-likelihood; tie-aware accuracy; teacher-forced states.
    """
    if num_blocks is None or num_walls is None or agent_id is None:
        raise ValueError("For gridworld_ensemble, num_blocks, num_walls, and agent_id must be provided")
    K = sample_size  # ensemble size (n_hyp)
    print(f"Gridworld ensemble mode: K={K} programs, ROTE-aligned weighted eval. num_blocks={num_blocks}, num_walls={num_walls}, agent_id={agent_id}")

    if client_kwargs is None:
        client_kwargs = {}
    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    seed_code = load_seed_program(seed_program_path)

    if output_dir is None:
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        mode = "non_strict"
        output_root = "generated_outputs_ablation" if ablation else "generated_outputs"
        run_dir = ablation if ablation else f"run_{timestamp}"
        output_dir = f"{output_root}/gridworld_ensemble/{mode}/{run_dir}/agent_{agent_id}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file_path = output_path / "wandb_metrics.jsonl" if wandb is not None else None

    # Baseline: single seed (train + test)
    print(f"\n{'='*80}\nBASELINE EVALUATION (seed program, single)\n{'='*80}")
    baseline_train_eval = evaluate_gridworld_program(
        seed_code, data_path, num_blocks, num_walls, agent_id,
        num_datapoints=80, num_steps=20, verbose=True, n_seeds=n_eval_seeds,
        evaluate_on_observed=True,
    )
    baseline_test_eval = evaluate_gridworld_program(
        seed_code, data_path, num_blocks, num_walls, agent_id,
        num_datapoints=20, num_steps=20, verbose=True, n_seeds=n_eval_seeds,
        evaluate_on_observed=False,
    )
    print(f"Baseline train accuracy: {baseline_train_eval['accuracy']:.4f}, test: {baseline_test_eval['accuracy']:.4f}")

    baseline_results = {
        "train_accuracy": baseline_train_eval["accuracy"],
        "test_accuracy": baseline_test_eval["accuracy"],
        "train_correct": baseline_train_eval["correct"],
        "train_total": baseline_train_eval["total"],
        "test_correct": baseline_test_eval["correct"],
        "test_total": baseline_test_eval["total"],
    }
    if wandb is not None:
        wandb.log({
            f"a{agent_id}_train_accuracy": baseline_train_eval["accuracy"],
            f"a{agent_id}_test_accuracy": baseline_test_eval["accuracy"],
            f"a{agent_id}_is_baseline": 1,
        }, step=0)
        if log_file_path is not None:
            with open(log_file_path, "a") as f:
                f.write(json.dumps({"step": 0, "iteration": -1, f"a{agent_id}_train_accuracy": baseline_train_eval["accuracy"], f"a{agent_id}_test_accuracy": baseline_test_eval["accuracy"], f"a{agent_id}_is_baseline": 1}) + "\n")

    # K elite pools; each element (code, fitness=train_acc, test_acc, program_id, None, None)
    elite_pools = []
    for k in range(K):
        elite_pools.append([
            (seed_code, baseline_train_eval["accuracy"], baseline_test_eval["accuracy"], "baseline", None, None)
        ])

    max_elite_size = max(sample_size * 2, 20)

    for iteration in range(n_iterations):
        iteration_step = iteration + 1  # 1-indexed to match wandb
        print(f"\n{'='*80}\nIteration {iteration_step}/{n_iterations} (ensemble size K={K})\n{'='*80}")
        iter_dir = output_path / f"iteration_{iteration_step}"
        iter_dir.mkdir(exist_ok=True)

        for k in range(K):
            candidates_dir = iter_dir / f"member_{k}"
            candidates_dir.mkdir(exist_ok=True)

            num_parents_to_use = min(sample_size, len(elite_pools[k]))
            selected_parents = elite_pools[k][:num_parents_to_use]
            parent_codes = [p[0] for p in selected_parents]
            parent_train_accs = [p[1] for p in selected_parents]

            candidate_codes = generate_gridworld_program_variants(
                client=client,
                model_name=model_name,
                template_code=seed_code,
                parent_codes=parent_codes,
                n_variants=n_candidates_per_iteration,
                parent_train_accuracies=parent_train_accs,
                max_workers=max_workers,
            )

            candidate_results = []
            for idx, code in enumerate(candidate_codes):
                (candidates_dir / f"candidate_{idx}.py").write_text(code or "")
                if not code:
                    candidate_results.append({"idx": idx, "code": "", "train_acc": 0.0, "test_acc": 0.0, "fitness": 0.0, "valid": False})
                    continue
                train_eval = evaluate_gridworld_program(
                    code, data_path, num_blocks, num_walls, agent_id,
                    num_datapoints=80, num_steps=20, n_seeds=n_eval_seeds,
                    evaluate_on_observed=True,
                )
                test_eval = evaluate_gridworld_program(
                    code, data_path, num_blocks, num_walls, agent_id,
                    num_datapoints=20, num_steps=20, n_seeds=n_eval_seeds,
                    evaluate_on_observed=False,
                )
                train_acc = train_eval["accuracy"]
                test_acc = test_eval["accuracy"]
                candidate_results.append({
                    "idx": idx, "code": code, "train_acc": train_acc, "test_acc": test_acc,
                    "fitness": train_acc,
                    "train_correct": train_eval["correct"], "test_correct": test_eval["correct"],
                    "train_total": train_eval["total"], "test_total": test_eval["total"],
                    "valid": train_eval["errors"] == 0,
                })

            valid_results = [r for r in candidate_results if r["valid"]]
            if valid_results:
                valid_results.sort(key=lambda x: x["fitness"], reverse=True)
                for r in valid_results:
                    program_id = f"iteration_{iteration_step}_member_{k}_candidate_{r['idx']}"
                    elite_pools[k].append((r["code"], r["fitness"], r["test_acc"], program_id, None, None))
                elite_pools[k].sort(key=lambda x: x[1], reverse=True)
                elite_pools[k] = elite_pools[k][:max_elite_size]

        # After all K members updated: compute weights from first-20-step log-likelihood, then ensemble test
        if K > 0 and elite_pools[0]:
            best_codes_iter = [elite_pools[k][0][0] for k in range(K)]
            weights_iter = compute_gridworld_ensemble_weights(
                best_codes_iter, data_path, num_blocks, num_walls, agent_id,
                num_datapoints=80, n_seeds=n_eval_seeds,
            )
            ensemble_test_eval_iter = evaluate_gridworld_ensemble_test(
                best_codes_iter, data_path, num_blocks, num_walls, agent_id,
                weights=weights_iter,
                top_k=top_k,
                num_datapoints=20, num_steps=20, verbose=False, n_seeds=n_eval_seeds,
            )
            ensemble_test_acc_iter = ensemble_test_eval_iter["accuracy"]
            # Best individual program ID this iteration (member with highest train acc)
            best_k = max(range(K), key=lambda k: elite_pools[k][0][1])
            best_program_id = elite_pools[best_k][0][3]
            avg_train = np.mean([elite_pools[k][0][1] for k in range(K)])
            if wandb is not None:
                log_dict = {
                    f"a{agent_id}_train_accuracy": avg_train,
                    f"a{agent_id}_test_accuracy": ensemble_test_acc_iter,
                    f"a{agent_id}_best_program_id": best_program_id,
                }
                wandb.log(log_dict, step=iteration + 1)
                if log_file_path is not None:
                    with open(log_file_path, "a") as f:
                        f.write(json.dumps({"step": iteration + 1, "iteration": iteration_step, **log_dict}) + "\n")

    # Best program per member; weights from log-likelihood on first 20 steps
    best_codes = [elite_pools[k][0][0] for k in range(K)]
    final_weights = compute_gridworld_ensemble_weights(
        best_codes, data_path, num_blocks, num_walls, agent_id,
        num_datapoints=80, n_seeds=n_eval_seeds,
    )
    ensemble_test_eval = evaluate_gridworld_ensemble_test(
        best_codes, data_path, num_blocks, num_walls, agent_id,
        weights=final_weights,
        top_k=top_k,
        num_datapoints=20, num_steps=20, verbose=True, n_seeds=n_eval_seeds,
    )
    mean_train_acc = np.mean([elite_pools[k][0][1] for k in range(K)])
    ensemble_test_acc = ensemble_test_eval["accuracy"]

    results = {
        "baseline": baseline_results,
        "overall_best_train": {"train_accuracy": mean_train_acc, "test_accuracy": ensemble_test_acc, "program_id": "ensemble"},
        "overall_best_test": {"train_accuracy": mean_train_acc, "test_accuracy": ensemble_test_acc, "program_id": "ensemble"},
        "ensemble_test_accuracy": ensemble_test_acc,
        "ensemble_size": K,
    }
    (output_path / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'='*80}\nEvolution Complete (gridworld_ensemble)\n{'='*80}")
    print(f"Mean train accuracy (over K best): {mean_train_acc:.4f}")
    print(f"Ensemble test accuracy (weighted, tie-aware): {ensemble_test_acc:.4f}")
    print(f"Baseline test accuracy (single): {baseline_test_eval['accuracy']:.4f}")
    print(f"Results saved to: {output_path / 'results.json'}")

    return {
        "participant_id": agent_id,
        "train_acc": mean_train_acc,
        "test_acc": ensemble_test_acc,
    }


def _make_unique_run_dir(base_dir: Path) -> Path:
    for i in range(1000):
        candidate = base_dir if i == 0 else Path(f"{base_dir}_{i}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create unique run directory from {base_dir}")


def _write_command_line_log(run_dir: Path) -> Path:
    """Persist interpreter + argv under run_dir/log/command.txt (path also printed for SLURM logs)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "log"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / "command.txt"
    cmd = shlex.join([sys.executable, *sys.argv])
    stamp = datetime.now().isoformat(timespec="seconds")
    body = f"# saved {stamp}\n# cwd: {os.getcwd()}\n# host: {socket.gethostname()}\n{cmd}\n"
    path.write_text(body, encoding="utf-8")
    return path


def _write_run_metadata(
    run_dir: Path,
    *,
    error_feedback_mode: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist resolved defaults that may be absent from command.txt."""
    log_dir = run_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "run_metadata.json"
    payload: Dict[str, Any] = {
        "error_feedback_mode": _normalize_error_feedback_mode(error_feedback_mode)
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _merge_run_metadata(run_dir: Path, extra: Mapping[str, Any]) -> Path:
    """Update log/run_metadata.json in place without dropping prior keys."""
    log_dir = Path(run_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "run_metadata.json"
    payload: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            payload = loaded
    payload.update(dict(extra))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _csv_fieldnames_from_rows(rows: List[Dict[str, Any]]) -> List[str]:
    """Union of dict keys across rows, preserving order from the first row then appending extras."""
    if not rows:
        return []
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def _round_floats_for_csv_row(row: Dict[str, Any], ndigits: int = 4) -> Dict[str, Any]:
    """Round finite floats for CSV output; keep ints, None, bools, and other types unchanged."""
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


def main():
    """Main entry point."""
    import argparse
    
    _teh_dataset_choices = sorted(PARTICIPANT_DATASETS | set(PSYCH101_LEGACY_ALIASES))
    parser = argparse.ArgumentParser(
        description="TEH: Template Evolution on Psych-101 binary cognitive datasets (structured trials)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=PETERSON2021USING_ALIAS,
        choices=_teh_dataset_choices,
        help=(
            "Psych-101 binary alias (1peterson2021using, ...), mixed_gambles (local CSV), "
            "or external alias (bergert_nosofsky_2007, guan_2020_stopping, steyvers_2009_bandit). "
            "Unprefixed legacy Psych-101 names are accepted and normalized."
        ),
    )
    parser.add_argument(
        "--mixed_gambles_csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="CSV path for --dataset mixed_gambles (default: datasets/mixed_gambles/data_all_2021-01-08.csv).",
    )
    parser.add_argument(
        "--psych_dataset_split",
        type=str,
        default=DEFAULT_PSYCH_DATASET_SPLIT,
        choices=sorted({"train", "test"}),
        help=(
            "Psych-101 HF participant corpus: train=marcelbinz/Psych-101, "
            "test=marcelbinz/Psych-101-test (default: train). Ignored for mixed_gambles."
        ),
    )
    parser.add_argument(
        "--local_dataset",
        type=str,
        default=None,
        help="Optional path from datasets.load_from_disk for Psych-101 (else Hugging Face hub).",
    )
    parser.add_argument(
        "--no_llm_prompt",
        action="store_true",
        help="Skip LLM prompt generation; merge base loglik prompt with dataset description only.",
    )
    parser.add_argument(
        "--dataset_prompt_evolution_iterations",
        type=int,
        default=0,
        help=(
            "Optional dataset-prompt evolution rounds after one-shot auto prompt "
            "(default: 0 = one-shot only)."
        ),
    )
    parser.add_argument(
        "--dataset_prompt_evolution_population",
        type=int,
        default=3,
        help="Beam size for dataset-prompt evolution (default: 3).",
    )
    parser.add_argument(
        "--dataset_prompt_evolution_children",
        type=int,
        default=3,
        help="Children per evolution round (default: 3; roles repair/model/explore).",
    )
    parser.add_argument(
        "--dataset_prompt_dev_participants",
        type=int,
        default=3,
        help="Number of valid-list ordinals 0..K-1 used to score prompts (default: 3).",
    )
    parser.add_argument(
        "--dataset_prompt_eval_candidates",
        type=int,
        default=5,
        help="Programs generated per prompt×dev-participant during prompt scoring (default: 5).",
    )
    parser.add_argument(
        "--dataset_prompt_example_char_budget",
        type=int,
        default=DEFAULT_EXAMPLE_CHAR_BUDGET,
        help=(
            "Char budget for full-JSON train examples in auto prompt generation "
            f"(default: {DEFAULT_EXAMPLE_CHAR_BUDGET})."
        ),
    )
    parser.add_argument(
        "--dataset_prompt_history_max_entries",
        type=int,
        default=DEFAULT_HISTORY_MAX_ENTRIES,
        help=(
            "Max history entries per prompt-generation example "
            f"(default: {DEFAULT_HISTORY_MAX_ENTRIES})."
        ),
    )
    parser.add_argument(
        "--dataset_prompt_max_examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help=(
            "Max complete JSON train examples in auto prompt generation "
            f"(default: {DEFAULT_MAX_EXAMPLES})."
        ),
    )
    parser.add_argument(
        "--dataset_prompt_file",
        type=str,
        default=None,
        help=(
            "Optional path to a fixed evolution prompt written as infer_single_choice.txt "
            "(skips reference / LLM / merge). Example: "
            "prompts/teh/cursor_designed/7hilbig2014generalized.txt"
        ),
    )
    parser.add_argument(
        "--prefer_auto_llm_prompt",
        action="store_true",
        help=(
            "Skip registered hand-written dataset prompts and generate the evolution "
            "prompt from task metadata plus observed train+val examples. "
            "Implied when --dataset_prompt_evolution_iterations > 0."
        ),
    )
    parser.add_argument(
        "--base_prompt",
        type=str,
        default="prompts/teh/infer_single_choice.txt",
        help=(
            "Path to base loglik evolution prompt template "
            "(default: prompts/teh/infer_single_choice.txt)."
        ),
    )
    parser.add_argument(
        "--seed_path",
        type=str,
        default=None,
        help=(
            "Path to seed program. Default: persona_code_example/te_vanilla/choices13k.py"
        ),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Unused in TEH (legacy Template_evo). Psych-101 uses Hugging Face; mixed_gambles uses --mixed_gambles_csv.",
    )
    parser.add_argument(
        "--loop_mode",
        type=str,
        default="random",
        choices=["random", "sequential"],
        help="Legacy gridworld only (not supported in TEH CLI).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Legacy gridworld only (not supported in TEH CLI).",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=None,
        help="Legacy gridworld only (not supported in TEH CLI).",
    )
    parser.add_argument(
        "--num_walls",
        type=int,
        default=None,
        help="Legacy gridworld only (not supported in TEH CLI).",
    )
    parser.add_argument(
        "--agent_id",
        type=int,
        default=None,
        help="Legacy gridworld only (not supported in TEH CLI).",
    )
    parser.add_argument(
        "--participant_scope",
        type=str,
        default="single",
        choices=["single", "range", "ordinals", "all"],
        help=(
            "How to select participants for TEH binary-loglik datasets. "
            "'single' uses --single_participant_id (raw id). "
            "'range' uses --range_start_ordinal/--range_end_ordinal (inclusive) into valid_participant_ids.json. "
            "'ordinals' uses --ordinals (0-based indices into that list). "
            "'all' runs all valid ids, optionally capped by --all_max_participants."
        ),
    )
    parser.add_argument(
        "--single_participant_id",
        type=int,
        default=0,
        help="Raw participant id when --participant_scope single (must appear in valid_participant_ids.json). Default 0.",
    )
    parser.add_argument(
        "--range_start_ordinal",
        type=int,
        default=None,
        help="0-based start index into valid_participant_ids.json when --participant_scope range.",
    )
    parser.add_argument(
        "--range_end_ordinal",
        type=int,
        default=None,
        help="0-based inclusive end index into valid_participant_ids.json when --participant_scope range.",
    )
    parser.add_argument(
        "--all_max_participants",
        type=int,
        default=None,
        help="When --participant_scope all: use only the first N valid raw ids from JSON. Omit to run all valids.",
    )
    parser.add_argument(
        "--ordinals",
        nargs="+",
        type=int,
        default=None,
        metavar="I",
        help=(
            "When --participant_scope ordinals: 0-based ordinals into datasets/*/valid_participant_ids.json "
            "(same ordering as range), not raw participant ids. Example: --ordinals 0 4 9"
        ),
    )
    parser.add_argument(
        "--num_agents_to_sample",
        type=int,
        default=1,
        help="Legacy gridworld only (not supported in TEH CLI).",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=5,
        help="Number of evolution iterations",
    )
    parser.add_argument(
        "--early_stop_iters",
        type=int,
        default=-1,
        help=(
            "Early stopping patience for evolution/refinement/global loops. "
            "<=0 disables (default: -1). If >0, stop when pool-best fitness improves by "
            f"< {_EARLY_STOP_MIN_IMPROVEMENT:.3f} for this many consecutive iterations."
        ),
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=10,
        help="Number of candidate programs per iteration",
    )
    parser.add_argument(
        "--fresh_n_candidates",
        type=int,
        default=0,
        help=(
            "Max fresh children per iteration from the seed/baseline parent only (independent of "
            "the elite pool). 0 disables fresh candidates (default). When >0, count decays each "
            "iteration: fresh_n = max(1, floor(fresh_n_candidates * (1 - iter_idx / total_iters))), "
            "clamped to n_candidates. Evolution, refinement, and global phase each decay over "
            "their own iteration counts."
        ),
    )
    parser.add_argument(
        "--explore_candidates",
        type=int,
        default=0,
        help=(
            "Before per-participant evolution, generate this many candidates, evaluate them, "
            "and merge valid programs into the initial elite pool (default: 0 = disabled). "
            "With --initial_pool_programs, or with --global_phase plus "
            "--explore_from_population_parents, the budget is split across those handoff "
            "programs; otherwise candidates are generated from --seed_path only."
        ),
    )
    parser.add_argument(
        "--explore_from_population_parents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After --global_phase, generate --explore_candidates from the population "
            "handoff programs (same parent path as --initial_pool_programs) instead of "
            "the vanilla seed. Default: False (EMNLP: live global explore stays seed-only). "
            "No effect on --initial_pool_programs, which already explores from those files. "
            "Requires --global_phase (or an initial pool) and --explore_candidates > 0."
        ),
    )
    parser.add_argument(
        "--explore_population_top_k",
        type=int,
        default=0,
        metavar="K",
        help=(
            "When exploring from handoff/population parents, use only the first K programs "
            "in global/handoff order (0 = all; 1 = sole rank-1 population program, Stage B). "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--explore_seed_candidates",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Of --explore_candidates, generate this many from the vanilla seed. The rest "
            "are split across handoff/population parents. Default 0 = all from handoff "
            "parents when those exist (Stage B/D). Stage F uses 25 of 50."
        ),
    )
    parser.add_argument(
        "--explore_prompt_source_program",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Stage E: inject this live source rank-1 choose() program into the participant "
            "explore prompt (cross-task suffix). Explore still generates from --seed_path "
            "unless an initial pool is also set. Requires --explore_prompt_source_dataset."
        ),
    )
    parser.add_argument(
        "--explore_prompt_source_dataset",
        type=str,
        default=None,
        metavar="DATASET",
        help="Dataset alias for --explore_prompt_source_program (task description + example trial).",
    )
    parser.add_argument(
        "--global_prompt_source_program",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Stage G: inject this live source rank-1 choose() program into the "
            "target population (global) prompt. Global still starts from --seed_path. "
            "Requires --global_phase and --global_prompt_source_dataset. "
            "Does not apply to participant explore."
        ),
    )
    parser.add_argument(
        "--global_prompt_source_dataset",
        type=str,
        default=None,
        metavar="DATASET",
        help=(
            "Dataset alias for --global_prompt_source_program. "
            "If omitted while --global_prompt_source_program is set, T-PICS looks up the "
            "official source for --dataset in --t_pics_source_config "
            "(default: analysis/config/transfer_source/t_pics_score_weighted_temp_fix.yaml)."
        ),
    )
    parser.add_argument(
        "--t_pics",
        action="store_true",
        default=False,
        help=(
            "Independent T-PICS: train the source population in this run, then the target. "
            "If --t_pics_source is omitted, look up the official source for --dataset. "
            "Not the gated dual-arm transfer pipeline (see --t_pics_gated_transfer)."
        ),
    )
    parser.add_argument(
        "--t_pics_gated_transfer",
        action="store_true",
        default=False,
        help=(
            "Final default T-PICS transfer pipeline: reuse frozen G.1 source pops from "
            "analysis/config/T-PICS/Transfer_source/v2/"
            "occurrence_eb_schema4_iter10_sa40_v2.yaml "
            "(unless --t_pics_gated_independent); "
            "run two matched 5-iter target-population arms (control vs transfer); "
            "select transfer only if mean target train_val (train+val) loglik is "
            "strictly greater; "
            "explore 50 from the retained target rank-1; then 10 person iterations. "
            "Does not change generic TEH defaults."
        ),
    )
    parser.add_argument(
        "--t_pics_gated_independent",
        action="store_true",
        default=False,
        help=(
            "Optional gated T-PICS mode: look up only the official selected source "
            "dataset from --t_pics_source_config, then train a live 10-iter G.1 "
            "source population in this run. Does not reuse frozen G.1 programs. "
            "Requires --t_pics_gated_transfer. G.2 dual-arm gate, G.3 rank-1 "
            "explore, and person evolution stay the same. "
            "Or pass --t_pics_gated_source DATASET to name the live source "
            "without a transfer map."
        ),
    )
    parser.add_argument(
        "--t_pics_gated_source",
        type=str,
        default=None,
        metavar="DATASET",
        help=(
            "With --t_pics_gated_independent: use this source dataset for the live "
            "G.1 (under <job>/source_population/) instead of looking up a frozen "
            "transfer map. Does not require --t_pics_source_config. Ignored unless "
            "--t_pics_gated_independent is set."
        ),
    )
    parser.add_argument(
        "--t_pics_source",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        metavar="DATASET",
        help=(
            "T-PICS source dataset for a live source population in this run. "
            "Pass --t_pics_source with no value, or --t_pics without this flag, to auto-select "
            "from --t_pics_source_config (default: "
            "analysis/config/transfer_source/t_pics_score_weighted_temp_fix.yaml). "
            "Do not combine with --global_prompt_source_program. "
            "Requires --global_phase, --explore_from_population_parents, "
            "--explore_population_top_k 1, --explore_candidates > 0, and "
            "--no-refinement_phase."
        ),
    )
    parser.add_argument(
        "--t_pics_source_config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "YAML target→source map for T-PICS auto-select, or a run config "
            "with source_map + step1.source_pops (e.g. "
            "analysis/config/misc/Sep17_T-PICS/config_T-PICS.yaml). "
            "Default map: analysis/config/transfer_source/"
            "t_pics_score_weighted_temp_fix.yaml "
            "(skips pre-fix CPC18 / Speekenbrink / Frey Risk / Badham). "
            "A run config also fills --global_prompt_source_program from the "
            "recorded Step 1 rank-1 when that flag is omitted. "
            "--t_pics_gated_transfer defaults this path to "
            "analysis/config/T-PICS/Transfer_source/v2/"
            "occurrence_eb_schema4_iter10_sa40_v2.yaml "
            "(written after v2 G.1 + schema-v4 + Occurrence-EB). "
            "Frozen v1 map: analysis/config/T-PICS/Transfer_source/"
            "occurrence_eb_schema4_iter10.yaml."
        ),
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Legacy gridworld only (not supported in TEH CLI).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Number of parent programs to use when generating each child (default: 3)",
    )
    parser.add_argument(
        "--sample_parents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled (default), pick parent programs uniformly at random without replacement from the "
            "elite pool. When disabled, use the top sample_size programs by fitness. "
            "Does not apply to legacy gridworld ensemble mode."
        ),
    )
    parser.add_argument(
        "--sampled_parents_decay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --sample_parents is enabled (default), decay the number of randomly sampled parents "
            "toward 0 over iterations (final iteration uses top parents only). "
            "Disable with --no-sampled_parents_decay to keep uniform random sampling each iteration."
        ),
    )
    parser.add_argument(
        "--elite_pool_size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max size of the elite program pool after each iteration (best programs kept). "
            "Default: max(2 * sample_size, 20). Must be >= 1 when set."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Legacy gridworld_ensemble only (not supported in TEH CLI).",
    )
    parser.add_argument(
        "--n_eval_seeds",
        type=int,
        default=1,
        help=(
            "Number of evaluation runs per program (averaged for final accuracy). "
            "Default: 1 (sufficient for deterministic choose; raise for stochastic programs)."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        help="LLM model name for generation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "local"],
        help="LLM mode: default uses OpenAI API; local routes to vLLM server",
    )
    parser.add_argument(
        "--llm_server_url",
        type=str,
        default=os.getenv("VLLM_LOCAL_URL", "http://localhost:8000/v1"),
        help="Base URL for local vLLM server when --mode local",
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=os.getenv("VLLM_LOCAL_API_KEY", "EMPTY"),
        help="API key for local vLLM server when --mode local",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        metavar="LABEL",
        help=(
            "Ablation label. When set, logs are written under generated_outputs_ablation/ and "
            "the run folder uses LABEL (e.g. --ablation population -> .../teh/.../population)."
        ),
    )
    parser.add_argument(
        "--mem_trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write passive MEM traces for source population and target evolution "
            "(global_phase/mem_trace.jsonl and participant_*/mem_trace.jsonl; "
            "iteration_context + candidate rows; no extra LLM calls; default: on). "
            "Disable with --no-mem_trace."
        ),
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable wandb logging. Default is enabled.",
    )
    parser.add_argument(
        "--filter_mixed_gambles",
        action="store_true",
        default=False,
        help=(
            "For mixed_gambles: keep only gain_loss trials. Default False (all trial types). "
            "Affects which valid_participant_ids.json variant is used for ordinal resolution."
        ),
    )
    parser.add_argument(
        "--fitness_metric",
        type=str,
        default="loglik",
        choices=["accuracy", "loglik"],
        help=(
            "Fitness for selection (default: loglik). TEH participant datasets require loglik "
            "(Bernoulli log-likelihood on train with val/test splits)."
        ),
    )
    parser.add_argument(
        "--evolution_selection_score",
        type=str,
        default="train_val",
        choices=["train", "train_val"],
        help=(
            "Score for evolution/explore/global pool ranking (default: train_val). "
            "'train' ranks by train loglik only. "
            "'train_val' ranks by trial-count-weighted mean of train+val loglik "
            "(falls back to train loglik with a warning if val is missing)."
        ),
    )
    parser.add_argument(
        "--mdl_lambda",
        type=float,
        default=0.0,
        help=(
            "Optional MDL ranking overlay: mdl_score = n * selection_score - mdl_lambda * "
            "program_ast_size. Default 0.0 leaves ranking unchanged. When >0, programs compete "
            "by mdl_score but reported fitness, early stopping, MEM traces, and test loglik "
            "keep the original mean log-likelihood."
        ),
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=TEH_WANDB_PROJECT,
        help=f"W&B project name (default: {TEH_WANDB_PROJECT}).",
    )
    parser.add_argument(
        "--cpc18_official_mse",
        action="store_true",
        help="Legacy CPC18 only (not supported in TEH CLI).",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="within_participant",
        choices=["within_participant", "across_participants"],
        help=(
            "within_participant (default): train/val/test per participant. "
            f"across_participants: pool train across participants ({PETERSON2021USING_ALIAS} only)."
        ),
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.6,
        help="Global train ratio for splitting (default: 0.6).",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Seed for deterministic splitting (default: 0).",
    )
    parser.add_argument(
        "--max_observed_trials_per_participant",
        type=int,
        default=None,
        metavar="N",
        help=(
            "After the train/val/test split, keep at most N train+val observations per "
            "participant (sampled proportionally from train and val; test is never changed). "
            "Omitted or <=0 disables the cap (full data). Uses --split_seed. "
            "Under --limited_data_protocol structure_aware, structure_aware_v2, or "
            "structure_aware_v3 this is the same N as --limited_train_val when that flag is omitted."
        ),
    )
    add_limited_data_cli_arguments(parser)
    parser.add_argument(
        "--max_prompt_train_trials",
        type=int,
        default=60,
        help=(
            "Max trials serialized into each LLM prompt (total across train and validation when both "
            "are injected). With --max_prompt_trials_per_problem > 0, sample (max // per_problem) "
            "blocks at up to per_problem trials each, plus (max %% per_problem) extra trials, from the "
            "union of available splits. With per_problem=0, flat-random sample of max trials. "
            "0 = no cap (full split in prompt). Default 60 (PICS v3 prompt-display ceiling)."
        ),
    )
    parser.add_argument(
        "--max_prompt_trials_per_problem",
        type=int,
        default=5,
        help=(
            "Trials per sampled block in LLM prompts (0 = flat sample of --max_prompt_train_trials only). "
            "Block = Choice13k/mixed_gambles gamble signature, CPC18 (problem_id, block_id), or problem_id."
        ),
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=1024,
        help=(
            "Max output tokens per candidate generation request "
            "(default: 1024 for PICS v3 / Qwen 32768 context)."
        ),
    )
    parser.add_argument(
        "--llm_reasoning_effort",
        type=str,
        default=None,
        choices=("low", "medium", "high", "none", "off"),
        help=(
            "Harmony / gpt-oss reasoning effort stamped on every chat.completions.create "
            "via extra_body.reasoning_effort (low|medium|high). Use low to shrink the "
            "analysis channel so more of --llm_max_tokens lands in final content. "
            "none/off leaves the server default. Ignored for non-Harmony models."
        ),
    )
    parser.add_argument(
        "--hard_prompt_token_cap",
        type=int,
        default=PICS_V3_HARD_PROMPT_TOKEN_CAP,
        help=(
            "Hard input-token budget for each LLM prompt (estimated via --prompt_token_estimator). "
            "Prompts are structurally truncated before calling vLLM; never sent if still over cap. "
            f"Effective cap is min(this value, {VLLM_CONTEXT} - --llm_max_tokens) so input+output "
            f"fit the vLLM context (default: {PICS_V3_HARD_PROMPT_TOKEN_CAP}; with "
            f"--llm_max_tokens 1024 that remains {PICS_V3_HARD_PROMPT_TOKEN_CAP})."
        ),
    )
    parser.add_argument(
        "--strict_prompt_budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When a prompt remains over --hard_prompt_token_cap after truncation, raise a clear error "
            "instead of calling the LLM (default: True)."
        ),
    )
    parser.add_argument(
        "--prompt_token_estimator",
        type=str,
        default="char4",
        choices=("char4",),
        help=(
            "Token estimator for prompt budgeting: char4 = ceil(len/4) (default: char4). "
            "Under structure_aware_v2 / structure_aware_v3 the Qwen chat tokenizer is used on "
            "chat-templated system+user input so trial-count truncation matches vLLM."
        ),
    )
    parser.add_argument(
        "--prompt_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save full LLM prompt + raw responses when all candidates sanitize to empty; "
            "use with --prompt_debug_exit to stop the run after writing debug artifacts."
        ),
    )
    parser.add_argument(
        "--prompt_debug_on_no_valid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "On 'No runtime-valid programs' in evolution, write prompt_debug/ artifacts "
            "(default: True). Does not exit unless --prompt_debug_exit is also set."
        ),
    )
    parser.add_argument(
        "--prompt_debug_exit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After saving prompt_debug/ on empty candidates or no runtime-valid iteration, "
            "exit the process (default: False)."
        ),
    )
    parser.add_argument(
        "--max_parent_chars",
        type=int,
        default=PICS_V3_MAX_PARENT_CHARS,
        help=(
            "Max characters per parent program inserted into LLM prompts (0 = no truncation). "
            f"Candidate code is never truncated for evaluation. Default: {PICS_V3_MAX_PARENT_CHARS} "
            "(PICS v3; existing ~70%% head + marker + ~30%% tail compaction)."
        ),
    )
    parser.add_argument(
        "--max_error_prompt_chars",
        type=int,
        default=1200,
        help=(
            "Max characters reserved for compact past invalid-program errors in each "
            "evolution prompt (0 disables the section). Default: 1200 (legacy). "
            "T-PICS workers pass 0 explicitly."
        ),
    )
    parser.add_argument(
        "--error-feedback-mode",
        "--error_feedback_mode",
        dest="error_feedback_mode",
        choices=sorted(ERROR_FEEDBACK_MODES),
        default=DEFAULT_ERROR_FEEDBACK_MODE,
        help=(
            "Error feedback semantics: legacy reproduces the pre-f169cb4 cumulative "
            "EMNLP behavior; new uses previous-iteration-only contextual groups. "
            "Default: legacy."
        ),
    )
    parser.add_argument(
        "--warn_parent_truncation_ratio",
        type=float,
        default=0.5,
        help=(
            "Print a warning when truncated_parents / sample_size >= this ratio "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Parallel LLM requests when generating candidate children per iteration (default: 5).",
    )
    parser.add_argument(
        "--parallel_participants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run participants in parallel (participant_workers=max_workers//n_candidates, "
            "candidate_workers_per_participant=n_candidates). Default: True. Used for "
            "--phase all, --phase evolution, --phase refine, and --participant_scope all "
            "(not across_participants or gridworld). Pass --no-parallel_participants to disable. "
            "Shared experiment CSVs are updated on the main thread only."
        ),
    )
    parser.add_argument(
        "--gate_phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After evolution on choice13k only (with val split): apply external val_loglik consistency "
            "gate during test evaluation and report gated_test_loglik (default: False)."
        ),
    )
    parser.add_argument(
        "--refinement_phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After evolution on choice13k / cpc18 loglik / mixed_gambles loglik: if val_loglik < "
            "--refinement_val_threshold, run refinement with validation trials in the prompt and "
            "report refinement test loglik as gated_test_loglik; otherwise copy evolution "
            "test_loglik into gated_test_loglik for experiment CSVs (default: True)."
        ),
    )
    parser.add_argument(
        "--refinement_iters",
        type=int,
        default=5,
        help="Number of refinement iterations when --refinement_phase is enabled (default: 5).",
    )
    parser.add_argument(
        "--refinement_val_threshold",
        type=float,
        default=-1.0,
        help=(
            "Run refinement only when val_loglik is below this value (default: -1.0). "
            "Applies to --phase all (post-evolution) and --phase refine."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=sorted(_RUN_PHASES),
        default="all",
        help=(
            "Pipeline stage: all=evolution then optional refinement; evolution=evolution only; "
            "refine=refinement-only from --prev_exp_path (default: all)."
        ),
    )
    parser.add_argument(
        "--global_phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --phase all: run cross-participant global evolution on pooled train trials "
            "before per-participant evolution (loglik only; uses --global_iters and the same "
            "parent/candidate args as evolution). Default: False."
        ),
    )
    parser.add_argument(
        "--global_iters",
        type=int,
        default=10,
        help="Number of global-phase iterations when --global_phase is enabled (default: 10).",
    )
    parser.add_argument(
        "--prev_exp_path",
        type=str,
        default=None,
        help=(
            "Prior run directory for --phase refine (e.g. generated_outputs/choice13k/non_strict/"
            "run_260517_091545). Each participant_* folder must contain best_program.py."
        ),
    )
    parser.add_argument(
        "--initial_pool_programs",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "Load these choose() programs into the participant initial elite pool "
            "(rescored on each participant's train/val). Cannot be combined with --global_phase. "
            "Vanilla --seed_path is not included unless listed here."
        ),
    )
    parser.add_argument(
        "--initial_pool_dir",
        nargs="+",
        default=None,
        metavar="DIR",
        help=(
            "Optional global_elite_pool directory (pool_manifest.json + ranked .py files) "
            "to load as the participant initial pool. Repeat to mix two live pools "
            "(Stage F: source pool then target pool). Can be combined with "
            "--initial_pool_programs. Cannot be combined with --global_phase."
        ),
    )

    args = parser.parse_args()
    t_pics_gated = bool(getattr(args, "t_pics_gated_transfer", False))
    t_pics_gated_independent = bool(getattr(args, "t_pics_gated_independent", False))
    t_pics_gated_source = str(getattr(args, "t_pics_gated_source", "") or "").strip() or None
    if t_pics_gated_independent and not t_pics_gated:
        print("Error: --t_pics_gated_independent requires --t_pics_gated_transfer.")
        return
    if t_pics_gated_source and not t_pics_gated_independent:
        print(
            "Error: --t_pics_gated_source requires --t_pics_gated_independent "
            "(and --t_pics_gated_transfer)."
        )
        return
    if t_pics_gated:
        if bool(getattr(args, "t_pics", False)) or getattr(args, "t_pics_source", None) is not None:
            print(
                "Error: --t_pics_gated_transfer cannot be combined with "
                "--t_pics / --t_pics_source (those are the leftover non-gated live-source "
                "pipeline). Use --t_pics_gated_independent for a live G.1 inside gated T-PICS "
                "(optionally with --t_pics_gated_source DATASET)."
            )
            return
        apply_gated_cli_defaults(args)
    if args.ablation is not None:
        args.ablation = args.ablation.strip()
        if not args.ablation:
            print("Error: --ablation must be a non-empty label when provided.")
            return
        if "/" in args.ablation or "\\" in args.ablation:
            print("Error: --ablation label must not contain path separators ('/' or '\\').")
            return
    args.dataset = normalize_psych101_dataset_alias(args.dataset)
    if t_pics_gated:
        if getattr(args, "global_prompt_source_program", None) or getattr(
            args, "global_prompt_source_dataset", None
        ):
            print(
                "Error: --t_pics_gated_transfer sets source-conditioning on the transfer arm "
                "from the frozen YAML; do not pass --global_prompt_source_program / "
                "--global_prompt_source_dataset."
            )
            return
        if getattr(args, "explore_prompt_source_program", None):
            print(
                "Error: --t_pics_gated_transfer cannot be combined with "
                "--explore_prompt_source_program (discarded arm must not leak into explore)."
            )
            return
        if int(args.explore_population_top_k) != 1:
            print(
                "Error: --t_pics_gated_transfer requires --explore_population_top_k 1 "
                f"(sole retained target rank-1 parent; got {args.explore_population_top_k})."
            )
            return
        if not args.explore_from_population_parents:
            print(
                "Error: --t_pics_gated_transfer requires --explore_from_population_parents."
            )
            return
        if args.refinement_phase:
            print("Error: --t_pics_gated_transfer requires --no-refinement_phase.")
            return
        if not args.global_phase:
            print("Error: --t_pics_gated_transfer requires --global_phase.")
            return
        if int(args.explore_candidates) <= 0:
            print("Error: --t_pics_gated_transfer requires --explore_candidates > 0.")
            return
        if t_pics_gated_source:
            try:
                entry = make_explicit_independent_source_entry(
                    target=str(args.dataset),
                    source=str(t_pics_gated_source),
                )
            except ValueError as exc:
                print(f"Error: {exc}")
                return
            print(
                f"[T-PICS gated] preflight OK (explicit independent source; "
                f"target={entry.target} source={entry.selected_source}; "
                f"no transfer map required; live G.1)"
            )
        else:
            try:
                cfg_path = Path(str(args.t_pics_source_config))
                if not cfg_path.is_absolute():
                    cfg_path = (_REPO_ROOT / cfg_path).resolve()
                cfg = load_frozen_transfer_config(cfg_path)
                errors = validate_frozen_transfer_config(
                    cfg, require_files=not t_pics_gated_independent
                )
            except (OSError, ValueError, KeyError) as exc:
                print(f"Error: T-PICS gated transfer config failed: {exc}")
                return
            if errors:
                print("Error: T-PICS gated transfer config failed validation:")
                for err in errors:
                    print(f"  - {err}")
                return
            try:
                selected_source_for_target(str(args.dataset), config=cfg)
            except (ValueError, KeyError) as exc:
                print(f"Error: {exc}")
                return
            print(
                f"[T-PICS gated] preflight OK ({cfg.path}; "
                f"{len(cfg.targets)} targets, {len(cfg.source_runs)} source runs"
                + (
                    "; independent live G.1, YAML source-dataset lookup only"
                    if t_pics_gated_independent
                    else "; rank-1 choose() validated; no GPU yet"
                )
                + ")"
            )
    if args.fitness_metric == "loglik" and not is_binary_loglik_dataset(args.dataset) and not (
        args.dataset == "cpc18" and not args.cpc18_official_mse
    ):
        print(
            "Error: --fitness_metric loglik requires a TEH binary-loglik dataset "
            f"({sorted(PARTICIPANT_DATASETS)})."
        )
        return
    if not (0.0 < args.split_ratio < 1.0):
        print(f"Error: --split_ratio must be in (0,1), got {args.split_ratio}.")
        return
    if float(getattr(args, "mdl_lambda", 0.0) or 0.0) < 0.0:
        print(f"Error: --mdl_lambda must be >= 0, got {args.mdl_lambda}.")
        return
    if (
        args.split_mode == "across_participants"
        and args.dataset != PETERSON2021USING_ALIAS
    ):
        print(
            f"Error: --split_mode across_participants is only supported with "
            f"--dataset {PETERSON2021USING_ALIAS}."
        )
        return
    if args.max_prompt_train_trials < 0:
        print("Error: --max_prompt_train_trials must be >= 0 (0 = no cap).")
        return
    if args.max_prompt_trials_per_problem < 0:
        print("Error: --max_prompt_trials_per_problem must be >= 0.")
        return
    if args.llm_max_tokens < 64:
        print("Error: --llm_max_tokens must be >= 64.")
        return
    if args.hard_prompt_token_cap < 256:
        print("Error: --hard_prompt_token_cap must be >= 256.")
        return
    requested_hard_cap = int(args.hard_prompt_token_cap)
    try:
        effective_hard_cap = effective_hard_prompt_token_cap(
            requested_hard_cap,
            int(args.llm_max_tokens),
            vllm_context=VLLM_CONTEXT,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    if effective_hard_cap < 256:
        print(
            "Error: effective hard_prompt_token_cap must be >= 256 "
            f"(got {effective_hard_cap} from hard={requested_hard_cap}, "
            f"llm_max_tokens={args.llm_max_tokens}, vllm_context={VLLM_CONTEXT})."
        )
        return
    if effective_hard_cap != requested_hard_cap:
        print(
            f"[TEH] clamping hard_prompt_token_cap {requested_hard_cap} -> {effective_hard_cap} "
            f"so input + llm_max_tokens={args.llm_max_tokens} fits vLLM context {VLLM_CONTEXT}"
        )
    args.hard_prompt_token_cap = int(effective_hard_cap)
    try:
        args.llm_reasoning_effort = normalize_llm_reasoning_effort(
            getattr(args, "llm_reasoning_effort", None)
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    if args.llm_reasoning_effort:
        print(
            f"[TEH] llm_reasoning_effort={args.llm_reasoning_effort} "
            "(Harmony analysis channel; stamped on every chat.completions.create)"
        )
    if args.max_parent_chars < 0:
        print("Error: --max_parent_chars must be >= 0 (0 = no truncation).")
        return
    if args.max_error_prompt_chars < 0:
        print("Error: --max_error_prompt_chars must be >= 0.")
        return
    if not (0.0 <= args.warn_parent_truncation_ratio <= 1.0):
        print("Error: --warn_parent_truncation_ratio must be in [0, 1].")
        return
    if args.max_workers < 1:
        print("Error: --max_workers must be >= 1.")
        return
    if args.n_candidates < 1:
        print("Error: --n_candidates must be >= 1.")
        return
    if args.fresh_n_candidates < 0 or args.fresh_n_candidates > args.n_candidates:
        print(
            "Error: --fresh_n_candidates must satisfy 0 <= fresh_n_candidates <= n_candidates "
            f"(got fresh_n_candidates={args.fresh_n_candidates}, n_candidates={args.n_candidates})."
        )
        return
    if args.explore_candidates < 0:
        print("Error: --explore_candidates must be >= 0.")
        return
    if args.explore_population_top_k < 0:
        print("Error: --explore_population_top_k must be >= 0 (0 = all handoff parents).")
        return
    if int(args.explore_seed_candidates) < 0:
        print("Error: --explore_seed_candidates must be >= 0.")
        return
    if int(args.explore_seed_candidates) > int(args.explore_candidates):
        print(
            "Error: --explore_seed_candidates cannot exceed --explore_candidates "
            f"(got {args.explore_seed_candidates} > {args.explore_candidates})."
        )
        return
    src_prog = getattr(args, "explore_prompt_source_program", None)
    src_ds = getattr(args, "explore_prompt_source_dataset", None)
    if bool(src_prog) != bool(src_ds):
        print(
            "Error: --explore_prompt_source_program and --explore_prompt_source_dataset "
            "must be set together."
        )
        return
    if src_prog and int(args.explore_candidates) <= 0:
        print("Error: --explore_prompt_source_program requires --explore_candidates > 0.")
        return
    g_prog = getattr(args, "global_prompt_source_program", None)
    g_ds = getattr(args, "global_prompt_source_dataset", None)
    t_pics_flag = bool(getattr(args, "t_pics", False))
    t_pics_source_raw = getattr(args, "t_pics_source", None)
    t_pics_source_alias = None
    t_pics_cfg_raw = getattr(args, "t_pics_source_config", None)
    if t_pics_cfg_raw:
        t_pics_cfg = Path(str(t_pics_cfg_raw)).expanduser()
        if not t_pics_cfg.is_absolute():
            t_pics_cfg = (_REPO_ROOT / t_pics_cfg).resolve()
    else:
        t_pics_cfg = DEFAULT_T_PICS_SOURCE_CONFIG
    want_independent = t_pics_flag or t_pics_source_raw is not None
    if (
        not want_independent
        and t_pics_cfg_raw
        and not g_prog
        and args.global_phase
        and has_t_pics_step1_source_pops(t_pics_cfg)
    ):
        try:
            src, best, jid = resolve_t_pics_reuse_source(
                str(args.dataset), config_path=t_pics_cfg
            )
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"Error: T-PICS run config cannot resolve Step 1 reuse: {exc}")
            return
        args.global_prompt_source_program = str(best)
        args.global_prompt_source_dataset = src
        g_prog = args.global_prompt_source_program
        g_ds = src
        print(
            f"[T-PICS] reuse Step 1 {src} job {jid} -> {best} ({t_pics_cfg})"
        )
    if want_independent:
        try:
            if is_auto_t_pics_source_token(t_pics_source_raw):
                t_pics_source_alias = official_t_pics_source(
                    str(args.dataset), config_path=t_pics_cfg
                )
                print(
                    f"[T-PICS] auto source for {args.dataset} -> {t_pics_source_alias} "
                    f"({t_pics_cfg})"
                )
            else:
                t_pics_source_alias = normalize_t_pics_dataset(str(t_pics_source_raw))
        except (ValueError, KeyError) as exc:
            print(f"Error: {exc}")
            return
        if t_pics_source_alias not in _PARTICIPANT_DATASETS:
            print(
                f"Error: --t_pics_source {t_pics_source_alias!r} is not a TEH participant dataset."
            )
            return
        try:
            target_alias = normalize_t_pics_dataset(str(args.dataset))
        except ValueError:
            target_alias = str(args.dataset)
        if t_pics_source_alias == target_alias:
            print("Error: --t_pics_source must differ from --dataset (self is excluded).")
            return
        if g_prog:
            print(
                "Error: --t_pics / --t_pics_source trains the source population in this run; "
                "do not also pass --global_prompt_source_program."
            )
            return
        if g_ds and normalize_t_pics_dataset(str(g_ds)) != t_pics_source_alias:
            print(
                "Error: --global_prompt_source_dataset must match --t_pics_source "
                f"({t_pics_source_alias!r}), got {g_ds!r}."
            )
            return
        if not args.global_phase:
            print("Error: --t_pics / --t_pics_source requires --global_phase.")
            return
        if not args.explore_from_population_parents:
            print(
                "Error: --t_pics / --t_pics_source requires --explore_from_population_parents "
                "(T-PICS explore is from the target transfer-pop rank-1)."
            )
            return
        if int(args.explore_population_top_k) != 1:
            print(
                "Error: --t_pics / --t_pics_source requires --explore_population_top_k 1 "
                f"(got {args.explore_population_top_k})."
            )
            return
        if int(args.explore_candidates) <= 0:
            print("Error: --t_pics / --t_pics_source requires --explore_candidates > 0.")
            return
        if args.refinement_phase:
            print(
                "Error: --t_pics / --t_pics_source requires --no-refinement_phase "
                "(T-PICS has no refinement/finetune)."
            )
            return
        if src_prog:
            print(
                "Error: --t_pics / --t_pics_source cannot be combined with "
                "--explore_prompt_source_program (that is Method E)."
            )
            return
        if not is_auto_t_pics_source_token(t_pics_source_raw):
            try:
                official = official_t_pics_source(
                    str(args.dataset), config_path=t_pics_cfg
                )
            except KeyError:
                official = None
            if official is not None and official != t_pics_source_alias:
                print(
                    f"[WARN] Official T-PICS source for {args.dataset} is {official}, "
                    f"but --t_pics_source is {t_pics_source_alias}."
                )
        args.t_pics_source = t_pics_source_alias
        if not g_ds:
            args.global_prompt_source_dataset = t_pics_source_alias
        g_ds = args.global_prompt_source_dataset
    if bool(g_prog) != bool(g_ds) and not t_pics_source_alias:
        if g_prog and not g_ds:
            try:
                g_ds = official_t_pics_source(
                    str(args.dataset), config_path=t_pics_cfg
                )
                args.global_prompt_source_dataset = g_ds
                print(
                    f"[T-PICS] auto source dataset for reuse -> {g_ds} "
                    f"({t_pics_cfg})"
                )
            except (ValueError, KeyError) as exc:
                print(
                    "Error: --global_prompt_source_dataset omitted and no official T-PICS "
                    f"source for {args.dataset}: {exc}"
                )
                return
        else:
            print(
                "Error: --global_prompt_source_program and --global_prompt_source_dataset "
                "must be set together."
            )
            return
    if g_prog and not args.global_phase:
        print("Error: --global_prompt_source_program requires --global_phase.")
        return
    if args.explore_from_population_parents:
        if not (args.global_phase or bool(args.initial_pool_programs or args.initial_pool_dir)):
            print(
                "Error: --explore_from_population_parents requires --global_phase "
                "or --initial_pool_programs/--initial_pool_dir."
            )
            return
        if int(args.explore_candidates) <= 0:
            print(
                "Error: --explore_from_population_parents requires --explore_candidates > 0."
            )
            return
    if args.early_stop_iters is not None and int(args.early_stop_iters) < -1:
        print("Error: --early_stop_iters must be -1 (disabled) or >= 0.")
        return
    if args.refinement_iters < 1:
        print("Error: --refinement_iters must be >= 1.")
        return
    if args.global_iters < 1:
        print("Error: --global_iters must be >= 1.")
        return
    if args.global_phase and args.phase != "all":
        print("Error: --global_phase requires --phase all.")
        return
    if args.global_phase and args.dataset not in _PARTICIPANT_DATASETS:
        print("Error: --global_phase requires a participant dataset (choice13k, cpc18, mixed_gambles).")
        return
    if args.global_phase and args.split_mode != "within_participant":
        print("Error: --global_phase requires --split_mode within_participant.")
        return
    if args.global_phase and args.fitness_metric != "loglik":
        print("Error: --global_phase requires --fitness_metric loglik.")
        return
    has_initial_pool = bool(args.initial_pool_programs or args.initial_pool_dir)
    if has_initial_pool and args.global_phase:
        print(
            "Error: --initial_pool_programs/--initial_pool_dir cannot be combined with "
            "--global_phase. Load existing population programs OR run a live global phase, not both."
        )
        return
    if has_initial_pool and args.fitness_metric != "loglik":
        print("Error: --initial_pool_programs/--initial_pool_dir requires --fitness_metric loglik.")
        return
    if has_initial_pool and args.split_mode != "within_participant":
        print(
            "Error: --initial_pool_programs/--initial_pool_dir requires "
            "--split_mode within_participant."
        )
        return
    if args.max_observed_trials_per_participant is not None and args.max_observed_trials_per_participant < 0:
        print("Error: --max_observed_trials_per_participant must be >= 0 when set (0 disables).")
        return
    try:
        from utils.teh.limited_data_protocol import resolve_limited_data_budget

        resolve_limited_data_budget(
            protocol=args.limited_data_protocol,
            max_observed_trials_per_participant=args.max_observed_trials_per_participant,
            limited_train_val=args.limited_train_val,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    if args.phase not in _RUN_PHASES:
        print(f"Error: --phase must be one of {sorted(_RUN_PHASES)}, got {args.phase!r}.")
        return
    if args.phase == "refine":
        if args.dataset not in _LOGlik_VAL_SPLIT_DATASETS:
            print(
                "Error: --phase refine only supports loglik datasets with val split: "
                f"{sorted(_LOGlik_VAL_SPLIT_DATASETS)}."
            )
            return
        if args.dataset == "cpc18" and args.cpc18_official_mse:
            print("Error: --phase refine does not support CPC18 official MSE mode.")
            return
        if args.fitness_metric != "loglik":
            print("Error: --phase refine requires --fitness_metric loglik.")
            return
        if not args.prev_exp_path:
            print("Error: --phase refine requires --prev_exp_path.")
            return
        if not Path(args.prev_exp_path).exists():
            print(f"Error: --prev_exp_path does not exist: {args.prev_exp_path}")
            return

    prompt_contract_scope(
        uses_training_only_sa40(
            getattr(args, "limited_data_protocol", "structure_aware_v3")
        )
    ).__enter__()
    if args.phase == "evolution" and args.refinement_phase:
        print("Note: --refinement_phase is ignored when --phase evolution.")
    if args.elite_pool_size is not None and args.elite_pool_size < 1:
        print("Error: --elite_pool_size must be >= 1 when set.")
        return
    mixed_gambles_gain_loss_only = bool(getattr(args, "filter_mixed_gambles", False))
    psych_dataset_split = _effective_psych_dataset_split(
        args.dataset, args.psych_dataset_split
    )

    if args.dataset not in PARTICIPANT_DATASETS:
        print(f"Error: unknown TEH dataset {args.dataset!r}")
        return
    if args.fitness_metric != "loglik":
        print(
            "Error: TEH requires --fitness_metric loglik for participant datasets "
            f"({sorted(PARTICIPANT_DATASETS)}). Accuracy mode is not supported "
            "(logging, refinement, and best-program tracking assume log-likelihood)."
        )
        return
    if is_psych101_dataset(args.dataset) and not PSYCH101_BINARY_DATASETS[args.dataset].get(
        "implemented"
    ):
        print(
            f"Error: parser for {args.dataset!r} is not implemented yet "
            f"(expected: {PSYCH101_BINARY_DATASETS[args.dataset].get('parser')!r})."
        )
        return

    if args.dataset in _PARTICIPANT_DATASETS:
        if args.participant_scope == "ordinals":
            if not args.ordinals:
                print(
                    "Error: --participant_scope ordinals requires --ordinals with at least one integer "
                    "(e.g. --ordinals 0 4 9)."
                )
                return
        elif args.ordinals is not None:
            print("Error: --ordinals is only valid with --participant_scope ordinals.")
            return

    # Create timestamp once at the beginning to ensure consistency between wandb name and folder name
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    output_root_dir = "generated_outputs_ablation" if args.ablation else "generated_outputs"
    run_dir_name = args.ablation if args.ablation else f"run_{timestamp}"
    
    # Optional wandb setup (gated T-PICS inits later, after output_dir exists).
    wandb_enabled = False
    wandb = None
    log_file_path = None
    if not args.no_log and not t_pics_gated:
        try:
            import wandb as _wandb
            wandb = _wandb
            run_name = teh_wandb_run_name(
                args.dataset,
                timestamp,
                args.participant_scope,
                psych_dataset_split=psych_dataset_split,
                range_start=args.range_start_ordinal,
                range_end=args.range_end_ordinal,
                ordinals=args.ordinals,
            )
            if float(getattr(args, "mdl_lambda", 0.0) or 0.0) > 0.0:
                run_name = f"{run_name}_mdl_l{args.mdl_lambda}"
            wandb.init(
                project=str(getattr(args, "wandb_project", None) or TEH_WANDB_PROJECT),
                name=run_name,
                config=vars(args),
                reinit=False,
            )
            wandb_enabled = True
            
        except Exception as e:
            print(f"wandb logging disabled: {e}")
            wandb_enabled = False
    
    # Setup client kwargs
    client_kwargs = {}
    if args.mode == "local":
        client_kwargs = {
            "api_key": args.llm_api_key,
            "base_url": args.llm_server_url,
        }
    
    # Determine which participants to process
    participants_to_process: List[int] = []
    if args.dataset in _PARTICIPANT_DATASETS:
        try:
            participants_to_process = resolve_participants_for_scope(
                dataset=args.dataset,
                repo_root=_REPO_ROOT,
                participant_scope=args.participant_scope,
                single_participant_id=args.single_participant_id,
                range_start_ordinal=args.range_start_ordinal,
                range_end_ordinal=args.range_end_ordinal,
                all_max_participants=args.all_max_participants,
                participant_ordinals=args.ordinals,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
                split_ratio=args.split_ratio,
                split_seed=args.split_seed,
                psych_dataset_split=psych_dataset_split,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        if args.agent_id is not None:
            participants_to_process = [args.agent_id]
        else:
            participants_to_process = list(range(args.num_agents_to_sample))

    if wandb is not None and args.dataset in _PARTICIPANT_DATASETS and not t_pics_gated:
        from utils.teh.t_pics_gated_wandb import suppress_dynamic_participant_wandb_scalars

        if not suppress_dynamic_participant_wandb_scalars():
            for pid in participants_to_process:
                wandb.define_metric(f"p{pid}_step")
                wandb.define_metric(f"p{pid}/*", step_metric=f"p{pid}_step")

    if args.dataset in _PARTICIPANT_DATASETS:
        _valid_ids_path = valid_participant_ids_path(
            args.dataset,
            filter_mixed_gambles=bool(getattr(args, "filter_mixed_gambles", False)),
            psych_dataset_split=psych_dataset_split,
        )
        if args.participant_scope == "single":
            print(
                f"Participant scope: single -> using raw participant id "
                f"{args.single_participant_id}."
            )
        elif args.participant_scope == "range":
            print(
                "Participant scope: range -> using inclusive ordinal slice "
                f"[{args.range_start_ordinal}, {args.range_end_ordinal}] from "
                f"{_valid_ids_path}."
            )
        elif args.participant_scope == "ordinals":
            print(
                "Participant scope: ordinals -> using raw participant ids at 0-based ordinals "
                f"{list(args.ordinals)} from {_valid_ids_path} "
                "(duplicate ordinals collapse to one id; order follows first occurrence)."
            )
        else:
            cap_text = (
                f"first {args.all_max_participants} valid ids"
                if args.all_max_participants is not None
                else "all valid ids"
            )
            print(
                f"Participant scope: all -> using {cap_text} from {_valid_ids_path}."
            )
    if is_psych101_dataset(args.dataset):
        print(
            f"Psych-101 HF corpus: {psych_dataset_split} -> {hf_id_for_psych_dataset_split(psych_dataset_split)}"
        )
    print(
        f"TEH split settings: dataset={args.dataset}, split_mode={args.split_mode}, "
        f"split_ratio={args.split_ratio:.3f}, split_seed={args.split_seed}"
        + (
            f", max_observed_trials_per_participant={args.max_observed_trials_per_participant}"
            if args.max_observed_trials_per_participant
            else ""
        )
        + (
            f", limited_data_protocol={args.limited_data_protocol}"
            if str(args.limited_data_protocol).strip().lower() not in {"", "off"}
            else ""
        )
        + (
            f", limited_train_val={args.limited_train_val}"
            if args.limited_train_val
            else ""
        )
    )
    if args.dataset in _PARTICIPANT_DATASETS and participants_to_process:
        _print_selected_participants_trial_summary(
            args.dataset,
            participants_to_process,
            repo_root=_REPO_ROOT,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=args.local_dataset,
        )

    base_run_dir = None
    if args.output_dir is None:
        default_run_dir = Path(
            teh_output_base_dir(
                args.dataset,
                timestamp,
                psych_dataset_split=psych_dataset_split,
                ablation=args.ablation,
            )
        )
        base_run_dir = str(_make_unique_run_dir(default_run_dir))
    elif len(participants_to_process) > 1:
        # Multiple participants with custom output_dir: use that as base directory
        base_run_dir = args.output_dir
        Path(base_run_dir).mkdir(parents=True, exist_ok=True)
    else:
        # Single participant with custom output_dir: use parent directory if it looks like a participant dir
        # Otherwise use the directory itself
        output_path = Path(args.output_dir)
        if output_path.name.startswith("participant_"):
            # It's a participant directory, use parent as base
            base_run_dir = str(output_path.parent)
        else:
            # It's already a base directory
            base_run_dir = args.output_dir
        Path(base_run_dir).mkdir(parents=True, exist_ok=True)

    cmd_log = _write_command_line_log(Path(base_run_dir))
    print(f"Wrote full command line to {cmd_log}")
    gated_meta: Optional[Dict[str, Any]] = None
    if t_pics_gated:
        explicit_source = str(getattr(args, "t_pics_gated_source", "") or "").strip() or None
        if explicit_source:
            _gated_entry = make_explicit_independent_source_entry(
                target=str(args.dataset),
                source=explicit_source,
            )
            _gated_rank1 = resolve_gated_transfer_source_rank1(
                independent=True,
                entry=_gated_entry,
                run_root=Path(base_run_dir),
            )
            gated_meta = gated_run_metadata(
                config_path=Path("(none)"),
                target=str(args.dataset),
                selected_source=_gated_entry.selected_source,
                selected_source_rank1=_gated_rank1,
                selector_name="explicit_t_pics_gated_source",
                extra={
                    "error_feedback_mode": _normalize_error_feedback_mode(
                        args.error_feedback_mode
                    ),
                    "global_iters": int(args.global_iters),
                    "n_iterations": int(args.n_iterations),
                    "explore_candidates": int(args.explore_candidates),
                    "explore_population_top_k": int(args.explore_population_top_k),
                    "independent_source_population": True,
                    "reused_frozen_g1_rank1": False,
                    "explicit_t_pics_gated_source": True,
                    "t_pics_gated_source": _gated_entry.selected_source,
                },
            )
        else:
            cfg_path = Path(str(args.t_pics_source_config))
            if not cfg_path.is_absolute():
                cfg_path = (_REPO_ROOT / cfg_path).resolve()
            _gated_cfg = load_frozen_transfer_config(cfg_path)
            _gated_entry = selected_source_for_target(str(args.dataset), config=_gated_cfg)
            _gated_rank1 = resolve_gated_transfer_source_rank1(
                independent=t_pics_gated_independent,
                entry=_gated_entry,
                run_root=Path(base_run_dir),
            )
            gated_meta = gated_run_metadata(
                config_path=_gated_cfg.path,
                target=str(args.dataset),
                selected_source=_gated_entry.selected_source,
                selected_source_rank1=_gated_rank1,
                selector_name=_gated_cfg.selector_name,
                extra={
                    "error_feedback_mode": _normalize_error_feedback_mode(
                        args.error_feedback_mode
                    ),
                    "global_iters": int(args.global_iters),
                    "n_iterations": int(args.n_iterations),
                    "explore_candidates": int(args.explore_candidates),
                    "explore_population_top_k": int(args.explore_population_top_k),
                    "independent_source_population": bool(t_pics_gated_independent),
                    "reused_frozen_g1_rank1": not bool(t_pics_gated_independent),
                },
            )
    metadata_log = _write_run_metadata(
        Path(base_run_dir),
        error_feedback_mode=args.error_feedback_mode,
        extra=gated_meta,
    )
    print(f"Wrote resolved run metadata to {metadata_log}")

    seed_program_path = _resolve_default_seed_program_path(args, 0)

    teh_client = maybe_wrap_openai_client(
        OpenAI(**client_kwargs) if client_kwargs else OpenAI(),
        getattr(args, "llm_reasoning_effort", None),
    )
    evo_iters = int(getattr(args, "dataset_prompt_evolution_iterations", 0) or 0)
    prefer_auto_llm_prompt = bool(
        evo_iters > 0 or getattr(args, "prefer_auto_llm_prompt", False) or t_pics_gated
    )
    # PICS v3 global-only G.1 with --prefer_auto_llm_prompt must fail closed (no
    # reference / merge fallback). Scoped to structure_aware_v3 so historical v1 /
    # preliminary-v2 G.1 keep their previous soft-fallback behavior. There is no
    # --require_auto_llm_prompt CLI flag; this wires setup_teh_run_prompts(...).
    g1_require_auto = bool(
        getattr(args, "prefer_auto_llm_prompt", False)
        and getattr(args, "global_phase", False)
        and int(getattr(args, "n_iterations", 0) or 0) == 0
        and str(getattr(args, "limited_data_protocol", "") or "") == "structure_aware_v3"
    )
    run_prompts_dir = setup_teh_run_prompts(
        Path(base_run_dir),
        args.dataset,
        Path(seed_program_path),
        client=teh_client,
        model_name=args.model_name,
        use_llm=not args.no_llm_prompt,
        base_prompt_path=args.base_prompt,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
        filter_mixed_gambles=mixed_gambles_gain_loss_only,
        psych_dataset_split=psych_dataset_split,
        example_char_budget=int(
            getattr(args, "dataset_prompt_example_char_budget", DEFAULT_EXAMPLE_CHAR_BUDGET)
        ),
        history_max_entries=int(
            getattr(args, "dataset_prompt_history_max_entries", DEFAULT_HISTORY_MAX_ENTRIES)
        ),
        max_examples=int(getattr(args, "dataset_prompt_max_examples", DEFAULT_MAX_EXAMPLES)),
        prefer_auto_llm_prompt=prefer_auto_llm_prompt,
        dataset_prompt_file=getattr(args, "dataset_prompt_file", None),
        split_ratio=float(args.split_ratio),
        split_seed=int(args.split_seed),
        speekenbrink_split=str(getattr(args, "speekenbrink_split", "chronological")),
        data_dir=getattr(args, "data_dir", None),
        limited_data_protocol=str(args.limited_data_protocol),
        limited_train_val=args.limited_train_val,
        max_observed_trials_per_participant=args.max_observed_trials_per_participant,
        require_auto_llm_prompt=bool(t_pics_gated or g1_require_auto),
        llm_decoding_seed=(int(args.split_seed) + 90_000) if t_pics_gated else None,
    )
    print(f"TEH run prompts directory: {run_prompts_dir}")
    seed_program_path = str(run_prompts_dir / "seed_program.py")
    prompt_meta_path = Path(run_prompts_dir) / "prompt_meta.json"
    prompt_meta: Dict[str, Any] = {}
    if prompt_meta_path.is_file():
        try:
            loaded_prompt_meta = json.loads(prompt_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_prompt_meta = {}
        if isinstance(loaded_prompt_meta, dict):
            prompt_meta = loaded_prompt_meta
            _merge_run_metadata(
                Path(base_run_dir),
                {
                    "prompt_provenance": {
                        "prompt_mode": prompt_meta.get("prompt_mode"),
                        "infer_prompt_path": prompt_meta.get("infer_prompt_path"),
                        "infer_prompt_sha256": prompt_meta.get("infer_prompt_sha256"),
                        "require_auto_llm_prompt": prompt_meta.get(
                            "require_auto_llm_prompt"
                        ),
                        "llm_decoding_seed": prompt_meta.get("llm_decoding_seed"),
                        "shared_by_g2_arms": bool(t_pics_gated),
                    }
                },
            )

    if t_pics_gated:
        wandb = init_gated_wandb_reporter(
            args=args,
            output_root=Path(base_run_dir),
            project=str(getattr(args, "wandb_project", None) or "teh_t_pics_v2"),
            selected_source=str(_gated_entry.selected_source),
            source_job_id=(
                "live" if t_pics_gated_independent else str(_gated_entry.job_id)
            ),
            source_rank1=Path(_gated_rank1),
            source_config_path=Path(
                (gated_meta or {}).get("t_pics_source_config") or "(none)"
            ),
            source_config_sha256=(gated_meta or {}).get("t_pics_source_config_sha256"),
            prompt_mode=prompt_meta.get("prompt_mode"),
            expected_participant_ids=[int(p) for p in participants_to_process],
            selected_dir=run_layout(Path(base_run_dir))["selected"],
            repo_root=_REPO_ROOT,
            enabled=not bool(args.no_log),
        )
        wandb_enabled = bool(getattr(wandb, "_enabled", False))
        _merge_run_metadata(
            Path(base_run_dir),
            {
                "wandb_run_id": getattr(wandb, "run_id", None),
                "wandb_project": getattr(wandb, "project", None),
                "wandb_group": "t_pics_gated_main",
                "wandb_job_type": "t_pics_gated",
            },
        )

    if evo_iters > 0:
        valid_for_prompt = load_valid_participant_ids_from_json(
            args.dataset,
            _REPO_ROOT,
            filter_mixed_gambles=mixed_gambles_gain_loss_only,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
        )
        n_dev = max(1, int(getattr(args, "dataset_prompt_dev_participants", 3)))
        if not valid_for_prompt:
            raise ValueError(
                "dataset_prompt_evolution requires valid participant ids, but none were found."
            )
        dev_ids = [int(valid_for_prompt[i]) for i in range(min(n_dev, len(valid_for_prompt)))]
        seed_text = Path(seed_program_path).read_text(encoding="utf-8")
        baseline_prompt = (run_prompts_dir / "infer_single_choice.txt").read_text(
            encoding="utf-8"
        )
        mutate_art = run_prompts_dir / "dataset_prompt_evolution" / "mutations"

        def _mutate(parent: str, role: str, feedback: str) -> str:
            return _mutate_dataset_prompt_via_llm(
                teh_client,
                args.model_name,
                parent,
                role,
                feedback,
                max_tokens=int(getattr(args, "llm_max_tokens", 2048) or 2048),
                artifacts_dir=mutate_art,
            )

        def _score(rendered: str) -> Dict[str, Any]:
            return _score_dataset_prompt_lightweight(
                rendered_prompt=rendered,
                dataset=args.dataset,
                client=teh_client,
                model_name=args.model_name,
                seed_program=seed_text,
                prompts_dir=run_prompts_dir,
                dev_participant_ids=dev_ids,
                eval_candidates=int(
                    getattr(args, "dataset_prompt_eval_candidates", 5) or 5
                ),
                split_ratio=float(args.split_ratio),
                split_seed=int(args.split_seed),
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
                psych_dataset_split=psych_dataset_split,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
                llm_max_tokens=int(getattr(args, "llm_max_tokens", 800) or 800),
                max_workers=int(getattr(args, "max_workers", 5) or 5),
            )

        evo_summary = maybe_run_dataset_prompt_evolution(
            iterations=evo_iters,
            prompts_dir=run_prompts_dir,
            baseline_prompt=baseline_prompt,
            population=int(getattr(args, "dataset_prompt_evolution_population", 3) or 3),
            children_per_round=int(
                getattr(args, "dataset_prompt_evolution_children", 3) or 3
            ),
            dev_participant_ids=dev_ids,
            eval_candidates=int(getattr(args, "dataset_prompt_eval_candidates", 5) or 5),
            seed_program=seed_text,
            mutate_fn=_mutate,
            score_fn=_score,
        )
        if evo_summary:
            print(
                "[TEH] Dataset-prompt evolution finished: "
                f"best={evo_summary.get('best_prompt_id')} "
                f"val={evo_summary.get('best_mean_val_fitness')}"
            )

    evolution_run_phase = "evolution" if args.phase == "evolution" else "all"

    global_elite_for_handoff: Optional[List[Tuple[Any, ...]]] = None
    has_initial_pool = bool(args.initial_pool_programs or args.initial_pool_dir)
    global_prompt_suffix = None
    source_best_loglik: Optional[float] = None
    global_client = teh_client
    t_pics_source = getattr(args, "t_pics_source", None)
    if t_pics_source:
        if seed_program_path is None:
            print("Error: --t_pics_source requires a seed program (--seed_path or dataset default).")
            if wandb is not None:
                wandb.finish()
            return
        try:
            source_pids, clamp_note = resolve_t_pics_source_participant_ids(
                source_dataset=str(t_pics_source),
                repo_root=_REPO_ROOT,
                participant_scope=args.participant_scope,
                single_participant_id=args.single_participant_id,
                range_start_ordinal=args.range_start_ordinal,
                range_end_ordinal=args.range_end_ordinal,
                all_max_participants=args.all_max_participants,
                participant_ordinals=args.ordinals,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
                split_ratio=args.split_ratio,
                split_seed=args.split_seed,
                psych_dataset_split=psych_dataset_split,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            if wandb is not None:
                wandb.finish()
            return
        if clamp_note:
            print(f"[T-PICS] {clamp_note}")
        try:
            source_best, source_best_loglik = _run_t_pics_source_population(
                source_dataset=str(t_pics_source),
                source_participants=source_pids,
                output_dir=Path(base_run_dir) / "source_population",
                seed_program_path=seed_program_path,
                args=args,
                client=global_client,
                wandb_module=wandb,
                psych_dataset_split=psych_dataset_split,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            if wandb is not None:
                wandb.finish()
            return
        args.global_prompt_source_program = str(source_best)
        args.global_prompt_source_dataset = str(t_pics_source)
    if args.global_prompt_source_program:
        try:
            global_prompt_suffix = _cross_task_source_suffix(
                source_dataset=str(args.global_prompt_source_dataset),
                program_path=str(args.global_prompt_source_program),
                args=args,
                psych_dataset_split=psych_dataset_split,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
                best_loglik=source_best_loglik,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            if wandb is not None:
                wandb.finish()
            return
        print(
            "[INFO] Stage G population prompt includes live source rank-1 "
            f"({args.global_prompt_source_dataset}: {args.global_prompt_source_program})"
        )
    if has_initial_pool:
        try:
            global_elite_for_handoff = _load_initial_pool_from_cli(
                program_paths=args.initial_pool_programs,
                pool_dirs=args.initial_pool_dir,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            if wandb is not None:
                wandb.finish()
            return
        print(
            f"Loaded {len(global_elite_for_handoff)} initial-pool program(s) for "
            f"participant handoff: {[p[3] for p in global_elite_for_handoff]}"
        )
    elif t_pics_gated:
        if seed_program_path is None:
            print("Error: --t_pics_gated_transfer requires a seed program (--seed_path or dataset default).")
            if wandb is not None:
                wandb.finish()
            return
        try:
            global_elite_for_handoff = _run_t_pics_gated_population_arms(
                args=args,
                run_root=Path(base_run_dir),
                participants=[int(p) for p in participants_to_process],
                seed_program_path=seed_program_path,
                client=global_client,
                wandb_module=wandb,
                psych_dataset_split=psych_dataset_split,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
                run_prompts_dir=str(run_prompts_dir),
            )
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as exc:
            print(f"Error: {exc}")
            if wandb is not None:
                if hasattr(wandb, "finish"):
                    try:
                        wandb.finish(completed=False, failure_reason=str(exc))
                    except Exception:
                        wandb.finish()
            raise
        selected_dir = run_layout(Path(base_run_dir))["selected"]
        selected_dir.mkdir(parents=True, exist_ok=True)
        base_run_dir = str(selected_dir)
        print(
            f"[T-PICS gated] downstream explore+person from retained pool only "
            f"({len(global_elite_for_handoff)} programs) -> {base_run_dir}"
        )
        if hasattr(wandb, "attach_context"):
            wandb.attach_context(
                expected_participant_ids=[int(p) for p in participants_to_process],
                selected_dir=Path(base_run_dir),
                n_iterations=int(args.n_iterations),
                explore_candidates=int(args.explore_candidates),
                output_root=Path(base_run_dir).parent,
            )
        if hasattr(wandb, "sync_progress_from_disk"):
            wandb.sync_progress_from_disk()
    elif args.global_phase and args.phase == "all" and args.dataset in _PARTICIPANT_DATASETS:
        if seed_program_path is None:
            print("Error: --global_phase requires a seed program (--seed_path or dataset default).")
            if wandb is not None:
                wandb.finish()
            return
        global_elite_for_handoff = run_global_evolution_phase(
            dataset=args.dataset,
            participants=[int(p) for p in participants_to_process],
            seed_program_path=seed_program_path,
            n_iterations=args.global_iters,
            n_candidates_per_iteration=args.n_candidates,
            fresh_n_candidates=args.fresh_n_candidates,
            sample_size=args.sample_size,
            sample_parents=args.sample_parents,
            sampled_parents_decay=args.sampled_parents_decay,
            elite_pool_size=args.elite_pool_size,
            model_name=args.model_name,
            client=global_client,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            data_path=args.data_path,
            filter_mixed_gambles=mixed_gambles_gain_loss_only,
            max_prompt_train_trials=args.max_prompt_train_trials,
            max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
            llm_max_tokens=args.llm_max_tokens,
            max_workers=args.max_workers,
            n_eval_seeds=args.n_eval_seeds,
            output_dir=Path(base_run_dir),
            save_artifacts=True,
            wandb_module=wandb,
            run_prompts_dir=str(run_prompts_dir),
            psych_dataset_split=psych_dataset_split,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
            max_parent_chars=args.max_parent_chars,
            warn_parent_truncation_ratio=args.warn_parent_truncation_ratio,
            early_stop_iters=args.early_stop_iters,
            hard_prompt_token_cap=args.hard_prompt_token_cap,
            strict_prompt_budget=args.strict_prompt_budget,
            prompt_token_estimator=args.prompt_token_estimator,
            prompt_debug=args.prompt_debug,
            prompt_debug_on_no_valid=args.prompt_debug_on_no_valid,
            prompt_debug_exit=args.prompt_debug_exit,
            evolution_selection_score=args.evolution_selection_score,
            max_error_prompt_chars=args.max_error_prompt_chars,
            error_feedback_mode=args.error_feedback_mode,
            max_observed_trials_per_participant=args.max_observed_trials_per_participant,
            limited_data_protocol=args.limited_data_protocol,
            speekenbrink_split=getattr(args, "speekenbrink_split", "chronological"),
            limited_train_val=args.limited_train_val,
            mdl_lambda=args.mdl_lambda,
            mem_trace=args.mem_trace,
            prompt_suffix=global_prompt_suffix,
        )

    explore_from_handoff_parents = resolve_explore_from_handoff_parents(
        has_initial_pool=has_initial_pool,
        explore_from_population_parents=bool(args.explore_from_population_parents),
    )
    explore_prompt_suffix = None
    if args.explore_prompt_source_program:
        try:
            explore_prompt_suffix = _cross_task_source_suffix(
                source_dataset=str(args.explore_prompt_source_dataset),
                program_path=str(args.explore_prompt_source_program),
                args=args,
                psych_dataset_split=psych_dataset_split,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            if wandb is not None:
                wandb.finish()
            return
        print(
            "[INFO] Stage E explore prompt includes live source rank-1 "
            f"({args.explore_prompt_source_dataset}: {args.explore_prompt_source_program})"
        )
    if (
        explore_from_handoff_parents
        and bool(args.explore_from_population_parents)
        and args.global_phase
        and not has_initial_pool
    ):
        top_k = int(args.explore_population_top_k)
        parent_desc = (
            "all population handoff programs"
            if top_k <= 0
            else f"the top-{top_k} population program(s) in global order"
        )
        n_handoff = 0 if global_elite_for_handoff is None else len(global_elite_for_handoff)
        print(
            f"[INFO] --explore_from_population_parents: "
            f"{int(args.explore_candidates)} explore candidates from {parent_desc} "
            f"(handoff pool size={n_handoff})."
        )

    if args.phase == "refine":
        if args.dataset not in _PARTICIPANT_DATASETS:
            print("Error: --phase refine requires a participant dataset.")
            if wandb is not None:
                wandb.finish()
            return
        if args.split_mode == "across_participants":
            print("Error: --phase refine does not support --split_mode across_participants.")
            if wandb is not None:
                wandb.finish()
            return
        refine_client = maybe_wrap_openai_client(
            OpenAI(**client_kwargs) if client_kwargs else OpenAI(),
            getattr(args, "llm_reasoning_effort", None),
        )
        try:
            run_loglik_refine_from_prev_experiment(
                dataset=args.dataset,
                client=refine_client,
                model_name=args.model_name,
                participants=[int(p) for p in participants_to_process],
                prev_exp_path=Path(args.prev_exp_path),
                output_dir=Path(base_run_dir),
                split_ratio=args.split_ratio,
                split_seed=args.split_seed,
                data_path=args.data_path,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
                n_iterations=args.refinement_iters,
                n_candidates_per_iteration=args.n_candidates,
                fresh_n_candidates=args.fresh_n_candidates,
                sample_size=args.sample_size,
                sample_parents=args.sample_parents,
                sampled_parents_decay=args.sampled_parents_decay,
                elite_pool_size=args.elite_pool_size,
                max_prompt_train_trials=args.max_prompt_train_trials,
                max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                llm_max_tokens=args.llm_max_tokens,
                max_workers=args.max_workers,
                n_eval_seeds=args.n_eval_seeds,
                wandb_module=wandb,
                parallel_participants=bool(args.parallel_participants),
                refinement_val_threshold=args.refinement_val_threshold,
                fitness_metric=args.fitness_metric,
                cpc18_official_mse=args.cpc18_official_mse,
                run_prompts_dir=str(run_prompts_dir),
                psych_dataset_split=psych_dataset_split,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
                max_parent_chars=args.max_parent_chars,
                warn_parent_truncation_ratio=args.warn_parent_truncation_ratio,
                early_stop_iters=args.early_stop_iters,
                hard_prompt_token_cap=args.hard_prompt_token_cap,
                strict_prompt_budget=args.strict_prompt_budget,
                prompt_token_estimator=args.prompt_token_estimator,
                max_error_prompt_chars=args.max_error_prompt_chars,
                error_feedback_mode=args.error_feedback_mode,
                max_observed_trials_per_participant=args.max_observed_trials_per_participant,
                limited_data_protocol=args.limited_data_protocol,
            speekenbrink_split=getattr(args, "speekenbrink_split", "chronological"),
                limited_train_val=args.limited_train_val,
                mdl_lambda=args.mdl_lambda,
            )
        finally:
            if wandb is not None:
                wandb.finish()
            return

    if (
        is_binary_loglik_dataset(args.dataset)
        and args.split_mode == "across_participants"
    ):
        selected_participants = list(participants_to_process)
        if len(selected_participants) < 2:
            print("Error: across_participants split requires at least 2 selected participants.")
            if wandb is not None:
                wandb.finish()
            return
        rng = np.random.default_rng(args.split_seed)
        shuffled = list(selected_participants)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * args.split_ratio)
        split_idx = max(1, min(split_idx, len(shuffled) - 1))
        train_participants = shuffled[:split_idx]
        test_participants = shuffled[split_idx:]
        print(
            f"split_mode={args.split_mode}, split_ratio={args.split_ratio:.3f}, "
            f"train_participants={len(train_participants)}, test_participants={len(test_participants)}"
        )

        from data_modules.psych101_binary import get_filtered_psych101_split

        filtered_psych = get_filtered_psych101_split(
            args.dataset,
            split=psych_dataset_split,
            local_dataset=args.local_dataset,
        )
        train_trials: List[Dict[str, Any]] = []
        test_trials: List[Dict[str, Any]] = []
        for pid in train_participants:
            exp = get_psych101_binary_experiment(
                args.dataset,
                pid,
                split=psych_dataset_split,
                local_dataset=args.local_dataset,
                filtered_split=filtered_psych,
            )
            train_trials.extend(experiment_to_trial_dicts(exp))
        for pid in test_participants:
            exp = get_psych101_binary_experiment(
                args.dataset,
                pid,
                split=psych_dataset_split,
                local_dataset=args.local_dataset,
                filtered_split=filtered_psych,
            )
            test_trials.extend(experiment_to_trial_dicts(exp))
        print(f"Across-participants trial counts: train={len(train_trials)}, test={len(test_trials)}")

        try:
            run_evolution(
                seed_program_path=seed_program_path,
                dataset=args.dataset,
                participant_id=0,
                data_path=args.data_path,
                num_blocks=getattr(args, "num_blocks", None),
                num_walls=getattr(args, "num_walls", None),
                agent_id=getattr(args, "agent_id", None),
                n_iterations=args.n_iterations,
                n_candidates_per_iteration=args.n_candidates,
                fresh_n_candidates=args.fresh_n_candidates,
                explore_candidates=args.explore_candidates,
                model_name=args.model_name,
                client_kwargs=client_kwargs if client_kwargs else None,
                output_dir=base_run_dir,
                wandb=wandb,
                n_eval_seeds=args.n_eval_seeds,
                sample_size=args.sample_size,
                sample_parents=args.sample_parents,
                sampled_parents_decay=args.sampled_parents_decay,
                elite_pool_size=args.elite_pool_size,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
                save_artifacts=True,
                all_data_mode=False,
                choice13k_experiment=None,
                fitness_metric=args.fitness_metric,
                split_ratio=args.split_ratio,
                split_seed=args.split_seed,
                choice13k_train_trials_override=train_trials,
                choice13k_test_trials_override=test_trials,
                choice13k_simple_logging=True,
                max_prompt_train_trials=args.max_prompt_train_trials,
                max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                llm_max_tokens=args.llm_max_tokens,
                gate_phase=args.gate_phase,
                run_phase=evolution_run_phase,
                refinement_phase=args.refinement_phase,
                refinement_iters=args.refinement_iters,
                refinement_val_threshold=args.refinement_val_threshold,
                max_workers=args.max_workers,
                run_prompts_dir=str(run_prompts_dir),
                psych_dataset_split=psych_dataset_split,
                ablation=args.ablation,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
                max_parent_chars=args.max_parent_chars,
                warn_parent_truncation_ratio=args.warn_parent_truncation_ratio,
                early_stop_iters=args.early_stop_iters,
                hard_prompt_token_cap=args.hard_prompt_token_cap,
                strict_prompt_budget=args.strict_prompt_budget,
                prompt_token_estimator=args.prompt_token_estimator,
                prompt_debug=args.prompt_debug,
                prompt_debug_on_no_valid=args.prompt_debug_on_no_valid,
                prompt_debug_exit=args.prompt_debug_exit,
                evolution_selection_score=args.evolution_selection_score,
                max_error_prompt_chars=args.max_error_prompt_chars,
                error_feedback_mode=args.error_feedback_mode,
                mem_trace=args.mem_trace,
                max_observed_trials_per_participant=args.max_observed_trials_per_participant,
                limited_data_protocol=args.limited_data_protocol,
            speekenbrink_split=getattr(args, "speekenbrink_split", "chronological"),
                limited_train_val=args.limited_train_val,
                explore_from_handoff_parents=explore_from_handoff_parents,
                explore_population_top_k=args.explore_population_top_k,
                mdl_lambda=args.mdl_lambda,
                explore_prompt_suffix=explore_prompt_suffix,
                explore_seed_candidates=int(args.explore_seed_candidates),
                t_pics_gated_transfer=bool(t_pics_gated),
            )
        finally:
            if wandb is not None:
                wandb.finish()
        return

    # participant_scope=all: process listed participants and save compact CSV outputs only (no per-participant artifacts)
    if args.dataset in _PARTICIPANT_DATASETS and args.participant_scope == "all":
        details_file = Path(base_run_dir) / "participants_details.csv"
        summary_file = Path(base_run_dir) / "summary.csv"
        details_loglik_file = Path(base_run_dir) / "participant_details_loglik.csv"
        summary_loglik_file = Path(base_run_dir) / "summary_loglik.csv"
        participant_details = []
        participant_details_loglik = []

        print(
            f"Participant scope=all using precomputed valid ids. "
            f"Total participants to process: {len(participants_to_process)}."
        )

        parallel_participants = bool(args.parallel_participants)
        participant_workers, candidate_workers_per_participant = _parallel_participant_pool_sizes(
            args.max_workers, args.n_candidates, parallel_participants
        )
        if parallel_participants:
            print(
                "[INFO] Parallel participants enabled: "
                f"participant_workers={participant_workers}, "
                f"candidate_workers_per_participant={candidate_workers_per_participant}"
            )

        def _run_all_mode_participant(participant_id: int) -> Dict[str, Any]:
            participant_start = datetime.now()
            participant_summary = run_evolution(
                seed_program_path=seed_program_path,
                dataset=args.dataset,
                participant_id=participant_id,
                data_path=args.data_path,
                num_blocks=getattr(args, "num_blocks", None),
                num_walls=getattr(args, "num_walls", None),
                agent_id=getattr(args, "agent_id", None),
                n_iterations=args.n_iterations,
                n_candidates_per_iteration=args.n_candidates,
                fresh_n_candidates=args.fresh_n_candidates,
                explore_candidates=args.explore_candidates,
                model_name=args.model_name,
                client_kwargs=client_kwargs if client_kwargs else None,
                output_dir=base_run_dir,
                wandb=wandb,
                n_eval_seeds=args.n_eval_seeds,
                sample_size=args.sample_size,
                sample_parents=args.sample_parents,
                sampled_parents_decay=args.sampled_parents_decay,
                elite_pool_size=args.elite_pool_size,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
                save_artifacts=False,
                all_data_mode=True,
                choice13k_experiment=None,
                fitness_metric=args.fitness_metric,
                split_ratio=args.split_ratio,
                split_seed=args.split_seed,
                max_prompt_train_trials=args.max_prompt_train_trials,
                max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                llm_max_tokens=args.llm_max_tokens,
                cpc18_official_mse=args.cpc18_official_mse,
                gate_phase=args.gate_phase,
                run_phase=evolution_run_phase,
                refinement_phase=args.refinement_phase,
                refinement_iters=args.refinement_iters,
                refinement_val_threshold=args.refinement_val_threshold,
                max_workers=candidate_workers_per_participant,
                global_elite_parents=global_elite_for_handoff,
                run_prompts_dir=str(run_prompts_dir),
                psych_dataset_split=psych_dataset_split,
                ablation=args.ablation,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
                max_parent_chars=args.max_parent_chars,
                warn_parent_truncation_ratio=args.warn_parent_truncation_ratio,
                early_stop_iters=args.early_stop_iters,
                hard_prompt_token_cap=args.hard_prompt_token_cap,
                strict_prompt_budget=args.strict_prompt_budget,
                prompt_token_estimator=args.prompt_token_estimator,
                prompt_debug=args.prompt_debug,
                prompt_debug_on_no_valid=args.prompt_debug_on_no_valid,
                prompt_debug_exit=args.prompt_debug_exit,
                evolution_selection_score=args.evolution_selection_score,
                max_error_prompt_chars=args.max_error_prompt_chars,
                error_feedback_mode=args.error_feedback_mode,
                mem_trace=args.mem_trace,
                max_observed_trials_per_participant=args.max_observed_trials_per_participant,
                limited_data_protocol=args.limited_data_protocol,
            speekenbrink_split=getattr(args, "speekenbrink_split", "chronological"),
                limited_train_val=args.limited_train_val,
                explore_from_handoff_parents=explore_from_handoff_parents,
                explore_population_top_k=args.explore_population_top_k,
                mdl_lambda=args.mdl_lambda,
                explore_prompt_suffix=explore_prompt_suffix,
                explore_seed_candidates=int(args.explore_seed_candidates),
                t_pics_gated_transfer=bool(t_pics_gated),
            )
            runtime_sec = (datetime.now() - participant_start).total_seconds()
            details_row = {
                "participant_id": participant_id,
                "train_fitness": participant_summary.get("train_fitness"),
                "test_fitness": participant_summary.get("test_fitness"),
                "total_runtime": runtime_sec,
                "seed_program_train_fitness": participant_summary.get("seed_program_train_fitness"),
                "seed_program_test_fitness": participant_summary.get("seed_program_test_fitness"),
            }
            loglik_row: Dict[str, Any] = {
                "participant_id": participant_id,
                "train_loglik": participant_summary.get("train_loglik"),
            }
            if _uses_train_val_test_loglik_split(
                args.dataset, args.fitness_metric, cpc18_official_mse=args.cpc18_official_mse
            ):
                if participant_summary.get("val_loglik") is not None:
                    loglik_row["val_loglik"] = participant_summary.get("val_loglik")
                loglik_row["test_loglik"] = participant_summary.get("test_loglik")
                _gated_ll = _gated_loglik_for_participant_summary(participant_summary)
                if _gated_ll is not None:
                    loglik_row["gated_test_loglik"] = _gated_ll
            else:
                loglik_row["test_loglik"] = participant_summary.get("test_loglik")
            return {
                "participant_id": participant_id,
                "details_row": details_row,
                "loglik_row": loglik_row,
            }

        try:
            if parallel_participants:
                all_mode_results: List[Dict[str, Any]] = []
                with ThreadPoolExecutor(max_workers=participant_workers) as pool:
                    futures = {
                        pool.submit(_run_all_mode_participant, int(pid)): int(pid)
                        for pid in participants_to_process
                    }
                    for fut in tqdm(
                        as_completed(futures), total=len(futures), desc="Participants"
                    ):
                        all_mode_results.append(fut.result())
                all_mode_results.sort(key=lambda r: int(r["participant_id"]))
            else:
                all_mode_results = []
                for participant_id in tqdm(participants_to_process, desc="Participants"):
                    all_mode_results.append(_run_all_mode_participant(int(participant_id)))

            participant_details = [r["details_row"] for r in all_mode_results]
            participant_details_loglik = [r["loglik_row"] for r in all_mode_results]

            with open(details_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "participant_id",
                        "train_fitness",
                        "test_fitness",
                        "total_runtime",
                        "seed_program_train_fitness",
                        "seed_program_test_fitness",
                    ],
                )
                writer.writeheader()
                writer.writerows(_round_floats_for_csv_rows(participant_details))

            avg_train_fitness = float(np.mean([d["train_fitness"] for d in participant_details]))
            avg_test_fitness = float(np.mean([d["test_fitness"] for d in participant_details]))
            with open(summary_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["num_of_participants", "avg_train_fitness", "avg_test_fitness"],
                )
                writer.writeheader()
                writer.writerow(
                    _round_floats_for_csv_row(
                        {
                            "num_of_participants": len(participant_details),
                            "avg_train_fitness": avg_train_fitness,
                            "avg_test_fitness": avg_test_fitness,
                        }
                    )
                )

            _ll_fields = ["participant_id", "train_loglik"]
            if _uses_train_val_test_loglik_split(
                args.dataset, args.fitness_metric, cpc18_official_mse=args.cpc18_official_mse
            ):
                _ll_fields.extend(["val_loglik", "test_loglik", "gated_test_loglik"])
            else:
                _ll_fields.append("test_loglik")
            with open(details_loglik_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_ll_fields)
                writer.writeheader()
                writer.writerows(_round_floats_for_csv_rows(participant_details_loglik))

            train_loglik_values = [
                d["train_loglik"] for d in participant_details_loglik if d["train_loglik"] is not None
            ]
            test_loglik_values = [
                d["test_loglik"] for d in participant_details_loglik if d["test_loglik"] is not None
            ]
            val_loglik_values = [
                d["val_loglik"] for d in participant_details_loglik if d.get("val_loglik") is not None
            ]
            gated_test_loglik_values = [
                d["gated_test_loglik"]
                for d in participant_details_loglik
                if d.get("gated_test_loglik") is not None
            ]
            avg_train_loglik = float(np.mean(train_loglik_values)) if train_loglik_values else None
            avg_test_loglik = float(np.mean(test_loglik_values)) if test_loglik_values else None
            _sum_ll_row = {
                "num_of_participants": len(participant_details_loglik),
                "avg_train_loglik": avg_train_loglik,
                "avg_test_loglik": avg_test_loglik,
            }
            _sum_ll_fields = ["num_of_participants", "avg_train_loglik", "avg_test_loglik"]
            if is_binary_loglik_dataset(args.dataset):
                _sum_ll_fields.extend(["avg_val_loglik", "avg_gated_test_loglik"])
                _sum_ll_row["avg_val_loglik"] = (
                    float(np.mean(val_loglik_values)) if val_loglik_values else None
                )
                _sum_ll_row["avg_gated_test_loglik"] = (
                    float(np.mean(gated_test_loglik_values)) if gated_test_loglik_values else None
                )
            with open(summary_loglik_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_sum_ll_fields)
                writer.writeheader()
                writer.writerow(_round_floats_for_csv_row(_sum_ll_row))
        finally:
            if wandb is not None:
                wandb.finish()
        return
    
    # Initialize participants summary (list for CSV)
    participants_summary = []
    participants_loglik_summary = []
    # Determine summary file location (use base_run_dir if available, otherwise use output_dir or its parent)
    if base_run_dir is not None:
        summary_file = Path(base_run_dir) / "participants_summary.csv"
    elif args.output_dir is not None:
        output_path = Path(args.output_dir)
        if output_path.name.startswith("participant_"):
            # It's a participant directory, use parent
            summary_file = output_path.parent / "participants_summary.csv"
        else:
            # Use the directory itself
            summary_file = output_path / "participants_summary.csv"
    else:
        # Auto-generated single participant - will be determined after first run
        summary_file = None
    summary_loglik_file = (
        Path(base_run_dir) / "summary_loglik.csv"
        if base_run_dir is not None
        else None
    )
    details_loglik_file = (
        Path(base_run_dir) / "participant_details_loglik.csv"
        if base_run_dir is not None
        else None
    )
    
    # Handle gridworld: ROTE code setting (episode-based, test split, prefix=20, ensemble from prefix only)
    if args.dataset == "gridworld" and args.loop_mode != "sequential":
        num_blocks_arg = getattr(args, 'num_blocks', None)
        num_walls_arg = getattr(args, 'num_walls', None)
        agent_id_arg = getattr(args, 'agent_id', 0)
        if num_blocks_arg is None or num_walls_arg is None:
            print("Error: For gridworld (ROTE) provide --num_blocks and --num_walls.")
            if wandb is not None:
                wandb.finish()
            return
        seed_path = args.seed_path
        if seed_path is None:
            seed_path = find_template_program_for_gridworld(num_blocks_arg, num_walls_arg, agent_id_arg)
            if seed_path is None:
                print(f"Warning: No template found for num_blocks={num_blocks_arg}, num_walls={num_walls_arg}, agent_id={agent_id_arg}; using default.")
                seed_path = "persona_code_example/vanilla.py"
            else:
                print(f"Auto-detected seed program: {seed_path}")
        output_dir = base_run_dir if base_run_dir else f"{output_root_dir}/gridworld/non_strict/{run_dir_name}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        client = maybe_wrap_openai_client(
            OpenAI(**client_kwargs) if client_kwargs else OpenAI(),
            getattr(args, "llm_reasoning_effort", None),
        )
        episode_results, mean_test_acc = run_evolution_gridworld_rote_episodes(
            seed_program_path=seed_path,
            data_path=args.data_path,
            num_blocks=num_blocks_arg,
            num_walls=num_walls_arg,
            agent_id=agent_id_arg,
            num_episodes=getattr(args, 'num_episodes', 10),
            K=args.n_candidates,
            N=args.n_iterations,
            n_candidates_per_iteration=max(1, args.n_candidates // 2),
            model_name=args.model_name,
            client=client,
            output_dir=output_dir,
            wandb=wandb,
            max_workers=args.max_workers,
        )
        if wandb is not None:
            wandb.log({"gridworld_mean_episode_test_acc": mean_test_acc})
            wandb.finish()
        return

    # Handle gridworld_ensemble: same as gridworld multi-agent but with run_evolution_gridworld_ensemble
    if args.dataset == "gridworld_ensemble":
        num_blocks_arg = getattr(args, 'num_blocks', None)
        num_walls_arg = getattr(args, 'num_walls', None)
        agent_id_arg = getattr(args, 'agent_id', None)
        if (num_blocks_arg is not None and num_walls_arg is not None and
            args.loop_mode != "sequential" and
            (args.num_agents_to_sample > 1 or agent_id_arg is None)):
            print(f"\n{'='*80}")
            print(f"Processing gridworld_ensemble: {args.num_agents_to_sample} agent types for problem: num_blocks={num_blocks_arg}, num_walls={num_walls_arg}")
            print(f"{'='*80}")
            if agent_id_arg is not None and args.num_agents_to_sample == 1:
                agent_types_to_process = [agent_id_arg]
            else:
                agent_types_to_process = list(range(args.num_agents_to_sample))
            for agent_id in tqdm(agent_types_to_process, desc="Agent types"):
                print(f"\n{'='*80}\nProcessing agent type {agent_id} (gridworld_ensemble)\n{'='*80}")
                if args.seed_path is None:
                    detected_seed_path = find_template_program_for_gridworld(num_blocks_arg, num_walls_arg, agent_id)
                    if detected_seed_path is None:
                        print(f"Warning: Could not auto-detect template for agent_id={agent_id}, skipping...")
                        continue
                    agent_seed_path = detected_seed_path
                    print(f"Auto-detected seed program: {agent_seed_path}")
                else:
                    agent_seed_path = args.seed_path
                if base_run_dir is not None:
                    agent_output_dir = os.path.join(base_run_dir, f"agent_{agent_id}")
                else:
                    mode = "non_strict"
                    agent_output_dir = f"{output_root_dir}/gridworld_ensemble/{mode}/{run_dir_name}/agent_{agent_id}"
                agent_summary = run_evolution_gridworld_ensemble(
                    seed_program_path=agent_seed_path,
                    participant_id=agent_id,
                    data_path=args.data_path,
                    num_blocks=num_blocks_arg,
                    num_walls=num_walls_arg,
                    agent_id=agent_id,
                    n_iterations=args.n_iterations,
                    n_candidates_per_iteration=args.n_candidates,
                    model_name=args.model_name,
                    client_kwargs=client_kwargs if client_kwargs else None,
                    output_dir=agent_output_dir,
                    wandb=wandb,
                    n_eval_seeds=args.n_eval_seeds,
                    sample_size=args.sample_size,
                    top_k=getattr(args, 'top_k', 0),
                    max_workers=args.max_workers,
                    ablation=args.ablation,
                )
                if agent_summary is not None and summary_file is not None:
                    participants_summary.append({
                        'agent_id': agent_id,
                        'num_blocks': num_blocks_arg,
                        'num_walls': num_walls_arg,
                        'train_acc': agent_summary.get('train_acc'),
                        'test_acc': agent_summary.get('test_acc'),
                    })
                    with open(summary_file, 'w', newline='') as f:
                        fieldnames = ['agent_id', 'num_blocks', 'num_walls', 'train_acc', 'test_acc']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(_round_floats_for_csv_rows(participants_summary))
                    print(f"\nSummary updated: {summary_file}")
            if wandb is not None:
                wandb.finish()
            return
    
    # Handle sequential mode for gridworld or gridworld_ensemble
    if (args.dataset == "gridworld" or args.dataset == "gridworld_ensemble") and args.loop_mode == "sequential":
        all_problem_configs = get_all_problem_configs()
        num_agent_types = 10  # Total number of agent types
        total_configs = len(all_problem_configs)
        use_ensemble = args.dataset == "gridworld_ensemble"
        out_subdir = "gridworld_ensemble" if use_ensemble else "gridworld"
        
        # Calculate which config and agent to use for each epoch
        def get_config_and_agents_for_epoch(epoch_idx):
            """Get (num_blocks, num_walls, agent_indices_list) for a given epoch index."""
            if epoch_idx >= total_configs:
                return None, None, None
            num_blocks, num_walls = all_problem_configs[epoch_idx]
            # Use first num_agents_to_sample agent types
            agent_indices = list(range(min(args.num_agents_to_sample, num_agent_types)))
            return num_blocks, num_walls, agent_indices
        
        # Process each epoch
        epochs_to_process = min(args.num_epochs, total_configs)
        for epoch in range(epochs_to_process):
            num_blocks, num_walls, agent_indices = get_config_and_agents_for_epoch(epoch)
            if num_blocks is None:
                break
            
            # Process all agent types for this epoch
            for agent_id in agent_indices:
                print(f"\n{'='*80}")
                print(f"Processing epoch {epoch+1}/{epochs_to_process} - Problem: num_blocks={num_blocks}, num_walls={num_walls}, Agent: {agent_id}" + (" (gridworld_ensemble)" if use_ensemble else ""))
                print(f"{'='*80}")
                
                # Determine seed program path for this agent type
                if args.seed_path is None:
                    # Auto-detect template program
                    detected_seed_path = find_template_program_for_gridworld(num_blocks, num_walls, agent_id)
                    if detected_seed_path is None:
                        print(f"Warning: Could not auto-detect template program for num_blocks={num_blocks}, num_walls={num_walls}, agent_id={agent_id}")
                        print(f"Expected location: persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}/")
                        print("Skipping this agent type...")
                        continue
                    epoch_seed_path = detected_seed_path
                    print(f"Auto-detected seed program: {epoch_seed_path}")
                else:
                    epoch_seed_path = args.seed_path
                
                # Construct output directory
                if base_run_dir is not None:
                    participant_output_dir = os.path.join(base_run_dir, f"epoch_{epoch}", f"agent_{agent_id}")
                else:
                    mode = "non_strict"
                    participant_output_dir = f"{output_root_dir}/{out_subdir}/{mode}/{run_dir_name}/epoch_{epoch}/agent_{agent_id}"
                
                if use_ensemble:
                    participant_summary = run_evolution_gridworld_ensemble(
                        seed_program_path=epoch_seed_path,
                        participant_id=agent_id,
                        data_path=args.data_path,
                        num_blocks=num_blocks,
                        num_walls=num_walls,
                        agent_id=agent_id,
                        n_iterations=args.n_iterations,
                        n_candidates_per_iteration=args.n_candidates,
                        model_name=args.model_name,
                        client_kwargs=client_kwargs if client_kwargs else None,
                        output_dir=participant_output_dir,
                        wandb=wandb,
                        n_eval_seeds=args.n_eval_seeds,
                        sample_size=args.sample_size,
                        top_k=getattr(args, 'top_k', 0),
                        max_workers=args.max_workers,
                        ablation=args.ablation,
                    )
                else:
                    participant_summary = run_evolution(
                        seed_program_path=epoch_seed_path,
                        dataset=args.dataset,
                        participant_id=agent_id,
                        data_path=args.data_path,
                        num_blocks=num_blocks,
                        num_walls=num_walls,
                        agent_id=agent_id,
                        n_iterations=args.n_iterations,
                        n_candidates_per_iteration=args.n_candidates,
                        fresh_n_candidates=args.fresh_n_candidates,
                        explore_candidates=args.explore_candidates,
                        model_name=args.model_name,
                        client_kwargs=client_kwargs if client_kwargs else None,
                        output_dir=participant_output_dir,
                        wandb=wandb,
                        n_eval_seeds=args.n_eval_seeds,
                        sample_size=args.sample_size,
                        sample_parents=args.sample_parents,
                        sampled_parents_decay=args.sampled_parents_decay,
                        elite_pool_size=args.elite_pool_size,
                        filter_mixed_gambles=mixed_gambles_gain_loss_only,
                        fitness_metric=args.fitness_metric,
                        split_ratio=args.split_ratio,
                        split_seed=args.split_seed,
                        max_prompt_train_trials=args.max_prompt_train_trials,
                        max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                        llm_max_tokens=args.llm_max_tokens,
                        max_parent_chars=args.max_parent_chars,
                        warn_parent_truncation_ratio=args.warn_parent_truncation_ratio,
                        gate_phase=args.gate_phase,
                        run_phase=evolution_run_phase,
                        refinement_phase=args.refinement_phase,
                        refinement_iters=args.refinement_iters,
                        refinement_val_threshold=args.refinement_val_threshold,
                        max_workers=args.max_workers,
                        early_stop_iters=args.early_stop_iters,
                        hard_prompt_token_cap=args.hard_prompt_token_cap,
                        strict_prompt_budget=args.strict_prompt_budget,
                        prompt_token_estimator=args.prompt_token_estimator,
                        prompt_debug=args.prompt_debug,
                        prompt_debug_on_no_valid=args.prompt_debug_on_no_valid,
                        prompt_debug_exit=args.prompt_debug_exit,
                        evolution_selection_score=args.evolution_selection_score,
                        max_error_prompt_chars=args.max_error_prompt_chars,
                        error_feedback_mode=args.error_feedback_mode,
                        ablation=args.ablation,
                        mem_trace=args.mem_trace,
                max_observed_trials_per_participant=args.max_observed_trials_per_participant,
                limited_data_protocol=args.limited_data_protocol,
            speekenbrink_split=getattr(args, "speekenbrink_split", "chronological"),
                limited_train_val=args.limited_train_val,
                explore_from_handoff_parents=explore_from_handoff_parents,
                explore_population_top_k=args.explore_population_top_k,
                mdl_lambda=args.mdl_lambda,
                explore_prompt_suffix=explore_prompt_suffix,
                explore_seed_candidates=int(args.explore_seed_candidates),
                t_pics_gated_transfer=bool(t_pics_gated),
                    )
                
                # Update summary (build row with only CSV columns; participant_summary uses 'participant_id' key)
                if participant_summary is not None and summary_file is not None:
                    participants_summary.append({
                        'epoch': epoch,
                        'num_blocks': num_blocks,
                        'num_walls': num_walls,
                        'agent_id': agent_id,
                        'train_acc': participant_summary.get('train_acc'),
                        'test_acc': participant_summary.get('test_acc'),
                    })
                    # Write CSV file
                    with open(summary_file, 'w', newline='') as f:
                        fieldnames = ['epoch', 'num_blocks', 'num_walls', 'agent_id', 'train_acc', 'test_acc']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(_round_floats_for_csv_rows(participants_summary))
                    print(f"\nSummary updated: {summary_file}")
    else:
        # Original logic for choice13k or random mode
        # Run evolution for each participant
        parallel_participants = bool(args.parallel_participants)
        participant_workers, candidate_workers_per_participant = _parallel_participant_pool_sizes(
            args.max_workers, args.n_candidates, parallel_participants
        )
        if parallel_participants:
            print(
                "[INFO] Parallel participants enabled: "
                f"participant_workers={participant_workers}, "
                f"candidate_workers_per_participant={candidate_workers_per_participant}"
            )

        def _participant_output_dir(participant_id: int) -> Optional[str]:
            if base_run_dir is not None:
                return os.path.join(base_run_dir, f"participant_{participant_id}")
            return args.output_dir

        def _ensure_summary_paths_from_participant_dir(participant_output_dir: Optional[str]) -> None:
            nonlocal summary_file, summary_loglik_file, details_loglik_file
            if summary_file is not None or participant_output_dir is None:
                return
            output_path = Path(participant_output_dir)
            if output_path.name.startswith("participant_"):
                summary_file = output_path.parent / "participants_summary.csv"
                summary_loglik_file = output_path.parent / "summary_loglik.csv"
                details_loglik_file = output_path.parent / "participant_details_loglik.csv"
            else:
                summary_file = output_path / "participants_summary.csv"
                summary_loglik_file = output_path / "summary_loglik.csv"
                details_loglik_file = output_path / "participant_details_loglik.csv"

        def _loglik_row_from_summary(participant_summary: Dict[str, Any]) -> Dict[str, Any]:
            _ps_log: Dict[str, Any] = {
                "participant_id": participant_summary.get("participant_id"),
                "train_loglik": participant_summary.get("train_loglik"),
            }
            if _uses_train_val_test_loglik_split(
                args.dataset, args.fitness_metric, cpc18_official_mse=args.cpc18_official_mse
            ):
                if participant_summary.get("val_loglik") is not None:
                    _ps_log["val_loglik"] = participant_summary.get("val_loglik")
                _ps_log["test_loglik"] = participant_summary.get("test_loglik")
                _gated_ll = _gated_loglik_for_participant_summary(participant_summary)
                if _gated_ll is not None:
                    _ps_log["gated_test_loglik"] = _gated_ll
            else:
                _ps_log["test_loglik"] = participant_summary.get("test_loglik")
            return _ps_log

        def _write_main_loop_experiment_csvs() -> None:
            """Rewrite experiment-level CSVs from participants_summary / participants_loglik_summary."""
            if summary_file is None or not participants_summary:
                return
            fieldnames = _csv_fieldnames_from_rows(participants_summary)
            with open(summary_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(_round_floats_for_csv_rows(participants_summary))
            print(f"\nParticipants summary updated: {summary_file}")

            if args.participant_scope not in ("range", "ordinals"):
                return
            if details_loglik_file is not None:
                _det_ll_fields = ["participant_id", "train_loglik"]
                if _uses_train_val_test_loglik_split(
                    args.dataset, args.fitness_metric, cpc18_official_mse=args.cpc18_official_mse
                ):
                    _det_ll_fields.extend(["val_loglik", "test_loglik", "gated_test_loglik"])
                else:
                    _det_ll_fields.append("test_loglik")
                with open(details_loglik_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=_det_ll_fields)
                    writer.writeheader()
                    writer.writerows(_round_floats_for_csv_rows(participants_loglik_summary))
            if summary_loglik_file is not None:
                train_ll_vals = [
                    d["train_loglik"]
                    for d in participants_loglik_summary
                    if d["train_loglik"] is not None
                ]
                test_ll_vals = [
                    d["test_loglik"]
                    for d in participants_loglik_summary
                    if d["test_loglik"] is not None
                ]
                val_ll_vals = [
                    d["val_loglik"]
                    for d in participants_loglik_summary
                    if d.get("val_loglik") is not None
                ]
                gated_test_ll_vals = [
                    d["gated_test_loglik"]
                    for d in participants_loglik_summary
                    if d.get("gated_test_loglik") is not None
                ]
                _agg_ll = {
                    "num_of_participants": len(participants_loglik_summary),
                    "avg_train_loglik": float(np.mean(train_ll_vals)) if train_ll_vals else None,
                    "avg_test_loglik": float(np.mean(test_ll_vals)) if test_ll_vals else None,
                }
                _agg_ll_fields = ["num_of_participants", "avg_train_loglik", "avg_test_loglik"]
            if _uses_train_val_test_loglik_split(
                args.dataset, args.fitness_metric, cpc18_official_mse=args.cpc18_official_mse
            ):
                _agg_ll_fields.extend(["avg_val_loglik", "avg_gated_test_loglik"])
                _agg_ll["avg_val_loglik"] = (
                    float(np.mean(val_ll_vals)) if val_ll_vals else None
                )
                _agg_ll["avg_gated_test_loglik"] = (
                    float(np.mean(gated_test_ll_vals)) if gated_test_ll_vals else None
                )
                with open(summary_loglik_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=_agg_ll_fields)
                    writer.writeheader()
                    writer.writerow(_round_floats_for_csv_row(_agg_ll))

        def _append_main_loop_summaries(participant_summary: Optional[Dict[str, Any]]) -> None:
            if participant_summary is None or summary_file is None:
                return
            with _SHARED_EXPERIMENT_CSV_LOCK:
                participants_summary.append(participant_summary)
                if args.participant_scope in ("range", "ordinals"):
                    participants_loglik_summary.append(_loglik_row_from_summary(participant_summary))
                _write_main_loop_experiment_csvs()
            if t_pics_gated and hasattr(wandb, "sync_progress_from_disk"):
                wandb.sync_progress_from_disk()

        def _flush_main_loop_csvs_from_completed(
            completed: Dict[int, Optional[Dict[str, Any]]],
            participant_ids_ordered: List[int],
        ) -> None:
            """Rebuild experiment CSVs from finished participants (parallel mode; main thread only)."""
            with _SHARED_EXPERIMENT_CSV_LOCK:
                participants_summary.clear()
                participants_loglik_summary.clear()
                for pid in participant_ids_ordered:
                    summary = completed.get(int(pid))
                    if summary is None:
                        continue
                    participants_summary.append(summary)
                    if args.participant_scope in ("range", "ordinals"):
                        participants_loglik_summary.append(_loglik_row_from_summary(summary))
                _write_main_loop_experiment_csvs()
            if t_pics_gated and hasattr(wandb, "sync_progress_from_disk"):
                wandb.sync_progress_from_disk()

        def _run_main_loop_participant(participant_id: int) -> Optional[Dict[str, Any]]:
            print(f"\n{'='*80}")
            print(f"Processing participant {participant_id}")
            print(f"{'='*80}")
            participant_output_dir = _participant_output_dir(participant_id)
            if not parallel_participants:
                _ensure_summary_paths_from_participant_dir(participant_output_dir)
            if t_pics_gated and participant_output_dir is not None:
                if participant_run_is_complete(
                    Path(participant_output_dir),
                    expected_n_iterations=int(args.n_iterations),
                    expected_explore_candidates=int(args.explore_candidates),
                ):
                    print(
                        f"[T-PICS gated] skip complete participant {participant_id} -> "
                        f"{participant_output_dir}"
                    )
                    try:
                        payload = json.loads(
                            (Path(participant_output_dir) / "results.json").read_text(
                                encoding="utf-8"
                            )
                        )
                    except (OSError, json.JSONDecodeError):
                        payload = {}
                    best = payload.get("overall_best_train") or {}
                    summary = {
                        "participant_id": int(participant_id),
                        "train_loglik": best.get("train_loglik"),
                        "val_loglik": best.get("val_loglik"),
                        "test_loglik": best.get("test_loglik"),
                        "train_fitness": best.get("train_loglik", best.get("train_fitness")),
                        "test_fitness": best.get("test_loglik", best.get("test_fitness")),
                    }
                    return summary

            seed_program_path_local = _resolve_default_seed_program_path(args, participant_id)
            if seed_program_path_local is None:
                if args.dataset in ("gridworld", "gridworld_ensemble"):
                    num_blocks = getattr(args, "num_blocks", None)
                    num_walls = getattr(args, "num_walls", None)
                    agent_id = getattr(args, "agent_id", None)
                    if num_blocks is not None and num_walls is not None and agent_id is not None:
                        detected_seed_path = find_template_program_for_gridworld(
                            num_blocks, num_walls, agent_id
                        )
                        if detected_seed_path is None:
                            print(
                                f"Error: Could not auto-detect template program for "
                                f"num_blocks={num_blocks}, num_walls={num_walls}, agent_id={agent_id}"
                            )
                            print(
                                f"Expected location: persona_code_example/gridworld/"
                                f"num_blocks{num_blocks}_num_walls{num_walls}/"
                            )
                            return None
                        seed_program_path_local = detected_seed_path
                        print(f"Auto-detected seed program: {seed_program_path_local}")
                    else:
                        print(
                            "Error: For gridworld/gridworld_ensemble without --seed_path, "
                            "must provide --num_blocks, --num_walls, and --agent_id"
                        )
                        return None

            if args.dataset == "gridworld_ensemble":
                num_blocks = getattr(args, "num_blocks", None)
                num_walls = getattr(args, "num_walls", None)
                agent_id = getattr(args, "agent_id", participant_id)
                if num_blocks is None or num_walls is None:
                    print("Error: For gridworld_ensemble must provide --num_blocks and --num_walls")
                    return None
                return run_evolution_gridworld_ensemble(
                    seed_program_path=seed_program_path_local,
                    participant_id=participant_id,
                    data_path=args.data_path,
                    num_blocks=num_blocks,
                    num_walls=num_walls,
                    agent_id=agent_id,
                    n_iterations=args.n_iterations,
                    n_candidates_per_iteration=args.n_candidates,
                    model_name=args.model_name,
                    client_kwargs=client_kwargs if client_kwargs else None,
                    output_dir=participant_output_dir,
                    wandb=wandb,
                    n_eval_seeds=args.n_eval_seeds,
                    sample_size=args.sample_size,
                    top_k=getattr(args, "top_k", 0),
                    max_workers=candidate_workers_per_participant,
                    ablation=args.ablation,
                )

            return run_evolution(
                seed_program_path=seed_program_path_local,
                dataset=args.dataset,
                participant_id=participant_id,
                data_path=args.data_path,
                num_blocks=getattr(args, "num_blocks", None),
                num_walls=getattr(args, "num_walls", None),
                agent_id=getattr(args, "agent_id", None),
                n_iterations=args.n_iterations,
                n_candidates_per_iteration=args.n_candidates,
                fresh_n_candidates=args.fresh_n_candidates,
                explore_candidates=args.explore_candidates,
                model_name=args.model_name,
                client_kwargs=client_kwargs if client_kwargs else None,
                output_dir=participant_output_dir,
                wandb=wandb,
                n_eval_seeds=args.n_eval_seeds,
                sample_size=args.sample_size,
                sample_parents=args.sample_parents,
                sampled_parents_decay=args.sampled_parents_decay,
                elite_pool_size=args.elite_pool_size,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
                fitness_metric=args.fitness_metric,
                split_ratio=args.split_ratio,
                split_seed=args.split_seed,
                max_prompt_train_trials=args.max_prompt_train_trials,
                max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                llm_max_tokens=args.llm_max_tokens,
                cpc18_official_mse=args.cpc18_official_mse,
                gate_phase=args.gate_phase,
                run_phase=evolution_run_phase,
                refinement_phase=args.refinement_phase,
                refinement_iters=args.refinement_iters,
                refinement_val_threshold=args.refinement_val_threshold,
                max_workers=candidate_workers_per_participant,
                global_elite_parents=global_elite_for_handoff,
                run_prompts_dir=str(run_prompts_dir),
                psych_dataset_split=psych_dataset_split,
                ablation=args.ablation,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
                max_parent_chars=args.max_parent_chars,
                warn_parent_truncation_ratio=args.warn_parent_truncation_ratio,
                early_stop_iters=args.early_stop_iters,
                hard_prompt_token_cap=args.hard_prompt_token_cap,
                strict_prompt_budget=args.strict_prompt_budget,
                prompt_token_estimator=args.prompt_token_estimator,
                prompt_debug=args.prompt_debug,
                prompt_debug_on_no_valid=args.prompt_debug_on_no_valid,
                prompt_debug_exit=args.prompt_debug_exit,
                evolution_selection_score=args.evolution_selection_score,
                max_error_prompt_chars=args.max_error_prompt_chars,
                error_feedback_mode=args.error_feedback_mode,
                mem_trace=args.mem_trace,
                max_observed_trials_per_participant=args.max_observed_trials_per_participant,
                limited_data_protocol=args.limited_data_protocol,
                speekenbrink_split=getattr(args, "speekenbrink_split", "chronological"),
                limited_train_val=args.limited_train_val,
                explore_from_handoff_parents=explore_from_handoff_parents,
                explore_population_top_k=args.explore_population_top_k,
                mdl_lambda=args.mdl_lambda,
                explore_prompt_suffix=explore_prompt_suffix,
                explore_seed_candidates=int(args.explore_seed_candidates),
                t_pics_gated_transfer=bool(t_pics_gated),
            )

        try:
            skip_all_people = bool(
                t_pics_gated
                and selected_stage_is_complete(
                    Path(base_run_dir),
                    [int(p) for p in participants_to_process],
                    expected_n_iterations=int(args.n_iterations),
                    expected_explore_candidates=int(args.explore_candidates),
                )
            )
            if skip_all_people:
                print(
                    f"[T-PICS gated] skip complete selected-arm person stage -> {base_run_dir}"
                )
            elif parallel_participants:
                if summary_file is None and base_run_dir is not None:
                    summary_file = Path(base_run_dir) / "participants_summary.csv"
                    summary_loglik_file = Path(base_run_dir) / "summary_loglik.csv"
                    details_loglik_file = Path(base_run_dir) / "participant_details_loglik.csv"
                elif summary_file is None and participants_to_process:
                    _ensure_summary_paths_from_participant_dir(
                        _participant_output_dir(int(participants_to_process[0]))
                    )
                pids_ordered = sorted(int(p) for p in participants_to_process)
                completed_by_pid: Dict[int, Optional[Dict[str, Any]]] = {}
                with ThreadPoolExecutor(max_workers=participant_workers) as pool:
                    futures = {
                        pool.submit(_run_main_loop_participant, int(pid)): int(pid)
                        for pid in participants_to_process
                    }
                    for fut in tqdm(
                        as_completed(futures), total=len(futures), desc="Participants"
                    ):
                        pid = futures[fut]
                        completed_by_pid[pid] = fut.result()
                        _flush_main_loop_csvs_from_completed(completed_by_pid, pids_ordered)
            else:
                for participant_id in tqdm(participants_to_process, desc="Participants"):
                    participant_summary = _run_main_loop_participant(int(participant_id))
                    _append_main_loop_summaries(participant_summary)
            if t_pics_gated and selected_stage_is_complete(
                Path(base_run_dir),
                [int(p) for p in participants_to_process],
                expected_n_iterations=int(args.n_iterations),
                expected_explore_candidates=int(args.explore_candidates),
            ):
                marker = Path(base_run_dir) / "STAGE_COMPLETE.json"
                marker.write_text(
                    json.dumps(
                        {
                            "stage": "selected",
                            "n_iterations": int(args.n_iterations),
                            "explore_candidates": int(args.explore_candidates),
                            "participant_ids": [int(p) for p in participants_to_process],
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if hasattr(wandb, "publish_final"):
                    wandb.publish_final()
        finally:
            if wandb is not None:
                if t_pics_gated:
                    exc = sys.exc_info()[1]
                    wandb.finish(
                        completed=(
                            exc is None
                            and selected_stage_is_complete(
                                Path(base_run_dir),
                                [int(p) for p in participants_to_process],
                                expected_n_iterations=int(args.n_iterations),
                                expected_explore_candidates=int(args.explore_candidates),
                            )
                        ),
                        failure_reason=None if exc is None else str(exc),
                    )
                else:
                    wandb.finish()


if __name__ == "__main__":
    main()
