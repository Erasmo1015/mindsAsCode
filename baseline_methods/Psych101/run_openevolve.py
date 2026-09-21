#!/usr/bin/env python3
# See documentation at baseline_methods/Psych101/docs/Documentation_Openevolve.md
# OpenEvolve intentionally keeps vanilla/original prompt behavior independent from TEH prompt-generation.

"""
Usage: 

python baseline_methods/Psych101/run_openevolve.py \
  --dataset 1peterson2021using \
  --psych_dataset_split train \
  --participant_scope range \
  --range_start_ordinal 0 \
  --range_end_ordinal 29 \
  --api_base http://localhost:8000/v1 \
  --n_iterations 350 \
  --parallel_participants 1 \
  --parallel_evaluations 10 \
  --limited_data_protocol structure_aware_v3 \
  --limited_train_val 40 \
  --max_prompt_train_trials 60 \
  --hard_prompt_token_cap 14000 \
  --max_model_len 16384

OpenEvolve baseline for Psych-101 / ICLR datasets (vanilla prompt, full OpenEvolve machinery).

Uses reference_repos/openevolve as a library. Does NOT use TEH/PICS prompt engineering.
Dataset description is the registry task text (Choice13k / mixed-gambles vanilla files
only for those two schemas). The choose() interface is a short mechanically generated
Bernoulli or categorical contract. Official island inspirations are prompt-only
contextual examples (not co-parents) and are dropped before observed examples if
the 14000-token Qwen input ceiling is exceeded.

Evolution optimizes trial-pooled mean log-likelihood on the observed train+val union
(combined_score). Per-split train_loglik and val_loglik are logged separately. Test
log-likelihood is computed only after evolution on the participant's
best-by-observed-union program.

OpenEvolve still uses islands / MAP-Elites / archive for parent selection. The LLM
sees one current mutable parent. Official optional contextual blocks (previous
attempts, artifacts, top/diverse programs, island inspirations) default OFF for
ICLR lean runs (matched to EMNLP production: num_top=0, num_diverse=0,
include_artifacts=False, enable_artifacts=False, no previous-attempt history).
They remain CLI-toggleable. Observed train+val examples fill the prompt. The
complete MAP-Elites database is never serialized.

ICLR freeze (matched to PICS v3 g5e50p30): --n_iterations 350, --parallel_participants 1,
--parallel_evaluations 10, SA40 (--limited_data_protocol structure_aware_v3
--limited_train_val 40; v1 `structure_aware` / preliminary-v2 remain available for
replay), first-30 ordinals from teh_datasets.yaml, --hard_prompt_token_cap 14000,
--max_model_len 16384, --llm_max_tokens 1024, --max_prompt_train_trials 60 (prompt
display only; fitness uses the complete retained train+val union). Do not copy the
obsolete 10×10 people×eval concurrency, 32k/30k context, or PICS --max_workers 100.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    PSYCH101_BINARY_DATASETS,
    PSYCH101_LEGACY_ALIASES,
    get_psych101_binary_experiment,
    hf_id_for_psych_dataset_split,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
    split_psych_experiment,
)
from data_modules.external import (
    EXTERNAL_DATASET_META,
    is_external_dataset,
)
# utils.teh.* here: dataset registry + participant-id paths only (not TEH prompts/runtime).
from utils.psych101_openevolve_pool import WORKER_VANILLA as _WORKER_VANILLA
from utils.teh.limited_data_protocol import (
    LIMITED_DATA_MANIFEST_CSV_FILENAME,
    LIMITED_DATA_MANIFEST_JSONL_FILENAME,
    LIMITED_DATA_PROTOCOL_OFF,
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
    add_limited_data_cli_arguments,
    append_limited_data_manifest_jsonl,
    load_participant_limited_splits,
    normalize_limited_data_protocol,
    resolve_limited_data_budget,
    rewrite_limited_data_csv_from_jsonl,
    should_persist_limited_data_manifest,
)
from utils.teh.participant_ids import load_valid_participant_ids
from utils.teh.baseline_wandb_completion import (
    apply_wandb_payload,
    merge_wandb_payload,
    wandb_completion_fields,
)
from utils.teh.teh_datasets import (
    PARTICIPANT_DATASETS,
    dataset_task_description,
    emnlp_ordinal_range,
    is_binary_loglik_dataset,
    is_categorical_output_dataset,
    is_mixed_gambles_dataset,
)
from utils.teh_psych.categorical_eval import evaluate_categorical_program

try:
    from openevolve import OpenEvolve
    from openevolve.config import Config
    from openevolve.process_parallel import ProcessParallelController
except ImportError:
    # Allow --help / CPU tests without the gitignored checkout. Production main()
    # still fail-fasts via require_openevolve_checkout() and never clones.
    OpenEvolve = object  # type: ignore[misc,assignment]
    Config = object  # type: ignore[misc,assignment]
    ProcessParallelController = object  # type: ignore[misc,assignment]

WANDB_PROJECT = "openevolve"
CHOICE13K_LOGLIK_EPS = 1e-9
# Clipped per-trial p is in [eps, 1-eps], so any valid mean loglik is
# >= log(eps) ≈ -20.72. Official OpenEvolve checkpoint JSON uses json.dump
# (IEEE -inf becomes non-standard `-Infinity`; the visualizer then sanitizes
# it to None, which would drop combined_score and let error=0.0 win). Use a
# finite failure floor strictly worse than any clipped observed mean.
FAILED_COMBINED_SCORE = math.log(CHOICE13K_LOGLIK_EPS) - 1.0
DEFAULT_SEED_PATH = _REPO_ROOT / "persona_code_example" / "openevolve_vanilla" / "choices13k.py"
DEFAULT_CATEGORICAL_SEED_PATH = _REPO_ROOT / "persona_code_example" / "teh" / "categorical_uniform.py"
# Deprecated runner may still load these; ICLR OE never reads them as # Task text.
_LEGACY_VANILLA_TASK_FILES = frozenset(
    (
        (
            _REPO_ROOT
            / "prompts"
            / "openevolve_vanilla"
            / "choices13k"
            / "infer_single_choice.txt"
        ).resolve(),
        (
            _REPO_ROOT
            / "prompts"
            / "openevolve_vanilla"
            / "mixed_gambles"
            / "infer_single_choice.txt"
        ).resolve(),
    )
)
ICLR_FROZEN_N_ITERATIONS = 350
ICLR_FROZEN_PARALLEL_PARTICIPANTS = 1
# Throughput only (does not change candidate count). Match historical H100 OE jobs
# that used --parallel_evaluations 10; keep people=1 to avoid 10×10 request storms.
ICLR_FROZEN_PARALLEL_EVALUATIONS = 10
ICLR_FROZEN_LIMITED_DATA_PROTOCOL = LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE
ICLR_V2_LIMITED_DATA_PROTOCOL = "structure_aware_v2"
ICLR_V3_LIMITED_DATA_PROTOCOL = "structure_aware_v3"
# Matched PICS v3 g5e50p30: structure_aware_v3 + 16k/14k/1024 + first-30 ordinals.
ICLR_DEFAULT_LIMITED_DATA_PROTOCOL = ICLR_V3_LIMITED_DATA_PROTOCOL
ICLR_FROZEN_LIMITED_TRAIN_VAL = 40
ICLR_FROZEN_MAX_PROMPT_TRAIN_TRIALS = 60  # T-PICS prompt-display cap on train+val union
ICLR_FROZEN_SPLIT_RATIO = 0.6
ICLR_FROZEN_SPLIT_SEED = 0
ICLR_FROZEN_LLM_MAX_TOKENS = 1024
ICLR_FROZEN_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
# Official OpenEvolve PromptConfig defaults are 3/2/True; ICLR lean freeze turns
# those optional contextual blocks OFF (EMNLP production used 1/0/False; we go
# fully off so prompts stay parent+task+examples only).
ICLR_FROZEN_NUM_DIVERSE_PROGRAMS = 0
ICLR_FROZEN_NUM_TOP_PROGRAMS = 0
ICLR_FROZEN_INCLUDE_ARTIFACTS = False
ICLR_FROZEN_ENABLE_ARTIFACTS = False
ICLR_FROZEN_INCLUDE_PREVIOUS_ATTEMPTS = False
ICLR_FROZEN_LOG_PROMPTS = False
# Disk hygiene defaults (implementation detail; not scientific knobs).
# WARNING keeps OE's logger quiet; 0 checkpoint interval = only at run end.
ICLR_FROZEN_LOG_LEVEL = "WARNING"
ICLR_FROZEN_CHECKPOINT_INTERVAL = 0  # 0 => resolve to n_iterations (single final dump)
ICLR_FROZEN_COMPACT_OE_ARTIFACTS = True
# Active freeze = PICS v3 16k-class pair (14000 input + 1024 out ≤ 16384).
ICLR_FROZEN_INPUT_TOKEN_CEILING = 14000
ICLR_FROZEN_VLLM_MAX_MODEL_LEN = 16384
# Historical unmatched OE context (pre-g5e50p30); do not launch as paper baseline.
ICLR_HISTORICAL_32K_INPUT_TOKEN_CEILING = 30000
ICLR_HISTORICAL_32K_VLLM_MAX_MODEL_LEN = 32768
# Alias kept for older imports/tests that named the 14k cap "preliminary-v2".
ICLR_PRELIMINARY_V2_INPUT_TOKEN_CEILING = ICLR_FROZEN_INPUT_TOKEN_CEILING
ICLR_FROZEN_MAX_PROGRAM_CHARS = 10000  # audited production bound for packing tests
EXPECTED_OPENEVOLVE_GIT_SHA = "411fb59c886c18704caaffb611e17cf9e7d824d2"
# Token-budget example ladder used only after all optional contextual blocks are omitted.
ICLR_PROMPT_EXAMPLE_REDUCTION_CAPS = (60, 40, 30, 20, 10, 5)
ICLR_RANGE_ORDINAL_CLI_DEFAULTS = (0, 29)
QWEN_TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
_ARTIFACTS_MARKER = "# Evaluation artifacts (official OpenEvolve; parent artifacts, not co-parents)"
_PREVIOUS_MARKER = "# Previous attempts (official OpenEvolve; contextual examples, not co-parents)"
_TOP_MARKER = "# Top performing programs (official OpenEvolve; contextual examples, not co-parents)"
_DIVERSE_MARKER = "# Diverse programs (official OpenEvolve; contextual examples, not co-parents)"
_INSPIRATIONS_MARKER = "# Inspiration programs (official OpenEvolve; contextual examples, not co-parents)"
_OPTIONAL_DROP_MARKERS = (
    (_INSPIRATIONS_MARKER, "inspirations"),
    (_DIVERSE_MARKER, "diverse"),
    (_TOP_MARKER, "top"),
    (_PREVIOUS_MARKER, "previous_attempts"),
    (_ARTIFACTS_MARKER, "artifacts"),
)
_QWEN_TOKENIZER = None
_QWEN_TOKENIZER_FAILED = False
# Five-new aliases (loaders/prompts/seeds are registered via PARTICIPANT_DATASETS;
# this set is documentation only and does not gate CLI or evaluation).
FOCUS_DATASETS = frozenset(
    {
        "14kool2016when",
        "13schulz2020finding",
        "bergert_nosofsky_2007",
        "guan_2020_stopping",
        "steyvers_2009_bandit",
    }
)
# Tracked source of truth for the choose() contract. No disk interface templates.
CHOOSE_API_BERNOULLI = """\
Implement def choose(problem, history).
- problem: dict of current-trial task features.
- history: list of previously realized trials.
- return: a Python float in [0, 1] equal to P(action=1).
- Action mapping: action 0 is problem["option_keys"][0]; action 1 is problem["option_keys"][1].
Do not return a dict.
"""
CHOOSE_API_CATEGORICAL = """\
Implement def choose(problem, history).
- problem: dict of current-trial task features.
- history: list of previously realized trials.
- Valid action ids: [opt["action"] for opt in problem["options"]] if problem["options"] is present, else problem["option_keys"].
{n_actions_line}
- return: dict[int, float] mapping every valid action id to a finite non-negative probability; values must sum to 1.
Do not return a single Bernoulli float when there are more than two actions.
"""

_SHARED_CSV_LOCK = threading.Lock()
_WANDB_LOG_LOCK = threading.Lock()
_PARTICIPANT_THREAD_CTX = threading.local()
_OE_CONTROLLER_HOOKED = False

_ORIG_BUILD_PROMPT = None
_ORIG_GENERATE_WITH_CONTEXT = None
_ORIG_EVALUATE_PROGRAM = None
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


class RequiredPromptOverflowError(RuntimeError):
    """Raised when system/task/interface/current-parent cannot fit the input ceiling."""


@dataclass
class TruncationState:
    estimated_tokens: int = 0
    examples_available: int = 0
    examples_included: int = 0
    prompt_trial_count: int = 0
    prompt_train_trials: int = 0
    prompt_val_trials: int = 0
    artifacts_requested: int = 0
    artifacts_kept: int = 0
    previous_attempts_requested: int = 0
    previous_attempts_kept: int = 0
    inspirations_requested: int = 0
    inspirations_kept: int = 0
    top_programs_requested: int = 0
    top_programs_kept: int = 0
    diverse_programs_requested: int = 0
    diverse_programs_kept: int = 0
    trim_reason: str = ""
    steps: List[str] = field(default_factory=list)


def _set_thread_participant_ctx(ctx: Dict[str, Any]) -> None:
    _PARTICIPANT_THREAD_CTX.data = ctx


def _get_thread_participant_ctx() -> Dict[str, Any]:
    ctx = getattr(_PARTICIPANT_THREAD_CTX, "data", None)
    if ctx is None:
        raise RuntimeError("participant context not set on this thread")
    return ctx


def _clear_thread_participant_ctx() -> None:
    if hasattr(_PARTICIPANT_THREAD_CTX, "data"):
        del _PARTICIPANT_THREAD_CTX.data


def _run_openevolve(oe: OpenEvolve, iterations: int) -> Any:
    """
    Run OpenEvolve evolution in this thread.

    OpenEvolve registers SIGINT/SIGTERM handlers in controller.run(), which raises
    ValueError outside the main thread. With --parallel_participants > 1, skip signal
    setup in worker threads (graceful shutdown via Ctrl+C still works on main thread).
    """
    import signal

    if threading.current_thread() is threading.main_thread():
        return asyncio.run(oe.run(iterations=iterations))

    _orig_signal = signal.signal

    def _noop_signal(signum: int, handler: Any) -> Any:
        return None

    signal.signal = _noop_signal
    try:
        return asyncio.run(oe.run(iterations=iterations))
    finally:
        signal.signal = _orig_signal


def _effective_psych_dataset_split(dataset: str, psych_dataset_split: str) -> str:
    if is_mixed_gambles_dataset(dataset) or is_external_dataset(dataset):
        return DEFAULT_PSYCH_DATASET_SPLIT
    return normalize_psych_dataset_split(psych_dataset_split)


def resolve_openevolve_seed_and_prompt(
    dataset: str,
    *,
    seed_path: Optional[str] = None,
    base_prompt: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Pick seed; keep CLI prompt path (TEH/PICS reference files are not OE vanilla)."""
    alias = normalize_psych101_dataset_alias(dataset)
    seed = Path(seed_path) if seed_path else DEFAULT_SEED_PATH
    prompt = Path(base_prompt) if base_prompt else Path()
    using_default_seed = Path(seed).resolve() == DEFAULT_SEED_PATH.resolve()

    if using_default_seed and is_categorical_output_dataset(alias):
        seed = DEFAULT_CATEGORICAL_SEED_PATH
    return seed, prompt


def dataset_n_actions(dataset: str) -> Optional[int]:
    alias = normalize_psych101_dataset_alias(dataset)
    if is_external_dataset(alias):
        n = EXTERNAL_DATASET_META.get(alias, {}).get("n_actions")
        return int(n) if n is not None else None
    spec = PSYCH101_BINARY_DATASETS.get(alias) or {}
    if spec.get("n_actions") is not None:
        return int(spec["n_actions"])
    if is_categorical_output_dataset(alias):
        return None
    return 2


def vanilla_dataset_description(dataset: str, *, base_prompt: Optional[str] = None) -> str:
    """Registered task_description for all 15. Never loads TEH/PICS or vanilla infer files."""
    if base_prompt:
        custom = Path(base_prompt)
        if custom.is_file() and custom.resolve() not in _LEGACY_VANILLA_TASK_FILES:
            return custom.read_text(encoding="utf-8")
    return dataset_task_description(dataset)


def choose_api_text(*, categorical: bool, n_actions: Optional[int] = None) -> str:
    """Evaluator-derived interface contract (Bernoulli vs categorical).

    Tracked in-runner strings are the only source of truth. No disk interface
    templates are read.
    """
    text = CHOOSE_API_CATEGORICAL if categorical else CHOOSE_API_BERNOULLI
    if categorical:
        n = int(n_actions) if n_actions else 0
        if n > 0:
            n_line = (
                f"This dataset has K={n} actions; return a key for each valid id "
                f"(typically 0..{n - 1})."
            )
        else:
            n_line = (
                "K is len(problem['options']) if present, else len(problem['option_keys'])."
            )
        text = text.replace("{n_actions_line}", n_line)
    else:
        text = text.replace("{n_actions_line}", "")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def vanilla_llm_user_prefix(dataset: str, *, base_prompt: Optional[str] = None) -> str:
    """Task + API sections actually sent to the LLM (before program/metrics/examples)."""
    alias = normalize_psych101_dataset_alias(dataset)
    categorical = is_categorical_output_dataset(alias)
    return "\n".join(
        [
            "# Task (vanilla — no TEH prompt engineering)",
            vanilla_dataset_description(alias, base_prompt=base_prompt).strip(),
            "",
            "# API",
            choose_api_text(categorical=categorical, n_actions=dataset_n_actions(alias)),
            "",
            "# Output format (required for parser)",
            _FULL_REWRITE_OUTPUT_FORMAT.strip(),
        ]
    )


def apply_iclr_frozen_range_ordinals(args: Any) -> None:
    """Replace leftover CLI placeholder ranges with teh_datasets.yaml ranges."""
    if getattr(args, "participant_scope", "range") != "range":
        return
    if getattr(args, "ordinals", None) is not None:
        return
    start = int(getattr(args, "range_start_ordinal", 0))
    end = int(getattr(args, "range_end_ordinal", 29))
    # (0, 29) = current argparse default; (0, 49) = pre-g5e50p30 leftover.
    if (start, end) not in {ICLR_RANGE_ORDINAL_CLI_DEFAULTS, (0, 49)}:
        return
    yaml_start, yaml_end = emnlp_ordinal_range(args.dataset)
    args.range_start_ordinal = yaml_start
    args.range_end_ordinal = yaml_end


def openevolve_checkout_sha(root: Path) -> str:
    """Read HEAD of an existing OpenEvolve git checkout. Never clones or fetches."""
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise FileNotFoundError(
            f"cannot read git HEAD at {root}: {err or 'git rev-parse failed'}"
        )
    sha = proc.stdout.strip()
    if not sha:
        raise FileNotFoundError(f"empty git HEAD at {root}")
    return sha


def require_openevolve_checkout(
    *,
    root: Optional[Path] = None,
    expected_sha: str = EXPECTED_OPENEVOLVE_GIT_SHA,
) -> str:
    """Fail-fast before a production run. Does not clone, install, or modify the checkout."""
    checkout = Path(root) if root is not None else _OPENVOLVE_ROOT
    pkg = checkout / "openevolve"
    if not checkout.is_dir() or not pkg.is_dir():
        raise SystemExit(
            f"OpenEvolve checkout missing at {checkout}. "
            f"Clone https://github.com/codelion/openevolve.git to that path and "
            f"checkout {expected_sha}. This runner will not clone or install it."
        )
    try:
        sha = openevolve_checkout_sha(checkout)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"OpenEvolve checkout at {checkout} is not a git worktree ({exc}). "
            f"Expected commit {expected_sha}. This runner will not modify it."
        ) from exc
    if sha != expected_sha:
        raise SystemExit(
            f"OpenEvolve at {checkout} is {sha}, expected {expected_sha}. "
            "Checkout that commit before a production run. This runner will not "
            "clone, fetch, or modify the dependency."
        )
    return sha


def iclr_frozen_argv(dataset: str, *, api_base: str = "http://localhost:8000/v1") -> List[str]:
    alias = normalize_psych101_dataset_alias(dataset)
    start, end = emnlp_ordinal_range(alias)
    return [
        "--dataset",
        alias,
        "--psych_dataset_split",
        "train",
        "--participant_scope",
        "range",
        "--range_start_ordinal",
        str(start),
        "--range_end_ordinal",
        str(end),
        "--split_ratio",
        str(ICLR_FROZEN_SPLIT_RATIO),
        "--split_seed",
        str(ICLR_FROZEN_SPLIT_SEED),
        "--n_iterations",
        str(ICLR_FROZEN_N_ITERATIONS),
        "--parallel_participants",
        str(ICLR_FROZEN_PARALLEL_PARTICIPANTS),
        "--parallel_evaluations",
        str(ICLR_FROZEN_PARALLEL_EVALUATIONS),
        "--num_diverse_programs",
        str(ICLR_FROZEN_NUM_DIVERSE_PROGRAMS),
        "--num_top_programs",
        str(ICLR_FROZEN_NUM_TOP_PROGRAMS),
        "--no-include_artifacts",
        "--no-enable_artifacts",
        "--no-include_previous_attempts",
        "--no-cascade_evaluation",
        "--no-use_llm_feedback",
        "--limited_data_protocol",
        ICLR_DEFAULT_LIMITED_DATA_PROTOCOL,
        "--limited_train_val",
        str(ICLR_FROZEN_LIMITED_TRAIN_VAL),
        "--max_prompt_train_trials",
        str(ICLR_FROZEN_MAX_PROMPT_TRAIN_TRIALS),
        "--model",
        ICLR_FROZEN_MODEL,
        "--llm_max_tokens",
        str(ICLR_FROZEN_LLM_MAX_TOKENS),
        "--hard_prompt_token_cap",
        str(ICLR_FROZEN_INPUT_TOKEN_CEILING),
        "--max_model_len",
        str(ICLR_FROZEN_VLLM_MAX_MODEL_LEN),
        "--api_base",
        api_base,
    ]


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pooled_observed_loglik(
    train_ll: float,
    val_ll: float,
    n_train: int,
    n_val: int,
) -> float:
    """Trial-count-weighted mean loglik on the observed train+val union."""
    n_tr = max(0, int(n_train))
    n_vl = max(0, int(n_val))
    denom = n_tr + n_vl
    if denom <= 0:
        return float(train_ll)
    if n_vl <= 0:
        return float(train_ll)
    val_f = _safe_float(val_ll)
    if val_f is None:
        return float(train_ll)
    return (n_tr * float(train_ll) + n_vl * val_f) / denom


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


_ENKAVI_ORACLE_KEYS = ("probe_in_set",)
_KOOL_STAGE1_UNOBSERVED_KEYS = (
    "planet",
    "alien_options",
    "stage1_action",
    "spaceship",
    "reward",
    "treasure",
)
_KOOL_CURRENT_OUTCOME_KEYS = ("reward", "treasure")


def sanitize_trial_for_program(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Drop oracle/future fields from the problem dict passed to choose() / compact text.

    History entries are unchanged (those are already-observed outcomes).
    """
    nt = dict(trial)
    problem = dict(trial.get("problem") or {})
    for key in _ENKAVI_ORACLE_KEYS:
        problem.pop(key, None)
    alias = str(problem.get("dataset_alias") or "")
    schema = str(problem.get("schema_type") or "")
    if schema == "kool_twostep" or alias == "14kool2016when":
        stage = int(problem.get("stage") or 1)
        drop = _KOOL_STAGE1_UNOBSERVED_KEYS if stage == 1 else _KOOL_CURRENT_OUTCOME_KEYS
        for key in drop:
            problem.pop(key, None)
    nt["problem"] = problem
    return nt


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
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    speekenbrink_split: str = "chronological",
    return_manifest: bool = False,
):
    """Train/val/test with optional sparse/limited-data protocol (same as TEH/MLE)."""
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
        train_trials = [sanitize_trial_for_program(t) for t in train_trials]
        val_trials = [sanitize_trial_for_program(t) for t in val_trials]
        test_trials = [sanitize_trial_for_program(t) for t in test_trials]
        return train_trials, val_trials, test_trials, audit, manifest
    train_trials = [sanitize_trial_for_program(t) for t in train_trials]
    val_trials = [sanitize_trial_for_program(t) for t in val_trials]
    test_trials = [sanitize_trial_for_program(t) for t in test_trials]
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


def evaluate_loglik(
    choose_fn: Callable,
    trials: List[Dict[str, Any]],
    *,
    dataset: Optional[str] = None,
) -> Dict[str, float]:
    if not trials:
        return {"avg_loglik": float("-inf"), "total": 0, "errors": 0}
    alias = normalize_psych101_dataset_alias(dataset) if dataset else None
    if alias and is_categorical_output_dataset(alias):
        return evaluate_categorical_program(choose_fn, trials)
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
    """Prompt-display subsample from train+val union only (never test).

    Fitness/evaluation still uses the complete retained train+val lists. This
    cap matches gated T-PICS ``--max_prompt_train_trials`` on the combined
    observed union. When the pool already has ``<= max_trials`` observations,
    every example is kept (no per-problem subsample). Per-problem grouping
    applies only when more than ``max_trials`` exist.
    """
    pool = list(train_trials) + list(val_trials)
    train_ids = {id(t) for t in train_trials}
    if max_trials <= 0 or not pool:
        return [], 0, 0

    orig_index = {id(t): i for i, t in enumerate(pool)}

    def _sort_chronological(selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(selected, key=lambda t: orig_index[id(t)])

    if len(pool) <= max_trials:
        selected = _sort_chronological(pool)
        n_train = sum(1 for t in selected if id(t) in train_ids)
        return selected, n_train, len(selected) - n_train

    rng = np.random.default_rng(subsample_seed)

    if max_trials_per_problem <= 0:
        idx = rng.choice(len(pool), size=max_trials, replace=False)
        selected = _sort_chronological([pool[int(i)] for i in idx])
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

    if len(out) < max_trials:
        remaining = [t for t in pool if id(t) not in used_ids]
        if remaining:
            n_pick = min(max_trials - len(out), len(remaining))
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


def _compact_history_entry(h: Dict[str, Any]) -> str:
    """Prior-trial compact token. Includes observed reward/value; never current-trial labels."""
    bits = [f"a{int(h.get('action', 0))}"]
    if h.get("stage") is not None:
        bits.append(f"st{h.get('stage')}")
    if h.get("position") is not None:
        bits.append(f"pos={h.get('position')}")
    fb = h.get("feedback")
    reward = h.get("reward")
    if isinstance(fb, dict):
        if "correct_category" in fb:
            bits.append(f"corr={fb.get('correct_category')}")
            bits.append(f"ok={int(bool(fb.get('is_correct')))}")
        if fb.get("planet") is not None:
            bits.append(f"pl={fb.get('planet')}")
    elif fb is not None:
        bits.append(f"r{fb}")
        if reward is not None and reward != fb:
            bits.append(f"r{reward}")
    elif reward is not None:
        bits.append(f"r{reward}")
    if h.get("value") is not None:
        bits.append(f"v{h.get('value')}")
    if h.get("planet") is not None and not (isinstance(fb, dict) and fb.get("planet") is not None):
        bits.append(f"pl={h.get('planet')}")
    return ",".join(bits)


def _compact_history(history: List[Dict[str, Any]], max_items: int = 6) -> str:
    if not history:
        return "[]"
    tail = history[-max_items:]
    parts = [_compact_history_entry(h) if isinstance(h, dict) else str(h) for h in tail]
    return "[" + ";".join(parts) + "]"


def format_trial_compact(trial: Dict[str, Any], split_label: str) -> str:
    trial = sanitize_trial_for_program(trial)
    problem = trial.get("problem") or {}
    alias = str(problem.get("dataset_alias") or "")
    schema = str(problem.get("schema_type") or "")
    if "gamble_A" in problem:
        core = _compact_gamble(problem)
    elif schema == "bergert_pairwise" or alias == "bergert_nosofsky_2007":
        oa = problem.get("option_A") or {}
        ob = problem.get("option_B") or {}
        core = (
            f"bergert pid={problem.get('problem_id')} "
            f"A_cues={oa.get('cues')} B_cues={ob.get('cues')} "
            f"keys={list(problem.get('option_keys') or [])}"
        )
    elif schema == "guan_stopping" or alias == "guan_2020_stopping":
        core = (
            f"guan env={problem.get('environment')} L={problem.get('sequence_length')} "
            f"pos={problem.get('position')} vals={list(problem.get('values_observed') or [])}"
        )
    elif alias == "steyvers_2009_bandit":
        core = (
            f"steyvers game={problem.get('game')} trial={problem.get('trial')} "
            f"n_arms={problem.get('n_arms')} keys={list(problem.get('option_keys') or [])}"
        )
    elif schema == "categorical_bandit" or alias == "13schulz2020finding":
        core = (
            f"schulz round={problem.get('round')} trial={problem.get('trial')} "
            f"n_arms={problem.get('n_arms')} keys={list(problem.get('option_keys') or [])}"
        )
    elif schema == "kool_twostep" or alias == "14kool2016when":
        stage = int(problem.get("stage") or 1)
        keys = list(problem.get("option_keys") or [])
        if stage == 1:
            core = (
                f"kool stage=1 day={problem.get('presented_day')} "
                f"keys={keys} "
                f"spaceship={problem.get('spaceship_options') or keys}"
            )
        else:
            core = (
                f"kool stage=2 day={problem.get('presented_day')} "
                f"keys={keys} "
                f"spaceship={problem.get('spaceship')} "
                f"aliens={problem.get('alien_options')} "
                f"s1={problem.get('stage1_action')} "
                f"planet={problem.get('planet')}"
            )
    elif "balloon_id" in problem and "pump_count_before" in problem:
        core = (
            f"balloon={problem.get('balloon_id')} "
            f"pump_n={problem.get('pump_count_before')} "
            f"acc={problem.get('accumulated_points_before')}"
        )
    elif alias == "3frey2017cct" or "cards_flipped" in problem:
        core = (
            f"cct round={problem.get('round_id')} flipped={problem.get('cards_flipped')} "
            f"score={problem.get('current_score')} "
            f"remain={problem.get('n_cards_remaining')}/{problem.get('n_cards_total')} "
            f"loss_n={problem.get('n_loss_cards')} "
            f"gain={problem.get('gain_amount')} loss={problem.get('loss_amount')} "
            f"keys={list(problem.get('option_keys') or [])}"
        )
    elif "ratings_A" in problem or "option_A_features" in problem:
        core = (
            f"hilbig A={problem.get('option_A_features')} B={problem.get('option_B_features')} "
            f"ratA={problem.get('ratings_A')} ratB={problem.get('ratings_B')} "
            f"keys={list(problem.get('option_keys') or [])}"
        )
    elif "memory_set_letters" in problem and "probe_letter" in problem:
        core = (
            "memory_set="
            f"{problem.get('memory_set_letters', [])} "
            f"probe={problem.get('probe_letter')}"
        )
    elif "stimulus_features" in problem and (
        problem.get("task") == "category_learning" or "correct_category" in problem
    ):
        sf = problem.get("stimulus_features") or {}
        keys = problem.get("option_keys") or []
        core = (
            f"stim=({sf.get('size')},{sf.get('color')},{sf.get('shape')}) "
            f"keys={list(keys)} "
            f"rule_block={problem.get('rule_block_id')}"
        )
    elif "cards" in problem:
        core = f"cards={list(problem.get('cards') or [])} keys={list(problem.get('option_keys') or [])}"
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
    """Fallback length estimator. Prefer chat_input_token_count for budget checks."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def get_qwen_tokenizer():
    """Load the production Qwen tokenizer from the local HF cache (CPU, no download)."""
    global _QWEN_TOKENIZER, _QWEN_TOKENIZER_FAILED
    if _QWEN_TOKENIZER is not None:
        return _QWEN_TOKENIZER
    if _QWEN_TOKENIZER_FAILED:
        return None
    try:
        from transformers import AutoTokenizer

        _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
            QWEN_TOKENIZER_NAME,
            local_files_only=True,
            trust_remote_code=True,
        )
        return _QWEN_TOKENIZER
    except Exception:
        _QWEN_TOKENIZER_FAILED = True
        return None


def conservative_chat_token_count(system: str, user: str) -> int:
    """Upper-bound stand-in when the Qwen tokenizer is unavailable.

    Qwen measured ~0.40 tokens/char on 10k-char Python; 0.5 tokens/char is safer.
    """
    return 16 + (len(system or "") + len(user or "") + 1) // 2


def chat_input_token_count(system: str, user: str) -> int:
    """Token count of the complete vLLM chat input, including template and generation prefix."""
    tokenizer = get_qwen_tokenizer()
    if tokenizer is None:
        return conservative_chat_token_count(system, user)
    ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system or ""},
            {"role": "user", "content": user or ""},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    return int(len(ids))


def _program_fitness_score(prog: Dict[str, Any]) -> str:
    metrics = prog.get("metrics") or {}
    score = metrics.get("combined_score")
    try:
        return f"{float(score):.4f}" if score is not None else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def split_official_optional_programs(
    top_programs: Optional[List[Dict[str, Any]]],
    inspirations: Optional[List[Dict[str, Any]]],
    *,
    num_top: int,
    num_diverse: int,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split worker-supplied lists the way 411fb59 PromptSampler does.

    ``top_programs`` is already the island-sorted slice
    ``island_programs[:num_top + num_diverse]``. First ``num_top`` become the
    Top Performing Programs section. The leftover is passed through
    ``random.sample(..., num_diverse)`` (official sampler.py:399), preserving
    seeded shuffle order rather than score-prefix order. Inspirations already
    shown as top/diverse are dropped by id.
    """

    def _usable(prog: Any) -> bool:
        return isinstance(prog, dict) and bool(str(prog.get("code") or "").strip())

    tops_in = [p for p in (top_programs or []) if _usable(p)]
    n_top = max(0, int(num_top))
    n_div = max(0, int(num_diverse))
    selected_top = tops_in[:n_top]
    remaining = tops_in[n_top:]
    k = min(n_div, len(remaining))
    if k > 0:
        sample_fn = rng.sample if rng is not None else random.sample
        diverse = sample_fn(remaining, k)
    else:
        diverse = []
    shown_ids = {p.get("id") for p in selected_top + diverse if p.get("id") is not None}
    insp: List[Dict[str, Any]] = []
    for prog in inspirations or []:
        if not _usable(prog):
            continue
        pid = prog.get("id")
        if pid is not None and pid in shown_ids:
            continue
        insp.append(prog)
    return selected_top, diverse, insp


def format_official_artifacts_section(artifacts: Optional[Dict[str, Any]]) -> str:
    """411fb59 ``_render_artifacts`` under Last Execution Output."""
    if not artifacts:
        return ""
    sections = [_ARTIFACTS_MARKER, "", "## Last Execution Output", ""]
    for key, value in artifacts.items():
        if isinstance(value, bytes):
            try:
                content = value.decode("utf-8", errors="replace")
            except Exception:
                content = f"<binary data: {len(value)} bytes>"
        else:
            content = str(value)
        if len(content) > 20 * 1024:
            content = content[: 20 * 1024] + "\n... (truncated)"
        sections.append(f"### {key}\n```\n{content}\n```")
    return "\n".join(sections).strip()


def format_official_previous_attempts(programs: Optional[List[Dict[str, Any]]]) -> str:
    """411fb59 previous-attempt template (changes/performance/outcome; not full code)."""
    usable = [p for p in (programs or []) if isinstance(p, dict)]
    if not usable:
        return ""
    selected = usable[-min(3, len(usable)) :]
    blocks = [_PREVIOUS_MARKER, "", "## Previous Attempts", ""]
    for i, program in enumerate(reversed(selected)):
        attempt_number = len(usable) - i
        changes = (
            program.get("changes_description")
            or (program.get("metadata") or {}).get("changes")
            or "unknown changes"
        )
        metrics = program.get("metrics") or {}
        parts: List[str] = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    parts.append(f"{name}: {float(value):.4f}")
                except (TypeError, ValueError):
                    parts.append(f"{name}: {value}")
            else:
                parts.append(f"{name}: {value}")
        performance = ", ".join(parts) if parts else "n/a"
        blocks.append(
            f"### Attempt {attempt_number}\n"
            f"- Changes: {changes}\n"
            f"- Performance: {performance}\n"
            f"- Outcome: combined_score comparison with parent metrics\n"
        )
    return "\n".join(blocks).strip()


def format_official_top_section(programs: List[Dict[str, Any]]) -> str:
    """411fb59 ``top_program`` template under a Top Performing Programs header."""
    if not programs:
        return ""
    blocks = [_TOP_MARKER, "", "## Top Performing Programs", ""]
    for i, prog in enumerate(programs):
        code = str(prog.get("code") or "").strip()
        if not code:
            continue
        blocks.append(
            f"### Program {i + 1} (Score: {_program_fitness_score(prog)})\n"
            f"```python\n{code}\n```\n"
            "Key features: combined_score\n"
        )
    text = "\n".join(blocks).strip()
    return text if "```python" in text else ""


def format_official_diverse_section(programs: List[Dict[str, Any]]) -> str:
    """411fb59 leftover island programs under the Diverse Programs fragment title."""
    if not programs:
        return ""
    blocks = [_DIVERSE_MARKER, "", "## Diverse Programs", ""]
    for i, prog in enumerate(programs):
        code = str(prog.get("code") or "").strip()
        if not code:
            continue
        blocks.append(
            f"### Program D{i + 1} (Score: {_program_fitness_score(prog)})\n"
            f"```python\n{code}\n```\n"
            "Key features: Alternative approach to combined_score\n"
        )
    text = "\n".join(blocks).strip()
    return text if "```python" in text else ""


def format_official_inspiration_section(inspirations: List[Dict[str, Any]]) -> str:
    """Official OpenEvolve inspiration examples (411fb59 templates). Empty if none kept."""
    if not inspirations:
        return ""
    blocks = [
        _INSPIRATIONS_MARKER,
        "",
        "## Inspiration Programs",
        "",
        "These programs represent diverse approaches and creative solutions that may inspire new ideas:",
        "",
    ]
    for i, prog in enumerate(inspirations):
        code = str(prog.get("code") or "").strip()
        if not code:
            continue
        metrics = prog.get("metrics") or {}
        score = metrics.get("combined_score")
        try:
            score_s = f"{float(score):.4f}" if score is not None else "n/a"
        except (TypeError, ValueError):
            score_s = "n/a"
        blocks.append(
            f"### Inspiration {i + 1} (Score: {score_s}, Type: island inspiration)\n"
            f"```python\n{code}\n```\n"
            "Unique approach: alternative implementation from the same island.\n"
        )
    text = "\n".join(blocks).strip()
    return text if "```python" in text else ""


def _drop_optional_program_sections(user_text: str) -> str:
    """Drop optional blocks in reverse official template order."""
    text = user_text
    for marker, _name in _OPTIONAL_DROP_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
    return text


def _drop_inspiration_section(user_text: str) -> str:
    return _drop_optional_program_sections(user_text)


def truncate_vanilla_messages(
    *,
    task_text: str,
    program_code: str,
    metrics: Dict[str, Any],
    trials_compact: str,
    max_prompt_tokens: int,
    reserved_completion_tokens: int,
    model_context_len: int,
    categorical: bool = False,
    n_actions: Optional[int] = None,
    interface_text: Optional[str] = None,
    top_programs: Optional[List[Dict[str, Any]]] = None,
    diverse_programs: Optional[List[Dict[str, Any]]] = None,
    inspirations: Optional[List[Dict[str, Any]]] = None,
    previous_programs: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], TruncationState]:
    """
    Fit the prompt under the Qwen input ceiling.

    Required: system, task, choose() contract, current parent. Observed train+val
    examples outrank optional official context. Optional blocks are appended in
    411fb59 template order (artifacts, previous attempts, top, diverse,
    inspirations) while they fit, then dropped in reverse before examples are
    reduced. Never includes test.
    """
    del reserved_completion_tokens  # counted by the chat template + separate max_tokens
    state = TruncationState()
    all_lines = [ln for ln in trials_compact.splitlines() if ln.strip()]
    state.examples_available = len(all_lines)
    ceiling = min(int(max_prompt_tokens), ICLR_FROZEN_INPUT_TOKEN_CEILING)
    ceiling = min(ceiling, int(model_context_len) - ICLR_FROZEN_LLM_MAX_TOKENS)

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
        api = (interface_text or "").strip() or choose_api_text(
            categorical=categorical, n_actions=n_actions
        )
        parts = [
            "# Task (vanilla — no TEH prompt engineering)",
            task_text.strip(),
            "",
            "# API",
            api,
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
    required_user = build_user("", include_metrics=False)
    required_tokens = chat_input_token_count(system, required_user)
    if required_tokens > ceiling:
        state.estimated_tokens = required_tokens
        state.trim_reason = "required_overflow"
        state.steps.append("required_overflow")
        raise RequiredPromptOverflowError(
            f"required system/task/interface/parent prompt is {required_tokens} tokens, "
            f"exceeds ceiling {ceiling}"
        )

    include_metrics = True
    example_lines = list(all_lines)
    trials_text = "\n".join(example_lines)
    user = build_user(trials_text, include_metrics)
    state.estimated_tokens = chat_input_token_count(system, user)
    state.prompt_trial_count = len(example_lines)
    state.examples_included = len(example_lines)
    fitted = state.estimated_tokens <= ceiling

    if not fitted:
        include_metrics = False
        state.steps.append("drop_metrics")
        user = build_user(trials_text, include_metrics)
        state.estimated_tokens = chat_input_token_count(system, user)
        fitted = state.estimated_tokens <= ceiling

    if not fitted:
        for cap in ICLR_PROMPT_EXAMPLE_REDUCTION_CAPS:
            if len(all_lines) <= cap:
                continue
            example_lines = all_lines[:cap]
            trials_text = "\n".join(example_lines)
            state.steps.append(f"cap_prompt_trials_{cap}")
            user = build_user(trials_text, include_metrics)
            state.estimated_tokens = chat_input_token_count(system, user)
            state.prompt_trial_count = len(example_lines)
            state.examples_included = len(example_lines)
            if state.estimated_tokens <= ceiling:
                fitted = True
                state.trim_reason = f"reduced_examples_to_{cap}"
                break

    if not fitted:
        state.steps.append("minimal_trials_5")
        example_lines = all_lines[:5]
        trials_text = "\n".join(example_lines)
        user = build_user(trials_text, include_metrics=False)
        state.estimated_tokens = chat_input_token_count(system, user)
        state.prompt_trial_count = len(example_lines)
        state.examples_included = len(example_lines)
        state.trim_reason = "reduced_examples_to_5"
        if state.estimated_tokens > ceiling:
            state.trim_reason = "required_plus_examples_overflow"
            raise RequiredPromptOverflowError(
                f"required content plus minimal examples is {state.estimated_tokens} tokens, "
                f"exceeds ceiling {ceiling}"
            )

    base_user = user
    artifacts_section = format_official_artifacts_section(artifacts)
    previous_section = format_official_previous_attempts(previous_programs)
    state.artifacts_requested = 1 if artifacts_section else 0
    state.previous_attempts_requested = 1 if previous_section else 0
    state.top_programs_requested = len(list(top_programs or []))
    state.diverse_programs_requested = len(list(diverse_programs or []))
    state.inspirations_requested = len(list(inspirations or []))

    def _try_section(kind: str, section: str) -> bool:
        nonlocal base_user
        if not section:
            return False
        candidate = base_user + "\n\n" + section
        est = chat_input_token_count(system, candidate)
        if est > ceiling:
            state.steps.append(f"drop_{kind}")
            return False
        base_user = candidate
        state.estimated_tokens = est
        return True

    if artifacts_section and _try_section("artifacts", artifacts_section):
        state.artifacts_kept = 1
        state.steps.append("keep_artifacts")
    if previous_section and _try_section("previous_attempts", previous_section):
        state.previous_attempts_kept = 1
        state.steps.append("keep_previous_attempts")

    program_base = base_user
    kept_top: List[Dict[str, Any]] = []
    kept_div: List[Dict[str, Any]] = []
    kept_insp: List[Dict[str, Any]] = []

    def _try_keep(kind: str, kept: List[Dict[str, Any]], prog: Dict[str, Any]) -> bool:
        nonlocal base_user
        trial_top = list(kept_top)
        trial_div = list(kept_div)
        trial_insp = list(kept_insp)
        if kind == "top":
            trial_top = kept + [prog]
        elif kind == "diverse":
            trial_div = kept + [prog]
        else:
            trial_insp = kept + [prog]
        parts = [
            format_official_top_section(trial_top),
            format_official_diverse_section(trial_div),
            format_official_inspiration_section(trial_insp),
        ]
        extra = "\n\n".join(p for p in parts if p)
        candidate = program_base + (("\n\n" + extra) if extra else "")
        est = chat_input_token_count(system, candidate)
        if est > ceiling:
            state.steps.append(f"drop_{kind}_{len(kept)}")
            return False
        kept.append(prog)
        base_user = candidate
        state.estimated_tokens = est
        return True

    for prog in list(top_programs or []):
        if isinstance(prog, dict) and str(prog.get("code") or "").strip():
            _try_keep("top", kept_top, prog)
    for prog in list(diverse_programs or []):
        if isinstance(prog, dict) and str(prog.get("code") or "").strip():
            _try_keep("diverse", kept_div, prog)
    for prog in list(inspirations or []):
        if isinstance(prog, dict) and str(prog.get("code") or "").strip():
            _try_keep("inspiration", kept_insp, prog)
    state.top_programs_kept = len(kept_top)
    state.diverse_programs_kept = len(kept_div)
    state.inspirations_kept = len(kept_insp)
    if kept_top:
        state.steps.append(f"keep_top_{len(kept_top)}")
    if kept_div:
        state.steps.append(f"keep_diverse_{len(kept_div)}")
    if kept_insp:
        state.steps.append(f"keep_inspirations_{len(kept_insp)}")
    dropped_optional = any(s.startswith("drop_") for s in state.steps)
    if dropped_optional and not state.trim_reason:
        state.trim_reason = "dropped_optional_blocks"
    return {"system": system, "user": base_user}, state


def _estimate_total_prompt_tokens(system_message: str, messages: List[Dict[str, str]], reserved: int) -> int:
    user_text = messages[-1].get("content", "") if messages else ""
    # Chat-template input tokens; reserved generation is a separate vLLM max_tokens.
    del reserved
    return chat_input_token_count(system_message or "", user_text)


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
        "examples_available": state.examples_available,
        "examples_included": state.examples_included,
        "prompt_trial_count": state.prompt_trial_count,
        "prompt_train_trials": ctx.get("prompt_train_trials"),
        "prompt_val_trials": ctx.get("prompt_val_trials"),
        "artifacts_requested": state.artifacts_requested,
        "artifacts_included": state.artifacts_kept,
        "previous_attempts_requested": state.previous_attempts_requested,
        "previous_attempts_included": state.previous_attempts_kept,
        "top_programs_requested": state.top_programs_requested,
        "top_programs_included": state.top_programs_kept,
        "diverse_programs_requested": state.diverse_programs_requested,
        "diverse_programs_included": state.diverse_programs_kept,
        "inspirations_requested": state.inspirations_requested,
        "inspirations_included": state.inspirations_kept,
        "inspirations_kept": state.inspirations_kept,
        "top_programs_kept": state.top_programs_kept,
        "diverse_programs_kept": state.diverse_programs_kept,
        "trim_reason": state.trim_reason,
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
    """Drop optional program blocks first; never strip the choose() contract or parent."""
    del reserved
    steps: List[str] = []
    est = chat_input_token_count(system_message or "", user_text)
    if est <= cap:
        return user_text, steps
    for marker, name in _OPTIONAL_DROP_MARKERS:
        if marker in user_text:
            user_text = user_text.split(marker, 1)[0].rstrip()
            steps.append(f"guard_drop_{name}")
            est = chat_input_token_count(system_message or "", user_text)
            if est <= cap:
                return user_text, steps
    # Drop example lines from the end of the examples block only.
    marker = "# Example trials (train+val only; compact)"
    if marker in user_text:
        head, tail = user_text.split(marker, 1)
        example_lines = [ln for ln in tail.splitlines() if ln.strip()]
        while example_lines and est > cap:
            example_lines = example_lines[:-1]
            steps.append("guard_drop_example_line")
            user_text = head + marker + ("\n" + "\n".join(example_lines) if example_lines else "")
            est = chat_input_token_count(system_message or "", user_text)
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

    OpenEvolve database/islands still select a single parent. Official optional
    whole-program context comes from the worker kwargs: ``previous_programs`` is
    the island top-``num_top`` list, ``top_programs`` is the island-sorted slice
    of size num_top+num_diverse (diverse taken via official ``random.sample``),
    ``inspirations`` is ``sample_from_island(..., num_inspirations=num_diverse)``,
    and ``program_artifacts`` is the parent artifact dict when include_artifacts
    is on. Optional blocks are packed after required task/contract/parent/examples
    and dropped before reducing examples.
    """
    ctx = _WORKER_VANILLA
    task_text = ctx.get("task_text", "")
    program_code = current_program or ""
    metrics = program_metrics or {}
    trials_compact = ctx.get("trials_compact", "")
    reserved = int(ctx.get("llm_max_tokens", ICLR_FROZEN_LLM_MAX_TOKENS))
    model_len = int(ctx.get("max_model_len", ICLR_FROZEN_VLLM_MAX_MODEL_LEN))
    max_prompt_tokens = min(
        int(ctx.get("hard_prompt_token_cap", ICLR_FROZEN_INPUT_TOKEN_CEILING)),
        ICLR_FROZEN_INPUT_TOKEN_CEILING,
        model_len - reserved,
    )
    cfg = getattr(self, "config", None)
    num_top = int(getattr(cfg, "num_top_programs", ICLR_FROZEN_NUM_TOP_PROGRAMS))
    num_diverse = int(getattr(cfg, "num_diverse_programs", ICLR_FROZEN_NUM_DIVERSE_PROGRAMS))
    include_artifacts = bool(getattr(cfg, "include_artifacts", ICLR_FROZEN_INCLUDE_ARTIFACTS))
    include_previous = bool(ctx.get("include_previous_attempts", ICLR_FROZEN_INCLUDE_PREVIOUS_ATTEMPTS))
    top, diverse, insp = split_official_optional_programs(
        kwargs.get("top_programs") or [],
        kwargs.get("inspirations") or [],
        num_top=num_top,
        num_diverse=num_diverse,
    )
    artifacts = kwargs.get("program_artifacts") if include_artifacts else None
    previous_programs = (kwargs.get("previous_programs") or []) if include_previous else []
    try:
        prompt, state = truncate_vanilla_messages(
            task_text=task_text,
            program_code=program_code,
            metrics=metrics,
            trials_compact=trials_compact,
            max_prompt_tokens=max_prompt_tokens,
            reserved_completion_tokens=reserved,
            model_context_len=model_len,
            categorical=bool(ctx.get("categorical", False)),
            n_actions=ctx.get("n_actions"),
            interface_text=ctx.get("interface_text"),
            top_programs=top,
            diverse_programs=diverse,
            inspirations=insp,
            previous_programs=previous_programs,
            artifacts=artifacts if isinstance(artifacts, dict) else None,
        )
    except RequiredPromptOverflowError:
        state = TruncationState(trim_reason="required_overflow", steps=["required_overflow"])
        ctx["last_truncation"] = state
        _append_prompt_diagnostic(
            ctx, source="build_prompt", state=state, status="required_overflow", safety_guard_passed=False
        )
        raise
    state.prompt_train_trials = int(ctx.get("prompt_train_trials", 0))
    state.prompt_val_trials = int(ctx.get("prompt_val_trials", 0))
    ctx["last_truncation"] = state
    truncating = any(
        s.startswith(("cap_", "drop_", "minimal_", "guard_")) for s in state.steps
    )
    status = "truncated" if truncating else "ok"
    _append_prompt_diagnostic(ctx, source="build_prompt", state=state, status=status, safety_guard_passed=None)
    return prompt


def _patched_generate_with_context(self, system_message, messages, **kwargs):
    """Final safety guard only — never rebuilds the task prompt from message content."""
    ctx = _WORKER_VANILLA
    if ctx and messages:
        reserved = int(ctx.get("llm_max_tokens", ICLR_FROZEN_LLM_MAX_TOKENS))
        model_len = int(ctx.get("max_model_len", ICLR_FROZEN_VLLM_MAX_MODEL_LEN))
        cap = min(
            int(ctx.get("hard_prompt_token_cap", ICLR_FROZEN_INPUT_TOKEN_CEILING)),
            ICLR_FROZEN_INPUT_TOKEN_CEILING,
            model_len - reserved,
        )
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


def _adapt_official_failure_metrics_for_loglik(metrics: Any) -> Any:
    """Map official failed-eval metrics so fitness cannot beat valid negative loglik.

    OpenEvolve (audit 411fb59 and the ProcessParallel APIs this runner imports)
    scores a failed ``evaluate()`` / timeout / unexpected return as
    ``{"error": 0.0}`` or ``{"error": 0.0, "timeout": True}`` with no
    ``combined_score``. ``get_fitness_score`` then uses that 0.0. That is the
    worst value for accuracy-style maximize; it outranks valid log-likelihood.
    Attach ``FAILED_COMBINED_SCORE`` only when official failure metrics are
    missing ``combined_score``. Successful returns are unchanged.
    """
    if not isinstance(metrics, dict):
        return metrics
    if "combined_score" in metrics:
        return metrics
    if (not metrics) or ("error" in metrics) or (metrics.get("timeout") is True):
        adapted = dict(metrics)
        adapted["combined_score"] = float(FAILED_COMBINED_SCORE)
        return adapted
    return metrics


async def _patched_evaluate_program(self, program_code: str, program_id: str = ""):
    metrics = await _ORIG_EVALUATE_PROGRAM(self, program_code, program_id)
    return _adapt_official_failure_metrics_for_loglik(metrics)


def _install_runtime_patches() -> None:
    global _ORIG_BUILD_PROMPT, _ORIG_GENERATE_WITH_CONTEXT, _ORIG_EVALUATE_PROGRAM
    import openevolve.evaluator as oe_evaluator
    import openevolve.llm.openai as oe_openai
    import openevolve.prompt.sampler as oe_sampler

    if _ORIG_BUILD_PROMPT is None:
        _ORIG_BUILD_PROMPT = oe_sampler.PromptSampler.build_prompt
        oe_sampler.PromptSampler.build_prompt = _patched_build_prompt
    if _ORIG_GENERATE_WITH_CONTEXT is None:
        _ORIG_GENERATE_WITH_CONTEXT = oe_openai.OpenAILLM.generate_with_context
        oe_openai.OpenAILLM.generate_with_context = _patched_generate_with_context
    if _ORIG_EVALUATE_PROGRAM is None:
        _ORIG_EVALUATE_PROGRAM = oe_evaluator.Evaluator.evaluate_program
        oe_evaluator.Evaluator.evaluate_program = _patched_evaluate_program


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


def _install_global_openevolve_controller_hook() -> None:
    """
    Install once before any participant runs.

    OpenEvolve.run() constructs ProcessParallelController internally; the hooked class
    reads per-thread participant context (safe for parallel_participants > 1).
    """
    global _OE_CONTROLLER_HOOKED
    if _OE_CONTROLLER_HOOKED:
        return
    import openevolve.controller as oc

    if not hasattr(oc, "_Orig_ProcessParallelController"):
        oc._Orig_ProcessParallelController = oc.ProcessParallelController

    class _ThreadLocalVanillaController(VanillaProcessParallelController):
        def __init__(
            self,
            config: Config,
            evaluation_file: str,
            database: Any,
            evolution_tracer=None,
            file_suffix: str = ".py",
        ):
            participant_ctx = _get_thread_participant_ctx()
            VanillaProcessParallelController.__init__(
                self,
                config,
                evaluation_file,
                database,
                evolution_tracer,
                file_suffix,
                participant_ctx,
            )

    oc.ProcessParallelController = _ThreadLocalVanillaController
    _OE_CONTROLLER_HOOKED = True


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
        # selected is prompt-display only; evaluator JSON still has full train+val.
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
            "categorical": bool(ctx.get("categorical", False)),
            "n_actions": ctx.get("n_actions"),
            "interface_text": ctx.get("interface_text"),
            "diagnostics_path": ctx.get("diagnostics_path"),
            "include_previous_attempts": bool(
                ctx.get("include_previous_attempts", ICLR_FROZEN_INCLUDE_PREVIOUS_ATTEMPTS)
            ),
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
    alias = normalize_psych101_dataset_alias(dataset)
    if is_external_dataset(alias):
        return f"generated_outputs/external/{alias}/openevolve/run_{timestamp}"
    if is_mixed_gambles_dataset(alias):
        return f"generated_outputs/mixed_gambles/openevolve/run_{timestamp}"
    split = _effective_psych_dataset_split(dataset, psych_dataset_split)
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


def _render_evaluator_py(
    evolution_split_path: Path,
    *,
    split_ratio: float,
    categorical: bool = False,
) -> str:
    split_path = str(evolution_split_path.resolve())
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")
    categorical_lit = "True" if categorical else "False"
    return f'''"""Auto-generated OpenEvolve evaluator (observed-union objective; no test access).

Loads the complete retained train+val JSON. Prompt example caps do not apply here.
"""
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

SPLIT_PATH = Path(r\"{split_path}\")
SPLIT_RATIO = {float(split_ratio)}
CHOICE13K_LOGLIK_EPS = {CHOICE13K_LOGLIK_EPS}
CATEGORICAL = {categorical_lit}


def _pooled_observed_loglik(train_ll: float, val_ll: float, n_train: int, n_val: int) -> float:
    n_tr = max(0, int(n_train))
    n_vl = max(0, int(n_val))
    denom = n_tr + n_vl
    if denom <= 0 or n_vl <= 0:
        return float(train_ll)
    if not math.isfinite(float(val_ll)):
        return float(train_ll)
    return (n_tr * float(train_ll) + n_vl * float(val_ll)) / denom


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


def _valid_action_ids(problem: Dict[str, Any]) -> List[int]:
    options = problem.get("options") or []
    ids: List[int] = []
    for opt in options:
        if isinstance(opt, dict) and "action" in opt:
            ids.append(int(opt["action"]))
    if ids:
        return ids
    keys = problem.get("option_keys") or []
    out: List[int] = []
    for i, k in enumerate(keys):
        try:
            out.append(int(k))
        except (TypeError, ValueError):
            out.append(i)
    return out


def _coerce_categorical(probs_raw: Any, valid_ids: List[int]) -> Dict[int, float]:
    K = len(valid_ids)
    if K < 1:
        raise ValueError("no valid action ids")
    expected = set(valid_ids)
    if isinstance(probs_raw, dict):
        raw = probs_raw
    elif K == 2 and isinstance(probs_raw, (float, int, bool)):
        p1 = float(probs_raw)
        if not (math.isfinite(p1) and 0.0 <= p1 <= 1.0):
            raise ValueError(f"invalid Bernoulli-style categorical output: {{p1!r}}")
        raw = {{0: 1.0 - p1, 1: p1}}
    else:
        raise ValueError(
            f"malformed categorical choose() output: {{type(probs_raw).__name__}}"
        )
    probs = {{aid: 0.0 for aid in valid_ids}}
    for key, val in raw.items():
        try:
            aid = int(key)
        except (TypeError, ValueError):
            continue
        if aid not in expected:
            continue
        try:
            p = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(p) and p >= 0.0:
            probs[aid] = p
    total = sum(probs.values())
    if total <= 0.0:
        raise ValueError("categorical probabilities have no positive mass")
    return {{aid: probs[aid] / total for aid in valid_ids}}


def evaluate_trials(choose_fn: Callable, trials: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trials:
        return {{"avg_loglik": float("-inf"), "n": 0}}
    ll = 0.0
    if CATEGORICAL:
        for t in trials:
            problem = t.get("problem") or {{}}
            y = int(t["action"])
            valid_ids = _valid_action_ids(problem)
            if not valid_ids:
                raise ValueError("no valid action ids")
            probs = _coerce_categorical(choose_fn(problem, t.get("history") or []), valid_ids)
            p = _clamp(float(probs.get(y, 0.0)))
            ll += math.log(p)
        return {{"avg_loglik": float(ll / len(trials)), "n": len(trials)}}
    for t in trials:
        y = int(t["action"])
        p = _clamp(_parse_choose_output(choose_fn(t["problem"], t["history"])))
        ll += y * math.log(p) + (1 - y) * math.log(1.0 - p)
    return {{"avg_loglik": float(ll / len(trials)), "n": len(trials)}}


def evaluate(program_path: str) -> Dict[str, float]:
    path = Path(program_path)
    code = path.read_text(encoding="utf-8")
    choose_fn = compile_program(code)
    if choose_fn is None:
        raise RuntimeError("no choose()")
    train_trials, val_trials = _load_splits()
    n_train = len(train_trials)
    n_val = len(val_trials)
    train_eval = evaluate_trials(choose_fn, train_trials)
    val_eval = evaluate_trials(choose_fn, val_trials)
    train_ll = float(train_eval["avg_loglik"])
    val_ll = float(val_eval["avg_loglik"]) if n_val > 0 else train_ll
    combined = _pooled_observed_loglik(train_ll, val_ll, n_train, n_val)
    if not math.isfinite(combined):
        raise RuntimeError("non-finite observed combined_score")
    return {{
        "combined_score": combined,
        "observed_loglik": combined,
        "train_loglik": train_ll,
        "val_loglik": val_ll,
        "n_train": float(n_train),
        "n_val": float(n_val),
    }}
'''


def _resolve_checkpoint_interval(checkpoint_interval: int, iterations: int) -> int:
    """Map CLI interval onto OE's positive interval.

    ``<= 0`` means "only checkpoint at the end of the run" (single dump), which is
    enough for our post-hoc best-program scan and much smaller on disk.
    """
    iters = max(1, int(iterations))
    if int(checkpoint_interval) <= 0:
        return iters
    return min(int(checkpoint_interval), iters)


def _build_config(args, iterations: int) -> Config:
    cfg = Config()
    cfg.max_iterations = iterations
    cfg.checkpoint_interval = _resolve_checkpoint_interval(args.checkpoint_interval, iterations)
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
        "Evolve Python code for human choice prediction. Return a complete program file."
    )
    cfg.prompt.num_top_programs = args.num_top_programs
    cfg.prompt.num_diverse_programs = args.num_diverse_programs
    cfg.prompt.include_artifacts = args.include_artifacts
    cfg.prompt.use_template_stochasticity = False  # official optional; always off
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
    cfg.database.log_prompts = bool(args.log_prompts)
    cfg.database.random_seed = args.random_seed

    cfg.evaluator.timeout = args.evaluator_timeout
    cfg.evaluator.max_retries = args.evaluator_max_retries
    cfg.evaluator.parallel_evaluations = args.parallel_evaluations
    cfg.evaluator.cascade_evaluation = args.cascade_evaluation
    cfg.evaluator.use_llm_feedback = args.use_llm_feedback
    cfg.evaluator.enable_artifacts = args.enable_artifacts
    return cfg


def _find_best_program_by_observed_loglik(checkpoint_dir: Path) -> Tuple[Optional[Path], Optional[float]]:
    """Select the reported program by observed-union combined_score (never test)."""
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
        observed_ll = _safe_float(metrics.get("combined_score"))
        if observed_ll is None:
            observed_ll = _safe_float(metrics.get("observed_loglik"))
        if observed_ll is None:
            n_train = int(metrics.get("n_train") or 0)
            n_val = int(metrics.get("n_val") or 0)
            train_ll = _safe_float(metrics.get("train_loglik"))
            val_ll = _safe_float(metrics.get("val_loglik"))
            if train_ll is not None:
                observed_ll = _pooled_observed_loglik(
                    train_ll, val_ll if val_ll is not None else train_ll, n_train, n_val
                )
        if observed_ll is None:
            continue
        if best_ll is None or observed_ll > best_ll:
            best_ll = observed_ll
            code = data.get("code")
            if code:
                best_path = prog_file
    return best_path, best_ll


def _program_code_from_json(prog_json: Path) -> str:
    data = json.loads(prog_json.read_text(encoding="utf-8"))
    return data.get("code") or ""


def _is_failed_oe_metrics(metrics: Dict[str, Any]) -> bool:
    """True when metrics look like an official failed eval (or our failure floor)."""
    if not isinstance(metrics, dict):
        return True
    if metrics.get("timeout") is True:
        return True
    if "error" in metrics and "combined_score" not in metrics:
        return True
    cs = _safe_float(metrics.get("combined_score"))
    if cs is not None and cs <= FAILED_COMBINED_SCORE + 1e-9:
        return True
    if "error" in metrics and cs is not None and cs <= FAILED_COMBINED_SCORE + 1e-9:
        return True
    return False


def extract_evolution_records_from_checkpoint(
    checkpoint_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compact per-child records from one OE checkpoint (no full program code).

    ``combined_score`` is the observed train+val pooled loglik used for evolution.
    Returns ``(summary, rows)`` where ``rows`` is one entry per program in the
    checkpoint (typically ≤ population; small CSV).
    """
    programs_dir = checkpoint_dir / "programs"
    rows: List[Dict[str, Any]] = []
    if not programs_dir.is_dir():
        summary = {
            "checkpoint": checkpoint_dir.name,
            "n_programs": 0,
            "n_valid": 0,
            "n_failed": 0,
            "n_timeout": 0,
            "best_combined_score": None,
            "best_iteration": None,
            "best_program_id": None,
        }
        return summary, rows

    best_cs: Optional[float] = None
    best_iter: Optional[int] = None
    best_id: Optional[str] = None
    n_failed = 0
    n_timeout = 0
    n_valid = 0

    for prog_file in sorted(programs_dir.glob("*.json")):
        try:
            data = json.loads(prog_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = data.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        failed = _is_failed_oe_metrics(metrics)
        timeout = bool(metrics.get("timeout") is True)
        if timeout:
            n_timeout += 1
        if failed:
            n_failed += 1
        else:
            n_valid += 1
        cs = _safe_float(metrics.get("combined_score"))
        if cs is None:
            cs = _safe_float(metrics.get("observed_loglik"))
        iteration = data.get("iteration_found", data.get("iteration"))
        try:
            iteration_i = int(iteration) if iteration is not None else None
        except (TypeError, ValueError):
            iteration_i = None
        pid = str(data.get("id") or prog_file.stem)
        row = {
            "iteration": iteration_i,
            "program_id": pid,
            "combined_score": cs,
            "train_loglik": _safe_float(metrics.get("train_loglik")),
            "val_loglik": _safe_float(metrics.get("val_loglik")),
            "failed": int(failed),
            "timeout": int(timeout),
        }
        rows.append(row)
        if cs is not None and (best_cs is None or cs > best_cs):
            best_cs = cs
            best_iter = iteration_i
            best_id = pid

    rows.sort(
        key=lambda r: (
            r["iteration"] is None,
            r["iteration"] if r["iteration"] is not None else 10**9,
            r["program_id"],
        )
    )
    summary = {
        "checkpoint": checkpoint_dir.name,
        "n_programs": len(rows),
        "n_valid": n_valid,
        "n_failed": n_failed,
        "n_timeout": n_timeout,
        "best_combined_score": best_cs,
        "best_iteration": best_iter,
        "best_program_id": best_id,
    }
    return summary, rows


def write_evolution_records(participant_dir: Path, checkpoint_dir: Path) -> Dict[str, Any]:
    """Persist compact evolution metrics next to ``best_program.py`` (keeps code out)."""
    summary, rows = extract_evolution_records_from_checkpoint(checkpoint_dir)
    summary_path = participant_dir / "evolution_summary.json"
    scores_path = participant_dir / "evolution_iteration_scores.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = [
        "iteration",
        "program_id",
        "combined_score",
        "train_loglik",
        "val_loglik",
        "failed",
        "timeout",
    ]
    with scores_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_round_floats_for_csv_rows(rows))
    out = dict(summary)
    out["evolution_summary_path"] = str(summary_path)
    out["evolution_iteration_scores_path"] = str(scores_path)
    return out


def compact_openevolve_output(oe_output: Path, *, keep_latest_checkpoint: bool = False) -> Dict[str, Any]:
    """Drop bulky OE dumps after best program + compact metrics are written.

    Keeps ``openevolve_output/best/`` (official best copy). By default removes all
    ``checkpoints/`` trees (full program JSON archives). Replaces ``logs/`` with a
    tiny stub so the folder still exists but is not multi‑MB INFO spam.
    """
    report: Dict[str, Any] = {
        "checkpoints_removed": [],
        "checkpoints_kept": [],
        "logs_compacted": False,
    }
    ckpt_root = oe_output / "checkpoints"
    if ckpt_root.is_dir():
        ckpt_dirs = sorted(
            [p for p in ckpt_root.glob("checkpoint_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_")[-1]) if "_" in p.name else 0,
        )
        keep: Optional[Path] = ckpt_dirs[-1] if (keep_latest_checkpoint and ckpt_dirs) else None
        for d in ckpt_dirs:
            if keep is not None and d.resolve() == keep.resolve():
                report["checkpoints_kept"].append(d.name)
                continue
            shutil.rmtree(d, ignore_errors=True)
            report["checkpoints_removed"].append(d.name)
        if keep is None and ckpt_root.is_dir() and not any(ckpt_root.iterdir()):
            shutil.rmtree(ckpt_root, ignore_errors=True)

    logs_dir = oe_output / "logs"
    if logs_dir.is_dir():
        for child in list(logs_dir.iterdir()):
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                pass
        stub = logs_dir / "COMPACTED.txt"
        stub.write_text(
            "OpenEvolve INFO logs discarded (compact_oe_artifacts default). "
            "See participant evolution_summary.json / evolution_iteration_scores.csv "
            "and best_program.py for retained metrics and the final program.\n",
            encoding="utf-8",
        )
        report["logs_compacted"] = True
    return report


def _mean_loglik_rows(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for r in rows:
        v = _safe_float(r.get(key))
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def participant_oe_run_is_complete(
    participant_dir: Path,
    *,
    expected_n_iterations: int,
) -> bool:
    """True when this person already finished OE and can be skipped on resume.

    Adapter-only (does not touch the OpenEvolve library). Requires:
    ``best_program.py`` with a loadable ``choose``, and ``results.json`` with
    ``status=="ok"``, finite ``test_loglik``, and
    ``n_iterations_completed >= expected_n_iterations``. Failed / partial /
    corrupt people are **not** skipped so a requeue can retry them.
    """
    path = Path(participant_dir)
    best = path / "best_program.py"
    results_path = path / "results.json"
    if not (best.is_file() and results_path.is_file()):
        return False
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if str(payload.get("status") or "") != "ok":
        return False
    if _safe_float(payload.get("test_loglik")) is None:
        return False
    try:
        n_done = int(payload.get("n_iterations_completed"))
    except (TypeError, ValueError):
        return False
    if n_done < int(expected_n_iterations):
        return False
    try:
        code = best.read_text(encoding="utf-8")
    except OSError:
        return False
    if compile_program(code) is None:
        return False
    return True


def try_load_completed_oe_participant_row(
    participant_dir: Path,
    *,
    expected_n_iterations: int,
) -> Optional[Dict[str, Any]]:
    """Return existing ``results.json`` if complete; else ``None`` (do not mutate disk)."""
    if not participant_oe_run_is_complete(
        participant_dir, expected_n_iterations=expected_n_iterations
    ):
        return None
    results_path = Path(participant_dir) / "results.json"
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    # Soft marker for logs/CSV consumers; do not rewrite results.json.
    out = dict(payload)
    out["resumed_from_disk"] = True
    return out


def _wandb_log_loglik_summary(
    wandb_module: Any,
    rows: List[Dict[str, Any]],
    *,
    last_row: Dict[str, Any],
    expected_n: int,
    run_finished: bool = False,
) -> None:
    """Upload running diagnostic logliks plus the gated-style completion contract."""
    diagnostic: Dict[str, Any] = {
        "n_completed": len(rows),
        "last_participant_id": last_row.get("participant_id"),
        "last_test_loglik": _safe_float(last_row.get("test_loglik")),
        "last_observed_loglik": _safe_float(last_row.get("observed_loglik")),
        "last_train_loglik": _safe_float(last_row.get("train_loglik")),
        "last_val_loglik": _safe_float(last_row.get("val_loglik")),
        "avg_test_loglik": _mean_loglik_rows(rows, "test_loglik"),
        "avg_observed_loglik": _mean_loglik_rows(rows, "observed_loglik"),
        "avg_train_loglik": _mean_loglik_rows(rows, "train_loglik"),
        "avg_val_loglik": _mean_loglik_rows(rows, "val_loglik"),
    }
    completion = wandb_completion_fields(
        rows,
        expected_n=expected_n,
        run_finished=run_finished,
        distinguish_failures=True,
    )
    payload = merge_wandb_payload(diagnostic, completion)
    with _WANDB_LOG_LOCK:
        apply_wandb_payload(wandb_module, payload)


def run_participant(
    args,
    participant_id: int,
    participant_ordinal: int,
    run_dir: Path,
    wandb_module: Any = None,
) -> Dict[str, Any]:
    participant_dir = run_dir / f"participant_{participant_id}"
    # Person-level resume (adapter only): skip OE evolution when this person already
    # finished under the same --output_dir (Slurm requeue / manual restart). Does not
    # call into OpenEvolve and does not rewrite results.json.
    if bool(getattr(args, "skip_completed_participants", True)):
        resumed = try_load_completed_oe_participant_row(
            participant_dir,
            expected_n_iterations=int(args.n_iterations),
        )
        if resumed is not None:
            print(
                f"[OE resume] skip complete participant {participant_id} "
                f"(ordinal={participant_ordinal}) -> {participant_dir}"
            )
            return resumed

    exp_dir = participant_dir / "openevolve_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_trials, val_trials, test_trials, _audit, manifest = trials_for_participant(
        args.dataset,
        participant_id,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        filter_mixed_gambles=args.filter_mixed_gambles,
        psych_dataset_split=_effective_psych_dataset_split(args.dataset, args.psych_dataset_split),
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
        max_observed_trials_per_participant=getattr(
            args, "max_observed_trials_per_participant", None
        ),
        limited_data_protocol=getattr(args, "limited_data_protocol", "off"),
        limited_train_val=getattr(args, "limited_train_val", None),
        speekenbrink_split=getattr(args, "speekenbrink_split", "chronological"),
        return_manifest=True,
    )
    manifest_jsonl = run_dir / "log" / LIMITED_DATA_MANIFEST_JSONL_FILENAME
    if should_persist_limited_data_manifest(manifest):
        append_limited_data_manifest_jsonl(manifest_jsonl, manifest)

    evolution_split_path = exp_dir / "trials_evolution_split.json"
    _write_evolution_split_json(evolution_split_path, train_trials, val_trials)
    posthoc_test_path = participant_dir / "trials_test_posthoc.json"
    _write_posthoc_test_json(posthoc_test_path, test_trials)

    seed_path, _base_prompt_path = resolve_openevolve_seed_and_prompt(
        args.dataset,
        seed_path=args.seed_path,
        base_prompt=args.base_prompt,
    )
    categorical = is_categorical_output_dataset(args.dataset)
    n_actions = dataset_n_actions(args.dataset)
    task_text = vanilla_dataset_description(args.dataset, base_prompt=args.base_prompt)
    interface_text = choose_api_text(categorical=categorical, n_actions=n_actions)
    (exp_dir / "vanilla_task_prompt.txt").write_text(task_text, encoding="utf-8")
    (exp_dir / "vanilla_interface_contract.txt").write_text(interface_text, encoding="utf-8")

    initial_src = seed_path.resolve()
    initial_dst = exp_dir / "initial_program.py"
    shutil.copy2(initial_src, initial_dst)

    evaluator_path = exp_dir / "evaluator.py"
    evaluator_path.write_text(
        _render_evaluator_py(
            evolution_split_path,
            split_ratio=args.split_ratio,
            categorical=categorical,
        ),
        encoding="utf-8",
    )

    cfg = _build_config(args, args.n_iterations)
    cfg.to_yaml(exp_dir / "config.yaml")

    diagnostics_path = participant_dir / "prompt_truncation_diagnostics.jsonl"
    participant_ctx = {
        "participant_id": participant_id,
        "train_trials": train_trials,
        "val_trials": val_trials,
        "task_text": task_text,
        "interface_text": interface_text,
        "n_actions": n_actions,
        "max_prompt_train_trials": args.max_prompt_train_trials,
        "max_prompt_trials_per_problem": args.max_prompt_trials_per_problem,
        "split_seed": args.split_seed,
        "hard_prompt_token_cap": args.hard_prompt_token_cap,
        "llm_max_tokens": args.llm_max_tokens,
        "max_model_len": args.max_model_len,
        "diagnostics_path": str(diagnostics_path),
        "categorical": categorical,
        "include_previous_attempts": bool(args.include_previous_attempts),
    }

    oe_output = participant_dir / "openevolve_output"
    oe = OpenEvolve(
        initial_program_path=str(initial_dst),
        evaluation_file=str(evaluator_path),
        config=cfg,
        output_dir=str(oe_output),
    )
    oe.config.database.novelty_llm = oe.llm_ensemble
    status = "ok"
    error_msg = ""
    n_completed = 0
    best_program_path = participant_dir / "best_program.py"
    train_ll = val_ll = test_ll = observed_ll = None
    best_observed_ll: Optional[float] = None

    _set_thread_participant_ctx(participant_ctx)
    evo_summary: Dict[str, Any] = {}
    compact_report: Dict[str, Any] = {}
    try:
        best_program = _run_openevolve(oe, args.n_iterations)
        n_completed = oe.database.last_iteration if oe.database.last_iteration else args.n_iterations

        ckpt_root = oe_output / "checkpoints"
        ckpt_dirs = sorted(
            [p for p in ckpt_root.glob("checkpoint_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_")[-1]) if "_" in p.name else 0,
        )
        latest_ckpt = ckpt_dirs[-1] if ckpt_dirs else None
        # Lean interval (= n_iterations) usually dumps once at the end; if OE skipped
        # the dump (early stop / off-cycle), force one snapshot so we can still write
        # compact evolution metrics and select best_program.py before pruning.
        if latest_ckpt is None and hasattr(oe, "_save_checkpoint"):
            try:
                oe._save_checkpoint(int(n_completed) if n_completed else int(args.n_iterations))
                ckpt_dirs = sorted(
                    [p for p in ckpt_root.glob("checkpoint_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[-1]) if "_" in p.name else 0,
                )
                latest_ckpt = ckpt_dirs[-1] if ckpt_dirs else None
            except Exception:
                latest_ckpt = None

        if latest_ckpt is not None:
            prog_json, best_observed_ll = _find_best_program_by_observed_loglik(latest_ckpt)
            if prog_json is not None:
                code = _program_code_from_json(prog_json)
                best_program_path.write_text(code, encoding="utf-8")
            elif best_program is not None:
                best_program_path.write_text(best_program.code, encoding="utf-8")
            # Compact metrics BEFORE pruning checkpoints (needs programs/*.json).
            try:
                evo_summary = write_evolution_records(participant_dir, latest_ckpt)
                if best_observed_ll is None:
                    best_observed_ll = _safe_float(evo_summary.get("best_combined_score"))
            except Exception as e:
                evo_summary = {"error": f"evolution_records_failed: {e}"}
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
                train_ll = evaluate_loglik(choose_fn, train_trials, dataset=args.dataset)["avg_loglik"]
                val_ll = evaluate_loglik(choose_fn, val_trials, dataset=args.dataset)["avg_loglik"]
                observed_ll = _pooled_observed_loglik(
                    float(train_ll),
                    float(val_ll) if val_trials else float(train_ll),
                    len(train_trials),
                    len(val_trials),
                )
                # Held-out test is scored once, after program selection.
                test_ll = evaluate_loglik(choose_fn, test_trials, dataset=args.dataset)["avg_loglik"]
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

        # After best_program.py + compact metrics exist, drop bulky OE dumps.
        if bool(getattr(args, "compact_oe_artifacts", True)):
            try:
                compact_report = compact_openevolve_output(oe_output, keep_latest_checkpoint=False)
            except Exception as e:
                compact_report = {"error": f"compact_failed: {e}"}
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        traceback.print_exc()
    finally:
        _clear_thread_participant_ctx()

    row = {
        "participant_id": participant_id,
        "participant_ordinal": participant_ordinal,
        "train_loglik": train_ll,
        "val_loglik": val_ll,
        "observed_loglik": observed_ll,
        "test_loglik": test_ll,
        "n_train": len(train_trials),
        "n_val": len(val_trials),
        "n_test": len(test_trials),
        "n_iterations_requested": args.n_iterations,
        "n_iterations_completed": n_completed,
        "best_program_path": str(best_program_path) if best_program_path.is_file() else "",
        "status": status,
        "error": error_msg,
        "best_observed_loglik_in_pool": best_observed_ll,
        "n_programs_in_pool": evo_summary.get("n_programs"),
        "n_valid_programs": evo_summary.get("n_valid"),
        "n_failed_programs": evo_summary.get("n_failed"),
        "n_timeout_programs": evo_summary.get("n_timeout"),
        "prompt_trials_resampled_per_candidate": True,
        "prompt_trials_source": "train+val_union",
        "evolution_summary": evo_summary or None,
        "compact_oe_artifacts": compact_report or None,
    }
    (participant_dir / "results.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def _write_experiment_csvs(run_dir: Path, detail_rows: List[Dict[str, Any]]) -> None:
    details_fields = [
        "participant_id",
        "participant_ordinal",
        "train_loglik",
        "val_loglik",
        "observed_loglik",
        "test_loglik",
        "n_train",
        "n_val",
        "n_test",
    ]
    details_path = run_dir / "participant_details_loglik.csv"
    with details_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_round_floats_for_csv_rows(detail_rows))

    train_vals = [r["train_loglik"] for r in detail_rows if r.get("train_loglik") is not None]
    val_vals = [r["val_loglik"] for r in detail_rows if r.get("val_loglik") is not None]
    observed_vals = [r["observed_loglik"] for r in detail_rows if r.get("observed_loglik") is not None]
    test_vals = [r["test_loglik"] for r in detail_rows if r.get("test_loglik") is not None]
    summary_row = {
        "num_of_participants": len(detail_rows),
        "avg_train_loglik": float(np.mean(train_vals)) if train_vals else None,
        "avg_val_loglik": float(np.mean(val_vals)) if val_vals else None,
        "avg_observed_loglik": float(np.mean(observed_vals)) if observed_vals else None,
        "avg_test_loglik": float(np.mean(test_vals)) if test_vals else None,
    }
    summary_fields = [
        "num_of_participants",
        "avg_train_loglik",
        "avg_val_loglik",
        "avg_observed_loglik",
        "avg_test_loglik",
    ]
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
        "observed_loglik",
        "test_loglik",
        "n_train",
        "n_val",
        "n_test",
        "n_iterations_requested",
        "n_iterations_completed",
        "n_programs_in_pool",
        "n_valid_programs",
        "n_failed_programs",
        "n_timeout_programs",
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
    p.add_argument("--range_end_ordinal", type=int, default=29)
    p.add_argument("--ordinals", nargs="+", type=int, default=None)
    p.add_argument("--all_max_participants", type=int, default=None)
    p.add_argument("--filter_mixed_gambles", action="store_true")
    p.add_argument("--split_mode", type=str, default="within_participant", choices=["within_participant", "across_participants"])
    p.add_argument("--split_ratio", type=float, default=ICLR_FROZEN_SPLIT_RATIO)
    p.add_argument("--split_seed", type=int, default=ICLR_FROZEN_SPLIT_SEED)
    p.add_argument("--seed_path", type=str, default=str(DEFAULT_SEED_PATH))
    p.add_argument(
        "--base_prompt",
        type=str,
        default=None,
        help=(
            "Optional override file for # Task text. Frozen ICLR uses registered "
            "task_description for all 15 datasets; the legacy vanilla infer files "
            "are never loaded."
        ),
    )
    p.add_argument(
        "--max_observed_trials_per_participant",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Sparse-data: after the train/val/test split, keep at most N combined "
            "train+val observations per participant (test untouched). Same as TEH/MLE."
        ),
    )
    add_limited_data_cli_arguments(p)
    p.set_defaults(
        limited_data_protocol=ICLR_DEFAULT_LIMITED_DATA_PROTOCOL,
        limited_train_val=ICLR_FROZEN_LIMITED_TRAIN_VAL,
    )
    p.add_argument("--n_iterations", type=int, default=ICLR_FROZEN_N_ITERATIONS)
    p.add_argument(
        "--checkpoint_interval",
        type=int,
        default=ICLR_FROZEN_CHECKPOINT_INTERVAL,
        help=(
            "OE checkpoint cadence. Default 0 = only at end of the run (single dump). "
            "Positive N checkpoints every N iterations. After selection we still write "
            "best_program.py + compact evolution_*.{json,csv}; with --compact_oe_artifacts "
            "the bulky checkpoint trees are then deleted."
        ),
    )
    p.add_argument(
        "--max_prompt_train_trials",
        type=int,
        default=ICLR_FROZEN_MAX_PROMPT_TRAIN_TRIALS,
        help=(
            "Legacy name. Prompt-display cap on the combined observed train+validation "
            "union (T-PICS 60). Fitness still uses the complete retained train+val set. "
            "0 = no prompt cap. Under structure_aware_v3, SA40 participants have at most "
            "40 retained train+val observations (exact-40 for Kool), so 40 and 60 select "
            "the same examples whenever |TV|<=40."
        ),
    )
    p.add_argument("--max_prompt_trials_per_problem", type=int, default=5)
    p.add_argument("--model", type=str, default=ICLR_FROZEN_MODEL)
    p.add_argument("--api_base", type=str, default=None)
    p.add_argument("--vllm_url", type=str, default=os.environ.get("VLLM_LOCAL_URL", "http://localhost:8000/v1"))
    p.add_argument("--llm_api_key", type=str, default=os.environ.get("VLLM_LOCAL_API_KEY", "EMPTY"))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--llm_max_tokens", type=int, default=ICLR_FROZEN_LLM_MAX_TOKENS)
    p.add_argument("--max_model_len", type=int, default=ICLR_FROZEN_VLLM_MAX_MODEL_LEN)
    p.add_argument("--hard_prompt_token_cap", type=int, default=ICLR_FROZEN_INPUT_TOKEN_CEILING)
    p.add_argument("--llm_timeout", type=int, default=300)
    p.add_argument("--llm_retries", type=int, default=3)
    p.add_argument("--parallel_evaluations", type=int, default=ICLR_FROZEN_PARALLEL_EVALUATIONS)
    p.add_argument(
        "--parallel_participants",
        type=int,
        default=ICLR_FROZEN_PARALLEL_PARTICIPANTS,
        help="Number of participants to evolve concurrently (thread pool). ICLR freeze: 1.",
    )
    p.add_argument("--evaluator_timeout", type=int, default=120)
    p.add_argument("--evaluator_max_retries", type=int, default=2)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument(
        "--log_level",
        type=str,
        default=ICLR_FROZEN_LOG_LEVEL,
        help="OpenEvolve logger level (default WARNING to keep openevolve_output/logs small).",
    )
    p.add_argument(
        "--compact_oe_artifacts",
        action=argparse.BooleanOptionalAction,
        default=ICLR_FROZEN_COMPACT_OE_ARTIFACTS,
        help=(
            "After each person, keep best_program.py + evolution_summary.json + "
            "evolution_iteration_scores.csv, keep openevolve_output/best/, and delete "
            "bulky checkpoints/ plus replace logs/ with a stub (default: on)."
        ),
    )
    p.add_argument(
        "--skip_completed_participants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Person-level resume: if participant_*/results.json is already complete "
            "(status=ok, finite test_loglik, n_iterations_completed>=n_iterations, "
            "best_program.py loads), skip OpenEvolve for that person and reuse the "
            "row (default: on). Same --output_dir required. Does not resume mid-person "
            "OE iterations. Use --no-skip_completed_participants to force re-evolve."
        ),
    )
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
    p.add_argument("--num_top_programs", type=int, default=ICLR_FROZEN_NUM_TOP_PROGRAMS)
    p.add_argument(
        "--num_diverse_programs",
        type=int,
        default=ICLR_FROZEN_NUM_DIVERSE_PROGRAMS,
        help=(
            "Official OpenEvolve inspiration/diverse count (contextual examples, not co-parents). "
            "ICLR lean default 0 (official PromptConfig default at 411fb59 is 2)."
        ),
    )
    p.add_argument(
        "--include_artifacts",
        action=argparse.BooleanOptionalAction,
        default=ICLR_FROZEN_INCLUDE_ARTIFACTS,
        help="Official optional: render parent evaluation artifacts into the prompt. Lean default off.",
    )
    p.add_argument(
        "--include_previous_attempts",
        action=argparse.BooleanOptionalAction,
        default=ICLR_FROZEN_INCLUDE_PREVIOUS_ATTEMPTS,
        help="Official optional: inject previous_programs history into the prompt. Lean default off.",
    )
    p.add_argument(
        "--use_llm_feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Official optional: extra LLM code-quality scoring. Always off for ICLR.",
    )
    p.add_argument(
        "--cascade_evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Official optional: multi-stage cascade evaluation. Always off for ICLR.",
    )
    p.add_argument(
        "--enable_artifacts",
        action=argparse.BooleanOptionalAction,
        default=ICLR_FROZEN_ENABLE_ARTIFACTS,
        help="Official optional: capture evaluator artifacts side-channel. Lean default off.",
    )
    p.add_argument(
        "--log_prompts",
        action=argparse.BooleanOptionalAction,
        default=ICLR_FROZEN_LOG_PROMPTS,
        help="Official optional: store full prompts in the OE database. Lean default off.",
    )
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
    if args.parallel_participants < 1:
        raise ValueError(f"--parallel_participants must be >= 1, got {args.parallel_participants}")
    if args.parallel_evaluations < 1:
        raise ValueError(f"--parallel_evaluations must be >= 1, got {args.parallel_evaluations}")
    apply_iclr_frozen_range_ordinals(args)
    oe_sha = require_openevolve_checkout()
    if normalize_limited_data_protocol(args.limited_data_protocol) == LIMITED_DATA_PROTOCOL_OFF:
        # ICLR default is SA40 (limited_train_val=40). Full-data reruns pass --limited_data_protocol off.
        args.limited_train_val = None
    try:
        resolve_limited_data_budget(
            protocol=args.limited_data_protocol,
            max_observed_trials_per_participant=args.max_observed_trials_per_participant,
            limited_train_val=args.limited_train_val,
        )
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from e

    seed_resolved, _prompt_path = resolve_openevolve_seed_and_prompt(
        args.dataset,
        seed_path=args.seed_path,
        base_prompt=args.base_prompt,
    )
    args.seed_path = str(seed_resolved)

    parallel_participants = int(args.parallel_participants)
    parallel_evaluations = int(args.parallel_evaluations)
    approx_concurrency = parallel_participants * parallel_evaluations
    print(f"parallel_participants={parallel_participants}")
    print(f"parallel_evaluations={parallel_evaluations}")
    print(f"approx_total_concurrency={approx_concurrency}")
    print(f"openevolve_git_sha={oe_sha}")
    print(f"dataset={args.dataset} categorical={is_categorical_output_dataset(args.dataset)}")
    print(f"seed_path={args.seed_path}")
    print(f"base_prompt={args.base_prompt}")
    if approx_concurrency > 100:
        print(
            f"WARNING: approx_total_concurrency={approx_concurrency} > 100; "
            "consider lowering --parallel_participants or --parallel_evaluations."
        )

    psych_dataset_split = _effective_psych_dataset_split(args.dataset, args.psych_dataset_split)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else Path(
        openevolve_output_base_dir(args.dataset, timestamp, psych_dataset_split=psych_dataset_split)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_meta = dict(vars(args))
    run_meta["openevolve_git_sha"] = oe_sha
    run_meta["openevolve_expected_git_sha"] = EXPECTED_OPENEVOLVE_GIT_SHA
    run_meta["openevolve_root"] = str(_OPENVOLVE_ROOT)
    (run_dir / "run_config.json").write_text(
        json.dumps(run_meta, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "log").mkdir(exist_ok=True)

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
            try:
                apply_wandb_payload(
                    wandb_module,
                    wandb_completion_fields(
                        [],
                        expected_n=len(participants),
                        run_finished=False,
                        distinguish_failures=True,
                    ),
                )
            except Exception as e:
                print(f"wandb log failed: {e}")
        except Exception as e:
            print(f"wandb disabled (init failed): {e}")

    print(f"OpenEvolve vanilla baseline | dataset={args.dataset} | psych_split={psych_dataset_split}")
    print(f"HF corpus: {hf_id_for_psych_dataset_split(psych_dataset_split)}")
    print(f"Participants: {len(participants)} | n_iterations={args.n_iterations} | output={run_dir}")
    print(
        "Split: split_ratio=0.6 -> 60% train, remainder 50/50 val/test blocks; "
        "SA40 observed = retained train+val; evolution combined_score = trial-pooled "
        "mean loglik on the complete observed union; "
        f"prompt examples from train+val only (display cap "
        f"{int(args.max_prompt_train_trials)}, fitness uncapped); "
        "test scored once after selecting the best-by-observed-union program; "
        "dataset mean = equal-person mean of person-level test loglik."
    )
    print(
        "Prompt: vanilla/minimal (task + choose() contract + parent + train/val examples). "
        "OpenEvolve islands/MAP-Elites/archive still select one formal parent; official "
        "optional contextual blocks (artifacts, previous attempts, 3 island-best top, "
        "random.sample diverse leftover, 2 inspirations) are packed in 411fb59 template "
        "order and dropped before reducing observed examples if the 14000-token Qwen "
        "input ceiling is exceeded. The MAP-Elites database is not serialized into the prompt."
    )

    _patch_process_parallel_worker()
    _install_runtime_patches()
    _install_global_openevolve_controller_hook()

    detail_rows: List[Dict[str, Any]] = []
    detail_rows_lock = threading.Lock()

    def _participant_row(pid: int) -> Dict[str, Any]:
        try:
            return run_participant(args, pid, pid_to_ordinal.get(pid, -1), run_dir, wandb_module)
        except Exception as e:
            traceback.print_exc()
            return {
                "participant_id": pid,
                "participant_ordinal": pid_to_ordinal.get(pid, -1),
                "status": "failed",
                "error": str(e),
                "n_iterations_requested": args.n_iterations,
                "n_iterations_completed": 0,
            }

    def _record_participant_row(row: Dict[str, Any]) -> None:
        with detail_rows_lock:
            detail_rows.append(row)
            sorted_rows = sorted(detail_rows, key=lambda r: int(r.get("participant_id", 0)))
            with _SHARED_CSV_LOCK:
                _write_experiment_csvs(run_dir, sorted_rows)
            if wandb_module is not None:
                try:
                    _wandb_log_loglik_summary(
                        wandb_module,
                        sorted_rows,
                        last_row=row,
                        expected_n=len(participants),
                        run_finished=False,
                    )
                except Exception as e:
                    print(f"wandb log failed: {e}")

    if parallel_participants > 1:
        with ThreadPoolExecutor(max_workers=parallel_participants) as pool:
            futures = {pool.submit(_participant_row, pid): pid for pid in participants}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="participants"):
                _record_participant_row(fut.result())
    else:
        for pid in tqdm(participants, desc="participants"):
            _record_participant_row(_participant_row(pid))

    if wandb_module is not None:
        try:
            last = detail_rows[-1] if detail_rows else {}
            _wandb_log_loglik_summary(
                wandb_module,
                detail_rows,
                last_row=last,
                expected_n=len(participants),
                run_finished=True,
            )
        except Exception as e:
            print(f"wandb log failed: {e}")
        wandb_module.finish()

    rewrite_limited_data_csv_from_jsonl(
        run_dir / "log" / LIMITED_DATA_MANIFEST_JSONL_FILENAME,
        run_dir / "log" / LIMITED_DATA_MANIFEST_CSV_FILENAME,
    )
    print(f"Done. CSVs: {run_dir / 'participant_details_loglik.csv'}, {run_dir / 'summary_loglik.csv'}")


if __name__ == "__main__":
    # ``100 \ --no_log`` (backslash before space) makes bash pass ``' --no_log'``; strip that.
    sys.argv = [a.strip() if a.startswith(" ") else a for a in sys.argv]
    main()
