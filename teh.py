"""
teh.py — TEH (Template Evolution HuggingFace): Psych-101 binary cognitive datasets.

Evolved programs implement choose(problem, history) -> float = P(action=1) on structured trials
parsed from Psych-101 / Psych-101-test natural-language rows (see data_modules/psych101_binary.py).

Psych-101 binary aliases (e.g. peterson2021using) plus local mixed_gambles. Legacy choice13k/cpc18/
gridworld entrypoints are not supported here (use dataset aliases instead of choice13k).
"""

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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Tuple, Set
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
from data_modules.psych101_binary import (
    PSYCH101_BINARY_DATASETS,
    get_psych101_binary_experiments,
    is_psych101_dataset,
)
from utils.teh.teh_datasets import (
    IMPLEMENTED_PSYCH101_ALIASES,
    LOGlik_VAL_SPLIT_DATASETS,
    MIXED_GAMBLES,
    PARTICIPANT_DATASETS,
    is_binary_loglik_dataset,
    is_mixed_gambles_dataset,
    uses_train_val_test_loglik_split,
)
from utils.teh.teh_runtime import (
    DEFAULT_SEED_PROGRAM,
    TEH_WANDB_PROJECT,
    setup_teh_run_prompts,
    teh_output_base_dir,
    teh_wandb_run_name,
    valid_participant_ids_path,
)

_REPO_ROOT = Path(__file__).resolve().parent
_PARTICIPANT_DATASETS = PARTICIPANT_DATASETS
BEST_PROGRAM_FILENAME = "best_program.py"

# Serializes writes to experiment-level CSVs (participant_details_loglik, summary_loglik, etc.)
# when --parallel_participants is enabled. Per-participant dirs are not locked (isolated paths).
_SHARED_EXPERIMENT_CSV_LOCK = threading.Lock()
_RUN_PHASES = frozenset({"all", "evolution", "refine"})


def _elite_pool_capacity(sample_size: int, elite_pool_size: Optional[int]) -> int:
    """Max programs retained in the elite pool (after sorting by fitness, best first)."""
    if elite_pool_size is None:
        return max(sample_size * 2, 20)
    return max(1, int(elite_pool_size))


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


_WANDB_PARTICIPANT_CHART_SUFFIXES = (
    "train_loglik",
    "val_loglik",
    "test_loglik",
    "gated_test_loglik",
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
    """Log participant metrics to W&B with te_aggregate-style slash chart grouping."""
    wandb_module.log(_wandb_participant_chart_dict(log_dict, participant_id, step), step=step)


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
    return evaluate_choice13k_program(choose_fn, trials, verbose=verbose, n_seeds=n_seeds)


def _resolve_default_seed_program_path(args: Any, participant_id: int) -> Optional[str]:
    """Default seed path for TEH Psych-101 binary datasets."""
    if args.seed_path is not None:
        return args.seed_path
    return str(DEFAULT_SEED_PROGRAM)


def _parallel_generate_children(
    n_children: int,
    generate_one: Callable[[], str],
    *,
    max_workers: int = 5,
    desc: str = "Generating candidate programs",
) -> List[str]:
    """Run independent LLM child generations in parallel; preserve child index order."""
    if n_children <= 0:
        return []
    workers = max(1, min(int(max_workers), n_children))
    if workers == 1:
        return [generate_one() for _ in tqdm(range(n_children), desc=desc)]

    results: List[Optional[str]] = [None] * n_children
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(generate_one): i for i in range(n_children)}
        with tqdm(total=n_children, desc=f"{desc} (workers={workers})") as pbar:
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                pbar.update(1)
    return [r if r is not None else "" for r in results]


def load_valid_participant_ids_from_json(
    dataset: str,
    repo_root: Path,
    filter_mixed_gambles: bool = False,
    *,
    split_ratio: float = 0.8,
    split_seed: int = 42,
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
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
    if participant_scope == "single":
        if single_participant_id not in valid:
            raise ValueError(
                f"--single_participant_id={single_participant_id} is not in the precomputed valid list "
                f"({len(valid)} ids). Check datasets/*/valid_participant_ids.json or your id."
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
    """Extract executable Python from LLM output, removing prose/markdown wrappers.

    Strategy:
    1) Try python fenced blocks, then generic fenced blocks, then raw text.
    2) Keep candidates that satisfy required markers (if provided).
    3) If marker appears later in the text, also try trimming prefix prose.
    4) Return first syntax-valid candidate.
    """
    if not text:
        return ""

    candidates: List[str] = []
    candidates.extend(_extract_fenced_blocks(text))
    candidates.append(text.strip())

    expanded: List[str] = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        expanded.append(c)
        if required_markers:
            for marker in required_markers:
                i = c.find(marker)
                if i > 0:
                    expanded.append(c[i:].strip())

    seen = set()
    ordered: List[str] = []
    for c in expanded:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for c in ordered:
        if required_markers and not any(m in c for m in required_markers):
            continue
        if _passes_python_syntax(c):
            return c
    return ""


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


def compile_program(code_str: str) -> Optional[Callable]:
    """Safely compile program code and return choose callable if present."""
    # Provide minimal safe builtins needed for the program to run
    # Only include what's necessary for pure Python computation
    import builtins
    import math
    safe_builtins = {
        'zip': zip,
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'sum': sum,
        'abs': abs,
        'min': min,
        'max': max,
        'float': float,
        'int': int,
        'str': str,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'bool': bool,
        'isinstance': isinstance,
        'hasattr': hasattr,
        'getattr': getattr,
        '__import__': __import__,  # Needed for dynamic imports like __import__("math")
    }
    global_ns = {
        "__builtins__": safe_builtins,
        "__import__": __import__,  # Make __import__ directly available in global namespace
        "math": math,  # Pre-import math module for convenience
    }
    local_ns = {}
    try:
        exec(code_str, global_ns, local_ns)
    except Exception as e:
        # For debugging: uncomment to see what went wrong
        # print(f"Compilation error: {e}")
        return None
    choose_fn = local_ns.get("choose") or global_ns.get("choose")
    if callable(choose_fn):
        return choose_fn
    return None


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


def wrap_choose_with_consistency_gate(
    choose_fn: Callable,
    val_loglik: float,
) -> Callable:
    """Return choose(problem, history) that applies the external consistency gate."""

    def gated_choose(problem: Any, history: Any) -> float:
        p_raw = choose_fn(problem, history)
        raw_p = _parse_choice13k_choose_output(p_raw)
        return apply_consistency_gate_to_probability(raw_p, val_loglik)

    return gated_choose


def run_choice13k_gate_phase(
    choose_fn: Callable,
    val_loglik: float,
    test_trials: List[Dict[str, Any]],
    *,
    n_eval_seeds: int = 1,
) -> Optional[float]:
    """Evaluate test log-likelihood with external consistency gate (source unchanged)."""
    if not test_trials:
        return None
    try:
        gated_fn = wrap_choose_with_consistency_gate(choose_fn, float(val_loglik))
        gated_eval = evaluate_choice13k_program(gated_fn, test_trials, n_seeds=n_eval_seeds)
        return float(gated_eval["avg_loglik"])
    except Exception as e:
        print(f"Warning: gate phase failed: {e}")
        return None


def _refinement_combined_fitness(train_loglik: float, val_loglik: float) -> float:
    """Refinement selection fitness: equal weight on train and validation log-likelihood."""
    return 0.5 * float(train_loglik) + 0.5 * float(val_loglik)


def _train_loglik_from_elite_tuple(parent_tuple: Tuple[Any, ...]) -> float:
    """Train loglik from evolution or refinement elite tuple."""
    program_id = str(parent_tuple[3]) if len(parent_tuple) > 3 else ""
    if program_id.startswith("refinement_"):
        if len(parent_tuple) > 6 and parent_tuple[6] is not None:
            return float(parent_tuple[6])
    # Evolution pool: index 1 is train loglik; index 6 is train accuracy for legacy tuples.
    return float(parent_tuple[1])


def _evolution_elite_to_refinement_pool(
    elite_parents: List[Tuple[Any, ...]],
    elite_val_logliks: List[Optional[float]],
) -> Tuple[List[Tuple[Any, ...]], List[Optional[float]]]:
    """Copy evolution elite pool into refinement format; preserve evolution order (no sort)."""
    refine_parents: List[Tuple[Any, ...]] = []
    refine_vals: List[Optional[float]] = []
    for parent, val_ll in zip(elite_parents, elite_val_logliks):
        program_id = str(parent[3])
        train_ll = _train_loglik_from_elite_tuple(parent)
        val_ll_f = _safe_float(val_ll)
        combined = (
            _refinement_combined_fitness(train_ll, val_ll_f)
            if val_ll_f is not None
            else train_ll
        )
        refine_parents.append(
            (parent[0], combined, None, program_id, None, None, train_ll)
        )
        refine_vals.append(val_ll_f)
    return refine_parents, refine_vals


def _save_evolution_elite_pool(
    output_path: Path,
    elite_parents: List[Tuple[Any, ...]],
    elite_val_logliks: List[Optional[float]],
) -> Path:
    """Persist evolution-phase elite pool programs and manifest (evolution sort order)."""
    pool_dir = output_path / "evolution_elite_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for rank, (parent, val_ll) in enumerate(zip(elite_parents, elite_val_logliks)):
        program_id = str(parent[3])
        train_ll = _train_loglik_from_elite_tuple(parent)
        val_ll_f = _safe_float(val_ll)
        combined = (
            _refinement_combined_fitness(train_ll, val_ll_f)
            if val_ll_f is not None
            else train_ll
        )
        safe_name = re.sub(r"[^\w.\-]+", "_", program_id) or "program"
        filename = f"{rank:03d}_{safe_name}.py"
        (pool_dir / filename).write_text(parent[0] or "", encoding="utf-8")
        manifest.append(
            {
                "rank": rank,
                "program_id": program_id,
                "filename": filename,
                "train_loglik": train_ll,
                "val_loglik": val_ll_f,
                "train_val_loglik": combined,
                "evolution_fitness": _safe_float(parent[1]),
            }
        )
    (pool_dir / "pool_manifest.json").write_text(
        json.dumps({"n_programs": len(manifest), "programs": manifest}, indent=2),
        encoding="utf-8",
    )
    return pool_dir


def _load_evolution_elite_pool(
    pool_dir: Path,
) -> Tuple[List[Tuple[Any, ...]], List[Optional[float]]]:
    """Load evolution elite pool from saved manifest + program files."""
    manifest_path = pool_dir / "pool_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing evolution elite pool manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    programs = payload.get("programs", payload if isinstance(payload, list) else [])
    elite_parents: List[Tuple[Any, ...]] = []
    elite_val_logliks: List[Optional[float]] = []
    for entry in programs:
        filename = entry["filename"]
        code = (pool_dir / filename).read_text(encoding="utf-8")
        train_ll = float(entry["train_loglik"])
        val_ll = _safe_float(entry.get("val_loglik"))
        combined = _safe_float(entry.get("train_val_loglik"))
        if combined is None and val_ll is not None:
            combined = _refinement_combined_fitness(train_ll, val_ll)
        elif combined is None:
            combined = train_ll
        program_id = str(entry.get("program_id", filename))
        elite_parents.append(
            (code, combined, None, program_id, None, None, train_ll)
        )
        elite_val_logliks.append(val_ll)
    return elite_parents, elite_val_logliks


def _collect_pooled_train_trials_for_participants(
    dataset: str,
    participant_ids: List[int],
    *,
    split_ratio: float,
    split_seed: int,
    data_path: str = "data",
    filter_mixed_gambles: bool = False,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
) -> List[Dict[str, Any]]:
    """Concatenate per-participant train splits (same splits as evolution uses)."""
    pooled: List[Dict[str, Any]] = []
    for pid in participant_ids:
        train_trials, _, _ = _trials_for_loglik_participant(
            dataset,
            int(pid),
            split_ratio=split_ratio,
            split_seed=split_seed,
            data_path=data_path,
            filter_mixed_gambles=filter_mixed_gambles,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
        )
        pooled.extend(train_trials)
    return pooled


def _save_global_elite_pool(
    global_dir: Path,
    elite_parents: List[Tuple[Any, ...]],
) -> Path:
    """Persist global-phase elite pool (global sort order)."""
    pool_dir = global_dir / "global_elite_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for rank, parent in enumerate(elite_parents):
        program_id = str(parent[3])
        train_ll = _train_loglik_from_elite_tuple(parent)
        safe_name = re.sub(r"[^\w.\-]+", "_", program_id) or "program"
        filename = f"{rank:03d}_{safe_name}.py"
        (pool_dir / filename).write_text(parent[0] or "", encoding="utf-8")
        manifest.append(
            {
                "rank": rank,
                "program_id": program_id,
                "filename": filename,
                "global_train_loglik": train_ll,
                "global_fitness": _safe_float(parent[1]),
            }
        )
    (pool_dir / "pool_manifest.json").write_text(
        json.dumps({"n_programs": len(manifest), "programs": manifest}, indent=2),
        encoding="utf-8",
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


def _global_elite_to_participant_elite(
    global_parents: List[Tuple[Any, ...]],
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    *,
    dataset: str,
    n_eval_seeds: int,
) -> Tuple[List[Tuple[Any, ...]], List[Optional[float]]]:
    """Map global pool into participant elite tuples; preserve global order (no sort)."""
    elite_parents: List[Tuple[Any, ...]] = []
    elite_val_logliks: List[Optional[float]] = []
    for parent in global_parents:
        code = parent[0]
        src_id = str(parent[3])
        program_id = src_id if src_id.startswith("global_") else f"global_{src_id}"
        train_ll = _train_loglik_from_elite_tuple(parent)
        val_ll: Optional[float] = None
        test_acc = 0.0
        train_acc = 0.0
        choose_fn = compile_program(code)
        if choose_fn is not None:
            train_eval = _evaluate_loglik_for_dataset(
                dataset, choose_fn, train_trials, n_seeds=n_eval_seeds
            )
            train_ll = float(train_eval["avg_loglik"])
            train_acc = float(train_eval["accuracy"])
            test_acc = train_acc
            if val_trials:
                val_eval = _evaluate_loglik_for_dataset(
                    dataset, choose_fn, val_trials, n_seeds=n_eval_seeds
                )
                val_ll = float(val_eval["avg_loglik"])
        elite_parents.append(
            (code, train_ll, test_acc, program_id, None, None, train_acc)
        )
        elite_val_logliks.append(val_ll)
    return elite_parents, elite_val_logliks


def run_global_evolution_phase(
    *,
    dataset: str,
    participants: List[int],
    seed_program_path: str,
    n_iterations: int,
    n_candidates_per_iteration: int,
    sample_size: int,
    sample_parents: bool,
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
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
) -> List[Tuple[Any, ...]]:
    """
    Cross-participant evolution on pooled train trials (loglik fitness).

    Runs before per-participant evolution when ``--global_phase`` and ``--phase all``.
    """
    participant_ids = [int(p) for p in participants]
    pooled_train = _collect_pooled_train_trials_for_participants(
        dataset,
        participant_ids,
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
    print(f"\n{'='*80}")
    print(
        f"Global phase: {n_iterations} iteration(s), "
        f"{len(participant_ids)} participant(s), {len(pooled_train)} pooled train trials"
    )
    print(f"{'='*80}")

    global_dir = output_dir / "global_phase"
    if save_artifacts:
        global_dir.mkdir(parents=True, exist_ok=True)

    seed_code = load_seed_program(seed_program_path)
    seed_fn = compile_program(seed_code)
    if seed_fn is None:
        raise RuntimeError(f"Failed to compile seed program for global phase: {seed_program_path}")
    baseline_eval = _evaluate_loglik_for_dataset(
        dataset, seed_fn, pooled_train, n_seeds=n_eval_seeds
    )
    baseline_ll = float(baseline_eval["avg_loglik"])
    elite_parents: List[Tuple[Any, ...]] = [
        (
            seed_code,
            baseline_ll,
            None,
            "global_baseline",
            None,
            None,
            baseline_ll,
        )
    ]
    print(f"Global baseline train loglik: {baseline_ll:.6f}")

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

        num_parents_to_use = min(sample_size, len(elite_parents))
        if sample_parents and len(elite_parents) > 0:
            rng = np.random.default_rng(
                int(split_seed) + 50_000 + int(iteration_step) * 1_000_003
            )
            idxs = rng.choice(len(elite_parents), size=num_parents_to_use, replace=False)
            selected_parents = [elite_parents[int(j)] for j in idxs]
            print(
                f"\nUsing {num_parents_to_use} sampled global parent(s) "
                f"(pool size={len(elite_parents)}):"
            )
        else:
            selected_parents = elite_parents[:num_parents_to_use]
            print(f"\nUsing top {num_parents_to_use} global parent(s):")

        for i, parent_tuple in enumerate(selected_parents):
            prog_id = parent_tuple[3]
            train_ll = _train_loglik_from_elite_tuple(parent_tuple)
            print(f"  Parent {i+1}: {prog_id} (global_train_loglik={train_ll:.4f})")

        parent_codes = [p[0] for p in selected_parents]
        parent_train_lls = [_train_loglik_from_elite_tuple(p) for p in selected_parents]

        candidate_codes = generate_program_variants(
            client=client,
            model_name=model_name,
            parent_programs=parent_codes,
            train_trials=pooled_train,
            n_variants=n_candidates_per_iteration,
            max_tokens=llm_max_tokens,
            dataset=dataset,
            parent_train_accuracies=parent_train_lls,
            max_prompt_train_trials=max_prompt_train_trials,
            max_prompt_trials_per_problem=max_prompt_trials_per_problem,
            prompt_train_trials_seed=int(split_seed) + 60_000 + iteration_step,
            fitness_metric="loglik",
            max_workers=max_workers,
            run_prompts_dir=run_prompts_dir,
        )

        selected_results: List[Dict[str, Any]] = []
        for idx, code in enumerate(candidate_codes):
            if iter_dir is not None:
                (iter_dir / "candidates" / f"candidate_{idx}.py").write_text(code or "")
            code = _sanitize_llm_python_candidate(code, required_markers=("def choose(",))
            if not code:
                continue
            choose_fn = compile_program(code)
            if choose_fn is None:
                continue
            try:
                train_eval = _evaluate_loglik_for_dataset(
                    dataset, choose_fn, pooled_train, n_seeds=n_eval_seeds
                )
            except (AssertionError, TypeError, ValueError):
                continue
            if train_eval.get("errors", 0) != 0:
                continue
            train_loglik = float(train_eval["avg_loglik"])
            selected_results.append(
                {
                    "idx": idx,
                    "code": code,
                    "train_loglik": train_loglik,
                    "fitness": train_loglik,
                }
            )

        if not selected_results:
            print("Warning: No runtime-valid global candidates; keeping elite pool.")
        else:
            selected_results.sort(key=lambda x: x["fitness"], reverse=True)
            best = selected_results[0]
            print(
                f"  Best global candidate {best['idx']}: "
                f"train_loglik={best['train_loglik']:.6f}"
            )
            for result in selected_results:
                program_id = f"global_iteration_{iteration_step}_candidate_{result['idx']}"
                elite_parents.append(
                    (
                        result["code"],
                        result["fitness"],
                        None,
                        program_id,
                        None,
                        None,
                        result["train_loglik"],
                    )
                )

        elite_parents.sort(key=lambda x: x[1], reverse=True)
        elite_cap = _elite_pool_capacity(sample_size, elite_pool_size)
        elite_parents = elite_parents[:elite_cap]
        pool_best_ll = _train_loglik_from_elite_tuple(elite_parents[0])
        print(
            f"\nGlobal elite set updated: {len(elite_parents)} programs "
            f"(cap={elite_cap}, pool_best_train_loglik={pool_best_ll:.6f})"
        )

        if iter_dir is not None:
            metrics = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_runtime_valid": len(selected_results),
                "pool_best_program_id": elite_parents[0][3],
                "pool_best_global_train_loglik": pool_best_ll,
            }
            (iter_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )

        if wandb_module is not None:
            wandb_module.log(
                {
                    "global/train_loglik": pool_best_ll,
                    "global/pool_size": len(elite_parents),
                    "global/iteration": iteration_step,
                },
                step=iteration_step,
            )

    if save_artifacts:
        pool_dir = _save_global_elite_pool(global_dir, elite_parents)
        print(f"Saved global elite pool ({len(elite_parents)} programs) -> {pool_dir}")
        pool_best_code = elite_parents[0][0]
        pool_best_global_train_ll = _train_loglik_from_elite_tuple(elite_parents[0])
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
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
        )
        global_results: Dict[str, Any] = {
            "phase": "global",
            "dataset": dataset,
            "n_participants": len(participant_ids),
            "participant_ids": participant_ids,
            "n_iterations": int(n_iterations),
            "n_pooled_train_trials": len(pooled_train),
            "pool_size": len(elite_parents),
            "pool_best_program_id": str(elite_parents[0][3]),
            "pool_best_global_train_loglik": pool_best_global_train_ll,
            "baseline_global_train_loglik": baseline_ll,
        }
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


def _cap_and_subsample_prompt_trials(
    trials: List[Dict[str, Any]],
    *,
    max_trials: int,
    max_trials_per_problem: int,
    subsample_seed: int,
    label: str,
) -> List[Dict[str, Any]]:
    """Select trials for LLM prompts: sample blocks then optional extra singles.

    When ``max_trials_per_problem`` > 0, sample ``max_trials // max_trials_per_problem`` blocks
    (up to ``max_trials_per_problem`` trials each), then ``max_trials % max_trials_per_problem``
    additional trials from the remainder pool. When ``max_trials_per_problem`` is 0, sample up to
    ``max_trials`` trials with no block structure. ``max_trials`` <= 0 disables capping.
    """
    trials = list(trials)
    if max_trials <= 0:
        return trials
    if not trials:
        return []

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

    out = _sort_chronological(out)
    extra_note = f", +{n_extra} extra trial(s)" if n_extra else ""
    print(
        f"[LLM prompt] Using {len(out)} of {len(trials)} {label} trials "
        f"({n_blocks_sampled} sampled block(s) x up to {per_block}{extra_note}, "
        f"max={max_trials}, seed={subsample_seed})."
    )
    return out


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
    sample_size: int,
    sample_parents: bool,
    elite_pool_size: Optional[int],
    participant_id: int,
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
    run_prompts_dir: Optional[str] = None,
) -> Optional[float]:
    """
    Refinement: val trials in prompt; pool sorted by train_val_loglik only after iteration 1+.

    When ``evolution_elite_parents`` / ``evolution_elite_val_logliks`` are provided, the full
    evolution elite pool is copied in evolution order (no re-sort). Otherwise falls back to a
    single-program pool from ``initial_code``.
    """
    if not val_trials or not test_trials or n_iterations < 1:
        return None

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
            initial_train_loglik, initial_val_loglik
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

        num_parents_to_use = min(sample_size, len(elite_parents))
        if sample_parents and len(elite_parents) > 0:
            pid_key = int(participant_id) if participant_id is not None else 0
            rng = np.random.default_rng(
                int(split_seed) + 9_000_000 + int(iteration_step) * 1_000_003 + pid_key * 17_179
            )
            idxs = rng.choice(len(elite_parents), size=num_parents_to_use, replace=False)
            selected_parents = [elite_parents[int(j)] for j in idxs]
            selected_val_lls = [elite_val_logliks[int(j)] for j in idxs]
            print(
                f"\nUsing {num_parents_to_use} sampled refinement parent(s) "
                f"(elite pool size={len(elite_parents)}):"
            )
        else:
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

        print(
            f"[LLM prompt] Refinement uses {len(val_for_prompt)} validation trials only "
            f"(not train+val; cap via max_prompt_train_trials={max_prompt_train_trials})."
        )
        candidate_codes = generate_program_variants(
            client=client,
            model_name=model_name,
            parent_programs=parent_codes,
            train_trials=train_trials,
            n_variants=n_candidates_per_iteration,
            max_tokens=llm_max_tokens,
            dataset=dataset,
            parent_train_accuracies=parent_train_lls,
            parent_val_logliks=selected_val_lls,
            max_prompt_train_trials=max_prompt_train_trials,
            max_prompt_trials_per_problem=max_prompt_trials_per_problem,
            prompt_train_trials_seed=split_seed,
            fitness_metric=fitness_metric,
            max_workers=max_workers,
            prompt_suffix=refine_suffix,
            prompt_observation_trials=val_for_prompt,
            run_prompts_dir=run_prompts_dir,
        )

        candidate_results: List[Dict[str, Any]] = []
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
            choose_fn = compile_program(code)
            if choose_fn is None:
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
            except (AssertionError, TypeError, ValueError):
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
            train_loglik = float(train_eval["avg_loglik"])
            val_loglik = float(val_eval["avg_loglik"])
            train_val_loglik = (
                _refinement_combined_fitness(train_loglik, val_loglik)
                if runtime_valid
                else float("-inf")
            )
            candidate_results.append(
                {
                    "idx": idx,
                    "code": code,
                    "train_val_loglik": train_val_loglik,
                    "train_loglik": train_loglik,
                    "val_loglik": val_loglik,
                    "runtime_valid": runtime_valid,
                }
            )

        selected_results = [r for r in candidate_results if r.get("runtime_valid", False)]
        iter_best_result: Optional[Dict[str, Any]] = None
        if not selected_results:
            print("Warning: No runtime-valid refinement candidates; keeping elite pool.")
        else:
            selected_results.sort(key=lambda x: x["train_val_loglik"], reverse=True)
            iter_best_result = selected_results[0]
            print(
                f"  Best refinement candidate {iter_best_result['idx']}: "
                f"train_val_loglik={iter_best_result['train_val_loglik']:.4f} "
                f"(0.5*train+0.5*val), train_loglik={iter_best_result['train_loglik']:.4f}, "
                f"val_loglik={iter_best_result['val_loglik']:.4f}"
            )
            for result in selected_results:
                program_id = f"refinement_{iteration_step}_candidate_{result['idx']}"
                elite_parents.append(
                    (
                        result["code"],
                        result["train_val_loglik"],
                        None,
                        program_id,
                        None,
                        None,
                        result["train_loglik"],
                    )
                )
                elite_val_logliks.append(result.get("val_loglik"))

        paired = list(zip(elite_parents, elite_val_logliks))
        if iteration_step >= 1:
            paired.sort(key=lambda x: x[0][1], reverse=True)
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
            metrics: Dict[str, Any] = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_runtime_valid": len(selected_results),
                "candidate_results": [
                    {
                        "idx": r["idx"],
                        "train_loglik": r.get("train_loglik"),
                        "val_loglik": r.get("val_loglik"),
                        "train_val_loglik": r.get("train_val_loglik"),
                        "runtime_valid": r.get("runtime_valid", False),
                    }
                    for r in candidate_results
                ],
                **pool_metrics,
            }
            if iter_best_result is not None:
                metrics["iter_best_program_id"] = (
                    f"refinement_{iteration_step}_candidate_{iter_best_result['idx']}"
                )
                metrics["iter_best_train_val_loglik"] = iter_best_result["train_val_loglik"]
                metrics["iter_best_train_loglik"] = iter_best_result["train_loglik"]
                metrics["iter_best_val_loglik"] = iter_best_result["val_loglik"]
            else:
                metrics["iter_best_program_id"] = None
                metrics["iter_best_train_val_loglik"] = None
                metrics["iter_best_train_loglik"] = None
                metrics["iter_best_val_loglik"] = None
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
                    float(initial_train_loglik), float(initial_val_loglik)
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
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
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
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
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
    local_dataset: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    experiments = get_psych101_binary_experiments(
        dataset_alias,
        n_participants=int(participant_id) + 1,
        split="test",
        local_dataset=local_dataset,
    )
    if participant_id >= len(experiments):
        raise ValueError(
            f"participant_id={participant_id} out of range (only {len(experiments)} experiments loaded)."
        )
    train_trials, val_trials, test_trials, _options = split_trials(
        experiments[participant_id], split_ratio=split_ratio, split_seed=split_seed
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
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
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
        raise ValueError(f"Unsupported TEH dataset for loglik split: {dataset!r}")
    return _psych101_trials_for_participant(
        dataset,
        participant_id,
        split_ratio=split_ratio,
        split_seed=split_seed,
        local_dataset=local_dataset,
    )


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
    sample_size: int,
    sample_parents: bool,
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
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
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
    train_trials, val_trials, test_trials = _trials_for_loglik_participant(
        dataset,
        participant_id,
        split_ratio=split_ratio,
        split_seed=split_seed,
        data_path=data_path,
        filter_mixed_gambles=filter_mixed_gambles,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
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
        "sample_size": sample_size,
        "sample_parents": sample_parents,
        "elite_pool_size": elite_pool_size,
        "participant_id": int(participant_id),
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
        "run_prompts_dir": run_prompts_dir,
    }
    if evolution_pool_dir.is_dir() and (evolution_pool_dir / "pool_manifest.json").exists():
        ref_parents, ref_vals = _load_evolution_elite_pool(evolution_pool_dir)
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
    sample_size: int,
    sample_parents: bool,
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
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
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
            sample_size=sample_size,
            sample_parents=sample_parents,
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
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
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
    """Convert trials to numbered text for prompt.
    
    Supports both Choice13k and CPC18 formats.
    
    Args:
        trials: List of trial dictionaries
        dataset: "choice13k" or "cpc18"
    """
    lines = []
    for idx, t in enumerate(trials):
        if dataset == "cpc18":
            # CPC18 format: problem has Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr
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
            # Choice13k format
            prob_a = t["problem"]["gamble_A"]["probs"]
            rew_a = t["problem"]["gamble_A"]["rewards"]
            prob_b = t["problem"]["gamble_B"]["probs"]
            rew_b = t["problem"]["gamble_B"]["rewards"]
            has_fb = t["problem"].get("has_feedback", False)
            action = t["action"]
            lines.append(
                f"{idx+1}. Problem: Option A probs {prob_a} rewards {rew_a}; "
                f"Option B probs {prob_b} rewards {rew_b}; has_feedback={has_fb}; "
                f"Observed action: {action}"
            )
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

    for seed in range(n_seeds):
        correct = 0
        errors = 0
        for t in trials:
            try:
                pred = choose_fn(t["problem"], t["history"])
                if pred is not None and pred == t["action"]:
                    correct += 1
            except Exception as e:
                errors += 1
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

    def _one_pass(seed_idx: int) -> Tuple[float, float, int]:
        loglik_acc = 0.0
        correct = 0
        errors = 0
        for t in trials:
            y = int(t["action"])
            try:
                p_raw = choose_fn(t["problem"], t["history"])
                p_use = _parse_choice13k_choose_output(p_raw)
            except Exception as e:
                errors += 1
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

    def _one_pass(seed_idx: int) -> Tuple[float, float, int]:
        loglik_acc = 0.0
        correct = 0
        errors = 0
        for t in trials:
            y = int(t["action"])
            try:
                p_raw = choose_fn(t["problem"], t["history"])
            except Exception as e:
                errors += 1
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
                            pred = choose_fn(trial["problem"], trial["history"])
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


def generate_gridworld_initial_candidates(
    client: OpenAI,
    model_name: str,
    template_code: str,
    prefix_text: str,
    n_candidates: int,
    max_tokens: int = 2000,
    max_workers: int = 5,
) -> List[str]:
    """Generate K initial candidate programs for one episode. Prompt injects episode prefix (ROTE-style).
    Used at episode start; no parent code, only environment description + prefix observations + template.
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    prompt_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "gridworld", "non_strict", "infer_single_fsm.txt")
    code_template_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "gridworld", "non_strict", "single_code_template.txt")
    try:
        base_prompt = open(prompt_path).read()
        code_template = open(code_template_path).read()
    except FileNotFoundError:
        base_prompt = "You are a robot viewing agents acting in an object-centric environment. Model the agent's behavior as FSM code. Experiences (state, action):"
        code_template = "Implement the FSM code. Actions: [0,1,2,3,4,5] = stay, right, left, down, up, interact."
    full_prompt = f"""{base_prompt}

Observed trajectory (first 20 steps) for this episode:
{prefix_text}

{code_template}

Output ONLY runnable Python code (no explanations, no markdown fences, no preamble). Generate the variant now:"""

    def _generate_one() -> str:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
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

    def _generate_one() -> str:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
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
        code_template = open(code_template_path).read()
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
        code_template = """Please implement code to model the logic of the agent's behavior as demonstrated by the experiences. Here is the template for the agent's FSM class. Please implement the FSM code for an agent following the template. The code needs to be directly runnable on the inputs of state and return an action based on an observation. Make sure the agent always returns an action in the list [0, 1, 2, 3, 4, 5] corresponding to "stay", "right", "left", "down", "up", "interact"."""
    
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

{code_template}

Output format: Provide ONLY runnable Python code (no explanations, no markdown fences, no preamble).
The variant must be a complete, runnable program.

Generate the variant now:"""

    best_parent = parent_codes[0] if parent_codes else ""

    def _generate_one() -> str:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": base_prompt_template_final}],
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
        except Exception as e:
            print(f"Warning: Failed to generate gridworld program variant: {e}")
        return best_parent

    return _parallel_generate_children(
        n_variants,
        _generate_one,
        max_workers=max_workers,
        desc="Generating gridworld variants",
    )[:n_variants]


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
        code_template = open(code_template_path).read()
    except FileNotFoundError as e:
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
        - gamble_A: {"probs": List[float], "rewards": List[float]}
        - gamble_B: {"probs": List[float], "rewards": List[float]}
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

Provide only the code for choose(...) as a complete function body.
"""
            code_template = """
def choose(problem, history):
    \"\"\"
    problem: dict with gamble_A/gamble_B (probs, rewards), option_keys, has_feedback
    history: list of dicts with keys action (int) and feedback (float or None)
    return: float, probability of choosing option 1 (Option B)
    \"\"\"
    # Write your decision logic here.
    # You can use probabilities, rewards, and history.
    # Must return a single float in [0, 1].
    # The returned value is the probability of choosing Option B (action 1).
    return 0.5
"""
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
"""
            code_template = """
def choose(problem, history):
    \"\"\"
    problem: dict with gamble_A/gamble_B (probs, rewards), option_keys, has_feedback
    history: list of dicts with keys action (int) and feedback (float or None)
    return: int index (0 for Option A, 1 for Option B)
    \"\"\"
    # Write your decision logic here.
    # You can use probabilities, rewards, and history.
    # Must return 0 or 1.
    return 0
"""
    
    # Trials serialized into the prompt (evaluation still uses full train_trials elsewhere).
    # Refinement passes prompt_observation_trials=val only; evolution uses train_trials.
    obs_label = "validation" if prompt_observation_trials is not None else "train"
    if prompt_observation_trials is not None:
        # Refinement pre-samples val trials once for the whole phase; do not resample here.
        trials_for_prompt = list(prompt_observation_trials)
    else:
        trials_for_prompt = _cap_and_subsample_prompt_trials(
            train_trials,
            max_trials=max_prompt_train_trials,
            max_trials_per_problem=max_prompt_trials_per_problem,
            subsample_seed=prompt_train_trials_seed,
            label=obs_label,
        )
    # Note: dataset parameter not available here, but this function is only called for Choice13k/CPC18
    # We'll detect format from trial structure
    if trials_for_prompt and "problem" in trials_for_prompt[0]:
        if "gamble_A" in trials_for_prompt[0]["problem"]:
            dataset_type = "choice13k"
        else:
            dataset_type = "cpc18"
    else:
        dataset_type = "choice13k"
    state_text = format_trials_to_text(trials_for_prompt, dataset=dataset_type)

    extra_state_text = ""
    if extra_prompt_trials is not None and prompt_observation_trials is None:
        extra_for_prompt = _cap_and_subsample_prompt_trials(
            extra_prompt_trials,
            max_trials=max_prompt_train_trials,
            max_trials_per_problem=max_prompt_trials_per_problem,
            subsample_seed=int(prompt_train_trials_seed) + 424_242,
            label="validation",
        )
        extra_state_text = (
            f"\n\n{extra_prompt_trials_label}:\n"
            f"{format_trials_to_text(extra_for_prompt, dataset=dataset_type)}\n"
        )

    if prompt_suffix:
        base_prompt = f"{base_prompt.rstrip()}\n\n{prompt_suffix.strip()}\n"

    # Include parent programs as reference
    num_parents = len(parent_programs)
    if num_parents == 1:
        parent_context = f"\n\nReference program (parent):\n```python\n{parent_programs[0]}\n```\n\n"
        if (
            parent_train_accuracies
            and len(parent_train_accuracies) > 0
            and parent_train_accuracies[0] is not None
            and fitness_metric == "loglik"
            and is_binary_loglik_dataset(dataset)
        ):
            train_ll = parent_train_accuracies[0]
            if parent_val_logliks and len(parent_val_logliks) > 0 and parent_val_logliks[0] is not None:
                parent_context += (
                    f"Parent performance: train_loglik={train_ll:.4f}, "
                    f"val_loglik={parent_val_logliks[0]:.4f}\n\n"
                )
            else:
                parent_context += f"Parent performance: train_loglik={train_ll:.4f}\n\n"
        parent_context += "Generate a variant that improves upon or explores alternatives to the parent program.\n"
    else:
        parent_context = f"\n\nReference parent programs ({num_parents} elite programs):\n"
        for i, parent_program in enumerate(parent_programs):
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
                        metric_str = (
                            f" (train_loglik: {parent_metric:.4f}, "
                            f"val_loglik: {parent_val_logliks[i]:.4f})"
                        )
                    else:
                        metric_label = (
                            "train_loglik"
                            if fitness_metric == "loglik" and is_binary_loglik_dataset(dataset)
                            else "train_acc"
                        )
                        metric_str = f" ({metric_label}: {parent_metric:.4f})"
                else:
                    metric_str = ""
                parent_context += f"\nParent {i+1}{metric_str}:\n```python\n{parent_program}\n```\n"
        
        if dataset == "cpc18" and cpc18_official_mse and parent_train_mses:
            avg_mse = sum(parent_train_mses) / len(parent_train_mses)
            min_mse = min(parent_train_mses)
            parent_context += f"\nParent performance on training data:\n"
            parent_context += f"- Average train block-MSE: {avg_mse:.2f}\n"
            parent_context += f"- Best train block-MSE: {min_mse:.2f}\n"
            parent_context += f"\nIMPORTANT for CPC18:\n"
            parent_context += f"The official CPC18 metric is block-level MSE (lower is better).\n"
            parent_context += f"Your goal is to reduce block-level MSE.\n"
            parent_context += f"Current best: train_block-MSE={min_mse:.2f}\n"
            if min_mse > 50:
                parent_context += f"\nNOTE: Current MSE is HIGH (>50). Focus on reducing MSE significantly.\n"
            elif min_mse > 30:
                parent_context += f"\nNOTE: Current MSE is MODERATE (30-50). Try to reduce MSE further.\n"
            else:
                parent_context += f"\nNOTE: Current MSE is LOW (<30). Fine-tune to improve further.\n"
        
        parent_context += "\nGenerate a variant that combines the best ideas from these parent programs.\n"
    
    prompt_text = f"{base_prompt}\n{state_text}{extra_state_text}\n{parent_context}\n{code_template}"

    def _generate_one() -> str:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                top_p=0.95,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            return _sanitize_llm_python_candidate(content, required_markers=("def choose(",))
        except Exception as e:
            print(f"Warning: Failed to generate program variant: {e}")
            return ""

    return _parallel_generate_children(
        n_variants,
        _generate_one,
        max_workers=max_workers,
        desc="Generating candidate programs",
    )


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
    model_name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    client_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    wandb=None,
    n_eval_seeds: int = 3,
    sample_size: int = 10,
    sample_parents: bool = True,
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
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
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
    """
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
        print(
            f"[Split] Train: {len(train_trials)}, Val: {len(val_trials)}, Test: {len(test_trials)} "
            f"(seed={split_seed}, ratio={split_ratio:.3f})"
        )
    else:
        print(f"Loading Psych-101 dataset {dataset} for participant {participant_id}...")
        if choice13k_experiment is not None:
            exp = choice13k_experiment
        else:
            experiments = get_psych101_binary_experiments(
                dataset,
                n_participants=participant_id + 1,
                split="test",
                local_dataset=local_dataset,
            )
            exp = experiments[participant_id]
        train_trials, val_trials, test_trials, options = split_trials(
            exp, split_ratio=split_ratio, split_seed=split_seed
        )
        print(
            f"[Split] Train: {len(train_trials)}, Val: {len(val_trials)}, Test: {len(test_trials)} "
            f"(seed={split_seed}, ratio={split_ratio:.3f})"
        )
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        output_dir = (
            f"{teh_output_base_dir(dataset, timestamp)}/participant_{participant_id}"
        )
    output_path = Path(output_dir)
    if save_artifacts:
        output_path.mkdir(parents=True, exist_ok=True)
    
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
        if participant_id is not None:
            _wandb_log_participant_metrics(wandb, baseline_log_dict, int(participant_id), 0)
        else:
            wandb.log(baseline_log_dict, step=0)
        
        # Also save baseline to local JSONL file
        if log_file_path is not None:
            baseline_entry = {
                "step": 0,
                "iteration": -1,  # Baseline is before iteration 0
                **baseline_log_dict
            }
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
            _baseline_fit = (
                baseline_train_eval["avg_loglik"]
                if fitness_metric == "loglik"
                else baseline_train_eval["accuracy"]
            )
            elite_parents = [(
                seed_code,
                _baseline_fit,
                baseline_test_eval["accuracy"],
                "baseline",
                None,
                None,
                baseline_train_eval["accuracy"],
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
        )
        global_pool_handoff = True
        print(
            f"\nEvolution initial pool: {len(elite_parents)} program(s) from global phase "
            f"(global order preserved; per-participant sort by train loglik starts after iteration 1)."
        )
        if save_artifacts and output_path is not None:
            run_root = (
                output_path.parent
                if output_path.name.startswith("participant_")
                else output_path
            )
            src_pool = run_root / "global_phase" / "global_elite_pool"
            if src_pool.is_dir():
                dst_pool = output_path / "initial_pool_from_global"
                if dst_pool.exists():
                    shutil.rmtree(dst_pool)
                shutil.copytree(src_pool, dst_pool)
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

    # Evolution loop (uses elite_parents pool for parent selection, not a single parent_program)
    simple_iterations_rows: List[Dict[str, Any]] = []
    simple_iterations_dir = None
    if choice13k_simple_logging and is_binary_loglik_dataset(dataset) and save_artifacts:
        simple_iterations_dir = output_path / "iterations"
        simple_iterations_dir.mkdir(parents=True, exist_ok=True)
    for iteration in range(n_iterations):
        iteration_step = iteration + 1  # 1-indexed to match wandb (0 = baseline)
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
        num_parents_to_use = min(sample_size, len(elite_parents))
        if sample_parents and len(elite_parents) > 0:
            pid_key = int(participant_id) if participant_id is not None else 0
            rng = np.random.default_rng(
                int(split_seed) + int(iteration_step) * 1_000_003 + pid_key * 17_179
            )
            idxs = rng.choice(len(elite_parents), size=num_parents_to_use, replace=False)
            selected_parents = [elite_parents[int(j)] for j in idxs]
            print(
                f"\nUsing {num_parents_to_use} uniformly sampled parent(s) from elite pool "
                f"(size={len(elite_parents)}, sample_size={sample_size}, sample_parents=True):"
            )
        else:
            selected_parents = elite_parents[:num_parents_to_use]
            print(
                f"\nUsing {num_parents_to_use} top parent(s) from elite set "
                f"(sample_size={sample_size}, sample_parents=False):"
            )
        parent_codes = [p[0] for p in selected_parents]

        if is_cpc18_mse:
            for i, parent_tuple in enumerate(selected_parents):
                code, fitness, test_mse, prog_id, train_mse, test_mse = parent_tuple
                print(f"  Parent {i+1}: {prog_id} (train_mse={train_mse:.2f}, test_mse={test_mse:.2f}, fitness={fitness:.2f})")
        else:
            for i, parent_tuple in enumerate(selected_parents):
                code, fitness, test_acc, prog_id, _, _, train_acc_prompt = parent_tuple
                if fitness_metric == "loglik" and is_binary_loglik_dataset(dataset):
                    print(f"  Parent {i+1}: {prog_id} (train_loglik={fitness:.4f}, test_acc={test_acc:.4f})")
                else:
                    print(f"  Parent {i+1}: {prog_id} (train_acc={train_acc_prompt:.4f}, test_acc={test_acc:.4f})")
        
        parent_train_accs = None
        parent_train_mses = None
        parent_test_mses = None
        if is_cpc18_mse:
            parent_train_mses = [p[4] for p in selected_parents if p[4] is not None]
            parent_test_mses = [p[5] for p in selected_parents if p[5] is not None]
        else:
            # In loglik mode, feed train fitness (log-likelihood) into prompt context.
            if fitness_metric == "loglik" and is_binary_loglik_dataset(dataset):
                parent_train_accs = [p[1] for p in selected_parents]
            else:
                parent_train_accs = [p[6] for p in selected_parents]
        
        # Generate candidate programs (full code, not just parameters)
        print(f"\nGenerating {n_candidates_per_iteration} candidate programs...")
        candidate_codes = generate_program_variants(
            client=client,
            model_name=model_name,
            parent_programs=parent_codes,
            train_trials=train_trials,
            n_variants=n_candidates_per_iteration,
            max_tokens=llm_max_tokens,
            dataset=dataset,
            parent_train_accuracies=parent_train_accs,
            max_prompt_train_trials=max_prompt_train_trials,
            max_prompt_trials_per_problem=max_prompt_trials_per_problem,
            prompt_train_trials_seed=split_seed,
            fitness_metric=fitness_metric,
            cpc18_official_mse=False,
            max_workers=max_workers,
            run_prompts_dir=run_prompts_dir,
        )
        
        # Evaluate candidates
        print(f"\nEvaluating candidates...")
        candidate_results = []
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
                mse_valid = train_mse_eval.get("valid", True) and test_mse_eval.get("valid", True)
                if not mse_valid:
                    candidate_results.append({
                        "idx": idx,
                        "code": code,
                        "train_acc": train_eval["accuracy"],
                        "test_acc": test_eval["accuracy"],
                        "train_mse": float('inf'),
                        "test_mse": float('inf'),
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
                choose_fn = compile_program(code)
                _worst = float("-inf") if fitness_metric == "loglik" else 0.0
                if choose_fn is None:
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
                    # Always run test pass for runtime-valid checks; test metrics may be hidden later.
                    test_eval = evaluate_cpc18_split_program(choose_fn, test_trials, n_seeds=n_eval_seeds)
                    val_eval = (
                        evaluate_cpc18_split_program(choose_fn, val_trials, n_seeds=n_eval_seeds)
                        if val_trials
                        else None
                    )
                except (TypeError, ValueError, AssertionError):
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
                test_acc = test_eval["accuracy"] if test_eval is not None else None
                train_loglik = train_eval["avg_loglik"]
                test_loglik = test_eval["avg_loglik"] if test_eval is not None else None
                val_loglik = val_eval["avg_loglik"] if val_eval is not None else None
                runtime_valid = (train_eval.get("errors", 0) == 0) and (
                    test_eval is None or test_eval.get("errors", 0) == 0
                )
                fitness = train_loglik if fitness_metric == "loglik" else train_acc
                if not runtime_valid:
                    fitness = -1e9 if fitness_metric == "loglik" else float("-inf")
                _row = {
                    "idx": idx,
                    "code": code,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_loglik": train_loglik,
                    "test_loglik": test_loglik,
                    "fitness": fitness,
                    "train_correct": train_eval["correct"],
                    "test_correct": test_eval["correct"] if test_eval is not None else None,
                    "train_total": train_eval["total"],
                    "test_total": test_eval["total"] if test_eval is not None else None,
                    "valid": True,
                    "runtime_valid": runtime_valid,
                }
                if val_eval is not None:
                    _row["val_loglik"] = val_loglik
                candidate_results.append(_row)
            elif is_binary_loglik_dataset(dataset):
                choose_fn = compile_program(code)
                _worst = float("-inf") if fitness_metric == "loglik" else 0.0
                if choose_fn is None:
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
                    train_eval = evaluate_choice13k_program(choose_fn, train_trials, n_seeds=n_eval_seeds)
                    # Always run test pass for runtime-valid checks; test metrics may be hidden later.
                    test_eval = evaluate_choice13k_program(choose_fn, test_trials, n_seeds=n_eval_seeds)
                    val_eval = (
                        evaluate_choice13k_program(choose_fn, val_trials, n_seeds=n_eval_seeds)
                        if val_trials
                        else None
                    )
                except (AssertionError, TypeError, ValueError):
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
                test_acc = test_eval["accuracy"] if test_eval is not None else None
                train_loglik = train_eval["avg_loglik"]
                test_loglik = test_eval["avg_loglik"] if test_eval is not None else None
                val_loglik = val_eval["avg_loglik"] if val_eval is not None else None
                runtime_valid = (train_eval.get("errors", 0) == 0) and (
                    test_eval is None or test_eval.get("errors", 0) == 0
                )
                fitness = train_loglik if fitness_metric == "loglik" else train_acc
                if not runtime_valid:
                    fitness = -1e9 if fitness_metric == "loglik" else float("-inf")
                _row = {
                    "idx": idx,
                    "code": code,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_loglik": train_loglik,
                    "test_loglik": test_loglik,
                    "fitness": fitness,
                    "train_correct": train_eval["correct"],
                    "test_correct": test_eval["correct"] if test_eval is not None else None,
                    "train_total": train_eval["total"],
                    "test_total": test_eval["total"] if test_eval is not None else None,
                    "valid": True,
                    "runtime_valid": runtime_valid,
                }
                if val_eval is not None:
                    _row["val_loglik"] = val_loglik
                candidate_results.append(_row)
            
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
        if selected_results:
            runtime_valid_evolved_found = True
            # Sort by fitness (for CPC18: -MSE, for others: accuracy)
            selected_results.sort(key=lambda x: x["fitness"], reverse=True)
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
                print(f"\nTop performers (by train avg log-likelihood, higher is better):")
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
                    print(
                        f"  {i+1}. Candidate {result['idx']}: "
                        f"train_loglik={result['train_loglik']:.6f}, "
                        f"test_loglik={_test_ll}{_val_part}, "
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
                    elite_parents.append((
                        result["code"],
                        result["fitness"],
                        result["test_acc"],
                        program_id,
                        None,
                        None,
                        result["train_acc"],
                    ))
                else:
                    elite_parents.append((
                        result["code"],
                        result["fitness"],
                        result["test_acc"],
                        program_id,
                        None,
                        None,
                        result["train_acc"],
                    ))
                if track_elite_val_loglik:
                    elite_val_logliks.append(_safe_float(result.get("val_loglik")))

            # Sort elite set by fitness (descending) and keep top programs.
            # Global handoff: preserve global order through iteration 1; sort from iteration 2+.
            should_sort_elite = (not global_pool_handoff) or (iteration_step >= 1)
            if track_elite_val_loglik:
                paired_elite = list(zip(elite_parents, elite_val_logliks))
                if should_sort_elite:
                    paired_elite.sort(key=lambda x: x[0][1], reverse=True)
                elite_cap = _elite_pool_capacity(sample_size, elite_pool_size)
                paired_elite = paired_elite[:elite_cap]
                elite_parents = [p[0] for p in paired_elite]
                elite_val_logliks = [p[1] for p in paired_elite]
            else:
                if should_sort_elite:
                    elite_parents.sort(key=lambda x: x[1], reverse=True)
                elite_cap = _elite_pool_capacity(sample_size, elite_pool_size)
                elite_parents = elite_parents[:elite_cap]

            print(f"\nElite set updated: {len(elite_parents)} programs (elite_pool_cap={elite_cap})")

            # Use the updated elite-pool best for per-iteration reporting.
            iter_best_code, iter_best_fitness, _, iter_best_program_id = elite_parents[0][:4]
            iter_best_train_acc = best_result["train_acc"]
            iter_best_test_acc = best_result["test_acc"]
            iter_best_train_loglik = best_result.get("train_loglik")
            iter_best_test_loglik = best_result.get("test_loglik")
            iter_best_val_loglik = best_result.get("val_loglik")
            if fitness_metric == "loglik" and (is_cpc18_split or is_binary_loglik_dataset(dataset)):
                iter_best_fn = compile_program(iter_best_code)
                if iter_best_fn is not None:
                    if is_cpc18_split:
                        iter_best_train_eval = evaluate_cpc18_split_program(
                            iter_best_fn, train_trials, n_seeds=n_eval_seeds
                        )
                        iter_best_test_eval = evaluate_cpc18_split_program(
                            iter_best_fn, test_trials, n_seeds=n_eval_seeds
                        )
                    else:
                        iter_best_train_eval = evaluate_choice13k_program(
                            iter_best_fn, train_trials, n_seeds=n_eval_seeds
                        )
                        iter_best_test_eval = evaluate_choice13k_program(
                            iter_best_fn, test_trials, n_seeds=n_eval_seeds
                        )
                    iter_best_train_acc = iter_best_train_eval["accuracy"]
                    iter_best_test_acc = iter_best_test_eval["accuracy"]
                    iter_best_train_loglik = iter_best_train_eval["avg_loglik"]
                    iter_best_test_loglik = iter_best_test_eval["avg_loglik"]
                    iter_best_fitness = iter_best_train_loglik
                    if val_trials:
                        iter_best_val_eval = _evaluate_loglik_for_dataset(
                            dataset, iter_best_fn, val_trials, n_seeds=n_eval_seeds
                        )
                        iter_best_val_loglik = iter_best_val_eval["avg_loglik"]
                else:
                    # Should be rare; keep loop stable if a pool entry cannot recompile.
                    iter_best_test_acc = None
                    iter_best_test_loglik = None
                    iter_best_val_loglik = None

            if choice13k_simple_logging and is_binary_loglik_dataset(dataset) and save_artifacts and simple_iterations_dir is not None:
                (simple_iterations_dir / f"iteration_{iteration_step}.py").write_text(iter_best_code or "")
                _simp_row = {
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
                if fitness_metric == "loglik":
                    _test_better2 = (
                        iter_best_test_loglik is not None
                        and iter_best_test_loglik > overall_best_test["test_loglik"]
                    )
                else:
                    _test_better2 = best_result["test_acc"] > overall_best_test["test_accuracy"]
                if _test_better2:
                    overall_best_test = {
                        "train_accuracy": iter_best_train_acc,
                        "test_accuracy": iter_best_test_acc,
                        "train_loglik": iter_best_train_loglik,
                        "test_loglik": iter_best_test_loglik,
                        "program_id": iter_best_program_id,
                    }
                    if val_trials and fitness_metric == "loglik":
                        overall_best_test["val_loglik"] = iter_best_val_loglik
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
                if fitness_metric == "loglik":
                    _test_better = (
                        iter_best_test_loglik is not None
                        and iter_best_test_loglik > overall_best_test["test_loglik"]
                    )
                else:
                    _test_better = best_result["test_acc"] > overall_best_test["test_accuracy"]
                if _test_better:
                    overall_best_test = {
                        "train_accuracy": iter_best_train_acc,
                        "test_accuracy": iter_best_test_acc,
                        "train_loglik": iter_best_train_loglik,
                        "test_loglik": iter_best_test_loglik,
                        "program_id": iter_best_program_id,
                    }
                    if val_trials:
                        overall_best_test["val_loglik"] = iter_best_val_loglik
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
        
        # Save iteration results
        best_program_id = None
        if selected_results:
            best_program_id = iter_best_program_id
        
        if is_cpc18_mse:
            metrics = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
                "best_program_id": best_program_id,
                "candidate_results": [
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
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_mse": selected_results[0]["train_mse"] if selected_results else None,
                "best_test_mse": selected_results[0]["test_mse"] if selected_results else None,
            }
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
            metrics = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
                "best_program_id": best_program_id,
                "candidate_results": _cpc_cand_rows,
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_acc": iter_best_train_acc if selected_results else None,
                "best_test_acc": iter_best_test_acc if selected_results else None,
                "best_train_loglik": iter_best_train_loglik if selected_results else None,
                "best_test_loglik": iter_best_test_loglik if selected_results else None,
            }
            if val_trials:
                metrics["best_val_loglik"] = iter_best_val_loglik if selected_results else None
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
                _cand_rows.append(_cr)
            metrics = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
                "best_program_id": best_program_id,
                "candidate_results": _cand_rows,
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_acc": iter_best_train_acc if selected_results else None,
                "best_test_acc": iter_best_test_acc if selected_results else None,
                "best_train_loglik": iter_best_train_loglik if selected_results else None,
                "best_test_loglik": iter_best_test_loglik if selected_results else None,
            }
            if val_trials:
                metrics["best_val_loglik"] = iter_best_val_loglik if selected_results else None
        else:
            metrics = {
                "iteration": iteration_step,
                "n_candidates": n_candidates_per_iteration,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
                "best_program_id": best_program_id,
                "candidate_results": [
                    {
                        "idx": r["idx"],
                        "train_acc": r["train_acc"],
                        "test_acc": r["test_acc"],
                        "valid": r["valid"],
                        "runtime_valid": r.get("runtime_valid", r["valid"]),
                    }
                    for r in candidate_results
                ],
                "best_train_acc": best_fitness if selected_results else None,
                "best_test_acc": selected_results[0]["test_acc"] if selected_results else None,
            }
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
            }
        else:
            summary = {
                "iteration": iteration_step,
                "best_train_acc": best_fitness if selected_results else None,
                "best_test_acc": selected_results[0]["test_acc"] if selected_results else None,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
            }
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
                log_entry = {
                    "step": iteration + 1,
                    "iteration": iteration_step,
                    **log_dict
                }
                with open(log_file_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
    
    # Final summary and save comprehensive results.json
    print(f"\n{'='*80}")
    print("Evolution Complete")
    print(f"{'='*80}")

    # Select final best program directly from the final elite pool (already sorted by train fitness).
    # This guarantees final reporting is paired from one candidate.
    if ((is_binary_loglik_dataset(dataset)) or is_cpc18_split) and (not runtime_valid_evolved_found):
        print(
            "[WARN] No runtime-valid evolved program found; using baseline program for final reporting.",
            flush=True,
        )
        final_best_code = seed_code
        final_best_program_id = "baseline"
    else:
        final_best_code, _, _, final_best_program_id = (
            elite_parents[0][0],
            elite_parents[0][1],
            elite_parents[0][2],
            elite_parents[0][3],
        )
    match = re.match(r"iteration_(\d+)_candidate_(\d+)$", final_best_program_id)
    if match is not None:
        best_iteration = int(match.group(1))
        best_candidate_idx = int(match.group(2))
    elif final_best_program_id == "baseline":
        best_iteration = -1
        best_candidate_idx = -1
    else:
        best_iteration = None
        best_candidate_idx = None
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
            "program_file": best_program_filename,
        }
        if final_val_eval is not None:
            overall_best_train["val_loglik"] = final_val_eval["avg_loglik"]
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
            "program_file": best_program_filename,
        }
        overall_best_test = dict(overall_best_train)

    if save_artifacts and track_elite_val_loglik and output_path is not None:
        pool_dir = _save_evolution_elite_pool(
            output_path, elite_parents, elite_val_logliks
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
            elite_parents, elite_val_logliks
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
            sample_size=sample_size,
            sample_parents=sample_parents,
            elite_pool_size=elite_pool_size,
            participant_id=int(participant_id),
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
            run_prompts_dir=run_prompts_dir,
        )
        refinement_ran = gated_test_loglik is not None
        if gated_test_loglik is not None:
            overall_best_train["gated_test_loglik"] = gated_test_loglik
            overall_best_test["gated_test_loglik"] = gated_test_loglik
            if wandb is not None and participant_id is not None:
                refine_step = int(n_iterations) + int(refinement_iters)
                refine_log = {
                    f"p{participant_id}_gated_test_loglik": gated_test_loglik,
                    f"p{participant_id}_train_loglik": overall_best_train.get("train_loglik"),
                    f"p{participant_id}_val_loglik": overall_best_train.get("val_loglik"),
                    f"p{participant_id}_test_loglik": overall_best_train.get("test_loglik"),
                    f"p{participant_id}_train_fitness": overall_best_train.get("train_loglik"),
                    f"p{participant_id}_test_fitness": overall_best_test.get("test_loglik"),
                }
                _wandb_log_participant_metrics(wandb, refine_log, int(participant_id), refine_step)
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
                        wandb, gate_log, int(participant_id), int(n_iterations) + 1
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
        output_dir = f"generated_outputs/gridworld_ensemble/{mode}/run_{timestamp}/agent_{agent_id}"
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
    
    _teh_dataset_choices = sorted(PARTICIPANT_DATASETS)
    parser = argparse.ArgumentParser(
        description="TEH: Template Evolution on Psych-101 binary cognitive datasets (structured trials)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="peterson2021using",
        choices=_teh_dataset_choices,
        help=(
            "Psych-101 binary alias (peterson2021using, plonsky2018when, ...) or mixed_gambles (local CSV)."
        ),
    )
    parser.add_argument(
        "--mixed_gambles_csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="CSV path for --dataset mixed_gambles (default: datasets/mixed_gambles/data_all_2021-01-08.csv).",
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
        "--n_candidates",
        type=int,
        default=10,
        help="Number of candidate programs per iteration",
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
        default=3,
        help="Number of evaluation runs per program (averaged for final accuracy). Default: 3",
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
            "across_participants: pool train across participants (peterson2021using only)."
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
        "--max_prompt_train_trials",
        type=int,
        default=1_000_000,
        help=(
            "Max trials serialized into each LLM prompt. With --max_prompt_trials_per_problem > 0, "
            "sample (max // per_problem) blocks at up to per_problem trials each, plus "
            "(max %% per_problem) extra trials. With per_problem=0, flat-random sample of max trials. "
            "0 = no cap (full split in prompt). Default 1000000."
        ),
    )
    parser.add_argument(
        "--max_prompt_trials_per_problem",
        type=int,
        default=0,
        help=(
            "Trials per sampled block in LLM prompts (0 = flat sample of --max_prompt_train_trials only). "
            "Block = Choice13k/mixed_gambles gamble signature, CPC18 (problem_id, block_id), or problem_id."
        ),
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=800,
        help="Max output tokens per candidate generation request (reduces context-overflow failures).",
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

    args = parser.parse_args()
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
    if args.split_mode == "across_participants" and args.dataset != "peterson2021using":
        print("Error: --split_mode across_participants is only supported with --dataset peterson2021using.")
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
    if args.max_workers < 1:
        print("Error: --max_workers must be >= 1.")
        return
    if args.n_candidates < 1:
        print("Error: --n_candidates must be >= 1.")
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
    if args.phase == "evolution" and args.refinement_phase:
        print("Note: --refinement_phase is ignored when --phase evolution.")
    if args.elite_pool_size is not None and args.elite_pool_size < 1:
        print("Error: --elite_pool_size must be >= 1 when set.")
        return
    mixed_gambles_gain_loss_only = bool(getattr(args, "filter_mixed_gambles", False))

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
    
    # Optional wandb setup
    wandb_enabled = False
    wandb = None
    log_file_path = None
    if not args.no_log:
        try:
            import wandb as _wandb
            wandb = _wandb
            run_name = teh_wandb_run_name(
                args.dataset,
                timestamp,
                args.participant_scope,
                range_start=args.range_start_ordinal,
                range_end=args.range_end_ordinal,
                ordinals=args.ordinals,
            )
            wandb.init(
                project=TEH_WANDB_PROJECT,
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

    if wandb is not None and args.dataset in _PARTICIPANT_DATASETS:
        for pid in participants_to_process:
            wandb.define_metric(f"p{pid}_step")
            wandb.define_metric(f"p{pid}/*", step_metric=f"p{pid}_step")

    if args.dataset in _PARTICIPANT_DATASETS:
        if args.participant_scope == "single":
            print(
                f"Participant scope: single -> using raw participant id "
                f"{args.single_participant_id}."
            )
        elif args.participant_scope == "range":
            print(
                "Participant scope: range -> using inclusive ordinal slice "
                f"[{args.range_start_ordinal}, {args.range_end_ordinal}] from valid_participant_ids.json."
            )
        elif args.participant_scope == "ordinals":
            print(
                "Participant scope: ordinals -> using raw participant ids at 0-based ordinals "
                f"{list(args.ordinals)} from valid_participant_ids.json "
                "(duplicate ordinals collapse to one id; order follows first occurrence)."
            )
        else:
            cap_text = (
                f"first {args.all_max_participants} valid ids"
                if args.all_max_participants is not None
                else "all valid ids"
            )
            print(
                f"Participant scope: all -> using {cap_text} from valid_participant_ids.json."
            )
    print(
        f"TEH split settings: dataset={args.dataset}, split_mode={args.split_mode}, "
        f"split_ratio={args.split_ratio:.3f}, split_seed={args.split_seed}"
    )

    base_run_dir = None
    if args.output_dir is None:
        base_run_dir = teh_output_base_dir(args.dataset, timestamp)
        Path(base_run_dir).mkdir(parents=True, exist_ok=True)
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

    seed_program_path = _resolve_default_seed_program_path(args, 0)

    teh_client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    run_prompts_dir = setup_teh_run_prompts(
        Path(base_run_dir),
        args.dataset,
        Path(seed_program_path),
        client=teh_client,
        model_name=args.model_name,
        use_llm=not args.no_llm_prompt,
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
        filter_mixed_gambles=mixed_gambles_gain_loss_only,
    )
    print(f"TEH run prompts directory: {run_prompts_dir}")
    seed_program_path = str(run_prompts_dir / "seed_program.py")

    evolution_run_phase = "evolution" if args.phase == "evolution" else "all"

    global_elite_for_handoff: Optional[List[Tuple[Any, ...]]] = None
    if args.global_phase and args.phase == "all" and args.dataset in _PARTICIPANT_DATASETS:
        if seed_program_path is None:
            print("Error: --global_phase requires a seed program (--seed_path or dataset default).")
            if wandb is not None:
                wandb.finish()
            return
        global_client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
        global_elite_for_handoff = run_global_evolution_phase(
            dataset=args.dataset,
            participants=[int(p) for p in participants_to_process],
            seed_program_path=seed_program_path,
            n_iterations=args.global_iters,
            n_candidates_per_iteration=args.n_candidates,
            sample_size=args.sample_size,
            sample_parents=args.sample_parents,
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
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
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
        refine_client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
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
                sample_size=args.sample_size,
                sample_parents=args.sample_parents,
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
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
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

        max_pid = max(selected_participants)
        experiments = get_psych101_binary_experiments(
            args.dataset,
            n_participants=max_pid + 1,
            split="test",
            local_dataset=args.local_dataset,
        )
        train_trials: List[Dict[str, Any]] = []
        test_trials: List[Dict[str, Any]] = []
        for pid in train_participants:
            p_trials, _ = experiment_to_trials(experiments[pid])
            train_trials.extend(p_trials)
        for pid in test_participants:
            p_trials, _ = experiment_to_trials(experiments[pid])
            test_trials.extend(p_trials)
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
                model_name=args.model_name,
                client_kwargs=client_kwargs if client_kwargs else None,
                output_dir=base_run_dir,
                wandb=wandb,
                n_eval_seeds=args.n_eval_seeds,
                sample_size=args.sample_size,
                sample_parents=args.sample_parents,
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
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
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
                model_name=args.model_name,
                client_kwargs=client_kwargs if client_kwargs else None,
                output_dir=base_run_dir,
                wandb=wandb,
                n_eval_seeds=args.n_eval_seeds,
                sample_size=args.sample_size,
                sample_parents=args.sample_parents,
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
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
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
        output_dir = base_run_dir if base_run_dir else f"generated_outputs/gridworld/non_strict/run_{timestamp}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
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
                    agent_output_dir = f"generated_outputs/gridworld_ensemble/{mode}/run_{timestamp}/agent_{agent_id}"
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
                    participant_output_dir = f"generated_outputs/{out_subdir}/{mode}/run_{timestamp}/epoch_{epoch}/agent_{agent_id}"
                
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
                        model_name=args.model_name,
                        client_kwargs=client_kwargs if client_kwargs else None,
                        output_dir=participant_output_dir,
                        wandb=wandb,
                        n_eval_seeds=args.n_eval_seeds,
                        sample_size=args.sample_size,
                        sample_parents=args.sample_parents,
                        elite_pool_size=args.elite_pool_size,
                        filter_mixed_gambles=mixed_gambles_gain_loss_only,
                        fitness_metric=args.fitness_metric,
                        split_ratio=args.split_ratio,
                        split_seed=args.split_seed,
                        max_prompt_train_trials=args.max_prompt_train_trials,
                        max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                        llm_max_tokens=args.llm_max_tokens,
                        gate_phase=args.gate_phase,
                        run_phase=evolution_run_phase,
                        refinement_phase=args.refinement_phase,
                        refinement_iters=args.refinement_iters,
                        refinement_val_threshold=args.refinement_val_threshold,
                        max_workers=args.max_workers,
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

        def _run_main_loop_participant(participant_id: int) -> Optional[Dict[str, Any]]:
            print(f"\n{'='*80}")
            print(f"Processing participant {participant_id}")
            print(f"{'='*80}")
            participant_output_dir = _participant_output_dir(participant_id)
            if not parallel_participants:
                _ensure_summary_paths_from_participant_dir(participant_output_dir)

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
                model_name=args.model_name,
                client_kwargs=client_kwargs if client_kwargs else None,
                output_dir=participant_output_dir,
                wandb=wandb,
                n_eval_seeds=args.n_eval_seeds,
                sample_size=args.sample_size,
                sample_parents=args.sample_parents,
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
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
            )

        try:
            if parallel_participants:
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
        finally:
            if wandb is not None:
                wandb.finish()


if __name__ == "__main__":
    main()
