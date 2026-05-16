"""
ROTE_evo_non_strict.py: Iterative evolution loop over executable Choice13k and Gridworld programs.

Non-strict version: Generates full program code without parameter restrictions.
This version allows the LLM to generate entirely new choose(problem, history) implementations for Choice13k,
or full FSMAgent class implementations for Gridworld, not restricted to parameter-only changes.

The evolution process:
1. Starts with seed program (configurable via --seed_path)
2. Generates candidate program variants per iteration (full code, not just parameters)
3. Evaluates each program on dataset (Choice13k or Gridworld)
4. Reports performance and selects best performers
5. Uses best programs as parents for next generation

Participant selection (choice13k / cpc18 / mixed_gambles):
- --participant_scope single (default): one raw id via --single_participant_id (must be in
  datasets/*/valid_participant_ids.json from utils/tools/collect_participant_ids.py).
- --participant_scope range: raw ids = valid_list[--range_start_ordinal : --range_end_ordinal+1]
  (inclusive end, 0-based ordinals into that JSON list).
- --participant_scope ordinals: raw ids = valid_list[i] for each i in --ordinals (0-based ordinals, same list as range).
- --participant_scope all: every raw id in JSON, optional cap --all_max_participants (first N).
- Gridworld / gridworld_ensemble: use --num_agents_to_sample and --agent_id; participant_scope is ignored.

Choice13k within-participant split: problems (blocks) are partitioned with ``--split_ratio`` as the **train
fraction** of blocks; remaining blocks are split equally between validation and test. Per-trial history stays
within-block only. Across-participants mode still pools trials per participant, then splits the pooled test pool
into validation and test for data-driven runs.
"""

import math
import os
import re
import json
import csv
import difflib
import shlex
import shutil
import socket
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
from datetime import datetime
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax
from datasets import load_dataset

# Import data loading (this is acceptable as it's a data module, not ROTE/evo code)
from data_modules.choice13k import get_choice13k_experiments, Experiment, Block
from data_modules import choice13k as choice13k_module
from data_modules.cpc18 import load_cpc18_track2_data, split_cpc18_trials, ParticipantData
from agent import AgentExecutionFramework
from utils.rbu import (
    StructureScoreParseError,
    compute_rbu,
    count_tokens_approx,
    parse_all_participant_structure_scores,
)

# Repo root for datasets/*/valid_participant_ids.json (ordinal resolution for choice13k / cpc18 / mixed_gambles).
_REPO_ROOT = Path(__file__).resolve().parent
_PARTICIPANT_DATASETS = frozenset({"choice13k", "cpc18", "mixed_gambles"})

_RBU_STRUCTURE_SCORE_ALL_FILENAME = "Structure_score_all.txt"
_RBU_PREPARED_INSTRUCTION_HEADER = "\n\n## Dataset-specific scoring instruction (prepared once for this run)\n"
# cl100k_base (and char/3.5 fallback) often undercount vs the server's tokenizer for Qwen-class models.
_RBU_STRUCTURE_TOKEN_ESTIMATE_SLACK = 1.12


def _elite_pool_capacity(sample_size: int, elite_pool_size: Optional[int]) -> int:
    """Max programs retained in the elite pool (after sorting by fitness, best first)."""
    if elite_pool_size is None:
        return max(sample_size * 2, 20)
    return max(1, int(elite_pool_size))


def load_valid_participant_ids_from_json(
    dataset: str, repo_root: Path, filter_mixed_gambles: bool
) -> List[int]:
    """Load precomputed valid raw participant ids (same ordering as utils/tools/collect_participant_ids.py)."""
    if dataset == "choice13k":
        path = repo_root / "datasets" / "choice13k" / "valid_participant_ids.json"
    elif dataset == "cpc18":
        path = repo_root / "datasets" / "cpc18" / "valid_participant_ids.json"
    elif dataset == "mixed_gambles":
        name = (
            "valid_participant_ids_gain_loss.json"
            if filter_mixed_gambles
            else "valid_participant_ids.json"
        )
        path = repo_root / "datasets" / "mixed_gambles" / name
    else:
        raise ValueError(f"load_valid_participant_ids_from_json: unsupported dataset {dataset!r}")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing valid participant list: {path}. "
            f"Generate it with: python utils/tools/collect_participant_ids.py --dataset {dataset}"
            + (" --filter_mixed_gambles" if dataset == "mixed_gambles" and filter_mixed_gambles else "")
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["valid_participant_ids"])


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
) -> List[int]:
    """
    Build the list of raw participant ids to process for choice13k / cpc18 / mixed_gambles.

    - participant_scope=single: one raw id (--single_participant_id).
    - participant_scope=range: inclusive ordinal slice into valid_participant_ids.json.
    - participant_scope=ordinals: raw ids at listed 0-based ordinals (--ordinals), same ordering as range.
    - participant_scope=all: all raw ids from JSON, optionally capped by --all_max_participants (first N valid).
    """
    valid = load_valid_participant_ids_from_json(dataset, repo_root, filter_mixed_gambles)
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
from plot_and_eval import get_all_problem_configs, make_dataloader
from environment import AutomaticityEnv, State


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


def _cap_prompt_trials_per_problem(
    trials: List[Dict[str, Any]], max_trials_per_problem: int
) -> List[Dict[str, Any]]:
    """Cap prompt trials per problem (prompt-only; evaluation still uses full train split)."""
    if max_trials_per_problem <= 0:
        return list(trials)
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    order: List[Any] = []
    for t in trials:
        if "problem_id" in t:
            key = ("problem_id", t["problem_id"])
        else:
            p = t.get("problem", {})
            ga = p.get("gamble_A", {})
            gb = p.get("gamble_B", {})
            ga_probs = ga.get("probs", [])
            gb_probs = gb.get("probs", [])
            if ga_probs is None:
                ga_probs = []
            if gb_probs is None:
                gb_probs = []
            key = (
                "problem_sig",
                tuple(ga.get("rewards", [])),
                tuple(ga_probs),
                tuple(gb.get("rewards", [])),
                tuple(gb_probs),
            )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(t)
    capped: List[Dict[str, Any]] = []
    for key in order:
        capped.extend(grouped[key][:max_trials_per_problem])
    return capped


def load_mixed_gambles_data(
    csv_path: str,
    participant_id: int,
    filter_gain_loss_only: bool = False,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """Load mixed_gambles CSV, filter by subject == participant_id, and split by disjoint problem signatures.

    Each row: Option A (gamble) = [gain, loss] with probs [0.5, 0.5]; Option B (certain) = [cert] with probs [1.0].
    Raw CSV `took_gamble`: 1 = chose gamble, 0 = chose certain. TE option index: action = 1 - took_gamble
    (0 = Option A gamble_A / accept gamble, 1 = Option B gamble_B / certain). history = [] (no temporal dependence).

    Args:
        filter_gain_loss_only: If True, keep only gamble_type == "gain_loss" trials (Section 4.2 mixed gambles).
            If False (default), include all trial types.
    """
    option_keys = [0, 1]  # 0 = Option A (gamble_A), 1 = Option B (gamble_B certain)
    all_trials = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["subject"]) != participant_id:
                continue
            # Optional: use only mixed-gamble trials (gain_loss). Section 4.2 models 165 mixed gambles per participant.
            if filter_gain_loss_only and row.get("gamble_type") != "gain_loss":
                continue
            gain, loss, cert = float(row["gain"]), float(row["loss"]), float(row["cert"])
            took_gamble = int(row["took_gamble"])
            action = 1 - took_gamble
            all_trials.append({
                "problem": {
                    "gamble_A": {"rewards": [gain, loss], "probs": [0.5, 0.5]},
                    "gamble_B": {"rewards": [cert], "probs": [1.0]},
                    "option_keys": option_keys,
                    "has_feedback": False,
                },
                "history": [],
                "options": option_keys,
                "action": action,
                "problem_signature": (gain, loss, cert),
            })
    if len(all_trials) == 0:
        raise ValueError(f"No rows found for subject {participant_id} in {csv_path}")
    if filter_gain_loss_only and not getattr(load_mixed_gambles_data, "_printed_gain_loss", False):
        print("[Mixed Gambles] Using gain_loss trials only.")
        load_mixed_gambles_data._printed_gain_loss = True
    # Split by unique (gain, loss, cert) signatures so train/test never share the same problem.
    signatures = sorted({t["problem_signature"] for t in all_trials})
    if len(signatures) < 2:
        raise ValueError(
            f"mixed_gambles participant {participant_id} has <2 unique problems; cannot build disjoint train/test."
        )
    rng = np.random.default_rng(split_seed)
    shuffled = list(signatures)
    rng.shuffle(shuffled)
    split_point = int(len(shuffled) * split_ratio)
    split_point = max(1, min(split_point, len(shuffled) - 1))
    train_sigs = set(shuffled[:split_point])
    test_sigs = set(shuffled[split_point:])
    train_trials = [t for t in all_trials if t["problem_signature"] in train_sigs]
    test_trials = [t for t in all_trials if t["problem_signature"] in test_sigs]
    for t in train_trials:
        t.pop("problem_signature", None)
    for t in test_trials:
        t.pop("problem_signature", None)
    return train_trials, test_trials, option_keys


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


def _safe_prob_for_observed_action(
    choose_fn: Callable, trial: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Return (p_observed_action, loglik_term) for one trial.
    Raises on runtime/semantic errors; caller decides fallback.
    """
    y = int(trial["action"])
    p_raw = choose_fn(trial["problem"], trial["history"])
    p = float(p_raw)
    if not np.isfinite(p):
        raise ValueError("non-finite probability")
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    p_obs = p if y == 1 else (1.0 - p)
    ll = math.log(p_obs)
    return p_obs, ll


def _coerce_output_to_prob_b(raw_output: Any) -> float:
    """Coerce choose(...) output to clipped Bernoulli P(action=1)."""
    if isinstance(raw_output, bool):
        p_use = 1.0 if raw_output else 0.0
    elif isinstance(raw_output, (int, np.integer)) and int(raw_output) in (0, 1):
        p_use = float(int(raw_output))
    elif isinstance(raw_output, (float, np.floating)):
        p_use = float(raw_output)
    else:
        raise TypeError(f"choose must return float or 0/1, got {type(raw_output)}")
    if not np.isfinite(p_use):
        raise ValueError("non-finite probability")
    if not (0.0 <= p_use <= 1.0):
        raise ValueError(f"invalid probability: {p_use!r}")
    return float(min(max(p_use, 1e-9), 1.0 - 1e-9))


def _compute_confidence_penalty(choose_fn: Callable, trials: List[Dict[str, Any]]) -> float:
    """
    Mean confidence penalty on train trials:
      mean((p - 0.5)^2), where p is clipped P(action=1).
    """
    if not trials:
        return 0.0
    vals: List[float] = []
    for t in trials:
        p = _coerce_output_to_prob_b(choose_fn(t["problem"], t["history"]))
        vals.append((p - 0.5) ** 2)
    return float(np.mean(vals)) if vals else 0.0


def _compute_selection_score(
    train_loglik: float,
    *,
    rbu_lambda: float,
    residual_behavioral_uncertainty: float,
    confidence_penalty: float,
) -> float:
    """Regularized score: train_loglik - λ * (RBU^2) * confidence_penalty (pass BIR as rate when RBU is disabled)."""
    r = float(residual_behavioral_uncertainty)
    return float(train_loglik) - float(rbu_lambda) * (r * r) * float(confidence_penalty)


def _build_diagnostic_trials_text(
    parent_code: str,
    train_trials: List[Dict[str, Any]],
    dataset: str,
    num_diagnostic_trials: Optional[int],
) -> str:
    if not num_diagnostic_trials or num_diagnostic_trials <= 0 or not train_trials:
        return ""
    choose_fn = compile_program(parent_code)
    if choose_fn is None:
        return ""

    scored = []
    for t in train_trials:
        try:
            p_obs, ll = _safe_prob_for_observed_action(choose_fn, t)
            scored.append((ll, p_obs, t))
        except Exception:
            continue
    if not scored:
        return ""

    k_bad = max(1, num_diagnostic_trials // 2)
    k_good = max(1, num_diagnostic_trials - k_bad)
    scored_sorted = sorted(scored, key=lambda x: x[0])
    bad = scored_sorted[: min(k_bad, len(scored_sorted))]
    good = scored_sorted[-min(k_good, len(scored_sorted)) :]

    def _trial_block(name: str, items: List[Tuple[float, float, Dict[str, Any]]]) -> List[str]:
        lines = [f"### {name}"]
        for i, (ll, p_obs, t) in enumerate(items, start=1):
            prob = t["problem"]
            action = int(t["action"])
            lines.append(f"- trial_{i}: observed_action={action}, p_observed={p_obs:.6f}, log_p_observed={ll:.6f}")
            if dataset in {"choice13k", "mixed_gambles"}:
                lines.append(
                    f"  problem: gamble_A probs={prob['gamble_A']['probs']} rewards={prob['gamble_A']['rewards']}; "
                    f"gamble_B probs={prob['gamble_B']['probs']} rewards={prob['gamble_B']['rewards']}; "
                    f"has_feedback={prob.get('has_feedback', False)}"
                )
            else:
                lines.append(
                    "  problem: "
                    f"Ha={prob.get('Ha')}, pHa={prob.get('pHa')}, La={prob.get('La')}, "
                    f"Hb={prob.get('Hb')}, pHb={prob.get('pHb')}, Lb={prob.get('Lb')}, "
                    f"Amb={prob.get('Amb')}, Corr={prob.get('Corr')}"
                )
        return lines

    lines = [
        "\n## Participant diagnostic trials (selected by log-likelihood under current parent)",
        "Use these diagnostics only as hints; do not overfit to specific IDs.",
        "Note: The diagnostic trials below are provided only to illustrate problem structure and feature types. "
        "They are not behavioral labels or evidence favoring Option A or Option B. "
        "Do not infer participant preferences from the diagnostic trial set itself.",
        *_trial_block("Low log-likelihood (hard) trials", bad),
        *_trial_block("High log-likelihood (easy) trials", good),
    ]
    return "\n".join(lines) + "\n"


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


def _build_val_examples_prompt_block(val_trials: List[Dict[str, Any]], max_n: int) -> str:
    """Serialize up to ``max_n`` validation trials for the LLM prompt (data-driven evolution)."""
    if max_n <= 0 or not val_trials:
        return ""
    take = val_trials[:max_n]
    body = format_trials_to_text(take, dataset="choice13k")
    return (
        f"\n## Validation examples (at most {len(take)} trials; held out from training blocks)\n"
        f"{body}\n"
    )


def _te_data_driven_choice13k_prompt_path(max_prompt_val: int) -> str:
    """Evolution prompt template: val-trials variant when validation examples are injected."""
    base = _REPO_ROOT / "prompts" / "te_data_driven" / "evolution"
    if int(max_prompt_val) > 0:
        return str(base / "choices13k_val_trials.txt")
    return str(base / "choices13k.txt")


def _evolution_stage_cutoff(n_iterations: int) -> int:
    """Last 1-indexed iteration that uses stage-1 prompt (first half, rounded up)."""
    return (max(1, int(n_iterations)) + 1) // 2


def _default_evolution_stage_prompt_paths() -> Tuple[Path, Path]:
    base = _REPO_ROOT / "prompts" / "te_data_driven" / "evolution_stages"
    return base / "stage1.txt", base / "stage2.txt"


def _te_data_driven_evolution_stage_prompt_path(
    iteration_step: int,
    n_iterations: int,
    *,
    stage1_path: Optional[Path] = None,
    stage2_path: Optional[Path] = None,
    max_prompt_val: int = 0,
    fallback_path: Optional[str] = None,
) -> str:
    """Stage-1 for first half of iterations, stage-2 for the rest (data-driven evolution)."""
    s1 = Path(stage1_path) if stage1_path is not None else _default_evolution_stage_prompt_paths()[0]
    s2 = Path(stage2_path) if stage2_path is not None else _default_evolution_stage_prompt_paths()[1]
    cutoff = _evolution_stage_cutoff(n_iterations)
    chosen = s1 if int(iteration_step) <= cutoff else s2
    if chosen.is_file() and chosen.read_text(encoding="utf-8").strip():
        return str(chosen.resolve())
    fb = fallback_path or _te_data_driven_choice13k_prompt_path(max_prompt_val)
    print(
        f"[LLM prompt] Warning: stage prompt empty or missing ({chosen}); "
        f"falling back to {fb}"
    )
    return fb


def _pool_split_val_test(
    trials: List[Dict[str, Any]], split_seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split a pooled trial list into validation and test when only a single ``test`` pool exists
    (e.g. across-participant overrides). Uses a single random cut preserving trial order.
    """
    if len(trials) < 2:
        raise ValueError(
            "data_driven pooled test split requires at least 2 trials to form val and test; got "
            f"{len(trials)}."
        )
    rng = np.random.default_rng(int(split_seed))
    cut = int(rng.integers(1, len(trials)))
    return trials[:cut], trials[cut:]


def _trial_problem_identity_key(trial: Dict[str, Any]) -> str:
    """Stable key for grouping trials by underlying problem (Choice13k / mixed_gambles vs CPC18)."""
    prob = trial["problem"]
    if isinstance(prob, dict) and "gamble_A" in prob and "gamble_B" in prob:
        return make_problem_key(prob)
    return json.dumps(prob, sort_keys=True, default=str)


def make_problem_key(problem: Dict[str, Any]) -> str:
    """
    Deterministic identity key for a gamble problem.

    Includes:
    - gamble_A probs/rewards
    - gamble_B probs/rewards
    - option_keys
    - has_feedback

    Ignores:
    - history
    - action
    """
    payload = {
        "gamble_A": {
            "probs": problem["gamble_A"].get("probs"),
            "rewards": problem["gamble_A"].get("rewards"),
        },
        "gamble_B": {
            "probs": problem["gamble_B"].get("probs"),
            "rewards": problem["gamble_B"].get("rewards"),
        },
        "option_keys": problem.get("option_keys"),
        "has_feedback": bool(problem.get("has_feedback", False)),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def compute_behavioral_inconsistency_rate(train_trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute participant-level Behavioral Inconsistency Rate (BIR) on train trials.

    A problem group is inconsistent when both actions (0 and 1) appear among
    repeated trials of the same underlying gamble problem.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for trial in train_trials:
        key = _trial_problem_identity_key(trial)
        grouped.setdefault(key, []).append(trial)

    num_groups = len(grouped)
    inconsistent = 0
    for group in grouped.values():
        actions = {int(t["action"]) for t in group}
        if 0 in actions and 1 in actions:
            inconsistent += 1

    bir = (float(inconsistent) / float(num_groups)) if num_groups > 0 else 0.0
    return {
        "num_train_trials": int(len(train_trials)),
        "num_problem_groups": int(num_groups),
        "num_inconsistent_problem_groups": int(inconsistent),
        "behavioral_inconsistency_rate": float(bir),
    }


_BEHAVIORAL_INCONSISTENCY_CSV_FIELDS = [
    "participant_ordinal",
    "num_train_trials",
    "num_train_problem_groups",
    "num_inconsistent_problem_groups",
    "BIR",
    "behavioral_inconsistency_rate",
    "rbu",
    "structure_score",
]


def _build_behavioral_inconsistency_rows(
    participant_trials: Dict[int, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[int, float]]:
    """One row dict per participant with BIR metrics; ``rbu`` / ``structure_score`` filled later."""
    rows: List[Dict[str, Any]] = []
    bir_by_participant: Dict[int, float] = {}
    for participant_ordinal in sorted(participant_trials.keys()):
        metrics = compute_behavioral_inconsistency_rate(participant_trials[participant_ordinal])
        bir = float(metrics["behavioral_inconsistency_rate"])
        bir_by_participant[int(participant_ordinal)] = bir
        rows.append(
            {
                "participant_ordinal": int(participant_ordinal),
                "num_train_trials": int(metrics["num_train_trials"]),
                "num_train_problem_groups": int(metrics["num_problem_groups"]),
                "num_inconsistent_problem_groups": int(metrics["num_inconsistent_problem_groups"]),
                "BIR": bir,
                "behavioral_inconsistency_rate": bir,
                "rbu": bir,
                "structure_score": "",
            }
        )
    return rows, bir_by_participant


def _round_behavioral_inconsistency_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Round every numeric value to 2 decimals for ``behavioral_inconsistency_rate.csv``."""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if v == "" or v is None:
            out[k] = v
        elif isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, np.integer)):
            out[k] = round(float(v), 2)
        elif isinstance(v, (float, np.floating)):
            x = float(v)
            out[k] = round(x, 2) if math.isfinite(x) else x
        else:
            out[k] = v
    return out


def _round_behavioral_inconsistency_csv_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_round_behavioral_inconsistency_csv_row(r) for r in rows]


def _write_behavioral_inconsistency_csv(base_run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    """Write ``analysis/behavioral_inconsistency_rate.csv`` with one row per participant."""
    analysis_dir = base_run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_csv = analysis_dir / "behavioral_inconsistency_rate.csv"
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        out_rows.append({k: row[k] for k in _BEHAVIORAL_INCONSISTENCY_CSV_FIELDS})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_BEHAVIORAL_INCONSISTENCY_CSV_FIELDS))
        w.writeheader()
        w.writerows(_round_behavioral_inconsistency_csv_rows(out_rows))

    print("\n[Behavioral Inconsistency Analysis]")
    if not rows:
        print("No participant train trials available for BIR computation.")
        print(f"Saved: {out_csv}")
        return

    bir_values = [float(r["behavioral_inconsistency_rate"]) for r in rows]
    for row in rows:
        print(
            "participant {pid}: BIR={bir:.2f} ({inc}/{tot} inconsistent groups)".format(
                pid=int(row["participant_ordinal"]),
                bir=float(row["behavioral_inconsistency_rate"]),
                inc=int(row["num_inconsistent_problem_groups"]),
                tot=int(row["num_train_problem_groups"]),
            )
        )
    print(
        "BIR summary: mean={mean:.2f}, min={minv:.2f}, max={maxv:.2f}".format(
            mean=float(np.mean(bir_values)),
            minv=float(np.min(bir_values)),
            maxv=float(np.max(bir_values)),
        )
    )
    print(f"Saved: {out_csv}")


@dataclass
class TeAggregateScoringState:
    """Pre-evolution outputs for te_aggregate within-participant mode (BIR / RBU / structure)."""

    bir_report_rows: List[Dict[str, Any]]
    participant_bir_map: Dict[int, float]
    participant_rbu_map: Dict[int, float]
    participant_structure_score_map: Dict[int, float]
    participant_structure_components: Dict[int, Dict[str, float]]
    participant_profiles: Dict[int, str]


def _te_aggregate_run_profile_warmup_if_enabled(
    *,
    args: Any,
    participants_to_process: List[int],
    participant_trials: Dict[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    profile_client: OpenAI,
    base_run_dir: str,
) -> Dict[int, str]:
    """Phase-1 text profile per participant when ``--profile_warmup`` is True; else empty dict."""
    participant_profiles: Dict[int, str] = {}
    if not args.profile_warmup:
        print("[Phase1] skipped (--profile_warmup False); phase 2 uses seed program only.")
        return participant_profiles
    _text_profile_prompt_path(args.dataset)
    print(
        f"[Phase1] text-profile warmup for {len(participants_to_process)} participant(s); "
        f"prompt: prompts/Template_evo/{args.dataset}/text_profile/text_profile.txt"
    )
    for participant_id in tqdm(participants_to_process, desc="Participants (phase1 profile)"):
        tr, _ = participant_trials[participant_id]
        pinfo = run_text_profile_warmup_participant(
            dataset=args.dataset,
            participant_id=participant_id,
            train_trials=tr,
            client=profile_client,
            model_name=args.model_name,
            base_run_dir=base_run_dir,
        )
        participant_profiles[participant_id] = str(pinfo.get("profile_text") or "")
    return participant_profiles


def _merge_structure_parse_into_bir_rows(
    bir_report_rows: List[Dict[str, Any]],
    parsed: Optional[Dict[int, Tuple[float, Dict[str, float]]]],
    *,
    use_rbu: bool,
    structure_weight: float,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, Dict[str, float]]]:
    """
    Fill ``rbu`` / ``structure_score`` on each BIR row from parsed structure scores (when ``use_rbu``),
    or set ``rbu`` to BIR when not using RBU.
    """
    participant_rbu_map: Dict[int, float] = {}
    participant_structure_score_map: Dict[int, float] = {}
    participant_structure_components: Dict[int, Dict[str, float]] = {}
    if use_rbu and parsed is not None:
        for r in bir_report_rows:
            pid = int(r["participant_ordinal"])
            bir_val = float(r["behavioral_inconsistency_rate"])
            s_v, comps = parsed[pid]
            rbu_v = compute_rbu(bir_val, float(s_v), structure_weight=float(structure_weight))
            r["structure_score"] = float(s_v)
            r["rbu"] = float(rbu_v)
            participant_rbu_map[pid] = float(rbu_v)
            participant_structure_score_map[pid] = float(s_v)
            participant_structure_components[pid] = comps
    else:
        for r in bir_report_rows:
            pid = int(r["participant_ordinal"])
            participant_rbu_map[pid] = float(r["behavioral_inconsistency_rate"])
    return participant_rbu_map, participant_structure_score_map, participant_structure_components


def _te_aggregate_run_scoring_stage(
    *,
    args: Any,
    base_run_dir: str,
    participants_to_process: List[int],
    bir_train_only: Dict[int, List[Dict[str, Any]]],
    participant_trials: Dict[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    profile_client: OpenAI,
    rbu_prepare_path: Path,
    rbu_use_path: Path,
) -> TeAggregateScoringState:
    """
    BIR report, optional RBU instruction + structure-score LLM, profile warmup, write
    ``analysis/behavioral_inconsistency_rate.csv``.
    """
    bir_report_rows, participant_bir_map = _build_behavioral_inconsistency_rows(bir_train_only)

    if args.use_rbu:
        rbu_llm_write_run_instruction(
            client=profile_client,
            model_name=args.model_name,
            prepare_instruction_path=rbu_prepare_path,
            run_dir=Path(base_run_dir),
        )
    else:
        print("[RBU] skipped (--use_rbu False); phase 2 uses BIR as the regularization rate.")

    participant_profiles = _te_aggregate_run_profile_warmup_if_enabled(
        args=args,
        participants_to_process=participants_to_process,
        participant_trials=participant_trials,
        profile_client=profile_client,
        base_run_dir=base_run_dir,
    )

    run_instruction_file = Path(base_run_dir) / "instruction.txt"
    analysis_dir = Path(base_run_dir) / "analysis"
    if args.use_rbu:
        print(
            "[RBU] structure_score is computed as mean(clipped evidence values in participant JSON 'evidence')."
        )
        n_part = len(participants_to_process)
        max_resp_toks = min(32768, max(4096, 768 * max(1, n_part)))
        raw_scores, _, _ = rbu_llm_write_all_participant_structure_scores(
            client=profile_client,
            model_name=args.model_name,
            use_instruction_path=rbu_use_path,
            run_instruction_path=run_instruction_file,
            participant_ids=list(participants_to_process),
            participant_train_trials=bir_train_only,
            dataset=args.dataset,
            analysis_dir=analysis_dir,
            structure_prompt_max_tokens=int(args.structure_prompt_max_tokens),
            max_response_tokens=max_resp_toks,
            model_context_tokens=int(args.structure_model_context_tokens),
        )
        score_all_path = analysis_dir / _RBU_STRUCTURE_SCORE_ALL_FILENAME
        try:
            parsed = parse_all_participant_structure_scores(
                raw_scores,
                expected_participant_ids=tuple(sorted(int(x) for x in participants_to_process)),
            )
        except StructureScoreParseError as exc:
            raise RuntimeError(
                f"RBU combined structure score parse failed path={score_all_path}: {exc}\n"
                f"raw_output (first 2400 chars):\n{raw_scores[:2400]!r}"
            ) from exc
        participant_rbu_map, participant_structure_score_map, participant_structure_components = (
            _merge_structure_parse_into_bir_rows(
                bir_report_rows,
                parsed,
                use_rbu=True,
                structure_weight=float(args.structure_weight),
            )
        )
    else:
        participant_rbu_map, participant_structure_score_map, participant_structure_components = (
            _merge_structure_parse_into_bir_rows(
                bir_report_rows,
                None,
                use_rbu=False,
                structure_weight=float(args.structure_weight),
            )
        )

    _write_behavioral_inconsistency_csv(Path(base_run_dir), bir_report_rows)

    return TeAggregateScoringState(
        bir_report_rows=bir_report_rows,
        participant_bir_map=participant_bir_map,
        participant_rbu_map=participant_rbu_map,
        participant_structure_score_map=participant_structure_score_map,
        participant_structure_components=participant_structure_components,
        participant_profiles=participant_profiles,
    )


def _resolve_scoring_input_path(path: str) -> Path:
    """Resolve ``--structure_path`` (often repo-relative) to an existing file."""
    raw = Path(path).expanduser()
    if raw.is_file():
        return raw.resolve()
    cand = (_REPO_ROOT / raw).resolve()
    if cand.is_file():
        return cand
    cwd_try = (Path.cwd() / raw).resolve()
    if cwd_try.is_file():
        return cwd_try
    raise FileNotFoundError(
        f"--structure_path not found: {path!r} (tried absolute path, cwd-relative, and repo-root-relative)."
    )


def _log_choice13k_evolution_loaded_rates(
    scoring: TeAggregateScoringState,
    participant_ids: List[int],
    *,
    use_rbu: bool,
) -> None:
    """Log BIR (recomputed on this run's train split), structure_score (from file if RBU), and RBU."""
    print(
        "\n[phase_option=evolution] BIR recomputed from current train data; "
        "RBU = clip(BIR - structure_weight * S) using this run's --structure_weight."
    )
    if use_rbu:
        print("  structure_score S parsed from --structure_path (prior combined structure-score output).")
    else:
        print("  (--use_rbu False: regularization rate is BIR; --structure_path ignored.)")
    for pid in sorted(int(x) for x in participant_ids):
        bir = float(scoring.participant_bir_map.get(pid, 0.0))
        rbu = float(scoring.participant_rbu_map.get(pid, bir))
        if use_rbu:
            s_v = scoring.participant_structure_score_map.get(pid)
            s_str = f"{float(s_v):.4f}" if s_v is not None else "n/a"
            print(f"  participant {pid}: BIR={bir:.4f}, S={s_str}, RBU={rbu:.4f}")
        else:
            print(f"  participant {pid}: BIR={bir:.4f}, RBU={rbu:.4f}")


def _te_aggregate_run_evolution_stage(
    *,
    args: Any,
    base_run_dir: str,
    seed_program_path: Optional[str],
    participants_to_process: List[int],
    participant_trials: Dict[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    scoring: TeAggregateScoringState,
    wandb: Any,
    client_kwargs: Optional[Dict[str, Any]],
    mixed_gambles_gain_loss_only: bool,
    _wandb_log_with_global_step: Any,
) -> None:
    """Phase-2 per-participant ``run_evolution`` and aggregate CSV summaries."""
    participant_bir_map = scoring.participant_bir_map
    participant_rbu_map = scoring.participant_rbu_map
    participant_structure_score_map = scoring.participant_structure_score_map
    participant_structure_components = scoring.participant_structure_components
    participant_profiles = scoring.participant_profiles

    participants_summary: List[Dict[str, Any]] = []
    participants_loglik_summary: List[Dict[str, Any]] = []
    details_loglik_file = Path(base_run_dir) / "participant_details_loglik.csv"
    summary_loglik_file = Path(base_run_dir) / "summary_loglik.csv"
    summary_file = Path(base_run_dir) / "summary.csv"
    for participant_id in tqdm(participants_to_process, desc="Participants (phase2 adapt)"):
        participant_output_dir = os.path.join(base_run_dir, f"participant_{participant_id}")
        profile_str = participant_profiles.get(participant_id) if args.profile_warmup else None
        bir_val = float(participant_bir_map.get(participant_id, 0.0))
        rbu_val = float(participant_rbu_map.get(participant_id, bir_val))
        struct_score: Optional[float] = None
        struct_comps: Optional[Dict[str, float]] = None
        if args.use_rbu:
            struct_score = participant_structure_score_map.get(participant_id)
            _comps = participant_structure_components.get(participant_id)
            struct_comps = _comps if _comps else None
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
            max_prompt_train=args.max_prompt_train,
            max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
            llm_max_tokens=args.llm_max_tokens,
            cpc18_official_mse=False,
            adaptation_mode=True,
            aggregate_base_code=None,
            participant_text_profile=profile_str,
            num_diagnostic_trials=getattr(args, "num_diagnostic_trials", None),
            max_prompt_val=int(getattr(args, "max_prompt_val", 10)),
            lambda_complexity=args.lambda_complexity,
            lambda_change=args.lambda_change,
            hard_participant_train_loglik_threshold=args.hard_participant_train_loglik_threshold,
            hard_participant_warmup_iters=args.hard_participant_warmup_iters,
            early_stop=args.early_stop,
            debug_continue_after_early_stop=args.debug_continue_after_early_stop,
            wandb_log_fn=_wandb_log_with_global_step,
            local_dataset=args.local_dataset,
            rbu_lambda=args.uncertainty_lambda,
            use_rbu=bool(args.use_rbu),
            participant_bir=bir_val,
            participant_rbu=float(rbu_val),
            rbu_prompt_threshold=args.uncertainty_threshold,
            structure_score=struct_score,
            structure_components=struct_comps,
        )
        best_src = Path(participant_output_dir) / "best_program.py"
        if best_src.exists():
            best_dst = Path(participant_output_dir) / f"best_adapted_program_participant_{participant_id}.py"
            best_dst.write_text(best_src.read_text(encoding="utf-8"), encoding="utf-8")
        participants_summary.append(participant_summary)
        participants_loglik_summary.append(
            {
                "participant_id": participant_summary.get("participant_id"),
                "train_loglik": participant_summary.get("train_loglik"),
                "test_loglik": participant_summary.get("test_loglik"),
                "selection_score": participant_summary.get("selection_score"),
            }
        )

        with open(details_loglik_file, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["participant_id", "train_loglik", "test_loglik", "selection_score"]
            )
            w.writeheader()
            w.writerows(_round_floats_for_csv_rows(participants_loglik_summary))
        train_ll_vals = [d["train_loglik"] for d in participants_loglik_summary if d["train_loglik"] is not None]
        test_ll_vals = [d["test_loglik"] for d in participants_loglik_summary if d["test_loglik"] is not None]
        with open(summary_loglik_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["num_of_participants", "avg_train_loglik", "avg_test_loglik"])
            w.writeheader()
            w.writerow(
                _round_floats_for_csv_row(
                    {
                        "num_of_participants": len(participants_loglik_summary),
                        "avg_train_loglik": float(np.mean(train_ll_vals)) if train_ll_vals else None,
                        "avg_test_loglik": float(np.mean(test_ll_vals)) if test_ll_vals else None,
                    }
                )
            )
        with open(summary_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(participant_summary.keys()))
            w.writeheader()
            w.writerows(_round_floats_for_csv_rows(participants_summary))


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
                if verbose and errors <= 3 and seed == 0:  # Only print first 3 errors from first seed
                    print(f"  Evaluation error: {e}")
        acc = correct / total if total > 0 else 0.0
        accuracies.append(acc)
    
    # Average across seeds
    avg_acc = np.mean(accuracies) if accuracies else 0.0
    # Use first seed's error count for reporting
    correct = int(avg_acc * total)
    errors = total - correct if n_seeds == 1 else 0  # Error count only meaningful for single seed
    
    result = {"accuracy": avg_acc, "total": total, "correct": correct, "errors": errors}
    if verbose and errors > 0 and n_seeds == 1:
        print(f"  Total evaluation errors: {errors}/{total}")
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
    seed_errors: List[int] = []

    def _one_pass(seed_idx: int) -> Tuple[float, float, int]:
        loglik_acc = 0.0
        correct = 0
        errors = 0
        for t in trials:
            y = int(t["action"])
            try:
                p_raw = choose_fn(t["problem"], t["history"])
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
            except Exception as e:
                errors += 1
                if verbose and errors <= 3 and seed_idx == 0:
                    print(f"  Evaluation error: {e}")
                continue

        avg_ll = (loglik_acc / total if total > 0 else 0.0) if errors == 0 else float("-inf")
        acc = correct / total if total > 0 else 0.0
        return avg_ll, acc, errors

    for seed in range(n_seeds):
        avg_ll, acc, errs = _one_pass(seed)
        seed_avg_logliks.append(avg_ll)
        seed_avg_accs.append(acc)
        seed_errors.append(errs)

    avg_acc = float(np.mean(seed_avg_accs)) if seed_avg_accs else 0.0
    avg_loglik = float(np.mean(seed_avg_logliks)) if seed_avg_logliks else float("-inf")
    correct = int(round(avg_acc * total))
    total_errors = int(max(seed_errors)) if seed_errors else 0
    return {
        "accuracy": avg_acc,
        "avg_loglik": avg_loglik,
        "total": total,
        "correct": correct,
        "errors": total_errors,
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
    seed_errors: List[int] = []

    def _one_pass(seed_idx: int) -> Tuple[float, float, int]:
        loglik_acc = 0.0
        correct = 0
        errors = 0
        for t in trials:
            y = int(t["action"])
            try:
                p_raw = choose_fn(t["problem"], t["history"])
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
            except Exception as e:
                errors += 1
                if verbose and errors <= 3 and seed_idx == 0:
                    print(f"  Evaluation error: {e}")
                continue

        avg_ll = (loglik_acc / total if total > 0 else 0.0) if errors == 0 else float("-inf")
        acc = correct / total if total > 0 else 0.0
        return avg_ll, acc, errors

    for seed in range(n_seeds):
        avg_ll, acc, errs = _one_pass(seed)
        seed_avg_logliks.append(avg_ll)
        seed_avg_accs.append(acc)
        seed_errors.append(errs)

    avg_acc = float(np.mean(seed_avg_accs)) if seed_avg_accs else 0.0
    avg_loglik = float(np.mean(seed_avg_logliks)) if seed_avg_logliks else float("-inf")
    correct = int(round(avg_acc * total))
    total_errors = int(max(seed_errors)) if seed_errors else 0
    return {
        "accuracy": avg_acc,
        "avg_loglik": avg_loglik,
        "total": total,
        "correct": correct,
        "errors": total_errors,
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
    candidates = []
    for _ in range(n_candidates):
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
            if code and ('class FSMAgent' in code or 'def act' in code):
                candidates.append(code)
            else:
                candidates.append(template_code)
        except Exception:
            candidates.append(template_code)
    return candidates


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
    variants = []
    for _ in range(n_variants):
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
            if code and ('class FSMAgent' in code or 'def act' in code):
                variants.append(code)
            else:
                variants.append(parent_codes[0])
        except Exception:
            variants.append(parent_codes[0])
    return variants


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
            client, model_name, seed_code, prefix_text, n_candidates=K,
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

    # Generate variants one at a time to avoid huge prompts (especially with multiple parents)
    variants = []
    best_parent = parent_codes[0] if parent_codes else ""
    
    for _ in tqdm(range(n_variants), desc="Generating gridworld variants"):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": base_prompt_template_final}],
                temperature=0.7,
                top_p=0.95,
                max_tokens=max_tokens,  # Per variant - keep original max_tokens
            )
            content = response.choices[0].message.content
            
            code = _sanitize_llm_python_candidate(
                content, required_markers=("class FSMAgent", "def act(")
            )
            if code and ('class FSMAgent' in code or 'def act' in code):
                variants.append(code)
            else:
                variants.append(best_parent)
        except Exception as e:
            print(f"Warning: Failed to generate program variant: {e}")
            # Fallback: use best parent code
            variants.append(best_parent)
    
    return variants[:n_variants]


def generate_program_variants(
    client: OpenAI,
    model_name: str,
    parent_programs: List[str],
    train_trials: List[Dict[str, Any]],
    n_variants: int = 10,
    max_tokens: int = 800,
    parent_train_accuracies: Optional[List[float]] = None,
    parent_train_mses: Optional[List[float]] = None,
    dataset: str = "choice13k",
    max_prompt_train: int = 10,
    max_prompt_trials_per_problem: int = 0,
    prompt_train_trials_seed: int = 0,
    fitness_metric: str = "accuracy",
    cpc18_official_mse: bool = True,
    include_train_trials_in_prompt: bool = True,
    base_program_code: Optional[str] = None,
    diagnostic_trials_text: str = "",
    extra_prompt_instructions: str = "",
    parent_metric_label_override: Optional[str] = None,
    parent_runtime_errors: Optional[List[Optional[str]]] = None,
    val_trials_for_prompt: Optional[List[Dict[str, Any]]] = None,
    max_prompt_val: int = 0,
    parent_val_logliks: Optional[List[Optional[float]]] = None,
    choice13k_loglik_prompt_path: Optional[str] = None,
    show_train_omitted_message: bool = False,
) -> List[str]:
    """
    Generate full program variants based on parent program and training trials.
    
    This generates complete choose(problem, history) implementations without
    restrictions on structure or logic - only the function signature is fixed.
    """
    # Load prompts from file
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    if (
        dataset == "cpc18"
        and not cpc18_official_mse
        and fitness_metric == "loglik"
    ):
        prompt_path = os.path.join(
            PROJECT_ROOT, "prompts", "Template_evo", "cpc18", "non_strict", "loglik", "infer_single_choice.txt"
        )
        code_template_path = os.path.join(
            PROJECT_ROOT, "prompts", "Template_evo", "cpc18", "non_strict", "loglik", "single_code_template.txt"
        )
    elif dataset == "cpc18":
        prompt_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "cpc18", "non_strict", "infer_single_choice.txt")
        code_template_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "cpc18", "non_strict", "single_code_template.txt")
    elif dataset == "mixed_gambles":
        if fitness_metric == "loglik":
            prompt_path = os.path.join(
                PROJECT_ROOT, "prompts", "Template_evo", "mixed_gambles", "non_strict", "loglik", "infer_single_choice.txt"
            )
            code_template_path = os.path.join(
                PROJECT_ROOT, "prompts", "Template_evo", "mixed_gambles", "non_strict", "loglik", "single_code_template.txt"
            )
        else:
            prompt_path = os.path.join(
                PROJECT_ROOT, "prompts", "Template_evo", "mixed_gambles", "non_strict", "infer_single_choice.txt"
            )
            code_template_path = os.path.join(
                PROJECT_ROOT, "prompts", "Template_evo", "mixed_gambles", "non_strict", "single_code_template.txt"
            )
    else:
        if fitness_metric == "loglik":
            if choice13k_loglik_prompt_path:
                prompt_path = choice13k_loglik_prompt_path
            else:
                prompt_path = os.path.join(
                    PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "non_strict", "loglik", "infer_single_choice.txt"
                )
            code_template_path = os.path.join(
                PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "non_strict", "loglik", "single_code_template.txt"
            )
        else:
            prompt_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "non_strict", "infer_single_choice.txt")
            code_template_path = os.path.join(PROJECT_ROOT, "prompts", "Template_evo", "choice13k", "non_strict", "single_code_template.txt")
    
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
    
    # Format training trials for context (evaluation elsewhere still uses full train_trials).
    state_text = ""
    if include_train_trials_in_prompt:
        if max_prompt_train <= 0:
            trials_for_prompt: List[Dict[str, Any]] = []
            print("[LLM prompt] Omitting train trials (max_prompt_train=0).")
        else:
            trials_for_prompt = list(train_trials)
            trials_for_prompt = _cap_prompt_trials_per_problem(
                trials_for_prompt, max_prompt_trials_per_problem
            )
            if len(trials_for_prompt) > max_prompt_train:
                rng = np.random.default_rng(prompt_train_trials_seed)
                perm = rng.permutation(len(trials_for_prompt))
                sel = perm[:max_prompt_train]
                trials_for_prompt = [trials_for_prompt[i] for i in sel]
                print(
                    f"[LLM prompt] Using {len(trials_for_prompt)} of {len(train_trials)} train trials "
                    f"(max_prompt_train={max_prompt_train}, seed={prompt_train_trials_seed})."
                )
        if trials_for_prompt and "problem" in trials_for_prompt[0]:
            if "gamble_A" in trials_for_prompt[0]["problem"]:
                dataset_type = "choice13k"
            else:
                dataset_type = "cpc18"
        else:
            dataset_type = "choice13k"
        state_text = (
            format_trials_to_text(trials_for_prompt, dataset=dataset_type) if trials_for_prompt else ""
        )
    elif show_train_omitted_message:
        state_text = "Train-trial history is intentionally omitted for this phase.\n"
    else:
        state_text = ""
    
    # Include parent programs as reference
    num_parents = len(parent_programs)
    if num_parents == 1:
        parent_context = f"\n\nReference program (parent):\n```python\n{parent_programs[0]}\n```\n\n"
        pm0 = parent_train_accuracies[0] if parent_train_accuracies and len(parent_train_accuracies) >= 1 else None
        if (
            pm0 is not None
            and fitness_metric == "loglik"
            and dataset == "choice13k"
            and parent_val_logliks
            and len(parent_val_logliks) >= 1
            and parent_val_logliks[0] is not None
        ):
            parent_context += (
                f"Parent scores: train_loglik={pm0:.6f}, val_loglik={float(parent_val_logliks[0]):.6f}\n\n"
            )
        parent_context += "Generate a variant that improves upon or explores alternatives to the parent program.\n"
    else:
        parent_context = f"\n\nReference parent programs ({num_parents} elite programs):\n"
        for i, parent_program in enumerate(parent_programs):
            if dataset == "cpc18" and cpc18_official_mse:
                mse = parent_train_mses[i] if (parent_train_mses and i < len(parent_train_mses)) else None
                mse_str = f" (train_block-MSE: {mse:.2f})" if mse is not None else ""
                err_str = ""
                if parent_runtime_errors and i < len(parent_runtime_errors) and parent_runtime_errors[i]:
                    err_str = f"\nRuntime error previously observed: {parent_runtime_errors[i]}"
                parent_context += f"\nParent {i+1}{mse_str}{err_str}:\n```python\n{parent_program}\n```\n"
            else:
                parent_metric = (
                    parent_train_accuracies[i]
                    if parent_train_accuracies and i < len(parent_train_accuracies)
                    else None
                )
                if parent_metric is not None:
                    is_loglik_prompt_metric = (
                        fitness_metric == "loglik"
                        and (dataset in {"choice13k", "mixed_gambles"} or (dataset == "cpc18" and not cpc18_official_mse))
                    )
                    metric_label = parent_metric_label_override or (
                        "train_loglik"
                        if is_loglik_prompt_metric
                        else "train_acc"
                    )
                    if (
                        is_loglik_prompt_metric
                        and dataset == "choice13k"
                        and parent_val_logliks is not None
                        and i < len(parent_val_logliks)
                        and parent_val_logliks[i] is not None
                    ):
                        metric_str = (
                            f" (train_loglik: {parent_metric:.4f}, val_loglik: {float(parent_val_logliks[i]):.4f})"
                        )
                    else:
                        metric_str = f" ({metric_label}: {parent_metric:.4f})"
                else:
                    metric_str = ""
                err_str = ""
                if parent_runtime_errors and i < len(parent_runtime_errors) and parent_runtime_errors[i]:
                    err_str = f"\nRuntime error previously observed: {parent_runtime_errors[i]}"
                parent_context += f"\nParent {i+1}{metric_str}{err_str}:\n```python\n{parent_program}\n```\n"
        
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
    
    base_context = ""
    if base_program_code:
        base_context = (
            "\n\nReference aggregate/base program:\n"
            f"```python\n{base_program_code}\n```\n"
        )
    val_examples_block = _build_val_examples_prompt_block(val_trials_for_prompt or [], max_prompt_val)
    prompt_text = (
        f"{base_prompt}\n"
        f"{extra_prompt_instructions}\n"
        f"{state_text}\n"
        f"{val_examples_block}\n"
        f"{diagnostic_trials_text}\n"
        f"{base_context}\n"
        f"{parent_context}\n"
        f"{code_template}"
    )
    
    programs = []
    for _ in tqdm(range(n_variants), desc="Generating candidate programs"):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                top_p=0.95,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            code = _sanitize_llm_python_candidate(content, required_markers=("def choose(",))
            programs.append(code)
        except Exception as e:
            print(f"Warning: Failed to generate program variant: {e}")
            programs.append("")
    return programs


def _evaluate_loglik_for_dataset(
    dataset: str,
    choose_fn: Callable,
    trials: List[Dict[str, Any]],
    *,
    n_eval_seeds: int,
) -> Dict[str, float]:
    if dataset == "cpc18":
        return evaluate_cpc18_split_program(choose_fn, trials, n_seeds=n_eval_seeds)
    return evaluate_choice13k_program(choose_fn, trials, n_seeds=n_eval_seeds)


def _complexity_penalty(code: str) -> float:
    lines = [ln for ln in code.splitlines() if ln.strip()]
    return float(len(lines))


def _change_penalty(code: str, base_code: str) -> float:
    a = [ln.rstrip() for ln in base_code.splitlines()]
    b = [ln.rstrip() for ln in code.splitlines()]
    sm = difflib.SequenceMatcher(a=a, b=b)
    return float(1.0 - sm.ratio())


def _json_safe_obj(obj: Any, neg_inf_fallback: float = -1e9, pos_inf_fallback: float = 1e9) -> Any:
    """Recursively convert NaN/Inf values to finite sentinels for JSON output."""
    if isinstance(obj, dict):
        return {k: _json_safe_obj(v, neg_inf_fallback, pos_inf_fallback) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe_obj(v, neg_inf_fallback, pos_inf_fallback) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe_obj(v, neg_inf_fallback, pos_inf_fallback) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if np.isnan(v) or np.isneginf(v):
            return neg_inf_fallback
        if np.isposinf(v):
            return pos_inf_fallback
        return v
    return obj


def _json_dumps_safe(obj: Any, **kwargs) -> str:
    return json.dumps(_json_safe_obj(obj), **kwargs)


def _one_line_runtime_error(exc: Exception) -> str:
    msg = str(exc).strip()
    if msg:
        return msg.splitlines()[-1][:240]
    return exc.__class__.__name__


def _probe_runtime_error_line(
    choose_fn: Callable,
    trials: List[Dict[str, Any]],
    max_trials: int = 256,
) -> str:
    for t in trials[:max_trials]:
        try:
            p_raw = choose_fn(t["problem"], t["history"])
            if isinstance(p_raw, bool) or (
                isinstance(p_raw, (int, np.integer)) and int(p_raw) in (0, 1)
            ):
                p_use = 1.0 if int(p_raw) == 1 else 0.0
            elif isinstance(p_raw, float):
                p_use = p_raw
            else:
                raise TypeError(f"choose must return float or 0/1, got {type(p_raw)}")
            if not (0.0 <= p_use <= 1.0):
                raise ValueError(f"invalid probability: {p_use!r}")
        except Exception as exc:
            return _one_line_runtime_error(exc)
    return "runtime error (not captured)"


def _approx_prompt_token_count(text: str) -> int:
    """Lightweight token budget estimate (~4 chars/token for English-ish text)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


_PROFILE_TRIALS_HEADER = "\n\n## Training trials (chronological)\n"
# Text-profile phase: CPC18 trial lines are long; cap serialized trials (~approx tokens) separately from choice13k.
_TEXT_PROFILE_CPC18_MAX_TRIAL_BLOCK_TOKENS = 6000


def _text_profile_prompt_path(dataset: str) -> Path:
    path = _REPO_ROOT / "prompts" / "Template_evo" / dataset / "text_profile" / "text_profile.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Text-profile warmup prompt missing (required when --profile_warmup): {path}"
        )
    return path


def _format_trials_for_text_profile(train_trials: List[Dict[str, Any]], dataset: str) -> str:
    """Serialize train trials (+ history when present) for the text-profile LLM prompt."""
    lines: List[str] = []
    fmt_ds = "cpc18" if dataset == "cpc18" else "choice13k"
    for idx, t in enumerate(train_trials):
        hist = json.dumps(t.get("history", []), default=str)
        if fmt_ds == "cpc18":
            prob = t["problem"]
            action = t["action"]
            lines.append(
                f"{idx + 1}. Problem: Option A (Ha={prob['Ha']}, pHa={prob['pHa']}, La={prob['La']}, "
                f"LotShapeA={prob['LotShapeA']}, LotNumA={prob['LotNumA']}); "
                f"Option B (Hb={prob['Hb']}, pHb={prob['pHb']}, Lb={prob['Lb']}, "
                f"LotShapeB={prob['LotShapeB']}, LotNumB={prob['LotNumB']}); "
                f"Amb={prob['Amb']}, Corr={prob['Corr']}; Observed action: {action}; prior_history={hist}"
            )
        else:
            prob_a = t["problem"]["gamble_A"]["probs"]
            rew_a = t["problem"]["gamble_A"]["rewards"]
            prob_b = t["problem"]["gamble_B"]["probs"]
            rew_b = t["problem"]["gamble_B"]["rewards"]
            has_fb = t["problem"].get("has_feedback", False)
            action = t["action"]
            lines.append(
                f"{idx + 1}. Problem: Option A probs {prob_a} rewards {rew_a}; "
                f"Option B probs {prob_b} rewards {rew_b}; has_feedback={has_fb}; "
                f"Observed action: {action}; prior_history={hist}"
            )
    return "\n".join(lines)


def _rbu_default_prepare_instruction_path(dataset: str) -> Path:
    p = _REPO_ROOT / "prompts" / "Template_evo" / dataset / "text_profile" / "prepare_instruction.txt"
    if not p.is_file():
        raise FileNotFoundError(f"RBU prepare_instruction.txt missing for dataset={dataset!r}: {p}")
    return p


def _rbu_default_use_instruction_path(dataset: str) -> Path:
    p = _REPO_ROOT / "prompts" / "Template_evo" / dataset / "text_profile" / "use_instruction.txt"
    if not p.is_file():
        raise FileNotFoundError(f"RBU use_instruction.txt missing for dataset={dataset!r}: {p}")
    return p


def rbu_llm_write_run_instruction(
    *,
    client: OpenAI,
    model_name: str,
    prepare_instruction_path: Path,
    run_dir: Path,
    max_tokens: int = 4096,
) -> None:
    """One LLM call at run start; writes ``run_dir / instruction.txt`` (always fresh for this run)."""
    prompt = prepare_instruction_path.read_text(encoding="utf-8")
    out_path = run_dir / "instruction.txt"
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError(
            f"RBU dataset instruction LLM returned empty content; expected scoring instruction text in {out_path}"
        )
    out_path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")


def _rbu_build_all_participant_trials_block(
    participant_ids: List[int],
    train_by_pid: Dict[int, List[Dict[str, Any]]],
    trials_per_participant: int,
    dataset: str,
) -> str:
    """Serialize the first ``trials_per_participant`` training trials for each participant (same k for all)."""
    parts: List[str] = []
    for pid in sorted(int(x) for x in participant_ids):
        tr = train_by_pid.get(pid, [])
        trials = tr[:trials_per_participant] if trials_per_participant > 0 else []
        parts.append(
            f"\n\n## Participant {pid} — training trials (training split only; no test trials)\n"
        )
        parts.append(_format_trials_for_text_profile(trials, dataset))
    return "".join(parts)


def rbu_llm_write_all_participant_structure_scores(
    *,
    client: OpenAI,
    model_name: str,
    use_instruction_path: Path,
    run_instruction_path: Path,
    participant_ids: List[int],
    participant_train_trials: Dict[int, List[Dict[str, Any]]],
    dataset: str,
    analysis_dir: Path,
    structure_prompt_max_tokens: int,
    max_response_tokens: int,
    model_context_tokens: int = 32768,
    token_estimate_slack: float = _RBU_STRUCTURE_TOKEN_ESTIMATE_SLACK,
) -> Tuple[str, int, int]:
    """
    One LLM call for all participants. Writes ``analysis_dir / Structure_score_all.txt`` (raw LLM output).

    The API counts **prompt + max_tokens** against the model context window; trial packing uses an inflated
    token estimate (``token_estimate_slack``) so cl100k / char heuristics stay below both
    ``--structure_prompt_max_tokens`` and ``model_context_tokens - max_completion``.

    Returns ``(raw_llm_text, trials_per_participant_used, estimated_prompt_tokens)``.
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)
    use_txt = use_instruction_path.read_text(encoding="utf-8")
    run_txt = run_instruction_path.read_text(encoding="utf-8")
    prefix = use_txt + _RBU_PREPARED_INSTRUCTION_HEADER + run_txt

    pids = sorted(int(x) for x in participant_ids)
    train_by: Dict[int, List[Dict[str, Any]]] = {pid: list(participant_train_trials[pid]) for pid in pids}
    if not pids:
        raise ValueError("rbu_llm_write_all_participant_structure_scores: empty participant_ids")
    min_full = min(len(train_by[pid]) for pid in pids)

    slack = float(token_estimate_slack)
    if slack < 1.0:
        raise ValueError("token_estimate_slack must be >= 1.0")

    def _inflate(n: int) -> int:
        return int(math.ceil(float(n) * slack))

    completion_reserve = int(max_response_tokens) + 64
    if int(model_context_tokens) <= completion_reserve:
        raise ValueError(
            f"model_context_tokens={model_context_tokens} must exceed max_response_tokens={max_response_tokens} + 64"
        )
    # Fit prompt under user cap AND under (context window − requested completion), both in inflated units.
    prompt_cap = min(int(structure_prompt_max_tokens), int(model_context_tokens) - completion_reserve)
    if prompt_cap < 1:
        raise RuntimeError(
            f"RBU structure prompt: effective prompt cap is {prompt_cap} "
            f"(structure_prompt_max_tokens={structure_prompt_max_tokens}, "
            f"model_context_tokens={model_context_tokens}, max_response_tokens={max_response_tokens})."
        )

    if _inflate(count_tokens_approx(prefix)) > prompt_cap:
        raise RuntimeError(
            f"RBU structure prompt: use_instruction + instruction alone inflate to "
            f"{_inflate(count_tokens_approx(prefix))} tokens (slack={slack}), exceeds effective prompt cap {prompt_cap}."
        )

    def _total_tokens_for_k(k: int) -> int:
        body = _rbu_build_all_participant_trials_block(pids, train_by, k, dataset) if k > 0 else ""
        return count_tokens_approx(prefix + body)

    lo, hi = 0, min_full
    best_k = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _inflate(_total_tokens_for_k(mid)) <= prompt_cap:
            best_k = mid
            lo = mid + 1
        else:
            hi = mid - 1

    final_body = _rbu_build_all_participant_trials_block(pids, train_by, best_k, dataset) if best_k > 0 else ""
    user_content = prefix + final_body
    est_tokens = count_tokens_approx(user_content)
    est_inflated = _inflate(est_tokens)
    max_resp_eff = min(int(max_response_tokens), int(model_context_tokens) - est_inflated - 64)
    if max_resp_eff < 256:
        raise RuntimeError(
            f"RBU structure prompt leaves insufficient room for completion "
            f"(raw_prompt_est={est_tokens}, inflated={est_inflated}, model_context_tokens={model_context_tokens}). "
            "Increase --structure_model_context_tokens, reduce --structure_prompt_max_tokens, or use fewer trials."
        )

    print(
        f"[RBU] structure_prompt_max_tokens={structure_prompt_max_tokens} "
        f"effective_prompt_cap={prompt_cap} (inflated est., slack={slack}) "
        f"estimated_prompt_tokens={est_tokens} inflated_prompt_est={est_inflated} "
        f"model_context_tokens={model_context_tokens} max_completion_tokens={max_resp_eff} "
        f"num_participants={len(pids)} training_trials_per_participant={best_k}"
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.2,
        top_p=0.95,
        max_tokens=max_resp_eff,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("RBU combined structure-score LLM returned empty output.")
    out_path = analysis_dir / _RBU_STRUCTURE_SCORE_ALL_FILENAME
    out_path.write_text(raw + ("\n" if not raw.endswith("\n") else ""), encoding="utf-8")
    return raw, best_k, est_tokens


def _select_train_trials_under_token_budget(
    template_text: str,
    train_trials: List[Dict[str, Any]],
    dataset: str,
    max_prompt_tokens: int,
    *,
    max_trial_block_tokens: Optional[int] = None,
    trials_block_header: str = _PROFILE_TRIALS_HEADER,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    """Greedy prefix of train_trials to include in the text-profile prompt.

    If ``max_trial_block_tokens`` is set (CPC18), only the serialized **trials** block is capped
    that many approximate tokens; template + header are always kept in full.

    Otherwise, the full user message (template + header + trials) must fit within
    ``max_prompt_tokens`` approximate tokens.
    """
    prefix = template_text + trials_block_header
    prefix_tokens = _approx_prompt_token_count(prefix)

    if max_trial_block_tokens is not None:
        if not train_trials:
            return [], prefix_tokens, False
        full_block = _format_trials_for_text_profile(train_trials, dataset)
        if _approx_prompt_token_count(full_block) <= max_trial_block_tokens:
            return (
                list(train_trials),
                prefix_tokens + _approx_prompt_token_count(full_block),
                False,
            )
        selected: List[Dict[str, Any]] = []
        for t in train_trials:
            cand = selected + [t]
            block = _format_trials_for_text_profile(cand, dataset)
            if _approx_prompt_token_count(block) <= max_trial_block_tokens:
                selected = cand
            else:
                break
        if not selected:
            selected = [train_trials[0]]
            block = _format_trials_for_text_profile(selected, dataset)
            return (
                selected,
                prefix_tokens + _approx_prompt_token_count(block),
                len(train_trials) > 1,
            )
        block = _format_trials_for_text_profile(selected, dataset)
        return (
            selected,
            prefix_tokens + _approx_prompt_token_count(block),
            len(selected) < len(train_trials),
        )

    if not train_trials:
        approx = _approx_prompt_token_count(prefix)
        return [], approx, False
    full_block = _format_trials_for_text_profile(train_trials, dataset)
    full_prompt = prefix + full_block
    if _approx_prompt_token_count(full_prompt) <= max_prompt_tokens:
        return list(train_trials), _approx_prompt_token_count(full_prompt), False
    selected = []
    for t in train_trials:
        cand = selected + [t]
        block = _format_trials_for_text_profile(cand, dataset)
        tot = _approx_prompt_token_count(prefix + block)
        if tot <= max_prompt_tokens:
            selected = cand
        else:
            break
    if not selected:
        selected = [train_trials[0]]
        block = _format_trials_for_text_profile(selected, dataset)
        tot = _approx_prompt_token_count(prefix + block)
        return selected, tot, len(train_trials) > 1
    block = _format_trials_for_text_profile(selected, dataset)
    tot = _approx_prompt_token_count(prefix + block)
    return selected, tot, len(selected) < len(train_trials)


def run_text_profile_warmup_participant(
    *,
    dataset: str,
    participant_id: int,
    train_trials: List[Dict[str, Any]],
    client: OpenAI,
    model_name: str,
    base_run_dir: str,
    max_prompt_tokens: int = 10_000,
    profile_response_max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Phase 1 (te_aggregate): one LLM call to write participant_{id}/profile.txt."""
    template_path = _text_profile_prompt_path(dataset)
    template_text = template_path.read_text(encoding="utf-8")
    participant_dir = Path(base_run_dir) / f"participant_{participant_id}"
    participant_dir.mkdir(parents=True, exist_ok=True)
    trial_cap: Optional[int] = (
        _TEXT_PROFILE_CPC18_MAX_TRIAL_BLOCK_TOKENS if dataset == "cpc18" else None
    )
    selected, approx_tokens, truncated = _select_train_trials_under_token_budget(
        template_text,
        train_trials,
        dataset,
        max_prompt_tokens,
        max_trial_block_tokens=trial_cap,
    )
    trials_block = _format_trials_for_text_profile(selected, dataset)
    user_content = f"{template_text}{_PROFILE_TRIALS_HEADER}{trials_block}"
    cap_note = f" trial_block_cap≈{trial_cap}" if trial_cap is not None else ""
    print(
        f"[text-profile] participant={participant_id} "
        f"trials_in_prompt={len(selected)}/{len(train_trials)} "
        f"approx_prompt_tokens={approx_tokens} truncated={truncated}{cap_note}"
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.35,
        top_p=0.95,
        max_tokens=profile_response_max_tokens,
    )
    profile_text = (resp.choices[0].message.content or "").strip()
    profile_path = participant_dir / "profile.txt"
    profile_path.write_text(profile_text + ("\n" if profile_text else ""), encoding="utf-8")
    meta: Dict[str, Any] = {
        "participant_id": participant_id,
        "n_train_trials_total": len(train_trials),
        "n_train_trials_in_prompt": len(selected),
        "approx_prompt_tokens": approx_tokens,
        "truncated": truncated,
        "template_path": str(template_path),
        "profile_path": str(profile_path),
    }
    if trial_cap is not None:
        meta["max_trial_block_tokens_approx"] = trial_cap
    (participant_dir / "text_profile_meta.json").write_text(_json_dumps_safe(meta, indent=2), encoding="utf-8")
    return {**meta, "profile_text": profile_text}


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
    n_eval_seeds: int = 1,
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
    max_prompt_train: int = 10,
    max_prompt_trials_per_problem: int = 0,
    llm_max_tokens: int = 800,
    cpc18_official_mse: bool = False,
    adaptation_mode: bool = False,
    aggregate_base_code: Optional[str] = None,
    participant_text_profile: Optional[str] = None,
    max_prompt_val: int = 10,
    data_driven_mode: bool = False,
    choice13k_val_trials_override: Optional[List[Dict[str, Any]]] = None,
    choice13k_loglik_prompt_path: Optional[str] = None,
    use_evolution_stages: bool = False,
    evolution_stage1_prompt_path: Optional[str] = None,
    evolution_stage2_prompt_path: Optional[str] = None,
    num_diagnostic_trials: Optional[int] = None,
    lambda_complexity: float = 0.0,
    lambda_change: float = 0.0,
    hard_participant_train_loglik_threshold: float = -0.6,
    hard_participant_warmup_iters: int = 5,
    early_stop: bool = False,
    debug_continue_after_early_stop: bool = False,
    wandb_log_fn=None,
    local_dataset: Optional[str] = None,
    rbu_lambda: float = 30.0,
    use_rbu: bool = True,
    participant_bir: float = 0.0,
    participant_rbu: float = 0.0,
    rbu_prompt_threshold: float = 0.6,
    structure_score: Optional[float] = None,
    structure_components: Optional[Dict[str, float]] = None,
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
    if fitness_metric == "loglik" and not (
        dataset in {"choice13k", "mixed_gambles"} or (dataset == "cpc18" and not cpc18_official_mse)
    ):
        raise ValueError(
            "fitness_metric='loglik' is only supported for choice13k/mixed_gambles, or for cpc18 when "
            "not using the official MSE protocol (cpc18_official_mse=False)."
        )
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")

    val_trials: Optional[List[Dict[str, Any]]] = None

    # Initialize client
    if client_kwargs is None:
        client_kwargs = {}
    client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
    
    # Load seed program
    print(f"Loading seed program from {seed_program_path}...")
    seed_code = load_seed_program(seed_program_path)
    
    # Branch based on dataset
    is_cpc18_mse = False
    is_cpc18_split = False
    if dataset == "gridworld":
        if num_blocks is None or num_walls is None or agent_id is None:
            raise ValueError("For gridworld, num_blocks, num_walls, and agent_id must be provided")
        print(f"Gridworld mode (non-strict): num_blocks={num_blocks}, num_walls={num_walls}, agent_id={agent_id}")
        train_trials = None  # Not used for gridworld
        test_trials = None
        options = None
        test_observed_blocks = None
    elif dataset == "cpc18":
        # Load CPC18 Track II data
        # Use datasets/cpc18 as default data_path if not specified
        cpc18_data_path = data_path if data_path != "data" else "datasets/cpc18"
        print(f"Loading CPC18 Track II data for participant {participant_id} from {cpc18_data_path}...")
        participant_data = load_cpc18_track2_data(data_path=cpc18_data_path, participant_id=participant_id)
        is_cpc18_mse = bool(cpc18_official_mse)
        is_cpc18_split = not is_cpc18_mse
        # Official: all trials, block MSE. Otherwise: per-participant holdout (problem or trial split).
        train_trials, test_trials, test_observed_blocks = split_cpc18_trials(
            participant_data,
            train_ratio=0.8,
            cpc18_official_mse=is_cpc18_mse,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        if is_cpc18_mse:
            print(f"CPC18 official: ALL {len(train_trials)} trials; block MSE vs Data-to-predict-Track-2")
            print(f"Problems: {len(participant_data.problems)} total; block targets: {len(test_observed_blocks)}")
        else:
            print(
                f"CPC18 held-out split: train={len(train_trials)} test={len(test_trials)} "
                f"(split_ratio={split_ratio:.3f} seed={split_seed})"
            )
            print(f"Problems: {len(participant_data.problems)} total (partition by problem when possible).")
        options = None
    elif dataset == "mixed_gambles":
        csv_path = "datasets/mixed_gambles/data_all_2021-01-08.csv"
        print(f"Loading mixed_gambles data for participant (subject) {participant_id} from {csv_path}...")
        train_trials, test_trials, options = load_mixed_gambles_data(
            csv_path,
            participant_id,
            filter_gain_loss_only=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        print(f"[Split] Train: {len(train_trials)}, Test: {len(test_trials)} (seed={split_seed}, ratio={split_ratio:.3f})")
        test_observed_blocks = None
    else:
        # Load Choice13k data
        if choice13k_train_trials_override is not None and choice13k_test_trials_override is not None:
            train_trials = choice13k_train_trials_override
            test_trials = choice13k_test_trials_override
            options = train_trials[0]["options"] if train_trials else ([0, 1] if test_trials else [])
            print("Loading Choice13k data from across-participants split (precomputed).")
            print(f"[Split] Train trials: {len(train_trials)}, Test trials: {len(test_trials)}")
            if data_driven_mode:
                if choice13k_val_trials_override is not None:
                    val_trials = choice13k_val_trials_override
                else:
                    val_trials, test_trials = _pool_split_val_test(test_trials, split_seed)
                print(
                    f"[Split] data_driven val/test from pooled test: val={len(val_trials)}, test={len(test_trials)}"
                )
        else:
            print(f"Loading Choice13k data for participant {participant_id}...")
            if choice13k_experiment is not None:
                exp = choice13k_experiment
            else:
                experiments = get_choice13k_experiments(
                    n_participants=participant_id + 1,
                    local_dataset=local_dataset,
                )
                exp = experiments[participant_id]
            train_trials, val_trials, test_trials, options = split_trials(
                exp, split_ratio=split_ratio, split_seed=split_seed
            )
            print(
                f"[Split] Train: {len(train_trials)}, Val: {len(val_trials)}, Test: {len(test_trials)} "
                f"(seed={split_seed}, train_fraction={split_ratio:.3f})"
            )
        test_observed_blocks = None
    
    use_data_driven_choice13k = (
        bool(data_driven_mode)
        and dataset == "choice13k"
        and fitness_metric == "loglik"
        and val_trials is not None
        and len(val_trials) > 0
    )
    if data_driven_mode and dataset == "choice13k" and fitness_metric == "loglik":
        if val_trials is None or len(val_trials) == 0:
            raise ValueError("data_driven_mode for choice13k+loglik requires non-empty val_trials.")

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        mode = (
            "te_dr"
            if dataset == "choice13k"
            else ("te_aggregate" if dataset in _PARTICIPANT_DATASETS else "non_strict")
        )
        if dataset == "gridworld":
            output_dir = f"generated_outputs/gridworld/{mode}/run_{timestamp}/epoch_0/agent_{agent_id}"
        elif dataset == "cpc18":
            output_dir = f"generated_outputs/cpc18/{mode}/run_{timestamp}/participant_{participant_id}"
        elif dataset == "mixed_gambles":
            output_dir = f"generated_outputs/mixed_gambles/{mode}/run_{timestamp}/participant_{participant_id}"
        else:
            output_dir = f"generated_outputs/choice13k/{mode}/run_{timestamp}/participant_{participant_id}"
    output_path = Path(output_dir)
    if save_artifacts:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up local log file for wandb metrics (if wandb is enabled)
    log_file_path = None
    if wandb is not None and save_artifacts and not (choice13k_simple_logging and dataset == "choice13k"):
        log_file_path = output_path / "wandb_metrics.jsonl"
    
    # ===== BASELINE EVALUATION =====
    print(f"\n{'='*80}")
    if adaptation_mode:
        print(
            "BASELINE EVALUATION (phase 2): evaluating initializer from "
            f"{seed_program_path}\n"
            "  (te_aggregate phase 2 adapts from this seed program; optional text profile is injected into prompts.)"
        )
    else:
        print(f"BASELINE EVALUATION: Evaluating seed program ({seed_program_path})")
    print(f"{'='*80}")
    
    baseline_val_eval = None
    if dataset == "gridworld":
        # Train: Evaluate on first 20 steps (matching ROTE's training/weighting phase)
        baseline_train_eval = evaluate_gridworld_program(
            seed_code, data_path, num_blocks, num_walls, agent_id,
            num_datapoints=80, num_steps=20, verbose=True, n_seeds=n_eval_seeds,
            evaluate_on_observed=True  # Match ROTE's training: evaluate on first 20 steps
        )
        # Test: Evaluate on future steps (matching ROTE's evaluation phase)
        baseline_test_eval = evaluate_gridworld_program(
            seed_code, data_path, num_blocks, num_walls, agent_id,
            num_datapoints=20, num_steps=20, verbose=True, n_seeds=n_eval_seeds,
            evaluate_on_observed=False  # Match ROTE's evaluation: evaluate on future steps
        )
    elif dataset == "cpc18":
        baseline_fn = compile_program(seed_code)
        if baseline_fn is None:
            print("ERROR: Failed to compile baseline program!")
            return None
        if is_cpc18_mse:
            baseline_train_eval = evaluate_cpc18_program(
                baseline_fn, train_trials, verbose=True, n_seeds=n_eval_seeds
            )
            baseline_test_eval = evaluate_cpc18_program(
                baseline_fn, test_trials, verbose=True, n_seeds=n_eval_seeds
            )
            train_observed_blocks = test_observed_blocks
            baseline_train_mse_eval = evaluate_cpc18_mse(
                baseline_fn, train_trials, train_observed_blocks, verbose=True, n_seeds=n_eval_seeds
            )
            baseline_test_mse_eval = evaluate_cpc18_mse(
                baseline_fn, test_trials, test_observed_blocks, verbose=True, n_seeds=n_eval_seeds
            )
        else:
            baseline_train_mse_eval = None
            baseline_test_mse_eval = None
            baseline_train_eval = evaluate_cpc18_split_program(
                baseline_fn, train_trials, verbose=True, n_seeds=n_eval_seeds
            )
            baseline_test_eval = evaluate_cpc18_split_program(
                baseline_fn, test_trials, verbose=True, n_seeds=n_eval_seeds
            )
    else:
        baseline_fn = compile_program(seed_code)
        if baseline_fn is None:
            print("ERROR: Failed to compile baseline program!")
            return None
        if dataset == "choice13k" or (dataset == "mixed_gambles" and fitness_metric == "loglik"):
            baseline_train_eval = evaluate_choice13k_program(
                baseline_fn, train_trials, verbose=True, n_seeds=n_eval_seeds
            )
            baseline_test_eval = evaluate_choice13k_program(
                baseline_fn, test_trials, verbose=True, n_seeds=n_eval_seeds
            )
            if use_data_driven_choice13k and val_trials is not None:
                baseline_val_eval = evaluate_choice13k_program(
                    baseline_fn, val_trials, verbose=True, n_seeds=n_eval_seeds
                )
            else:
                baseline_val_eval = None
        else:
            baseline_train_eval = evaluate_program(baseline_fn, train_trials, verbose=True, n_seeds=n_eval_seeds)
            baseline_test_eval = evaluate_program(baseline_fn, test_trials, verbose=True, n_seeds=n_eval_seeds)
    
    print(f"\nBaseline Performance:")
    print(f"  Train accuracy: {baseline_train_eval['accuracy']:.4f} ({baseline_train_eval['correct']}/{baseline_train_eval['total']})")
    print(f"  Test accuracy: {baseline_test_eval['accuracy']:.4f} ({baseline_test_eval['correct']}/{baseline_test_eval['total']})")
    if dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split:
        print(
            f"  Train avg log-likelihood: {baseline_train_eval['avg_loglik']:.6f}, "
            f"test: {baseline_test_eval['avg_loglik']:.6f}"
        )
        if use_data_driven_choice13k and baseline_val_eval is not None:
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
    if dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split:
        baseline_results["train_loglik"] = baseline_train_eval["avg_loglik"]
        baseline_results["test_loglik"] = baseline_test_eval["avg_loglik"]
        if use_data_driven_choice13k and baseline_val_eval is not None:
            baseline_results["val_loglik"] = baseline_val_eval["avg_loglik"]

    # BIR-regularized baseline (step 0): same formulas as per-iteration pool-best logs.
    # Must run before W&B / JSONL baseline so iter -1 is comparable to later steps.
    baseline_confidence_penalty = 0.0
    baseline_selection_score = None
    if (
        adaptation_mode
        and (not use_data_driven_choice13k)
        and fitness_metric == "loglik"
        and (dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split)
    ):
        baseline_confidence_penalty = (
            _compute_confidence_penalty(baseline_fn, train_trials) if baseline_fn is not None else 0.0
        )
        baseline_selection_score = _compute_selection_score(
            float(baseline_train_eval["avg_loglik"]),
            rbu_lambda=rbu_lambda,
            residual_behavioral_uncertainty=float(participant_rbu),
            confidence_penalty=float(baseline_confidence_penalty),
        )
        if dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split:
            baseline_results["confidence_penalty"] = float(baseline_confidence_penalty)
            baseline_results["selection_score"] = float(baseline_selection_score)

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
        else:
            if all_data_mode:
                if dataset in {"choice13k", "mixed_gambles"} and fitness_metric == "loglik":
                    baseline_log_dict = {
                        f"p{participant_id}_train_fitness": baseline_train_eval["avg_loglik"],
                        f"p{participant_id}_test_fitness": baseline_test_eval["avg_loglik"],
                        f"p{participant_id}_train_acc": baseline_train_eval["accuracy"],
                        f"p{participant_id}_test_acc": baseline_test_eval["accuracy"],
                        f"p{participant_id}_train_loglik": baseline_train_eval["avg_loglik"],
                        f"p{participant_id}_test_loglik": baseline_test_eval["avg_loglik"],
                    }
                    if use_data_driven_choice13k and dataset == "choice13k" and baseline_val_eval is not None:
                        baseline_log_dict[f"p{participant_id}_val_loglik"] = baseline_val_eval["avg_loglik"]
                        baseline_log_dict["val_loglik"] = baseline_val_eval["avg_loglik"]
                else:
                    baseline_log_dict = {
                        f"p{participant_id}_train_fitness": baseline_train_eval["accuracy"],
                        f"p{participant_id}_test_fitness": baseline_test_eval["accuracy"],
                    }
                    if dataset in {"choice13k", "mixed_gambles"}:
                        baseline_log_dict[f"p{participant_id}_train_acc"] = baseline_train_eval["accuracy"]
                        baseline_log_dict[f"p{participant_id}_test_acc"] = baseline_test_eval["accuracy"]
                        baseline_log_dict[f"p{participant_id}_train_loglik"] = baseline_train_eval["avg_loglik"]
                        baseline_log_dict[f"p{participant_id}_test_loglik"] = baseline_test_eval["avg_loglik"]
            else:
                baseline_log_dict = {
                    f"p{participant_id}_train_accuracy": baseline_train_eval["accuracy"],
                    f"p{participant_id}_test_accuracy": baseline_test_eval["accuracy"],
                    f"p{participant_id}_is_baseline": 1,
                }
                if dataset in {"choice13k", "mixed_gambles"}:
                    baseline_log_dict[f"p{participant_id}_train_loglik"] = baseline_train_eval["avg_loglik"]
                    baseline_log_dict[f"p{participant_id}_test_loglik"] = baseline_test_eval["avg_loglik"]
                    baseline_log_dict[f"p{participant_id}_train_acc"] = baseline_train_eval["accuracy"]
                    baseline_log_dict[f"p{participant_id}_test_acc"] = baseline_test_eval["accuracy"]
                    if use_data_driven_choice13k and dataset == "choice13k" and baseline_val_eval is not None:
                        baseline_log_dict[f"p{participant_id}_val_loglik"] = baseline_val_eval["avg_loglik"]
                        baseline_log_dict["val_loglik"] = baseline_val_eval["avg_loglik"]
        if participant_id is not None:
            baseline_log_dict[f"p{participant_id}_step"] = 0
            if (
                (dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split)
                and baseline_train_eval is not None
                and baseline_test_eval is not None
            ):
                baseline_log_dict[f"p{participant_id}/train_loglik"] = baseline_train_eval.get("avg_loglik", None)
                baseline_log_dict[f"p{participant_id}/test_loglik"] = baseline_test_eval.get("avg_loglik", None)
                if use_data_driven_choice13k and dataset == "choice13k" and baseline_val_eval is not None:
                    baseline_log_dict[f"p{participant_id}/val_loglik"] = baseline_val_eval.get("avg_loglik", None)
                    baseline_log_dict["val_loglik"] = baseline_val_eval.get("avg_loglik", None)
                if adaptation_mode and fitness_metric == "loglik" and baseline_selection_score is not None:
                    baseline_log_dict[f"p{participant_id}/confidence_penalty"] = baseline_confidence_penalty
                    baseline_log_dict[f"p{participant_id}/selection_score"] = baseline_selection_score
                    baseline_log_dict[f"p{participant_id}_confidence_penalty"] = baseline_confidence_penalty
                    baseline_log_dict[f"p{participant_id}_selection_score"] = baseline_selection_score
        if wandb_log_fn is not None:
            wandb_log_fn(baseline_log_dict)
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
                f.write(_json_dumps_safe(baseline_entry) + "\n")

    # Initialize best program tracking with baseline
    if is_cpc18_mse and baseline_train_mse_eval is not None:
        best_fitness = -baseline_train_mse_eval['mse']
    elif is_cpc18_split and fitness_metric == "loglik":
        best_fitness = (
            baseline_selection_score
            if adaptation_mode and not use_data_driven_choice13k
            else baseline_train_eval["avg_loglik"]
        )
    elif is_cpc18_split:
        best_fitness = baseline_train_eval["accuracy"]
    elif dataset in {"choice13k", "mixed_gambles"} and fitness_metric == "loglik":
        best_fitness = (
            baseline_selection_score
            if adaptation_mode and not use_data_driven_choice13k
            else baseline_train_eval["avg_loglik"]
        )
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
        if dataset in {"choice13k", "mixed_gambles"}:
            overall_best_train["train_loglik"] = baseline_train_eval["avg_loglik"]
            overall_best_train["test_loglik"] = baseline_test_eval["avg_loglik"]
            overall_best_test["train_loglik"] = baseline_train_eval["avg_loglik"]
            overall_best_test["test_loglik"] = baseline_test_eval["avg_loglik"]
            if use_data_driven_choice13k and baseline_val_eval is not None:
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
            (
                baseline_selection_score
                if adaptation_mode and not use_data_driven_choice13k
                else baseline_train_eval["avg_loglik"]
            )
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
        if dataset == "choice13k":
            _baseline_fit = (
                (
                    baseline_selection_score
                    if adaptation_mode and not use_data_driven_choice13k
                    else baseline_train_eval["avg_loglik"]
                )
                if fitness_metric == "loglik"
                else baseline_train_eval["accuracy"]
            )
            _baseline_val_ll = (
                baseline_val_eval["avg_loglik"]
                if use_data_driven_choice13k and baseline_val_eval is not None
                else None
            )
            if use_data_driven_choice13k:
                elite_parents = [(
                    seed_code,
                    _baseline_fit,
                    baseline_test_eval["accuracy"],
                    "baseline",
                    None,
                    None,
                    baseline_train_eval["accuracy"],
                    _baseline_val_ll,
                )]
            else:
                elite_parents = [(
                    seed_code,
                    _baseline_fit,
                    baseline_test_eval["accuracy"],
                    "baseline",
                    None,
                    None,
                    baseline_train_eval["accuracy"],
                )]
        elif dataset == "mixed_gambles":
            _baseline_fit = (
                (baseline_selection_score if adaptation_mode else baseline_train_eval["avg_loglik"])
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
    runtime_error_bank: List[Dict[str, Any]] = []
    participant_stopped_early = False
    participant_early_stop_iteration: Optional[int] = None
    frozen_best_code: Optional[str] = None
    frozen_best_program_id: Optional[str] = None
    frozen_overall_best_train: Optional[Dict[str, Any]] = None
    frozen_overall_best_test: Optional[Dict[str, Any]] = None

    # Evolution loop (uses elite_parents pool for parent selection, not a single parent_program)
    simple_iterations_rows: List[Dict[str, Any]] = []
    simple_iterations_dir = None
    if choice13k_simple_logging and dataset == "choice13k" and save_artifacts:
        simple_iterations_dir = output_path / "iterations"
        simple_iterations_dir.mkdir(parents=True, exist_ok=True)
    for iteration in range(n_iterations):
        iteration_step = iteration + 1  # 1-indexed to match wandb (0 = baseline)
        iter_best_val_loglik: Optional[float] = None
        # Baseline seed is used only for initialization / first generation.
        # From iteration 2 onward, never allow it in the elite parent pool.
        if iteration_step >= 2:
            elite_parents = [p for p in elite_parents if p[3] != "baseline"]
        print(f"\n{'='*80}")
        print(f"Iteration {iteration_step}/{n_iterations}")
        print(f"{'='*80}")
        
        iter_dir = None
        candidates_dir = None
        if save_artifacts and not (choice13k_simple_logging and dataset == "choice13k"):
            iter_dir = output_path / f"iteration_{iteration_step}"
            iter_dir.mkdir(exist_ok=True)
            candidates_dir = iter_dir / "candidates"
            candidates_dir.mkdir(exist_ok=True)
        
        # Select sample_size parents: either top programs by fitness or uniform sample without replacement
        # from the trimmed elite pool (see _elite_pool_capacity / elite_pool_size).
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
        parent_runtime_errors: List[Optional[str]] = [None for _ in selected_parents]
        def _fmt_opt(v: Optional[float], ndigits: int = 4) -> str:
            if v is None:
                return "N/A"
            return f"{v:.{ndigits}f}"
        if is_cpc18_mse:
            for i, parent_tuple in enumerate(selected_parents):
                code, fitness, test_mse, prog_id, train_mse, test_mse = parent_tuple
                print(
                    f"  Parent {i+1}: {prog_id} "
                    f"(train_mse={_fmt_opt(train_mse, 2)}, test_mse={_fmt_opt(test_mse, 2)}, fitness={_fmt_opt(fitness, 2)})"
                )
        else:
            for i, parent_tuple in enumerate(selected_parents):
                code, fitness, test_acc, prog_id, _, _, train_acc_prompt = parent_tuple[:7]
                if fitness_metric == "loglik" and (dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split):
                    if (
                        use_data_driven_choice13k
                        and dataset == "choice13k"
                        and len(parent_tuple) > 7
                        and parent_tuple[7] is not None
                    ):
                        print(
                            f"  Parent {i+1}: {prog_id} "
                            f"(train_loglik={_fmt_opt(fitness)}, val_loglik={_fmt_opt(float(parent_tuple[7]))})"
                        )
                    else:
                        metric_label = (
                            "selection_score"
                            if adaptation_mode and (not use_data_driven_choice13k)
                            else "train_loglik"
                        )
                        print(
                            f"  Parent {i+1}: {prog_id} "
                            f"({metric_label}={_fmt_opt(fitness)})"
                        )
                else:
                    print(
                        f"  Parent {i+1}: {prog_id} "
                        f"(train_acc={_fmt_opt(train_acc_prompt)}, test_acc={_fmt_opt(test_acc)})"
                    )
        
        parent_train_accs = None
        parent_train_mses = None
        parent_test_mses = None
        if is_cpc18_mse:
            parent_train_mses = [p[4] for p in selected_parents if p[4] is not None]
            parent_test_mses = [p[5] for p in selected_parents if p[5] is not None]
        else:
            # In loglik mode, feed train fitness (log-likelihood) into prompt context.
            if fitness_metric == "loglik" and (dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split):
                parent_train_accs = [p[1] for p in selected_parents]
            else:
                parent_train_accs = [p[6] for p in selected_parents]
        parent_val_logliks: Optional[List[Optional[float]]] = None
        if use_data_driven_choice13k and dataset == "choice13k":
            parent_val_logliks = [
                float(p[7]) if len(p) > 7 and p[7] is not None else None for p in selected_parents
            ]
        needs_fallback = (
            len(selected_parents) < sample_size
            or (len(selected_parents) == 1 and selected_parents[0][3] == "baseline")
        )
        if needs_fallback and runtime_error_bank:
            selected_ids = {p[3] for p in selected_parents}
            candidates = [r for r in runtime_error_bank if r["program_id"] not in selected_ids]
            finite = [r for r in candidates if np.isfinite(float(r["fitness"])) and float(r["fitness"]) > -1e8]
            if finite:
                finite.sort(key=lambda r: float(r["fitness"]), reverse=True)
                extras = finite[:3]
            else:
                rng = np.random.default_rng(split_seed + iteration_step)
                take = min(3, len(candidates))
                if take > 0:
                    idxs = rng.choice(len(candidates), size=take, replace=False)
                    extras = [candidates[int(i)] for i in idxs]
                else:
                    extras = []
            if extras:
                print(f"Adding {len(extras)} fallback parent(s) with runtime error context.")
            for e in extras:
                parent_codes.append(e["code"])
                parent_runtime_errors.append(e.get("runtime_error_line"))
                if parent_train_accs is not None:
                    parent_train_accs.append(float(e["fitness"]))
                if parent_train_mses is not None:
                    parent_train_mses.append(None)
        
        iter_loglik_prompt_path = choice13k_loglik_prompt_path
        if use_evolution_stages and use_data_driven_choice13k:
            stage_cutoff = _evolution_stage_cutoff(n_iterations)
            stage_num = 1 if iteration_step <= stage_cutoff else 2
            iter_loglik_prompt_path = _te_data_driven_evolution_stage_prompt_path(
                iteration_step,
                n_iterations,
                stage1_path=(
                    Path(evolution_stage1_prompt_path)
                    if evolution_stage1_prompt_path
                    else None
                ),
                stage2_path=(
                    Path(evolution_stage2_prompt_path)
                    if evolution_stage2_prompt_path
                    else None
                ),
                max_prompt_val=max_prompt_val,
                fallback_path=choice13k_loglik_prompt_path,
            )
            print(
                f"[LLM prompt] Evolution stage {stage_num} "
                f"(iteration {iteration_step}/{n_iterations}, stage-1 through {stage_cutoff}): "
                f"{iter_loglik_prompt_path}"
            )

        # Generate candidate programs (full code, not just parameters)
        print(f"\nGenerating {n_candidates_per_iteration} candidate programs...")
        diagnostic_text = ""
        # In data-driven Choice13k+loglik, held-out validation trials are injected via --max_prompt_val
        # (_build_val_examples_prompt_block), not via --num_diagnostic_trials. Skip diagnostic blocks then to
        # avoid duplicating "extra trial context" and to keep prompts smaller (te_aggregate phase-2 still uses
        # num_diagnostic_trials when not on this path).
        if adaptation_mode and (not use_data_driven_choice13k) and num_diagnostic_trials:
            diagnostic_text = _build_diagnostic_trials_text(
                parent_code=parent_codes[0] if parent_codes else seed_code,
                train_trials=train_trials if train_trials is not None else [],
                dataset=dataset,
                num_diagnostic_trials=num_diagnostic_trials,
            )
        adaptation_instruction = ""
        # Same split as diagnostics: data-driven run uses te_data_driven template + val examples; omit the
        # phase-2 "minimal adaptation" / RBU-BIR instruction block which targets te_aggregate-style prompts.
        if adaptation_mode and (not use_data_driven_choice13k):
            profile_block = ""
            if participant_text_profile and participant_text_profile.strip():
                profile_block = (
                    "\n\nParticipant behavioral profile (qualitative summary from training observations; "
                    "treat as soft guidance, not hard constraints):\n"
                    f"{participant_text_profile.strip()}\n"
                )
            base_adapt = (
                "Modify the seed program minimally for this participant. "
                "Small structural edits are allowed, but avoid full rewrites unless necessary. "
                "Prefer simple changes that improve log-likelihood and generalize to unseen trials.\n"
            )
            if float(participant_rbu) < float(rbu_prompt_threshold):
                adaptation_instruction = profile_block + base_adapt
            else:
                if use_rbu:
                    rbu_note = (
                        f"This participant has residual behavioral uncertainty RBU = {float(participant_rbu):.3f} "
                        f"(behavioral inconsistency rate BIR = {float(participant_bir):.3f} is diagnostic only); "
                        "avoid unjustified extreme probabilities such as 0.99/0.01 and prefer calibrated rules."
                    )
                    adaptation_instruction = (
                        profile_block
                        + base_adapt
                        + "Phase 2 uses an RBU-regularized score: "
                        + f"selection_score = train_loglik - {float(rbu_lambda):.3f} * (RBU^2) * mean((p - 0.5)^2), "
                        + f"with RBU = {float(participant_rbu):.3f}.\n"
                        + f"{rbu_note}\n"
                    )
                else:
                    bir_note = (
                        f"This participant has BIR = {float(participant_bir):.2f}; "
                        "avoid unjustified extreme probabilities such as 0.99/0.01 and prefer calibrated rules."
                    )
                    adaptation_instruction = (
                        profile_block
                        + base_adapt
                        + f"Phase 2 uses a BIR-regularized score: "
                        + f"selection_score = train_loglik - {float(rbu_lambda):.3f} * (BIR^2) * mean((p - 0.5)^2), "
                        + f"with BIR = {float(participant_bir):.3f}.\n"
                        + f"{bir_note}\n"
                    )
        if dataset == "gridworld":
            candidate_codes = generate_gridworld_program_variants(
                client=client,
                model_name=model_name,
                template_code=seed_code,
                parent_codes=parent_codes,
                n_variants=n_candidates_per_iteration,
                parent_train_accuracies=parent_train_accs,
            )
        else:
            candidate_codes = generate_program_variants(
                client=client,
                model_name=model_name,
                parent_programs=parent_codes,
                train_trials=train_trials,
                n_variants=n_candidates_per_iteration,
                max_tokens=llm_max_tokens,
                dataset=dataset,
                parent_train_accuracies=parent_train_accs if (dataset != "cpc18" or is_cpc18_split) else None,
                parent_train_mses=parent_train_mses if is_cpc18_mse else None,
                max_prompt_train=max_prompt_train,
                max_prompt_trials_per_problem=max_prompt_trials_per_problem,
                prompt_train_trials_seed=split_seed,
                fitness_metric=fitness_metric,
                cpc18_official_mse=is_cpc18_mse,
                include_train_trials_in_prompt=(
                    ((not adaptation_mode) and (not use_data_driven_choice13k))
                    or (use_data_driven_choice13k and max_prompt_train > 0)
                ),
                show_train_omitted_message=(
                    adaptation_mode and (not use_data_driven_choice13k)
                ),
                base_program_code=(
                    aggregate_base_code if adaptation_mode and (not use_data_driven_choice13k) else None
                ),
                diagnostic_trials_text=diagnostic_text,
                extra_prompt_instructions=adaptation_instruction,
                parent_runtime_errors=parent_runtime_errors,
                parent_metric_label_override=(
                    "selection_score"
                    if adaptation_mode
                    and fitness_metric == "loglik"
                    and (not use_data_driven_choice13k)
                    else None
                ),
                val_trials_for_prompt=val_trials if use_data_driven_choice13k else None,
                max_prompt_val=max_prompt_val,
                parent_val_logliks=parent_val_logliks,
                choice13k_loglik_prompt_path=(
                    iter_loglik_prompt_path if use_data_driven_choice13k else None
                ),
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
                if dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split:
                    empty_row["train_loglik"] = float("-inf")
                    empty_row["test_loglik"] = float("-inf")
                    empty_row["fitness"] = float("-inf") if fitness_metric == "loglik" else 0.0
                    empty_row["runtime_valid"] = False
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
                    candidate_results.append({
                        "idx": idx,
                        "code": code,
                        "train_acc": 0.0,
                        "test_acc": 0.0,
                        "train_loglik": float("-inf"),
                        "test_loglik": float("-inf"),
                        "fitness": _worst,
                        "valid": False,
                        "runtime_valid": False,
                    })
                    continue
                try:
                    train_eval = evaluate_cpc18_split_program(choose_fn, train_trials, n_seeds=n_eval_seeds)
                    test_eval = (
                        None
                        if adaptation_mode
                        else evaluate_cpc18_split_program(choose_fn, test_trials, n_seeds=n_eval_seeds)
                    )
                except (TypeError, ValueError, AssertionError):
                    candidate_results.append({
                        "idx": idx,
                        "code": code,
                        "train_acc": 0.0,
                        "test_acc": 0.0,
                        "train_loglik": float("-inf"),
                        "test_loglik": float("-inf"),
                        "fitness": _worst,
                        "valid": False,
                        "runtime_valid": False,
                    })
                    continue
                train_acc = train_eval["accuracy"]
                test_acc = test_eval["accuracy"] if test_eval is not None else None
                train_loglik = train_eval["avg_loglik"]
                test_loglik = test_eval["avg_loglik"] if test_eval is not None else None
                runtime_valid = (train_eval.get("errors", 0) == 0) and (
                    test_eval is None or test_eval.get("errors", 0) == 0
                )
                confidence_penalty = None
                selection_score = None
                fitness = train_loglik if fitness_metric == "loglik" else train_acc
                if not runtime_valid:
                    fitness = -1e9 if fitness_metric == "loglik" else float("-inf")
                elif adaptation_mode and fitness_metric == "loglik":
                    confidence_penalty = _compute_confidence_penalty(choose_fn, train_trials)
                    selection_score = _compute_selection_score(
                        float(train_loglik),
                        rbu_lambda=rbu_lambda,
                        residual_behavioral_uncertainty=float(participant_rbu),
                        confidence_penalty=float(confidence_penalty),
                    )
                    fitness = selection_score
                candidate_results.append({
                    "idx": idx,
                    "code": code,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_loglik": train_loglik,
                    "test_loglik": test_loglik,
                    "confidence_penalty": confidence_penalty,
                    "selection_score": selection_score,
                    "fitness": fitness,
                    "train_correct": train_eval["correct"],
                    "test_correct": test_eval["correct"] if test_eval is not None else None,
                    "train_total": train_eval["total"],
                    "test_total": test_eval["total"] if test_eval is not None else None,
                    "valid": True,
                    "runtime_valid": runtime_valid,
                })
                if not runtime_valid:
                    err_line = _probe_runtime_error_line(choose_fn, train_trials)
                    candidate_results[-1]["runtime_error_line"] = err_line
                    runtime_error_bank.append(
                        {
                            "program_id": f"iter{iteration_step}_cand{idx}",
                            "code": code,
                            "fitness": fitness,
                            "runtime_error_line": err_line,
                        }
                    )
            elif dataset == "choice13k":
                choose_fn = compile_program(code)
                _worst = float("-inf") if fitness_metric == "loglik" else 0.0
                if choose_fn is None:
                    er = {
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
                    if use_data_driven_choice13k:
                        er["val_loglik"] = float("-inf")
                    candidate_results.append(er)
                    continue
                try:
                    train_eval = evaluate_choice13k_program(choose_fn, train_trials, n_seeds=n_eval_seeds)
                    test_eval = (
                        None
                        if adaptation_mode and (not use_data_driven_choice13k)
                        else evaluate_choice13k_program(choose_fn, test_trials, n_seeds=n_eval_seeds)
                    )
                    if use_data_driven_choice13k and val_trials is not None:
                        val_eval = evaluate_choice13k_program(choose_fn, val_trials, n_seeds=n_eval_seeds)
                    else:
                        val_eval = None
                except (AssertionError, TypeError, ValueError):
                    er = {
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
                    if use_data_driven_choice13k:
                        er["val_loglik"] = float("-inf")
                    candidate_results.append(er)
                    continue
                train_acc = train_eval["accuracy"]
                test_acc = test_eval["accuracy"] if test_eval is not None else None
                train_loglik = train_eval["avg_loglik"]
                test_loglik = test_eval["avg_loglik"] if test_eval is not None else None
                val_loglik = val_eval["avg_loglik"] if val_eval is not None else None
                runtime_valid = train_eval.get("errors", 0) == 0
                if test_eval is not None:
                    runtime_valid = runtime_valid and test_eval.get("errors", 0) == 0
                if val_eval is not None:
                    runtime_valid = runtime_valid and val_eval.get("errors", 0) == 0
                confidence_penalty = None
                selection_score = None
                fitness = train_loglik if fitness_metric == "loglik" else train_acc
                if not runtime_valid:
                    fitness = -1e9 if fitness_metric == "loglik" else float("-inf")
                elif (
                    adaptation_mode
                    and fitness_metric == "loglik"
                    and (not use_data_driven_choice13k)
                ):
                    confidence_penalty = _compute_confidence_penalty(choose_fn, train_trials)
                    selection_score = _compute_selection_score(
                        float(train_loglik),
                        rbu_lambda=rbu_lambda,
                        residual_behavioral_uncertainty=float(participant_rbu),
                        confidence_penalty=float(confidence_penalty),
                    )
                    fitness = selection_score
                row = {
                    "idx": idx,
                    "code": code,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_loglik": train_loglik,
                    "test_loglik": test_loglik,
                    "confidence_penalty": confidence_penalty,
                    "selection_score": selection_score,
                    "fitness": fitness,
                    "train_correct": train_eval["correct"],
                    "test_correct": test_eval["correct"] if test_eval is not None else None,
                    "train_total": train_eval["total"],
                    "test_total": test_eval["total"] if test_eval is not None else None,
                    "valid": True,
                    "runtime_valid": runtime_valid,
                }
                if use_data_driven_choice13k:
                    row["val_loglik"] = val_loglik
                candidate_results.append(row)
                if not runtime_valid:
                    err_line = _probe_runtime_error_line(choose_fn, train_trials)
                    candidate_results[-1]["runtime_error_line"] = err_line
                    runtime_error_bank.append(
                        {
                            "program_id": f"iter{iteration_step}_cand{idx}",
                            "code": code,
                            "fitness": fitness,
                            "runtime_error_line": err_line,
                        }
                    )
            else:
                # mixed_gambles: support accuracy and optional log-likelihood fitness
                choose_fn = compile_program(code)
                if choose_fn is None:
                    row = {
                        "idx": idx,
                        "code": code,
                        "train_acc": 0.0,
                        "test_acc": 0.0,
                        "fitness": 0.0,
                        "valid": False,
                        "runtime_valid": False,
                    }
                    if fitness_metric == "loglik":
                        row["train_loglik"] = float("-inf")
                        row["test_loglik"] = float("-inf")
                        row["fitness"] = float("-inf")
                    candidate_results.append(row)
                    continue
                if fitness_metric == "loglik":
                    train_eval = evaluate_choice13k_program(choose_fn, train_trials, n_seeds=n_eval_seeds)
                    test_eval = (
                        None
                        if adaptation_mode
                        else evaluate_choice13k_program(choose_fn, test_trials, n_seeds=n_eval_seeds)
                    )
                else:
                    train_eval = evaluate_program(choose_fn, train_trials, n_seeds=n_eval_seeds)
                    test_eval = (
                        None
                        if adaptation_mode
                        else evaluate_program(choose_fn, test_trials, n_seeds=n_eval_seeds)
                    )
                train_acc = train_eval["accuracy"]
                test_acc = test_eval["accuracy"] if test_eval is not None else None
                fitness = train_eval["avg_loglik"] if fitness_metric == "loglik" else train_acc
                runtime_valid = True
                confidence_penalty = None
                selection_score = None
                if fitness_metric == "loglik":
                    runtime_valid = (train_eval.get("errors", 0) == 0) and (
                        test_eval is None or test_eval.get("errors", 0) == 0
                    )
                    if not runtime_valid:
                        fitness = -1e9
                    elif adaptation_mode:
                        confidence_penalty = _compute_confidence_penalty(choose_fn, train_trials)
                        selection_score = _compute_selection_score(
                            float(train_eval["avg_loglik"]),
                            rbu_lambda=rbu_lambda,
                            residual_behavioral_uncertainty=float(participant_rbu),
                            confidence_penalty=float(confidence_penalty),
                        )
                        fitness = selection_score
                row = {
                    "idx": idx,
                    "code": code,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "fitness": fitness,
                    "train_correct": train_eval["correct"],
                    "test_correct": test_eval["correct"] if test_eval is not None else None,
                    "train_total": train_eval["total"],
                    "test_total": test_eval["total"] if test_eval is not None else None,
                    "valid": True,
                    "runtime_valid": runtime_valid,
                }
                if fitness_metric == "loglik":
                    row["train_loglik"] = train_eval["avg_loglik"]
                    row["test_loglik"] = None
                    row["confidence_penalty"] = confidence_penalty
                    row["selection_score"] = selection_score
                candidate_results.append(row)
                if fitness_metric == "loglik" and not runtime_valid:
                    err_line = _probe_runtime_error_line(choose_fn, train_trials)
                    candidate_results[-1]["runtime_error_line"] = err_line
                    runtime_error_bank.append(
                        {
                            "program_id": f"iter{iteration_step}_cand{idx}",
                            "code": code,
                            "fitness": fitness,
                            "runtime_error_line": err_line,
                        }
                    )
            
        # Report results
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1} Results:")
        print(f"{'='*80}")
        
        compile_valid_results = [r for r in candidate_results if r.get("valid", False)]
        # For probability-evaluation datasets, selection must use runtime-valid programs only.
        if dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split:
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
                title = (
                    "Top performers (by selection_score, higher is better):"
                    if adaptation_mode
                    else "Top performers (by train avg log-likelihood, higher is better):"
                )
                print(f"\n{title}")
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
                        f"selection_score={_fmt_opt(result.get('selection_score'), 6)}, "
                        f"confidence_penalty={_fmt_opt(result.get('confidence_penalty'), 6)}, "
                        f"train_acc={result['train_acc']:.4f}, "
                        f"test_acc={_test_acc}"
                    )
            elif dataset in {"choice13k", "mixed_gambles"} and fitness_metric == "loglik":
                title = (
                    "Top performers (by selection_score, higher is better):"
                    if adaptation_mode and (not use_data_driven_choice13k)
                    else "Top performers (by train avg log-likelihood, higher is better):"
                )
                print(f"\n{title}")
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
                    if use_data_driven_choice13k and dataset == "choice13k":
                        print(
                            f"  {i+1}. Candidate {result['idx']}: "
                            f"train_loglik={result['train_loglik']:.6f}, "
                            f"val_loglik={_val_ll}, "
                            f"test_loglik={_test_ll}, "
                            f"train_acc={result['train_acc']:.4f}, "
                            f"test_acc={_test_acc}"
                        )
                    else:
                        print(
                            f"  {i+1}. Candidate {result['idx']}: "
                            f"train_loglik={result['train_loglik']:.6f}, "
                            f"test_loglik={_test_ll}, "
                            f"selection_score={_fmt_opt(result.get('selection_score'), 6)}, "
                            f"confidence_penalty={_fmt_opt(result.get('confidence_penalty'), 6)}, "
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
            elif dataset in {"choice13k", "mixed_gambles"} and fitness_metric == "loglik":
                print(f"  Train accuracy: {best_result['train_acc']:.4f}")
                if use_data_driven_choice13k and dataset == "choice13k" and best_result.get("val_loglik") is not None:
                    print(f"  Val avg log-likelihood: {best_result['val_loglik']:.6f}")
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
                    if use_data_driven_choice13k and dataset == "choice13k":
                        elite_parents.append((
                            result["code"],
                            result["fitness"],
                            result["test_acc"],
                            program_id,
                            None,
                            None,
                            result["train_acc"],
                            result.get("val_loglik"),
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
            # Remove seed baseline from pool-best selection/logging from iteration 1 onward.
            if iteration_step >= 1:
                elite_parents = [p for p in elite_parents if p[3] != "baseline"]

            # Sort elite set by fitness (descending) and keep top programs
            # For CPC18: fitness = -MSE (higher is better)
            # For others: fitness = accuracy (higher is better)
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
            iter_best_confidence_penalty = best_result.get("confidence_penalty")
            iter_best_selection_score = best_result.get("selection_score")
            if fitness_metric == "loglik" and (is_cpc18_split or dataset in {"choice13k", "mixed_gambles"}):
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
                        if use_data_driven_choice13k and val_trials is not None:
                            iter_best_val_eval = evaluate_choice13k_program(
                                iter_best_fn, val_trials, n_seeds=n_eval_seeds
                            )
                            iter_best_val_loglik = iter_best_val_eval["avg_loglik"]
                    iter_best_train_acc = iter_best_train_eval["accuracy"]
                    iter_best_test_acc = iter_best_test_eval["accuracy"]
                    iter_best_train_loglik = iter_best_train_eval["avg_loglik"]
                    iter_best_test_loglik = iter_best_test_eval["avg_loglik"]
                    if adaptation_mode and (not use_data_driven_choice13k):
                        iter_best_confidence_penalty = _compute_confidence_penalty(iter_best_fn, train_trials)
                        iter_best_selection_score = _compute_selection_score(
                            float(iter_best_train_loglik),
                            rbu_lambda=rbu_lambda,
                            residual_behavioral_uncertainty=float(participant_rbu),
                            confidence_penalty=float(iter_best_confidence_penalty),
                        )
                        iter_best_fitness = float(iter_best_selection_score)
                    else:
                        iter_best_fitness = float(iter_best_train_loglik)
                else:
                    # Should be rare; keep loop stable if a pool entry cannot recompile.
                    iter_best_test_acc = None
                    iter_best_test_loglik = None
                    iter_best_val_loglik = None

            if choice13k_simple_logging and dataset == "choice13k" and save_artifacts and simple_iterations_dir is not None:
                (simple_iterations_dir / f"iteration_{iteration_step}.py").write_text(iter_best_code or "")
                row_it = {
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
                    "selection_score": iter_best_selection_score,
                    "confidence_penalty": iter_best_confidence_penalty,
                }
                if use_data_driven_choice13k:
                    row_it["val_loglik"] = iter_best_val_loglik
                simple_iterations_rows.append(row_it)
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
            elif dataset == "choice13k":
                if fitness_metric == "loglik":
                    _train_better = (
                        iter_best_train_loglik is not None
                        and iter_best_train_loglik > overall_best_train["train_loglik"]
                    )
                else:
                    _train_better = best_result["train_acc"] > overall_best_train["train_accuracy"]
                if _train_better:
                    ob = {
                        "train_accuracy": iter_best_train_acc,
                        "test_accuracy": iter_best_test_acc,
                        "train_loglik": iter_best_train_loglik,
                        "test_loglik": iter_best_test_loglik,
                        "program_id": iter_best_program_id,
                    }
                    if use_data_driven_choice13k:
                        ob["val_loglik"] = iter_best_val_loglik
                    overall_best_train = ob
                if fitness_metric == "loglik":
                    _test_better = (
                        iter_best_test_loglik is not None
                        and iter_best_test_loglik > overall_best_test["test_loglik"]
                    )
                else:
                    _test_better = best_result["test_acc"] > overall_best_test["test_accuracy"]
                if _test_better:
                    ob2 = {
                        "train_accuracy": iter_best_train_acc,
                        "test_accuracy": iter_best_test_acc,
                        "train_loglik": iter_best_train_loglik,
                        "test_loglik": iter_best_test_loglik,
                        "program_id": iter_best_program_id,
                    }
                    if use_data_driven_choice13k:
                        ob2["val_loglik"] = iter_best_val_loglik
                    overall_best_test = ob2
            elif dataset == "mixed_gambles":
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
            if (dataset in {"choice13k", "mixed_gambles"}) or is_cpc18_split:
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
                        "train_loglik": r.get("train_loglik"),
                        "test_loglik": r.get("test_loglik"),
                        "selection_score": r.get("selection_score"),
                        "confidence_penalty": r.get("confidence_penalty"),
                        "fitness": r.get("fitness"),
                        "valid": r["valid"],
                        "runtime_valid": r.get("runtime_valid", r["valid"]),
                    }
                    for r in candidate_results
                ],
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_acc": iter_best_train_acc if selected_results else None,
                "best_test_acc": iter_best_test_acc if selected_results else None,
                "best_train_loglik": iter_best_train_loglik if selected_results else None,
                "best_test_loglik": iter_best_test_loglik if selected_results else None,
                "best_selection_score": iter_best_selection_score if selected_results else None,
                "best_confidence_penalty": iter_best_confidence_penalty if selected_results else None,
            }
        elif dataset in {"choice13k", "mixed_gambles"}:
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
                        "train_loglik": r.get("train_loglik"),
                        "test_loglik": r.get("test_loglik"),
                        "val_loglik": r.get("val_loglik"),
                        "selection_score": r.get("selection_score"),
                        "confidence_penalty": r.get("confidence_penalty"),
                        "fitness": r.get("fitness"),
                        "valid": r["valid"],
                        "runtime_valid": r.get("runtime_valid", r["valid"]),
                    }
                    for r in candidate_results
                ],
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_acc": iter_best_train_acc if selected_results else None,
                "best_test_acc": iter_best_test_acc if selected_results else None,
                "best_train_loglik": iter_best_train_loglik if selected_results else None,
                "best_test_loglik": iter_best_test_loglik if selected_results else None,
                "best_val_loglik": (
                    iter_best_val_loglik
                    if (selected_results and use_data_driven_choice13k and dataset == "choice13k")
                    else None
                ),
            }
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
            (iter_dir / "metrics.json").write_text(_json_dumps_safe(metrics, indent=2))
        
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
                "best_selection_score": iter_best_selection_score if selected_results else None,
                "best_confidence_penalty": iter_best_confidence_penalty if selected_results else None,
                "n_valid": len(compile_valid_results),
                "n_runtime_valid": len(selected_results),
            }
        elif dataset in {"choice13k", "mixed_gambles"}:
            summary = {
                "iteration": iteration_step,
                "best_train_fitness": best_fitness if selected_results else None,
                "best_train_acc": iter_best_train_acc if selected_results else None,
                "best_test_acc": iter_best_test_acc if selected_results else None,
                "best_train_loglik": iter_best_train_loglik if selected_results else None,
                "best_test_loglik": iter_best_test_loglik if selected_results else None,
                "best_val_loglik": (
                    iter_best_val_loglik
                    if (selected_results and use_data_driven_choice13k and dataset == "choice13k")
                    else None
                ),
                "best_selection_score": iter_best_selection_score if selected_results else None,
                "best_confidence_penalty": iter_best_confidence_penalty if selected_results else None,
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
        print(f"\nSummary: {_json_dumps_safe(summary, indent=2)}")

        # Hard-participant early-stop rule for phase-2 adaptation under loglik fitness.
        if (
            adaptation_mode
            and early_stop
            and fitness_metric == "loglik"
            and (dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split)
            and iteration_step >= hard_participant_warmup_iters
        ):
            cur_best_train_loglik = overall_best_train.get("train_loglik")
            if (
                cur_best_train_loglik is not None
                and cur_best_train_loglik < hard_participant_train_loglik_threshold
            ):
                first_early_stop = not participant_stopped_early
                participant_stopped_early = True
                if first_early_stop:
                    participant_early_stop_iteration = iteration_step
                    print(
                        "[EARLY STOP] participant adaptation stopped: "
                        f"best_train_loglik={cur_best_train_loglik:.6f} < "
                        f"threshold={hard_participant_train_loglik_threshold:.6f} "
                        f"after warmup_iters={hard_participant_warmup_iters}."
                    )
                if debug_continue_after_early_stop:
                    if first_early_stop:
                        frozen_best_code = elite_parents[0][0]
                        frozen_best_program_id = elite_parents[0][3]
                        frozen_overall_best_train = dict(overall_best_train)
                        frozen_overall_best_test = dict(overall_best_test)
                        print(
                            "[DEBUG] Continuing evolution for logging; "
                            "final metrics will use the early-stop snapshot.",
                            flush=True,
                        )
                else:
                    break
        
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
                        log_dict[f"p{participant_id}_avg_train_accuracy"] = np.mean(
                            [r["train_acc"] for r in valid_results]
                        )
                        if adaptation_mode and fitness_metric == "loglik":
                            log_dict[f"p{participant_id}_selection_score"] = iter_best_selection_score
                            log_dict[f"p{participant_id}_confidence_penalty"] = iter_best_confidence_penalty
                        if fitness_metric != "loglik":
                            log_dict[f"p{participant_id}_avg_test_accuracy"] = np.mean(
                                [r["test_acc"] for r in valid_results]
                            )
            elif dataset in {"choice13k", "mixed_gambles"}:
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
                        if use_data_driven_choice13k and dataset == "choice13k":
                            log_dict[f"p{participant_id}_val_loglik"] = iter_best_val_loglik
                            log_dict["val_loglik"] = iter_best_val_loglik
                        if adaptation_mode and fitness_metric == "loglik":
                            log_dict[f"p{participant_id}_selection_score"] = iter_best_selection_score
                            log_dict[f"p{participant_id}_confidence_penalty"] = iter_best_confidence_penalty
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
                        if use_data_driven_choice13k and dataset == "choice13k":
                            log_dict[f"p{participant_id}_val_loglik"] = iter_best_val_loglik
                            log_dict["val_loglik"] = iter_best_val_loglik
                        log_dict[f"p{participant_id}_train_fitness"] = best_fitness
                        log_dict[f"p{participant_id}_test_fitness"] = (
                            iter_best_test_loglik
                            if fitness_metric == "loglik"
                            else iter_best_test_acc
                        )
                        log_dict[f"p{participant_id}_avg_train_accuracy"] = np.mean([r["train_acc"] for r in valid_results])
                        if adaptation_mode and fitness_metric == "loglik":
                            log_dict[f"p{participant_id}_selection_score"] = iter_best_selection_score
                            log_dict[f"p{participant_id}_confidence_penalty"] = iter_best_confidence_penalty
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
                log_dict[f"p{participant_id}_step"] = iteration + 1
                if dataset in {"choice13k", "mixed_gambles"} or is_cpc18_split:
                    log_dict[f"p{participant_id}/train_loglik"] = iter_best_train_loglik
                    log_dict[f"p{participant_id}/test_loglik"] = iter_best_test_loglik
                    if use_data_driven_choice13k and dataset == "choice13k":
                        log_dict[f"p{participant_id}/val_loglik"] = iter_best_val_loglik
                        log_dict["val_loglik"] = iter_best_val_loglik
                    if adaptation_mode and fitness_metric == "loglik" and (not use_data_driven_choice13k):
                        log_dict[f"p{participant_id}/selection_score"] = iter_best_selection_score
                        log_dict[f"p{participant_id}/confidence_penalty"] = iter_best_confidence_penalty
                        wb_tag = "[W&B RBU]" if use_rbu else "[W&B BIR]"
                        print(
                            wb_tag,
                            participant_id,
                            iteration_step,
                            iter_best_selection_score,
                            iter_best_confidence_penalty,
                            f"rate={'RBU' if use_rbu else 'BIR'}={float(participant_rbu):.4f}",
                        )
            if wandb_log_fn is not None:
                wandb_log_fn(log_dict)
            else:
                wandb.log(log_dict, step=iteration + 1)  # Step starts at 1 (baseline is step=0)
            
            # Also save to local JSONL file
            if save_artifacts and log_file_path is not None:
                log_entry = {
                    "step": iteration + 1,
                    "iteration": iteration_step,
                    **log_dict
                }
                with open(log_file_path, "a") as f:
                    f.write(_json_dumps_safe(log_entry) + "\n")
    
    # Final summary and save comprehensive results.json
    print(f"\n{'='*80}")
    print("Evolution Complete")
    print(f"{'='*80}")

    # Select final best program directly from the final elite pool (already sorted by train fitness).
    # This guarantees final reporting is paired from one candidate.
    if (
        participant_stopped_early
        and debug_continue_after_early_stop
        and frozen_best_code is not None
        and frozen_best_program_id is not None
    ):
        final_best_code = frozen_best_code
        final_best_program_id = frozen_best_program_id
        if frozen_overall_best_train is not None:
            overall_best_train = frozen_overall_best_train
        if frozen_overall_best_test is not None:
            overall_best_test = frozen_overall_best_test
        print(
            "[INFO] Final reporting is frozen at early-stop snapshot "
            f"(iteration {participant_early_stop_iteration})."
        )
    elif ((dataset in {"choice13k", "mixed_gambles"}) or is_cpc18_split) and (not runtime_valid_evolved_found):
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
        best_program_filename = f"best_program_fr_iter{best_iteration}_cand{best_candidate_idx}.py"
    elif final_best_program_id == "baseline":
        best_iteration = -1
        best_candidate_idx = -1
        best_program_filename = "best_program_fr_baseline.py"
    else:
        best_iteration = None
        best_candidate_idx = None
        safe_program_id = re.sub(r"[^A-Za-z0-9_.-]", "_", final_best_program_id)
        best_program_filename = f"best_program_fr_{safe_program_id}.py"

    if save_artifacts:
        (output_path / best_program_filename).write_text(final_best_code or "")

    final_best_fn = compile_program(final_best_code)
    if final_best_fn is None:
        raise RuntimeError(
            f"Final best program failed to compile: {final_best_program_id}"
        )

    final_confidence_penalty: Optional[float] = None
    final_selection_score: Optional[float] = None
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
    elif is_cpc18_split:
        final_train_eval = evaluate_cpc18_split_program(final_best_fn, train_trials, n_seeds=n_eval_seeds)
        final_test_eval = evaluate_cpc18_split_program(final_best_fn, test_trials, n_seeds=n_eval_seeds)
        if adaptation_mode and fitness_metric == "loglik":
            final_confidence_penalty = _compute_confidence_penalty(final_best_fn, train_trials)
            final_selection_score = _compute_selection_score(
                float(final_train_eval["avg_loglik"]),
                rbu_lambda=rbu_lambda,
                residual_behavioral_uncertainty=float(participant_rbu),
                confidence_penalty=float(final_confidence_penalty),
            )
        overall_best_train = {
            "train_accuracy": final_train_eval["accuracy"],
            "test_accuracy": final_test_eval["accuracy"],
            "train_loglik": final_train_eval["avg_loglik"],
            "test_loglik": final_test_eval["avg_loglik"],
            "confidence_penalty": final_confidence_penalty,
            "selection_score": final_selection_score,
            "program_id": final_best_program_id,
            "origin_iteration": best_iteration,
            "origin_candidate_idx": best_candidate_idx,
            "program_file": best_program_filename,
        }
        overall_best_test = dict(overall_best_train)
    elif dataset == "choice13k" or (dataset == "mixed_gambles" and fitness_metric == "loglik"):
        final_train_eval = evaluate_choice13k_program(final_best_fn, train_trials, n_seeds=n_eval_seeds)
        final_test_eval = evaluate_choice13k_program(final_best_fn, test_trials, n_seeds=n_eval_seeds)
        final_val_eval = None
        if use_data_driven_choice13k and dataset == "choice13k" and val_trials is not None:
            final_val_eval = evaluate_choice13k_program(final_best_fn, val_trials, n_seeds=n_eval_seeds)
        if adaptation_mode and fitness_metric == "loglik" and (not use_data_driven_choice13k):
            final_confidence_penalty = _compute_confidence_penalty(final_best_fn, train_trials)
            final_selection_score = _compute_selection_score(
                float(final_train_eval["avg_loglik"]),
                rbu_lambda=rbu_lambda,
                residual_behavioral_uncertainty=float(participant_rbu),
                confidence_penalty=float(final_confidence_penalty),
            )
        overall_best_train = {
            "train_accuracy": final_train_eval["accuracy"],
            "test_accuracy": final_test_eval["accuracy"],
            "train_loglik": final_train_eval["avg_loglik"],
            "test_loglik": final_test_eval["avg_loglik"],
            "confidence_penalty": final_confidence_penalty,
            "selection_score": final_selection_score,
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
        (output_path / "results.json").write_text(_json_dumps_safe(results, indent=2))
    if save_artifacts and choice13k_simple_logging and dataset == "choice13k":
        if simple_iterations_rows:
            _it_fields = list(simple_iterations_rows[0].keys())
            with open(output_path / "iterations.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_it_fields)
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
        if use_data_driven_choice13k:
            summary_row["val_loglik"] = overall_best_train.get("val_loglik")
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
        elif dataset == "choice13k":
            print(f"Final best train accuracy: {overall_best_train['train_accuracy']:.4f} (from {overall_best_train['program_id']})")
            print(f"Final best test accuracy: {overall_best_test['test_accuracy']:.4f} (from {overall_best_test['program_id']})")
            print(
                f"Final best train avg log-likelihood: {overall_best_train['train_loglik']:.6f}, "
                f"test: {overall_best_test['test_loglik']:.6f}"
            )
            if use_data_driven_choice13k and overall_best_train.get("val_loglik") is not None:
                print(f"Final best val avg log-likelihood: {overall_best_train['val_loglik']:.6f}")
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
            "stopped_early": participant_stopped_early,
            "stopped_iteration": participant_early_stop_iteration,
        }
    elif is_cpc18_split:
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
                baseline_train_eval["avg_loglik"]
                if fitness_metric == "loglik"
                else baseline_train_eval["accuracy"]
            ),
            "seed_program_test_fitness": (
                baseline_test_eval["avg_loglik"]
                if fitness_metric == "loglik"
                else baseline_test_eval["accuracy"]
            ),
            "stopped_early": participant_stopped_early,
            "stopped_iteration": participant_early_stop_iteration,
            "behavioral_inconsistency_rate": float(participant_bir),
            "confidence_penalty": final_confidence_penalty,
            "selection_score": final_selection_score,
        }
    elif dataset == "choice13k":
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
            "stopped_early": participant_stopped_early,
            "stopped_iteration": participant_early_stop_iteration,
            "behavioral_inconsistency_rate": float(participant_bir),
            "confidence_penalty": final_confidence_penalty,
            "selection_score": final_selection_score,
        }
        if use_data_driven_choice13k:
            result["val_loglik"] = overall_best_train.get("val_loglik")
            result["seed_program_val_fitness"] = baseline_results.get("val_loglik")
    elif dataset == "mixed_gambles" and fitness_metric == "loglik":
        result = {
            "participant_id": participant_id,
            "train_acc": overall_best_train["train_accuracy"],
            "test_acc": overall_best_test["test_accuracy"],
            "train_loglik": overall_best_train["train_loglik"],
            "test_loglik": overall_best_test["test_loglik"],
            "train_fitness": overall_best_train["train_loglik"],
            "test_fitness": overall_best_test["test_loglik"],
            "seed_program_train_fitness": baseline_results["train_loglik"],
            "seed_program_test_fitness": baseline_results["test_loglik"],
            "stopped_early": participant_stopped_early,
            "stopped_iteration": participant_early_stop_iteration,
            "behavioral_inconsistency_rate": float(participant_bir),
            "confidence_penalty": final_confidence_penalty,
            "selection_score": final_selection_score,
        }
    else:
        result = {
            "participant_id": participant_id if dataset in ["choice13k", "cpc18", "mixed_gambles"] else agent_id,
            "train_acc": overall_best_train['train_accuracy'],
            "test_acc": overall_best_test['test_accuracy'],
            "train_fitness": overall_best_train['train_accuracy'],
            "test_fitness": overall_best_test['test_accuracy'],
            "seed_program_train_fitness": baseline_results['train_accuracy'],
            "seed_program_test_fitness": baseline_results['test_accuracy'],
            "stopped_early": participant_stopped_early,
            "stopped_iteration": participant_early_stop_iteration,
        }
    if adaptation_mode and fitness_metric == "loglik" and isinstance(result, dict) and "behavioral_inconsistency_rate" in result:
        result["residual_behavioral_uncertainty"] = float(participant_rbu)
        result["rbu_squared"] = float(participant_rbu) ** 2
        result["use_rbu"] = bool(use_rbu)
        if structure_score is not None:
            result["structure_score"] = float(structure_score)
        if structure_components is not None:
            result["structure_components"] = _json_dumps_safe(structure_components)
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
    n_eval_seeds: int = 1,
    sample_size: int = 3,
    top_k: int = 0,
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
                f.write(_json_dumps_safe({"step": 0, "iteration": -1, f"a{agent_id}_train_accuracy": baseline_train_eval["accuracy"], f"a{agent_id}_test_accuracy": baseline_test_eval["accuracy"], f"a{agent_id}_is_baseline": 1}) + "\n")

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
                        f.write(_json_dumps_safe({"step": iteration + 1, "iteration": iteration_step, **log_dict}) + "\n")

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
    (output_path / "results.json").write_text(_json_dumps_safe(results, indent=2))

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


def _parse_use_rbu_arg(value: str) -> bool:
    """Argparse type for --use_rbu True|False."""
    v = str(value).strip().lower()
    if v in ("true", "t", "1", "yes", "y"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    raise ValueError("Invalid --use_rbu value: use True or False (e.g. --use_rbu False).")


def _parse_profile_warmup_arg(value: str) -> bool:
    """Argparse type for --profile_warmup True|False."""
    v = str(value).strip().lower()
    if v in ("true", "t", "1", "yes", "y"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    raise ValueError(
        "Invalid --profile_warmup value: use True or False (e.g. --profile_warmup False)."
    )


def _parse_early_stop_arg(value: str) -> bool:
    """Argparse type for --early_stop True|False."""
    v = str(value).strip().lower()
    if v in ("true", "t", "1", "yes", "y"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    raise ValueError("Invalid --early_stop value: use True or False (e.g. --early_stop False).")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ROTE Evolution (Non-Strict): Iterative evolution of Choice13k and Gridworld programs")
    parser.add_argument(
        "--dataset",
        type=str,
        default="choice13k",
        choices=["choice13k", "gridworld", "gridworld_ensemble", "cpc18", "mixed_gambles"],
        help="Dataset to use: choice13k, gridworld, gridworld_ensemble (ensemble ablation), cpc18 (Track II), or mixed_gambles",
    )
    parser.add_argument(
        "--seed_path",
        type=str,
        default=None,
        help="Path to seed program (starting persona). If not set, auto-detects from persona_code_example/gridworld/ for gridworld. Default for choice13k: persona_code_example/vanilla.py. Default for cpc18: persona_code_example/cpc18/hard.py. Default for mixed_gambles: persona_code_example/hard_Qwen.py",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to data directory (for gridworld) or CPC18 Track II data directory (for cpc18, default: datasets/cpc18)",
    )
    parser.add_argument(
        "--loop_mode",
        type=str,
        default="random",
        choices=["random", "sequential"],
        help="Loop mode for gridworld: sequential evaluates problem configs systematically",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs (problem configs) to evaluate in sequential mode",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=None,
        help="Number of blocks in the problem (for gridworld)",
    )
    parser.add_argument(
        "--num_walls",
        type=int,
        default=None,
        help="Number of walls in the problem (for gridworld)",
    )
    parser.add_argument(
        "--agent_id",
        type=int,
        default=None,
        help="Agent type ID to evaluate (0-indexed, for gridworld). If None and num_agents_to_sample > 1, processes all agent types.",
    )
    parser.add_argument(
        "--participant_scope",
        type=str,
        default="single",
        choices=["single", "range", "ordinals", "all"],
        help=(
            "For choice13k / cpc18 / mixed_gambles: how to select participants. "
            "'single' uses --single_participant_id (raw id). "
            "'range' uses --range_start_ordinal/--range_end_ordinal (inclusive) into datasets/*/valid_participant_ids.json. "
            "'ordinals' uses --ordinals (0-based indices into that same list, not raw ids). "
            "'all' runs all raw ids from that JSON, optionally capped by --all_max_participants. "
            "Ignored for gridworld (use --num_agents_to_sample / --agent_id)."
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
        help="Gridworld / gridworld_ensemble only: number of agent types (0 .. num-1) when --agent_id is None. Ignored for choice13k/cpc18/mixed_gambles.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=10,
        help="Phase 2 per-participant adaptation iterations (default: 10)",
    )
    parser.add_argument(
        "--profile_warmup",
        type=_parse_profile_warmup_arg,
        default=True,
        metavar="True|False",
        help=(
            "te_aggregate (participant datasets): pass True to run phase-1 text-profile warmup before phase-2 "
            "(one LLM call per participant, saves participant_*/profile.txt); pass False to skip phase 1. "
            "Default: True."
        ),
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
        help="Gridworld (ROTE): number of episodes to evaluate (test-split trajectories). Default: 10",
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
            "Does not apply to gridworld_ensemble member pools."
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
        "--max_prompt_val",
        type=int,
        default=10,
        help=(
            "Data-driven Choice13k: max validation trials to include in each LLM generation prompt "
            "(structure + calibration context). Default: 10."
        ),
    )
    parser.add_argument(
        "--evolution_stages",
        action="store_true",
        help=(
            "Enable 2-stage evolution prompts for data-driven Choice13k+loglik: "
            "stage1.txt for the first half of iterations, stage2.txt for the rest. "
            "Default: off (single choices13k template)."
        ),
    )
    _stage1_default, _stage2_default = _default_evolution_stage_prompt_paths()
    parser.add_argument(
        "--evolution_stage1_prompt",
        type=Path,
        default=_stage1_default,
        help=f"Stage-1 prompt for the first half of iterations (default: {_stage1_default}).",
    )
    parser.add_argument(
        "--evolution_stage2_prompt",
        type=Path,
        default=_stage2_default,
        help=f"Stage-2 prompt for the remaining iterations (default: {_stage2_default}).",
    )
    parser.add_argument(
        "--lambda_complexity",
        type=float,
        default=0.0,
        help="Phase 2 regularization weight for code-complexity penalty.",
    )
    parser.add_argument(
        "--lambda_change",
        type=float,
        default=0.0,
        help="Phase 2 regularization weight for change-from-reference penalty (when used).",
    )
    parser.add_argument(
        "--uncertainty_lambda",
        "--rbu_lambda",
        dest="uncertainty_lambda",
        type=float,
        default=30.0,
        help=(
            "Phase 2 uncertainty-rate confidence regularization weight λ in "
            "selection_score = train_loglik - λ * (rate^2) * mean((p-0.5)^2) on train; "
            "rate is RBU when --use_rbu True, else BIR. "
            "Alias: --rbu_lambda (deprecated). Default: 30."
        ),
    )
    parser.add_argument(
        "--uncertainty_threshold",
        "--rbu_threshold",
        dest="uncertainty_threshold",
        type=float,
        default=0.6,
        help=(
            "Phase 2 adaptation prompts: if the regularization rate (RBU with --use_rbu True, else BIR) is strictly "
            "below this value, omit score/confidence wording in extra instructions. "
            "Alias: --rbu_threshold (deprecated). Default: 0.6."
        ),
    )
    parser.add_argument(
        "--use_rbu",
        type=_parse_use_rbu_arg,
        default=True,
        help="If True (default), run structure-score LLM steps and use RBU = clip(BIR - w * S, 0, 1) with w from "
        "--structure_weight in phase-2 selection. If False, use BIR as the regularization rate (ablation).",
    )
    parser.add_argument(
        "--structure_weight",
        "--rbu_structure_weight",
        dest="structure_weight",
        type=float,
        default=0.5,
        help=(
            "Multiplier w in RBU = clip(BIR - w * structure_score, 0, 1) when --use_rbu True. "
            "Alias: --rbu_structure_weight (deprecated). Default: 0.5."
        ),
    )
    parser.add_argument(
        "--structure_prompt_max_tokens",
        type=int,
        default=24000,
        metavar="N",
        help=(
            "Target cap (inflated token estimate, see --structure_model_context_tokens) for the combined RBU "
            "structure-scoring *input* (use_instruction + instruction + symmetric trial prefix per participant). "
            "The packer also enforces input + max completion <= model context. Default: 24000."
        ),
    )
    parser.add_argument(
        "--structure_model_context_tokens",
        type=int,
        default=32768,
        metavar="N",
        help=(
            "Full context window for the structure-scoring chat model (prompt + max completion). Used to reserve "
            "completion tokens and to cap inflated prompt estimates (default: 32768). Set to your server's "
            "context_length when it differs."
        ),
    )
    parser.add_argument(
        "--prepare_instruction_path",
        type=str,
        default=None,
        help="Override path to prepare_instruction.txt for RBU (default: per-dataset Template_evo/.../prepare_instruction.txt).",
    )
    parser.add_argument(
        "--use_instruction_path",
        type=str,
        default=None,
        help="Override path to use_instruction.txt for RBU (default: per-dataset Template_evo/.../use_instruction.txt).",
    )
    parser.add_argument(
        "--phase_option",
        type=str,
        choices=["all", "score", "evolution"],
        default="all",
        help=(
            "choice13k within-participant two-phase te_aggregate only: "
            "'all' (default) run BIR, structure-score LLM, then evolution; "
            "'score' stop after writing analysis/behavioral_inconsistency_rate.csv; "
            "'evolution' skip scoring LLM calls: recompute BIR on this run, parse S from --structure_path, "
            "then run evolution (RBU uses this run's --structure_weight)."
        ),
    )
    parser.add_argument(
        "--structure_path",
        type=str,
        default=None,
        help=(
            "choice13k --phase_option evolution with --use_rbu True: path to a prior run's combined "
            "structure-score file (same format as analysis/Structure_score_all.txt). "
            "BIR is recomputed on the current train split; S is read from this file; RBU uses this run's "
            "--structure_weight. Example: generated_outputs/choice13k/te_dr/<run>/analysis/Structure_score_all.txt"
        ),
    )
    parser.add_argument(
        "--hard_participant_train_loglik_threshold",
        type=float,
        default=-0.6,
        help=(
            "When --early_stop true: after warmup, stop adaptation if best train_loglik stays below this value."
        ),
    )
    parser.add_argument(
        "--hard_participant_warmup_iters",
        type=int,
        default=5,
        help="When --early_stop true: number of iterations before applying the train_loglik threshold check.",
    )
    parser.add_argument(
        "--early_stop",
        type=_parse_early_stop_arg,
        default=False,
        metavar="BOOL",
        help=(
            "Hard-participant early-stop in phase-2 loglik adaptation: after warmup, break the loop if best "
            "train_loglik stays below --hard_participant_train_loglik_threshold. Use True or False (default: False)."
        ),
    )
    parser.add_argument(
        "--debug_continue_after_early_stop",
        action="store_true",
        help=(
            "When early-stop triggers in phase-2, keep running remaining iterations for logs/plots "
            "but freeze final reporting to the early-stop best snapshot."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="For gridworld_ensemble: if > 0 and < sample_size, use top_k programs by weight only (0 = use all)",
    )
    parser.add_argument(
        "--n_eval_seeds",
        type=int,
        default=1,
        help="Number of evaluation runs per program (averaged). Default: 1",
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
        default="accuracy",
        choices=["accuracy", "loglik"],
        help=(
            "Fitness for selection (default: accuracy). For choice13k, or cpc18 without --cpc18_official_mse, "
            "loglik = mean Bernoulli log-likelihood on held-out-consistent train; use with --fitness_metric loglik for CPC18 split."
        ),
    )
    parser.add_argument(
        "--cpc18_official_mse",
        action="store_true",
        help=(
            "CPC18 only: official competition protocol (all trial data, block-level MSE vs Data-to-predict-Track-2). "
            "Omit (default) for a per-participant held-out problem/trial split with split_ratio/split_seed, "
            "comparable to mixed_gambles/choice13k; use with --fitness_metric loglik for the Bernoulli log-likelihood line."
        ),
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="within_participant",
        choices=["within_participant", "across_participants"],
        help="Choice13k split mode: within_participant (default) or across_participants.",
    )
    parser.add_argument(
        "--local_dataset",
        nargs="?",
        const="datasets/downloaded/choices13k/Psych-101-test",
        type=str,
        default=None,
        help=(
            "Choice13k only: path to a local Hugging Face dataset saved with datasets.save_to_disk "
            "(default when flag is provided without value: datasets/downloaded/choices13k/Psych-101-test). "
            "When set, skips remote HF download."
        ),
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help=(
            "Train fraction of problems (blocks) for Choice13k within-participant split; "
            "remaining blocks are split equally between validation and test (e.g. 0.8 → 80%% train, 10%% val, 10%% test)."
        ),
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Seed for deterministic splitting (default: 0).",
    )
    parser.add_argument(
        "--max_prompt_train",
        "--max_prompt_train_trials",
        dest="max_prompt_train",
        type=int,
        default=10,
        help=(
            "Train trials in each LLM generation prompt. 0 = omit train trials entirely. "
            "N>0: random subsample of at most N train trials (all train trials if fewer than N). "
            "Independent from --max_prompt_val. Default: 10. Alias: --max_prompt_train_trials."
        ),
    )
    parser.add_argument(
        "--max_prompt_trials_per_problem",
        type=int,
        default=0,
        help="Cap serialized train trials per problem in LLM prompts (0 = no per-problem cap).",
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=800,
        help="Max output tokens per candidate generation request (reduces context-overflow failures).",
    )

    args = parser.parse_args()
    if args.fitness_metric == "loglik" and args.dataset not in {"choice13k", "mixed_gambles"} and not (
        args.dataset == "cpc18" and not args.cpc18_official_mse
    ):
        print(
            "Error: --fitness_metric loglik is only supported for --dataset choice13k/mixed_gambles, "
            "or for cpc18 without --cpc18_official_mse (held-out split mode)."
        )
        return
    if not (0.0 < args.split_ratio < 1.0):
        print(f"Error: --split_ratio must be in (0,1), got {args.split_ratio}.")
        return
    if args.split_mode == "across_participants" and args.dataset != "choice13k":
        print("Error: --split_mode across_participants is only supported with --dataset choice13k.")
        return
    if args.local_dataset is not None and args.dataset != "choice13k":
        print("Warning: --local_dataset is only used for --dataset choice13k; ignoring it.")
    if args.max_prompt_train < 0:
        print("Error: --max_prompt_train must be >= 0 (0 = omit train trials from prompt).")
        return
    if args.max_prompt_trials_per_problem < 0:
        print("Error: --max_prompt_trials_per_problem must be >= 0.")
        return
    if args.llm_max_tokens < 64:
        print("Error: --llm_max_tokens must be >= 64.")
        return
    if args.n_iterations < 1:
        print("Error: --n_iterations must be >= 1.")
        return
    if args.max_prompt_val < 0:
        print("Error: --max_prompt_val must be >= 0.")
        return
    if args.lambda_complexity < 0.0 or args.lambda_change < 0.0:
        print("Error: --lambda_complexity and --lambda_change must be >= 0.")
        return
    if args.uncertainty_lambda < 0.0:
        print("Error: --uncertainty_lambda must be >= 0.")
        return
    if args.uncertainty_threshold < 0.0:
        print("Error: --uncertainty_threshold must be >= 0.")
        return
    if args.structure_weight < 0.0:
        print("Error: --structure_weight must be >= 0.")
        return
    if int(args.structure_prompt_max_tokens) < 1:
        print("Error: --structure_prompt_max_tokens must be >= 1.")
        return
    if int(args.structure_model_context_tokens) < 256:
        print("Error: --structure_model_context_tokens must be >= 256.")
        return
    if args.hard_participant_warmup_iters < 1:
        print("Error: --hard_participant_warmup_iters must be >= 1.")
        return
    if args.phase_option != "all" and args.dataset != "choice13k":
        print("Error: --phase_option other than 'all' is only supported for --dataset choice13k.")
        return
    if args.phase_option != "all" and getattr(args, "participant_scope", None) == "all":
        print(
            "Error: --phase_option score|evolution is not compatible with --participant_scope all "
            "(use single, range, or ordinals)."
        )
        return
    if args.phase_option == "evolution" and args.dataset == "choice13k" and args.use_rbu:
        if not args.structure_path or not str(args.structure_path).strip():
            print(
                "Error: --structure_path is required when --phase_option evolution (choice13k) and --use_rbu True."
            )
            return
        try:
            _resolve_scoring_input_path(str(args.structure_path))
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return
    if args.structure_path and str(args.structure_path).strip() and args.phase_option != "evolution":
        print("Error: --structure_path is only allowed with --phase_option evolution.")
        return
    mixed_gambles_gain_loss_only = bool(getattr(args, "filter_mixed_gambles", False))

    use_evolution_stages = (
        args.evolution_stages
        and args.dataset == "choice13k"
        and args.fitness_metric == "loglik"
    )
    evolution_stage1_prompt_path = str(Path(args.evolution_stage1_prompt).expanduser().resolve())
    evolution_stage2_prompt_path = str(Path(args.evolution_stage2_prompt).expanduser().resolve())
    if use_evolution_stages:
        stage_cutoff = _evolution_stage_cutoff(args.n_iterations)
        print(
            f"[Config] 2-stage evolution prompts: stage1 iterations 1-{stage_cutoff}, "
            f"stage2 iterations {stage_cutoff + 1}-{args.n_iterations}"
        )
        print(f"  stage1: {evolution_stage1_prompt_path}")
        print(f"  stage2: {evolution_stage2_prompt_path}")

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
            # Include dataset in run name
            dataset_prefix = args.dataset if args.dataset else "choice13k"
            run_tag = "te_dr" if args.dataset in _PARTICIPANT_DATASETS else "non_strict"
            run_name = f"{dataset_prefix}_{run_tag}_{timestamp}"
            if args.dataset in _PARTICIPANT_DATASETS:
                if args.participant_scope == "single":
                    run_name = f"{run_name}_participant_{args.single_participant_id}"
                elif args.participant_scope == "range":
                    run_name = (
                        f"{run_name}_ordinals_{args.range_start_ordinal}_to_{args.range_end_ordinal}"
                    )
                elif args.participant_scope == "ordinals":
                    tag = "_".join(str(x) for x in (args.ordinals or []))
                    if len(tag) > 120:
                        tag = tag[:120] + "_etc"
                    run_name = f"{run_name}_ordinals_{tag}"
                else:
                    run_name = f"{run_name}_all_valid"
            else:
                if args.agent_id is not None:
                    run_name = f"{run_name}_agent_{args.agent_id}"
                else:
                    run_name = f"{run_name}_agents_0to{args.num_agents_to_sample-1}"
            wandb.init(
                project="ROTE_evo",
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
    if args.dataset == "choice13k":
        print(
            f"Choice13k split settings: split_mode={args.split_mode}, "
            f"split_ratio={args.split_ratio:.3f}, split_seed={args.split_seed}"
        )
    if args.dataset == "cpc18":
        print(
            f"CPC18: official_mse={args.cpc18_official_mse}, "
            f"split_ratio={args.split_ratio:.3f}, split_seed={args.split_seed}, "
            f"fitness_metric={args.fitness_metric}"
        )

    wandb_global_step = 0

    def _wandb_log_with_global_step(metrics: Dict[str, Any]) -> None:
        nonlocal wandb_global_step
        if wandb is None:
            return
        wandb.log(metrics, step=wandb_global_step)
        wandb_global_step += 1

    if wandb is not None and args.dataset in _PARTICIPANT_DATASETS:
        for pid in participants_to_process:
            wandb.define_metric(f"p{pid}_step")
            wandb.define_metric(f"p{pid}/*", step_metric=f"p{pid}_step")
    
    # Create base run directory and save seed program once
    base_run_dir = None
    if args.output_dir is None:
        # Auto-generated output: create base run directory (use same timestamp)
        mode = "te_aggregate" if args.dataset in _PARTICIPANT_DATASETS else "non_strict"
        if args.dataset == "gridworld":
            base_run_dir = f"generated_outputs/gridworld/{mode}/run_{timestamp}"
        elif args.dataset == "gridworld_ensemble":
            base_run_dir = f"generated_outputs/gridworld_ensemble/{mode}/run_{timestamp}"
        elif args.dataset == "cpc18":
            base_run_dir = f"generated_outputs/cpc18/{mode}/run_{timestamp}"
        elif args.dataset == "mixed_gambles":
            base_run_dir = f"generated_outputs/mixed_gambles/{mode}/run_{timestamp}"
        elif args.dataset == "choice13k":
            base_run_dir = f"generated_outputs/choice13k/te_dr/run_{timestamp}"
        else:
            base_run_dir = f"generated_outputs/{args.dataset}/{mode}/run_{timestamp}"
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

    # Determine seed program path
    if args.seed_path is None:
        if args.dataset == "gridworld" or args.dataset == "gridworld_ensemble":
            # Auto-detect template program for gridworld / gridworld_ensemble (per epoch or per agent)
            seed_program_path = None
        elif args.dataset == "cpc18":
            # Default for CPC18 Track II
            seed_program_path = "persona_code_example/cpc18/hard.py"
        elif args.dataset == "mixed_gambles":
            seed_program_path = "persona_code_example/hard_Qwen.py"
        else:
            # Default for choice13k
            seed_program_path = "persona_code_example/vanilla.py"
    else:
        seed_program_path = args.seed_path
    
    # Load and save seed program once in the experiment folder (if we have a single seed)
    if seed_program_path is not None:
        seed_code = load_seed_program(seed_program_path)
        (Path(base_run_dir) / "seed_program.py").write_text(seed_code)
        if args.participant_scope != "all":
            print(f"Seed program saved to: {Path(base_run_dir) / 'seed_program.py'}")

    if args.dataset == "choice13k" and args.split_mode == "across_participants":
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
        experiments = get_choice13k_experiments(
            n_participants=max_pid + 1,
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

        te_dr_prompt = _te_data_driven_choice13k_prompt_path(args.max_prompt_val)
        try:
            run_evolution(
                seed_program_path=seed_program_path,
                dataset="choice13k",
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
                wandb_log_fn=_wandb_log_with_global_step,
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
                max_prompt_train=args.max_prompt_train,
                max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                llm_max_tokens=args.llm_max_tokens,
                adaptation_mode=False,
                data_driven_mode=args.fitness_metric == "loglik",
                max_prompt_val=args.max_prompt_val,
                choice13k_loglik_prompt_path=te_dr_prompt if args.fitness_metric == "loglik" else None,
                use_evolution_stages=use_evolution_stages,
                evolution_stage1_prompt_path=evolution_stage1_prompt_path,
                evolution_stage2_prompt_path=evolution_stage2_prompt_path,
                hard_participant_train_loglik_threshold=args.hard_participant_train_loglik_threshold,
                hard_participant_warmup_iters=args.hard_participant_warmup_iters,
                early_stop=args.early_stop,
                debug_continue_after_early_stop=args.debug_continue_after_early_stop,
                local_dataset=args.local_dataset,
                rbu_lambda=args.uncertainty_lambda,
                use_rbu=False,
                participant_bir=0.0,
                participant_rbu=0.0,
                rbu_prompt_threshold=args.uncertainty_threshold,
            )
        finally:
            if wandb is not None:
                wandb.finish()
            return

    if args.dataset in _PARTICIPANT_DATASETS:
        if args.dataset != "choice13k":
            print(
                "Error: te_dr.py supports data-driven evolution for --dataset choice13k only. "
                "Use te_aggregate.py for cpc18 / mixed_gambles participant workflows."
            )
            if wandb is not None:
                wandb.finish()
            return
        if args.fitness_metric != "loglik":
            print("Error: te_dr.py requires --fitness_metric loglik.")
            if wandb is not None:
                wandb.finish()
            return
        if args.split_mode != "within_participant":
            print(
                "Error: te_dr.py requires --split_mode within_participant "
                "(train/val/test split by problem block)."
            )
            if wandb is not None:
                wandb.finish()
            return

        te_dr_prompt = _te_data_driven_choice13k_prompt_path(args.max_prompt_val)
        print("\n=== Data-driven single-phase evolution (Choice13k) ===")
        pid_list = list(participants_to_process)
        run_root = Path(base_run_dir)
        details_loglik_file = run_root / "participant_details_loglik.csv"
        summary_loglik_file = run_root / "summary_loglik.csv"
        participants_loglik_summary: List[Dict[str, Any]] = []
        try:
            for participant_id in tqdm(pid_list, desc="Participants"):
                participant_output_dir = os.path.join(base_run_dir, f"participant_{participant_id}")
                participant_summary = run_evolution(
                    seed_program_path=seed_program_path,
                    dataset="choice13k",
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
                    wandb_log_fn=_wandb_log_with_global_step,
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
                    max_prompt_train=args.max_prompt_train,
                    max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                    llm_max_tokens=args.llm_max_tokens,
                    adaptation_mode=False,
                    data_driven_mode=True,
                    max_prompt_val=args.max_prompt_val,
                    choice13k_loglik_prompt_path=te_dr_prompt,
                    use_evolution_stages=use_evolution_stages,
                    evolution_stage1_prompt_path=evolution_stage1_prompt_path,
                    evolution_stage2_prompt_path=evolution_stage2_prompt_path,
                    hard_participant_train_loglik_threshold=args.hard_participant_train_loglik_threshold,
                    hard_participant_warmup_iters=args.hard_participant_warmup_iters,
                    early_stop=args.early_stop,
                    debug_continue_after_early_stop=args.debug_continue_after_early_stop,
                    local_dataset=args.local_dataset,
                    rbu_lambda=args.uncertainty_lambda,
                    use_rbu=False,
                    participant_bir=0.0,
                    participant_rbu=0.0,
                    rbu_prompt_threshold=args.uncertainty_threshold,
                )
                if participant_summary is not None:
                    participants_loglik_summary.append(
                        {
                            "participant_id": participant_summary.get("participant_id"),
                            "train_loglik": participant_summary.get("train_loglik"),
                            "val_loglik": participant_summary.get("val_loglik"),
                            "test_loglik": participant_summary.get("test_loglik"),
                            "selection_score": participant_summary.get("selection_score"),
                        }
                    )
                    with open(details_loglik_file, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "participant_id",
                                "train_loglik",
                                "val_loglik",
                                "test_loglik",
                                "selection_score",
                            ],
                        )
                        writer.writeheader()
                        writer.writerows(_round_floats_for_csv_rows(participants_loglik_summary))
                    train_ll_vals = [
                        d["train_loglik"] for d in participants_loglik_summary if d["train_loglik"] is not None
                    ]
                    val_ll_vals = [
                        d["val_loglik"] for d in participants_loglik_summary if d.get("val_loglik") is not None
                    ]
                    test_ll_vals = [
                        d["test_loglik"] for d in participants_loglik_summary if d["test_loglik"] is not None
                    ]
                    with open(summary_loglik_file, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "num_of_participants",
                                "avg_train_loglik",
                                "avg_val_loglik",
                                "avg_test_loglik",
                            ],
                        )
                        writer.writeheader()
                        writer.writerow(
                            _round_floats_for_csv_row(
                                {
                                    "num_of_participants": len(participants_loglik_summary),
                                    "avg_train_loglik": float(np.mean(train_ll_vals)) if train_ll_vals else None,
                                    "avg_val_loglik": float(np.mean(val_ll_vals)) if val_ll_vals else None,
                                    "avg_test_loglik": float(np.mean(test_ll_vals)) if test_ll_vals else None,
                                }
                            )
                        )
                    print(
                        f"\nUpdated run-level loglik CSVs: {details_loglik_file} "
                        f"and {summary_loglik_file} ({len(participants_loglik_summary)} participant(s))."
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

        te_dr_prompt_all = _te_data_driven_choice13k_prompt_path(args.max_prompt_val)

        try:
            for participant_id in tqdm(participants_to_process, desc="Participants"):
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
                    wandb_log_fn=_wandb_log_with_global_step,
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
                    max_prompt_train=args.max_prompt_train,
                    max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                    llm_max_tokens=args.llm_max_tokens,
                    cpc18_official_mse=args.cpc18_official_mse,
                    adaptation_mode=False,
                    data_driven_mode=(args.dataset == "choice13k" and args.fitness_metric == "loglik"),
                    max_prompt_val=args.max_prompt_val,
                    choice13k_loglik_prompt_path=(
                        te_dr_prompt_all if args.dataset == "choice13k" and args.fitness_metric == "loglik" else None
                    ),
                    use_evolution_stages=use_evolution_stages,
                    evolution_stage1_prompt_path=evolution_stage1_prompt_path,
                    evolution_stage2_prompt_path=evolution_stage2_prompt_path,
                    hard_participant_train_loglik_threshold=args.hard_participant_train_loglik_threshold,
                    hard_participant_warmup_iters=args.hard_participant_warmup_iters,
                    early_stop=args.early_stop,
                    debug_continue_after_early_stop=args.debug_continue_after_early_stop,
                    local_dataset=args.local_dataset,
                    rbu_lambda=args.uncertainty_lambda,
                    use_rbu=False,
                    participant_bir=0.0,
                    participant_rbu=0.0,
                    rbu_prompt_threshold=args.uncertainty_threshold,
                )
                runtime_sec = (datetime.now() - participant_start).total_seconds()

                participant_details.append({
                    "participant_id": participant_id,
                    "train_fitness": participant_summary.get("train_fitness"),
                    "test_fitness": participant_summary.get("test_fitness"),
                    "total_runtime": runtime_sec,
                    "seed_program_train_fitness": participant_summary.get("seed_program_train_fitness"),
                    "seed_program_test_fitness": participant_summary.get("seed_program_test_fitness"),
                })
                participant_details_loglik.append({
                    "participant_id": participant_id,
                    "train_loglik": participant_summary.get("train_loglik"),
                    "val_loglik": participant_summary.get("val_loglik"),
                    "test_loglik": participant_summary.get("test_loglik"),
                    "selection_score": participant_summary.get("selection_score"),
                })

                with open(details_file, "w", newline="") as f:
                    fieldnames = [
                        "participant_id",
                        "train_fitness",
                        "test_fitness",
                        "total_runtime",
                        "seed_program_train_fitness",
                        "seed_program_test_fitness",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(_round_floats_for_csv_rows(participant_details))

                avg_train_fitness = float(np.mean([d["train_fitness"] for d in participant_details]))
                avg_test_fitness = float(np.mean([d["test_fitness"] for d in participant_details]))
                with open(summary_file, "w", newline="") as f:
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

                with open(details_loglik_file, "w", newline="") as f:
                    fieldnames = ["participant_id", "train_loglik", "val_loglik", "test_loglik", "selection_score"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(_round_floats_for_csv_rows(participant_details_loglik))

                train_loglik_values = [
                    d["train_loglik"] for d in participant_details_loglik if d["train_loglik"] is not None
                ]
                val_loglik_values = [
                    d["val_loglik"] for d in participant_details_loglik if d.get("val_loglik") is not None
                ]
                test_loglik_values = [
                    d["test_loglik"] for d in participant_details_loglik if d["test_loglik"] is not None
                ]
                avg_train_loglik = float(np.mean(train_loglik_values)) if train_loglik_values else None
                avg_val_loglik = float(np.mean(val_loglik_values)) if val_loglik_values else None
                avg_test_loglik = float(np.mean(test_loglik_values)) if test_loglik_values else None
                with open(summary_loglik_file, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "num_of_participants",
                            "avg_train_loglik",
                            "avg_val_loglik",
                            "avg_test_loglik",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        _round_floats_for_csv_row(
                            {
                                "num_of_participants": len(participant_details_loglik),
                                "avg_train_loglik": avg_train_loglik,
                                "avg_val_loglik": avg_val_loglik,
                                "avg_test_loglik": avg_test_loglik,
                            }
                        )
                    )
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
                        wandb_log_fn=_wandb_log_with_global_step,
                        n_eval_seeds=args.n_eval_seeds,
                        sample_size=args.sample_size,
                        top_k=getattr(args, 'top_k', 0),
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
                        wandb_log_fn=_wandb_log_with_global_step,
                        n_eval_seeds=args.n_eval_seeds,
                        sample_size=args.sample_size,
                        sample_parents=args.sample_parents,
                        elite_pool_size=args.elite_pool_size,
                        filter_mixed_gambles=mixed_gambles_gain_loss_only,
                        fitness_metric=args.fitness_metric,
                        split_ratio=args.split_ratio,
                        split_seed=args.split_seed,
                        max_prompt_train=args.max_prompt_train,
                        max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                        llm_max_tokens=args.llm_max_tokens,
                        hard_participant_train_loglik_threshold=args.hard_participant_train_loglik_threshold,
                        hard_participant_warmup_iters=args.hard_participant_warmup_iters,
                        early_stop=args.early_stop,
                        debug_continue_after_early_stop=args.debug_continue_after_early_stop,
                        local_dataset=args.local_dataset,
                        rbu_lambda=args.uncertainty_lambda,
                        use_rbu=False,
                        participant_bir=0.0,
                        participant_rbu=0.0,
                        rbu_prompt_threshold=args.uncertainty_threshold,
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
        try:
            for participant_id in tqdm(participants_to_process, desc="Participants"):
                print(f"\n{'='*80}")
                print(f"Processing participant {participant_id}")
                print(f"{'='*80}")
                # If base_run_dir is set, construct participant-specific output_dir
                if base_run_dir is not None:
                    participant_output_dir = os.path.join(base_run_dir, f"participant_{participant_id}")
                else:
                    participant_output_dir = args.output_dir
                
                # If summary_file is None (auto-generated single participant), determine it now
                if summary_file is None and participant_output_dir is not None:
                    output_path = Path(participant_output_dir)
                    if output_path.name.startswith("participant_"):
                        summary_file = output_path.parent / "participants_summary.csv"
                        summary_loglik_file = output_path.parent / "summary_loglik.csv"
                        details_loglik_file = output_path.parent / "participant_details_loglik.csv"
                    else:
                        summary_file = output_path / "participants_summary.csv"
                        summary_loglik_file = output_path / "summary_loglik.csv"
                        details_loglik_file = output_path / "participant_details_loglik.csv"
                
                # Determine seed program path
                if args.seed_path is None:
                    if args.dataset == "gridworld" or args.dataset == "gridworld_ensemble":
                        num_blocks = getattr(args, 'num_blocks', None)
                        num_walls = getattr(args, 'num_walls', None)
                        agent_id = getattr(args, 'agent_id', None)
                        if num_blocks is not None and num_walls is not None and agent_id is not None:
                            detected_seed_path = find_template_program_for_gridworld(num_blocks, num_walls, agent_id)
                            if detected_seed_path is None:
                                print(f"Error: Could not auto-detect template program for num_blocks={num_blocks}, num_walls={num_walls}, agent_id={agent_id}")
                                print(f"Expected location: persona_code_example/gridworld/num_blocks{num_blocks}_num_walls{num_walls}/")
                                continue
                            seed_program_path = detected_seed_path
                            print(f"Auto-detected seed program: {seed_program_path}")
                        else:
                            print("Error: For gridworld/gridworld_ensemble without --seed_path, must provide --num_blocks, --num_walls, and --agent_id")
                            continue
                    elif args.dataset == "cpc18":
                        seed_program_path = "persona_code_example/cpc18/hard.py"
                    elif args.dataset == "mixed_gambles":
                        seed_program_path = "persona_code_example/hard_Qwen.py"
                    else:
                        seed_program_path = "persona_code_example/vanilla.py"
                else:
                    seed_program_path = args.seed_path
                
                if args.dataset == "gridworld_ensemble":
                    num_blocks = getattr(args, 'num_blocks', None)
                    num_walls = getattr(args, 'num_walls', None)
                    agent_id = getattr(args, 'agent_id', participant_id)
                    if num_blocks is None or num_walls is None:
                        print("Error: For gridworld_ensemble must provide --num_blocks and --num_walls")
                        continue
                    participant_summary = run_evolution_gridworld_ensemble(
                        seed_program_path=seed_program_path,
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
                        top_k=getattr(args, 'top_k', 0),
                    )
                else:
                    participant_summary = run_evolution(
                        seed_program_path=seed_program_path,
                        dataset=args.dataset,
                        participant_id=participant_id,
                        data_path=args.data_path,
                        num_blocks=getattr(args, 'num_blocks', None),
                        num_walls=getattr(args, 'num_walls', None),
                        agent_id=getattr(args, 'agent_id', None),
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
                        max_prompt_train=args.max_prompt_train,
                        max_prompt_trials_per_problem=args.max_prompt_trials_per_problem,
                        llm_max_tokens=args.llm_max_tokens,
                        cpc18_official_mse=args.cpc18_official_mse,
                        hard_participant_train_loglik_threshold=args.hard_participant_train_loglik_threshold,
                        hard_participant_warmup_iters=args.hard_participant_warmup_iters,
                        early_stop=args.early_stop,
                        debug_continue_after_early_stop=args.debug_continue_after_early_stop,
                        local_dataset=args.local_dataset,
                        rbu_lambda=args.uncertainty_lambda,
                        use_rbu=False,
                        participant_bir=0.0,
                        participant_rbu=0.0,
                        rbu_prompt_threshold=args.uncertainty_threshold,
                    )
                
                # Update participants summary after each participant completes
                if participant_summary is not None and summary_file is not None:
                    participants_summary.append(participant_summary)
                    # Write CSV file
                    with open(summary_file, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=list(participant_summary.keys()))
                        writer.writeheader()
                        writer.writerows(_round_floats_for_csv_rows(participants_summary))
                    print(f"\nParticipants summary updated: {summary_file}")

                    if args.participant_scope in ("range", "ordinals"):
                        participants_loglik_summary.append({
                            "participant_id": participant_summary.get("participant_id"),
                            "train_loglik": participant_summary.get("train_loglik"),
                            "val_loglik": participant_summary.get("val_loglik"),
                            "test_loglik": participant_summary.get("test_loglik"),
                            "selection_score": participant_summary.get("selection_score"),
                        })
                        if details_loglik_file is not None:
                            with open(details_loglik_file, "w", newline="") as f:
                                writer = csv.DictWriter(
                                    f,
                                    fieldnames=[
                                        "participant_id",
                                        "train_loglik",
                                        "val_loglik",
                                        "test_loglik",
                                        "selection_score",
                                    ],
                                )
                                writer.writeheader()
                                writer.writerows(_round_floats_for_csv_rows(participants_loglik_summary))
                        if summary_loglik_file is not None:
                            train_ll_vals = [
                                d["train_loglik"] for d in participants_loglik_summary if d["train_loglik"] is not None
                            ]
                            val_ll_vals = [
                                d["val_loglik"] for d in participants_loglik_summary if d.get("val_loglik") is not None
                            ]
                            test_ll_vals = [
                                d["test_loglik"] for d in participants_loglik_summary if d["test_loglik"] is not None
                            ]
                            with open(summary_loglik_file, "w", newline="") as f:
                                writer = csv.DictWriter(
                                    f,
                                    fieldnames=[
                                        "num_of_participants",
                                        "avg_train_loglik",
                                        "avg_val_loglik",
                                        "avg_test_loglik",
                                    ],
                                )
                                writer.writeheader()
                                writer.writerow(
                                    _round_floats_for_csv_row(
                                        {
                                            "num_of_participants": len(participants_loglik_summary),
                                            "avg_train_loglik": float(np.mean(train_ll_vals)) if train_ll_vals else None,
                                            "avg_val_loglik": float(np.mean(val_ll_vals)) if val_ll_vals else None,
                                            "avg_test_loglik": float(np.mean(test_ll_vals)) if test_ll_vals else None,
                                        }
                                    )
                                )
        finally:
            if wandb is not None:
                wandb.finish()


if __name__ == "__main__":
    main()
