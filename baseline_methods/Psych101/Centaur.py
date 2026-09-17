#!/usr/bin/env python3
"""
Centaur (Llama 3.1 + Psych-101-style prompts) baseline for TEH loglik datasets.

Evaluates held-out test trials with explicit trial log-likelihood (PICS-fair Option P):
  - Bernoulli: P(action=1) from softmax over <<key0>> vs <<key1>> suffix logprobs
  - Categorical (K-way): softmax over <<display_key_i>> suffixes; LL = log p[action]

Supports Psych-101 binaries, Kool/Schulz, and external Bergert/Guan/Steyvers.
Sparse / limited-data protocol matches TEH/MLE (caps train+val; test untouched).
SA40 / sequential tasks: prefixes are built on the concatenated train+val+test
timeline; only test indices are scored. Do not pass a test-only list when
history is longer than the test index (Python negative wrap leaks later test
problems).

Model loading / suffix scoring follows reference_repos/Llama-3.1-Centaur-70B/test_adapter.py
(Unsloth FastLanguageModel, teacher-forcing loss on suffix tokens only — not trainer.evaluate()).

Example:
  python baseline_methods/Psych101/Centaur.py --dataset 3frey2017cct --psych_dataset_split train \\
    --participant_scope single --single_participant_id 0 --fitness_metric loglik \\
    --smoke_prompt_only

Requires GPU + conda env with torch, unsloth, transformers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PSYCH101_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_PSYCH101_DIR) not in sys.path:
    sys.path.insert(0, str(_PSYCH101_DIR))

from centaur_prompts import (  # noqa: E402
    centaur_display_keys,
    try_build_extended_centaur_prefix,
)
from data_modules.mixed_gambles import DEFAULT_CSV_PATH  # noqa: E402
from data_modules.psych101_binary import (  # noqa: E402
    DEFAULT_PSYCH_DATASET_SPLIT,
    get_psych101_binary_experiment,
    is_psych101_dataset,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
)
from data_modules.external import (  # noqa: E402
    EXTERNAL_DATASETS,
    external_dataset_task_description,
    is_external_dataset,
)
from utils.teh.limited_data_protocol import (  # noqa: E402
    LIMITED_DATA_MANIFEST_CSV_FILENAME,
    LIMITED_DATA_MANIFEST_JSONL_FILENAME,
    add_limited_data_cli_arguments,
    append_limited_data_manifest_jsonl,
    load_participant_limited_splits,
    resolve_limited_data_budget,
    rewrite_limited_data_csv_from_jsonl,
    should_persist_limited_data_manifest,
)
from utils.teh.participant_ids import load_valid_participant_ids  # noqa: E402
from utils.teh.teh_datasets import (  # noqa: E402
    IMPLEMENTED_PSYCH101_ALIASES,
    is_binary_loglik_dataset,
)

# Five new datasets + all implemented Psych-101 aliases.
CENTAUR_FOCUS_DATASETS = frozenset(
    {
        "14kool2016when",
        "13schulz2020finding",
        "bergert_nosofsky_2007",
        "guan_2020_stopping",
        "steyvers_2009_bandit",
    }
)
PSYCH101_CENTAUR_DATASETS = sorted(
    set(IMPLEMENTED_PSYCH101_ALIASES) | set(EXTERNAL_DATASETS)
)


def _effective_psych_dataset_split(psych_dataset_split: str) -> str:
    return normalize_psych_dataset_split(psych_dataset_split)


def load_valid_participant_ids_from_json(
    dataset: str,
    repo_root: Path,
    *,
    split_ratio: float = 0.6,
    split_seed: int = 0,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    auto_prepare: bool = True,
) -> List[int]:
    if not is_binary_loglik_dataset(dataset):
        raise ValueError(f"Unsupported dataset {dataset!r}")
    return load_valid_participant_ids(
        dataset,
        repo_root,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=_effective_psych_dataset_split(psych_dataset_split),
        local_dataset=local_dataset,
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
    split_ratio: float = 0.6,
    split_seed: int = 0,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
) -> List[int]:
    valid = load_valid_participant_ids_from_json(
        dataset,
        repo_root,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    if participant_scope == "single":
        if single_participant_id not in valid:
            raise ValueError(
                f"--single_participant_id={single_participant_id} not in valid list ({len(valid)} ids)."
            )
        return [single_participant_id]
    if participant_scope == "range":
        if range_start_ordinal is None or range_end_ordinal is None:
            raise ValueError("range scope requires --range_start_ordinal and --range_end_ordinal.")
        if (
            range_start_ordinal < 0
            or range_end_ordinal >= len(valid)
            or range_start_ordinal > range_end_ordinal
        ):
            raise ValueError(
                f"Invalid ordinal range [{range_start_ordinal}, {range_end_ordinal}] "
                f"for list length {len(valid)}."
            )
        return valid[range_start_ordinal : range_end_ordinal + 1]
    if participant_scope == "ordinals":
        if not participant_ordinals:
            raise ValueError("--participant_scope ordinals requires --ordinals.")
        out: List[int] = []
        seen: set[int] = set()
        for o in participant_ordinals:
            oi = int(o)
            if oi < 0 or oi >= len(valid):
                raise ValueError(f"Ordinal {oi} out of range 0..{len(valid) - 1}.")
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


def centaur_output_base_dir(
    dataset: str,
    timestamp: str,
    *,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
) -> str:
    alias = normalize_psych101_dataset_alias(dataset)
    if is_external_dataset(alias):
        return f"generated_outputs/external/{alias}/centaur/run_{timestamp}"
    split = normalize_psych_dataset_split(psych_dataset_split)
    return f"generated_outputs/psych101_{split}/centaur/{alias}/run_{timestamp}"


def check_centaur_runtime_deps(*, load_unsloth_api: bool = True) -> Dict[str, Any]:
    """Import the Centaur scoring stack without loading the 70B weights or requiring CUDA."""
    report: Dict[str, Any] = {"ok": True, "packages": [], "cuda_available": None}
    specs = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("unsloth", "unsloth"),
        ("bitsandbytes", "bitsandbytes"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
    ]
    for label, mod in specs:
        entry: Dict[str, Any] = {"package": label, "ok": False, "version": None, "error": None}
        try:
            m = __import__(mod)
            entry["ok"] = True
            entry["version"] = getattr(m, "__version__", None)
        except Exception as e:
            report["ok"] = False
            entry["error"] = f"{type(e).__name__}: {e}"
        report["packages"].append(entry)
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        report["cuda_available"] = False
    if load_unsloth_api:
        api: Dict[str, Any] = {
            "package": "unsloth.FastLanguageModel",
            "ok": False,
            "version": None,
            "error": None,
        }
        try:
            from unsloth import FastLanguageModel  # noqa: F401

            api["ok"] = True
        except Exception as e:
            report["ok"] = False
            api["error"] = f"{type(e).__name__}: {e}"
        report["packages"].append(api)
    return report


def _centaur_history_span(
    trials: List[Dict[str, Any]], trial_index: int
) -> Tuple[int, List[Dict[str, Any]]]:
    """Map ``history`` onto earlier rows in ``trials``. Reject negative wrap."""
    if trial_index < 0 or trial_index >= len(trials):
        raise ValueError(
            f"Centaur prefix trial_index={trial_index} out of range 0..{len(trials) - 1}"
        )
    hist = list(trials[trial_index].get("history") or [])
    start = trial_index - len(hist)
    if start < 0:
        raise ValueError(
            f"Centaur prefix: history_len={len(hist)} exceeds trial_index={trial_index} "
            f"in a {len(trials)}-trial list. Score test trials on the concatenated "
            f"train+val+test timeline so history maps onto earlier rows "
            f"(Python negative wrap would leak later test problems)."
        )
    return start, hist


def _centaur_prompt_timeline(
    train_trials: Sequence[Dict[str, Any]],
    val_trials: Sequence[Dict[str, Any]],
    test_trials: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Full split timeline for prefixes; indices to score (test only)."""
    prompt = list(train_trials) + list(val_trials) + list(test_trials)
    test_start = len(train_trials) + len(val_trials)
    score_indices = list(range(test_start, test_start + len(test_trials)))
    return prompt, score_indices


def _task_instruction_for_participant(
    dataset: str,
    participant_id: int,
    *,
    psych_dataset_split: str,
    local_dataset: Optional[str],
) -> str:
    alias = normalize_psych101_dataset_alias(dataset)
    if is_external_dataset(alias):
        return external_dataset_task_description(alias)
    if is_psych101_dataset(alias):
        exp = get_psych101_binary_experiment(
            alias,
            int(participant_id),
            split=psych_dataset_split,
            local_dataset=local_dataset,
        )
        return str(exp.instruction or "")
    return ""


# ----- Psych-101 transcript-style prompt construction -----

GAMBLE_TASK_INTRO = (
    "You will encounter a series of gambling problems where you have to select between two options.\n"
    "You can select an option by pressing the corresponding key.\n"
    "For some problems, you are told the points you received and missed out on after each selection, "
    "while for others this information is suppressed.\n"
    "In cases where the probabilities are unknown, they sum up to one and remain constant within a problem.\n"
)


def _action_key(keys: List[str], action: int) -> str:
    if 0 <= action < len(keys):
        return str(keys[action])
    return "?"


def _schema_b_subtype(problem: Dict[str, Any]) -> str:
    if "memory_set_letters" in problem and "probe_letter" in problem:
        return "memory_probe"
    if "stimulus_features" in problem and (
        "correct_category" in problem or problem.get("task") == "category_learning"
    ):
        return "category_learning"
    if "tree_features" in problem:
        return "tree"
    if "cards" in problem or problem.get("features", {}).get("task") == "weather_prediction":
        return "weather"
    if "ratings_A" in problem or "option_A_features" in problem:
        return "product"
    return "binary"


def _format_gamble_option_line(letter: str, rewards: List[float], probs: Optional[List[float]]) -> str:
    if probs is None:
        parts = [f"either {float(r)} points with unknown chance" for r in rewards]
        body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", or " + parts[-1]
        return f"Option {letter} delivers {body}."
    chunks = [
        f"{float(rewards[i])} points with {float(probs[i]) * 100:.1f}% chance"
        for i in range(len(rewards))
    ]
    inner = chunks[0] if len(chunks) == 1 else ", ".join(chunks[:-1]) + ", or " + chunks[-1]
    return f"Option {letter} delivers {inner}."


def _format_gamble_problem(problem: Dict[str, Any]) -> str:
    keys = problem["option_keys"]
    ga = problem.get("gamble_A", {})
    gb = problem.get("gamble_B", {})
    return (
        f"{_format_gamble_option_line(keys[0], list(ga.get('rewards', [])), ga.get('probs'))}\n"
        f"{_format_gamble_option_line(keys[1], list(gb.get('rewards', [])), gb.get('probs'))}"
    )


def _gamble_history_line(past_problem: Dict[str, Any], h: Dict[str, Any]) -> str:
    keys = past_problem["option_keys"]
    letter = _action_key(keys, int(h["action"]))
    line = f"You press <<{letter}>>."
    if past_problem.get("has_feedback") and h.get("feedback") is not None:
        line += f" You receive {float(h['feedback'])} points by selecting this option."
    return line


def _cct_round_header(problem: Dict[str, Any]) -> str:
    return (
        f"Round {problem['round_id']}:\n"
        f"You will be awarded {problem['gain_amount']} points for turning over a gain card.\n"
        f"You will lose {problem['loss_amount']} points for turning over a loss card.\n"
        f"There are {problem['n_loss_cards']} loss cards in this round."
    )


def _cct_history_line(keys: List[str], h: Dict[str, Any]) -> str:
    act = int(h["action"])
    key = _action_key(keys, act)
    if act == 1:
        score = h.get("current_score", 0)
        return (
            f"You press <<{key}>> and stop the round and claim your payout. "
            f"Your final score for this round is {score}."
        )
    score = h.get("current_score", 0)
    return f"You press <<{key}>> and turn over a gain card. Your current score is {score}."


def _weather_trial_lines(problem: Dict[str, Any]) -> str:
    cards = problem.get("cards", [])
    cards_str = ", ".join(f"card {c}" for c in cards)
    return f"You are seeing the following: {cards_str}."


def _weather_history_line(past_problem: Dict[str, Any], h: Dict[str, Any]) -> str:
    keys = past_problem["option_keys"]
    letter = _action_key(keys, int(h["action"]))
    lines = [_weather_trial_lines(past_problem), f"You press <<{letter}>>."]
    if h.get("was_correct") is not None:
        wo = h.get("weather_outcome", past_problem.get("weather_outcome", "fine"))
        verdict = "correct" if h.get("was_correct") else "wrong"
        lines.append(f"You are {verdict}, the weather is {wo}.")
    return "\n".join(lines)


def _product_problem_lines(problem: Dict[str, Any]) -> str:
    keys = problem["option_keys"]
    ra = problem.get("ratings_A", [])
    rb = problem.get("ratings_B", [])
    return (
        f"Product {keys[0]} ratings: {ra}.\n"
        f"Product {keys[1]} ratings: {rb}."
    )


def _product_history_line(past_problem: Dict[str, Any], h: Dict[str, Any]) -> str:
    keys = past_problem["option_keys"]
    letter = _action_key(keys, int(h["action"]))
    return f"{_product_problem_lines(past_problem)}\nYou press <<{letter}>>."


def _tree_problem_line(problem: Dict[str, Any]) -> str:
    tf = problem.get("tree_features", {})
    leaf = tf.get("leafiness", 0)
    branch = tf.get("branchiness", 0)
    garden = problem.get("garden", "North")
    return (
        f"You get a tree with level {leaf} of leafiness and level {branch} of branchiness "
        f"in the {garden} garden."
    )


def _tree_history_line(past_problem: Dict[str, Any], h: Dict[str, Any]) -> str:
    keys = past_problem["option_keys"]
    letter = _action_key(keys, int(h["action"]))
    line = f"{_tree_problem_line(past_problem)}\nYou press <<{letter}>>"
    if h.get("feedback") is not None:
        line += f" and get {int(float(h['feedback']))} points"
    line += "."
    return line


def _bandit_game_header(problem: Dict[str, Any]) -> str:
    keys = problem.get("machine_options") or problem.get("option_keys", [])
    n_trials = problem.get("n_trials_game", "?")
    gid = problem.get("game_id", "?")
    if len(keys) >= 2:
        return (
            f"Game {gid}. There are {n_trials} trials labeled {keys[0]} and {keys[1]}."
        )
    return f"Game {gid}. There are {n_trials} trials."


def _bandit_history_line(h: Dict[str, Any]) -> str:
    key = h.get("machine_options", h.get("option_keys", ["?", "?"]))
    if isinstance(key, list):
        letter = _action_key(key, int(h["action"]))
    else:
        letter = "?"
    payoff = h.get("payoff", h.get("feedback", 0))
    if h.get("phase") == "instructed":
        return f"You are instructed to press {letter} and get {int(float(payoff))} points."
    return f"You press <<{letter}>> and get {int(float(payoff))} points."


def _build_gamble_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    cur = trials[trial_index]["problem"]
    start, hist = _centaur_history_span(trials, trial_index)
    parts: List[str] = []
    intro = instruction.strip() if instruction else GAMBLE_TASK_INTRO.rstrip()
    parts.append(intro)
    for j, h in enumerate(hist):
        parts.append(_gamble_history_line(trials[start + j]["problem"], h))
    parts.append(_format_gamble_problem(cur))
    parts.append("You press ")
    return "\n\n".join(parts)


def _build_cct_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    cur = trials[trial_index]["problem"]
    keys = cur["option_keys"]
    start, hist = _centaur_history_span(trials, trial_index)
    parts: List[str] = []
    if instruction.strip():
        parts.append(instruction.strip()[:2000])
    parts.append(_cct_round_header(cur))
    for j, h in enumerate(hist):
        parts.append(_cct_history_line(keys, h))
    parts.append("You press ")
    return "\n\n".join(parts)


def _balloon_problem_line(problem: Dict[str, Any]) -> str:
    balloon_id = problem.get("balloon_id", "?")
    pump_n = problem.get("pump_count_before", 0)
    acc = problem.get("accumulated_points_before", 0)
    return (
        f"Balloon {balloon_id}: You have pumped {pump_n} times and accumulated {acc} points. "
        f"You can pump once more or stop and cash out."
    )


def _balloon_history_line(past_problem: Dict[str, Any], h: Dict[str, Any]) -> str:
    keys = past_problem.get("option_keys", [])
    letter = _action_key(keys, int(h["action"]))
    line = f"You press <<{letter}>>."
    marker = str(h.get("outcome_marker", "ongoing"))
    if marker == "cashout":
        fb = h.get("feedback", 0)
        line += f" You stop inflating the balloon and get {int(float(fb))} points."
    elif marker == "explode" or h.get("exploded"):
        line += " The balloon was inflated too much and explodes."
    return line


def _build_schema_b_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    cur = trials[trial_index]["problem"]
    subtype = _schema_b_subtype(cur)
    start, hist = _centaur_history_span(trials, trial_index)
    parts: List[str] = []
    if instruction.strip():
        parts.append(instruction.strip()[:2000])
    for j, h in enumerate(hist):
        pp = trials[start + j]["problem"]
        if subtype == "weather":
            parts.append(_weather_history_line(pp, h))
        elif subtype == "tree":
            parts.append(_tree_history_line(pp, h))
        elif subtype == "memory_probe":
            ms = pp.get("memory_set_letters", [])
            probe = pp.get("probe_letter", "?")
            letter = _action_key(pp.get("option_keys", []), int(h["action"]))
            parts.append(
                f"You are shown the letters {ms}. You see the letter {probe}. You press <<{letter}>>."
            )
        elif subtype == "category_learning":
            sf = pp.get("stimulus_features", {})
            size = sf.get("size", "?")
            color = sf.get("color", "?")
            shape = sf.get("shape", "?")
            letter = _action_key(pp.get("option_keys", []), int(h["action"]))
            corr = h.get("correct_category", pp.get("correct_category", "?"))
            parts.append(
                f"You see a {size} {color} {shape}. You press <<{letter}>>. The correct category is {corr}."
            )
        else:
            parts.append(_product_history_line(pp, h))
    if subtype == "weather":
        parts.append(_weather_trial_lines(cur))
    elif subtype == "tree":
        parts.append(_tree_problem_line(cur))
    elif subtype == "memory_probe":
        parts.append(
            f"You are shown the letters {cur.get('memory_set_letters', [])}. "
            f"You see the letter {cur.get('probe_letter', '?')}."
        )
    elif subtype == "category_learning":
        sf = cur.get("stimulus_features", {})
        parts.append(
            f"You see a {sf.get('size', '?')} {sf.get('color', '?')} {sf.get('shape', '?')}."
        )
    else:
        parts.append(_product_problem_lines(cur))
    parts.append("You press ")
    return "\n\n".join(parts)


def _build_bandit_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    cur = trials[trial_index]["problem"]
    start, hist = _centaur_history_span(trials, trial_index)
    L = len(hist)
    parts: List[str] = []
    if instruction.strip():
        parts.append(instruction.strip()[:2000])
    if (
        start == 0
        or L == 0
        or trials[start - 1]["problem"].get("game_id") != cur.get("game_id")
    ):
        parts.append(_bandit_game_header(cur))
    for j, h in enumerate(hist):
        parts.append(_bandit_history_line(h))
    parts.append("You press ")
    return "\n\n".join(parts)


def _build_balloon_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    cur = trials[trial_index]["problem"]
    start, hist = _centaur_history_span(trials, trial_index)
    L = len(hist)
    parts: List[str] = []
    if instruction.strip():
        parts.append(instruction.strip()[:2000])
    if (
        start == 0
        or L == 0
        or trials[start - 1]["problem"].get("balloon_id") != cur.get("balloon_id")
    ):
        parts.append(_balloon_problem_line(cur))
    for j, h in enumerate(hist):
        parts.append(_balloon_history_line(trials[start + j]["problem"], h))
    parts.append("You press ")
    return "\n\n".join(parts)


def _build_generic_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    cur = trials[trial_index]["problem"]
    keys = cur.get("option_keys", [])
    start, hist = _centaur_history_span(trials, trial_index)
    parts: List[str] = []
    if instruction.strip():
        parts.append(instruction.strip()[:2000])
    for j, h in enumerate(hist):
        letter = _action_key(trials[start + j]["problem"].get("option_keys", keys), int(h["action"]))
        parts.append(f"You press <<{letter}>>.")
    parts.append("You press ")
    return "\n\n".join(parts)


def build_centaur_prompt_prefix_indexed(
    trials: List[Dict[str, Any]],
    trial_index: int,
    *,
    instruction: str = "",
) -> str:
    """Build Psych-101-style prefix for trials[trial_index] (ends with 'You press ')."""
    extended = try_build_extended_centaur_prefix(
        trials, trial_index, instruction=instruction
    )
    if extended is not None:
        return extended
    schema = str(trials[trial_index]["problem"].get("schema_type", "?"))
    if schema == "A" and (
        "gamble_A" in trials[trial_index]["problem"]
        or "gamble_B" in trials[trial_index]["problem"]
    ):
        return _build_gamble_prefix(trials, trial_index, instruction=instruction)
    if schema == "D":
        if "balloon_id" in trials[trial_index]["problem"]:
            return _build_balloon_prefix(trials, trial_index, instruction=instruction)
        return _build_cct_prefix(trials, trial_index, instruction=instruction)
    if schema == "B":
        return _build_schema_b_prefix(trials, trial_index, instruction=instruction)
    if schema == "C":
        return _build_bandit_prefix(trials, trial_index, instruction=instruction)
    return _build_generic_prefix(trials, trial_index, instruction=instruction)


# ----- Centaur model (reference: test_adapter.py loading; suffix logprob scoring) -----


class CentaurChooser:
    """Loads Centaur once; scores action probs via normalized <<key>> suffix logprobs."""

    def __init__(
        self,
        model_name: str,
        *,
        max_seq_length: int = 32768,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.task_instruction: str = ""
        self._model = None
        self._tokenizer = None
        self.last_prob_debug: Dict[str, Any] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from unsloth import FastLanguageModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to load Centaur (Unsloth).")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer

    def _suffix_logprob_detailed(self, prefix: str, suffix: str) -> Tuple[float, Optional[str]]:
        import torch

        self._ensure_loaded()
        model = self._model
        tokenizer = self._tokenizer
        device = next(model.parameters()).device

        text = prefix + suffix
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        pref = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        plen = int(pref["input_ids"].shape[1])
        if plen >= input_ids.shape[1]:
            return float("-inf"), "empty_suffix_after_tokenization"

        labels = input_ids.clone()
        labels[:, :plen] = -100

        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            if loss is None:
                return float("-inf"), "loss_none"
            if not torch.isfinite(loss):
                return float("nan"), f"loss_non_finite:{float(loss.item())}"
            n = int((labels != -100).sum().item())
            if n <= 0:
                return float("-inf"), "no_suffix_tokens_after_masking"
            score = float(-loss.item() * n)
            if not math.isfinite(score):
                return score, f"logprob_non_finite:{score}"
            return score, None

    def action_probs_from_suffixes(
        self, trials: List[Dict[str, Any]], trial_index: int
    ) -> List[float]:
        """Softmax over Centaur display-key suffix logprobs; length K = n_actions."""
        problem = trials[trial_index]["problem"]
        keys = centaur_display_keys(problem)
        if len(keys) < 2:
            raise ValueError(f"Expected >=2 display keys, got {keys!r}")

        prefix = build_centaur_prompt_prefix_indexed(
            trials, trial_index, instruction=self.task_instruction
        )
        lps: List[float] = []
        reasons: List[Optional[str]] = []
        for key in keys:
            lp, reason = self._suffix_logprob_detailed(prefix, f"<<{key}>>.")
            lps.append(lp)
            reasons.append(reason)
        m = max(lps)
        exps = [math.exp(lp - m) for lp in lps]
        denom = sum(exps)
        if denom <= 0 or not math.isfinite(denom):
            self.last_prob_debug = {
                "fallback_source": "invalid_denom",
                "lps": lps,
                "reasons": reasons,
                "keys": keys,
            }
            return [1.0 / len(keys)] * len(keys)
        probs = [e / denom for e in exps]
        self.last_prob_debug = {
            "fallback_source": None,
            "lps": lps,
            "keys": keys,
            "probs": probs,
        }
        return [float(min(max(p, 1e-9), 1.0 - 1e-9)) for p in probs]

    def prob_choose_second_option(self, trials: List[Dict[str, Any]], trial_index: int) -> float:
        """Bernoulli P(action=1); for K=2 this is the second display-key probability."""
        probs = self.action_probs_from_suffixes(trials, trial_index)
        if len(probs) != 2:
            raise ValueError(
                f"prob_choose_second_option requires K=2, got K={len(probs)} "
                f"(use action_probs_from_suffixes for categorical)"
            )
        return float(probs[1])


def evaluate_centaur_on_trials(
    chooser: CentaurChooser,
    trials: List[Dict[str, Any]],
    *,
    verbose: bool = False,
    n_seeds: int = 1,
    debug_prob: bool = False,
    debug_limit: int = 5,
    score_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if score_indices is None:
        indices = list(range(len(trials)))
    else:
        indices = [int(i) for i in score_indices]
        for i in indices:
            if i < 0 or i >= len(trials):
                raise ValueError(
                    f"Centaur score index {i} out of range 0..{len(trials) - 1}"
                )
    total = len(indices)
    seed_avg_logliks: List[float] = []
    seed_avg_accs: List[float] = []

    def _one_pass(seed_idx: int) -> Tuple[float, float, int]:
        loglik_acc = 0.0
        correct = 0
        errors = 0
        for pos, i in enumerate(indices):
            y = int(trials[i]["action"])
            try:
                probs = chooser.action_probs_from_suffixes(trials, i)
                if y < 0 or y >= len(probs):
                    raise ValueError(f"action {y} out of range for K={len(probs)}")
                p_obs = float(probs[y])
                pred = int(max(range(len(probs)), key=lambda a: probs[a]))
            except Exception as e:
                errors += 1
                if verbose and errors <= 3 and seed_idx == 0:
                    print(f"  Evaluation error trial {i}: {e}")
                keys = centaur_display_keys(trials[i]["problem"])
                k = max(2, len(keys))
                p_obs = 1.0 / k
                pred = 0
            p_obs = min(max(p_obs, 1e-9), 1.0 - 1e-9)
            loglik_acc += math.log(p_obs)
            correct += int(pred == y)
            if debug_prob and seed_idx == 0 and pos < debug_limit:
                print(f"  [debug] trial={i} y={y} p_obs={p_obs:.6f} dbg={chooser.last_prob_debug}")
        avg_ll = loglik_acc / total if total else 0.0
        acc = correct / total if total else 0.0
        return avg_ll, acc, errors

    for seed in range(n_seeds):
        avg_ll, acc, _ = _one_pass(seed)
        seed_avg_logliks.append(avg_ll)
        seed_avg_accs.append(acc)

    return {
        "accuracy": float(np.mean(seed_avg_accs)) if seed_avg_accs else 0.0,
        "avg_loglik": float(np.mean(seed_avg_logliks)) if seed_avg_logliks else float("-inf"),
        "total": total,
        "correct": int(round(float(np.mean(seed_avg_accs)) * total)) if total else 0,
    }


def collect_centaur_predictions(
    chooser: CentaurChooser,
    trials: List[Dict[str, Any]],
    *,
    participant_id: int,
    dataset: str,
    split_name: str,
    score_indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    if score_indices is None:
        indices = list(range(len(trials)))
    else:
        indices = [int(i) for i in score_indices]
    rows: List[Dict[str, Any]] = []
    for local_i, i in enumerate(indices):
        t = trials[i]
        y = int(t["action"])
        keys = centaur_display_keys(t["problem"])
        p_obs: Optional[float] = None
        pred: Optional[int] = None
        error = ""
        probs_json = ""
        try:
            probs = chooser.action_probs_from_suffixes(trials, i)
            probs_json = json.dumps([float(p) for p in probs])
            if 0 <= y < len(probs):
                p_obs = float(probs[y])
            pred = int(max(range(len(probs)), key=lambda a: probs[a]))
        except Exception as e:
            error = str(e)
        rows.append(
            {
                "participant_id": participant_id,
                "dataset": dataset,
                "split": split_name,
                "trial_index": local_i,
                "prompt_trial_index": i,
                "option_key_0": keys[0] if len(keys) > 0 else "",
                "option_key_1": keys[1] if len(keys) > 1 else "",
                "n_keys": len(keys),
                "actual_action": y,
                "pred_prob_observed": p_obs,
                "pred_prob_action1": (
                    float(json.loads(probs_json)[1])
                    if probs_json and len(json.loads(probs_json)) == 2
                    else p_obs
                ),
                "pred_action": pred,
                "probs_json": probs_json,
                "history_len": len(t.get("history", [])),
                "schema_type": t.get("problem", {}).get("schema_type", ""),
                "error": error,
            }
        )
    return rows


def _load_centaur_trials(
    dataset: str,
    participant_row_index: int,
    *,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    speekenbrink_split: str = "chronological",
    data_dir: Optional[str] = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    str,
    Any,
]:
    """Load train/val/test with optional sparse/limited-data protocol; return instruction + manifest."""
    alias = normalize_psych101_dataset_alias(dataset)
    train_trials, val_trials, test_trials, _audit, manifest = load_participant_limited_splits(
        alias,
        int(participant_row_index),
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        speekenbrink_split=speekenbrink_split,
        data_dir=data_dir,
    )
    instruction = _task_instruction_for_participant(
        alias,
        int(participant_row_index),
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    return train_trials, val_trials, test_trials, instruction, manifest


# Back-compat alias used by older call sites / tests.
def _load_psych101_trials(
    dataset: str,
    participant_row_index: int,
    *,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str,
    local_dataset: Optional[str],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    str,
]:
    train, val, test, instruction, _manifest = _load_centaur_trials(
        dataset,
        participant_row_index,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    return train, val, test, instruction


def _safe_mean_numeric(values: List[Any]) -> Optional[float]:
    vals = [
        float(v)
        for v in values
        if isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))
    ]
    return float(np.mean(vals)) if vals else None


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


def _write_command_line_log(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "log"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / "command.txt"
    cmd = shlex.join([sys.executable, *sys.argv])
    stamp = datetime.now().isoformat(timespec="seconds")
    path.write_text(
        f"# saved {stamp}\n# cwd: {os.getcwd()}\n# host: {socket.gethostname()}\n{cmd}\n",
        encoding="utf-8",
    )
    return path


def _write_loglik_csvs(
    base_run_dir: Path,
    participant_loglik: List[Dict[str, Any]],
) -> None:
    base_run_dir.mkdir(parents=True, exist_ok=True)
    details_path = base_run_dir / "participant_details_loglik.csv"
    summary_path = base_run_dir / "summary_loglik.csv"
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "train_loglik",
                "val_loglik",
                "test_loglik",
                "test_accuracy",
            ],
        )
        w.writeheader()
        w.writerows(_round_floats_for_csv_rows(participant_loglik))
    tr = [d["train_loglik"] for d in participant_loglik if d.get("train_loglik") is not None]
    va = [d["val_loglik"] for d in participant_loglik if d.get("val_loglik") is not None]
    te = [d["test_loglik"] for d in participant_loglik if d.get("test_loglik") is not None]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "num_of_participants",
                "avg_train_loglik",
                "avg_val_loglik",
                "avg_test_loglik",
            ],
        )
        w.writeheader()
        w.writerow(
            _round_floats_for_csv_row(
                {
                    "num_of_participants": len(participant_loglik),
                    "avg_train_loglik": _safe_mean_numeric(tr),
                    "avg_val_loglik": _safe_mean_numeric(va),
                    "avg_test_loglik": _safe_mean_numeric(te),
                }
            )
        )


def _write_predictions_csv(base_run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    log_dir = base_run_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "predictions_vs_actual.csv"
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_smoke_prompt_check(
    dataset: str,
    participant_row_index: int,
    *,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    n_trials: int = 3,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    speekenbrink_split: str = "chronological",
) -> Dict[str, Any]:
    """Validate loaders, display keys, and prompt prefixes without loading the model."""
    train, val, test, instruction, manifest = _load_centaur_trials(
        dataset,
        participant_row_index,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        speekenbrink_split=speekenbrink_split,
    )
    prompt_trials, score_indices = _centaur_prompt_timeline(
        train, val, test
    )
    if not prompt_trials:
        raise ValueError("No trials available for smoke check.")
    if score_indices:
        sample_idx = score_indices[: min(n_trials, len(score_indices))]
    else:
        sample_idx = list(range(min(n_trials, len(prompt_trials))))
    samples = []
    for i in sample_idx:
        prob = prompt_trials[i]["problem"]
        keys = centaur_display_keys(prob)
        if len(keys) < 2:
            raise ValueError(f"Trial {i}: expected >=2 display keys, got {keys!r}")
        prefix = build_centaur_prompt_prefix_indexed(
            prompt_trials, i, instruction=instruction
        )
        if not prefix.rstrip().endswith("You press"):
            raise ValueError(f"Trial {i}: prefix must end with 'You press ', got tail={prefix[-40:]!r}")
        samples.append(
            {
                "trial_index": i,
                "schema_type": prob.get("schema_type"),
                "dataset_alias": prob.get("dataset_alias"),
                "option_keys": prob.get("option_keys"),
                "display_keys": keys,
                "action": prompt_trials[i]["action"],
                "history_len": len(prompt_trials[i].get("history", [])),
                "prefix_tail": prefix[-400:],
                "suffixes": [f"<<{k}>>." for k in keys],
            }
        )
    return {
        "dataset": normalize_psych101_dataset_alias(dataset),
        "participant_row_index": participant_row_index,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "limited_data_protocol": getattr(manifest, "protocol", None),
        "retained_n_train": getattr(manifest, "retained_n_train", None),
        "retained_n_val": getattr(manifest, "retained_n_val", None),
        "retained_n_test": getattr(manifest, "retained_n_test", None),
        "instruction_chars": len(instruction),
        "samples": samples,
    }


def _evaluate_participant(
    chooser: CentaurChooser,
    *,
    dataset: str,
    participant_row_index: int,
    split_ratio: float,
    split_seed: int,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    n_eval_seeds: int,
    debug_prob: bool,
    debug_limit: int,
    max_observed_trials_per_participant: Optional[int] = None,
    limited_data_protocol: str = "off",
    limited_train_val: Optional[int] = None,
    speekenbrink_split: str = "chronological",
    manifest_jsonl_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    train_trials, val_trials, test_trials, instruction, manifest = _load_centaur_trials(
        dataset,
        participant_row_index,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        max_observed_trials_per_participant=max_observed_trials_per_participant,
        limited_data_protocol=limited_data_protocol,
        limited_train_val=limited_train_val,
        speekenbrink_split=speekenbrink_split,
    )
    if manifest_jsonl_path is not None and should_persist_limited_data_manifest(manifest):
        append_limited_data_manifest_jsonl(manifest_jsonl_path, manifest)
    chooser.task_instruction = instruction
    prompt_trials, score_indices = _centaur_prompt_timeline(
        train_trials, val_trials, test_trials
    )
    print(
        f"[Split] {dataset} participant row {participant_row_index}: "
        f"train={len(train_trials)}, val={len(val_trials)}, test={len(test_trials)} "
        f"(ratio={split_ratio:.3f}, seed={split_seed}; "
        f"Centaur prefix on train+val+test n={len(prompt_trials)}, "
        f"scores test only n={len(score_indices)}; protocol={manifest.protocol})"
    )
    test_eval = evaluate_centaur_on_trials(
        chooser,
        prompt_trials,
        n_seeds=n_eval_seeds,
        debug_prob=debug_prob,
        debug_limit=debug_limit,
        score_indices=score_indices,
    )
    preds = collect_centaur_predictions(
        chooser,
        prompt_trials,
        participant_id=participant_row_index,
        dataset=normalize_psych101_dataset_alias(dataset),
        split_name="test",
        score_indices=score_indices,
    )
    summary = {
        "participant_id": participant_row_index,
        "train_loglik": None,
        "val_loglik": None,
        "test_loglik": test_eval["avg_loglik"],
        "test_accuracy": test_eval["accuracy"],
    }
    return summary, preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Centaur baseline for TEH loglik datasets (held-out test trial loglik; "
            "Bernoulli or K-way categorical)."
        )
    )
    _choices = sorted(
        set(PSYCH101_CENTAUR_DATASETS)
        | {normalize_psych101_dataset_alias(a) for a in PSYCH101_CENTAUR_DATASETS}
    )
    parser.add_argument("--dataset", type=str, default=None, choices=_choices)
    parser.add_argument(
        "--psych_dataset_split",
        type=str,
        default=DEFAULT_PSYCH_DATASET_SPLIT,
        choices=["train", "test"],
    )
    parser.add_argument("--local_dataset", type=str, default=None)
    parser.add_argument(
        "--participant_scope",
        type=str,
        default="single",
        choices=["single", "range", "ordinals", "all"],
    )
    parser.add_argument("--single_participant_id", type=int, default=0)
    parser.add_argument("--range_start_ordinal", type=int, default=None)
    parser.add_argument("--range_end_ordinal", type=int, default=None)
    parser.add_argument("--ordinals", nargs="+", type=int, default=None)
    parser.add_argument("--all_max_participants", type=int, default=None)
    parser.add_argument("--split_ratio", type=float, default=0.6)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--fitness_metric", type=str, default="loglik", choices=["loglik"])
    parser.add_argument(
        "--centaur_model",
        type=str,
        default="marcelbinz/Llama-3.1-Centaur-70B-adapter",
    )
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument("--n_eval_seeds", type=int, default=1)
    parser.add_argument("--debug_prob", action="store_true", default=False)
    parser.add_argument("--debug_prob_limit", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--max_observed_trials_per_participant",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Sparse-data: after the train/val/test split, keep at most N combined "
            "train+val observations per participant (test untouched). Same CLI as TEH/MLE."
        ),
    )
    add_limited_data_cli_arguments(parser)
    parser.add_argument(
        "--check_deps",
        action="store_true",
        help=(
            "Import Centaur runtime packages (torch/transformers/unsloth/...) without "
            "loading the model or requiring a GPU. Exit 1 if any import fails."
        ),
    )
    parser.add_argument(
        "--smoke_prompt_only",
        action="store_true",
        help="Validate parsing/prompts for participant 0 (or --single_participant_id) without GPU.",
    )
    parser.add_argument(
        "--smoke_all_datasets",
        action="store_true",
        help="Run --smoke_prompt_only for every Centaur dataset alias (no GPU).",
    )
    parser.add_argument(
        "--smoke_focus_datasets",
        action="store_true",
        help="Like --smoke_all_datasets but only the five new focus datasets.",
    )
    args = parser.parse_args()

    if args.check_deps:
        report = check_centaur_runtime_deps()
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            missing = [
                p["package"] for p in report["packages"] if not p.get("ok")
            ]
            print(
                "[ERROR] Centaur deps missing: "
                + ", ".join(missing)
                + ". Install with: pip install -e '.[centaur]' (see setup_env.sh --with-centaur).",
                file=sys.stderr,
            )
            sys.exit(1)
        print("[INFO] Centaur import check passed (model not loaded).")
        return

    if args.fitness_metric != "loglik":
        print("Only --fitness_metric loglik is supported.")
        sys.exit(1)
    if not (0.0 < args.split_ratio < 1.0):
        print("--split_ratio must be in (0, 1).")
        sys.exit(1)
    if not args.smoke_all_datasets and not args.smoke_focus_datasets and not args.dataset:
        print("--dataset is required unless --smoke_all_datasets / --smoke_focus_datasets is set.")
        sys.exit(1)
    try:
        resolve_limited_data_budget(
            protocol=args.limited_data_protocol,
            max_observed_trials_per_participant=args.max_observed_trials_per_participant,
            limited_train_val=args.limited_train_val,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    dataset = normalize_psych101_dataset_alias(args.dataset) if args.dataset else ""
    psych_split = _effective_psych_dataset_split(args.psych_dataset_split)
    sparse_kwargs = dict(
        max_observed_trials_per_participant=args.max_observed_trials_per_participant,
        limited_data_protocol=args.limited_data_protocol,
        limited_train_val=args.limited_train_val,
        speekenbrink_split=getattr(args, "speekenbrink_split", "chronological"),
    )

    smoke_aliases: List[str] = []
    if args.smoke_focus_datasets:
        smoke_aliases = sorted(CENTAUR_FOCUS_DATASETS)
    elif args.smoke_all_datasets:
        smoke_aliases = list(PSYCH101_CENTAUR_DATASETS)

    if smoke_aliases:
        for alias in smoke_aliases:
            print(f"\n=== smoke {alias} ===")
            try:
                valid = load_valid_participant_ids_from_json(
                    alias,
                    REPO_ROOT,
                    split_ratio=args.split_ratio,
                    split_seed=args.split_seed,
                    psych_dataset_split=psych_split,
                    local_dataset=args.local_dataset,
                )
                pid = int(valid[0]) if valid else 0
                info = run_smoke_prompt_check(
                    alias,
                    pid,
                    split_ratio=args.split_ratio,
                    split_seed=args.split_seed,
                    psych_dataset_split=psych_split,
                    local_dataset=args.local_dataset,
                    **sparse_kwargs,
                )
                print(json.dumps(info, indent=2))
            except Exception as e:
                print(f"FAIL {alias}: {e}")
        return

    participants = resolve_participants_for_scope(
        dataset=dataset,
        repo_root=REPO_ROOT,
        participant_scope=args.participant_scope,
        single_participant_id=args.single_participant_id,
        range_start_ordinal=args.range_start_ordinal,
        range_end_ordinal=args.range_end_ordinal,
        all_max_participants=args.all_max_participants,
        participant_ordinals=args.ordinals,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        psych_dataset_split=psych_split,
        local_dataset=args.local_dataset,
    )

    if args.smoke_prompt_only:
        pid = participants[0]
        info = run_smoke_prompt_check(
            dataset,
            pid,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            psych_dataset_split=psych_split,
            local_dataset=args.local_dataset,
            **sparse_kwargs,
        )
        print(json.dumps(info, indent=2))
        return

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_run_dir = Path(
        args.output_dir
        if args.output_dir
        else centaur_output_base_dir(dataset, timestamp, psych_dataset_split=psych_split)
    )
    cmd_log = _write_command_line_log(base_run_dir)
    print(f"Wrote full command line to {cmd_log}")
    manifest_jsonl = base_run_dir / "log" / LIMITED_DATA_MANIFEST_JSONL_FILENAME

    chooser = CentaurChooser(args.centaur_model, max_seq_length=args.max_seq_length)
    participant_loglik: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    for pid in tqdm(participants, desc="Participants"):
        summ, preds = _evaluate_participant(
            chooser,
            dataset=dataset,
            participant_row_index=pid,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            psych_dataset_split=psych_split,
            local_dataset=args.local_dataset,
            n_eval_seeds=args.n_eval_seeds,
            debug_prob=args.debug_prob,
            debug_limit=args.debug_prob_limit,
            manifest_jsonl_path=manifest_jsonl,
            **sparse_kwargs,
        )
        participant_loglik.append(summ)
        prediction_rows.extend(preds)

    _write_loglik_csvs(base_run_dir, participant_loglik)
    _write_predictions_csv(base_run_dir, prediction_rows)
    rewrite_limited_data_csv_from_jsonl(
        manifest_jsonl, base_run_dir / "log" / LIMITED_DATA_MANIFEST_CSV_FILENAME
    )
    print(f"Wrote participant_details_loglik.csv and summary_loglik.csv under {base_run_dir}")


if __name__ == "__main__":
    main()
