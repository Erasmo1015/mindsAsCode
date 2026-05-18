"""
Psych-101 binary cognitive datasets for TEH (Template Evolution HuggingFace).

Loads marcelbinz/Psych-101 or Psych-101-test rows and parses NL transcripts into
PsychExperiment objects with schema-specific structured trial dicts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from datasets import load_dataset, load_from_disk

from data_modules.choice13k import _hf_token_for_datasets

TEST_HF_ID = "marcelbinz/Psych-101-test"
TRAIN_HF_ID = "marcelbinz/Psych-101"

PSYCH_DATASET_SPLITS = frozenset({"train", "test"})
DEFAULT_PSYCH_DATASET_SPLIT = "train"


class ParsedTrial(NamedTuple):
    action: int
    feedback: Optional[Any]
    problem_fields: Dict[str, Any]


class PsychBlock(NamedTuple):
    trials: List[ParsedTrial]
    option_keys: List[str]
    problem_static: Dict[str, Any]
    schema_type: str


class PsychExperiment(NamedTuple):
    instruction: str
    blocks: List[PsychBlock]
    dataset_alias: str
    schema_type: str


def normalize_psych_dataset_split(split: str) -> str:
    s = str(split).strip().lower()
    if s not in PSYCH_DATASET_SPLITS:
        raise ValueError(
            f"psych_dataset_split must be one of {sorted(PSYCH_DATASET_SPLITS)}, got {split!r}"
        )
    return s


def hf_id_for_psych_dataset_split(split: str) -> str:
    s = normalize_psych_dataset_split(split)
    return TRAIN_HF_ID if s == "train" else TEST_HF_ID


PSYCH101_BINARY_DATASETS: Dict[str, Dict[str, Any]] = {
    "peterson2021using": {
        "experiment_id": "peterson2021using/exp1.csv",
        "display_name": "Choice13k",
        "schema_type": "A",
        "parser": "choice13k",
        "implemented": True,
        "task_description": (
            "Risky two-option gambles with explicit outcome probabilities (Choice13k / peterson2021using)."
        ),
    },
    "plonsky2018when": {
        "experiment_id": "plonsky2018when/exp1.csv",
        "display_name": "CPC18 Psych-101",
        "schema_type": "A",
        "parser": "option_delivers_extended",
        "implemented": True,
        "task_description": (
            "Two-option gambles (CPC18); includes no-feedback and feedback trials with "
            "'You press <<X>> and gain/lose ...' lines."
        ),
    },
    "wulff2018description": {
        "experiment_id": "wulff2018description/exp1.csv",
        "display_name": "Wulff description",
        "schema_type": "A",
        "parser": "lottery_offers",
        "implemented": True,
        "task_description": "Decisions from description; Lottery W/H offers format.",
    },
    "speekenbrink2008learning": {
        "experiment_id": "speekenbrink2008learning/exp1.csv",
        "display_name": "Speekenbrink learning",
        "schema_type": "B",
        "parser": "weather_cards",
        "implemented": True,
        "task_description": "Weather prediction from tarot cards; binary press keys per participant.",
    },
    "sadeghiyeh2020temporal": {
        "experiment_id": "sadeghiyeh2020temporal/exp1.csv",
        "display_name": "Sadeghiyeh temporal/bandit",
        "schema_type": "C",
        "parser": "slot_machine_bandit",
        "implemented": True,
        "task_description": "Two-arm bandit slot machines by game (instructed then free trials).",
    },
    "hilbig2014generalized": {
        "experiment_id": "hilbig2014generalized/exp1.csv",
        "display_name": "Hilbig generalized",
        "schema_type": "B",
        "parser": "product_ratings",
        "implemented": True,
        "task_description": "Product choice with expert rating vectors.",
    },
    "frey2017cct": {
        "experiment_id": "frey2017cct/exp1.csv",
        "display_name": "Frey CCT",
        "schema_type": "D",
        "parser": "columbia_card_task",
        "implemented": True,
        "task_description": "Columbia Card Task (flip E vs stop C) per round.",
    },
    "flesch2018comparing": {
        "experiment_id": "flesch2018comparing/exp1.csv",
        "display_name": "Flesch comparing",
        "schema_type": "B",
        "parser": "tree_accept_reject",
        "implemented": True,
        "task_description": "Tree accept/reject in North/South gardens.",
    },
}


def is_psych101_dataset(dataset_alias: str) -> bool:
    return dataset_alias in PSYCH101_BINARY_DATASETS


def experiment_id_for_alias(dataset_alias: str) -> str:
    if dataset_alias not in PSYCH101_BINARY_DATASETS:
        raise KeyError(f"Unknown Psych-101 dataset alias: {dataset_alias!r}")
    return PSYCH101_BINARY_DATASETS[dataset_alias]["experiment_id"]


def schema_type_for_alias(dataset_alias: str) -> str:
    return PSYCH101_BINARY_DATASETS[dataset_alias]["schema_type"]


def _parse_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    spec = PSYCH101_BINARY_DATASETS[dataset_alias]
    if not spec.get("implemented"):
        raise NotImplementedError(
            f"Parser for dataset alias {dataset_alias!r} is not implemented yet "
            f"(expected parser type: {spec.get('parser')!r})."
        )
    from data_modules.psych101_parsers import PARSER_DISPATCH

    parser_id = spec["parser"]
    fn = PARSER_DISPATCH.get(parser_id)
    if fn is None:
        raise NotImplementedError(f"Unknown parser id {parser_id!r} for {dataset_alias!r}")
    return fn(row, dataset_alias)


def _load_hf_split(
    split: str,
    *,
    local_dataset: Optional[str] = None,
    hf_test: str = TEST_HF_ID,
    hf_train: str = TRAIN_HF_ID,
):
    if local_dataset:
        path = Path(local_dataset).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local dataset path does not exist: {path}")
        dataset = load_from_disk(str(path))
    else:
        tok = _hf_token_for_datasets()
        ds_kw = {"token": tok} if tok else {}
        hf_id = hf_test if split == "test" else hf_train
        dataset = load_dataset(hf_id, **ds_kw)
    if split not in dataset:
        raise ValueError(f"Split {split!r} not in dataset; have {list(dataset.keys())}")
    return dataset[split]


def get_filtered_psych101_split(
    dataset_alias: str,
    split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
):
    """HF rows for one Psych-101 experiment id (shows datasets `Filter` progress once)."""
    if dataset_alias not in PSYCH101_BINARY_DATASETS:
        raise KeyError(
            f"Unknown dataset alias {dataset_alias!r}. "
            f"Known: {sorted(PSYCH101_BINARY_DATASETS)}"
        )
    exp_id = experiment_id_for_alias(dataset_alias)
    split_ds = _load_hf_split(split, local_dataset=local_dataset)
    return split_ds.filter(lambda ex, e=exp_id: ex["experiment"] == e)


def parse_psych101_binary_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    """Parse one HF row into a PsychExperiment."""
    return _parse_row(row, dataset_alias)


def get_psych101_binary_experiments(
    dataset_alias: str,
    n_participants: int = 10,
    split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
) -> List[PsychExperiment]:
    filtered = get_filtered_psych101_split(
        dataset_alias, split=split, local_dataset=local_dataset
    )
    n = min(n_participants, len(filtered))
    experiments: List[PsychExperiment] = []
    for i in range(n):
        row = dict(filtered[i])
        experiments.append(parse_psych101_binary_row(row, dataset_alias))
    return experiments


def _merge_problem(
    block: PsychBlock,
    trial: ParsedTrial,
    *,
    dataset_alias: str,
    experiment_id: str,
) -> Dict[str, Any]:
    problem = dict(block.problem_static)
    problem.update(trial.problem_fields)
    problem.setdefault("option_keys", list(block.option_keys))
    problem.setdefault("schema_type", block.schema_type)
    problem["dataset_alias"] = dataset_alias
    problem["experiment_id"] = experiment_id
    return problem


def experiment_to_trial_dicts(
    exp: PsychExperiment,
    *,
    dataset_alias: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert PsychExperiment to evaluator trial dicts (schema-specific problem fields)."""
    alias = dataset_alias or exp.dataset_alias
    exp_id = experiment_id or experiment_id_for_alias(alias)
    all_trials: List[Dict[str, Any]] = []
    history_accum: List[Dict[str, Any]] = []
    for block in exp.blocks:
        block_history: List[Dict[str, Any]] = []
        for trial in block.trials:
            problem = _merge_problem(
                block, trial, dataset_alias=alias, experiment_id=exp_id
            )
            all_trials.append(
                {
                    "problem": problem,
                    "history": list(block_history),
                    "options": list(block.option_keys),
                    "action": trial.action,
                }
            )
            entry: Dict[str, Any] = {"action": trial.action}
            if trial.feedback is not None:
                entry["feedback"] = trial.feedback
            for k, v in trial.problem_fields.items():
                if k not in entry:
                    entry[k] = v
            block_history.append(entry)
    return all_trials


def trials_from_blocks(
    exp: PsychExperiment,
    block_indices: set,
    *,
    dataset_alias: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Trials from selected blocks; history resets per block."""
    alias = dataset_alias or exp.dataset_alias
    exp_id = experiment_id or experiment_id_for_alias(alias)
    out: List[Dict[str, Any]] = []
    for bi, block in enumerate(exp.blocks):
        if bi not in block_indices:
            continue
        block_history: List[Dict[str, Any]] = []
        for trial in block.trials:
            problem = _merge_problem(
                block, trial, dataset_alias=alias, experiment_id=exp_id
            )
            out.append(
                {
                    "problem": problem,
                    "history": list(block_history),
                    "options": list(block.option_keys),
                    "action": trial.action,
                }
            )
            entry: Dict[str, Any] = {"action": trial.action}
            if trial.feedback is not None:
                entry["feedback"] = trial.feedback
            for k, v in trial.problem_fields.items():
                if k not in entry:
                    entry[k] = v
            block_history.append(entry)
    return out


def _expand_single_block_to_pseudo_blocks(exp: PsychExperiment) -> PsychExperiment:
    """When one global block holds all trials, chunk into pseudo-blocks for TEH split."""
    if len(exp.blocks) != 1:
        return exp
    block = exp.blocks[0]
    n = len(block.trials)
    if n < 3:
        return exp
    target_blocks = max(3, min(30, n // 15))
    chunk_size = max(1, (n + target_blocks - 1) // target_blocks)
    new_blocks: List[PsychBlock] = []
    for i in range(0, n, chunk_size):
        chunk = block.trials[i : i + chunk_size]
        if not chunk:
            continue
        new_blocks.append(
            PsychBlock(
                trials=chunk,
                option_keys=list(block.option_keys),
                problem_static=dict(block.problem_static),
                schema_type=block.schema_type,
            )
        )
    if len(new_blocks) < 3:
        return exp
    return PsychExperiment(
        instruction=exp.instruction,
        blocks=new_blocks,
        dataset_alias=exp.dataset_alias,
        schema_type=exp.schema_type,
    )


def split_psych_experiment(
    exp: PsychExperiment,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], list]:
    """Split by block (problem/game/round); history does not cross blocks."""
    exp = _expand_single_block_to_pseudo_blocks(exp)
    n_blocks = len(exp.blocks)
    if n_blocks < 3:
        raise ValueError(
            f"Psych-101 train/val/test split requires at least 3 problems (blocks); got {n_blocks}."
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

    train_trials = trials_from_blocks(exp, train_blocks)
    val_trials = trials_from_blocks(exp, val_blocks)
    test_trials = trials_from_blocks(exp, test_blocks)
    options = exp.blocks[0].option_keys
    return train_trials, val_trials, test_trials, options


def parse_coverage_stats(text: str, exp: PsychExperiment) -> Dict[str, Any]:
    n_press = len(re.findall(r"You press <<", text))
    n_instructed = len(re.findall(r"You are instructed to press", text, re.I))
    n_actions = n_press + n_instructed
    n_trials = sum(len(b.trials) for b in exp.blocks)
    denom = n_actions if n_instructed else n_press
    return {
        "n_presses_in_text": n_press,
        "n_instructed_actions": n_instructed,
        "n_trials_parsed": n_trials,
        "parse_coverage": round(n_trials / denom, 4) if denom else 0.0,
    }


def format_trial_for_prompt(trial: Dict[str, Any], index: int) -> str:
    """One-line summary of a parsed trial for infer_single_choice prompt generation."""
    p = trial["problem"]
    schema = p.get("schema_type", "?")
    action = trial["action"]
    keys = p.get("option_keys", [])
    hist_len = len(trial.get("history", []))
    if schema == "A":
        ga = p.get("gamble_A", {})
        gb = p.get("gamble_B", {})
        return (
            f"{index}. [gamble] gamble_A probs={ga.get('probs')} rewards={ga.get('rewards')}; "
            f"gamble_B probs={gb.get('probs')} rewards={gb.get('rewards')}; "
            f"option_keys={keys}; has_feedback={p.get('has_feedback')}; "
            f"action={action} (key={keys[action] if action < len(keys) else '?'}); history_len={hist_len}"
        )
    if schema == "B":
        if "cards" in p or "cards" in trial.get("history", [{}])[0] if trial.get("history") else False:
            return (
                f"{index}. [weather] cards={p.get('cards')}; option_keys={keys}; "
                f"weather_outcome={p.get('weather_outcome')}; action={action}; history_len={hist_len}"
            )
        if "tree_features" in p:
            return (
                f"{index}. [tree] features={p.get('tree_features')}; garden={p.get('garden')}; "
                f"phase={p.get('phase')}; option_keys={keys}; action={action}; history_len={hist_len}"
            )
        return (
            f"{index}. [product] ratings_A={p.get('ratings_A')}; ratings_B={p.get('ratings_B')}; "
            f"option_keys={keys}; action={action}; history_len={hist_len}"
        )
    if schema == "C":
        return (
            f"{index}. [bandit] game_id={p.get('game_id')}; phase={p.get('phase')}; "
            f"trial_index={p.get('trial_index')}; machine_options={p.get('machine_options')}; "
            f"option_keys={keys}; action={action}; history_len={hist_len}"
        )
    if schema == "D":
        return (
            f"{index}. [cct] round={p.get('round_id')}; score={p.get('current_score')}; "
            f"flipped={p.get('cards_flipped')}; remaining={p.get('n_cards_remaining')}; "
            f"gain={p.get('gain_amount')}; loss={p.get('loss_amount')}; n_loss={p.get('n_loss_cards')}; "
            f"option_keys={keys}; action={action} (E=flip,C=stop); history_len={hist_len}"
        )
    return f"{index}. problem_keys={list(p.keys())}; action={action}; history_len={hist_len}"


def format_trials_for_prompt(trials: List[Dict[str, Any]], max_trials: int = 8) -> str:
    lines = [
        format_trial_for_prompt(t, i + 1)
        for i, t in enumerate(trials[:max_trials])
    ]
    return "\n".join(lines)
