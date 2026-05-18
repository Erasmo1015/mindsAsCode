"""
Psych-101 binary cognitive datasets for TEH (Template Evolution HuggingFace).

Loads marcelbinz/Psych-101 or Psych-101-test rows and parses NL transcripts into
Experiment objects compatible with Template_evo split_trials / experiment_to_trials.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from datasets import load_dataset, load_from_disk

# Reuse choice13k structured types (same as legacy choice13k pipeline).
from data_modules.choice13k import (
    Block,
    Experiment,
    Gamble,
    Trial,
    _convert_to_experiment as _choice13k_convert_to_experiment,
    _extract_gamble_info,
    _extract_has_feedback,
    _hf_token_for_datasets,
)

TEST_HF_ID = "marcelbinz/Psych-101-test"
TRAIN_HF_ID = "marcelbinz/Psych-101"

# choice13k-style press + CPC18 feedback lines (`and gain` / `and lose`)
_RE_TRIAL_PRESS = re.compile(
    r"(You press <<([A-Z])>>"
    r"(?:\.(?:\s*You receive (-?\d+\.?\d*) points.*?)?"
    r"| and (?:gain|lose) (-?\d+\.?\d*) points.*?))"
    r"(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_trials_extended(
    trials_str: str, gamble_info: str, option_keys: List[str]
) -> List[Trial]:
    trials: List[Trial] = []
    history_prefix = gamble_info.strip() + "\n"
    for match in _RE_TRIAL_PRESS.finditer(trials_str):
        full_trial_str = match.group(1).strip()
        key = match.group(2).upper()
        if key not in option_keys:
            continue
        action = option_keys.index(key)
        fb_dot = match.group(3)
        fb_gain = match.group(4)
        feedback = None
        if fb_dot:
            feedback = float(fb_dot)
        elif fb_gain:
            feedback = float(fb_gain)
        trials.append(
            Trial(
                action=action,
                feedback=feedback,
                history=history_prefix.strip(),
            )
        )
        history_prefix += "You press <<" + key + ">>.\n" + full_trial_str + "\n"
    return trials


def _convert_to_block_extended(block_str: str) -> Block:
    gamble_info = block_str.split("\nYou press")[0]
    trials_str = block_str[len(gamble_info) :].lstrip()
    gamble_A, gamble_B, option_keys = _extract_gamble_info(gamble_info)
    trials = _extract_trials_extended(trials_str, gamble_info, option_keys)
    return Block(
        trials=trials,
        gamble_A=gamble_A,
        gamble_B=gamble_B,
        has_feedback=_extract_has_feedback(block_str),
        option_keys=option_keys,
        gamble_info_text=gamble_info.strip(),
    )


def _convert_to_experiment_extended(row: Dict[str, Any]) -> Experiment:
    data = dict(row)
    data["text"] = data["text"].replace("\n\n\n\nOption", "\n\nOption")
    data["text"] = data["text"].replace("\n\n\nOption", "\n\nOption")
    instruction = data["text"].split("\n\nOption")[0]
    trials_str = data["text"][len(instruction) :].lstrip()
    blocks: List[Block] = []
    for block_str in trials_str.split("\n\n"):
        if not block_str.strip():
            continue
        if "Option " not in block_str and not blocks:
            continue
        if "Option " not in block_str:
            continue
        blocks.append(_convert_to_block_extended(block_str))
    return Experiment(instruction=instruction, blocks=blocks)


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
        "implemented": False,
        "task_description": "Decisions from description; Lottery W/H offers format.",
    },
    "speekenbrink2008learning": {
        "experiment_id": "speekenbrink2008learning/exp1.csv",
        "display_name": "Speekenbrink learning",
        "schema_type": "B",
        "parser": "weather_cards",
        "implemented": False,
        "task_description": "Weather prediction from tarot cards; binary E/J.",
    },
    "sadeghiyeh2020temporal": {
        "experiment_id": "sadeghiyeh2020temporal/exp1.csv",
        "display_name": "Sadeghiyeh temporal/bandit",
        "schema_type": "C",
        "parser": "slot_machine_bandit",
        "implemented": False,
        "task_description": "Two-arm bandit slot machines (Psych-101 text).",
    },
    "hilbig2014generalized": {
        "experiment_id": "hilbig2014generalized/exp1.csv",
        "display_name": "Hilbig generalized",
        "schema_type": "B",
        "parser": "product_ratings",
        "implemented": False,
        "task_description": "Product choice with expert rating vectors.",
    },
    "frey2017cct": {
        "experiment_id": "frey2017cct/exp1.csv",
        "display_name": "Frey CCT",
        "schema_type": "C",
        "parser": "columbia_card_task",
        "implemented": False,
        "task_description": "Columbia Card Task (flip vs stop).",
    },
    "flesch2018comparing": {
        "experiment_id": "flesch2018comparing/exp1.csv",
        "display_name": "Flesch comparing",
        "schema_type": "B",
        "parser": "tree_accept_reject",
        "implemented": False,
        "task_description": "Tree accept/reject in North/South gardens.",
    },
}


def is_psych101_dataset(dataset_alias: str) -> bool:
    return dataset_alias in PSYCH101_BINARY_DATASETS


def experiment_id_for_alias(dataset_alias: str) -> str:
    if dataset_alias not in PSYCH101_BINARY_DATASETS:
        raise KeyError(f"Unknown Psych-101 dataset alias: {dataset_alias!r}")
    return PSYCH101_BINARY_DATASETS[dataset_alias]["experiment_id"]


def _parse_row(row: Dict[str, Any], dataset_alias: str) -> Experiment:
    spec = PSYCH101_BINARY_DATASETS[dataset_alias]
    if not spec.get("implemented"):
        raise NotImplementedError(
            f"Parser for dataset alias {dataset_alias!r} is not implemented yet "
            f"(expected parser type: {spec.get('parser')!r}, schema_type={spec.get('schema_type')!r})."
        )
    parser = spec["parser"]
    if parser == "choice13k":
        return _choice13k_convert_to_experiment(row)
    if parser == "option_delivers_extended":
        return _convert_to_experiment_extended(row)
    raise NotImplementedError(f"Unknown parser id {parser!r} for {dataset_alias!r}")


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


def get_psych101_binary_experiments(
    dataset_alias: str,
    n_participants: int = 10,
    split: str = "test",
    local_dataset: Optional[str] = None,
) -> List[Experiment]:
    """
    Load and parse Psych-101 participant rows for a registered binary dataset alias.

    Participant index i is the i-th row in the filtered HF split (same as legacy choice13k).
    """
    if dataset_alias not in PSYCH101_BINARY_DATASETS:
        raise KeyError(
            f"Unknown dataset alias {dataset_alias!r}. "
            f"Known: {sorted(PSYCH101_BINARY_DATASETS)}"
        )
    exp_id = experiment_id_for_alias(dataset_alias)
    split_ds = _load_hf_split(split, local_dataset=local_dataset)
    filtered = split_ds.filter(lambda ex, e=exp_id: ex["experiment"] == e)
    n = min(n_participants, len(filtered))
    experiments: List[Experiment] = []
    for i in range(n):
        row = dict(filtered[i])
        experiments.append(_parse_row(row, dataset_alias))
    return experiments


def experiment_to_trial_dicts(
    exp: Experiment,
    *,
    dataset_alias: str,
    experiment_id: str,
) -> List[Dict[str, Any]]:
    """Convert Experiment to evaluator trial dicts (gamble schema A)."""
    if not exp.blocks:
        return []
    options = exp.blocks[0].option_keys
    all_trials: List[Dict[str, Any]] = []
    history_accum: List[Dict[str, Any]] = []
    for block in exp.blocks:
        for trial in block.trials:
            problem: Dict[str, Any] = {
                "gamble_A": {
                    "probs": block.gamble_A.probs,
                    "rewards": block.gamble_A.rewards,
                },
                "gamble_B": {
                    "probs": block.gamble_B.probs,
                    "rewards": block.gamble_B.rewards,
                },
                "option_keys": list(options),
                "has_feedback": block.has_feedback,
                "dataset_alias": dataset_alias,
                "experiment_id": experiment_id,
            }
            all_trials.append(
                {
                    "problem": problem,
                    "history": list(history_accum),
                    "options": list(options),
                    "action": trial.action,
                }
            )
            history_accum.append(
                {"action": trial.action, "feedback": trial.feedback}
            )
    return all_trials


def parse_coverage_stats(text: str, exp: Experiment) -> Dict[str, Any]:
    """Press recovery rate for smoke tests."""
    n_press = len(re.findall(r"You press <<", text))
    n_trials = sum(len(b.trials) for b in exp.blocks)
    return {
        "n_presses_in_text": n_press,
        "n_trials_parsed": n_trials,
        "parse_coverage": round(n_trials / n_press, 4) if n_press else 0.0,
    }
