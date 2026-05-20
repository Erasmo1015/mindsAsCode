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


# CLI / output-folder aliases (numbered); HF experiment_id strings are unchanged.
PSYCH101_BINARY_DATASETS: Dict[str, Dict[str, Any]] = {
    "1peterson2021using": {
        "experiment_id": "peterson2021using/exp1.csv",
        "display_name": "Choice13k",
        "schema_type": "A",
        "parser": "choice13k",
        "implemented": True,
        "task_description": (
            "Risky two-option gambles with explicit outcome probabilities (Choice13k / peterson2021using)."
        ),
    },
    "2plonsky2018when": {
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
    "3frey2017cct": {
        "experiment_id": "frey2017cct/exp1.csv",
        "display_name": "Frey CCT",
        "schema_type": "D",
        "parser": "columbia_card_task",
        "implemented": True,
        "task_description": "Columbia Card Task (flip E vs stop C) per round.",
    },
    "4wulff2018description": {
        "experiment_id": "wulff2018description/exp1.csv",
        "display_name": "Wulff description",
        "schema_type": "A",
        "parser": "lottery_offers",
        "implemented": True,
        "task_description": "Decisions from description; Lottery W/H offers format.",
    },
    "5speekenbrink2008learning": {
        "experiment_id": "speekenbrink2008learning/exp1.csv",
        "display_name": "Speekenbrink learning",
        "schema_type": "B",
        "parser": "weather_cards",
        "implemented": True,
        "task_description": "Weather prediction from tarot cards; binary press keys per participant.",
    },
    "6sadeghiyeh2020temporal": {
        "experiment_id": "sadeghiyeh2020temporal/exp1.csv",
        "display_name": "Sadeghiyeh temporal/bandit",
        "schema_type": "C",
        "parser": "slot_machine_bandit",
        "implemented": True,
        "task_description": "Two-arm bandit slot machines by game (instructed then free trials).",
    },
    "7hilbig2014generalized": {
        "experiment_id": "hilbig2014generalized/exp1.csv",
        "display_name": "Hilbig generalized",
        "schema_type": "B",
        "parser": "product_ratings",
        "implemented": True,
        "task_description": "Product choice with expert rating vectors.",
    },
    "8flesch2018comparing": {
        "experiment_id": "flesch2018comparing/exp1.csv",
        "display_name": "Flesch comparing",
        "schema_type": "B",
        "parser": "tree_accept_reject",
        "implemented": True,
        "task_description": "Tree accept/reject in North/South gardens.",
    },
}

# Unprefixed aliases accepted on CLI for backward compatibility.
PSYCH101_LEGACY_ALIASES: Dict[str, str] = {
    legacy: numbered
    for numbered, spec in PSYCH101_BINARY_DATASETS.items()
    if (legacy := str(spec["experiment_id"]).split("/")[0]) != numbered
}

PETERSON2021USING_ALIAS = "1peterson2021using"


def normalize_psych101_dataset_alias(dataset_alias: str) -> str:
    """Map legacy unprefixed alias to numbered CLI alias (e.g. peterson2021using -> 1peterson2021using)."""
    alias = str(dataset_alias).strip()
    if alias in PSYCH101_BINARY_DATASETS:
        return alias
    return PSYCH101_LEGACY_ALIASES.get(alias, alias)


def is_psych101_dataset(dataset_alias: str) -> bool:
    return normalize_psych101_dataset_alias(dataset_alias) in PSYCH101_BINARY_DATASETS


def experiment_id_for_alias(dataset_alias: str) -> str:
    alias = normalize_psych101_dataset_alias(dataset_alias)
    if alias not in PSYCH101_BINARY_DATASETS:
        raise KeyError(f"Unknown Psych-101 dataset alias: {dataset_alias!r}")
    return PSYCH101_BINARY_DATASETS[alias]["experiment_id"]


def schema_type_for_alias(dataset_alias: str) -> str:
    alias = normalize_psych101_dataset_alias(dataset_alias)
    return PSYCH101_BINARY_DATASETS[alias]["schema_type"]


def _parse_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    alias = normalize_psych101_dataset_alias(dataset_alias)
    spec = PSYCH101_BINARY_DATASETS[alias]
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
    return fn(row, alias)


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
    alias = normalize_psych101_dataset_alias(dataset_alias)
    if alias not in PSYCH101_BINARY_DATASETS:
        raise KeyError(
            f"Unknown dataset alias {dataset_alias!r}. "
            f"Known: {sorted(PSYCH101_BINARY_DATASETS)}"
        )
    exp_id = experiment_id_for_alias(alias)
    split_ds = _load_hf_split(split, local_dataset=local_dataset)
    return split_ds.filter(lambda ex, e=exp_id: ex["experiment"] == e)


def parse_psych101_binary_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    """Parse one HF row into a PsychExperiment."""
    return _parse_row(row, dataset_alias)


def get_psych101_binary_experiment(
    dataset_alias: str,
    participant_row_index: int,
    split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    *,
    filtered_split: Optional[Any] = None,
) -> PsychExperiment:
    """Load one participant by 0-based index in the filtered HF split (no sequential re-parse)."""
    alias = normalize_psych101_dataset_alias(dataset_alias)
    filtered = (
        filtered_split
        if filtered_split is not None
        else get_filtered_psych101_split(
            alias, split=split, local_dataset=local_dataset
        )
    )
    n_rows = len(filtered)
    if participant_row_index < 0 or participant_row_index >= n_rows:
        raise IndexError(
            f"participant_row_index={participant_row_index} out of range for "
            f"{alias!r} split={split!r} (filtered rows={n_rows})"
        )
    row = dict(filtered[participant_row_index])
    return parse_psych101_binary_row(row, alias)


def get_psych101_binary_experiments(
    dataset_alias: str,
    n_participants: int = 10,
    split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
) -> List[PsychExperiment]:
    alias = normalize_psych101_dataset_alias(dataset_alias)
    filtered = get_filtered_psych101_split(
        alias, split=split, local_dataset=local_dataset
    )
    n = min(n_participants, len(filtered))
    return [
        get_psych101_binary_experiment(
            alias,
            i,
            split=split,
            local_dataset=local_dataset,
            filtered_split=filtered,
        )
        for i in range(n)
    ]


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


def _action_key_label(keys: List[str], action: int) -> str:
    if 0 <= action < len(keys):
        return keys[action]
    return "?"


def _schema_b_subtype(problem: Dict[str, Any]) -> str:
    if "tree_features" in problem:
        return "tree"
    if "cards" in problem or problem.get("features", {}).get("task") == "weather_prediction":
        return "weather"
    if "ratings_A" in problem or "option_A_features" in problem:
        return "product"
    return "binary"


def _action_semantics_for_schema(
    keys: List[str], schema: str, problem: Dict[str, Any], *, is_gamble: bool
) -> str:
    if len(keys) < 2:
        return "action=0 is first option; action=1 is second; return P(action=1)."
    k0, k1 = keys[0], keys[1]
    if schema == "A" and is_gamble:
        keys_note = f"option_keys={list(keys)!r}" if keys else "option_keys has two labels"
        return (
            "action=0 selects option_keys[0] (first gamble; problem['gamble_A']); "
            "action=1 selects option_keys[1] (second gamble; problem['gamble_B']); "
            f"return P(action=1). {keys_note} — use numeric action indices in code, "
            "not letter matching between option_keys labels and gamble_A/gamble_B names."
        )
    if schema == "D":
        return (
            f"action=0 -> flip/turn card (key {k0}); "
            f"action=1 -> stop/claim payout (key {k1}); return P(action=1)=P(stop)."
        )
    if schema == "B":
        subtype = _schema_b_subtype(problem)
        if subtype == "weather":
            return (
                f"action=0 -> rainy prediction (key {k0}); "
                f"action=1 -> fine prediction (key {k1}); return P(action=1)."
            )
        if subtype == "tree":
            return (
                f"action=0 -> reject tree (key {k0}); "
                f"action=1 -> accept/plant tree (key {k1}); return P(action=1)."
            )
        if subtype == "product":
            return (
                f"action=0 -> product {k0}; action=1 -> product {k1}; return P(action=1)."
            )
    if schema == "C":
        return (
            f"action=0 -> machine {k0}; action=1 -> machine {k1}; return P(action=1)."
        )
    return (
        f"action=0 -> option_keys[0] ({k0}); "
        f"action=1 -> option_keys[1] ({k1}); return P(action=1)."
    )


def summarize_runtime_schema_for_prompt(trials: List[Dict[str, Any]]) -> str:
    """Markdown-friendly summary of problem/history schema from parsed trial examples."""
    if not trials:
        return "- (no parsed trial examples provided)"

    schemas: set = set()
    problem_keys: set = set()
    history_keys: set = set()
    option_keys_samples: List[List[str]] = []
    has_gamble = False

    for trial in trials:
        p = trial["problem"]
        schemas.add(str(p.get("schema_type", "?")))
        for key in p:
            if key in ("dataset_alias", "experiment_id"):
                continue
            problem_keys.add(key)
            if key in ("gamble_A", "gamble_B"):
                has_gamble = True
        keys = p.get("option_keys")
        if isinstance(keys, list) and keys and keys not in option_keys_samples:
            option_keys_samples.append(list(keys))
        for entry in trial.get("history", []):
            if isinstance(entry, dict):
                history_keys.update(entry.keys())

    lines = [
        f"- schema_type(s): {', '.join(sorted(schemas))}",
        f"- is_gamble_A/B_task: {has_gamble}",
        f"- problem keys observed: {sorted(problem_keys)}",
    ]
    if option_keys_samples:
        lines.append(f"- option_keys example(s): {option_keys_samples[:3]}")
    if history_keys:
        core_hist = sorted(k for k in history_keys if k in ("action", "feedback"))
        extra_hist = sorted(k for k in history_keys if k not in ("action", "feedback"))
        if core_hist:
            lines.append(f"- history core keys: {core_hist}")
        if extra_hist:
            lines.append(
                f"- history may also carry prior-trial context fields: {extra_hist}"
            )

    for trial in trials:
        keys = trial["problem"].get("option_keys", [])
        if isinstance(keys, list) and len(keys) >= 2:
            schema = str(trial["problem"].get("schema_type", "?"))
            sem = _action_semantics_for_schema(
                keys, schema, trial["problem"], is_gamble=has_gamble
            )
            lines.append(f"- action semantics: {sem}")
            break

    if has_gamble:
        lines.append(
            "- gamble tasks: problem includes gamble_A/gamble_B dicts with probs/rewards; "
            "probs may be None for unknown probabilities."
        )
    else:
        lines.append(
            "- not a gamble task: do NOT document gamble_A/gamble_B (absent from examples)."
        )

    return "\n".join(lines)


def format_trial_for_prompt(trial: Dict[str, Any], index: int) -> str:
    """One-line summary of a parsed trial for infer_single_choice prompt generation."""
    p = trial["problem"]
    schema = p.get("schema_type", "?")
    action = trial["action"]
    keys = p.get("option_keys", [])
    key_lbl = _action_key_label(keys, action)
    hist = trial.get("history", [])
    hist_len = len(hist)
    hist_fb = hist[-1].get("feedback") if hist else None

    if schema == "A":
        ga = p.get("gamble_A", {})
        gb = p.get("gamble_B", {})
        if "gamble_A" in p or "gamble_B" in p:
            return (
                f"{index}. [gamble/A] gamble_A probs={ga.get('probs')} rewards={ga.get('rewards')}; "
                f"gamble_B probs={gb.get('probs')} rewards={gb.get('rewards')}; "
                f"option_keys={keys}; has_feedback={p.get('has_feedback')}; "
                f"action={action} (key={key_lbl}); history_len={hist_len}"
                + (f"; last_feedback={hist_fb}" if hist_fb is not None else "")
            )
        return (
            f"{index}. [binary/A] option_keys={keys}; problem_keys={sorted(k for k in p if k not in ('dataset_alias', 'experiment_id'))}; "
            f"action={action} (key={key_lbl}); history_len={hist_len}"
        )

    if schema == "B":
        subtype = _schema_b_subtype(p)
        if subtype == "weather":
            return (
                f"{index}. [weather/B] cards={p.get('cards')}; weather_outcome={p.get('weather_outcome')}; "
                f"was_correct={p.get('was_correct')}; option_keys={keys}; "
                f"action={action} (key={key_lbl}); history_len={hist_len}"
                + (f"; last_feedback={hist_fb}" if hist_fb is not None else "")
            )
        if subtype == "tree":
            return (
                f"{index}. [tree/B] tree_features={p.get('tree_features')}; garden={p.get('garden')}; "
                f"phase={p.get('phase')}; option_keys={keys}; "
                f"action={action} (key={key_lbl}); history_len={hist_len}"
                + (f"; last_feedback={hist_fb}" if hist_fb is not None else "")
            )
        return (
            f"{index}. [product/B] ratings_A={p.get('ratings_A')}; ratings_B={p.get('ratings_B')}; "
            f"option_keys={keys}; action={action} (key={key_lbl}); history_len={hist_len}"
        )

    if schema == "C":
        return (
            f"{index}. [bandit/C] game_id={p.get('game_id')}; n_trials_game={p.get('n_trials_game')}; "
            f"phase={p.get('phase')}; trial_index={p.get('trial_index')}; payoff={p.get('payoff')}; "
            f"machine_options={p.get('machine_options')}; option_keys={keys}; "
            f"action={action} (key={key_lbl}); history_len={hist_len}"
            + (f"; last_feedback={hist_fb}" if hist_fb is not None else "")
        )

    if schema == "D":
        k0, k1 = (keys[0], keys[1]) if len(keys) >= 2 else ("?", "?")
        return (
            f"{index}. [cct/D] round_id={p.get('round_id')}; current_score={p.get('current_score')}; "
            f"cards_flipped={p.get('cards_flipped')}; n_cards_remaining={p.get('n_cards_remaining')}; "
            f"gain_amount={p.get('gain_amount')}; loss_amount={p.get('loss_amount')}; "
            f"n_loss_cards={p.get('n_loss_cards')}; option_keys={keys} "
            f"(action=0 flip {k0}, action=1 stop {k1}); "
            f"action={action} (key={key_lbl}); history_len={hist_len}"
        )

    meta_keys = sorted(k for k in p if k not in ("dataset_alias", "experiment_id"))
    return (
        f"{index}. [schema={schema}] option_keys={keys}; problem_keys={meta_keys}; "
        f"action={action} (key={key_lbl}); history_len={hist_len}"
    )


def format_trials_for_prompt(trials: List[Dict[str, Any]], max_trials: int = 8) -> str:
    lines = [
        format_trial_for_prompt(t, i + 1)
        for i, t in enumerate(trials[:max_trials])
    ]
    return "\n".join(lines)
