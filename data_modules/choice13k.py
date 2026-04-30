"""
Choice13k dataset loader replicated from llm_evo_cog (read-only reference).
"""
import math
import os
import re
from pathlib import Path
from typing import List, NamedTuple, Optional

from datasets import load_dataset, load_from_disk


def _hf_token_for_datasets():
    """Read token from env (batch jobs); same vars as Hugging Face CLI / hub."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class Gamble(NamedTuple):
    probs: List[float]
    rewards: List[float]


class Trial(NamedTuple):
    action: int
    feedback: Optional[float]
    history: str


class Block(NamedTuple):
    trials: List[Trial]
    gamble_A: Gamble
    gamble_B: Gamble
    has_feedback: bool
    option_keys: List[str]   # e.g. ["P", "U"]
    gamble_info_text: str


class Experiment(NamedTuple):
    blocks: List[Block]
    instruction: str


def _extract_gamble_info(gamble_info_str: str):
    gamble_info_str = gamble_info_str.strip()
    lines = [l.strip() for l in gamble_info_str.splitlines() if l.strip().startswith("Option")]
    gambles = {}
    option_keys = []

    for line in lines:
        m = re.match(r"Option\s+([A-Z]) delivers (.*)", line)
        if not m:
            raise ValueError(f"Could not parse gamble line: {line}")
        opt, outcomes_str = m.groups()
        option_keys.append(opt)

        probs, rewards = [], []
        matches = re.findall(r"(-?\d+\.?\d*).*?with\s+(\d+\.?\d*)% chance", outcomes_str)

        if matches:
            for reward, prob in matches:
                rewards.append(float(reward))
                probs.append(float(prob) / 100.0)
            assert math.isclose(sum(probs), 1.0, rel_tol=1e-4), f"Probs do not add up to 1, Probs: {probs}, Sum: {sum(probs)}"
        else:
            rewards = [float(r) for r in re.findall(r"(-?\d+\.?\d*) points", outcomes_str)]
            probs = None

        gambles[opt] = Gamble(probs=probs, rewards=rewards)

    gamble_A, gamble_B = [gambles[k] for k in option_keys]
    return gamble_A, gamble_B, option_keys


def _extract_trials(trials_str: str, gamble_info: str, option_keys: List[str]):
    trials = []
    trial_regex = r"(You press <<([A-Z])>>\..*?(?:You receive (-?\d+\.?\d*) points.*?)?(?:\n|$))"
    history_prefix = gamble_info.strip() + "\n"

    for match in re.finditer(trial_regex, trials_str, flags=re.DOTALL):
        full_trial_str = match.group(1).strip()
        key = match.group(2)
        action = option_keys.index(key)
        feedback = float(match.group(3)) if match.group(3) else None
        history = history_prefix.strip()
        trials.append(Trial(action=action, feedback=feedback, history=history))
        history_prefix += full_trial_str + "\n"

    return trials


def _extract_has_feedback(block_str: str):
    return "You receive" in block_str


def _convert_to_block(block_str: str) -> Block:
    gamble_info = block_str.split("\nYou press")[0]
    trials_str = block_str[len(gamble_info):].lstrip()

    gamble_A, gamble_B, option_keys = _extract_gamble_info(gamble_info)
    trials = _extract_trials(trials_str, gamble_info, option_keys)

    return Block(
        trials=trials,
        gamble_A=gamble_A,
        gamble_B=gamble_B,
        has_feedback=_extract_has_feedback(block_str),
        option_keys=option_keys,
        gamble_info_text=gamble_info.strip(),
    )


def _convert_to_experiment(data) -> Experiment:
    data['text'] = data['text'].replace("\n\n\n\nOption", "\n\nOption")
    data['text'] = data['text'].replace("\n\n\nOption", "\n\nOption")
    instruction = data['text'].split("\n\nOption")[0]
    trials_str = data['text'][len(instruction):].lstrip()

    blocks = []
    for block_str in trials_str.split("\n\n"):
        blocks.append(_convert_to_block(block_str))

    return Experiment(
        instruction=instruction,
        blocks=blocks
    )


def get_choice13k_experiments(n_participants: int = 10, local_dataset: Optional[str] = None):
    if local_dataset:
        dataset_path = Path(local_dataset).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Local dataset path does not exist: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))
    else:
        tok = _hf_token_for_datasets()
        ds_kw = {"token": tok} if tok else {}
        dataset = load_dataset("marcelbinz/Psych-101-test", **ds_kw)
    test_split = dataset['test']
    choices13k_ds = test_split.filter(lambda ex: ex['experiment'] == 'peterson2021using/exp1.csv')

    experiments = []
    for i in range(min(n_participants, len(choices13k_ds))):
        experiments.append(_convert_to_experiment(choices13k_ds[i]))
    return experiments

