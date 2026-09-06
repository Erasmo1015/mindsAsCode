"""
Steyvers, Lee & Wagenmakers (2009) — 4-arm bandit.

Local TEH adapter (categorical):
  choose(problem, history) -> dict[int, float] over arms {0,1,2,3}

Raw CSV coding:
  choice ∈ {1,2,3,4}  (1-indexed arm id)
Internal TEH coding:
  action = choice - 1 ∈ {0,1,2,3}

Never expose rewardRateChosen / gameRewardRates / future rewards.
History resets at each game; contains only prior (action, reward).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_modules.mixed_gambles import three_way_unit_counts

DATASET_ALIAS = "steyvers_2009_bandit"
DATASET_NAME = DATASET_ALIAS
DISPLAY_NAME = "Steyvers, Lee & Wagenmakers (2009) 4-arm bandit"
OUTPUT_TYPE = "categorical"
N_ACTIONS = 4
SPLIT_UNIT = "game"
RAW_CHOICE_MIN = 1
RAW_CHOICE_MAX = 4

DEFAULT_DATA_DIR = "datasets/external/steyvers_2009_bandit"
TASK_DESCRIPTION = (
    "Four-armed bandit: each game has 15 trials. On each trial choose one arm "
    "and observe a binary reward. History resets every game. "
    "choose(problem, history) returns a probability distribution over actions "
    "{0,1,2,3} (internal 0-based arms; raw CSV choices were 1-4)."
)

_OPTION_DICTS = [{"action": i} for i in range(N_ACTIONS)]
_OPTION_KEYS = list(range(N_ACTIONS))


def resolve_data_dir(data_dir: Optional[str | Path] = None) -> Path:
    if data_dir is None:
        return Path(DEFAULT_DATA_DIR)
    return Path(data_dir)


def resolve_trials_path(data_dir: Optional[str | Path] = None) -> Path:
    p = resolve_data_dir(data_dir) / "trials.csv"
    if not p.is_file():
        raise FileNotFoundError(
            f"Steyvers trials.csv not found at {p}. "
            "Run: python scripts/setup_external_steyvers_2009_bandit.py"
        )
    return p


def raw_choice_to_action(raw_choice: int) -> int:
    c = int(raw_choice)
    if c < RAW_CHOICE_MIN or c > RAW_CHOICE_MAX:
        raise ValueError(f"raw choice must be in 1..4, got {c}")
    return c - 1


def list_participant_ids(data_dir: Optional[str | Path] = None) -> List[int]:
    path = resolve_trials_path(data_dir)
    ids: set[int] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.add(int(row["participant"]))
    return sorted(ids)


def load_participant_raw_trials(
    participant_id: int,
    *,
    data_dir: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    path = resolve_trials_path(data_dir)
    by_game: Dict[int, List[Dict[str, str]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["participant"]) != int(participant_id):
                continue
            # Intentionally ignore rewardRateChosen (latent generating rate).
            g = int(row["game"])
            by_game.setdefault(g, []).append(row)

    if not by_game:
        raise ValueError(
            f"No Steyvers trials for participant {participant_id} under {resolve_data_dir(data_dir)}"
        )

    out: List[Dict[str, Any]] = []
    for game in sorted(by_game):
        rows = sorted(by_game[game], key=lambda r: int(r["trial"]))
        if len(rows) != 15:
            raise ValueError(
                f"participant {participant_id} game {game}: expected 15 trials, got {len(rows)}"
            )
        history: List[Dict[str, Any]] = []
        for row in rows:
            trial_idx = int(row["trial"])
            action = raw_choice_to_action(int(row["choice"]))
            reward = int(row["reward"])
            if reward not in (0, 1):
                raise ValueError(f"reward must be 0/1, got {reward}")
            problem = {
                "dataset_alias": DATASET_ALIAS,
                "game": int(game),
                "trial": int(trial_idx),
                "n_arms": N_ACTIONS,
                "options": [dict(o) for o in _OPTION_DICTS],
                "option_keys": list(_OPTION_KEYS),
                "has_feedback": True,
                # Document coding for prompts / auditors.
                "raw_choice_coding": "1-4",
                "internal_action_coding": "0-3",
            }
            out.append(
                {
                    "problem": problem,
                    "history": [dict(h) for h in history],
                    "options": list(_OPTION_KEYS),
                    "action": int(action),
                    "participant_id": int(participant_id),
                    "problem_signature": int(game),
                }
            )
            # Only participant-observable feedback enters history.
            history.append({"action": int(action), "reward": int(reward)})
    n_games = len({t["problem_signature"] for t in out})
    if n_games != 20:
        raise ValueError(
            f"participant {participant_id}: expected 20 games, got {n_games}"
        )
    if len(out) != 300:
        raise ValueError(
            f"participant {participant_id}: expected 300 trials, got {len(out)}"
        )
    return out


def load_steyvers_2009_bandit_trials(
    participant_id: int,
    *,
    data_dir: Optional[str | Path] = None,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """
    Load one participant; split by whole games (never split trials within a game).

    Returns (train_trials, val_trials, test_trials, option_keys).
    """
    all_trials = load_participant_raw_trials(participant_id, data_dir=data_dir)
    signatures = sorted({int(t["problem_signature"]) for t in all_trials})
    if len(signatures) < 3:
        raise ValueError(
            f"steyvers participant {participant_id} has <3 games; cannot split."
        )
    rng = np.random.default_rng(split_seed)
    shuffled = list(signatures)
    rng.shuffle(shuffled)
    n_train, n_val, n_test = three_way_unit_counts(len(shuffled), split_ratio)
    train_sigs = set(shuffled[:n_train])
    val_sigs = set(shuffled[n_train : n_train + n_val])
    test_sigs = set(shuffled[n_train + n_val :])
    assert not (train_sigs & val_sigs)
    assert not (train_sigs & test_sigs)
    assert not (val_sigs & test_sigs)

    def _filter(sigs: set) -> List[Dict[str, Any]]:
        out = []
        for t in all_trials:
            if int(t["problem_signature"]) not in sigs:
                continue
            tt = dict(t)
            tt.pop("problem_signature", None)
            tt.pop("participant_id", None)
            out.append(tt)
        return out

    return _filter(train_sigs), _filter(val_sigs), _filter(test_sigs), list(_OPTION_KEYS)


load_steyvers_trials = load_steyvers_2009_bandit_trials
