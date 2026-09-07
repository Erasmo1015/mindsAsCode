"""
Schulz et al. (2020) Experiment 4 — 8-arm bandit (Psych-101 execution source).

Psych-101 experiment id: schulz2020finding/exp4.csv
Original reference (validation only): ericschulz/banditdata dynamicdata.csv

Raw Psych-101 coding:
  pressed keys <<1>> .. <<8>>
Internal TEH coding:
  action = int(press) - 1  ∈ {0..7}

History resets every round (30 rounds × 10 pulls).
Do NOT expose cond / rcond / SRS-RSR structure labels (absent from participant text).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data_modules.psych101_binary import (
    ParsedTrial,
    PsychBlock,
    PsychExperiment,
    split_psych_experiment,
)

DATASET_ALIAS = "13schulz2020finding"
EXPERIMENT_ID = "schulz2020finding/exp4.csv"
DISPLAY_NAME = "Schulz et al. (2020) Exp4 8-arm bandit"
OUTPUT_TYPE = "categorical"
N_ACTIONS = 8
SPLIT_UNIT = "round"
RAW_PRESS_MIN = 1
RAW_PRESS_MAX = 8

DEFAULT_VALIDATION_CSV = "datasets/external/schulz2020_exp4/dynamicdata.csv"

TASK_DESCRIPTION = (
    "Eight-armed bandit over 30 rounds of 10 trials. On each trial choose one option "
    "(keys 1–8 in the transcript; internal actions 0–7) and observe a numeric reward. "
    "Options reset each round. choose(problem, history) returns a probability "
    "distribution over actions 0..7. History is within-round only."
)

_RE_ROUND = re.compile(r"You are playing round\s+(\d+)\s*:", re.I)
_RE_PRESS = re.compile(
    r"You press <<([1-8])>> and get (-?\d+(?:\.\d+)?) points\.?",
    re.I,
)

_OPTION_DICTS = [{"action": i} for i in range(N_ACTIONS)]
_OPTION_KEYS = list(range(N_ACTIONS))


def raw_press_to_action(raw_press: int | str) -> int:
    p = int(raw_press)
    if p < RAW_PRESS_MIN or p > RAW_PRESS_MAX:
        raise ValueError(f"raw press must be in 1..8, got {p}")
    return p - 1


def parse_schulz2020_exp4_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    text = str(row["text"])
    m0 = _RE_ROUND.search(text)
    instruction = text[: m0.start()].strip() if m0 else text.split("\n\n")[0].strip()
    parts = re.split(r"(?=You are playing round\s+\d+\s*:)", text)
    blocks: List[PsychBlock] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        rm = _RE_ROUND.match(part)
        if not rm:
            continue
        round_id = int(rm.group(1))
        presses = list(_RE_PRESS.finditer(part))
        if len(presses) != 10:
            raise ValueError(
                f"schulz exp4 round {round_id}: expected 10 presses, got {len(presses)}"
            )
        trials: List[ParsedTrial] = []
        for trial_idx, pm in enumerate(presses, start=1):
            action = raw_press_to_action(pm.group(1))
            reward = float(pm.group(2))
            trials.append(
                ParsedTrial(
                    action=action,
                    feedback=reward,
                    # Participant-visible state only (no current reward / no cond labels).
                    problem_fields={
                        "round": round_id,
                        "trial": trial_idx,
                    },
                )
            )
        static = {
            "schema_type": "categorical_bandit",
            "option_keys": list(_OPTION_KEYS),
            "round": round_id,
            "n_arms": N_ACTIONS,
            "has_feedback": True,
            "raw_press_coding": "1-8",
            "internal_action_coding": "0-7",
        }
        blocks.append(
            PsychBlock(
                trials=trials,
                option_keys=list(_OPTION_KEYS),
                problem_static=static,
                schema_type="categorical_bandit",
            )
        )
    if len(blocks) != 30:
        raise ValueError(f"schulz exp4: expected 30 rounds, got {len(blocks)}")
    total = sum(len(b.trials) for b in blocks)
    if total != 300:
        raise ValueError(f"schulz exp4: expected 300 trials, got {total}")
    return PsychExperiment(
        instruction=instruction,
        blocks=blocks,
        dataset_alias=dataset_alias,
        schema_type="categorical_bandit",
    )


def finalize_schulz_categorical_trials(
    trials: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert Psych-101 trial dicts to Steyvers-compatible categorical shape.

    - problem['options'] = [{'action': i}, ...] for i in 0..7
    - history entries use {'action', 'reward'} only
    - strip any experimenter condition fields if present
    """
    banned = {
        "cond",
        "rcond",
        "structure",
        "srs",
        "rsr",
        "pos",
        "neg",
        "ran",
        "block",
        "reward",
        "feedback",
        "out",
    }
    out: List[Dict[str, Any]] = []
    for t in trials:
        problem = dict(t.get("problem") or {})
        for k in list(problem.keys()):
            if k.lower() in banned or k.lower().startswith("reward_rate"):
                problem.pop(k, None)
        problem["n_arms"] = N_ACTIONS
        problem["options"] = [dict(o) for o in _OPTION_DICTS]
        problem["option_keys"] = list(_OPTION_KEYS)
        problem["has_feedback"] = True
        problem.setdefault("raw_press_coding", "1-8")
        problem.setdefault("internal_action_coding", "0-7")
        history = []
        for h in t.get("history") or []:
            reward = h.get("reward", h.get("feedback"))
            if reward is None:
                raise ValueError("schulz history entry missing reward/feedback")
            history.append(
                {
                    "action": int(h["action"]),
                    "reward": float(reward),
                }
            )
        action = int(t["action"])
        if action < 0 or action >= N_ACTIONS:
            raise ValueError(f"action out of range: {action}")
        out.append(
            {
                "problem": problem,
                "history": history,
                "options": list(_OPTION_KEYS),
                "action": action,
            }
        )
    return out


def split_schulz2020_exp4_experiment(
    exp: PsychExperiment,
    *,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """Split by whole rounds; categorical finalize runs inside split_psych_experiment."""
    train, val, test, options = split_psych_experiment(
        exp, split_ratio=split_ratio, split_seed=split_seed
    )
    return train, val, test, list(options)


def resolve_validation_csv(path: Optional[str | Path] = None) -> Path:
    p = Path(path) if path is not None else Path(DEFAULT_VALIDATION_CSV)
    return p


def load_dynamicdata_subject_rows(
    subject_id: int,
    *,
    csv_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Load one subject from author dynamicdata.csv (validation/reference only)."""
    import csv

    path = resolve_validation_csv(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Schulz validation CSV not found at {path}. "
            "Run: python scripts/setup_external_schulz2020_exp4.py"
        )
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["id"]) != int(subject_id):
                continue
            rows.append(
                {
                    "id": int(row["id"]),
                    "trial": int(row["trial"]),
                    "round": int(row["round"]),
                    "arm": int(row["arm"]),  # 1..8 (raw key coding)
                    "out": float(row["out"]),
                    # cond/rcond/block intentionally ignored for TEH problem/history
                }
            )
    rows.sort(key=lambda r: (r["round"], r["trial"]))
    if len(rows) != 300:
        raise ValueError(
            f"dynamicdata subject {subject_id}: expected 300 rows, got {len(rows)}"
        )
    return rows


def fingerprint_reward_sequence(rewards: Sequence[float], *, ndigits: int = 6) -> Tuple[float, ...]:
    return tuple(round(float(r), ndigits) for r in rewards)


def match_psych_row_to_dynamicdata_id(
    exp: PsychExperiment,
    *,
    csv_path: Optional[str | Path] = None,
) -> int:
    """
    Map a parsed Psych-101 participant to dynamicdata `id` via reward fingerprint.

    Psych-101 `participant` field is not the author CSV id.
    """
    import csv

    psych_rewards: List[float] = []
    for block in exp.blocks:
        for trial in block.trials:
            psych_rewards.append(float(trial.feedback))
    target = fingerprint_reward_sequence(psych_rewards)

    path = resolve_validation_csv(csv_path)
    by_id: Dict[int, List[float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = int(row["id"])
            by_id.setdefault(sid, []).append(float(row["out"]))

    matches = [
        sid
        for sid, outs in by_id.items()
        if fingerprint_reward_sequence(outs) == target
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one dynamicdata id matching Psych rewards; got {matches}"
        )
    return matches[0]


def assert_psych_matches_dynamicdata(
    exp: PsychExperiment,
    *,
    csv_path: Optional[str | Path] = None,
    subject_id: Optional[int] = None,
    reward_atol: float = 1e-6,
) -> int:
    """
    Compare Psych-101 parsed trials to dynamicdata.csv on arm, reward, order, rounds.

    Returns matched dynamicdata subject id.
    """
    sid = (
        int(subject_id)
        if subject_id is not None
        else match_psych_row_to_dynamicdata_id(exp, csv_path=csv_path)
    )
    ref = load_dynamicdata_subject_rows(sid, csv_path=csv_path)
    flat: List[ParsedTrial] = []
    for block in exp.blocks:
        flat.extend(block.trials)
    if len(flat) != len(ref):
        raise AssertionError(f"trial count mismatch: psych={len(flat)} csv={len(ref)}")

    for i, (pt, rr) in enumerate(zip(flat, ref)):
        raw_press = int(pt.action) + 1
        if raw_press != int(rr["arm"]):
            raise AssertionError(
                f"row {i}: arm mismatch psych_press={raw_press} csv_arm={rr['arm']}"
            )
        if abs(float(pt.feedback) - float(rr["out"])) > reward_atol:
            raise AssertionError(
                f"row {i}: reward mismatch psych={pt.feedback} csv={rr['out']}"
            )
        # Round boundaries: every 10 pulls advance round; trial 1..10 within round.
        expected_round = (i // 10) + 1
        expected_trial = (i % 10) + 1
        if int(rr["round"]) != expected_round or int(rr["trial"]) != expected_trial:
            raise AssertionError(
                f"row {i}: csv round/trial={rr['round']}/{rr['trial']} "
                f"expected {expected_round}/{expected_trial}"
            )
        pf_round = int(pt.problem_fields.get("round", -1))
        pf_trial = int(pt.problem_fields.get("trial", -1))
        if pf_round != expected_round or pf_trial != expected_trial:
            raise AssertionError(
                f"row {i}: psych round/trial={pf_round}/{pf_trial} "
                f"expected {expected_round}/{expected_trial}"
            )
    return sid
