"""
Bergert & Nosofsky (2007) — pairwise cue-based multi-attribute choice.

Local TEH adapter (Bernoulli): choose(problem, history) -> float = P(action=1).

Action coding (preserve CSV `choice`; verified vs cue1-TTB):
  action=1 -> option_A (alternativeA)
  action=0 -> option_B (alternativeB)

History is empty: each pairwise problem is treated as an independent decision unit
(no within-problem sequential state in this extract).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_modules.mixed_gambles import three_way_unit_counts

DATASET_ALIAS = "bergert_nosofsky_2007"
DATASET_NAME = DATASET_ALIAS
DISPLAY_NAME = "Bergert & Nosofsky (2007) cue-based pairwise choice"
OUTPUT_TYPE = "bernoulli"
N_ACTIONS = 2
SPLIT_UNIT = "problem"

DEFAULT_DATA_DIR = "datasets/external/bergert_nosofsky_2007"
TASK_DESCRIPTION = (
    "Pairwise multi-attribute choice: two alternatives with six binary cues. "
    "Participants choose option_A or option_B. "
    "action=1 means choose option_A; action=0 means choose option_B. "
    "choose(problem, history) returns P(action=1). History is empty."
)

_CUE_KEYS = ("cue1", "cue2", "cue3", "cue4", "cue5", "cue6")


def resolve_data_dir(data_dir: Optional[str | Path] = None) -> Path:
    if data_dir is None:
        return Path(DEFAULT_DATA_DIR)
    return Path(data_dir)


def _require_files(data_dir: Path) -> Dict[str, Path]:
    paths = {
        "trials": data_dir / "trials.csv",
        "problems": data_dir / "problems.csv",
        "alternatives": data_dir / "alternatives.csv",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Bergert dataset files missing. Expected under "
            f"{data_dir}. Missing: {missing}. "
            "Run: python scripts/setup_external_bergert_nosofsky_2007.py"
        )
    # Raw analysis metadata only (NOT participant-visible at decision time):
    cue_path = data_dir / "cueValidities.csv"
    if cue_path.is_file():
        paths["cue_validities"] = cue_path
    return paths


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_alternative_cues(row: Dict[str, str]) -> Dict[str, int]:
    return {k: int(row[k]) for k in _CUE_KEYS}


def load_bergert_tables(
    data_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load and index raw tables (no participant filtering)."""
    paths = _require_files(resolve_data_dir(data_dir))
    alternatives_rows = _read_csv_dicts(paths["alternatives"])
    alternatives: Dict[int, Dict[str, int]] = {}
    for row in alternatives_rows:
        alt_id = int(row["alternative"])
        alternatives[alt_id] = _parse_alternative_cues(row)

    problems: Dict[int, Tuple[int, int]] = {}
    for row in _read_csv_dicts(paths["problems"]):
        pid = int(row["problem"])
        problems[pid] = (int(row["alternativeA"]), int(row["alternativeB"]))

    # Ecological cue validities are analysis metadata for this extract.
    # Stojic et al. (2016) cite Bergert & Nosofsky (2007) as a paradigm where
    # participants *learn* cue weights rather than receiving them explicitly.
    # Do NOT expose these on choose(problem, history).
    cue_validities_meta: List[Dict[str, float]] = []
    if "cue_validities" in paths:
        for row in _read_csv_dicts(paths["cue_validities"]):
            cue_validities_meta.append(
                {
                    "cue": int(row["cue"]),
                    "validity": float(row["validity"]),
                    "log_odds_weight": float(row["logOddsWeight"]),
                }
            )
        cue_validities_meta.sort(key=lambda d: d["cue"])

    trials = _read_csv_dicts(paths["trials"])
    return {
        "alternatives": alternatives,
        "problems": problems,
        "cue_validities_meta": cue_validities_meta,
        "trials": trials,
        "paths": paths,
    }


def list_participant_ids(data_dir: Optional[str | Path] = None) -> List[int]:
    tables = load_bergert_tables(data_dir)
    ids = sorted({int(r["participant"]) for r in tables["trials"]})
    return ids


def _option_payload(alternative_id: int, cues: Dict[str, int]) -> Dict[str, Any]:
    return {
        "alternative_id": int(alternative_id),
        "cues": {k: int(cues[k]) for k in _CUE_KEYS},
    }


def build_trial_dict(
    *,
    participant_id: int,
    problem_id: int,
    alternative_a: int,
    alternative_b: int,
    choice: int,
    alternatives: Dict[int, Dict[str, int]],
) -> Dict[str, Any]:
    if alternative_a not in alternatives:
        raise KeyError(f"Unknown alternativeA={alternative_a} for problem {problem_id}")
    if alternative_b not in alternatives:
        raise KeyError(f"Unknown alternativeB={alternative_b} for problem {problem_id}")
    if choice not in (0, 1):
        raise ValueError(f"choice must be 0 or 1, got {choice}")

    option_keys = [0, 1]
    problem = {
        "dataset_alias": DATASET_ALIAS,
        "problem_id": int(problem_id),
        "option_A": _option_payload(alternative_a, alternatives[alternative_a]),
        "option_B": _option_payload(alternative_b, alternatives[alternative_b]),
        "option_keys": option_keys,
        "has_feedback": False,
        # Document inverted-vs-B naming: action 1 selects option_A.
        "action_means_option_A_when_1": True,
    }
    return {
        "problem": problem,
        "history": [],
        "options": option_keys,
        "action": int(choice),
        "participant_id": int(participant_id),
        "problem_signature": int(problem_id),
    }


def load_participant_raw_trials(
    participant_id: int,
    *,
    data_dir: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    tables = load_bergert_tables(data_dir)
    problems = tables["problems"]
    alternatives = tables["alternatives"]
    out: List[Dict[str, Any]] = []
    for row in tables["trials"]:
        if int(row["participant"]) != int(participant_id):
            continue
        problem_id = int(row["problem"])
        if problem_id not in problems:
            raise KeyError(f"trial references unknown problem_id={problem_id}")
        alt_a_p, alt_b_p = problems[problem_id]
        alt_a = int(row["alternativeA"])
        alt_b = int(row["alternativeB"])
        if (alt_a, alt_b) != (alt_a_p, alt_b_p):
            raise ValueError(
                f"participant {participant_id} problem {problem_id}: "
                f"trials.csv alternatives {(alt_a, alt_b)} != problems.csv {(alt_a_p, alt_b_p)}"
            )
        out.append(
            build_trial_dict(
                participant_id=participant_id,
                problem_id=problem_id,
                alternative_a=alt_a,
                alternative_b=alt_b,
                choice=int(row["choice"]),
                alternatives=alternatives,
            )
        )
    if not out:
        raise ValueError(
            f"No Bergert trials for participant {participant_id} under {resolve_data_dir(data_dir)}"
        )
    out.sort(key=lambda t: int(t["problem"]["problem_id"]))
    return out


def load_bergert_nosofsky_2007_trials(
    participant_id: int,
    *,
    data_dir: Optional[str | Path] = None,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """
    Load one participant; split by problem_id (40 pairwise items).

    Returns (train_trials, val_trials, test_trials, option_keys).
    """
    all_trials = load_participant_raw_trials(participant_id, data_dir=data_dir)
    signatures = sorted({int(t["problem_signature"]) for t in all_trials})
    if len(signatures) < 3:
        raise ValueError(
            f"bergert participant {participant_id} has <3 problems; cannot split."
        )
    rng = np.random.default_rng(split_seed)
    shuffled = list(signatures)
    rng.shuffle(shuffled)
    n_train, n_val, n_test = three_way_unit_counts(len(shuffled), split_ratio)
    train_sigs = set(shuffled[:n_train])
    val_sigs = set(shuffled[n_train : n_train + n_val])
    test_sigs = set(shuffled[n_train + n_val :])
    assert len(train_sigs | val_sigs | test_sigs) == len(signatures)
    assert not (train_sigs & val_sigs)
    assert not (train_sigs & test_sigs)
    assert not (val_sigs & test_sigs)

    train_trials = [t for t in all_trials if t["problem_signature"] in train_sigs]
    val_trials = [t for t in all_trials if t["problem_signature"] in val_sigs]
    test_trials = [t for t in all_trials if t["problem_signature"] in test_sigs]
    for t in train_trials + val_trials + test_trials:
        t.pop("problem_signature", None)
        t.pop("participant_id", None)
    option_keys = [0, 1]
    return train_trials, val_trials, test_trials, option_keys


# Alias used by registry / participant-id collectors
load_bergert_trials = load_bergert_nosofsky_2007_trials
