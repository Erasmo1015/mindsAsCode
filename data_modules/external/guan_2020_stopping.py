"""
Guan et al. (2020) optimal stopping — author RiskProject MAT adapter.

Source of truth:
  datasets/external/guan_2020_stopping/OptimalStopping.mat
  (from https://github.com/maimeguan/RiskProject Data/OptimalStopping.mat)

Do NOT use Michael behavioralDataRepository trials.csv (missing value5–8 for L=8).

Bernoulli API:
  choose(problem, history) -> float = P(action=1) = P(stop)

Action coding:
  0 = continue
  1 = stop

Each stopping problem expands to decision positions 1..stop_position.
At position t, problem.values_observed contains ONLY values 1..t.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_modules.mixed_gambles import three_way_unit_counts

DATASET_ALIAS = "guan_2020_stopping"
DATASET_NAME = DATASET_ALIAS
DISPLAY_NAME = "Guan et al. (2020) optimal stopping"
OUTPUT_TYPE = "bernoulli"
N_ACTIONS = 2
SPLIT_UNIT = "stopping_problem"

DEFAULT_DATA_DIR = "datasets/external/guan_2020_stopping"
DEFAULT_MAT_NAME = "OptimalStopping.mat"

CONDITION_NAMES = (
    "length4_neutral",
    "length4_plentiful",
    "length8_neutral",
    "length8_plentiful",
)

TASK_DESCRIPTION = (
    "Secretary-style optimal stopping: observe sequence values one-by-one and "
    "decide continue (0) or stop (1). Sequence length is 4 or 8; environment is "
    "neutral or plentiful. choose(problem, history) returns P(stop). "
    "History is prior continue decisions within the current stopping problem only."
)


def resolve_data_dir(data_dir: Optional[str | Path] = None) -> Path:
    if data_dir is None:
        return Path(DEFAULT_DATA_DIR)
    return Path(data_dir)


def resolve_mat_path(data_dir: Optional[str | Path] = None) -> Path:
    d = resolve_data_dir(data_dir)
    p = d / DEFAULT_MAT_NAME
    if not p.is_file():
        raise FileNotFoundError(
            f"Guan OptimalStopping.mat not found at {p}. "
            "Run: python scripts/setup_external_guan_2020_stopping.py"
        )
    return p


def _load_mat_struct(mat_path: Path) -> Any:
    from scipy.io import loadmat

    raw = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "OptimalStopping" not in raw:
        raise KeyError(f"{mat_path} missing OptimalStopping struct")
    return raw["OptimalStopping"]


def _subject_complete(dec: np.ndarray, stim: np.ndarray, nstim: np.ndarray, s: int) -> bool:
    for c in range(int(nstim.shape[0])):
        L = int(nstim[c])
        if not np.isfinite(dec[s, :, c]).all():
            return False
        if not np.isfinite(stim[s, :, c, :L]).all():
            return False
    return True


def _build_participant_index(os_struct: Any) -> List[Dict[str, Any]]:
    """
    Map TEH participant_id 1..56 to MAT subject indices with complete data.

    Alignment verified against Michael CSV participant IDs (same people).
    Incomplete MAT rows (2 of 58) are dropped.
    """
    dec = np.asarray(os_struct.decisions, dtype=float)
    stim = np.asarray(os_struct.stimuli, dtype=float)
    nstim = np.asarray(os_struct.nstim).astype(int)
    ok = [s for s in range(dec.shape[0]) if _subject_complete(dec, stim, nstim, s)]
    if len(ok) != 56:
        raise ValueError(f"Expected 56 complete OptimalStopping subjects, found {len(ok)}")
    return [
        {
            "participant_id": i + 1,
            "mat_subject_index": int(s),
        }
        for i, s in enumerate(ok)
    ]


def load_guan_raw(data_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    mat_path = resolve_mat_path(data_dir)
    os_struct = _load_mat_struct(mat_path)
    dec = np.asarray(os_struct.decisions, dtype=float)
    stim = np.asarray(os_struct.stimuli, dtype=float)
    nstim = np.asarray(os_struct.nstim).astype(int)
    if int(os_struct.ncond) != 4 or int(os_struct.nprob) != 40:
        raise ValueError(
            f"Unexpected OptimalStopping shape: ncond={os_struct.ncond}, nprob={os_struct.nprob}"
        )
    if tuple(int(x) for x in nstim) != (4, 4, 8, 8):
        raise ValueError(f"Unexpected nstim={nstim}")
    participants = _build_participant_index(os_struct)
    return {
        "mat_path": mat_path,
        "decisions": dec,
        "stimuli": stim,
        "nstim": nstim,
        "participants": participants,
        "condition_names": CONDITION_NAMES,
        "condition_note": str(getattr(os_struct, "condition", "")),
        "distributions_note": str(getattr(os_struct, "distributions", "")),
    }


def list_participant_ids(data_dir: Optional[str | Path] = None) -> List[int]:
    return [int(p["participant_id"]) for p in load_guan_raw(data_dir)["participants"]]


def _participant_mat_index(raw: Dict[str, Any], participant_id: int) -> int:
    for p in raw["participants"]:
        if int(p["participant_id"]) == int(participant_id):
            return int(p["mat_subject_index"])
    raise ValueError(f"Unknown Guan participant_id={participant_id}")


def _environment_for_condition(condition: str) -> str:
    if condition.endswith("_plentiful"):
        return "plentiful"
    if condition.endswith("_neutral"):
        return "neutral"
    raise ValueError(f"Unrecognized condition {condition!r}")


def expand_stopping_problem(
    *,
    participant_id: int,
    condition_index: int,
    condition: str,
    problem_id: int,
    sequence_length: int,
    stop_position: int,
    full_values: Sequence[float],
) -> List[Dict[str, Any]]:
    """Expand one stopping problem into continue/stop decision trials."""
    L = int(sequence_length)
    S = int(stop_position)
    if L not in (4, 8):
        raise ValueError(f"sequence_length must be 4 or 8, got {L}")
    if not (1 <= S <= L):
        raise ValueError(f"stop_position {S} out of range for L={L}")
    values = [float(full_values[i]) for i in range(L)]
    if any(not np.isfinite(v) for v in values):
        raise ValueError("non-finite stimulus values in stopping problem")

    env = _environment_for_condition(condition)
    unit_key = (int(condition_index), int(problem_id))
    option_keys = [0, 1]
    trials: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    for t in range(1, S + 1):
        observed = values[:t]
        # Leakage invariant: never expose future values.
        assert len(observed) == t
        action = 1 if t == S else 0
        problem = {
            "dataset_alias": DATASET_ALIAS,
            "condition": condition,
            "condition_index": int(condition_index),
            "environment": env,
            "problem_id": int(problem_id),
            "sequence_length": L,
            "position": int(t),
            "values_observed": list(observed),
            "option_keys": option_keys,
            "has_feedback": False,
        }
        trials.append(
            {
                "problem": problem,
                "history": [dict(h) for h in history],
                "options": option_keys,
                "action": int(action),
                "participant_id": int(participant_id),
                "problem_signature": unit_key,
                # analysis-only (stripped before return from split loader)
                "_full_values_len": L,
                "_stop_position": S,
            }
        )
        if action == 0:
            history.append(
                {
                    "action": 0,
                    "position": int(t),
                    "value": float(values[t - 1]),
                }
            )
    # Exactly S-1 continues + 1 stop.
    assert sum(1 for t in trials if t["action"] == 0) == S - 1
    assert trials[-1]["action"] == 1
    return trials


def load_participant_raw_trials(
    participant_id: int,
    *,
    data_dir: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    raw = load_guan_raw(data_dir)
    s = _participant_mat_index(raw, participant_id)
    dec = raw["decisions"]
    stim = raw["stimuli"]
    nstim = raw["nstim"]
    out: List[Dict[str, Any]] = []
    for c, condition in enumerate(CONDITION_NAMES):
        L = int(nstim[c])
        for p in range(40):
            stop_pos = int(dec[s, p, c])
            full_vals = [float(stim[s, p, c, i]) for i in range(L)]
            out.extend(
                expand_stopping_problem(
                    participant_id=participant_id,
                    condition_index=c,
                    condition=condition,
                    problem_id=p + 1,
                    sequence_length=L,
                    stop_position=stop_pos,
                    full_values=full_vals,
                )
            )
    # 160 stopping problems / person
    n_units = len({t["problem_signature"] for t in out})
    if n_units != 160:
        raise ValueError(
            f"Guan participant {participant_id}: expected 160 stopping problems, got {n_units}"
        )
    return out


def load_guan_2020_stopping_trials(
    participant_id: int,
    *,
    data_dir: Optional[str | Path] = None,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """
    Load one participant; split by complete stopping problem units.

    Returns (train_trials, val_trials, test_trials, option_keys).
    """
    all_trials = load_participant_raw_trials(participant_id, data_dir=data_dir)
    signatures = sorted({t["problem_signature"] for t in all_trials})
    if len(signatures) < 3:
        raise ValueError(
            f"guan participant {participant_id} has <3 stopping problems; cannot split."
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
            if t["problem_signature"] not in sigs:
                continue
            tt = dict(t)
            tt.pop("problem_signature", None)
            tt.pop("participant_id", None)
            tt.pop("_full_values_len", None)
            tt.pop("_stop_position", None)
            out.append(tt)
        return out

    option_keys = [0, 1]
    return _filter(train_sigs), _filter(val_sigs), _filter(test_sigs), option_keys


load_guan_trials = load_guan_2020_stopping_trials
