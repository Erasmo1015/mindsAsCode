#!/usr/bin/env python3
"""
Logistic MLE baselines for Choice13k, CPC18 (Track II), and Mixed Gambles.

Data loading/splitting conventions align with `Template_evo_non_strict.py` where noted:
  - choice13k within_participant: **problem (block)-level** split — train/test use disjoint gamble pairs;
    blocks are shuffled with --split_seed, assigned by --split_ratio; trial `history` is within-block only
  - choice13k across_participants: still pools all trials per participant (full `experiment_to_trials`)
  - choice13k across_participants: same participant shuffle/split as TE (--participant_scope, --split_ratio, --split_seed)
  - cpc18: NO artificial trial split (official MSE computed)
  - mixed_gambles: participant filtering + optional gain_loss + trial split via --split_ratio / --split_seed

Method differs from Template Evolution:
  - Fit logistic MLE on the training split
  - Evaluate on the test split

python baseline_methods/MLE.py --dataset cpc18 --participant_id 0
python baseline_methods/MLE.py --dataset choice13k --split_mode across_participants \\
  --participant_scope range --range_start_ordinal 0 --range_end_ordinal 9 \\
  --split_ratio 0.9 --split_seed 0 --fitness_metric loglik
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

# Avoid importing Choice13k at module import time:
# data_modules/choice13k -> datasets -> depends on torch metadata in this environment.
# We lazily import it only when `--dataset choice13k` is selected.
if TYPE_CHECKING:
    from data_modules.choice13k import Experiment

# Ensure repo root is importable when running as:
#   python baseline_methods/MLE.py ...
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DATA_MODULES_DIR = _REPO_ROOT / "data_modules"


def load_valid_participant_ids_from_json(
    dataset: str, repo_root: Path, filter_mixed_gambles: bool
) -> List[int]:
    """Load precomputed valid raw participant ids (same as Template_evo_non_strict)."""
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
    filter_mixed_gambles: bool,
) -> List[int]:
    """Same ordinal/raw-id resolution as Template_evo_non_strict.resolve_participants_for_scope."""
    valid = load_valid_participant_ids_from_json(dataset, repo_root, filter_mixed_gambles)
    if participant_scope == "single":
        if single_participant_id not in valid:
            raise ValueError(
                f"--single_participant_id={single_participant_id} is not in the precomputed valid list "
                f"({len(valid)} ids)."
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
    if participant_scope == "all":
        if all_max_participants is not None:
            n = max(0, int(all_max_participants))
            return valid[:n]
        return list(valid)
    raise ValueError(f"Unknown participant_scope: {participant_scope!r}")


def _load_data_module(module_filename: str, module_name: str) -> Any:
    """Load a python file as a module without importing the `data_modules` package.

    This avoids executing `data_modules/__init__.py` (which currently imports Choice13k
    and can fail in environments where `datasets` depends on torch metadata).
    """
    module_path = _DATA_MODULES_DIR / module_filename
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def experiment_to_trials(exp: Any) -> Tuple[List[Dict[str, Any]], list]:
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
    exp: Any, block_indices: set[int]
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


def split_trials(
    exp: Experiment,
    split_ratio: float = 0.9,
    split_seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], list]:
    """Train/test split by **problem** (Choice13k block): disjoint gamble pairs; ratio/seed apply to block counts."""
    n_blocks = len(exp.blocks)
    if n_blocks < 2:
        raise ValueError(
            f"Choice13k within_participant split requires at least 2 problems (blocks); got {n_blocks}."
        )
    rng = np.random.default_rng(split_seed)
    perm = np.arange(n_blocks)
    rng.shuffle(perm)
    split_idx = int(n_blocks * split_ratio)
    split_idx = max(1, min(split_idx, n_blocks - 1))
    train_blocks = set(perm[:split_idx].tolist())
    test_blocks = set(perm[split_idx:].tolist())
    train_trials = trials_from_blocks_chronological(exp, train_blocks)
    test_trials = trials_from_blocks_chronological(exp, test_blocks)
    options = exp.blocks[0].option_keys
    return train_trials, test_trials, options


def load_mixed_gambles_data(
    csv_path: str,
    participant_id: int,
    filter_gain_loss_only: bool = False,
    split_ratio: float = 0.9,
    split_seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """Load mixed_gambles CSV, filter by subject == participant_id, convert to choice13k-style trials with train/test split.

    Each row: Option A (gamble) = [gain, loss] with probs [0.5, 0.5]; Option B (certain) = [cert] with probs [1.0].
    Raw CSV `took_gamble`: 1 = chose gamble, 0 = chose certain. TE option index: action = 1 - took_gamble
    (0 = Option A gamble_A, 1 = Option B gamble_B certain). history = [] (no temporal dependence).

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
            all_trials.append(
                {
                    "problem": {
                        "gamble_A": {"rewards": [gain, loss], "probs": [0.5, 0.5]},
                        "gamble_B": {"rewards": [cert], "probs": [1.0]},
                        "option_keys": option_keys,
                        "has_feedback": False,
                    },
                    "history": [],
                    "options": option_keys,
                    "action": action,
                }
            )
    if len(all_trials) == 0:
        raise ValueError(f"No rows found for subject {participant_id} in {csv_path}")
    if filter_gain_loss_only and not getattr(load_mixed_gambles_data, "_printed_gain_loss", False):
        print("[Mixed Gambles] Using gain_loss trials only.")
        load_mixed_gambles_data._printed_gain_loss = True
    # Random train/test split (reproducible).
    # The dataset is ordered by stimulus structure,
    # so row-order split creates artificial distribution shift.
    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(all_trials))
    rng.shuffle(indices)
    split_point = int(len(all_trials) * split_ratio)
    train_trials = [all_trials[i] for i in indices[:split_point]]
    test_trials = [all_trials[i] for i in indices[split_point:]]
    return train_trials, test_trials, option_keys


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def nll_logistic(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Binary logistic regression NLL.

    params = [beta, bias], p(y=1)=sigmoid(beta*x + bias)
    """
    beta, bias = params[0], params[1]
    p = sigmoid(beta * x + bias)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def expected_value_from_choice13k_gamble(gamble: Dict[str, Any]) -> float:
    """Compute EV from `problem['gamble_A'/'gamble_B']` structure."""
    rewards = gamble.get("rewards", [])
    probs = gamble.get("probs", None)
    if rewards is None or len(rewards) == 0:
        return 0.0
    if probs is None:
        # Deterministic fallback (matches minimal reasonable interpretation).
        return float(rewards[0])
    return float(np.sum(np.array(probs, dtype=np.float64) * np.array(rewards, dtype=np.float64)))


def fit_logistic_ev_diff_choice13k(train_trials: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Fit beta/bias where x = EV(gamble_B) - EV(gamble_A), y = action."""
    x = []
    y = []
    for t in train_trials:
        p = t["problem"]
        ev_a = expected_value_from_choice13k_gamble(p["gamble_A"])
        ev_b = expected_value_from_choice13k_gamble(p["gamble_B"])
        x.append(ev_b - ev_a)
        y.append(int(t["action"]))
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    res = minimize(
        lambda params: nll_logistic(params, x_arr, y_arr),
        x0=[1.0, 0.0],
        method="L-BFGS-B",
        bounds=[(-50.0, 50.0), (-50.0, 50.0)],
    )
    return float(res.x[0]), float(res.x[1])


def predict_action_logistic(beta: float, bias: float, x: float) -> int:
    p = float(sigmoid(beta * x + bias))
    return 1 if p >= 0.5 else 0


def eval_mean_loglik_choice13k_ev(trials: List[Dict[str, Any]], beta: float, bias: float) -> float:
    """Mean Bernoulli log-likelihood under logistic EV-diff model (same convention as Template_evo loglik)."""
    if not trials:
        return float("nan")
    total = 0.0
    for t in trials:
        p = t["problem"]
        ev_a = expected_value_from_choice13k_gamble(p["gamble_A"])
        ev_b = expected_value_from_choice13k_gamble(p["gamble_B"])
        x = ev_b - ev_a
        pr = float(sigmoid(beta * x + bias))
        pr = min(max(pr, 1e-9), 1.0 - 1e-9)
        y = int(t["action"])
        total += y * np.log(pr) + (1.0 - y) * np.log(1.0 - pr)
    return float(total / len(trials))


def choice13k_across_participants_train_test(
    selected_participants: List[int],
    split_ratio: float,
    split_seed: int,
    choice13k_mod: Any,
) -> Tuple[List[int], List[int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Shuffle/split participant ids then concatenate all trials per side (matches Template_evo_non_strict)."""
    if len(selected_participants) < 2:
        raise ValueError("across_participants requires at least 2 selected participants.")
    rng = np.random.default_rng(split_seed)
    shuffled = list(selected_participants)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * split_ratio)
    split_idx = max(1, min(split_idx, len(shuffled) - 1))
    train_participants = shuffled[:split_idx]
    test_participants = shuffled[split_idx:]
    max_pid = max(selected_participants)
    experiments = choice13k_mod.get_choice13k_experiments(n_participants=max_pid + 1)
    train_trials: List[Dict[str, Any]] = []
    test_trials: List[Dict[str, Any]] = []
    for pid in train_participants:
        tr, _ = experiment_to_trials(experiments[pid])
        train_trials.extend(tr)
    for pid in test_participants:
        tr, _ = experiment_to_trials(experiments[pid])
        test_trials.extend(tr)
    return train_participants, test_participants, train_trials, test_trials


def fit_logistic_ev_diff_cpc18(train_trials: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Fit beta/bias where x = EV(B) - EV(A), using Ha/pHa/La and Hb/pHb/Lb."""
    x = []
    y = []
    for t in train_trials:
        p = t["problem"]
        # Minimal EV feature from parameters present in TE's problem dict.
        ev_a = float(p["pHa"] * p["Ha"] + (1.0 - p["pHa"]) * p["La"])
        ev_b = float(p["pHb"] * p["Hb"] + (1.0 - p["pHb"]) * p["Lb"])
        x.append(ev_b - ev_a)
        y.append(int(t["action"]))
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    res = minimize(
        lambda params: nll_logistic(params, x_arr, y_arr),
        x0=[1.0, 0.0],
        method="L-BFGS-B",
        bounds=[(-50.0, 50.0), (-50.0, 50.0)],
    )
    return float(res.x[0]), float(res.x[1])


def fit_logistic_mixed_gambles(train_trials: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Fit omega/lambda for mixed gambles.

    Utility = G - omega * L, P(choose gamble_A)=sigmoid(lambda*utility).
    Trial action is TE-encoded: 0 = gamble_A, 1 = gamble_B (certain). Bernoulli y = 1 - action = 1 iff chose gamble.
    """
    G = []
    L = []
    y = []
    for t in train_trials:
        r = t["problem"]["gamble_A"]["rewards"]
        g = float(r[0])
        l = abs(float(r[1]))
        G.append(g)
        L.append(l)
        y.append(int(t["action"]))
    G_arr = np.asarray(G, dtype=np.float64)
    L_arr = np.asarray(L, dtype=np.float64)
    y_action = np.asarray(y, dtype=np.float64)  # 0=gamble_A, 1=certain
    y_chose_gamble = 1.0 - y_action  # Bernoulli target for P(gamble)

    def nll(params: np.ndarray) -> float:
        omega, lam = params[0], params[1]
        utility = G_arr - omega * L_arr
        p = sigmoid(lam * utility)
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        return -np.sum(y_chose_gamble * np.log(p) + (1.0 - y_chose_gamble) * np.log(1.0 - p))

    bounds = [(1e-5, 10.0), (1e-5, 20.0)]
    res = minimize(nll, x0=[1.0, 1.0], method="L-BFGS-B", bounds=bounds)
    omega_hat, lam_hat = float(res.x[0]), float(res.x[1])
    return omega_hat, lam_hat


def eval_accuracy_from_predict_fn(
    trials: List[Dict[str, Any]],
    predict_action: Callable[[Dict[str, Any]], int],
) -> Dict[str, float]:
    correct = 0
    total = len(trials)
    for t in trials:
        pred = predict_action(t)
        if pred == int(t["action"]):
            correct += 1
    acc = correct / total if total > 0 else 0.0
    return {"accuracy": float(acc), "total": int(total), "correct": int(correct)}


def eval_cpc18_mse_from_predict_fn(
    trials: List[Dict[str, Any]],
    observed_blocks: Dict[int, np.ndarray],
    predict_action: Callable[[Dict[str, Any]], int],
) -> Dict[str, Any]:
    """Compute official CPC18 block-level MSE (matches Template_evo_non_strict formula)."""
    problems_dict: Dict[int, List[Dict[str, Any]]] = {}
    for tr in trials:
        pid = int(tr["problem_id"])
        problems_dict.setdefault(pid, []).append(tr)

    all_mse = []
    for problem_id, problem_trials in problems_dict.items():
        if problem_id not in observed_blocks:
            continue
        blocks_dict: Dict[int, List[Dict[str, Any]]] = {}
        for tr in problem_trials:
            bid = int(tr["block_id"])
            blocks_dict.setdefault(bid, []).append(tr)

        pred_rates = np.zeros(5, dtype=np.float64)
        for block_id in range(1, 6):
            block_trials = blocks_dict.get(block_id, [])
            if not block_trials:
                continue
            preds = [int(predict_action(tr) == 1) for tr in block_trials]
            pred_rates[block_id - 1] = float(np.mean(preds))

        obs_rates = observed_blocks[problem_id]
        mse = 100.0 * float(np.mean((pred_rates - obs_rates) ** 2))
        all_mse.append(mse)

    mse_val = float(np.mean(all_mse)) if all_mse else float("inf")
    return {"mse": mse_val, "valid": True, "n_problems": len(all_mse)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Logistic MLE baselines (TE-compatible data handling)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="choice13k",
        choices=["choice13k", "cpc18", "mixed_gambles"],
        help="Dataset to use: choice13k, cpc18, or mixed_gambles",
    )
    parser.add_argument(
        "--participant_id",
        type=int,
        default=None,
        help="Specific participant ID to evaluate (0-indexed). If None, evaluates 0..num_agents_to_sample-1.",
    )
    parser.add_argument(
        "--num_agents_to_sample",
        type=int,
        default=1,
        help="Number of participants to process when --participant_id is None.",
    )
    parser.add_argument(
        "--all_data",
        action="store_true",
        help="Process all valid participants for the selected dataset.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to data directory (for cpc18; default: datasets/cpc18 when --data_path=data).",
    )
    parser.add_argument(
        "--filter_mixed_gambles",
        action="store_true",
        default=False,
        help=(
            "For mixed_gambles: keep only gain_loss trials. Default False (all trial types; "
            "larger valid-participant set under --all_data and in collect_participant_ids)."
        ),
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="within_participant",
        choices=["within_participant", "across_participants"],
        help="choice13k only: within_participant = problem (block)-level split per participant; "
        "across_participants = shuffle/split participant ids then one global MLE (TE-aligned).",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.9,
        help=(
            "choice13k within_participant: fraction of problems (blocks) for train; "
            "choice13k across_participants: fraction of participants for train; "
            "mixed_gambles: fraction of trials for train."
        ),
    )
    parser.add_argument("--split_seed", type=int, default=0, help="RNG seed for splits (TE-aligned).")
    parser.add_argument(
        "--participant_scope",
        type=str,
        default=None,
        choices=["single", "range", "all"],
        help="choice13k only: resolve participants from datasets/*/valid_participant_ids.json (same ordinals as TE).",
    )
    parser.add_argument(
        "--single_participant_id",
        type=int,
        default=0,
        help="Raw participant id when --participant_scope single.",
    )
    parser.add_argument(
        "--range_start_ordinal",
        type=int,
        default=None,
        help="Inclusive start index into the sorted valid id list (with --participant_scope range).",
    )
    parser.add_argument(
        "--range_end_ordinal",
        type=int,
        default=None,
        help="Inclusive end index into the sorted valid id list (with --participant_scope range).",
    )
    parser.add_argument(
        "--all_max_participants",
        type=int,
        default=None,
        help="With --participant_scope all: cap how many valid ids to use (None = all).",
    )
    parser.add_argument(
        "--fitness_metric",
        type=str,
        default="acc",
        choices=["acc", "loglik"],
        help="Primary metric for printed summary (both acc and mean loglik are computed for choice13k).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated).",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable wandb logging. Default is enabled.",
    )
    args = parser.parse_args()
    num_agents_arg_explicit = "--num_agents_to_sample" in sys.argv
    mixed_gambles_gain_loss_only = bool(getattr(args, "filter_mixed_gambles", False))

    if not (0.0 < args.split_ratio < 1.0):
        raise ValueError("--split_ratio must be strictly between 0 and 1.")
    if args.participant_scope is not None and args.dataset != "choice13k":
        raise ValueError("--participant_scope is only supported for --dataset choice13k.")
    if args.split_mode == "across_participants":
        if args.dataset != "choice13k":
            raise ValueError("--split_mode across_participants is only supported for --dataset choice13k.")
        if args.participant_scope is None:
            raise ValueError("--split_mode across_participants requires --participant_scope.")
        if args.all_data:
            raise ValueError("--split_mode across_participants is incompatible with --all_data; use --participant_scope all.")
    if args.participant_scope is not None and args.all_data:
        raise ValueError("Use either --participant_scope or --all_data, not both.")

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # choice13k: single global MLE on pooled train trials; test = held-out participants' trials.
    if args.dataset == "choice13k" and args.split_mode == "across_participants":
        selected_participants = resolve_participants_for_scope(
            dataset="choice13k",
            repo_root=_REPO_ROOT,
            participant_scope=args.participant_scope,
            single_participant_id=args.single_participant_id,
            range_start_ordinal=args.range_start_ordinal,
            range_end_ordinal=args.range_end_ordinal,
            all_max_participants=args.all_max_participants,
            filter_mixed_gambles=False,
        )
        if args.output_dir is None:
            base_run_dir_ap = f"generated_outputs/choice13k/MLE/run_{timestamp}"
        else:
            base_run_dir_ap = args.output_dir
        Path(base_run_dir_ap).mkdir(parents=True, exist_ok=True)

        wandb_ap = None
        if not args.no_log:
            try:
                import wandb as _wandb

                wandb_ap = _wandb
                run_name = f"choice13k_MLE_across_{timestamp}"
                wandb_ap.init(
                    project="ROTE_evo",
                    name=run_name,
                    config=vars(args),
                    reinit=False,
                )
            except Exception as e:
                print(f"wandb logging disabled: {e}")
                wandb_ap = None

        choice13k_mod = _load_data_module("choice13k.py", "choice13k_data_module")
        train_p, test_p, train_trials, test_trials = choice13k_across_participants_train_test(
            selected_participants, args.split_ratio, args.split_seed, choice13k_mod
        )
        beta_hat, bias_hat = fit_logistic_ev_diff_choice13k(train_trials)

        def predict_action_ap(tr: Dict[str, Any]) -> int:
            p = tr["problem"]
            ev_a = expected_value_from_choice13k_gamble(p["gamble_A"])
            ev_b = expected_value_from_choice13k_gamble(p["gamble_B"])
            x = ev_b - ev_a
            return predict_action_logistic(beta_hat, bias_hat, x)

        train_acc_eval = eval_accuracy_from_predict_fn(train_trials, predict_action_ap)
        test_acc_eval = eval_accuracy_from_predict_fn(test_trials, predict_action_ap)
        train_ll = eval_mean_loglik_choice13k_ev(train_trials, beta_hat, bias_hat)
        test_ll = eval_mean_loglik_choice13k_ev(test_trials, beta_hat, bias_hat)
        results_ap = {
            "method": "logistic_MLE",
            "dataset": "choice13k",
            "split_mode": "across_participants",
            "train_participants": train_p,
            "test_participants": test_p,
            "n_train_participants": len(train_p),
            "n_test_participants": len(test_p),
            "fitted_params": {"beta": beta_hat, "bias": bias_hat},
            "train_accuracy": train_acc_eval["accuracy"],
            "test_accuracy": test_acc_eval["accuracy"],
            "train_mean_loglik": train_ll,
            "test_mean_loglik": test_ll,
            "n_train_trials": train_acc_eval["total"],
            "n_test_trials": test_acc_eval["total"],
            "train_correct": train_acc_eval["correct"],
            "test_correct": test_acc_eval["correct"],
        }
        (Path(base_run_dir_ap) / "results.json").write_text(json.dumps(results_ap, indent=2))

        if wandb_ap is not None:
            wandb_ap.log(
                {
                    "train_accuracy": train_acc_eval["accuracy"],
                    "test_accuracy": test_acc_eval["accuracy"],
                    "train_mean_loglik": train_ll,
                    "test_mean_loglik": test_ll,
                    "is_baseline": 1,
                },
                step=0,
            )
            wandb_ap.finish()

        if args.fitness_metric == "loglik":
            print(
                f"\n[MLE baseline] choice13k across_participants train_mean_loglik={train_ll:.6f} "
                f"test_mean_loglik={test_ll:.6f}"
            )
        else:
            print(
                f"\n[MLE baseline] choice13k across_participants train_acc={train_acc_eval['accuracy']:.4f} "
                f"test_acc={test_acc_eval['accuracy']:.4f}"
            )
        print(f"Results saved under: {base_run_dir_ap}")
        return

    wandb = None
    if not args.no_log:
        try:
            import wandb as _wandb
            wandb = _wandb

            dataset_prefix = args.dataset if args.dataset else "choice13k"
            run_name = f"{dataset_prefix}_MLE_{timestamp}"
            if args.participant_id is not None:
                run_name = f"{run_name}_participant_{args.participant_id}"
            elif getattr(args, "participant_scope", None) is not None:
                run_name = f"{run_name}_scope_{args.participant_scope}"
            else:
                run_name = f"{run_name}_participants_0to{args.num_agents_to_sample-1}"

            wandb.init(
                project="ROTE_evo",
                name=run_name,
                config=vars(args),
                reinit=False,
            )
        except Exception as e:
            print(f"wandb logging disabled: {e}")
            wandb = None

    # Determine which participants to process
    choice13k_exp_cache: Dict[int, Any] = {}
    if args.dataset == "choice13k" and args.participant_scope is not None:
        participants_to_process = resolve_participants_for_scope(
            dataset="choice13k",
            repo_root=_REPO_ROOT,
            participant_scope=args.participant_scope,
            single_participant_id=args.single_participant_id,
            range_start_ordinal=args.range_start_ordinal,
            range_end_ordinal=args.range_end_ordinal,
            all_max_participants=args.all_max_participants,
            filter_mixed_gambles=False,
        )
    elif args.all_data:
        valid_participants: List[int] = []
        if args.dataset == "cpc18":
            cpc18_data_path = args.data_path if args.data_path != "data" else "datasets/cpc18"
            raw_file = Path(cpc18_data_path) / "raw-comp-set-data-Track-2.csv"
            if not raw_file.exists():
                raise FileNotFoundError(f"Could not find CPC18 raw data file: {raw_file}")
            unique_subj_ids = set()
            with open(raw_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    unique_subj_ids.add(int(row["SubjID"]))
            cpc18_mod = _load_data_module("cpc18.py", "cpc18_data_module")
            for participant_id in range(len(unique_subj_ids)):
                try:
                    participant_data = cpc18_mod.load_cpc18_track2_data(
                        data_path=cpc18_data_path, participant_id=participant_id
                    )
                    train_trials, test_trials, _ = cpc18_mod.split_cpc18_trials(
                        participant_data, train_ratio=0.8, cpc18_official_mse=True
                    )
                    if len(train_trials) > 0 and len(test_trials) > 0:
                        valid_participants.append(participant_id)
                except Exception:
                    continue
        elif args.dataset == "mixed_gambles":
            csv_path = "datasets/mixed_gambles/data_all_2021-01-08.csv"
            unique_subjects = set()
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    unique_subjects.add(int(row["subject"]))
            for participant_id in sorted(unique_subjects):
                try:
                    train_trials, test_trials, _ = load_mixed_gambles_data(
                        csv_path,
                        participant_id,
                        filter_gain_loss_only=mixed_gambles_gain_loss_only,
                        split_ratio=args.split_ratio,
                        split_seed=args.split_seed,
                    )
                    if len(train_trials) > 0 and len(test_trials) > 0:
                        valid_participants.append(participant_id)
                except Exception:
                    continue
        else:
            # choice13k: enumerate all candidates from filtered dataset length, validate each safely.
            choice13k_mod = _load_data_module("choice13k.py", "choice13k_data_module")
            dataset = choice13k_mod.load_dataset("marcelbinz/Psych-101-test")
            test_split = dataset["test"]
            choices13k_ds = test_split.filter(lambda ex: ex["experiment"] == "peterson2021using/exp1.csv")
            for participant_id in range(len(choices13k_ds)):
                try:
                    exp = choice13k_mod._convert_to_experiment(choices13k_ds[participant_id])
                    train_trials, test_trials, _ = split_trials(
                        exp, split_ratio=args.split_ratio, split_seed=args.split_seed
                    )
                    if len(train_trials) > 0 and len(test_trials) > 0:
                        valid_participants.append(participant_id)
                        choice13k_exp_cache[participant_id] = exp
                except Exception:
                    continue

        if num_agents_arg_explicit:
            valid_participants = valid_participants[: max(0, int(args.num_agents_to_sample))]
        participants_to_process = valid_participants
        print(
            f"All data mode activated by --all_data. All participants will be processed. "
            f"Total num of valid participants: {len(participants_to_process)}."
        )
    elif args.participant_id is not None:
        participants_to_process = [args.participant_id]
    else:
        participants_to_process = list(range(args.num_agents_to_sample))

    # Base run dir (match Template_evo_non_strict pattern)
    base_run_dir: Optional[str] = None
    if args.output_dir is None:
        mode = "MLE"
        if args.dataset == "cpc18":
            base_run_dir = f"generated_outputs/cpc18/{mode}/run_{timestamp}"
        elif args.dataset == "mixed_gambles":
            base_run_dir = f"generated_outputs/mixed_gambles/{mode}/run_{timestamp}"
        else:
            base_run_dir = f"generated_outputs/choice13k/{mode}/run_{timestamp}"
        Path(base_run_dir).mkdir(parents=True, exist_ok=True)
    elif len(participants_to_process) > 1:
        base_run_dir = args.output_dir
        Path(base_run_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output_dir)
        if output_path.name.startswith("participant_"):
            base_run_dir = str(output_path.parent)
        else:
            base_run_dir = args.output_dir
        Path(base_run_dir).mkdir(parents=True, exist_ok=True)

    assert base_run_dir is not None
    summary_file = Path(base_run_dir) / ("summary.csv" if args.all_data else "participants_summary.csv")
    details_file = Path(base_run_dir) / "participants_details.csv"

    participants_summary: List[Dict[str, Any]] = []
    if args.all_data:
        summary_fieldnames = ["num_of_participants", "avg_train_fitness", "avg_test_fitness"]
        details_fieldnames = ["participant_id", "train_fitness", "test_fitness", "total_runtime"]
        participants_details: List[Dict[str, Any]] = []
    else:
        if args.dataset == "cpc18":
            summary_fieldnames = ["participant_id", "train_fitness", "train_mse", "test_mse"]
        elif args.dataset == "choice13k":
            summary_fieldnames = [
                "participant_id",
                "train_acc",
                "test_acc",
                "train_mean_loglik",
                "test_mean_loglik",
            ]
        else:
            summary_fieldnames = ["participant_id", "train_acc", "test_acc"]

    for participant_id in tqdm(participants_to_process, desc="Participants"):
        participant_start = datetime.now()
        if args.dataset == "choice13k":
            if args.all_data and participant_id in choice13k_exp_cache:
                exp = choice13k_exp_cache[participant_id]
            else:
                choice13k_mod = _load_data_module("choice13k.py", "choice13k_data_module")
                experiments = choice13k_mod.get_choice13k_experiments(n_participants=participant_id + 1)
                exp = experiments[participant_id]
            train_trials, test_trials, _ = split_trials(
                exp, split_ratio=args.split_ratio, split_seed=args.split_seed
            )
        elif args.dataset == "cpc18":
            cpc18_data_path = args.data_path if args.data_path != "data" else "datasets/cpc18"
            cpc18_mod = _load_data_module("cpc18.py", "cpc18_data_module")
            participant_data = cpc18_mod.load_cpc18_track2_data(
                data_path=cpc18_data_path, participant_id=participant_id
            )
            train_trials, test_trials, test_observed_blocks = cpc18_mod.split_cpc18_trials(
                participant_data, train_ratio=0.8, cpc18_official_mse=True
            )
        elif args.dataset == "mixed_gambles":
            csv_path = "datasets/mixed_gambles/data_all_2021-01-08.csv"
            train_trials, test_trials, _ = load_mixed_gambles_data(
                csv_path,
                participant_id,
                filter_gain_loss_only=mixed_gambles_gain_loss_only,
                split_ratio=args.split_ratio,
                split_seed=args.split_seed,
            )
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        # Fit MLE and predict
        if args.dataset == "mixed_gambles":
            omega_hat, lam_hat = fit_logistic_mixed_gambles(train_trials)

            def predict_action(tr: Dict[str, Any]) -> int:
                r = tr["problem"]["gamble_A"]["rewards"]
                g = float(r[0])
                l = abs(float(r[1]))
                utility = g - omega_hat * l
                p = float(sigmoid(lam_hat * utility))
                return 0 if p >= 0.5 else 1

            train_acc_eval = eval_accuracy_from_predict_fn(train_trials, predict_action)
            test_acc_eval = eval_accuracy_from_predict_fn(test_trials, predict_action)
            results = {
                "method": "logistic_MLE",
                "dataset": args.dataset,
                "participant_id": participant_id,
                "fitted_params": {"omega": omega_hat, "lambda": lam_hat},
                "train_accuracy": train_acc_eval["accuracy"],
                "test_accuracy": test_acc_eval["accuracy"],
                "n_train": train_acc_eval["total"],
                "n_test": test_acc_eval["total"],
                "train_correct": train_acc_eval["correct"],
                "test_correct": test_acc_eval["correct"],
            }

        elif args.dataset == "choice13k":
            beta_hat, bias_hat = fit_logistic_ev_diff_choice13k(train_trials)

            def predict_action(tr: Dict[str, Any]) -> int:
                p = tr["problem"]
                ev_a = expected_value_from_choice13k_gamble(p["gamble_A"])
                ev_b = expected_value_from_choice13k_gamble(p["gamble_B"])
                x = ev_b - ev_a
                return predict_action_logistic(beta_hat, bias_hat, x)

            train_acc_eval = eval_accuracy_from_predict_fn(train_trials, predict_action)
            test_acc_eval = eval_accuracy_from_predict_fn(test_trials, predict_action)
            train_ll = eval_mean_loglik_choice13k_ev(train_trials, beta_hat, bias_hat)
            test_ll = eval_mean_loglik_choice13k_ev(test_trials, beta_hat, bias_hat)
            results = {
                "method": "logistic_MLE",
                "dataset": args.dataset,
                "participant_id": participant_id,
                "fitted_params": {"beta": beta_hat, "bias": bias_hat},
                "train_accuracy": train_acc_eval["accuracy"],
                "test_accuracy": test_acc_eval["accuracy"],
                "train_mean_loglik": train_ll,
                "test_mean_loglik": test_ll,
                "n_train": train_acc_eval["total"],
                "n_test": test_acc_eval["total"],
                "train_correct": train_acc_eval["correct"],
                "test_correct": test_acc_eval["correct"],
            }

        elif args.dataset == "cpc18":
            beta_hat, bias_hat = fit_logistic_ev_diff_cpc18(train_trials)

            def predict_action(tr: Dict[str, Any]) -> int:
                p = tr["problem"]
                ev_a = float(p["pHa"] * p["Ha"] + (1.0 - p["pHa"]) * p["La"])
                ev_b = float(p["pHb"] * p["Hb"] + (1.0 - p["pHb"]) * p["Lb"])
                x = ev_b - ev_a
                return predict_action_logistic(beta_hat, bias_hat, x)

            train_acc_eval = eval_accuracy_from_predict_fn(train_trials, predict_action)
            test_acc_eval = eval_accuracy_from_predict_fn(test_trials, predict_action)
            train_mse_eval = eval_cpc18_mse_from_predict_fn(
                train_trials, test_observed_blocks, predict_action
            )
            test_mse_eval = eval_cpc18_mse_from_predict_fn(
                test_trials, test_observed_blocks, predict_action
            )

            results = {
                "method": "logistic_MLE",
                "dataset": args.dataset,
                "participant_id": participant_id,
                "fitted_params": {"beta": beta_hat, "bias": bias_hat},
                "train_accuracy": train_acc_eval["accuracy"],
                "test_accuracy": test_acc_eval["accuracy"],
                "train_mse": train_mse_eval["mse"],
                "test_mse": test_mse_eval["mse"],
                "n_train": train_acc_eval["total"],
                "n_test": test_acc_eval["total"],
                "train_correct": train_acc_eval["correct"],
                "test_correct": test_acc_eval["correct"],
            }

        else:
            raise AssertionError("Unreachable")

        # Update participants summary/details
        if args.dataset == "cpc18":
            train_fitness = -float(results.get("train_mse", float("nan")))
            test_fitness = -float(results.get("test_mse", float("nan")))
            row = {
                "participant_id": participant_id,
                "train_fitness": train_fitness,
                "train_mse": float(results.get("train_mse", float("nan"))),
                "test_mse": float(results.get("test_mse", float("nan"))),
            }
        else:
            train_fitness = float(results.get("train_accuracy", float("nan")))
            test_fitness = float(results.get("test_accuracy", float("nan")))
            row = {
                "participant_id": participant_id,
                "train_acc": float(results.get("train_accuracy", float("nan"))),
                "test_acc": float(results.get("test_accuracy", float("nan"))),
            }
            if args.dataset == "choice13k":
                row["train_mean_loglik"] = float(results.get("train_mean_loglik", float("nan")))
                row["test_mean_loglik"] = float(results.get("test_mean_loglik", float("nan")))

        if args.all_data:
            runtime_sec = (datetime.now() - participant_start).total_seconds()
            details_row = {
                "participant_id": participant_id,
                "train_fitness": train_fitness,
                "test_fitness": test_fitness,
                "total_runtime": runtime_sec,
            }
            participants_details.append(details_row)
            with open(details_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=details_fieldnames)
                writer.writeheader()
                writer.writerows(participants_details)

            avg_train_fitness = float(np.mean([r["train_fitness"] for r in participants_details])) if participants_details else 0.0
            avg_test_fitness = float(np.mean([r["test_fitness"] for r in participants_details])) if participants_details else 0.0
            with open(summary_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "num_of_participants": len(participants_details),
                        "avg_train_fitness": avg_train_fitness,
                        "avg_test_fitness": avg_test_fitness,
                    }
                )
        else:
            # Save outputs in TE-like structure
            participant_output_dir = os.path.join(base_run_dir, f"participant_{participant_id}")
            Path(participant_output_dir).mkdir(parents=True, exist_ok=True)
            (Path(participant_output_dir) / "results.json").write_text(json.dumps(results, indent=2))

            participants_summary.append(row)
            with open(summary_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
                writer.writeheader()
                writer.writerows(participants_summary)

        # W&B logging (episode-free; baseline)
        if wandb is not None:
            prefix = f"p{participant_id}"
            if args.dataset == "cpc18":
                train_fitness = float(row["train_fitness"])
                wandb.log(
                    {
                        f"{prefix}_train_fitness": train_fitness,
                        f"{prefix}_train_mse": float(row["train_mse"]),
                        f"{prefix}_test_mse": float(row["test_mse"]),
                        f"{prefix}_is_baseline": 1,
                        # Keep accuracy for debugging (not used for selection)
                        f"{prefix}_train_accuracy": float(results.get("train_accuracy", float("nan"))),
                        f"{prefix}_test_accuracy": float(results.get("test_accuracy", float("nan"))),
                    },
                    step=0,
                )
            else:
                log_payload = {
                    f"{prefix}_train_accuracy": row["train_acc"],
                    f"{prefix}_test_accuracy": row["test_acc"],
                    f"{prefix}_is_baseline": 1,
                }
                if args.dataset == "choice13k":
                    log_payload[f"{prefix}_train_mean_loglik"] = float(
                        results.get("train_mean_loglik", float("nan"))
                    )
                    log_payload[f"{prefix}_test_mean_loglik"] = float(
                        results.get("test_mean_loglik", float("nan"))
                    )
                wandb.log(log_payload, step=0)

    if wandb is not None:
        wandb.finish()

    # Print final mean across processed participants
    if args.all_data:
        train_mean = float(np.mean([r["train_fitness"] for r in participants_details])) if participants_details else 0.0
        test_mean = float(np.mean([r["test_fitness"] for r in participants_details])) if participants_details else 0.0
        print(f"\n[MLE baseline] dataset={args.dataset} participants={participants_to_process}")
        print(f"Mean train fitness: {train_mean:.4f}")
        print(f"Mean test fitness:  {test_mean:.4f}")
    elif args.dataset == "cpc18":
        train_mean = float(np.mean([r["train_fitness"] for r in participants_summary])) if participants_summary else 0.0
        test_mean = float(np.mean([r["test_mse"] for r in participants_summary])) if participants_summary else 0.0
        print(f"\n[MLE baseline] dataset={args.dataset} participants={participants_to_process}")
        print(f"Mean train fitness (-MSE): {train_mean:.4f}")
        print(f"Mean test MSE (official):  {test_mean:.4f}")
    else:
        print(f"\n[MLE baseline] dataset={args.dataset} participants={participants_to_process}")
        if args.dataset == "choice13k":
            train_mean_acc = (
                float(np.mean([r["train_acc"] for r in participants_summary])) if participants_summary else 0.0
            )
            test_mean_acc = (
                float(np.mean([r["test_acc"] for r in participants_summary])) if participants_summary else 0.0
            )
            train_mean_ll = (
                float(np.mean([r["train_mean_loglik"] for r in participants_summary]))
                if participants_summary
                else 0.0
            )
            test_mean_ll = (
                float(np.mean([r["test_mean_loglik"] for r in participants_summary]))
                if participants_summary
                else 0.0
            )
            if args.fitness_metric == "loglik":
                print(f"Mean train loglik: {train_mean_ll:.6f}")
                print(f"Mean test loglik:  {test_mean_ll:.6f}")
            else:
                print(f"Mean train accuracy: {train_mean_acc:.4f}")
                print(f"Mean test accuracy:  {test_mean_acc:.4f}")
        else:
            train_mean = float(np.mean([r["train_acc"] for r in participants_summary])) if participants_summary else 0.0
            test_mean = float(np.mean([r["test_acc"] for r in participants_summary])) if participants_summary else 0.0
            print(f"Mean train accuracy: {train_mean:.4f}")
            print(f"Mean test accuracy:  {test_mean:.4f}")
    print(f"Results saved under: {base_run_dir}")


if __name__ == "__main__":
    main()

