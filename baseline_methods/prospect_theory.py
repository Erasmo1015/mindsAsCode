#!/usr/bin/env python3
"""
Prospect Theory MLE baselines for Choice13k, CPC18 (Track II), and Mixed Gambles.

Data loading/splitting conventions are copied to match `Template_evo_non_strict.py`
for the three datasets exactly:
  - choice13k: fixed 80/20 split with RNG seed=42 (history accumulated in order)
  - cpc18: NO artificial split (train_ratio ignored; use all trials; CPC18 official MSE computed)
  - mixed_gambles: participant filtering + optional gain_loss filtering + 80/20 split with RNG seed=42

Method differs from Template Evolution:
  - Fit a simple prospect-theory model by MLE on the training split
  - Evaluate on the test split (CPC18: report MSE + train_fitness = -train_mse)
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
#   python baseline_methods/prospect_theory.py ...
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DATA_MODULES_DIR = _REPO_ROOT / "data_modules"


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


# ===== Dataset loading / splitting (copied to match Template_evo_non_strict.py) =====

def split_trials(exp: Experiment) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], list]:
    """Split trials into train/test 80/20 (fixed split, matching ROTE); return (train, test, options)."""
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
    # Random train/test split (reproducible).
    # The dataset is ordered by stimulus structure,
    # so row-order split creates artificial distribution shift.
    rng = np.random.default_rng(42)
    indices = np.arange(len(all_trials))
    rng.shuffle(indices)
    split_point = int(len(all_trials) * 0.8)
    train_trials = [all_trials[i] for i in indices[:split_point]]
    test_trials = [all_trials[i] for i in indices[split_point:]]
    return train_trials, test_trials, options


def load_mixed_gambles_data(
    csv_path: str, participant_id: int, filter_gain_loss_only: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """Load mixed_gambles CSV, filter by subject == participant_id, convert to choice13k-style trials with 80/20 split.

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
    rng = np.random.default_rng(42)
    indices = np.arange(len(all_trials))
    rng.shuffle(indices)
    split_point = int(len(all_trials) * 0.8)
    train_trials = [all_trials[i] for i in indices[:split_point]]
    test_trials = [all_trials[i] for i in indices[split_point:]]
    return train_trials, test_trials, option_keys


# ===== Prospect theory model =====

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def v_value(x: np.ndarray, alpha: float, lam: float) -> np.ndarray:
    """Prospect theory value function v(x)."""
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[pos] = np.power(x[pos], alpha)
    out[~pos] = -lam * np.power(-x[~pos], alpha)
    return out


def w_prelec(p: np.ndarray, gamma: float) -> np.ndarray:
    """Prelec probability weighting w(p)=exp(-(-log(p))^gamma)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p, dtype=np.float64)
    out[p >= 1.0] = 1.0
    mask = (p > 0.0) & (p < 1.0)
    if np.any(mask):
        pm = np.clip(p[mask], 1e-12, 1.0 - 1e-12)
        out[mask] = np.exp(-np.power(-np.log(pm), gamma))
    return out


def _ensure_probs(rewards: List[float], probs: Optional[List[float]]) -> List[float]:
    """If probs is None, interpret as deterministic if single reward else uniform."""
    if probs is None:
        if len(rewards) <= 1:
            return [1.0]
        return [1.0 / len(rewards)] * len(rewards)
    return probs


def subjective_value_gamble(
    rewards: List[float],
    probs: Optional[List[float]],
    alpha: float,
    lam: float,
    gamma: float,
) -> float:
    probs_use = _ensure_probs(rewards, probs)
    x = np.asarray(rewards, dtype=np.float64)
    p = np.asarray(probs_use, dtype=np.float64)
    # Probabilities are always passed through the Prelec weighting function.
    # If `probs` is None, we use a minimal fallback distribution (deterministic for single-outcome,
    # otherwise uniform) and still weight via Prelec.
    w = w_prelec(p, gamma)
    return float(np.sum(w * v_value(x, alpha, lam)))


def nll_prospect_theory_choiceA(
    params: np.ndarray,
    VA: np.ndarray,
    VB: np.ndarray,
    y_chooseA: np.ndarray,
) -> float:
    """NLL with P(choose A)=sigmoid(beta*(VA-VB))."""
    beta = float(params[0])
    pA = sigmoid(beta * (VA - VB))
    pA = np.clip(pA, 1e-9, 1.0 - 1e-9)
    return float(-np.sum(y_chooseA * np.log(pA) + (1.0 - y_chooseA) * np.log(1.0 - pA)))


def fit_prospect_theory_gamble_choice(
    train_trials: List[Dict[str, Any]],
    action_is_chooseA: Callable[[int], bool],
    gambleA_getter: Callable[[Dict[str, Any]], Tuple[List[float], Optional[List[float]]]],
    gambleB_getter: Callable[[Dict[str, Any]], Tuple[List[float], Optional[List[float]]]],
    *,
    dataset: Optional[str] = None,
    participant_id: Optional[int] = None,
) -> Dict[str, float]:
    """Fit alpha, lambda, gamma, beta by MLE for two-option gamble-choice."""
    default_theta = np.array([0.8, 2.0, 1.0, 1.0], dtype=np.float64)  # alpha, lambda, gamma, beta
    if len(train_trials) == 0:
        print(
            f"[Warning][prospect_theory] Degenerate train set (empty). "
            f"Using default params. dataset={dataset} participant_id={participant_id}"
        )
        return {"alpha": float(default_theta[0]), "lambda": float(default_theta[1]), "gamma": float(default_theta[2]), "beta": float(default_theta[3])}

    # We fit alpha/lambda/gamma/beta jointly. VA/VB depend on alpha/lambda/gamma.
    yA = np.asarray([1.0 if action_is_chooseA(int(t["action"])) else 0.0 for t in train_trials], dtype=np.float64)
    if np.all(yA == yA[0]):
        print(
            f"[Warning][prospect_theory] Degenerate train labels (single class). "
            f"Using default params. dataset={dataset} participant_id={participant_id}"
        )
        return {"alpha": float(default_theta[0]), "lambda": float(default_theta[1]), "gamma": float(default_theta[2]), "beta": float(default_theta[3])}

    def nll(theta: np.ndarray) -> float:
        alpha, lam, gamma, beta = float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])
        VA = []
        VB = []
        for tr in train_trials:
            p = tr["problem"]
            rA, prA = gambleA_getter(p)
            rB, prB = gambleB_getter(p)
            VA.append(subjective_value_gamble(rA, prA, alpha, lam, gamma))
            VB.append(subjective_value_gamble(rB, prB, alpha, lam, gamma))
        VA_arr = np.asarray(VA, dtype=np.float64)
        VB_arr = np.asarray(VB, dtype=np.float64)
        # beta used only in choice rule
        pA = sigmoid(beta * (VA_arr - VB_arr))
        pA = np.clip(pA, 1e-9, 1.0 - 1e-9)
        return float(-np.sum(yA * np.log(pA) + (1.0 - yA) * np.log(1.0 - pA)))

    bounds = [(0.01, 2.0), (0.01, 10.0), (0.01, 5.0), (0.01, 20.0)]
    res = minimize(
        nll,
        x0=default_theta,
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not getattr(res, "success", False):
        msg = getattr(res, "message", "")
        print(
            f"[Warning][prospect_theory] Optimization failed; using best found params. "
            f"dataset={dataset} participant_id={participant_id} message={msg}"
        )
    alpha_hat, lam_hat, gamma_hat, beta_hat = (float(res.x[0]), float(res.x[1]), float(res.x[2]), float(res.x[3]))
    return {"alpha": alpha_hat, "lambda": lam_hat, "gamma": gamma_hat, "beta": beta_hat}


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
    parser = argparse.ArgumentParser(description="Prospect Theory baseline (TE-compatible data handling)")
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
        help="For mixed_gambles dataset: keep only gain_loss trials (matches Template_evo_non_strict).",
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

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    wandb = None
    if not args.no_log:
        try:
            import wandb as _wandb
            wandb = _wandb

            dataset_prefix = args.dataset if args.dataset else "choice13k"
            run_name = f"{dataset_prefix}_prospect_theory_{timestamp}"
            if args.participant_id is not None:
                run_name = f"{run_name}_participant_{args.participant_id}"
            elif args.all_data:
                run_name = f"{run_name}_all_data"
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
    if args.all_data:
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
                        participant_data, train_ratio=0.8
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
                        csv_path, participant_id, filter_gain_loss_only=args.filter_mixed_gambles
                    )
                    if len(train_trials) > 0 and len(test_trials) > 0:
                        valid_participants.append(participant_id)
                except Exception:
                    continue
        else:
            choice13k_mod = _load_data_module("choice13k.py", "choice13k_data_module")
            dataset = choice13k_mod.load_dataset("marcelbinz/Psych-101-test")
            test_split = dataset["test"]
            choices13k_ds = test_split.filter(lambda ex: ex["experiment"] == "peterson2021using/exp1.csv")
            for participant_id in range(len(choices13k_ds)):
                try:
                    exp = choice13k_mod._convert_to_experiment(choices13k_ds[participant_id])
                    train_trials, test_trials, _ = split_trials(exp)
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

    # Base run dir (match Template_evo_non_strict / baseline_methods/MLE.py pattern)
    base_run_dir: Optional[str] = None
    if args.output_dir is None:
        mode = "prospect_theory"
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
        summary_fieldnames = (
            ["participant_id", "train_fitness", "train_mse", "test_mse"]
            if args.dataset == "cpc18"
            else ["participant_id", "train_acc", "test_acc"]
        )

    for participant_id in tqdm(participants_to_process, desc="Participants"):
        participant_start = datetime.now()
        # Load data (TE conventions)
        if args.dataset == "choice13k":
            if args.all_data and participant_id in choice13k_exp_cache:
                exp = choice13k_exp_cache[participant_id]
            else:
                choice13k_mod = _load_data_module("choice13k.py", "choice13k_data_module")
                experiments = choice13k_mod.get_choice13k_experiments(n_participants=participant_id + 1)
                exp = experiments[participant_id]
            train_trials, test_trials, _ = split_trials(exp)
        elif args.dataset == "cpc18":
            cpc18_data_path = args.data_path if args.data_path != "data" else "datasets/cpc18"
            cpc18_mod = _load_data_module("cpc18.py", "cpc18_data_module")
            participant_data = cpc18_mod.load_cpc18_track2_data(
                data_path=cpc18_data_path, participant_id=participant_id
            )
            train_trials, test_trials, test_observed_blocks = cpc18_mod.split_cpc18_trials(
                participant_data, train_ratio=0.8
            )
        elif args.dataset == "mixed_gambles":
            csv_path = "datasets/mixed_gambles/data_all_2021-01-08.csv"
            train_trials, test_trials, _ = load_mixed_gambles_data(
                csv_path, participant_id, filter_gain_loss_only=args.filter_mixed_gambles
            )
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        # Fit prospect-theory model + evaluate
        if args.dataset == "choice13k":
            # Choice13k: action=0 means choose gamble_A, action=1 means choose gamble_B.
            params = fit_prospect_theory_gamble_choice(
                train_trials,
                action_is_chooseA=lambda a: a == 0,
                gambleA_getter=lambda prob: (prob["gamble_A"]["rewards"], prob["gamble_A"]["probs"]),
                gambleB_getter=lambda prob: (prob["gamble_B"]["rewards"], prob["gamble_B"]["probs"]),
                dataset=args.dataset,
                participant_id=participant_id,
            )

            def predict_action(tr: Dict[str, Any]) -> int:
                p = tr["problem"]
                VA = subjective_value_gamble(p["gamble_A"]["rewards"], p["gamble_A"]["probs"], params["alpha"], params["lambda"], params["gamma"])
                VB = subjective_value_gamble(p["gamble_B"]["rewards"], p["gamble_B"]["probs"], params["alpha"], params["lambda"], params["gamma"])
                pA = float(sigmoid(params["beta"] * (VA - VB)))
                return 0 if pA >= 0.5 else 1

            train_eval = eval_accuracy_from_predict_fn(train_trials, predict_action)
            test_eval = eval_accuracy_from_predict_fn(test_trials, predict_action)

            results = {
                "method": "prospect_theory_MLE",
                "dataset": args.dataset,
                "participant_id": participant_id,
                "fitted_params": params,
                "train_accuracy": train_eval["accuracy"],
                "test_accuracy": test_eval["accuracy"],
                "n_train": train_eval["total"],
                "n_test": test_eval["total"],
                "train_correct": train_eval["correct"],
                "test_correct": test_eval["correct"],
            }

        elif args.dataset == "mixed_gambles":
            # Mixed gambles TE: action=0 means choose gamble_A, action=1 means choose gamble_B (certain).
            params = fit_prospect_theory_gamble_choice(
                train_trials,
                action_is_chooseA=lambda a: a == 0,
                gambleA_getter=lambda prob: (prob["gamble_A"]["rewards"], prob["gamble_A"]["probs"]),
                gambleB_getter=lambda prob: (prob["gamble_B"]["rewards"], prob["gamble_B"]["probs"]),
                dataset=args.dataset,
                participant_id=participant_id,
            )

            def predict_action(tr: Dict[str, Any]) -> int:
                p = tr["problem"]
                VA = subjective_value_gamble(p["gamble_A"]["rewards"], p["gamble_A"]["probs"], params["alpha"], params["lambda"], params["gamma"])
                VB = subjective_value_gamble(p["gamble_B"]["rewards"], p["gamble_B"]["probs"], params["alpha"], params["lambda"], params["gamma"])
                pA = float(sigmoid(params["beta"] * (VA - VB)))
                return 0 if pA >= 0.5 else 1

            train_eval = eval_accuracy_from_predict_fn(train_trials, predict_action)
            test_eval = eval_accuracy_from_predict_fn(test_trials, predict_action)

            results = {
                "method": "prospect_theory_MLE",
                "dataset": args.dataset,
                "participant_id": participant_id,
                "fitted_params": params,
                "train_accuracy": train_eval["accuracy"],
                "test_accuracy": test_eval["accuracy"],
                "n_train": train_eval["total"],
                "n_test": test_eval["total"],
                "train_correct": train_eval["correct"],
                "test_correct": test_eval["correct"],
            }

        elif args.dataset == "cpc18":
            # CPC18: action=0 means choose A, action=1 means choose B.
            # We'll model P(choose B) = sigmoid(beta*(V_B - V_A)).
            default_theta = np.array([0.8, 2.0, 1.0, 1.0], dtype=np.float64)  # alpha, lambda, gamma, beta
            yB = np.asarray([1.0 if int(t["action"]) == 1 else 0.0 for t in train_trials], dtype=np.float64)
            if len(train_trials) == 0:
                print(
                    f"[Warning][prospect_theory] Degenerate train set (empty). "
                    f"Using default params. dataset={args.dataset} participant_id={participant_id}"
                )
                params = {"alpha": float(default_theta[0]), "lambda": float(default_theta[1]), "gamma": float(default_theta[2]), "beta": float(default_theta[3])}
            elif np.all(yB == yB[0]):
                print(
                    f"[Warning][prospect_theory] Degenerate train labels (single class). "
                    f"Using default params. dataset={args.dataset} participant_id={participant_id}"
                )
                params = {"alpha": float(default_theta[0]), "lambda": float(default_theta[1]), "gamma": float(default_theta[2]), "beta": float(default_theta[3])}
            else:

                def nll(theta: np.ndarray) -> float:
                    alpha, lam, gamma, beta = float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])
                    VB = []
                    VA = []
                    for tr in train_trials:
                        prob = tr["problem"]
                        # Option A
                        rA = [float(prob["Ha"]), float(prob["La"])]
                        pA = [float(prob["pHa"]), float(1.0 - prob["pHa"])]
                        # Option B
                        rB = [float(prob["Hb"]), float(prob["Lb"])]
                        pB = [float(prob["pHb"]), float(1.0 - prob["pHb"])]
                        VA.append(subjective_value_gamble(rA, pA, alpha, lam, gamma))
                        VB.append(subjective_value_gamble(rB, pB, alpha, lam, gamma))
                    VA_arr = np.asarray(VA, dtype=np.float64)
                    VB_arr = np.asarray(VB, dtype=np.float64)
                    p_choose_B = sigmoid(beta * (VB_arr - VA_arr))
                    p_choose_B = np.clip(p_choose_B, 1e-9, 1.0 - 1e-9)
                    return float(-np.sum(yB * np.log(p_choose_B) + (1.0 - yB) * np.log(1.0 - p_choose_B)))

                bounds = [(0.01, 2.0), (0.01, 10.0), (0.01, 5.0), (0.01, 20.0)]
                res = minimize(
                    nll,
                    x0=default_theta,
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                if not getattr(res, "success", False):
                    msg = getattr(res, "message", "")
                    print(
                        f"[Warning][prospect_theory] Optimization failed; using best found params. "
                        f"dataset={args.dataset} participant_id={participant_id} message={msg}"
                    )
                params = {"alpha": float(res.x[0]), "lambda": float(res.x[1]), "gamma": float(res.x[2]), "beta": float(res.x[3])}

            def predict_action(tr: Dict[str, Any]) -> int:
                prob = tr["problem"]
                rA = [float(prob["Ha"]), float(prob["La"])]
                pA = [float(prob["pHa"]), float(1.0 - prob["pHa"])]
                rB = [float(prob["Hb"]), float(prob["Lb"])]
                pB = [float(prob["pHb"]), float(1.0 - prob["pHb"])]
                VA = subjective_value_gamble(rA, pA, params["alpha"], params["lambda"], params["gamma"])
                VB = subjective_value_gamble(rB, pB, params["alpha"], params["lambda"], params["gamma"])
                pB_choice = float(sigmoid(params["beta"] * (VB - VA)))
                return 1 if pB_choice >= 0.5 else 0

            train_acc_eval = eval_accuracy_from_predict_fn(train_trials, predict_action)
            test_acc_eval = eval_accuracy_from_predict_fn(test_trials, predict_action)
            # TE convention: train_observed_blocks == test_observed_blocks
            train_mse_eval = eval_cpc18_mse_from_predict_fn(train_trials, test_observed_blocks, predict_action)
            test_mse_eval = eval_cpc18_mse_from_predict_fn(test_trials, test_observed_blocks, predict_action)

            results = {
                "method": "prospect_theory_MLE",
                "dataset": args.dataset,
                "participant_id": participant_id,
                "fitted_params": params,
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
                "train_acc": train_fitness,
                "test_acc": test_fitness,
            }

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
            participant_output_dir = os.path.join(base_run_dir, f"participant_{participant_id}")
            Path(participant_output_dir).mkdir(parents=True, exist_ok=True)
            (Path(participant_output_dir) / "results.json").write_text(json.dumps(results, indent=2))

            if args.dataset == "cpc18":
                participants_summary.append(row)
            else:
                participants_summary.append(
                    {
                        "participant_id": participant_id,
                        "train_acc": float(results.get("train_accuracy", float("nan"))),
                        "test_acc": float(results.get("test_accuracy", float("nan"))),
                    }
                )
            with open(summary_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
                writer.writeheader()
                writer.writerows(participants_summary)

        # W&B logging (TE-aligned keys)
        if wandb is not None:
            prefix = f"p{participant_id}"
            if args.dataset == "cpc18":
                wandb.log(
                    {
                        f"{prefix}_train_fitness": float(row["train_fitness"]),
                        f"{prefix}_train_mse": float(row["train_mse"]),
                        f"{prefix}_test_mse": float(row["test_mse"]),
                        f"{prefix}_is_baseline": 1,
                        f"{prefix}_train_accuracy": float(results.get("train_accuracy", float("nan"))),
                        f"{prefix}_test_accuracy": float(results.get("test_accuracy", float("nan"))),
                    },
                    step=0,
                )
            else:
                wandb.log(
                    {
                        f"{prefix}_train_accuracy": float(row["train_acc"]),
                        f"{prefix}_test_accuracy": float(row["test_acc"]),
                        f"{prefix}_is_baseline": 1,
                    },
                    step=0,
                )

    if wandb is not None:
        wandb.finish()

    # Final mean across processed participants (TE-style reporting)
    if args.all_data:
        train_mean = float(np.mean([r["train_fitness"] for r in participants_details])) if participants_details else 0.0
        test_mean = float(np.mean([r["test_fitness"] for r in participants_details])) if participants_details else 0.0
        print(f"\n[Prospect Theory baseline] dataset={args.dataset} participants={participants_to_process}")
        print(f"Mean train fitness: {train_mean:.4f}")
        print(f"Mean test fitness:  {test_mean:.4f}")
    elif args.dataset == "cpc18":
        train_mean = float(np.mean([r["train_fitness"] for r in participants_summary])) if participants_summary else 0.0
        test_mean = float(np.mean([r["test_mse"] for r in participants_summary])) if participants_summary else 0.0
        print(f"\n[Prospect Theory baseline] dataset={args.dataset} participants={participants_to_process}")
        print(f"Mean train fitness (-MSE): {train_mean:.4f}")
        print(f"Mean test MSE (official):  {test_mean:.4f}")
    else:
        train_mean = float(np.mean([r["train_acc"] for r in participants_summary])) if participants_summary else 0.0
        test_mean = float(np.mean([r["test_acc"] for r in participants_summary])) if participants_summary else 0.0
        print(f"\n[Prospect Theory baseline] dataset={args.dataset} participants={participants_to_process}")
        print(f"Mean train accuracy: {train_mean:.4f}")
        print(f"Mean test accuracy:  {test_mean:.4f}")
    print(f"Results saved under: {base_run_dir}")


if __name__ == "__main__":
    main()

