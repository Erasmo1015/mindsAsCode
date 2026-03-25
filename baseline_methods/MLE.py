#!/usr/bin/env python3
"""
Logistic MLE baselines for Choice13k, CPC18 (Track II), and Mixed Gambles.

Data loading/splitting conventions are copied to match `Template_evo_non_strict.py`
for the three datasets exactly:
  - choice13k: fixed 80/20 split with RNG seed=42 (history accumulated in order)
  - cpc18: NO artificial split (train_ratio ignored; use all trials; CPC18 official MSE computed)
  - mixed_gambles: participant filtering + optional gain_loss filtering + 80/20 split with RNG seed=42

Method differs from Template Evolution:
  - Fit logistic MLE on the training split
  - Evaluate on the test split

python baseline_methods/MLE.py --dataset cpc18 --participant_id 0
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

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    wandb = None
    if not args.no_log:
        try:
            import wandb as _wandb
            wandb = _wandb

            dataset_prefix = args.dataset if args.dataset else "choice13k"
            run_name = f"{dataset_prefix}_MLE_{timestamp}"
            if args.participant_id is not None:
                run_name = f"{run_name}_participant_{args.participant_id}"
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
    if args.participant_id is not None:
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
    summary_file = Path(base_run_dir) / "participants_summary.csv"

    participants_summary: List[Dict[str, Any]] = []
    summary_fieldnames = (
        ["participant_id", "train_fitness", "train_mse", "test_mse"]
        if args.dataset == "cpc18"
        else ["participant_id", "train_acc", "test_acc"]
    )

    for participant_id in tqdm(participants_to_process, desc="Participants"):
        if args.dataset == "choice13k":
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
            results = {
                "method": "logistic_MLE",
                "dataset": args.dataset,
                "participant_id": participant_id,
                "fitted_params": {"beta": beta_hat, "bias": bias_hat},
                "train_accuracy": train_acc_eval["accuracy"],
                "test_accuracy": test_acc_eval["accuracy"],
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

        # Save outputs in TE-like structure
        participant_output_dir = os.path.join(base_run_dir, f"participant_{participant_id}")
        Path(participant_output_dir).mkdir(parents=True, exist_ok=True)
        (Path(participant_output_dir) / "results.json").write_text(json.dumps(results, indent=2))

        # Update participants summary with TE-compatible fieldnames
        if args.dataset == "cpc18":
            train_mse = float(results.get("train_mse", float("nan")))
            test_mse = float(results.get("test_mse", float("nan")))
            row = {
                "participant_id": participant_id,
                # TE convention: fitness = -MSE (higher is better)
                "train_fitness": -train_mse,
                "train_mse": train_mse,
                "test_mse": test_mse,
            }
        else:
            row = {
                "participant_id": participant_id,
                "train_acc": float(results.get("train_accuracy", float("nan"))),
                "test_acc": float(results.get("test_accuracy", float("nan"))),
            }
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
                wandb.log(
                    {
                        f"{prefix}_train_accuracy": row["train_acc"],
                        f"{prefix}_test_accuracy": row["test_acc"],
                        f"{prefix}_is_baseline": 1,
                    },
                    step=0,
                )

    if wandb is not None:
        wandb.finish()

    # Print final mean across processed participants
    if args.dataset == "cpc18":
        train_mean = float(np.mean([r["train_fitness"] for r in participants_summary])) if participants_summary else 0.0
        test_mean = float(np.mean([r["test_mse"] for r in participants_summary])) if participants_summary else 0.0
        print(f"\n[MLE baseline] dataset={args.dataset} participants={participants_to_process}")
        print(f"Mean train fitness (-MSE): {train_mean:.4f}")
        print(f"Mean test MSE (official):  {test_mean:.4f}")
    else:
        train_mean = float(np.mean([r["train_acc"] for r in participants_summary])) if participants_summary else 0.0
        test_mean = float(np.mean([r["test_acc"] for r in participants_summary])) if participants_summary else 0.0
        print(f"\n[MLE baseline] dataset={args.dataset} participants={participants_to_process}")
        print(f"Mean train accuracy: {train_mean:.4f}")
        print(f"Mean test accuracy:  {test_mean:.4f}")
    print(f"Results saved under: {base_run_dir}")


if __name__ == "__main__":
    main()

