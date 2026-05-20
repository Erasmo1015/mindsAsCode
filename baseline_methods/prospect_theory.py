#!/usr/bin/env python3
"""
Prospect Theory MLE baseline for TEH participant datasets (Psych-101 binary aliases + mixed_gambles).

CLI flags for dataset, participants, and splits match `teh.py` and `baseline_methods/MLE.py`.
Fit prospect theory by MLE on the combined train+val split; report accuracy and mean Bernoulli
log-likelihood.

python baseline_methods/prospect_theory.py --dataset 1peterson2021using --psych_dataset_split train \\
  --participant_scope range --range_start_ordinal 0 --range_end_ordinal 9 \\
  --split_mode within_participant --split_ratio 0.6 --split_seed 0 --fitness_metric loglik
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials
from baseline_methods.psych101_features import prospect_gamble_getters
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    PETERSON2021USING_ALIAS,
    PSYCH101_LEGACY_ALIASES,
    experiment_to_trial_dicts,
    get_psych101_binary_experiment,
    get_filtered_psych101_split,
    hf_id_for_psych_dataset_split,
    is_psych101_dataset,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
    split_psych_experiment,
)
from utils.teh.participant_ids import load_valid_participant_ids
from utils.teh.teh_datasets import (
    PARTICIPANT_DATASETS,
    is_binary_loglik_dataset,
    is_mixed_gambles_dataset,
)

_PARTICIPANT_DATASETS = PARTICIPANT_DATASETS
_PETERSON_ALIAS = PETERSON2021USING_ALIAS


def _effective_psych_dataset_split(dataset: str, psych_dataset_split: str) -> str:
    if is_mixed_gambles_dataset(dataset):
        return DEFAULT_PSYCH_DATASET_SPLIT
    return normalize_psych_dataset_split(psych_dataset_split)


def load_valid_participant_ids_from_json(
    dataset: str,
    repo_root: Path,
    filter_mixed_gambles: bool = False,
    *,
    split_ratio: float = 0.8,
    split_seed: int = 42,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
    auto_prepare: bool = True,
) -> List[int]:
    if not is_binary_loglik_dataset(dataset):
        raise ValueError(f"load_valid_participant_ids_from_json: unsupported dataset {dataset!r}")
    return load_valid_participant_ids(
        dataset,
        repo_root,
        filter_mixed_gambles=filter_mixed_gambles,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=_effective_psych_dataset_split(dataset, psych_dataset_split),
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        auto_prepare=auto_prepare,
    )


def resolve_participants_for_scope(
    *,
    dataset: str,
    repo_root: Path,
    participant_scope: str,
    single_participant_id: int,
    range_start_ordinal: Optional[int],
    range_end_ordinal: Optional[int],
    all_max_participants: Optional[int],
    participant_ordinals: Optional[List[int]],
    filter_mixed_gambles: bool,
    split_ratio: float = 0.8,
    split_seed: int = 42,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
    local_dataset: Optional[str] = None,
    mixed_gambles_csv: str = DEFAULT_CSV_PATH,
) -> List[int]:
    valid = load_valid_participant_ids_from_json(
        dataset,
        repo_root,
        filter_mixed_gambles,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=psych_dataset_split,
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
    )
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
    if participant_scope == "ordinals":
        if not participant_ordinals:
            raise ValueError(
                "--participant_scope ordinals requires --ordinals with one or more integers "
                "(0-based indices into valid_participant_ids.json), e.g. --ordinals 0 4 9."
            )
        out: List[int] = []
        seen: set[int] = set()
        for o in participant_ordinals:
            oi = int(o)
            if oi < 0 or oi >= len(valid):
                raise ValueError(
                    f"Ordinal {oi} is out of range for valid list of length {len(valid)} "
                    f"(valid indices: 0..{len(valid) - 1})."
                )
            pid = int(valid[oi])
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out
    if participant_scope == "all":
        if all_max_participants is not None:
            n = max(0, int(all_max_participants))
            return valid[:n]
        return list(valid)
    raise ValueError(f"Unknown participant_scope: {participant_scope!r}")


def pt_output_base_dir(
    dataset: str,
    timestamp: str,
    *,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
) -> str:
    if is_mixed_gambles_dataset(dataset):
        return f"generated_outputs/mixed_gambles/prospect_theory/run_{timestamp}"
    split = normalize_psych_dataset_split(psych_dataset_split)
    return f"generated_outputs/psych101_{split}/prospect_theory/{dataset}/run_{timestamp}"


def trials_for_participant(
    dataset: str,
    participant_id: int,
    *,
    split_ratio: float,
    split_seed: int,
    filter_mixed_gambles: bool,
    psych_dataset_split: str,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if is_mixed_gambles_dataset(dataset):
        csv_path = mixed_gambles_csv or DEFAULT_CSV_PATH
        train_trials, val_trials, test_trials, _ = load_mixed_gambles_trials(
            participant_id,
            csv_path=csv_path,
            filter_gain_loss_only=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        return train_trials, val_trials, test_trials
    if not is_psych101_dataset(dataset):
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    exp = get_psych101_binary_experiment(
        dataset,
        int(participant_id),
        split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    train_trials, val_trials, test_trials, _ = split_psych_experiment(
        exp, split_ratio=split_ratio, split_seed=split_seed
    )
    return train_trials, val_trials, test_trials


def psych_across_participants_train_test(
    dataset: str,
    selected_participants: List[int],
    split_ratio: float,
    split_seed: int,
    *,
    psych_dataset_split: str,
    local_dataset: Optional[str],
) -> Tuple[List[int], List[int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(selected_participants) < 2:
        raise ValueError("across_participants requires at least 2 selected participants.")
    rng = np.random.default_rng(split_seed)
    shuffled = list(selected_participants)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * split_ratio)
    split_idx = max(1, min(split_idx, len(shuffled) - 1))
    train_participants = shuffled[:split_idx]
    test_participants = shuffled[split_idx:]
    filtered = get_filtered_psych101_split(
        dataset, split=psych_dataset_split, local_dataset=local_dataset
    )
    train_trials: List[Dict[str, Any]] = []
    test_trials: List[Dict[str, Any]] = []
    for pid in train_participants:
        exp = get_psych101_binary_experiment(
            dataset,
            pid,
            split=psych_dataset_split,
            local_dataset=local_dataset,
            filtered_split=filtered,
        )
        train_trials.extend(experiment_to_trial_dicts(exp))
    for pid in test_participants:
        exp = get_psych101_binary_experiment(
            dataset,
            pid,
            split=psych_dataset_split,
            local_dataset=local_dataset,
            filtered_split=filtered,
        )
        test_trials.extend(experiment_to_trial_dicts(exp))
    return train_participants, test_participants, train_trials, test_trials


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
    gambleA_getter: Callable[
        [Dict[str, Any], Optional[List[Dict[str, Any]]]],
        Tuple[List[float], Optional[List[float]]],
    ],
    gambleB_getter: Callable[
        [Dict[str, Any], Optional[List[Dict[str, Any]]]],
        Tuple[List[float], Optional[List[float]]],
    ],
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
            hist = tr.get("history")
            rA, prA = gambleA_getter(p, hist)
            rB, prB = gambleB_getter(p, hist)
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


def eval_mean_loglik_choice13k_prospect(
    trials: List[Dict[str, Any]],
    params: Dict[str, float],
) -> float:
    """Mean Bernoulli log-likelihood under fitted prospect model (P(choose A)=sigmoid(beta*(VA-VB)))."""
    if not trials:
        return float("nan")
    alpha = float(params["alpha"])
    lam = float(params["lambda"])
    gamma = float(params["gamma"])
    beta = float(params["beta"])
    total = 0.0
    ga, gb = prospect_gamble_getters(trials[0]["problem"])
    for tr in trials:
        p = tr["problem"]
        hist = tr.get("history")
        r_a, pr_a = ga(p, hist)
        r_b, pr_b = gb(p, hist)
        va = subjective_value_gamble(r_a, pr_a, alpha, lam, gamma)
        vb = subjective_value_gamble(r_b, pr_b, alpha, lam, gamma)
        p_choose_a = float(sigmoid(beta * (va - vb)))
        p_choose_a = min(max(p_choose_a, 1e-9), 1.0 - 1e-9)
        y = int(tr["action"])
        if y == 0:
            total += np.log(p_choose_a)
        else:
            total += np.log(1.0 - p_choose_a)
    return float(total / len(trials))


def _round_floats_for_csv_row(row: Dict[str, Any], ndigits: int = 4) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (float, np.floating)):
            x = float(v)
            out[k] = round(x, ndigits) if math.isfinite(x) else x
        else:
            out[k] = v
    return out


def _round_floats_for_csv_rows(rows: List[Dict[str, Any]], ndigits: int = 4) -> List[Dict[str, Any]]:
    return [_round_floats_for_csv_row(r, ndigits) for r in rows]


def _loglik_row_from_results(participant_id: int, results: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "participant_id": participant_id,
        "train_loglik": results.get("train_mean_loglik"),
        "val_loglik": results.get("val_mean_loglik"),
        "test_loglik": results.get("test_mean_loglik"),
    }


def _write_experiment_loglik_csvs(
    base_run_dir: str | Path,
    participant_details_loglik: List[Dict[str, Any]],
) -> None:
    base = Path(base_run_dir)
    details_fields = ["participant_id", "train_loglik", "val_loglik", "test_loglik"]
    with (base / "participant_details_loglik.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_round_floats_for_csv_rows(participant_details_loglik))

    train_vals = [d["train_loglik"] for d in participant_details_loglik if d.get("train_loglik") is not None]
    val_vals = [d["val_loglik"] for d in participant_details_loglik if d.get("val_loglik") is not None]
    test_vals = [d["test_loglik"] for d in participant_details_loglik if d.get("test_loglik") is not None]
    summary_row = {
        "num_of_participants": len(participant_details_loglik),
        "avg_train_loglik": float(np.mean(train_vals)) if train_vals else None,
        "avg_test_loglik": float(np.mean(test_vals)) if test_vals else None,
        "avg_val_loglik": float(np.mean(val_vals)) if val_vals else None,
        "avg_gated_test_loglik": None,
    }
    summary_fields = [
        "num_of_participants",
        "avg_train_loglik",
        "avg_test_loglik",
        "avg_val_loglik",
        "avg_gated_test_loglik",
    ]
    with (base / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(_round_floats_for_csv_row(summary_row))


def _make_predict_action(
    params: Dict[str, float],
    gambleA_getter: Callable[
        [Dict[str, Any], Optional[List[Dict[str, Any]]]],
        Tuple[List[float], Optional[List[float]]],
    ],
    gambleB_getter: Callable[
        [Dict[str, Any], Optional[List[Dict[str, Any]]]],
        Tuple[List[float], Optional[List[float]]],
    ],
) -> Callable[[Dict[str, Any]], int]:
    def predict_action(tr: Dict[str, Any]) -> int:
        p = tr["problem"]
        hist = tr.get("history")
        rA, prA = gambleA_getter(p, hist)
        rB, prB = gambleB_getter(p, hist)
        va = subjective_value_gamble(rA, prA, params["alpha"], params["lambda"], params["gamma"])
        vb = subjective_value_gamble(rB, prB, params["alpha"], params["lambda"], params["gamma"])
        p_a = float(sigmoid(params["beta"] * (va - vb)))
        return 0 if p_a >= 0.5 else 1

    return predict_action


def _fit_and_evaluate_participant(
    dataset: str,
    participant_id: int,
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fit_trials = train_trials + val_trials
    if not fit_trials:
        raise ValueError(f"No training trials for participant {participant_id}.")
    sample_problem = fit_trials[0]["problem"]
    ga, gb = prospect_gamble_getters(sample_problem)
    params = fit_prospect_theory_gamble_choice(
        fit_trials,
        action_is_chooseA=lambda a: a == 0,
        gambleA_getter=ga,
        gambleB_getter=gb,
        dataset=dataset,
        participant_id=participant_id,
    )
    predict_action = _make_predict_action(params, ga, gb)
    train_acc = eval_accuracy_from_predict_fn(train_trials, predict_action)
    val_acc = eval_accuracy_from_predict_fn(val_trials, predict_action)
    test_acc = eval_accuracy_from_predict_fn(test_trials, predict_action)
    return {
        "method": "prospect_theory_MLE",
        "dataset": dataset,
        "participant_id": participant_id,
        "fitted_params": params,
        "train_accuracy": train_acc["accuracy"],
        "val_accuracy": val_acc["accuracy"],
        "test_accuracy": test_acc["accuracy"],
        "train_mean_loglik": eval_mean_loglik_choice13k_prospect(train_trials, params),
        "val_mean_loglik": eval_mean_loglik_choice13k_prospect(val_trials, params),
        "test_mean_loglik": eval_mean_loglik_choice13k_prospect(test_trials, params),
        "n_train": train_acc["total"],
        "n_val": val_acc["total"],
        "n_test": test_acc["total"],
    }


def _print_across_summary(
    args: argparse.Namespace,
    train_ll: float,
    test_ll: float,
    train_acc_eval: Dict[str, float],
    test_acc_eval: Dict[str, float],
    base_run_dir: str,
) -> None:
    if args.fitness_metric == "loglik":
        print(
            f"\n[Prospect Theory baseline] {args.dataset} across_participants "
            f"train_mean_loglik={train_ll:.6f} test_mean_loglik={test_ll:.6f}"
        )
    else:
        print(
            f"\n[Prospect Theory baseline] {args.dataset} across_participants "
            f"train_acc={train_acc_eval['accuracy']:.4f} test_acc={test_acc_eval['accuracy']:.4f}"
        )
    print(f"Results saved under: {base_run_dir}")


def _print_within_summary(
    args: argparse.Namespace,
    participants_summary: List[Dict[str, Any]],
    participants_to_process: List[int],
    base_run_dir: str,
) -> None:
    print(f"\n[Prospect Theory baseline] dataset={args.dataset} participants={participants_to_process}")
    if not participants_summary:
        print("No participants processed.")
        print(f"Results saved under: {base_run_dir}")
        return
    if args.fitness_metric == "loglik":
        train_ll = float(
            np.mean([r["train_mean_loglik"] for r in participants_summary if r["train_mean_loglik"] is not None])
        )
        test_ll = float(
            np.mean([r["test_mean_loglik"] for r in participants_summary if r["test_mean_loglik"] is not None])
        )
        print(f"Mean train loglik: {train_ll:.6f}")
        print(f"Mean test loglik:  {test_ll:.6f}")
    else:
        train_acc = float(np.mean([r["train_acc"] for r in participants_summary if r["train_acc"] is not None]))
        test_acc = float(np.mean([r["test_acc"] for r in participants_summary if r["test_acc"] is not None]))
        print(f"Mean train accuracy: {train_acc:.4f}")
        print(f"Mean test accuracy:  {test_acc:.4f}")
    print(f"Results saved under: {base_run_dir}")


def _add_te_compat_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_path", type=str, default="data", help=argparse.SUPPRESS)
    parser.add_argument("--seed_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no_llm_prompt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--n_iterations", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--n_candidates", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--sample_size", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument(
        "--sample_parents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--elite_pool_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n_eval_seeds", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--model_name", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--mode", type=str, default="default", help=argparse.SUPPRESS)
    parser.add_argument("--llm_server_url", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--llm_api_key", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--max_prompt_train_trials", type=int, default=1_000_000, help=argparse.SUPPRESS)
    parser.add_argument("--max_prompt_trials_per_problem", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--llm_max_tokens", type=int, default=800, help=argparse.SUPPRESS)
    parser.add_argument("--max_workers", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument(
        "--parallel_participants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gate_phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--refinement_phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--refinement_iters", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--refinement_val_threshold", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--phase", type=str, default="all", help=argparse.SUPPRESS)
    parser.add_argument(
        "--global_phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--global_iters", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--prev_exp_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cpc18_official_mse", action="store_true", help=argparse.SUPPRESS)


def main() -> None:
    _teh_dataset_choices = sorted(_PARTICIPANT_DATASETS | set(PSYCH101_LEGACY_ALIASES))
    parser = argparse.ArgumentParser(
        description="Prospect Theory MLE baseline (TEH-compatible dataset / participant / split CLI)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=_PETERSON_ALIAS,
        choices=_teh_dataset_choices,
        help=(
            "Psych-101 binary alias (1peterson2021using, 2plonsky2018when, ...) or mixed_gambles (local CSV)."
        ),
    )
    parser.add_argument(
        "--mixed_gambles_csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="CSV path for --dataset mixed_gambles (default: datasets/mixed_gambles/data_all_2021-01-08.csv).",
    )
    parser.add_argument(
        "--psych_dataset_split",
        type=str,
        default=DEFAULT_PSYCH_DATASET_SPLIT,
        choices=sorted({"train", "test"}),
        help=(
            "Psych-101 HF participant corpus: train=marcelbinz/Psych-101, "
            "test=marcelbinz/Psych-101-test (default: train). Ignored for mixed_gambles."
        ),
    )
    parser.add_argument(
        "--local_dataset",
        type=str,
        default=None,
        help="Optional path from datasets.load_from_disk for Psych-101 (else Hugging Face hub).",
    )
    parser.add_argument(
        "--filter_mixed_gambles",
        action="store_true",
        default=False,
        help=(
            "For mixed_gambles: keep only gain_loss trials. Default False (all trial types). "
            "Affects which valid_participant_ids.json variant is used for ordinal resolution."
        ),
    )
    parser.add_argument(
        "--participant_scope",
        type=str,
        default="single",
        choices=["single", "range", "ordinals", "all"],
        help=(
            "How to select participants. 'single' uses --single_participant_id (raw id). "
            "'range' uses --range_start_ordinal/--range_end_ordinal (inclusive) into valid_participant_ids.json. "
            "'ordinals' uses --ordinals (0-based indices into that list). "
            "'all' runs all valid ids, optionally capped by --all_max_participants."
        ),
    )
    parser.add_argument(
        "--single_participant_id",
        type=int,
        default=0,
        help="Raw participant id when --participant_scope single (must appear in valid_participant_ids.json). Default 0.",
    )
    parser.add_argument(
        "--range_start_ordinal",
        type=int,
        default=None,
        help="0-based start index into valid_participant_ids.json when --participant_scope range.",
    )
    parser.add_argument(
        "--range_end_ordinal",
        type=int,
        default=None,
        help="0-based inclusive end index into valid_participant_ids.json when --participant_scope range.",
    )
    parser.add_argument(
        "--all_max_participants",
        type=int,
        default=None,
        help="When --participant_scope all: use only the first N valid raw ids from JSON. Omit to run all valids.",
    )
    parser.add_argument(
        "--ordinals",
        nargs="+",
        type=int,
        default=None,
        metavar="I",
        help=(
            "When --participant_scope ordinals: 0-based ordinals into datasets/*/valid_participant_ids.json "
            "(same ordering as range), not raw participant ids. Example: --ordinals 0 4 9"
        ),
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="within_participant",
        choices=["within_participant", "across_participants"],
        help=(
            "within_participant (default): train/val/test per participant. "
            "across_participants: pool train across participants (Psych-101 binary datasets)."
        ),
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.6,
        help="Global train ratio for splitting (default: 0.6).",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Seed for deterministic splitting (default: 0).",
    )
    parser.add_argument(
        "--fitness_metric",
        type=str,
        default="loglik",
        choices=["accuracy", "loglik"],
        help="Primary metric for printed summary (accuracy and mean loglik are always computed when applicable).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated).",
    )
    parser.add_argument(
        "--no_log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable wandb logging (default: disabled). Pass --no-no-log to enable.",
    )
    _add_te_compat_args(parser)
    args = parser.parse_args()
    args.dataset = normalize_psych101_dataset_alias(args.dataset)
    mixed_gambles_gain_loss_only = bool(args.filter_mixed_gambles)
    psych_dataset_split = _effective_psych_dataset_split(args.dataset, args.psych_dataset_split)

    if not is_binary_loglik_dataset(args.dataset):
        print(f"Error: --dataset must be one of {sorted(_PARTICIPANT_DATASETS)}.")
        sys.exit(1)
    if not (0.0 < args.split_ratio < 1.0):
        print(f"Error: --split_ratio must be in (0,1), got {args.split_ratio}.")
        sys.exit(1)
    if args.split_mode == "across_participants" and not is_psych101_dataset(args.dataset):
        print(
            "Error: --split_mode across_participants is only supported for Psych-101 binary datasets "
            "(not mixed_gambles)."
        )
        sys.exit(1)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    if args.split_mode == "across_participants":
        try:
            selected_participants = resolve_participants_for_scope(
                dataset=args.dataset,
                repo_root=_REPO_ROOT,
                participant_scope=args.participant_scope,
                single_participant_id=args.single_participant_id,
                range_start_ordinal=args.range_start_ordinal,
                range_end_ordinal=args.range_end_ordinal,
                all_max_participants=args.all_max_participants,
                participant_ordinals=args.ordinals,
                filter_mixed_gambles=mixed_gambles_gain_loss_only,
                split_ratio=args.split_ratio,
                split_seed=args.split_seed,
                psych_dataset_split=psych_dataset_split,
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        if len(selected_participants) < 2:
            print("Error: across_participants split requires at least 2 selected participants.")
            sys.exit(1)
        base_run_dir_ap = (
            args.output_dir
            if args.output_dir
            else pt_output_base_dir(args.dataset, timestamp, psych_dataset_split=psych_dataset_split)
        )
        Path(base_run_dir_ap).mkdir(parents=True, exist_ok=True)
        train_p, test_p, train_trials, test_trials = psych_across_participants_train_test(
            args.dataset,
            selected_participants,
            args.split_ratio,
            args.split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=args.local_dataset,
        )
        ga, gb = prospect_gamble_getters(train_trials[0]["problem"])
        params = fit_prospect_theory_gamble_choice(
            train_trials,
            action_is_chooseA=lambda a: a == 0,
            gambleA_getter=ga,
            gambleB_getter=gb,
            dataset=args.dataset,
            participant_id=None,
        )
        predict_action_ap = _make_predict_action(params, ga, gb)
        train_acc_eval = eval_accuracy_from_predict_fn(train_trials, predict_action_ap)
        test_acc_eval = eval_accuracy_from_predict_fn(test_trials, predict_action_ap)
        train_ll = eval_mean_loglik_choice13k_prospect(train_trials, params)
        test_ll = eval_mean_loglik_choice13k_prospect(test_trials, params)
        results_ap = {
            "method": "prospect_theory_MLE",
            "dataset": args.dataset,
            "split_mode": "across_participants",
            "train_participants": train_p,
            "test_participants": test_p,
            "fitted_params": params,
            "train_accuracy": train_acc_eval["accuracy"],
            "test_accuracy": test_acc_eval["accuracy"],
            "train_mean_loglik": train_ll,
            "test_mean_loglik": test_ll,
            "n_train_trials": train_acc_eval["total"],
            "n_test_trials": test_acc_eval["total"],
        }
        (Path(base_run_dir_ap) / "results.json").write_text(json.dumps(results_ap, indent=2))
        summary_fields = [
            "num_of_participants",
            "avg_train_loglik",
            "avg_test_loglik",
            "avg_val_loglik",
            "avg_gated_test_loglik",
        ]
        summary_row = _round_floats_for_csv_row(
            {
                "num_of_participants": len(selected_participants),
                "avg_train_loglik": train_ll,
                "avg_test_loglik": test_ll,
                "avg_val_loglik": None,
                "avg_gated_test_loglik": None,
            }
        )
        with (Path(base_run_dir_ap) / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerow(summary_row)
        _print_across_summary(args, train_ll, test_ll, train_acc_eval, test_acc_eval, base_run_dir_ap)
        return

    try:
        participants_to_process = resolve_participants_for_scope(
            dataset=args.dataset,
            repo_root=_REPO_ROOT,
            participant_scope=args.participant_scope,
            single_participant_id=args.single_participant_id,
            range_start_ordinal=args.range_start_ordinal,
            range_end_ordinal=args.range_end_ordinal,
            all_max_participants=args.all_max_participants,
            participant_ordinals=args.ordinals,
            filter_mixed_gambles=mixed_gambles_gain_loss_only,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            psych_dataset_split=psych_dataset_split,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if is_psych101_dataset(args.dataset):
        print(
            f"Psych-101 HF corpus: {psych_dataset_split} -> {hf_id_for_psych_dataset_split(psych_dataset_split)}"
        )
    print(
        f"Prospect Theory split settings: dataset={args.dataset}, split_mode={args.split_mode}, "
        f"split_ratio={args.split_ratio:.3f}, split_seed={args.split_seed}"
    )

    base_run_dir = args.output_dir
    if base_run_dir is None:
        base_run_dir = pt_output_base_dir(args.dataset, timestamp, psych_dataset_split=psych_dataset_split)
    Path(base_run_dir).mkdir(parents=True, exist_ok=True)

    participant_details_loglik: List[Dict[str, Any]] = []
    participants_summary: List[Dict[str, Any]] = []

    for participant_id in tqdm(participants_to_process, desc="Participants"):
        train_trials, val_trials, test_trials = trials_for_participant(
            args.dataset,
            participant_id,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
            filter_mixed_gambles=mixed_gambles_gain_loss_only,
            psych_dataset_split=psych_dataset_split,
            local_dataset=args.local_dataset,
            mixed_gambles_csv=args.mixed_gambles_csv,
        )
        results = _fit_and_evaluate_participant(
            args.dataset,
            participant_id,
            train_trials,
            val_trials,
            test_trials,
        )
        participants_summary.append(
            {
                "participant_id": participant_id,
                "train_acc": results.get("train_accuracy"),
                "test_acc": results.get("test_accuracy"),
                "val_acc": results.get("val_accuracy"),
                "train_mean_loglik": results.get("train_mean_loglik"),
                "val_mean_loglik": results.get("val_mean_loglik"),
                "test_mean_loglik": results.get("test_mean_loglik"),
            }
        )
        participant_details_loglik.append(_loglik_row_from_results(participant_id, results))
        participant_output_dir = os.path.join(base_run_dir, f"participant_{participant_id}")
        Path(participant_output_dir).mkdir(parents=True, exist_ok=True)
        (Path(participant_output_dir) / "results.json").write_text(json.dumps(results, indent=2))

    _write_experiment_loglik_csvs(base_run_dir, participant_details_loglik)
    _print_within_summary(args, participants_summary, participants_to_process, base_run_dir)


if __name__ == "__main__":
    main()

