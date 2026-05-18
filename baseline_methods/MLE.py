#!/usr/bin/env python3
"""
Logistic MLE baseline for TEH participant datasets (Psych-101 binary aliases + mixed_gambles).

CLI flags for dataset, participants, and splits match `teh.py` exactly. Fit logistic MLE on the
train split; report accuracy and mean Bernoulli log-likelihood on train/val/test.

python baseline_methods/MLE.py --dataset peterson2021using --psych_dataset_split train \\
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
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    experiment_to_trial_dicts,
    get_psych101_binary_experiments,
    hf_id_for_psych_dataset_split,
    is_psych101_dataset,
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
_PETERSON_ALIAS = "peterson2021using"


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
    """Load valid participant ids (same as `teh.load_valid_participant_ids_from_json`)."""
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
    """Same semantics as `teh.resolve_participants_for_scope`."""
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


def mle_output_base_dir(
    dataset: str,
    timestamp: str,
    *,
    psych_dataset_split: str = DEFAULT_PSYCH_DATASET_SPLIT,
) -> str:
    if is_mixed_gambles_dataset(dataset):
        return f"generated_outputs/mixed_gambles/MLE/run_{timestamp}"
    split = normalize_psych_dataset_split(psych_dataset_split)
    return f"generated_outputs/psych101_{split}/MLE/{dataset}/run_{timestamp}"


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
    """Train/val/test trials for one participant (TEH split conventions)."""
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
    experiments = get_psych101_binary_experiments(
        dataset,
        n_participants=int(participant_id) + 1,
        split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    if participant_id >= len(experiments):
        raise ValueError(
            f"participant_id={participant_id} out of range (only {len(experiments)} experiments loaded)."
        )
    train_trials, val_trials, test_trials, _ = split_psych_experiment(
        experiments[participant_id], split_ratio=split_ratio, split_seed=split_seed
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
    """Shuffle/split participant ids, pool all trials per side (matches `teh.py` across_participants)."""
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
    experiments = get_psych101_binary_experiments(
        dataset,
        n_participants=max_pid + 1,
        split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    train_trials: List[Dict[str, Any]] = []
    test_trials: List[Dict[str, Any]] = []
    for pid in train_participants:
        train_trials.extend(experiment_to_trial_dicts(experiments[pid]))
    for pid in test_participants:
        test_trials.extend(experiment_to_trial_dicts(experiments[pid]))
    return train_participants, test_participants, train_trials, test_trials


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


def expected_value_from_gamble(gamble: Dict[str, Any]) -> float:
    """Compute EV from gamble_A / gamble_B problem fields."""
    rewards = gamble.get("rewards", [])
    probs = gamble.get("probs", None)
    if rewards is None or len(rewards) == 0:
        return 0.0
    if probs is None:
        return float(rewards[0])
    return float(np.sum(np.array(probs, dtype=np.float64) * np.array(rewards, dtype=np.float64)))


def ev_diff_feature(problem: Dict[str, Any]) -> float:
    """EV(option B) - EV(option A) when gambles or CPC18-style fields exist; else 0 (intercept-only)."""
    if "gamble_A" in problem and "gamble_B" in problem:
        ev_a = expected_value_from_gamble(problem["gamble_A"])
        ev_b = expected_value_from_gamble(problem["gamble_B"])
        return ev_b - ev_a
    if "pHa" in problem:
        ev_a = float(problem["pHa"] * problem["Ha"] + (1.0 - problem["pHa"]) * problem["La"])
        ev_b = float(problem["pHb"] * problem["Hb"] + (1.0 - problem["pHb"]) * problem["Lb"])
        return ev_b - ev_a
    return 0.0


def fit_logistic_ev_diff(train_trials: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Fit beta/bias where x = ev_diff_feature(problem), y = action."""
    x = [ev_diff_feature(t["problem"]) for t in train_trials]
    y = [int(t["action"]) for t in train_trials]
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


def eval_mean_loglik_ev_diff(trials: List[Dict[str, Any]], beta: float, bias: float) -> float:
    """Mean Bernoulli log-likelihood under logistic EV-diff model."""
    if not trials:
        return float("nan")
    total = 0.0
    for t in trials:
        x = ev_diff_feature(t["problem"])
        pr = float(sigmoid(beta * x + bias))
        pr = min(max(pr, 1e-9), 1.0 - 1e-9)
        y = int(t["action"])
        total += y * np.log(pr) + (1.0 - y) * np.log(1.0 - pr)
    return float(total / len(trials))


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


def eval_mean_loglik_mixed_gambles(
    trials: List[Dict[str, Any]], omega: float, lam: float
) -> float:
    """Mean Bernoulli log-likelihood for mixed-gambles utility model."""
    if not trials:
        return float("nan")
    total = 0.0
    for t in trials:
        r = t["problem"]["gamble_A"]["rewards"]
        g = float(r[0])
        l = abs(float(r[1]))
        utility = g - omega * l
        p = float(sigmoid(lam * utility))
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        y_chose_gamble = 1.0 - float(int(t["action"]))
        total += y_chose_gamble * np.log(p) + (1.0 - y_chose_gamble) * np.log(1.0 - p)
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
    """Write participant_details_loglik.csv and summary.csv (TEH summary_loglik schema, 4 dp)."""
    base = Path(base_run_dir)
    details_fields = ["participant_id", "train_loglik", "val_loglik", "test_loglik"]
    details_path = base / "participant_details_loglik.csv"
    with details_path.open("w", newline="", encoding="utf-8") as f:
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
    summary_path = base / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(_round_floats_for_csv_row(summary_row))


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


def _fit_and_evaluate_participant(
    dataset: str,
    participant_id: int,
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fit MLE on train; report metrics on train/val/test."""
    if is_mixed_gambles_dataset(dataset):
        omega_hat, lam_hat = fit_logistic_mixed_gambles(train_trials)

        def predict_action(tr: Dict[str, Any]) -> int:
            r = tr["problem"]["gamble_A"]["rewards"]
            g = float(r[0])
            l = abs(float(r[1]))
            utility = g - omega_hat * l
            p = float(sigmoid(lam_hat * utility))
            return 0 if p >= 0.5 else 1

        fitted = {"omega": omega_hat, "lambda": lam_hat}
        loglik_fn = lambda trials: eval_mean_loglik_mixed_gambles(trials, omega_hat, lam_hat)
    else:
        beta_hat, bias_hat = fit_logistic_ev_diff(train_trials)

        def predict_action(tr: Dict[str, Any]) -> int:
            return predict_action_logistic(beta_hat, bias_hat, ev_diff_feature(tr["problem"]))

        fitted = {"beta": beta_hat, "bias": bias_hat}
        loglik_fn = lambda trials: eval_mean_loglik_ev_diff(trials, beta_hat, bias_hat)

    train_acc = eval_accuracy_from_predict_fn(train_trials, predict_action)
    val_acc = eval_accuracy_from_predict_fn(val_trials, predict_action)
    test_acc = eval_accuracy_from_predict_fn(test_trials, predict_action)
    out: Dict[str, Any] = {
        "method": "logistic_MLE",
        "dataset": dataset,
        "participant_id": participant_id,
        "fitted_params": fitted,
        "train_accuracy": train_acc["accuracy"],
        "val_accuracy": val_acc["accuracy"],
        "test_accuracy": test_acc["accuracy"],
        "n_train": train_acc["total"],
        "n_val": val_acc["total"],
        "n_test": test_acc["total"],
    }
    out["train_mean_loglik"] = loglik_fn(train_trials)
    out["val_mean_loglik"] = loglik_fn(val_trials)
    out["test_mean_loglik"] = loglik_fn(test_trials)
    return out


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
            f"\n[MLE baseline] {args.dataset} across_participants "
            f"train_mean_loglik={train_ll:.6f} test_mean_loglik={test_ll:.6f}"
        )
    else:
        print(
            f"\n[MLE baseline] {args.dataset} across_participants "
            f"train_acc={train_acc_eval['accuracy']:.4f} test_acc={test_acc_eval['accuracy']:.4f}"
        )
    print(f"Results saved under: {base_run_dir}")


def _print_within_summary(
    args: argparse.Namespace,
    participants_summary: List[Dict[str, Any]],
    participants_to_process: List[int],
    base_run_dir: str,
) -> None:
    print(f"\n[MLE baseline] dataset={args.dataset} participants={participants_to_process}")
    if not participants_summary:
        print("No participants processed.")
        print(f"Results saved under: {base_run_dir}")
        return
    if args.fitness_metric == "loglik":
        train_ll = float(np.mean([r["train_mean_loglik"] for r in participants_summary if r["train_mean_loglik"] is not None]))
        test_ll = float(np.mean([r["test_mean_loglik"] for r in participants_summary if r["test_mean_loglik"] is not None]))
        print(f"Mean train loglik: {train_ll:.6f}")
        print(f"Mean test loglik:  {test_ll:.6f}")
    else:
        train_acc = float(np.mean([r["train_acc"] for r in participants_summary if r["train_acc"] is not None]))
        test_acc = float(np.mean([r["test_acc"] for r in participants_summary if r["test_acc"] is not None]))
        print(f"Mean train accuracy: {train_acc:.4f}")
        print(f"Mean test accuracy:  {test_acc:.4f}")
    print(f"Results saved under: {base_run_dir}")


def _add_te_compat_args(parser: argparse.ArgumentParser) -> None:
    """TEH-only flags accepted for shell compatibility (ignored by MLE)."""
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
    _teh_dataset_choices = sorted(_PARTICIPANT_DATASETS)
    parser = argparse.ArgumentParser(
        description="Logistic MLE baseline (TEH-compatible dataset / participant / split CLI)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=_PETERSON_ALIAS,
        choices=_teh_dataset_choices,
        help=(
            "Psych-101 binary alias (peterson2021using, plonsky2018when, ...) or mixed_gambles (local CSV)."
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
            "across_participants: pool train across participants (peterson2021using only)."
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
        action="store_true",
        help="Disable wandb logging. Default is enabled.",
    )
    _add_te_compat_args(parser)
    args = parser.parse_args()
    mixed_gambles_gain_loss_only = bool(args.filter_mixed_gambles)
    psych_dataset_split = _effective_psych_dataset_split(args.dataset, args.psych_dataset_split)

    if not is_binary_loglik_dataset(args.dataset):
        print(f"Error: --dataset must be one of {sorted(_PARTICIPANT_DATASETS)}.")
        sys.exit(1)
    if not (0.0 < args.split_ratio < 1.0):
        print(f"Error: --split_ratio must be in (0,1), got {args.split_ratio}.")
        sys.exit(1)
    if args.split_mode == "across_participants" and args.dataset != _PETERSON_ALIAS:
        print("Error: --split_mode across_participants is only supported with --dataset peterson2021using.")
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
            else mle_output_base_dir(args.dataset, timestamp, psych_dataset_split=psych_dataset_split)
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
        beta_hat, bias_hat = fit_logistic_ev_diff(train_trials)

        def predict_action_ap(tr: Dict[str, Any]) -> int:
            return predict_action_logistic(beta_hat, bias_hat, ev_diff_feature(tr["problem"]))

        train_acc_eval = eval_accuracy_from_predict_fn(train_trials, predict_action_ap)
        test_acc_eval = eval_accuracy_from_predict_fn(test_trials, predict_action_ap)
        train_ll = eval_mean_loglik_ev_diff(train_trials, beta_hat, bias_hat)
        test_ll = eval_mean_loglik_ev_diff(test_trials, beta_hat, bias_hat)
        results_ap = {
            "method": "logistic_MLE",
            "dataset": args.dataset,
            "split_mode": "across_participants",
            "train_participants": train_p,
            "test_participants": test_p,
            "fitted_params": {"beta": beta_hat, "bias": bias_hat},
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
        f"MLE split settings: dataset={args.dataset}, split_mode={args.split_mode}, "
        f"split_ratio={args.split_ratio:.3f}, split_seed={args.split_seed}"
    )

    base_run_dir = args.output_dir
    if base_run_dir is None:
        base_run_dir = mle_output_base_dir(args.dataset, timestamp, psych_dataset_split=psych_dataset_split)
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

