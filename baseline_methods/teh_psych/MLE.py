#!/usr/bin/env python3
"""
Categorical logistic MLE baseline on teh_psych-parsed Psych-101 experiments.

Uses a frozen parse-plan cache (no LLM) and the same row sampling / prediction
filtering / train-val-test split as ``prototype/teh_psych.py``.

Model: see ``baseline_methods/teh_psych/features.py`` (binary logistic EV-style
diff for K=2; softmax over history-mean features for K>2).

Example::

  python baseline_methods/teh_psych/MLE.py \\
    --experiment_ids wilson2014humans/exp1.csv,frey2017cct/exp1.csv \\
    --reuse_parse_plan_cache \\
    --parse_plan_cache_dir generated_outputs_teh_psych/run_260722_004944/parse_plan_cache \\
    --require_cached_parse_plan \\
    --max_participants_per_experiment 10 \\
    --range_start_ordinal 0 --range_end_ordinal 9
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baseline_methods.teh_psych.common import (
    add_teh_psych_baseline_args,
    group_trials_by_participant,
    parse_experiment_ids,
    trial_weighted_mean_loglik,
    write_comparable_dataset_results_csv,
    write_experiment_loglik_csvs,
    write_json,
)
from baseline_methods.teh_psych.features import binary_feature_diff, option_feature_vector
from utils.teh_psych.categorical_eval import valid_action_ids_from_problem
from utils.teh_psych.dataset_loop import (
    discover_psych101_train_experiments,
    load_psych101_split,
    safe_experiment_id_for_path,
)
from utils.teh_psych.load_trials import (
    CachedParsePlanError,
    load_experiment_trials_from_parse_plan,
)

LOGLIK_EPS = 1e-9


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def nll_logistic(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Binary logistic NLL; params=[beta, bias], p(y=1)=sigmoid(beta*x+bias)."""
    beta, bias = params[0], params[1]
    p = sigmoid(beta * x + bias)
    p = np.clip(p, LOGLIK_EPS, 1.0 - LOGLIK_EPS)
    return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_logistic_binary(train_trials: List[Dict[str, Any]]) -> Tuple[float, float]:
    x = [binary_feature_diff(t["problem"], t.get("history")) for t in train_trials]
    y = [int(t.get("target_action", t["action"])) for t in train_trials]
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    res = minimize(
        lambda params: nll_logistic(params, x_arr, y_arr),
        x0=[1.0, 0.0],
        method="L-BFGS-B",
        bounds=[(-50.0, 50.0), (-50.0, 50.0)],
    )
    return float(res.x[0]), float(res.x[1])


def _softmax(u: np.ndarray) -> np.ndarray:
    z = u - np.max(u)
    e = np.exp(np.clip(z, -50, 50))
    s = e.sum()
    if s <= 0:
        return np.ones_like(u) / len(u)
    return e / s


def nll_softmax_beta(beta: float, feature_rows: List[np.ndarray], y: np.ndarray) -> float:
    total = 0.0
    for feats, yi in zip(feature_rows, y):
        p = _softmax(float(beta) * feats)
        yi_i = int(yi)
        if yi_i < 0 or yi_i >= len(p):
            total += -np.log(LOGLIK_EPS)
        else:
            total += -np.log(max(float(p[yi_i]), LOGLIK_EPS))
    return float(total)


def fit_softmax_beta(train_trials: List[Dict[str, Any]]) -> float:
    feature_rows = [
        np.asarray(option_feature_vector(t["problem"], t.get("history")), dtype=np.float64)
        for t in train_trials
    ]
    y = np.asarray(
        [int(t.get("target_action", t["action"])) for t in train_trials], dtype=np.int64
    )
    res = minimize(
        lambda th: nll_softmax_beta(float(th[0]), feature_rows, y),
        x0=[1.0],
        method="L-BFGS-B",
        bounds=[(-50.0, 50.0)],
    )
    return float(res.x[0])


def _trial_is_binary(trial: Dict[str, Any]) -> bool:
    return len(valid_action_ids_from_problem(trial.get("problem") or {})) == 2


def trials_are_binary(trials: List[Dict[str, Any]]) -> bool:
    return bool(trials) and all(_trial_is_binary(t) for t in trials)


def eval_mean_loglik_binary(trials: List[Dict[str, Any]], beta: float, bias: float) -> float:
    if not trials:
        return float("nan")
    total = 0.0
    for t in trials:
        x = binary_feature_diff(t["problem"], t.get("history"))
        pr = float(sigmoid(np.asarray(beta * x + bias)))
        pr = min(max(pr, LOGLIK_EPS), 1.0 - LOGLIK_EPS)
        y = int(t.get("target_action", t["action"]))
        total += y * np.log(pr) + (1.0 - y) * np.log(1.0 - pr)
    return float(total / len(trials))


def eval_mean_loglik_softmax(trials: List[Dict[str, Any]], beta: float) -> float:
    if not trials:
        return float("nan")
    total = 0.0
    for t in trials:
        feats = np.asarray(option_feature_vector(t["problem"], t.get("history")), dtype=np.float64)
        p = _softmax(beta * feats)
        y = int(t.get("target_action", t["action"]))
        total += np.log(max(float(p[y]) if 0 <= y < len(p) else LOGLIK_EPS, LOGLIK_EPS))
    return float(total / len(trials))


def eval_accuracy(trials: List[Dict[str, Any]], predict_fn: Callable[[Dict[str, Any]], int]) -> float:
    if not trials:
        return float("nan")
    correct = sum(
        1
        for t in trials
        if predict_fn(t) == int(t.get("target_action", t["action"]))
    )
    return float(correct / len(trials))


def fit_model(
    fit_trials: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Callable[[List[Dict[str, Any]]], float], Callable[[Dict[str, Any]], int]]:
    """Fit binary logistic or categorical softmax; return params, loglik_fn, predict_fn."""
    if not fit_trials:
        raise ValueError("No trials to fit")
    if trials_are_binary(fit_trials):
        beta_hat, bias_hat = fit_logistic_binary(fit_trials)
        fitted: Dict[str, Any] = {"beta": beta_hat, "bias": bias_hat, "model": "binary_logistic"}

        def loglik_fn(trials: List[Dict[str, Any]]) -> float:
            return eval_mean_loglik_binary(trials, beta_hat, bias_hat)

        def predict_fn(tr: Dict[str, Any]) -> int:
            x = binary_feature_diff(tr["problem"], tr.get("history"))
            p = float(sigmoid(np.asarray(beta_hat * x + bias_hat)))
            return 1 if p >= 0.5 else 0

    else:
        beta_hat = fit_softmax_beta(fit_trials)
        fitted = {"beta": beta_hat, "model": "categorical_softmax"}

        def loglik_fn(trials: List[Dict[str, Any]]) -> float:
            return eval_mean_loglik_softmax(trials, beta_hat)

        def predict_fn(tr: Dict[str, Any]) -> int:
            feats = np.asarray(
                option_feature_vector(tr["problem"], tr.get("history")), dtype=np.float64
            )
            return int(np.argmax(_softmax(beta_hat * feats)))

    return fitted, loglik_fn, predict_fn


def fit_and_evaluate_participant(
    experiment_id: str,
    participant_id: str,
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fit_trials = train_trials + val_trials
    fitted, loglik_fn, predict_fn = fit_model(fit_trials)
    return {
        "method": "teh_psych_logistic_MLE",
        "experiment_id": experiment_id,
        "participant_id": participant_id,
        "fitted_params": fitted,
        "train_accuracy": eval_accuracy(train_trials, predict_fn),
        "val_accuracy": eval_accuracy(val_trials, predict_fn),
        "test_accuracy": eval_accuracy(test_trials, predict_fn),
        "n_train": len(train_trials),
        "n_val": len(val_trials),
        "n_test": len(test_trials),
        "train_mean_loglik": loglik_fn(train_trials),
        "val_mean_loglik": loglik_fn(val_trials),
        "test_mean_loglik": loglik_fn(test_trials),
    }


def default_output_dir(timestamp: Optional[str] = None) -> Path:
    ts = timestamp or datetime.now().strftime("%y%m%d_%H%M%S")
    # Under generated_outputs_teh_psych (often a symlink to scratch/careAIDrive).
    return (
        _REPO_ROOT
        / "generated_outputs_teh_psych"
        / "baseline_methods"
        / "MLE"
        / f"run_{ts}"
    )


def run_one_experiment(
    experiment_id: str,
    *,
    cache_dir: Path,
    split_ds,
    args: argparse.Namespace,
    exp_out: Path,
) -> Dict[str, Any]:
    print(f"[MLE] {experiment_id}: loading+parsing rows…", flush=True)
    bundle = load_experiment_trials_from_parse_plan(
        experiment_id,
        cache_dir,
        require_cached=True,
        split_ds=split_ds,
        psych_dataset_split=args.psych_dataset_split,
        local_dataset=args.local_dataset,
        max_participants=args.max_participants_per_experiment,
        range_start_ordinal=args.range_start_ordinal,
        range_end_ordinal=args.range_end_ordinal,
        do_split=True,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        min_pooled_prediction_trials=args.min_pooled_prediction_trials,
        show_progress=bool(getattr(args, "show_progress", True)),
    )
    print(
        f"[MLE] {experiment_id}: parsed n_pred={len(bundle.prediction_trials)} "
        f"(train/val/test={len(bundle.train_trials)}/{len(bundle.val_trials)}/{len(bundle.test_trials)})",
        flush=True,
    )
    train_by = group_trials_by_participant(bundle.train_trials)
    val_by = group_trials_by_participant(bundle.val_trials)
    test_by = group_trials_by_participant(bundle.test_trials)
    participants = sorted(set(train_by) | set(val_by) | set(test_by))

    # Primary experiment metric: one pooled model (same split as teh_psych).
    # Per-participant fits often have empty test under pooled (row,block) splits when
    # participants have few blocks — those alone would yield test_ll=None.
    print(f"[MLE] {experiment_id}: fitting pooled model…", flush=True)
    global_fit_trials = bundle.train_trials + bundle.val_trials
    if not global_fit_trials:
        raise ValueError(f"No train+val trials for experiment {experiment_id}")
    global_params, global_loglik_fn, global_predict_fn = fit_model(global_fit_trials)
    global_metrics = {
        "fitted_params": global_params,
        "train_mean_loglik": global_loglik_fn(bundle.train_trials),
        "val_mean_loglik": global_loglik_fn(bundle.val_trials),
        "test_mean_loglik": global_loglik_fn(bundle.test_trials),
        "train_accuracy": eval_accuracy(bundle.train_trials, global_predict_fn),
        "val_accuracy": eval_accuracy(bundle.val_trials, global_predict_fn),
        "test_accuracy": eval_accuracy(bundle.test_trials, global_predict_fn),
    }
    write_json(exp_out / "pooled_model_results.json", global_metrics)

    details: List[Dict[str, Any]] = []
    participant_results: List[Dict[str, Any]] = []
    pid_iter = participants
    if bool(getattr(args, "show_progress", True)):
        pid_iter = tqdm(
            participants,
            desc=f"MLE fit participants ({experiment_id})",
            leave=False,
        )
    for pid in pid_iter:
        train_p = train_by.get(pid, [])
        val_p = val_by.get(pid, [])
        test_p = test_by.get(pid, [])
        if not (train_p + val_p):
            continue
        res = fit_and_evaluate_participant(
            experiment_id, pid, train_p, val_p, test_p
        )
        res["parse_plan_path"] = str(bundle.plan_path)
        res["parse_plan_sha256"] = bundle.plan_sha256
        participant_results.append(res)
        pid_dir = exp_out / f"participant_{pid}"
        pid_dir.mkdir(parents=True, exist_ok=True)
        write_json(pid_dir / "results.json", res)
        details.append(
            {
                "participant_id": pid,
                "train_loglik": res["train_mean_loglik"],
                "val_loglik": res["val_mean_loglik"],
                "test_loglik": res["test_mean_loglik"],
            }
        )

    write_experiment_loglik_csvs(exp_out, details)
    summary = {
        "method": "teh_psych_logistic_MLE",
        "experiment_id": experiment_id,
        "status": "success",
        "parse_plan_path": str(bundle.plan_path),
        "parse_plan_sha256": bundle.plan_sha256,
        "n_rows_used": len(bundle.rows),
        "n_prediction_trials": len(bundle.prediction_trials),
        "n_train": len(bundle.train_trials),
        "n_val": len(bundle.val_trials),
        "n_test": len(bundle.test_trials),
        "num_actions_min": bundle.action_summary.get("num_actions_min"),
        "num_actions_max": bundle.action_summary.get("num_actions_max"),
        "is_variable_k": bundle.action_summary.get("is_variable_k"),
        "num_participants_fit": len(participant_results),
        "n_participants_with_test": sum(
            1 for r in participant_results if int(r.get("n_test") or 0) > 0
        ),
        # Primary = pooled model on same train/val/test as teh_psych.
        "avg_train_loglik": global_metrics["train_mean_loglik"],
        "avg_val_loglik": global_metrics["val_mean_loglik"],
        "avg_test_loglik": global_metrics["test_mean_loglik"],
        "pooled_fitted_params": global_params,
        "per_participant_trial_weighted_test_loglik": trial_weighted_mean_loglik(
            participant_results, split="test"
        ),
        "model_doc": (
            "Primary avg_*_loglik: one pooled MLE on train+val, eval on each split "
            "(aligned with teh_psych pooled split). "
            "K=2 binary logistic on f1-f0; K>2 softmax(beta*f_k); "
            "f_k=history mean feedback (+ label-aligned stim). "
            "Per-participant fits kept under participant_*/ for diagnostics."
        ),
    }
    write_json(exp_out / "experiment_summary.json", summary)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_teh_psych_baseline_args(parser)
    args = parser.parse_args(argv)

    if not args.reuse_parse_plan_cache:
        print("error: --reuse_parse_plan_cache is required for teh_psych baselines", file=sys.stderr)
        return 2
    if not args.require_cached_parse_plan:
        print(
            "error: --require_cached_parse_plan is required (no LLM fallback)",
            file=sys.stderr,
        )
        return 2
    if not args.parse_plan_cache_dir:
        print("error: --parse_plan_cache_dir is required", file=sys.stderr)
        return 2

    cache_dir = Path(args.parse_plan_cache_dir).expanduser()
    if not cache_dir.is_dir():
        print(f"error: parse_plan_cache_dir not found: {cache_dir}", file=sys.stderr)
        return 2

    out_root = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "command.json", {"argv": list(sys.argv if argv is None else argv)})

    split_ds = load_psych101_split(args.psych_dataset_split, local_dataset=args.local_dataset)
    experiment_ids = parse_experiment_ids(args.experiment_ids)
    max_exp = None if str(args.max_experiments).lower() == "all" else int(args.max_experiments)
    ordered = discover_psych101_train_experiments(
        split_ds, experiment_ids=experiment_ids, max_experiments=max_exp
    )
    if not ordered:
        print("error: no experiments selected", file=sys.stderr)
        return 2

    results: List[Dict[str, Any]] = []
    exp_iter = ordered
    if bool(getattr(args, "show_progress", True)):
        exp_iter = tqdm(ordered, desc="MLE experiments", unit="exp")
    for eid in exp_iter:
        exp_out = out_root / "experiments" / safe_experiment_id_for_path(eid)
        exp_out.mkdir(parents=True, exist_ok=True)
        try:
            summary = run_one_experiment(
                eid, cache_dir=cache_dir, split_ds=split_ds, args=args, exp_out=exp_out
            )
            results.append(summary)
            print(
                f"[MLE] {eid}: test_ll={summary.get('avg_test_loglik')} "
                f"participants={summary.get('num_participants_fit')}",
                flush=True,
            )
        except CachedParsePlanError as exc:
            err = {
                "method": "teh_psych_logistic_MLE",
                "experiment_id": eid,
                "status": "cached_plan_failed",
                "error": str(exc),
            }
            write_json(exp_out / "experiment_summary.json", err)
            results.append(err)
            print(f"[MLE] {eid}: CACHED PLAN FAILURE: {exc}", file=sys.stderr, flush=True)
        except Exception as exc:
            err = {
                "method": "teh_psych_logistic_MLE",
                "experiment_id": eid,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(exp_out / "experiment_summary.json", err)
            results.append(err)
            print(f"[MLE] {eid}: FAILED: {exc}", file=sys.stderr, flush=True)

        # Refresh joinable CSVs after each experiment (partial runs stay usable).
        write_json(out_root / "dataset_results.json", results)
        write_comparable_dataset_results_csv(
            out_root, results, method="teh_psych_logistic_MLE"
        )

    n_ok = sum(1 for r in results if r.get("status") == "success")
    print(
        f"Done. {n_ok}/{len(results)} experiments succeeded. Output: {out_root}\n"
        f"  Compare LLs with teh_psych via: {out_root / 'dataset_results.csv'} "
        f"or {out_root / 'loglik_comparison.csv'}",
        flush=True,
    )
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
