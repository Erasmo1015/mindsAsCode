#!/usr/bin/env python3
"""
Prospect Theory MLE baseline on teh_psych-parsed Psych-101 experiments.

Uses a frozen parse-plan cache (no LLM) and the same sampling / prediction /
split path as ``prototype/teh_psych.py``.

Optimization matches ``baseline_methods/prospect_theory.py`` (alpha, lambda,
gamma, beta; L-BFGS-B; Bernoulli P(choose A)=sigmoid(beta*(VA-VB))) **when**
each trial provides explicit binary gambles in stimulus.

Experiments without explicit ``(rewards, probs)`` structure are marked
``unsupported_prospect_theory`` — outcomes/probabilities are never invented.

Example::

  python baseline_methods/teh_psych/prospect_theory.py \\
    --experiment_ids peterson2021using/exp1.csv \\
    --reuse_parse_plan_cache \\
    --parse_plan_cache_dir generated_outputs_teh_psych/run_260722_004944/parse_plan_cache \\
    --require_cached_parse_plan
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baseline_methods.prospect_theory import (  # reuse kernels + optimizer
    eval_accuracy_from_predict_fn,
    fit_prospect_theory_gamble_choice,
    sigmoid,
    subjective_value_gamble,
)
from baseline_methods.teh_psych.common import (
    add_teh_psych_baseline_args,
    group_trials_by_participant,
    parse_experiment_ids,
    trial_weighted_mean_loglik,
    write_comparable_dataset_results_csv,
    write_experiment_loglik_csvs,
    write_json,
)
from baseline_methods.teh_psych.features import (
    experiment_prospect_support,
    extract_explicit_binary_gambles,
)
from utils.teh_psych.dataset_loop import (
    discover_psych101_train_experiments,
    load_psych101_split,
    safe_experiment_id_for_path,
)
from utils.teh_psych.load_trials import (
    CachedParsePlanError,
    load_experiment_trials_from_parse_plan,
)


def _gamble_getters():
    def ga(p: Dict[str, Any], _h=None):
        gg = extract_explicit_binary_gambles(p)
        if gg is None:
            raise ValueError("missing explicit gambles")
        return gg[0]

    def gb(p: Dict[str, Any], _h=None):
        gg = extract_explicit_binary_gambles(p)
        if gg is None:
            raise ValueError("missing explicit gambles")
        return gg[1]

    return ga, gb


def _make_predict_action(params: Dict[str, float], ga, gb):
    def predict_action(tr: Dict[str, Any]) -> int:
        p = tr["problem"]
        hist = tr.get("history")
        rA, prA = ga(p, hist)
        rB, prB = gb(p, hist)
        va = subjective_value_gamble(rA, prA, params["alpha"], params["lambda"], params["gamma"])
        vb = subjective_value_gamble(rB, prB, params["alpha"], params["lambda"], params["gamma"])
        p_a = float(sigmoid(params["beta"] * (va - vb)))
        return 0 if p_a >= 0.5 else 1

    return predict_action


def eval_mean_loglik_prospect(
    trials: List[Dict[str, Any]],
    params: Dict[str, float],
    ga,
    gb,
) -> float:
    """Mean Bernoulli loglik with P(choose A)=sigmoid(beta*(VA-VB))."""
    if not trials:
        return float("nan")
    alpha = float(params["alpha"])
    lam = float(params["lambda"])
    gamma = float(params["gamma"])
    beta = float(params["beta"])
    total = 0.0
    for tr in trials:
        p = tr["problem"]
        hist = tr.get("history")
        r_a, pr_a = ga(p, hist)
        r_b, pr_b = gb(p, hist)
        va = subjective_value_gamble(r_a, pr_a, alpha, lam, gamma)
        vb = subjective_value_gamble(r_b, pr_b, alpha, lam, gamma)
        p_choose_a = float(sigmoid(beta * (va - vb)))
        p_choose_a = min(max(p_choose_a, 1e-9), 1.0 - 1e-9)
        y = int(tr.get("target_action", tr["action"]))
        if y == 0:
            total += np.log(p_choose_a)
        else:
            total += np.log(1.0 - p_choose_a)
    return float(total / len(trials))


def fit_and_evaluate_participant(
    experiment_id: str,
    participant_id: str,
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    test_trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fit_trials = train_trials + val_trials
    if not fit_trials:
        raise ValueError(f"No train+val trials for participant {participant_id}")
    ga, gb = _gamble_getters()
    params = fit_prospect_theory_gamble_choice(
        fit_trials,
        action_is_chooseA=lambda a: a == 0,
        gambleA_getter=ga,
        gambleB_getter=gb,
        dataset=experiment_id,
        participant_id=None,
    )
    predict_action = _make_predict_action(params, ga, gb)
    train_acc = eval_accuracy_from_predict_fn(train_trials, predict_action)
    val_acc = eval_accuracy_from_predict_fn(val_trials, predict_action)
    test_acc = eval_accuracy_from_predict_fn(test_trials, predict_action)
    return {
        "method": "teh_psych_prospect_theory_MLE",
        "experiment_id": experiment_id,
        "participant_id": participant_id,
        "fitted_params": params,
        "train_accuracy": train_acc["accuracy"],
        "val_accuracy": val_acc["accuracy"],
        "test_accuracy": test_acc["accuracy"],
        "n_train": len(train_trials),
        "n_val": len(val_trials),
        "n_test": len(test_trials),
        "train_mean_loglik": eval_mean_loglik_prospect(train_trials, params, ga, gb),
        "val_mean_loglik": eval_mean_loglik_prospect(val_trials, params, ga, gb),
        "test_mean_loglik": eval_mean_loglik_prospect(test_trials, params, ga, gb),
    }


def default_output_dir(timestamp: Optional[str] = None) -> Path:
    ts = timestamp or datetime.now().strftime("%y%m%d_%H%M%S")
    return (
        _REPO_ROOT
        / "generated_outputs_teh_psych"
        / "baseline_methods"
        / "prospect_theory"
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
    print(f"[PT] {experiment_id}: loading+parsing rows…", flush=True)
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
        f"[PT] {experiment_id}: parsed n_pred={len(bundle.prediction_trials)} "
        f"(train/val/test={len(bundle.train_trials)}/{len(bundle.val_trials)}/{len(bundle.test_trials)})",
        flush=True,
    )
    supported, reason, frac = experiment_prospect_support(bundle.prediction_trials)
    meta = {
        "method": "teh_psych_prospect_theory_MLE",
        "experiment_id": experiment_id,
        "parse_plan_path": str(bundle.plan_path),
        "parse_plan_sha256": bundle.plan_sha256,
        "n_rows_used": len(bundle.rows),
        "n_prediction_trials": len(bundle.prediction_trials),
        "n_train": len(bundle.train_trials),
        "n_val": len(bundle.val_trials),
        "n_test": len(bundle.test_trials),
        "num_actions_min": bundle.action_summary.get("num_actions_min"),
        "num_actions_max": bundle.action_summary.get("num_actions_max"),
        "prospect_support_fraction": frac,
        "prospect_support_reason": reason,
    }
    if not supported:
        out = {
            **meta,
            "status": "unsupported_prospect_theory",
            "num_participants_fit": 0,
        }
        write_json(exp_out / "experiment_summary.json", out)
        return out

    train_by = group_trials_by_participant(bundle.train_trials)
    val_by = group_trials_by_participant(bundle.val_trials)
    test_by = group_trials_by_participant(bundle.test_trials)
    participants = sorted(set(train_by) | set(val_by) | set(test_by))

    details: List[Dict[str, Any]] = []
    participant_results: List[Dict[str, Any]] = []
    pid_iter = participants
    if bool(getattr(args, "show_progress", True)):
        pid_iter = tqdm(
            participants,
            desc=f"PT fit participants ({experiment_id})",
            leave=False,
        )
    for pid in pid_iter:
        train_p = train_by.get(pid, [])
        val_p = val_by.get(pid, [])
        test_p = test_by.get(pid, [])
        if not (train_p + val_p):
            continue
        res = fit_and_evaluate_participant(experiment_id, pid, train_p, val_p, test_p)
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
        **meta,
        "status": "success",
        "num_participants_fit": len(participant_results),
        "n_participants_with_test": sum(
            1 for r in participant_results if int(r.get("n_test") or 0) > 0
        ),
        "avg_train_loglik": trial_weighted_mean_loglik(participant_results, split="train"),
        "avg_val_loglik": trial_weighted_mean_loglik(participant_results, split="val"),
        "avg_test_loglik": trial_weighted_mean_loglik(participant_results, split="test"),
    }
    write_json(exp_out / "experiment_summary.json", summary)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_teh_psych_baseline_args(parser)
    args = parser.parse_args(argv)

    if not args.reuse_parse_plan_cache:
        print("error: --reuse_parse_plan_cache is required", file=sys.stderr)
        return 2
    if not args.require_cached_parse_plan:
        print("error: --require_cached_parse_plan is required", file=sys.stderr)
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
    n_supported = 0
    n_unsupported = 0
    exp_iter = ordered
    if bool(getattr(args, "show_progress", True)):
        exp_iter = tqdm(ordered, desc="PT experiments", unit="exp")
    for eid in exp_iter:
        exp_out = out_root / "experiments" / safe_experiment_id_for_path(eid)
        exp_out.mkdir(parents=True, exist_ok=True)
        try:
            summary = run_one_experiment(
                eid, cache_dir=cache_dir, split_ds=split_ds, args=args, exp_out=exp_out
            )
            results.append(summary)
            if summary.get("status") == "success":
                n_supported += 1
                print(
                    f"[PT] {eid}: supported test_ll={summary.get('avg_test_loglik')}",
                    flush=True,
                )
            elif summary.get("status") == "unsupported_prospect_theory":
                n_unsupported += 1
                print(
                    f"[PT] {eid}: unsupported — {summary.get('prospect_support_reason')}",
                    flush=True,
                )
            else:
                print(f"[PT] {eid}: {summary.get('status')}", flush=True)
        except CachedParsePlanError as exc:
            err = {
                "method": "teh_psych_prospect_theory_MLE",
                "experiment_id": eid,
                "status": "cached_plan_failed",
                "error": str(exc),
            }
            write_json(exp_out / "experiment_summary.json", err)
            results.append(err)
            print(f"[PT] {eid}: CACHED PLAN FAILURE: {exc}", file=sys.stderr, flush=True)
        except Exception as exc:
            err = {
                "method": "teh_psych_prospect_theory_MLE",
                "experiment_id": eid,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(exp_out / "experiment_summary.json", err)
            results.append(err)
            print(f"[PT] {eid}: FAILED: {exc}", file=sys.stderr, flush=True)

        write_json(
            out_root / "dataset_results.json",
            {
                "results": results,
                "n_supported": n_supported,
                "n_unsupported": n_unsupported,
                "n_cached_plan_failed": sum(
                    1 for r in results if r.get("status") == "cached_plan_failed"
                ),
            },
        )
        write_comparable_dataset_results_csv(
            out_root, results, method="teh_psych_prospect_theory_MLE"
        )

    print(
        f"Done. PT supported={n_supported} unsupported={n_unsupported} "
        f"of {len(results)}. Output: {out_root}\n"
        f"  Compare LLs with teh_psych via: {out_root / 'dataset_results.csv'} "
        f"or {out_root / 'loglik_comparison.csv'}",
        flush=True,
    )
    # Unsupported is an expected outcome, not a hard failure.
    hard_fail = any(
        r.get("status") in ("failed", "cached_plan_failed") for r in results
    )
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
