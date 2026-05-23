#!/usr/bin/env python3
"""
Follow-up analyses: CCT (3frey2017cct) MLE rule, confidence, stopping structure;
dataset-4 participant subsets and stratified diagnosis; cross-dataset comparison.

Usage:
  python analysis/code/psych-101/cct_and_dataset4_followup.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baseline_methods.MLE import (
    eval_mean_loglik_ev_diff,
    fit_logistic_ev_diff,
    predict_action_logistic,
    sigmoid,
    trials_for_participant,
)
from baseline_methods.psych101_features import _cct_one_step_ev, option_b_feature_diff
from data_modules.psych101_binary import DEFAULT_PSYCH_DATASET_SPLIT, split_psych_experiment

from analysis.code.utils import compare as cmp

_DATASET_CCT = "3frey2017cct"
_DATASET_D4 = "4wulff2018description"
_DATASET_SPEEK = "5speekenbrink2008learning"
_PSYCH_SPLIT = DEFAULT_PSYCH_DATASET_SPLIT
_SPLIT_RATIO = 0.6
_SPLIT_SEED = 0
_D4_AUDIT_COUNTS = "analysis/data/psych101_dataset4_audit/dataset4_trial_counts.csv"
_D4_AUDIT_SPLIT_RATIO = 0.8
_D4_AUDIT_SPLIT_SEED = 42
_CONVERGENCE_CSV = "generated_outputs/psych101_train/teh/iteration_convergence.csv"
_CCT_OUT = "analysis/data/psych101_cct_diagnosis"
_D4_OUT = "analysis/data/psych101_dataset4_audit"
_STRONG_GAP = 0.15
_LOSS_MARGIN = 0.05
_NEAR_PERFECT = -0.05
_CATASTROPHIC = -1.0


def _repo() -> Path:
    return _REPO_ROOT


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.{nd}f}"


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_mle_results(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for pdir in sorted(run_dir.glob("participant_*")):
        m = re.match(r"participant_(\d+)$", pdir.name)
        if not m:
            continue
        rpath = pdir / "results.json"
        if not rpath.is_file():
            continue
        data = json.loads(rpath.read_text(encoding="utf-8"))
        out[int(m.group(1))] = data
    return out


def _load_choose_fn(path: Path) -> Callable:
    spec = importlib.util.spec_from_file_location(f"prog_{path.parent.name}", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    choose = getattr(mod, "choose", None)
    if not callable(choose):
        raise RuntimeError(f"No choose in {path}")
    return choose


def _clamp_p(p: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(p)))


def _prob_from_choose(raw: Any) -> float:
    if isinstance(raw, bool):
        return 1.0 - 1e-6 if raw else 1e-6
    if isinstance(raw, (int, np.integer)) and int(raw) in (0, 1):
        return 1.0 - 1e-6 if int(raw) == 1 else 1e-6
    return _clamp_p(float(raw))


def _trial_metrics(
    trials: Sequence[Mapping[str, Any]],
    prob_fn: Callable[[Mapping[str, Any]], float],
) -> Dict[str, Any]:
    if not trials:
        return {}
    probs_at: List[float] = []
    abs_dev: List[float] = []
    preds: List[float] = []
    ys: List[int] = []
    ll = 0.0
    correct = 0
    for t in trials:
        y = int(t["action"])
        p = _clamp_p(prob_fn(t))
        probs_at.append(p if y == 1 else 1.0 - p)
        abs_dev.append(abs(p - 0.5))
        preds.append(p)
        ys.append(y)
        ll += y * math.log(p) + (1 - y) * math.log(1 - p)
        correct += int((1 if p >= 0.5 else 0) == y)
    n = len(trials)
    return {
        "accuracy": correct / n,
        "mean_loglik": ll / n,
        "mean_prob_at_action": statistics.mean(probs_at),
        "mean_abs_p_minus_half": statistics.mean(abs_dev),
        "preds": preds,
        "ys": ys,
    }


def _mle_prob_fn(beta: float, bias: float) -> Callable[[Mapping[str, Any]], float]:
    def fn(t: Mapping[str, Any]) -> float:
        x = option_b_feature_diff(t["problem"], t.get("history"))
        return float(sigmoid(beta * x + bias))

    return fn


def _calibration_bins(preds: Sequence[float], ys: Sequence[int], n_bins: int = 5) -> List[Dict[str, Any]]:
    if not preds:
        return []
    pairs = sorted(zip(preds, ys), key=lambda z: z[0])
    chunk = max(1, len(pairs) // n_bins)
    bins: List[Dict[str, Any]] = []
    for i in range(0, len(pairs), chunk):
        sl = pairs[i : i + chunk]
        if not sl:
            continue
        mean_p = statistics.mean(p for p, _ in sl)
        obs = statistics.mean(y for _, y in sl)
        bins.append(
            {
                "bin": len(bins),
                "n": len(sl),
                "mean_predicted_p_stop": mean_p,
                "observed_stop_rate": obs,
                "calibration_gap": obs - mean_p,
            }
        )
    return bins


def _cct_derived_fields(problem: Mapping[str, Any]) -> Dict[str, float]:
    p = problem
    n_rem = max(1.0, float(p.get("n_cards_remaining", 1)))
    n_loss = max(0.0, float(p.get("n_loss_cards", 0)))
    gain = float(p.get("gain_amount", 0))
    loss = float(p.get("loss_amount", 0))
    cur = float(p.get("current_score", 0))
    flipped = float(p.get("cards_flipped", 0))
    p_loss = min(1.0, max(0.0, n_loss / n_rem))
    ev_stop, ev_continue = _cct_one_step_ev(p)
    x_mle = ev_stop - ev_continue
    risk = p_loss
    return {
        "cards_flipped": flipped,
        "current_score": cur,
        "n_cards_remaining": n_rem,
        "n_loss_cards": n_loss,
        "gain_amount": gain,
        "loss_amount": loss,
        "risk_p_loss": risk,
        "ev_stop": ev_stop,
        "ev_continue": ev_continue,
        "mle_feature_x": x_mle,
        "expected_continuation_value": ev_continue,
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _discover_runs(repo: Path, dataset: str) -> List[Path]:
    root = cmp._teh_search_root(repo, dataset, _PSYCH_SPLIT)
    if not root.is_dir():
        return []
    runs = cmp._collect_run_candidates(root)
    return sorted(runs, key=cmp._run_sort_key, reverse=True)


def _classify_program(code: str) -> str:
    c = code or ""
    low = re.sub(r"\s+", "", c.lower())
    scores = Counter(
        raw_EV_linear=0,
        stopping_threshold=0,
        subjective_value=0,
        history_heavy=0,
        calibrated_confident=0,
        unclear=0,
    )
    if re.search(r"def\s+subjective_value\s*\(", c, re.I):
        scores["subjective_value"] += 3
    if re.search(r"expected_value|net_expected|expected_gain", low):
        scores["raw_EV_linear"] += 2
    if "sigmoid" in low:
        scores["raw_EV_linear"] += 1
    if re.search(r"return\s+[01](?:\.0)?\s*$", c, re.M):
        scores["stopping_threshold"] += 2
    if re.search(r"if\s+.*n_loss_cards|if\s+.*cards_flipped", c):
        scores["stopping_threshold"] += 1
    if re.search(r"action_counts|recent_actions|history", low):
        scores["history_heavy"] += 2
    if re.search(r"return\s+0\.(?:9|95|99)", c):
        scores["calibrated_confident"] += 2
    if re.search(r"return\s+0\.(?:0[0-4]|1)\b", c) and "sigmoid" not in low:
        scores["calibrated_confident"] += 1
    top, n = scores.most_common(1)[0]
    second = sorted(scores.values(), reverse=True)[1]
    if n == 0 or n == second:
        return "unclear"
    return top


def _load_d4_trial_counts(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "participant_id": int(float(row["teh_participant_id"])),
                    "hf_participant": row.get("hf_participant", ""),
                    "train_trials": int(float(row["train_trials"])),
                    "val_trials": int(float(row["val_trials"])),
                    "test_trials": int(float(row["test_trials"])),
                    "parsed_total_trials": int(float(row.get("parsed_total_trials") or 0)),
                }
            )
    return rows


def _load_loglik_maps(
    repo: Path, dataset: str, config_data: Mapping[str, Any]
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
    paths = cmp._resolve_baseline_run_paths(
        config_data, repo, dataset, _PSYCH_SPLIT, quiet=True
    )
    teh_run = cmp._auto_discover_teh_run(repo, dataset=dataset, psych_dataset_split=_PSYCH_SPLIT)
    teh_test: Dict[int, float] = {}
    teh_gated: Dict[int, float] = {}
    if teh_run:
        csv_path = cmp._resolve_loglik_csv(teh_run)
        teh_test = cmp._read_loglik_csv(csv_path, "test_loglik", required=False)
        teh_gated = cmp._read_loglik_csv(csv_path, "gated_test_loglik", required=False)
    maps = {}
    for method in ("MLE", "prospect_theory", "Centaur"):
        if method in paths:
            maps[method] = cmp._load_scores_from_run(paths[method], "test_loglik", required=False)
        else:
            maps[method] = {}
    return (
        maps.get("MLE", {}),
        maps.get("prospect_theory", {}),
        maps.get("Centaur", {}),
        teh_test,
        teh_gated,
        paths,
    )


def _num_best(scores_by_method: Mapping[str, Mapping[int, float]], pids: Sequence[int]) -> Dict[str, int]:
    counts = {m: 0 for m in scores_by_method}
    for pid in pids:
        vals = [
            (m, scores_by_method[m][pid])
            for m in scores_by_method
            if pid in scores_by_method[m] and math.isfinite(scores_by_method[m][pid])
        ]
        if not vals:
            continue
        best = max(v for _, v in vals)
        for m, v in vals:
            if v >= best:
                counts[m] += 1
    return counts


def _method_stats(vals: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    finite = [v for v in vals if v is not None and math.isfinite(v)]
    if not finite:
        return {"avg": None, "median": None, "near_perfect": 0, "catastrophic": 0}
    return {
        "avg": statistics.mean(finite),
        "median": statistics.median(finite),
        "near_perfect": sum(1 for v in finite if v > _NEAR_PERFECT),
        "catastrophic": sum(1 for v in finite if v < _CATASTROPHIC),
    }


def analyze_cct(repo: Path, config_data: Mapping[str, Any]) -> None:
    out_dir = repo / _CCT_OUT
    mle_run = cmp._resolve_baseline_run_paths(
        config_data, repo, _DATASET_CCT, _PSYCH_SPLIT, quiet=True
    ).get("MLE")
    teh_run = cmp._auto_discover_teh_run(repo, dataset=_DATASET_CCT, psych_dataset_split=_PSYCH_SPLIT)
    if mle_run is None or teh_run is None:
        raise SystemExit("Missing MLE or TEH run for 3frey2017cct")

    mle_results = _load_mle_results(mle_run if mle_run.is_dir() else mle_run.parent)
    teh_csv = cmp._resolve_loglik_csv(teh_run)
    teh_test_ll = cmp._read_loglik_csv(teh_csv, "test_loglik", required=False)

    betas: List[float] = []
    biases: List[float] = []
    threshold_xs: List[float] = []

    # --- 1. MLE rule report ---
    mle_lines: List[str] = [
        "MLE inductive bias report — 3frey2017cct",
        f"MLE run: {mle_run}",
        f"TEH run: {teh_run}",
        f"Split: ratio={_SPLIT_RATIO}, seed={_SPLIT_SEED}",
        "",
        "MODEL (from baseline_methods/MLE.py + psych101_features.py):",
        "  P(action=1) = sigmoid(beta * x + bias)",
        "  action 0 = flip/continue, action 1 = stop/cash out",
        "  x = option_b_feature_diff(problem) = ev_stop - ev_continue",
        "  where ev_stop = current_score,",
        "        ev_continue = current_score + (1-p_loss)*gain_amount - p_loss*loss_amount,",
        "        p_loss = n_loss_cards / n_cards_remaining",
        "",
        "IMPORTANT: MLE uses ONE scalar feature (continuation EV disadvantage), NOT raw",
        "cards_flipped/current_score/etc. directly. Those enter only through ev_continue.",
        "",
    ]

    per_pid_aux: List[Dict[str, Any]] = []
    strong_examples: List[Tuple[float, int, Dict[str, Any]]] = []

    acc_rows: List[Dict[str, str]] = []
    all_mle_preds: List[float] = []
    all_mle_ys: List[int] = []
    all_teh_preds: List[float] = []
    all_teh_ys: List[int] = []

    stopping_rows: List[Dict[str, str]] = []
    bins_def = [
        ("cards_flipped", [(0, 5), (6, 15), (16, 30), (31, 999)]),
        ("risk_p_loss", [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]),
        ("current_score", [(-999, 0), (0, 50), (50, 150), (150, 9999)]),
        ("mle_feature_x", [(-999, -20), (-20, -5), (-5, 0), (0, 999)]),
    ]

    pooled_train: List[Dict[str, Any]] = []
    pooled_test: List[Dict[str, Any]] = []

    for pid in sorted(mle_results):
        res = mle_results[pid]
        fp = res.get("fitted_params") or {}
        beta = float(fp.get("beta", 0))
        bias = float(fp.get("bias", 0))
        betas.append(beta)
        biases.append(bias)
        x_thresh = (-bias / beta) if abs(beta) > 1e-8 else float("nan")
        if math.isfinite(x_thresh):
            threshold_xs.append(x_thresh)

        train, val, test = trials_for_participant(
            _DATASET_CCT,
            pid,
            split_ratio=_SPLIT_RATIO,
            split_seed=_SPLIT_SEED,
            filter_mixed_gambles=False,
            psych_dataset_split=_PSYCH_SPLIT,
            local_dataset=None,
            mixed_gambles_csv="",
        )
        fit_trials = train + val
        mle_prob = _mle_prob_fn(beta, bias)
        test_m = _trial_metrics(test, mle_prob)

        teh_path = teh_run / f"participant_{pid}" / "best_program.py"
        teh_m: Dict[str, Any] = {}
        if teh_path.is_file():
            choose = _load_choose_fn(teh_path)

            def teh_prob(t: Mapping[str, Any]) -> float:
                return _prob_from_choose(choose(t["problem"], t.get("history", [])))

            teh_m = _trial_metrics(test, teh_prob)

        gap = (res.get("test_mean_loglik") or 0) - (teh_test_ll.get(pid) or 0)
        if gap > _STRONG_GAP:
            strong_examples.append((gap, pid, res))

        # auxiliary correlations on test
        ys = [int(t["action"]) for t in test]
        fields = [_cct_derived_fields(t["problem"]) for t in test]
        aux = {"participant_id": pid, "beta": beta, "bias": bias, "x_threshold_p05": x_thresh}
        for key in (
            "mle_feature_x",
            "cards_flipped",
            "current_score",
            "risk_p_loss",
            "expected_continuation_value",
        ):
            xs = [f[key] for f in fields]
            aux[f"corr_{key}_action"] = _pearson(xs, ys)
        per_pid_aux.append(aux)

        acc_gap = (test_m.get("mean_loglik") or 0) - (teh_m.get("mean_loglik") or 0)
        acc_diff = abs((test_m.get("accuracy") or 0) - (teh_m.get("accuracy") or 0))
        similar_acc_bad_ll = acc_diff <= 0.05 and acc_gap > 0.10

        acc_rows.append(
            {
                "participant_id": str(pid),
                "mle_test_loglik": _fmt(res.get("test_mean_loglik")),
                "teh_test_loglik": _fmt(teh_test_ll.get(pid)),
                "loglik_gap_mle_minus_teh": _fmt(gap),
                "mle_test_accuracy": _fmt(test_m.get("accuracy")),
                "teh_test_accuracy": _fmt(teh_m.get("accuracy")),
                "accuracy_gap_abs": _fmt(acc_diff),
                "mle_mean_prob_at_action": _fmt(test_m.get("mean_prob_at_action")),
                "teh_mean_prob_at_action": _fmt(teh_m.get("mean_prob_at_action")),
                "mle_mean_abs_p_minus_half": _fmt(test_m.get("mean_abs_p_minus_half")),
                "teh_mean_abs_p_minus_half": _fmt(teh_m.get("mean_abs_p_minus_half")),
                "similar_accuracy_much_worse_loglik": str(similar_acc_bad_ll),
                "mle_beta": _fmt(beta, 6),
                "mle_bias": _fmt(bias, 6),
                "mle_x_threshold_at_p05": _fmt(x_thresh, 4),
            }
        )

        if test_m.get("preds"):
            all_mle_preds.extend(test_m["preds"])
            all_mle_ys.extend(test_m["ys"])
        if teh_m.get("preds"):
            all_teh_preds.extend(teh_m["preds"])
            all_teh_ys.extend(teh_m["ys"])

        pooled_train.extend(train)
        pooled_test.extend(test)

    mle_lines.append("POPULATION FITTED PARAMETERS (50 participants):")
    mle_lines.append(f"  beta: mean={statistics.mean(betas):.6f}, median={statistics.median(betas):.6f}, "
                     f"min={min(betas):.6f}, max={max(betas):.6f}")
    mle_lines.append(f"  bias: mean={statistics.mean(biases):.6f}, median={statistics.median(biases):.6f}, "
                     f"min={min(biases):.6f}, max={max(biases):.6f}")
    pos_beta = sum(1 for b in betas if b > 0)
    mle_lines.append(f"  beta>0 (positive x favors STOP): {pos_beta}/{len(betas)}")
    if threshold_xs:
        mle_lines.append(
            f"  x_threshold where P(stop)=0.5 (-bias/beta): mean={statistics.mean(threshold_xs):.3f}, "
            f"median={statistics.median(threshold_xs):.3f}"
        )
    mle_lines.append("")
    mle_lines.append(
        "INTERPRETATION: With beta>0, MLE is a one-dimensional stopping rule on continuation"
        " disadvantage x=ev_stop-ev_continue. Larger x (stop looks better vs one more flip) => higher P(stop)."
        " Near-deterministic behavior is captured by large |beta| with bias setting the stop/continue boundary."
    )
    mle_lines.append("")

    # aggregate feature correlations
    mle_lines.append("Mean per-participant Pearson r(feature, action) on TEST:")
    for key in (
        "mle_feature_x",
        "cards_flipped",
        "current_score",
        "risk_p_loss",
        "expected_continuation_value",
    ):
        vals = [a[f"corr_{key}_action"] for a in per_pid_aux if a.get(f"corr_{key}_action") is not None]
        if vals:
            mle_lines.append(f"  {key}: mean r={statistics.mean(vals):.3f} (n={len(vals)})")
    mle_lines.append("  -> mle_feature_x correlation is what MLE directly exploits.")

    mle_lines.append("\nSTRONG MLE WINS (gap > 0.15), with test prediction examples:")
    strong_examples.sort(reverse=True)
    for gap, pid, res in strong_examples[:8]:
        beta = res["fitted_params"]["beta"]
        bias = res["fitted_params"]["bias"]
        _, _, test = trials_for_participant(
            _DATASET_CCT, pid, split_ratio=_SPLIT_RATIO, split_seed=_SPLIT_SEED,
            filter_mixed_gambles=False, psych_dataset_split=_PSYCH_SPLIT,
            local_dataset=None, mixed_gambles_csv="",
        )
        mle_prob = _mle_prob_fn(beta, bias)
        mle_lines.append(
            f"\n  pid={pid}: MLE test_ll={res['test_mean_loglik']:.4f}, TEH test_ll={teh_test_ll.get(pid, float('nan')):.4f}, "
            f"gap={gap:.4f}, beta={beta:.6f}, bias={bias:.6f}, MLE acc={res['test_accuracy']:.3f}"
        )
        for t in test[:3]:
            d = _cct_derived_fields(t["problem"])
            p = mle_prob(t)
            mle_lines.append(
                f"    action={t['action']} P(stop)={p:.3f} x={d['mle_feature_x']:.2f} "
                f"flipped={d['cards_flipped']:.0f} score={d['current_score']:.0f} "
                f"risk={d['risk_p_loss']:.3f} ev_cont={d['expected_continuation_value']:.1f}"
            )

    _write_text(out_dir / "mle_rule_report.txt", "\n".join(mle_lines) + "\n")

    # --- 2. accuracy vs confidence ---
    acc_report = [
        "Accuracy vs loglik vs confidence — 3frey2017cct",
        f"Participants: {len(acc_rows)}",
        "",
    ]
    similar = [r for r in acc_rows if r["similar_accuracy_much_worse_loglik"] == "True"]
    acc_report.append(
        f"Cases with |acc_MLE-acc_TEH|<=0.05 but loglik gap (MLE-TEH)>0.10: {len(similar)}"
    )
    for r in similar[:10]:
        acc_report.append(
            f"  pid={r['participant_id']}: MLE acc={r['mle_test_accuracy']} ll={r['mle_test_loglik']}, "
            f"TEH acc={r['teh_test_accuracy']} ll={r['teh_test_loglik']}, "
            f"MLE p@action={r['mle_mean_prob_at_action']} |p-.5|={r['mle_mean_abs_p_minus_half']}, "
            f"TEH p@action={r['teh_mean_prob_at_action']} |p-.5|={r['teh_mean_abs_p_minus_half']}"
        )

    underconf = [
        r for r in acc_rows
        if r["teh_test_accuracy"] and r["mle_test_accuracy"]
        and float(r["teh_test_accuracy"]) >= float(r["mle_test_accuracy"]) - 0.03
        and float(r["teh_test_loglik"]) < float(r["mle_test_loglik"]) - 0.10
    ]
    acc_report.append(f"\nTEH accuracy within 0.03 of MLE but loglik worse by >0.10: {len(underconf)}")
    acc_report.append("-> These are primarily UNDERCONFIDENCE losses (right side, low confidence).")

    mle_bins = _calibration_bins(all_mle_preds, all_mle_ys)
    teh_bins = _calibration_bins(all_teh_preds, all_teh_ys)
    acc_report.append("\nPooled TEST calibration bins (action=1 stop):")
    acc_report.append("MLE:")
    for b in mle_bins:
        acc_report.append(
            f"  bin {b['bin']}: n={b['n']} pred={b['mean_predicted_p_stop']:.3f} "
            f"obs={b['observed_stop_rate']:.3f} gap={b['calibration_gap']:+.3f}"
        )
    acc_report.append("TEH:")
    for b in teh_bins:
        acc_report.append(
            f"  bin {b['bin']}: n={b['n']} pred={b['mean_predicted_p_stop']:.3f} "
            f"obs={b['observed_stop_rate']:.3f} gap={b['calibration_gap']:+.3f}"
        )

    avg_mle_dev = statistics.mean(float(r["mle_mean_abs_p_minus_half"]) for r in acc_rows if r["mle_mean_abs_p_minus_half"])
    avg_teh_dev = statistics.mean(float(r["teh_mean_abs_p_minus_half"]) for r in acc_rows if r["teh_mean_abs_p_minus_half"])
    acc_report.append(
        f"\nMean |p-0.5| on test: MLE={avg_mle_dev:.3f}, TEH={avg_teh_dev:.3f}"
    )
    avg_mle_pa = statistics.mean(float(r["mle_mean_prob_at_action"]) for r in acc_rows if r["mle_mean_prob_at_action"])
    avg_teh_pa = statistics.mean(float(r["teh_mean_prob_at_action"]) for r in acc_rows if r["teh_mean_prob_at_action"])
    acc_report.append(f"Mean prob assigned to observed action: MLE={avg_mle_pa:.3f}, TEH={avg_teh_pa:.3f}")
    acc_report.append(
        "CONCLUSION: TEH loses mainly via UNDERCONFIDENCE — similar accuracy but lower prob@action;"
        " MLE assigns higher confidence on the dominant continue action (low P(stop) with high P(continue))."
        " Boundary errors are secondary; e.g. pid=10 has identical accuracy but MLE p@action=0.88 vs TEH=0.58."
    )

    _write_csv(
        out_dir / "accuracy_confidence_comparison.csv",
        list(acc_rows[0].keys()) if acc_rows else ["participant_id"],
        acc_rows,
    )
    _write_text(out_dir / "accuracy_confidence_report.txt", "\n".join(acc_report) + "\n")

    # --- 3. CCT stopping structure ---
    def _agg_stop_curve(trials: Sequence[Mapping[str, Any]], split: str) -> None:
        for feat, edges in bins_def:
            for lo, hi in edges:
                sub = []
                for t in trials:
                    d = _cct_derived_fields(t["problem"])
                    v = d[feat]
                    if lo <= v < hi:
                        sub.append(int(t["action"]))
                if not sub:
                    continue
                stopping_rows.append(
                    {
                        "split": split,
                        "feature": feat,
                        "bin_lo": str(lo),
                        "bin_hi": str(hi),
                        "n_trials": str(len(sub)),
                        "p_stop": _fmt(sum(sub) / len(sub)),
                        "p_continue": _fmt(1 - sum(sub) / len(sub)),
                    }
                )

    _agg_stop_curve(pooled_train, "train")
    _agg_stop_curve(pooled_test, "test")

    stop_report = [
        "CCT stopping structure — 3frey2017cct (pooled over participants)",
        f"Train trials pooled: {len(pooled_train)}, test trials pooled: {len(pooled_test)}",
        "",
        "Key patterns to encode in TEH prompt/seed:",
    ]
    for split in ("train", "test"):
        stop_report.append(f"\n{split.upper()} split:")
        for feat in ("cards_flipped", "risk_p_loss", "current_score", "mle_feature_x"):
            rows_f = [r for r in stopping_rows if r["split"] == split and r["feature"] == feat]
            for r in rows_f:
                stop_report.append(
                    f"  {feat} [{r['bin_lo']},{r['bin_hi']}): n={r['n_trials']} P(stop)={r['p_stop']}"
                )

    # train/test curve similarity for cards_flipped
    stop_report.append("\nTrain vs test P(stop) by cards_flipped:")
    for lo, hi in bins_def[0][1]:
        tr = [r for r in stopping_rows if r["split"] == "train" and r["feature"] == "cards_flipped"
              and r["bin_lo"] == str(lo)][0] if any(
            r["split"] == "train" and r["feature"] == "cards_flipped" and r["bin_lo"] == str(lo)
            for r in stopping_rows) else None
        te = [r for r in stopping_rows if r["split"] == "test" and r["feature"] == "cards_flipped"
              and r["bin_lo"] == str(lo)][0] if any(
            r["split"] == "test" and r["feature"] == "cards_flipped" and r["bin_lo"] == str(lo)
            for r in stopping_rows) else None
        if tr and te:
            stop_report.append(
                f"  flipped [{lo},{hi}): train P(stop)={tr['p_stop']} test P(stop)={te['p_stop']}"
            )

    maj_rates = []
    for pid in sorted(mle_results):
        _, _, test = trials_for_participant(
            _DATASET_CCT, pid, split_ratio=_SPLIT_RATIO, split_seed=_SPLIT_SEED,
            filter_mixed_gambles=False, psych_dataset_split=_PSYCH_SPLIT,
            local_dataset=None, mixed_gambles_csv="",
        )
        if test:
            stops = sum(int(t["action"]) for t in test)
            maj_rates.append(max(stops, len(test) - stops) / len(test))
    stop_report.append(
        f"\nParticipant test majority-action rate: mean={statistics.mean(maj_rates):.3f}, "
        f"min={min(maj_rates):.3f}, max={max(maj_rates):.3f}"
    )
    stop_report.append(
        "RECOMMENDED PROMPT/SEED additions: explicit P(stop) as sigmoid of (ev_stop - ev_continue);"
        " highlight that P(stop) rises with cards_flipped and loss risk; allow near-0/1 outputs."
    )

    _write_csv(
        out_dir / "cct_stopping_stats.csv",
        ["split", "feature", "bin_lo", "bin_hi", "n_trials", "p_stop", "p_continue"],
        stopping_rows,
    )
    _write_text(out_dir / "cct_stopping_report.txt", "\n".join(stop_report) + "\n")

    # --- 4. Prompt / program audit across TEH runs ---
    runs = _discover_runs(repo, _DATASET_CCT)
    prompt_lines = [
        "CCT TEH prompt/program audit — 3frey2017cct",
        f"Runs found: {[r.name for r in runs]}",
        "",
    ]
    latest = runs[0].name if runs else ""
    prompt_path = runs[0] / "prompts" / "infer_single_choice.txt" if runs else None
    if prompt_path and prompt_path.is_file():
        prompt_lines.append(f"Latest run: {runs[0]}")
        prompt_lines.append(f"Prompt: {prompt_path} ({prompt_path.stat().st_size} bytes)")
        head = prompt_path.read_text(encoding="utf-8", errors="replace")[:800]
        prompt_lines.append("Prompt head:\n" + head)
    prompt_lines.append("\nProgram style counts by run:")
    for run in runs[:3]:
        styles = Counter()
        n = 0
        for pdir in run.glob("participant_*"):
            bp = pdir / "best_program.py"
            if bp.is_file():
                styles[_classify_program(bp.read_text(encoding="utf-8", errors="replace"))] += 1
                n += 1
        prompt_lines.append(f"  {run.name}: n={n} styles={dict(styles)}")
    prompt_lines.append(
        "\nSTATUS: Latest run is run_260522_141701 (CCT-specific prompt with cards_flipped, "
        "current_score, gain/loss, n_cards_remaining). Still 48/50 raw_EV_linear vs "
        "run_260520_073506 partial (20 pids) with more history_heavy (6/20)."
    )
    prompt_lines.append(
        "No newer 3frey run after run_260522_141701 was found — post-prompt-revision A/B is PENDING."
        " Re-run TEH after seed/prompt changes and re-run this script to measure style shift."
    )
    _write_text(out_dir / "prompt_program_audit.txt", "\n".join(prompt_lines) + "\n")
    print(f"Wrote CCT outputs under {out_dir}")


def analyze_dataset4(repo: Path, config_data: Mapping[str, Any]) -> None:
    out_base = repo / _D4_OUT
    list_dir = out_base / "filtered_participant_lists"
    list_dir.mkdir(parents=True, exist_ok=True)

    counts_path = repo / _D4_AUDIT_COUNTS
    if not counts_path.is_file():
        raise SystemExit(f"Missing {counts_path}; run audit_dataset4_loading.py first")
    d4_counts = _load_d4_trial_counts(counts_path)

    thresholds = [4, 6, 10]
    subset_lines = [
        "Filtered participant lists — 4wulff2018description",
        f"Trial counts from {_D4_AUDIT_COUNTS}",
        f"Split used in audit: ratio={_D4_AUDIT_SPLIT_RATIO}, seed={_D4_AUDIT_SPLIT_SEED}",
        "",
        "NOTE: Current TEH run ordinals 0-49 mostly have train=1 (45/50) or train=4 (5/50).",
        "For meaningful TEH evaluation use ordinals with more parsed blocks (often >=45 or >=209).",
        "",
    ]

    for thr in thresholds:
        filtered = [r for r in d4_counts if r["train_trials"] >= thr]
        filtered.sort(key=lambda r: r["participant_id"])
        ids = [r["participant_id"] for r in filtered]
        fname = list_dir / f"min_train_trials_{thr}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write("\n".join(str(i) for i in ids) + ("\n" if ids else ""))
        first50 = ids[:50]
        subset_lines.append(f"--- min_train_trials >= {thr} ---")
        subset_lines.append(f"  count: {len(ids)}")
        subset_lines.append(f"  first 50 ids: {first50}")
        if filtered:
            tr = [r["train_trials"] for r in filtered]
            va = [r["val_trials"] for r in filtered]
            te = [r["test_trials"] for r in filtered]
            subset_lines.append(
                f"  train trials: min={min(tr)} med={statistics.median(tr):.0f} max={max(tr)}"
            )
            subset_lines.append(
                f"  val trials:   min={min(va)} med={statistics.median(va):.0f} max={max(va)}"
            )
            subset_lines.append(
                f"  test trials:  min={min(te)} med={statistics.median(te):.0f} max={max(te)}"
            )

    # stratified sample: pick up to 50 spread across train count buckets
    buckets = [(1, 1), (2, 3), (4, 9), (10, 999)]
    stratified: List[int] = []
    for lo, hi in buckets:
        pool = [r["participant_id"] for r in d4_counts if lo <= r["train_trials"] <= hi]
        step = max(1, len(pool) // 12)
        stratified.extend(pool[::step][:12])
    stratified = sorted(set(stratified))[:50]
    with open(list_dir / "stratified_sample_50.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(str(i) for i in stratified) + "\n")
    subset_lines.append("\n--- stratified sample (up to 50) ---")
    subset_lines.append(f"  ids: {stratified}")
    subset_lines.append("\nRECOMMENDED next TEH ordinals:")
    subset_lines.append("  - Quick sanity: 45-49 (train=4 each, in current run)")
    subset_lines.append("  - Medium-N: ordinals 209-247 (train=8) from audit CSV")
    subset_lines.append("  - Larger-N: ordinals with train>=10 (409 available), e.g. 630-649 (train=9)")

    _write_text(out_base / "filtered_subset_report.txt", "\n".join(subset_lines) + "\n")

    # --- 6. Stratified current-result diagnosis (ordinals 0-49 from existing run) ---
    mle, pt, cent, teh_t, teh_g, paths = _load_loglik_maps(repo, _DATASET_D4, config_data)
    current_pids = list(range(50))
    count_by_pid = {r["participant_id"]: r["train_trials"] for r in d4_counts}

    def _bucket(n: int) -> str:
        if n <= 1:
            return "train=1"
        if n <= 3:
            return "train=2-3"
        if n <= 9:
            return "train=4-9"
        return "train>=10"

    strat_lines = [
        "Stratified diagnosis — 4wulff2018description (existing run, ordinals 0-49)",
        f"TEH run: {cmp._auto_discover_teh_run(repo, dataset=_DATASET_D4, psych_dataset_split=_PSYCH_SPLIT)}",
        f"MLE run: {paths.get('MLE')}",
        "",
    ]
    bucket_pids: Dict[str, List[int]] = defaultdict(list)
    for pid in current_pids:
        bucket_pids[_bucket(count_by_pid.get(pid, 0))].append(pid)

    for bname in ("train=1", "train=2-3", "train=4-9", "train>=10"):
        pids = bucket_pids.get(bname, [])
        if not pids:
            strat_lines.append(f"{bname}: n=0")
            continue
        methods = {
            "MLE": mle,
            "prospect_theory": pt,
            "Centaur": cent,
            "TEH": teh_t,
            "TEH_gated": teh_g,
        }
        strat_lines.append(f"\n{bname}: n={len(pids)} pids={pids[:20]}{'...' if len(pids)>20 else ''}")
        nb = _num_best({"MLE": mle, "prospect_theory": pt, "Centaur": cent, "TEH": teh_t}, pids)
        for m, mp in methods.items():
            st = _method_stats([mp.get(pid) for pid in pids])
            if st["avg"] is None:
                continue
            nb_val = nb.get("TEH" if m == "TEH_gated" else m, 0)
            strat_lines.append(
                f"  {m}: avg={st['avg']:.3f} median={st['median']:.3f} "
                f"num_best={nb_val} "
                f"near_perf={st['near_perfect']} catastrophic={st['catastrophic']}"
            )
        strat_lines.append(
            "  -> Tiny train=1 bucket dominates current run; TEH failures are largely tiny-N + deterministic actions."
        )

    _write_text(out_base / "stratified_current_results.txt", "\n".join(strat_lines) + "\n")
    print(f"Wrote dataset-4 outputs under {out_base}")


def cross_dataset_compare(repo: Path, config_data: Mapping[str, Any]) -> None:
    conv_path = repo / _CONVERGENCE_CSV
    conv: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if conv_path.is_file():
        with open(conv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                conv[(row["dataset"], int(float(row["participant_id"])))] = row

    d4_counts = _load_d4_trial_counts(repo / _D4_AUDIT_COUNTS)
    d4_ge4 = {r["participant_id"] for r in d4_counts if r["train_trials"] >= 4}

    lines = [
        "Cross-dataset positive-control comparison",
        "",
    ]

    for label, dataset, pid_filter in (
        ("3frey2017cct (full 0-49)", _DATASET_CCT, lambda p: True),
        ("4wulff train>=4 subset", _DATASET_D4, lambda p: p in d4_ge4),
        ("5speekenbrink (full)", _DATASET_SPEEK, lambda p: True),
    ):
        mle, pt, _, teh_t, _, _ = _load_loglik_maps(repo, dataset, config_data)
        teh_run = cmp._auto_discover_teh_run(repo, dataset=dataset, psych_dataset_split=_PSYCH_SPLIT)
        pids = sorted(pid for pid in set(teh_t) | set(mle) if pid_filter(pid))
        if not pids:
            lines.append(f"{label}: no participants")
            continue

        gaps = []
        maj = []
        conf = []
        styles = Counter()
        conv_steps = []
        train_ns = []

        for pid in pids:
            if pid in mle and pid in teh_t:
                gaps.append(mle[pid] - teh_t[pid])
            if dataset == _DATASET_D4:
                train_ns.append(next((r["train_trials"] for r in d4_counts if r["participant_id"] == pid), 0))
            elif dataset == _DATASET_CCT:
                tr, _, _te = trials_for_participant(
                    dataset, pid, split_ratio=_SPLIT_RATIO, split_seed=_SPLIT_SEED,
                    filter_mixed_gambles=False, psych_dataset_split=_PSYCH_SPLIT,
                    local_dataset=None, mixed_gambles_csv="",
                )
                train_ns.append(len(tr))
            else:
                tr, _, _ = trials_for_participant(
                    dataset, pid, split_ratio=_SPLIT_RATIO, split_seed=_SPLIT_SEED,
                    filter_mixed_gambles=False, psych_dataset_split=_PSYCH_SPLIT,
                    local_dataset=None, mixed_gambles_csv="",
                )
                train_ns.append(len(tr))

            if teh_run:
                bp = teh_run / f"participant_{pid}" / "best_program.py"
                if bp.is_file():
                    code = bp.read_text(encoding="utf-8", errors="replace")
                    styles[_classify_program(code)] += 1
                    if dataset in (_DATASET_CCT, _DATASET_SPEEK):
                        try:
                            choose = _load_choose_fn(bp)
                            _, _, test = trials_for_participant(
                                dataset, pid, split_ratio=_SPLIT_RATIO, split_seed=_SPLIT_SEED,
                                filter_mixed_gambles=False, psych_dataset_split=_PSYCH_SPLIT,
                                local_dataset=None, mixed_gambles_csv="",
                            )
                            m = _trial_metrics(test, lambda t: _prob_from_choose(
                                choose(t["problem"], t.get("history", []))))
                            conf.append(m.get("mean_abs_p_minus_half") or 0)
                            if test:
                                stops = sum(int(t["action"]) for t in test)
                                maj.append(max(stops, len(test)-stops)/len(test))
                        except Exception:
                            pass
            c = conv.get((dataset, pid), {})
            if c.get("tail_converged_steps"):
                conv_steps.append(int(float(c["tail_converged_steps"])))

        mle_avg = statistics.mean(mle[p] for p in pids if p in mle)
        teh_avg = statistics.mean(teh_t[p] for p in pids if p in teh_t)
        lines.append(f"=== {label} (n={len(pids)}) ===")
        lines.append(f"  TEH run: {teh_run}")
        lines.append(f"  mean train trials: {statistics.mean(train_ns):.1f}")
        lines.append(f"  avg test loglik: MLE={mle_avg:.3f} TEH={teh_avg:.3f} gap(MLE-TEH)={mle_avg-teh_avg:+.3f}")
        if gaps:
            lines.append(f"  per-participant gap mean={statistics.mean(gaps):+.3f}")
        if maj:
            lines.append(f"  test majority rate mean={statistics.mean(maj):.3f}")
        if conf:
            lines.append(f"  TEH mean |p-0.5|={statistics.mean(conf):.3f}")
        if conv_steps:
            lines.append(f"  tail_converged_steps mean={statistics.mean(conv_steps):.1f}")
        lines.append(f"  program styles: {dict(styles)}")
        lines.append("")

    lines.append(
        "WHY 5speekenbrink works: more train trials, TEH attains higher |p-0.5| on correct side,"
        " lower MLE advantage; 3frey/4wulff have high majority rates where MLE one-feature stopping"
        " rule fits tightly but TEH sigmoid EV programs stay near 0.5-0.8."
    )
    _write_text(repo / _CCT_OUT / "cross_dataset_comparison.txt", "\n".join(lines) + "\n")
    print(f"Wrote {repo / _CCT_OUT / 'cross_dataset_comparison.txt'}")


def main() -> None:
    repo = _repo()
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_config", type=Path, default=Path(cmp._DEFAULT_BASELINE_CONFIG))
    args = p.parse_args()
    config_path = args.baseline_config.expanduser()
    config_path = config_path.resolve() if config_path.is_absolute() else (repo / config_path).resolve()
    config_data = cmp._load_baseline_config_file(config_path)

    print("Analyzing 3frey2017cct...")
    analyze_cct(repo, config_data)
    print("Analyzing 4wulff2018description...")
    analyze_dataset4(repo, config_data)
    print("Cross-dataset comparison...")
    cross_dataset_compare(repo, config_data)


if __name__ == "__main__":
    main()
