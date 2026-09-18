#!/usr/bin/env python3
"""Schema-v4 hierarchical source-selection pipeline (occurrence / fitness / combined).

Freezes signatures WITHOUT transfer outcomes, then evaluates on the old
unaffected 6×6 improve_test_loglik table only.

Approaches
----------
A) Hierarchical motif-occurrence MEM (empirical-Bayes logit random intercept)
B) Hierarchical fitness-effect MEM (exact ΔF; MixedLM random slopes)
C) Combined: concat(zscore(occurrence), zscore(fitness slopes)) equal blocks

Selection: argmax cosine among SIX allowlist; self excluded; lex tie-break.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mem.schema_population_motif_v4 import (  # noqa: E402
    BEHAVIORAL_MOTIFS_V4,
)

MOTIFS: Tuple[str, ...] = tuple(BEHAVIORAL_MOTIFS_V4)
TRANSITION_SUFFIXES: Tuple[str, ...] = ("added", "removed", "modified")

# Old unaffected six-source allowlist (targets ∩ sources for 6×6 eval).
SIX: Tuple[str, ...] = (
    "1peterson2021using",
    "3frey2017cct",
    "4wulff2018description",
    "7hilbig2014generalized",
    "11enkavi2019recentprobes",
    "mixed_gambles",
)

DATASET_LABELS: Dict[str, str] = {
    "1peterson2021using": "Choice13k",
    "2plonsky2018when": "CPC18",
    "3frey2017cct": "Frey CCT",
    "4wulff2018description": "Wulff",
    "5speekenbrink2008learning": "Speekenbrink",
    "7hilbig2014generalized": "Hilbig",
    "10frey2017risk": "Frey Risk",
    "11enkavi2019recentprobes": "Enkavi",
    "12badham2017deficits": "Badham",
    "mixed_gambles": "Mixed Gambles",
    "bergert_nosofsky_2007": "Bergert",
    "guan_2020_stopping": "Guan",
    "steyvers_2009_bandit": "Steyvers",
    "13schulz2020finding": "Schulz",
    "14kool2016when": "Kool",
}

# Prespecified fitness-effect support gates (do NOT force sparse effects).
MIN_POSITIVE_ROWS = 20
MIN_DATASETS_WITH_POSITIVE = 3
MIN_DATASETS_WITH_BOTH = 2

DEFAULT_ANN_ROOT = Path(
    "/careAIDrive/zichang/repo/mindsAsCode/analysis_2026Sep/mem/"
    "t_pics_source_pop10_schema_v4/annotations"
)
DEFAULT_MANIFEST = Path(
    "/careAIDrive/zichang/repo/mindsAsCode/analysis_2026Sep/mem/"
    "t_pics_source_pop10_schema_v4/program_manifest_all.json"
)
DEFAULT_CFG = (
    _REPO_ROOT / "analysis/config/misc/Sep17_T-PICS/config_T-PICS_schema4.yaml"
)
DEFAULT_OUT = Path(
    "/careAIDrive/zichang/repo/mindsAsCode/analysis_2026Sep/mem/"
    "t_pics_source_pop10_schema_v4/source_selection_v4"
)
DEFAULT_TRANSFER = (
    _REPO_ROOT
    / "generated_outputs_transfer_old/teh_transfer/run_260616_235649"
    / "summary_csv/single_transfer/matrix_improve_test_loglik.csv"
)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(math.log(p / (1.0 - p)))


def invlogit(x: float) -> float:
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def zscore_across_rows(mat: np.ndarray) -> np.ndarray:
    """Column-wise z-score across datasets (rows). Constant cols → 0."""
    x = np.asarray(mat, dtype=float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=0)
    out = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        if not np.isfinite(sd[j]) or sd[j] < 1e-12:
            out[:, j] = 0.0
        else:
            out[:, j] = (x[:, j] - mu[j]) / sd[j]
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# IO: annotations, config, scores
# ---------------------------------------------------------------------------


def resolve_repo() -> Path:
    if (_REPO_ROOT / "teh.py").exists():
        return _REPO_ROOT
    alt = Path("/common/home/users/z/zichang.ge.2023/repo/mindsAsCode")
    return alt if (alt / "teh.py").exists() else _REPO_ROOT


def load_run_dirs(cfg_path: Path, repo: Path) -> Dict[str, Path]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    pops = cfg["step1"]["source_pops"]
    out: Dict[str, Path] = {}
    for ds, meta in pops.items():
        out[str(ds)] = (repo / meta["run_dir"]).resolve()
    return out


def load_annotations_union(
    ann_root: Path,
    *,
    prefer_by_dataset: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Union annotations by resume_key; by_dataset preferred over shards."""
    by_dataset: Dict[str, Dict[str, Any]] = {}
    shards: Dict[str, Dict[str, Any]] = {}

    bd = ann_root / "by_dataset"
    if bd.is_dir():
        for path in sorted(bd.rglob("annotations_population_v4.jsonl")):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                rk = str(rec["resume_key"])
                by_dataset[rk] = rec

    for path in sorted(ann_root.glob("shard_*/annotations_population_v4.jsonl")):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            rk = str(rec["resume_key"])
            shards[rk] = rec

    if prefer_by_dataset:
        merged = dict(shards)
        merged.update(by_dataset)  # by_dataset wins
        preferred_n = len(by_dataset)
    else:
        merged = dict(by_dataset)
        merged.update(shards)
        preferred_n = len(shards)

    rows = list(merged.values())
    df = pd.DataFrame(rows)
    stats = {
        "ann_root": str(ann_root),
        "n_by_dataset_keys": int(len(by_dataset)),
        "n_shard_keys": int(len(shards)),
        "n_union_keys": int(len(merged)),
        "n_only_shards": int(len(set(shards) - set(by_dataset))),
        "n_only_by_dataset": int(len(set(by_dataset) - set(shards))),
        "prefer_by_dataset": prefer_by_dataset,
        "n_preferred_layer": preferred_n,
    }
    return df, stats


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def build_score_lookup(run_dir: Path) -> Dict[str, float]:
    """Recover selection scores available on disk for one T-PICS run.

    Sources (in increasing priority on write order, last wins for ties):
      * global_elite_pool/pool_manifest.json → program_id: global_fitness
      * per-iter metrics pool_best_program_id → pool_best_selection_score
      * results.json baseline_global_train_loglik → global_baseline / baseline

    Non-elite candidates without a persisted score remain missing.
    """
    run_dir = Path(run_dir)
    gp = run_dir / "global_phase"
    scores: Dict[str, float] = {}

    pool_path = gp / "global_elite_pool" / "pool_manifest.json"
    if pool_path.is_file():
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        for prog in pool.get("programs", []):
            pid = str(prog.get("program_id") or "")
            gf = _safe_float(prog.get("global_fitness"))
            if pid and gf is not None:
                scores[pid] = gf

    for it in range(1, 64):
        mp = gp / f"iteration_{it}" / "metrics.json"
        if not mp.is_file():
            break
        m = json.loads(mp.read_text(encoding="utf-8"))
        pid = m.get("pool_best_program_id")
        sel = _safe_float(m.get("pool_best_selection_score"))
        if pid and sel is not None:
            scores[str(pid)] = sel

    results_path = gp / "results.json"
    if results_path.is_file():
        res = json.loads(results_path.read_text(encoding="utf-8"))
        base = _safe_float(res.get("baseline_global_train_loglik"))
        if base is not None:
            scores["global_baseline"] = base
            scores["baseline"] = base

    return scores


def presence_vector(rec: Dict[str, Any]) -> Dict[str, int]:
    """Binary presence for each BEHAVIORAL_MOTIFS_V4 motif."""
    out: Dict[str, int] = {}
    presence = rec.get("presence")
    state = rec.get("candidate_motif_state") or rec.get("program_motif_state") or []
    state_set = set(state) if isinstance(state, list) else set()
    for m in MOTIFS:
        if isinstance(presence, dict) and m in presence:
            out[m] = int(bool(presence[m]))
        else:
            out[m] = int(m in state_set)
    return out


def transition_flags(rec: Dict[str, Any]) -> Dict[str, int]:
    flags: Dict[str, int] = {}
    for suffix, key in (
        ("added", "added_motifs"),
        ("removed", "removed_motifs"),
        ("modified", "modified_motifs"),
    ):
        vals = rec.get(key) or []
        if not isinstance(vals, list):
            vals = []
        s = set(str(x) for x in vals)
        for m in MOTIFS:
            flags[f"{m}_{suffix}"] = int(m in s)
    return flags


# ---------------------------------------------------------------------------
# Analysis panel
# ---------------------------------------------------------------------------


def build_analysis_panel(
    ann: pd.DataFrame,
    run_dirs: Dict[str, Path],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    score_cache: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, Any]] = []
    n_exact = 0
    n_exact_scored = 0
    n_missing_cand_score = 0
    n_missing_parent_score = 0

    for _, rec in ann.iterrows():
        d = rec.to_dict()
        ds = str(d.get("dataset"))
        run_dir = run_dirs.get(ds)
        if run_dir is None:
            continue
        if ds not in score_cache:
            score_cache[ds] = build_score_lookup(run_dir)

        scores = score_cache[ds]
        cand_id = str(d.get("program_id") or d.get("candidate_id") or "")
        parent_id = str(d.get("parent") or d.get("parent_id") or "")
        cand_score = scores.get(cand_id)
        parent_score = scores.get(parent_id)
        exact = bool(d.get("reference_is_exact"))
        if exact:
            n_exact += 1
            if cand_score is None:
                n_missing_cand_score += 1
            if parent_score is None:
                n_missing_parent_score += 1

        delta_f = None
        if cand_score is not None and parent_score is not None:
            delta_f = float(cand_score) - float(parent_score)
            if exact:
                n_exact_scored += 1

        row: Dict[str, Any] = {
            "resume_key": d.get("resume_key"),
            "dataset": ds,
            "run_id": d.get("run_id"),
            "iteration": int(d["iteration"]) if pd.notna(d.get("iteration")) else None,
            "program_id": cand_id,
            "parent_id": parent_id,
            "source": d.get("source"),
            "reference_type": d.get("reference_type") or d.get("reference_kind"),
            "reference_is_exact": exact,
            "reference_is_proxy": bool(d.get("reference_is_proxy")),
            "candidate_selection_score": cand_score,
            "parent_selection_score": parent_score,
            "delta_f": delta_f,
            "score_recovered": cand_score is not None and parent_score is not None,
            "confidence": _safe_float(d.get("confidence")),
        }
        row.update(presence_vector(d))
        row.update(transition_flags(d))
        rows.append(row)

    df = pd.DataFrame(rows)
    stats = {
        "n_rows": int(len(df)),
        "n_datasets": int(df["dataset"].nunique()) if len(df) else 0,
        "n_exact": n_exact,
        "n_exact_with_delta_f": n_exact_scored,
        "n_exact_missing_candidate_score": n_missing_cand_score,
        "n_exact_missing_parent_score": n_missing_parent_score,
        "score_recovery_note": (
            "ΔF uses recovered elite/metrics scores: global_elite_pool "
            "global_fitness, per-iter pool_best_selection_score, and "
            "results.json baseline_global_train_loglik for global_baseline. "
            "Non-elite candidates typically lack persisted selection scores "
            "and are excluded from fitness-effect MEM rows."
        ),
        "datasets": sorted(df["dataset"].unique().tolist()) if len(df) else [],
    }
    return df, stats


# ---------------------------------------------------------------------------
# A) Occurrence MEM (empirical Bayes logit random intercept)
# ---------------------------------------------------------------------------


def fit_occurrence_eb(
    panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Jeffreys-smoothed MoM EB shrink; signature = invlogit(α + b_d)."""
    datasets = sorted(panel["dataset"].unique().tolist())
    model_rows: List[Dict[str, Any]] = []
    eb_rows: List[Dict[str, Any]] = []
    sig_rows: List[Dict[str, Any]] = []

    # Precompute per-dataset n and motif counts
    by_ds = {ds: panel[panel["dataset"] == ds] for ds in datasets}
    n_by_ds = {ds: int(len(g)) for ds, g in by_ds.items()}

    for motif in MOTIFS:
        k_total = 0
        n_total = 0
        ds_stats: List[Dict[str, Any]] = []
        for ds in datasets:
            g = by_ds[ds]
            n = int(len(g))
            k = int(pd.to_numeric(g[motif], errors="coerce").fillna(0).sum())
            k_total += k
            n_total += n
            raw = k / n if n else 0.0
            # Jeffreys: p = (k + 0.5) / (n + 1)
            sm = (k + 0.5) / (n + 1.0)
            y = logit(sm)
            samp_var = 1.0 / (n * sm * (1.0 - sm))
            ds_stats.append(
                {
                    "dataset": ds,
                    "n": n,
                    "k": k,
                    "raw": raw,
                    "smoothed": sm,
                    "logit": y,
                    "samp_var": samp_var,
                }
            )

        alpha = logit((k_total + 0.5) / (n_total + 1.0))
        logits = np.array([s["logit"] for s in ds_stats], dtype=float)
        samp_vars = np.array([s["samp_var"] for s in ds_stats], dtype=float)
        between = float(np.var(logits, ddof=1)) if len(logits) > 1 else 0.0
        mean_samp = float(np.mean(samp_vars)) if len(samp_vars) else 0.0
        tau2 = max(0.0, between - mean_samp)
        warning = "tau2_at_boundary_zero" if tau2 <= 0.0 else ""
        n_ds_any = sum(1 for s in ds_stats if s["k"] > 0)
        n_ds_all = sum(1 for s in ds_stats if s["k"] == s["n"] and s["n"] > 0)
        extreme = sum(
            1 for s in ds_stats if s["raw"] <= 1e-12 or s["raw"] >= 1.0 - 1e-12
        ) / max(1, len(ds_stats))

        model_rows.append(
            {
                "motif": motif,
                "n_pos": k_total,
                "n": n_total,
                "n_datasets_any": n_ds_any,
                "n_datasets_all_positive": n_ds_all,
                "alpha": alpha,
                "tau2": tau2,
                "inv_logit_alpha": invlogit(alpha),
                "between_logit_var": between,
                "mean_sampling_var_logit": mean_samp,
                "between_prev_sd": float(np.std([s["raw"] for s in ds_stats], ddof=0)),
                "frac_datasets_extreme_prev": float(extreme),
                "method": "empirical_bayes_logit_random_intercept",
                "warning": warning,
                "singular_or_boundary": bool(tau2 <= 0.0),
            }
        )

        sig_map: Dict[str, float] = {}
        for s in ds_stats:
            v = float(s["samp_var"])
            y = float(s["logit"])
            if tau2 <= 0.0:
                w = 0.0
            else:
                w = tau2 / (tau2 + v)
            eb_logit = alpha + w * (y - alpha)
            post_var = (
                0.0 if tau2 <= 0.0 else (tau2 * v) / (tau2 + v)
            )
            eb_prob = invlogit(eb_logit)
            sig_map[s["dataset"]] = eb_prob
            eb_rows.append(
                {
                    "motif": motif,
                    "dataset": s["dataset"],
                    "dataset_label": DATASET_LABELS.get(s["dataset"], s["dataset"]),
                    "n_programs": s["n"],
                    "n_positive": s["k"],
                    "raw_prevalence": s["raw"],
                    "smoothed_prevalence": s["smoothed"],
                    "obs_logit": y,
                    "sampling_var_logit": v,
                    "shrinkage_weight": w,
                    "eb_logit_mean": eb_logit,
                    "eb_logit_se": math.sqrt(post_var) if post_var > 0 else 0.0,
                    "eb_prob": eb_prob,
                    "alpha": alpha,
                    "tau2": tau2,
                    "b_hat": eb_logit - alpha,
                }
            )

    model_df = pd.DataFrame(model_rows)
    eb_df = pd.DataFrame(eb_rows)

    # Signature matrix: datasets × motifs = eb_prob
    sig_pivot = eb_df.pivot(index="dataset", columns="motif", values="eb_prob")
    sig_pivot = sig_pivot.reindex(index=datasets, columns=list(MOTIFS))
    for ds in datasets:
        row = {"dataset": ds, "dataset_label": DATASET_LABELS.get(ds, ds), "n_programs": n_by_ds[ds]}
        for m in MOTIFS:
            row[m] = float(sig_pivot.loc[ds, m])
        sig_rows.append(row)
    sig_df = pd.DataFrame(sig_rows)

    meta = {
        "method": "empirical_bayes_logit_random_intercept",
        "formula": "logit(p_{d,j,e}) = alpha_e + b_{d,e}, b ~ N(0, tau^2)",
        "smoothing": "Jeffreys p=(k+0.5)/(n+1); alpha=logit(overall Jeffreys)",
        "tau2": "MoM max(0, Var_d(logit_d) - mean sampling_var); sampling_var=1/(n p (1-p))",
        "signature": "invlogit(alpha + b_hat) per motif",
        "motifs": list(MOTIFS),
        "n_datasets": len(datasets),
        "peeked_transfer": False,
    }
    return sig_df, eb_df, model_df, meta


# ---------------------------------------------------------------------------
# B) Fitness-effect MEM
# ---------------------------------------------------------------------------


def _transition_columns() -> List[str]:
    return [f"{m}_{s}" for m in MOTIFS for s in TRANSITION_SUFFIXES]


def fitness_support_audit(
    exact_df: pd.DataFrame,
    *,
    min_positive_rows: int = MIN_POSITIVE_ROWS,
    min_datasets_with_positive: int = MIN_DATASETS_WITH_POSITIVE,
    min_datasets_with_both: int = MIN_DATASETS_WITH_BOTH,
) -> Dict[str, Any]:
    cols = [c for c in _transition_columns() if c in exact_df.columns]
    motifs_out: List[Dict[str, Any]] = []
    for col in cols:
        vals = pd.to_numeric(exact_df[col], errors="coerce").fillna(0).astype(int)
        n_pos = int((vals == 1).sum())
        g = exact_df.assign(_v=vals).groupby("dataset")["_v"]
        n_ds_pos = int((g.max() >= 1).sum())
        n_ds_both = int(((g.max() >= 1) & (g.min() <= 0)).sum())
        reasons: List[str] = []
        if n_pos < min_positive_rows:
            reasons.append("below_min_positive_rows")
        if n_ds_pos < min_datasets_with_positive:
            reasons.append("below_min_datasets_with_positive")
        if n_ds_both < min_datasets_with_both:
            reasons.append("below_min_datasets_with_both")
        if int(vals.nunique()) < 2:
            reasons.append("constant_predictor")
        motifs_out.append(
            {
                "transition": col,
                "n_positive_rows": n_pos,
                "n_datasets_with_positive": n_ds_pos,
                "n_datasets_with_both": n_ds_both,
                "supported": len(reasons) == 0,
                "exclusion_reasons": reasons,
            }
        )
    supported = [m["transition"] for m in motifs_out if m["supported"]]
    return {
        "min_positive_rows": min_positive_rows,
        "min_datasets_with_positive": min_datasets_with_positive,
        "min_datasets_with_both": min_datasets_with_both,
        "n_exact_scored_rows": int(len(exact_df)),
        "transitions": motifs_out,
        "supported_transitions": supported,
        "n_supported": len(supported),
        "note": (
            "Sparse / unsupported transitions are NOT forced into the fitness "
            "signature. Gates are prespecified and do not use transfer outcomes."
        ),
    }


def _extract_dataset_slopes(
    result: Any,
    *,
    focal: str,
    fixed_coef: float,
) -> Tuple[Dict[str, float], Optional[float]]:
    """θ̂_d = μ̂ + b̂_d for random slope on focal."""
    slopes: Dict[str, float] = {}
    var_slope: Optional[float] = None
    try:
        cov = result.cov_re
        if hasattr(cov, "iloc"):
            cols = list(cov.columns)
            if focal in cols:
                var_slope = float(cov.loc[focal, focal])
            elif len(cols) >= 2:
                var_slope = float(cov.iloc[1, 1])
        elif hasattr(cov, "shape"):
            arr = np.asarray(cov, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                var_slope = float(arr[1, 1])
    except Exception:
        var_slope = None

    try:
        re_df = result.random_effects
        for pid, re_vec in re_df.items():
            dev = None
            if hasattr(re_vec, "get"):
                dev = re_vec.get(focal)
                if dev is None and len(re_vec) >= 2:
                    try:
                        # Series values: Intercept, then focal
                        keys = list(re_vec.index)
                        if focal in keys:
                            dev = float(re_vec[focal])
                        else:
                            dev = float(list(re_vec.values)[1])
                    except Exception:
                        dev = None
                elif dev is None and len(re_vec) == 1:
                    dev = 0.0
                else:
                    dev = float(dev) if dev is not None else None
            else:
                arr = np.asarray(re_vec, dtype=float).ravel()
                dev = float(arr[1]) if arr.size >= 2 else 0.0
            if dev is None:
                continue
            slopes[str(pid)] = float(fixed_coef) + float(dev)
    except Exception:
        return {}, var_slope
    return slopes, var_slope


def fit_fitness_mixedlm(
    exact_df: pd.DataFrame,
    supported: Sequence[str],
    all_datasets: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise SystemExit(
            "statsmodels is required for fitness MixedLM "
            "(conda env evo312)."
        ) from exc

    work = exact_df.copy()
    work["delta_f"] = pd.to_numeric(work["delta_f"], errors="coerce")
    work["iteration"] = pd.to_numeric(work["iteration"], errors="coerce")
    work = work.dropna(subset=["delta_f", "iteration", "dataset"])
    work["dataset"] = work["dataset"].astype(str)
    # Center iteration (prespecified covariate; not tuned on transfer).
    work["iteration_c"] = work["iteration"] - float(work["iteration"].mean())

    fit_records: Dict[str, Any] = {}
    # dataset → transition → slope
    slope_map: Dict[str, Dict[str, float]] = {ds: {} for ds in all_datasets}

    for T in supported:
        if T not in work.columns:
            fit_records[T] = {"status": "missing_column"}
            continue
        w = work.copy()
        w[T] = pd.to_numeric(w[T], errors="coerce").fillna(0).astype(int)
        n_pos = int((w[T] == 1).sum())
        formula = f"delta_f ~ {T} + iteration_c"
        re_formula = f"1 + {T}"
        warnings_list: List[str] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula, w, groups=w["dataset"], re_formula=re_formula
                )
                result = model.fit(method="lbfgs", reml=True, maxiter=200)
                for c in caught:
                    warnings_list.append(str(c.message))
        except Exception as exc:
            fit_records[T] = {
                "status": "fit_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "formula": formula,
                "re_formula": re_formula,
                "n_rows": int(len(w)),
                "n_positive": n_pos,
            }
            continue

        params = result.fe_params
        mu = float(params.get(T, float("nan")))
        slopes, var_slope = _extract_dataset_slopes(
            result, focal=T, fixed_coef=mu
        )
        status = "ok"
        if not bool(result.converged):
            status = "not_converged"
        # Fill missing datasets with population mean μ̂
        for ds in all_datasets:
            slope_map[ds][T] = float(slopes.get(ds, mu)) if math.isfinite(mu) else float(
                slopes.get(ds, 0.0)
            )

        fit_records[T] = {
            "status": status,
            "formula": formula,
            "re_formula": re_formula,
            "n_rows": int(len(w)),
            "n_positive": n_pos,
            "converged": bool(result.converged),
            "fixed_coef_mu": mu,
            "fixed_se": float(result.bse_fe.get(T, float("nan")))
            if hasattr(result.bse_fe, "get")
            else None,
            "fixed_pvalue": float(result.pvalues.get(T, float("nan")))
            if hasattr(result.pvalues, "get")
            else None,
            "random_slope_variance": var_slope,
            "n_datasets_with_re": len(slopes),
            "warnings": warnings_list[:20],
            "llf": float(result.llf) if result.llf is not None else None,
        }

    # Only include transitions that fitted successfully into the signature.
    used = [
        T
        for T in supported
        if fit_records.get(T, {}).get("status") in ("ok", "not_converged")
        and math.isfinite(float(fit_records[T].get("fixed_coef_mu", float("nan"))))
    ]
    sig_rows: List[Dict[str, Any]] = []
    for ds in all_datasets:
        row: Dict[str, Any] = {
            "dataset": ds,
            "dataset_label": DATASET_LABELS.get(ds, ds),
        }
        for T in used:
            row[T] = slope_map[ds].get(T, float("nan"))
        sig_rows.append(row)
    sig_df = pd.DataFrame(sig_rows)

    meta = {
        "method": "mixedlm_delta_f_random_slope",
        "formula_template": "delta_f ~ T + iteration_c + (1 + T | dataset)",
        "exact_rows_only": True,
        "supported_transitions_prespecified": list(supported),
        "signature_transitions": used,
        "dropped_fit_failures": [
            T for T in supported if T not in used
        ],
        "fits": fit_records,
        "peeked_transfer": False,
        "score_recovery_note": (
            "Fitness ΔF uses recovered elite/metrics scores (non-elite lack "
            "persisted scores)."
        ),
    }
    return sig_df, meta


# ---------------------------------------------------------------------------
# C) Combined signatures
# ---------------------------------------------------------------------------


def build_combined_signatures(
    occ_sig: pd.DataFrame,
    fit_sig: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    datasets = sorted(set(occ_sig["dataset"]).union(set(fit_sig["dataset"])))
    occ_cols = list(MOTIFS)
    fit_cols = [c for c in fit_sig.columns if c not in ("dataset", "dataset_label")]

    occ_mat = np.zeros((len(datasets), len(occ_cols)), dtype=float)
    fit_mat = np.zeros((len(datasets), max(1, len(fit_cols))), dtype=float)
    for i, ds in enumerate(datasets):
        orow = occ_sig[occ_sig["dataset"] == ds]
        for j, c in enumerate(occ_cols):
            occ_mat[i, j] = float(orow.iloc[0][c]) if len(orow) else 0.0
        if fit_cols:
            frow = fit_sig[fit_sig["dataset"] == ds]
            for j, c in enumerate(fit_cols):
                if len(frow) and c in frow.columns:
                    v = frow.iloc[0][c]
                    fit_mat[i, j] = float(v) if pd.notna(v) else 0.0
                else:
                    fit_mat[i, j] = 0.0

    occ_z = zscore_across_rows(occ_mat)
    if fit_cols:
        fit_z = zscore_across_rows(fit_mat)
    else:
        fit_z = np.zeros((len(datasets), 0), dtype=float)

    # Equal blocks after z-score: each block contributes equal L2 mass via
    # explicit sqrt(0.5) scaling of each half (equal weight when both present).
    rows: List[Dict[str, Any]] = []
    feature_names: List[str] = [f"occ_z__{c}" for c in occ_cols] + [
        f"fit_z__{c}" for c in fit_cols
    ]
    for i, ds in enumerate(datasets):
        row: Dict[str, Any] = {
            "dataset": ds,
            "dataset_label": DATASET_LABELS.get(ds, ds),
        }
        # Store raw z-scored halves; selection uses concat of equal-weight halves.
        vec_parts: List[np.ndarray] = []
        if occ_z.shape[1]:
            half = occ_z[i] * math.sqrt(0.5)
            for j, c in enumerate(occ_cols):
                row[f"occ_z__{c}"] = float(half[j])
            vec_parts.append(half)
        if fit_z.shape[1]:
            half = fit_z[i] * math.sqrt(0.5)
            for j, c in enumerate(fit_cols):
                row[f"fit_z__{c}"] = float(half[j])
            vec_parts.append(half)
        rows.append(row)

    meta = {
        "method": "combined_equal_block_zscore",
        "occurrence_dims": occ_cols,
        "fitness_dims": fit_cols,
        "feature_names": feature_names,
        "weighting": (
            "zscore each block across 15 datasets, then scale each block by "
            "sqrt(1/2) so occurrence and fitness contribute equal weight"
        ),
        "peeked_transfer": False,
        "n_occurrence_dims": len(occ_cols),
        "n_fitness_dims": len(fit_cols),
    }
    return pd.DataFrame(rows), meta


# ---------------------------------------------------------------------------
# Selection + eval (transfer GT only here)
# ---------------------------------------------------------------------------


def signature_vectors(
    sig_df: pd.DataFrame,
    cols: Sequence[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for _, row in sig_df.iterrows():
        ds = str(row["dataset"])
        out[ds] = np.array([float(row[c]) for c in cols], dtype=float)
    return out


def select_sources(
    vecs: Dict[str, np.ndarray],
    *,
    targets: Sequence[str],
    allowlist: Sequence[str],
) -> pd.DataFrame:
    """Argmax cosine among allowlist; self excluded; lex tie-break on source id."""
    rows: List[Dict[str, Any]] = []
    allow = list(allowlist)
    for target in targets:
        if target not in vecs:
            continue
        cands = [s for s in allow if s != target and s in vecs]
        if not cands:
            continue
        scored = [(cosine(vecs[target], vecs[s]), s) for s in cands]
        # max cosine, then lexicographic source id
        scored.sort(key=lambda x: (-x[0], x[1]))
        best_cos, best_src = scored[0]
        ties = [s for c, s in scored if abs(c - best_cos) <= 1e-12]
        top3 = scored[:3]
        rows.append(
            {
                "target_id": target,
                "target": DATASET_LABELS.get(target, target),
                "selected_source_id": best_src,
                "selected_source": DATASET_LABELS.get(best_src, best_src),
                "cosine": best_cos,
                "n_ties": len(ties),
                "tie_break": "max_cosine_then_lexicographic_source_id",
                "tied_source_ids": ";".join(ties),
                "top3_source_ids": ";".join(s for _, s in top3),
                "top3_cosines": ";".join(f"{c:.6f}" for c, _ in top3),
            }
        )
    return pd.DataFrame(rows)


def load_transfer_matrix(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, index_col=0)
    m.index = m.index.astype(str)
    m.columns = m.columns.astype(str)
    return m


def evaluate_6x6(
    ranks: pd.DataFrame,
    improve: pd.DataFrame,
    *,
    method: str,
    allowlist: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    detail_rows: List[Dict[str, Any]] = []
    improves: List[float] = []
    regrets: List[float] = []
    top1 = 0
    random_pos: List[float] = []

    for _, r in ranks.iterrows():
        target = str(r["target_id"])
        if target not in improve.index:
            continue
        sources = [
            s
            for s in allowlist
            if s != target and s in improve.columns and pd.notna(improve.loc[target, s])
        ]
        if not sources:
            continue
        benefit = improve.loc[target, sources].astype(float)
        selected = str(r["selected_source_id"])
        if selected not in benefit.index:
            continue
        sel_imp = float(benefit.loc[selected])
        oracle = benefit.idxmax()
        oracle_imp = float(benefit.max())
        hit = int(selected == oracle)
        regret = oracle_imp - sel_imp
        rank = int(benefit.rank(ascending=False, method="min").loc[selected])
        rand_p = float((benefit > 0).mean())

        improves.append(sel_imp)
        regrets.append(regret)
        top1 += hit
        random_pos.append(rand_p)

        detail_rows.append(
            {
                "method": method,
                "target_id": target,
                "target": DATASET_LABELS.get(target, target),
                "selected_source_id": selected,
                "selected_source": DATASET_LABELS.get(selected, selected),
                "cosine": float(r["cosine"]),
                "n_ties": int(r["n_ties"]),
                "tie_break": r["tie_break"],
                "top3_source_ids": r["top3_source_ids"],
                "improve_test_loglik": sel_imp,
                "positive": bool(sel_imp > 0),
                "nonnegative": bool(sel_imp >= 0),
                "rank_among_allowlist": rank,
                "n_allowlist_sources": len(sources),
                "oracle_source_id": oracle,
                "oracle_source": DATASET_LABELS.get(str(oracle), str(oracle)),
                "oracle_improve_test_loglik": oracle_imp,
                "top1_hit": hit,
                "regret_vs_oracle": regret,
                "random_source_positive_rate_for_target": rand_p,
            }
        )

    n = len(improves)
    arr = np.array(improves, dtype=float) if improves else np.array([])
    summary = {
        "method": method,
        "n_targets": n,
        "candidate_pool": "SIX_old_unaffected",
        "positive": float(np.mean(arr > 0)) if n else None,
        "nonnegative": float(np.mean(arr >= 0)) if n else None,
        "mean": float(np.mean(arr)) if n else None,
        "median": float(np.median(arr)) if n else None,
        "worst": float(np.min(arr)) if n else None,
        "top1": float(top1 / n) if n else None,
        "mean_regret": float(np.mean(regrets)) if regrets else None,
        "random_plus": float(np.mean(random_pos)) if random_pos else None,
        "random_plus_definition": (
            "mean over targets of fraction of allowlisted other sources "
            "with improve_test_loglik > 0"
        ),
    }
    return pd.DataFrame(detail_rows), summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    path: Path,
    *,
    panel_stats: Dict[str, Any],
    ann_stats: Dict[str, Any],
    occ_meta: Dict[str, Any],
    fit_support: Dict[str, Any],
    fit_meta: Dict[str, Any],
    comb_meta: Dict[str, Any],
    eval_summaries: Dict[str, Dict[str, Any]],
    frozen_spec: Dict[str, Any],
    transfer_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Source selection v4 (schema-v4 hierarchical MEM)")
    lines.append("")
    lines.append(
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "Three approaches freeze signatures **without** transfer outcomes, "
        "then evaluate on the old unaffected 6×6 "
        "`improve_test_loglik` table only."
    )
    lines.append("")
    lines.append(
        "- **Do not** use category-based T-PICS 6×6 results."
    )
    lines.append(
        "- Hypers / support gates / EB formulas are prespecified; "
        "transfer GT is loaded only at evaluation."
    )
    lines.append(
        f"- Transfer GT: `{transfer_path}`"
    )
    lines.append(
        "- Selection: argmax cosine among SIX allowlist; self excluded; "
        "lexicographic source-id tie-break."
    )
    lines.append("")
    lines.append("### SIX (old unaffected)")
    lines.append("")
    for ds in SIX:
        lines.append(f"- `{ds}` ({DATASET_LABELS.get(ds, ds)})")
    lines.append("")

    lines.append("## Data coverage")
    lines.append("")
    lines.append(
        f"- Annotation union keys: **{ann_stats.get('n_union_keys')}** "
        f"(by_dataset={ann_stats.get('n_by_dataset_keys')}, "
        f"shards={ann_stats.get('n_shard_keys')}, "
        f"shard-only={ann_stats.get('n_only_shards')})"
    )
    lines.append(
        f"- Analysis panel rows: **{panel_stats.get('n_rows')}** across "
        f"{panel_stats.get('n_datasets')} datasets"
    )
    lines.append(
        f"- Exact reference rows: {panel_stats.get('n_exact')}; "
        f"with recovered ΔF: {panel_stats.get('n_exact_with_delta_f')}"
    )
    lines.append("")
    lines.append("### Fitness ΔF score recovery")
    lines.append("")
    lines.append(str(panel_stats.get("score_recovery_note", "")))
    lines.append("")
    lines.append(
        "**Important:** fitness ΔF uses recovered elite/metrics scores "
        "(non-elite lack persisted scores)."
    )
    lines.append("")

    lines.append("## A) Hierarchical motif-occurrence MEM")
    lines.append("")
    lines.append(f"- Model: `{occ_meta.get('formula')}`")
    lines.append(f"- Smoothing: {occ_meta.get('smoothing')}")
    lines.append(f"- τ²: {occ_meta.get('tau2')}")
    lines.append(f"- Signature: {occ_meta.get('signature')}")
    lines.append("")

    lines.append("## B) Hierarchical fitness-effect MEM")
    lines.append("")
    lines.append(
        f"- Exact rows only; formula template: "
        f"`{fit_meta.get('formula_template')}`"
    )
    lines.append(
        f"- Support gates: min_positive_rows="
        f"{fit_support.get('min_positive_rows')}, "
        f"min_datasets_with_positive="
        f"{fit_support.get('min_datasets_with_positive')}, "
        f"min_datasets_with_both="
        f"{fit_support.get('min_datasets_with_both')}"
    )
    lines.append(
        f"- Supported transitions: "
        f"{fit_support.get('supported_transitions')}"
    )
    lines.append(
        f"- Signature transitions (fitted): "
        f"{fit_meta.get('signature_transitions')}"
    )
    lines.append(
        "- Sparse effects are **not** forced into the signature."
    )
    lines.append("")

    lines.append("## C) Combined")
    lines.append("")
    lines.append(f"- {comb_meta.get('weighting')}")
    lines.append(
        f"- Occurrence dims={comb_meta.get('n_occurrence_dims')}, "
        f"fitness dims={comb_meta.get('n_fitness_dims')}"
    )
    lines.append("")

    lines.append("## Frozen selection (no transfer peek)")
    lines.append("")
    lines.append(
        f"- FROZEN_SPEC methods: {list(frozen_spec.get('methods', {}).keys())}"
    )
    lines.append(
        f"- peeked_transfer_at_freeze: "
        f"**{frozen_spec.get('peeked_transfer_at_freeze')}**"
    )
    lines.append("")

    lines.append("## Eval on old unaffected 6×6")
    lines.append("")
    lines.append(
        "| Method | positive | nonnegative | mean | median | worst | "
        "top-1 | mean regret | random+ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("occurrence", "fitness", "combined"):
        s = eval_summaries.get(key, {})
        def _f(x: Any, nd: int = 4) -> str:
            if x is None:
                return "—"
            try:
                return f"{float(x):.{nd}f}"
            except (TypeError, ValueError):
                return "—"

        lines.append(
            f"| {key} | {_f(s.get('positive'))} | {_f(s.get('nonnegative'))} | "
            f"{_f(s.get('mean'))} | {_f(s.get('median'))} | {_f(s.get('worst'))} | "
            f"{_f(s.get('top1'))} | {_f(s.get('mean_regret'))} | "
            f"{_f(s.get('random_plus'))} |"
        )
    lines.append("")
    lines.append(
        "`random+` = mean over targets of the fraction of allowlisted other "
        "sources with improve_test_loglik > 0."
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("See this directory for `analysis_panel.csv`, occurrence_*, "
                 "fitness_*, `combined_signatures.csv`, `FROZEN_SPEC.json`, "
                 "frozen ranks, and eval tables.")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    repo = resolve_repo()
    ann_root = Path(args.ann_root)
    manifest_path = Path(args.manifest)
    cfg_path = Path(args.config)
    out_dir = Path(args.out_dir)
    transfer_path = Path(args.transfer_matrix)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] repo={repo}", flush=True)
    print(f"[INFO] ann_root={ann_root}", flush=True)
    print(f"[INFO] out_dir={out_dir}", flush=True)

    run_dirs = load_run_dirs(cfg_path, repo)
    print(f"[INFO] n_run_dirs={len(run_dirs)}", flush=True)

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_manifest = int(man.get("n_programs") or len(man.get("programs", [])))

    ann, ann_stats = load_annotations_union(ann_root, prefer_by_dataset=True)
    print(
        f"[INFO] annotations union={ann_stats['n_union_keys']} "
        f"(expect ~{n_manifest})",
        flush=True,
    )
    if ann_stats["n_union_keys"] != n_manifest:
        print(
            f"[WARN] union keys {ann_stats['n_union_keys']} != "
            f"manifest {n_manifest}",
            flush=True,
        )

    panel, panel_stats = build_analysis_panel(ann, run_dirs)
    panel_stats["n_manifest_programs"] = n_manifest
    panel_stats["ann_stats"] = ann_stats
    panel.to_csv(out_dir / "analysis_panel.csv", index=False)
    _write_json(out_dir / "panel_stats.json", panel_stats)
    print(
        f"[INFO] panel rows={len(panel)} exact_scored="
        f"{panel_stats['n_exact_with_delta_f']}",
        flush=True,
    )

    # ---- A) Occurrence (freeze; no transfer) ----
    occ_sig, occ_eb, occ_model, occ_meta = fit_occurrence_eb(panel)
    occ_sig.to_csv(out_dir / "occurrence_signatures.csv", index=False)
    occ_eb.to_csv(out_dir / "occurrence_dataset_eb.csv", index=False)
    occ_model.to_csv(out_dir / "occurrence_model_summary.csv", index=False)
    _write_json(out_dir / "occurrence_model_summary.json", {
        "meta": occ_meta,
        "rows": occ_model.to_dict(orient="records"),
    })
    _write_json(out_dir / "occurrence_signatures.json", {
        "meta": occ_meta,
        "signatures": occ_sig.to_dict(orient="records"),
    })

    all_datasets = sorted(panel["dataset"].unique().tolist())

    # ---- B) Fitness (freeze; no transfer) ----
    exact = panel[
        (panel["reference_is_exact"] == True)  # noqa: E712
        & panel["delta_f"].notna()
    ].copy()
    fit_support = fitness_support_audit(exact)
    _write_json(out_dir / "fitness_support.json", fit_support)
    supported = list(fit_support["supported_transitions"])
    print(
        f"[INFO] fitness exact_scored={len(exact)} "
        f"supported_transitions={supported}",
        flush=True,
    )
    fit_sig, fit_meta = fit_fitness_mixedlm(exact, supported, all_datasets)
    fit_sig.to_csv(out_dir / "fitness_signatures.csv", index=False)
    # Long slopes table
    fit_cols = [c for c in fit_sig.columns if c not in ("dataset", "dataset_label")]
    slope_rows: List[Dict[str, Any]] = []
    for _, row in fit_sig.iterrows():
        for T in fit_cols:
            slope_rows.append(
                {
                    "dataset": row["dataset"],
                    "dataset_label": row["dataset_label"],
                    "transition": T,
                    "theta_hat": row[T],
                    "mu_hat": fit_meta.get("fits", {}).get(T, {}).get("fixed_coef_mu"),
                }
            )
    pd.DataFrame(slope_rows).to_csv(out_dir / "fitness_slopes.csv", index=False)
    _write_json(out_dir / "fitness_model_fits.json", fit_meta)
    _write_json(out_dir / "fitness_signatures.json", {
        "meta": {
            k: v for k, v in fit_meta.items() if k != "fits"
        },
        "signatures": fit_sig.to_dict(orient="records"),
        "supported_transitions": supported,
        "signature_transitions": fit_cols,
    })

    # ---- C) Combined ----
    comb_sig, comb_meta = build_combined_signatures(occ_sig, fit_sig)
    comb_sig.to_csv(out_dir / "combined_signatures.csv", index=False)
    _write_json(out_dir / "combined_signatures.json", {
        "meta": comb_meta,
        "signatures": comb_sig.to_dict(orient="records"),
    })

    # ---- Freeze ranks (still no transfer) ----
    methods: Dict[str, Dict[str, Any]] = {}

    occ_vecs = signature_vectors(occ_sig, list(MOTIFS))
    methods["occurrence"] = {
        "vector_columns": list(MOTIFS),
        "signature_file": "occurrence_signatures.csv",
        "n_dims": len(MOTIFS),
    }

    if fit_cols:
        fit_vecs = signature_vectors(fit_sig, fit_cols)
    else:
        # Degenerate: zero vector per dataset (documented)
        fit_vecs = {ds: np.zeros(1, dtype=float) for ds in all_datasets}
        methods["fitness_note"] = "no_supported_fitted_transitions"
    methods["fitness"] = {
        "vector_columns": fit_cols,
        "signature_file": "fitness_signatures.csv",
        "n_dims": len(fit_cols),
    }

    comb_cols = [
        c
        for c in comb_sig.columns
        if c.startswith("occ_z__") or c.startswith("fit_z__")
    ]
    comb_vecs = signature_vectors(comb_sig, comb_cols)
    methods["combined"] = {
        "vector_columns": comb_cols,
        "signature_file": "combined_signatures.csv",
        "n_dims": len(comb_cols),
    }

    frozen_spec = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "peeked_transfer_at_freeze": False,
        "six_allowlist": list(SIX),
        "selection_rule": (
            "argmax cosine among SIX allowlist; self excluded; "
            "lexicographic source id tie-break"
        ),
        "motifs_v4": list(MOTIFS),
        "occurrence": occ_meta,
        "fitness_support_gates": {
            "min_positive_rows": MIN_POSITIVE_ROWS,
            "min_datasets_with_positive": MIN_DATASETS_WITH_POSITIVE,
            "min_datasets_with_both": MIN_DATASETS_WITH_BOTH,
        },
        "fitness_supported_transitions": supported,
        "fitness_signature_transitions": fit_cols,
        "combined": comb_meta,
        "methods": methods,
        "score_recovery_note": panel_stats["score_recovery_note"],
        "transfer_matrix_for_eval_only": str(transfer_path),
        "do_not_use_category_based_t_pics_6x6": True,
    }
    _write_json(out_dir / "FROZEN_SPEC.json", frozen_spec)

    # Ranks for 6×6 targets and all 15 targets (still no GT)
    method_vecs = {
        "occurrence": occ_vecs,
        "fitness": fit_vecs,
        "combined": comb_vecs,
    }
    for name, vecs in method_vecs.items():
        r6 = select_sources(vecs, targets=list(SIX), allowlist=list(SIX))
        r6.to_csv(out_dir / f"frozen_ranks_6x6_{name}.csv", index=False)
        r15 = select_sources(
            vecs, targets=all_datasets, allowlist=list(SIX)
        )
        r15.to_csv(out_dir / f"frozen_ranks_15targets_{name}.csv", index=False)

    # ---- Eval: load transfer GT only now ----
    print(f"[INFO] loading transfer GT for eval: {transfer_path}", flush=True)
    if not transfer_path.is_file():
        raise FileNotFoundError(f"Transfer matrix missing: {transfer_path}")
    improve = load_transfer_matrix(transfer_path)

    eval_summaries: Dict[str, Dict[str, Any]] = {}
    for name in ("occurrence", "fitness", "combined"):
        ranks = pd.read_csv(out_dir / f"frozen_ranks_6x6_{name}.csv")
        detail, summary = evaluate_6x6(
            ranks, improve, method=name, allowlist=list(SIX)
        )
        detail.to_csv(out_dir / f"eval_6x6_detail_{name}.csv", index=False)
        eval_summaries[name] = summary
        print(
            f"[INFO] eval {name}: positive={summary.get('positive')} "
            f"mean={summary.get('mean')} top1={summary.get('top1')}",
            flush=True,
        )

    _write_json(
        out_dir / "eval_6x6_summary.json",
        {
            "transfer_matrix": str(transfer_path),
            "six_allowlist": list(SIX),
            "methods": eval_summaries,
            "metrics": [
                "positive",
                "nonnegative",
                "mean",
                "median",
                "worst",
                "top1",
                "mean_regret",
                "random_plus",
            ],
        },
    )

    write_report(
        out_dir / "SOURCE_SELECTION_V4_REPORT.md",
        panel_stats=panel_stats,
        ann_stats=ann_stats,
        occ_meta=occ_meta,
        fit_support=fit_support,
        fit_meta=fit_meta,
        comb_meta=comb_meta,
        eval_summaries=eval_summaries,
        frozen_spec=frozen_spec,
        transfer_path=transfer_path,
    )
    print(f"[INFO] done → {out_dir}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ann_root", type=Path, default=DEFAULT_ANN_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--transfer_matrix", type=Path, default=DEFAULT_TRANSFER)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
