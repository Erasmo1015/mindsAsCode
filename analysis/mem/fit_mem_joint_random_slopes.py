#!/usr/bin/env python3
"""Fit joint multi-motif MixedLM with participant random slopes.

Shared fixed-effect design is **not** auto-selected for Schema v5. Pass an
explicit formula / fixed list after running coverage_eligibility_v5.py.
The legacy default below is schema-v3 era (includes risk_*) and must not be
assumed for participant_transition_v5:

  delta_f ~ history_added + value_added + risk_added + learning_added
          + history_modified + value_modified + risk_modified + learning_modified
          + iteration

Random effects are a subset of those motif columns (plus intercept), chosen via
``--random_slopes``. Primary optimizer is REML + lbfgs (matches focal pipeline);
optional secondary methods (cg, powell) are also fit and stored.

Eligibility (schema_v3/v5 CSV):
  --eligibility_mode off        use all rows; legacy v2 OK
  --eligibility_mode fe_adjust  exploratory: add eligible_* as FE covariates
                                (does **not** put each effect on its risk set)
  --eligibility_mode restrict   **validated joint construction**: keep only
                                rows in the intersection of the appropriate
                                risk set for every included motif effect.
                                Reports exact eligibility rules in outputs.
                                For ``*_modified``, risk set =
                                retained-construct set
                                (reference_has AND candidate_has);
                                contrast is modified vs retained_unmodified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.mem.bh_fdr import bh_fdr  # noqa: E402
from utils.mem.schema_participant_transition_v5 import (  # noqa: E402
    risk_set_definition,
)


DEFAULT_FIXED_EFFECTS = [
    "history_added",
    "value_added",
    "risk_added",
    "learning_added",
    "history_modified",
    "value_modified",
    "risk_modified",
    "learning_modified",
    "iteration",
]


def _require_statsmodels():
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise SystemExit(
            "statsmodels is required. Install with: pip install statsmodels"
        ) from exc
    return smf


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _motif_effects(fixed_effects: Sequence[str], random_slopes: Sequence[str]) -> List[str]:
    """Directional construct effects included in the joint design (excl. iteration)."""
    out: List[str] = []
    seen = set()
    for t in list(fixed_effects) + list(random_slopes):
        if t == "iteration" or str(t).startswith("eligible_"):
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def apply_joint_eligibility_restrict(
    df: pd.DataFrame,
    *,
    motif_effects: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Intersect appropriate risk sets for every included motif effect.

    Row kept iff eligible_<effect>==1 for all effects in ``motif_effects``.
    """
    if not motif_effects:
        raise ValueError("motif_effects must be non-empty for eligibility_mode=restrict")

    definitions: List[Dict[str, str]] = []
    missing: List[str] = []
    per_effect: Dict[str, Any] = {}
    mask = pd.Series(True, index=df.index)
    n_before = int(len(df))

    for effect in motif_effects:
        try:
            defn = risk_set_definition(effect)
        except ValueError:
            # Non-directional FE (e.g. structural_*): skip risk-set gate.
            definitions.append(
                {
                    "effect": effect,
                    "risk_set_name": "none_non_directional",
                    "rule": "not gated by construct eligibility",
                }
            )
            continue
        elig_col = defn["eligibility_column"]
        definitions.append(defn)
        if elig_col not in df.columns:
            missing.append(elig_col)
            continue
        if df[elig_col].isna().any():
            raise ValueError(
                f"eligibility_mode=restrict: {elig_col} has nulls; "
                "missing state ≠ ineligible"
            )
        elig = pd.to_numeric(df[elig_col], errors="coerce").fillna(0).astype(int)
        n_elig = int((elig == 1).sum())
        per_effect[effect] = {
            **defn,
            "n_rows_in_risk_set": n_elig,
            "n_rows_out_of_risk_set": int(n_before - n_elig),
        }
        mask &= elig == 1

    if missing:
        raise ValueError(
            "eligibility_mode=restrict requires eligibility columns for every "
            f"included motif effect; missing: {missing}"
        )

    work = df.loc[mask].copy()
    report: Dict[str, Any] = {
        "mode": "restrict",
        "construction": (
            "intersection: row kept iff it lies in the appropriate risk set "
            "for every included motif effect (eligible_<effect>==1 for all)"
        ),
        "n_rows_before": n_before,
        "n_rows_after": int(len(work)),
        "n_rows_excluded": int(n_before - len(work)),
        "included_effects": list(motif_effects),
        "risk_set_definitions": definitions,
        "per_effect": per_effect,
        "modified_effects_note": (
            "For *_modified effects, eligible_c_modified is the "
            "retained-construct modification risk set "
            "(reference_has_c AND candidate_has_c), not the broader "
            "pre-transition opportunity reference_has_c / "
            "modification_opportunity_c. Within that set the contrast is "
            "c_modified vs retained_unmodified_c."
        ),
    }
    return work, report


def prepare_joint_frame(
    df: pd.DataFrame,
    *,
    fixed_effects: Sequence[str],
    random_slopes: Sequence[str],
    phase: Optional[str] = "evolution",
    eligibility_report: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Coerce motif columns, drop incomplete rows, return fingerprint meta."""
    work = df.copy()
    if phase and "phase" in work.columns:
        work = work[work["phase"] == phase].copy()

    needed = set(fixed_effects) | set(random_slopes) | {
        "delta_f",
        "participant_id",
        "iteration",
    }
    missing = [c for c in needed if c not in work.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    work["delta_f"] = pd.to_numeric(work["delta_f"], errors="coerce")
    work["iteration"] = pd.to_numeric(work["iteration"], errors="coerce")
    work["participant_id"] = work["participant_id"].astype(str)

    motif_cols = [
        c
        for c in list(dict.fromkeys(list(fixed_effects) + list(random_slopes)))
        if c != "iteration"
    ]
    for c in motif_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0).astype(int)
        # Reject constant motif used as RE slope
        if c in random_slopes and work[c].nunique(dropna=True) < 2:
            raise ValueError(f"random slope '{c}' is constant after prep")

    before = len(work)
    work = work.dropna(subset=["delta_f", "participant_id", "iteration"]).copy()
    work = work.reset_index(drop=True)

    # Fingerprint of analysis rows (sorted index hash of original indices if available)
    idx_bytes = ",".join(map(str, work.index.tolist())).encode("utf-8")
    # Prefer original CSV row order after dropna: use stable content hash
    content_cols = ["participant_id", "delta_f", "iteration"] + motif_cols
    content = work[content_cols].to_csv(index=False).encode("utf-8")
    fingerprint = hashlib.sha256(content).hexdigest()

    support = {}
    for c in motif_cols:
        pos = work[c] == 1
        support[c] = {
            "n_positive_rows": int(pos.sum()),
            "n_participants_with_pos": int(work.loc[pos, "participant_id"].nunique()),
            "n_participants_both_levels": int(
                work.groupby("participant_id")[c]
                .apply(lambda s: s.nunique() >= 2)
                .sum()
            ),
        }

    meta = {
        "n_rows_before_dropna": int(before),
        "n_rows": int(len(work)),
        "n_participants": int(work["participant_id"].nunique()),
        "phase": phase,
        "fingerprint_sha256": fingerprint,
        "motif_support": support,
        "motif_columns": motif_cols,
        "eligibility": eligibility_report,
    }
    return work, meta


def _re_labels(random_slopes: Sequence[str]) -> List[str]:
    return ["Group"] + list(random_slopes)


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, np.nan)
    np.fill_diagonal(corr, 1.0)
    return corr


def _extract_random_effects(
    result,
    *,
    random_slopes: Sequence[str],
    fe_params: pd.Series,
) -> pd.DataFrame:
    """Participant BLUPs: b_hat and theta = mu + b for each slope (+ intercept)."""
    labels = _re_labels(random_slopes)
    rows: List[Dict[str, Any]] = []
    re_df = result.random_effects
    for pid, re_vec in re_df.items():
        if hasattr(re_vec, "index"):
            # Series keyed by RE names
            vals = {str(k): float(re_vec[k]) for k in re_vec.index}
            # Normalize Group / Intercept naming
            if "Group" not in vals:
                for alt in ("Intercept", "intercept", labels[0]):
                    if alt in vals:
                        vals["Group"] = vals[alt]
                        break
            ordered = []
            for lab in labels:
                if lab in vals:
                    ordered.append(vals[lab])
                elif lab != "Group" and lab in vals:
                    ordered.append(vals[lab])
                else:
                    # positional fallback
                    arr = np.asarray(re_vec, dtype=float).ravel()
                    ordered = list(arr)
                    break
            else:
                arr = np.asarray(ordered, dtype=float)
        else:
            arr = np.asarray(re_vec, dtype=float).ravel()

        if arr.size < len(labels):
            arr = np.pad(arr, (0, len(labels) - arr.size))
        row: Dict[str, Any] = {"participant_id": pid}
        # intercept
        b0 = float(arr[0])
        mu0 = float(fe_params.get("Intercept", 0.0))
        row["b_hat_Intercept"] = b0
        row["theta_Intercept"] = mu0 + b0
        for j, slope in enumerate(random_slopes, start=1):
            b = float(arr[j]) if j < arr.size else 0.0
            mu = float(fe_params.get(slope, float("nan")))
            row[f"b_hat_{slope}"] = b
            row[f"theta_{slope}"] = mu + b
        rows.append(row)
    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)


def _try_hessian(result) -> Dict[str, Any]:
    out: Dict[str, Any] = {"hessian_status": "unavailable"}
    try:
        hess = None
        if hasattr(result, "model") and hasattr(result.model, "hessian"):
            try:
                hess = result.model.hessian(result.params)
            except Exception as exc:
                out["hessian_status"] = "unavailable"
                out["hessian_error"] = f"{type(exc).__name__}: {exc}"
                return out
        if hess is None:
            hess = getattr(result, "hess", None) or getattr(result, "hessian", None)
        if hess is None:
            return out
        arr = np.asarray(hess, dtype=float)
        out["shape"] = list(arr.shape)
        try:
            eigs = np.linalg.eigvalsh(0.5 * (arr + arr.T))
            out["min_eig"] = float(np.min(eigs))
            out["max_eig"] = float(np.max(eigs))
            out["hessian_status"] = "ok"
        except Exception as exc:
            out["hessian_status"] = "unavailable"
            out["eig_error"] = str(exc)
    except Exception as exc:
        out["hessian_status"] = "unavailable"
        out["error"] = str(exc)
    return out


def _fit_one_method(
    model,
    *,
    method: str,
    reml: bool,
    maxiter: int,
    start_params=None,
) -> Tuple[Any, Dict[str, Any]]:
    warnings_list: List[str] = []
    fit_kwargs: Dict[str, Any] = {
        "method": method,
        "reml": reml,
        "maxiter": maxiter,
    }
    if start_params is not None:
        fit_kwargs["start_params"] = start_params
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = model.fit(**fit_kwargs)
        except Exception as exc:
            return None, {
                "method": method,
                "status": "fit_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "warnings": [str(w.message) for w in caught],
            }
        warnings_list = [str(w.message) for w in caught]
    return result, {
        "method": method,
        "status": "ok" if result.converged else "not_converged",
        "converged": bool(result.converged),
        "llf": float(result.llf) if result.llf is not None else None,
        "warnings": warnings_list,
    }


def fit_joint_random_slopes(
    df: pd.DataFrame,
    *,
    random_slopes: Sequence[str],
    fixed_effects: Sequence[str],
    methods: Sequence[str] = ("lbfgs", "cg", "powell"),
    primary_method: str = "lbfgs",
    reml: bool = True,
    maxiter: int = 200,
    phase: Optional[str] = "evolution",
    eligibility_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    smf = _require_statsmodels()
    work, prep_meta = prepare_joint_frame(
        df,
        fixed_effects=fixed_effects,
        random_slopes=random_slopes,
        phase=phase,
        eligibility_report=eligibility_report,
    )

    # Every RE slope must appear as FE
    for s in random_slopes:
        if s not in fixed_effects:
            raise ValueError(f"random slope '{s}' missing from fixed_effects")

    fe_terms = " + ".join(fixed_effects)
    formula = f"delta_f ~ {fe_terms}"
    re_formula = "1 + " + " + ".join(random_slopes)
    re_names = _re_labels(random_slopes)

    model = smf.mixedlm(
        formula, work, groups=work["participant_id"], re_formula=re_formula
    )

    method_results: Dict[str, Any] = {}
    primary = None
    primary_meta: Dict[str, Any] = {}
    start = None

    # Fit primary first to seed others when possible
    ordered_methods = [primary_method] + [
        m for m in methods if m != primary_method
    ]
    for method in ordered_methods:
        result, meta = _fit_one_method(
            model,
            method=method,
            reml=reml,
            maxiter=maxiter,
            start_params=start if method != primary_method else None,
        )
        method_results[method] = {"fit_meta": meta}
        if result is None:
            continue
        if start is None:
            try:
                start = np.asarray(result.params, dtype=float)
            except Exception:
                start = None
        cov_m = result.cov_re
        cov_arr = np.asarray(
            cov_m.values if hasattr(cov_m, "values") else cov_m, dtype=float
        )
        method_results[method].update(
            {
                "fe_params": {str(k): float(v) for k, v in result.fe_params.items()},
                "llf": float(result.llf) if result.llf is not None else None,
                "converged": bool(result.converged),
                "cov_re_diag": [float(x) for x in np.diag(cov_arr)],
                "scale": float(result.scale) if result.scale is not None else None,
            }
        )
        if method == primary_method:
            primary = result
            primary_meta = meta

    if primary is None:
        return {
            "status": "fit_failed",
            "formula": formula,
            "re_formula": re_formula,
            "prep": prep_meta,
            "method_results": method_results,
            "primary_method": primary_method,
        }

    # Covariance / correlation of RE
    cov = primary.cov_re
    if hasattr(cov, "values"):
        cov_mat = np.asarray(cov.values, dtype=float)
        cov_index = [str(x) for x in cov.index]
    else:
        cov_mat = np.asarray(cov, dtype=float)
        cov_index = list(re_names)

    # Align labels if needed
    if len(cov_index) != cov_mat.shape[0]:
        cov_index = list(re_names)[: cov_mat.shape[0]]

    corr_mat = _cov_to_corr(cov_mat)
    try:
        eigs = np.linalg.eigvalsh(0.5 * (cov_mat + cov_mat.T))
        min_eig = float(np.min(eigs))
        max_eig = float(np.max(eigs))
        cond = float(max_eig / min_eig) if min_eig > 0 else float("inf")
    except Exception:
        eigs = np.array([])
        min_eig = max_eig = cond = float("nan")

    hess_diag = _try_hessian(primary)

    # FE table with BH on motif FE terms only (ordered random_slopes list, not iteration)
    fe_params = primary.fe_params
    bse = primary.bse_fe
    pvalues = primary.pvalues
    conf = primary.conf_int()
    fe_rows = []
    motif_p: List[Optional[float]] = []
    motif_idx: List[int] = []
    for name in fe_params.index:
        coef = float(fe_params[name])
        se = float(bse[name]) if name in bse.index else float("nan")
        p = float(pvalues[name]) if name in pvalues.index else float("nan")
        ci_low = ci_high = None
        try:
            if name in conf.index:
                ci_low = float(conf.loc[name, 0])
                ci_high = float(conf.loc[name, 1])
        except Exception:
            pass
        row = {
            "term": str(name),
            "coef": coef,
            "se": se,
            "pvalue_raw": p if math.isfinite(p) else None,
            "qvalue_bh": None,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "in_bh_family": str(name) in set(random_slopes),
        }
        if row["in_bh_family"]:
            motif_idx.append(len(fe_rows))
            motif_p.append(row["pvalue_raw"])
        fe_rows.append(row)

    qvals = bh_fdr(motif_p)
    for i, q in zip(motif_idx, qvals):
        fe_rows[i]["qvalue_bh"] = q

    # Participant effects
    part_df = _extract_random_effects(
        primary, random_slopes=random_slopes, fe_params=fe_params
    )

    # Pairwise both-levels support among RE slopes
    pairwise = {}
    for i, a in enumerate(random_slopes):
        for b in random_slopes[i + 1 :]:
            both = 0
            for pid, g in work.groupby("participant_id"):
                if g[a].nunique() >= 2 and g[b].nunique() >= 2:
                    both += 1
            pairwise[f"{a}__{b}"] = both

    scale = float(primary.scale) if primary.scale is not None else None
    singular = bool(
        (math.isfinite(min_eig) and min_eig < 1e-10)
        or any(float(cov_mat[i, i]) < 1e-10 for i in range(cov_mat.shape[0]))
    )

    return {
        "status": primary_meta.get("status", "ok"),
        "formula": formula,
        "re_formula": re_formula,
        "re_labels": cov_index,
        "random_slopes": list(random_slopes),
        "fixed_effects": list(fixed_effects),
        "prep": prep_meta,
        "primary_method": primary_method,
        "reml": reml,
        "maxiter": maxiter,
        "converged": bool(primary.converged),
        "singular_or_boundary": singular,
        "llf": float(primary.llf) if primary.llf is not None else None,
        "scale": scale,
        "cov_re": cov_mat.tolist(),
        "corr_re": corr_mat.tolist(),
        "eigenvalues": [float(x) for x in eigs.tolist()] if eigs.size else [],
        "min_eig": min_eig,
        "max_eig": max_eig,
        "cond_number": cond,
        "hessian": hess_diag,
        "fixed_effects_table": fe_rows,
        "participant_effects": part_df,
        "motif_support": prep_meta["motif_support"],
        "pairwise_both_levels": pairwise,
        "method_results": method_results,
    }


def write_heatmaps(
    cov: np.ndarray,
    corr: np.ndarray,
    labels: Sequence[str],
    out_dir: Path,
) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}
    for name, mat, cmap, vmin, vmax in [
        ("cov_re_heatmap.png", cov, "viridis", None, None),
        ("corr_re_heatmap.png", corr, "coolwarm", -1.0, 1.0),
    ]:
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(labels) + 2), max(5, 0.7 * len(labels) + 2)))
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(name.replace("_", " ").replace(".png", ""))
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[name] = str(path)
    return paths


def write_outputs(fit: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist failure diagnostics without requiring a successful MixedLM extract.
    if fit.get("status") == "fit_failed" or "participant_effects" not in fit:
        (out_dir / "fit_summary.json").write_text(
            json.dumps(fit, indent=2, default=str), encoding="utf-8"
        )
        (out_dir / "summary.md").write_text(
            "# Joint random-slope fit\n\n"
            f"- status: `{fit.get('status')}`\n"
            f"- formula: `{fit.get('formula')}`\n"
            f"- re_formula: `{fit.get('re_formula')}`\n"
            f"- method_results: see `fit_summary.json`\n",
            encoding="utf-8",
        )
        return

    part = fit["participant_effects"]
    if not isinstance(part, pd.DataFrame):
        part = pd.DataFrame(part)

    part.to_csv(out_dir / "participant_effects.csv", index=False)
    # also b_hat / theta wide already; write slope-only long form
    long_rows = []
    for _, r in part.iterrows():
        for s in fit["random_slopes"]:
            long_rows.append(
                {
                    "participant_id": r["participant_id"],
                    "slope": s,
                    "b_hat": r.get(f"b_hat_{s}"),
                    "theta": r.get(f"theta_{s}"),
                }
            )
    pd.DataFrame(long_rows).to_csv(out_dir / "participant_thetas_long.csv", index=False)

    pd.DataFrame(fit["fixed_effects_table"]).to_csv(
        out_dir / "fixed_effects.csv", index=False
    )
    cov = np.asarray(fit["cov_re"], dtype=float)
    corr = np.asarray(fit["corr_re"], dtype=float)
    labels = fit["re_labels"]
    pd.DataFrame(cov, index=labels, columns=labels).to_csv(out_dir / "cov_re.csv")
    pd.DataFrame(corr, index=labels, columns=labels).to_csv(out_dir / "corr_re.csv")

    heat = write_heatmaps(cov, corr, labels, out_dir)

    # Support tables
    support_rows = [
        {"column": k, **v} for k, v in (fit.get("motif_support") or {}).items()
    ]
    pd.DataFrame(support_rows).to_csv(out_dir / "motif_support.csv", index=False)
    pd.DataFrame(
        [{"pair": k, "n_participants_both_levels": v} for k, v in fit.get("pairwise_both_levels", {}).items()]
    ).to_csv(out_dir / "pairwise_both_levels.csv", index=False)

    diag = {
        k: fit[k]
        for k in [
            "status",
            "formula",
            "re_formula",
            "re_labels",
            "random_slopes",
            "fixed_effects",
            "eligibility_mode",
            "eligibility_report",
            "prep",
            "primary_method",
            "reml",
            "maxiter",
            "converged",
            "singular_or_boundary",
            "llf",
            "scale",
            "cov_re",
            "corr_re",
            "eigenvalues",
            "min_eig",
            "max_eig",
            "cond_number",
            "hessian",
            "fixed_effects_table",
            "motif_support",
            "pairwise_both_levels",
            "method_results",
        ]
        if k in fit
    }
    diag["heatmaps"] = heat
    # participant_effects is a DataFrame — omit from JSON (on disk as CSV)
    (out_dir / "fit_diagnostics.json").write_text(
        json.dumps(diag, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Markdown summary
    lines = [
        f"# Joint random-slope fit",
        "",
        f"- status: `{fit.get('status')}`",
        f"- formula: `{fit.get('formula')}`",
        f"- re_formula: `{fit.get('re_formula')}`",
        f"- eligibility_mode: `{fit.get('eligibility_mode')}`",
        f"- n_rows: {fit.get('prep', {}).get('n_rows')}",
        f"- fingerprint: `{fit.get('prep', {}).get('fingerprint_sha256')}`",
        f"- primary: `{fit.get('primary_method')}` reml={fit.get('reml')} maxiter={fit.get('maxiter')}",
        f"- converged: {fit.get('converged')}",
        f"- llf: {fit.get('llf')}",
        f"- scale: {fit.get('scale')}",
        f"- min_eig(Σ): {fit.get('min_eig')}",
        f"- cond(Σ): {fit.get('cond_number')}",
        f"- singular_or_boundary: {fit.get('singular_or_boundary')}",
        f"- hessian: {fit.get('hessian', {}).get('hessian_status')}",
        "",
        "## Eligibility construction",
        "",
    ]
    elig = fit.get("eligibility_report") or (fit.get("prep") or {}).get("eligibility")
    if elig:
        lines.append(f"- mode: `{elig.get('mode')}`")
        lines.append(f"- construction: {elig.get('construction')}")
        if elig.get("n_rows_before") is not None:
            lines.append(
                f"- rows: {elig.get('n_rows_before')} → {elig.get('n_rows_after')} "
                f"(excluded {elig.get('n_rows_excluded')})"
            )
        if elig.get("modified_effects_note"):
            lines.append(f"- modified note: {elig.get('modified_effects_note')}")
        lines.append("")
        lines.append("| effect | risk set | rule | n_in_set |")
        lines.append("|---|---|---|---:|")
        for effect, info in (elig.get("per_effect") or {}).items():
            lines.append(
                f"| {effect} | {info.get('risk_set_name')} | `{info.get('rule')}` | "
                f"{info.get('n_rows_in_risk_set')} |"
            )
        lines.append("")
    else:
        lines.append(
            "- mode: `off` or `fe_adjust` (no risk-set intersection; "
            "fe_adjust alone is not validated joint eligibility)."
        )
        lines.append("")

    lines += [
        "## Fixed effects (motif BH family)",
        "",
        "| term | coef | se | p | q_BH |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in fit.get("fixed_effects_table") or []:
        if not r.get("in_bh_family"):
            continue
        lines.append(
            f"| {r['term']} | {r['coef']:.6g} | {r['se']:.6g} | "
            f"{r['pvalue_raw']} | {r['qvalue_bh']} |"
        )
    lines += ["", "## Optimizer comparison", ""]
    for method, bundle in (fit.get("method_results") or {}).items():
        meta = bundle.get("fit_meta") or {}
        lines.append(
            f"- `{method}`: status={meta.get('status')} converged={bundle.get('converged', meta.get('converged'))} "
            f"llf={bundle.get('llf', meta.get('llf'))}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--random_slopes",
        required=True,
        help="Comma-separated motif columns in the RE formula (after intercept).",
    )
    parser.add_argument(
        "--fixed_effects",
        default=",".join(DEFAULT_FIXED_EFFECTS),
        help="Comma-separated FE terms (must include all random slopes + iteration).",
    )
    parser.add_argument("--methods", default="lbfgs,cg,powell")
    parser.add_argument("--primary_method", default="lbfgs")
    parser.add_argument("--reml", action="store_true", default=True)
    parser.add_argument("--no_reml", action="store_true")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--phase", default="evolution")
    parser.add_argument(
        "--assert_n_rows",
        type=int,
        default=0,
        help="If >0, assert prepared n_rows equals this value.",
    )
    parser.add_argument(
        "--eligibility_mode",
        choices=["off", "fe_adjust", "restrict"],
        default="off",
        help="off: all rows. fe_adjust: exploratory eligible_* FE covariates "
        "(insufficient alone). restrict: intersect appropriate risk sets for "
        "every included motif effect; report construction in summary.",
    )
    args = parser.parse_args()

    reml = False if args.no_reml else True
    random_slopes = _parse_csv_list(args.random_slopes)
    fixed_effects = _parse_csv_list(args.fixed_effects)
    methods = _parse_csv_list(args.methods)
    eligibility_mode = str(args.eligibility_mode)

    df = pd.read_csv(args.input_csv)
    eligibility_report: Optional[Dict[str, Any]] = None

    # Apply phase filter before eligibility so risk-set counts match the analysis frame.
    if args.phase and "phase" in df.columns:
        df = df[df["phase"] == args.phase].copy()

    if eligibility_mode == "restrict":
        motif_fes = _motif_effects(fixed_effects, random_slopes)
        try:
            df, eligibility_report = apply_joint_eligibility_restrict(
                df, motif_effects=motif_fes
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            f"[fit_joint] eligibility_mode=restrict "
            f"n_rows={eligibility_report['n_rows_after']} "
            f"(from {eligibility_report['n_rows_before']}); "
            f"effects={motif_fes}",
            flush=True,
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "eligibility_construction.json").write_text(
            json.dumps(eligibility_report, indent=2) + "\n", encoding="utf-8"
        )

    if eligibility_mode == "fe_adjust":
        motif_fes = [
            t
            for t in fixed_effects
            if t != "iteration" and not str(t).startswith("eligible_")
        ]
        missing = []
        elig_terms: List[str] = []
        for t in motif_fes:
            elig = f"eligible_{t}"
            if elig not in df.columns:
                missing.append(elig)
            else:
                elig_terms.append(elig)
        if missing:
            raise SystemExit(
                "eligibility_mode=fe_adjust requires schema_v3 eligibility columns; "
                f"missing: {missing}. Legacy v2 CSVs refused. "
                "Note: FE adjustment does not intersect eligibility sets and is "
                "not a validated joint eligibility model."
            )
        if any(df[c].isna().any() for c in elig_terms):
            raise SystemExit(
                "eligibility_mode=fe_adjust: nulls in eligible_* columns; "
                "refusing (missing state ≠ zero)."
            )
        # Drop weak / near-constant motif FEs (except required random slopes) and
        # near-constant eligible_* covariates. nunique>=2 alone is not enough:
        # e.g. value_added with 2 positives still singularizes the MixedLM design.
        phase = args.phase or None
        work = df
        if phase and "phase" in df.columns and (df["phase"] == phase).any():
            work = df[df["phase"] == phase]
        min_pos = 5
        min_both = 2
        keep_fe: List[str] = []
        for t in fixed_effects:
            if t == "iteration" or t in random_slopes:
                keep_fe.append(t)
                continue
            if t not in work.columns:
                continue
            s = pd.to_numeric(work[t], errors="coerce").fillna(0)
            n_pos = int((s == 1).sum())
            n_both = int(
                work.assign(_x=s)
                .groupby("participant_id")["_x"]
                .nunique()
                .ge(2)
                .sum()
            )
            if s.nunique(dropna=True) < 2 or n_pos < min_pos or n_both < min_both:
                print(
                    f"[fit_joint] drop weak FE {t} (n_pos={n_pos}, n_both={n_both})",
                    flush=True,
                )
                continue
            keep_fe.append(t)
        fixed_effects = list(dict.fromkeys(keep_fe))
        for e in elig_terms:
            if e in fixed_effects:
                continue
            # Only adjust eligibility for motifs that remain in the FE design.
            motif = e[len("eligible_") :]
            if motif not in fixed_effects:
                print(
                    f"[fit_joint] skip eligibility FE {e} (motif not in FE)",
                    flush=True,
                )
                continue
            s = pd.to_numeric(work[e], errors="coerce").fillna(0)
            n0 = int((s == 0).sum())
            n1 = int((s == 1).sum())
            if s.nunique(dropna=True) < 2 or min(n0, n1) < min_pos:
                print(
                    f"[fit_joint] drop weak eligibility FE {e} (n0={n0}, n1={n1})",
                    flush=True,
                )
                continue
            fixed_effects.append(e)
        print(
            f"[fit_joint] fe_adjust fixed_effects={fixed_effects}",
            flush=True,
        )

    fit = fit_joint_random_slopes(
        df,
        random_slopes=random_slopes,
        fixed_effects=fixed_effects,
        methods=methods,
        primary_method=args.primary_method,
        reml=reml,
        maxiter=args.maxiter,
        phase=None,
        eligibility_report=eligibility_report,
    )
    fit["eligibility_mode"] = eligibility_mode
    fit["eligibility_report"] = eligibility_report
    if args.assert_n_rows and fit.get("prep", {}).get("n_rows") != args.assert_n_rows:
        raise SystemExit(
            f"n_rows={fit.get('prep', {}).get('n_rows')} != assert {args.assert_n_rows}"
        )

    write_outputs(fit, Path(args.output_dir))
    print(f"[fit_joint] Wrote {args.output_dir}", flush=True)
    print(
        json.dumps(
            {
                "status": fit.get("status"),
                "eligibility_mode": eligibility_mode,
                "n_rows": fit.get("prep", {}).get("n_rows"),
                "fingerprint": fit.get("prep", {}).get("fingerprint_sha256"),
                "llf": fit.get("llf"),
                "min_eig": fit.get("min_eig"),
                "cond": fit.get("cond_number"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
