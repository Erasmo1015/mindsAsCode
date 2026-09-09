#!/usr/bin/env python3
"""Fit one-focal-motif MixedLM models with participant random slopes.

For each focal directional motif M:

  delta_f ~ M + other_supported_controls + iteration + (1 + M | participant_id)

Fits added / removed / modified separately (never averages opposite directions).
CPU-only statsmodels MixedLM. Reports fixed effects, random-slope variance,
participant-specific slopes, convergence / singularity diagnostics, and BH-FDR
across planned motif tests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.mem.predictor_support import motif_support_report  # noqa: E402
from utils.mem.schema_v2 import (  # noqa: E402
    DIRECTIONAL_SUFFIXES,
    all_directional_behavioral_columns,
)


def _require_statsmodels():
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise SystemExit(
            "statsmodels is required. Install with: pip install statsmodels"
        ) from exc
    return smf


def _bh_fdr(pvals: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini–Hochberg q-values; None stays None."""
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None and math.isfinite(p)]
    m = len(indexed)
    out: List[Optional[float]] = [None] * len(pvals)
    if m == 0:
        return out
    indexed.sort(key=lambda t: t[1])
    prev = 1.0
    ranks = {}
    for rank, (i, p) in enumerate(indexed, start=1):
        q = min(prev, p * m / rank)
        prev = q
        ranks[i] = q
    # Enforce monotonicity from largest p upward
    running = 1.0
    for i, p in reversed(indexed):
        running = min(running, ranks[i])
        out[i] = running
    return out


def _participant_slopes(
    result,
    *,
    focal: str,
    fixed_coef: float,
) -> Tuple[List[Dict[str, Any]], Optional[float], int]:
    """Extract participant-specific slopes = fixed + random deviation for focal."""
    slopes: List[Dict[str, Any]] = []
    var_slope: Optional[float] = None
    n_informative = 0
    try:
        re_df = result.random_effects
        cov = result.cov_re
        # MixedLM cov_re is VarianceComponents-like; try matrix form.
        if hasattr(cov, "iloc"):
            # pandas DataFrame indexed by effect names
            cols = list(cov.columns)
            if focal in cols:
                var_slope = float(cov.loc[focal, focal])
            elif len(cols) >= 2:
                var_slope = float(cov.iloc[1, 1])
        elif hasattr(cov, "shape"):
            import numpy as np

            arr = np.asarray(cov, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                var_slope = float(arr[1, 1])
            elif arr.ndim == 2 and arr.shape[0] == 1:
                var_slope = None
        for pid, re_vec in re_df.items():
            # re_vec may be Series with Group/focal keys or ndarray
            if hasattr(re_vec, "get"):
                dev = re_vec.get(focal)
                if dev is None and len(re_vec) >= 2:
                    # second entry typically slope when formula is 1 + M
                    try:
                        dev = float(list(re_vec.values)[1])
                    except Exception:
                        dev = None
                elif dev is None and len(re_vec) == 1:
                    # intercept-only fallback
                    dev = 0.0
                else:
                    dev = float(dev) if dev is not None else None
            else:
                import numpy as np

                arr = np.asarray(re_vec, dtype=float).ravel()
                dev = float(arr[1]) if arr.size >= 2 else 0.0
            if dev is None:
                continue
            slope = float(fixed_coef) + float(dev)
            slopes.append(
                {
                    "participant_id": pid,
                    "random_deviation": float(dev),
                    "participant_slope": slope,
                }
            )
            if abs(float(dev)) > 1e-12:
                n_informative += 1
    except Exception as exc:
        return [], None, 0
    return slopes, var_slope, n_informative


def fit_focal_motif(
    df: pd.DataFrame,
    *,
    focal: str,
    controls: Sequence[str],
    include_phase_effects: bool,
    structural_controls: Sequence[str],
) -> Dict[str, Any]:
    smf = _require_statsmodels()
    work = df.copy()
    work["delta_f"] = pd.to_numeric(work["delta_f"], errors="coerce")
    work["iteration"] = pd.to_numeric(work.get("iteration"), errors="coerce")
    work["participant_id"] = work["participant_id"].astype(str)
    work[focal] = pd.to_numeric(work[focal], errors="coerce").fillna(0).astype(int)
    for c in list(controls) + list(structural_controls):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0).astype(int)

    # Drop rows missing outcome / grouping
    work = work.dropna(subset=["delta_f", "participant_id", "iteration"])
    n_pos = int((work[focal] == 1).sum())
    n_parts = int(work.loc[work[focal] == 1, "participant_id"].nunique())
    if n_pos < 5 or n_parts < 2:
        return {
            "focal": focal,
            "status": "skipped_unsupported",
            "reason": "too_few_positive_rows_or_participants",
            "n_positive": n_pos,
            "n_participants_with_positive": n_parts,
            "n_rows": int(len(work)),
        }

    fixed_terms = [focal] + [c for c in controls if c != focal and c in work.columns]
    fixed_terms += [c for c in structural_controls if c in work.columns]
    fixed_terms.append("iteration")
    if include_phase_effects and "phase" in work.columns and work["phase"].nunique() > 1:
        fixed_terms.append("C(phase)")
        # Motif × phase interaction when phases differ in ΔF meaning.
        fixed_terms.append(f"C(phase):{focal}")
    if (
        include_phase_effects
        and "reference_type" in work.columns
        and work["reference_type"].nunique() > 1
    ):
        fixed_terms.append("C(reference_type)")

    fe = " + ".join(fixed_terms)
    # statsmodels MixedLM formula uses vc or re_formula
    formula = f"delta_f ~ {fe}"
    re_formula = f"1 + {focal}"

    warnings_list: List[str] = []
    status = "ok"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(formula, work, groups=work["participant_id"], re_formula=re_formula)
            result = model.fit(method="lbfgs", reml=True, maxiter=200)
            for w in caught:
                warnings_list.append(str(w.message))
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        status = "fit_failed"
        if "Singular" in msg or "singular" in msg.lower():
            status = "singular_or_boundary"
        return {
            "focal": focal,
            "status": status,
            "error": msg,
            "formula": formula,
            "re_formula": re_formula,
            "n_rows": int(len(work)),
            "n_positive": n_pos,
            "n_participants_with_positive": n_parts,
        }

    # Singularity / Hessian diagnostics
    singular = False
    hess_diag: Dict[str, Any] = {}
    try:
        import numpy as np

        hess = getattr(result, "hess", None) or getattr(result, "hessian", None)
        if hess is not None:
            arr = np.asarray(hess, dtype=float)
            hess_diag["shape"] = list(arr.shape)
            try:
                eigs = np.linalg.eigvalsh(arr)
                hess_diag["min_eig"] = float(np.min(eigs))
                hess_diag["max_eig"] = float(np.max(eigs))
                if abs(hess_diag["min_eig"]) < 1e-8:
                    singular = True
            except Exception as exc:
                hess_diag["eig_error"] = str(exc)
        # cov_re near-zero slope variance often indicates boundary/singular RE
        cov = result.cov_re
        if hasattr(cov, "iloc") and cov.shape[0] >= 2:
            if float(cov.iloc[1, 1]) < 1e-10:
                singular = True
                warnings_list.append("near_zero_random_slope_variance")
    except Exception as exc:
        hess_diag["error"] = str(exc)

    if not result.converged:
        status = "not_converged"
    elif singular:
        status = "singular_or_boundary"

    # Fixed effect for focal
    params = result.fe_params
    bse = result.bse_fe
    pvalues = result.pvalues
    conf = result.conf_int()
    fixed_coef = float(params.get(focal, float("nan")))
    fixed_se = float(bse.get(focal, float("nan"))) if hasattr(bse, "get") else float("nan")
    fixed_p = float(pvalues.get(focal, float("nan"))) if hasattr(pvalues, "get") else float("nan")
    ci_low = ci_high = None
    try:
        if focal in conf.index:
            ci_low = float(conf.loc[focal, 0])
            ci_high = float(conf.loc[focal, 1])
    except Exception:
        pass

    slopes, var_slope, n_info = _participant_slopes(result, focal=focal, fixed_coef=fixed_coef)

    return {
        "focal": focal,
        "status": status,
        "formula": formula,
        "re_formula": re_formula,
        "n_rows": int(len(work)),
        "n_positive": n_pos,
        "n_participants_with_positive": n_parts,
        "n_informative_participants": n_info,
        "converged": bool(result.converged),
        "warnings": warnings_list,
        "singular_or_boundary": singular,
        "hessian": hess_diag,
        "fixed_effect": {
            "coef": fixed_coef,
            "se": fixed_se,
            "pvalue": fixed_p,
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "random_slope_variance": var_slope,
        "participant_slopes": slopes,
        "llf": float(result.llf) if result.llf is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--focal_motifs",
        default="",
        help="Comma-separated focals; default = all supported directional columns.",
    )
    parser.add_argument(
        "--min_positive_rows",
        type=int,
        default=5,
        help="Support threshold for controls and focals.",
    )
    parser.add_argument("--min_participants_with_pos", type=int, default=2)
    parser.add_argument(
        "--structural_controls",
        default="",
        help="Optional structural_* columns as fixed controls only.",
    )
    parser.add_argument(
        "--combine_phases",
        action="store_true",
        help="Fit on all phases with phase/reference factors (default: evolution only).",
    )
    parser.add_argument(
        "--phase",
        default="evolution",
        help="Phase filter when not --combine_phases (default: evolution).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "reference_type" not in df.columns and "reference_kind" in df.columns:
        df["reference_type"] = df["reference_kind"]

    if not args.combine_phases and args.phase and "phase" in df.columns:
        df = df[df["phase"] == args.phase].copy()

    support = motif_support_report(
        df,
        min_positive_rows=args.min_positive_rows,
        min_participants_with_pos=args.min_participants_with_pos,
    )
    supported = list(support["supported_columns"])
    if args.focal_motifs.strip():
        focals = [x.strip() for x in args.focal_motifs.split(",") if x.strip()]
    else:
        focals = list(supported)

    structural = [x.strip() for x in args.structural_controls.split(",") if x.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictor_support.json").write_text(
        json.dumps(support, indent=2), encoding="utf-8"
    )

    results: List[Dict[str, Any]] = []
    for focal in focals:
        if focal not in all_directional_behavioral_columns():
            results.append(
                {
                    "focal": focal,
                    "status": "skipped_unsupported",
                    "reason": "not_a_directional_behavioral_column",
                }
            )
            continue
        # Controls: other supported motifs; never opposite-direction averaging.
        controls = [c for c in supported if c != focal]
        # Prefer not to explode FE dim: keep same-direction and common others.
        # Cap controls at 8 densest supported columns excluding focal.
        controls_sorted = sorted(
            controls,
            key=lambda c: next(
                (m["n_positive_rows"] for m in support["motifs"] if m["column"] == c),
                0,
            ),
            reverse=True,
        )[:8]
        print(f"[fit_rs] Fitting focal={focal} controls={controls_sorted}", flush=True)
        res = fit_focal_motif(
            df,
            focal=focal,
            controls=controls_sorted,
            include_phase_effects=bool(args.combine_phases),
            structural_controls=structural,
        )
        results.append(res)
        # Per-focal participant slopes CSV
        if res.get("participant_slopes"):
            pd.DataFrame(res["participant_slopes"]).to_csv(
                out_dir / f"participant_slopes_{focal}.csv", index=False
            )

    # Multiple-testing correction across planned focals with raw p-values.
    raw_p = [
        (r.get("fixed_effect") or {}).get("pvalue") if r.get("status") != "skipped_unsupported" else None
        for r in results
    ]
    qvals = _bh_fdr(raw_p)
    for r, q in zip(results, qvals):
        if r.get("fixed_effect") is not None:
            r["fixed_effect"]["qvalue_bh"] = q
            r["fixed_effect"]["pvalue_raw"] = r["fixed_effect"].get("pvalue")

    summary = {
        "n_models": len(results),
        "phase_filter": None if args.combine_phases else args.phase,
        "combine_phases": bool(args.combine_phases),
        "supported_controls_universe": supported,
        "exclusions_from_support": support["excluded_columns"],
        "models": [
            {
                k: v
                for k, v in r.items()
                if k != "participant_slopes"  # keep summary lean; slopes on disk
            }
            for r in results
        ],
    }
    (out_dir / "random_slope_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    # Flat coefficients table
    rows = []
    for r in results:
        fe = r.get("fixed_effect") or {}
        rows.append(
            {
                "focal": r.get("focal"),
                "status": r.get("status"),
                "coef": fe.get("coef"),
                "se": fe.get("se"),
                "pvalue_raw": fe.get("pvalue_raw", fe.get("pvalue")),
                "qvalue_bh": fe.get("qvalue_bh"),
                "ci_low": fe.get("ci_low"),
                "ci_high": fe.get("ci_high"),
                "random_slope_variance": r.get("random_slope_variance"),
                "n_informative_participants": r.get("n_informative_participants"),
                "converged": r.get("converged"),
                "singular_or_boundary": r.get("singular_or_boundary"),
                "formula": r.get("formula"),
                "re_formula": r.get("re_formula"),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "random_slope_coefficients.csv", index=False)
    print(f"[fit_rs] Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
