#!/usr/bin/env python3
"""Fit one-focal-motif MixedLM models with participant random slopes.

For each focal directional motif M:

  delta_f ~ M + other_supported_controls + iteration
    + (1 + M | dataset::run_id::participant_id)

Raw ``participant_id`` alone is **forbidden** as the RE grouping key (ordinals
restart per dataset). Both focal and joint fitters require ``dataset``,
``run_id``, and ``participant_id`` and validate group uniqueness.

Fits added / removed / modified separately (never averages opposite directions).
CPU-only statsmodels MixedLM. Reports fixed effects, random-slope variance,
participant-specific slopes, convergence / singularity diagnostics, and BH-FDR
across planned motif tests.

Eligibility (schema_v3/v5 CSV):
  --eligibility_mode off      (default) use all rows; legacy v2 CSVs OK
  --eligibility_mode restrict require eligible_<focal> column; subset to
                      eligible==1. Refuses legacy / missing-state CSVs.
  Focal restrict ≠ joint FE-adjust (see fit_mem_joint_random_slopes.py).

For ``*_modified`` focals under restrict, ``eligible_c_modified`` is the
**retained-construct modification risk set** (reference_has_c AND
candidate_has_c), not the broader pre-transition opportunity
``reference_has_c`` / ``modification_opportunity_c``. Within that set the
contrast is ``c_modified`` vs ``retained_unmodified_c``.

Do not hard-code focals to every construct-operation combination: run
coverage_eligibility_v5.py first, then pass --focal_motifs for effects with
within-participant variation. Preserve reassessment of history_modified,
value_modified, feedback_added, feedback_modified on final Schema-v5 runs.
"""

from __future__ import annotations

import argparse
import math
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.mem.bh_fdr import bh_fdr  # noqa: E402
from analysis.mem.predictor_support import motif_support_report  # noqa: E402
from utils.mem.schema_v2 import (  # noqa: E402
    DIRECTIONAL_SUFFIXES,
    all_directional_behavioral_columns as all_directional_behavioral_columns_v2,
)
from utils.mem.schema_participant_transition_v5 import (  # noqa: E402
    all_directional_behavioral_columns as all_directional_behavioral_columns_v5,
    risk_set_definition,
)
from utils.mem.participant_semantic_postprocess_v5 import (  # noqa: E402
    filter_frame_for_construct_effect_fitting,
)
from analysis.mem.mem_grouping import (  # noqa: E402
    GROUPING_KEY_NAME,
    MEM_GROUP_COL,
    MemGroupingError,
    attach_validated_mem_group,
    make_global_participant_key,
)


def all_directional_behavioral_columns() -> List[str]:
    """Union of v2 and v5 directional columns (CSV may be either schema)."""
    seen = set()
    out: List[str] = []
    for c in list(all_directional_behavioral_columns_v2()) + list(
        all_directional_behavioral_columns_v5()
    ):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _require_statsmodels():
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise SystemExit(
            "statsmodels is required. Install with: pip install statsmodels"
        ) from exc
    return smf


# Deprecated local name kept for any external imports; delegates to bh_fdr.
def _bh_fdr(pvals: Sequence[Optional[float]]) -> List[Optional[float]]:
    return bh_fdr(pvals)

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


# Re-export make_global_participant_key is already imported from mem_grouping.


def fit_focal_motif(
    df: pd.DataFrame,
    *,
    focal: str,
    controls: Sequence[str],
    include_phase_effects: bool,
    structural_controls: Sequence[str],
    optimizer_methods: Sequence[str] = ("lbfgs",),
    reml: bool = True,
    maxiter: int = 200,
    center_iteration: bool = False,
    scale_delta_f: bool = False,
) -> Dict[str, Any]:
    smf = _require_statsmodels()
    work = df.copy()
    work["delta_f"] = pd.to_numeric(work["delta_f"], errors="coerce")
    work["iteration"] = pd.to_numeric(work.get("iteration"), errors="coerce")
    work["participant_id"] = work["participant_id"].astype(str)
    try:
        work, group_meta = attach_validated_mem_group(work)
    except MemGroupingError as exc:
        return {
            "focal": focal,
            "status": "grouping_failed",
            "error": str(exc),
            "grouping_key": GROUPING_KEY_NAME,
        }
    work[focal] = pd.to_numeric(work[focal], errors="coerce").fillna(0).astype(int)
    for c in list(controls) + list(structural_controls):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0).astype(int)

    # Optional numerically equivalent transforms (same linear model up to scaling).
    delta_scale = 1.0
    if scale_delta_f:
        sd = float(pd.to_numeric(work["delta_f"], errors="coerce").std(ddof=1) or 1.0)
        if sd > 0:
            work["delta_f"] = work["delta_f"] / sd
            delta_scale = sd
    if center_iteration and "iteration" in work.columns:
        work["iteration"] = work["iteration"] - work["iteration"].mean()

    # Drop rows missing outcome / grouping
    work = work.dropna(subset=["delta_f", MEM_GROUP_COL, "iteration"])
    n_pos = int((work[focal] == 1).sum())
    n_parts = int(work.loc[work[focal] == 1, MEM_GROUP_COL].nunique())
    n_datasets = (
        int(work["dataset"].nunique()) if "dataset" in work.columns and len(work) else 0
    )
    if n_pos < 5 or n_parts < 2:
        return {
            "focal": focal,
            "status": "skipped_unsupported",
            "reason": "too_few_positive_rows_or_participants",
            "n_positive": n_pos,
            "n_participants_with_positive": n_parts,
            "n_datasets": n_datasets,
            "n_rows": int(len(work)),
            "grouping_key": GROUPING_KEY_NAME,
            "grouping_meta": group_meta,
        }

    fixed_terms = [focal] + [c for c in controls if c != focal and c in work.columns]
    fixed_terms += [c for c in structural_controls if c in work.columns]
    # Drop constant FE predictors (e.g. explore slice has iteration==0 for all rows).
    kept: List[str] = []
    dropped_constant: List[str] = []
    for term in fixed_terms:
        if term not in work.columns:
            continue
        nunq = int(pd.to_numeric(work[term], errors="coerce").nunique(dropna=True))
        if nunq < 2 and term != focal:
            dropped_constant.append(term)
            continue
        kept.append(term)
    if "iteration" in work.columns and int(pd.to_numeric(work["iteration"], errors="coerce").nunique(dropna=True)) >= 2:
        kept.append("iteration")
    elif "iteration" in work.columns:
        dropped_constant.append("iteration")
    fixed_terms = kept
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

    fe = " + ".join(fixed_terms) if fixed_terms else "1"
    # statsmodels MixedLM formula uses vc or re_formula
    formula = f"delta_f ~ {fe}"
    re_formula = f"1 + {focal}"

    warnings_list: List[str] = []
    if dropped_constant:
        warnings_list.append(f"dropped_constant_fe:{','.join(dropped_constant)}")
    if center_iteration:
        warnings_list.append("centered_iteration")
    if scale_delta_f and delta_scale != 1.0:
        warnings_list.append(f"scaled_delta_f_by_{delta_scale:.6g}")

    status = "ok"
    result = None
    used_method = None
    method_attempts: List[Dict[str, Any]] = []
    last_exc: Optional[str] = None
    model = smf.mixedlm(
        formula, work, groups=work[MEM_GROUP_COL], re_formula=re_formula
    )
    for method in optimizer_methods:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit_res = model.fit(method=method, reml=reml, maxiter=maxiter)
                msgs = [str(w.message) for w in caught]
            method_attempts.append(
                {
                    "method": method,
                    "converged": bool(fit_res.converged),
                    "warnings": msgs,
                }
            )
            result = fit_res
            used_method = method
            warnings_list.extend(msgs)
            if bool(fit_res.converged):
                # Prefer first converged method with finite, non-pathological FE SE.
                try:
                    se_try = float(fit_res.bse_fe.get(focal, float("nan")))
                    coef_try = float(fit_res.fe_params.get(focal, float("nan")))
                    if (
                        math.isfinite(se_try)
                        and math.isfinite(coef_try)
                        and se_try <= max(10.0, 50.0 * (abs(coef_try) + 0.05))
                    ):
                        break
                except Exception:
                    break
                # else keep trying next optimizer
        except Exception as exc:
            last_exc = f"{type(exc).__name__}: {exc}"
            method_attempts.append(
                {"method": method, "converged": False, "error": last_exc}
            )
            continue

    if result is None:
        status = "fit_failed"
        if last_exc and ("Singular" in last_exc or "singular" in last_exc.lower()):
            status = "singular_or_boundary"
        return {
            "focal": focal,
            "status": status,
            "error": last_exc,
            "formula": formula,
            "re_formula": re_formula,
            "optimizer_attempts": method_attempts,
            "n_rows": int(len(work)),
            "n_positive": n_pos,
            "n_participants_with_positive": n_parts,
            "grouping_key": GROUPING_KEY_NAME,
            "grouping_meta": group_meta,
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
    # Rescale FE to original delta_f units if outcome was standardized.
    if delta_scale != 1.0:
        fixed_coef = fixed_coef * delta_scale
        fixed_se = fixed_se * delta_scale if fixed_se == fixed_se else fixed_se
        if ci_low is not None:
            ci_low = ci_low * delta_scale
        if ci_high is not None:
            ci_high = ci_high * delta_scale

    slopes, var_slope, n_info = _participant_slopes(result, focal=focal, fixed_coef=fixed_coef)
    if delta_scale != 1.0 and var_slope is not None:
        var_slope = float(var_slope) * (delta_scale ** 2)

    # Intercept variance from cov_re when available
    var_intercept = None
    try:
        cov = result.cov_re
        if hasattr(cov, "iloc") and cov.shape[0] >= 1:
            var_intercept = float(cov.iloc[0, 0])
            if delta_scale != 1.0:
                var_intercept = var_intercept * (delta_scale ** 2)
    except Exception:
        pass

    return {
        "focal": focal,
        "status": status,
        "formula": formula,
        "re_formula": re_formula,
        "grouping_key": GROUPING_KEY_NAME,
        "grouping_column": MEM_GROUP_COL,
        "grouping_meta": group_meta,
        "n_rows": int(len(work)),
        "n_positive": n_pos,
        "n_participants_with_positive": n_parts,
        "n_groups": int(work[MEM_GROUP_COL].nunique()),
        "n_datasets": n_datasets,
        "n_informative_participants": n_info,
        "converged": bool(result.converged),
        "optimizer": used_method,
        "optimizer_attempts": method_attempts,
        "reml": reml,
        "center_iteration": center_iteration,
        "scale_delta_f": scale_delta_f,
        "delta_f_scale": delta_scale,
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
        "random_intercept_variance": var_intercept,
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
        "--max_motif_controls",
        type=int,
        default=8,
        help="Max other supported directional motifs as FE controls "
        "(0 = primary focal-only: delta_f ~ focal + iteration).",
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
    parser.add_argument(
        "--eligibility_mode",
        choices=["off", "restrict"],
        default="off",
        help="off: all rows. restrict: subset each focal to eligible_<focal>==1 "
        "(requires schema_v3 eligibility columns; refuses legacy).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "reference_type" not in df.columns and "reference_kind" in df.columns:
        df["reference_type"] = df["reference_kind"]

    if not args.combine_phases and args.phase and "phase" in df.columns:
        df = df[df["phase"] == args.phase].copy()

    df, n_excl_unresolved = filter_frame_for_construct_effect_fitting(df)
    if n_excl_unresolved:
        print(
            f"[fit_rs] excluded {n_excl_unresolved} unresolved NMC adjudication "
            "row(s) from construct-transition effect fitting",
            flush=True,
        )

    eligibility_mode = str(args.eligibility_mode)
    if eligibility_mode == "restrict":
        # Pre-check that at least one eligible_* column exists; per-focal checked below.
        elig_any = [c for c in df.columns if str(c).startswith("eligible_")]
        if not elig_any:
            raise SystemExit(
                "eligibility_mode=restrict requires schema_v3 eligibility columns "
                "(eligible_<motif>_<direction>). Legacy v2 CSVs lack state/eligibility; "
                "do not treat directional X=0 as ineligible. Refusing."
            )

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
        df_focal = df
        if eligibility_mode == "restrict":
            elig_col = f"eligible_{focal}"
            if elig_col not in df.columns:
                raise SystemExit(
                    f"eligibility_mode=restrict requires column {elig_col!r} "
                    f"for focal {focal!r}. Missing/null eligibility is not treated "
                    "as zero; refusing legacy or incomplete state-aware CSV."
                )
            if df[elig_col].isna().any():
                raise SystemExit(
                    f"eligibility_mode=restrict: {elig_col} has null values; "
                    "refusing (missing state ≠ ineligible)."
                )
            df_focal = df[df[elig_col].astype(int) == 1].copy()
            # For modified focals: contrast is modified vs retained_unmodified
            # inside the retained-construct risk set.
            if focal.endswith("_modified"):
                construct = focal[: -len("_modified")]
                ret_col = f"retained_unmodified_{construct}"
                if ret_col in df_focal.columns:
                    mod_vals = pd.to_numeric(df_focal[focal], errors="coerce").fillna(0).astype(int)
                    ret_vals = pd.to_numeric(df_focal[ret_col], errors="coerce").fillna(0).astype(int)
                    # Within risk set, exactly one of modified / retained_unmodified.
                    bad = ((mod_vals + ret_vals) != 1).sum()
                    if int(bad) > 0:
                        print(
                            f"[fit_rs] warning: {focal}: {bad} rows in retained-construct "
                            f"set are not a clean modified vs retained_unmodified partition",
                            flush=True,
                        )
        # Controls: other supported motifs; never opposite-direction averaging.
        # Primary one-focal models use --max_motif_controls 0.
        max_ctrl = max(0, int(args.max_motif_controls))
        controls = [c for c in supported if c != focal] if max_ctrl > 0 else []
        controls_sorted = sorted(
            controls,
            key=lambda c: next(
                (m["n_positive_rows"] for m in support["motifs"] if m["column"] == c),
                0,
            ),
            reverse=True,
        )[:max_ctrl]
        print(
            f"[fit_rs] Fitting focal={focal} controls={controls_sorted} "
            f"n_rows={len(df_focal)} eligibility_mode={eligibility_mode}",
            flush=True,
        )
        res = fit_focal_motif(
            df_focal,
            focal=focal,
            controls=controls_sorted,
            include_phase_effects=bool(args.combine_phases),
            structural_controls=structural,
        )
        if eligibility_mode == "restrict":
            res["eligibility_mode"] = "restrict"
            res["eligibility_column"] = f"eligible_{focal}"
            try:
                res["risk_set"] = risk_set_definition(focal)
            except ValueError:
                res["risk_set"] = None
            res["n_rows_eligible"] = int(len(df_focal))
            res["n_rows_full"] = int(len(df))
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
    qvals = bh_fdr(raw_p)
    for r, q in zip(results, qvals):
        if r.get("fixed_effect") is not None:
            r["fixed_effect"]["qvalue_bh"] = q
            r["fixed_effect"]["pvalue_raw"] = r["fixed_effect"].get("pvalue")

    summary = {
        "n_models": len(results),
        "eligibility_mode": eligibility_mode,
        "phase_filter": None if args.combine_phases else args.phase,
        "combine_phases": bool(args.combine_phases),
        "n_excluded_unresolved_nmc_adjudication": int(n_excl_unresolved),
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
                "random_intercept_variance": r.get("random_intercept_variance"),
                "random_slope_variance": r.get("random_slope_variance"),
                "n_informative_participants": r.get("n_informative_participants"),
                "n_rows": r.get("n_rows", r.get("n_rows_eligible")),
                "n_positive": r.get("n_positive"),
                "n_groups": r.get("n_groups"),
                "n_datasets": r.get("n_datasets"),
                "converged": r.get("converged"),
                "singular_or_boundary": r.get("singular_or_boundary"),
                "optimizer": r.get("optimizer"),
                "grouping_key": r.get("grouping_key"),
                "formula": r.get("formula"),
                "re_formula": r.get("re_formula"),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "random_slope_coefficients.csv", index=False)
    print(f"[fit_rs] Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
