#!/usr/bin/env python3
"""Fit participant-random-intercept MEM on schema-v2 directional ΔF predictors.

Behavioral predictors must be selected explicitly via --behavioral_predictors.
Constant / missing predictors are reported and excluded from the formula (never
silently dropped without a reason). Structural columns may be added as optional
controls via --structural_controls.

Does not use primary_edit. Does not read test metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mem.schema_v2 import (  # noqa: E402
    SCHEMA_VERSION,
    all_directional_behavioral_columns,
    all_structural_columns,
)


def _require_statsmodels():
    try:
        import statsmodels.formula.api as smf  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "statsmodels is required for fit_mem.py. Install with:\n"
            "  pip install statsmodels\n"
            "or: pip install 'mindsascode[analysis]'"
        ) from exc
    import statsmodels.formula.api as smf

    return smf


def _init_wandb(
    *,
    project: str,
    enabled: bool,
    config: Dict[str, Any],
    run_name: Optional[str],
):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is required unless --no_log is set. Install wandb or pass --no_log."
        ) from exc
    name = run_name or f"mem_fit_v2_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    return wandb.init(project=project, name=name, config=config)


def _wandb_safe_key(name: str) -> str:
    return str(name).replace("[", "_").replace("]", "_").replace(".", "_")


def _parse_column_list(raw: Optional[str]) -> List[str]:
    if raw is None or not str(raw).strip():
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def select_predictors(
    df: pd.DataFrame,
    *,
    requested_behavioral: Sequence[str],
    requested_structural: Sequence[str],
    min_count: int,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Return (included_behavioral, included_structural, exclusion_reports).

    Never auto-selects rare predictors. Never silently drops without a reason row.
    """
    allowed_b = set(all_directional_behavioral_columns())
    allowed_s = set(all_structural_columns())
    reports: List[Dict[str, Any]] = []
    included_b: List[str] = []
    included_s: List[str] = []

    if not requested_behavioral:
        raise SystemExit(
            "Must pass --behavioral_predictors explicitly "
            "(comma-separated directional columns, e.g. history_added,risk_modified). "
            f"Available: {', '.join(all_directional_behavioral_columns())}"
        )

    for col in requested_behavioral:
        if col not in allowed_b:
            reports.append(
                {
                    "column": col,
                    "role": "behavioral",
                    "included": False,
                    "reason": "not_a_directional_behavioral_column",
                }
            )
            continue
        if col not in df.columns:
            reports.append(
                {
                    "column": col,
                    "role": "behavioral",
                    "included": False,
                    "reason": "missing_from_csv",
                }
            )
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        n_pos = int((vals == 1).sum())
        n_unique = int(vals.nunique(dropna=True))
        if n_unique < 2:
            reports.append(
                {
                    "column": col,
                    "role": "behavioral",
                    "included": False,
                    "reason": "constant_predictor",
                    "n_positive": n_pos,
                    "n_unique": n_unique,
                }
            )
            continue
        if n_pos < int(min_count):
            reports.append(
                {
                    "column": col,
                    "role": "behavioral",
                    "included": False,
                    "reason": "below_min_count",
                    "n_positive": n_pos,
                    "min_count": int(min_count),
                }
            )
            continue
        included_b.append(col)
        reports.append(
            {
                "column": col,
                "role": "behavioral",
                "included": True,
                "reason": "ok",
                "n_positive": n_pos,
            }
        )

    for col in requested_structural:
        if col not in allowed_s:
            reports.append(
                {
                    "column": col,
                    "role": "structural_control",
                    "included": False,
                    "reason": "not_a_structural_column",
                }
            )
            continue
        if col not in df.columns:
            reports.append(
                {
                    "column": col,
                    "role": "structural_control",
                    "included": False,
                    "reason": "missing_from_csv",
                }
            )
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        n_pos = int((vals == 1).sum())
        n_unique = int(vals.nunique(dropna=True))
        if n_unique < 2:
            reports.append(
                {
                    "column": col,
                    "role": "structural_control",
                    "included": False,
                    "reason": "constant_predictor",
                    "n_positive": n_pos,
                }
            )
            continue
        if n_pos < int(min_count):
            reports.append(
                {
                    "column": col,
                    "role": "structural_control",
                    "included": False,
                    "reason": "below_min_count",
                    "n_positive": n_pos,
                    "min_count": int(min_count),
                }
            )
            continue
        included_s.append(col)
        reports.append(
            {
                "column": col,
                "role": "structural_control",
                "included": True,
                "reason": "ok",
                "n_positive": n_pos,
            }
        )

    if not included_b:
        raise SystemExit(
            "No usable behavioral predictors after exclusion checks. "
            f"Reports: {json.dumps(reports, indent=2)}"
        )
    return included_b, included_s, reports


def build_formula(behavioral: Sequence[str], structural: Sequence[str]) -> str:
    terms = list(behavioral) + list(structural) + ["iteration"]
    return "delta_f ~ " + " + ".join(terms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for model summary.txt, coefficients.csv, predictor_report.json",
    )
    parser.add_argument(
        "--behavioral_predictors",
        type=str,
        required=True,
        help=(
            "Comma-separated directional columns to include as fixed effects "
            "(e.g. history_added,history_modified,risk_added). Required; no auto-select."
        ),
    )
    parser.add_argument(
        "--structural_controls",
        type=str,
        default="",
        help="Optional comma-separated structural_* columns as controls (not mixed into behavioral).",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Exclude requested predictors with fewer than this many positives (reported).",
    )
    parser.add_argument("--wandb_project", type=str, default="teh_mem")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_detail_level",
        type=str,
        default="final_only",
        choices=["final_only", "full"],
        help=(
            "How much to upload to Weights & Biases. "
            "'final_only' logs only compact end-of-fit summary metrics; "
            "'full' also uploads per-term coefficients, tables, and artifacts."
        ),
    )
    parser.add_argument("--no_log", action="store_true")
    args = parser.parse_args()

    smf = _require_statsmodels()
    df = pd.read_csv(args.input_csv)
    if "schema_version" in df.columns:
        bad = df[df["schema_version"].fillna(-1).astype(int) != SCHEMA_VERSION]
        if len(bad):
            raise SystemExit(
                f"CSV has {len(bad)} rows with schema_version != {SCHEMA_VERSION}"
            )

    required = {"delta_f", "iteration", "participant_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")
    if "primary_edit" in df.columns:
        warnings.warn(
            "CSV still contains primary_edit; schema v2 fit ignores it.",
            RuntimeWarning,
            stacklevel=1,
        )

    df = df.copy()
    df["delta_f"] = pd.to_numeric(df["delta_f"], errors="coerce")
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df = df.dropna(subset=["delta_f", "iteration", "participant_id"])
    if df.empty:
        raise SystemExit("No usable rows after dropping NA in required columns.")

    req_b = _parse_column_list(args.behavioral_predictors)
    req_s = _parse_column_list(args.structural_controls)
    included_b, included_s, reports = select_predictors(
        df,
        requested_behavioral=req_b,
        requested_structural=req_s,
        min_count=int(args.min_count),
    )
    formula = build_formula(included_b, included_s)
    n_participants = int(df["participant_id"].nunique())
    if n_participants < 2:
        warnings.warn(
            f"Only {n_participants} participant_id level(s); random intercept is not identified.",
            RuntimeWarning,
            stacklevel=1,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "predictor_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "requested_behavioral": req_b,
                "requested_structural": req_s,
                "included_behavioral": included_b,
                "included_structural": included_s,
                "formula": formula,
                "reports": reports,
                "min_count": int(args.min_count),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {report_path}", flush=True)
    for r in reports:
        if not r.get("included"):
            print(
                f"[fit] EXCLUDED {r['role']} {r['column']}: {r['reason']}",
                flush=True,
            )

    wb = _init_wandb(
        project=str(args.wandb_project),
        enabled=not bool(args.no_log),
        run_name=args.wandb_run_name,
        config={
            "schema_version": SCHEMA_VERSION,
            "input_csv": str(Path(args.input_csv).resolve()),
            "output_dir": str(out_dir.resolve()),
            "formula": formula,
            "behavioral_predictors": included_b,
            "structural_controls": included_s,
            "groups": "participant_id",
            "n_rows": int(len(df)),
            "n_participants": n_participants,
        },
    )

    print(
        f"Fitting MixedLM on n={len(df)} rows, participants={n_participants}, "
        f"formula={formula}",
        flush=True,
    )
    model = smf.mixedlm(formula, data=df, groups=df["participant_id"])
    result = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(method="lbfgs", reml=True)
        for w in caught:
            print(f"FIT WARNING: {w.category.__name__}: {w.message}", file=sys.stderr)
    except Exception as exc:
        fit_error = f"{type(exc).__name__}: {exc}"
        print(f"FIT FAILED: {fit_error}", file=sys.stderr)
        if wb is not None:
            import wandb

            wandb.summary["fit_failed"] = 1
            wandb.summary["fit_error"] = fit_error
            wandb.finish(exit_code=1)
        raise SystemExit(1) from exc

    assert result is not None
    converged = bool(getattr(result, "converged", True))
    singular_re = False
    if not converged:
        warnings.warn("MixedLM reports converged=False.", RuntimeWarning, stacklevel=1)
        print("WARNING: model did not converge.", file=sys.stderr)

    cov_re = getattr(result, "cov_re", None)
    if cov_re is not None:
        try:
            import numpy as np

            arr = np.asarray(cov_re, dtype=float)
            if arr.size and float(np.min(np.linalg.eigvalsh(arr))) <= 1e-10:
                singular_re = True
                warnings.warn(
                    "Random-effect covariance appears singular/near-singular.",
                    RuntimeWarning,
                    stacklevel=1,
                )
                print("WARNING: singular/near-singular random effects.", file=sys.stderr)
        except Exception as exc:
            print(f"WARNING: could not diagnose cov_re singularity: {exc}", file=sys.stderr)

    summary_path = out_dir / "summary.txt"
    coef_path = out_dir / "coefficients.csv"
    summary_text = str(result.summary())
    summary_path.write_text(summary_text, encoding="utf-8")
    coef = result.params.rename("coef").to_frame()
    if getattr(result, "bse", None) is not None:
        coef["stderr"] = result.bse
    if getattr(result, "pvalues", None) is not None:
        coef["pvalue"] = result.pvalues
    coef.to_csv(coef_path)
    print(summary_text)
    print(f"Wrote {summary_path}")
    print(f"Wrote {coef_path}")

    if wb is not None:
        import wandb

        final_summary: Dict[str, Any] = {
            "fit_failed": 0,
            "schema_version": SCHEMA_VERSION,
            "n_rows": int(len(df)),
            "n_participants": n_participants,
            "n_behavioral_predictors": len(included_b),
            "n_structural_controls": len(included_s),
            "delta_f_mean": float(df["delta_f"].mean()),
            "delta_f_std": float(df["delta_f"].std(ddof=1)) if len(df) > 1 else 0.0,
            "converged": int(converged),
            "singular_random_effects": int(singular_re),
            "llf": float(getattr(result, "llf", float("nan"))),
        }
        try:
            import numpy as np

            if cov_re is not None:
                final_summary["group_var"] = float(
                    np.asarray(cov_re, dtype=float).reshape(-1)[0]
                )
        except Exception:
            pass
        wandb.log(final_summary)
        for key, value in final_summary.items():
            wandb.summary[_wandb_safe_key(key)] = value
        wandb.summary["formula"] = formula
        wandb.summary["input_csv"] = str(Path(args.input_csv).resolve())
        wandb.summary["summary_path"] = str(summary_path.resolve())
        wandb.summary["coefficients_path"] = str(coef_path.resolve())
        wandb.summary["predictor_report_path"] = str(report_path.resolve())
        wandb.summary["model_summary"] = summary_text[:8000]

        if args.wandb_detail_level == "full":
            detailed_scalars: Dict[str, Any] = {}
            for name, row in coef.iterrows():
                key = _wandb_safe_key(name)
                detailed_scalars[f"coef/{key}"] = float(row["coef"])
                if "pvalue" in row and pd.notna(row["pvalue"]):
                    detailed_scalars[f"pvalue/{key}"] = float(row["pvalue"])
            if detailed_scalars:
                wandb.log(detailed_scalars)
            wandb.log(
                {
                    "coefficients": wandb.Table(
                        dataframe=coef.reset_index().rename(columns={"index": "term"})
                    ),
                    "predictor_report": wandb.Table(dataframe=pd.DataFrame(reports)),
                }
            )
            art = wandb.Artifact(
                name=f"mem_fit_v2_{wb.id}",
                type="mem_fit",
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "n_rows": int(len(df)),
                    "n_participants": n_participants,
                },
            )
            art.add_file(str(summary_path))
            art.add_file(str(coef_path))
            art.add_file(str(report_path))
            art.add_file(str(Path(args.input_csv).resolve()))
            wb.log_artifact(art)
        print(
            f"[wandb] Logged to project={args.wandb_project} run={wb.name} url={wb.url}",
            flush=True,
        )
        wandb.finish()


if __name__ == "__main__":
    main()
