#!/usr/bin/env python3
"""Fit a minimal participant-random-intercept MEM on PICS edit ΔF.

Model (feasibility only):
  delta_f ~ C(primary_edit) + iteration
  random intercept: participant_id

Does not silently drop convergence / singularity failures.

Optional W&B logging (project default: teh_mem) uploads summary text, coefficient
table, and scalar diagnostics so results survive local disk / server shutdown.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _require_statsmodels():
    try:
        import statsmodels.formula.api as smf  # noqa: F401
        from statsmodels.regression.mixed_linear_model import MixedLM  # noqa: F401
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
    name = run_name or f"mem_fit_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    return wandb.init(project=project, name=name, config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for model summary.txt and coefficients.csv",
    )
    parser.add_argument("--min_motif_count", type=int, default=5)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="teh_mem",
        help="W&B project name (default: teh_mem).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name (default: mem_fit_YYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    args = parser.parse_args()

    smf = _require_statsmodels()
    df = pd.read_csv(args.input_csv)
    required = {"delta_f", "primary_edit", "iteration", "participant_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["delta_f"] = pd.to_numeric(df["delta_f"], errors="coerce")
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df = df.dropna(subset=["delta_f", "primary_edit", "iteration", "participant_id"])
    if df.empty:
        raise SystemExit("No usable rows after dropping NA in required columns.")

    n_participants = int(df["participant_id"].nunique())
    motif_counts = Counter(df["primary_edit"].astype(str))
    if n_participants < 2:
        warnings.warn(
            f"Only {n_participants} participant_id level(s); random intercept is not identified.",
            RuntimeWarning,
            stacklevel=1,
        )

    rare = {m: c for m, c in motif_counts.items() if c < int(args.min_motif_count)}
    if rare:
        warnings.warn(
            "Rare primary_edit categories (count < "
            f"{args.min_motif_count}): {dict(sorted(rare.items(), key=lambda x: x[1]))}. "
            "Coefficients for these levels may be unstable.",
            RuntimeWarning,
            stacklevel=1,
        )

    wb = _init_wandb(
        project=str(args.wandb_project),
        enabled=not bool(args.no_log),
        run_name=args.wandb_run_name,
        config={
            "input_csv": str(Path(args.input_csv).resolve()),
            "output_dir": str(Path(args.output_dir).resolve()),
            "min_motif_count": int(args.min_motif_count),
            "formula": "delta_f ~ C(primary_edit) + iteration",
            "groups": "participant_id",
            "n_rows": int(len(df)),
            "n_participants": n_participants,
            "n_motifs": len(motif_counts),
        },
    )

    print(
        f"Fitting MixedLM on n={len(df)} rows, "
        f"participants={n_participants}, motifs={len(motif_counts)}",
        flush=True,
    )
    # Use formula API MixedLM for random intercept.
    model = smf.mixedlm(
        "delta_f ~ C(primary_edit) + iteration",
        data=df,
        groups=df["participant_id"],
    )
    fit_failed = False
    fit_error = ""
    result = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(method="lbfgs", reml=True)
        for w in caught:
            print(f"FIT WARNING: {w.category.__name__}: {w.message}", file=sys.stderr)
    except Exception as exc:
        fit_failed = True
        fit_error = f"{type(exc).__name__}: {exc}"
        print(f"FIT FAILED: {fit_error}", file=sys.stderr)
        if wb is not None:
            import wandb

            wandb.log({"fit_failed": 1, "fit_error": fit_error})
            wandb.finish(exit_code=1)
        raise SystemExit(1) from exc

    assert result is not None
    converged = bool(getattr(result, "converged", True))
    singular_re = False
    # Surface singularity / convergence issues explicitly.
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
        except Exception as exc:  # diagnostic only
            print(f"WARNING: could not diagnose cov_re singularity: {exc}", file=sys.stderr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

        motif_mean = (
            df.groupby("primary_edit", dropna=False)["delta_f"]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        scalars: Dict[str, Any] = {
            "fit_failed": 0,
            "n_rows": int(len(df)),
            "n_participants": n_participants,
            "n_motifs": len(motif_counts),
            "delta_f_mean": float(df["delta_f"].mean()),
            "delta_f_std": float(df["delta_f"].std(ddof=1)) if len(df) > 1 else 0.0,
            "converged": int(converged),
            "singular_random_effects": int(singular_re),
            "llf": float(getattr(result, "llf", float("nan"))),
        }
        # Group variance if available
        try:
            import numpy as np

            if cov_re is not None:
                scalars["group_var"] = float(np.asarray(cov_re, dtype=float).reshape(-1)[0])
        except Exception:
            pass
        for name, row in coef.iterrows():
            key = str(name).replace("[", "_").replace("]", "_").replace(".", "_")
            scalars[f"coef/{key}"] = float(row["coef"])
            if "pvalue" in row and pd.notna(row["pvalue"]):
                scalars[f"pvalue/{key}"] = float(row["pvalue"])

        wandb.log(scalars)
        wandb.log(
            {
                "coefficients": wandb.Table(dataframe=coef.reset_index().rename(columns={"index": "term"})),
                "motif_counts": wandb.Table(
                    dataframe=pd.DataFrame(
                        {"primary_edit": list(motif_counts.keys()), "count": list(motif_counts.values())}
                    )
                ),
                "motif_delta_f_summary": wandb.Table(dataframe=motif_mean),
            }
        )
        art = wandb.Artifact(
            name=f"mem_fit_{wb.id}",
            type="mem_fit",
            metadata={"n_rows": int(len(df)), "n_participants": n_participants},
        )
        art.add_file(str(summary_path))
        art.add_file(str(coef_path))
        art.add_file(str(Path(args.input_csv).resolve()))
        wb.log_artifact(art)
        # Also keep a readable summary in the run page.
        wandb.summary["model_summary"] = summary_text[:8000]
        print(f"[wandb] Logged to project={args.wandb_project} run={wb.name} url={wb.url}", flush=True)
        wandb.finish()


if __name__ == "__main__":
    main()
