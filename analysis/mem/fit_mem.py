#!/usr/bin/env python3
"""Fit a minimal participant-random-intercept MEM on PICS edit ΔF.

Model (feasibility only):
  delta_f ~ C(primary_edit) + iteration
  random intercept: participant_id

Does not silently drop convergence / singularity failures.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

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

    n_participants = df["participant_id"].nunique()
    if n_participants < 2:
        warnings.warn(
            f"Only {n_participants} participant_id level(s); random intercept is not identified.",
            RuntimeWarning,
            stacklevel=1,
        )

    motif_counts = Counter(df["primary_edit"].astype(str))
    rare = {m: c for m, c in motif_counts.items() if c < int(args.min_motif_count)}
    if rare:
        warnings.warn(
            "Rare primary_edit categories (count < "
            f"{args.min_motif_count}): {dict(sorted(rare.items(), key=lambda x: x[1]))}. "
            "Coefficients for these levels may be unstable.",
            RuntimeWarning,
            stacklevel=1,
        )

    print(
        f"Fitting MixedLM on n={len(df)} rows, "
        f"participants={n_participants}, motifs={len(motif_counts)}"
    )
    # Use formula API MixedLM for random intercept.
    model = smf.mixedlm(
        "delta_f ~ C(primary_edit) + iteration",
        data=df,
        groups=df["participant_id"],
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(method="lbfgs", reml=True)
        for w in caught:
            print(f"FIT WARNING: {w.category.__name__}: {w.message}", file=sys.stderr)
    except Exception as exc:
        print(f"FIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # Surface singularity / convergence issues explicitly.
    if not bool(getattr(result, "converged", True)):
        warnings.warn("MixedLM reports converged=False.", RuntimeWarning, stacklevel=1)
        print("WARNING: model did not converge.", file=sys.stderr)

    cov_re = getattr(result, "cov_re", None)
    if cov_re is not None:
        try:
            import numpy as np

            arr = np.asarray(cov_re, dtype=float)
            if arr.size and float(np.min(np.linalg.eigvalsh(arr))) <= 1e-10:
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
    summary_path.write_text(str(result.summary()), encoding="utf-8")
    coef = result.params.rename("coef").to_frame()
    if getattr(result, "bse", None) is not None:
        coef["stderr"] = result.bse
    if getattr(result, "pvalues", None) is not None:
        coef["pvalue"] = result.pvalues
    coef.to_csv(coef_path)
    print(result.summary())
    print(f"Wrote {summary_path}")
    print(f"Wrote {coef_path}")


if __name__ == "__main__":
    main()
