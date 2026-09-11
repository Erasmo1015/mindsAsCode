#!/usr/bin/env python3
"""Rewrite BH-FDR q-values for an existing focal random-slope coefficients CSV.

Writes sibling files under --output_dir (does not overwrite the original CSV).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.mem.bh_fdr import bh_fdr  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coefficients_csv",
        required=True,
        help="Existing random_slope_coefficients.csv",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--summary_json",
        default="",
        help="Optional random_slope_summary.json to mirror corrected q-values",
    )
    args = p.parse_args()

    src = Path(args.coefficients_csv)
    df = pd.read_csv(src)
    if "pvalue_raw" not in df.columns:
        raise SystemExit("coefficients CSV missing pvalue_raw")
    raw = [
        None if pd.isna(v) else float(v)
        for v in df["pvalue_raw"].tolist()
    ]
    # Only correct rows that were fitted with a finite p (status ok / not skipped)
    q = bh_fdr(raw)
    out = df.copy()
    out["qvalue_bh_broken_original"] = (
        df["qvalue_bh"] if "qvalue_bh" in df.columns else None
    )
    out["qvalue_bh"] = q
    out["bh_family"] = "focal_models_17_or_planned"
    out["bh_note"] = (
        "Corrected with statsmodels multipletests fdr_bh over finite pvalue_raw; "
        "original on-disk qvalue_bh was broken (constant)."
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "random_slope_coefficients_bh_fixed.csv"
    out.to_csv(out_csv, index=False)

    meta = {
        "source_coefficients_csv": str(src),
        "n_rows": int(len(out)),
        "n_finite_p": int(sum(1 for x in raw if x is not None)),
        "qvalue_unique_original": (
            int(df["qvalue_bh"].nunique(dropna=True))
            if "qvalue_bh" in df.columns
            else None
        ),
        "qvalue_unique_corrected": int(out["qvalue_bh"].nunique(dropna=True)),
        "method": "statsmodels.stats.multitest.multipletests method=fdr_bh",
        "output_csv": str(out_csv),
    }

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        by_focal = {
            str(r["focal"]): (None if pd.isna(r["qvalue_bh"]) else float(r["qvalue_bh"]))
            for _, r in out.iterrows()
        }
        for m in summary.get("models") or []:
            fe = m.get("fixed_effect")
            if not isinstance(fe, dict):
                continue
            focal = m.get("focal")
            if focal in by_focal:
                fe["qvalue_bh_broken_original"] = fe.get("qvalue_bh")
                fe["qvalue_bh"] = by_focal[focal]
        out_summary = out_dir / "random_slope_summary_bh_fixed.json"
        out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        meta["output_summary_json"] = str(out_summary)

    (out_dir / "bh_correction_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
