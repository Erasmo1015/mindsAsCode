#!/usr/bin/env python3
"""Predictor-support report for schema-v2 directional motif columns.

Reports prevalence / co-occurrence / rarity for every directional motif.
Does NOT select predictors by significance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mem.schema_v2 import all_directional_behavioral_columns  # noqa: E402


def motif_support_report(
    df: pd.DataFrame,
    *,
    min_positive_rows: int = 5,
    min_participants_with_pos: int = 2,
) -> Dict[str, Any]:
    cols = [c for c in all_directional_behavioral_columns() if c in df.columns]
    missing = [c for c in all_directional_behavioral_columns() if c not in df.columns]
    pid_col = "participant_id" if "participant_id" in df.columns else None
    phase_col = "phase" if "phase" in df.columns else None
    ds_col = "dataset" if "dataset" in df.columns else None

    motifs: List[Dict[str, Any]] = []
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        n_pos = int((vals == 1).sum())
        n_neg = int((vals == 0).sum())
        n_unique = int(vals.nunique(dropna=True))
        row: Dict[str, Any] = {
            "column": col,
            "n_positive_rows": n_pos,
            "n_negative_rows": n_neg,
            "n_unique": n_unique,
            "constant": n_unique < 2,
            "too_rare_rows": n_pos < int(min_positive_rows),
        }
        if pid_col:
            g = df.assign(_v=vals).groupby(pid_col)["_v"]
            parts_pos = int((g.max() >= 1).sum())
            parts_both = int(((g.max() >= 1) & (g.min() <= 0)).sum())
            row["n_participants_with_positive"] = parts_pos
            row["n_participants_with_both_pos_neg"] = parts_both
            row["too_rare_participants"] = parts_pos < int(min_participants_with_pos)
        if phase_col:
            row["prevalence_by_phase"] = (
                df.assign(_v=vals).groupby(phase_col)["_v"].mean().round(4).to_dict()
            )
        if ds_col:
            row["prevalence_by_dataset"] = (
                df.assign(_v=vals).groupby(ds_col)["_v"].mean().round(4).to_dict()
            )
        # Exclusion recommendation (support only; not significance).
        reasons = []
        if row["constant"]:
            reasons.append("constant_predictor")
        if row["too_rare_rows"]:
            reasons.append("below_min_positive_rows")
        if row.get("too_rare_participants"):
            reasons.append("below_min_participants_with_positive")
        row["supported"] = len(reasons) == 0
        row["exclusion_reasons"] = reasons
        motifs.append(row)

    # Pairwise correlations among non-constant columns with enough variance.
    corr_cols = [m["column"] for m in motifs if not m["constant"] and m["n_positive_rows"] > 0]
    corr = None
    cooccur = None
    if len(corr_cols) >= 2:
        mat = df[corr_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        corr = mat.corr().round(4).to_dict()
        # Co-occurrence: P(col_j=1 | col_i=1)
        cooccur = {}
        for a in corr_cols:
            cooccur[a] = {}
            a_pos = mat[a] == 1
            denom = int(a_pos.sum())
            for b in corr_cols:
                if denom == 0:
                    cooccur[a][b] = None
                else:
                    cooccur[a][b] = round(float(((mat[b] == 1) & a_pos).sum()) / denom, 4)

    supported = [m["column"] for m in motifs if m["supported"]]
    excluded = [
        {"column": m["column"], "reasons": m["exclusion_reasons"]}
        for m in motifs
        if not m["supported"]
    ]
    return {
        "n_rows": int(len(df)),
        "missing_columns": missing,
        "min_positive_rows": int(min_positive_rows),
        "min_participants_with_pos": int(min_participants_with_pos),
        "motifs": motifs,
        "supported_columns": supported,
        "excluded_columns": excluded,
        "pairwise_correlation": corr,
        "pairwise_cooccurrence_given_positive": cooccur,
        "note": (
            "Support filters are count/variance only. Do not interpret as significance "
            "screening. Opposite directions (added/removed/modified) are never averaged."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_positive_rows", type=int, default=5)
    parser.add_argument("--min_participants_with_pos", type=int, default=2)
    parser.add_argument(
        "--phase",
        default="",
        help="Optional phase filter (e.g. evolution, explore).",
    )
    parser.add_argument(
        "--reference_type",
        default="",
        help="Optional reference_type filter (e.g. pool_best_proxy, seed_baseline).",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    if args.phase and "phase" in df.columns:
        df = df[df["phase"] == args.phase].copy()
    if args.reference_type:
        rt_col = "reference_type" if "reference_type" in df.columns else "reference_kind"
        if rt_col in df.columns:
            df = df[df[rt_col] == args.reference_type].copy()
    report = motif_support_report(
        df,
        min_positive_rows=args.min_positive_rows,
        min_participants_with_pos=args.min_participants_with_pos,
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "supported": report["supported_columns"]}, indent=2))


if __name__ == "__main__":
    main()
