#!/usr/bin/env python3
"""Coverage + eligibility counts for participant Schema-v5 analysis CSVs.

Does **not** select a final focal/joint model. Reports, for every construct ×
operation (added/modified/removed):

  - raw positive counts
  - eligible-row counts (eligible_<construct>_<op>==1)
  - within-participant variation (participants with both pos and neg on
    the eligible subset)
  - whether each PRIMARY_FOCAL_CANDIDATE still has support

Risk-set notes:
  - eligible_c_added / _removed: addition / removal risk sets
  - eligible_c_modified: **retained-construct** modification risk set
    (reference_has AND candidate_has), not the broader
    modification_opportunity_c (= reference_has alone)
  - For modified focals, contrast is modified vs retained_unmodified
    within the retained-construct set

Joint models should use ``--eligibility_mode restrict`` (intersection of
these risk sets), not ``fe_adjust`` alone.

Run after build_dataset --schema_version 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mem.schema_participant_transition_v5 import (  # noqa: E402
    BEHAVIORAL_MOTIFS,
    DIRECTIONAL_SUFFIXES,
    PRIMARY_FOCAL_CANDIDATES,
    all_directional_behavioral_columns,
    eligible_column,
)


def _counts_for_column(
    df: pd.DataFrame,
    col: str,
    *,
    elig_col: str | None,
    min_positive_rows: int,
    min_parts_both: int,
) -> Dict[str, Any]:
    if col not in df.columns:
        return {"column": col, "status": "missing_column"}
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    row: Dict[str, Any] = {
        "column": col,
        "n_rows_total": int(len(df)),
        "n_positive_total": int((vals == 1).sum()),
        "n_negative_total": int((vals == 0).sum()),
    }
    work = df
    if elig_col is not None:
        if elig_col not in df.columns:
            row["eligibility_status"] = "missing_eligible_column"
            return row
        elig = pd.to_numeric(df[elig_col], errors="coerce").fillna(0).astype(int)
        row["n_eligible"] = int((elig == 1).sum())
        work = df[elig == 1].copy()
        vals_e = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)
        row["n_positive_eligible"] = int((vals_e == 1).sum())
        row["n_negative_eligible"] = int((vals_e == 0).sum())
    else:
        vals_e = vals
        row["n_eligible"] = int(len(df))
        row["n_positive_eligible"] = row["n_positive_total"]
        row["n_negative_eligible"] = row["n_negative_total"]

    if "participant_id" in work.columns and len(work):
        g = work.assign(_v=vals_e.values).groupby("participant_id")["_v"]
        parts_pos = int((g.max() >= 1).sum())
        parts_both = int(((g.max() >= 1) & (g.min() <= 0)).sum())
        row["n_participants_with_positive_eligible"] = parts_pos
        row["n_participants_with_both_pos_neg_eligible"] = parts_both
        row["has_within_participant_variation"] = parts_both >= int(min_parts_both)
    else:
        row["has_within_participant_variation"] = False

    row["enough_positive_rows"] = row["n_positive_eligible"] >= int(min_positive_rows)
    row["supported_for_focal_consideration"] = bool(
        row["enough_positive_rows"] and row.get("has_within_participant_variation")
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_positive_rows", type=int, default=5)
    parser.add_argument("--min_participants_with_both", type=int, default=2)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    effects: List[Dict[str, Any]] = []
    for motif in BEHAVIORAL_MOTIFS:
        for direction in DIRECTIONAL_SUFFIXES:
            col = f"{motif}_{direction}"
            elig = eligible_column(motif, direction)
            effects.append(
                _counts_for_column(
                    df,
                    col,
                    elig_col=elig,
                    min_positive_rows=args.min_positive_rows,
                    min_parts_both=args.min_participants_with_both,
                )
            )

    primary = []
    for name in PRIMARY_FOCAL_CANDIDATES:
        match = next((e for e in effects if e.get("column") == name), None)
        primary.append(
            {
                "column": name,
                "supported_for_focal_consideration": bool(
                    match and match.get("supported_for_focal_consideration")
                ),
                "detail": match,
            }
        )

    out = {
        "schema_hint": 5 if "probability_used_added" in df.columns else "unknown",
        "n_rows": int(len(df)),
        "n_participants": int(df["participant_id"].nunique())
        if "participant_id" in df.columns
        else None,
        "constructs": list(BEHAVIORAL_MOTIFS),
        "directional_columns_expected": all_directional_behavioral_columns(),
        "effects": effects,
        "primary_focal_candidates": primary,
        "supported_effects": [
            e["column"]
            for e in effects
            if e.get("supported_for_focal_consideration")
        ],
        "note": (
            "Do not hard-code a joint/focal model to every construct-operation. "
            "Choose focals from supported_effects after inspecting this report."
        ),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"[coverage_v5] rows={out['n_rows']} supported={out['supported_effects']} "
        f"wrote {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
