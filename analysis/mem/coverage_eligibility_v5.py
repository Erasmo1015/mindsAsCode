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
    parser.add_argument(
        "--phase",
        type=str,
        default="",
        help="If set, filter rows to this phase before counting (never silently pool).",
    )
    parser.add_argument(
        "--by_phase",
        action="store_true",
        help="Write one coverage object per phase under key phase_reports "
        "(in addition to overall on the filtered/full frame).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.phase:
        if "phase" not in df.columns:
            raise SystemExit("--phase set but CSV has no phase column")
        df = df[df["phase"] == args.phase].copy()

    def _unresolved_audit(frame: pd.DataFrame) -> Dict[str, Any]:
        """Retain unresolved NMC rows in coverage; report how many / status breakdown."""
        n = int(len(frame))
        if "exclude_from_construct_effect_fitting" in frame.columns:
            excl = (
                pd.to_numeric(
                    frame["exclude_from_construct_effect_fitting"], errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )
            n_excl = int((excl == 1).sum())
        elif "semantic_resolution_status" in frame.columns:
            n_excl = int(
                (
                    frame["semantic_resolution_status"].fillna("resolved").astype(str)
                    == "nmc_needs_adjudication"
                ).sum()
            )
        else:
            n_excl = 0
        by_status: Dict[str, int] = {}
        if "semantic_resolution_status" in frame.columns:
            by_status = {
                str(k): int(v)
                for k, v in frame["semantic_resolution_status"]
                .fillna("missing")
                .astype(str)
                .value_counts()
                .items()
            }
        return {
            "n_rows_in_coverage": n,
            "n_unresolved_nmc_adjudication": n_excl,
            "semantic_resolution_status_counts": by_status,
            "note": (
                "Unresolved NMC rows are retained in coverage/audit counts; "
                "focal/joint construct-transition fitters exclude them."
            ),
        }

    def _report(frame: pd.DataFrame) -> Dict[str, Any]:
        effects: List[Dict[str, Any]] = []
        for motif in BEHAVIORAL_MOTIFS:
            for direction in DIRECTIONAL_SUFFIXES:
                col = f"{motif}_{direction}"
                elig = eligible_column(motif, direction)
                effects.append(
                    _counts_for_column(
                        frame,
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
        return {
            "schema_hint": 5 if "probability_used_added" in frame.columns else "unknown",
            "n_rows": int(len(frame)),
            "n_participants": int(frame["participant_id"].nunique())
            if "participant_id" in frame.columns
            else None,
            "phase_filter": args.phase or None,
            "phases_present": sorted(frame["phase"].dropna().unique().tolist())
            if "phase" in frame.columns
            else None,
            "unresolved_nmc_audit": _unresolved_audit(frame),
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
                "Choose focals from supported_effects after inspecting this report. "
                "Never silently pool explore and evolution. "
                "Unresolved NMC rows remain in this coverage report."
            ),
        }

    out = _report(df)
    if args.by_phase and "phase" in df.columns and not args.phase:
        out["phase_reports"] = {
            str(ph): _report(df[df["phase"] == ph].copy())
            for ph in sorted(df["phase"].dropna().unique().tolist())
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
