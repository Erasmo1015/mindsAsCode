#!/usr/bin/env python3
"""Compare joint multi-motif θ̂ against focal one-motif participant slopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> Optional[float]:
    if len(a) < 3:
        return None
    s = pd.Series(a)
    t = pd.Series(b)
    if method == "pearson":
        v = s.corr(t, method="pearson")
    else:
        v = s.corr(t, method="spearman")
    return None if pd.isna(v) else float(v)


def compare_slope(
    joint_df: pd.DataFrame,
    focal_path: Path,
    slope: str,
    focal_var: Optional[float],
    joint_var: Optional[float],
) -> Dict[str, Any]:
    if not focal_path.is_file():
        return {"slope": slope, "status": "missing_focal", "path": str(focal_path)}
    focal = pd.read_csv(focal_path)
    focal["participant_id"] = focal["participant_id"].astype(str)
    j = joint_df[["participant_id", f"theta_{slope}"]].rename(
        columns={f"theta_{slope}": "theta_joint"}
    )
    j["participant_id"] = j["participant_id"].astype(str)
    f = focal.rename(columns={"participant_slope": "theta_focal"})[
        ["participant_id", "theta_focal"]
    ]
    m = j.merge(f, on="participant_id", how="inner")
    if m.empty:
        return {"slope": slope, "status": "no_overlap"}
    a = m["theta_joint"].to_numpy(dtype=float)
    b = m["theta_focal"].to_numpy(dtype=float)
    sign_flip = int(np.sum(np.sign(a) * np.sign(b) < 0))
    return {
        "slope": slope,
        "status": "ok",
        "n": int(len(m)),
        "pearson": _corr(a, b, "pearson"),
        "spearman": _corr(a, b, "spearman"),
        "mae": float(np.mean(np.abs(a - b))),
        "sign_flips": sign_flip,
        "sign_flip_rate": float(sign_flip / len(m)),
        "joint_mean_theta": float(np.mean(a)),
        "focal_mean_theta": float(np.mean(b)),
        "joint_random_slope_variance": joint_var,
        "focal_random_slope_variance": focal_var,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint_dir", required=True, help="One joint model output dir")
    p.add_argument(
        "--focal_dir",
        required=True,
        help="Focal fit_random_slopes_normal directory",
    )
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    joint_dir = Path(args.joint_dir)
    focal_dir = Path(args.focal_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    part = pd.read_csv(joint_dir / "participant_effects.csv")
    diag = json.loads((joint_dir / "fit_diagnostics.json").read_text(encoding="utf-8"))
    slopes: List[str] = list(diag.get("random_slopes") or [])

    # Focal variances from coefficients table
    coef_path = focal_dir / "random_slope_coefficients.csv"
    focal_var_map: Dict[str, Optional[float]] = {}
    if coef_path.is_file():
        coef = pd.read_csv(coef_path)
        for _, r in coef.iterrows():
            focal_var_map[str(r["focal"])] = (
                None
                if pd.isna(r.get("random_slope_variance"))
                else float(r["random_slope_variance"])
            )

    # Joint variances from cov_re diag (skip intercept = index 0)
    cov = np.asarray(diag.get("cov_re"), dtype=float)
    labels = list(diag.get("re_labels") or [])
    joint_var_map: Dict[str, Optional[float]] = {}
    for i, lab in enumerate(labels):
        if lab == "Group" or lab.lower() == "intercept":
            continue
        joint_var_map[lab] = float(cov[i, i]) if i < cov.shape[0] else None

    rows = []
    for s in slopes:
        rows.append(
            compare_slope(
                part,
                focal_dir / f"participant_slopes_{s}.csv",
                s,
                focal_var_map.get(s),
                joint_var_map.get(s),
            )
        )

    out_csv = out_dir / f"comparison_{joint_dir.name}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "joint_dir": str(joint_dir),
        "focal_dir": str(focal_dir),
        "n_slopes": len(rows),
        "rows": rows,
    }
    (out_dir / f"comparison_{joint_dir.name}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
