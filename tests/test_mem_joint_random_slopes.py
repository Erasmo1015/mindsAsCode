"""Tests for joint multi-motif random-slope MixedLM fitter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.mem.fit_mem_joint_random_slopes import (
    DEFAULT_FIXED_EFFECTS,
    fit_joint_random_slopes,
    prepare_joint_frame,
    write_outputs,
)


def _synth_df(n_participants: int = 12, n_per: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    motifs = [c for c in DEFAULT_FIXED_EFFECTS if c != "iteration"]
    slopes = ["history_added", "value_added", "risk_added"]
    for pid in range(n_participants):
        b = rng.normal(0, 0.25, size=len(slopes))
        for t in range(n_per):
            x = {s: int(rng.random() < 0.35) for s in motifs}
            # force both levels early in each participant for RE slopes
            if t < 2:
                x["history_added"] = t
                x["value_added"] = 1 - t
                x["risk_added"] = t
            mu = 0.1 + 0.4 * x["history_added"] + 0.25 * x["value_added"] + 0.15 * x["risk_added"]
            mu += float(
                b[0] * x["history_added"] + b[1] * x["value_added"] + b[2] * x["risk_added"]
            )
            rows.append(
                {
                    "participant_id": str(pid),
                    "delta_f": mu + rng.normal(0, 0.35),
                    "iteration": float(t),
                    "phase": "evolution",
                    **x,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_fingerprint_and_constant_reject():
    df = _synth_df()
    work, meta = prepare_joint_frame(
        df,
        fixed_effects=DEFAULT_FIXED_EFFECTS,
        random_slopes=["history_added", "value_added"],
        phase="evolution",
    )
    assert meta["n_rows"] == len(work)
    assert len(meta["fingerprint_sha256"]) == 64

    bad = df.copy()
    bad["risk_added"] = 0
    with pytest.raises(ValueError, match="constant"):
        prepare_joint_frame(
            bad,
            fixed_effects=DEFAULT_FIXED_EFFECTS,
            random_slopes=["history_added", "risk_added"],
            phase="evolution",
        )


def test_joint_fit_ordered_labels_and_theta(tmp_path: Path):
    df = _synth_df()
    slopes = ["history_added", "value_added"]
    fit = fit_joint_random_slopes(
        df,
        random_slopes=slopes,
        fixed_effects=DEFAULT_FIXED_EFFECTS,
        methods=("lbfgs",),
        primary_method="lbfgs",
        reml=True,
        maxiter=100,
        phase="evolution",
    )
    assert fit["status"] in ("ok", "not_converged", "singular_or_boundary")
    assert fit["re_labels"][0] in ("Group", "Intercept") or "Group" in fit["re_labels"][0]
    # RE labels should include slopes in order after intercept
    assert fit["random_slopes"] == slopes
    part = fit["participant_effects"]
    assert "theta_history_added" in part.columns
    # theta = mu + b
    mu = next(
        r["coef"]
        for r in fit["fixed_effects_table"]
        if r["term"] == "history_added"
    )
    recon = part["b_hat_history_added"] + mu
    assert np.allclose(recon, part["theta_history_added"], atol=1e-8)

    out = tmp_path / "joint_out"
    write_outputs(fit, out)
    assert (out / "fit_diagnostics.json").is_file()
    assert (out / "participant_effects.csv").is_file()
    assert (out / "cov_re_heatmap.png").is_file()
    diag = json.loads((out / "fit_diagnostics.json").read_text())
    assert diag["prep"]["fingerprint_sha256"] == fit["prep"]["fingerprint_sha256"]


def test_identical_row_fingerprint_across_slope_subsets():
    df = _synth_df()
    _, m1 = prepare_joint_frame(
        df,
        fixed_effects=DEFAULT_FIXED_EFFECTS,
        random_slopes=["history_added", "value_added"],
        phase="evolution",
    )
    _, m2 = prepare_joint_frame(
        df,
        fixed_effects=DEFAULT_FIXED_EFFECTS,
        random_slopes=["history_added", "value_added", "risk_added"],
        phase="evolution",
    )
    # Same FE motif set → same content columns → same fingerprint
    assert m1["fingerprint_sha256"] == m2["fingerprint_sha256"]
    assert m1["n_rows"] == m2["n_rows"]
