"""Tests for BH-FDR utility used by MEM focal and joint fits."""

from __future__ import annotations

import math

from analysis.mem.bh_fdr import bh_fdr


def test_bh_fdr_known_vector():
    # Classic example: p = [0.01, 0.04, 0.03, 0.005] with m=4
    # Sorted: 0.005, 0.01, 0.03, 0.04
    # Raw BH: 0.005*4/1=0.02, 0.01*4/2=0.02, 0.03*4/3=0.04, 0.04*4/4=0.04
    # Monotone from largest p: all become 0.02, 0.02, 0.04, 0.04 after cummin from end
    # Actually from largest: q4=0.04, q3=min(0.04,0.04)=0.04, q2=min(0.04,0.02)=0.02, q1=min(0.02,0.02)=0.02
    p = [0.01, 0.04, 0.03, 0.005]
    q = bh_fdr(p)
    assert q[3] == 0.02  # 0.005
    assert abs(q[0] - 0.02) < 1e-12
    assert abs(q[2] - 0.04) < 1e-12
    assert abs(q[1] - 0.04) < 1e-12


def test_bh_fdr_preserves_none_and_nan():
    p = [0.01, None, float("nan"), 0.04]
    q = bh_fdr(p)
    assert q[1] is None
    assert q[2] is None
    assert q[0] is not None and q[3] is not None
    assert q[0] <= q[3] or math.isclose(q[0], q[3])


def test_bh_fdr_not_constant_for_spread_pvalues():
    """Regression: broken ascending min-pass collapsed all q to m*p_min."""
    p = [1e-5, 0.2, 0.5, 0.9]
    q = bh_fdr(p)
    assert len(set(round(x, 12) for x in q if x is not None)) > 1
    # Smallest p should get the smallest q
    assert q[0] == min(x for x in q if x is not None)
