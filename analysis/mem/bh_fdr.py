"""Benjamini–Hochberg FDR correction for MEM motif tests.

Uses statsmodels' vectorized multipletests. Preserves None / non-finite slots
so skipped models do not enter the correction family.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


def bh_fdr(pvals: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Return BH-adjusted q-values aligned with ``pvals``.

    Finite p-values form one correction family. ``None`` and non-finite entries
    remain ``None`` and are excluded from the family size.
    """
    out: List[Optional[float]] = [None] * len(pvals)
    indexed = [
        (i, float(p))
        for i, p in enumerate(pvals)
        if p is not None and math.isfinite(float(p))
    ]
    if not indexed:
        return out

    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "statsmodels is required for BH-FDR. Install with: pip install statsmodels"
        ) from exc

    idxs = [i for i, _ in indexed]
    vals = [p for _, p in indexed]
    _reject, qvals, _alphacSidak, _alphacBonf = multipletests(vals, method="fdr_bh")
    for i, q in zip(idxs, qvals):
        out[i] = float(q)
    return out


# Backward-compatible alias used by fit_mem_random_slopes
_bh_fdr = bh_fdr
