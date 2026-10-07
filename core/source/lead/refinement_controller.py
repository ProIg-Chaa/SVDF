"""Small, deterministic helpers for causal TALR refinement control."""

from __future__ import annotations

import math
from collections.abc import Sequence


def nearest_rank_quantile(values: Sequence[float], quantile: float) -> float:
    """Return the deterministic nearest-rank quantile of prior observations."""
    if not values:
        raise ValueError("nearest-rank quantile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(float(quantile) * len(ordered)))
    return ordered[rank - 1]
