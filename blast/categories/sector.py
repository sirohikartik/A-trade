"""Sector strength sub-score — sector breadth (% above EMA20)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_sector(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Score relative sector participation using ``sector_breadth``."""
    breadth = df.get("sector_breadth", 0.5).fillna(0.5)
    low = float(thresholds.get("sector_breadth_low", 0.40))
    high = float(thresholds.get("sector_breadth_high", 0.70))
    span = max(high - low, 1e-9)

    score = np.clip((breadth - low) / span, 0, 1)
    return pd.Series(score, index=df.index)
