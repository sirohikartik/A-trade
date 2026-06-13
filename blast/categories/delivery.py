"""Delivery accumulation sub-score — NSE delivery % vs thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_delivery(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Score delivery participation when bhavcopy data is present."""
    dpct = df.get("delivery_pct", pd.Series(np.nan, index=df.index))
    min_d = float(thresholds.get("delivery_pct_min", 50))

    has_data = dpct.notna() & (dpct >= min_d)
    score = np.where(
        has_data,
        0.3 + 0.7 * np.clip((dpct - min_d) / (80 - min_d), 0, 1),
        0.0,
    )
    above = df.get("delivery_above_avg", 0).fillna(0).astype(int) == 1
    score = np.where(has_data, np.clip(score + above.astype(float) * 0.20, 0, 1), 0.0)

    return pd.Series(score, index=df.index)
