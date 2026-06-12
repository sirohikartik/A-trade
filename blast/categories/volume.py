"""Volume surge sub-score — volume ratio vs 20-day average."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_volume(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Score elevated volume relative to recent average."""
    vr = df.get("volume_ratio", 0).fillna(0)
    min_vr = float(thresholds.get("volume_ratio_min", 2.0))

    base = np.where(
        vr < min_vr,
        0.0,
        0.4 + 0.6 * np.clip((vr - min_vr) / (min_vr * 1.5), 0, 1),
    )
    breakout = df.get("breakout_volume", 0).fillna(0).astype(int) == 1
    vol_up = df.get("vol_trend_up", 0).fillna(0).astype(int) == 1
    score = base + breakout.astype(float) * 0.10 + vol_up.astype(float) * 0.05

    return pd.Series(np.clip(score, 0, 1), index=df.index)
