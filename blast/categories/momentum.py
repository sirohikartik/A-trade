"""Momentum sub-score — RSI zone, MACD, rising ADX."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_momentum(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Score RSI, MACD, and ADX momentum signals."""
    rsi = df.get("rsi", 50).fillna(50)
    rsi_min = float(thresholds.get("rsi_min", 55))
    rsi_max = float(thresholds.get("rsi_max", 75))
    mid = (rsi_min + rsi_max) / 2
    half_range = max((rsi_max - rsi_min) / 2, 1e-9)

    in_zone = (rsi >= rsi_min) & (rsi <= rsi_max)
    above_zone = rsi > rsi_max
    dist = np.abs(rsi - mid) / half_range
    rsi_score = np.where(
        in_zone,
        0.45 * (1 - 0.3 * dist),
        np.where(above_zone, 0.10, 0.0),
    )

    macd_b = df.get("macd_bullish", 0).fillna(0).astype(int) == 1
    macd_r = df.get("macd_hist_rising", 0).fillna(0).astype(int) == 1
    adx_r = df.get("adx_rising", 0).fillna(0).astype(int) == 1

    score = (
        rsi_score
        + macd_b.astype(float) * 0.30
        + macd_r.astype(float) * 0.15
        + adx_r.astype(float) * 0.10
    )

    return pd.Series(np.clip(score, 0, 1), index=df.index)
