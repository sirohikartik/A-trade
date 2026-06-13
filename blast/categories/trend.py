"""Trend strength sub-score — EMA stack, slope, ADX."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_trend(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Score trend alignment from EMA stack, slope, and ADX strength."""
    is_bull = df.get("ema_stack_bull", 0).fillna(0).astype(int) == 1
    partial = (
        df.get("above_ema200", 0).fillna(0).astype(float) * 0.15
        + df.get("above_ema50", 0).fillna(0).astype(float) * 0.10
        + df.get("above_ema20", 0).fillna(0).astype(float) * 0.05
    )
    score = np.where(is_bull, 0.50, partial)

    slope = df.get("ema20_slope", 0).fillna(0)
    score = score + np.clip(slope * 5, 0, 0.25)

    adx = df.get("adx_strong", 0).fillna(0).astype(int) == 1
    score = score + adx.astype(float) * 0.25

    return pd.Series(np.clip(score, 0, 1), index=df.index)
