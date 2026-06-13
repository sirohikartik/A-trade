"""Breakout quality sub-score — 52w/20d highs and close position."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_breakout(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Score proximity to highs and intraday close strength."""
    new_52 = df.get("new_52w_high", 0).fillna(0).astype(int) == 1
    near_52 = df.get("near_52w_high", 0).fillna(0).astype(int) == 1
    near_20 = df.get("near_20d_high", 0).fillna(0).astype(int) == 1
    cp = df.get("close_position", 0.5).fillna(0.5)

    score = np.where(new_52, 0.50, np.where(near_52, 0.35, 0.0))
    score = score + near_20.astype(float) * 0.30
    score = score + np.where(cp >= 0.7, 0.20, 0.0)

    return pd.Series(np.clip(score, 0, 1), index=df.index)
