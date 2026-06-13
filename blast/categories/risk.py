"""Risk / liquidity sub-score — ATR extension, overextension, liquidity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_risk(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Score favorable risk-reward and liquidity (higher = lower risk)."""
    atr_pct = df.get("atr_pct", 0).fillna(0)
    max_atr = float(thresholds.get("atr_pct_max", 0.06))
    max_pullback = float(thresholds.get("max_pullback_ema20", 0.025))

    atr_ok = df.get("atr_risk_ok", 0).fillna(0).astype(int) == 1
    score = atr_ok.astype(float) * 0.50

    excess = np.where(atr_pct > max_atr, (atr_pct - max_atr) / max_atr, 0.0)
    score = score - np.clip(excess, 0, 1) * 0.30

    pullback = df.get("pullback_ema20", 0).fillna(0)
    pull_pen = np.where(pullback > max_pullback, np.clip((pullback - max_pullback) / max_pullback, 0, 1), 0.0)
    score = score - pull_pen * 0.20

    adv = df.get("avg_daily_value", df.get("close", 0) * df.get("vol_ma20", 1)).fillna(0)
    liquidity = np.log1p(adv)
    liquidity_norm = np.clip(liquidity / 22.0, 0, 1)
    score = score + liquidity_norm * 0.30

    return pd.Series(np.clip(score, 0, 1), index=df.index)
