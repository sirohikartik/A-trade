"""Liquidity and macro filters — separate from blast_score (never zero the score)."""

from __future__ import annotations

import pandas as pd

from blast._config import get_filter_config


def eligible_mask(df: pd.DataFrame, segment: str | None = None) -> pd.Series:
    """Rows that may enter the daily ranking pool."""
    cfg = get_filter_config()
    mask = pd.Series(True, index=df.index)

    min_price = cfg["min_price"]
    if min_price > 0 and "close" in df.columns:
        mask &= df["close"].fillna(0) >= min_price

    adv_floor = cfg.get("min_avg_daily_value", {})
    if segment and segment in adv_floor:
        floor = float(adv_floor[segment])
        if floor > 0 and "avg_daily_value" in df.columns:
            mask &= df["avg_daily_value"].fillna(0) >= floor

    if cfg.get("use_macro_filter", True) and "market_breadth" in df.columns:
        min_breadth = cfg["min_market_breadth"]
        mask &= df["market_breadth"].fillna(0.5) >= min_breadth

    return mask


def eligible_mask_by_segment(df: pd.DataFrame) -> pd.Series:
    """Per-row eligibility when the frame mixes cap segments."""
    if "segment" not in df.columns:
        return eligible_mask(df)

    out = pd.Series(False, index=df.index)
    for segment, grp in df.groupby("segment", sort=False):
        out.loc[grp.index] = eligible_mask(grp, segment=str(segment)).to_numpy()
    return out
