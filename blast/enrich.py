"""Load processed segment parquets and derive sector breadth.

``sector_breadth`` is the fraction of symbols above EMA20 within each
(date, sector) group — required by the sector category scorer.
"""

from __future__ import annotations

import os

import pandas as pd

from blast._config import ATRADE_ROOT, SECTOR_MAP_PATH


def load_sector_map() -> dict[str, str]:
    """Return symbol → sector mapping from ``data/sector_map.csv``."""
    if not os.path.isfile(SECTOR_MAP_PATH):
        return {}
    df = pd.read_csv(SECTOR_MAP_PATH)
    df = df.dropna(subset=["symbol"])
    sectors = df["sector"].fillna("Unknown").astype(str)
    return dict(zip(df["symbol"].astype(str), sectors))


def add_sector_breadth(df: pd.DataFrame, sector_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Attach ``sector`` and ``sector_breadth`` columns for blast scoring."""
    sector_map = sector_map or load_sector_map()
    out = df.copy()
    out["sector"] = out["symbol"].astype(str).map(sector_map).fillna("Unknown")

    above = out.get("above_ema20", (out["close"] > out["ema20"]).astype(int)).fillna(0)
    out["_above_ema20"] = above.astype(int)
    out["sector_breadth"] = out.groupby(["date", "sector"])["_above_ema20"].transform("mean")
    out = out.drop(columns=["_above_ema20"])
    return out


def load_segment_frame(segment: str | None = None) -> pd.DataFrame:
    """Load a segment parquet (or ``all``) with sector breadth attached."""
    seg_dir = os.path.join(ATRADE_ROOT, "data", "processed", "segments")
    if segment:
        path = os.path.join(seg_dir, f"{segment}.parquet")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
    else:
        path = os.path.join(seg_dir, "all.parquet")
        df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return add_sector_breadth(df)
