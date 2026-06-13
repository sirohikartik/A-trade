"""Live blast scoring for the Nifty 500 scanner.

Transforms per-symbol indicator rows from ``Collector/start.py`` into a
scored, filtered batch using segment-specific tuned weights and alert bands.
"""

from __future__ import annotations

import os

import pandas as pd

from blast._config import ATRADE_ROOT, load_blast_config, load_segment_weights
from blast.enrich import load_sector_map
from blast.filters import eligible_mask_by_segment
from blast.scoring import compute_blast_score, grade_alerts


def load_segment_map() -> dict[str, str]:
    """Symbol → large_cap / mid_cap / small_cap from processed segment map."""
    path = os.path.join(ATRADE_ROOT, "data", "processed", "segment_map.parquet")
    if os.path.isfile(path):
        df = pd.read_parquet(path)
        return dict(zip(df["symbol"].astype(str), df["segment"].astype(str)))
    return {}


def prepare_latest_rows(rows: list[pd.Series]) -> pd.DataFrame:
    """Combine per-symbol latest indicator rows into one dated frame."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        df["date"] = pd.Timestamp.today().normalize()
    else:
        df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    if "vol_ma20" in df.columns and "avg_daily_value" not in df.columns:
        df["avg_daily_value"] = df["close"].astype(float) * df["vol_ma20"].astype(float)
    seg_map = load_segment_map()
    df["segment"] = df["symbol"].map(seg_map).fillna("small_cap")
    return df


def enrich_live_breadths(df: pd.DataFrame) -> pd.DataFrame:
    """Add sector, market_breadth, and sector_breadth for a single scan snapshot."""
    out = df.copy()
    sector_map = load_sector_map()
    out["sector"] = out["symbol"].map(sector_map).fillna("Unknown")
    if "above_ema50" in out.columns:
        out["market_breadth"] = float(out["above_ema50"].fillna(0).mean())
    else:
        out["market_breadth"] = 0.5
    above = out.get("above_ema20", (out["close"] > out["ema20"]).astype(int)).fillna(0)
    out["_above_ema20"] = above.astype(int)
    out["sector_breadth"] = out.groupby("sector")["_above_ema20"].transform("mean")
    return out.drop(columns=["_above_ema20"])


def min_alert_score() -> float:
    """Minimum blast_score to surface an alert (default: B band = 65)."""
    bands = load_blast_config().get("alert_bands", {})
    return float(bands.get("b", 65))


def score_scan_batch(df: pd.DataFrame, min_score: float | None = None) -> pd.DataFrame:
    """Score all symbols; keep rows that pass eligibility and min alert band."""
    if df.empty:
        return df

    min_score = min_alert_score() if min_score is None else min_score
    enriched = enrich_live_breadths(df)
    scored_parts: list[pd.DataFrame] = []

    for segment, seg_df in enriched.groupby("segment", sort=False):
        weights = load_segment_weights(str(segment))
        part = compute_blast_score(seg_df.copy(), weights=weights)
        part["alert_grade"] = grade_alerts(part["blast_score"])
        scored_parts.append(part)

    scored = pd.concat(scored_parts, ignore_index=True)
    eligible = eligible_mask_by_segment(scored)
    passed = scored.loc[eligible & (scored["blast_score"] >= min_score)].copy()
    return passed.sort_values("blast_score", ascending=False).reset_index(drop=True)
