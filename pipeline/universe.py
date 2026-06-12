"""
Rank Nifty 500 symbols by historical edge on the training window.

Uses only train-period rows (no validation leakage) to pick top-N names
for a focused ML/DL dataset.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

_COLLECTOR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Collector")
if _COLLECTOR not in sys.path:
    sys.path.insert(0, _COLLECTOR)

import config as cfg


def _load_settings() -> dict:
    with open(cfg.SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _win_rate(mask: pd.Series, outcome: pd.Series) -> float:
    subset = outcome[mask]
    if len(subset) < 1:
        return 0.0
    return float(subset.mean())


def score_symbol(train_df: pd.DataFrame, weights: dict, min_signals: int) -> dict | None:
    """Per-symbol profitability score from train-period rule buckets."""
    if len(train_df) < min_signals:
        return None

    outcome = train_df["outcome_reference"].astype(float)
    n = len(outcome)

    setup_mask = train_df.get("setup_ready", pd.Series(0, index=train_df.index)) == 1
    breakout_mask = (
        (train_df.get("new_20d_high", 0) == 1)
        & (train_df.get("volume_ratio", 0) >= 2.0)
        & (train_df.get("atr_risk_ok", 0) == 1)
    )
    momentum_mask = (
        (train_df.get("rsi_bull_zone", 0) == 1)
        & (train_df.get("macd_bullish", 0) == 1)
        & (train_df.get("adx_rising", 0) == 1)
    )
    delivery_mask = train_df.get("delivery_strong", 0) == 1

    setup_n = int(setup_mask.sum())
    if setup_n < min_signals:
        return None

    setup_wr = _win_rate(setup_mask, outcome)
    breakout_wr = _win_rate(breakout_mask, outcome)
    momentum_wr = _win_rate(momentum_mask, outcome)
    delivery_wr = _win_rate(delivery_mask, outcome)

    trend_consistency = float((train_df.get("ema_stack_bull", 0) == 1).mean())
    vol_edge = float(train_df.loc[setup_mask, "volume_ratio"].mean()) if setup_n else 0.0
    vol_edge = min(vol_edge / 4.0, 1.0)

    adv = train_df.get("avg_daily_value", train_df["close"] * train_df.get("vol_ma20", 1))
    liquidity = float(np.log1p(adv.median()))
    liquidity_norm = min(liquidity / 22.0, 1.0)

    avg_return = float(train_df.loc[setup_mask, "reference_swing_return_pct"].mean()) if setup_n else 0.0
    return_norm = float(np.clip(avg_return * 10 + 0.5, 0, 1))

    w = weights
    composite = (
        w.get("setup_win_rate", 0.30) * setup_wr
        + w.get("breakout_win_rate", 0.20) * breakout_wr
        + w.get("momentum_win_rate", 0.15) * momentum_wr
        + w.get("delivery_win_rate", 0.10) * delivery_wr
        + w.get("trend_consistency", 0.10) * trend_consistency
        + w.get("volume_edge", 0.05) * vol_edge
        + w.get("liquidity", 0.05) * liquidity_norm
        + w.get("avg_return", 0.05) * return_norm
    )

    return {
        "symbol": train_df["symbol"].iloc[0],
        "segment": train_df["segment"].iloc[0] if "segment" in train_df.columns else "unknown",
        "train_rows": n,
        "setup_signals": setup_n,
        "setup_win_rate": round(setup_wr, 4),
        "breakout_win_rate": round(breakout_wr, 4),
        "momentum_win_rate": round(momentum_wr, 4),
        "delivery_win_rate": round(delivery_wr, 4),
        "trend_consistency": round(trend_consistency, 4),
        "volume_edge": round(vol_edge, 4),
        "liquidity_score": round(liquidity_norm, 4),
        "avg_setup_return": round(avg_return, 4),
        "composite_score": round(composite, 4),
    }


def rank_universe(df: pd.DataFrame) -> pd.DataFrame:
    settings = _load_settings()
    uni = settings.get("universe", {})
    weights = uni.get("ranking_weights", {})
    min_signals = int(uni.get("min_setup_signals", 8))
    if cfg.fetch_recent_days() is not None:
        min_signals = max(1, min(min_signals, 2))

    from pipeline.splits import mask_split, resolve_splits

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dev_mode = cfg.fetch_recent_days() is not None
    bounds = resolve_splits(df, dev_mode=dev_mode)
    train = df[mask_split(df, "train", bounds)]

    rows: list[dict] = []
    for symbol, grp in train.groupby("symbol"):
        scored = score_symbol(grp, weights, min_signals)
        if scored:
            rows.append(scored)

    if not rows:
        raise RuntimeError("No symbols passed ranking thresholds on train data.")

    ranked = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    ranked["rank"] = range(1, len(ranked) + 1)

    out_path = os.path.join(cfg.PROCESSED_DIR, "universe_rankings.parquet")
    os.makedirs(cfg.PROCESSED_DIR, exist_ok=True)
    ranked.to_parquet(out_path, index=False)
    return ranked


def select_top_n(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    ranked = rank_universe(df)
    top_n = min(top_n, len(ranked))
    selected = ranked.head(top_n)
    symbols = selected["symbol"].tolist()

    filtered = df[df["symbol"].isin(symbols)].copy()

    meta = {
        "top_n": top_n,
        "symbols": symbols,
        "ranking_path": os.path.join(cfg.PROCESSED_DIR, "universe_rankings.parquet"),
        "selection_rules": _load_settings().get("universe", {}),
        "top_symbols": selected.to_dict(orient="records"),
    }
    meta_path = cfg.UNIVERSE_TOP_N_PATH
    os.makedirs(cfg.DATASETS_DIR, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    filtered_path = os.path.join(cfg.SEGMENTS_DIR, f"top_{top_n}.parquet")
    filtered.to_parquet(filtered_path, index=False)

    print(f"Universe: selected top {top_n} symbols (of {len(ranked)} ranked)")
    print(f"  Rankings: {meta['ranking_path']}")
    print(f"  Selection: {meta_path}")
    print(f"  Filtered rows: {len(filtered)}")
    for _, row in selected.head(5).iterrows():
        print(
            f"    #{int(row['rank'])} {row['symbol']:<12} "
            f"score={row['composite_score']:.3f} setup_wr={row['setup_win_rate']:.1%}"
        )

    return filtered, symbols, selected
