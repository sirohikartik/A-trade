"""Compute indicators, merge delivery, label outcomes."""

from __future__ import annotations

import glob
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yaml
from tqdm import tqdm

_COLLECTOR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Collector")
if _COLLECTOR not in sys.path:
    sys.path.insert(0, _COLLECTOR)

import config as cfg
import db as store
from delivery import load_all_delivery
from indicators import add_delivery_features, add_indicators
from labeling import label_stock
from segments import build_segment_map


def _load_settings() -> dict:
    with open(cfg.SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _process_symbol_pass1(
    symbol: str,
    segment_map: dict,
    delivery_df: pd.DataFrame,
    min_history: int,
    min_price: float,
    min_adv_map: dict,
) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None]:
    raw = store.load_ohlcv(cfg.DB_PATH, symbol, min_bars=1)
    if raw is None or len(raw) < min_history:
        return symbol, None, None

    df = raw.reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = symbol
    df = add_indicators(df)

    if not delivery_df.empty:
        sym_del = delivery_df[delivery_df["symbol"] == symbol][
            ["date", "delivery_qty", "delivery_pct"]
        ]
        df = df.merge(sym_del, on="date", how="left")

    df = add_delivery_features(df)
    df["avg_daily_value"] = df["close"] * df["vol_ma20"]
    segment = segment_map.get(symbol, "small_cap")
    df["segment"] = segment

    if min_price > 0:
        df = df[df["close"] >= min_price]

    min_adv = float(min_adv_map.get(segment, 0))
    if min_adv > 0 and cfg.fetch_recent_days() is None:
        df = df[df["avg_daily_value"].fillna(0) >= min_adv]

    if len(df) < max(5, min_history // 10):
        return symbol, None, None

    breadth = df[["date", "above_ema50"]].copy()
    return symbol, df, breadth


def _process_symbol_pass2(
    fpath: str,
    market_breadth: pd.DataFrame,
    segment_map: dict,
) -> pd.DataFrame | None:
    df = pd.read_parquet(fpath)
    symbol = os.path.basename(fpath).replace(".parquet", "")
    segment = df["segment"].iloc[0] if "segment" in df.columns else segment_map.get(symbol, "small_cap")

    if "market_breadth" in df.columns:
        df = df.drop(columns=["market_breadth"])
    df = df.merge(market_breadth, on="date", how="left")
    df["market_breadth"] = df["market_breadth"].fillna(0.5)
    df.to_parquet(fpath, index=False)

    labeled = label_stock(df, segment)
    return labeled if len(labeled) > 0 else None


def process_all_symbols(symbols: list[str] | None = None) -> None:
    """Build segment parquets: indicators, delivery, reference labels, breadth."""
    settings = _load_settings()
    filters = settings.get("filters", {})
    min_history = cfg.min_bars_required()
    min_price = float(filters.get("min_price", 20))
    min_adv_map = filters.get("min_avg_daily_value", {})

    if symbols is None:
        symbols = store.list_symbols(cfg.DB_PATH)

    if not symbols:
        raise RuntimeError("No OHLCV symbols in DuckDB. Run fetch step first.")

    segment_df = (
        pd.read_parquet(cfg.SEGMENT_MAP_PATH)
        if os.path.isfile(cfg.SEGMENT_MAP_PATH)
        else build_segment_map(symbols)
    )
    segment_map = dict(zip(segment_df["symbol"], segment_df["segment"]))

    delivery_df = load_all_delivery()
    if os.path.isdir(cfg.COMBINED_DIR):
        shutil.rmtree(cfg.COMBINED_DIR)
    os.makedirs(cfg.COMBINED_DIR, exist_ok=True)
    os.makedirs(cfg.SEGMENTS_DIR, exist_ok=True)

    breadth_chunks: list[pd.DataFrame] = []
    processed = 0

    workers = min(cfg.WORKERS, max(1, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_symbol_pass1,
                symbol,
                segment_map,
                delivery_df,
                min_history,
                min_price,
                min_adv_map,
            ): symbol
            for symbol in symbols
        }
        with tqdm(total=len(futures), desc="Indicators", unit="sym") as bar:
            for future in as_completed(futures):
                symbol, df, breadth = future.result()
                if df is not None and breadth is not None:
                    df.to_parquet(os.path.join(cfg.COMBINED_DIR, f"{symbol}.parquet"), index=False)
                    breadth_chunks.append(breadth)
                    processed += 1
                bar.update(1)
                bar.set_postfix(ok=processed)

    print(f"Pass 1: {processed} symbols with indicators")

    if not breadth_chunks:
        print("No symbols processed.")
        return

    all_breadth = pd.concat(breadth_chunks)
    market_breadth = all_breadth.groupby("date")["above_ema50"].mean().reset_index()
    market_breadth.rename(columns={"above_ema50": "market_breadth"}, inplace=True)
    market_breadth.to_parquet(cfg.MARKET_BREADTH_PATH, index=False)

    files = [
        os.path.join(cfg.COMBINED_DIR, f"{sym}.parquet")
        for sym in symbols
        if os.path.isfile(os.path.join(cfg.COMBINED_DIR, f"{sym}.parquet"))
    ]
    labeled_chunks: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_symbol_pass2, fpath, market_breadth, segment_map): fpath
            for fpath in files
        }
        with tqdm(total=len(futures), desc="Labeling", unit="sym") as bar:
            for future in as_completed(futures):
                labeled = future.result()
                if labeled is not None:
                    labeled_chunks.append(labeled)
                bar.update(1)
                bar.set_postfix(labeled=len(labeled_chunks))

    if not labeled_chunks:
        print("No labeled rows produced.")
        return

    full = pd.concat(labeled_chunks, ignore_index=True)
    for seg in ("large_cap", "mid_cap", "small_cap"):
        seg_df = full[full["segment"] == seg]
        if len(seg_df) == 0:
            continue
        seg_df.to_parquet(os.path.join(cfg.SEGMENTS_DIR, f"{seg}.parquet"), index=False)
        print(
            f"  {seg}: {len(seg_df)} rows | "
            f"ref={seg_df['outcome_reference'].mean():.1%}"
        )

    full.to_parquet(os.path.join(cfg.SEGMENTS_DIR, "all.parquet"), index=False)
