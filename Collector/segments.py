"""NSE index-based market cap segmentation."""

from __future__ import annotations

import io

import pandas as pd
import requests

import config as cfg


def fetch_index_constituents() -> dict[str, set[str]]:
    urls = {
        "nifty100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
        "midcap150": "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        "smallcap250": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.nseindia.com",
    }
    result: dict[str, set[str]] = {}

    for name, url in urls.items():
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            sym_col = next(c for c in df.columns if "symbol" in c.lower())
            result[name] = set(df[sym_col].astype(str).str.strip())
            print(f"  {name}: {len(result[name])} symbols")
        except Exception as exc:
            print(f"  Failed {name}: {exc}")
            result[name] = set()

    return result


def classify_symbol(symbol: str, constituents: dict[str, set[str]]) -> str:
    if symbol in constituents.get("nifty100", set()):
        return "large_cap"
    if symbol in constituents.get("midcap150", set()):
        return "mid_cap"
    if symbol in constituents.get("smallcap250", set()):
        return "small_cap"
    return "small_cap"


def build_segment_map(symbols: list[str]) -> pd.DataFrame:
    print("Building segment map from NSE index constituents...")
    constituents = fetch_index_constituents()
    rows = [{"symbol": sym, "segment": classify_symbol(sym, constituents)} for sym in symbols]
    df = pd.DataFrame(rows)
    import os

    os.makedirs(cfg.PROCESSED_DIR, exist_ok=True)
    df.to_parquet(cfg.SEGMENT_MAP_PATH, index=False)
    print(df["segment"].value_counts().to_string())
    return df
