"""
NSE Breakout Scanner

1. Load Nifty 500 symbols from NSE
2. Download daily OHLCV (Yahoo Finance) → DuckDB
3. Apply technical filters and save passing symbols to DuckDB

Tune fetch length and filters in config.py
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

_COLLECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if _COLLECTOR_DIR not in sys.path:
    sys.path.insert(0, _COLLECTOR_DIR)

import config as cfg
import db as store

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

REQUIRED_COLS = (
    "close",
    "volume",
    "EMA20",
    "EMA50",
    "EMA200",
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "ADX",
    "ATR",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def print_settings() -> None:
    print("=" * 50)
    print("SETTINGS (edit Collector/config.py to change)")
    print("=" * 50)
    print(f"  Years of history     : {cfg.YEARS_OF_HISTORY}")
    print(f"  Calendar buffer days : {cfg.CALENDAR_BUFFER_DAYS}")
    print(f"  Lookback days (total): {cfg.LOOKBACK_DAYS}")
    print(f"  Min bars per symbol  : {cfg.MIN_BARS}")
    print(f"  Force refresh        : {cfg.FORCE_REFRESH_ON_RUN}")
    print(f"  Download workers     : {cfg.WORKERS}")
    print(f"  Min price (INR)      : {cfg.MIN_PRICE}")
    print(f"  Min volume           : {cfg.MIN_VOLUME:,}")
    print(f"  Max ATR % of price   : {cfg.MAX_ATR_PCT * 100:.1f}%")
    print(f"  Min pass score       : {cfg.MIN_SCORE}")
    print(f"  Database             : {cfg.DB_PATH}")
    print("=" * 50)
    print()


def ensure_dirs() -> None:
    os.makedirs(os.path.dirname(cfg.DB_PATH), exist_ok=True)


def get_nifty500_symbols() -> list[str]:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=20)

    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(BytesIO(resp.content))
    col = next(c for c in df.columns if c.strip().lower() == "symbol")
    symbols = df[col].astype(str).str.strip()
    return symbols[
        symbols.ne("")
        & symbols.ne("nan")
        & ~symbols.str.upper().str.startswith("DUMMY")
    ].tolist()


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df[["open", "high", "low", "close", "volume"]].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    return df.sort_index()


def fetch_history_from_yahoo(yf_symbol: str) -> pd.DataFrame | None:
    end = date.today()
    start = end - timedelta(days=cfg.LOOKBACK_DAYS)

    df = yf.download(
        yf_symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=False,
    )

    if df is None or df.empty:
        return None

    df = _normalize_ohlcv(df)
    return df if len(df) >= cfg.MIN_BARS else None


def save_history(symbol: str, df: pd.DataFrame) -> None:
    store.save_ohlcv(cfg.DB_PATH, symbol, df)


def load_history(symbol: str) -> pd.DataFrame | None:
    df = store.load_ohlcv(cfg.DB_PATH, symbol, cfg.MIN_BARS)
    return _normalize_ohlcv(df) if df is not None else None


def get_history(symbol: str, yf_symbol: str, force_refresh: bool = False) -> pd.DataFrame | None:
    if not force_refresh:
        cached = load_history(symbol)
        if cached is not None:
            return cached

    df = fetch_history_from_yahoo(yf_symbol)
    if df is not None:
        save_history(symbol, df)
    return df


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr = _atr(high, low, close, length)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / length, min_periods=length, adjust=False
    ).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / length, min_periods=length, adjust=False
    ).mean() / atr

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = _ema(df["close"], 20)
    df["EMA50"] = _ema(df["close"], 50)
    df["EMA200"] = _ema(df["close"], 200)
    df["RSI"] = _rsi(df["close"], 14)

    macd_line, signal = _macd(df["close"])
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = signal
    df["ADX"] = _adx(df["high"], df["low"], df["close"], 14)
    df["ATR"] = _atr(df["high"], df["low"], df["close"], 14)
    return df


def latest_row_valid(latest: pd.Series) -> bool:
    for col in REQUIRED_COLS:
        val = latest.get(col)
        if val is None or pd.isna(val):
            return False
    return True


def scan_symbol(df: pd.DataFrame, symbol: str) -> dict | None:
    df = add_indicators(df)
    latest = df.iloc[-1]

    if not latest_row_valid(latest):
        return None
    if latest["close"] < cfg.MIN_PRICE:
        return None
    if latest["volume"] < cfg.MIN_VOLUME:
        return None

    atr_pct = float(latest["ATR"]) / float(latest["close"])
    if atr_pct > cfg.MAX_ATR_PCT:
        return None

    if not (
        latest["close"] > latest["EMA20"] > latest["EMA50"] > latest["EMA200"]
    ):
        return None

    high20 = df["high"].iloc[-21:-1].max()
    if latest["close"] <= high20:
        return None

    avg_vol20 = df["volume"].tail(20).mean()
    if avg_vol20 <= 0:
        return None

    volume_ratio = float(latest["volume"]) / float(avg_vol20)
    if volume_ratio <= 2:
        return None

    rsi = float(latest["RSI"])
    if not (55 <= rsi <= 75):
        return None
    if latest["MACD"] <= latest["MACD_SIGNAL"]:
        return None
    if latest["ADX"] <= 20:
        return None

    score = min(volume_ratio * 10, 25) + min(float(latest["ADX"]), 25)
    score += max(0.0, rsi - 50) + 10
    if score < cfg.MIN_SCORE:
        return None

    first_date = df.index.min()
    last_date = df.index.max()

    return {
        "symbol": symbol,
        "score": round(score, 2),
        "close": round(float(latest["close"]), 2),
        "volume_ratio": round(volume_ratio, 2),
        "rsi": round(rsi, 2),
        "adx": round(float(latest["ADX"]), 2),
        "atr_pct": round(atr_pct * 100, 2),
        "bars": len(df),
        "history_from": str(first_date.date()),
        "history_to": str(last_date.date()),
    }


def process_symbol(
    args: tuple[str, str], force_refresh: bool = False
) -> tuple[dict | None, dict]:
    symbol, yf_symbol = args
    time.sleep(random.uniform(cfg.SLEEP_MIN, cfg.SLEEP_MAX))

    meta = {"symbol": symbol, "downloaded": False, "bars": 0, "error": None}

    try:
        df = get_history(symbol, yf_symbol, force_refresh=force_refresh)
        if df is None:
            meta["error"] = "insufficient_history"
            return None, meta

        meta["downloaded"] = True
        meta["bars"] = len(df)
        meta["from"] = str(df.index.min().date())
        meta["to"] = str(df.index.max().date())

        return scan_symbol(df, symbol), meta

    except Exception as exc:
        meta["error"] = str(exc)
        print(f"  ERROR {symbol}: {exc}")
        return None, meta


def print_results(passed_stocks: list[dict], run_id: int) -> None:
    print(f"\n{'=' * 50}")
    print(f"Done — {len(passed_stocks)} stocks passed (run_id={run_id})")
    print(f"DuckDB: {cfg.DB_PATH}")
    print(f"  Tables: ohlcv_daily, symbol_meta, scan_runs, scan_passed")
    print(f"{'=' * 50}\n")

    if not passed_stocks:
        return

    print(
        f"{'SYMBOL':<15} {'SCORE':>6} {'CLOSE':>8} {'RSI':>6} "
        f"{'ADX':>6} {'VOL_RATIO':>10} {'BARS':>6}"
    )
    print("-" * 65)
    for s in passed_stocks:
        print(
            f"{s['symbol']:<15} {s['score']:>6.1f} {s['close']:>8.2f} "
            f"{s['rsi']:>6.1f} {s['adx']:>6.1f} {s['volume_ratio']:>9.1f}x "
            f"{s['bars']:>6}"
        )


def run(force_refresh: bool | None = None) -> None:
    if force_refresh is None:
        force_refresh = cfg.FORCE_REFRESH_ON_RUN

    ensure_dirs()
    print_settings()

    print("Fetching Nifty 500 symbols...")
    symbols = get_nifty500_symbols()
    print(f"Got {len(symbols)} symbols\n")

    pairs = list(zip(symbols, [s + ".NS" for s in symbols]))
    passed_stocks: list[dict] = []
    download_meta: list[dict] = []
    total = len(pairs)
    done = 0

    with ThreadPoolExecutor(max_workers=cfg.WORKERS) as executor:
        futures = {
            executor.submit(process_symbol, p, force_refresh): p[0] for p in pairs
        }

        for future in as_completed(futures):
            done += 1
            result, meta = future.result()
            download_meta.append(meta)

            if result:
                passed_stocks.append(result)
                print(
                    f"  PASS [{done}/{total}] {result['symbol']:15s} "
                    f"Score={result['score']:.1f}  RSI={result['rsi']}  "
                    f"ADX={result['adx']}  VolRatio={result['volume_ratio']}x"
                )
            elif done % 50 == 0:
                print(f"  ... [{done}/{total}] processed")

    passed_stocks.sort(key=lambda x: x["score"], reverse=True)
    ok_downloads = sum(1 for m in download_meta if m.get("downloaded"))

    summary = {
        "run_date": date.today().isoformat(),
        "years_requested": cfg.YEARS_OF_HISTORY,
        "lookback_days": cfg.LOOKBACK_DAYS,
        "symbols_total": total,
        "symbols_with_history": ok_downloads,
        "symbols_passed_scan": len(passed_stocks),
        "database": cfg.DB_PATH,
    }
    run_id = store.save_scan_run(cfg.DB_PATH, summary, passed_stocks)
    print_results(passed_stocks, run_id)


if __name__ == "__main__":
    run()
