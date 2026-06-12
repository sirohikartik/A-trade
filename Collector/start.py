"""Live Nifty 500 scanner with blast scoring.

Workflow:
    1. Load Nifty 500 symbols from NSE
    2. Download daily OHLCV (Yahoo Finance) → DuckDB
    3. Compute indicators per symbol
    4. Score batch via ``blast.live.score_scan_batch`` (per-segment tuned weights)
    5. Persist passing symbols (blast_score ≥ B band) to DuckDB scan tables

Tune fetch length and filters in ``config/settings.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from io import BytesIO

import pandas as pd
import requests
import yfinance as yf

_COLLECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
_ATRADE_ROOT = os.path.dirname(_COLLECTOR_DIR)
for _path in (_COLLECTOR_DIR, _ATRADE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config as cfg
import db as store
from blast.live import min_alert_score, prepare_latest_rows, score_scan_batch
from indicators import add_indicators

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

REQUIRED_COLS = (
    "close",
    "volume",
    "ema20",
    "ema50",
    "ema200",
    "rsi",
    "macd",
    "macd_signal",
    "adx",
    "atr",
    "volume_ratio",
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
    print(f"  Min blast score (B)  : {min_alert_score()}")
    print(f"  Database             : {cfg.DB_PATH}")
    print("=" * 50)
    print()


def ensure_dirs() -> None:
    os.makedirs(os.path.dirname(cfg.DB_PATH), exist_ok=True)


def clear_generated_data() -> None:
    """Remove DuckDB and any leftover local artifacts (not used by current code)."""
    for path in (
        cfg.DB_PATH,
        cfg.DB_PATH + ".wal",
        os.path.join(cfg.BASE_DIR, "data", "history"),
        os.path.join(cfg.BASE_DIR, "targets"),
    ):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)


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


def fetch_history_from_yahoo(yf_symbol: str, lookback_days: int | None = None) -> pd.DataFrame | None:
    end = date.today()
    days = lookback_days if lookback_days is not None else cfg.lookback_days()
    start = end - timedelta(days=days)
    min_bars = cfg.min_bars_required()

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
    return df if len(df) >= min_bars else None


def save_history(symbol: str, df: pd.DataFrame) -> None:
    store.save_ohlcv(cfg.DB_PATH, symbol, df)
    save_raw_ohlcv(symbol, df)


def save_raw_ohlcv(symbol: str, df: pd.DataFrame) -> None:
    """Plain OHLCV parquet alongside DuckDB (data/raw/ohlcv/)."""
    os.makedirs(cfg.RAW_OHLCV_DIR, exist_ok=True)
    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: "date"})
    out["symbol"] = symbol
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out.to_parquet(os.path.join(cfg.RAW_OHLCV_DIR, f"{symbol}.parquet"), index=False)


def load_history(symbol: str, min_bars: int | None = None) -> pd.DataFrame | None:
    bars = min_bars if min_bars is not None else cfg.min_bars_required()
    df = store.load_ohlcv(cfg.DB_PATH, symbol, bars)
    return _normalize_ohlcv(df) if df is not None else None


def get_history(
    symbol: str,
    yf_symbol: str,
    force_refresh: bool = False,
    lookback_days: int | None = None,
) -> pd.DataFrame | None:
    min_bars = cfg.min_bars_required()
    if not force_refresh:
        cached = load_history(symbol, min_bars=min_bars)
        if cached is not None:
            return cached

    df = fetch_history_from_yahoo(yf_symbol, lookback_days=lookback_days)
    if df is not None:
        save_history(symbol, df)
    return df


def latest_row_valid(latest: pd.Series) -> bool:
    for col in REQUIRED_COLS:
        val = latest.get(col)
        if val is None or pd.isna(val):
            return False
    return True


def extract_latest_row(df: pd.DataFrame, symbol: str) -> pd.Series | None:
    """Build indicator row for blast scoring (filters applied in score_scan_batch)."""
    work = df.reset_index()
    if work.columns[0] != "date":
        work = work.rename(columns={work.columns[0]: "date"})
    enriched = add_indicators(work)
    if enriched.empty:
        return None

    latest = enriched.iloc[-1]
    if not latest_row_valid(latest):
        return None

    row = latest.copy()
    row["symbol"] = symbol
    if "date" not in row.index or pd.isna(row.get("date")):
        row["date"] = enriched["date"].iloc[-1]
    row["bars"] = len(enriched)
    row["history_from"] = str(pd.to_datetime(enriched["date"].iloc[0]).date())
    row["history_to"] = str(pd.to_datetime(enriched["date"].iloc[-1]).date())
    return row


def rows_to_passed_records(scored: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for _, row in scored.iterrows():
        atr_pct = float(row.get("atr_pct", 0))
        records.append(
            {
                "symbol": row["symbol"],
                "score": round(float(row["blast_score"]), 2),
                "close": round(float(row["close"]), 2),
                "volume_ratio": round(float(row.get("volume_ratio", 0)), 2),
                "rsi": round(float(row.get("rsi", 0)), 2),
                "adx": round(float(row.get("adx", 0)), 2),
                "atr_pct": round(atr_pct * 100 if atr_pct <= 1 else atr_pct, 2),
                "bars": int(row.get("bars", 0)),
                "history_from": row.get("history_from", ""),
                "history_to": row.get("history_to", ""),
                "alert_grade": row.get("alert_grade", ""),
                "segment": row.get("segment", ""),
            }
        )
    return records


def process_symbol(
    args: tuple[str, str], force_refresh: bool = False
) -> tuple[pd.Series | None, dict]:
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

        return extract_latest_row(df, symbol), meta

    except Exception as exc:
        meta["error"] = str(exc)
        print(f"  ERROR {symbol}: {exc}")
        return None, meta


def print_results(passed_stocks: list[dict], run_id: int) -> None:
    print(f"\n{'=' * 50}")
    print(f"Done — {len(passed_stocks)} stocks passed (run_id={run_id})")
    print(f"DuckDB: {cfg.DB_PATH}")
    print("  Tables: ohlcv_daily, symbol_meta, scan_runs, scan_passed")
    print("  Latest picks: SELECT * FROM scan_passed WHERE run_id =", run_id)
    print(f"{'=' * 50}\n")

    if not passed_stocks:
        return

    print(
        f"{'SYMBOL':<15} {'SCORE':>6} {'GRADE':>5} {'SEG':>10} {'CLOSE':>8} "
        f"{'RSI':>6} {'ADX':>6} {'VOL':>8}"
    )
    print("-" * 80)
    for s in passed_stocks:
        print(
            f"{s['symbol']:<15} {s['score']:>6.1f} {s.get('alert_grade', ''):>5} "
            f"{s.get('segment', ''):>10} {s['close']:>8.2f} "
            f"{s['rsi']:>6.1f} {s['adx']:>6.1f} {s['volume_ratio']:>7.1f}x"
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
    latest_rows: list[pd.Series] = []
    download_meta: list[dict] = []
    total = len(pairs)
    done = 0

    with ThreadPoolExecutor(max_workers=cfg.WORKERS) as executor:
        futures = {
            executor.submit(process_symbol, p, force_refresh): p[0] for p in pairs
        }

        for future in as_completed(futures):
            done += 1
            row, meta = future.result()
            download_meta.append(meta)

            if row is not None:
                latest_rows.append(row)
            if done % 50 == 0:
                print(f"  ... [{done}/{total}] downloaded")

    print(f"\nScoring {len(latest_rows)} symbols with blast (tuned per-segment weights)...")
    frame = prepare_latest_rows(latest_rows)
    scored = score_scan_batch(frame)
    passed_stocks = rows_to_passed_records(scored)
    for s in passed_stocks[:20]:
        print(
            f"  PASS {s['symbol']:15s} blast={s['score']:.1f} grade={s.get('alert_grade')} "
            f"seg={s.get('segment')}"
        )
    if len(passed_stocks) > 20:
        print(f"  ... and {len(passed_stocks) - 20} more")
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
    parser = argparse.ArgumentParser(description="NSE breakout scanner → DuckDB")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete DuckDB and local artifacts, then download and scan from scratch",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download all symbols from Yahoo (ignore DuckDB cache)",
    )
    args = parser.parse_args()

    if args.fresh:
        clear_generated_data()
        print("Cleared local data. Starting fresh download.\n")

    run(force_refresh=args.refresh or args.fresh)
