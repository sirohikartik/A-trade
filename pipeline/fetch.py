"""Bulk OHLCV fetch for Nifty 500 universe."""

from __future__ import annotations

import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

_COLLECTOR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Collector")
if _COLLECTOR not in sys.path:
    sys.path.insert(0, _COLLECTOR)

import config as cfg
from start import ensure_dirs, get_history, get_nifty500_symbols


def fetch_all_ohlcv(
    force_refresh: bool = False,
    max_symbols: int | None = None,
) -> list[str]:
    ensure_dirs()
    recent = cfg.fetch_recent_days()
    if recent:
        print(f"Fetching Nifty 500 OHLCV (last {recent} calendar days, dev mode)...")
    else:
        print(f"Fetching Nifty 500 OHLCV ({cfg.YEARS_OF_HISTORY} years)...")

    symbols = get_nifty500_symbols()
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
        print(f"  Limited to {max_symbols} symbols (--max-symbols)")

    pairs = list(zip(symbols, [s + ".NS" for s in symbols]))
    total = len(pairs)
    ok_symbols: list[str] = []
    min_bars = cfg.min_bars_required()
    lookback = cfg.lookback_days()

    def _one(pair: tuple[str, str]) -> tuple[str, bool]:
        symbol, yf_symbol = pair
        time.sleep(random.uniform(cfg.SLEEP_MIN, cfg.SLEEP_MAX))
        try:
            df = get_history(
                symbol,
                yf_symbol,
                force_refresh=force_refresh,
                lookback_days=lookback,
            )
            return symbol, df is not None and len(df) >= min_bars
        except Exception:
            return symbol, False

    with ThreadPoolExecutor(max_workers=cfg.WORKERS) as pool:
        futures = {pool.submit(_one, p): p[0] for p in pairs}
        with tqdm(total=total, desc="OHLCV fetch", unit="sym") as bar:
            for future in as_completed(futures):
                symbol, ok = future.result()
                if ok:
                    ok_symbols.append(symbol)
                bar.update(1)
                bar.set_postfix(ok=len(ok_symbols), min_bars=min_bars)

    print(
        f"OHLCV complete: {len(ok_symbols)}/{total} symbols | "
        f"DuckDB + raw parquet -> {cfg.DB_PATH}, {cfg.RAW_OHLCV_DIR}"
    )
    return ok_symbols
