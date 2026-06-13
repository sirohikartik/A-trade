#!/usr/bin/env python3
"""Build A-trade research datasets — five-phase CLI pipeline.

Phases:
    1. Fetch OHLCV (Yahoo) → DuckDB + raw parquet
    2. Fetch NSE bhavcopy delivery
    3. Cap-segment map (large / mid / small)
    4. Indicators + reference 3-day swing labels → segment parquets
    5. Train/val/test splits (+ optional top-N universe filter)

Usage (from ``A-trade/``)::

    python build_dataset.py
    python build_dataset.py --days 30 --max-symbols 20
    python build_dataset.py --top-n 100
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
COLLECTOR = os.path.join(ROOT, "Collector")
PIPELINE = os.path.join(ROOT, "pipeline")
for path in (ROOT, COLLECTOR, PIPELINE):
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd

import config as cfg
from segments import build_segment_map
from delivery import fetch_all_delivery

from pipeline.fetch import fetch_all_ohlcv
from pipeline.process import process_all_symbols
from pipeline.splits import format_splits_banner, resolve_splits, write_splits_file


def _load_processed_frame() -> pd.DataFrame:
    seg_path = os.path.join(cfg.SEGMENTS_DIR, "all.parquet")
    if os.path.isfile(seg_path):
        return pd.read_parquet(seg_path)

    parts = []
    for seg in ("large_cap", "mid_cap", "small_cap"):
        p = os.path.join(cfg.SEGMENTS_DIR, f"{seg}.parquet")
        if os.path.isfile(p):
            parts.append(pd.read_parquet(p))
    if not parts:
        raise RuntimeError("No segment parquets found. Run process step first.")
    return pd.concat(parts, ignore_index=True)


def finalize_splits(df: pd.DataFrame | None = None, top_n: int | None = None, dev_mode: bool = False) -> dict:
    """Resolve train/val/test splits and optionally filter to top-N symbols."""
    if df is None:
        df = _load_processed_frame()

    if top_n is not None and top_n > 0:
        from pipeline.universe import select_top_n

        df, symbols, ranked = select_top_n(df, top_n)
        universe_info = {
            "top_n": top_n,
            "symbols": symbols,
            "top_5": ranked.head(5)[["rank", "symbol", "composite_score", "setup_win_rate"]].to_dict(
                orient="records"
            ),
        }
    else:
        universe_info = None

    df["date"] = pd.to_datetime(df["date"])
    split_bounds = resolve_splits(df, dev_mode=dev_mode)
    splits_path = write_splits_file(split_bounds)
    print(format_splits_banner(split_bounds))

    train_mask = (df["date"] >= pd.Timestamp(split_bounds["train"]["start"])) & (
        df["date"] <= pd.Timestamp(split_bounds["train"]["end"])
    )
    train_rows = int(train_mask.sum()) if len(df) else 0

    return {
        "splits_path": splits_path,
        "splits": split_bounds,
        "universe": universe_info,
        "total_rows": len(df),
        "train_rows": train_rows,
        "outcome_rate": float(df["outcome_reference"].mean()) if "outcome_reference" in df.columns and len(df) else 0.0,
    }


def _print_banner(top_n: int | None, dev_mode: bool) -> None:
    print("=" * 60)
    print("A-trade Research Pipeline")
    print("=" * 60)
    recent = cfg.fetch_recent_days()
    if recent:
        print(f"  Fetch window     : last {recent} calendar days (dev/test)")
        print(f"  Min bars/symbol  : {cfg.min_bars_required()} (dev)")
    else:
        print(f"  Years of history : {cfg.YEARS_OF_HISTORY}")
        print(f"  Lookback days    : {cfg.lookback_days()}")
        print(f"  Min bars/symbol  : {cfg.min_bars_required()}")
    print(f"  Split config     : {cfg.SPLITS_CONFIG_PATH}")
    print(f"  Split output     : {cfg.SPLITS_PATH}")
    print(f"  Universe         : {'top ' + str(top_n) if top_n else 'all Nifty 500'}")
    print(f"  DuckDB (OHLCV)   : {cfg.DB_PATH}")
    print(f"  Raw OHLCV        : {cfg.RAW_OHLCV_DIR}")
    print(f"  Raw bhavcopy     : {cfg.RAW_BHAVCOPY_DIR}")
    print(f"  Processed        : {cfg.SEGMENTS_DIR}")
    if dev_mode:
        print("  Mode             : DEV (auto 60/20/20 train/val/test split)")
    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build A-trade research datasets")
    parser.add_argument("--refresh", action="store_true", help="Re-download all OHLCV from Yahoo")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip OHLCV download (use DuckDB cache)")
    parser.add_argument("--skip-delivery", action="store_true", help="Skip NSE bhavcopy download")
    parser.add_argument("--fresh", action="store_true", help="Clear data/ and re-download everything")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        metavar="N",
        help="Fetch only the last N calendar days (dev/test; lowers min-bar requirement)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        metavar="N",
        help="Limit universe to first N Nifty 500 symbols (dev/test)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="Use only top N symbols ranked by train-period setup profitability (default: all)",
    )
    args = parser.parse_args()

    cfg.set_runtime(fetch_days=args.days, max_symbols=args.max_symbols)
    dev_mode = args.days is not None

    t0 = time.time()
    _print_banner(args.top_n, dev_mode)

    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    os.makedirs(cfg.RAW_OHLCV_DIR, exist_ok=True)
    os.makedirs(cfg.PROCESSED_DIR, exist_ok=True)

    if args.fresh:
        if os.path.isdir(cfg.DATA_DIR):
            shutil.rmtree(cfg.DATA_DIR, ignore_errors=True)
        print("Cleared all generated data (DuckDB, raw OHLCV, bhavcopy, processed).\n")
        os.makedirs(cfg.DATA_DIR, exist_ok=True)
        os.makedirs(cfg.RAW_OHLCV_DIR, exist_ok=True)
        os.makedirs(cfg.PROCESSED_DIR, exist_ok=True)

    if args.skip_fetch:
        import db as store

        symbols = store.list_symbols(cfg.DB_PATH)
        print(f"Phase 1/5 skipped - {len(symbols)} symbols in DuckDB\n")
    else:
        print("Phase 1/5 - Fetch OHLCV -> DuckDB + raw parquet")
        symbols = fetch_all_ohlcv(
            force_refresh=args.refresh or args.fresh,
            max_symbols=args.max_symbols,
        )
        print()

    if not symbols:
        import db as store

        symbols = store.list_symbols(cfg.DB_PATH)
    if not symbols:
        raise SystemExit("No OHLCV data available. Remove --skip-fetch or check network.")

    if args.skip_delivery:
        print("Phase 2/5 - Delivery fetch skipped\n")
    else:
        print("Phase 2/5 - Fetch delivery raw bhavcopy")
        fetch_all_delivery()
        print()

    print("Phase 3/5 - Build segment map")
    build_segment_map(symbols)
    print()

    print("Phase 4/5 - Indicators + reference labels")
    process_all_symbols(symbols)
    print()

    print("Phase 5/5 - Splits + optional universe top-N")
    if args.top_n:
        print(f"  Ranking train-period edge -> selecting top {args.top_n} symbols")
    summary = finalize_splits(top_n=args.top_n, dev_mode=dev_mode)
    print()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Done in {elapsed / 60:.1f} min")
    if summary.get("splits_path"):
        print(f"  Splits written : {summary['splits_path']}")
    print(f"  Total rows     : {summary.get('total_rows', 0)}")
    print(f"  Train rows     : {summary.get('train_rows', 0)}")
    print(f"  Outcome rate   : {summary.get('outcome_rate', 0):.1%} (reference 3-day)")
    if summary.get("universe"):
        print(f"  Top-N symbols  : {summary['universe']['top_n']}")
    print("=" * 60)

    cfg.clear_runtime()


if __name__ == "__main__":
    main()
