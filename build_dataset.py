#!/usr/bin/env python3
"""
Build ML tabular + DL sequence datasets for A-trade — single command pipeline.

Usage (from A-trade/):
  python build_dataset.py
  python build_dataset.py --days 30 --max-symbols 20    # quick test run
  python build_dataset.py --top-n 100
  python build_dataset.py --days 30 --max-symbols 20 --skip-delivery
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

import config as cfg
from segments import build_segment_map
from delivery import fetch_all_delivery

from pipeline.fetch import fetch_all_ohlcv
from pipeline.process import process_all_symbols
from pipeline.features import build_datasets


def _print_banner(top_n: int | None, dev_mode: bool) -> None:
    print("=" * 60)
    print("A-trade Dataset Builder (ML tabular + DL sequences)")
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
    print(f"  Split output     : {os.path.join(cfg.DATASETS_DIR, 'splits.yaml')}")
    print(f"  Universe         : {'top ' + str(top_n) if top_n else 'all Nifty 500'}")
    print(f"  DuckDB (OHLCV)   : {cfg.DB_PATH}")
    print(f"  Raw OHLCV        : {cfg.RAW_OHLCV_DIR}")
    print(f"  Raw bhavcopy     : {cfg.RAW_BHAVCOPY_DIR}")
    print(f"  ML tabular       : {cfg.TABULAR_DIR}")
    print(f"  DL sequences     : {cfg.SEQUENCES_DIR}")
    if dev_mode:
        print("  Mode             : DEV (auto 60/20/20 train/val/test split)")
    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML tabular + DL sequence datasets")
    parser.add_argument("--refresh", action="store_true", help="Re-download all OHLCV from Yahoo")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip OHLCV download (use DuckDB cache)")
    parser.add_argument("--skip-delivery", action="store_true", help="Skip NSE bhavcopy download")
    parser.add_argument("--fresh", action="store_true", help="Clear DuckDB and re-download everything")
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
    os.makedirs(cfg.DATASETS_DIR, exist_ok=True)
    os.makedirs(cfg.TABULAR_DIR, exist_ok=True)
    os.makedirs(cfg.SEQUENCES_DIR, exist_ok=True)

    if args.fresh:
        if os.path.isdir(cfg.DATA_DIR):
            shutil.rmtree(cfg.DATA_DIR, ignore_errors=True)
        print("Cleared all generated data (DuckDB, raw OHLCV, bhavcopy, processed, datasets).\n")

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

    print("Phase 4/5 - Indicators + dual labels")
    process_all_symbols(symbols)
    print()

    print("Phase 5/5 - Build ML tabular + DL sequence datasets")
    if args.top_n:
        print(f"  Ranking train-period edge -> selecting top {args.top_n} symbols")
    summary = build_datasets(top_n=args.top_n, dev_mode=dev_mode)
    print()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Done in {elapsed / 60:.1f} min")
    if summary.get("splits_path"):
        print(f"  Splits written : {summary['splits_path']}")
    for profile, info in summary.get("profiles", {}).items():
        print(
            f"  {profile}: ML {info['train_rows']}/{info['val_rows']}/{info['test_rows']} "
            f"(train/val/test) | DL {info['dl_train_samples']}/{info['dl_val_samples']}/"
            f"{info['dl_test_samples']}"
        )
    print(f"  Features: {summary.get('feature_count', 0)} | seq_len: {summary.get('sequence_length')}")
    print(f"  Manifest: {summary.get('manifest_path')}")
    print("=" * 60)

    cfg.clear_runtime()


if __name__ == "__main__":
    main()
