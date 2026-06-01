"""
Scanner settings — edit values here before running Collector/start.py
"""

import os

# Project root (A-trade/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# DATA FETCH (Yahoo Finance daily bars)
# =============================================================================

# How many years of history to download per symbol.
YEARS_OF_HISTORY = 3

# Extra calendar days on top of YEARS_OF_HISTORY (weekends, holidays, gaps).
CALENDAR_BUFFER_DAYS = 45

# Total calendar span requested from Yahoo: ~3 years + buffer → ~1140 days.
LOOKBACK_DAYS = YEARS_OF_HISTORY * 365 + CALENDAR_BUFFER_DAYS

# Minimum number of daily bars required to keep a symbol (EMA200 needs ~200).
MIN_BARS = 220

# Re-download from Yahoo even if symbol already exists in DuckDB.
FORCE_REFRESH_ON_RUN = False

# =============================================================================
# DOWNLOAD CONCURRENCY
# =============================================================================

WORKERS = 5
SLEEP_MIN = 0.2
SLEEP_MAX = 0.5

# =============================================================================
# SCAN FILTERS (breakout / momentum)
# =============================================================================

MIN_PRICE = 20
MIN_VOLUME = 100_000
MAX_ATR_PCT = 0.06
MIN_SCORE = 65

# =============================================================================
# PATHS (usually leave as-is)
# =============================================================================

DB_PATH = os.path.join(BASE_DIR, "data", "atrade.duckdb")
