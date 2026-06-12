"""
Scanner and pipeline settings — edit values here or override via config/settings.yaml
"""

import os

import yaml

# Project root (A-trade/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.yaml")


def _load_yaml() -> dict:
    if os.path.isfile(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_YAML = _load_yaml()
_DATA = _YAML.get("data", {})
_FETCH = _YAML.get("fetch", {})
_FILTERS = _YAML.get("filters", {})

# Runtime overrides (set by build_dataset.py --days / --max-symbols)
_RUNTIME: dict = {}


def set_runtime(**kwargs) -> None:
    _RUNTIME.update({k: v for k, v in kwargs.items() if v is not None})


def clear_runtime() -> None:
    _RUNTIME.clear()


# =============================================================================
# DATA FETCH (Yahoo Finance daily bars)
# =============================================================================

YEARS_OF_HISTORY = int(_DATA.get("years_of_history", 5))
CALENDAR_BUFFER_DAYS = int(_DATA.get("calendar_buffer_days", 45))
_FETCH_RECENT_DAYS = _DATA.get("fetch_recent_days")
MIN_BARS_DEFAULT = int(_FILTERS.get("min_history_days", 250))
DEV_MIN_BARS = int(_FILTERS.get("dev_min_history_days", 5))
FORCE_REFRESH_ON_RUN = False

TRAIN_END = _DATA.get("train_end", "2023-06-30")
VALIDATION_START = _DATA.get("validation_start", "2023-07-01")
SPLITS_CONFIG_PATH = os.path.join(CONFIG_DIR, "splits.yaml")


def fetch_recent_days() -> int | None:
    if _RUNTIME.get("fetch_days") is not None:
        return int(_RUNTIME["fetch_days"])
    if _FETCH_RECENT_DAYS is not None:
        return int(_FETCH_RECENT_DAYS)
    return None


def lookback_days() -> int:
    recent = fetch_recent_days()
    if recent is not None:
        return int(recent) + CALENDAR_BUFFER_DAYS
    return YEARS_OF_HISTORY * 365 + CALENDAR_BUFFER_DAYS


def min_bars_required() -> int:
    if fetch_recent_days() is not None:
        return DEV_MIN_BARS
    return MIN_BARS_DEFAULT


# Back-compat module-level names (default production values)
LOOKBACK_DAYS = lookback_days()
MIN_BARS = min_bars_required()

# =============================================================================
# DOWNLOAD CONCURRENCY
# =============================================================================

WORKERS = int(_FETCH.get("workers", 5))
SLEEP_MIN = float(_FETCH.get("sleep_min", 0.2))
SLEEP_MAX = float(_FETCH.get("sleep_max", 0.5))
DELIVERY_WORKERS = int(_FETCH.get("delivery_workers", 8))

# =============================================================================
# SCAN FILTERS (breakout / momentum)
# =============================================================================

MIN_PRICE = float(_FILTERS.get("min_price", 20))
MIN_VOLUME = int(_FILTERS.get("min_volume", 100_000))
MAX_ATR_PCT = float(_YAML.get("reference_rules", {}).get("atr_pct_max", 0.06))
MIN_SCORE = 65

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "atrade.duckdb")
RAW_OHLCV_DIR = os.path.join(DATA_DIR, "raw", "ohlcv")
RAW_BHAVCOPY_DIR = os.path.join(DATA_DIR, "raw", "bhavcopy")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
COMBINED_DIR = os.path.join(PROCESSED_DIR, "combined")
SEGMENTS_DIR = os.path.join(PROCESSED_DIR, "segments")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
TABULAR_DIR = os.path.join(DATASETS_DIR, "tabular")
SEQUENCES_DIR = os.path.join(DATASETS_DIR, "sequences")
# Back-compat alias
ML_DIR = TABULAR_DIR
SEGMENT_MAP_PATH = os.path.join(PROCESSED_DIR, "segment_map.parquet")
UNIVERSE_RANKINGS_PATH = os.path.join(PROCESSED_DIR, "universe_rankings.parquet")
UNIVERSE_TOP_N_PATH = os.path.join(DATASETS_DIR, "universe_top_n.json")
MARKET_BREADTH_PATH = os.path.join(PROCESSED_DIR, "market_breadth.parquet")
FEATURE_MANIFEST_PATH = os.path.join(DATASETS_DIR, "manifest.yaml")

LABELS_PATH = os.path.join(CONFIG_DIR, "labels.yaml")
ML_CONFIG_PATH = os.path.join(CONFIG_DIR, "ml.yaml")
