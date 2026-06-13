"""Blast configuration and weight loading.

Resolves paths under ``blast/config/blast.yaml`` and ``config/settings.yaml``,
and loads tuned weight YAMLs from ``data/blast/`` (active → standard → blueprint).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

BLAST_ROOT = os.path.dirname(os.path.abspath(__file__))
ATRADE_ROOT = os.path.dirname(BLAST_ROOT)
BLAST_CONFIG_PATH = os.path.join(BLAST_ROOT, "config", "blast.yaml")
SETTINGS_PATH = os.path.join(ATRADE_ROOT, "config", "settings.yaml")
SECTOR_MAP_PATH = os.path.join(ATRADE_ROOT, "data", "sector_map.csv")
BLAST_DATA_DIR = os.path.join(ATRADE_ROOT, "data", "blast")
STANDARD_WEIGHTS_DIR = os.path.join(BLAST_DATA_DIR, "standard")
ACTIVE_WEIGHTS_DIR = BLAST_DATA_DIR


@lru_cache(maxsize=1)
def load_blast_config() -> dict[str, Any]:
    """Cached parse of ``blast/config/blast.yaml``."""
    with open(BLAST_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_thresholds() -> dict[str, Any]:
    """Merge ``reference_rules`` from settings with blast-specific threshold overrides."""
    blast = load_blast_config().get("thresholds", {})
    rules = load_settings().get("reference_rules", {})
    merged = dict(rules)
    merged.update(blast)
    return merged


def get_weights() -> dict[str, int]:
    """Blueprint default weights (sum 100) before EA tuning."""
    return dict(load_blast_config().get("weights_default", {}))


def _read_weights_yaml(path: str) -> dict[str, int] | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    weights = doc.get("weights")
    if not weights:
        return None
    return {str(k): int(v) for k, v in weights.items()}


def load_standard_weights(segment: str) -> dict[str, int]:
    """Frozen standard weights (run_01 baseline)."""
    path = os.path.join(STANDARD_WEIGHTS_DIR, f"weights_{segment}.yaml")
    weights = _read_weights_yaml(path)
    if weights:
        return weights
    return get_weights()


def load_segment_weights(segment: str) -> dict[str, int]:
    """Active tuned weights; falls back to standard then blueprint."""
    for directory in (ACTIVE_WEIGHTS_DIR, STANDARD_WEIGHTS_DIR):
        path = os.path.join(directory, f"weights_{segment}.yaml")
        weights = _read_weights_yaml(path)
        if weights:
            return weights
    return get_weights()


def get_filter_config() -> dict[str, Any]:
    """Liquidity and macro filter thresholds for the daily ranking pool."""
    blast_filters = load_blast_config().get("filters", {})
    settings_filters = load_settings().get("filters", {})
    macro = settings_filters.get("macro", {})
    return {
        "min_price": float(settings_filters.get("min_price", 20)),
        "min_avg_daily_value": settings_filters.get("min_avg_daily_value", {}),
        "min_market_breadth": float(
            blast_filters.get("min_market_breadth", macro.get("min_market_breadth", 0.40))
        ),
        "use_macro_filter": bool(blast_filters.get("use_macro_filter", True)),
    }
