"""Load resolved splits from data/splits.yaml."""

from __future__ import annotations

import os

import pandas as pd
import yaml

from blast._config import ATRADE_ROOT


def load_splits_path() -> str:
    return os.path.join(ATRADE_ROOT, "data", "splits.yaml")


def load_splits() -> dict:
    """Parse ``data/splits.yaml`` written by ``build_dataset.py``."""
    path = load_splits_path()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def mask_split(df: pd.DataFrame, split_name: str, bounds: dict | None = None) -> pd.Series:
    """Boolean mask for rows inside train, validation, or test date range."""
    bounds = bounds or load_splits()
    dates = pd.to_datetime(df["date"])
    info = bounds[split_name]
    if split_name == "test" and not info.get("enabled", False):
        return pd.Series(False, index=df.index)

    start = pd.Timestamp(info["start"])
    end = pd.Timestamp(info["end"])
    return (dates >= start) & (dates <= end)
