"""Resolve and persist train / validation / test date ranges."""

from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd
import yaml

_COLLECTOR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Collector")
if _COLLECTOR not in sys.path:
    sys.path.insert(0, _COLLECTOR)

import config as cfg

SPLITS_CONFIG_PATH = os.path.join(cfg.CONFIG_DIR, "splits.yaml")
SPLITS_OUTPUT_PATH = cfg.SPLITS_PATH


def _load_splits_config() -> dict:
    if os.path.isfile(SPLITS_CONFIG_PATH):
        with open(SPLITS_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _to_ts(value: str | date | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def auto_split_ranges(dates: pd.Series, ratios: dict) -> dict:
    """Split sorted unique dates into train / val / test by ratio."""
    unique = sorted(pd.to_datetime(dates).dt.normalize().unique())
    n = len(unique)
    if n < 3:
        raise ValueError(f"Need at least 3 trading days for splits, got {n}")

    tr = float(ratios.get("train_ratio", 0.60))
    vr = float(ratios.get("validation_ratio", 0.20))
    te = float(ratios.get("test_ratio", 0.20))
    total = tr + vr + te
    tr, vr, te = tr / total, vr / total, te / total

    i_train_end = max(1, int(n * tr)) - 1
    i_val_end = max(i_train_end + 1, int(n * (tr + vr))) - 1
    i_val_end = min(i_val_end, n - 2)

    train_end = unique[i_train_end]
    val_start = unique[i_train_end + 1]
    val_end = unique[i_val_end]
    test_start = unique[i_val_end + 1]
    test_end = unique[-1]

    return {
        "mode": "dev_auto",
        "data_first": str(unique[0].date()),
        "data_last": str(unique[-1].date()),
        "train": {"start": str(unique[0].date()), "end": str(train_end.date())},
        "validation": {"start": str(val_start.date()), "end": str(val_end.date())},
        "test": {
            "start": str(test_start.date()),
            "end": str(test_end.date()),
            "enabled": True,
        },
    }


def resolve_splits(df: pd.DataFrame, dev_mode: bool = False) -> dict:
    """Return split boundaries; dev_mode uses ratio split on actual dates."""
    dates = pd.to_datetime(df["date"])
    data_first = dates.min()
    data_last = dates.max()

    if dev_mode:
        cfg_doc = _load_splits_config()
        ratios = cfg_doc.get("dev_auto_split", {})
        return auto_split_ranges(dates, ratios)

    cfg_doc = _load_splits_config()
    split_cfg = cfg_doc.get("splits", {})

    train_start = _to_ts(split_cfg.get("train", {}).get("start")) or data_first
    train_end = _to_ts(split_cfg.get("train", {}).get("end")) or data_last
    val_start = _to_ts(split_cfg.get("validation", {}).get("start")) or train_end + pd.Timedelta(days=1)
    val_end = _to_ts(split_cfg.get("validation", {}).get("end")) or data_last
    test_cfg = split_cfg.get("test", {})
    test_enabled = bool(test_cfg.get("enabled", False))
    test_start = _to_ts(test_cfg.get("start")) if test_enabled else None
    test_end = _to_ts(test_cfg.get("end")) if test_enabled else None
    if test_enabled and test_start is None:
        test_start = val_end + pd.Timedelta(days=1)
    if test_enabled and test_end is None:
        test_end = data_last

    return {
        "mode": "config",
        "data_first": str(data_first.date()),
        "data_last": str(data_last.date()),
        "train": {"start": str(train_start.date()), "end": str(train_end.date())},
        "validation": {"start": str(val_start.date()), "end": str(val_end.date())},
        "test": {
            "start": str(test_start.date()) if test_enabled and test_start is not None else None,
            "end": str(test_end.date()) if test_enabled and test_end is not None else None,
            "enabled": test_enabled,
        },
    }


def mask_split(df: pd.DataFrame, split_name: str, bounds: dict) -> pd.Series:
    dates = pd.to_datetime(df["date"])
    info = bounds[split_name]
    if split_name == "test" and not info.get("enabled", False):
        return pd.Series(False, index=df.index)

    start = pd.Timestamp(info["start"])
    end = pd.Timestamp(info["end"])
    return (dates >= start) & (dates <= end)


def write_splits_file(bounds: dict) -> str:
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    doc = {
        "description": "Resolved train / validation / test date ranges for research and blast EA",
        **bounds,
    }
    with open(SPLITS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
    return SPLITS_OUTPUT_PATH


def format_splits_banner(bounds: dict) -> str:
    lines = [
        f"  Data window      : {bounds['data_first']} to {bounds['data_last']} ({bounds['mode']})",
        f"  Train            : {bounds['train']['start']} to {bounds['train']['end']}",
        f"  Validation       : {bounds['validation']['start']} to {bounds['validation']['end']}",
    ]
    if bounds.get("test", {}).get("enabled"):
        lines.append(
            f"  Test             : {bounds['test']['start']} to {bounds['test']['end']}"
        )
    else:
        lines.append("  Test             : disabled")
    lines.append(f"  Splits file      : {SPLITS_OUTPUT_PATH}")
    return "\n".join(lines)
