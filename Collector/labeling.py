"""Outcome labeling — reference 3-day swing profile."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yaml

import config as cfg

_LABELS_CFG: dict[str, Any] | None = None


def load_labels_config() -> dict[str, Any]:
    global _LABELS_CFG
    if _LABELS_CFG is None:
        with open(cfg.LABELS_PATH, "r", encoding="utf-8") as f:
            _LABELS_CFG = yaml.safe_load(f)
    return _LABELS_CFG


def get_profile_params(profile: str, segment: str) -> dict[str, float | int]:
    profile_cfg = load_labels_config()["profiles"][profile]
    if profile_cfg.get("per_segment"):
        seg_cfg = profile_cfg["segments"][segment]
        return {
            "target_pct": seg_cfg["target_pct"],
            "stop_pct": seg_cfg["stop_pct"],
            "forward_days": seg_cfg["forward_days"],
        }
    return {
        "target_pct": profile_cfg["target_pct"],
        "stop_pct": profile_cfg["stop_pct"],
        "forward_days": profile_cfg["forward_days"],
    }


def label_single_stock(
    df: pd.DataFrame,
    target_pct: float,
    stop_pct: float,
    forward_days: int,
) -> dict[str, list]:
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)

    outcomes, entry_prices = [], []
    ambiguous_flags, hit_target_flags, hit_stop_flags = [], [], []
    return_pcts, mfe_list, mae_list = [], [], []

    for i in range(n):
        if i + 1 >= n:
            outcomes.append(np.nan)
            entry_prices.append(np.nan)
            ambiguous_flags.append(False)
            hit_target_flags.append(False)
            hit_stop_flags.append(False)
            return_pcts.append(np.nan)
            mfe_list.append(np.nan)
            mae_list.append(np.nan)
            continue

        entry = float(df.iloc[i + 1]["open"])
        target = entry * (1 + target_pct)
        stop = entry * (1 - stop_pct)

        outcome = np.nan
        ambiguous = False
        hit_target = False
        hit_stop = False
        max_fav = 0.0
        max_adv = 0.0
        final_return = 0.0

        for j in range(1, forward_days + 1):
            if i + j >= n:
                break

            row = df.iloc[i + j]
            high = float(row["high"])
            low = float(row["low"])

            day_open = float(row["open"])
            day_close = float(row["close"])

            max_fav = max(max_fav, (high - entry) / entry)
            max_adv = min(max_adv, (low - entry) / entry)

            day_hit_target = high >= target
            day_hit_stop = low <= stop

            if day_hit_target and not day_hit_stop:
                outcome = 1
                hit_target = True
                final_return = target_pct
                break
            if day_hit_stop and not day_hit_target:
                outcome = 0
                hit_stop = True
                final_return = -stop_pct
                break
            if day_hit_target and day_hit_stop:
                ambiguous = True
                outcome = 1 if day_close > day_open else 0
                hit_target = day_hit_target
                hit_stop = day_hit_stop
                final_return = (day_close - entry) / entry
                break

        if np.isnan(outcome):
            outcome = 0
            if i + forward_days < n:
                last_close = float(df.iloc[min(i + forward_days, n - 1)]["close"])
                final_return = (last_close - entry) / entry

        outcomes.append(outcome)
        entry_prices.append(entry)
        ambiguous_flags.append(ambiguous)
        hit_target_flags.append(hit_target)
        hit_stop_flags.append(hit_stop)
        return_pcts.append(final_return)
        mfe_list.append(max_fav)
        mae_list.append(max_adv)

    return {
        "outcome": outcomes,
        "entry_price": entry_prices,
        "ambiguous": ambiguous_flags,
        "hit_target": hit_target_flags,
        "hit_stop": hit_stop_flags,
        "return_pct": return_pcts,
        "max_favorable_excursion": mfe_list,
        "max_adverse_excursion": mae_list,
    }


def apply_profile_labels(df: pd.DataFrame, profile: str, segment: str) -> pd.DataFrame:
    params = get_profile_params(profile, segment)
    labels = label_single_stock(
        df,
        target_pct=float(params["target_pct"]),
        stop_pct=float(params["stop_pct"]),
        forward_days=int(params["forward_days"]),
    )

    df = df.copy()
    outcome_col = load_labels_config()["output_columns"][profile]
    df[outcome_col] = labels["outcome"]
    df[f"{profile}_return_pct"] = labels["return_pct"]
    df[f"{profile}_mfe"] = labels["max_favorable_excursion"]
    df[f"{profile}_mae"] = labels["max_adverse_excursion"]
    df[f"{profile}_hit_target"] = labels["hit_target"]
    df[f"{profile}_hit_stop"] = labels["hit_stop"]
    df[f"{profile}_ambiguous"] = labels["ambiguous"]

    if "entry_price" not in df.columns:
        df["entry_price"] = labels["entry_price"]

    return df


def label_stock(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Apply the active label profile and set ``outcome`` = ``outcome_reference``."""
    """Apply reference_swing labels and set primary outcome columns."""
    df = apply_profile_labels(df, "reference_swing", segment)

    df["outcome"] = df["outcome_reference"]
    df["ambiguous"] = df["reference_swing_ambiguous"]

    fwd = int(get_profile_params("reference_swing", segment)["forward_days"])

    df = df.dropna(subset=["outcome_reference"])
    if len(df) > fwd:
        df = df.iloc[:-fwd]

    return df


# Back-compat alias
label_stock_all_profiles = label_stock
