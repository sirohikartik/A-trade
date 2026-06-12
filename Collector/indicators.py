"""Technical indicators — Reference-aligned, no lookahead."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yaml
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

import config as cfg

_CONFIG: dict[str, Any] | None = None


def _load_settings() -> dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        with open(cfg.SETTINGS_PATH, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f) or {}
    return _CONFIG


def get_reference_rules() -> dict[str, Any]:
    return _load_settings().get("reference_rules", {})


def get_indicator_params() -> dict[str, Any]:
    return _load_settings().get("indicators", {})


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        return df
    out = df.reset_index()
    if out.columns[0] != "date":
        out = out.rename(columns={out.columns[0]: "date"})
    return out


def add_indicators(df: pd.DataFrame, rules: dict[str, Any] | None = None) -> pd.DataFrame:
    rules = rules or get_reference_rules()
    params = get_indicator_params()

    df = _ensure_date_column(df).sort_values("date").copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    rsi_period = params.get("rsi_period", 14)
    adx_period = params.get("adx_period", 14)
    atr_period = params.get("atr_period", 14)
    bb_period = params.get("bb_period", 20)
    bb_std = params.get("bb_std", 2)
    vol_ma = params.get("volume_ma_period", 20)
    adx_lb = params.get("adx_rising_lookback", 3)

    rsi_min = rules.get("rsi_min", 55)
    rsi_max = rules.get("rsi_max", 75)
    adx_min = rules.get("adx_min", 20)
    vol_min = rules.get("volume_ratio_min", 2.0)
    delivery_min = rules.get("delivery_pct_min", 50)
    atr_max = rules.get("atr_pct_max", 0.06)
    near_20d = rules.get("near_20d_high_pct", 0.98)
    near_52w = rules.get("near_52w_high_pct", 0.97)

    df["ema20"] = EMAIndicator(close, 20).ema_indicator()
    df["ema50"] = EMAIndicator(close, 50).ema_indicator()
    df["ema200"] = EMAIndicator(close, 200).ema_indicator()

    df["ema_stack_bull"] = (
        (close > df["ema20"])
        & (df["ema20"] > df["ema50"])
        & (df["ema50"] > df["ema200"])
    ).astype(int)

    df["above_ema20"] = (close > df["ema20"]).astype(int)
    df["above_ema50"] = (close > df["ema50"]).astype(int)
    df["above_ema200"] = (close > df["ema200"]).astype(int)
    df["ema20_slope"] = df["ema20"].diff(5) / df["ema20"].shift(5)
    df["pullback_ema20"] = (close - df["ema20"]).abs() / df["ema20"]

    open_p = df["open"]
    body = (open_p - close).abs()
    lower_wick = np.minimum(open_p, close) - low
    upper_wick = high - np.maximum(open_p, close)

    df["candle_hammer"] = (
        (lower_wick > 2 * body) & (upper_wick < body) & (body > 0)
    ).astype(int)
    df["candle_engulfing"] = (
        (close > open_p.shift(1))
        & (open_p < close.shift(1))
        & (close > open_p)
        & (close.shift(1) < open_p.shift(1))
    ).astype(int)

    df["rsi"] = RSIIndicator(close, rsi_period).rsi()
    df["rsi_oversold"] = (df["rsi"] < 45).astype(int)
    df["rsi_bull_zone"] = ((df["rsi"] >= rsi_min) & (df["rsi"] <= rsi_max)).astype(int)

    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    macd_ind = MACD(close)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()
    df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)
    df["macd_hist_rising"] = (df["macd_hist"] > df["macd_hist"].shift(1)).astype(int)

    df["adx"] = ADXIndicator(high, low, close, adx_period).adx()
    df["adx_strong"] = (df["adx"] > adx_min).astype(int)
    df["adx_rising"] = (
        (df["adx"] > adx_min) & (df["adx"] > df["adx"].shift(adx_lb))
    ).astype(int)

    df["atr"] = AverageTrueRange(high, low, close, atr_period).average_true_range()
    df["atr_pct"] = df["atr"] / close
    df["atr_risk_ok"] = (df["atr_pct"] <= atr_max).astype(int)

    bb = BollingerBands(close, bb_period, bb_std)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    df["bb_squeeze"] = (
        df["bb_width"] < df["bb_width"].rolling(50).mean() * 0.75
    ).astype(int)
    df["bb_squeeze_breakout"] = (
        (df["bb_squeeze"].shift(1) == 1) & (df["bb_width"] > df["bb_width"].shift(1))
    ).astype(int)

    df["vol_ma20"] = vol.rolling(vol_ma).mean()
    df["volume_ratio"] = vol / df["vol_ma20"]
    df["vol_ma5"] = vol.rolling(5).mean()
    df["vol_trend_up"] = (df["vol_ma5"] > df["vol_ma20"]).astype(int)

    df["high_20d"] = high.rolling(20).max()
    df["high_52w"] = high.rolling(params.get("high_52w_period", 252)).max()
    df["low_20d"] = low.rolling(20).min()

    prev_high_20d = df["high_20d"].shift(1)
    df["new_20d_high"] = (close > prev_high_20d).astype(int)
    df["near_20d_high"] = (close >= df["high_20d"] * near_20d).astype(int)
    df["near_52w_high"] = (close >= df["high_52w"] * near_52w).astype(int)
    df["new_52w_high"] = (close >= df["high_52w"]).astype(int)
    df["above_prev_high"] = (close > high.shift(1)).astype(int)
    df["breakout_volume"] = (
        (df["new_20d_high"] == 1) & (df["volume_ratio"] >= vol_min)
    ).astype(int)

    df["close_position"] = (close - low) / (high - low + 1e-9)

    df["setup_ready"] = (
        (df["ema_stack_bull"] == 1)
        & ((df["new_20d_high"] == 1) | (df["above_prev_high"] == 1))
        & (df["volume_ratio"] >= vol_min)
        & (df["rsi_bull_zone"] == 1)
        & (df["macd_bullish"] == 1)
        & (df["adx_rising"] == 1)
        & (df["atr_risk_ok"] == 1)
    ).astype(int)

    return df


def add_delivery_features(df: pd.DataFrame, rules: dict[str, Any] | None = None) -> pd.DataFrame:
    rules = rules or get_reference_rules()
    delivery_min = rules.get("delivery_pct_min", 50)

    if "delivery_pct" not in df.columns:
        df["delivery_pct_ma10"] = np.nan
        df["delivery_above_avg"] = 0
        df["delivery_strong"] = 0
        return df

    df["delivery_pct"] = df["delivery_pct"].ffill()
    df["delivery_pct_ma10"] = df["delivery_pct"].rolling(10).mean()
    df["delivery_above_avg"] = (df["delivery_pct"] > df["delivery_pct_ma10"]).astype(int)
    df["delivery_strong"] = (
        (df["delivery_pct"] > delivery_min) & (df["delivery_above_avg"] == 1)
    ).astype(int)

    has_delivery = df["delivery_pct"].notna()
    df.loc[has_delivery, "setup_ready"] = (
        (df.loc[has_delivery, "setup_ready"] == 1)
        & (df.loc[has_delivery, "delivery_strong"] == 1)
    ).astype(int)

    return df
