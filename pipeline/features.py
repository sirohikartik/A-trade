"""Build ML tabular + DL sequence train/validation datasets."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yaml

_COLLECTOR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Collector")
if _COLLECTOR not in sys.path:
    sys.path.insert(0, _COLLECTOR)

import config as cfg


def _load_ml_config() -> dict:
    with open(cfg.ML_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_settings() -> dict:
    with open(cfg.SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _outcome_column(profile: str) -> str:
    with open(cfg.LABELS_PATH, "r", encoding="utf-8") as f:
        labels = yaml.safe_load(f)
    return labels["output_columns"][profile]


def _feature_list(ml_cfg: dict) -> list[str]:
    cols = ml_cfg.get("feature_columns", {})
    return list(cols.get("raw", [])) + list(cols.get("binary", []))


def _add_zscores(df: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        roll_mean = out.groupby("symbol")[col].transform(
            lambda s: s.rolling(window, min_periods=20).mean()
        )
        roll_std = out.groupby("symbol")[col].transform(
            lambda s: s.rolling(window, min_periods=20).std()
        )
        out[f"{col}_z"] = (out[col] - roll_mean) / roll_std.replace(0, np.nan)
    return out


def _add_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby("symbol")[col].shift(lag)
    return out


def _prepare_frame(
    df: pd.DataFrame, ml_cfg: dict, settings: dict
) -> tuple[pd.DataFrame, list[str], list[str]]:
    filters = settings.get("filters", {})
    atr_max = settings.get("reference_rules", {}).get("atr_pct_max", 0.06)

    if "atr_risk_ok" in df.columns:
        df = df[df["atr_risk_ok"] == 1]
    elif "atr_pct" in df.columns:
        df = df[df["atr_pct"] <= atr_max]

    min_price = float(filters.get("min_price", 20))
    df = df[df["close"] >= min_price]

    ml_section = settings.get("ml", {})
    z_window = int(ml_section.get("rolling_zscore_window", 60))
    lag_periods = ml_section.get("lag_periods", [1, 2, 3])

    df = _add_zscores(df, ml_cfg.get("zscore_columns", []), z_window)
    df = _add_lags(df, ml_cfg.get("lag_features", []), lag_periods)

    base_feats = _feature_list(ml_cfg)
    z_feats = [f"{c}_z" for c in ml_cfg.get("zscore_columns", []) if c in df.columns]
    lag_feats = []
    for col in ml_cfg.get("lag_features", []):
        for lag in lag_periods:
            name = f"{col}_lag{lag}"
            if name in df.columns:
                lag_feats.append(name)

    feature_cols = [c for c in base_feats + z_feats + lag_feats if c in df.columns]
    meta_cols = ["symbol", "date", "segment"]
    return df, feature_cols, meta_cols


def _build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
) -> dict[str, np.ndarray]:
    """Stack (samples, seq_len, features) tensors for DL models."""
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    sym_list: list[str] = []
    date_list: list[str] = []

    feat = df[feature_cols].astype(np.float32)
    labels = df[target_col].astype(int)
    symbols = df["symbol"].astype(str)
    dates = pd.to_datetime(df["date"])

    for sym in df["symbol"].unique():
        idx = df.index[df["symbol"] == sym]
        arr = feat.loc[idx].values
        lab = labels.loc[idx].values
        dts = dates.loc[idx].values
        sym_arr = symbols.loc[idx].values

        for i in range(seq_len, len(idx)):
            window = arr[i - seq_len : i]
            if np.isnan(window).any():
                continue
            x_list.append(window)
            y_list.append(int(lab[i]))
            sym_list.append(str(sym_arr[i]))
            date_list.append(str(dts[i])[:10])

    if not x_list:
        return {}

    return {
        "X": np.stack(x_list, axis=0),
        "y": np.array(y_list, dtype=np.int8),
        "symbol": np.array(sym_list, dtype=object),
        "date": np.array(date_list, dtype=object),
    }


def build_datasets(
    df: pd.DataFrame | None = None,
    top_n: int | None = None,
    dev_mode: bool = False,
) -> dict:
    ml_cfg = _load_ml_config()
    settings = _load_settings()

    if df is None:
        seg_path = os.path.join(cfg.SEGMENTS_DIR, "all.parquet")
        if os.path.isfile(seg_path):
            df = pd.read_parquet(seg_path)
        else:
            parts = []
            for seg in ("large_cap", "mid_cap", "small_cap"):
                p = os.path.join(cfg.SEGMENTS_DIR, f"{seg}.parquet")
                if os.path.isfile(p):
                    parts.append(pd.read_parquet(p))
            if not parts:
                raise RuntimeError("No segment parquets found. Run process step first.")
            df = pd.concat(parts, ignore_index=True)

    universe_info = None
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

    from pipeline.splits import format_splits_banner, mask_split, resolve_splits, write_splits_file

    df["date"] = pd.to_datetime(df["date"])
    split_bounds = resolve_splits(df, dev_mode=dev_mode)
    splits_path = write_splits_file(split_bounds)
    print(format_splits_banner(split_bounds))

    df, feature_cols, meta_cols = _prepare_frame(df, ml_cfg, settings)

    core = [c for c in ml_cfg.get("feature_columns", {}).get("raw", []) if c in df.columns]
    df = df.dropna(subset=core, how="any")

    os.makedirs(cfg.TABULAR_DIR, exist_ok=True)
    os.makedirs(cfg.SEQUENCES_DIR, exist_ok=True)
    os.makedirs(cfg.DATASETS_DIR, exist_ok=True)

    seq_len = int(settings.get("ml", {}).get("sequence_length", 20))
    if dev_mode:
        seq_len = min(seq_len, 10)
    profiles = ml_cfg.get("label_profiles_for_training", ["reference_swing", "segment_swing"])
    test_enabled = bool(split_bounds.get("test", {}).get("enabled", False))
    summary: dict = {
        "profiles": {},
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "sequence_length": seq_len,
        "universe": universe_info,
        "splits": split_bounds,
        "splits_path": splits_path,
    }

    for profile in profiles:
        target_col = _outcome_column(profile)
        if target_col not in df.columns:
            continue

        sub = df.dropna(subset=[target_col]).copy()
        sub["label"] = sub[target_col].astype(int)

        train = sub[mask_split(sub, "train", split_bounds)]
        val = sub[mask_split(sub, "validation", split_bounds)]
        test = sub[mask_split(sub, "test", split_bounds)] if test_enabled else sub.iloc[0:0]

        keep = meta_cols + feature_cols + ["label", target_col]
        keep = [c for c in keep if c in sub.columns]

        suffix = f"_{profile}" if not top_n else f"_top{top_n}_{profile}"
        train_path = os.path.join(cfg.TABULAR_DIR, f"ml_train{suffix}.parquet")
        val_path = os.path.join(cfg.TABULAR_DIR, f"ml_val{suffix}.parquet")
        test_path = os.path.join(cfg.TABULAR_DIR, f"ml_test{suffix}.parquet")
        train[keep].to_parquet(train_path, index=False)
        val[keep].to_parquet(val_path, index=False)
        if test_enabled and len(test):
            test[keep].to_parquet(test_path, index=False)

        train_seq = _build_sequences(train, feature_cols, target_col, seq_len)
        val_seq = _build_sequences(val, feature_cols, target_col, seq_len)
        test_seq = _build_sequences(test, feature_cols, target_col, seq_len) if test_enabled else {}

        seq_train_path = os.path.join(cfg.SEQUENCES_DIR, f"dl_train{suffix}.npz")
        seq_val_path = os.path.join(cfg.SEQUENCES_DIR, f"dl_val{suffix}.npz")
        seq_test_path = os.path.join(cfg.SEQUENCES_DIR, f"dl_test{suffix}.npz")
        if train_seq:
            np.savez_compressed(seq_train_path, **train_seq)
        if val_seq:
            np.savez_compressed(seq_val_path, **val_seq)
        if test_seq:
            np.savez_compressed(seq_test_path, **test_seq)

        profile_info = {
            "target_column": target_col,
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "train_positive_rate": float(train["label"].mean()) if len(train) else 0.0,
            "val_positive_rate": float(val["label"].mean()) if len(val) else 0.0,
            "test_positive_rate": float(test["label"].mean()) if len(test) else 0.0,
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path if test_enabled and len(test) else None,
            "dl_train_path": seq_train_path if train_seq else None,
            "dl_val_path": seq_val_path if val_seq else None,
            "dl_test_path": seq_test_path if test_seq else None,
            "dl_train_samples": int(train_seq["X"].shape[0]) if train_seq else 0,
            "dl_val_samples": int(val_seq["X"].shape[0]) if val_seq else 0,
            "dl_test_samples": int(test_seq["X"].shape[0]) if test_seq else 0,
            "dl_feature_shape": list(train_seq["X"].shape[1:]) if train_seq else None,
        }
        summary["profiles"][profile] = profile_info

        print(
            f"  {profile}: ML train={len(train)} val={len(val)} test={len(test)} | "
            f"DL train={profile_info['dl_train_samples']} val={profile_info['dl_val_samples']} "
            f"test={profile_info['dl_test_samples']}"
        )

    manifest = {
        "feature_columns": feature_cols,
        "meta_columns": meta_cols,
        "splits": split_bounds,
        "splits_path": splits_path,
        "years_of_history": cfg.YEARS_OF_HISTORY,
        "fetch_recent_days": cfg.fetch_recent_days(),
        "sequence_length": seq_len,
        "null_policy": "drop rows with NaN in raw feature_columns",
        "ml_format": "parquet (one row per symbol-day)",
        "dl_format": "npz with X (samples, seq_len, features), y, symbol, date",
        "raw_ohlcv_dir": cfg.RAW_OHLCV_DIR,
        "duckdb_path": cfg.DB_PATH,
        "raw_bhavcopy_dir": cfg.RAW_BHAVCOPY_DIR,
        "universe": universe_info,
        "profiles": summary["profiles"],
    }
    manifest_name = "manifest.yaml" if not top_n else f"manifest_top{top_n}.yaml"
    manifest_path = os.path.join(cfg.DATASETS_DIR, manifest_name)
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    summary["manifest_path"] = manifest_path
    print(f"Feature manifest: {manifest_path}")
    return summary
