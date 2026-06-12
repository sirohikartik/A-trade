# A-trade

Indian equity research pipeline built around **Nifty 500**. Fetches daily OHLCV and NSE bhavcopy delivery data, computes Reference-aligned technical indicators, labels swing-trade outcomes, and exports **ML tabular** and **DL sequence** datasets with one command.

## Quick start

```powershell
cd A-trade
pip install -r requirements.txt
python build_dataset.py
```

That builds **both** ML parquet and DL `.npz` files in phase 5. There is no separate DL script.

### Recommended dev / smoke test

```powershell
# Wipe generated data, fetch 120 days, 15 symbols, include bhavcopy
python build_dataset.py --days 120 --max-symbols 15 --fresh
```

Use `--skip-delivery` only when you want a faster run without delivery features.

## Commands cheat sheet

Run from the `A-trade/` folder:

| Goal | Command |
|------|---------|
| Full pipeline — ML + DL, all Nifty 500 | `python build_dataset.py` |
| Full pipeline — top N by train-period edge | `python build_dataset.py --top-n 100` |
| Rebuild from cached OHLCV (skip download) | `python build_dataset.py --skip-fetch --skip-delivery` |
| Dev smoke test (longer window + bhavcopy) | `python build_dataset.py --days 120 --max-symbols 15 --fresh` |
| Fast dev (no delivery) | `python build_dataset.py --days 30 --max-symbols 20 --skip-delivery` |

### CLI flags

| Flag | Description |
|------|-------------|
| `--days N` | Fetch only last N calendar days; lowers min bars; auto 60/20/20 split |
| `--max-symbols N` | Limit to first N Nifty 500 tickers |
| `--top-n N` | Rank on train window only; export top N symbols |
| `--skip-fetch` | Reuse cached DuckDB OHLCV |
| `--skip-delivery` | Skip NSE bhavcopy (no `delivery_pct` from exchange) |
| `--refresh` | Force re-download OHLCV from Yahoo |
| `--fresh` | Delete entire `data/` folder and rebuild from scratch |

## Pipeline (five phases)

| Phase | What it does | Output |
|-------|----------------|--------|
| 1 | Download Nifty 500 OHLCV (Yahoo) | `data/atrade.duckdb` + `data/raw/ohlcv/{SYMBOL}.parquet` |
| 2 | Download NSE bhavcopy delivery % | `data/raw/bhavcopy/{YYYY-MM-DD}.parquet` |
| 3 | Classify large/mid/small cap | `data/processed/segment_map.parquet` |
| 4 | Indicators + dual labels | `data/processed/combined/`, `data/processed/segments/` |
| 5 | ML + DL datasets | `data/datasets/tabular/`, `data/datasets/sequences/` |

Progress bars (`tqdm`) run on OHLCV fetch, bhavcopy fetch, indicators, and labeling. Symbol-level work uses parallel thread pools (`fetch.workers`, `fetch.delivery_workers` in config).

## Output paths

| Artifact | Path |
|----------|------|
| ML tabular | `data/datasets/tabular/ml_{train,val,test}_{profile}.parquet` |
| DL sequences | `data/datasets/sequences/dl_{train,val,test}_{profile}.npz` |
| Resolved splits | `data/datasets/splits.yaml` |
| Feature manifest | `data/datasets/manifest.yaml` |
| Raw OHLCV | `data/raw/ohlcv/{SYMBOL}.parquet` |
| DuckDB | `data/atrade.duckdb` (table `ohlcv_daily`) |
| Bhavcopy | `data/raw/bhavcopy/{YYYY-MM-DD}.parquet` |
| Bhavcopy failures | `data/raw/failed_bhavcopy_dates.csv` (if any) |

With `--top-n N`, filenames include `top{N}` (e.g. `ml_train_top100_reference_swing.parquet`).

---

## Dataset shapes and columns

### ML tabular (parquet)

**Layout:** one row per **symbol-day**.

**Typical shape:** `(rows, 57)` where rows depend on universe, history, and split.

| Column group | Count | Names |
|--------------|-------|-------|
| Meta | 3 | `symbol`, `date`, `segment` |
| Target | 1 | `outcome_reference` or `outcome_segment` |
| Features | 52 | See feature list below |
| Training alias | 1 | `label` (copy of target) |

**Example row counts** (15 symbols, `--days 120`, dev auto-split):

| File | Rows | Target | Positive rate (approx.) |
|------|------|--------|-------------------------|
| `ml_train_reference_swing.parquet` | 417 | `outcome_reference` | 23% |
| `ml_val_reference_swing.parquet` | 297 | `outcome_reference` | 37% |
| `ml_test_reference_swing.parquet` | 243 | `outcome_reference` | 26% |
| `ml_train_segment_swing.parquet` | 417 | `outcome_segment` | 16% |
| `ml_val_segment_swing.parquet` | 297 | `outcome_segment` | 42% |
| `ml_test_segment_swing.parquet` | 243 | `outcome_segment` | 19% |

**Feature columns (52)** — defined in `config/ml.yaml`, extended in export:

| Type | Examples |
|------|----------|
| Raw indicators | `rsi`, `adx`, `atr_pct`, `volume_ratio`, `ema20_slope`, `bb_width`, `delivery_pct`, `market_breadth`, `macd_hist`, `stoch_k`, … |
| Binary flags | `ema_stack_bull`, `setup_ready`, `delivery_strong`, `new_20d_high`, `breakout_volume`, … |
| Z-scores | `rsi_z`, `adx_z`, `delivery_pct_z`, … (rolling window in `settings.yaml` → `ml.rolling_zscore_window`) |
| Lags | `rsi_lag1..3`, `volume_ratio_lag1..3`, `macd_hist_lag1..3` |

Rows with NaN in core raw features are dropped before export (`null_policy` in manifest).

**Inspect in Python:**

```python
import pandas as pd

df = pd.read_parquet("data/datasets/tabular/ml_train_reference_swing.parquet")
print(df.shape)
print(df.columns.tolist())
print(df[["symbol", "date", "segment", "outcome_reference", "rsi", "delivery_pct", "label"]].head())
```

### DL sequences (npz)

**Layout:** one sample per **symbol-day**; each sample is the last `seq_len` timesteps of features ending on that date.

**Typical `X` shape:** `(samples, seq_len, n_features)` e.g. `(117, 10, 52)` in dev mode.

| Key | Shape | Description |
|-----|-------|-------------|
| `X` | `(N, seq_len, F)` | Feature windows (float32) |
| `y` | `(N,)` | Binary label (int8) |
| `symbol` | `(N,)` | Ticker string |
| `date` | `(N,)` | Signal date (end of window) |

**Example sample counts** (same 15-symbol / 120-day run):

| File | X shape | Notes |
|------|---------|-------|
| `dl_train_reference_swing.npz` | (117, 10, 52) | Fewer than ML train rows (needs full window) |
| `dl_val_reference_swing.npz` | (147, 10, 52) | |
| `dl_test_reference_swing.npz` | (93, 10, 52) | |

`segment_swing` NPZ files use the same `X` but different `y` (segment-specific outcomes).

**Why ML rows > DL samples:** ML keeps every clean symbol-day. DL drops days that do not have `seq_len` prior bars in the same split.

**Load in Python:**

```python
import numpy as np

d = np.load("data/datasets/sequences/dl_train_reference_swing.npz", allow_pickle=True)
X, y, symbols, dates = d["X"], d["y"], d["symbol"], d["date"]
print(X.shape)  # (samples, seq_len, features)
print(symbols[0], dates[0], y[0])
print(X[0, -1, :5])  # last timestep, first 5 features
```

---

## Train / validation / test splits

### Production (no `--days`)

Edit **`config/splits.yaml`**:

| Split | Default range |
|-------|----------------|
| Train | earliest data → `2023-06-30` |
| Validation | `2023-07-01` → `2024-12-31` |
| Test | `2025-01-01` → latest (`enabled: true`) |

`null` start/end means “use earliest/latest bar in processed data”.

### Dev mode (`--days N`)

Splits are computed automatically from loaded trading days using ratios in `config/splits.yaml` → `dev_auto_split`:

```yaml
dev_auto_split:
  train_ratio: 0.60
  validation_ratio: 0.20
  test_ratio: 0.20
```

**Example resolved ranges** (`--days 120`, 15 symbols) written to `data/datasets/splits.yaml`:

| Split | Start | End |
|-------|--------|-----|
| Train | 2025-12-29 | 2026-03-25 |
| Validation | 2026-03-27 | 2026-04-28 |
| Test | 2026-04-29 | 2026-05-27 |

After every build, **`data/datasets/splits.yaml`** and **`data/datasets/manifest.yaml`** record the actual ranges used. Top-N ranking always uses the **train** split only (no validation leakage).

---

## Configuration reference

All YAML lives under `config/`. Python loads via `Collector/config.py`.

### `config/settings.yaml`

| Section | Key | Default | Purpose |
|---------|-----|---------|---------|
| `data` | `years_of_history` | `5` | Production OHLCV lookback |
| `data` | `calendar_buffer_days` | `45` | Extra days for indicators / bhavcopy |
| `data` | `fetch_recent_days` | `null` | Set in YAML or override with `--days` |
| `filters` | `min_history_days` | `250` | Min bars per symbol (production) |
| `filters` | `dev_min_history_days` | `5` | Min bars when `--days` is set |
| `indicators` | `ema_periods`, `rsi_period`, … | see file | Passed to `indicators.py` |
| `reference_rules` | `rsi_min`, `volume_ratio_min`, … | see file | Setup / breakout thresholds |
| `ml` | `rolling_zscore_window` | `60` | Z-score lookback |
| `ml` | `lag_periods` | `[1, 2, 3]` | Lag feature offsets |
| `ml` | `sequence_length` | `20` | DL window length (capped at **10** in dev `--days` mode) |
| `universe` | `min_setup_signals` | `8` | Min setups to rank for `--top-n` |
| `universe` | `ranking_weights` | see file | Top-N composite score weights |
| `fetch` | `workers` | `5` | Parallel OHLCV downloads |
| `fetch` | `delivery_workers` | `8` | Parallel bhavcopy downloads |

**Lookback math:**

- Production: `years_of_history * 365 + calendar_buffer_days`
- Dev (`--days N`): `N + calendar_buffer_days` (used for both Yahoo and bhavcopy)

### `config/splits.yaml`

Production date boundaries and dev auto-split ratios (see above).

### `config/labels.yaml`

| Profile | Target | Stop | Horizon | Column |
|---------|--------|------|---------|--------|
| `reference_swing` | +4% | -2.5% | 3 days | `outcome_reference` |
| `segment_swing` | 8–12% (by cap) | 4–6% | 10–15 days | `outcome_segment` |

Entry is simulated at **next-day open**.

### `config/ml.yaml`

Lists `feature_columns` (raw + binary), `zscore_columns`, `lag_features`, and `label_profiles_for_training`. Add or remove features here to change exported columns (re-run phase 5 or full pipeline).

---

## Top-N universe selection

When `--top-n` is set, symbols are ranked using **only the train split**.

| Signal bucket | Weight | Rule |
|---------------|--------|------|
| Setup win rate | 30% | `outcome_reference` when `setup_ready == 1` |
| Breakout win rate | 20% | `new_20d_high` + volume + ATR filter |
| Momentum win rate | 15% | RSI zone + MACD + ADX rising |
| Delivery win rate | 10% | `delivery_strong == 1` |
| Trend consistency | 10% | % days `ema_stack_bull` |
| Volume edge | 5% | Avg volume ratio on setup days |
| Liquidity | 5% | Median daily value (log-scaled) |
| Avg setup return | 5% | Mean swing return on setups |

Outputs: `data/processed/universe_rankings.parquet`, `data/datasets/universe_top_n.json`, `data/processed/segments/top_N.parquet`.

---

## Indicators

Implemented in `Collector/indicators.py`:

- Trend: EMA 20/50/200, stack, slope, pullback
- Momentum: RSI, MACD, Stochastic, ADX
- Volatility: ATR, Bollinger width / squeeze / breakout
- Volume: ratio, trend
- Breakout: 20d / 52w highs, `breakout_volume`
- Delivery: `delivery_pct` from bhavcopy, above avg, strong (>50%)
- Composite: `setup_ready`

Thresholds: `config/settings.yaml` → `reference_rules`.

---

## Project layout

```
A-trade/
  build_dataset.py          # Main entry — ML + DL export
  Collector/
    start.py                # Live breakout scanner
    config.py               # Paths + YAML loader
    db.py                   # DuckDB OHLCV
    indicators.py
    labeling.py
    segments.py
    delivery.py             # NSE bhavcopy
  pipeline/
    fetch.py                # Bulk OHLCV (parallel + tqdm)
    process.py              # Indicators + labels
    splits.py               # Train/val/test resolution
    universe.py             # Top-N ranking
    features.py             # ML parquet + DL npz
  config/
    settings.yaml
    splits.yaml
    labels.yaml
    ml.yaml
  data/                     # Generated (gitignored) — delete with --fresh
```

## Live scanner (separate)

```powershell
python Collector/start.py
python Collector/start.py --refresh
```

Uses a lighter inline indicator set for **today's** breakouts. The dataset pipeline uses the full `indicators.py` module for research.

## Cleaning generated data

```powershell
# Option 1: CLI flag on next run
python build_dataset.py --fresh ...

# Option 2: manual
Remove-Item -Recurse -Force data
```

Everything under `data/` is gitignored. Only config and source are versioned.
