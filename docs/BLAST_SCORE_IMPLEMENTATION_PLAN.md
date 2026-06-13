# Blast Scoring Engine + Hybrid EA — Implementation Plan

> Saved from design session (June 2026). Reference blueprints: `Reference/AI_Stock_Research_Agent_Indian_Markets.pptx`, `Reference/AI_Agent_Tech_Stack_Integration_Indian_Stocks.pptx`.

## Scope decisions

- **News/catalyst removed** from scoring and EA for now. Live news stays a future **alert-stage** filter + AI explain step (Reference Step 10–11), not a scored factor until historical `news_events` exist.
- **7 factors**, weights **sum to exactly 100**. The blueprint’s 10 catalyst points are **redistributed proportionally** across the remaining factors.
- **Location**: new `A-trade/blast/` package; consumes **processed segment parquets** (`data/processed/segments/*.parquet`) with `outcome_reference` and indicator columns from `Collector/indicators.py`.
- **Splits**: train/val/test from `data/splits.yaml` (written by `build_dataset.py` phase 5).
- **Optimizer**: hybrid EA (not pure GA) targeting **Precision@10** on train; **validation** picks the winning run.

---

## Phase 0 — Pipeline cleanup (complete)

Prerequisite before blast implementation:

- Removed ML tabular export, DL `.npz` sequences, and `config/ml.yaml`.
- Single label profile: **reference_swing** → `outcome_reference` (+4%, −2.5%, 3 days, next-open).
- Pipeline ends at processed parquets + splits; no `data/datasets/` tree.
- Run locally: `python build_dataset.py --days 120 --max-symbols 15 --fresh` (smoke) or full `python build_dataset.py --fresh`.

---

## 7-factor model (no news)

Blueprint had 90 points without catalyst. Scale to 100:

| Factor | Blueprint pts | **Initial weight** | Role |
|--------|---------------|------------------|------|
| Trend strength | 20 | **22** | EMA stack, slope, ADX trend |
| Breakout quality | 20 | **22** | 20d/52w highs, close strength |
| Volume surge | 15 | **17** | Volume ratio vs 20d avg |
| Delivery accumulation | 15 | **17** | Delivery % vs threshold & avg |
| Momentum indicators | 10 | **11** | RSI zone, MACD, ADX rising |
| Sector strength | 5 | **6** | Sector breadth (% above EMA20) |
| Risk / liquidity | 5 | **5** | ATR risk, extension, liquidity |
| ~~News/catalyst~~ | ~~10~~ | **0** | Deferred |
| **Total** | | **100** | |

Alert bands (unchanged from Reference): **85+ A+**, **75–84 A**, **65–74 B**, **<65 ignore**.

Formula:

```
blast_score = Σ (w_i × subscore_i)     where each subscore_i ∈ [0, 1], Σ w_i = 100
```

---

## Required indicators (minimal set)

Only columns needed for the 7 sub-scorers plus filters. All already exist in `Collector/indicators.py` and `config/settings.yaml` → `reference_rules` **except sector breadth** (new derived column).

### Per-factor indicator map + initial sub-score logic

Port proven shapes from `stats_testing/steps/scoring.py`; keep formulas stable so EA only tunes **weights**, not sub-score code, in v1.

#### 1. Trend (`subscore_trend`) — weight 22

| Indicator | Type | Use in sub-score |
|-----------|------|------------------|
| `ema_stack_bull` | binary | +0.50 if true |
| `above_ema200` | binary | +0.15 partial if stack false |
| `above_ema50` | binary | +0.10 partial |
| `above_ema20` | binary | +0.05 partial |
| `ema20_slope` | continuous | +min(0.25, slope × 5) if slope > 0 |
| `adx_strong` | binary | +0.25 if ADX > threshold |

`subscore_trend = clip(sum, 0, 1)`

#### 2. Breakout (`subscore_breakout`) — weight 22

| Indicator | Type | Use in sub-score |
|-----------|------|------------------|
| `new_52w_high` | binary | +0.50 |
| `near_52w_high` | binary | +0.35 (if not new 52w) |
| `near_20d_high` | binary | +0.30 |
| `close_position` | continuous | +0.20 if ≥ 0.7 (close in upper range of day) |
| `new_20d_high` | binary | optional tie-break in `explain.py` only |

#### 3. Volume (`subscore_volume`) — weight 17

| Indicator | Type | Use in sub-score |
|-----------|------|------------------|
| `volume_ratio` | continuous | 0 below `volume_ratio_min` (default 2.0); else `0.4 + 0.6 × min((vr - min) / (min × 1.5), 1)` |
| `breakout_volume` | binary | +0.10 bonus (capped at 1.0) |
| `vol_trend_up` | binary | +0.05 bonus |

Thresholds from `config/settings.yaml` → `reference_rules`.

#### 4. Delivery (`subscore_delivery`) — weight 17

| Indicator | Type | Use in sub-score |
|-----------|------|------------------|
| `delivery_pct` | continuous | 0 if NaN or < min (50%); else scale min→0.3, 80%→1.0 |
| `delivery_above_avg` | binary | +0.20 if true |
| `delivery_strong` | binary | used in explain; folded into above_avg path |

If `delivery_pct` missing for a row → `subscore_delivery = 0` (do not block other factors).

#### 5. Momentum (`subscore_momentum`) — weight 11

| Indicator | Type | Use in sub-score |
|-----------|------|------------------|
| `rsi` | continuous | peak score in zone 55–75 (`rsi_bull_zone`); 0 below; +0.10 above 75 |
| `macd_bullish` | binary | +0.30 |
| `macd_hist_rising` | binary | +0.15 |
| `adx_rising` | binary | +0.10 |

#### 6. Sector (`subscore_sector`) — weight 6

| Indicator | Type | Use in sub-score |
|-----------|------|------------------|
| `sector_breadth` | continuous **new** | % of same-sector symbols with close > EMA20 on signal date; map 40%→0, 70%→1 linear |

Build from `archive_v1_backtest/scorer.py` sector breadth pattern + `data/sector_map.csv`. If sector unknown → 0.5 neutral.

#### 7. Risk / liquidity (`subscore_risk`) — weight 5

Scored as **positive** contribution (higher = safer), not a penalty subtracted elsewhere.

| Indicator | Type | Use in sub-score |
|-----------|------|------------------|
| `atr_risk_ok` | binary | +0.50 base if ATR% ≤ max |
| `atr_pct` | continuous | deduct proportionally if above max (inverted to 0–1) |
| `pullback_ema20` | continuous | penalize if pullback > 2.5% extension |
| `liquidity_norm` | continuous **derived** | `log1p(median close×volume over 20d)` scaled 0–1 vs segment ADTV floors |

---

## Indicators NOT in blast score (filter stage only)

These **must not zero out** `blast_score`:

| Feature | Where it goes |
|---------|----------------|
| `market_breadth` | `filters.py` — macro pass if ≥ 0.40 |
| `candle_hammer`, `candle_engulfing` | Removed from score gatekeepers; optional soft filter later |
| `setup_ready` | Diagnostic / explain only; not a hard score gate |
| Min price, min ADTV | `filters.py` from `settings.yaml` filters |

**Eligible pool for ranking**: all symbol-days passing filters; **every** eligible row gets a blast score (may be low).

---

## Hybrid EA optimizer

### Technique stack

| Step | Method | Purpose |
|------|--------|---------|
| **0 — Seeds** | Blueprint defaults + lift-proportional weights + 5 random simplex points | Avoid cold-start |
| **1 — Global** | **Differential Evolution** (`scipy.optimize.differential_evolution`) on 7 softmax params θ | Robust on noisy Precision@10 |
| **2 — Local** | **CMA-ES** or **Nelder-Mead** polish starting from DE best | Fine-tune weight boundaries |
| **3 — Restarts** | 3 independent DE runs (different seeds) | Reduce local-optima luck |
| **4 — Selection** | Pick run with **highest validation Precision@10** (not train) | Anti-overfit gate |
| **5 — Repair** | Round weights to integers summing to 100 (largest remainder) | Deployable YAML |

### Fitness function

```
For each trading day d in split:
  pool_d = rows passing filters.py on day d
  top10_d = symbols with highest blast_score in pool_d (up to 10)
  prec_d = wins(top10_d) / len(top10_d)

precision_at_10 = mean(prec_d) over days where len(pool_d) >= 10

fitness = precision_at_10
        - λ * max(0, 10 - median_daily_pool_size) / 10
        - μ * false_alert_rate_top10
```

- **Train**: maximize fitness (DE/CMA-ES).
- **Val**: select best candidate / restart.
- **Test**: report once, never optimize.

**Constraint**: `w = 100 × softmax(θ)` → always sums to 100.

### Optimizer parameters (starting point)

| Param | Value |
|-------|-------|
| DE population | 60 |
| DE generations | 150 |
| CMA-ES generations | 50 |
| Restarts | 3 |
| λ (sparse penalty) | 0.05 |
| μ (false alert) | 0.03 |
| Per segment | large_cap, mid_cap, small_cap separately |

---

## Derived system structure

### Directory layout

```
A-trade/
  blast/
    __init__.py
    config/
      blast.yaml
    categories/
      trend.py
      breakout.py
      volume.py
      delivery.py
      momentum.py
      sector.py
      risk.py
    scoring.py
    filters.py
    explain.py
    optimize/
      simplex.py
      fitness.py
      lift_seeds.py
      hybrid_ea.py
      evaluate.py
    tune_weights.py
  data/
    sector_map.csv
  docs/
    BLAST_SCORE_IMPLEMENTATION_PLAN.md   # this file
```

### Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `categories/*.py` | Pure functions: row or vector → subscore ∈ [0,1] |
| `scoring.py` | Weighted sum; expose `subscore_*` columns for explain |
| `filters.py` | Liquidity, macro breadth, min price — boolean mask |
| `explain.py` | `{factor: points_earned, reasons: [...]}` for alerts |
| `optimize/fitness.py` | Daily groupby rank, Precision@K, penalties |
| `optimize/hybrid_ea.py` | Run DE→CMA-ES per segment; return best θ |
| `optimize/evaluate.py` | Wilson CI, lift vs base rate, P@5/P@10/P@20 |
| `tune_weights.py` | Load `data/processed/segments/*.parquet` + `data/splits.yaml`; write `data/blast/weights_{segment}.yaml` |

### Config sketch (`blast/config/blast.yaml`)

```yaml
weights_default:
  trend: 22
  breakout: 22
  volume: 17
  delivery: 17
  momentum: 11
  sector: 6
  risk: 5

alert_bands:
  a_plus: 85
  a: 75
  b: 65

thresholds:
  volume_ratio_min: 2.0
  rsi_min: 55
  rsi_max: 75
  delivery_pct_min: 50
  atr_pct_max: 0.06
  max_pullback_ema20: 0.025

optimizer:
  objective: precision_at_10
  de_popsize: 60
  de_maxiter: 150
  cma_maxiter: 50
  n_restarts: 3
  sparse_penalty: 0.05
  false_alert_penalty: 0.03

filters:
  min_market_breadth: 0.40
  use_macro_filter: true
```

---

## Implementation phases

### Phase 0 — Pipeline cleanup

Done. Research pipeline outputs processed parquets and `data/splits.yaml`; ML/DL exports removed.

### Phase 1 — Scoring engine (no EA yet)

1. Scaffold `A-trade/blast/` package
2. Add `sector_map.csv` + `sector_breadth` enrichment
3. Implement 7 category scorers + `scoring.py` + `filters.py`
4. Smoke test: score one parquet split; print top-10 per day with explain breakdown
5. Verify all filtered rows can receive a non-forced-zero score

### Phase 2 — Fitness + baselines

1. `fitness.py` with daily Precision@10
2. Report metrics for blueprint default weights on train/val/test
3. Lift-proportional seed baseline comparison

### Phase 3 — Hybrid EA

1. `hybrid_ea.py` + `tune_weights.py` CLI
2. Run per segment; val-select best of 3 restarts
3. Write `data/blast/weights_{segment}.yaml`
4. Final test report (single shot)

### Phase 4 — Integration

1. Done: `blast/live.py` + `Collector/start.py` use tuned per-segment weights
2. Add news as alert filter only (not scored)
3. Optional catalyst sub-score when `news_events` table exists

---

## Fitness label (v1)

Use existing `outcome_reference` from `config/labels.yaml`: +4% target, −2.5% stop, 3 days, next-open entry.

---

## Success criteria

| Metric | Target (directional) |
|--------|----------------------|
| Validation Precision@10 | Beat blueprint defaults by ≥ 5 pp |
| Val lift vs base rate | Positive on at least 2 of 3 segments |
| Median daily pool size | ≥ 10 symbols after filters |
| Overfit gap train−val P@10 | < 10 pp |
| Explainability | Every top-10 row has factor breakdown |

---

## Implementation checklist

- [x] Add `A-trade/blast/` package: categories, scoring.py, filters.py, config/blast.yaml
- [x] Add sector_map + daily sector_breadth column to processed parquet
- [x] Implement 7 category scorers (0-1) with blueprint default weights
- [x] `blast/optimize/fitness.py`: daily Precision@10, sparse-day penalty, false-alert rate
- [x] `blast/optimize/hybrid_ea.py`: lift seeds + DE + Nelder-Mead polish, val-selected winner
- [x] `blast/tune_weights.py` CLI + YAML output (weights, metrics train/val/test)
- [x] `blast/live.py` + `Collector/start.py` live scanner integration
- [ ] Optional: align labeling with pessimistic same-bar rule for EA fitness parity
