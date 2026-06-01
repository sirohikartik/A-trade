# =============================================================
#  NSE Breakout Scanner  —  start.py
#  Uses Yahoo Finance (yfinance) — no auth, no blocking
# =============================================================

import os
import json
import time
import random
import requests
import pandas as pd
import pandas_ta as ta
from io import BytesIO
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
TARGET_DIR    = "targets"
OUTPUT_FILE   = os.path.join(TARGET_DIR, "filtered_stocks.json")
WORKERS       = 5          # Yahoo is more tolerant than NSE
SLEEP_MIN     = 0.2
SLEEP_MAX     = 0.5
LOOKBACK_DAYS = 400
MIN_BARS      = 220
MIN_PRICE     = 20
MIN_VOLUME    = 100_000
MAX_ATR_PCT   = 0.06
MIN_SCORE     = 65

os.makedirs(TARGET_DIR, exist_ok=True)

# -------------------------------------------------------------
# STEP 1 — Get Nifty 500 symbols
# -------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com",
}

def get_nifty500_symbols():
    url  = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    df   = pd.read_csv(BytesIO(resp.content))
    col  = next(c for c in df.columns if c.strip().lower() == "symbol")
    return df[col].str.strip().tolist()

print("Fetching Nifty 500 symbols...")
symbols = get_nifty500_symbols()
print(f"Got {len(symbols)} symbols\n")

# Yahoo Finance needs ".NS" suffix for NSE stocks
yf_symbols = [s + ".NS" for s in symbols]

# -------------------------------------------------------------
# STEP 2 — Fetch history via yfinance
# -------------------------------------------------------------
def fetch_history(yf_symbol: str) -> pd.DataFrame | None:
    end   = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)

    df = yf.download(
        yf_symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=True,
    )

    if df is None or df.empty:
        return None

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open":   "open",
        "High":   "high",
        "Low":    "low",
        "Close":  "close",
        "Volume": "volume",
    })

    df = df[["open", "high", "low", "close", "volume"]].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    df = df.sort_index()

    return df

# -------------------------------------------------------------
# STEP 3 — Per-symbol scan
# -------------------------------------------------------------
def process_symbol(args):
    symbol, yf_symbol = args
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    try:
        df = fetch_history(yf_symbol)

        if df is None or len(df) < MIN_BARS:
            return None

        # --------------------------------------------------
        # INDICATORS
        # --------------------------------------------------
        df["EMA20"]  = ta.ema(df["close"], length=20)
        df["EMA50"]  = ta.ema(df["close"], length=50)
        df["EMA200"] = ta.ema(df["close"], length=200)
        df["RSI"]    = ta.rsi(df["close"], length=14)

        macd_df           = ta.macd(df["close"])
        df["MACD"]        = macd_df.iloc[:, 0]
        df["MACD_SIGNAL"] = macd_df.iloc[:, 1]

        adx_df   = ta.adx(high=df["high"], low=df["low"], close=df["close"])
        df["ADX"] = adx_df["ADX_14"]
        df["ATR"] = ta.atr(high=df["high"], low=df["low"], close=df["close"])

        latest = df.iloc[-1]

        # --------------------------------------------------
        # PHASE 1 — Basic filters
        # --------------------------------------------------
        if latest["close"] < MIN_PRICE:
            return None

        if latest["volume"] < MIN_VOLUME:
            return None

        atr_pct = latest["ATR"] / latest["close"]
        if atr_pct > MAX_ATR_PCT:
            return None

        # --------------------------------------------------
        # PHASE 2 — Trend & breakout
        # --------------------------------------------------
        trend_ok = (
            latest["close"]
            > latest["EMA20"]
            > latest["EMA50"]
            > latest["EMA200"]
        )
        if not trend_ok:
            return None

        high20 = df["high"].iloc[-21:-1].max()
        if latest["close"] <= high20:
            return None

        avg_vol20    = df["volume"].tail(20).mean()
        volume_ratio = latest["volume"] / avg_vol20
        if volume_ratio <= 2:
            return None

        rsi = latest["RSI"]
        if not (55 <= rsi <= 75):
            return None

        if latest["MACD"] <= latest["MACD_SIGNAL"]:
            return None

        if latest["ADX"] <= 20:
            return None

        # --------------------------------------------------
        # SCORE
        # --------------------------------------------------
        score  = 0
        score += min(volume_ratio * 10, 25)
        score += min(latest["ADX"], 25)
        score += max(0, rsi - 50)
        score += 10

        if score < MIN_SCORE:
            return None

        return {
            "symbol":       symbol,
            "score":        round(score, 2),
            "close":        round(float(latest["close"]), 2),
            "volume_ratio": round(float(volume_ratio), 2),
            "rsi":          round(float(rsi), 2),
            "adx":          round(float(latest["ADX"]), 2),
            "atr_pct":      round(float(atr_pct * 100), 2),
        }

    except Exception as e:
        print(f"  ERROR {symbol}: {e}")
        return None

# -------------------------------------------------------------
# STEP 4 — Run scan
# -------------------------------------------------------------
passed_stocks = []
total = len(symbols)
done  = 0

print(f"Scanning {total} symbols with {WORKERS} workers...\n")

pairs = list(zip(symbols, yf_symbols))

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(process_symbol, p): p[0] for p in pairs}

    for future in as_completed(futures):
        done += 1
        result = future.result()

        if result:
            passed_stocks.append(result)
            print(
                f"  PASS [{done}/{total}] {result['symbol']:15s} "
                f"Score={result['score']:.1f}  RSI={result['rsi']}  "
                f"ADX={result['adx']}  VolRatio={result['volume_ratio']}x"
            )
        elif done % 50 == 0:
            print(f"  ... [{done}/{total}] scanned")

# -------------------------------------------------------------
# STEP 5 — Sort & save
# -------------------------------------------------------------
passed_stocks.sort(key=lambda x: x["score"], reverse=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(passed_stocks, f, indent=4)

print(f"\n{'='*50}")
print(f"Scan complete. {len(passed_stocks)} stocks passed.")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"{'='*50}\n")

if passed_stocks:
    print(f"{'SYMBOL':<15} {'SCORE':>6} {'CLOSE':>8} {'RSI':>6} {'ADX':>6} {'VOL_RATIO':>10}")
    print("-" * 55)
    for s in passed_stocks:
        print(
            f"{s['symbol']:<15} {s['score']:>6.1f} "
            f"{s['close']:>8.2f} {s['rsi']:>6.1f} "
            f"{s['adx']:>6.1f} {s['volume_ratio']:>9.1f}x"
        )
