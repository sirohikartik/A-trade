"""DuckDB storage for OHLCV history and scan results."""

from __future__ import annotations

import os
import threading
from datetime import datetime

import duckdb
import pandas as pd

_lock = threading.Lock()


def db_path(base_dir: str) -> str:
    return os.path.join(base_dir, "data", "atrade.duckdb")


def connect(path: str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return duckdb.connect(path, read_only=read_only)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv_daily (
            symbol   VARCHAR NOT NULL,
            date     DATE NOT NULL,
            open     DOUBLE,
            high     DOUBLE,
            low      DOUBLE,
            close    DOUBLE,
            volume   DOUBLE,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS symbol_meta (
            symbol      VARCHAR PRIMARY KEY,
            bars        INTEGER,
            first_date  DATE,
            last_date   DATE,
            updated_at  TIMESTAMP
        )
        """
    )
    conn.execute("CREATE SEQUENCE IF NOT EXISTS scan_run_id_seq START 1")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_runs (
            run_id                 BIGINT PRIMARY KEY DEFAULT nextval('scan_run_id_seq'),
            run_date               DATE NOT NULL,
            years_requested        INTEGER,
            lookback_days          INTEGER,
            symbols_total          INTEGER,
            symbols_with_history   INTEGER,
            symbols_passed         INTEGER,
            created_at             TIMESTAMP DEFAULT current_timestamp
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_passed (
            run_id         BIGINT NOT NULL,
            symbol         VARCHAR NOT NULL,
            score          DOUBLE,
            close          DOUBLE,
            volume_ratio   DOUBLE,
            rsi            DOUBLE,
            adx            DOUBLE,
            atr_pct        DOUBLE,
            bars           INTEGER,
            history_from   DATE,
            history_to     DATE,
            PRIMARY KEY (run_id, symbol)
        )
        """
    )


def _df_to_ohlcv_rows(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    date_col = out.columns[0]
    out = out.rename(columns={date_col: "date"})
    out["symbol"] = symbol
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out[["symbol", "date", "open", "high", "low", "close", "volume"]]


def save_ohlcv(path: str, symbol: str, df: pd.DataFrame) -> None:
    rows = _df_to_ohlcv_rows(symbol, df)
    first_date = rows["date"].min()
    last_date = rows["date"].max()
    now = datetime.utcnow()

    with _lock:
        conn = connect(path)
        try:
            init_schema(conn)
            conn.execute("DELETE FROM ohlcv_daily WHERE symbol = ?", [symbol])
            conn.register("_batch", rows)
            conn.execute(
                """
                INSERT INTO ohlcv_daily
                SELECT symbol, date, open, high, low, close, volume FROM _batch
                """
            )
            conn.unregister("_batch")
            conn.execute("DELETE FROM symbol_meta WHERE symbol = ?", [symbol])
            conn.execute(
                """
                INSERT INTO symbol_meta
                (symbol, bars, first_date, last_date, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [symbol, len(rows), first_date, last_date, now],
            )
        finally:
            conn.close()


def load_ohlcv(path: str, symbol: str, min_bars: int) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    with _lock:
        conn = connect(path, read_only=True)
        try:
            df = conn.execute(
                """
                SELECT date, open, high, low, close, volume
                FROM ohlcv_daily
                WHERE symbol = ?
                ORDER BY date
                """,
                [symbol],
            ).df()
        finally:
            conn.close()

    if df.empty or len(df) < min_bars:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df.sort_index()


def save_scan_run(path: str, summary: dict, passed: list[dict]) -> int:
    with _lock:
        conn = connect(path)
        try:
            init_schema(conn)
            run_id = conn.execute(
                """
                INSERT INTO scan_runs (
                    run_date, years_requested, lookback_days,
                    symbols_total, symbols_with_history, symbols_passed
                )
                VALUES (?, ?, ?, ?, ?, ?)
                RETURNING run_id
                """,
                [
                    summary["run_date"],
                    summary["years_requested"],
                    summary["lookback_days"],
                    summary["symbols_total"],
                    summary["symbols_with_history"],
                    summary["symbols_passed_scan"],
                ],
            ).fetchone()[0]

            if passed:
                rows = pd.DataFrame(passed)
                rows["run_id"] = run_id
                rows = rows[
                    [
                        "run_id",
                        "symbol",
                        "score",
                        "close",
                        "volume_ratio",
                        "rsi",
                        "adx",
                        "atr_pct",
                        "bars",
                        "history_from",
                        "history_to",
                    ]
                ]
                rows["history_from"] = pd.to_datetime(rows["history_from"]).dt.date
                rows["history_to"] = pd.to_datetime(rows["history_to"]).dt.date
                conn.register("_passed", rows)
                conn.execute("INSERT INTO scan_passed SELECT * FROM _passed")
                conn.unregister("_passed")

            return int(run_id)
        finally:
            conn.close()
