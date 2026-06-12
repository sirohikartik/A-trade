"""NSE bhavcopy delivery data fetch."""

from __future__ import annotations

import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal
import requests

import config as cfg

_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/csv,text/plain,*/*",
                "Referer": "https://www.nseindia.com/report-listing/equDailyRpt",
            }
        )
        _thread_local.session = session
    return _thread_local.session


def get_nse_trading_days(from_date: str, to_date: str) -> list:
    nse = mcal.get_calendar("NSE")
    schedule = nse.schedule(start_date=from_date, end_date=to_date)
    return schedule.index.date.tolist()


def _build_api_url(day: date) -> str:
    date_str = day.strftime("%d-%b-%Y")
    return (
        "https://www.nseindia.com/api/reports?archives="
        '[{"name":"Full Bhavcopy and Security Deliverable data",'
        '"type":"archives","category":"capital-market","section":"equities"}]'
        f"&date={date_str}&type=equities&mode=single"
    )


def fetch_bhavcopy_for_date(day: date) -> tuple[date, str, str | None]:
    os.makedirs(cfg.RAW_BHAVCOPY_DIR, exist_ok=True)
    save_path = os.path.join(cfg.RAW_BHAVCOPY_DIR, f"{day}.parquet")
    if os.path.exists(save_path):
        return day, "skip", None

    session = _get_session()
    try:
        resp = session.get(_build_api_url(day), timeout=25)
        resp.raise_for_status()

        ct = resp.headers.get("content-type", "")
        if "csv" not in ct and "text/plain" not in ct:
            return day, "fail", f"Unexpected content-type: {ct}"
        if len(resp.content) < 100:
            return day, "fail", f"Response too small ({len(resp.content)} bytes)"

        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if "series" in df.columns:
            df = df[df["series"].astype(str).str.strip() == "EQ"]

        df["date"] = day

        for col in list(df.columns):
            if "deliv_qty" in col or "deliv_qnty" in col or col == "deliverable_qty":
                df = df.rename(columns={col: "delivery_qty"})
            elif "deliv_per" in col or col in ("deliverable_per", "deliverable_pct"):
                df = df.rename(columns={col: "delivery_pct"})

        if "delivery_pct" not in df.columns:
            for col in df.columns:
                if "deliv" in col and "per" in col:
                    df = df.rename(columns={col: "delivery_pct"})
                    break

        if "delivery_pct" not in df.columns:
            return day, "fail", f"No delivery_pct column. Cols: {list(df.columns)[:10]}"

        if "delivery_qty" not in df.columns:
            for col in df.columns:
                if "deliv" in col and ("qty" in col or "qnty" in col):
                    df = df.rename(columns={col: "delivery_qty"})
                    break

        df["delivery_pct"] = pd.to_numeric(
            df["delivery_pct"].astype(str).str.replace("-", "").str.strip(),
            errors="coerce",
        )
        if "delivery_qty" in df.columns:
            df["delivery_qty"] = pd.to_numeric(
                df["delivery_qty"].astype(str).str.replace("-", "").str.replace(",", "").str.strip(),
                errors="coerce",
            )

        df["symbol"] = df["symbol"].astype(str).str.strip()
        keep = ["symbol", "date", "delivery_pct"]
        if "delivery_qty" in df.columns:
            keep.insert(2, "delivery_qty")

        result = df[keep].dropna(subset=["delivery_pct"])
        result.to_parquet(save_path, index=False)
        return day, "ok", None

    except Exception as exc:
        return day, "fail", f"{type(exc).__name__}: {exc}"


def fetch_all_delivery(from_date: str | None = None, to_date: str | None = None) -> None:
    from tqdm import tqdm

    if to_date is None:
        to_date = date.today().isoformat()
    if from_date is None:
        start = date.today() - timedelta(days=cfg.lookback_days())
        from_date = start.isoformat()

    os.makedirs(cfg.RAW_BHAVCOPY_DIR, exist_ok=True)
    trading_days = get_nse_trading_days(from_date, to_date)
    pending = [
        d
        for d in trading_days
        if not os.path.exists(os.path.join(cfg.RAW_BHAVCOPY_DIR, f"{d}.parquet"))
    ]

    print(f"Delivery fetch: {len(trading_days)} trading days, {len(pending)} pending")

    if not pending:
        print("All bhavcopy dates already cached.")
        return

    failed: list[tuple[str, str]] = []
    ok = 0

    with ThreadPoolExecutor(max_workers=cfg.DELIVERY_WORKERS) as pool:
        futures = {pool.submit(fetch_bhavcopy_for_date, day): day for day in pending}
        with tqdm(total=len(pending), desc="Bhavcopy raw", unit="day") as bar:
            for future in as_completed(futures):
                day, status, error = future.result()
                if status == "ok":
                    ok += 1
                elif status == "fail":
                    failed.append((str(day), error or ""))
                bar.update(1)
                bar.set_postfix(ok=ok, fail=len(failed))

    print(f"Bhavcopy done: {ok} succeeded, {len(failed)} failed")
    if failed:
        fail_path = os.path.join(cfg.DATA_DIR, "raw", "failed_bhavcopy_dates.csv")
        pd.DataFrame(failed, columns=["date", "error"]).to_csv(fail_path, index=False)
        print(f"Failed dates logged to {fail_path}")


def load_all_delivery() -> pd.DataFrame:
    import glob

    files = glob.glob(os.path.join(cfg.RAW_BHAVCOPY_DIR, "*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
