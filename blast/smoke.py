#!/usr/bin/env python3
"""Smoke test CLI — score one session date and print top-N with factor breakdown."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from blast.enrich import load_segment_frame
from blast.explain import explain_row
from blast.scoring import compute_blast_score, rank_eligible_day


def main() -> None:
    parser = argparse.ArgumentParser(description="Blast score smoke test")
    parser.add_argument("--segment", choices=["large_cap", "mid_cap", "small_cap"], default=None)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest date in data)")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    df = load_segment_frame(args.segment)
    if args.date:
        day = df[df["date"] == args.date].copy()
        label = args.date
    else:
        latest = df["date"].max()
        day = df[df["date"] == latest].copy()
        label = str(latest.date())

    top = rank_eligible_day(day, segment=args.segment, top_n=args.top_n)
    print(f"Blast smoke — {label} — segment={args.segment or 'all'} — pool={len(day)} eligible={len(top)}")
    print("-" * 72)

    if top.empty:
        print("No eligible symbols for this day.")
        return

    for _, row in top.iterrows():
        info = explain_row(row)
        factors = ", ".join(f"{k}:{v}" for k, v in info["factors"].items())
        print(
            f"{row['symbol']:<12} score={info['blast_score']:5.1f} "
            f"grade={info['alert_grade']:<6} outcome_ref={row.get('outcome_reference', 'n/a')}"
        )
        print(f"             {factors}")


if __name__ == "__main__":
    main()
