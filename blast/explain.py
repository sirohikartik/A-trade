"""Human-readable factor breakdown for ranked blast picks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from blast._config import get_weights


def explain_row(row: pd.Series, weights: dict[str, int] | None = None) -> dict[str, Any]:
    """Return weighted point contribution per category for one scored row."""
    weights = weights or get_weights()
    factors: dict[str, float] = {}
    reasons: list[str] = []

    for name, w in weights.items():
        sub_col = f"subscore_{name}"
        if w <= 0 or sub_col not in row.index:
            continue
        sub = float(row[sub_col]) if pd.notna(row[sub_col]) else 0.0
        points = round(w * sub, 2)
        factors[name] = points
        if sub >= 0.7:
            reasons.append(f"Strong {name} ({sub:.0%})")

    return {
        "symbol": row.get("symbol"),
        "date": str(row.get("date", ""))[:10],
        "blast_score": round(float(row.get("blast_score", 0)), 1),
        "alert_grade": row.get("alert_grade"),
        "factors": factors,
        "reasons": reasons,
    }
