"""Weighted blast score from seven category sub-scores.

Public API:
    compute_subscores   — add subscore_* columns in [0, 1]
    compute_blast_score — weighted sum → blast_score in [0, 100]
    grade_alerts        — map score to A+ / A / B / ignore bands
    rank_eligible_day   — filter, score, and return top-N for one session date
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from blast._config import get_thresholds, get_weights, load_blast_config
from blast.categories import CATEGORY_FUNCS
from blast.filters import eligible_mask


def compute_subscores(df: pd.DataFrame, thresholds: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run each category scorer and attach ``subscore_<category>`` columns."""
    thresholds = thresholds or get_thresholds()
    out = df.copy()
    for name, fn in CATEGORY_FUNCS.items():
        out[f"subscore_{name}"] = fn(out, thresholds)
    return out


def compute_blast_score(
    df: pd.DataFrame,
    weights: dict[str, int] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Combine sub-scores with integer weights (sum 100) into ``blast_score``."""
    weights = weights or get_weights()
    thresholds = thresholds or get_thresholds()
    out = compute_subscores(df, thresholds)

    total = np.zeros(len(out))
    for name, w in weights.items():
        col = f"subscore_{name}"
        if w > 0 and col in out.columns:
            total += w * out[col].fillna(0).to_numpy()

    out["blast_score"] = np.clip(total, 0, 100)
    return out


def grade_alerts(scores: pd.Series) -> pd.Series:
    """Map numeric scores to alert grades using bands from ``blast/config/blast.yaml``."""
    bands = load_blast_config().get("alert_bands", {})
    a_plus = float(bands.get("a_plus", 85))
    a = float(bands.get("a", 75))
    b = float(bands.get("b", 65))

    return pd.Series(
        np.select(
            [scores >= a_plus, scores >= a, scores >= b],
            ["A+", "A", "B"],
            default="ignore",
        ),
        index=scores.index,
    )


def rank_eligible_day(
    day_df: pd.DataFrame,
    segment: str | None = None,
    top_n: int = 10,
    weights: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Score and return top-N eligible symbols for one date."""
    mask = eligible_mask(day_df, segment=segment)
    pool = day_df.loc[mask].copy()
    if pool.empty:
        return pool

    scored = compute_blast_score(pool, weights=weights)
    scored["alert_grade"] = grade_alerts(scored["blast_score"])
    return (
        scored.sort_values("blast_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
