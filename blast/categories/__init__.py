"""Seven blast category scorers — each returns a sub-score in [0, 1].

Categories map to indicator columns from ``Collector/indicators.py``.
Weights are tuned by the hybrid EA; sub-score formulas stay fixed in v1.
"""

from blast.categories.breakout import score_breakout
from blast.categories.delivery import score_delivery
from blast.categories.momentum import score_momentum
from blast.categories.risk import score_risk
from blast.categories.sector import score_sector
from blast.categories.trend import score_trend
from blast.categories.volume import score_volume

CATEGORY_FUNCS = {
    "trend": score_trend,
    "breakout": score_breakout,
    "volume": score_volume,
    "delivery": score_delivery,
    "momentum": score_momentum,
    "sector": score_sector,
    "risk": score_risk,
}

__all__ = list(CATEGORY_FUNCS.keys()) + ["CATEGORY_FUNCS"]
