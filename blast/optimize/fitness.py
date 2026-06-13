"""Daily Precision@K fitness with sparse-pool and false-alert penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from blast._config import get_thresholds, load_blast_config
from blast.filters import eligible_mask
from blast.optimize.simplex import CATEGORY_ORDER, weights_to_vector
from blast.optimize.splits import mask_split
from blast.scoring import compute_subscores


@dataclass
class DayPool:
    """Eligible symbols for one trading day — precomputed sub-scores and labels."""

    subscores: np.ndarray  # (n_eligible, n_categories)
    outcomes: np.ndarray  # (n_eligible,)


@dataclass
class FitnessContext:
    """Precomputed pools for fast weight search."""

    segment: str
    split_name: str
    day_pools: list[DayPool]
    base_rate: float
    category_order: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.category_order is None:
            self.category_order = list(CATEGORY_ORDER)

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        segment: str,
        split_name: str,
        min_pool: int = 10,
        top_k: int = 10,
    ) -> FitnessContext:
        """Build daily pools from a split slice for fast weight search."""
        cfg = load_blast_config().get("optimizer", {})
        min_pool = int(cfg.get("min_pool_size", min_pool))
        top_k = int(cfg.get("top_k", top_k))

        split_mask = mask_split(df, split_name)
        sub = compute_subscores(df.loc[split_mask].copy(), get_thresholds())
        eligible = eligible_mask(sub, segment=segment)
        sub = sub.loc[eligible].copy()

        if sub.empty:
            return cls(segment=segment, split_name=split_name, day_pools=[], base_rate=0.0)

        base_rate = float(sub["outcome_reference"].mean())
        sub_cols = [f"subscore_{c}" for c in CATEGORY_ORDER]
        sub_matrix = sub[sub_cols].fillna(0).to_numpy(dtype=float)
        outcomes = sub["outcome_reference"].fillna(0).to_numpy(dtype=float)
        dates = sub["date"].to_numpy()

        day_pools: list[DayPool] = []
        for date in np.unique(dates):
            idx = np.where(dates == date)[0]
            if len(idx) < min_pool:
                continue
            day_pools.append(
                DayPool(
                    subscores=sub_matrix[idx],
                    outcomes=outcomes[idx],
                )
            )

        return cls(
            segment=segment,
            split_name=split_name,
            day_pools=day_pools,
            base_rate=base_rate,
        )

    def precision_at_k(
        self,
        weights: dict[str, int] | np.ndarray,
        k: int = 10,
        return_details: bool = False,
    ) -> float | dict[str, Any]:
        """Mean Precision@K across days, minus sparse-pool and false-alert penalties."""
        cfg = load_blast_config().get("optimizer", {})
        sparse_penalty = float(cfg.get("sparse_penalty", 0.05))
        false_alert_penalty = float(cfg.get("false_alert_penalty", 0.03))

        if isinstance(weights, dict):
            w = weights_to_vector(weights, self.category_order)
        else:
            w = np.asarray(weights, dtype=float)

        precisions: list[float] = []
        false_rates: list[float] = []
        pool_sizes: list[int] = []

        for pool in self.day_pools:
            scores = pool.subscores @ w
            n = len(scores)
            if n < k:
                continue
            top = np.argpartition(-scores, k - 1)[:k]
            wins = pool.outcomes[top]
            precisions.append(float(wins.mean()))
            false_rates.append(float(1.0 - wins.mean()))
            pool_sizes.append(n)

        if not precisions:
            result = {
                "precision_at_k": 0.0,
                "fitness": -1.0,
                "n_days": 0,
                "median_pool_size": 0,
                "false_alert_rate": 1.0,
                "base_rate": self.base_rate,
            }
            return result if return_details else 0.0

        p_at_k = float(np.mean(precisions))
        median_pool = float(np.median(pool_sizes))
        false_alert = float(np.mean(false_rates))

        sparse = max(0.0, k - median_pool) / k
        fitness = p_at_k - sparse_penalty * sparse - false_alert_penalty * false_alert

        result = {
            "precision_at_k": p_at_k,
            "fitness": fitness,
            "n_days": len(precisions),
            "median_pool_size": median_pool,
            "false_alert_rate": false_alert,
            "base_rate": self.base_rate,
            "lift": p_at_k - self.base_rate,
        }
        return result if return_details else fitness
