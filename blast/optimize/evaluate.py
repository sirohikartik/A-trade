"""Evaluation metrics — Wilson CI, lift, Precision@K."""

from __future__ import annotations

import math
from typing import Any

from blast.optimize.fitness import FitnessContext
from blast.optimize.simplex import CATEGORY_ORDER


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95% CI)."""
    if n <= 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    low = (centre - margin) / denom
    high = (centre + margin) / denom
    return max(0.0, low), min(1.0, high)


def evaluate_weights(
    contexts: dict[str, FitnessContext],
    weights: dict[str, int],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Precision@K, lift, and Wilson CI per train/val/test context."""
    k_values = k_values or [5, 10, 20]
    report: dict[str, Any] = {"weights": weights, "splits": {}}

    for split_name, ctx in contexts.items():
        split_report: dict[str, Any] = {"base_rate": round(ctx.base_rate, 4)}
        for k in k_values:
            details = ctx.precision_at_k(weights, k=k, return_details=True)
            assert isinstance(details, dict)
            n_days = int(details["n_days"])
            prec = float(details["precision_at_k"])
            wins = int(round(prec * k * n_days)) if n_days else 0
            trials = k * n_days
            lo, hi = wilson_ci(wins, trials) if trials else (0.0, 0.0)
            split_report[f"precision_at_{k}"] = round(prec, 4)
            split_report[f"precision_at_{k}_ci"] = [round(lo, 4), round(hi, 4)]
            split_report[f"lift_at_{k}"] = round(prec - ctx.base_rate, 4)
            if k == 10:
                split_report["fitness"] = round(float(details["fitness"]), 4)
                split_report["n_days"] = n_days
                split_report["median_pool_size"] = round(float(details["median_pool_size"]), 1)
        report["splits"][split_name] = split_report

    return report


def format_report(report: dict[str, Any]) -> str:
    """Pretty-print an ``evaluate_weights`` report for logs and CLI."""
    lines = ["Weights: " + ", ".join(f"{k}={v}" for k, v in report["weights"].items())]
    for split_name, info in report["splits"].items():
        lines.append(f"\n[{split_name}]")
        lines.append(f"  base_rate={info['base_rate']:.1%}")
        for key, val in info.items():
            if key.startswith("precision_at_") and not key.endswith("_ci"):
                k = key.replace("precision_at_", "")
                lift = info.get(f"lift_at_{k}", 0)
                ci = info.get(f"precision_at_{k}_ci", [0, 0])
                lines.append(
                    f"  P@{k}={val:.1%}  lift={lift:+.1%}  CI=[{ci[0]:.1%}, {ci[1]:.1%}]"
                )
        if "n_days" in info:
            lines.append(f"  days={info['n_days']}  median_pool={info.get('median_pool_size', 0)}")
    return "\n".join(lines)
