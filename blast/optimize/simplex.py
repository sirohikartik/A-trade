"""Softmax weights and integer repair (sum = 100)."""

from __future__ import annotations

import numpy as np

CATEGORY_ORDER = [
    "trend",
    "breakout",
    "volume",
    "delivery",
    "momentum",
    "sector",
    "risk",
]


def softmax_weights(theta: np.ndarray, categories: list[str] | None = None) -> dict[str, int]:
    """Map unconstrained logits to integer weights summing to 100."""
    categories = categories or CATEGORY_ORDER
    theta = np.asarray(theta, dtype=float)
    if len(theta) != len(categories):
        raise ValueError(f"Expected {len(categories)} weights, got {len(theta)}")

    shifted = theta - theta.max()
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    from blast._config import load_blast_config

    min_w = int(load_blast_config().get("optimizer", {}).get("min_weight_per_category", 3))
    raw = {cat: 100.0 * p for cat, p in zip(categories, probs)}
    return repair_integer_weights(raw, categories, min_per_category=min_w)


def repair_integer_weights(
    raw: dict[str, float],
    categories: list[str] | None = None,
    min_per_category: int = 3,
) -> dict[str, int]:
    """Largest-remainder method to integers summing to 100."""
    categories = categories or CATEGORY_ORDER
    floor_total = min_per_category * len(categories)
    if floor_total >= 100:
        raise ValueError("min_per_category too high for 100-point scale")

    remaining = 100 - floor_total
    probs = np.array([max(raw.get(c, 0), 0) for c in categories], dtype=float)
    if probs.sum() <= 0:
        probs = np.ones(len(categories))
    probs = probs / probs.sum()

    floors = {c: min_per_category + int(np.floor(remaining * p)) for c, p in zip(categories, probs)}
    remainder = 100 - sum(floors.values())
    fractional = sorted(
        ((raw.get(c, 0) - floors[c], c) for c in categories),
        reverse=True,
    )
    for i in range(max(0, remainder)):
        floors[fractional[i % len(categories)][1]] += 1
    return floors


def weights_to_vector(weights: dict[str, int], categories: list[str] | None = None) -> np.ndarray:
    categories = categories or CATEGORY_ORDER
    return np.array([float(weights.get(c, 0)) for c in categories], dtype=float)
