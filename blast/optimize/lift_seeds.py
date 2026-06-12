"""Seed weight vectors for EA — blueprint, lift-proportional, random."""

from __future__ import annotations

import numpy as np

from blast._config import get_weights, load_standard_weights
from blast.optimize.fitness import FitnessContext
from blast.optimize.simplex import CATEGORY_ORDER, repair_integer_weights, softmax_weights


def blueprint_seed() -> dict[str, int]:
    return repair_integer_weights({c: float(get_weights().get(c, 0)) for c in CATEGORY_ORDER})


def standard_seed(ctx: FitnessContext) -> dict[str, int]:
    weights = load_standard_weights(ctx.segment)
    if sum(weights.get(c, 0) for c in CATEGORY_ORDER) == 100:
        return {c: int(weights[c]) for c in CATEGORY_ORDER}
    return repair_integer_weights(
        {c: float(weights.get(c, 0)) for c in CATEGORY_ORDER}
    )


def lift_proportional_seed(ctx: FitnessContext) -> dict[str, int]:
    """Allocate weights by category lift over base rate on train pools."""
    if not ctx.day_pools:
        return blueprint_seed()

    lifts = []
    for cat_idx, cat in enumerate(CATEGORY_ORDER):
        wins_high = 0.0
        n_high = 0
        for pool in ctx.day_pools:
            col = pool.subscores[:, cat_idx]
            if len(col) == 0:
                continue
            threshold = np.median(col)
            mask = col >= threshold
            if mask.sum() == 0:
                continue
            wins_high += pool.outcomes[mask].sum()
            n_high += mask.sum()
        rate = wins_high / n_high if n_high else ctx.base_rate
        lifts.append(max(rate - ctx.base_rate, 0.001))

    total = sum(lifts)
    raw = {c: 100.0 * lift / total for c, lift in zip(CATEGORY_ORDER, lifts)}
    return repair_integer_weights(raw)


def random_seed(rng: np.random.Generator) -> dict[str, int]:
    theta = rng.normal(size=len(CATEGORY_ORDER))
    return softmax_weights(theta)


def generate_seeds(ctx: FitnessContext, n_random: int = 5, rng: np.random.Generator | None = None) -> list[dict[str, int]]:
    """Initial weight vectors: standard, blueprint, lift-proportional, plus random."""
    rng = rng or np.random.default_rng()
    seeds = [standard_seed(ctx), blueprint_seed(), lift_proportional_seed(ctx)]
    seen = {tuple(sorted(s.items())) for s in seeds}
    while len(seeds) < 3 + n_random:
        s = random_seed(rng)
        key = tuple(sorted(s.items()))
        if key not in seen:
            seen.add(key)
            seeds.append(s)
    return seeds
