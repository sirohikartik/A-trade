"""Hybrid EA: seeds + DE + Nelder-Mead polish, val-selected winner."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

from blast._config import load_blast_config

logger = logging.getLogger("blast.tune")
from blast.optimize.evaluate import evaluate_weights
from blast.optimize.fitness import FitnessContext
from blast.optimize.lift_seeds import generate_seeds
from blast.optimize.simplex import CATEGORY_ORDER, softmax_weights, weights_to_vector


def _theta_from_weights(weights: dict[str, int]) -> np.ndarray:
    w = weights_to_vector(weights) + 1e-6
    return np.log(w / w.sum())


def _de_objective(theta: np.ndarray, ctx: FitnessContext) -> float:
    weights = softmax_weights(theta)
    return -float(ctx.precision_at_k(weights))


def _polish(theta0: np.ndarray, ctx: FitnessContext, maxiter: int) -> np.ndarray:
    bounds = [(-4.0, 4.0)] * len(CATEGORY_ORDER)

    def obj(theta: np.ndarray) -> float:
        return _de_objective(theta, ctx)

    res = minimize(
        obj,
        theta0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4},
    )
    return res.x if res.success else theta0


class _DEProgress:
    def __init__(self, restart: int, train_ctx: FitnessContext, maxiter: int, log_every: int = 10) -> None:
        self.restart = restart
        self.train_ctx = train_ctx
        self.maxiter = maxiter
        self.log_every = log_every
        self.iteration = 0

    def __call__(self, xk: np.ndarray, convergence: float = 0.0) -> bool:
        self.iteration += 1
        if self.iteration == 1 or self.iteration % self.log_every == 0 or self.iteration == self.maxiter:
            weights = softmax_weights(xk)
            train_p10 = float(self.train_ctx.precision_at_k(weights, k=10))
            logger.info(
                "DE restart %d  iter %d/%d  conv=%.4f  train_P@10=%.3f  weights=%s",
                self.restart,
                self.iteration,
                self.maxiter,
                float(convergence),
                train_p10,
                weights,
            )
        return False


def _log_weights(label: str, weights: dict[str, int], train_ctx: FitnessContext, val_ctx: FitnessContext) -> None:
    train_p10 = float(train_ctx.precision_at_k(weights, k=10))
    val_p10 = float(val_ctx.precision_at_k(weights, k=10))
    logger.info("%s  train_P@10=%.3f  val_P@10=%.3f  weights=%s", label, train_p10, val_p10, weights)


def tune_segment(
    train_ctx: FitnessContext,
    val_ctx: FitnessContext,
    test_ctx: FitnessContext | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Search integer weights via seeds, differential evolution, and polish; pick best val P@10."""
    cfg = load_blast_config().get("optimizer", {})
    if quick:
        popsize = int(cfg.get("quick_de_popsize", 15))
        maxiter = int(cfg.get("quick_de_maxiter", 25))
        n_restarts = int(cfg.get("quick_n_restarts", 1))
        polish_iter = int(cfg.get("quick_polish_maxiter", 30))
    else:
        popsize = int(cfg.get("de_popsize", 60))
        maxiter = int(cfg.get("de_maxiter", 150))
        n_restarts = int(cfg.get("n_restarts", 3))
        polish_iter = int(cfg.get("polish_maxiter", 50))

    bounds = [(-4.0, 4.0)] * len(CATEGORY_ORDER)
    candidates: list[tuple[str, dict[str, int]]] = []

    logger.info(
        "Starting %s  quick=%s  popsize=%d  maxiter=%d  restarts=%d  polish=%d  train_days=%d  val_days=%d",
        train_ctx.segment,
        quick,
        popsize,
        maxiter,
        n_restarts,
        polish_iter,
        len(train_ctx.day_pools),
        len(val_ctx.day_pools),
    )

    seeds = generate_seeds(train_ctx)
    logger.info("Evaluating %d seed vectors", len(seeds))
    for i, seed_w in enumerate(seeds):
        _log_weights(f"seed_{i}", seed_w, train_ctx, val_ctx)
        candidates.append((f"seed_{i}", seed_w))

    log_every = int(cfg.get("de_log_every", 10))
    rng = np.random.default_rng(42)
    for restart in range(n_restarts):
        seed = int(rng.integers(0, 1_000_000))
        logger.info("DE restart %d/%d starting (rng_seed=%d)", restart + 1, n_restarts, seed)
        progress = _DEProgress(restart, train_ctx, maxiter, log_every=log_every)
        result = differential_evolution(
            _de_objective,
            bounds,
            args=(train_ctx,),
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-3,
            polish=False,
            updating="immediate",
            workers=1,
            callback=progress,
        )
        theta = result.x
        logger.info(
            "DE restart %d finished  nit=%d  success=%s  train_fitness=%.4f",
            restart,
            result.nit,
            result.success,
            -float(result.fun),
        )
        if polish_iter > 0:
            logger.info("DE restart %d polishing (%d iter max)", restart, polish_iter)
            theta = _polish(theta, train_ctx, polish_iter)
        de_weights = softmax_weights(theta)
        _log_weights(f"de_{restart}", de_weights, train_ctx, val_ctx)
        candidates.append((f"de_{restart}", de_weights))

    best_val = -1.0
    best_weights: dict[str, int] | None = None
    best_tag = ""
    candidate_rows: list[dict[str, Any]] = []

    for tag, weights in candidates:
        val_details = val_ctx.precision_at_k(weights, k=10, return_details=True)
        assert isinstance(val_details, dict)
        val_p10 = float(val_details["precision_at_k"])
        train_details = train_ctx.precision_at_k(weights, k=10, return_details=True)
        assert isinstance(train_details, dict)
        candidate_rows.append(
            {
                "tag": tag,
                "val_p10": round(val_p10, 4),
                "train_p10": round(float(train_details["precision_at_k"]), 4),
                "weights": weights,
            }
        )
        if val_p10 > best_val:
            best_val = val_p10
            best_weights = weights
            best_tag = tag

    logger.info("Candidate leaderboard (val P@10):")
    for row in sorted(candidate_rows, key=lambda r: r["val_p10"], reverse=True):
        logger.info(
            "  %-8s  val=%.3f  train=%.3f  %s",
            row["tag"],
            row["val_p10"],
            row["train_p10"],
            row["weights"],
        )
    logger.info("Selected winner: %s  val_P@10=%.3f", best_tag, best_val)

    if best_weights is None:
        best_weights = candidates[0][1]
        best_tag = candidates[0][0]

    contexts = {"train": train_ctx, "validation": val_ctx}
    if test_ctx is not None:
        contexts["test"] = test_ctx

    report = evaluate_weights(contexts, best_weights)
    report["segment"] = train_ctx.segment
    report["selected"] = best_tag
    report["candidates"] = candidate_rows
    return report
