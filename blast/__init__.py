"""Blast scoring engine — seven-factor weighted score for swing setups.

See ``blast.scoring`` for core scoring and ``blast.tune_weights`` for EA tuning.
"""

from blast.scoring import compute_blast_score, rank_eligible_day

__all__ = ["compute_blast_score", "rank_eligible_day"]
