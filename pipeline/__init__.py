"""A-trade research pipeline — fetch, process, split, universe selection.

Modules:
    fetch    — bulk Yahoo OHLCV download into DuckDB
    process  — indicators, delivery merge, reference labels, segment parquets
    splits   — train / validation / test date ranges → data/splits.yaml
    universe — train-period edge ranking for optional top-N subset
"""
