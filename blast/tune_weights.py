#!/usr/bin/env python3
"""CLI to tune blast factor weights per cap segment.

Runs hybrid EA (seeds + differential evolution + Nelder-Mead), selects the
candidate with best validation Precision@10, and writes ``data/blast/weights_*.yaml``.
Use ``--run-label`` to archive logs and weights under ``data/blast/runs/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from blast._config import get_weights
from blast.enrich import load_segment_frame
from blast.optimize.evaluate import evaluate_weights, format_report
from blast.optimize.fitness import FitnessContext
from blast.optimize.hybrid_ea import tune_segment

SEGMENTS = ("large_cap", "mid_cap", "small_cap")
OUTPUT_DIR = os.path.join(ROOT, "data", "blast")
LOG = logging.getLogger("blast.tune")


def setup_logging(log_path: str | None = None) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if log_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(OUTPUT_DIR, f"tune_{stamp}.log")

    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    LOG.addHandler(file_handler)
    LOG.addHandler(stream_handler)
    LOG.info("Log file: %s", log_path)
    return log_path


def run_baseline(segment: str) -> None:
    """Log train/val/test metrics for blueprint default weights."""
    df = load_segment_frame(segment)
    weights = get_weights()
    contexts = {
        "train": FitnessContext.from_frame(df, segment, "train"),
        "validation": FitnessContext.from_frame(df, segment, "validation"),
        "test": FitnessContext.from_frame(df, segment, "test"),
    }
    report = evaluate_weights(contexts, weights)
    report["segment"] = segment
    report["mode"] = "baseline"
    LOG.info("\n%s", format_report(report))


def run_tune(segment: str, quick: bool) -> dict:
    """Tune one segment and return the evaluation report for the winner."""
    LOG.info("=" * 60)
    LOG.info("Loading segment frame: %s", segment)
    t_load = time.time()
    df = load_segment_frame(segment)
    LOG.info("Loaded %d rows in %.1fs", len(df), time.time() - t_load)

    t_ctx = time.time()
    train_ctx = FitnessContext.from_frame(df, segment, "train")
    val_ctx = FitnessContext.from_frame(df, segment, "validation")
    test_ctx = FitnessContext.from_frame(df, segment, "test")
    LOG.info("Built fitness contexts in %.1fs", time.time() - t_ctx)

    t_tune = time.time()
    report = tune_segment(train_ctx, val_ctx, test_ctx, quick=quick)
    LOG.info("Tuning finished in %.1fs", time.time() - t_tune)
    LOG.info("Selected: %s", report["selected"])
    LOG.info("\n%s", format_report(report))
    return report


def save_report(segment: str, report: dict) -> str:
    """Write active weights YAML for a segment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"weights_{segment}.yaml")
    doc = {
        "segment": segment,
        "selected": report.get("selected"),
        "weights": report["weights"],
        "metrics": report.get("splits", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune blast factor weights")
    parser.add_argument(
        "--segment",
        choices=list(SEGMENTS),
        action="append",
        help="Segment to tune (repeatable; default: all)",
    )
    parser.add_argument("--baseline-only", action="store_true", help="Report default weights only")
    parser.add_argument("--quick", action="store_true", help="Fast DE settings for smoke runs")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log path (default: data/blast/tune_YYYYMMDD_HHMMSS.log)",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Archive outputs to data/blast/runs/<label>/ when complete (e.g. run_02)",
    )
    args = parser.parse_args()

    log_path = setup_logging(args.log_file)
    segments = args.segment or list(SEGMENTS)
    mode = "quick" if args.quick else "full"
    LOG.info("Blast weight tuning  mode=%s  segments=%s", mode, ", ".join(segments))

    t0 = time.time()

    if args.baseline_only:
        for seg in segments:
            run_baseline(seg)
        return

    run_label = args.run_label
    if run_label:
        LOG.info("Results will be archived to data/blast/runs/%s/", run_label)

    for i, seg in enumerate(segments, start=1):
        LOG.info("Segment %d/%d: %s", i, len(segments), seg)
        report = run_tune(seg, quick=args.quick)
        path = save_report(seg, report)
        LOG.info("Saved %s", path)

    if run_label:
        archive_dir = os.path.join(OUTPUT_DIR, "runs", run_label)
        os.makedirs(archive_dir, exist_ok=True)
        for seg in segments:
            src = os.path.join(OUTPUT_DIR, f"weights_{seg}.yaml")
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(archive_dir, f"weights_{seg}.yaml"))
        shutil.copy2(log_path, os.path.join(archive_dir, os.path.basename(log_path)))
        LOG.info("Archived run to %s", archive_dir)

    elapsed = (time.time() - t0) / 60
    LOG.info("Done in %.1f min  log=%s", elapsed, log_path)


if __name__ == "__main__":
    main()
