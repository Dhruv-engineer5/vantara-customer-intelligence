"""Single-command pipeline: raw data -> cleaned -> features -> models -> segments -> explanations -> report.

Usage: python -m src.pipeline [--skip-report]
"""

from __future__ import annotations

import sys

from src.data import clean
from src.explainability import explain
from src.features import build
from src.models import train
from src.recommend import recommender
from src.report import diagrams, final_report
from src.segmentation import cluster
from src.utils.config import load_config, rpath
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


def main(skip_report: bool = False) -> None:
    """Run every stage in order; each stage logs its own summary."""
    cfg = load_config()
    clean.run(cfg)
    build.run(cfg)
    train.run(cfg)
    cluster.run(cfg)
    recommender.run(cfg)
    explain.run(cfg)
    rpath(cfg, "invoices").unlink(missing_ok=True)  # rebuilt from the fresh cleaned data on next seed
    from src.db.session import build_invoices

    build_invoices(cfg)
    if not skip_report:
        diagrams.run()
        final_report.run(cfg)
    log.info("pipeline complete")


if __name__ == "__main__":
    main("--skip-report" in sys.argv)
