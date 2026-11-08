"""Structured logging setup (no print statements in the pipeline)."""

from __future__ import annotations

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a key=value style format, level from LOG_LEVEL (default INFO)."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
        )
    return logging.getLogger(name)
