"""Reusable logging configuration for the application."""

from __future__ import annotations

import logging
import sys

from src.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with a consistent console formatter."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        ),
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
