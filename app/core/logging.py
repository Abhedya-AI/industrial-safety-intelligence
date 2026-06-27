"""Unified logging configuration."""

import logging

from app.core.settings import get_settings

settings = get_settings()


def setup_logging() -> None:
    """Initialize logging settings."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
