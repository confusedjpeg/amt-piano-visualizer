"""
Structured logging configuration using loguru.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting pipeline", run_id="abc123")
"""

from __future__ import annotations

import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Configure the global loguru logger.

    Removes any existing handlers and installs a new stderr handler
    with a structured format.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


def get_logger(name: str = __name__) -> logger.__class__:
    """Return a logger bound with the given module name.

    Args:
        name: Module name (typically ``__name__``).

    Returns:
        A loguru logger instance with the module name bound as context.
    """
    return logger.bind(module=name)


# Apply default configuration on import
setup_logging()
