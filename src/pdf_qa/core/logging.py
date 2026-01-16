"""Logging configuration with Rich handler support."""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: str = "INFO", console: Optional[Console] = None) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Optional Rich console for output. Creates new one if not provided.

    Returns:
        Configured logger instance for the pdf_qa package.
    """
    if console is None:
        console = Console(stderr=True, force_terminal=True)

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure Rich handler with proper formatting
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_level=True,
    )
    rich_handler.setLevel(numeric_level)

    # Ensure logs start on new line
    formatter = logging.Formatter("\n%(message)s\n" if numeric_level <= logging.DEBUG else "%(message)s\n")
    rich_handler.setFormatter(formatter)

    # Get the pdf_qa logger
    logger = logging.getLogger("pdf_qa")
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(rich_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "pdf_qa") -> logging.Logger:
    """Get a logger instance for the given name.

    Args:
        name: Logger name, typically module name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# Module-level logger for convenience
logger = get_logger()
