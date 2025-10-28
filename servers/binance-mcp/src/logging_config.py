"""Structlog-based logging configuration."""

from __future__ import annotations

import logging
from typing import Final

import structlog

_CONFIGURED: Final[dict[str, bool]] = {"value": False}


def configure_logging() -> None:
    """Configure structlog for structured JSON output."""
    if _CONFIGURED["value"]:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    _CONFIGURED["value"] = True


__all__ = ["configure_logging"]
