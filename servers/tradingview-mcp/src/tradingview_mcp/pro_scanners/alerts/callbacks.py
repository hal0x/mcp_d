
"""Alert router callbacks and HALv1 integration stubs."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from ..models import AlertRouteResult


logger = logging.getLogger(__name__)


async def log_dispatch_result(result: AlertRouteResult) -> None:
    """Default callback that logs routing outcomes.

    Intended to be replaced or wrapped by HALv1-specific handlers once
    the bot exposes an acknowledgement endpoint.
    """
    logger.info(
        "Alert dispatched: status=%s symbol=%s timeframe=%s details=%s",
        result.status,
        result.payload.symbol,
        result.payload.timeframe,
        result.details,
    )


CallbackType = Callable[[AlertRouteResult], Awaitable[None] | None]


__all__ = ["log_dispatch_result", "CallbackType"]
