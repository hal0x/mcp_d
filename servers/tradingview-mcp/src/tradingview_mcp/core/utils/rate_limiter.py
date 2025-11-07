"""Token bucket rate limiter with jitter for external API calls."""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    waited_seconds: float
    tokens_consumed: float


class TokenBucketRateLimiter:
    """Thread-safe token bucket implementation used for TradingView calls."""

    def __init__(self, requests_per_minute: int, burst: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if burst <= 0:
            raise ValueError("burst must be positive")

        self._capacity = float(burst)
        self._tokens = float(burst)
        self._rate_per_second = float(requests_per_minute) / 60.0
        self._updated_at = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated_at)
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_second)
        self._updated_at = now

    def acquire(self, tokens: float = 1.0, jitter: float = 0.2) -> RateLimitResult:
        """Acquire tokens, blocking if necessary. Returns how long we waited."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")

        with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._updated_at = now
                return RateLimitResult(waited_seconds=0.0, tokens_consumed=tokens)

            deficit = tokens - self._tokens
            wait_seconds = deficit / self._rate_per_second
            self._tokens = 0.0
            self._updated_at = now + wait_seconds

        jitter = max(0.0, min(1.0, jitter))
        jitter_delay = random.uniform(0, wait_seconds * jitter) if wait_seconds > 0 else 0.0
        total_sleep = wait_seconds + jitter_delay
        if total_sleep > 0:
            logger.debug("Rate limiter sleeping for %.2fs (wait %.2fs + jitter %.2fs)", total_sleep, wait_seconds, jitter_delay)
            time.sleep(total_sleep)
        return RateLimitResult(waited_seconds=total_sleep, tokens_consumed=tokens)
