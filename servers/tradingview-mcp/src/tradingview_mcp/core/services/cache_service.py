"""Shared caching infrastructure for TradingView MCP tools."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - redis is optional at runtime
    redis = None  # type: ignore[assignment]

from tradingview_mcp.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    errors: int = 0

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return round(self.hits / total, 4)


class CacheService:
    """Redis-backed cache with in-memory fallback and metric tracking."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._redis_client: Optional["redis.Redis[str]"] = None
        self._redis_error_logged = False
        self._redis_lock = threading.Lock()
        self._memory_cache: Dict[str, tuple[float, Any]] = {}
        self._memory_lock = threading.Lock()
        self._stats = CacheStats()

    def _redis_enabled(self) -> bool:
        return bool(self._settings.redis_url and redis is not None and self._settings.redis_enabled)

    def _get_redis(self) -> Optional["redis.Redis[str]"]:
        if not self._redis_enabled():
            return None
        if self._redis_client is not None:
            return self._redis_client
        with self._redis_lock:
            if self._redis_client is None:
                try:
                    self._redis_client = redis.Redis.from_url(
                        self._settings.redis_url,
                        decode_responses=True,
                        socket_timeout=self._settings.redis_socket_timeout,
                    )
                    # probe connection once
                    self._redis_client.ping()
                    logger.info("Connected to Redis cache at %s", self._settings.redis_url)
                except Exception as exc:  # pragma: no cover - best effort
                    if not self._redis_error_logged:
                        logger.warning("Redis unavailable, falling back to in-memory cache: %s", exc)
                        self._redis_error_logged = True
                    self._redis_client = None
        return self._redis_client

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if absent."""
        try:
            client = self._get_redis()
            if client is not None:
                value = client.get(key)
                if value is not None:
                    self._stats.hits += 1
                    return json.loads(value)
        except Exception as exc:  # pragma: no cover - best effort
            self._stats.errors += 1
            logger.debug("Redis get failed for %s: %s", key, exc)

        with self._memory_lock:
            record = self._memory_cache.get(key)
            if record:
                expires_at, payload = record
                if expires_at > time.monotonic():
                    self._stats.hits += 1
                    return payload
                self._memory_cache.pop(key, None)
        self._stats.misses += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store value in cache with TTL."""
        serialized: str
        try:
            serialized = json.dumps(value, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to serialize cache entry %s: %s", key, exc)
            return

        expires_at = time.monotonic() + max(ttl_seconds, 1)

        try:
            client = self._get_redis()
            if client is not None:
                client.setex(key, ttl_seconds, serialized)
                return
        except Exception as exc:  # pragma: no cover - best effort
            self._stats.errors += 1
            logger.debug("Redis set failed for %s: %s", key, exc)

        with self._memory_lock:
            self._memory_cache[key] = (expires_at, value)

    def purge_expired(self) -> None:
        """Remove stale records from in-memory cache."""
        deadline = time.monotonic()
        with self._memory_lock:
            expired = [key for key, (expires_at, _) in self._memory_cache.items() if expires_at <= deadline]
            for key in expired:
                self._memory_cache.pop(key, None)
        if expired:
            logger.debug("Purged %d expired cache records", len(expired))

    def stats(self) -> CacheStats:
        """Return accumulated cache statistics."""
        return self._stats


_CACHE_SERVICE: Optional[CacheService] = None
_CACHE_LOCK = threading.Lock()


def get_cache_service() -> CacheService:
    """Return global cache service instance."""
    global _CACHE_SERVICE
    if _CACHE_SERVICE is None:
        with _CACHE_LOCK:
            if _CACHE_SERVICE is None:
                _CACHE_SERVICE = CacheService()
    return _CACHE_SERVICE


__all__ = ["CacheService", "CacheStats", "get_cache_service"]
