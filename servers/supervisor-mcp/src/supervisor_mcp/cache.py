"""Redis cache helpers for Supervisor MCP."""

from __future__ import annotations

from typing import Optional

from redis.asyncio import Redis

from .config import Settings, get_settings

_redis: Optional[Redis] = None


def get_redis(settings: Optional[Settings] = None) -> Redis:
    """Return global Redis client (lazy-initialized)."""
    global _redis
    if _redis is None:
        settings = settings or get_settings()
        _redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def close_redis() -> None:
    """Close Redis connection pool."""
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None

