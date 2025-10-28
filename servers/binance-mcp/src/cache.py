"""Redis cache helpers for Binance MCP."""

from __future__ import annotations

import json
import structlog
from typing import Any

from redis import asyncio as redis

from .config import get_config

logger = structlog.get_logger(__name__)

_redis_client: redis.Redis | None = None


async def init_redis() -> None:
    """Ensures Redis connection is initialised for startup."""
    client = await _get_client()
    if client is not None:
        try:
            await client.ping()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to ping Redis during startup")


async def close_redis() -> None:
    """Close Redis connection on shutdown."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.close()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to close Redis connection")
        finally:
            _redis_client = None


async def get_cached_json(key: str) -> Any | None:
    client = await _get_client()
    if client is None:
        return None
    try:
        data = await client.get(key)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to get Redis key %s", key)
        return None
    if not data:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON stored in Redis for key %s", key)
        return None


async def set_cached_json(key: str, payload: Any, ttl: int | None = None) -> None:
    client = await _get_client()
    if client is None:
        return
    config = get_config()
    expiration = ttl or config.redis_cache_ttl
    try:
        await client.set(key, json.dumps(payload, default=str), ex=expiration)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to set Redis key %s", key)


async def _get_client() -> redis.Redis | None:
    global _redis_client
    config = get_config()
    if not config.has_redis:
        return None
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(
                config.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to create Redis client")
            _redis_client = None
    return _redis_client


__all__ = ["init_redis", "close_redis", "get_cached_json", "set_cached_json"]
