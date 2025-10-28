"""Redis cache helpers for pro scanners."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, List, Sequence

from redis import asyncio as redis

from ..config import RedisConfig
from ..models import DerivativeSnapshot, ScannerSignal


logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, config: RedisConfig):
        self._config = config
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        if self._client is not None:
            return

        hosts: list[str] = [self._config.host]
        fallback_host = os.getenv("TRADINGVIEW_REDIS_FALLBACK_HOST") or "127.0.0.1"
        if self._config.host in {"redis", "localhost"} and fallback_host not in hosts:
            hosts.append(fallback_host)

        last_exc: Exception | None = None
        for host in hosts:
            try:
                url = f"redis://{host}:{self._config.port}/{self._config.db}"
                client = redis.from_url(url, decode_responses=True)
                await client.ping()
                self._client = client
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis connect attempt failed for %s:%s (%s)", host, self._config.port, exc)
                last_exc = exc

        if last_exc:
            raise last_exc

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def get_cached_signals(self, key: str) -> list[ScannerSignal] | None:
        raw = await self._get(key)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            return [ScannerSignal.model_validate(item) for item in payload]
        except Exception:  # noqa: BLE001
            logger.exception("Failed to decode cached signals for key %s", key)
            return None

    async def set_cached_signals(self, key: str, signals: Sequence[ScannerSignal], ttl: int) -> None:
        payload = json.dumps([signal.model_dump(mode="json") for signal in signals])
        await self._set(key, payload, ttl)

    async def get_derivatives_snapshot(self, key: str) -> DerivativeSnapshot | None:
        raw = await self._get(key)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            return DerivativeSnapshot.model_validate(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to decode derivatives snapshot for key %s", key)
            return None

    async def set_derivatives_snapshot(self, key: str, snapshot: DerivativeSnapshot) -> None:
        ttl = self._config.ttl.indicators
        payload = json.dumps(snapshot.model_dump(mode="json"))
        await self._set(key, payload, ttl)

    async def get_json(self, key: str) -> Any | None:
        raw = await self._get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to decode JSON for key %s", key)
            return None

    async def set_json(self, key: str, payload: Any, ttl: int | None = None) -> None:
        ttl_value = ttl if ttl is not None else self._config.ttl.indicators
        serialized = json.dumps(payload)
        await self._set(key, serialized, ttl_value)

    async def clear_namespace(self, namespace: str | None = None) -> int:
        """Remove cached entries optionally scoped by namespace."""
        try:
            client = self._require_client()
        except RuntimeError:
            logger.warning("Redis not available for cache clearing")
            return 0

        pattern = f"{namespace}:*" if namespace else "*"
        cursor = "0"
        removed = 0
        max_iterations = 1000  # Prevent infinite loops
        iteration = 0
        
        try:
            while iteration < max_iterations:
                cursor, keys = await client.scan(cursor=cursor, match=pattern, count=200)
                if keys:
                    await client.delete(*keys)
                    removed += len(keys)
                if cursor == "0":
                    break
                iteration += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error during cache clearing: %s", exc)
            # Return partial results if we encountered an error
        finally:
            if iteration >= max_iterations:
                logger.warning("Cache clearing stopped after %d iterations to prevent infinite loop", max_iterations)
        
        return removed

    async def _get(self, key: str) -> str | None:
        try:
            client = self._require_client()
        except RuntimeError:
            return None
        return await client.get(key)

    async def _set(self, key: str, value: str, ttl: int) -> None:
        try:
            client = self._require_client()
        except RuntimeError:
            return
        await client.set(key, value, ex=ttl)

    def _require_client(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("RedisCache not connected")
        return self._client

    @property
    def indicator_ttl(self) -> int:
        return self._config.ttl.indicators

    @property
    def alert_ttl(self) -> int:
        return self._config.ttl.alerts

    @property
    def session_ttl(self) -> int:
        return self._config.ttl.sessions


__all__ = ["RedisCache"]
