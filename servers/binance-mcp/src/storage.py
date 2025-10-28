"""PostgreSQL storage helpers for Binance MCP."""

from __future__ import annotations

from typing import Any, Optional

import asyncpg
import structlog

from .config import get_config

logger = structlog.get_logger(__name__)


class PostgresStorage:
    """Async PostgreSQL storage for risk metrics and limits."""

    def __init__(self) -> None:
        config = get_config()
        self._config = config
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        if not self._config.has_postgres:
            return
        kwargs: dict[str, Any] = {
            "host": self._config.postgres_host,
            "port": self._config.postgres_port,
            "database": self._config.postgres_database,
            "min_size": 1,
            "max_size": 5,
        }
        if self._config.postgres_user:
            kwargs["user"] = self._config.postgres_user
        if self._config.postgres_password:
            kwargs["password"] = self._config.postgres_password
        self._pool = await asyncpg.create_pool(**kwargs)
        await self._ensure_schema()
        logger.info("postgres_connected")

    async def disconnect(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None
        logger.info("postgres_disconnected")

    async def _ensure_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_metrics (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    total_equity NUMERIC,
                    available_balance NUMERIC,
                    unrealized_pnl NUMERIC,
                    margin_ratio NUMERIC,
                    positions_count INT,
                    largest_position TEXT
                );
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS position_limits (
                    symbol TEXT PRIMARY KEY,
                    max_size NUMERIC NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

    async def record_portfolio_metrics(self, metrics: dict[str, Any]) -> None:
        if self._pool is None:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO portfolio_metrics (
                    total_equity,
                    available_balance,
                    unrealized_pnl,
                    margin_ratio,
                    positions_count,
                    largest_position
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                metrics.get("total_equity"),
                metrics.get("available_balance"),
                metrics.get("unrealized_pnl"),
                metrics.get("margin_ratio"),
                metrics.get("positions_count"),
                metrics.get("largest_position"),
            )

    async def upsert_position_limit(
        self, symbol: str, max_size: float, updated_at: int
    ) -> None:
        if self._pool is None:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO position_limits(symbol, max_size, updated_at)
                VALUES ($1, $2, to_timestamp($3))
                ON CONFLICT(symbol)
                DO UPDATE SET max_size = EXCLUDED.max_size, updated_at = EXCLUDED.updated_at
                """,
                symbol,
                max_size,
                updated_at,
            )

    async def fetch_position_limits(self) -> list[dict[str, Any]]:
        if self._pool is None:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT symbol, max_size, EXTRACT(EPOCH FROM updated_at) AS updated_at FROM position_limits"
            )
        return [dict(row) for row in rows]


_storage: PostgresStorage | None = None


async def init_postgres() -> None:
    global _storage
    config = get_config()
    if not config.has_postgres:
        return
    if _storage is None:
        _storage = PostgresStorage()
    await _storage.connect()


def get_postgres_storage() -> PostgresStorage | None:
    return _storage


async def close_postgres() -> None:
    global _storage
    if _storage is None:
        return
    await _storage.disconnect()
    _storage = None


__all__ = ["init_postgres", "close_postgres", "get_postgres_storage", "PostgresStorage"]
