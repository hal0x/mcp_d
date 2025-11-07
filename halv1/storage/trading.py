"""PostgreSQL storage for trading alerts and feedback."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import asyncpg
import structlog

logger = structlog.get_logger(__name__)

_POOL: asyncpg.Pool | None = None


def _dsn() -> str:
    return os.getenv("TRADING_STORAGE_DSN") or os.getenv("HAL_TRADING_STORAGE_DSN", "postgresql://mcp:mcp@localhost:5432/mcp")


async def init_trading_storage() -> None:
    """Initialise asyncpg pool and ensure schema exists."""
    global _POOL
    if _POOL is not None:
        return
    dsn = _dsn()
    try:
        _POOL = await asyncpg.create_pool(dsn)
        async with _POOL.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_alerts (
                    id SERIAL PRIMARY KEY,
                    external_id TEXT,
                    symbol TEXT,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_feedback (
                    id SERIAL PRIMARY KEY,
                    alert_id INTEGER REFERENCES trading_alerts(id) ON DELETE CASCADE,
                    action TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
        logger.info("trading_storage_initialised", dsn=dsn)
    except Exception:  # pragma: no cover - best effort init
        logger.exception("Failed to initialise trading storage", dsn=dsn)
        _POOL = None


async def close_trading_storage() -> None:
    global _POOL
    if _POOL is None:
        return
    await _POOL.close()
    _POOL = None
    logger.info("trading_storage_closed")


async def insert_trading_alert(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    if _POOL is None:
        return None
    async with _POOL.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO trading_alerts(external_id, symbol, payload)
            VALUES ($1, $2, $3)
            RETURNING id, created_at
            """,
            str(payload.get("id")) if payload.get("id") is not None else None,
            payload.get("symbol"),
            json.dumps(payload, ensure_ascii=False),
        )
    return dict(row) if row else None


async def insert_trading_feedback(alert_id: int, action: str) -> None:
    if _POOL is None:
        return
    async with _POOL.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO trading_feedback(alert_id, action)
            VALUES ($1, $2)
            """,
            alert_id,
            action,
        )


async def fetch_trading_alert(alert_id: int) -> Dict[str, Any] | None:
    if _POOL is None:
        return None
    async with _POOL.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, external_id, symbol, payload, created_at FROM trading_alerts WHERE id = $1",
            alert_id,
        )
    if not row:
        return None
    payload = json.loads(row["payload"]) if row["payload"] else {}
    payload["id"] = row["id"]
    payload.setdefault("external_id", row["external_id"])
    payload["created_at"] = row["created_at"].isoformat() if row["created_at"] else None
    return payload


__all__ = [
    "init_trading_storage",
    "close_trading_storage",
    "insert_trading_alert",
    "insert_trading_feedback",
    "fetch_trading_alert",
]
