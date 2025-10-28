
"""PostgreSQL storage helpers."""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from ..config import PostgresConfig
from ..models import BacktestResult, ScannerSignal


logger = logging.getLogger(__name__)


class PostgresStore:
    def __init__(self, config: PostgresConfig):
        self._config = config
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is None:
            kwargs: dict[str, Any] = {
                "host": self._config.host,
                "port": self._config.port,
                "database": self._config.database,
                "min_size": 1,
                "max_size": self._config.pool_size,
            }
            if self._config.user:
                kwargs["user"] = self._config.user
            if self._config.password:
                kwargs["password"] = self._config.password
            self._pool = await asyncpg.create_pool(**kwargs)

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def record_signal(self, signal: ScannerSignal) -> None:
        query = """
            INSERT INTO signals (
                timestamp,
                symbol,
                timeframe,
                direction,
                entry,
                sl,
                tp,
                leverage,
                confidence,
                reasons
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10::jsonb
            )
        """
        values = (
            signal.generated_at,
            signal.symbol,
            signal.timeframe,
            signal.direction.value,
            signal.entry,
            signal.risk.stop_loss,
            json.dumps(list(signal.risk.take_profit)),
            signal.risk.leverage,
            signal.confidence,
            json.dumps(list(signal.reasons)),
        )
        await self._execute(query, *values)

    async def record_backtest(self, result: BacktestResult) -> None:
        query = """
            INSERT INTO backtests (
                strategy,
                profile,
                timeframe,
                symbol_universe,
                metrics,
                generated_signals,
                started_at,
                completed_at
            )
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7, $8)
        """
        values = (
            result.strategy,
            result.profile,
            result.timeframe,
            json.dumps(list(result.symbol_universe)),
            json.dumps(dict(result.metrics)),
            json.dumps([signal.model_dump(mode="json") for signal in result.signals]),
            result.started_at,
            result.completed_at,
        )
        await self._execute(query, *values)

    async def record_metrics_snapshot(self, result: BacktestResult) -> None:
        if not result.metrics:
            return
        metrics = result.metrics
        query = """
            INSERT INTO metrics (
                period,
                strategy,
                total_signals,
                take_rate,
                avg_confidence,
                avg_leverage,
                calculated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
        """
        values = (
            result.timeframe,
            result.strategy,
            int(metrics.get("total_signals", 0)),
            float(metrics.get("take_rate", 0.0)),
            float(metrics.get("avg_confidence", 0.0)),
            float(metrics.get("avg_leverage", 0.0)),
        )
        await self._execute(query, *values)

    async def record_feedback(self, signal_id: int, action: str) -> None:
        query = """
            INSERT INTO feedback (signal_id, action, timestamp)
            VALUES ($1, $2, NOW())
        """
        await self._execute(query, signal_id, action)

    async def fetch_recent_signals(self, limit: int = 20) -> list[dict[str, Any]]:
        query = """
            SELECT
                id,
                timestamp,
                symbol,
                timeframe,
                direction,
                entry,
                sl,
                tp,
                leverage,
                confidence,
                reasons
            FROM signals
            ORDER BY timestamp DESC
            LIMIT $1
        """
        return await self._fetch(query, limit)

    async def fetch_metrics(self, period_days: int = 14, strategy: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT
                strategy,
                period,
                total_signals,
                take_rate,
                avg_confidence,
                avg_leverage,
                calculated_at
            FROM metrics
            WHERE calculated_at >= NOW() - ($1::int * INTERVAL '1 day')
        """
        params: list[Any] = [period_days]
        if strategy:
            query += " AND strategy = $2"
            params.append(strategy)
        query += " ORDER BY calculated_at DESC LIMIT 100"
        return await self._fetch(query, *params)

    async def _execute(self, query: str, *args) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(query, *args)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to execute query: %s", exc)
                # Check for specific database errors
                if "relation" in str(exc) and "does not exist" in str(exc):
                    logger.error("Database table missing. Please run the database initialization script.")
                raise

    async def _fetch(self, query: str, *args) -> list[dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to fetch query results: %s", exc)
                # Check for specific database errors
                if "relation" in str(exc) and "does not exist" in str(exc):
                    logger.error("Database table missing. Please run the database initialization script.")
                raise

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresStore not connected")
        return self._pool


__all__ = ["PostgresStore"]
