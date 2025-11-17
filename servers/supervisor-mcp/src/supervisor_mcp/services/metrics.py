"""Metrics service with PostgreSQL persistence and Redis caching."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, AsyncContextManager

from redis.asyncio import Redis
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..pydantic_models import AggregationResult, Fact, Metric
from ..models.orm import AggregateORM, FactORM, MetricORM

SessionProvider = Callable[[], AsyncContextManager[AsyncSession]]


class MetricsService:
    """Service for collecting metrics and facts with persistent storage."""

    _CACHE_INDEX_KEY = "supervisor:agg:index"

    def __init__(
        self,
        session_provider: SessionProvider,
        redis_client: Optional[Redis] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._session_provider = session_provider
        self._redis = redis_client
        self._cache_ttl = cache_ttl_seconds or settings.redis_cache_ttl_seconds

    # ------------------------------------------------------------------ ingest
    async def ingest_metric(self, metric: Metric) -> None:
        """Persist a single metric."""
        await self.ingest_metrics([metric])

    async def ingest_metrics(self, metrics: List[Metric]) -> None:
        """Persist multiple metrics."""
        if not metrics:
            return

        records = [
            MetricORM(
                ts=metric.ts,
                name=metric.name,
                value=metric.value,
                tags=dict(metric.tags or {}),
            )
            for metric in metrics
        ]

        async with self._session_provider() as session:
            session.add_all(records)
            await session.commit()

        await self._invalidate_cache()

    async def ingest_fact(self, fact: Fact) -> None:
        """Persist a single fact."""
        await self.ingest_facts([fact])

    async def ingest_facts(self, facts: List[Fact]) -> None:
        """Persist multiple facts."""
        if not facts:
            return

        records = [
            FactORM(
                ts=fact.ts,
                kind=fact.kind,
                actor=fact.actor,
                correlation_id=fact.correlation_id,
                payload=dict(fact.payload or {}),
            )
            for fact in facts
        ]

        async with self._session_provider() as session:
            session.add_all(records)
            await session.commit()

        await self._invalidate_cache()

    # ------------------------------------------------------------------ queries
    async def query_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[Metric]:
        """Query metrics with optional filters."""
        async with self._session_provider() as session:
            stmt = select(MetricORM)
            if name:
                stmt = stmt.where(MetricORM.name == name)
            if start_time:
                stmt = stmt.where(MetricORM.ts >= start_time)
            if end_time:
                stmt = stmt.where(MetricORM.ts <= end_time)
            stmt = stmt.order_by(MetricORM.ts.asc())

            result = await session.execute(stmt)
            records = result.scalars().all()

        metrics = [self._metric_from_record(record) for record in records]
        if tags:
            metrics = [
                metric
                for metric in metrics
                if all(metric.tags.get(key) == value for key, value in tags.items())
            ]
        return metrics

    async def query_facts(
        self,
        kind: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        actor: Optional[str] = None,
    ) -> List[Fact]:
        """Query facts with optional filters."""
        async with self._session_provider() as session:
            stmt = select(FactORM)
            if kind:
                stmt = stmt.where(FactORM.kind == kind)
            if actor:
                stmt = stmt.where(FactORM.actor == actor)
            if start_time:
                stmt = stmt.where(FactORM.ts >= start_time)
            if end_time:
                stmt = stmt.where(FactORM.ts <= end_time)
            stmt = stmt.order_by(FactORM.ts.asc())

            result = await session.execute(stmt)
            records = result.scalars().all()

        return [self._fact_from_record(record) for record in records]

    # -------------------------------------------------------------- aggregation
    async def get_aggregation(self, kind: str = "business", window: str = "7d") -> AggregationResult:
        """Return aggregated metrics for a window, using Redis for caching."""
        cache_key = self._cache_key(kind, window)
        cached = await self._get_cached_aggregation(cache_key)
        if cached:
            return cached

        days = self._parse_window_days(window)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        period_metrics = await self.query_metrics(start_time=start_time, end_time=end_time)
        period_facts = await self.query_facts(start_time=start_time, end_time=end_time)

        aggregated_metrics = self._calculate_aggregates(kind, period_metrics, period_facts)

        result = AggregationResult(
            window=window,
            kind=kind,
            metrics=aggregated_metrics,
            facts_count=len(period_facts),
            period_start=start_time,
            period_end=end_time,
        )

        await self._store_aggregate(result)
        await self._cache_aggregation(cache_key, result)
        return result

    # ----------------------------------------------------------- helper methods
    def _metric_from_record(self, record: MetricORM) -> Metric:
        return Metric(
            ts=record.ts,
            name=record.name,
            value=record.value,
            tags=dict(record.tags or {}),
        )

    def _fact_from_record(self, record: FactORM) -> Fact:
        return Fact(
            ts=record.ts,
            kind=record.kind,
            actor=record.actor or "",
            correlation_id=record.correlation_id or "",
            payload=dict(record.payload or {}),
        )

    async def _store_aggregate(self, result: AggregationResult) -> None:
        async with self._session_provider() as session:
            stmt = select(AggregateORM).where(
                AggregateORM.window == result.window,
                AggregateORM.kind == result.kind,
                AggregateORM.period_start == result.period_start,
                AggregateORM.period_end == result.period_end,
            )
            existing = await session.scalar(stmt)

            if existing:
                existing.metrics = dict(result.metrics)
                existing.facts_count = result.facts_count
                existing.calculated_at = datetime.utcnow()
            else:
                session.add(
                    AggregateORM(
                        window=result.window,
                        kind=result.kind,
                        metrics=dict(result.metrics),
                        facts_count=result.facts_count,
                        period_start=result.period_start,
                        period_end=result.period_end,
                    )
                )

            await session.commit()

    def _calculate_aggregates(
        self,
        kind: str,
        period_metrics: List[Metric],
        period_facts: List[Fact],
    ) -> Dict[str, float]:
        if kind == "business":
            trade_facts = [fact for fact in period_facts if fact.kind == "Fact:Trade"]
            outcome_facts = [fact for fact in period_facts if fact.kind == "Fact:Outcome"]

            success_count = sum(1 for fact in outcome_facts if fact.payload.get("success"))
            total_outcomes = max(len(outcome_facts), 1)

            return {
                "total_trades": float(len(trade_facts)),
                "total_outcomes": float(len(outcome_facts)),
                "success_rate": float(success_count / total_outcomes),
            }

        rpc_metrics = [metric for metric in period_metrics if metric.name.startswith("rpc_")]
        if not rpc_metrics:
            return {}

        latency_values = [metric.value for metric in rpc_metrics if "latency" in metric.name]
        error_count = sum(1 for metric in rpc_metrics if "error" in metric.name)
        total_latency = sum(latency_values) if latency_values else 0.0
        latency_count = len(latency_values) or 1

        return {
            "total_rpc_calls": float(len(rpc_metrics)),
            "avg_latency": float(total_latency / latency_count),
            "error_rate": float(error_count / max(len(rpc_metrics), 1)),
        }

    async def _cache_aggregation(self, cache_key: str, result: AggregationResult) -> None:
        if not self._redis:
            return
        payload = {
            "window": result.window,
            "kind": result.kind,
            "metrics": result.metrics,
            "facts_count": result.facts_count,
            "period_start": result.period_start.isoformat(),
            "period_end": result.period_end.isoformat(),
        }
        await self._redis.setex(cache_key, self._cache_ttl, json.dumps(payload))
        await self._redis.sadd(self._CACHE_INDEX_KEY, cache_key)

    async def _get_cached_aggregation(self, cache_key: str) -> Optional[AggregationResult]:
        if not self._redis:
            return None
        cached = await self._redis.get(cache_key)
        if not cached:
            return None
        data = json.loads(cached)
        return AggregationResult(
            window=data["window"],
            kind=data["kind"],
            metrics=data["metrics"],
            facts_count=data["facts_count"],
            period_start=datetime.fromisoformat(data["period_start"]),
            period_end=datetime.fromisoformat(data["period_end"]),
        )

    async def _invalidate_cache(self) -> None:
        if not self._redis:
            return
        keys = await self._redis.smembers(self._CACHE_INDEX_KEY)
        if keys:
            await self._redis.delete(*keys)
        await self._redis.delete(self._CACHE_INDEX_KEY)

    def _cache_key(self, kind: str, window: str) -> str:
        return f"supervisor:agg:{kind}:{window}"

    def _parse_window_days(self, window: str) -> int:
        if window.endswith("d"):
            return max(int(window[:-1]), 1)
        return 7

    async def purge_expired_data(self) -> None:
        """Remove metrics/facts/aggregates outside retention window."""
        retention_days = get_settings().metrics_retention_days
        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        async with self._session_provider() as session:
            await session.execute(delete(MetricORM).where(MetricORM.ts < cutoff))
            await session.execute(delete(FactORM).where(FactORM.ts < cutoff))
            await session.execute(delete(AggregateORM).where(AggregateORM.period_end < cutoff))
            await session.commit()

        await self._invalidate_cache()
