"""Simple asyncio-based task scheduler."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping

from events.models import MessageReceived
from memory import EpisodeGraph, MemoryServiceAdapter, consolidate_graph
from services.event_bus import AsyncEventBus

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Run coroutines periodically."""

    def __init__(self) -> None:
        self._tasks: List[asyncio.Task[None]] = []
        self._jobs: List[tuple[Callable[[], Awaitable[None]], int, int]] = []
        self._goal_counters: Dict[int, int] = {}

    def add_periodic(
        self,
        coro_factory: Callable[[], Awaitable[None]],
        interval: int,
        initial_delay: int = 0,
    ) -> None:
        """Schedule ``coro_factory`` to run every ``interval`` seconds.

        If ``initial_delay`` > 0, waits that many seconds before the first run.
        """

        # Store the job so ``run`` can recreate tasks after a stop
        self._jobs.append((coro_factory, interval, initial_delay))

        # If ``run`` is already active, start the task immediately
        if self._tasks:
            self._start_job(coro_factory, interval, initial_delay)

    def _start_job(
        self,
        coro_factory: Callable[[], Awaitable[None]],
        interval: int,
        initial_delay: int,
    ) -> None:
        async def _runner(
            cf: Callable[[], Awaitable[None]] = coro_factory,
            iv: int = interval,
            delay: int = initial_delay,
        ) -> None:
            try:
                if delay and delay > 0:
                    await asyncio.sleep(delay)
                while True:
                    try:
                        await cf()
                    except asyncio.CancelledError:
                        break
                    except Exception as exc:  # pragma: no cover - background errors
                        logger.exception("Periodic task error: %s", exc)
                    await asyncio.sleep(iv)
            except asyncio.CancelledError:
                pass

        self._tasks.append(asyncio.create_task(_runner()))

    async def run(self) -> None:
        """Wait for all scheduled tasks (runs forever)."""
        # Materialize tasks from stored jobs only once
        if self._jobs and not self._tasks:
            for coro_factory, interval, initial_delay in self._jobs:
                self._start_job(coro_factory, interval, initial_delay)
        if not self._tasks:  # pragma: no cover - nothing scheduled
            return
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            # Running tasks may be cancelled via ``stop()``; ensure we exit quietly
            pass

    async def stop(self) -> None:
        """Cancel all scheduled tasks and wait for them to finish."""
        if not self._tasks:
            return
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def add_goal(
        self,
        bus: AsyncEventBus[MessageReceived],
        chat_id: int,
        goal: str,
        interval: int,
        initial_delay: int = 0,
    ) -> None:
        """Schedule a high-level goal to be published as :class:`MessageReceived`.

        Each run publishes the goal text to the ``incoming`` topic so that the
        agent processes it like a user message.
        """

        async def _publish() -> None:
            counter = self._goal_counters.get(chat_id, 0) + 1
            self._goal_counters[chat_id] = counter
            await bus.publish(
                "incoming",
                MessageReceived(chat_id=chat_id, message_id=counter, text=goal),
            )

        self.add_periodic(lambda: _publish(), interval, initial_delay)


@dataclass
class ClusterScheduleConfig:
    """Configuration for cluster maintenance tasks."""

    summarise_interval: int = 3600
    decay_interval: int = 3600
    recluster_interval: int = 3600
    decay_half_life: float = 86400.0
    missing_facts_interval: int = 3600

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "ClusterScheduleConfig":
        """Build config from a generic mapping.

        The mapping is expected to use the keys present in ``settings.yaml``
        under ``scheduler``.  This helper makes it easy to pass raw config
        dictionaries to :func:`schedule_cluster_jobs`.
        """

        return ClusterScheduleConfig(
            summarise_interval=int(
                data.get(
                    "cluster_summarise_interval_seconds",
                    data.get("summarise_interval", 3600),
                )
            ),
            decay_interval=int(
                data.get(
                    "cluster_decay_interval_seconds", data.get("decay_interval", 3600)
                )
            ),
            recluster_interval=int(
                data.get(
                    "cluster_recluster_interval_seconds",
                    data.get("recluster_interval", 3600),
                )
            ),
            decay_half_life=float(
                data.get(
                    "cluster_decay_half_life_seconds",
                    data.get("decay_half_life", 86400.0),
                )
            ),
            missing_facts_interval=int(
                data.get(
                    "missing_facts_interval_seconds",
                    data.get("missing_facts_interval", 3600),
                )
            ),
        )


async def summarise_clusters(manager: Any, summariser: Any) -> None:
    """Generate summaries for all clusters using ``summariser``."""

    manager.summarise(summariser)


async def decay_clusters(manager: Any, *, half_life: float, interval: float) -> None:
    """Apply freshness decay to clusters."""

    _decay_clusters(manager, half_life=half_life, interval=interval)


async def rebuild_clusters(manager: Any) -> None:
    """Recompute cluster structure or check for change points."""

    _recluster_or_check(manager)


async def analyse_missing_facts(manager: Any) -> None:
    """Inspect missing facts log and trigger follow-up actions."""

    path = getattr(manager, "missing_facts_path", None)
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    lines = [
        line.strip()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return
    logger.info("Missing facts detected: %d", len(lines))
    # Placeholder for reindexing/expansion logic
    p.write_text("", encoding="utf-8")


def _decay_clusters(
    manager: Any,
    *,
    half_life: float,
    interval: float,
    stale_threshold: float = 0.1,
) -> None:
    """Apply exponential decay to cluster scores removing stale ones."""

    now = time.time()
    stale: List[str] = []
    for cid, cluster in list(manager.clusters.items()):
        if half_life <= 0:
            factor = 0.0
        else:
            elapsed = max(interval, now - getattr(cluster, "freshness_ts", 0.0))
            factor = 0.5 ** (elapsed / half_life)
        cluster.freshness *= factor
        cluster.source_quality *= factor
        cluster.freshness_ts = now
        if cluster.freshness < stale_threshold:
            stale.append(cid)
    for cid in stale:
        del manager.clusters[cid]


def _recluster_or_check(manager: Any) -> None:
    """Trigger re-clustering or change-point checks."""

    if hasattr(manager, "rebuild_clusters"):
        manager.rebuild_clusters()
    elif hasattr(manager, "recluster"):
        manager.recluster()
    else:
        for cluster in manager.clusters.values():
            cluster.recompute_centroid()
            cluster.recompute_medoid()


def schedule_cluster_jobs(
    scheduler: TaskScheduler,
    manager: Any,
    summariser: Any,
    config: ClusterScheduleConfig | Mapping[str, Any] | None = None,
) -> None:
    """Create periodic jobs for cluster management.

    ``config`` may be provided either as :class:`ClusterScheduleConfig` or as a
    plain mapping (e.g. a dictionary loaded from YAML).  If ``None`` is passed,
    default intervals are used.
    """

    if config is None:
        cfg = ClusterScheduleConfig()
    elif isinstance(config, ClusterScheduleConfig):
        cfg = config
    else:
        cfg = ClusterScheduleConfig.from_mapping(config)

    scheduler.add_periodic(
        lambda: summarise_clusters(manager, summariser), cfg.summarise_interval
    )
    scheduler.add_periodic(
        lambda: decay_clusters(
            manager,
            half_life=cfg.decay_half_life,
            interval=cfg.decay_interval,
        ),
        cfg.decay_interval,
    )
    scheduler.add_periodic(lambda: rebuild_clusters(manager), cfg.recluster_interval)
    scheduler.add_periodic(
        lambda: analyse_missing_facts(manager), cfg.missing_facts_interval
    )


def schedule_memory_jobs(
    scheduler: TaskScheduler,
    memory: MemoryServiceAdapter,
    *,
    save_interval: int = 300,
    prune_interval: int = 3600,
    max_items: int = 1000,
) -> None:
    """Schedule periodic persistence and cleanup for ``memory``."""

    # Проверяем, есть ли работающий event loop
    try:
        asyncio.get_running_loop()
        # Если есть работающий loop, добавляем задачи
        scheduler.add_periodic(lambda: asyncio.to_thread(memory.save), save_interval)
        # Используем prune_long_term для совместимости с UnifiedMemory
        scheduler.add_periodic(
            lambda: asyncio.to_thread(memory.prune_long_term, max_items), prune_interval
        )
    except RuntimeError:
        # Если нет работающего loop, просто логируем предупреждение
        logger.warning("No running event loop, skipping memory job scheduling")


def schedule_schema_consolidation(
    scheduler: TaskScheduler,
    graph: EpisodeGraph,
    *,
    interval: int = 86400,
) -> None:
    """Schedule periodic schema consolidation for ``graph``.

    The job runs :func:`memory.schemas.consolidate_graph` every ``interval``
    seconds. It is typically executed once per day to group recent events into
    schemas and assign their identifiers in the episode graph.
    """

    try:
        asyncio.get_running_loop()
        scheduler.add_periodic(
            lambda: asyncio.to_thread(consolidate_graph, graph), interval
        )
    except RuntimeError:
        logger.warning(
            "No running event loop, skipping schema consolidation scheduling"
        )
