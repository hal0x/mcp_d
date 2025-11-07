import asyncio
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from unittest.mock import AsyncMock, patch

from index.cluster_manager import Cluster, ClusterManager
from index.vector_index import VectorEntry
from tasks.scheduler import (
    ClusterScheduleConfig,
    TaskScheduler,
    _decay_clusters,
    schedule_cluster_jobs,
)
from tasks.scheduler import (
    decay_clusters as real_decay_clusters,
)
from tasks.scheduler import (
    rebuild_clusters as real_rebuild_clusters,
)
from tasks.scheduler import (
    summarise_clusters as real_summarise_clusters,
)


class DummySummarizer:
    def summarize(self, texts):
        return "summary"


class DummyManager(ClusterManager):
    def __init__(self):
        super().__init__()
        self.summaries = 0
        self.rebuilds = 0

    def summarise(self, summariser, **kwargs):
        self.summaries += 1

    def rebuild_clusters(self):
        self.rebuilds += 1


class MockScheduler:
    def __init__(self):
        self.jobs = []

    def add_periodic(self, coro_factory, interval, initial_delay=0):
        self.jobs.append(
            {"coro": coro_factory, "interval": interval, "next": initial_delay}
        )

    async def run_for(self, total):
        t = 0
        while t < total:
            for job in self.jobs:
                if t >= job["next"]:
                    await job["coro"]()
                    job["next"] += job["interval"]
            t += 1


def test_scheduler_stop_cancels_tasks():
    async def main():
        scheduler = TaskScheduler()

        async def sleepy():
            await asyncio.sleep(10)

        scheduler.add_periodic(lambda: sleepy(), interval=1)

        run_task = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0)  # let the task start
        scheduled = list(scheduler._tasks)
        await scheduler.stop()
        await run_task
        assert all(t.done() for t in scheduled)
        assert scheduler._tasks == []

    asyncio.run(main())


def test_jobs_added_after_run_begin_execution():
    async def main():
        scheduler = TaskScheduler()
        first_called = asyncio.Event()
        second_called = asyncio.Event()

        async def first() -> None:
            first_called.set()

        async def second() -> None:
            second_called.set()

        scheduler.add_periodic(lambda: first(), interval=0)
        run_task = asyncio.create_task(scheduler.run())

        await asyncio.wait_for(first_called.wait(), timeout=1)
        assert len(scheduler._tasks) == 1

        scheduler.add_periodic(lambda: second(), interval=0)
        await asyncio.wait_for(second_called.wait(), timeout=1)
        assert len(scheduler._tasks) == 2

        await scheduler.stop()
        await run_task

    asyncio.run(main())


def test_cluster_jobs_run_on_schedule():
    scheduler = MockScheduler()
    manager = DummyManager()
    entry = VectorEntry("1", "text", [1.0, 0.0], {})
    cluster = Cluster(
        id="c",
        members=[entry],
        centroid=[1.0, 0.0],
        medoid=entry,
        freshness=0.2,
        freshness_ts=time.time(),
        source_quality=1.0,
    )
    manager.clusters = {"c": cluster}
    summariser = DummySummarizer()
    config = ClusterScheduleConfig(
        summarise_interval=2,
        decay_interval=3,
        recluster_interval=5,
        decay_half_life=6,
    )

    with (
        patch(
            "tasks.scheduler.summarise_clusters",
            AsyncMock(side_effect=real_summarise_clusters),
        ) as summ_mock,
        patch(
            "tasks.scheduler.decay_clusters",
            AsyncMock(side_effect=real_decay_clusters),
        ) as decay_mock,
        patch(
            "tasks.scheduler.rebuild_clusters",
            AsyncMock(side_effect=real_rebuild_clusters),
        ) as rebuild_mock,
    ):
        schedule_cluster_jobs(scheduler, manager, summariser, config)
        asyncio.run(scheduler.run_for(10))

    assert summ_mock.await_count == 5
    assert decay_mock.await_count == 4
    assert rebuild_mock.await_count == 2
    assert manager.summaries == 5
    assert manager.rebuilds == 2
    assert "c" not in manager.clusters


def test_cluster_jobs_accept_dict_config():
    scheduler = MockScheduler()
    manager = DummyManager()
    entry = VectorEntry("1", "text", [1.0, 0.0], {})
    cluster = Cluster(
        id="c",
        members=[entry],
        centroid=[1.0, 0.0],
        medoid=entry,
        freshness=0.2,
        freshness_ts=time.time(),
        source_quality=1.0,
    )
    manager.clusters = {"c": cluster}
    summariser = DummySummarizer()
    config_dict = {
        "cluster_summarise_interval_seconds": 2,
        "cluster_decay_interval_seconds": 3,
        "cluster_recluster_interval_seconds": 5,
        "cluster_decay_half_life_seconds": 6,
    }

    with (
        patch(
            "tasks.scheduler.summarise_clusters",
            AsyncMock(side_effect=real_summarise_clusters),
        ) as summ_mock,
        patch(
            "tasks.scheduler.decay_clusters",
            AsyncMock(side_effect=real_decay_clusters),
        ) as decay_mock,
        patch(
            "tasks.scheduler.rebuild_clusters",
            AsyncMock(side_effect=real_rebuild_clusters),
        ) as rebuild_mock,
    ):
        schedule_cluster_jobs(scheduler, manager, summariser, config_dict)
        asyncio.run(scheduler.run_for(10))

    assert summ_mock.await_count == 5
    assert decay_mock.await_count == 4
    assert rebuild_mock.await_count == 2
    assert manager.summaries == 5
    assert manager.rebuilds == 2
    assert "c" not in manager.clusters


def test_decay_clusters_reduces_quality():
    mgr = ClusterManager()
    entry = VectorEntry("1", "text", [1.0, 0.0], {"source": "news"})
    mgr.ingest([entry])
    cid = next(iter(mgr.clusters))
    cl = mgr.clusters[cid]
    initial_quality = cl.source_quality
    initial_fresh = cl.freshness
    _decay_clusters(mgr, half_life=6, interval=3)
    cl = mgr.clusters[cid]
    assert cl.source_quality < initial_quality
    assert cl.freshness < initial_fresh
