import asyncio
from typing import cast

from memory import MemoryServiceAdapter
from tasks.scheduler import TaskScheduler, schedule_memory_jobs


class DummyMemory(MemoryServiceAdapter):
    def __init__(self) -> None:
        super().__init__(path='test_memory.db')
        self.saves = 0
        self.prunes = 0
        self.prune_args: list[int] = []

    def save(self) -> None:  # pragma: no cover - simple counter
        self.saves += 1

    def prune_long_term(self, max_items: int = 1000) -> None:  # pragma: no cover
        self.prunes += 1
        self.prune_args.append(max_items)


def test_schedule_memory_jobs_runs() -> None:
    async def main() -> DummyMemory:
        mem = DummyMemory()
        scheduler = TaskScheduler()
        schedule_memory_jobs(
            scheduler,
            mem,
            save_interval=cast(int, 0.01),
            prune_interval=cast(int, 0.02),
            max_items=7,
        )
        runner = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.05)
        await scheduler.stop()
        await runner
        return mem

    mem = asyncio.run(main())
    assert mem.saves >= 1
    assert mem.prunes >= 1
    assert mem.prune_args and all(arg == 7 for arg in mem.prune_args)


def test_schedule_memory_jobs_multiple_memories() -> None:
    async def main() -> tuple[DummyMemory, DummyMemory]:
        mem1 = DummyMemory()
        mem2 = DummyMemory()
        scheduler = TaskScheduler()
        schedule_memory_jobs(
            scheduler,
            mem1,
            save_interval=cast(int, 0.01),
            prune_interval=cast(int, 0.02),
            max_items=3,
        )
        schedule_memory_jobs(
            scheduler,
            mem2,
            save_interval=cast(int, 0.01),
            prune_interval=cast(int, 0.02),
            max_items=5,
        )
        runner = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.05)
        await scheduler.stop()
        await runner
        return mem1, mem2

    mem1, mem2 = asyncio.run(main())
    assert mem1.saves >= 1 and mem2.saves >= 1
    assert mem1.prune_args and all(arg == 3 for arg in mem1.prune_args)
    assert mem2.prune_args and all(arg == 5 for arg in mem2.prune_args)
