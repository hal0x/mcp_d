import asyncio
import os
from types import SimpleNamespace

import main


class DummyScheduler:
    instances = []

    def __init__(self):
        self.__class__.instances.append(self)
        self.stopped = False
        self.goal_calls = []

    async def run(self):
        await asyncio.sleep(0)

    def add_periodic(self, func, interval, initial_delay=0):
        pass

    def add_goal(self, *args, **kwargs):
        self.goal_calls.append((args, kwargs))

    async def stop(self):
        self.stopped = True


class DummySummaryService:
    def __init__(self, *args, **kwargs):
        run_smoke.summary = self
        self.calls = 0
        self.stopped = False

    async def start(self):
        pass

    async def hourly_summary(self):
        self.calls += 1

    async def stop(self):
        self.stopped = True


class DummyChronicleService:
    def __init__(self, *args, **kwargs):
        run_smoke.chronicle = self
        self.stopped = False

    async def start(self):
        pass

    async def stop(self):
        self.stopped = True


class DummyTelethonService:
    async def ensure_connected(self):
        pass


class DummySearchClient:
    async def close(self):
        pass


class DummyEventBus:
    async def join(self):
        pass

    async def graceful_shutdown(self):
        pass

    def subscribe(self, *args, **kwargs):
        pass


class DummyUnifiedMemory:
    def save(self):
        pass


class DummyRaw:
    pass


class DummyVector:
    pass


class DummyCluster:
    def load(self, store) -> None:
        pass


class DummyThemeStore:
    def migrate_chat_names_to_sanitized(self) -> None:
        pass


class DummySummarizer:
    pass


class DummyCodeGen:
    pass


def minimal_config(tmp_path):
    return {
        "telegram": {
            "bot_token": "",
            "summary_chat_id": None,
        },
        "scheduler": {},
        "paths": {
            "index": str(tmp_path / "idx"),
            "raw": str(tmp_path / "raw"),
            "agent_memory": str(tmp_path / "agent_mem"),
        },
        "summary": {},
    }


async def run_smoke(tmp_path):
    os.environ["SMOKE_TEST"] = "1"
    main.load_config = lambda: minimal_config(tmp_path)
    main.MemoryServiceAdapter = lambda *a, **k: DummyUnifiedMemory()
    main.AsyncEventBus = lambda *a, **k: DummyEventBus()
    main.TaskScheduler = lambda *a, **k: DummyScheduler()
    main.SearchClient = lambda *a, **k: DummySearchClient()
    main.RawStorage = lambda *a, **k: DummyRaw()

    class DummyVectorIndex:
        @staticmethod
        def themed(*args, **kwargs):
            return DummyVector()

    main.VectorIndex = DummyVectorIndex
    main.ClusterManager = lambda *a, **k: DummyCluster()
    main.ThemeStore = lambda *a, **k: DummyThemeStore()
    main.TelethonIndexer = lambda *a, **k: object()
    main.TelethonService = lambda *a, **k: DummyTelethonService()
    main.Summarizer = lambda *a, **k: DummySummarizer()
    main.FinRLAgent = lambda *a, **k: SimpleNamespace(analyze=lambda text: None)
    main.CodeGenerator = lambda *a, **k: DummyCodeGen()
    main.AgentCore = lambda *a, **k: object()
    main.LLMTaskPlanner = lambda *a, **k: object()
    main.create_executor = lambda *a, **k: object()
    main.SummaryService = DummySummaryService
    main.ChronicleService = DummyChronicleService

    await main.main()
    del os.environ["SMOKE_TEST"]
    return DummyScheduler.instances, run_smoke.summary, run_smoke.chronicle


def test_smoke_mode_cleans_up(tmp_path):
    schedulers, summary, chronicle = asyncio.run(run_smoke(tmp_path))
    assert summary.calls == 1
    assert summary.stopped and chronicle.stopped
    assert len(schedulers) >= 2
    assert all(s.stopped for s in schedulers[:2])
    goal_calls = [call for s in schedulers for call in s.goal_calls]
    assert goal_calls and goal_calls[0][1]["chat_id"] == 134432210
