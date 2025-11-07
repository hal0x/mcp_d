import asyncio

from agent.core import AgentCore
from events.models import MessageReceived
from executor import CodeGenerator, SimpleCodeExecutor
from internet import SearchClient
from llm.base_client import LLMClient
from memory import UnifiedMemory
from planner import SimpleTaskPlanner
from services.event_bus import AsyncEventBus
from tasks.scheduler import TaskScheduler


class EchoLLM(LLMClient):
    def generate(self, prompt: str) -> str:  # pragma: no cover - simple
        mapping = {"add two and two": "result = 2 + 2"}
        for key, value in mapping.items():
            if key in prompt:
                return value
        return prompt

    def stream(self, prompt: str):  # pragma: no cover - simple
        yield self.generate(prompt)


def test_goal_scheduler_publishes_events():
    async def main() -> int:
        bus = AsyncEventBus(workers_per_topic=1)
        scheduler = TaskScheduler()
        seen: list[MessageReceived] = []

        async def capture(event: MessageReceived) -> None:
            seen.append(event)

        bus.subscribe("incoming", capture)
        scheduler.add_goal(bus, chat_id=1, goal="hello", interval=0.05)
        run_task = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.16)
        await scheduler.stop()
        await run_task
        await bus.join("incoming")
        await bus.graceful_shutdown()
        return len(seen)

    count = asyncio.run(main())
    assert count >= 2


def test_goal_scheduler_agent_core():
    async def main() -> list[str]:
        bus = AsyncEventBus(workers_per_topic=1)
        scheduler = TaskScheduler()
        planner = SimpleTaskPlanner()
        executor = SimpleCodeExecutor()
        search = SearchClient()
        memory = UnifiedMemory()
        generator = CodeGenerator(EchoLLM())
        AgentCore(bus, planner, executor, search, memory, generator, max_iterations=0)

        scheduler.add_goal(bus, chat_id=1, goal="add two and two", interval=0.05)
        run_task = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.12)
        await scheduler.stop()
        await run_task
        await bus.join()
        await bus.graceful_shutdown()
        await search.close()
        return memory.recall()

    results = asyncio.run(main())
    assert "4" in results


def test_goal_scheduler_message_ids_increment() -> None:
    async def main() -> list[int]:
        bus = AsyncEventBus(workers_per_topic=1)
        scheduler = TaskScheduler()
        seen: list[int] = []

        async def capture(event: MessageReceived) -> None:
            seen.append(event.message_id)

        bus.subscribe("incoming", capture)
        scheduler.add_goal(bus, chat_id=1, goal="ping", interval=0.05)
        run_task = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.12)
        await scheduler.stop()
        await run_task
        await bus.join("incoming")
        await bus.graceful_shutdown()
        return seen

    ids = asyncio.run(main())
    assert ids[:2] == [1, 2]
