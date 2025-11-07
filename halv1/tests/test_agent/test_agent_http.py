import asyncio
from typing import Iterable, cast

import pytest
from aiohttp import web

from agent import AgentCore
from events.models import MessageReceived
from executor import CodeGenerator, SimpleCodeExecutor
from internet import SearchClient
from memory import UnifiedMemory
from planner import Plan, PlanStep, Tool
from services.event_bus import AsyncEventBus


class DummyLLM:
    def generate(self, prompt: str) -> str:  # pragma: no cover - trivial
        return prompt

    def stream(self, prompt: str) -> Iterable[str]:  # pragma: no cover - not used
        yield prompt


class HTTPPlanner:
    def __init__(self, command: str) -> None:
        self.command = command

    def plan(
        self,
        request: str,
        context: list[str] | None = None,
        previous_results: list[str] | None = None,
    ) -> Plan:
        step = PlanStep(tool=Tool.HTTP, content=self.command)
        return Plan(steps=[step], context=["http"])


async def _start_server() -> tuple[web.AppRunner, str]:
    app = web.Application()

    async def handle_get(request: web.Request) -> web.Response:
        return web.Response(text="hi")

    async def handle_post(request: web.Request) -> web.Response:
        data = await request.text()
        return web.Response(text=data)

    app.router.add_get("/get", handle_get)
    app.router.add_post("/post", handle_post)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    return runner, base_url


def test_http_get_request() -> None:
    async def main() -> list[str]:
        runner, base = await _start_server()
        bus = AsyncEventBus(workers_per_topic=1)
        planner = HTTPPlanner(f"GET {base}/get")
        executor = SimpleCodeExecutor()
        search = SearchClient()
        memory = UnifiedMemory()
        generator = CodeGenerator(DummyLLM())
        AgentCore(bus, planner, executor, search, memory, generator, max_iterations=0)
        await bus.publish(
            "incoming", MessageReceived(chat_id=1, message_id=1, text="run")
        )
        await bus.join()
        await bus.graceful_shutdown()
        await search.close()
        await runner.cleanup()
        return cast(list[str], memory.recall())

    results = asyncio.run(main())
    assert results == ["hi"]


def test_http_post_request() -> None:
    async def main() -> list[str]:
        runner, base = await _start_server()
        bus = AsyncEventBus(workers_per_topic=1)
        planner = HTTPPlanner(f"POST {base}/post\nhello")
        executor = SimpleCodeExecutor()
        search = SearchClient()
        memory = UnifiedMemory()
        generator = CodeGenerator(DummyLLM())
        AgentCore(bus, planner, executor, search, memory, generator, max_iterations=0)
        await bus.publish(
            "incoming", MessageReceived(chat_id=1, message_id=1, text="run")
        )
        await bus.join()
        await bus.graceful_shutdown()
        await search.close()
        await runner.cleanup()
        return cast(list[str], memory.recall())

    results = asyncio.run(main())
    assert results == ["hello"]


def test_http_unknown_method_raises() -> None:
    async def main() -> None:
        bus = AsyncEventBus(workers_per_topic=1)
        planner = HTTPPlanner("PATCH http://example.com")
        executor = SimpleCodeExecutor()
        search = SearchClient()
        memory = UnifiedMemory()
        generator = CodeGenerator(DummyLLM())
        core = AgentCore(
            bus, planner, executor, search, memory, generator, max_iterations=0
        )
        step = PlanStep(tool=Tool.HTTP, content="PATCH http://example.com")
        try:
            await core._execute_http(step)
        finally:
            await search.close()

    with pytest.raises(ValueError):
        asyncio.run(main())
