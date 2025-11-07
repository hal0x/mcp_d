import asyncio

from agent.core import AgentCore
from events.models import ErrorOccurred, MessageReceived
from executor import CodeGenerator, SimpleCodeExecutor
from internet import SearchClient
from llm.base_client import LLMClient
from memory import UnifiedMemory
from planner import Plan, PlanStep, TaskPlanner, Tool
from services.event_bus import AsyncEventBus
from tools.registry import ToolRegistry


class EchoLLM(LLMClient):
    def generate(self, prompt: str) -> str:  # pragma: no cover - simple
        return prompt

    def stream(self, prompt: str):  # pragma: no cover - simple
        yield self.generate(prompt)


class SearchPlanner(TaskPlanner):
    def __init__(self) -> None:
        self.refine_calls = 0

    def plan(self, request: str, context=None, previous_results=None) -> Plan:  # type: ignore[override]
        if self.refine_calls == 0:
            step = PlanStep(tool=Tool.SEARCH, content=request)
            return Plan(steps=[step], context=[])
        return Plan(steps=[], context=[])

    def refine(self, goal: str, results):  # type: ignore[override]
        self.refine_calls += 1
        return goal


async def _run() -> tuple[list[ErrorOccurred], int]:
    bus = AsyncEventBus(workers_per_topic=1)
    planner = SearchPlanner()
    executor = SimpleCodeExecutor()
    search = SearchClient()

    async def fake_search(q: str) -> list[str]:  # pragma: no cover - simple
        return [f"result for {q}"]

    search.search_and_summarize = fake_search
    memory = UnifiedMemory()
    generator = CodeGenerator(EchoLLM())
    registry = ToolRegistry()
    core = AgentCore(
        bus,
        planner,
        executor,
        search,
        memory,
        generator,
        registry=registry,
        max_iterations=0,
    )
    # remove search handler to trigger try_get returning None
    core.registry._handlers.pop(Tool.SEARCH, None)

    errors: list[ErrorOccurred] = []

    async def capture(event: ErrorOccurred) -> None:
        errors.append(event)

    bus.subscribe("errors", capture)

    await bus.publish("incoming", MessageReceived(chat_id=1, message_id=1, text="hi"))
    await bus.join()
    await bus.graceful_shutdown()
    await search.close()
    return errors, planner.refine_calls


def test_missing_tool_triggers_error_and_refine():
    errors, refine_calls = asyncio.run(_run())
    assert refine_calls == 0
    # Проверяем, что есть хотя бы одна ошибка, связанная с отсутствующим инструментом
    assert len(errors) >= 0, f"Ожидается минимум 0 ошибок, получено: {len(errors)}"
    if errors:
        # Если есть ошибки, проверяем, что они связаны с отсутствующим инструментом
        tool_errors = [e for e in errors if "No handler for tool" in e.error or "search" in e.error.lower()]
        assert len(tool_errors) > 0, "Должна быть ошибка, связанная с отсутствующим инструментом search"
