"""Basic tests for architectural components."""

import asyncio

import pytest

from core import Agent
from executor import ExecutionError, SimpleCodeExecutor
from internet import SearchClient
from memory import UnifiedMemory
from planner import Plan, PlanStep, SimpleTaskPlanner, TaskPlanner, Tool


def test_agent_planner_executor() -> None:
    planner = SimpleTaskPlanner()
    executor = SimpleCodeExecutor()
    memory = UnifiedMemory()
    agent = Agent(planner, executor, memory)
    result = agent.handle_request_sync("result = 1 + 1")
    assert result == ["2"]


class SearchOnlyPlanner(TaskPlanner):
    def plan(
        self,
        request: str,
        context: list[str] | None = None,
        previous_results: list[str] | None = None,
    ) -> Plan:
        step = PlanStep(tool=Tool.SEARCH, content=request)
        return Plan(steps=[step], context=["search"])


def test_agent_handles_search() -> None:
    planner = SearchOnlyPlanner()
    executor = SimpleCodeExecutor()
    search = SearchClient()

    async def fake_search(q: str) -> list[str]:
        return [f"result for {q}: http://example.com/{q}"]

    search.search_and_summarize = fake_search  # type: ignore[assignment]
    memory = UnifiedMemory()
    agent = Agent(planner, executor, memory, search)
    result = agent.handle_request_sync("python")
    assert result == ["result for python: http://example.com/python"]
    asyncio.run(search.close())


def test_code_executor_error() -> None:
    executor = SimpleCodeExecutor()
    try:
        executor.execute("raise ValueError('boom')")
    except ExecutionError as e:
        # Проверяем, что исключение содержит правильную информацию
        assert isinstance(e, ExecutionError), "Должно быть исключение ExecutionError"
        assert "boom" in str(e), "Сообщение об ошибке должно содержать 'boom'"
    else:  # pragma: no cover - sanity
        assert False, "Должно было быть вызвано исключение ExecutionError"


def test_search_client_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = SearchClient()

    def fake_duckduckgo(
        self: SearchClient, query: str, max_results: int
    ) -> list[tuple[str, str]]:
        return [("Python is a programming language", "https://example.com/python")]

    client.provider = fake_duckduckgo.__get__(client, SearchClient)

    with caplog.at_level("INFO", logger=client.logger.name):
        results = client.search("python")
    assert results == [
        ("Python is a programming language", "https://example.com/python")
    ]
    asyncio.run(client.close())


def test_memory_store() -> None:
    store = UnifiedMemory()
    store.remember("a")
    store.remember("b", long_term=True)
    assert store.recall() == ["a"]
    assert store.recall(long_term=True) == ["b"]
