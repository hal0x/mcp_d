from __future__ import annotations

import pytest

from agent.core import AgentCore
from internet import SearchClient
from llm.base_client import LLMClient
from planner import PlanStep, Tool


class DummySearchClient(SearchClient):
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search_and_summarize(
        self,
        query: str,
        max_results: int = 3,
        llm: LLMClient | None = None,
        crawl_depth: int = 0,
    ) -> list[str]:
        self.queries.append(query)
        return [f"results for {query}"]


class EmptySearchClient(SearchClient):
    async def search_and_summarize(
        self,
        query: str,
        max_results: int = 3,
        llm: LLMClient | None = None,
        crawl_depth: int = 0,
    ) -> list[str]:
        return []


@pytest.fixture
def dummy_search_client() -> DummySearchClient:
    return DummySearchClient()


@pytest.mark.asyncio
async def test_execute_search_calls_search_and_summarize(
    dummy_search_client: DummySearchClient,
) -> None:
    core = object.__new__(AgentCore)
    core.search = dummy_search_client
    step = PlanStep(tool=Tool.SEARCH, content="latest tech news")
    result = await core._execute_search(step)
    assert dummy_search_client.queries == ["latest tech news"]
    assert result == {
        "stdout": "results for latest tech news",
        "stderr": "",
        "files": {},
    }


@pytest.mark.asyncio
async def test_execute_search_raises_on_empty_results() -> None:
    core = object.__new__(AgentCore)
    core.search = EmptySearchClient()
    step = PlanStep(tool=Tool.SEARCH, content="nothing here")
    with pytest.raises(RuntimeError):
        await core._execute_search(step)
