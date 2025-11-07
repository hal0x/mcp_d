import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from internet import SearchClient  # noqa: E402


class DummyTransport:
    def __init__(self, responses: Dict[Tuple[str, str], str]) -> None:
        self.responses = responses
        self.engine = "google"

    async def call(self, tool_name: str, arguments):
        key = (
            tool_name,
            arguments.get("query") or arguments.get("url"),
        )
        return self.responses.get(key, "")


class StubLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str):
        self.prompts.append(prompt)
        return ("summarized content", {})


def test_search_parses_markdown() -> None:
    markdown = (
        "1. [OpenAI](https://openai.com) - AI research lab\n"
        "2. [Example](https://example.com) - example result"
    )
    transport = DummyTransport({("search_engine", "openai news"): markdown})
    client = SearchClient(transport=transport)

    results = client.search("openai news", max_results=2)

    assert results == [
        ("OpenAI", "https://openai.com"),
        ("Example", "https://example.com"),
    ]


@pytest.mark.asyncio
async def test_fetch_async_returns_markdown() -> None:
    transport = DummyTransport(
        {
            ("scrape_as_markdown", "https://example.com"): "# Example Page\nContent",
        }
    )
    client = SearchClient(transport=transport)

    content = await client.fetch_async("https://example.com")

    assert "Example Page" in content


@pytest.mark.asyncio
async def test_search_and_summarize_uses_llm() -> None:
    markdown = "1. [Item](https://item.test) - description"
    transport = DummyTransport(
        {
            ("search_engine", "test query"): markdown,
            ("scrape_as_markdown", "https://item.test"): "## Heading\nDetails",
        }
    )
    llm = StubLLM()
    client = SearchClient(transport=transport, llm=llm)

    summaries = await client.search_and_summarize("test query")

    assert summaries and "https://item.test" in summaries[0]
    assert llm.prompts, "LLM should receive a prompt"
