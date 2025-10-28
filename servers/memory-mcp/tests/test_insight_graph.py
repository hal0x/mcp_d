"""Unit tests for the summary insight analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memory_mcp.analysis.insight_graph import InsightGraphResult, SummaryInsightAnalyzer


class StubEmbeddingClient:
    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * 8 for _ in texts]


class StubCollection:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def query(self, **_: Any) -> dict[str, Any]:
        return self._response


class StubChromaClient:
    def __init__(self, collection: StubCollection) -> None:
        self._collection = collection

    def get_collection(self, name: str) -> StubCollection:
        assert name == "telegram_summaries"
        return self._collection


@pytest.mark.asyncio()
async def test_insight_graph_builds_basic_relations(temp_dir: Path) -> None:
    """Analyzer should build nodes, tags, and similarity edges."""

    summary_a = temp_dir / "Alpha.md"
    summary_a.write_text(
        """---
chat: Alpha
purpose: Alpha purpose
participants:
  - Alice
  - Bob
topics_canon:
  - Topic One
  - Topic Two
tags_glossary:
  - tag: alpha_tag
    desc: Alpha tag
---

## Overview

Details about Alpha chat.

- [Session One](sessions/ALPHA-001.md)
""",
        encoding="utf-8",
    )

    summary_b = temp_dir / "Beta.md"
    summary_b.write_text(
        """---
chat: Beta
purpose: Beta purpose
participants:
  - Carol
topics_canon:
  - Topic Two
  - Topic Three
tags_glossary:
  - tag: beta_tag
    desc: Beta tag
---

## Highlights

Details about Beta chat.
""",
        encoding="utf-8",
    )

    stub_response = {
        "metadatas": [[{"chat_name": "Beta"}]],
        "distances": [[0.1]],
    }

    collection = StubCollection(response=stub_response)
    chroma_client = StubChromaClient(collection)

    analyzer = SummaryInsightAnalyzer(
        summaries_dir=temp_dir,
        chroma_path=temp_dir / "db",
        ollama_client=StubEmbeddingClient(),
        chroma_client=chroma_client,
        similarity_threshold=0.5,
    )

    async with analyzer:
        result: InsightGraphResult = await analyzer.analyze()

    graph = result.graph
    assert "Alpha" in graph
    assert "Beta" in graph
    assert any(data.get("relation") == "tag" for _, _, data in graph.edges(data=True))
    assert (
        graph.has_edge("Alpha", "Beta")
        and graph.get_edge_data("Alpha", "Beta").get("relation") == "similarity"
    )
    report = SummaryInsightAnalyzer.generate_report(result)
    assert "Insight Graph Summary" in report
    assert "Ключевые хабы обсуждений" in report
