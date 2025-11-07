import asyncio

from index.vector_index import VectorEntry
from retriever.retriever import Retriever


class DummyIndex:
    def __init__(self) -> None:
        self.entries = [
            VectorEntry("1", "finance text", [0.0], {"theme": "finance"}),
            VectorEntry("2", "other text", [0.0], {"theme": "other"}),
            VectorEntry("3", "no theme text", [0.0], {}),
        ]

        class BM25:
            def __init__(self, entries):
                self.entries = entries

            def search(self, query, top_k=25):
                return [(e.chunk_id, 1.0) for e in self.entries]

        self.bm25 = BM25(self.entries)
        self._faiss = None

    async def _embed(self, text):
        return [0.0]


class DummyClusterManager:
    def __init__(self) -> None:
        self.clusters = {}


def _run_select(theme):
    retriever = Retriever(
        DummyIndex(), DummyClusterManager(), get_active_theme=lambda: theme
    )
    return asyncio.run(retriever.select_context("query"))


def test_select_context_theme_none_returns_all():
    entries = _run_select(None)
    assert {e.chunk_id for e in entries} == {"1", "2", "3"}


def test_select_context_filters_finance():
    entries = _run_select("finance")
    assert {e.chunk_id for e in entries} == {"1"}
