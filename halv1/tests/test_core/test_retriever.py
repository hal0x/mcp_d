import asyncio
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from types import SimpleNamespace

from index.vector_index import VectorEntry
from retriever.retriever import Retriever


class DummyIndex:
    def __init__(self):
        self.search_args = None
        self.entries = [VectorEntry("1", "text", [1.0, 0.0], {})]

        class BM25:
            def search(self, query, top_k=25):
                return []

        self.bm25 = BM25()

    async def search(self, query, top_k=25):
        self.search_args = (query, top_k)
        return self.entries

    async def embed(self, text):
        return [1.0, 0.0]


class DummyClusterManager:
    def __init__(self):
        self.calls = None
        entry = VectorEntry("1", "text", [1.0, 0.0], {})
        self.clusters = {"c1": SimpleNamespace(members=[entry])}

    def get_insights(
        self,
        emb,
        top_k=5,
        cluster_weight=None,
        bm25_score=None,
        w5=0.1,
        w6=0.1,
    ):
        self.calls = (emb, top_k, cluster_weight or {}, bm25_score or {})
        return [{"id": "c1", "summary": "sum", "medoid": "med"}]


def test_query_with_insights_merges_results():
    idx = DummyIndex()
    mgr = DummyClusterManager()
    retriever = Retriever(idx, mgr)
    result = asyncio.run(
        retriever.query_with_insights("q", top_k_vectors=2, top_k_insights=3)
    )
    assert result.vector_hits and result.vector_hits[0].text == "text"
    assert result.insights == [
        {"summary": "sum", "medoid": "med", "fragments": ["text"]}
    ]
    assert "text" in result.context
    assert idx.search_args == ("q", 2)
    assert mgr.calls == ([1.0, 0.0], 3, {"c1": 1.0}, {})


def test_query_with_insights_two_pass_returns_brief():
    idx = DummyIndex()
    mgr = DummyClusterManager()
    retriever = Retriever(idx, mgr)
    result = asyncio.run(retriever.query_with_insights("q", two_pass=True))
    assert result.brief_context and "- sum: med" in result.brief_context
    assert "text" in result.context
