import asyncio
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from types import SimpleNamespace

from index.vector_index import VectorEntry
from retriever.context_builder import build_prompt


class DummyIndex:
    def __init__(self):
        # create several entries that belong to different clusters
        self.entries = [
            VectorEntry("1", "fact1", [1.0, 0.0], {}),
            VectorEntry("2", "fact2", [1.0, 0.0], {}),
            VectorEntry("3", "fact3", [1.0, 0.0], {}),
            VectorEntry("4", "fact4", [0.0, 1.0], {}),
            VectorEntry("5", "fact5", [0.0, 1.0], {}),
            VectorEntry("6", "fact6", [1.0, 1.0], {}),
            VectorEntry("7", "fact7", [1.0, 1.0], {}),
        ]

        class BM25:
            def search(self, q, top_k=25):
                return []

        self.bm25 = BM25()

    async def search(self, query, top_k=25):
        return self.entries

    async def embed(self, text):
        return [1.0, 0.0]


class DummyClusterManager:
    def __init__(self, entries):
        # map entries to cluster ids
        self.clusters = {
            "c1": SimpleNamespace(members=entries[:3]),
            "c2": SimpleNamespace(members=[entries[3]]),
            "c3": SimpleNamespace(members=[entries[4]]),
            "c4": SimpleNamespace(members=[entries[5]]),
            "c5": SimpleNamespace(members=[entries[6]]),
            "c6": SimpleNamespace(members=[]),
        }

    def get_insights(self, emb, top_k=5):
        insights = [
            {"id": "c1", "summary": "s1", "medoid": "m1"},
            {"id": "c2", "summary": "s2", "medoid": "m2"},
            {"id": "c3", "summary": "s3", "medoid": "m3"},
            {"id": "c4", "summary": "s4", "medoid": "m4"},
            {"id": "c5", "summary": "s5", "medoid": "m5"},
            {"id": "c6", "summary": "s6", "medoid": "m6"},
        ]
        return insights[:top_k]


def test_build_prompt_limits_insights_and_includes_facts():
    idx = DummyIndex()
    mgr = DummyClusterManager(idx.entries)
    prompt = asyncio.run(build_prompt("q", idx, mgr))
    # Ensure only top 5 insights are used
    summary_lines = [line for line in prompt.splitlines() if line.startswith("- ")]
    assert len(summary_lines) == 5
    # The first insight should include three raw facts
    frag_lines = [line for line in prompt.splitlines() if line.startswith("  * ")]
    assert frag_lines[:3] == ["  * fact1", "  * fact2", "  * fact3"]


def test_build_prompt_two_pass_returns_summary_and_detailed():
    idx = DummyIndex()
    mgr = DummyClusterManager(idx.entries)
    summary, detailed = asyncio.run(build_prompt("q", idx, mgr, two_pass=True))
    assert summary.count("- ") == 5
    assert "fact1" in detailed
    assert "fact1" not in summary
