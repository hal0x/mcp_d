import asyncio
from types import SimpleNamespace

from index.vector_index import VectorEntry
from retriever.insight_utils import build_insight_cards


class DummyIndex:
    def __init__(self) -> None:
        self.entries = [
            VectorEntry("a1", "f1", [], {}),
            VectorEntry("a2", "f2", [], {}),
            VectorEntry("b1", "g1", [], {}),
            VectorEntry("a3", "f3", [], {}),
            VectorEntry("a4", "f4", [], {}),
        ]

        class BM25:
            def search(self, query: str, top_k: int = 25):
                return [("a1", 0.9), ("b1", 0.8), ("a2", 0.5)]

        self.bm25 = BM25()

    async def search(self, query: str, top_k: int = 25):
        return self.entries[:top_k]

    async def embed(self, text: str):
        return [1.0, 0.0]


class DummyClusterManager:
    def __init__(self, entries: list[VectorEntry]) -> None:
        self.clusters = {
            "c1": SimpleNamespace(members=entries[:4]),
            "c2": SimpleNamespace(members=[entries[2]]),
        }
        self.calls: tuple[dict[str, float], dict[str, float]] | None = None

    def get_insights(
        self,
        emb,
        top_k: int = 5,
        *,
        cluster_weight: dict[str, float] | None = None,
        bm25_score: dict[str, float] | None = None,
        w5: float = 0.1,
        w6: float = 0.1,
    ):
        self.calls = (dict(cluster_weight or {}), dict(bm25_score or {}))
        return [
            {"id": "c1", "summary": "s1", "medoid": "m1"},
            {"id": "c2", "summary": "s2", "medoid": "m2"},
        ][:top_k]


def test_build_insight_cards_limits_fragments_and_passes_weights() -> None:
    idx = DummyIndex()
    mgr = DummyClusterManager(idx.entries)
    cards, hits = asyncio.run(
        build_insight_cards("q", idx, mgr, top_k_vectors=5, top_k_insights=5)
    )
    assert len(hits) == 5
    assert cards == [
        {"summary": "s1", "medoid": "m1", "fragments": ["f1", "f2", "f3"]},
        {"summary": "s2", "medoid": "m2", "fragments": ["g1"]},
    ]
    assert mgr.calls == ({"c1": 3.0, "c2": 1.0}, {"c1": 0.9, "c2": 0.8})
