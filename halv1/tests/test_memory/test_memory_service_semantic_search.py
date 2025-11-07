from __future__ import annotations

from memory import MemoryService


class DummyEmbeddings:
    def embed(self, text: str) -> list[float]:
        tokens = text.lower().split()
        dim = 64
        vec = [0.0] * dim
        for tok in tokens:
            h = hash(tok) % dim
            vec[h] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm:
            vec = [v / norm for v in vec]
        return vec


def test_semantic_search_long_term_scope(tmp_path) -> None:
    memory = MemoryService(tmp_path / "graph.db", embeddings_client=DummyEmbeddings())
    short_item = "cat animal"
    long_item = "dog animal"
    memory.write_event(short_item)
    memory.remember(long_item)

    assert memory.semantic_search("animal", long_term=False) == [short_item]
    assert memory.semantic_search("animal", long_term=True) == [long_item]
    assert memory.semantic_search("animal") == [short_item, long_item]


def test_semantic_search_empty_returns_no_items(tmp_path) -> None:
    memory = MemoryService(tmp_path / "graph.db", embeddings_client=DummyEmbeddings())
    memory.write_event("alpha")
    memory.remember("beta")
    assert memory.semantic_search("") == []
