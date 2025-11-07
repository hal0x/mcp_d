import hashlib

import numpy as np

from memory import UnifiedMemory


class DummyEmbeddings:
    def embed(self, text: str) -> list[float]:
        tokens = text.lower().split()
        dim = 64
        vec = np.zeros(dim, dtype=float)
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()


class FailingEmbeddings:
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("boom")


def test_semantic_search_returns_relevant_items() -> None:
    store = UnifiedMemory(embeddings_client=DummyEmbeddings())
    store.remember("Paris is the capital of France")
    store.remember("Berlin is the capital of Germany")
    results = store.semantic_search("France capital")
    assert results and "France" in results[0]


def test_semantic_search_long_term_scope() -> None:
    store = UnifiedMemory(embeddings_client=DummyEmbeddings())
    short_item = "Cat sits on the mat"
    long_item = "Dog plays in the park"
    store.remember(short_item)
    store.remember(long_item, long_term=True)

    long_results = store.semantic_search("dog", long_term=True)
    assert long_results == [long_item]

    short_results = store.semantic_search("dog", long_term=False)
    assert short_results == []

    assert store.semantic_search("hamster", long_term=True) == []
    assert store.semantic_search("hamster", long_term=False) == []


def test_semantic_search_respects_top_k() -> None:
    store = UnifiedMemory(embeddings_client=DummyEmbeddings())
    store.remember("apple fruit")
    store.remember("banana fruit")
    store.remember("cherry fruit")

    results = store.semantic_search("fruit", top_k=2)
    assert len(results) == 2
    
def test_embed_fallback_is_deterministic() -> None:
    failing_store = UnifiedMemory(embeddings_client=FailingEmbeddings())
    text = "Deterministic text"
    vec1 = failing_store._embed(text)
    vec2 = failing_store._embed(text)
    expected = DummyEmbeddings().embed(text)
    assert vec1 == vec2 == expected
