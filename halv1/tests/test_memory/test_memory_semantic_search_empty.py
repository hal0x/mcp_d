from __future__ import annotations

from memory import UnifiedMemory


def _make_store() -> UnifiedMemory:
    store = UnifiedMemory()
    store.remember("alpha")
    store.remember("beta", long_term=True)
    store.remember("gamma")
    return store


def test_semantic_search_empty_returns_no_items() -> None:
    store = _make_store()
    assert store.semantic_search("") == []
    assert store.semantic_search("", long_term=True) == []
    assert store.semantic_search("", long_term=False) == []
