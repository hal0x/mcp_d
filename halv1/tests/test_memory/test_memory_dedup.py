from __future__ import annotations

from memory import UnifiedMemory


def test_remember_skips_case_insensitive_duplicates() -> None:
    store = UnifiedMemory()
    store.remember("Hello")
    store.remember("hello")
    assert store.recall() == ["Hello"]


def test_remember_skips_duplicates_across_scopes() -> None:
    store = UnifiedMemory()
    store.remember("Alpha", long_term=True)
    store.remember("alpha")
    assert store.recall(long_term=True) == ["Alpha"]
    assert store.recall() == []


def test_remember_skips_duplicates_from_short_to_long() -> None:
    store = UnifiedMemory()
    store.remember("Beta")
    store.remember("beta", long_term=True)
    assert store.recall() == ["Beta"]
    assert store.recall(long_term=True) == []


def test_remember_dedup_across_memories() -> None:
    store = UnifiedMemory()
    store.remember("Alpha")
    store.remember("alpha", long_term=True)
    assert store.recall() == ["Alpha"]
    assert store.recall(long_term=True) == []

    store = UnifiedMemory()
    store.remember("Beta", long_term=True)
    store.remember("beta")
    assert store.recall(long_term=True) == ["Beta"]
    assert store.recall() == []
