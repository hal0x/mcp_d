"""Memory level behavior tests for UnifiedMemory."""

from memory import UnifiedMemory


def test_remember_and_recall_separate_levels() -> None:
    store = UnifiedMemory()
    store.remember("short")
    store.remember("long", long_term=True)

    assert store.recall() == ["short"]
    assert store.recall(long_term=True) == ["long"]


def test_summarization_moves_items_to_long_term() -> None:
    store = UnifiedMemory(short_term_limit=2)
    store.remember("a")
    store.remember("b")
    store.remember("c")  # triggers summarization

    assert store.recall() == []
    assert store.recall(long_term=True) == ["a; b; c"]
    assert store.search("a", long_term=True) == ["a; b; c"]
