"""Tests for summarization behavior in UnifiedMemory when short-term memory is empty."""

from memory import UnifiedMemory


def test_summarize_to_long_term_ignores_empty_short_term() -> None:
    store = UnifiedMemory()
    store.remember("existing", long_term=True)
    before = list(store.long_term)

    store._summarize_to_long_term()

    assert store.long_term == before
    assert store.short_term == []
