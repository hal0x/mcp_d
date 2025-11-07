from memory import UnifiedMemory


def test_prune_long_term_noop() -> None:
    store = UnifiedMemory()
    store.remember("alpha", long_term=True)
    before = store.recall(long_term=True)
    store.prune_long_term(2)
    assert store.recall(long_term=True) == before
