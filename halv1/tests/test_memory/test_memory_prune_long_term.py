from datetime import datetime, timezone

from memory import UnifiedMemory


class DummyArchive:
    def __init__(self) -> None:
        self.entries: list[str] = []

    def archive(self, entries) -> None:  # type: ignore[no-untyped-def]
        self.entries.extend(e.text for e in entries)


def test_prune_long_term_evicts_lowest_score() -> None:
    archive = DummyArchive()
    store = UnifiedMemory(archive=archive)
    now = datetime.now(timezone.utc).timestamp()

    store.remember("alpha", long_term=True)
    store.remember("beta", long_term=True)
    store.remember("gamma", long_term=True)

    store.long_term[0].importance = 0.9
    store.long_term[0].usage_freq = 0.9
    store.long_term[0].timestamp = now

    store.long_term[1].importance = 0.1
    store.long_term[1].usage_freq = 0.1
    store.long_term[1].timestamp = now

    store.long_term[2].importance = 0.2
    store.long_term[2].usage_freq = 0.2
    store.long_term[2].timestamp = now

    store.prune_long_term(2)

    assert [e.text for e in store.long_term] == ["alpha", "gamma"]
    assert archive.entries == ["beta"]


def test_prune_long_term_remove_all_when_zero() -> None:
    archive = DummyArchive()
    store = UnifiedMemory(archive=archive)
    store.remember("alpha", long_term=True)
    store.remember("beta", long_term=True)

    store.prune_long_term(0)

    assert store.recall(long_term=True) == ["alpha; beta"]
    assert archive.entries == ["alpha", "beta"]


def test_prune_long_term_preserves_frozen() -> None:
    archive = DummyArchive()
    store = UnifiedMemory(archive=archive)
    store.remember("alpha", long_term=True)
    store.remember("beta", long_term=True)
    store.remember("gamma", long_term=True)
    store.long_term[1].frozen = True

    store.prune_long_term(1)

    texts = [e.text for e in store.long_term]
    assert "beta" in texts  # frozen item remains
    assert "beta" not in archive.entries
    assert len(texts) == 2
