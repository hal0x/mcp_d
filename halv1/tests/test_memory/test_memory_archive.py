from __future__ import annotations

import numpy as np

from memory import MemoryArchive, UnifiedMemory


def test_archive_and_restore_short_term(tmp_path):
    store = UnifiedMemory()
    archive = MemoryArchive(tmp_path / "arc.jsonl", tmp_path / "arc.faiss")

    store.remember("foo")
    original = store.short_term[0].embedding
    assert store.forget("foo", archive=archive)
    assert store.recall() == []

    archive.restore_to(store)
    assert store.recall() == ["foo"]
    assert np.allclose(store.short_term[0].embedding, original, atol=1e-3)


def test_archive_and_restore_long_term(tmp_path):
    store = UnifiedMemory()
    archive = MemoryArchive(tmp_path / "arc.jsonl", tmp_path / "arc.faiss")

    store.remember("bar", long_term=True)
    original = store.long_term[0].embedding
    assert store.forget("bar", long_term=True, archive=archive)
    assert store.recall(long_term=True) == []

    archive.restore_to(store, long_term=True)
    assert store.recall(long_term=True) == ["bar"]
    assert np.allclose(store.long_term[0].embedding, original, atol=1e-3)
