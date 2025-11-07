"""Tests for persistence and summarization in :mod:`unified_memory`."""

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from memory import ProfileStore, UnifiedMemory


class DummyLLM:
    def __init__(self, response: str | None = None, *, error: bool = False) -> None:
        self.response = response
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:  # pragma: no cover - simple dummy
        self.prompts.append(prompt)
        if self.error:
            raise RuntimeError("LLM failure")
        return self.response or ""

    def stream(self, prompt: str) -> Iterable[str]:  # pragma: no cover - simple dummy
        yield self.generate(prompt)


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


def test_long_term_persistence(tmp_path: Path) -> None:
    path = tmp_path / "lt.db"
    emb = DummyEmbeddings()
    store = UnifiedMemory(path=path, embeddings_client=emb)
    store.remember("persisted", long_term=True)

    store2 = UnifiedMemory(path=path, embeddings_client=emb)
    assert store2.recall(long_term=True) == ["persisted"]


def test_short_term_threshold_and_search(tmp_path: Path) -> None:
    path = tmp_path / "lt.db"
    llm = DummyLLM("alpha; beta; gamma")
    emb = DummyEmbeddings()
    store = UnifiedMemory(
        path=path,
        short_term_limit=2,
        llm_client=llm,
        embeddings_client=emb,
    )
    store.remember("alpha")
    store.remember("beta")
    store.remember("gamma")  # triggers summarization

    assert store.recall() == []
    long_term = store.recall(long_term=True)
    assert long_term == ["alpha; beta; gamma"]
    assert llm.prompts  # ensure LLM was called
    prompt = llm.prompts[0]
    assert "user-provided data" in prompt
    assert "verify that the summary is concise and neutral" in prompt.lower()

    store2 = UnifiedMemory(path=path, short_term_limit=2, embeddings_client=emb)
    assert store2.recall(long_term=True) == long_term
    assert store2.search("beta") == ["alpha; beta; gamma"]


def test_summarization_llm_error_fallback(tmp_path: Path) -> None:
    path = tmp_path / "lt.db"
    llm = DummyLLM(error=True)
    emb = DummyEmbeddings()
    store = UnifiedMemory(
        path=path,
        short_term_limit=2,
        llm_client=llm,
        embeddings_client=emb,
    )
    store.remember("alpha")
    store.remember("beta")
    store.remember("gamma")  # triggers summarization with error

    long_term = store.recall(long_term=True)
    assert long_term == ["alpha; beta; gamma"]

    assert llm.prompts  # LLM attempted
    prompt = llm.prompts[0]
    assert "user-provided data" in prompt
    assert "verify that the summary is concise and neutral" in prompt.lower()

    store2 = UnifiedMemory(path=path, short_term_limit=2, embeddings_client=emb)
    assert store2.recall(long_term=True) == long_term


def test_prune_long_term_respects_max_items(tmp_path: Path) -> None:
    path = tmp_path / "lt.db"
    llm = DummyLLM("merged")
    emb = DummyEmbeddings()
    store = UnifiedMemory(path=path, embeddings_client=emb, llm_client=llm)
    for i in range(5):
        store.remember(f"item {i}", long_term=True)
    max_items = 3
    store.prune_long_term(max_items)
    assert len(store.long_term) <= max_items
    assert llm.prompts
    prompt = llm.prompts[0]
    assert "user-provided data" in prompt
    assert "verify that the summary is concise and neutral" in prompt.lower()

    store2 = UnifiedMemory(path=path, embeddings_client=emb)
    assert len(store2.long_term) <= max_items


def test_prune_long_term_max_items_zero(tmp_path: Path) -> None:
    path = tmp_path / "lt.db"
    emb = DummyEmbeddings()
    store = UnifiedMemory(path=path, embeddings_client=emb)
    for i in range(5):
        store.remember(f"item {i}", long_term=True)
    store.prune_long_term(0)
    assert len(store.long_term) == 1
    expected = "; ".join(f"item {i}" for i in range(5))
    assert store.recall(long_term=True) == [expected]


def test_prune_long_term_ignores_large_max_items(tmp_path: Path) -> None:
    path = tmp_path / "lt.db"
    emb = DummyEmbeddings()
    store = UnifiedMemory(path=path, embeddings_client=emb)
    for i in range(3):
        store.remember(f"item {i}", long_term=True)
    store.prune_long_term(10)
    assert len(store.long_term) == 3
    assert store.recall(long_term=True) == [f"item {i}" for i in range(3)]


def test_profile_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    store = ProfileStore(path)
    store.write("Hal")
    profile = store.read()
    assert profile is not None and profile.preferred_name == "Hal"


def test_profile_store_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profile.json"
    store = ProfileStore(path)

    def _raise(*args: object, **kwargs: object) -> None:
        raise PermissionError

    monkeypatch.setattr(Path, "write_text", _raise)
    with pytest.raises(PermissionError):
        store.write("Hal")


def test_profile_store_repeated_write(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    store = ProfileStore(path)
    store.write("One")
    store.write("Two")
    profile = store.read()
    assert profile is not None and profile.preferred_name == "Two"


def test_profile_store_corrupted_json(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text("{broken", encoding="utf-8")
    store = ProfileStore(path)
    with pytest.raises(ValueError):
        store.read()
