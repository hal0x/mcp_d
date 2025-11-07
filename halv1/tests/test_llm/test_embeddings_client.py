import asyncio
from typing import Any, Dict, List, Tuple

from llm.embeddings_client import AsyncEmbeddingsClient, EmbeddingsClient


class DummyAsyncClient(AsyncEmbeddingsClient):
    def __init__(self) -> None:
        super().__init__(model="test")
        self.calls: list[Dict[str, Any]] = []

    async def post_embeddings(
        self, payload: Dict[str, Any]
    ) -> Tuple[List[List[float]], int]:
        self.calls.append(payload)
        return [[1.0, 2.0, 3.0]], 0


class CountingAsyncClient(AsyncEmbeddingsClient):
    """Client that returns different embeddings on each call."""

    def __init__(self) -> None:
        super().__init__(model="test")
        self.calls: int = 0

    async def post_embeddings(
        self, payload: Dict[str, Any]
    ) -> Tuple[List[List[float]], int]:
        self.calls += 1
        return [[float(self.calls)]], 0


def test_async_embed_uses_post_embeddings() -> None:
    client = DummyAsyncClient()
    emb = asyncio.run(client.embed("hello"))
    assert emb == [1.0, 2.0, 3.0]
    assert client.calls == [{"model": "test", "input": "hello"}]


def test_sync_adapter_runs_async_client() -> None:
    async_client = DummyAsyncClient()
    client = EmbeddingsClient(async_client=async_client)
    vec = client.embed("world")
    assert vec == [1.0, 2.0, 3.0]
    assert async_client.calls == [{"model": "test", "input": "world"}]


def test_embed_returns_identical_vector_on_repeated_calls() -> None:
    client = CountingAsyncClient()
    first = asyncio.run(client.embed("repeat"))
    second = asyncio.run(client.embed("repeat"))
    assert first == second
    assert client.calls == 1
