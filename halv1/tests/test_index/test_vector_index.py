import asyncio
import pathlib
import sys
from typing import Any

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from index.vector_index import MIN_EMBED_NORM, VectorEntry, VectorIndex


class EmptyEmbedIndex(VectorIndex):
    async def _embed(self, text: str) -> list[float]:
        return []


async def _search(idx: VectorIndex, query: str, top_k: int = 25) -> list[VectorEntry]:
    return await idx.search(query, top_k=top_k)


def test_search_returns_empty_when_query_embedding_empty(
    tmp_path: pathlib.Path,
) -> None:
    idx = EmptyEmbedIndex(path=str(tmp_path / "idx.json"))
    idx.entries = [VectorEntry("1", "text", [1.0, 0.0], {})]
    assert asyncio.run(_search(idx, "query")) == []


class ZeroQueryIndex(VectorIndex):
    async def _embed(self, text: str) -> list[float]:
        return [0.0, 0.0]


def test_search_returns_empty_when_query_norm_zero(tmp_path: pathlib.Path) -> None:
    idx = ZeroQueryIndex(path=str(tmp_path / "idx.json"))
    idx.entries = [VectorEntry("1", "text", [1.0, 0.0], {})]
    assert asyncio.run(_search(idx, "query")) == []


class NormalQueryIndex(VectorIndex):
    async def _embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FixedEmbedIndex(VectorIndex):
    async def _embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def _embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class CaptureIndex(VectorIndex):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.captured: list[str] = []

    async def _embed(self, text: str) -> list[float]:
        self.captured.append(text)
        return [1.0, 0.0]

    async def _embed_many(self, texts: list[str]) -> list[list[float]]:
        self.captured.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def test_search_ignores_zero_norm_entries(tmp_path: pathlib.Path) -> None:
    idx = NormalQueryIndex(path=str(tmp_path / "idx.json"))
    idx.entries = [
        VectorEntry("1", "bad", [0.0, 0.0], {}),
        VectorEntry("2", "good", [1.0, 0.0], {}),
    ]
    results = asyncio.run(_search(idx, "query", top_k=5))
    assert [e.chunk_id for e in results] == ["2"]


def test_search_returns_empty_when_all_entry_norm_zero(
    tmp_path: pathlib.Path,
) -> None:
    idx = NormalQueryIndex(path=str(tmp_path / "idx.json"))
    idx.entries = [VectorEntry("1", "text", [0.0, 0.0], {})]
    assert asyncio.run(_search(idx, "query")) == []


def test_add_updates_existing_entry(tmp_path: pathlib.Path) -> None:
    idx = FixedEmbedIndex(path=str(tmp_path / "idx.json"))
    asyncio.run(idx.add("1", "a", {}))
    asyncio.run(idx.add("1", "b", {}))
    assert len(idx.entries) == 1
    assert idx._id_map["1"].text == "b"


def test_add_many_handles_duplicates(tmp_path: pathlib.Path) -> None:
    idx = FixedEmbedIndex(path=str(tmp_path / "idx.json"))
    asyncio.run(idx.add_many([("1", "a", {}), ("2", "b", {})]))
    asyncio.run(idx.add_many([("1", "c", {}), ("3", "d", {})]))
    assert len(idx.entries) == 3
    assert idx._id_map["1"].text == "c"


def test_add_normalizes_text_before_embedding(tmp_path: pathlib.Path) -> None:
    idx = CaptureIndex(path=str(tmp_path / "idx.json"))
    asyncio.run(idx.add("1", "<b>Hello</b> 😀 [source: tg]", {}))
    assert idx.captured == ["Hello"]


def test_add_many_normalizes_text_before_embedding(
    tmp_path: pathlib.Path,
) -> None:
    idx = CaptureIndex(path=str(tmp_path / "idx.json"))
    asyncio.run(
        idx.add_many(
            [
                ("1", "<i>a</i> 😊", {}),
                ("2", "[source: x]<b>b</b>", {}),
            ]
        )
    )
    assert idx.captured == ["a", "b"]


class SmallNormIndex(VectorIndex):
    async def _embed(self, text: str) -> list[float]:
        return [MIN_EMBED_NORM / 10, 0.0]

    async def _embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[MIN_EMBED_NORM / 10, 0.0] for _ in texts]


def test_add_skips_low_norm_embedding(tmp_path: pathlib.Path) -> None:
    idx = SmallNormIndex(path=str(tmp_path / "idx.json"))
    asyncio.run(idx.add("1", "text", {}))
    assert idx.entries == []


def test_add_many_skips_low_norm_embedding(tmp_path: pathlib.Path) -> None:
    idx = SmallNormIndex(path=str(tmp_path / "idx.json"))
    asyncio.run(idx.add_many([("1", "a", {}), ("2", "b", {})]))
    assert idx.entries == []


class WeightedIndex(VectorIndex):
    async def _embed(self, text: str) -> list[float]:
        return [0.0, 1.0]


def test_search_uses_weights(tmp_path: pathlib.Path) -> None:
    idx = WeightedIndex(path=str(tmp_path / "idx.json"))
    idx.entries = [
        VectorEntry("1", "a", [1.0, 0.0], {}),
        VectorEntry("2", "b", [1.0, 0.0], {}),
    ]
    idx.weights["1"] = 1
    idx.weights["2"] = 10
    results = asyncio.run(_search(idx, "q", top_k=2))
    assert [e.chunk_id for e in results] == ["2", "1"]
