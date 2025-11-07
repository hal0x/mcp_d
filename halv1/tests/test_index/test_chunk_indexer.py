import asyncio
import datetime as dt
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import index.chunk_indexer as chunk_indexer
from index.chunk_indexer import build_chunk_meta, index_message_chunks
from index.vector_index import VectorIndex


class CaptureIndex(VectorIndex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured: list[str] = []

    async def _embed(self, text: str):
        self.captured.append(text)
        return [1.0]

    async def _embed_many(self, texts):
        self.captured.extend(texts)
        return [[1.0] for _ in texts]


class DummyIndex:
    def __init__(self):
        self.added_many = []
        self.added_single = []

    async def add_many(self, items):
        self.added_many.append(items)

    async def add(self, chunk_id, text, meta):  # type: ignore[override]
        self.added_single.append((chunk_id, text, meta))


def test_empty_messages_no_indexing() -> None:
    idx = DummyIndex()
    count, newest = asyncio.run(index_message_chunks([], idx, "theme"))
    assert count == 0
    assert newest is None
    assert idx.added_many == []
    assert idx.added_single == []


def _msg(message_id: str, minutes: int, text: str):
    return {
        "chat_id": "chat",
        "message_id": message_id,
        "timestamp": dt.datetime(2024, 1, 1, 12, minutes, 0),
        "author": "user",
        "text": text,
        "reply_to": None,
    }


def test_messages_are_chunked_and_indexed():
    msgs = [_msg("1", 0, "hello"), _msg("2", 1, "world")]
    idx = DummyIndex()
    count, newest = asyncio.run(index_message_chunks(msgs, idx, "theme"))
    assert count == 2
    assert newest == dt.datetime(2024, 1, 1, 12, 0, 0)
    assert len(idx.added_many) == 1
    assert len(idx.added_many[0]) == 1
    chunk_id, text, meta = idx.added_many[0][0]
    assert text == "hello world"
    assert meta["chat"] == "chat"
    assert meta["message_ids"] == "1"
    assert meta["theme"] == "theme"
    assert meta["date"] == dt.datetime(2024, 1, 1, 12, 0, 0).isoformat()


def test_progress_and_cancellation():
    # Third message far enough to form a new chunk
    msgs = [_msg("1", 0, "a"), _msg("2", 1, "b"), _msg("3", 10, "c")]
    idx = DummyIndex()
    progress = []
    cancel_flag = {"val": False}

    def progress_cb(count: int) -> None:
        progress.append(count)
        if count >= 1:
            cancel_flag["val"] = True

    def is_cancelled() -> bool:
        return cancel_flag["val"]

    processed, _ = asyncio.run(
        index_message_chunks(
            msgs, idx, "theme", progress_cb=progress_cb, is_cancelled=is_cancelled
        )
    )

    # Only first chunk (merged message) should be indexed
    assert processed == 1
    assert len(idx.added_single) == 1
    assert progress[-1] == 1


def test_normalization_in_indexing(tmp_path):
    msgs = [
        _msg("1", 0, "<b>Hello</b> 😀"),
        _msg("2", 0, "world"),
    ]
    idx = CaptureIndex(path=tmp_path / "idx.json")
    asyncio.run(index_message_chunks(msgs, idx, "theme"))
    assert idx.captured == ["Hello world"]


def test_short_messages_are_merged_before_chunking(monkeypatch):
    captured = {}

    def fake_chunk_messages(msgs, **kwargs):
        captured["msgs"] = list(msgs)
        return []

    monkeypatch.setattr(chunk_indexer, "chunk_messages", fake_chunk_messages)

    msgs = [_msg("1", 0, "hi"), _msg("2", 0, "there")]
    idx = DummyIndex()
    asyncio.run(index_message_chunks(msgs, idx, "theme"))

    assert len(captured["msgs"]) == 1
    assert captured["msgs"][0]["text"] == "hi there"


def test_build_chunk_meta() -> None:
    chunk = {
        "chat_id": 1,
        "message_ids": [1, 2],
        "end_ts": dt.datetime(2024, 1, 1, 12, 0, 0),
    }
    meta = build_chunk_meta(chunk, "theme")
    assert meta == {
        "chat": "1",
        "message_ids": "1,2",
        "date": dt.datetime(2024, 1, 1, 12, 0, 0).isoformat(),
        "theme": "theme",
    }
