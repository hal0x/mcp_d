"""Helper to index messages in chunked form."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

import numpy as np

from .chunking import chunk_messages
from .preprocess import merge_short_messages, normalize_text
from .vector_index import VectorIndex


def build_chunk_meta(chunk: Dict[str, Any], theme: str) -> Dict[str, str]:
    """Return metadata dictionary for a chunk."""

    return {
        "chat": str(chunk["chat_id"]),
        "message_ids": ",".join(str(mid) for mid in chunk["message_ids"]),
        "date": chunk["end_ts"].isoformat(),
        "theme": theme,
    }


def _prepare_chunks(
    messages: Iterable[Dict[str, Any]],
    max_chunk_tokens: int = 4000,  # Используем новое значение по умолчанию
) -> Tuple[list[Dict[str, Any]], int]:
    msgs = list(messages)
    if not msgs:
        return [], 0
    original_count = len(msgs)
    msgs = merge_short_messages(msgs)
    chunks = chunk_messages(msgs, max_chunk_tokens=max_chunk_tokens)
    return chunks, original_count


async def _index_fast(
    chunks: Sequence[Dict[str, Any]], index: VectorIndex, theme: str
) -> datetime | None:
    newest: datetime | None = None
    items: list[tuple[str, str, dict[str, str]]] = []
    texts: list[str] = []
    for chunk in chunks:
        text = chunk["text"]
        if not text.strip():
            continue
        chunk_id = "_".join(str(mid) for mid in chunk["message_ids"])
        items.append((chunk_id, text, build_chunk_meta(chunk, theme)))
        texts.append(normalize_text(text))
        if newest is None or chunk["end_ts"] > newest:
            newest = chunk["end_ts"]
    if items:
        if hasattr(index, "_embed_many"):
            embeddings = await index._embed_many(texts)
        else:
            embeddings = [[1.0] for _ in texts]
        pairs = [
            (item, emb)
            for item, emb in zip(items, embeddings)
            if emb and np.linalg.norm(emb) > 0
        ]
        if pairs:
            filtered_items = [it for it, _ in pairs]
            filtered_embs = [emb for _, emb in pairs]
            try:
                await index.add_many(filtered_items, embeddings=filtered_embs)
            except TypeError:
                await index.add_many(filtered_items)
    return newest


async def _index_slow(
    chunks: Sequence[Dict[str, Any]],
    index: VectorIndex,
    theme: str,
    progress_cb: Callable[[int], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> Tuple[int, datetime | None]:
    processed = 0
    newest: datetime | None = None
    for chunk in chunks:
        if is_cancelled and is_cancelled():
            break
        chunk_id = "_".join(str(mid) for mid in chunk["message_ids"])
        await index.add(chunk_id, chunk["text"], build_chunk_meta(chunk, theme))
        processed += len(chunk["message_ids"])
        if progress_cb:
            try:
                progress_cb(processed)
            except Exception:
                pass
        if newest is None or chunk["end_ts"] > newest:
            newest = chunk["end_ts"]
    return processed, newest


async def index_message_chunks(
    messages: Iterable[Dict[str, Any]],
    index: VectorIndex,
    theme: str,
    progress_cb: Callable[[int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    max_chunk_tokens: int = 4000,  # Используем новое значение по умолчанию
) -> Tuple[int, datetime | None]:
    """Chunk ``messages`` and add resulting documents to ``index``.

    Parameters
    ----------
    messages:
        Iterable of message dicts to be chunked and indexed.
    index:
        Target :class:`VectorIndex` implementation.
    theme:
        Name of the active theme for metadata tagging.
    progress_cb:
        Optional callback receiving the total number of processed messages.
        Used to report progress back to the caller.
    is_cancelled:
        Optional callable returning ``True`` if the operation should abort.

    Returns
    -------
    Tuple[int, datetime | None]
        ``(message_count, newest_ts)`` where ``newest_ts`` is the timestamp of
        the last message in the newest produced chunk or ``None`` if no chunks
        were added.
    """

    chunks, original_count = _prepare_chunks(messages, max_chunk_tokens=max_chunk_tokens)
    if not chunks:
        return 0, None

    # Fast path when no progress tracking or cancellation is required.
    if not progress_cb and not is_cancelled:
        newest = await _index_fast(chunks, index, theme)
        return original_count, newest

    processed, newest = await _index_slow(
        chunks, index, theme, progress_cb=progress_cb, is_cancelled=is_cancelled
    )
    return processed, newest
