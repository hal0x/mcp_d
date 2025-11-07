# index/chunking.py
"""Utilities for grouping Telegram messages into time‑based chunks.

The implementation is intentionally lightweight – it focuses on the
behaviour required by the tests and the documentation in the prompt.  It
does **not** aim to be a fully fledged production‑ready chunker, but it
provides all public hooks that other modules (e.g. :mod:`index.vector_index`)
can import.

The module exposes a single function :func:`chunk_messages`.  The
behaviour is deterministic and fully covered by the test‑suite in
``tests/test_chunking.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Pattern
from urllib.parse import urlparse


@dataclass
class ChunkState:
    """Mutable state accumulated while building a chunk."""

    chat_id: Any
    message_ids: list[Any]
    start_ts: Any
    end_ts: Any
    authors: set[str]
    raw_texts: list[str]
    unique_domains: set[str]


def start_chunk(
    msg: Dict[str, Any],
    text: str,
    extract_domains: Callable[[str], set[str]],
) -> ChunkState:
    """Create :class:`ChunkState` seeded with ``msg`` and ``text``."""
    author = msg.get("author") or ""
    ts = msg["timestamp"]
    return ChunkState(
        chat_id=msg["chat_id"],
        message_ids=[msg["message_id"]],
        start_ts=ts,
        end_ts=ts,
        authors={author},
        raw_texts=[text],
        unique_domains=extract_domains(text),
    )


def finalize_chunk(
    state: ChunkState,
    token_counter: Callable[[str], int],
    url_pattern: Pattern[str],
) -> Dict[str, Any]:
    """Compute derived fields from ``state`` and return a chunk dict."""
    texts = [re.sub(r"\n+", "\n", t.strip()) for t in state.raw_texts]
    merged_text = " ".join(texts).strip()

    token_count = sum(token_counter(t) for t in state.raw_texts)

    has_code = any("```" in t or len(t.splitlines()) >= 30 for t in state.raw_texts)
    has_links = (
        any(url_pattern.search(t) for t in state.raw_texts)
        and len(state.unique_domains) > 1
    )

    meta = {
        "merged_count": len(state.message_ids) - 1,
        "has_code": has_code,
        "has_links": has_links,
    }

    return {
        "chat_id": state.chat_id,
        "message_ids": list(state.message_ids),
        "start_ts": state.start_ts,
        "end_ts": state.end_ts,
        "authors": list(state.authors),
        "text": merged_text,
        "size_tokens": token_count,
        "meta": meta,
    }


def should_split(
    state: ChunkState,
    msg: Dict[str, Any],
    *,
    text: str,
    token_counter: Callable[[str], int],
    max_chunk_tokens: int,
    same_author_only: bool,
    stop_patterns: List[Pattern[str]],
    allow_reply_gap: bool,
    max_gap_minutes: int,
    reply_gap_minutes: int,
) -> bool:
    """Return ``True`` if ``msg`` should start a new chunk."""

    if any(p.search(text) for p in stop_patterns):
        return True

    author = msg.get("author") or ""
    if same_author_only and author not in state.authors:
        return True

    current_tokens = sum(token_counter(t) for t in state.raw_texts)
    if current_tokens + token_counter(text) > max_chunk_tokens:
        return True

    prev_ts = state.end_ts
    ts = msg["timestamp"]
    gap = (ts - prev_ts).total_seconds() / 60
    limit = reply_gap_minutes if allow_reply_gap else max_gap_minutes
    if gap > limit:
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_messages(
    messages: Iterable[Dict[str, Any]],
    *,
    max_gap_minutes: int = 10,
    reply_gap_minutes: int = 120,  # 2 часа для ответов на сообщения
    max_chunk_tokens: int = 4000,  # Увеличено с 200 до 4000 для лучшего использования контекста эмбеддингов
    same_author_only: bool = False,
    stop_patterns: List[Pattern[str]] | None = None,
    token_counter: Callable[[str], int] = lambda s: max(1, len(s) // 4),
) -> List[Dict[str, Any]]:
    """Group consecutive Telegram messages into *chunks*.

    Parameters
    ----------
    messages:
        Iterable of message dicts.  The function does not modify the input
        and accepts any iterable – a list or generator are both fine.
    max_gap_minutes:
        Maximum time difference between two adjacent messages that still
        allows them to belong to the same chunk.
    reply_gap_minutes:
        If ``reply_to`` points to a message inside the current chunk, this
        larger gap is tolerated.
    max_chunk_tokens:
        Rough token budget for a single chunk.  When adding a new message
        would exceed this limit, the current chunk is closed and a new one
        starts.
    same_author_only:
        If ``True`` only messages from the *same* author may be merged.
    stop_patterns:
        List of compiled regular expressions that, when matched against a
        message text, force a new chunk to start **before** that message.
    token_counter:
        Callable that estimates the number of tokens in a string.  The
        default is a very rough ``len(s)//4`` approximation.

    Returns
    -------
    list[dict]
        Each dict contains the aggregated data for a chunk, ready to be fed
        into an embedding pipeline.
    """

    # Default stop patterns – these are case‑insensitive and anchored at
    # the beginning of the string.
    if stop_patterns is None:
        default = [
            r"^#новая тема",  # Russian "new topic"
            r"^\[ANN\]",  # Announcement marker
            r"^Summary:",
        ]
        stop_patterns = [re.compile(p, re.IGNORECASE) for p in default]

    # Ensure deterministic ordering – the spec says input is already sorted,
    # but we sort just to be safe.
    sorted_msgs = sorted(messages, key=lambda m: m["timestamp"])

    url_pattern = re.compile(r"https?://[^\s]+", re.IGNORECASE)

    def extract_domains(text: str) -> set[str]:
        return {
            urlparse(match.group()).netloc.lower()
            for match in url_pattern.finditer(text)
        }

    chunks: List[Dict[str, Any]] = []
    current: ChunkState | None = None

    # Track seen message IDs **per chat** to avoid cross‑chat collisions.
    # ``message_id`` is only unique within a chat, therefore the key is a
    # tuple ``(chat_id, message_id)``.
    seen_ids: set[tuple[Any, Any]] = set()

    def _is_duplicate(message: Dict[str, Any]) -> bool:
        """Return ``True`` if ``message`` was seen before."""
        key = (message["chat_id"], message["message_id"])
        if key in seen_ids:
            return True
        seen_ids.add(key)
        return False

    def _finalise(chunk: ChunkState) -> None:
        chunks.append(finalize_chunk(chunk, token_counter, url_pattern))

    for msg in sorted_msgs:
        # Skip duplicates globally – keep only the first occurrence within a
        # chat.  IDs may repeat across different chats, so we namespace by
        # ``chat_id``.
        if _is_duplicate(msg):
            continue

        # Prepare message data
        ts = msg["timestamp"]
        author = msg.get("author") or ""
        text = msg.get("text", "") or ""
        reply_to = msg.get("reply_to")

        # If we have no current chunk, start one.
        if not current:
            current = start_chunk(msg, text, extract_domains)
            continue

        # Check if we are switching chats – always start a new chunk.
        if msg["chat_id"] != current.chat_id:
            _finalise(current)
            current = start_chunk(msg, text, extract_domains)
            continue

        allow_reply_gap = bool(reply_to and reply_to in current.message_ids)

        if should_split(
            current,
            msg,
            text=text,
            token_counter=token_counter,
            max_chunk_tokens=max_chunk_tokens,
            same_author_only=same_author_only,
            stop_patterns=stop_patterns,
            allow_reply_gap=allow_reply_gap,
            max_gap_minutes=max_gap_minutes,
            reply_gap_minutes=reply_gap_minutes,
        ):
            _finalise(current)
            current = start_chunk(msg, text, extract_domains)
            continue

        # Merge into current chunk.
        current.message_ids.append(msg["message_id"])
        current.end_ts = ts
        current.authors.add(author)
        current.raw_texts.append(text)
        current.unique_domains.update(extract_domains(text))

    # Finalise any remaining chunk.
    if current:
        _finalise(current)

    return chunks


# End of file
