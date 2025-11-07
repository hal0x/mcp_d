# index/preprocess.py
"""Text normalisation and message preprocessing helpers.

This module exposes a small set of utilities that are used during the
ingestion pipeline before texts are embedded.  The focus is on lightweight
transformations that make the downstream vector index more robust.
"""
from __future__ import annotations

import html
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List

_EMOJI_RE = re.compile(
    "[\U0001f1e0-\U0001f6ff\U0001f300-\U0001f5ff\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff\U00002700-\U000027bf]",
    flags=re.UNICODE,
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")

_TAG_RE = re.compile(r"\[(?:source|date):[^\]]*\]", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Return a cleaned up version of ``text``.

    The function performs a couple of inexpensive normalisation steps:

    * HTML tags are stripped.
    * emoji characters are removed.
    * ``[date:...]`` and ``[source:...]`` style tags are stripped.
    * consecutive whitespace is collapsed to a single space.
    """

    if not text:
        return ""
    # Unescape any HTML entities first
    text = html.unescape(text)
    # Remove tags such as ``<b>`` or ``<a href=...>``
    text = _HTML_TAG_RE.sub("", text)
    # Remove simple [date:...] or [source:...] tags
    text = _TAG_RE.sub("", text)
    # Strip emoji characters
    text = _EMOJI_RE.sub("", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def merge_short_messages(
    messages: Iterable[Dict[str, Any]],
    *,
    delta: timedelta = timedelta(minutes=1),
    max_length: int = 50,
) -> List[Dict[str, Any]]:
    r"""Merge consecutive *short* messages within a sliding time window.

    Parameters
    ----------
    messages:
        Iterable of message dictionaries.  Each item must provide at least
        ``timestamp`` and ``text`` fields.  The function does **not** mutate the
        input.
    delta:
        Half of the time window size.  Messages whose timestamps fall within
        ``t \pm delta`` are candidates for merging.
    max_length:
        Maximum ``len(text)`` for a message to be considered *short*.

    Returns
    -------
    list[dict]
        New message dictionaries.  When several short messages fall into the
        same window they are merged by concatenating the text with a space and
        using the earliest timestamp.
    """

    msgs = sorted(messages, key=lambda m: m["timestamp"])
    result: List[Dict[str, Any]] = []
    buffer: List[Dict[str, Any]] = []
    window_end: datetime | None = None

    def _flush() -> None:
        nonlocal buffer, window_end
        if not buffer:
            return
        first = buffer[0]
        merged_text = " ".join(m.get("text", "") for m in buffer).strip()
        merged = dict(first)
        merged["text"] = merged_text
        result.append(merged)
        buffer = []
        window_end = None

    for msg in msgs:
        text = msg.get("text", "")
        ts: datetime = msg["timestamp"]
        if len(text) > max_length:
            _flush()
            result.append(dict(msg))
            continue
        if not buffer:
            buffer = [dict(msg)]
            window_end = ts + delta
            continue
        assert window_end is not None
        if ts <= window_end:
            buffer.append(dict(msg))
            window_end = ts + delta
        else:
            _flush()
            buffer = [dict(msg)]
            window_end = ts + delta
    _flush()
    return result


__all__ = ["normalize_text", "merge_short_messages"]
