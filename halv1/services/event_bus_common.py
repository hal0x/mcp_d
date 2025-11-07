"""Common utilities shared by event bus implementations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class LRUCache:
    """Simple LRU cache for event ids."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def add(self, key: str) -> None:
        self._data[key] = None
        self._data.move_to_end(key)
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def discard(self, key: str) -> None:
        """Remove key if present."""
        self._data.pop(key, None)


def derive_event_id(topic: str, event: Any) -> str:
    """Derive a stable string id for any event-like payload."""
    ev_id = getattr(event, "id", None)
    if isinstance(ev_id, str):
        return ev_id
    try:
        return f"{topic}:{hash(str(event))}"
    except Exception:
        return f"{topic}:{object.__repr__(event)}"


class NoopAwaitable:
    """Trivial awaitable returning None — useful for subscribe APIs."""

    def __await__(self) -> Any:  # pragma: no cover - trivial
        if False:
            yield None
        return None
