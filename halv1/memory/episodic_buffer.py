"""Transient storage for recent events with automatic expiration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, List

from events.models import Event


@dataclass
class _Config:
    ttl: timedelta
    max_size: int


class EpisodicBuffer:
    """Keep recent events for a limited time period.

    Parameters
    ----------
    ttl_days:
        Number of days to keep events. Must be between 1 and 7.
    max_size:
        Maximum number of events to keep in memory. When exceeded, the
        buffer is flushed to the provided ``flush_callback``.
    flush_callback:
        Optional callable receiving the list of events when the buffer is
        flushed. After flushing the buffer is cleared.
    """

    def __init__(
        self,
        *,
        ttl_days: int = 7,
        max_size: int = 1000,
        flush_callback: Callable[[List[Event]], None] | None = None,
    ) -> None:
        if not 1 <= ttl_days <= 7:
            raise ValueError("ttl_days must be between 1 and 7")
        self._cfg = _Config(ttl=timedelta(days=ttl_days), max_size=max_size)
        self._events: List[Event] = []
        self._flush_callback = flush_callback

    # ------------------------------------------------------------------
    def write(self, event: Event) -> None:
        """Store ``event`` in the buffer if it is within TTL."""

        self._expire()
        cutoff = self._now() - self._cfg.ttl
        if event.timestamp < cutoff:
            return
        self._events.append(event)
        if len(self._events) > self._cfg.max_size:
            self.flush()

    # ------------------------------------------------------------------
    def read(self) -> List[Event]:
        """Return a list of non-expired events."""

        self._expire()
        return list(self._events)

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Flush current events to L1 storage if configured."""

        if not self._events:
            return
        if self._flush_callback:
            self._flush_callback(list(self._events))
        self._events.clear()

    # ------------------------------------------------------------------
    def _expire(self) -> None:
        """Remove events older than the TTL."""

        cutoff = self._now() - self._cfg.ttl
        self._events = [e for e in self._events if e.timestamp >= cutoff]

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
