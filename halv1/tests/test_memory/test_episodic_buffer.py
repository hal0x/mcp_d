"""Tests for EpisodicBuffer."""

from datetime import datetime, timedelta, timezone

from pytest import MonkeyPatch

from events.models import Event
from memory.episodic_buffer import EpisodicBuffer


class TestEpisodicBuffer:
    def test_write_and_read(self) -> None:
        buf = EpisodicBuffer()
        ev = Event()
        buf.write(ev)
        assert buf.read() == [ev]

    def test_expired_event_ignored(self) -> None:
        buf = EpisodicBuffer(ttl_days=1)
        old = Event(timestamp=datetime.now(timezone.utc) - timedelta(days=2))
        buf.write(old)
        assert buf.read() == []

    def test_expiration_over_time(self, monkeypatch: MonkeyPatch) -> None:
        buf = EpisodicBuffer(ttl_days=1)
        ev = Event()
        buf.write(ev)

        future = ev.timestamp + timedelta(days=2)
        monkeypatch.setattr(EpisodicBuffer, "_now", staticmethod(lambda: future))
        assert buf.read() == []
