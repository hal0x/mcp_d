from datetime import datetime, timedelta, timezone

from scripts.consolidate import Event, consolidate


def make_events(now: datetime) -> list[Event]:
    """Create a week of events with varying values and a frozen one."""

    base = now - timedelta(days=6)
    events: list[Event] = []
    for i in range(7):
        ts = base + timedelta(days=i)
        # first three events have low value and one is frozen
        value = 1.0 if i < 3 else 10.0
        frozen = i == 1
        events.append(Event(id=str(i), timestamp=ts, value=value, frozen=frozen))
    return events


def test_consolidate_moves_low_scores_to_archive() -> None:
    now = datetime.now(timezone.utc)
    events = make_events(now)
    archive: list[Event] = []
    kept = consolidate(events, k=2, threshold=5.0, archive=archive, now=now)

    assert {e.id for e in archive} == {"0", "2", "3", "4"}
    assert {e.id for e in kept} == {"1", "5", "6"}


def test_consolidate_drops_low_scores_without_archive() -> None:
    now = datetime.now(timezone.utc)
    events = make_events(now)
    kept = consolidate(events, k=2, threshold=5.0, archive=None, now=now)

    assert {e.id for e in kept} == {"1", "5", "6"}
