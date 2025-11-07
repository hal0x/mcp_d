"""Create chronological summaries from cluster timelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Iterable, List, Tuple

from .cluster_manager import ClusterManager
from .summarizer import Summarizer


@dataclass
class TimeSlice:
    """Collected texts for a time window and its summary."""

    start: datetime
    texts: List[str] = field(default_factory=list)
    summary: str = ""


class Chronicle:
    """Build a simple history from cluster medoid timelines."""

    def __init__(self, manager: ClusterManager, summarizer: Summarizer) -> None:
        self.manager = manager
        self.summarizer = summarizer

    # ------------------------------------------------------------------
    def _collect_events(self) -> List[Tuple[float, str]]:
        """Return ``(timestamp, text)`` pairs from cluster timelines."""

        events: List[Tuple[float, str]] = []
        for cluster in self.manager.clusters.values():
            summary = cluster.summary or ""
            for entry in getattr(cluster, "timeline", []):
                events.append((entry.timestamp, entry.text))
                if summary:
                    events.append((entry.timestamp, summary))
        events.sort(key=lambda x: x[0])
        return events

    # ------------------------------------------------------------------
    def build(self, slice_hours: int = 24) -> List[TimeSlice]:
        """Group events into ``slice_hours`` windows and summarise them."""

        events = self._collect_events()
        if not events:
            return []
        
        # Validate slice_hours
        if slice_hours <= 0:
            slice_hours = 24  # Use default

        slice_seconds = slice_hours * 3600
        slices: List[TimeSlice] = []

        bucket = (events[0][0] // slice_seconds) * slice_seconds
        current = TimeSlice(start=datetime.fromtimestamp(bucket, UTC))
        for ts, text in events:
            b = (ts // slice_seconds) * slice_seconds
            if b != bucket:
                current.summary = self.summarizer.summarize(current.texts)
                slices.append(current)
                bucket = b
                current = TimeSlice(start=datetime.fromtimestamp(bucket, UTC))
            current.texts.append(text)

        if current.texts:
            current.summary = self.summarizer.summarize(current.texts)
            slices.append(current)

        return slices

    # ------------------------------------------------------------------
    def render(self, slices: Iterable[TimeSlice]) -> str:
        """Render ``slices`` into a plain text chronicle."""

        lines = []
        for sl in slices:
            ts = sl.start.isoformat()
            lines.append(f"{ts}: {sl.summary}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def chronicle(self, slice_hours: int = 24) -> str:
        """Convenience wrapper: build and render in one step."""

        return self.render(self.build(slice_hours))
