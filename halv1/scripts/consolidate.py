"""Utilities for consolidating event data.

This module clusters events and computes an eviction score for each one.
Events with a low score are moved to an archive (if provided) or removed.
Events marked as ``frozen`` are always kept and skipped during eviction
scoring.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, cast

import numpy as np
import numpy.typing as npt


@dataclass
class Event:
    """Simple event record used for consolidation.

    Attributes
    ----------
    id:
        Unique identifier of the event.
    timestamp:
        Time when the event occurred.
    value:
        Numeric value associated with the event used for scoring.
    frozen:
        If ``True`` the event will never be evicted during consolidation.
    """

    id: str
    timestamp: datetime
    value: float
    frozen: bool = False


def _kmeans(
    points: npt.NDArray[np.float64], k: int, max_iter: int = 100
) -> npt.NDArray[np.int64]:
    """Run a very small K-means clustering on ``points``.

    Parameters
    ----------
    points:
        Array of shape (n, m) representing ``n`` samples and ``m`` features.
    k:
        Number of clusters.
    max_iter:
        Maximum number of refinement iterations.

    Returns
    -------
    ndarray
        Cluster label for each point.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    rng = np.random.default_rng(0)
    centroids = points[rng.choice(len(points), k, replace=False)]
    labels = np.zeros(len(points), dtype=int)
    for _ in range(max_iter):
        # assign points to nearest centroid
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # recompute centroids
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centroids[i] = points[mask].mean(axis=0)
    return labels


def cluster_events(events: Iterable[Event], k: int) -> List[List[Event]]:
    """Cluster ``events`` into ``k`` groups.

    Clustering is performed using a simple K-means implementation on the
    two-dimensional feature space ``[age_in_days, value]`` where the age is
    measured relative to ``now``.
    """

    events = list(events)
    if not events:
        return []
    now = datetime.now(timezone.utc)
    features = np.array(
        [
            [
                (now - e.timestamp).total_seconds() / 86400.0,
                e.value,
            ]
            for e in events
        ],
        dtype=float,
    )
    labels = _kmeans(features, k)
    clusters: List[List[Event]] = [[] for _ in range(k)]
    for label, event in zip(labels, events):
        clusters[label].append(event)
    return clusters


def compute_evict_score(event: Event, *, now: Optional[datetime] = None) -> float:
    """Return eviction score for ``event``.

    The score is the event ``value`` decayed by its age in days.  Higher values
    indicate events that should be retained.
    """

    if now is None:
        now = datetime.now(timezone.utc)
    age_days = (now - event.timestamp).total_seconds() / 86400.0
    return event.value / (1.0 + age_days)


def consolidate(
    events: Iterable[Event],
    *,
    k: int,
    threshold: float,
    archive: Optional[List[Event]] = None,
    now: Optional[datetime] = None,
) -> List[Event]:
    """Consolidate ``events`` clustering them and archiving low-score items.

    Parameters
    ----------
    events:
        Iterable of events to process.
    k:
        Number of clusters to form for the incoming events.
    threshold:
        Events with an eviction score below this value are archived or
        discarded.
    archive:
        Optional list that will collect archived events.  If ``None`` the
        low-scoring events are simply dropped.
    now:
        Reference time used to compute age.  Defaults to ``datetime.now``.

    Returns
    -------
    list[Event]
        Events that remain after consolidation.
    """

    events = list(events)
    if not events:
        return []
    if now is None:
        now = datetime.now(timezone.utc)
    clusters = cluster_events(events, k)
    kept: List[Event] = []
    for cluster in clusters:
        for event in cluster:
            if event.frozen:
                kept.append(event)
                continue
            score = compute_evict_score(event, now=now)
            if score < threshold:
                if archive is not None:
                    archive.append(event)
            else:
                kept.append(event)
    return kept


def _parse_event(data: dict[str, object]) -> Event:
    """Convert a mapping loaded from JSON into an :class:`Event`.

    Parameters
    ----------
    data:
        Mapping containing the keys ``id``, ``timestamp``, ``value`` and
        optionally ``frozen``.
    """

    return Event(
        id=cast(str, data["id"]),
        timestamp=datetime.fromisoformat(cast(str, data["timestamp"])),
        value=float(cast(float | int, data["value"])),
        frozen=bool(data.get("frozen", False)),
    )


def main() -> None:
    """Command-line interface for event consolidation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="do not write output")
    parser.add_argument("--k", type=int, required=True, help="number of clusters")
    parser.add_argument(
        "--threshold", type=float, required=True, help="eviction score threshold"
    )
    parser.add_argument("--input", type=Path, required=True, help="input JSON file")
    parser.add_argument("--output", type=Path, required=True, help="output JSON file")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as fh:
        raw: list[dict[str, object]] = json.load(fh)
    events = [_parse_event(item) for item in raw]

    archive: List[Event] = []
    kept = consolidate(events, k=args.k, threshold=args.threshold, archive=archive)

    if not args.dry_run:
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(
                [
                    {
                        "id": e.id,
                        "timestamp": e.timestamp.isoformat(),
                        "value": e.value,
                        "frozen": e.frozen,
                    }
                    for e in kept
                ],
                fh,
            )


if __name__ == "__main__":
    main()
