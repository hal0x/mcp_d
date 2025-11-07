"""Utilities for clustering recent episodic events into schemas.

The module provides a minimal implementation that groups episodes with the
same ``text`` within a configurable time window (default: 7 days). Each
cluster produces a :class:`Schema` holding a ``schema_summary`` and the
associated episodes. A mapping from episode IDs to their schema enables
bidirectional lookups between schemas and episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Sequence, cast

from .episode_graph import EpisodeGraph


@dataclass(frozen=True)
class Episode:
    """Lightweight representation of an episodic event."""

    id: str
    text: str
    timestamp: datetime


@dataclass
class Schema:
    """Collection of related episodes with a shared summary."""

    schema_summary: str
    episodes: List[Episode]


@dataclass
class ConsolidationResult:
    """Result of clustering episodes into schemas.

    Attributes
    ----------
    schemas:
        List of resulting schemas.
    episode_to_schema:
        Mapping from episode ID to the schema it belongs to.
    """

    schemas: List[Schema]
    episode_to_schema: Dict[str, Schema]


# ---------------------------------------------------------------------------
def consolidate(
    episodes: Iterable[Episode],
    *,
    now: datetime | None = None,
    window_days: int = 7,
    embed: Callable[[str], Sequence[float]] | None = None,
    threshold: float = 0.8,
    summarize: Callable[[List[str]], str] | None = None,
) -> ConsolidationResult:
    """Cluster episodes occurring within ``window_days`` days.

    Parameters
    ----------
    episodes:
        Iterable of :class:`Episode` objects to consolidate.
    now:
        Reference time used to compute the cutoff window. Defaults to UTC now.
    window_days:
        Size of the time window in days. Episodes older than this are ignored.
    embed:
        Optional callable returning an embedding vector for a given text.
        When provided, cosine similarity between episode embeddings is used
        for clustering. When ``None`` episodes with identical text are
        grouped together (legacy behaviour).
    threshold:
        Cosine similarity threshold for assigning an episode to an existing
        cluster. Only used when ``embed`` is provided.
    summarize:
        Optional callable producing a summary for a cluster given a list of
        episode texts. If ``None`` a simple keyword-based summary is used.

    Returns
    -------
    ConsolidationResult
        Object containing schemas and episode-to-schema mapping.
    """

    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    recent = [ep for ep in episodes if ep.timestamp >= cutoff]

    if not recent:
        return ConsolidationResult(schemas=[], episode_to_schema={})

    def _default_summary(texts: List[str]) -> str:
        from collections import Counter

        words = Counter()
        for t in texts:
            words.update(t.lower().split())
        return " ".join([w for w, _ in words.most_common(3)]) if words else ""

    summarize = summarize or _default_summary

    # ------------------------------------------------------------------
    if embed is None:
        # Fallback to grouping by identical text for backwards compatibility.
        groups: Dict[str, List[Episode]] = {}
        for ep in recent:
            key = ep.text.strip().lower()
            groups.setdefault(key, []).append(ep)

        schemas: List[Schema] = []
        episode_map: Dict[str, Schema] = {}
        for key in sorted(groups):
            eps = sorted(groups[key], key=lambda e: e.timestamp)
            summary = summarize([e.text for e in eps]) or eps[0].text
            schema = Schema(schema_summary=summary, episodes=eps)
            schemas.append(schema)
            for ep in eps:
                episode_map[ep.id] = schema

        return ConsolidationResult(schemas=schemas, episode_to_schema=episode_map)

    # ------------------------------------------------------------------
    import numpy as np

    clusters: List[Dict[str, object]] = []
    for ep in recent:
        vec = np.array(embed(ep.text), dtype=float)
        if np.linalg.norm(vec) == 0:
            # Treat zero vectors as unique clusters
            clusters.append({"centroid": vec, "episodes": [ep]})
            continue

        best_idx = -1
        best_sim = -1.0
        for idx, c in enumerate(clusters):
            centroid = c["centroid"]
            denom = np.linalg.norm(centroid) * np.linalg.norm(vec)
            sim = float(np.dot(centroid, vec) / denom) if denom else 0.0
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx >= 0 and best_sim >= threshold:
            cl = clusters[best_idx]
            cl_eps = cast(List[Episode], cl["episodes"])
            cl_eps.append(ep)
            centroid = cast(np.ndarray, cl["centroid"])
            centroid = centroid + (vec - centroid) / len(cl_eps)
            cl["centroid"] = centroid
        else:
            clusters.append({"centroid": vec, "episodes": [ep]})

    schemas: List[Schema] = []
    episode_map: Dict[str, Schema] = {}
    for cl in clusters:
        eps = sorted(cast(List[Episode], cl["episodes"]), key=lambda e: e.timestamp)
        summary = summarize([e.text for e in eps]) or eps[0].text
        schema = Schema(schema_summary=summary, episodes=eps)
        schemas.append(schema)
        for ep in eps:
            episode_map[ep.id] = schema

    return ConsolidationResult(schemas=schemas, episode_to_schema=episode_map)


# ---------------------------------------------------------------------------
def consolidate_graph(
    graph: EpisodeGraph,
    *,
    now: datetime | None = None,
    window_days: int = 7,
) -> None:
    """Cluster recent events from ``graph`` and update schema assignments.

    The function fetches events from the ``items`` table, groups them using
    :func:`consolidate` and writes resulting schemas to the ``schemas`` table.
    Each processed event receives a ``schema_id`` pointing to its assigned
    schema.

    Parameters
    ----------
    graph:
        Episode graph containing events in the ``items`` table.
    now:
        Reference time used for consolidation. Defaults to ``datetime.now``.
    window_days:
        Size of the time window passed to :func:`consolidate`.
    """

    cur = graph.conn.cursor()
    cur.execute(
        "SELECT id, content, timestamp FROM items WHERE timestamp IS NOT NULL",
    )
    episodes = [
        Episode(
            id=str(row[0]),
            text=str(row[1]),
            timestamp=datetime.fromtimestamp(float(row[2]), timezone.utc),
        )
        for row in cur.fetchall()
    ]

    result = consolidate(episodes, now=now, window_days=window_days)

    for schema in result.schemas:
        cur.execute("SELECT id FROM schemas WHERE name = ?", (schema.schema_summary,))
        row = cur.fetchone()
        if row:
            schema_id = int(row[0])
        else:
            cur.execute(
                "INSERT INTO schemas (name, definition) VALUES (?, ?)",
                (schema.schema_summary, schema.schema_summary),
            )
            schema_id = cast(int, cur.lastrowid)

        for ep in schema.episodes:
            cur.execute(
                "UPDATE items SET schema_id = ? WHERE id = ?",
                (schema_id, int(ep.id)),
            )

    graph.conn.commit()
