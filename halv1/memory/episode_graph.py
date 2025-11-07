"""Graph structure for episodic memory stored in SQLite.

Nodes are stored in the ``items`` table from :mod:`memory.models` and
connections are tracked via the ``edges`` table.  The schema is created
through the migrations already bundled with the project.  Each inserted
node automatically updates the FTS5 and HNSW indexes via triggers defined
in the migration scripts.
"""

from __future__ import annotations

import json
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, cast

from .models import MemoryItem, init_db


@dataclass
class Node:
    """Simple representation of a graph node."""

    id: int
    content: str


class EpisodeGraph:
    """Persisted graph of episodic events and their relations."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self.conn = init_db(self._path)
        self._last_event_id: Optional[int] = None

    # ------------------------------------------------------------------
    def add_event(
        self, item: MemoryItem, *, related: Iterable[int] | None = None
    ) -> int:
        """Insert ``item`` as a new event node and optional edges.

        All metadata on :class:`~memory.models.MemoryItem` (timestamp,
        entities, importance, recall score, etc.) is written to the ``items``
        table.  ``related`` allows attaching semantic edges to existing nodes.
        """
        cur = self.conn.cursor()

        # Reset the temporal pointer if the referenced node has been deleted
        if self._last_event_id is not None:
            cur.execute("SELECT 1 FROM items WHERE id = ?", (self._last_event_id,))
            if cur.fetchone() is None:
                self._last_event_id = None
        cur.execute(
            (
                "INSERT INTO items (content, embedding, timestamp, modality, entities, topics, importance, recall_score, schema_id, frozen, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                item.content,
                self._to_blob(item.embedding),
                item.timestamp,
                item.modality,
                json.dumps(item.entities) if item.entities is not None else None,
                json.dumps(item.topics) if item.topics is not None else None,
                item.importance,
                item.recall_score,
                item.schema_id,
                int(item.frozen),
                item.source,
            ),
        )
        event_id = cast(int, cur.lastrowid)

        # temporal edge to previous event
        if self._last_event_id is not None:
            cur.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, relation) VALUES (?, ?, 'time')",
                (self._last_event_id, event_id),
            )
        self._last_event_id = event_id

        # entity edges
        for name in item.entities or []:
            ent_id = self._get_or_create_entity(name, cur)
            cur.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, relation) VALUES (?, ?, 'entity')",
                (event_id, ent_id),
            )

        # semantic edges
        for target_id in related or []:
            cur.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, relation) VALUES (?, ?, 'semantic')",
                (event_id, int(target_id)),
            )

        self.conn.commit()
        return event_id

    # ------------------------------------------------------------------
    def get_node(self, node_id: int) -> Node | None:
        """Return a :class:`Node` by ``node_id`` or ``None`` if missing."""

        cur = self.conn.cursor()
        cur.execute("SELECT id, content FROM items WHERE id = ?", (node_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Node(id=int(row[0]), content=str(row[1]))

    # ------------------------------------------------------------------
    def get_edges(self, source_id: int, relation: str | None = None) -> list[int]:
        """Return target IDs for edges from ``source_id``.

        Parameters
        ----------
        source_id:
            ID of the source node.
        relation:
            Optional relation filter (``time``, ``entity`` or ``semantic``).
        """

        cur = self.conn.cursor()
        if relation is None:
            cur.execute("SELECT target_id FROM edges WHERE source_id = ?", (source_id,))
        else:
            cur.execute(
                "SELECT target_id FROM edges WHERE source_id = ? AND relation = ?",
                (source_id, relation),
            )
        return [int(row[0]) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    def induce(self, node_ids: Sequence[int]) -> Dict[str, Dict[str, Any]]:
        """Return induced subgraph for ``node_ids``.

        The returned dictionary maps node ID strings to dictionaries
        containing at least ``neighbors`` (list of adjacent node IDs within
        ``node_ids``) and ``timestamp`` metadata pulled from the ``items``
        table. Additional metadata fields may be included in the future.
        """

        ids = [int(nid) for nid in node_ids]
        if not ids:
            return {}

        placeholders = ",".join(["?"] * len(ids))
        cur = self.conn.cursor()

        # Gather basic metadata for requested nodes
        cur.execute(
            f"SELECT id, timestamp FROM items WHERE id IN ({placeholders})",
            ids,
        )
        data: Dict[str, Dict[str, Any]] = {
            str(row[0]): {"neighbors": [], "timestamp": row[1]}
            for row in cur.fetchall()
        }

        # Collect edges where both endpoints are within ``ids``
        params = ids + ids
        cur.execute(
            (
                f"SELECT source_id, target_id FROM edges "
                f"WHERE source_id IN ({placeholders}) "
                f"AND target_id IN ({placeholders})"
            ),
            params,
        )
        for src, dst in cur.fetchall():
            key = str(src)
            if key in data:
                data[key]["neighbors"].append(str(dst))

        # Ensure all requested nodes are represented
        for nid in ids:
            key = str(nid)
            data.setdefault(key, {"neighbors": [], "timestamp": None})

        return data

    # ------------------------------------------------------------------
    def _get_or_create_entity(self, name: str, cur: sqlite3.Cursor) -> int:
        """Return node ID for ``name`` creating a node if necessary."""

        cur.execute("SELECT id FROM items WHERE content = ?", (name,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        emb = self._to_blob([0.0] * 384)
        cur.execute("INSERT INTO items (content, embedding) VALUES (?, ?)", (name, emb))
        return cast(int, cur.lastrowid)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_blob(embedding: Sequence[float]) -> bytes:
        """Serialize ``embedding`` into a BLOB for SQLite storage."""

        return array("f", embedding).tobytes()
