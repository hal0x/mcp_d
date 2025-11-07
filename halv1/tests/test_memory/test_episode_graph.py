import json
import sqlite3
from pathlib import Path

from memory import EpisodeGraph
from memory.models import MemoryItem

EMB = [0.0] * 384


def test_event_creation_and_edges(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.db"
    graph = EpisodeGraph(db_path)

    first_item = MemoryItem(
        "first event",
        EMB,
        timestamp=1,
        entities=["Alice"],
        importance=0.1,
        recall_score=0.2,
        schema_id=5,
        frozen=True,
        source="sensor-A",
    )
    first = graph.add_event(first_item)
    second_item = MemoryItem(
        "second event",
        EMB,
        timestamp=2,
        entities=["Alice", "Bob"],
        importance=0.3,
        recall_score=0.4,
        schema_id=6,
        source="sensor-B",
    )
    second = graph.add_event(second_item, related=[first])

    # nodes exist
    first_node = graph.get_node(first)
    second_node = graph.get_node(second)
    assert first_node is not None and first_node.content == "first event"
    assert second_node is not None and second_node.content == "second event"

    # entity nodes and metadata persistence
    conn: sqlite3.Connection = graph.conn
    cur = conn.cursor()
    cur.execute("SELECT id FROM items WHERE content = ?", ("Alice",))
    alice_id = cur.fetchone()[0]
    cur.execute("SELECT id FROM items WHERE content = ?", ("Bob",))
    bob_id = cur.fetchone()[0]

    # stored metadata for first event
    cur.execute(
        """
        SELECT timestamp, entities, importance, recall_score, schema_id, frozen, source
        FROM items WHERE id=?
        """,
        (first,),
    )
    row = cur.fetchone()
    assert row is not None
    ts, ents, importance, recall, schema_id, frozen, source = row
    assert ts == 1
    assert json.loads(ents) == ["Alice"]
    assert importance == 0.1
    assert recall == 0.2
    assert schema_id == 5
    assert frozen == 1
    assert source == "sensor-A"

    cur.execute("SELECT frozen FROM items WHERE id=?", (second,))
    assert cur.fetchone()[0] == 0

    # time edge
    cur.execute(
        "SELECT 1 FROM edges WHERE source_id=? AND target_id=? AND relation='time'",
        (first, second),
    )
    assert cur.fetchone()

    # semantic edge
    cur.execute(
        "SELECT 1 FROM edges WHERE source_id=? AND target_id=? AND relation='semantic'",
        (second, first),
    )
    assert cur.fetchone()

    # entity edges
    cur.execute(
        "SELECT 1 FROM edges WHERE source_id=? AND target_id=? AND relation='entity'",
        (second, alice_id),
    )
    assert cur.fetchone()
    cur.execute(
        "SELECT 1 FROM edges WHERE source_id=? AND target_id=? AND relation='entity'",
        (second, bob_id),
    )
    assert cur.fetchone()

    # FTS5 index
    cur.execute("SELECT rowid FROM items_fts WHERE content MATCH 'first'")
    assert cur.fetchone()[0] == first

    # HNSW index
    cur.execute("SELECT COUNT(*) FROM items_hnsw")
    assert cur.fetchone()[0] == 4
