from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory import EpisodeGraph
from memory.models import MemoryItem
from memory.schemas import consolidate_graph

EMB = [0.0] * 384


def test_consolidate_graph_assigns_schema_ids(tmp_path: "Path") -> None:
    db_path = tmp_path / "mem.db"
    graph = EpisodeGraph(db_path)
    now = datetime.now(timezone.utc)
    t1 = int((now - timedelta(days=1)).timestamp())
    t2 = int((now - timedelta(days=2)).timestamp())
    t3 = int((now - timedelta(days=3)).timestamp())
    t_old = int((now - timedelta(days=8)).timestamp())

    graph.add_event(MemoryItem("buy milk", EMB, timestamp=t1))
    graph.add_event(MemoryItem("buy milk", EMB, timestamp=t2))
    graph.add_event(MemoryItem("buy bread", EMB, timestamp=t3))
    old_id = graph.add_event(MemoryItem("buy milk", EMB, timestamp=t_old))

    consolidate_graph(graph, now=now)

    cur = graph.conn.cursor()
    cur.execute("SELECT name, id FROM schemas")
    rows = cur.fetchall()
    assert len(rows) == 2
    schema_map = {name: _id for name, _id in rows}

    cur.execute("SELECT schema_id FROM items WHERE content='buy bread'")
    assert cur.fetchone()[0] == schema_map["buy bread"]

    cur.execute(
        "SELECT schema_id FROM items WHERE content='buy milk' AND timestamp=?",
        (t1,),
    )
    assert cur.fetchone()[0] == schema_map["buy milk"]
    cur.execute(
        "SELECT schema_id FROM items WHERE content='buy milk' AND timestamp=?",
        (t2,),
    )
    assert cur.fetchone()[0] == schema_map["buy milk"]

    cur.execute("SELECT schema_id FROM items WHERE id=?", (old_id,))
    assert cur.fetchone()[0] is None
