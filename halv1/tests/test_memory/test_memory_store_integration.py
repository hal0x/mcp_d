from pathlib import Path

from memory import UnifiedMemory


def test_consolidate_flushes_events_to_graph(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.db"
    store = UnifiedMemory(episode_graph_path=db_path)
    store.write_event("hello")
    assert store.read_events() == ["hello"]
    store.consolidate()
    assert store.read_events() == []
    assert store.graph is not None
    cur = store.graph.conn.cursor()
    cur.execute("SELECT content FROM items")
    rows = cur.fetchall()
    assert rows and rows[0][0] == "hello"
