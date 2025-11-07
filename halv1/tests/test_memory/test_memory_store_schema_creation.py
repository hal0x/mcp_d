from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory import UnifiedMemory


def test_consolidate_creates_schema(tmp_path: Path) -> None:
    store = UnifiedMemory(episode_graph_path=tmp_path / "graph.db")
    pipeline = getattr(store.memory_service, "pipeline", None)
    if pipeline:
        pipeline.tau_merge = 1.1
    now = datetime.now(timezone.utc)
    store.write_event("buy milk", timestamp=now - timedelta(days=1))
    store.write_event("buy milk", timestamp=now - timedelta(days=2))
    store.write_event("buy bread", timestamp=now - timedelta(days=1))

    store.consolidate()

    assert store.graph is not None
    cur = store.graph.conn.cursor()
    cur.execute("SELECT id FROM schemas WHERE name='buy milk'")
    row = cur.fetchone()
    assert row is not None
    schema_id = row[0]
    cur.execute(
        "SELECT content FROM items WHERE schema_id=? ORDER BY timestamp",
        (schema_id,),
    )
    items = [r[0] for r in cur.fetchall()]
    assert items == ["buy milk", "buy milk"]
