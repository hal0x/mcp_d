from pathlib import Path

from memory import MemoryService


def test_remember_sets_frozen_flag(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    memory.remember("freeze", frozen=True)
    cur = memory.graph.conn.cursor()
    cur.execute("SELECT frozen FROM items WHERE content = ?", ("freeze",))
    row = cur.fetchone()
    assert row is not None and row[0] == 1


def test_write_event_sets_frozen_flag(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    memory.write_event("short", frozen=True)
    memory.buffer.flush()
    cur = memory.graph.conn.cursor()
    cur.execute("SELECT frozen FROM items WHERE content = ?", ("short",))
    row = cur.fetchone()
    assert row is not None and row[0] == 1
