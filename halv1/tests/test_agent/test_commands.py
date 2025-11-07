from pathlib import Path
from types import SimpleNamespace

from agent.commands import handle_command
from memory import MemoryService
from memory.models import MemoryItem


def test_remember_command_adds_memory(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    agent = SimpleNamespace()
    reply = handle_command("/remember hello", memory, agent)
    assert reply == "Запомнил: hello"
    assert "hello" in memory.recall(long_term=True)


def test_remember_command_sets_frozen(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    # baseline: direct remembers are not frozen
    memory.remember("plain")
    cur = memory.graph.conn.cursor()
    cur.execute("SELECT frozen FROM items WHERE content = ?", ("plain",))
    row = cur.fetchone()
    assert row is not None and row[0] == 0

    agent = SimpleNamespace()
    handle_command("/remember hello", memory, agent)
    cur.execute("SELECT frozen FROM items WHERE content = ?", ("hello",))
    row = cur.fetchone()
    assert row is not None and row[0] == 1


def test_forget_command_removes_memory(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    memory.remember("hello")
    agent = SimpleNamespace()
    reply = handle_command("/forget hello", memory, agent)
    assert reply == "Забыл: hello"
    assert not memory.search("hello")


def test_why_returns_path(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    graph = memory.graph
    emb = [0.0] * 384
    item1 = MemoryItem(content="first", embedding=emb, timestamp=1)
    id1 = graph.add_event(item1)
    item2 = MemoryItem(content="second", embedding=emb, timestamp=2, entities=["Alice"])
    id2 = graph.add_event(item2)
    entity_id = graph.get_edges(id2, "entity")[0]
    cur = graph.conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO edges (source_id, target_id, relation) VALUES (?, ?, 'semantic')",
        (entity_id, id1),
    )
    graph.conn.commit()
    reply = handle_command(f"/why {id1}", memory, agent=None)
    expected = [
        f"{id1}: first",
        f"time -> {id2}: second",
        f"entity -> {entity_id}: Alice",
        f"semantic -> {id1}: first",
    ]
    assert reply.splitlines() == expected


def test_snapshot_lists_memory(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    memory.remember("beta")
    memory.write_event("event1")
    agent = SimpleNamespace()
    reply = handle_command("/snapshot", memory, agent)
    assert reply is not None
    assert "event1" in reply and "beta" in reply


def test_snapshot_shows_non_empty_lists(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    memory.write_event("event1")
    agent = SimpleNamespace()
    reply = handle_command("/snapshot", memory, agent)
    assert reply is not None
    lines = reply.splitlines()
    idx = lines.index("Эпизоды:")
    assert lines[idx + 1] != "<пусто>"
    idx = lines.index("Краткосрочная память:")
    assert lines[idx + 1] != "<пусто>"
