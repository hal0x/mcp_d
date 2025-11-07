from memory import MemoryService
from memory.models import MemoryItem


def test_explain_returns_chain(tmp_path):
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

    path = memory.explain(id1)
    relations = [rel for rel, _ in path]
    ids = [node.id for _, node in path]
    assert relations == ["self", "time", "entity", "semantic"]
    assert ids == [id1, id2, entity_id, id1]
