import math
from pathlib import Path

import numpy as np

from memory import EpisodeGraph
from memory.models import MemoryItem
from memory.write_pipeline import EMBED_DIM, WritePipeline
from utils.vector_math import cosine_similarity


def _unit_vector(index: int) -> list[float]:
    vec = [0.0] * EMBED_DIM
    vec[index] = 1.0
    return vec


def test_nearest_uses_hnsw_index(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.db"
    graph = EpisodeGraph(db_path)

    def distance(blob: bytes, query: bytes) -> float:
        a = np.frombuffer(blob, dtype=np.float32)
        b = np.frombuffer(query, dtype=np.float32)
        return float(1.0 - cosine_similarity(a, b))

    graph.conn.create_function("distance", 2, distance)
    item_a = MemoryItem("a", _unit_vector(0), timestamp=1)
    item_b = MemoryItem("b", _unit_vector(1), timestamp=2)
    id_a = graph.add_event(item_a)
    graph.add_event(item_b)

    pipeline = WritePipeline(graph, k=1)
    queries: list[str] = []
    graph.conn.set_trace_callback(queries.append)
    neighbours = pipeline._nearest(_unit_vector(0))

    assert any("items_hnsw" in q for q in queries)
    assert neighbours and neighbours[0][0] == id_a


def test_nearest_falls_back_without_index(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.db"
    graph = EpisodeGraph(db_path)

    item_a = MemoryItem("a", _unit_vector(0), timestamp=1)
    item_b = MemoryItem("b", _unit_vector(1), timestamp=2)
    id_a = graph.add_event(item_a)
    graph.add_event(item_b)

    pipeline = WritePipeline(graph, k=1)
    queries: list[str] = []
    graph.conn.set_trace_callback(queries.append)
    neighbours = pipeline._nearest(_unit_vector(0))

    assert any("SELECT id, embedding FROM items" in q for q in queries)
    assert neighbours and neighbours[0][0] == id_a


def test_write_event_skips_when_not_useful(tmp_path: Path) -> None:
    graph = EpisodeGraph(tmp_path / "mem.db")
    pipeline = WritePipeline(graph, k=1, tau_s=0.3, tau_u=0.8)

    emb_a = _unit_vector(0)
    emb_b = [0.75, math.sqrt(1 - 0.75**2)] + [0.0] * (EMBED_DIM - 2)

    def fake_embed(text: str) -> list[float]:
        return emb_a if text == "a" else emb_b

    pipeline._embed = fake_embed  # type: ignore[misc]

    pipeline.write_event("a", timestamp=1)
    result_skip = pipeline.write_event("b", timestamp=2)
    assert result_skip.node_id is None

    cur = graph.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM items")
    assert cur.fetchone()[0] == 1

    result_force = pipeline.write_event("b", timestamp=3, frozen=True)
    assert result_force.node_id is not None
    cur.execute("SELECT COUNT(*) FROM items")
    assert cur.fetchone()[0] == 2
    cur.execute("SELECT frozen FROM items WHERE id = ?", (result_force.node_id,))
    assert cur.fetchone()[0] == 1
