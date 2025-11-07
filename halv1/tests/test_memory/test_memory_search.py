"""Tests for UnifiedMemory.search selecting memory based on long_term flag."""

import sys
from types import ModuleType

import pytest

# Stub out core.utils.json_io to avoid circular import in tests
json_io = ModuleType("json_io")
json_io.load_json = lambda path, default: default  # type: ignore[attr-defined]
json_io.save_json = lambda path, data, indent=None: None  # type: ignore[attr-defined]
json_io.parse_llm_json = lambda text, default=None: default  # type: ignore[attr-defined]
core_utils = ModuleType("core.utils")
core_utils.json_io = json_io  # type: ignore[attr-defined]
core_module = ModuleType("core")
core_module.utils = core_utils  # type: ignore[attr-defined]
core_module.__path__ = []
core_utils.__path__ = []
sys.modules.setdefault("core", core_module)
sys.modules.setdefault("core.utils", core_utils)
sys.modules.setdefault("core.utils.json_io", json_io)

from memory import EpisodeGraph, FTS5Index, MemoryService, UnifiedMemory  # noqa: E402
from memory.models import MemoryItem  # noqa: E402
from retriever.cross_encoder import CrossEncoder  # noqa: E402
from retriever.pipeline import Candidate, RetrievalPipeline  # noqa: E402


def test_search_long_term_flag_filters_memories() -> None:
    store = UnifiedMemory()
    short_item = "foo short"
    long_item = "foo long"
    store.remember(short_item)
    store.remember(long_item, long_term=True)

    assert store.search("foo", long_term=False) == [short_item]
    assert store.search("foo", long_term=True) == [long_item]
    assert store.search("foo") == [short_item, long_item]


def test_memory_service_search_returns_matches(tmp_path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    memory.remember("foo short")
    memory.remember("bar long")

    assert memory.search("foo") == ["foo short"]


def test_memory_service_search_long_term_flag(tmp_path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    memory.write_event("foo short")
    memory.remember("foo long")

    assert memory.search("foo", long_term=False) == ["foo short"]
    assert memory.search("foo", long_term=True) == ["foo long"]
    assert memory.search("foo") == ["foo short", "foo long"]


def test_ppr_rank_combines_features(tmp_path) -> None:
    EMB = [0.0] * 384
    graph = EpisodeGraph(tmp_path / "mem.db")
    id1 = graph.add_event(MemoryItem("doc1", EMB, timestamp=1000.0))
    id2 = graph.add_event(MemoryItem("doc2", EMB, timestamp=2000.0), related=[id1])
    id3 = graph.add_event(MemoryItem("doc3", EMB, timestamp=1500.0), related=[id2])

    candidates = [
        Candidate(str(id1), "doc1", [0.1], 0.6),
        Candidate(str(id2), "doc2", [0.2], 0.4),
        Candidate(str(id3), "doc3", [0.3], 0.8),
    ]
    pipe = RetrievalPipeline(None, None, graph, None)
    subgraph = graph.induce([c.node_id for c in candidates])
    ppr, recency, centrality = pipe._ppr_rank(subgraph, candidates)
    scores = {
        c.node_id: (
            0.55 * c.score
            + 0.20 * ppr[c.node_id]
            + 0.15 * recency.get(c.node_id, 0.0)
            + 0.10 * centrality.get(c.node_id, 0.0)
        )
        for c in candidates
    }

    assert scores[str(id3)] > scores[str(id2)] > scores[str(id1)]


def test_beam_search_prioritizes_combined_scores(tmp_path) -> None:
    EMB = [0.0] * 384
    graph = EpisodeGraph(tmp_path / "mem.db")
    id1 = graph.add_event(MemoryItem("doc1", EMB))
    id2 = graph.add_event(MemoryItem("doc2", EMB))

    candidates = [
        Candidate(str(id1), "doc1", [0.1], 0.9),
        Candidate(str(id2), "doc2", [0.2], 0.2),
    ]
    pipe = RetrievalPipeline(None, None, graph, None)
    subgraph = graph.induce([c.node_id for c in candidates])
    fast_scores = {str(id1): 0.1, str(id2): 0.9}
    result = pipe.beam_search(
        subgraph, candidates, fast_scores, beam_width=1, max_depth=0
    )

    assert [c.node_id for c in result] == [str(id2)]


def test_run_uses_beam_search_before_cross_encoder(monkeypatch, tmp_path) -> None:
    class DummyIndex:
        def __init__(self, results):
            self.results = results

        def search(self, query, top_k):
            return self.results

    class DummyModel:
        def __init__(self):
            self.calls = []

        def predict(self, pairs):
            self.calls.extend(pairs)
            return [0.0 for _ in pairs]

    EMB = [0.0] * 384
    graph = EpisodeGraph(tmp_path / "mem.db")
    id1 = graph.add_event(MemoryItem("doc1", EMB))
    id2 = graph.add_event(MemoryItem("doc2", EMB))

    candidates = [
        Candidate(str(id1), "doc1", [0.1], 0.9),
        Candidate(str(id2), "doc2", [0.2], 0.2),
    ]
    hnsw = DummyIndex(candidates)
    fts5 = FTS5Index(graph)
    model = DummyModel()
    ce = CrossEncoder(model=model)
    pipe = RetrievalPipeline(hnsw, fts5, graph, ce)

    def fake_beam(subgraph, cands, fast_scores, beam_width=5, max_depth=2):
        return [cands[0]]

    monkeypatch.setattr(pipe, "beam_search", fake_beam)

    pipe.run("q", top_k=1, candidate_k=2)

    assert model.calls == [("q", "doc1")]


def test_search_long_logs_and_handles_index_error(
    monkeypatch, tmp_path, caplog
) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    memory.remember("foo long")

    def failing_run(self, query, top_k=20):
        raise IndexError("corrupted index")

    monkeypatch.setattr(RetrievalPipeline, "run", failing_run)
    with caplog.at_level("WARNING"):
        result = memory.search("foo", long_term=True)

    assert result == []
    assert "IndexError" in caplog.text
    assert "corrupted index" in caplog.text


def test_search_long_propagates_unexpected_error(monkeypatch, tmp_path) -> None:
    memory = MemoryService(tmp_path / "graph.db")
    memory.remember("foo long")

    def failing_run(self, query, top_k=20):
        raise RuntimeError("boom")

    monkeypatch.setattr(RetrievalPipeline, "run", failing_run)
    with pytest.raises(RuntimeError):
        memory.search("foo", long_term=True)
