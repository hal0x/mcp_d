from memory import EpisodeGraph, FTS5Index
from memory.models import MemoryItem
from retriever.cross_encoder import CrossEncoder
from retriever.pipeline import Candidate, RetrievalPipeline


class DummyIndex:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k):  # pragma: no cover - simple passthrough
        return self.results


class DummyModel:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):  # pragma: no cover - deterministic
        self.calls.extend(pairs)
        return [self.scores[doc] for _, doc in pairs]


def test_generate_candidates_merges_sources(tmp_path):
    graph = EpisodeGraph(tmp_path / "mem.db")
    EMB = [0.0] * 384
    id1 = graph.add_event(MemoryItem("alpha", EMB))
    id2 = graph.add_event(MemoryItem("beta", EMB))
    id3 = graph.add_event(MemoryItem("gamma", EMB))
    hnsw = DummyIndex(
        [
            Candidate(str(id2), "beta", [float(id2)], -1.0),
            Candidate(str(id3), "gamma", [float(id3)], 0.2),
        ]
    )
    fts5 = FTS5Index(graph)
    pipe = RetrievalPipeline(hnsw, fts5, graph, None)
    cands = pipe.generate_candidates("alpha OR beta", top_k=3)
    ids = sorted(c.node_id for c in cands)
    assert ids == [str(id1), str(id2), str(id3)]
    cur = graph.conn.cursor()
    cur.execute("SELECT bm25(items_fts) FROM items_fts WHERE rowid=?", (id2,))
    bm25_score = float(cur.fetchone()[0])
    assert {c.node_id: c.score for c in cands}[str(id2)] == bm25_score


def test_pipeline_ranks_results_using_cross_encoder_and_mmr(tmp_path):
    graph = EpisodeGraph(tmp_path / "mem.db")
    EMB = [0.0] * 384
    id1 = graph.add_event(MemoryItem("alpha", EMB))
    id2 = graph.add_event(MemoryItem("beta", EMB), related=[id1])
    _id3 = graph.add_event(MemoryItem("gamma", EMB), related=[id2])
    hnsw = DummyIndex([Candidate(str(id1), "alpha", [float(id1)], 0.4)])
    fts5 = FTS5Index(graph)
    model = DummyModel({"alpha": 0.3, "beta": 0.5, "gamma": 0.4})
    ce = CrossEncoder(model=model)
    pipe = RetrievalPipeline(hnsw, fts5, graph, ce)
    query = "beta OR gamma"
    cands = pipe.generate_candidates(query, top_k=5)
    subgraph = pipe.induce_subgraph(cands)
    ppr, recency, centrality = pipe._ppr_rank(subgraph, cands)
    fast_scores = {
        c.node_id: (
            0.55 * c.score
            + 0.20 * ppr[c.node_id]
            + 0.15 * recency.get(c.node_id, 0.0)
            + 0.10 * centrality.get(c.node_id, 0.0)
        )
        for c in cands
    }
    cands = pipe.beam_search(subgraph, cands, fast_scores)
    expected_fast = {c.node_id: fast_scores[c.node_id] for c in cands}

    results = pipe.run(query, top_k=20, candidate_k=5)
    ids = [r.node_id for r in results]
    assert ids[0] == str(id2)
    fast_scores_run = {r.node_id: r.fast_score for r in results}
    assert fast_scores_run == expected_fast
    assert len(model.calls) == len(results)
