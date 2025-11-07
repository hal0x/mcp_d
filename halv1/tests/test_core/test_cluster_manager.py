import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from index.cluster_manager import Cluster, ClusterManager
from index.insight_store import InsightStore
from index.vector_index import VectorEntry


class DummySummarizer:
    def summarize(self, texts):
        # simple deterministic summariser for tests
        return " | ".join(texts)


def test_ingest_deduplicates_and_clusters():
    mgr = ClusterManager(dup_threshold=0.95, attach_threshold=0.8)
    entries = [
        VectorEntry("1", "alpha beta", [1.0, 0.0], {}),
        VectorEntry("2", "alpha beta duplicate", [0.98, 0.02], {}),
        VectorEntry("3", "gamma delta", [0.0, 1.0], {}),
    ]
    mgr.ingest(entries)
    # second entry is a duplicate -> only two stored
    assert len(mgr._entries) == 2
    assert len(mgr.clusters) == 2

    # original document weight is increased due to duplicate
    assert mgr.doc_weights["1"] == pytest.approx(2.0)
    cid = next(
        cid
        for cid, cl in mgr.clusters.items()
        if any(m.chunk_id == "1" for m in cl.members)
    )
    cl = mgr.clusters[cid]
    assert cl.freshness == pytest.approx(3.0)

    mgr.summarise(DummySummarizer())
    for cl in mgr.clusters.values():
        assert cl.summary
        assert cl.medoid is not None
        assert cl.timeline

    insights = mgr.get_insights([1.0, 0.0], top_k=2)
    assert len(insights) == 2
    # first insight corresponds to the cluster around [1,0]
    assert insights[0]["medoid"] == "alpha beta"


def test_ingest_updates_freshness_and_quality():
    mgr = ClusterManager()
    entry1 = VectorEntry("1", "a", [1.0, 0.0], {"source": "news", "chat": "c1"})
    entry2 = VectorEntry("2", "b", [0.7, 0.3], {"source": "user"})
    mgr.ingest([entry1])
    cid = next(iter(mgr.clusters))
    cl = mgr.clusters[cid]
    sq1 = mgr._compute_source_quality(entry1.metadata)
    assert cl.freshness > 0
    assert cl.freshness_ts > 0
    assert cl.source_quality == sq1
    ts1 = cl.freshness_ts
    mgr.ingest([entry2])
    cl = mgr.clusters[cid]
    sq2 = mgr._compute_source_quality(entry2.metadata)
    assert cl.freshness > 1.0
    assert cl.freshness_ts >= ts1
    expected = (sq1 + sq2) / 2
    assert cl.source_quality == pytest.approx(expected)


def test_duplicate_updates_weight_and_timestamp():
    mgr = ClusterManager(dup_threshold=0.95, attach_threshold=0.8)
    first = VectorEntry("1", "alpha", [1.0, 0.0], {})
    dup = VectorEntry("2", "alpha duplicate", [0.98, 0.02], {})
    mgr.ingest([first])
    cid = next(iter(mgr.clusters))
    cl = mgr.clusters[cid]
    ts = cl.freshness_ts
    mgr.ingest([dup])
    cl = mgr.clusters[cid]
    assert mgr.doc_weights["1"] == pytest.approx(2.0)
    assert cl.freshness == pytest.approx(3.0)
    assert cl.freshness_ts >= ts
    assert len(mgr._entries) == 1


def test_clusters_survive_restart(tmp_path):
    mgr = ClusterManager(dup_threshold=0.95, attach_threshold=0.8)
    entries = [
        VectorEntry("1", "alpha beta", [1.0, 0.0], {}),
        VectorEntry("3", "gamma delta", [0.0, 1.0], {}),
    ]
    mgr.ingest(entries)
    mgr.summarise(DummySummarizer())

    store = InsightStore(str(tmp_path / "insights.json"))
    mgr.save(store)

    mgr2 = ClusterManager()
    mgr2.load(store)

    # ensure centroids and members were restored correctly
    orig_centroids = {cid: cl.centroid for cid, cl in mgr.clusters.items()}
    loaded_centroids = {cid: cl.centroid for cid, cl in mgr2.clusters.items()}
    assert loaded_centroids == orig_centroids
    orig_members = {
        cid: [m.chunk_id for m in cl.members] for cid, cl in mgr.clusters.items()
    }
    loaded_members = {
        cid: [m.chunk_id for m in cl.members] for cid, cl in mgr2.clusters.items()
    }
    assert loaded_members == orig_members
    orig_timeline = {
        cid: [m.chunk_id for m in cl.timeline] for cid, cl in mgr.clusters.items()
    }
    loaded_timeline = {
        cid: [m.chunk_id for m in cl.timeline] for cid, cl in mgr2.clusters.items()
    }
    assert loaded_timeline == orig_timeline

    insights = mgr2.get_insights([1.0, 0.0], top_k=2)
    assert len(insights) == 2
    assert insights[0]["medoid"] == "alpha beta"
    # summaries should also persist
    assert insights[0]["summary"] == "alpha beta"


def test_get_insights_ranking_with_additional_scores():
    mgr = ClusterManager()
    entry_a = VectorEntry("a", "a", [1.0, 0.0], {})
    entry_b = VectorEntry("b", "b", [1.0, 0.0], {})
    cluster_a = Cluster(
        id="a",
        members=[entry_a],
        centroid=[1.0, 0.0],
        medoid=entry_a,
        pagerank=0.1,
        freshness=5.0,
        source_quality=0.5,
    )
    cluster_b = Cluster(
        id="b",
        members=[entry_b],
        centroid=[1.0, 0.0],
        medoid=entry_b,
        pagerank=0.9,
        freshness=5.0,
        source_quality=0.5,
    )
    mgr.clusters = {"a": cluster_a, "b": cluster_b}

    insights = mgr.get_insights([1.0, 0.0], top_k=2)
    assert insights[0]["medoid"] == "b"

    mgr.clusters["a"].pagerank = mgr.clusters["b"].pagerank
    mgr.clusters["a"].freshness = 10.0
    mgr.clusters["b"].freshness = 1.0
    insights = mgr.get_insights([1.0, 0.0], top_k=2)
    assert insights[0]["medoid"] == "a"

    mgr.clusters["a"].freshness = mgr.clusters["b"].freshness
    mgr.clusters["a"].source_quality = 0.2
    mgr.clusters["b"].source_quality = 0.9
    insights = mgr.get_insights([1.0, 0.0], top_k=2)
    assert insights[0]["medoid"] == "b"

    # reset scores for additional component tests
    mgr.clusters["a"].pagerank = mgr.clusters["b"].pagerank
    mgr.clusters["a"].freshness = mgr.clusters["b"].freshness
    mgr.clusters["a"].source_quality = mgr.clusters["b"].source_quality
    cluster_weights = {"a": 2.0, "b": 1.0}
    insights = mgr.get_insights([1.0, 0.0], top_k=2, cluster_weight=cluster_weights)
    assert insights[0]["medoid"] == "a"

    bm25_scores = {"a": 0.1, "b": 0.9}
    insights = mgr.get_insights([1.0, 0.0], top_k=2, bm25_score=bm25_scores)
    assert insights[0]["medoid"] == "b"


def test_get_insights_weighting_changes_order():
    mgr = ClusterManager()
    entry_a = VectorEntry("a", "a", [1.0, 0.0], {})
    entry_b = VectorEntry("b", "b", [0.0, 1.0], {})
    cluster_a = Cluster(
        id="a",
        members=[entry_a],
        centroid=[1.0, 0.0],
        medoid=entry_a,
        pagerank=0.1,
    )
    cluster_b = Cluster(
        id="b",
        members=[entry_b],
        centroid=[0.0, 1.0],
        medoid=entry_b,
        pagerank=0.9,
    )
    mgr.clusters = {"a": cluster_a, "b": cluster_b}

    query = [1.0, 0.0]
    insights = mgr.get_insights(query, top_k=2, w1=1.0, w2=0.0, w3=0.0, w4=0.0)
    assert insights[0]["medoid"] == "a"

    insights = mgr.get_insights(query, top_k=2, w1=0.0, w2=1.0, w3=0.0, w4=0.0)
    assert insights[0]["medoid"] == "b"


def test_drift_detection_splits_cluster():
    mgr = ClusterManager(attach_threshold=0.0)
    first = VectorEntry("1", "a", [1.0, 0.0], {})
    second = VectorEntry("2", "b", [0.0, 1.0], {})
    mgr.ingest([first])
    assert len(mgr.clusters) == 1
    mgr.ingest([second])
    # drift should trigger split creating a new cluster for the second entry
    assert len(mgr.clusters) == 2
    assert any(any(m.text == "b" for m in cl.members) for cl in mgr.clusters.values())


def test_cusum_detects_abrupt_centroid_shift():
    mgr = ClusterManager(attach_threshold=0.0)
    first = VectorEntry("1", "a", [1.0, 0.0], {})
    second = VectorEntry("2", "b", [0.6, 0.4], {})
    third = VectorEntry("3", "c", [0.0, 1.0], {})
    mgr.ingest([first])
    mgr.ingest([second])
    # after two similar entries no split should happen
    assert len(mgr.clusters) == 1
    cid = next(iter(mgr.clusters))
    assert len(mgr._centroid_history[cid]) == 2

    mgr.ingest([third])
    # abrupt shift should trigger split
    assert len(mgr.clusters) == 2
    # history entries contain timestamps and centroids
    for ts, cen in mgr._centroid_history[cid]:
        assert isinstance(ts, float)
        assert isinstance(cen, list)


def test_recompute_pagerank_uses_document_graph():
    mgr = ClusterManager(attach_threshold=1.0, dup_threshold=0.99, k_neighbors=1)
    e1 = VectorEntry("1", "a", [1.0, 0.0], {})
    e2 = VectorEntry("2", "b", [0.8, 0.2], {})
    e3 = VectorEntry("3", "c", [0.0, 1.0], {})
    mgr.ingest([e1, e2, e3])
    # graph should link each document to its nearest neighbour
    assert mgr._doc_graph["1"] == {"2"}
    assert mgr._doc_graph["2"] == {"1"}
    assert mgr._doc_graph["3"] == {"2"}
    mgr.recompute_pagerank()
    ranks = {cl.members[0].chunk_id: cl.pagerank for cl in mgr.clusters.values()}
    assert ranks["2"] > ranks["1"] > ranks["3"]


def test_rebuild_clusters_groups_entries():
    mgr = ClusterManager()
    entries = [
        VectorEntry("1", "a", [1.0, 0.0], {}),
        VectorEntry("2", "b", [1.1, 0.0], {}),
        VectorEntry("3", "c", [0.0, 1.0], {}),
    ]
    mgr._entries = entries.copy()
    mgr.rebuild_clusters(k=2)
    assert len(mgr.clusters) == 2
    total = sum(len(cl.members) for cl in mgr.clusters.values())
    assert total == 3


def test_apply_decay_and_mark_noise_accelerate_weights():
    mgr = ClusterManager()
    entry = VectorEntry("d1", "t", [1.0, 0.0], {})
    entry.timestamp = 0.0
    mgr._entries = [entry]
    mgr.doc_weights["d1"] = 1.0
    hl = mgr.weight_half_life
    mgr.apply_decay(hl)
    assert mgr.doc_weights["d1"] == pytest.approx(0.5)
    mgr.mark_noise("d1")
    assert mgr.doc_weights["d1"] == pytest.approx(0.25)
    mgr.apply_decay(hl)
    assert mgr.doc_weights["d1"] == pytest.approx(0.125)


def test_source_message_limit_archives_documents():
    mgr = ClusterManager(source_msg_limit=2)
    e1 = VectorEntry("1", "a", [1.0, 0.0], {"source": "s"})
    e2 = VectorEntry("2", "b", [0.0, 1.0], {"source": "s"})
    e3 = VectorEntry("3", "c", [1.0, 1.0], {"source": "s"})
    mgr.ingest([e1, e2])
    assert not mgr.archived_docs
    mgr.ingest([e3])
    assert {"1", "2", "3"} <= mgr.archived_docs
    assert mgr.doc_weights["1"] == pytest.approx(0.5)
    assert mgr.doc_weights["3"] == pytest.approx(0.5)
