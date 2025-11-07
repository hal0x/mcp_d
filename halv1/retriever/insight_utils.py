from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from index.cluster_manager import ClusterManager
from index.vector_index import VectorEntry
from retriever.index_protocol import IndexProtocol


async def build_insight_cards(
    query: str,
    index: IndexProtocol,
    clusters: ClusterManager,
    top_k_vectors: int = 25,
    top_k_insights: int = 5,
) -> Tuple[List[Dict[str, object]], List[VectorEntry]]:
    """Return insight cards and matching vector hits for ``query``.

    This helper encapsulates the boilerplate needed to fetch raw facts,
    compute cluster weights and assemble "insight cards".  It returns both
    the cards and the raw :class:`~index.vector_index.VectorEntry` objects so
    that callers can reuse the hits for additional processing.
    """

    top_k_insights = min(top_k_insights, 5)

    vector_hits: List[VectorEntry] = await index.search(query, top_k=top_k_vectors)
    query_emb: Sequence[float] = await index.embed(query)

    # map entries to their clusters
    entry_to_cluster: Dict[str, str] = {
        m.chunk_id: cid
        for cid, cl in clusters.clusters.items()
        for m in getattr(cl, "members", [])
    }

    # compute cluster weights based on the vector hits
    cluster_weight: defaultdict[str, float] = defaultdict(float)
    for hit in vector_hits:
        cid = entry_to_cluster.get(hit.chunk_id)
        if cid:
            cluster_weight[cid] += 1.0

    # compute BM25 scores per cluster
    bm25_hits = index.bm25.search(query, top_k=len(getattr(index, "entries", [])))
    bm25_score: defaultdict[str, float] = defaultdict(float)
    for doc_id, score in bm25_hits:
        cid = entry_to_cluster.get(doc_id)
        if cid and score > bm25_score[cid]:
            bm25_score[cid] = score

    # fetch top insights with optional weighting; fall back for simple managers
    try:
        raw_insights = clusters.get_insights(
            query_emb,
            top_k=top_k_insights,
            cluster_weight=cluster_weight,
            bm25_score=bm25_score,
            w5=0.1,
            w6=0.1,
        )
    except TypeError:  # pragma: no cover - simple managers may ignore weights
        raw_insights = clusters.get_insights(query_emb, top_k=top_k_insights)

    # merge insights with matching fragments
    cards: List[Dict[str, object]] = []
    for ins in raw_insights:
        cid = ins.get("id")
        fragments = [
            hit.text for hit in vector_hits if entry_to_cluster.get(hit.chunk_id) == cid
        ][:3]
        cards.append(
            {
                "summary": ins.get("summary", ""),
                "medoid": ins.get("medoid", ""),
                "fragments": fragments,
            }
        )

    return cards, vector_hits
