"""Retriever for answering questions using the vector index."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, cast

from index.cluster_manager import ClusterManager
from index.vector_index import VectorEntry
from retriever.index_protocol import IndexProtocol
from retriever.insight_utils import build_insight_cards


@dataclass
class RetrievalResult:
    """Container for raw vector hits, insight cards and context."""

    vector_hits: List[VectorEntry]
    insights: List[Dict[str, object]]
    context: str
    brief_context: str | None = None


class Retriever:
    """Perform similarity search over the :class:`VectorIndex`."""

    def __init__(
        self,
        index: IndexProtocol,
        cluster_manager: ClusterManager,
        *,
        get_active_theme: Callable[[], str] | None = None,
        context_config: Mapping[str, Any] | None = None,
    ):
        self.index: IndexProtocol = index
        self.cluster_manager = cluster_manager
        self.get_active_theme = get_active_theme
        self.context_cfg: Dict[str, Any] = dict(context_config or {})

    async def query(self, question: str, top_k: int = 25) -> List[VectorEntry]:
        """Return the most relevant entries for ``question``."""
        return await self.index.search(question, top_k=top_k)

    def _compose_context(
        self,
        insight_cards: Sequence[Mapping[str, object]],
        *,
        two_pass: bool,
    ) -> Tuple[str, str | None]:
        """Return context and optional brief context for ``insight_cards``."""

        ctx_parts: List[str] = []
        for card in insight_cards:
            lines = [f"- {card['summary']}: {card['medoid']}"]
            fragments = cast(List[str], card["fragments"])
            lines.extend(f"  * {f}" for f in fragments)
            ctx_parts.append("\n".join(lines))
        context = "\n\n".join(ctx_parts)
        brief_context = None
        if two_pass:
            brief_context = "\n".join(
                f"- {c['summary']}: {c['medoid']}" for c in insight_cards
            )
        return context, brief_context

    async def query_with_insights(
        self,
        question: str,
        top_k_vectors: int = 25,
        top_k_insights: int = 5,
        *,
        two_pass: bool = False,
    ) -> RetrievalResult:
        """Return vector hits and matching cluster insights for ``question``.

        When ``two_pass`` is ``True`` both a detailed ``context`` and a
        ``brief_context`` are returned so that the caller can first provide a
        short answer and then optionally request more details.
        """

        insight_cards, vector_hits = await build_insight_cards(
            question,
            self.index,
            self.cluster_manager,
            top_k_vectors=top_k_vectors,
            top_k_insights=top_k_insights,
        )
        context, brief_context = self._compose_context(insight_cards, two_pass=two_pass)
        return RetrievalResult(
            vector_hits=vector_hits,
            insights=insight_cards,
            context=context,
            brief_context=brief_context,
        )

    # ------------------------------------------------------------------
    async def select_context(
        self, question: str, *, default_theme: str | None = None
    ) -> List[VectorEntry]:
        """Select up to ``max_messages`` entries for context with balanced signals.

        Parameters
        ----------
        question:
            Query to search for related context.
        default_theme:
            Theme to assume for entries missing a ``theme`` metadata key.

        Signals used:
        - semantic: vector similarity and BM25 rank
        - recency: exponential decay by half-life
        - authority: cluster PageRank
        Enforces per-chat quotas for diversity.
        """

        max_messages = int(self.context_cfg.get("max_messages", 100))
        per_chat_max = int(self.context_cfg.get("per_chat_max", 40))
        half_life_h = float(self.context_cfg.get("recency_half_life_hours", 24))
        weights = self.context_cfg.get(
            "weights", {"semantic": 0.6, "recency": 0.25, "authority": 0.15}
        )
        w_sem = float(weights.get("semantic", 0.6))
        w_rec = float(weights.get("recency", 0.25))
        w_auth = float(weights.get("authority", 0.15))

        # Active theme filter (optional)
        theme = None
        try:
            if self.get_active_theme:
                theme = self.get_active_theme()
        except Exception:
            theme = None

        # Prepare semantic scores
        q_vec = await self.index._embed(question)
        vector_scores: Dict[str, float] = {}
        if (
            getattr(self.index, "_faiss", None) is not None
            and getattr(self.index._faiss, "index", None) is not None
        ):
            try:
                vector_scores = self.index._faiss.search(
                    q_vec, max_messages * 4, get_id=lambda e: e.chunk_id
                )
            except Exception:
                vector_scores = {}

        bm25_hits = self.index.bm25.search(question, top_k=max(200, max_messages * 4))
        bm25_scores = {cid: score for cid, score in bm25_hits}

        # Authority via cluster PageRank
        entry_to_cluster: Dict[str, str] = {
            m.chunk_id: cid
            for cid, cl in self.cluster_manager.clusters.items()
            for m in getattr(cl, "members", [])
        }
        cluster_pr: Dict[str, float] = {
            cid: cl.pagerank for cid, cl in self.cluster_manager.clusters.items()
        }
        max_pr = max(cluster_pr.values()) if cluster_pr else 0.0

        # Gather candidate entries (union of vector/BM25 top lists), filter by theme
        id_set = set(vector_scores) | set(bm25_scores)
        id_to_entry = {
            e.chunk_id: e
            for e in self.index.entries
            if theme is None or e.metadata.get("theme", default_theme) == theme
        }
        candidates = [id_to_entry[cid] for cid in id_set if cid in id_to_entry]
        if not candidates:
            # Fallback to simple top-k vector search
            return cast(
                List[VectorEntry], await self.index.search(question, top_k=max_messages)
            )

        # Compute recency score with exponential decay
        now = datetime.now(timezone.utc).timestamp()
        half_life_s = max(1.0, half_life_h * 3600.0)

        def recency_score(ts_iso: str | None, default_ts: float) -> float:
            if not ts_iso:
                ts = default_ts
            else:
                try:
                    # Accept both naive and aware ISO strings
                    dt = datetime.fromisoformat(ts_iso)
                    if dt.tzinfo is None:
                        ts = dt.replace(tzinfo=timezone.utc).timestamp()
                    else:
                        ts = dt.timestamp()
                except Exception:
                    ts = default_ts
            age = max(0.0, now - ts)
            return 0.5 ** (age / half_life_s)

        # Score each candidate
        scored: List[Tuple[VectorEntry, float]] = []
        # For semantic normalisation using ranks when similarity not available
        # Map top ranks to descending weights
        vector_rank = {
            cid: i + 1
            for i, cid in enumerate(
                sorted(vector_scores, key=vector_scores.get, reverse=True)
            )
        }
        max_vec_sim = max(vector_scores.values()) if vector_scores else 0.0
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 0.0

        for e in candidates:
            cid = e.chunk_id
            # semantic component combines normalised vector and BM25
            if max_vec_sim > 0.0 and cid in vector_scores:
                vec_norm = vector_scores[cid] / max_vec_sim
            else:
                # rank-based fallback
                r = vector_rank.get(cid, 0)
                vec_norm = 1.0 / r if r > 0 else 0.0
            bm_norm = (bm25_scores.get(cid, 0.0) / max_bm25) if max_bm25 else 0.0
            sem = 0.7 * vec_norm + 0.3 * bm_norm

            # recency component
            date_iso = (
                e.metadata.get("date")
                if isinstance(e.metadata.get("date"), str)
                else None
            )
            rec = recency_score(date_iso, getattr(e, "timestamp", now))

            # authority via cluster PageRank
            cl_id = entry_to_cluster.get(cid)
            pr = (
                (cluster_pr.get(cl_id, 0.0) / max_pr)
                if (cl_id and max_pr > 0.0)
                else 0.0
            )

            score = w_sem * sem + w_rec * rec + w_auth * pr
            scored.append((e, score))

        scored.sort(key=lambda t: t[1], reverse=True)

        # Diversity: enforce per-chat quotas, keep order by score
        per_chat_count: Dict[str, int] = {}
        selected: List[VectorEntry] = []
        for e, _ in scored:
            chat = str(e.metadata.get("chat", ""))
            cnt = per_chat_count.get(chat, 0)
            if cnt >= per_chat_max:
                continue
            per_chat_count[chat] = cnt + 1
            selected.append(e)
            if len(selected) >= max_messages:
                break

        return selected

    async def build_context_lines(self, question: str) -> List[str]:
        """Return formatted context lines, applying summarization if oversized.

        Produces lines in the format "[chat] ISO-date: text". If the combined
        length exceeds ``max_chars`` (from ``context_config`` or default 6000),
        returns compact per-chat and per-cluster mini-summaries instead of all
        raw lines.
        """

        max_chars = int(self.context_cfg.get("max_chars", 6000))
        entries = await self.select_context(question)
        lines: List[str] = []
        for e in entries:
            chat = str(e.metadata.get("chat", ""))
            date = str(e.metadata.get("date", ""))
            text = (e.text or "").strip().replace("\n", " ")
            if not text:
                continue
            lines.append(f"[{chat}] {date}: {text}")
        total_len = sum(len(x) for x in lines)
        if total_len <= max_chars:
            return lines

        # Oversized: build compact summaries by chat and cluster
        by_chat: Dict[str, List[str]] = {}
        for e in entries:
            chat = str(e.metadata.get("chat", "")) or "?"
            text = (e.text or "").strip().replace("\n", " ")
            if not text:
                continue
            by_chat.setdefault(chat, []).append(text)

        # Cluster summaries using manager medoids when available
        entry_to_cluster: Dict[str, str] = {
            m.chunk_id: cid
            for cid, cl in self.cluster_manager.clusters.items()
            for m in getattr(cl, "members", [])
        }
        cluster_to_texts: Dict[str, List[str]] = {}
        for e in entries:
            cid = entry_to_cluster.get(e.chunk_id)
            if not cid:
                continue
            cluster_to_texts.setdefault(cid, []).append(
                (e.text or "").strip().replace("\n", " ")
            )

        mini: List[str] = []
        # Per-chat mini summaries (top few fragments)
        for chat, texts in by_chat.items():
            frags = texts[:3]
            joined = "; ".join(t[:200] for t in frags)
            mini.append(f"[{chat}] summary: {joined}")
        # Per-cluster mini summaries
        for cid, texts in cluster_to_texts.items():
            cl = self.cluster_manager.clusters.get(cid)
            title = (
                (cl.summary or (cl.medoid.text if cl and cl.medoid else "")).strip()
                if cl
                else ""
            )
            if not title:
                title = (texts[0] if texts else "").strip()
            title = title.replace("\n", " ")[:240]
            mini.append(f"[cluster] {title}")

        # Limit the number of lines to avoid bloat
        max_lines = max(10, self.context_cfg.get("summary_max_lines", 30))
        return mini[:max_lines]
