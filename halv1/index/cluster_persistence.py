from __future__ import annotations

# mypy: ignore-errors
import json
import time
from typing import Dict, Iterable, List

from .cluster_model import Cluster
from .insight_store import InsightStore
from .vector_index import VectorEntry


class ClusterPersistenceMixin:
    """Persistence helpers for :class:`ClusterManager`."""

    # ------------------------------------------------------------------
    def save(self, store: InsightStore) -> None:
        """Persist all clusters to ``store``."""
        payload: Dict[str, Dict[str, object]] = {}
        for cid, cluster in self.clusters.items():
            medoid = cluster.medoid.__dict__ if cluster.medoid else None
            members = [m.__dict__ for m in cluster.members]
            timeline_entries = [
                m.__dict__ if isinstance(m, VectorEntry) else dict(m)
                for m in getattr(cluster, "timeline", [])
            ]
            if not timeline_entries and medoid:
                timeline_entries = [medoid]
            payload[cid] = {
                "centroid": list(cluster.centroid),
                "medoid": medoid,
                "summary": cluster.summary,
                "members": members,
                "pagerank": cluster.pagerank,
                "freshness_ts": cluster.freshness_ts,
                "freshness": cluster.freshness,
                "source_quality": cluster.source_quality,
                "timeline": timeline_entries,
            }
        store.set_clusters(payload)

    # ------------------------------------------------------------------
    def load(self, store: InsightStore) -> None:
        """Load clusters from ``store`` replacing current state."""
        self.clusters.clear()
        self._entries.clear()
        data = store.get_clusters()
        for cid, info in data.items():
            medoid_data = info.get("medoid")
            members_data = info.get("members", [])
            members = [VectorEntry(**m) for m in members_data if isinstance(m, dict)]
            medoid: VectorEntry | None = None
            if medoid_data:
                for m in members:
                    if m.chunk_id == medoid_data.get("chunk_id"):
                        medoid = m
                        break
                if medoid is None:
                    medoid = VectorEntry(**medoid_data)
                    members.append(medoid)
            timeline_data = info.get("timeline", [])
            timeline: List[VectorEntry] = []
            for item in timeline_data:
                if isinstance(item, dict):
                    try:
                        timeline.append(VectorEntry(**item))
                    except Exception:
                        continue
            cluster = Cluster(
                id=cid,
                members=members,
                centroid=list(info.get("centroid", [])),
                medoid=medoid,
                summary=info.get("summary"),
                pagerank=float(info.get("pagerank", 0.0)),
                freshness_ts=float(info.get("freshness_ts", 0.0)),
                freshness=float(info.get("freshness", 0.0)),
                source_quality=float(info.get("source_quality", 0.0)),
                timeline=timeline,
            )
            self.clusters[cid] = cluster
            self._entries.extend(members)
            self._centroid_history[cid] = [(time.time(), list(cluster.centroid))]
        self._rebuild_index()

    # ------------------------------------------------------------------
    def log_missing_facts(self, facts: Iterable[str]) -> None:
        """Append ``facts`` to the missing facts log."""
        filtered = [f.strip() for f in facts if f and f.strip()]
        if not filtered:
            return
        self.missing_facts_path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.time()
        with self.missing_facts_path.open("a", encoding="utf-8") as fh:
            for fact in filtered:
                record = {"ts": ts, "fact": fact}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
