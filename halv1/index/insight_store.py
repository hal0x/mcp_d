"""Persistent storage for derived cluster insights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from core.utils.json_io import load_json, save_json


class InsightStore:
    """Store clusters with their summaries and members in a JSON file.

    The data structure is ``{cluster_id: {centroid, medoid, summary, members}}``
    where ``medoid`` and ``members`` are serialised :class:`VectorEntry` dicts.
    Extra keys are preserved to allow future extensions.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        # cluster_id -> {centroid: [], medoid: {}, summary: str, members: []}
        self._clusters: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                data: Dict[str, Any] = load_json(self.path, {})
            except Exception:  # pragma: no cover - logged then re-raised
                logging.getLogger(__name__).exception(
                    "Failed to load insights from %s", self.path
                )
                raise
            if isinstance(data, dict):
                for cid, info in data.items():
                    if isinstance(info, dict):
                        base = {
                            "centroid": list(info.get("centroid", [])),
                            "medoid": info.get("medoid"),
                            "summary": info.get("summary"),
                            "members": list(info.get("members", [])),
                        }
                        extras = {k: v for k, v in info.items() if k not in base}
                        base.update(extras)
                        self._clusters[str(cid)] = base

    # ----------------------------- persistence -----------------------------
    def _save(self) -> None:
        save_json(self.path, self._clusters, indent=2)

    # ------------------------------- queries --------------------------------
    def list_clusters(self) -> List[str]:
        return sorted(self._clusters.keys())

    def get_cluster(self, cluster_id: str) -> Dict[str, Any] | None:
        """Return a copy of cluster data or ``None`` if missing.

        Mutating the returned mapping does **not** persist changes. Call
        :meth:`set_cluster` to save modifications.
        """

        data = self._clusters.get(cluster_id)
        return dict(data) if data is not None else None

    def get_clusters(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self._clusters.items()}

    def set_cluster(self, cluster_id: str, data: Dict[str, Any]) -> None:
        self._clusters[cluster_id] = dict(data)
        self._save()

    def set_clusters(self, clusters: Dict[str, Dict[str, Any]]) -> None:
        self._clusters = {str(k): dict(v) for k, v in clusters.items()}
        self._save()

    def delete_cluster(self, cluster_id: str) -> bool:
        if cluster_id in self._clusters:
            del self._clusters[cluster_id]
            self._save()
            return True
        return False
