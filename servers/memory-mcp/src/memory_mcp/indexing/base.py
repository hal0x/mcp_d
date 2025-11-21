"""Base abstractions for data indexers that feed the memory pipeline."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from ..models.memory import Attachment, MemoryRecord


@dataclass(slots=True)
class IndexingStats:
    """Statistics returned by indexers after processing."""

    records_indexed: int = 0
    sources_processed: int = 0
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class IndexingError(RuntimeError):
    """Raised when an indexer fails to produce records."""


class BaseIndexer(abc.ABC):
    """Abstract base class for data source indexers."""

    name: str = "base-indexer"

    def __enter__(self) -> "BaseIndexer":
        self.prepare()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def prepare(self) -> None:
        """Perform any setup required before iterating records."""

    @abc.abstractmethod
    def iter_records(self) -> Iterator[MemoryRecord]:
        """Yield normalized memory records from the data source."""

    def finalize(self) -> IndexingStats:
        """Return processing statistics after iteration completes."""
        return IndexingStats()

    def close(self) -> None:
        """Release any resources associated with the indexer."""

    # Utility helper -------------------------------------------------
    @staticmethod
    def ensure_iterable(value: Iterable[Any] | None) -> Iterable[Any]:
        return value or ()
