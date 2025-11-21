"""Модели данных для работы с памятью."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass(slots=True)
class Attachment:
    """Represents an additional payload attached to a memory record."""

    type: str
    uri: str | None = None
    text: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryRecord:
    """Normalized memory record produced by an indexer."""

    record_id: str
    source: str
    content: str
    timestamp: datetime
    author: str | None = None
    tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


