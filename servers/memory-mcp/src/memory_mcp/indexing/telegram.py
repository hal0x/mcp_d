"""Telegram chat indexer implementing the unified BaseIndexer contract."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from ..analysis.quality.utils.data_processor import load_chats_from_directory
from ..utils.processing.datetime_utils import parse_datetime_utc
from .base import Attachment, BaseIndexer, IndexingStats, MemoryRecord

logger = logging.getLogger(__name__)


def _parse_datetime(value: str | None) -> datetime:
    """Парсинг даты (использует общую утилиту)."""
    return parse_datetime_utc(value, default=datetime.now(timezone.utc))


class TelegramIndexer(BaseIndexer):
    """Index raw Telegram exports into normalized memory records."""

    name = "telegram-chat-indexer"

    def __init__(
        self,
        chats_dir: str | Path = "chats",
        *,
        selected_chats: Iterable[str] | None = None,
        source_name: str = "telegram",
    ) -> None:
        self.chats_dir = Path(chats_dir)
        self.selected_chats = list(selected_chats) if selected_chats else None
        self.source_name = source_name
        self._prepared = False
        self._loaded: Dict[str, List[dict]] = {}
        self._stats = IndexingStats()

    def prepare(self) -> None:
        if self._prepared:
            return
        logger.info("Загрузка чатов из %s", self.chats_dir)
        self._loaded = load_chats_from_directory(
            self.chats_dir, self.selected_chats
        )
        self._stats.sources_processed = len(self._loaded)
        self._prepared = True

    def iter_records(self) -> Iterator[MemoryRecord]:
        if not self._prepared:
            self.prepare()

        for chat_name, messages in self._loaded.items():
            chat_tag = chat_name.strip() or "unknown-chat"
            for idx, msg in enumerate(messages):
                metadata = msg.get("metadata", {})
                content = msg.get("text", "").strip()
                if not content:
                    continue

                record_id = metadata.get("message_id")
                if record_id is None:
                    record_id = f"{metadata.get('chat', chat_tag)}-{metadata.get('sender_id')}-{idx}"

                timestamp = _parse_datetime(msg.get("date"))
                record = MemoryRecord(
                    record_id=f"{self.source_name}:{chat_tag}:{record_id}",
                    source=self.source_name,
                    content=content,
                    timestamp=timestamp,
                    author=metadata.get("sender"),
                    tags=[chat_tag],
                    entities=[],
                    attachments=self._extract_attachments(metadata),
                    metadata=metadata,
                )
                self._stats.records_indexed += 1
                yield record

    def _extract_attachments(self, metadata: dict) -> List[Attachment]:
        attachments_raw = metadata.get("attachments")
        if not attachments_raw:
            return []
        attachments: List[Attachment] = []
        for item in attachments_raw:
            attachments.append(
                Attachment(
                    type=item.get("type", "file"),
                    uri=item.get("href") or item.get("file"),
                    text=item.get("text"),
                    metadata={k: v for k, v in item.items() if k not in {"type", "href", "file", "text"}},
                )
            )
        return attachments

    def finalize(self) -> IndexingStats:
        return self._stats

    def close(self) -> None:
        self._loaded.clear()
        self._prepared = False
