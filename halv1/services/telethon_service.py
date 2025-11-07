from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from index.cluster_manager import ClusterManager
from index.raw_storage import RawStorage
from index.telethon_indexer import TelethonIndexer
from index.theme_store import ThemeStore
from index.vector_index import VectorIndex

from .telethon_indexing import TelethonIndexingMixin

logger = logging.getLogger(__name__)


class TelethonService(TelethonIndexingMixin):
    """Service wrapper around :class:`TelethonIndexer`.

    Encapsulates operations for fetching messages via Telethon and
    persisting them into raw storage/vector index.
    """

    CACHE_EXPIRY_MINUTES = 30

    def __init__(
        self,
        tele_indexer: TelethonIndexer,
        raw_storage: RawStorage,
        vector_index: VectorIndex,
        cluster_manager: ClusterManager,
        theme_store: ThemeStore,
        index_state_path: Path,
        get_active_theme: Callable[[], str],
    ) -> None:
        self.tele_indexer = tele_indexer
        self.raw_storage = raw_storage
        self.vector_index = vector_index
        self.cluster_manager = cluster_manager
        self.theme_store = theme_store
        self.index_state_path = index_state_path
        self.get_active_theme = get_active_theme

        # Создаем файл last_indexed.txt если его нет
        try:
            self.index_state_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.index_state_path.exists():
                self.index_state_path.write_text("", encoding="utf-8")
                logger.info(f"Created last_indexed.txt at {self.index_state_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize last_indexed.txt: {e}")

        self.tele_lock = asyncio.Lock()
        self._chat_cache: Optional[List[str]] = None
        self._cache_timestamp: Optional[datetime] = None

    async def ensure_connected(self) -> None:
        """Ensure Telethon client is connected.

        Any connection error is logged but not raised to avoid breaking
        the rest of the application when Telethon credentials are
        missing or invalid.
        """

        try:
            await self.tele_indexer.ensure_connected()
        except Exception as exc:  # pragma: no cover - network/credential dependent
            logger.error("Telethon unavailable: %s", exc)

    async def recompute_pagerank(self) -> None:
        """Recompute PageRank scores using the cluster manager."""
        await asyncio.to_thread(self.cluster_manager.recompute_pagerank)

    # ------------------------------------------------------------------
    # Chat listing and caching
    # ------------------------------------------------------------------
    async def list_chats(self, force_refresh: bool = False) -> List[str]:
        """Return a list of chat titles available via Telethon."""

        now = datetime.now(UTC)
        cache_valid = (
            self._chat_cache is not None
            and self._cache_timestamp is not None
            and (now - self._cache_timestamp).total_seconds()
            < self.CACHE_EXPIRY_MINUTES * 60
        )
        if force_refresh or not cache_valid:
            logger.info("Refreshing chat cache from Telegram API")
            async with self.tele_lock:
                dialogs = await self.tele_indexer.list_dialogs()
            self._chat_cache = [title for title, _ in dialogs]
            self._cache_timestamp = now
            logger.info("Chat cache updated with %d chats", len(self._chat_cache))
        else:
            logger.info("Using cached chat list with %d chats", len(self._chat_cache))
        return self._chat_cache.copy() if self._chat_cache else []

    async def refresh_chat_cache(self) -> bool:
        try:
            await self.list_chats(force_refresh=True)
            return True
        except Exception as exc:  # pragma: no cover - network
            logger.exception("Failed to refresh chat cache: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Authorization helpers
    # ------------------------------------------------------------------
    async def telethon_is_authorized(self) -> bool:
        try:
            await self.tele_indexer.assert_authorized()
            return True
        except Exception:
            return False

    async def telethon_auth_request_code(self, phone: str) -> None:
        async with self.tele_lock:
            await self.tele_indexer.request_code(phone)

    async def telethon_auth_sign_in(
        self, phone: str, code: str, password: Optional[str]
    ) -> Dict[str, str]:
        async with self.tele_lock:
            return await self.tele_indexer.sign_in(phone, code, password)
