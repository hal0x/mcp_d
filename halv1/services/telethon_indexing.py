from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Dict, List

from index.chunk_indexer import index_message_chunks
from index.raw_storage import _sanitize_component

logger = logging.getLogger(__name__)


class TelethonIndexingMixin:
    """Indexing helpers for :class:`TelethonService`."""

    # ------------------------------------------------------------------
    def _write_last_indexed(self, ts: datetime) -> None:
        try:
            self.index_state_path.parent.mkdir(parents=True, exist_ok=True)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            self.index_state_path.write_text(ts.isoformat(), encoding="utf-8")
        except Exception:  # pragma: no cover - filesystem dependent
            logger.exception("Failed to write last_indexed state")

    # ------------------------------------------------------------------
    async def index_last(self, selected_titles: List[str], n: int) -> int:
        """Index last ``n`` messages for selected chats."""
        allowed_map = self.theme_store.get_chats(self.get_active_theme())
        allowed = set(allowed_map.keys())
        sanitized_selected = [_sanitize_component(t) for t in selected_titles]
        titles = [t for t in sanitized_selected if t in allowed] or list(allowed)

        async with self.tele_lock:
            dialogs = await self.tele_indexer.list_dialogs()
        sanitized_to_entity = {
            _sanitize_component(title): entity for title, entity in dialogs
        }

        errors: List[str] = []
        all_msgs: List[Dict[str, Any]] = []
        for title in titles:
            entity = sanitized_to_entity.get(title)
            display = allowed_map.get(title, title)
            if not entity:
                errors.append(f"Не найден чат: {display}")
                continue
            try:
                async with self.tele_lock:
                    agen = self.tele_indexer.iter_last_messages(entity, limit=n)
                async for m in agen:
                    text = (getattr(m, "message", "") or "").strip()
                    if not text:
                        continue
                    date_dt = (
                        m.date.replace(tzinfo=None)
                        if getattr(m, "date", None)
                        else datetime.now(UTC).replace(tzinfo=None)
                    )
                    chat = title
                    try:
                        self.raw_storage.save(
                            chat, {"date": date_dt.isoformat(), "text": text}
                        )
                        all_msgs.append(
                            {
                                "chat_id": chat,
                                "message_id": str(getattr(m, "id", "")),
                                "timestamp": date_dt,
                                "author": str(getattr(m, "sender_id", "")),
                                "text": text,
                                "reply_to": getattr(m, "reply_to_msg_id", None),
                            }
                        )
                    except Exception as exc:
                        logger.exception(
                            "Failed to persist message for chat %s: %s", display, exc
                        )
                        errors.append(f"Ошибка сохранения для чата '{display}': {exc}")
            except Exception as exc:
                logger.exception(
                    "Failed to iterate messages for chat %s: %s", display, exc
                )
                errors.append(f"Ошибка чтения сообщений чата '{display}': {exc}")

        count, newest = await index_message_chunks(
            all_msgs, self.vector_index, self.get_active_theme()
        )
        if count:
            await self.recompute_pagerank()
        if errors:
            logger.warning("Reindex completed with warnings: %s", "; ".join(errors))
        if newest is not None:
            self._write_last_indexed(newest)
        return count

    # ------------------------------------------------------------------
    async def index_since(
        self,
        selected_titles: List[str],
        since: datetime,
        progress_cb: Callable[[int], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> int:
        """Index messages for selected chats newer than ``since``."""
        allowed_map = self.theme_store.get_chats(self.get_active_theme())
        allowed = set(allowed_map.keys())
        sanitized_selected = [_sanitize_component(t) for t in selected_titles]
        titles = [t for t in sanitized_selected if t in allowed] or list(allowed)

        async with self.tele_lock:
            dialogs = await self.tele_indexer.list_dialogs()
        sanitized_to_entity = {
            _sanitize_component(title): entity for title, entity in dialogs
        }

        all_msgs: List[Dict[str, Any]] = []
        retrieved = 0
        since_naive = since.replace(tzinfo=None)
        for title in titles:
            if is_cancelled and is_cancelled():
                break
            entity = sanitized_to_entity.get(title)
            display = allowed_map.get(title, title)
            if not entity:
                logger.warning("Не найден чат для индексации: %s", display)
                continue
            try:
                async with self.tele_lock:
                    agen = self.tele_indexer.iter_messages_since(entity, since_naive)
                async for m in agen:
                    if is_cancelled and is_cancelled():
                        break
                    text = (getattr(m, "message", "") or "").strip()
                    if not text or text.startswith("/"):
                        continue
                    date_dt = (
                        m.date.replace(tzinfo=None)
                        if getattr(m, "date", None)
                        else datetime.now(UTC).replace(tzinfo=None)
                    )
                    chat = title
                    try:
                        self.raw_storage.save(
                            chat, {"date": date_dt.isoformat(), "text": text}
                        )
                        all_msgs.append(
                            {
                                "chat_id": chat,
                                "message_id": str(getattr(m, "id", "")),
                                "timestamp": date_dt,
                                "author": str(getattr(m, "sender_id", "")),
                                "text": text,
                                "reply_to": getattr(m, "reply_to_msg_id", None),
                            }
                        )
                        retrieved += 1
                        if progress_cb:
                            try:
                                progress_cb(retrieved)
                            except Exception:  # pragma: no cover - callback
                                pass
                    except Exception as exc:
                        logger.exception(
                            "Failed to persist message for chat %s: %s", display, exc
                        )
            except Exception as exc:
                logger.exception(
                    "Failed to iterate messages for chat %s since %s: %s",
                    display,
                    since,
                    exc,
                )

        if is_cancelled and is_cancelled():
            return retrieved

        count, newest = await index_message_chunks(
            all_msgs,
            self.vector_index,
            self.get_active_theme(),
            is_cancelled=is_cancelled,
        )
        if count:
            await self.recompute_pagerank()
        if newest is not None:
            self._write_last_indexed(newest)
        return count

    # ------------------------------------------------------------------
    async def dump_since(
        self,
        days: int,
        progress_cb: Callable[[int], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> int:
        """Fetch recent messages and save them only to raw storage."""
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        allowed_map = self.theme_store.get_chats(self.get_active_theme())
        allowed = set(allowed_map.keys())

        async with self.tele_lock:
            dialogs = await self.tele_indexer.list_dialogs()
        sanitized_to_entity = {
            _sanitize_component(title): entity for title, entity in dialogs
        }

        total = 0
        for title in allowed:
            if is_cancelled and is_cancelled():
                break
            entity = sanitized_to_entity.get(title)
            if not entity:
                continue
            try:
                async with self.tele_lock:
                    agen = self.tele_indexer.iter_messages_since(entity, cutoff)
                # Импортируем функцию извлечения данных
                from utils.message_extractor import extract_message_data
                
                async for m in agen:
                    if is_cancelled and is_cancelled():
                        break
                    text = (getattr(m, "message", "") or "").strip()
                    if not text or text.startswith("/"):
                        continue
                    
                    # Используем новую функцию извлечения расширенной структуры
                    msg_data = extract_message_data(m)
                    
                    self.raw_storage.save(msg_data["chat"], msg_data)
                    total += 1
                    if progress_cb:
                        try:
                            progress_cb(total)
                        except Exception:  # pragma: no cover
                            pass
            except Exception as exc:
                logger.exception("Failed to dump messages for chat %s: %s", title, exc)
        return total

    # ------------------------------------------------------------------
    async def index_dumped(
        self,
        days: int,
        progress_cb: Callable[[int], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> int:
        """Embed previously dumped messages and add them to the vector index."""
        import json

        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        base = self.raw_storage.base_path
        if not base.exists():
            return 0
        items: List[tuple[str, str, Dict[str, str]]] = []
        count = 0
        for chat_dir in base.iterdir():
            if is_cancelled and is_cancelled():
                break
            if not chat_dir.is_dir():
                continue
            chat = chat_dir.name
            for file in chat_dir.glob("*.json"):
                if is_cancelled and is_cancelled():
                    break
                try:
                    file_dt = datetime.fromisoformat(file.stem)
                    if file_dt.tzinfo is not None:
                        file_dt = file_dt.replace(tzinfo=None)
                except Exception:
                    continue
                if file_dt < cutoff:
                    continue
                try:
                    with file.open("r", encoding="utf-8") as f:
                        for idx, line in enumerate(f):
                            if is_cancelled and is_cancelled():
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                msg = json.loads(line)
                            except Exception:
                                continue
                            text = (msg.get("text") or "").strip()
                            if not text or text.startswith("/"):
                                continue
                            date_iso = msg.get("date") or file_dt.isoformat()
                            msg_id = str(
                                msg.get("id") or f"{chat}_{file_dt.isoformat()}_{idx}"
                            )
                            items.append(
                                (
                                    msg_id,
                                    text,
                                    {
                                        "chat": chat,
                                        "date": date_iso,
                                        "theme": self.get_active_theme(),
                                    },
                                ),
                            )
                            count += 1
                            if progress_cb:
                                try:
                                    progress_cb(count)
                                except Exception:  # pragma: no cover
                                    pass
                except Exception as exc:
                    logger.exception("Failed to read raw file %s: %s", file, exc)
        await self.vector_index.add_many(items)
        if items:
            await self.recompute_pagerank()
        return count
