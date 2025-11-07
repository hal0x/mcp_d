from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import yaml

from index.summarizer import Summarizer
from index.theme_store import ThemeStore
from index.vector_index import VectorIndex
from tasks.scheduler import TaskScheduler

from .telethon_service import TelethonService

logger = logging.getLogger(__name__)


class SummaryService:
    """Periodic summarization of indexed messages."""

    def __init__(
        self,
        summarizer: Summarizer,
        vector_index: VectorIndex,
        theme_store: ThemeStore,
        telethon_service: Optional[TelethonService],
        scheduler: TaskScheduler,
        summary_interval_file: Path,
        get_active_theme: Callable[[], str],
        get_bot: Callable[[], Optional[object]],
        summary_chat_id: Optional[int],
        summary_interval: int,
        user_name: Optional[str] = None,
        timezone: Optional[str] = None,
        config_path: Path = Path("config/settings.yaml"),
    ) -> None:
        self.summarizer = summarizer
        self.vector_index = vector_index
        self.theme_store = theme_store
        self.telethon_service = telethon_service
        self.scheduler = scheduler
        self.summary_interval_file = summary_interval_file
        self.get_active_theme = get_active_theme
        self.get_bot = get_bot
        self.summary_chat_id = summary_chat_id
        self.summary_interval = summary_interval
        self.scheduler_task: Optional[asyncio.Task] = None

        if user_name is None or timezone is None:
            try:
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                summary_cfg = cfg.get("summary", {})
                user_name = user_name or summary_cfg.get("user_name", "@hal0x")
                timezone = timezone or summary_cfg.get("timezone", "UTC")
            except Exception:
                logger.exception("Failed to load summary config")
                user_name = user_name or "@user"
                timezone = timezone or "UTC"
        self.user_name = user_name
        self.timezone = timezone

    async def start(self) -> None:
        self.scheduler.add_periodic(self.hourly_summary, self.summary_interval)
        
        # Проверяем, есть ли работающий event loop
        try:
            asyncio.get_running_loop()
            self.scheduler_task = asyncio.create_task(self.scheduler.run())
        except RuntimeError:
            # Если нет работающего loop, создаем пустую задачу
            self.scheduler_task = None
            logger.warning("No running event loop, skipping summary scheduler task")

    async def stop(self) -> None:
        await self.scheduler.stop()
        if self.scheduler_task:
            self.scheduler_task.cancel()
            await asyncio.gather(self.scheduler_task, return_exceptions=True)

    async def summarize_interval(self, hours: int) -> str:
        logger.info(
            "Summary requested for %sh, active theme: %s",
            hours,
            self.get_active_theme(),
        )
        cutoff = datetime.now(UTC) - timedelta(hours=hours)

        has_recent = False
        
        def _parse_iso_aware(s: str, default: datetime) -> datetime:
            try:
                dt = datetime.fromisoformat(s)
            except Exception:
                # Для невалидных дат возвращаем далёкое прошлое, чтобы исключить запись
                return datetime(1970, 1, 1, tzinfo=UTC)
            # Normalize to timezone-aware (assume UTC if naive)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
        for e in self.vector_index.entries:
            dt = _parse_iso_aware(e.metadata.get("date", cutoff.isoformat()), cutoff)
            if dt >= cutoff and e.metadata.get("theme") == self.get_active_theme():
                has_recent = True
                break
        if not has_recent and self.telethon_service:
            try:
                await self.telethon_service.index_since(
                    list(self.theme_store.get_chats(self.get_active_theme()).values()),
                    cutoff,
                )
            except Exception as exc:
                logger.exception("Backfill for summary failed: %s", exc)

        lines = []
        for e in self.vector_index.entries:
            dt = _parse_iso_aware(e.metadata.get("date", cutoff.isoformat()), cutoff)
            theme = e.metadata.get("theme", "unknown")
            if dt >= cutoff and theme == self.get_active_theme():
                chat = e.metadata.get("chat", "?")
                msg_id = e.chunk_id
                lines.append(f"{msg_id}|{chat}|?|{dt.isoformat()}|{e.text}")
        if not lines:
            return ""
        block = "\n".join(lines)
        return await asyncio.to_thread(
            self.summarizer.summarize_as_agent,
            mode="summary",
            user_name=self.user_name,
            theme=self.get_active_theme(),
            timezone=self.timezone,
            window_start=cutoff.isoformat(),
            window_end=datetime.now(UTC).isoformat(),
            now_iso=datetime.now(UTC).isoformat(),
            messages_block=block,
        )

    async def hourly_summary(self) -> None:
        try:
            cutoff = datetime.now(UTC) - timedelta(hours=1)
            lines = []
            def _parse_iso_aware(s: str, default: datetime) -> datetime:
                try:
                    dt = datetime.fromisoformat(s)
                except Exception:
                    return datetime(1970, 1, 1, tzinfo=UTC)
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

            for e in self.vector_index.entries[-300:]:
                dt = _parse_iso_aware(e.metadata.get("date", cutoff.isoformat()), cutoff)
                if dt >= cutoff and e.metadata.get("theme") == self.get_active_theme():
                    chat = e.metadata.get("chat", "?")
                    msg_id = e.chunk_id
                    lines.append(f"{msg_id}|{chat}|?|{dt.isoformat()}|{e.text}")
            if not lines:
                return
            block = "\n".join(lines)
            summary = await asyncio.to_thread(
                self.summarizer.summarize_as_agent,
                mode="summary",
                user_name=self.user_name,
                theme=self.get_active_theme(),
                timezone=self.timezone,
                window_start=cutoff.isoformat(),
                window_end=datetime.now(UTC).isoformat(),
                now_iso=datetime.now(UTC).isoformat(),
                messages_block=block,
            )
            bot = self.get_bot()
            if summary and bot and getattr(bot, "app", None) and self.summary_chat_id:
                try:
                    await bot.app.bot.send_message(
                        chat_id=self.summary_chat_id, text=summary
                    )
                except Exception as exc:
                    logger.exception("Failed to send summary via bot: %s", exc)
        except Exception as exc:  # pragma: no cover - logging
            logger.exception("Hourly summary failed: %s", exc)

    async def set_summary_interval(self, hours: int) -> str:
        self.summary_interval = hours * 3600
        try:
            self.summary_interval_file.parent.mkdir(parents=True, exist_ok=True)
            self.summary_interval_file.write_text(
                yaml.safe_dump(
                    {"summary_interval_seconds": self.summary_interval},
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
        except Exception:  # pragma: no cover - filesystem
            logger.exception("Failed to save summary interval override")
        await self.stop()
        self.scheduler.add_periodic(self.hourly_summary, self.summary_interval)
        self.scheduler_task = asyncio.create_task(self.scheduler.run())
        return f"Интервал сводки обновлён: {hours} ч."
