from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from index.chronicle import Chronicle
from index.cluster_manager import ClusterManager
from index.insight_store import InsightStore
from index.summarizer import Summarizer
from tasks.scheduler import TaskScheduler

logger = logging.getLogger(__name__)


class ChronicleService:
    """Periodic topic overview based on stored clusters."""

    def __init__(
        self,
        summarizer: Summarizer,
        insight_store: InsightStore,
        scheduler: TaskScheduler,
        get_bot: Callable[[], Optional[Any]],
        summary_chat_id: Optional[int],
        days_interval: int,
    ) -> None:
        self.summarizer = summarizer
        self.insight_store = insight_store
        self.scheduler = scheduler
        self.get_bot = get_bot
        self.summary_chat_id = summary_chat_id
        self.interval = int(days_interval * 86400)
        self.scheduler_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        self.scheduler.add_periodic(self.publish_chronicle, self.interval)
        
        # Проверяем, есть ли работающий event loop
        try:
            asyncio.get_running_loop()
            self.scheduler_task = asyncio.create_task(self.scheduler.run())
        except RuntimeError:
            # Если нет работающего loop, создаем пустую задачу
            self.scheduler_task = None
            logger.warning("No running event loop, skipping chronicle scheduler task")

    async def stop(self) -> None:
        await self.scheduler.stop()
        if self.scheduler_task:
            self.scheduler_task.cancel()
            await asyncio.gather(self.scheduler_task, return_exceptions=True)

    async def publish_chronicle(self) -> None:
        try:
            manager = ClusterManager()
            manager.load(self.insight_store)
            if not manager.clusters:
                return
            chronicle = Chronicle(manager, self.summarizer)
            slices = await asyncio.to_thread(chronicle.build)
            if not slices:
                return
            body = "Хроника событий:\n" + chronicle.render(slices)
            bot = self.get_bot()
            if bot and getattr(bot, "app", None) and self.summary_chat_id:
                await bot.app.bot.send_message(chat_id=self.summary_chat_id, text=body)
        except Exception as exc:  # pragma: no cover - logging
            logger.exception("Chronicle publication failed: %s", exc)
