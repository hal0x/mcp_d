"""Управление прогрессом индексации."""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from zoneinfo import ZoneInfo

from ...memory.qdrant_collections import QdrantCollectionsManager
from ...utils.datetime_utils import parse_datetime_utc
from ...utils.naming import slugify

logger = logging.getLogger(__name__)


class ProgressManager:
    """Отслеживание прогресса индексации."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantCollectionsManager],
        messages_collection: Optional[str],
        progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ):
        """Инициализирует менеджер прогресса.

        Args:
            qdrant_manager: Менеджер Qdrant коллекций
            messages_collection: Имя коллекции сообщений
            progress_callback: Callback функция для отслеживания прогресса
        """
        self.qdrant_manager = qdrant_manager
        self.messages_collection = messages_collection
        self.progress_callback = progress_callback

    def get_last_indexed_date(self, chat_name: str) -> Optional[datetime]:
        """
        Получить дату последнего проиндексированного сообщения для чата.

        Args:
            chat_name: Название чата

        Returns:
            Дата последнего проиндексированного сообщения или None
        """
        try:
            # Получаем последнее сообщение из коллекции messages для данного чата
            if self.qdrant_manager and self.messages_collection:
                result = self.qdrant_manager.get(
                    collection_name=self.messages_collection,
                    where={"chat": chat_name},
                    limit=1,
                    order_by="timestamp:desc" if hasattr(self.qdrant_manager, "get") else None,
                )

                if result and result.get("ids") and result.get("metadatas"):
                    # Пытаемся получить timestamp из метаданных
                    metadata = result["metadatas"][0]
                    timestamp_str = metadata.get("timestamp") or metadata.get("date_utc")
                    if timestamp_str:
                        return parse_datetime_utc(timestamp_str, use_zoneinfo=True)

                    # Если нет timestamp, пытаемся получить из документа или других полей
                    # Для обратной совместимости проверяем старый формат
                    last_date_str = metadata.get("last_indexed_date")
                    if last_date_str:
                        return parse_datetime_utc(last_date_str, use_zoneinfo=True)
        except Exception as e:
            logger.debug(f"Не удалось получить прогресс для {chat_name}: {e}")

        return None

    def save_indexing_progress(
        self,
        chat_name: str,
        last_message_date: datetime,
        messages_count: int,
        sessions_count: int,
    ):
        """
        Сохранить прогресс индексации для чата.

        Примечание: Прогресс теперь хранится в IndexingJobTracker (JSON файл),
        но этот метод оставлен для обратной совместимости и может использоваться
        для сохранения метаданных в Qdrant.

        Args:
            chat_name: Название чата
            last_message_date: Дата последнего проиндексированного сообщения
            messages_count: Количество проиндексированных сообщений
            sessions_count: Количество созданных сессий
        """
        # Прогресс теперь хранится в IndexingJobTracker через callback
        # Этот метод оставлен для обратной совместимости
        logger.debug(
            f"Прогресс для {chat_name}: "
            f"последнее сообщение {last_message_date.isoformat()}, "
            f"всего сообщений {messages_count}, сессий {sessions_count}"
        )

    def count_indexed_messages_in_chat(self, chat_name: str) -> int:
        """
        Подсчитывает количество уже проиндексированных сообщений в чате.

        Args:
            chat_name: Название чата

        Returns:
            int: количество проиндексированных сообщений
        """
        try:
            # Получаем все сообщения из коллекции chat_messages для данного чата
            existing_messages = None
            if self.qdrant_manager and self.messages_collection:
                existing_messages = self.qdrant_manager.get(
                    collection_name=self.messages_collection,
                    where={"chat": chat_name},
                )

            if existing_messages and existing_messages.get("ids") is not None:
                message_count = len(existing_messages["ids"])
                logger.info(
                    f"Найдено {message_count} уже проиндексированных сообщений в чате {chat_name}"
                )
                return message_count
            else:
                logger.info(f"В чате {chat_name} нет проиндексированных сообщений")
                return 0

        except Exception as e:
            logger.warning(
                f"Ошибка при подсчете проиндексированных сообщений для чата {chat_name}: {e}"
            )
            return 0

    def call_progress_callback(
        self, job_id: Optional[str], event: str, data: Dict[str, Any]
    ) -> None:
        """Вызвать callback прогресса, если он установлен."""
        if self.progress_callback and job_id:
            try:
                self.progress_callback(job_id, event, data)
            except Exception as e:
                logger.warning(f"Ошибка при вызове progress_callback: {e}")

