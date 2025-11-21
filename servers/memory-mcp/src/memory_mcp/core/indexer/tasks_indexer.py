"""Индексация задач (action items)."""

import logging
from typing import Any, Dict, Optional

from ...memory.qdrant_collections import QdrantCollectionsManager

logger = logging.getLogger(__name__)


class TasksIndexer:
    """Индексация action items."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantCollectionsManager],
        tasks_collection: Optional[str],
        embedding_client: Any,
    ):
        """Инициализирует индексатор задач.

        Args:
            qdrant_manager: Менеджер Qdrant коллекций
            tasks_collection: Имя коллекции задач
            embedding_client: Клиент для генерации эмбеддингов
        """
        self.qdrant_manager = qdrant_manager
        self.tasks_collection = tasks_collection
        self.embedding_client = embedding_client

    async def index_tasks(self, summary: Dict[str, Any]) -> int:
        """
        Индексация Action Items в коллекцию tasks

        Args:
            summary: Саммаризация сессии

        Returns:
            Количество проиндексированных задач
        """
        actions = summary.get("actions", [])
        session_id = summary["session_id"]
        chat = summary.get("meta", {}).get("chat_name", "")

        indexed_count = 0

        for i, action in enumerate(actions):
            confidence = action.get("confidence", 0.8)
            if confidence < 0.6:
                continue

            try:
                # Текст задачи
                task_text = action.get("text", "")
                if not task_text:
                    continue

                # Генерируем эмбеддинг
                async with self.embedding_client:
                    embeddings = await self.embedding_client.generate_embeddings(
                        [task_text]
                    )
                    embedding = embeddings[0]

                # Подготавливаем метаданные
                task_id = f"{session_id}-T{i+1:02d}"
                metadata = {
                    "task_id": task_id,
                    "session_id": session_id,
                    "chat": chat,
                    "owner": action.get("owner", ""),
                    "due": action.get("due", ""),
                    "priority": action.get("priority", "normal"),
                    "confidence": confidence,
                    "msg_id": action.get("msg_id", ""),
                    "topic_title": action.get("topic_title", ""),
                }

                # Добавляем в коллекцию
                if self.qdrant_manager and self.tasks_collection:
                    self.qdrant_manager.upsert(
                        collection_name=self.tasks_collection,
                        ids=[task_id],
                        documents=[task_text],
                        embeddings=[embedding],
                        metadatas=[metadata],
                    )

                indexed_count += 1

            except Exception as e:
                logger.error(
                    f"Ошибка при индексации задачи {i} в сессии {session_id}: {e}"
                )
                continue

        logger.info(
            f"Tasks: Проиндексировано {indexed_count} задач из сессии {session_id}"
        )
        return indexed_count

