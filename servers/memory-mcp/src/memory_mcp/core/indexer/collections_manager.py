"""Менеджер коллекций Qdrant для индексатора."""

import logging
from typing import Any, Optional

from ...memory.qdrant_collections import QdrantCollectionsManager

logger = logging.getLogger(__name__)


class CollectionsManager:
    """Управление Qdrant коллекциями (sessions, messages, tasks, clusters)."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantCollectionsManager],
        embedding_client: Optional[Any],
        force: bool = False,
    ):
        """Инициализирует менеджер коллекций.

        Args:
            qdrant_manager: Менеджер Qdrant коллекций
            embedding_client: Клиент для генерации эмбеддингов
            force: Принудительно пересоздать существующие коллекции
        """
        self.qdrant_manager = qdrant_manager
        self.embedding_client = embedding_client
        self.force = force

        # Имена коллекций
        self.sessions_collection: Optional[str] = None
        self.messages_collection: Optional[str] = None
        self.tasks_collection: Optional[str] = None
        self.clusters_collection: Optional[str] = None
        self.progress_collection: Optional[str] = None

    def _get_embedding_dimension(self) -> Optional[int]:
        """Получить размерность эмбеддингов из клиента."""
        if not self.embedding_client:
            return None

        # Пытаемся получить размерность из клиента
        if (
            hasattr(self.embedding_client, "_embedding_dimension")
            and self.embedding_client._embedding_dimension
        ):
            return self.embedding_client._embedding_dimension

        # Если размерность ещё не определена, делаем тестовый запрос
        try:
            import asyncio
            import nest_asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Если цикл уже запущен, создаём новый
                try:
                    nest_asyncio.apply()
                except ImportError:
                    pass

            async def _get_dim():
                async with self.embedding_client:
                    test_embedding = await self.embedding_client.get_embedding("test")
                    return len(test_embedding) if test_embedding else None

            try:
                dimension = asyncio.run(_get_dim())
                if dimension:
                    self.embedding_client._embedding_dimension = dimension
                return dimension
            except RuntimeError:
                return None
        except Exception as e:
            logger.debug(f"Не удалось определить размерность эмбеддингов: {e}")
            return None

    def _check_and_recreate_collection(
        self, collection_name: str, description: str, force_recreate: bool = False
    ) -> Optional[str]:
        """Проверить коллекцию Qdrant и пересоздать при несоответствии размерности."""
        if not self.qdrant_manager or not self.qdrant_manager.available():
            logger.warning(
                f"Qdrant недоступен, коллекция {collection_name} не будет создана"
            )
            return None

        # Обновляем размерность в менеджере, если нужно
        expected_dimension = self._get_embedding_dimension()
        if (
            expected_dimension
            and expected_dimension != self.qdrant_manager.vector_size
        ):
            logger.info(
                f"Обновление размерности Qdrant менеджера: "
                f"{self.qdrant_manager.vector_size} -> {expected_dimension}"
            )
            self.qdrant_manager.vector_size = expected_dimension

        # Создаем/проверяем коллекцию через менеджер
        if self.qdrant_manager.ensure_collection(
            collection_name, force_recreate=force_recreate
        ):
            count = self.qdrant_manager.count(collection_name)
            logger.info(
                f"Коллекция Qdrant {collection_name} готова ({count} записей)"
            )
            return collection_name  # Возвращаем имя коллекции вместо объекта
        else:
            logger.error(f"Не удалось создать коллекцию Qdrant {collection_name}")
            return None

    def initialize_collections(self):
        """Инициализирует коллекции Qdrant с проверкой размерности эмбеддингов."""
        if not self.qdrant_manager or not self.qdrant_manager.available():
            logger.warning("Qdrant недоступен, коллекции не будут инициализированы")
            self.sessions_collection = None
            self.messages_collection = None
            self.tasks_collection = None
            self.clusters_collection = None
            self.progress_collection = None
            return

        try:
            self.sessions_collection = self._check_and_recreate_collection(
                "chat_sessions",
                "Саммаризации сессий для векторного поиска (L1)",
                force_recreate=self.force,
            )

            self.messages_collection = self._check_and_recreate_collection(
                "chat_messages",
                "Сообщения с контекстом для уточняющего поиска (L2)",
                force_recreate=self.force,
            )

            self.tasks_collection = self._check_and_recreate_collection(
                "chat_tasks",
                "Action Items из сессий",
                force_recreate=self.force,
            )

            self.clusters_collection = self._check_and_recreate_collection(
                "session_clusters",
                "Тематические кластеры сессий",
                force_recreate=self.force,
            )

            # indexing_progress теперь хранится в SQLite через IndexingJobTracker
            # Qdrant не используется для прогресса
            self.progress_collection = None
            logger.info("Коллекции Qdrant инициализированы")

        except Exception as e:
            logger.error(f"Ошибка при инициализации коллекций: {e}")
            raise

