"""L1 индексация: сессии с саммаризацией."""

import logging
from typing import Any, Dict, Optional

from ...memory.qdrant_collections import QdrantCollectionsManager
from ...utils.url_validator import validate_embedding_text

logger = logging.getLogger(__name__)


class L1Indexer:
    """Индексация сессий с саммаризацией."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantCollectionsManager],
        sessions_collection: Optional[str],
        embedding_client: Any,
        vector_store: Optional[Any],
        ingestor: Optional[Any],
        graph: Optional[Any],
        collections_manager: Optional[Any] = None,
    ):
        """Инициализирует L1 индексатор.

        Args:
            qdrant_manager: Менеджер Qdrant коллекций
            sessions_collection: Имя коллекции сессий
            embedding_client: Клиент для генерации эмбеддингов
            vector_store: VectorStore для сохранения эмбеддингов
            ingestor: MemoryIngestor для синхронизации с графом
            graph: Граф памяти
            collections_manager: Менеджер коллекций (для пересоздания при ошибках)
        """
        self.qdrant_manager = qdrant_manager
        self.sessions_collection = sessions_collection
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.ingestor = ingestor
        self.graph = graph
        self.collections_manager = collections_manager

    async def index_session(self, summary: Dict[str, Any]) -> None:
        """
        Индексация сессии на уровне L1 (саммари + E1).

        Args:
            summary: Саммаризация сессии
        """
        session_id = summary["session_id"]

        meta = summary.get("meta", {})

        topics_text = "\n".join(
            f"{topic.get('title', '')}: {topic.get('summary', '')}"
            for topic in summary.get("topics", [])
        )
        claims_text = "\n".join(
            claim.get("summary", "") for claim in summary.get("claims", [])
        )
        discussion_text = "\n".join(
            item.get("quote", "") for item in summary.get("discussion", [])
        )
        entities_text = ", ".join(summary.get("entities", []))

        embedding_text = (
            f"Topics:\n{topics_text}\n\n"
            f"Claims:\n{claims_text}\n\n"
            f"Discussion:\n{discussion_text}\n\n"
            f"Entities: {entities_text}"
        )

        # Валидируем текст перед отправкой в эмбеддинг
        embedding_text, replaced_urls = validate_embedding_text(embedding_text)

        # Логируем замененные URL если они есть
        if replaced_urls:
            logger.warning(
                f"В сессии {session_id} заменены некорректные URL: {replaced_urls}"
            )

        # Генерируем эмбеддинг
        async with self.embedding_client:
            embeddings = await self.embedding_client.generate_embeddings([embedding_text])
            embedding = embeddings[0]

        # Подготавливаем метаданные
        metadata = {
            "session_id": session_id,
            "chat": meta.get("chat_name", ""),
            "profile": meta.get("profile", ""),
            "start_time_utc": meta.get("start_time_utc", ""),
            "end_time_utc": meta.get("end_time_utc", ""),
            "time_span": meta.get("time_span", ""),
            "message_count": meta.get("messages_total", 0),
            "dominant_language": meta.get("dominant_language", "unknown"),
            "chat_mode": meta.get("chat_mode", "group"),
            "topics_count": len(summary.get("topics", [])),
            "claims_count": len(summary.get("claims", [])),
            "quality_score": summary.get("quality", {}).get("score", 0),
            "replaced_urls": ",".join(replaced_urls) if replaced_urls else "",
        }

        # Добавляем в коллекцию Qdrant
        if self.qdrant_manager and self.sessions_collection:
            try:
                self.qdrant_manager.upsert(
                    collection_name=self.sessions_collection,
                    ids=[session_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[embedding_text],
                )
            except Exception as e:
                error_msg = str(e)
                if "embedding with dimension" in error_msg or "dimension" in error_msg.lower():
                    logger.warning(
                        f"Ошибка размерности эмбеддингов в коллекции chat_sessions: {error_msg}. "
                        "Пересоздаём коллекцию..."
                    )
                    # Пересоздаём коллекцию
                    if self.collections_manager:
                        self.sessions_collection = (
                            self.collections_manager._check_and_recreate_collection(
                                "chat_sessions",
                                "Саммаризации сессий для векторного поиска (L1)",
                                force_recreate=True,
                            )
                        )
                        if self.sessions_collection:
                            self.qdrant_manager.upsert(
                                collection_name=self.sessions_collection,
                                ids=[session_id],
                                embeddings=[embedding],
                                metadatas=[metadata],
                                documents=[embedding_text],
                            )
                            logger.info("Коллекция chat_sessions пересоздана и запись добавлена")
                else:
                    logger.error(f"Ошибка при добавлении сессии {session_id} в Qdrant: {e}")
        else:
            logger.warning("Qdrant недоступен, сессия не будет сохранена в векторное хранилище")

        # Синхронизация с графом памяти
        if self.ingestor and self.graph:
            from ...indexing import MemoryRecord
            from ...utils.datetime_utils import parse_datetime_utc

            try:
                # Парсим timestamp из метаданных
                start_time_utc = meta.get("start_time_utc", "")
                timestamp = (
                    parse_datetime_utc(start_time_utc, default=None) if start_time_utc else None
                )
                if not timestamp:
                    from datetime import datetime, timezone

                    timestamp = datetime.now(timezone.utc)

                # Создаём теги для сессии
                tags = []
                chat_name = meta.get("chat_name", "")
                if chat_name:
                    # Добавляем тег на основе имени чата
                    chat_tag = chat_name.lower().replace(" ", "_")
                    tags.append(chat_tag)

                # Создаём MemoryRecord для сессии
                record = MemoryRecord(
                    record_id=session_id,
                    source=meta.get("chat_name", "unknown"),
                    content=embedding_text,
                    timestamp=timestamp,
                    author=None,  # Сессии не имеют автора
                    tags=tags,  # Применяем теги
                    entities=summary.get("entities", []),
                    attachments=[],
                    metadata={
                        "chat": meta.get("chat_name", ""),
                        "profile": meta.get("profile", ""),
                        "start_time_utc": start_time_utc,
                        "end_time_utc": meta.get("end_time_utc", ""),
                        "time_span": meta.get("time_span", ""),
                        "message_count": meta.get("messages_total", 0),
                        "dominant_language": meta.get("dominant_language", "unknown"),
                        "chat_mode": meta.get("chat_mode", "group"),
                        "topics_count": len(summary.get("topics", [])),
                        "claims_count": len(summary.get("claims", [])),
                        "quality_score": summary.get("quality", {}).get("score", 0),
                        "session_type": "session_summary",  # Помечаем как саммари сессии
                    },
                )

                # Сохраняем в граф
                self.ingestor.ingest([record])

                # Сохраняем эмбеддинг в граф и Qdrant
                if embedding:
                    try:
                        # Сохраняем в граф
                        self.graph.update_node(session_id, embedding=embedding)

                        # Сохраняем в Qdrant для векторного поиска (коллекция memory-records)
                        # ВАЖНО: Это основная коллекция для поиска, должна содержать все записи, включая сессии
                        if self.vector_store and self.vector_store.available():
                            try:
                                payload_data = {
                                    "record_id": session_id,
                                    "source": meta.get("chat_name", ""),
                                    "tags": tags,  # Сохраняем теги в Qdrant
                                    "timestamp": start_time_utc.timestamp() if start_time_utc else 0,
                                    "timestamp_iso": start_time_utc.isoformat() if start_time_utc else "",
                                    "content_preview": summary.get("context", "")[:200],
                                    "session_type": "session_summary",
                                    "chat": meta.get("chat_name", ""),
                                }
                                chat_name = meta.get("chat_name")
                                if isinstance(chat_name, str):
                                    payload_data["chat"] = chat_name

                                self.vector_store.upsert(session_id, embedding, payload_data)
                                logger.debug(
                                    f"Эмбеддинг сессии сохранен в Qdrant (memory-records) для {session_id}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Ошибка при сохранении эмбеддинга сессии {session_id} в Qdrant (memory-records): {e}"
                                )
                    except Exception as e:
                        logger.debug(f"Ошибка при сохранении эмбеддинга сессии {session_id}: {e}")

                logger.debug(f"Синхронизирована сессия {session_id} с графом памяти")
            except Exception as e:
                logger.warning(f"Ошибка при синхронизации сессии {session_id} с графом: {e}")

        logger.info(f"L1: Проиндексирована сессия {session_id}")

