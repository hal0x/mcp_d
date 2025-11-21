"""L2 индексация: сообщения с группировкой."""

import hashlib
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from ...memory.qdrant_collections import QdrantCollectionsManager
from ...utils.context_optimizer import truncate_embedding_text
from ...utils.url_validator import validate_embedding_text

logger = logging.getLogger(__name__)


class L2Indexer:
    """Индексация сообщений с группировкой."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantCollectionsManager],
        messages_collection: Optional[str],
        embedding_client: Any,
        vector_store: Optional[Any],
        ingestor: Optional[Any],
        graph: Optional[Any],
        entity_extractor: Optional[Any] = None,
        enable_message_grouping: bool = True,
        message_grouping_strategy: str = "session",
        min_group_size: int = 3,
        max_group_size: int = 50,
        max_group_tokens: int = 8000,
        semantic_regrouper: Optional[Any] = None,
        adaptive_grouper: Optional[Any] = None,
        force: bool = False,
        collections_manager: Optional[Any] = None,
    ):
        """Инициализирует L2 индексатор.

        Args:
            qdrant_manager: Менеджер Qdrant коллекций
            messages_collection: Имя коллекции сообщений
            embedding_client: Клиент для генерации эмбеддингов
            vector_store: VectorStore для сохранения эмбеддингов
            ingestor: MemoryIngestor для синхронизации с графом
            graph: Граф памяти
            entity_extractor: Извлекатель сущностей
            enable_message_grouping: Включить группировку сообщений
            message_grouping_strategy: Стратегия группировки
            min_group_size: Минимальный размер группы
            max_group_size: Максимальный размер группы
            max_group_tokens: Максимальное количество токенов в группе
            semantic_regrouper: Семантический перегруппировщик
            adaptive_grouper: Адаптивный группировщик
            force: Принудительная переиндексация
            collections_manager: Менеджер коллекций (для пересоздания при ошибках)
        """
        self.qdrant_manager = qdrant_manager
        self.messages_collection = messages_collection
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.ingestor = ingestor
        self.graph = graph
        self.entity_extractor = entity_extractor
        self.enable_message_grouping = enable_message_grouping
        self.message_grouping_strategy = message_grouping_strategy
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.max_group_tokens = max_group_tokens
        self.semantic_regrouper = semantic_regrouper
        self.adaptive_grouper = adaptive_grouper
        self.force = force
        self.collections_manager = collections_manager

    def format_group_text(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Форматирует группу сообщений в единый текст для создания эмбеддинга.

        Args:
            messages: Список сообщений группы
            max_tokens: Максимальное количество токенов (опционально)

        Returns:
            Отформатированный текст группы
        """
        if not messages:
            return ""

        parts = []
        total_chars = 0
        max_chars = (max_tokens * 4) if max_tokens else None  # ~4 символа на токен

        for i, msg in enumerate(messages, 1):
            msg_text = msg.get("text", "").strip()
            if not msg_text:
                continue

            # Извлекаем автора
            author = "Unknown"
            from_field = msg.get("from") or {}
            if isinstance(from_field, dict):
                author = (
                    from_field.get("display")
                    or from_field.get("username")
                    or from_field.get("id")
                    or "Unknown"
                )
            elif isinstance(from_field, str):
                author = from_field

            # Форматируем сообщение
            formatted_msg = f"[MSG_{i}] {author}: {msg_text}"

            # Проверяем лимит токенов
            if max_chars and total_chars + len(formatted_msg) > max_chars:
                # Обрезаем последнее сообщение, если нужно
                remaining = max_chars - total_chars - len(f"[MSG_{i}] {author}: ")
                if remaining > 20:  # Минимум 20 символов
                    formatted_msg = f"[MSG_{i}] {author}: {msg_text[:remaining]}..."
                else:
                    break

            parts.append(formatted_msg)
            total_chars += len(formatted_msg)

        return "\n".join(parts)

    async def group_messages_for_embedding(
        self,
        messages: List[Dict[str, Any]],
        session_id: str,
        chat_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Группирует сообщения по смыслу для создания эмбеддингов.

        Args:
            messages: Список сообщений для группировки
            session_id: ID сессии
            chat_name: Название чата

        Returns:
            Список групп сообщений с метаданными
        """
        if not self.enable_message_grouping:
            # Если группировка отключена, возвращаем каждое сообщение как отдельную группу
            return [
                {
                    "group_id": f"{session_id}-M{i+1:04d}",
                    "messages": [msg],
                    "message_ids": [
                        msg.get("id") or msg.get("message_id") or f"msg_{i}"
                    ],
                    "strategy": "none",
                }
                for i, msg in enumerate(messages)
            ]

        strategy = self.message_grouping_strategy
        groups = []

        if strategy == "session":
            # Стратегия "session": используем всю сессию как одну группу (если размер подходит)
            if (
                len(messages) >= self.min_group_size
                and len(messages) <= self.max_group_size
            ):
                # Вся сессия как одна группа
                message_ids = [
                    msg.get("id") or msg.get("message_id") or f"msg_{i}"
                    for i, msg in enumerate(messages)
                ]
                groups.append(
                    {
                        "group_id": f"{session_id}-G001",
                        "messages": messages,
                        "message_ids": message_ids,
                        "strategy": "session",
                    }
                )
            else:
                # Разбиваем на подгруппы по max_group_size
                for i in range(0, len(messages), self.max_group_size):
                    group_messages = messages[i : i + self.max_group_size]
                    if len(group_messages) >= self.min_group_size:
                        message_ids = [
                            msg.get("id")
                            or msg.get("message_id")
                            or f"msg_{i+j}"
                            for j, msg in enumerate(group_messages)
                        ]
                        groups.append(
                            {
                                "group_id": f"{session_id}-G{i//self.max_group_size + 1:03d}",
                                "messages": group_messages,
                                "message_ids": message_ids,
                                "strategy": "session",
                            }
                        )

        elif strategy == "semantic" and self.semantic_regrouper:
            # Стратегия "semantic": семантическая перегруппировка через LLM
            try:
                # Создаем временные сессии из сообщений для перегруппировки
                temp_sessions = [
                    {
                        "session_id": f"{session_id}-temp",
                        "messages": messages,
                        "chat": chat_name,
                    }
                ]

                regrouped_sessions = await self.semantic_regrouper.regroup_sessions(
                    temp_sessions, chat_name
                )

                # Преобразуем перегруппированные сессии в группы
                for i, regrouped_session in enumerate(regrouped_sessions, 1):
                    group_messages = regrouped_session.get("messages", [])
                    if len(group_messages) >= self.min_group_size:
                        message_ids = [
                            msg.get("id")
                            or msg.get("message_id")
                            or f"msg_{j}"
                            for j, msg in enumerate(group_messages)
                        ]
                        groups.append(
                            {
                                "group_id": f"{session_id}-G{i:03d}",
                                "messages": group_messages,
                                "message_ids": message_ids,
                                "strategy": "semantic",
                                "theme": regrouped_session.get("theme"),
                                "rationale": regrouped_session.get("regroup_rationale"),
                            }
                        )
            except Exception as e:
                logger.warning(
                    f"Ошибка семантической перегруппировки, используем стратегию session: {e}"
                )
                # Fallback на стратегию session
                if len(messages) >= self.min_group_size:
                    message_ids = [
                        msg.get("id") or msg.get("message_id") or f"msg_{i}"
                        for i, msg in enumerate(messages)
                    ]
                    groups.append(
                        {
                            "group_id": f"{session_id}-G001",
                            "messages": messages,
                            "message_ids": message_ids,
                            "strategy": "session",
                        }
                    )

        elif strategy == "adaptive" and self.adaptive_grouper:
            # Стратегия "adaptive": адаптивная группировка с учетом размера контекста
            try:
                message_groups = self.adaptive_grouper.group_messages_adaptively(
                    messages, chat_name
                )

                for i, group_messages in enumerate(message_groups, 1):
                    if len(group_messages) >= self.min_group_size:
                        message_ids = [
                            msg.get("id") or msg.get("message_id") or f"msg_{j}"
                            for j, msg in enumerate(group_messages)
                        ]
                        groups.append(
                            {
                                "group_id": f"{session_id}-G{i:03d}",
                                "messages": group_messages,
                                "message_ids": message_ids,
                                "strategy": "adaptive",
                            }
                        )
            except Exception as e:
                logger.warning(
                    f"Ошибка адаптивной группировки, используем стратегию session: {e}"
                )
                # Fallback на стратегию session
                if len(messages) >= self.min_group_size:
                    message_ids = [
                        msg.get("id") or msg.get("message_id") or f"msg_{i}"
                        for i, msg in enumerate(messages)
                    ]
                    groups.append(
                        {
                            "group_id": f"{session_id}-G001",
                            "messages": messages,
                            "message_ids": message_ids,
                            "strategy": "session",
                        }
                    )

        else:
            # Неизвестная стратегия или компонент недоступен - используем session
            logger.warning(f"Стратегия {strategy} недоступна, используем session")
            if len(messages) >= self.min_group_size:
                message_ids = [
                    msg.get("id") or msg.get("message_id") or f"msg_{i}"
                    for i, msg in enumerate(messages)
                ]
                groups.append(
                    {
                        "group_id": f"{session_id}-G001",
                        "messages": messages,
                        "message_ids": message_ids,
                        "strategy": "session",
                    }
                )

        # Если не получилось создать группы, возвращаем каждое сообщение отдельно
        if not groups:
            logger.warning(
                f"Не удалось создать группы для сессии {session_id}, используем отдельные сообщения"
            )
            return [
                {
                    "group_id": f"{session_id}-M{i+1:04d}",
                    "messages": [msg],
                    "message_ids": [
                        msg.get("id") or msg.get("message_id") or f"msg_{i}"
                    ],
                    "strategy": "none",
                }
                for i, msg in enumerate(messages)
            ]

        logger.info(
            f"Создано {len(groups)} групп из {len(messages)} сообщений "
            f"для сессии {session_id} (стратегия: {strategy})"
        )
        return groups

    def detect_chat_mode(self, messages: List[Dict[str, Any]]) -> str:
        """Локальная эвристика определения типа чата: 'channel' или 'group'."""
        authors = []
        for m in messages:
            fr = m.get("from") or {}
            name = (
                fr.get("username") or fr.get("display") or fr.get("id") or "unknown"
            )
            authors.append(str(name))
        total = len([a for a in authors if a != "unknown"])
        if total == 0:
            return "group"
        cnt = Counter(a for a in authors if a != "unknown")
        top, top_count = cnt.most_common(1)[0]
        top_share = top_count / total
        unique = len(cnt)
        if (top_share >= 0.85 and unique <= 3 and total >= 5) or unique == 1:
            return "channel"
        return "group"

    def build_symmetric_context(
        self,
        messages: List[Dict[str, Any]],
        current_idx: int,
        max_messages: int = 10,
        max_chars: int = 1500,
    ) -> str:
        """
        Построение симметричного контекста для сообщения

        Args:
            messages: Список сообщений
            current_idx: Индекс текущего сообщения
            max_messages: Максимум сообщений в контексте (по умолчанию 10)
            max_chars: Максимум символов в контексте (по умолчанию 1500)

        Returns:
            Текст контекста
        """
        context_parts = []
        total_chars = 0

        # Добавляем парами: -1, +1, -2, +2, ...
        distance = 1
        while len(context_parts) < max_messages and distance <= max_messages // 2:
            # Предыдущее сообщение
            prev_idx = current_idx - distance
            if prev_idx >= 0:
                prev_text = messages[prev_idx].get("text", "").strip()
                if prev_text and total_chars + len(prev_text) <= max_chars:
                    context_parts.insert(0, prev_text)
                    total_chars += len(prev_text)

            # Следующее сообщение
            next_idx = current_idx + distance
            if next_idx < len(messages):
                next_text = messages[next_idx].get("text", "").strip()
                if next_text and total_chars + len(next_text) <= max_chars:
                    context_parts.append(next_text)
                    total_chars += len(next_text)

            distance += 1

        return " | ".join(context_parts)

    async def index_messages_l2(self, session: Dict[str, Any]) -> int:
        """
        Индексация сообщений на уровне L2 (с симметричным контекстом или группировкой)

        Args:
            session: Сессия

        Returns:
            Количество проиндексированных сообщений
        """
        messages = session["messages"]
        session_id = session["session_id"]
        chat = session["chat"]

        # Определим тип чата для подстройки контекста
        chat_mode = self.detect_chat_mode(messages)

        # Если включена группировка, используем групповые эмбеддинги
        if self.enable_message_grouping:
            return await self.index_messages_l2_grouped(session, messages, session_id, chat, chat_mode)

        # Иначе используем старую логику (отдельные сообщения с контекстом)
        # Упрощенная версия - полная реализация будет добавлена позже
        logger.warning("Индексация без группировки пока не реализована, используем группировку")
        return await self.index_messages_l2_grouped(session, messages, session_id, chat, chat_mode)

    async def index_messages_l2_grouped(
        self,
        session: Dict[str, Any],
        messages: List[Dict[str, Any]],
        session_id: str,
        chat: str,
        chat_mode: str,
    ) -> int:
        """
        Индексация сообщений на уровне L2 с группировкой по смыслу.

        Args:
            session: Сессия
            messages: Список сообщений
            session_id: ID сессии
            chat: Название чата
            chat_mode: Тип чата (channel/group)

        Returns:
            Количество проиндексированных сообщений
        """
        indexed_count = 0
        skipped_duplicates_count = 0

        # Группируем сообщения
        groups = await self.group_messages_for_embedding(messages, session_id, chat)

        groups_to_index = []

        for group in groups:
            try:
                group_id = group["group_id"]
                group_messages = group["messages"]
                message_ids = group["message_ids"]
                strategy = group.get("strategy", "session")

                # Проверяем дубликаты группы
                skipped_duplicate = False

                # Проверяем, существует ли уже эта группа в базе
                if not self.force and self.qdrant_manager and self.messages_collection:
                    try:
                        existing_group = self.qdrant_manager.get(
                            collection_name=self.messages_collection,
                            ids=[group_id]
                        )
                        if existing_group and existing_group.get("ids"):
                            logger.debug(f"Группа {group_id} уже существует, пропускаем")
                            skipped_duplicate = True
                    except Exception as e:
                        logger.debug(f"Не удалось проверить дубликаты группы: {e}")

                if skipped_duplicate:
                    skipped_duplicates_count += len(group_messages)
                    continue

                # Форматируем текст группы
                group_text = self.format_group_text(
                    group_messages,
                    max_tokens=self.max_group_tokens
                )

                if not group_text or len(group_text.strip()) < 10:
                    logger.warning(f"Группа {group_id} пуста, пропускаем")
                    continue

                # Валидируем текст перед отправкой в эмбеддинг
                group_text, replaced_urls = validate_embedding_text(group_text)

                # Сохраняем данные для батчевой обработки
                groups_to_index.append({
                    "group_id": group_id,
                    "group_text": group_text,
                    "group_messages": group_messages,
                    "message_ids": message_ids,
                    "strategy": strategy,
                    "message_count": len(group_messages),
                    "metadata": {
                        "group_id": group_id,
                        "message_ids": message_ids,
                        "message_count": len(group_messages),
                        "group_strategy": strategy,
                        "is_group_embedding": True,
                        "session_id": session_id,
                        "chat": chat,
                        "chat_mode": chat_mode,
                        "replaced_urls": ",".join(replaced_urls) if replaced_urls else "",
                        "theme": group.get("theme"),
                        "rationale": group.get("rationale"),
                    }
                })

            except Exception as e:
                logger.error(
                    f"Ошибка при подготовке группы {group.get('group_id', 'unknown')} "
                    f"в сессии {session_id} для индексации: {e}"
                )
                continue

        # Генерируем эмбеддинги батчами
        if groups_to_index:
            try:
                async with self.embedding_client:
                    # Собираем тексты для батча
                    batch_texts = [group["group_text"] for group in groups_to_index]

                    # Генерируем эмбеддинги батчем
                    embeddings = await self.embedding_client.generate_embeddings(batch_texts, batch_size=32)

                    # Добавляем в коллекцию батчем
                    ids = [group["group_id"] for group in groups_to_index]
                    documents = [group["group_text"] for group in groups_to_index]
                    metadatas = [group["metadata"] for group in groups_to_index]

                    # Сохраняем в Qdrant коллекцию chat_messages
                    if self.qdrant_manager and self.messages_collection:
                        try:
                            self.qdrant_manager.upsert(
                                collection_name=self.messages_collection,
                                ids=ids,
                                embeddings=embeddings,
                                metadatas=metadatas,
                                documents=documents,
                            )
                            logger.debug(f"Сохранено {len(ids)} групп в Qdrant коллекцию {self.messages_collection}")
                        except Exception as e:
                            error_msg = str(e)
                            if "dimension" in error_msg.lower():
                                logger.warning(
                                    f"Ошибка размерности эмбеддингов в коллекции chat_messages: {error_msg}. "
                                    "Пересоздаём коллекцию..."
                                )
                                if self.qdrant_manager and self.collections_manager:
                                    self.qdrant_manager.delete_collection(self.messages_collection)
                                    self.messages_collection = self.collections_manager._check_and_recreate_collection(
                                        "chat_messages",
                                        "Сообщения с контекстом для уточняющего поиска (L2)",
                                        force_recreate=True
                                    )
                                    if self.messages_collection:
                                        self.qdrant_manager.upsert(
                                            collection_name=self.messages_collection,
                                            ids=ids,
                                            embeddings=embeddings,
                                            metadatas=metadatas,
                                            documents=documents,
                                        )
                                        logger.info("Коллекция chat_messages пересоздана и записи добавлены")
                            else:
                                logger.error(f"Ошибка при добавлении групп в Qdrant: {e}")
                    else:
                        logger.warning("Qdrant недоступен, группы не будут сохранены в векторное хранилище")

                    # Синхронизация с графом памяти
                    if self.ingestor and self.graph:
                        logger.info(f"Синхронизация с графом: обрабатываем {len(groups_to_index)} групп")
                        from ...utils.datetime_utils import parse_datetime_utc
                        from datetime import datetime, timezone
                        from ...indexing import MemoryRecord

                        records_to_ingest = []
                        for idx, group_data in enumerate(groups_to_index):
                            try:
                                group_id = group_data["group_id"]
                                group_text = group_data["group_text"]
                                metadata = group_data["metadata"]
                                embedding = embeddings[idx] if idx < len(embeddings) else None

                                # Используем время первого сообщения группы
                                first_msg = group_data["group_messages"][0]
                                date_utc = first_msg.get("date_utc") or first_msg.get("date", "")
                                timestamp = parse_datetime_utc(date_utc, default=None) if date_utc else None
                                if not timestamp:
                                    timestamp = datetime.now(timezone.utc)

                                # Создаем запись для группы
                                record = MemoryRecord(
                                    record_id=group_id,
                                    source=chat,
                                    content=group_text,
                                    timestamp=timestamp,
                                    author=None,  # Группа может содержать сообщения от разных авторов
                                    tags=[],
                                    entities=[],
                                    attachments=[],
                                    metadata=metadata,
                                )

                                if embedding is not None and len(embedding) > 0:
                                    records_to_ingest.append((record, embedding))

                            except Exception as e:
                                logger.error(f"Ошибка при подготовке записи группы {group_data.get('group_id', 'unknown')}: {e}")
                                continue

                        # Инжестим записи в граф
                        if records_to_ingest:
                            try:
                                for record, embedding in records_to_ingest:
                                    try:
                                        if hasattr(embedding, 'tolist'):
                                            embedding = embedding.tolist()
                                        elif not isinstance(embedding, list):
                                            embedding = list(embedding)

                                        self.ingestor.ingest([record], embeddings=[embedding])

                                        # Также сохраняем в vector_store для поиска
                                        if self.vector_store and self.vector_store.available():
                                            payload_data = {
                                                "record_id": record.record_id,
                                                "source": record.source,
                                                "content": record.content,
                                                "timestamp": record.timestamp.isoformat() if record.timestamp else None,
                                                "tags": record.tags,
                                                "entities": record.entities,
                                                "metadata": record.metadata,
                                            }
                                            self.vector_store.upsert(record.record_id, embedding, payload_data)

                                    except Exception as e:
                                        logger.error(f"Ошибка при индексации группы {record.record_id}: {e}")
                                        continue

                                logger.info(
                                    f"Синхронизировано {len(records_to_ingest)} групп с графом памяти"
                                )
                            except Exception as e:
                                logger.error(f"Ошибка при синхронизации групп с графом: {e}")

                    indexed_count = sum(group["message_count"] for group in groups_to_index)

            except Exception as e:
                logger.error(
                    f"Ошибка при индексации групп в сессии {session_id}: {e}"
                )

        logger.info(
            f"L2 (группировка): Проиндексировано {indexed_count} сообщений "
            f"в {len(groups_to_index)} группах из сессии {session_id} "
            f"(пропущено дубликатов: {skipped_duplicates_count})"
        )
        return indexed_count


