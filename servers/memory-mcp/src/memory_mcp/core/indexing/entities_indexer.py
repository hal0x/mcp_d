"""Индексация сущностей."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EntitiesIndexer:
    """Индексация сущностей."""

    def __init__(
        self,
        entity_dictionary: Optional[Any],
        entity_vector_store: Optional[Any],
        graph: Optional[Any],
        embedding_client: Optional[Any],
    ):
        """Инициализирует индексатор сущностей.

        Args:
            entity_dictionary: Словарь сущностей
            entity_vector_store: VectorStore для сущностей
            graph: Граф памяти
            embedding_client: Клиент для генерации эмбеддингов
        """
        self.entity_dictionary = entity_dictionary
        self.entity_vector_store = entity_vector_store
        self.graph = graph
        self.embedding_client = embedding_client

    async def update_entity_nodes_with_descriptions(self) -> None:
        """Обновление EntityNode в графе с описаниями из словаря сущностей"""
        if not self.graph or not self.entity_dictionary:
            return

        try:
            from ...memory.storage.graph.graph_types import NodeType

            # Получаем все EntityNode из графа
            entity_nodes = self.graph.get_nodes_by_type(NodeType.ENTITY, limit=10000)

            updated_count = 0
            for node_data in entity_nodes:
                node_id = node_data.get("id")
                if not node_id:
                    continue

                # Получаем тип и имя сущности из узла
                entity_type = node_data.get("entity_type", "term")
                label = node_data.get("label", "")

                if not label:
                    continue

                # Получаем описание из словаря
                description = self.entity_dictionary.get_entity_description(
                    entity_type, label
                )

                if description:
                    # Обновляем узел с описанием
                    # Проверяем, нужно ли обновлять (если описание изменилось или отсутствует)
                    current_description = node_data.get("description")
                    if current_description != description:
                        # Обновляем через update_node или напрямую через SQL
                        try:
                            # Обновляем узел через метод update_node графа
                            # Сначала получаем текущие свойства
                            if node_id in self.graph.graph:
                                node_data = self.graph.graph.nodes[node_id]
                                current_properties = node_data.get("properties", {})

                                # Обновляем описание в свойствах
                                current_properties["description"] = description

                                # Обновляем узел через метод графа
                                self.graph.update_node(
                                    node_id,
                                    properties=current_properties,
                                )

                                # Также обновляем поле description напрямую, если узел это EntityNode
                                node_data["description"] = description

                                updated_count += 1
                                logger.debug(
                                    f"Обновлено описание для EntityNode {node_id}: {description[:50]}..."
                                )
                        except Exception as e:
                            logger.debug(
                                f"Ошибка при обновлении описания для EntityNode {node_id}: {e}"
                            )

            if updated_count > 0:
                logger.info(f"Обновлено {updated_count} EntityNode с описаниями")
        except Exception as e:
            logger.warning(f"Ошибка при обновлении EntityNode с описаниями: {e}")

    async def build_and_index_entities(self, chat_name: str) -> None:
        """
        Построение и индексация профилей сущностей в векторное хранилище

        Args:
            chat_name: Название чата (для логирования)
        """
        if (
            not self.entity_dictionary
            or not self.entity_vector_store
            or not self.graph
        ):
            return

        try:
            from ...memory.storage.graph.graph_types import NodeType, EntityNode

            # Собираем все сущности из словаря
            all_entities = []
            for entity_type in self.entity_dictionary.learned_dictionaries:
                for normalized_value in self.entity_dictionary.learned_dictionaries[
                    entity_type
                ]:
                    # Получаем оригинальное значение (из entity_counts или из графа)
                    entity_data = self.entity_dictionary.learned_dictionaries[
                        entity_type
                    ][normalized_value]
                    original_value = entity_data.get("original_value", normalized_value)

                    all_entities.append(
                        {
                            "entity_type": entity_type,
                            "normalized_value": normalized_value,
                            "original_value": original_value,
                        }
                    )

            if not all_entities:
                logger.info(f"Нет сущностей для индексации в чате {chat_name}")
                return

            logger.info(
                f"Начинаем индексацию {len(all_entities)} сущностей из чата {chat_name}"
            )

            indexed_count = 0
            failed_count = 0

            for entity_info in all_entities:
                try:
                    entity_type = entity_info["entity_type"]
                    normalized_value = entity_info["normalized_value"]
                    original_value = entity_info["original_value"]

                    # Получаем профиль сущности из словаря
                    profile = self.entity_dictionary.get_entity_profile(
                        entity_type, normalized_value
                    )

                    if not profile:
                        logger.debug(
                            f"Профиль не найден для {entity_type}={normalized_value}"
                        )
                        failed_count += 1
                        continue

                    # Формируем текст для эмбеддинга
                    embedding_text_parts = []

                    # Добавляем описание, если есть
                    description = profile.get("description", "")
                    if description:
                        embedding_text_parts.append(description)

                    # Добавляем алиасы
                    aliases = profile.get("aliases", [])
                    if aliases:
                        embedding_text_parts.append(f"Также известен как: {', '.join(aliases)}")

                    # Добавляем контекст использования (первые несколько чатов)
                    chats = profile.get("chats", [])
                    if chats:
                        embedding_text_parts.append(
                            f"Упоминается в: {', '.join(chats[:3])}"
                        )

                    embedding_text = " ".join(embedding_text_parts)

                    if not embedding_text:
                        # Если нет описания, используем имя и тип
                        embedding_text = f"{entity_type} {original_value}"

                    # Генерируем эмбеддинг
                    async with self.embedding_client:
                        embedding = await self.embedding_client.embed(embedding_text)

                    if not embedding or len(embedding) == 0:
                        logger.debug(
                            f"Не удалось сгенерировать эмбеддинг для {entity_type}={normalized_value}"
                        )
                        failed_count += 1
                        continue

                    # Формируем payload для Qdrant
                    entity_id = f"entity-{normalized_value.replace(' ', '-')}"
                    payload = {
                        "entity_type": entity_type,
                        "value": original_value,
                        "normalized_value": normalized_value,
                        "description": profile.get("description", ""),
                        "aliases": profile.get("aliases", []),
                        "importance": profile.get("importance", 0.5),
                        "mention_count": profile.get("mention_count", 0),
                        "chats": profile.get("chats", []),
                        "first_seen": profile.get("first_seen"),
                        "last_seen": profile.get("last_seen"),
                    }

                    # Сохраняем в EntityVectorStore
                    self.entity_vector_store.upsert_entity(entity_id, embedding, payload)

                    # Обновляем EntityNode в графе с эмбеддингом и полным описанием
                    if entity_id in self.graph.graph:
                        node_data = self.graph.graph.nodes[entity_id]
                        current_properties = node_data.get("properties", {})

                        # Обновляем описание и эмбеддинг
                        current_properties["description"] = profile.get("description", "")
                        current_properties["entity_profile"] = {
                            "mention_count": profile.get("mention_count", 0),
                            "chats": profile.get("chats", []),
                            "importance": profile.get("importance", 0.5),
                        }

                        self.graph.update_node(
                            entity_id,
                            properties=current_properties,
                            embedding=embedding,
                        )
                    else:
                        # Создаем новый EntityNode, если его нет в графе
                        entity_node = EntityNode(
                            id=entity_id,
                            label=original_value,
                            entity_type=entity_type,
                            aliases=profile.get("aliases", []),
                            description=profile.get("description"),
                            importance=profile.get("importance", 0.5),
                            properties={
                                "normalized_value": normalized_value,
                                "entity_profile": {
                                    "mention_count": profile.get("mention_count", 0),
                                    "chats": profile.get("chats", []),
                                    "importance": profile.get("importance", 0.5),
                                },
                            },
                            embedding=embedding,
                        )
                        self.graph.add_node(entity_node)

                    indexed_count += 1

                    if indexed_count % 10 == 0:
                        logger.debug(f"Проиндексировано {indexed_count} сущностей...")

                except Exception as e:
                    logger.warning(
                        f"Ошибка при индексации сущности {entity_type}={normalized_value}: {e}"
                    )
                    failed_count += 1
                    continue

            logger.info(
                f"Завершена индексация сущностей из чата {chat_name}: "
                f"{indexed_count} успешно, {failed_count} ошибок"
            )

        except Exception as e:
            logger.warning(
                f"Ошибка при построении и индексации профилей сущностей: {e}"
            )

