#!/usr/bin/env python3
"""
Обогащение поисковых запросов контекстом сущностей

Извлекает сущности из запроса, получает их полные профили и добавляет контекст
для улучшения понимания намерения поиска.
"""

import logging
from typing import Any, Dict, List, Optional

from ..analysis.entities import EntityExtractor, EntityDictionary
from ..memory.embeddings import build_embedding_service_from_env
from ..memory.storage.vector.vector_store import build_entity_vector_store_from_env, EntityVectorStore

logger = logging.getLogger(__name__)


class EntityContextEnricher:
    """Обогащение поисковых запросов контекстом сущностей."""

    def __init__(
        self,
        entity_dictionary: Optional[EntityDictionary] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        graph: Optional[Any] = None,
        entity_vector_store: Optional[EntityVectorStore] = None,
        embedding_service: Optional[Any] = None,
    ):
        """
        Инициализация обогатителя контекста сущностей

        Args:
            entity_dictionary: Словарь сущностей для получения профилей
            entity_extractor: Экстрактор сущностей для извлечения из запросов
            graph: Граф памяти для поиска связанных сущностей
            entity_vector_store: Векторное хранилище сущностей для семантического поиска
            embedding_service: Сервис для генерации эмбеддингов
        """
        self.entity_dictionary = entity_dictionary
        self.entity_extractor = entity_extractor or EntityExtractor(enable_learning=False)
        self.graph = graph
        self.entity_vector_store = entity_vector_store or build_entity_vector_store_from_env()
        self.embedding_service = embedding_service or build_embedding_service_from_env()

    def extract_entities_from_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Извлечение сущностей из поискового запроса

        Args:
            query: Поисковый запрос

        Returns:
            Список найденных сущностей с типом и значением
        """
        if not query or not query.strip():
            return []

        try:
            # Извлекаем сущности из запроса
            extracted = self.entity_extractor.extract_entities(query, chat_name="")

            entities = []
            # Собираем все типы сущностей
            for entity_type, values in extracted.items():
                if not values:
                    continue

                # Убираем дубликаты
                unique_values = list(set(values))

                for value in unique_values:
                    if value and str(value).strip():
                        entities.append({
                            "entity_type": entity_type,
                            "value": str(value),
                        })

            logger.debug(f"Извлечено {len(entities)} сущностей из запроса: {query[:50]}...")
            return entities

        except Exception as e:
            logger.warning(f"Ошибка при извлечении сущностей из запроса: {e}")
            return []

    def enrich_query_with_entity_context(
        self, query: str, entities: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Обогащение запроса контекстом сущностей с использованием полного профиля

        Args:
            query: Исходный поисковый запрос
            entities: Список сущностей (если None, извлекаются автоматически)

        Returns:
            Обогащенный запрос с контекстом сущностей
        """
        if not query or not query.strip():
            return query

        # Извлекаем сущности, если не переданы
        if entities is None:
            entities = self.extract_entities_from_query(query)

        if not entities or not self.entity_dictionary:
            return query

        # Собираем контексты для найденных сущностей с использованием полного профиля
        entity_contexts = []
        processed_entity_values = set()  # Для избежания дубликатов
        
        for entity in entities:
            entity_type = entity.get("entity_type")
            value = entity.get("value")

            if not entity_type or not value:
                continue
            
            # Избегаем обработки дубликатов
            entity_key = f"{entity_type}:{value.lower()}"
            if entity_key in processed_entity_values:
                continue
            processed_entity_values.add(entity_key)

            # Получаем полный профиль сущности
            profile = self.entity_dictionary.build_entity_profile(entity_type, value)
            
            if not profile:
                # Fallback на простое описание, если профиль не найден
                description = self.entity_dictionary.get_entity_description(entity_type, value)
                if description:
                    entity_contexts.append(f"{value}: {description}")
                continue

            # Формируем расширенный контекст из профиля
            context_parts = []
            
            # Основное описание
            description = profile.get("description")
            if description:
                context_parts.append(f"{value}: {description}")
            
            # Алиасы (альтернативные названия)
            aliases = profile.get("aliases", [])
            if aliases and len(aliases) > 1:
                # Исключаем само значение из алиасов
                other_aliases = [a for a in aliases if a.lower() != value.lower()][:3]
                if other_aliases:
                    context_parts.append(f"также известен как: {', '.join(other_aliases)}")
            
            # Ключевые контексты упоминаний (первые 2-3)
            contexts = profile.get("contexts", [])
            if contexts:
                key_contexts = contexts[:3]
                context_snippets = []
                for ctx in key_contexts:
                    content = ctx.get("content", "")
                    if content:
                        # Берем первые 100 символов контекста
                        snippet = content[:100].strip()
                        if snippet:
                            context_snippets.append(snippet)
                if context_snippets:
                    context_parts.append(f"контексты: {'; '.join(context_snippets)}")
            
            # Связанные сущности (топ-3 по весу)
            related_entities = profile.get("related_entities", [])
            if related_entities:
                top_related = sorted(
                    related_entities, 
                    key=lambda x: x.get("weight", 0.0), 
                    reverse=True
                )[:3]
                related_names = [r.get("label", "") for r in top_related if r.get("label")]
                if related_names:
                    context_parts.append(f"связано с: {', '.join(related_names)}")
            
            # Временные метки для актуальности
            last_seen = profile.get("last_seen")
            if last_seen:
                context_parts.append(f"последнее упоминание: {last_seen[:10]}")  # Только дата
            
            # Важность (для информации, не добавляем в запрос, но учитываем при весах)
            importance = profile.get("importance", 0.5)
            
            if context_parts:
                entity_context = " | ".join(context_parts)
                entity_contexts.append(entity_context)
                logger.debug(
                    f"Обогащен профиль сущности {entity_type}={value} "
                    f"(важность: {importance:.2f}, упоминаний: {profile.get('mention_count', 0)})"
                )
        
        # Поиск семантически похожих сущностей для расширения контекста
        if self.entity_vector_store and self.entity_vector_store.available():
            similar_entities = self.find_semantically_similar_entities(query, limit=3)
            for similar in similar_entities:
                similar_value = similar.get("value", "")
                similar_type = similar.get("entity_type", "")
                similar_score = similar.get("score", 0.0)
                
                # Добавляем только если достаточно похожи (score > 0.7) и не дубликат
                if similar_value and similar_score > 0.7:
                    similar_key = f"{similar_type}:{similar_value.lower()}"
                    if similar_key not in processed_entity_values:
                        processed_entity_values.add(similar_key)
                        similar_desc = similar.get("description", "")
                        if similar_desc:
                            entity_contexts.append(f"{similar_value} (похоже): {similar_desc[:100]}")
                            logger.debug(
                                f"Добавлена семантически похожая сущность: {similar_value} "
                                f"(score: {similar_score:.2f})"
                            )

        # Если есть контексты, добавляем их к запросу
        if entity_contexts:
            context_text = " [контекст: " + "; ".join(entity_contexts) + "]"
            enriched_query = query + context_text
            logger.debug(f"Запрос обогащен контекстом {len(entity_contexts)} сущностей")
            return enriched_query

        return query

    def get_related_entities(
        self, entity_type: str, value: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Получение связанных сущностей через граф

        Args:
            entity_type: Тип сущности
            value: Значение сущности
            limit: Максимальное количество связанных сущностей

        Returns:
            Список связанных сущностей с метаданными
        """
        if not self.graph:
            return []

        try:
            from ..memory.storage.graph.graph_types import EdgeType

            # Нормализуем значение для поиска ID узла
            normalized_value = value.lower().strip().replace(" ", "-")
            entity_id = f"entity-{normalized_value}"

            if entity_id not in self.graph.graph:
                return []

            # Получаем соседние EntityNode
            neighbors = self.graph.get_neighbors(entity_id, direction="both")

            related = []
            for neighbor_id, edge_data in neighbors:
                if neighbor_id in self.graph.graph:
                    neighbor_data = self.graph.graph.nodes[neighbor_id]
                    neighbor_type = neighbor_data.get("type")

                    # Собираем только EntityNode
                    if neighbor_type == "Entity" or (
                        isinstance(neighbor_type, str) and "Entity" in neighbor_type
                    ):
                        related.append({
                            "entity_id": neighbor_id,
                            "label": neighbor_data.get("label", ""),
                            "entity_type": neighbor_data.get("entity_type", ""),
                            "description": neighbor_data.get("description", ""),
                            "edge_type": edge_data.get("type", ""),
                            "weight": edge_data.get("weight", 0.0),
                        })

            # Сортируем по весу связи и ограничиваем
            related.sort(key=lambda x: x.get("weight", 0.0), reverse=True)
            return related[:limit]

        except Exception as e:
            logger.debug(f"Ошибка при получении связанных сущностей для {value}: {e}")
            return []

    def expand_query_with_related_entities(
        self, query: str, entities: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Расширение запроса связанными сущностями через граф

        Args:
            query: Исходный поисковый запрос
            entities: Список сущностей (если None, извлекаются автоматически)

        Returns:
            Расширенный запрос с связанными сущностями
        """
        if not query or not query.strip():
            return query

        # Извлекаем сущности, если не переданы
        if entities is None:
            entities = self.extract_entities_from_query(query)

        if not entities or not self.graph:
            return query

        # Собираем связанные сущности для каждой найденной сущности
        related_entity_names = []
        for entity in entities:
            entity_type = entity.get("entity_type")
            value = entity.get("value")

            if not entity_type or not value:
                continue

            # Получаем связанные сущности через граф
            related = self.get_related_entities(entity_type, value, limit=5)
            
            # Добавляем имена связанных сущностей с учетом веса связи
            for rel in related:
                label = rel.get("label", "")
                weight = rel.get("weight", 0.0)
                
                # Добавляем только сущности с достаточным весом связи (> 0.3)
                if label and weight > 0.3 and label not in related_entity_names:
                    related_entity_names.append(label)

        # Если есть связанные сущности, добавляем их к запросу
        if related_entity_names:
            related_text = " " + " ".join(related_entity_names[:5])  # Максимум 5 связанных
            expanded_query = query + related_text
            logger.debug(f"Запрос расширен {len(related_entity_names)} связанными сущностями")
            return expanded_query

        return query

    def find_semantically_similar_entities(
        self, query: str, entity_type: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Поиск семантически похожих сущностей через EntityVectorStore

        Args:
            query: Поисковый запрос
            entity_type: Фильтр по типу сущности (опционально)
            limit: Максимальное количество результатов

        Returns:
            Список похожих сущностей с метаданными
        """
        if not self.entity_vector_store or not self.entity_vector_store.available():
            return []

        if not self.embedding_service or not self.embedding_service.available():
            return []

        try:
            # Генерируем эмбеддинг запроса
            query_embedding = self.embedding_service.embed(query)
            if not query_embedding:
                return []

            # Ищем похожие сущности через EntityVectorStore
            search_results = self.entity_vector_store.search_entities(
                query_vector=query_embedding,
                entity_type=entity_type,
                limit=limit,
            )

            # Преобразуем результаты в формат словарей
            similar_entities = []
            for result in search_results:
                payload = result.payload
                similar_entities.append({
                    "entity_id": result.entity_id,
                    "score": result.score,
                    "entity_type": payload.get("entity_type", ""),
                    "value": payload.get("value", ""),
                    "normalized_value": payload.get("normalized_value", ""),
                    "description": payload.get("description", ""),
                    "aliases": payload.get("aliases", []),
                    "importance": payload.get("importance", 0.5),
                    "mention_count": payload.get("mention_count", 0),
                })

            logger.debug(
                f"Найдено {len(similar_entities)} семантически похожих сущностей "
                f"для запроса: {query[:50]}..."
            )
            return similar_entities

        except Exception as e:
            logger.warning(f"Ошибка при поиске семантически похожих сущностей: {e}")
            return []

    def get_entity_profile_summary(
        self, entity_type: str, value: str
    ) -> Optional[Dict[str, Any]]:
        """
        Получение краткой сводки профиля сущности для обогащения запроса

        Args:
            entity_type: Тип сущности
            value: Значение сущности

        Returns:
            Краткая сводка профиля или None
        """
        if not self.entity_dictionary:
            return None

        try:
            # Получаем полный профиль
            profile = self.entity_dictionary.build_entity_profile(entity_type, value)
            
            if profile:
                return {
                    "entity_type": profile.get("entity_type"),
                    "value": profile.get("value"),
                    "description": profile.get("description"),
                    "mention_count": profile.get("mention_count", 0),
                    "importance": profile.get("importance", 0.5),
                    "aliases": profile.get("aliases", []),
                    "related_entities_count": len(profile.get("related_entities", [])),
                    "contexts_count": profile.get("context_count", 0),
                }

            # Fallback на простое описание
            description = self.entity_dictionary.get_entity_description(entity_type, value)
            if not description:
                return None

            normalized_value = self.entity_dictionary._normalize_entity_value(value)
            if not normalized_value:
                return None

            total_count = self.entity_dictionary.entity_counts.get(entity_type, {}).get(
                normalized_value, 0
            )

            return {
                "entity_type": entity_type,
                "value": value,
                "description": description,
                "mention_count": total_count,
            }

        except Exception as e:
            logger.debug(f"Ошибка при получении профиля сущности {value}: {e}")
            return None

