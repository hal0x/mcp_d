#!/usr/bin/env python3
"""
Обогащение поисковых запросов контекстом сущностей

Извлекает сущности из запроса, получает их полные профили и добавляет контекст
для улучшения понимания намерения поиска.
"""

import logging
from typing import Any, Dict, List, Optional

from ..analysis.entity_extraction import EntityExtractor
from ..analysis.entity_dictionary import EntityDictionary

logger = logging.getLogger(__name__)


class EntityContextEnricher:
    """Обогащение поисковых запросов контекстом сущностей."""

    def __init__(
        self,
        entity_dictionary: Optional[EntityDictionary] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        graph: Optional[Any] = None,
    ):
        """
        Инициализация обогатителя контекста сущностей

        Args:
            entity_dictionary: Словарь сущностей для получения профилей
            entity_extractor: Экстрактор сущностей для извлечения из запросов
            graph: Граф памяти для поиска связанных сущностей
        """
        self.entity_dictionary = entity_dictionary
        self.entity_extractor = entity_extractor or EntityExtractor(enable_learning=False)
        self.graph = graph

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
        Обогащение запроса контекстом сущностей

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

        # Собираем контексты для найденных сущностей
        entity_contexts = []
        for entity in entities:
            entity_type = entity.get("entity_type")
            value = entity.get("value")

            if not entity_type or not value:
                continue

            # Получаем описание из словаря
            description = self.entity_dictionary.get_entity_description(entity_type, value)

            if description:
                entity_contexts.append(f"{value}: {description}")

        # Если есть контексты, добавляем их к запросу
        if entity_contexts:
            context_text = " [контекст: " + ", ".join(entity_contexts) + "]"
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
            from ..memory.graph_types import EdgeType

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
            # Получаем описание
            description = self.entity_dictionary.get_entity_description(entity_type, value)

            if not description:
                return None

            # Получаем базовую статистику
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

