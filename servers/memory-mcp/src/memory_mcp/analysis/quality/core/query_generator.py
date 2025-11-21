#!/usr/bin/env python3
"""
Генератор тестовых запросов для анализа качества

Автоматически генерирует тестовые запросы на основе реальных данных чатов:
- Простые фактологические запросы
- Контекстные запросы
- Аналитические запросы
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class QueryGenerator:
    """Генератор тестовых запросов для анализа качества"""

    def __init__(self):
        """Инициализация генератора запросов"""
        self.query_templates = {
            "factual": [
                "когда был упомянут {entity}?",
                "что говорили про {entity}?",
                "кто упоминал {entity}?",
                "найти сообщения про {entity}",
                "что обсуждали {entity}?",
            ],
            "contextual": [
                "что обсуждали про {entity} в {timeframe}?",
                "какие были мнения про {entity} в {timeframe}?",
                "что происходило с {entity} в {timeframe}?",
                "как развивалась ситуация с {entity} в {timeframe}?",
                "какие решения принимали по {entity} в {timeframe}?",
            ],
            "analytical": [
                "какие основные темы обсуждались в этом чате?",
                "какие важные решения принимались?",
                "какие проблемы возникали?",
                "какие планы обсуждались?",
                "какие результаты были достигнуты?",
            ],
        }

        logger.info("Инициализирован QueryGenerator")

    async def generate_queries_for_chat(
        self,
        chat_name: str,
        chat_data: List[Dict[str, Any]],
        max_queries: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Генерация тестовых запросов для конкретного чата

        Args:
            chat_name: Название чата
            chat_data: Данные чата (сообщения, сессии, задачи)
            max_queries: Максимальное количество запросов

        Returns:
            Список тестовых запросов
        """
        logger.info(f"Генерация запросов для чата: {chat_name}")

        # Извлекаем сущности из данных чата
        entities = self._extract_entities(chat_data)
        logger.info(f"Извлечено сущностей: {len(entities)}")

        # Извлекаем временные периоды
        timeframes = self._extract_timeframes(chat_data)
        logger.info(f"Извлечено временных периодов: {len(timeframes)}")

        # Генерируем запросы разных типов
        queries = []

        # Простые фактологические запросы
        factual_queries = self._generate_factual_queries(entities)
        queries.extend(factual_queries)

        # Контекстные запросы
        contextual_queries = self._generate_contextual_queries(entities, timeframes)
        queries.extend(contextual_queries)

        # Аналитические запросы
        analytical_queries = self._generate_analytical_queries()
        queries.extend(analytical_queries)

        # Ограничиваем количество запросов
        if len(queries) > max_queries:
            queries = random.sample(queries, max_queries)

        # Добавляем метаданные к каждому запросу
        for i, query in enumerate(queries):
            query["id"] = f"{chat_name}_{i+1}"
            query["chat_name"] = chat_name
            query["generated_at"] = datetime.now().isoformat()

        logger.info(f"Сгенерировано {len(queries)} тестовых запросов")
        return queries

    def _extract_entities(self, chat_data: List[Dict[str, Any]]) -> List[str]:
        """
        Извлечение сущностей из данных чата

        Args:
            chat_data: Данные чата

        Returns:
            Список сущностей
        """
        entities = set()

        for item in chat_data:
            # Извлекаем из текста сообщений
            text = item.get("text", "")
            if text:
                # Простое извлечение слов с заглавной буквы (имена, названия)
                words = text.split()
                for word in words:
                    if len(word) > 3 and word[0].isupper() and word.isalpha():
                        entities.add(word)

            # Извлекаем из метаданных
            metadata = item.get("metadata", {})
            if "user" in metadata:
                entities.add(metadata["user"])
            if "chat" in metadata:
                entities.add(metadata["chat"])

        # Фильтруем и возвращаем наиболее частые сущности
        entity_counts = {}
        for entity in entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1

        # Возвращаем топ-10 сущностей
        sorted_entities = sorted(
            entity_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [entity for entity, count in sorted_entities[:10]]

    def _extract_timeframes(self, chat_data: List[Dict[str, Any]]) -> List[str]:
        """
        Извлечение временных периодов из данных чата

        Args:
            chat_data: Данные чата

        Returns:
            Список временных периодов
        """
        timeframes = []

        # Извлекаем даты из сообщений
        dates = []
        for item in chat_data:
            date_str = item.get("date", "")
            if date_str:
                try:
                    # Парсим дату (предполагаем ISO формат)
                    from ...utils.datetime_utils import parse_datetime_utc

                    date = parse_datetime_utc(date_str, return_none_on_error=True)
                    if not date:
                        continue
                    dates.append(date)
                except Exception:
                    continue

        if not dates:
            return ["последний месяц", "последнюю неделю", "вчера"]

        # Группируем по периодам
        dates.sort()
        earliest = dates[0]
        latest = dates[-1]

        # Генерируем временные периоды
        if latest - earliest > timedelta(days=30):
            timeframes.extend(["январь", "февраль", "март", "апрель"])
        if latest - earliest > timedelta(days=7):
            timeframes.extend(["последнюю неделю", "прошлую неделю"])
        if latest - earliest > timedelta(days=1):
            timeframes.extend(["вчера", "позавчера"])

        timeframes.extend(["сегодня", "последний месяц", "последнюю неделю"])
        return timeframes[:5]  # Ограничиваем количество

    def _generate_factual_queries(self, entities: List[str]) -> List[Dict[str, Any]]:
        """
        Генерация простых фактологических запросов

        Args:
            entities: Список сущностей

        Returns:
            Список фактологических запросов
        """
        queries = []

        for entity in entities[:5]:  # Берем топ-5 сущностей
            for template in self.query_templates["factual"]:
                query_text = template.format(entity=entity)
                queries.append(
                    {
                        "query": query_text,
                        "type": "factual",
                        "entity": entity,
                        "expected_result_type": "specific_message",
                    }
                )

        return queries

    def _generate_contextual_queries(
        self, entities: List[str], timeframes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Генерация контекстных запросов

        Args:
            entities: Список сущностей
            timeframes: Список временных периодов

        Returns:
            Список контекстных запросов
        """
        queries = []

        # Комбинируем сущности с временными периодами
        for entity in entities[:3]:  # Берем топ-3 сущности
            for timeframe in timeframes[:3]:  # Берем топ-3 периода
                for template in self.query_templates["contextual"]:
                    query_text = template.format(entity=entity, timeframe=timeframe)
                    queries.append(
                        {
                            "query": query_text,
                            "type": "contextual",
                            "entity": entity,
                            "timeframe": timeframe,
                            "expected_result_type": "message_sequence",
                        }
                    )

        return queries

    def _generate_analytical_queries(self) -> List[Dict[str, Any]]:
        """
        Генерация аналитических запросов

        Returns:
            Список аналитических запросов
        """
        queries = []

        for template in self.query_templates["analytical"]:
            queries.append(
                {
                    "query": template,
                    "type": "analytical",
                    "expected_result_type": "summary",
                }
            )

        return queries

    def generate_custom_queries(
        self,
        chat_name: str,
        custom_queries: List[Union[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Генерация запросов из пользовательского списка

        Args:
            chat_name: Название чата
            custom_queries: Список пользовательских запросов

        Returns:
            Список тестовых запросов
        """
        queries = []

        for i, entry in enumerate(custom_queries):
            if isinstance(entry, str):
                query_text = entry
                query_payload: Dict[str, Any] = {}
            elif isinstance(entry, dict):
                query_text = entry.get("query", "").strip()
                query_payload = {k: v for k, v in entry.items() if k != "query"}
            else:
                logger.warning(
                    "Пропуск неподдерживаемого формата пользовательского запроса"
                )
                continue

            if not query_text:
                logger.warning("Пропуск пользовательского запроса без текста")
                continue

            query: Dict[str, Any] = {
                "id": f"{chat_name}_custom_{i+1}",
                "query": query_text,
                "type": query_payload.get("type", "custom"),
                "chat_name": chat_name,
                "generated_at": datetime.now().isoformat(),
                "expected_result_type": query_payload.get(
                    "expected_result_type", "unknown"
                ),
            }

            # Добавляем дополнительные поля, если они есть
            for key in ("entity", "timeframe", "metadata"):
                if key in query_payload:
                    query[key] = query_payload[key]

            queries.append(query)

        logger.info(f"Создано {len(queries)} пользовательских запросов")
        return queries

    def validate_query(self, query: Dict[str, Any]) -> bool:
        """
        Валидация тестового запроса

        Args:
            query: Тестовый запрос

        Returns:
            True если запрос валиден
        """
        required_fields = ["query", "type", "chat_name"]

        for field in required_fields:
            if field not in query:
                logger.warning(f"Отсутствует обязательное поле: {field}")
                return False

        if not query["query"].strip():
            logger.warning("Пустой текст запроса")
            return False

        valid_types = ["factual", "contextual", "analytical", "custom"]
        if query["type"] not in valid_types:
            logger.warning(f"Недопустимый тип запроса: {query['type']}")
            return False

        return True
