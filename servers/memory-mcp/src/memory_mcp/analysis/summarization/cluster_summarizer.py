#!/usr/bin/env python3
"""
Генерация сводок для кластеров сессий через LLM

Создаёт связные описания тематических кластеров.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ...core.adapters.lmql_adapter import LMQLAdapter, build_lmql_adapter_from_env

logger = logging.getLogger(__name__)


class ClusterSummarizer:
    """Генератор сводок для кластеров через LLM"""

    def __init__(self, embedding_client, lmql_adapter: Optional[LMQLAdapter] = None):
        """
        Инициализация

        Args:
            embedding_client: LangChainLLMAdapter для взаимодействия с LLM
            lmql_adapter: Опциональный LMQL адаптер для структурированной генерации.
                         Если не указан, создается из настроек окружения.
        """
        self.embedding_client = embedding_client
        try:
            self.lmql_adapter = lmql_adapter or build_lmql_adapter_from_env()
        except RuntimeError:
            self.lmql_adapter = None
        logger.info("Инициализирован ClusterSummarizer")

    async def summarize_cluster(
        self, cluster: Dict[str, Any], max_sessions_to_analyze: int = 10
    ) -> Dict[str, Any]:
        """
        Генерация сводки для кластера

        Args:
            cluster: Данные кластера
            max_sessions_to_analyze: Максимум сессий для анализа

        Returns:
            Сводка кластера
        """
        cluster_id = cluster["cluster_id"]
        sessions = cluster.get("sessions", [])
        session_count = len(sessions)

        logger.info(
            f"Генерация сводки для кластера {cluster_id} ({session_count} сессий)"
        )

        if not sessions:
            return {
                "title": f"Пустой кластер {cluster_id}",
                "description": "Нет сессий для анализа",
                "key_insights": [],
            }

        # Берём репрезентативную выборку сессий
        sample_sessions = sessions[:max_sessions_to_analyze]

        # Собираем ключевую информацию из сессий
        cluster_content = self._extract_cluster_content(sample_sessions, cluster)

        # Генерируем сводку через LMQL
        try:
            if self.lmql_adapter:
                logger.debug("Используется LMQL для генерации сводки кластера")
                summary = await self._generate_lmql_summary(cluster_content, cluster)
            else:
                raise RuntimeError(
                    "LMQL не настроен. Установите MEMORY_MCP_USE_LMQL=true и настройте модель"
                )
            return summary
        except Exception as e:
            logger.error(f"Ошибка генерации LMQL сводки: {e}")
            raise RuntimeError(
                f"Ошибка генерации LMQL сводки для кластера: {e}. "
                "Проверьте конфигурацию LMQL."
            ) from e

    def _extract_cluster_content(
        self, sessions: List[Dict[str, Any]], cluster: Dict[str, Any]
    ) -> str:
        """
        Извлечение ключевого контента из сессий кластера

        Args:
            sessions: Список сессий
            cluster: Данные кластера

        Returns:
            Текст для анализа
        """
        content_parts = []

        # Метаданные кластера
        content_parts.append("=== МЕТАДАННЫЕ КЛАСТЕРА ===")
        content_parts.append(f"Сессий: {cluster.get('session_count', 0)}")
        content_parts.append(f"Доминирующий чат: {cluster.get('dominant_chat', 'N/A')}")

        top_topics = cluster.get("top_topics", [])
        if top_topics:
            content_parts.append(f"Основные темы: {', '.join(top_topics[:5])}")

        top_entities = cluster.get("top_entities", [])
        if top_entities:
            content_parts.append(f"Ключевые сущности: {', '.join(top_entities[:10])}")

        time_range = cluster.get("time_range")
        if time_range:
            content_parts.append(
                f"Временной диапазон: {time_range.get('span_days', 0)} дней"
            )

        content_parts.append("")
        content_parts.append("=== СОДЕРЖАНИЕ СЕССИЙ ===")

        # Извлекаем ключевую информацию из каждой сессии
        for i, session in enumerate(sessions, 1):
            content_parts.append(
                f"\n--- Сессия {i} ({session.get('session_id', 'N/A')}) ---"
            )

            # Темы сессии
            topics = session.get("topics", [])
            if topics:
                content_parts.append("\nТемы:")
                for topic in topics[:3]:
                    if isinstance(topic, dict):
                        title = topic.get("title", "")
                        summary = topic.get("summary", "")
                        content_parts.append(f"  - {title}: {summary[:150]}")
                    else:
                        content_parts.append(f"  - {topic}")

            # Ключевые утверждения
            claims = session.get("claims", [])
            if claims:
                content_parts.append("\nКлючевые утверждения:")
                for claim in claims[:3]:
                    if isinstance(claim, dict):
                        claim_text = claim.get("summary", claim.get("text", ""))
                        content_parts.append(f"  - {claim_text[:150]}")

            # Действия/задачи
            actions = session.get("actions", [])
            if actions:
                content_parts.append("\nДействия:")
                for action in actions[:2]:
                    if isinstance(action, dict):
                        action_text = action.get("text", "")
                        content_parts.append(f"  - {action_text[:100]}")

        return "\n".join(content_parts)

    async def _generate_lmql_summary(
        self, cluster_content: str, cluster: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Генерация сводки через LMQL

        Args:
            cluster_content: Контент кластера
            cluster: Метаданные кластера

        Returns:
            Сводка

        Raises:
            RuntimeError: Если произошла ошибка при выполнении запроса
        """
        try:
            lmql_query_str = self._create_lmql_summary_query(cluster_content)
            response_data = await self.lmql_adapter.execute_json_query(
                prompt=f"""Ты - аналитик, который создаёт краткие сводки для тематических кластеров обсуждений в Telegram чатах.

Дано:
{cluster_content}

Задача:
1. Определи **одну главную тему**, которая объединяет все эти сессии (макс 10 слов)
2. Напиши краткое описание кластера (2-3 предложения)
3. Выдели 3-5 ключевых инсайтов из обсуждений""",
                json_schema='{"title": "[TITLE]", "description": "[DESCRIPTION]", "key_insights": [KEY_INSIGHTS]}',
                constraints="""
                    len(TOKENS(TITLE)) <= 10 and
                    3 <= len(KEY_INSIGHTS) <= 5
                """,
                temperature=0.3,
                max_tokens=2048,
            )

            return self._parse_lmql_response(response_data)

        except Exception as e:
            logger.error(f"Ошибка при использовании LMQL для генерации сводки: {e}")
            raise RuntimeError(f"Ошибка генерации сводки через LMQL: {e}") from e

    def _create_lmql_summary_query(self, cluster_content: str) -> str:
        """Создание LMQL запроса для генерации сводки.
        
        Args:
            cluster_content: Контент кластера
            
        Returns:
            Строка с LMQL запросом (для логирования/отладки)
        """
        # Этот метод возвращает строку для отладки, фактический запрос выполняется в execute_json_query
        return f"LMQL query for cluster summary: {cluster_content[:100]}..."

    def _parse_lmql_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг ответа LMQL в структуру сводки.
        
        Args:
            data: Данные из LMQL ответа
            
        Returns:
            Словарь со сводкой
        """
        return {
            "title": data.get("title", "Тематический кластер"),
            "description": data.get("description", ""),
            "key_insights": data.get("key_insights", []),
        }

    def _fallback_summary(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """
        Базовая сводка без LLM

        Args:
            cluster: Данные кластера

        Returns:
            Базовая сводка
        """
        top_topics = cluster.get("top_topics", [])
        top_entities = cluster.get("top_entities", [])
        session_count = cluster.get("session_count", 0)
        dominant_chat = cluster.get("dominant_chat", "различных чатов")

        # Формируем заголовок
        if top_topics:
            title = f"Кластер: {top_topics[0]}"
        elif top_entities:
            title = f"Обсуждение: {', '.join(top_entities[:3])}"
        else:
            title = f"Тематический кластер {cluster['cluster_id']}"

        # Формируем описание
        description = f"Объединяет {session_count} сессий "

        if top_topics:
            description += f"по темам: {', '.join(top_topics[:3])}. "

        if dominant_chat:
            description += f"Преимущественно из чата '{dominant_chat}'. "

        if top_entities:
            description += f"Ключевые упоминания: {', '.join(top_entities[:5])}."

        # Инсайты на основе метаданных
        insights = []

        time_range = cluster.get("time_range")
        if time_range and "span_days" in time_range:
            span = time_range["span_days"]
            if span > 30:
                insights.append(f"Обсуждение продолжалось в течение {span} дней")
            elif span > 7:
                insights.append(f"Активное обсуждение на протяжении {span} дней")

        chats = cluster.get("chats", {})
        if len(chats) > 1:
            insights.append(f"Тема обсуждается в {len(chats)} разных чатах")

        total_messages = cluster.get("total_messages", 0)
        if total_messages > 100:
            insights.append(f"Высокая активность: {total_messages} сообщений")

        if not insights:
            insights = ["Требуется дополнительный анализ для выявления инсайтов"]

        return {"title": title, "description": description, "key_insights": insights}


async def summarize_all_clusters(
    clusters: List[Dict[str, Any]], embedding_client, progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Генерация сводок для всех кластеров

    Args:
        clusters: Список кластеров
        embedding_client: LangChainLLMAdapter
        progress_callback: Функция для отчёта о прогрессе

    Returns:
        Список кластеров с сводками
    """
    summarizer = ClusterSummarizer(embedding_client)

    enriched_clusters = []

    for i, cluster in enumerate(clusters, 1):
        if progress_callback:
            progress_callback(i, len(clusters), cluster["cluster_id"])

        try:
            summary = await summarizer.summarize_cluster(cluster)
            cluster["summary"] = summary
            enriched_clusters.append(cluster)

            # Задержка между запросами
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Ошибка обработки кластера {cluster['cluster_id']}: {e}")
            cluster["summary"] = summarizer._fallback_summary(cluster)
            enriched_clusters.append(cluster)

    return enriched_clusters
