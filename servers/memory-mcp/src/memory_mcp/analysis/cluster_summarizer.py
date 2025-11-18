#!/usr/bin/env python3
"""
Генерация сводок для кластеров сессий через LLM

Создаёт связные описания тематических кластеров.
"""

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ClusterSummarizer:
    """Генератор сводок для кластеров через LLM"""

    def __init__(self, embedding_client):
        """
        Инициализация

        Args:
            embedding_client: LMStudioEmbeddingClient для взаимодействия с LLM
        """
        self.embedding_client = embedding_client
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

        # Генерируем сводку через LLM
        try:
            summary = await self._generate_llm_summary(cluster_content, cluster)
            return summary
        except Exception as e:
            logger.error(f"Ошибка генерации LLM сводки: {e}")
            # Возвращаем базовую сводку
            return self._fallback_summary(cluster)

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

    async def _generate_llm_summary(
        self, cluster_content: str, cluster: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Генерация сводки через LLM

        Args:
            cluster_content: Контент кластера
            cluster: Метаданные кластера

        Returns:
            Сводка
        """
        # Формируем промпт
        prompt = f"""Ты - аналитик, который создаёт краткие сводки для тематических кластеров обсуждений в Telegram чатах.

Дано:
{cluster_content}

Задача:
1. Определи **одну главную тему**, которая объединяет все эти сессии (макс 10 слов)
2. Напиши краткое описание кластера (2-3 предложения)
3. Выдели 3-5 ключевых инсайтов из обсуждений

Формат ответа (строго JSON):
{{
  "title": "Главная тема кластера",
  "description": "Краткое описание того, что обсуждается",
  "key_insights": [
    "Инсайт 1",
    "Инсайт 2",
    "Инсайт 3"
  ]
}}

Ответ:"""

        # Запрашиваем LLM
        try:
            async with self.embedding_client:
                response = await self.embedding_client.generate_summary(
                    prompt, max_tokens=131072, temperature=0.3  # Для gpt-oss-20b (максимальный лимит)
                )

            # Парсим JSON ответ
            import json
            import re

            # Пробуем найти JSON в ответе
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                summary_data = json.loads(json_match.group(0))
            else:
                raise ValueError("Не найден JSON в ответе LLM")

            logger.info("LLM сводка сгенерирована успешно")

            return {
                "title": summary_data.get("title", "Тематический кластер"),
                "description": summary_data.get("description", ""),
                "key_insights": summary_data.get("key_insights", []),
            }

        except Exception as e:
            logger.error(f"Ошибка парсинга LLM ответа: {e}")
            raise

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
        embedding_client: LMStudioEmbeddingClient
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
