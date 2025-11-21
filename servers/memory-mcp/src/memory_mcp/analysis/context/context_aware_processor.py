"""Контекстно-осознанная обработка с интеграцией smart_search."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from typing import TYPE_CHECKING

from ..aggregation.large_context_processor import LargeContextProcessor

if TYPE_CHECKING:
    from ..search import SmartSearchEngine

logger = logging.getLogger(__name__)


class ContextAwareProcessor:
    """
    Контекстно-осознанная обработка с интеграцией smart_search.
    
    Использует smart_search для:
    - Анализа контекста чата перед обработкой
    - Извлечения ключевых тем и паттернов
    - Адаптации стратегии группировки
    - Улучшения качества саммаризации
    """

    def __init__(
        self,
        large_context_processor: LargeContextProcessor,
        smart_search_engine: Optional["SmartSearchEngine"] = None,
        enable_smart_search: bool = True,
    ):
        """
        Инициализация контекстно-осознанного процессора.

        Args:
            large_context_processor: Процессор больших контекстов
            smart_search_engine: Движок smart_search (опционально)
            enable_smart_search: Включить использование smart_search
        """
        self.large_context_processor = large_context_processor
        self.smart_search_engine = smart_search_engine
        self.enable_smart_search = enable_smart_search and smart_search_engine is not None

    async def process_with_context_awareness(
        self,
        messages: List[Dict[str, Any]],
        chat_name: Optional[str] = None,
        summarization_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Обработка сообщений с учетом контекста через smart_search.

        Workflow:
        1. Перед обработкой сессии → smart_search для анализа контекста
        2. Извлечение ключевых тем и паттернов
        3. Адаптация группировки на основе найденных паттернов
        4. Обработка большим контекстом с учетом найденных тем
        5. Сохранение результатов для улучшения smart_search

        Args:
            messages: Список сообщений для обработки
            chat_name: Название чата
            summarization_prompt: Промпт для саммаризации

        Returns:
            Словарь с результатами обработки, включая контекстную информацию
        """
        context_insights = None

        # Шаг 1: Анализ контекста через smart_search
        if self.enable_smart_search and chat_name:
            context_insights = await self._analyze_context_with_smart_search(
                messages, chat_name
            )

        # Шаг 2: Адаптация промпта на основе контекста
        enhanced_prompt = self._enhance_prompt_with_context(
            summarization_prompt, context_insights, chat_name
        )

        # Шаг 3: Обработка большим контекстом
        result = await self.large_context_processor.process_large_context(
            messages, chat_name, enhanced_prompt
        )

        # Шаг 4: Добавляем контекстную информацию в результат
        result["context_insights"] = context_insights
        result["enhanced_prompt_used"] = enhanced_prompt != summarization_prompt

        return result

    async def _analyze_context_with_smart_search(
        self, messages: List[Dict[str, Any]], chat_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Анализ контекста чата через smart_search.

        Args:
            messages: Список сообщений
            chat_name: Название чата

        Returns:
            Словарь с инсайтами контекста или None
        """
        if not self.smart_search_engine:
            return None

        try:
            # Импортируем SmartSearchRequest локально для избежания циклических зависимостей
            from ..mcp.schema import SmartSearchRequest
            
            # Формируем запрос для анализа контекста
            # Используем первые и последние сообщения для понимания тематики
            sample_messages = messages[:10] + messages[-10:] if len(messages) > 20 else messages
            context_query = self._extract_context_query(sample_messages, chat_name)

            # Выполняем поиск для анализа контекста
            search_request = SmartSearchRequest(
                query=context_query,
                top_k=10,
                source="telegram",
                clarify=False,  # Не запрашиваем уточнения для анализа
            )

            search_response = await self.smart_search_engine.search(search_request)

            # Извлекаем ключевые темы и паттерны
            insights = self._extract_insights_from_search(search_response, chat_name)

            logger.info(
                f"Извлечено {len(insights.get('key_topics', []))} ключевых тем "
                f"для чата {chat_name} через smart_search"
            )

            return insights

        except Exception as e:
            logger.warning(
                f"Ошибка анализа контекста через smart_search для {chat_name}: {e}",
                exc_info=True,
            )
            return None

    def _extract_context_query(
        self, messages: List[Dict[str, Any]], chat_name: str
    ) -> str:
        """
        Извлечение поискового запроса из сообщений для анализа контекста.

        Args:
            messages: Список сообщений
            chat_name: Название чата

        Returns:
            Поисковый запрос
        """
        # Извлекаем ключевые слова из сообщений
        texts = []
        for msg in messages:
            text = msg.get("text") or msg.get("content") or ""
            if text and len(text) > 10:  # Игнорируем очень короткие сообщения
                texts.append(text[:200])  # Берем первые 200 символов

        # Формируем запрос из первых слов сообщений
        if texts:
            # Берем первые слова из каждого сообщения
            words = []
            for text in texts[:5]:  # Берем первые 5 сообщений
                words.extend(text.split()[:5])  # Первые 5 слов

            # Убираем дубликаты и формируем запрос
            unique_words = list(dict.fromkeys(words))[:10]  # Максимум 10 уникальных слов
            query = " ".join(unique_words)
        else:
            query = chat_name or "обсуждение"

        return query

    def _extract_insights_from_search(
        self, search_response: Any, chat_name: str
    ) -> Dict[str, Any]:
        """
        Извлечение инсайтов из результатов smart_search.

        Args:
            search_response: Ответ smart_search
            chat_name: Название чата

        Returns:
            Словарь с инсайтами
        """
        insights = {
            "key_topics": [],
            "important_entities": [],
            "temporal_patterns": [],
            "confidence": search_response.confidence_score if hasattr(search_response, 'confidence_score') else 0.0,
        }

        # Извлекаем ключевые темы из результатов поиска
        if hasattr(search_response, 'results') and search_response.results:
            # Анализируем топ-результаты для извлечения тем
            for result in search_response.results[:5]:
                content = result.content if hasattr(result, 'content') else str(result)
                # Простое извлечение тем (можно улучшить через LLM)
                topics = self._extract_topics_from_text(content)
                insights["key_topics"].extend(topics)

        # Убираем дубликаты
        insights["key_topics"] = list(dict.fromkeys(insights["key_topics"]))[:10]

        return insights

    def _extract_topics_from_text(self, text: str) -> List[str]:
        """
        Простое извлечение тем из текста.

        В будущем можно улучшить через LLM для более точного извлечения.

        Args:
            text: Текст для анализа

        Returns:
            Список тем
        """
        # Простая эвристика: ищем слова с заглавной буквы и часто встречающиеся слова
        import re

        topics = []

        # Слова с заглавной буквы (возможные имена, названия)
        capitalized = re.findall(r'\b[A-ZА-Я][a-zа-я]+\b', text)
        topics.extend(capitalized[:5])

        # Часто встречающиеся существительные (упрощенная версия)
        words = re.findall(r'\b[а-яА-Я]{4,}\b', text.lower())
        if words:
            from collections import Counter
            common_words = [word for word, count in Counter(words).most_common(5)]
            topics.extend(common_words)

        return topics[:10]  # Максимум 10 тем

    def _enhance_prompt_with_context(
        self,
        original_prompt: Optional[str],
        context_insights: Optional[Dict[str, Any]],
        chat_name: Optional[str],
    ) -> str:
        """
        Улучшение промпта на основе контекстных инсайтов.

        Args:
            original_prompt: Исходный промпт
            context_insights: Инсайты из smart_search
            chat_name: Название чата

        Returns:
            Улучшенный промпт
        """
        if not context_insights:
            return original_prompt or self._create_default_prompt(chat_name)

        # Формируем улучшенный промпт с учетом контекста
        enhanced_parts = []

        if original_prompt:
            enhanced_parts.append(original_prompt)
        else:
            enhanced_parts.append(self._create_default_prompt(chat_name))

        # Добавляем контекстную информацию
        if context_insights.get("key_topics"):
            topics = ", ".join(context_insights["key_topics"][:5])
            enhanced_parts.append(
                f"\n\nВажно: В этом чате часто обсуждаются следующие темы: {topics}. "
                f"Учти это при создании саммаризации."
            )

        return "\n".join(enhanced_parts)

    def _create_default_prompt(self, chat_name: Optional[str]) -> str:
        """Создание стандартного промпта."""
        return f"""Создай структурированную саммаризацию сообщений из чата {chat_name or ""}.

Выдели:
1. Основные темы обсуждения
2. Ключевые события и моменты
3. Участники и их вклад
4. Важные детали и контекст

Будь внимателен к контексту и связям между сообщениями."""

