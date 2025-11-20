"""Обработка больших контекстов для эффективного использования LLM."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.lmstudio_client import LMStudioEmbeddingClient
from ..core.ollama_client import OllamaEmbeddingClient
from .adaptive_message_grouper import AdaptiveMessageGrouper

logger = logging.getLogger(__name__)


class LargeContextProcessor:
    """
    Обработчик больших контекстов для эффективного использования LLM.
    
    Особенности:
    - Определение оптимального размера контекста на основе модели
    - Иерархическая обработка: сначала общий контекст, потом детализация
    - Кэширование промежуточных результатов
    - Интеграция с AdaptiveMessageGrouper для умной группировки
    """

    def __init__(
        self,
        max_tokens: int = 131072,  # Максимальный контекст модели
        prompt_reserve_tokens: int = 5000,
        hierarchical_threshold: int = 100000,  # Порог для иерархической обработки
        embedding_client: Optional[LMStudioEmbeddingClient | OllamaEmbeddingClient] = None,
        enable_hierarchical: bool = True,
        enable_caching: bool = True,
    ):
        """
        Инициализация обработчика больших контекстов.

        Args:
            max_tokens: Максимальный контекст модели
            prompt_reserve_tokens: Резерв токенов для промпта
            hierarchical_threshold: Порог для включения иерархической обработки
            embedding_client: Клиент для LLM (опционально)
            enable_hierarchical: Включить иерархическую обработку
            enable_caching: Включить кэширование промежуточных результатов
        """
        self.max_tokens = max_tokens
        self.prompt_reserve_tokens = prompt_reserve_tokens
        self.available_tokens = max_tokens - prompt_reserve_tokens
        self.hierarchical_threshold = hierarchical_threshold
        self.embedding_client = embedding_client
        self.enable_hierarchical = enable_hierarchical
        self.enable_caching = enable_caching

        # Инициализируем группировщик
        self.grouper = AdaptiveMessageGrouper(
            max_tokens=self.available_tokens,
            prompt_reserve_tokens=prompt_reserve_tokens,
            strategy="hybrid",
        )

        # Кэш для промежуточных результатов
        self._cache: Dict[str, Any] = {}

    def estimate_tokens(self, text: str) -> int:
        """Оценка количества токенов в тексте."""
        return self.grouper.estimate_tokens(text)

    def estimate_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Оценка количества токенов в группе сообщений."""
        return self.grouper.estimate_group_tokens(messages)

    def should_use_hierarchical(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Определение необходимости иерархической обработки.

        Args:
            messages: Список сообщений

        Returns:
            True, если нужна иерархическая обработка
        """
        if not self.enable_hierarchical:
            return False

        total_tokens = self.estimate_messages_tokens(messages)
        return total_tokens > self.hierarchical_threshold

    async def process_large_context(
        self,
        messages: List[Dict[str, Any]],
        chat_name: Optional[str] = None,
        summarization_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Обработка большого контекста с адаптивной стратегией.

        Алгоритм:
        1. Оценка размера контекста в токенах
        2. Если < hierarchical_threshold → обрабатываем целиком одним запросом
        3. Если > hierarchical_threshold → иерархическая обработка:
           - Уровень 1: Общая саммаризация всего контекста (сжатие до 20K токенов)
           - Уровень 2: Детальная обработка частей с учетом общей саммаризации

        Args:
            messages: Список сообщений для обработки
            chat_name: Название чата (опционально)
            summarization_prompt: Промпт для саммаризации (опционально)

        Returns:
            Словарь с результатами обработки:
            - summary: Общая саммаризация
            - detailed_summaries: Детальные саммаризации частей
            - groups: Группы сообщений
            - tokens_used: Использовано токенов
        """
        if not messages:
            return {
                "summary": "",
                "detailed_summaries": [],
                "groups": [],
                "tokens_used": 0,
            }

        total_tokens = self.estimate_messages_tokens(messages)
        logger.info(
            f"Обработка контекста: {len(messages)} сообщений, "
            f"~{total_tokens} токенов для чата {chat_name}"
        )

        # Проверяем кэш
        cache_key = self._get_cache_key(messages, chat_name)
        if self.enable_caching and cache_key in self._cache:
            logger.info(f"Использован кэш для {chat_name}")
            return self._cache[cache_key]

        # Если контекст помещается в один запрос
        if total_tokens <= self.available_tokens:
            logger.info(f"Контекст помещается в один запрос ({total_tokens} токенов)")
            result = await self._process_single_request(
                messages, chat_name, summarization_prompt
            )
        elif self.should_use_hierarchical(messages):
            logger.info(
                f"Используется иерархическая обработка "
                f"({total_tokens} токенов > {self.hierarchical_threshold})"
            )
            result = await self._process_hierarchical(
                messages, chat_name, summarization_prompt
            )
        else:
            # Группируем и обрабатываем по частям
            logger.info(f"Группировка и обработка по частям ({total_tokens} токенов)")
            result = await self._process_grouped(
                messages, chat_name, summarization_prompt
            )

        # Сохраняем в кэш
        if self.enable_caching:
            self._cache[cache_key] = result

        return result

    async def process_batch_sessions(
        self,
        sessions: List[Dict[str, Any]],
        chat_name: str,
        accumulative_context: Optional[str] = None,
        summarization_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Обработка батча сессий с накопительным контекстом.

        Args:
            sessions: Список сессий для обработки
            chat_name: Название чата
            accumulative_context: Накопительный контекст
            summarization_prompt: Промпт для саммаризации (опционально)

        Returns:
            Словарь с результатами обработки:
            - summary: Общая саммаризация
            - detailed_summaries: Детальные саммаризации для каждой сессии
            - groups: Группы сообщений
            - tokens_used: Использовано токенов
        """
        if not sessions:
            return {
                "summary": "",
                "detailed_summaries": [],
                "groups": [],
                "tokens_used": 0,
            }

        # Объединяем все сообщения из сессий
        all_messages = []
        session_boundaries = []  # Границы сессий в объединенном списке

        for session in sessions:
            session_start = len(all_messages)
            messages = session.get("messages", [])
            all_messages.extend(messages)
            session_end = len(all_messages)
            session_boundaries.append({
                "session_id": session.get("session_id", "unknown"),
                "start_index": session_start,
                "end_index": session_end,
                "original_session": session,
            })

        # Оцениваем размер накопительного контекста
        context_tokens = self.estimate_tokens(accumulative_context) if accumulative_context else 0
        available_for_messages = self.available_tokens - context_tokens

        # Если накопительный контекст слишком большой, используем минимальный резерв
        if available_for_messages <= 0:
            available_for_messages = self.available_tokens // 2

        # Оцениваем размер всех сообщений
        total_tokens = self.estimate_messages_tokens(all_messages)

        logger.info(
            f"Обработка батча сессий: {len(sessions)} сессий, "
            f"{len(all_messages)} сообщений, ~{total_tokens} токенов "
            f"(контекст: {context_tokens} токенов, доступно: {available_for_messages})"
        )

        # Если все помещается в один запрос
        if total_tokens <= available_for_messages:
            logger.info(f"Батч помещается в один запрос ({total_tokens} токенов)")
            result = await self._process_batch_single_request(
                all_messages, sessions, session_boundaries, chat_name,
                accumulative_context, summarization_prompt
            )
        elif self.should_use_hierarchical(all_messages):
            logger.info(
                f"Используется иерархическая обработка батча "
                f"({total_tokens} токенов > {self.hierarchical_threshold})"
            )
            result = await self._process_batch_hierarchical(
                all_messages, sessions, session_boundaries, chat_name,
                accumulative_context, summarization_prompt
            )
        else:
            # Группируем и обрабатываем по частям
            logger.info(f"Группировка батча и обработка по частям ({total_tokens} токенов)")
            result = await self._process_batch_grouped(
                all_messages, sessions, session_boundaries, chat_name,
                accumulative_context, summarization_prompt
            )

        return result

    async def _process_batch_single_request(
        self,
        all_messages: List[Dict[str, Any]],
        sessions: List[Dict[str, Any]],
        session_boundaries: List[Dict[str, Any]],
        chat_name: str,
        accumulative_context: Optional[str],
        prompt: Optional[str],
    ) -> Dict[str, Any]:
        """Обработка батча одним запросом."""
        # Формируем текст из сообщений
        context_text = self._format_messages_for_llm(all_messages)

        # Добавляем накопительный контекст в промпт, если он есть
        enhanced_prompt = prompt or ""
        if accumulative_context:
            enhanced_prompt = f"{enhanced_prompt}\n\nНакопительный контекст чата:\n{accumulative_context}\n" if prompt else f"Накопительный контекст чата:\n{accumulative_context}\n"

        # Генерируем саммаризацию
        summary = await self._generate_summary(context_text, enhanced_prompt, chat_name)

        # Создаем детальные саммаризации для каждой сессии (используем общую саммаризацию)
        detailed_summaries = [
            {"summary": summary, "session_id": session.get("session_id", "unknown")}
            for session in sessions
        ]

        return {
            "summary": summary,
            "detailed_summaries": detailed_summaries,
            "groups": [all_messages],
            "tokens_used": self.estimate_tokens(context_text) + self.estimate_tokens(enhanced_prompt),
        }

    async def _process_batch_hierarchical(
        self,
        all_messages: List[Dict[str, Any]],
        sessions: List[Dict[str, Any]],
        session_boundaries: List[Dict[str, Any]],
        chat_name: str,
        accumulative_context: Optional[str],
        prompt: Optional[str],
    ) -> Dict[str, Any]:
        """Иерархическая обработка батча."""
        # Используем стандартную иерархическую обработку
        enhanced_prompt = prompt or ""
        if accumulative_context:
            enhanced_prompt = f"{enhanced_prompt}\n\nНакопительный контекст чата:\n{accumulative_context}\n" if prompt else f"Накопительный контекст чата:\n{accumulative_context}\n"

        result = await self._process_hierarchical(
            all_messages, chat_name, enhanced_prompt
        )

        # Адаптируем результат для батча сессий
        detailed_summaries = result.get("detailed_summaries", [])
        if len(detailed_summaries) != len(sessions):
            # Если количество не совпадает, создаем саммаризации для каждой сессии
            overall_summary = result.get("summary", "")
            detailed_summaries = [
                {"summary": overall_summary, "session_id": session.get("session_id", "unknown")}
                for session in sessions
            ]

        return {
            "summary": result.get("summary", ""),
            "detailed_summaries": detailed_summaries,
            "groups": result.get("groups", []),
            "tokens_used": result.get("tokens_used", 0),
        }

    async def _process_batch_grouped(
        self,
        all_messages: List[Dict[str, Any]],
        sessions: List[Dict[str, Any]],
        session_boundaries: List[Dict[str, Any]],
        chat_name: str,
        accumulative_context: Optional[str],
        prompt: Optional[str],
    ) -> Dict[str, Any]:
        """Группировка батча и обработка по частям."""
        # Группируем сообщения
        groups = self.grouper.group_messages_adaptively(all_messages, chat_name)

        enhanced_prompt = prompt or ""
        if accumulative_context:
            enhanced_prompt = f"{enhanced_prompt}\n\nНакопительный контекст чата:\n{accumulative_context}\n" if prompt else f"Накопительный контекст чата:\n{accumulative_context}\n"

        # Обрабатываем каждую группу
        group_summaries = []
        total_tokens = 0

        for group in groups:
            group_text = self._format_messages_for_llm(group)
            group_summary = await self._generate_summary(group_text, enhanced_prompt, chat_name)
            group_summaries.append(group_summary)
            total_tokens += self.estimate_tokens(group_text) + self.estimate_tokens(enhanced_prompt)

        # Объединяем саммаризации групп
        overall_summary = "\n\n".join(group_summaries)

        # Создаем детальные саммаризации для каждой сессии
        # Распределяем саммаризации групп по сессиям на основе границ
        detailed_summaries = []
        for session in sessions:
            session_id = session.get("session_id", "unknown")
            # Используем общую саммаризацию для каждой сессии
            detailed_summaries.append({
                "summary": overall_summary,
                "session_id": session_id,
            })

        return {
            "summary": overall_summary,
            "detailed_summaries": detailed_summaries,
            "groups": groups,
            "tokens_used": total_tokens,
        }

    async def _process_single_request(
        self,
        messages: List[Dict[str, Any]],
        chat_name: Optional[str],
        prompt: Optional[str],
    ) -> Dict[str, Any]:
        """Обработка контекста одним запросом."""
        # Формируем текст из сообщений
        context_text = self._format_messages_for_llm(messages)

        # Генерируем саммаризацию
        summary = await self._generate_summary(context_text, prompt, chat_name)

        return {
            "summary": summary,
            "detailed_summaries": [summary],
            "groups": [messages],
            "tokens_used": self.estimate_tokens(context_text),
        }

    async def _process_hierarchical(
        self,
        messages: List[Dict[str, Any]],
        chat_name: Optional[str],
        prompt: Optional[str],
    ) -> Dict[str, Any]:
        """
        Иерархическая обработка большого контекста.

        Уровень 1: Общая саммаризация всего контекста
        Уровень 2: Детальная обработка частей с учетом общей саммаризации
        """
        # Уровень 1: Общая саммаризация
        logger.info("Уровень 1: Генерация общей саммаризации")

        # Создаем сжатую версию контекста для общей саммаризации
        compressed_context = self._compress_context(messages, target_tokens=20000)
        level1_prompt = self._create_hierarchical_prompt(
            compressed_context, level=1, chat_name=chat_name
        )
        overall_summary = await self._generate_summary(
            compressed_context, level1_prompt, chat_name
        )

        # Уровень 2: Детальная обработка частей
        logger.info("Уровень 2: Детальная обработка частей")

        # Группируем сообщения
        groups = self.grouper.group_messages_adaptively(messages, chat_name)

        detailed_summaries = []
        total_tokens = 0

        for i, group in enumerate(groups):
            group_text = self._format_messages_for_llm(group)
            group_tokens = self.estimate_tokens(group_text)

            # Создаем промпт с учетом общей саммаризации
            level2_prompt = self._create_hierarchical_prompt(
                group_text,
                level=2,
                chat_name=chat_name,
                overall_summary=overall_summary,
                group_index=i + 1,
                total_groups=len(groups),
            )

            group_summary = await self._generate_summary(
                group_text, level2_prompt, chat_name
            )

            detailed_summaries.append({
                "group_index": i + 1,
                "summary": group_summary,
                "messages_count": len(group),
                "tokens": group_tokens,
            })

            total_tokens += group_tokens

        return {
            "summary": overall_summary,
            "detailed_summaries": detailed_summaries,
            "groups": groups,
            "tokens_used": total_tokens,
            "hierarchical": True,
        }

    async def _process_grouped(
        self,
        messages: List[Dict[str, Any]],
        chat_name: Optional[str],
        prompt: Optional[str],
    ) -> Dict[str, Any]:
        """Обработка контекста по группам без иерархии."""
        # Группируем сообщения
        groups = self.grouper.group_messages_adaptively(messages, chat_name)

        summaries = []
        total_tokens = 0

        for i, group in enumerate(groups):
            group_text = self._format_messages_for_llm(group)
            group_tokens = self.estimate_tokens(group_text)

            group_prompt = prompt or self._create_default_prompt(
                chat_name, group_index=i + 1, total_groups=len(groups)
            )

            group_summary = await self._generate_summary(
                group_text, group_prompt, chat_name
            )

            summaries.append({
                "group_index": i + 1,
                "summary": group_summary,
                "messages_count": len(group),
                "tokens": group_tokens,
            })

            total_tokens += group_tokens

        # Объединяем саммаризации
        combined_summary = self._combine_summaries(summaries)

        return {
            "summary": combined_summary,
            "detailed_summaries": summaries,
            "groups": groups,
            "tokens_used": total_tokens,
            "hierarchical": False,
        }

    def _format_messages_for_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Форматирование сообщений для LLM."""
        lines = []
        for msg in messages:
            author = msg.get("from") or msg.get("author") or "Unknown"
            text = msg.get("text") or msg.get("content") or ""
            date = msg.get("date") or msg.get("timestamp") or ""

            if text:
                lines.append(f"[{date}] {author}: {text}")

        return "\n".join(lines)

    def _compress_context(
        self, messages: List[Dict[str, Any]], target_tokens: int
    ) -> str:
        """
        Сжатие контекста до целевого количества токенов.

        Стратегия:
        - Берем первые и последние сообщения
        - Добавляем сообщения с высокой важностью (если есть метрики)
        - Сжимаем длинные сообщения
        """
        total_tokens = self.estimate_messages_tokens(messages)

        if total_tokens <= target_tokens:
            return self._format_messages_for_llm(messages)

        # Берем первые 30% и последние 30% сообщений
        first_count = int(len(messages) * 0.3)
        last_count = int(len(messages) * 0.3)

        selected = messages[:first_count] + messages[-last_count:]

        # Если все еще слишком много, сжимаем сообщения
        compressed_text = self._format_messages_for_llm(selected)
        compressed_tokens = self.estimate_tokens(compressed_text)

        if compressed_tokens > target_tokens:
            # Сжимаем текст, обрезая длинные сообщения
            ratio = target_tokens / compressed_tokens
            max_chars_per_msg = int(200 * ratio)  # Примерно 200 символов на сообщение

            compressed_lines = []
            for msg in selected:
                author = msg.get("from") or msg.get("author") or "Unknown"
                text = msg.get("text") or msg.get("content") or ""
                if len(text) > max_chars_per_msg:
                    text = text[:max_chars_per_msg] + "..."
                if text:
                    compressed_lines.append(f"{author}: {text}")

            compressed_text = "\n".join(compressed_lines)

        return compressed_text

    def _create_hierarchical_prompt(
        self,
        context: str,
        level: int,
        chat_name: Optional[str] = None,
        overall_summary: Optional[str] = None,
        group_index: Optional[int] = None,
        total_groups: Optional[int] = None,
    ) -> str:
        """Создание промпта для иерархической обработки."""
        if level == 1:
            return f"""Создай краткую общую саммаризацию следующего контекста чата {chat_name or ""}.

Контекст содержит множество сообщений. Твоя задача - выделить основные темы, ключевые события и общий тон обсуждения.

Контекст:
{context[:50000]}  # Ограничиваем для промпта

Создай структурированную саммаризацию с разделами:
1. Основные темы
2. Ключевые события
3. Участники и их роли
4. Общий тон и атмосфера"""

        elif level == 2:
            group_info = f" (группа {group_index} из {total_groups})" if group_index else ""
            summary_context = f"\n\nОбщая саммаризация всего контекста:\n{overall_summary}\n" if overall_summary else ""

            return f"""Создай детальную саммаризацию части контекста чата {chat_name or ""}{group_info}.

{summary_context}

Детальный контекст группы:
{context[:80000]}  # Ограничиваем для промпта

Создай детальную саммаризацию, которая:
1. Соответствует общей саммаризации выше
2. Раскрывает детали этой части обсуждения
3. Выделяет важные моменты и переходы тем"""

        return "Создай саммаризацию следующего контекста."

    def _create_default_prompt(
        self,
        chat_name: Optional[str] = None,
        group_index: Optional[int] = None,
        total_groups: Optional[int] = None,
    ) -> str:
        """Создание стандартного промпта для саммаризации."""
        group_info = f" (группа {group_index} из {total_groups})" if group_index else ""
        return f"""Создай структурированную саммаризацию сообщений из чата {chat_name or ""}{group_info}.

Выдели:
1. Основные темы обсуждения
2. Ключевые события и моменты
3. Участники и их вклад
4. Важные детали и контекст"""

    async def _generate_summary(
        self,
        context: str,
        prompt: Optional[str],
        chat_name: Optional[str],
    ) -> str:
        """Генерация саммаризации через LLM."""
        if not self.embedding_client:
            raise ValueError("LLM клиент не настроен для генерации саммаризации")

        full_prompt = f"{prompt or self._create_default_prompt(chat_name)}\n\n{context}"

        async with self.embedding_client:
            summary = await self.embedding_client.generate_summary(
                prompt=full_prompt,
                temperature=0.3,
                max_tokens=30000,  # Уменьшено для предотвращения таймаутов
                top_p=0.9,
                presence_penalty=0.1,
            )

        return summary.strip()

    def _combine_summaries(self, summaries: List[Dict[str, Any]]) -> str:
        """Объединение нескольких саммаризаций в одну."""
        if not summaries:
            return ""

        if len(summaries) == 1:
            return summaries[0].get("summary", "")

        combined = []
        for summary_data in summaries:
            group_info = f"Группа {summary_data.get('group_index', '?')}:"
            summary_text = summary_data.get("summary", "")
            combined.append(f"{group_info}\n{summary_text}\n")

        return "\n---\n\n".join(combined)

    def _get_cache_key(
        self, messages: List[Dict[str, Any]], chat_name: Optional[str]
    ) -> str:
        """Генерация ключа кэша."""
        import hashlib
        import json

        # Используем первые и последние сообщения + количество для ключа
        key_data = {
            "chat": chat_name,
            "count": len(messages),
            "first": messages[0].get("text", "")[:100] if messages else "",
            "last": messages[-1].get("text", "")[:100] if messages else "",
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._cache.clear()

