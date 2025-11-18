"""Батч-обработка сессий для максимального использования контекста LLM."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.lmstudio_client import LMStudioEmbeddingClient
from ..core.ollama_client import OllamaEmbeddingClient
from .large_context_processor import LargeContextProcessor

logger = logging.getLogger(__name__)


class BatchSessionProcessor:
    """
    Процессор для батч-обработки сессий с максимальным использованием контекста LLM.
    
    Особенности:
    - Группирует несколько сессий/дней в один батч (до 131K токенов)
    - Адаптивное заполнение контекста
    - Учитывает накопительный контекст при расчете размера батча
    - Интеграция с LargeContextProcessor для оценки токенов
    """

    def __init__(
        self,
        max_tokens: int = 131072,  # Максимальный контекст модели
        prompt_reserve_tokens: int = 5000,
        context_reserve_tokens: int = 10000,  # Резерв для накопительного контекста
        embedding_client: Optional[LMStudioEmbeddingClient | OllamaEmbeddingClient] = None,
    ):
        """
        Инициализация батч-процессора.

        Args:
            max_tokens: Максимальный контекст модели
            prompt_reserve_tokens: Резерв токенов для промпта
            context_reserve_tokens: Резерв токенов для накопительного контекста
            embedding_client: Клиент для LLM (опционально)
        """
        self.max_tokens = max_tokens
        self.prompt_reserve_tokens = prompt_reserve_tokens
        self.context_reserve_tokens = context_reserve_tokens
        self.available_tokens = max_tokens - prompt_reserve_tokens - context_reserve_tokens
        self.embedding_client = embedding_client
        
        # Инициализируем LargeContextProcessor для оценки токенов
        self.large_context_processor = LargeContextProcessor(
            max_tokens=max_tokens,
            prompt_reserve_tokens=prompt_reserve_tokens,
            embedding_client=embedding_client,
            enable_hierarchical=False,  # Батч-обработка сама управляет размером
        )

    def estimate_session_tokens(self, session: Dict[str, Any]) -> int:
        """
        Оценка количества токенов в сессии.

        Args:
            session: Сессия для оценки

        Returns:
            Примерное количество токенов
        """
        messages = session.get("messages", [])
        return self.large_context_processor.estimate_messages_tokens(messages)

    def estimate_context_tokens(self, context: Optional[str]) -> int:
        """
        Оценка количества токенов в накопительном контексте.

        Args:
            context: Контекст для оценки

        Returns:
            Примерное количество токенов
        """
        if not context:
            return 0
        return self.large_context_processor.estimate_tokens(context)

    def create_batches(
        self,
        sessions: List[Dict[str, Any]],
        accumulative_context: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Создает батчи из сессий с учетом лимита токенов.

        Args:
            sessions: Список сессий для батчирования
            accumulative_context: Накопительный контекст (учитывается при расчете размера)

        Returns:
            Список батчей, каждый батч - список сессий
        """
        if not sessions:
            return []

        # Оцениваем размер накопительного контекста
        context_tokens = self.estimate_context_tokens(accumulative_context)
        available_for_sessions = self.available_tokens - context_tokens

        if available_for_sessions <= 0:
            logger.warning(
                f"Накопительный контекст слишком большой ({context_tokens} токенов), "
                f"недостаточно места для сессий"
            )
            # Используем минимальный резерв для сессий
            available_for_sessions = self.available_tokens // 2

        batches = []
        current_batch = []
        current_batch_tokens = 0

        for session in sessions:
            session_tokens = self.estimate_session_tokens(session)

            # Если одна сессия больше доступного места, добавляем её отдельно
            if session_tokens > available_for_sessions:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0

                logger.warning(
                    f"Сессия {session.get('session_id', 'unknown')} слишком большая "
                    f"({session_tokens} токенов), добавляем отдельным батчем"
                )
                batches.append([session])
                continue

            # Проверяем, поместится ли сессия в текущий батч
            if current_batch_tokens + session_tokens <= available_for_sessions:
                current_batch.append(session)
                current_batch_tokens += session_tokens
            else:
                # Текущий батч заполнен, начинаем новый
                if current_batch:
                    batches.append(current_batch)
                current_batch = [session]
                current_batch_tokens = session_tokens

        # Добавляем последний батч, если он не пустой
        if current_batch:
            batches.append(current_batch)

        logger.info(
            f"Создано {len(batches)} батчей из {len(sessions)} сессий "
            f"(доступно токенов: {available_for_sessions}, контекст: {context_tokens})"
        )

        return batches

    async def process_batch(
        self,
        batch: List[Dict[str, Any]],
        chat_name: str,
        accumulative_context: Optional[str] = None,
        processing_type: str = "summarize",  # "summarize" или "regroup"
    ) -> Dict[str, Any]:
        """
        Обрабатывает батч сессий через LLM с накопительным контекстом.

        Args:
            batch: Батч сессий для обработки
            chat_name: Название чата
            accumulative_context: Накопительный контекст
            processing_type: Тип обработки ("summarize" или "regroup")

        Returns:
            Результат обработки батча
        """
        if not batch:
            return {"sessions": [], "total_tokens": 0}

        # Объединяем все сообщения из батча
        all_messages = []
        session_boundaries = []  # Границы сессий в объединенном списке

        for session in batch:
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

        # Создаем промпт в зависимости от типа обработки
        if processing_type == "regroup":
            prompt = self._create_regroup_prompt(chat_name, accumulative_context, batch)
        else:  # summarize
            prompt = self._create_summarize_prompt(chat_name, accumulative_context, batch)

        # Обрабатываем через LargeContextProcessor
        try:
            result = await self.large_context_processor.process_large_context(
                all_messages,
                chat_name,
                prompt,
            )
        except Exception as e:
            logger.error(f"Ошибка при обработке большого контекста: {e}", exc_info=True)
            # В случае ошибки возвращаем оригинальные сессии без саммаризации
            result = {
                "summary": "",
                "detailed_summaries": [],
                "groups": [],
                "tokens_used": 0,
            }

        # Восстанавливаем структуру сессий из результата
        processed_sessions = self._reconstruct_sessions_from_batch(
            batch, session_boundaries, result
        )

        return {
            "sessions": processed_sessions,
            "total_tokens": result.get("tokens_used", 0) if isinstance(result, dict) else 0,
            "summary": result.get("summary", "") if isinstance(result, dict) else "",
            "detailed_summaries": result.get("detailed_summaries", []) if isinstance(result, dict) else [],
        }

    def _create_regroup_prompt(
        self,
        chat_name: str,
        accumulative_context: Optional[str],
        batch: List[Dict[str, Any]],
    ) -> str:
        """Создает промпт для семантической перегруппировки."""
        context_part = ""
        if accumulative_context:
            context_part = f"\n\nНакопительный контекст чата:\n{accumulative_context}\n"

        sessions_info = []
        for i, session in enumerate(batch, 1):
            session_id = session.get("session_id", f"session_{i}")
            messages = session.get("messages", [])
            time_range = session.get("time_range", "unknown")
            sessions_info.append(
                f"Сессия {i} ({session_id}, {time_range}): {len(messages)} сообщений"
            )

        return f"""Проанализируй следующие сессии из чата "{chat_name}" и перегруппируй их по смыслу и темам.
{context_part}
Сессии для анализа:
{chr(10).join(sessions_info)}

Задачи:
1. Определи основные темы и контекст каждой сессии
2. Объедини связанные по смыслу сессии в новые группы
3. Обоснуй каждую группировку
4. Сохрани хронологический порядок внутри групп

Верни структурированный ответ с перегруппированными сессиями."""

    def _create_summarize_prompt(
        self,
        chat_name: str,
        accumulative_context: Optional[str],
        batch: List[Dict[str, Any]],
    ) -> str:
        """Создает промпт для батч-саммаризации."""
        context_part = ""
        if accumulative_context:
            context_part = f"\n\nНакопительный контекст чата:\n{accumulative_context}\n"

        sessions_info = []
        for i, session in enumerate(batch, 1):
            session_id = session.get("session_id", f"session_{i}")
            messages = session.get("messages", [])
            time_range = session.get("time_range", "unknown")
            sessions_info.append(
                f"Сессия {i} ({session_id}, {time_range}): {len(messages)} сообщений"
            )

        return f"""Создай саммаризацию следующих сессий из чата "{chat_name}".
{context_part}
Сессии для саммаризации:
{chr(10).join(sessions_info)}

Укажи:
1. Основные темы обсуждения
2. Ключевые решения или события
3. Важные упоминания (люди, проекты)
4. Связи между сессиями

Саммаризация должна быть структурированной и учитывать контекст предыдущих сообщений."""

    def _reconstruct_sessions_from_batch(
        self,
        original_batch: List[Dict[str, Any]],
        session_boundaries: List[Dict[str, Any]],
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Восстанавливает структуру сессий из результата батч-обработки.

        Args:
            original_batch: Оригинальный батч сессий
            session_boundaries: Границы сессий
            result: Результат обработки от LargeContextProcessor

        Returns:
            Список обработанных сессий
        """
        processed_sessions = []

        # Если есть детальные саммаризации, используем их
        detailed_summaries = result.get("detailed_summaries", [])
        if detailed_summaries and len(detailed_summaries) == len(original_batch):
            for i, (session, summary_data) in enumerate(zip(original_batch, detailed_summaries)):
                processed_session = session.copy()
                processed_session["summary"] = summary_data.get("summary", "")
                processed_session["batch_processed"] = True
                processed_sessions.append(processed_session)
        else:
            # Используем общую саммаризацию для всех сессий
            overall_summary = result.get("summary", "")
            for session in original_batch:
                processed_session = session.copy()
                processed_session["summary"] = overall_summary
                processed_session["batch_processed"] = True
                processed_sessions.append(processed_session)

        return processed_sessions

