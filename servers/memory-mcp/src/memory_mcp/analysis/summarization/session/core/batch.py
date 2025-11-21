#!/usr/bin/env python3
"""
Методы батч-обработки для session_summarizer
"""

import logging
from typing import Any, Dict, List, Optional

from ...utils import detect_chat_mode, map_profile, prepare_conversation_text

logger = logging.getLogger(__name__)


async def summarize_batch_sessions(
    sessions: List[Dict[str, Any]],
    chat_name: str,
    large_context_processor,
    context_manager,
    accumulative_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Саммаризация нескольких сессий в одном запросе с использованием накопительного контекста.

    Args:
        sessions: Список сессий для саммаризации
        chat_name: Название чата
        large_context_processor: Процессор больших контекстов
        context_manager: Менеджер контекста
        accumulative_context: Накопительный контекст для улучшения понимания

    Returns:
        Список саммаризаций сессий
    """
    if not sessions:
        return []

    logger.info(
        f"Батч-саммаризация {len(sessions)} сессий для чата {chat_name}"
    )

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

    # Получаем контекст из предыдущих сессий
    previous_context = context_manager.get_previous_context(
        chat_name, sessions[0].get("session_id", "batch") if sessions else "batch"
    )

    # Формируем промпт с накопительным контекстом
    context_part = ""
    if accumulative_context:
        context_part = f"\n\nНакопительный контекст чата:\n{accumulative_context}\n"
    elif previous_context.get("recent_context"):
        context_part = f"\n\nКонтекст из предыдущих сессий:\n{previous_context['recent_context']}\n"

    # Определяем тип коммуникации (берем из первой сессии)
    chat_mode = detect_chat_mode(sessions[0].get("messages", []) if sessions else [])
    dominant_language = sessions[0].get("dominant_language", "ru") if sessions else "ru"

    # Формируем информацию о сессиях
    sessions_info = []
    for i, session in enumerate(sessions, 1):
        session_id = session.get("session_id", f"session_{i}")
        time_range = session.get("time_range", "unknown")
        message_count = len(session.get("messages", []))
        sessions_info.append(
            f"Сессия {i} ({session_id}, {time_range}): {message_count} сообщений"
        )

    # Подготавливаем текст разговора
    conversation_text = prepare_conversation_text(
        all_messages, max_messages=200, max_chars=100000
    )

    # Создаем промпт для батч-саммаризации
    prompt = f"""Создай саммаризацию следующих сессий из чата "{chat_name}".
{context_part}
Сессии для саммаризации:
{chr(10).join(sessions_info)}

Укажи для каждой сессии:
1. Основные темы обсуждения
2. Ключевые решения или события
3. Важные упоминания (люди, проекты)
4. Связи между сессиями

Сообщения:
{conversation_text}

Саммаризация должна быть структурированной и учитывать контекст предыдущих сообщений.
Для каждой сессии создай отдельную саммаризацию."""

    # Используем LargeContextProcessor для обработки большого контекста
    try:
        result = await large_context_processor.process_large_context(
            all_messages, chat_name, prompt
        )

        # Восстанавливаем структуру сессий из результата
        summaries = []
        detailed_summaries = result.get("detailed_summaries", [])

        # Если есть детальные саммаризации, используем их
        if detailed_summaries and len(detailed_summaries) == len(sessions):
            for i, (session, summary_data) in enumerate(zip(sessions, detailed_summaries)):
                session_summary = summary_data.get("summary", "")
                # Создаем полную саммаризацию в формате, совместимом с summarize_session
                summary_dict = {
                    "session_id": session.get("session_id", f"session_{i}"),
                    "chat": chat_name,
                    "summary": session_summary,
                    "topics": [{"summary": session_summary}],
                    "meta": {
                        "chat_name": chat_name,
                        "time_span": session.get("time_range", ""),
                        "profile": map_profile(chat_mode, session.get("messages", [])),
                    },
                    "batch_processed": True,
                }
                summaries.append(summary_dict)
        else:
            # Используем общую саммаризацию для всех сессий
            overall_summary = result.get("summary", "")
            for i, session in enumerate(sessions):
                summary_dict = {
                    "session_id": session.get("session_id", f"session_{i}"),
                    "chat": chat_name,
                    "summary": overall_summary,
                    "topics": [{"summary": overall_summary}],
                    "meta": {
                        "chat_name": chat_name,
                        "time_span": session.get("time_range", ""),
                        "profile": map_profile(chat_mode, session.get("messages", [])),
                    },
                    "batch_processed": True,
                }
                summaries.append(summary_dict)

        logger.info(
            f"Батч-саммаризация завершена: {len(summaries)} саммаризаций, "
            f"использовано {result.get('tokens_used', 0)} токенов"
        )

        return summaries

    except Exception as e:
        logger.error(f"Ошибка при батч-саммаризации: {e}")
        # В случае ошибки возвращаем пустые саммаризации
        return [
            {
                "session_id": session.get("session_id", f"session_{i}"),
                "chat": chat_name,
                "summary": f"Ошибка саммаризации: {str(e)}",
                "topics": [],
                "meta": {"chat_name": chat_name},
                "batch_processed": True,
            }
            for i, session in enumerate(sessions)
        ]

