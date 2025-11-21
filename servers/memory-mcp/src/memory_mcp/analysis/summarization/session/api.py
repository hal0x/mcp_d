#!/usr/bin/env python3
"""
Публичный API для session_summarizer
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from ...core.langchain_adapters import LangChainLLMAdapter
from ..utils.instruction_manager import InstructionManager
from ..segmentation.session_segmentation import SessionSegmenter
from .summarizer import SessionSummarizer

logger = logging.getLogger(__name__)


async def summarize_chat_sessions(
    messages: List[Dict[str, Any]],
    chat_name: str,
    embedding_client: Optional[LangChainLLMAdapter] = None,
    summaries_dir: Path = Path("artifacts/reports"),
    instruction_manager: Optional[InstructionManager] = None,
) -> List[Dict[str, Any]]:
    """
    Удобная функция для саммаризации всех сессий в чате

    Args:
        messages: Список сообщений
        chat_name: Название чата
        embedding_client: Клиент для генерации эмбеддингов и текста (LM Studio)
        summaries_dir: Директория с саммаризациями для контекста
        instruction_manager: Менеджер специальных инструкций

    Returns:
        Список саммаризаций сессий
    """
    # Сегментируем на сессии
    segmenter = SessionSegmenter()
    sessions = segmenter.segment_messages(messages, chat_name)

    # Саммаризируем каждую сессию
    summarizer = SessionSummarizer(
        embedding_client,
        summaries_dir,
        instruction_manager=instruction_manager,
    )
    summaries = []

    for session in sessions:
        try:
            summary = await summarizer.summarize_session(session)
            summaries.append(summary)

            # Небольшая задержка между запросами
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Ошибка при саммаризации сессии {session['session_id']}: {e}")
            continue

    return summaries


if __name__ == "__main__":
    # Тест модуля
    # Создаём тестовые сообщения
    base_time = datetime.now(ZoneInfo("UTC"))
    test_messages = [
        {
            "id": "1",
            "date_utc": base_time.isoformat(),
            "text": "Привет! Давайте обсудим TON",
            "from": {"username": "alice"},
            "language": "ru",
        },
        {
            "id": "2",
            "date_utc": (base_time + timedelta(minutes=5)).isoformat(),
            "text": "Да, нужно принять решение о бюджете",
            "from": {"username": "bob"},
            "language": "ru",
        },
    ]

    async def test():
        summaries = await summarize_chat_sessions(test_messages, "TestChat")
        print(f"Создано саммаризаций: {len(summaries)}")
        for summary in summaries:
            print(f"  {summary['session_id']}: {summary.get('context', 'N/A')[:100]}")

    asyncio.run(test())

