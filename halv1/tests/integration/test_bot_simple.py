"""
Простой интеграционный тест для проверки основных функций бота.

Этот тест проверяет базовую функциональность без сложной настройки.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
import pytest
import pytest_asyncio

from agent.core import AgentCore
from events.models import MessageReceived, ReplyReady
from services.event_bus import AsyncEventBus
from memory import MemoryServiceAdapter
from llm import create_llm_client
from planner import LLMTaskPlanner
from executor import create_executor
from internet import SearchClient


class SimpleBotTest:
    """Простой тест бота с минимальной настройкой."""

    def __init__(self):
        self.bus = AsyncEventBus()
        self.core = None
        self.agent_memory = None
        self.replies = []

    async def setup(self):
        """Минимальная настройка для тестирования."""
        # Настройка LLM клиента
        llm_client = create_llm_client("ollama", {"model": "gemma3n:e4b-it-q8_0"}, {})
        
        # Настройка памяти агента (в памяти)
        self.agent_memory = MemoryServiceAdapter(
            path=":memory:",
            embeddings_client=None,  # Отключаем embeddings для простоты
            short_term_limit=100,
            llm_client=llm_client,
        )

        # Настройка планировщика
        planner = LLMTaskPlanner(llm_client)

        # Настройка исполнителя
        executor = create_executor("docker", "venv")

        # Настройка поискового клиента
        search_client = SearchClient(llm=llm_client)

        # Создание ядра агента
        self.core = AgentCore(
            self.bus,
            planner,
            executor,
            search_client,
            self.agent_memory,
            None,  # code_generator
            registry=None
        )

        # Обработчик для сбора ответов
        async def collect_reply(event: ReplyReady) -> None:
            self.replies.append(event)
        
        self.bus.subscribe("reply_ready", collect_reply)

    async def send_message(self, text: str, chat_id: int = 12345) -> list:
        """Отправляет сообщение и возвращает ответы."""
        self.replies.clear()
        
        # Публикуем сообщение в event bus
        await self.bus.publish(
            "incoming",
            MessageReceived(chat_id=chat_id, message_id=1, text=text)
        )
        
        # Ждем обработки
        await self.bus.join()
        
        return self.replies.copy()

    async def cleanup(self):
        """Очистка."""
        if self.agent_memory:
            self.agent_memory.save()


@pytest.fixture
async def simple_bot():
    """Фикстура для простого теста бота."""
    bot = SimpleBotTest()
    await bot.setup()
    try:
        yield bot
    finally:
        await bot.cleanup()


class TestSimpleBot:
    """Простые тесты бота."""

    @pytest.mark.asyncio
    async def test_basic_message_processing(self, simple_bot):
        """Тест базовой обработки сообщений."""
        replies = await simple_bot.send_message("Привет!")
        
        print(f"Получено ответов: {len(replies)}")
        for i, reply in enumerate(replies):
            print(f"Ответ {i+1}: {reply.reply}")

    @pytest.mark.asyncio
    async def test_simple_question(self, simple_bot):
        """Тест простого вопроса."""
        replies = await simple_bot.send_message("Как дела?")
        
        print(f"Получено ответов: {len(replies)}")
        for i, reply in enumerate(replies):
            print(f"Ответ {i+1}: {reply.reply}")

    @pytest.mark.asyncio
    async def test_memory_usage(self, simple_bot):
        """Тест использования памяти."""
        # Отправляем информацию
        await simple_bot.send_message("Меня зовут Тест")
        
        # Проверяем, что информация сохранилась
        memory_content = simple_bot.agent_memory.recall()
        print(f"Содержимое памяти: {memory_content}")

    @pytest.mark.asyncio
    async def test_multiple_messages(self, simple_bot):
        """Тест нескольких сообщений подряд."""
        messages = [
            "Привет!",
            "Как дела?",
            "Что ты умеешь?",
            "Спасибо!"
        ]
        
        for msg in messages:
            replies = await simple_bot.send_message(msg)
            print(f"Сообщение: '{msg}' -> Ответов: {len(replies)}")
            if replies:
                print(f"  Ответ: {replies[0].reply}")


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "-s"])
