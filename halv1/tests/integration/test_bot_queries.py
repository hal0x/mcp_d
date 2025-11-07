"""
Интеграционный тест для проверки разных типов запросов к боту.

Этот тест проверяет:
1. Простые запросы с контекстом памяти
2. Сложные запросы (планирование и выполнение)
3. Команды бота
4. Обработку ошибок
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import pytest
import pytest_asyncio
import yaml

from agent.core import AgentCore
from events.models import MessageReceived, ReplyReady
from executor import create_executor
from index.vector_index import VectorIndex
from internet import SearchClient
from llm import create_llm_client
from memory import MemoryServiceAdapter
from planner import LLMTaskPlanner
from retriever.retriever import Retriever
from services.event_bus import AsyncEventBus
from index.cluster_manager import ClusterManager
from index.insight_store import InsightStore
from index.theme_store import ThemeStore
from llm.embeddings_client import AsyncEmbeddingsClient


class BotTestEnvironment:
    """Класс для настройки тестовой среды бота."""

    def __init__(self, tmp_dir: Path):
        self.tmp_dir = tmp_dir
        self.bus: AsyncEventBus = None
        self.core: AgentCore = None
        self.agent_memory: MemoryServiceAdapter = None
        self.vector_index: VectorIndex = None
        self.embeddings_client: AsyncEmbeddingsClient = None
        self.replies: List[ReplyReady] = []

    async def setup(self):
        """Настройка тестовой среды."""
        # Создаем временную конфигурацию
        config = {
            "llm": {"provider": "ollama", "model": "gemma3n:e4b-it-q8_0"},
            "embeddings": {"model": "dengcao/Qwen3-Embedding-4B:Q5_K_M"},
            "paths": {
                "memory_db": str(self.tmp_dir / "memory.db"),
                "index": str(self.tmp_dir / "index"),
                "raw": str(self.tmp_dir / "raw")
            }
        }

        # Создаем директории
        (self.tmp_dir / "index").mkdir(parents=True)
        (self.tmp_dir / "raw").mkdir(parents=True)

        # Настройка Event Bus
        self.bus = AsyncEventBus()
        
        # Обработчик для сбора ответов
        async def collect_reply(event: ReplyReady) -> None:
            self.replies.append(event)
        
        self.bus.subscribe("reply_ready", collect_reply)

        # Настройка LLM клиента
        llm_client = create_llm_client("ollama", config["llm"], {})
        
        # Настройка embeddings клиента
        self.embeddings_client = AsyncEmbeddingsClient(
            model=config["embeddings"]["model"],
            host="127.0.0.1",
            port=11434,
            provider="ollama"
        )

        # Настройка памяти агента
        self.agent_memory = MemoryServiceAdapter(
            path=":memory:",
            embeddings_client=self.embeddings_client,
            short_term_limit=100,
            llm_client=llm_client,
        )

        # Настройка векторного индекса
        self.vector_index = VectorIndex.themed(
            str(self.tmp_dir / "index"),
            "default",
            model_name=config["embeddings"]["model"],
            host="127.0.0.1",
            port=11434
        )

        # Настройка кластерного менеджера
        cluster_manager = ClusterManager()
        insight_store = InsightStore(str(self.tmp_dir / "insights.json"))
        cluster_manager.load(insight_store)

        # Настройка поисковика
        retriever = Retriever(
            self.vector_index,
            cluster_manager,
            get_active_theme=lambda: "default"
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
            registry=None,
            retriever=retriever
        )

        # Добавляем тестовые данные в память
        await self._add_test_data()

    async def _add_test_data(self):
        """Добавляем тестовые данные в память."""
        test_messages = [
            "Сегодня была отличная погода, я гулял в парке",
            "Вчера я читал книгу о машинном обучении",
            "Завтра у меня важная встреча с клиентом",
            "Я изучаю Python и создаю AI-ассистента",
            "Мой любимый цвет - синий"
        ]
        
        for i, msg in enumerate(test_messages):
            await self.vector_index.add(
                f"test_msg_{i}",
                msg,
                {"chat": "test", "date": "2024-01-01", "theme": "default"}
            )

    async def send_message(self, text: str, chat_id: int = 12345) -> List[ReplyReady]:
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
        """Очистка тестовой среды."""
        if self.vector_index:
            await self.vector_index.close()
        if self.embeddings_client:
            await self.embeddings_client.close()


@pytest.fixture
async def bot_env():
    """Фикстура для тестовой среды бота."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = BotTestEnvironment(Path(tmp_dir))
        await env.setup()
        try:
            yield env
        finally:
            await env.cleanup()


class TestBotQueries:
    """Тесты для разных типов запросов к боту."""

    @pytest.mark.asyncio
    async def test_simple_query_with_memory(self, bot_env):
        """Тест простого запроса с использованием контекста памяти."""
        # Запрос, который должен найти релевантный контекст
        replies = await bot_env.send_message("как дела?")
        
        # Проверяем, что получили ответ
        assert len(replies) > 0, "Должен быть получен ответ"
        
        reply = replies[0]
        assert reply.reply is not None
        assert len(reply.reply) > 0
        print(f"Ответ на простой запрос: {reply.reply}")

    @pytest.mark.asyncio
    async def test_memory_context_query(self, bot_env):
        """Тест запроса, который должен использовать контекст из памяти."""
        # Запрос, который должен найти информацию о погоде
        replies = await bot_env.send_message("что ты знаешь о погоде?")
        
        assert len(replies) > 0, "Должен быть получен ответ"
        
        reply = replies[0]
        assert reply.reply is not None
        print(f"Ответ с контекстом памяти: {reply.reply}")

    @pytest.mark.asyncio
    async def test_complex_query(self, bot_env):
        """Тест сложного запроса, требующего планирования."""
        # Запрос, который должен быть классифицирован как сложный
        replies = await bot_env.send_message("сделай сводку событий за сегодня")
        
        # Для сложных запросов ответ может прийти позже или через другой механизм
        # Пока просто проверяем, что нет ошибок
        print(f"Количество ответов на сложный запрос: {len(replies)}")

    @pytest.mark.asyncio
    async def test_command_query(self, bot_env):
        """Тест команд бота."""
        # Тестируем команду /help
        replies = await bot_env.send_message("/help")
        
        # Команды обрабатываются отдельно, могут не генерировать reply_ready
        print(f"Количество ответов на команду: {len(replies)}")

    @pytest.mark.asyncio
    async def test_empty_query(self, bot_env):
        """Тест пустого запроса."""
        replies = await bot_env.send_message("")
        
        # Пустой запрос может не генерировать ответ
        print(f"Количество ответов на пустой запрос: {len(replies)}")

    @pytest.mark.asyncio
    async def test_memory_persistence(self, bot_env):
        """Тест сохранения информации в память."""
        # Отправляем сообщение с новой информацией
        replies1 = await bot_env.send_message("Меня зовут Алексей")
        
        # Отправляем запрос, который должен использовать эту информацию
        replies2 = await bot_env.send_message("как меня зовут?")
        
        print(f"Ответ 1: {replies1[0].reply if replies1 else 'Нет ответа'}")
        print(f"Ответ 2: {replies2[0].reply if replies2 else 'Нет ответа'}")

    @pytest.mark.asyncio
    async def test_multiple_queries(self, bot_env):
        """Тест нескольких запросов подряд."""
        queries = [
            "Привет!",
            "Как дела?",
            "Что ты умеешь?",
            "Расскажи о погоде"
        ]
        
        all_replies = []
        for query in queries:
            replies = await bot_env.send_message(query)
            all_replies.extend(replies)
            print(f"Запрос: '{query}' -> Ответов: {len(replies)}")
        
        print(f"Всего ответов: {len(all_replies)}")


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "-s"])
