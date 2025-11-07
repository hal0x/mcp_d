#!/usr/bin/env python3
"""
Реальный тест для проверки работы бота с исправленным LLM.

Этот скрипт тестирует систему с правильной настройкой LLM клиента.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
import sys
import os

# Добавляем корневую директорию halv1 в путь
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.core import AgentCore
from events.models import MessageReceived, ReplyReady
from services.event_bus import AsyncEventBus
from memory import MemoryServiceAdapter
from llm import create_llm_client
from planner import LLMTaskPlanner
from executor import create_executor
from internet import SearchClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)


class RealBotTest:
    """Реальный тест бота с исправленным LLM."""

    def __init__(self):
        self.bus = AsyncEventBus()
        self.core = None
        self.agent_memory = None
        self.replies = []

    async def setup(self):
        """Настройка для тестирования."""
        logger.info("🔧 Настраиваем реальную тестовую среду...")
        
        # Настройка LLM клиента с правильными параметрами
        llm_config = {
            "provider": "ollama",
            "model": "gemma3n:e4b-it-q8_0",
            "host": "localhost",
            "port": 11434
        }
        
        llm_client = create_llm_client("ollama", llm_config, {})
        
        # Настройка памяти агента
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
            logger.info(f"📨 Получен ответ: {event.reply[:100]}...")
        
        self.bus.subscribe("reply_ready", collect_reply)
        
        logger.info("✅ Реальная тестовая среда настроена")

    async def send_message(self, text: str, chat_id: int = 12345) -> list:
        """Отправляет сообщение и возвращает ответы."""
        logger.info(f"📤 Отправляем сообщение: '{text}'")
        self.replies.clear()
        
        # Публикуем сообщение в event bus
        await self.bus.publish(
            "incoming",
            MessageReceived(chat_id=chat_id, message_id=1, text=text)
        )
        
        # Ждем обработки
        await self.bus.join()
        
        logger.info(f"📥 Получено ответов: {len(self.replies)}")
        return self.replies.copy()

    async def test_simple_queries(self):
        """Тестирует простые запросы."""
        test_queries = [
            "Привет!",
            "Как дела?",
            "Что ты умеешь?",
            "Расскажи о себе"
        ]
        
        for query in test_queries:
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 Тестируем запрос: '{query}'")
            
            replies = await self.send_message(query)
            
            if replies:
                for i, reply in enumerate(replies):
                    logger.info(f"📝 Ответ {i+1}: {reply.reply}")
            else:
                logger.warning("⚠️ Ответ не получен")
            
            # Небольшая пауза между запросами
            await asyncio.sleep(1)

    async def test_memory_queries(self):
        """Тестирует запросы с использованием памяти."""
        logger.info(f"\n{'='*50}")
        logger.info("🧪 Тестируем запросы с памятью")
        
        # Добавляем информацию в память
        await self.agent_memory.remember("Меня зовут Тест-Бот")
        await self.agent_memory.remember("Я умею отвечать на вопросы")
        await self.agent_memory.remember("Сегодня хорошая погода")
        
        # Проверяем, что информация сохранилась
        memory_content = self.agent_memory.recall()
        logger.info(f"📚 Содержимое памяти: {memory_content}")
        
        # Тестируем запросы, которые должны использовать память
        memory_queries = [
            "Как меня зовут?",
            "Что ты умеешь?",
            "Какая сегодня погода?"
        ]
        
        for query in memory_queries:
            logger.info(f"\n📤 Тестируем запрос с памятью: '{query}'")
            replies = await self.send_message(query)
            
            if replies:
                for i, reply in enumerate(replies):
                    logger.info(f"📝 Ответ {i+1}: {reply.reply}")
            else:
                logger.warning("⚠️ Ответ не получен")
            
            await asyncio.sleep(1)

    async def cleanup(self):
        """Очистка."""
        if self.agent_memory:
            self.agent_memory.save()
        logger.info("🧹 Очистка завершена")


async def main():
    """Главная функция."""
    logger.info("🚀 Запуск реального теста бота")
    
    # Проверяем, что мы в правильной директории
    if not Path("main.py").exists():
        logger.error("❌ Запустите скрипт из корневой директории проекта")
        return
    
    # Проверяем Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            logger.warning("⚠️ Ollama может быть недоступен")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось проверить Ollama: {e}")
    
    bot = RealBotTest()
    
    try:
        await bot.setup()
        await bot.test_simple_queries()
        await bot.test_memory_queries()
        
        logger.info("\n🎉 Реальный тест завершен успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время теста: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await bot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
