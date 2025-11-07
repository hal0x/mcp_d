#!/usr/bin/env python3
"""
Отладочный тест для проверки работы бота.

Этот скрипт тестирует систему без LLM, чтобы проверить базовую функциональность.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
import sys
import os

# Добавляем корневую директорию halv1 в путь
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from events.models import MessageReceived, ReplyReady
from services.event_bus import AsyncEventBus

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)


class DebugBotTest:
    """Отладочный тест бота без LLM."""

    def __init__(self):
        self.bus = AsyncEventBus()
        self.replies = []

    async def setup(self):
        """Настройка для тестирования."""
        logger.info("🔧 Настраиваем отладочную среду...")
        
        # Обработчик для сбора ответов
        async def collect_reply(event: ReplyReady) -> None:
            self.replies.append(event)
            logger.info(f"📨 Получен ответ: {event.reply[:100]}...")
        
        self.bus.subscribe("reply_ready", collect_reply)
        
        # Обработчик для входящих сообщений
        async def handle_message(event: MessageReceived) -> None:
            logger.info(f"📥 Получено сообщение: '{event.text}'")
            
            # Простой ответ без LLM
            reply_text = f"Получено сообщение: '{event.text}'"
            
            # Отправляем ответ
            await self.bus.publish("reply_ready", ReplyReady(
                chat_id=event.chat_id,
                message_id=event.message_id,
                reply=reply_text
            ))
        
        self.bus.subscribe("incoming", handle_message)
        
        logger.info("✅ Отладочная среда настроена")

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

    async def test_event_bus(self):
        """Тестирует работу event bus."""
        test_queries = [
            "Привет!",
            "Как дела?",
            "Что ты умеешь?",
            "Тест 1",
            "Тест 2"
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
            await asyncio.sleep(0.5)

    async def test_multiple_messages(self):
        """Тестирует несколько сообщений подряд."""
        logger.info(f"\n{'='*50}")
        logger.info("🧪 Тестируем несколько сообщений подряд")
        
        messages = ["Сообщение 1", "Сообщение 2", "Сообщение 3"]
        
        for i, msg in enumerate(messages):
            logger.info(f"📤 Отправляем сообщение {i+1}: '{msg}'")
            replies = await self.send_message(msg)
            
            if replies:
                logger.info(f"📝 Получен ответ: {replies[0].reply}")
            else:
                logger.warning("⚠️ Ответ не получен")
            
            await asyncio.sleep(0.2)

    async def cleanup(self):
        """Очистка."""
        logger.info("🧹 Очистка завершена")


async def main():
    """Главная функция."""
    logger.info("🚀 Запуск отладочного теста бота")
    
    # Проверяем, что мы в правильной директории
    if not Path("main.py").exists():
        logger.error("❌ Запустите скрипт из корневой директории проекта")
        return
    
    bot = DebugBotTest()
    
    try:
        await bot.setup()
        await bot.test_event_bus()
        await bot.test_multiple_messages()
        
        logger.info("\n🎉 Отладочный тест завершен успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время теста: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await bot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
