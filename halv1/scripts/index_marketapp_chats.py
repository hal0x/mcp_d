#!/usr/bin/env python3
"""
Скрипт для индексации чатов Marketapp с повышенными лимитами.
Скачивает все сообщения без ограничений, но с защитой от дубликатов.
"""

import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Добавляем корневую директорию проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from index.telethon_indexer import TelethonIndexer
from index.raw_storage import RawStorage
from utils.message_extractor import extract_message_data

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def index_marketapp_chats():
    """Индексирует чаты Marketapp с повышенными лимитами."""
    
    # Загружаем конфигурацию
    from main import load_config
    config = load_config()
    
    # Настройки Telethon
    tele_cfg = config.get("telethon", {})
    api_id = tele_cfg.get("api_id")
    api_hash = tele_cfg.get("api_hash")
    
    if not api_id or not api_hash:
        logger.error("❌ Не найдены TELETHON_API_ID или TELETHON_API_HASH")
        return False
    
    # Создаем индексатор
    tele_indexer = TelethonIndexer(
        api_id,
        api_hash,
        tele_cfg.get("session", "user"),
    )
    
    # Создаем хранилище
    raw_storage = RawStorage(config["paths"]["raw"])
    
    try:
        # Подключаемся к Telegram
        logger.info("🔌 Подключаемся к Telegram...")
        await tele_indexer.ensure_connected()
        await tele_indexer.assert_authorized()
        logger.info("✅ Подключение к Telegram успешно")
        
        # Запускаем индексацию
        logger.info("📚 Начинаем индексацию чатов Marketapp...")
        logger.info("💡 Система автоматически определит, нужно ли полная индексация или только новые сообщения")
        messages_count = 0
        start_time = datetime.now(UTC)
        last_log_time = start_time
        
        async for message in tele_indexer.index_once():
            # Проверяем, является ли это маркером завершения чата
            if isinstance(message, dict) and message.get("type") == "chat_completed":
                chat_name = message["chat_name"]
                new_messages = message["new_messages"]
                
                # Удаляем старые сообщения, если добавили новые (только для обычных чатов)
                if new_messages > 0:
                    # Для Marketapp чатов не удаляем старые сообщения
                    if "marketapp" in chat_name.lower():
                        logger.info(f"📊 Marketapp чат '{chat_name}': сохранены все {new_messages} новых сообщений (без удаления старых)")
                    else:
                        removed_count = raw_storage.trim_old_messages(chat_name, max_messages=50000)
                        if removed_count > 0:
                            logger.info(f"🗑️  Удалено {removed_count} старых сообщений из чата '{chat_name}'")
                continue
            
            # Извлекаем данные сообщения
            msg_data = extract_message_data(message)
            
            # Сохраняем сообщение в raw storage
            raw_storage.save(msg_data["chat"], msg_data)
            
            messages_count += 1
            
            # Логируем прогресс каждые 100 сообщений или каждые 30 секунд
            current_time = datetime.now(UTC)
            if messages_count % 100 == 0 or (current_time - last_log_time).total_seconds() > 30:
                elapsed = (current_time - start_time).total_seconds()
                speed = messages_count / elapsed if elapsed > 0 else 0
                msg_date = getattr(message, "date", current_time)
                logger.info(f"📚 {messages_count} сообщений | Чат: {msg_data['chat']} | Дата: {msg_date} | Скорость: {speed:.1f}/с")
                last_log_time = current_time
        
        # Сохраняем состояние индексации
        index_state_path = Path(config["paths"]["index"]).parent / "last_indexed.txt"
        index_state_path.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
        
        logger.info(f"✅ Индексация Marketapp чатов завершена: обработано {messages_count} сообщений")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка индексации: {e}")
        return False
    
    finally:
        # Отключаемся от Telegram
        try:
            if tele_indexer.client.is_connected():
                await tele_indexer.client.disconnect()
                logger.info("🔌 Отключились от Telegram")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка отключения от Telegram: {e}")


def main():
    """Основная функция."""
    logger.info("🚀 Запуск индексации чатов Marketapp с повышенными лимитами")
    
    # Проверяем, что чаты добавлены в белый список
    allowed_chats_file = Path("allowed_chats.txt")
    if not allowed_chats_file.exists():
        logger.error("❌ Файл allowed_chats.txt не найден")
        return 1
    
    with open(allowed_chats_file, "r", encoding="utf-8") as f:
        allowed_chats = [line.strip() for line in f if line.strip()]
    
    marketapp_chats = [chat for chat in allowed_chats if "marketapp" in chat.lower()]
    
    if not marketapp_chats:
        logger.error("❌ Чаты Marketapp не найдены в allowed_chats.txt")
        logger.info("💡 Добавьте чаты с 'marketapp' в названии в файл allowed_chats.txt")
        return 1
    
    logger.info(f"📋 Найдены чаты Marketapp: {marketapp_chats}")
    
    # Запускаем индексацию
    success = asyncio.run(index_marketapp_chats())
    
    if success:
        logger.info("🎉 Индексация завершена успешно!")
        return 0
    else:
        logger.error("💥 Индексация завершилась с ошибками")
        return 1


if __name__ == "__main__":
    exit(main())
