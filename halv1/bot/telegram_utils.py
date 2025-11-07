"""Утилиты для работы с Telegram API."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

try:
    from telegram.error import TelegramError, TimedOut
except Exception:  # pragma: no cover - optional dependency
    TelegramError = TimedOut = None  # type: ignore


def throttle_updates(interval_seconds: float = 1.0):
    """Декоратор для ограничения частоты обновлений сообщений.
    
    Args:
        interval_seconds: Минимальный интервал между обновлениями в секундах
    """
    last_update = {}
    
    def decorator(func):
        async def wrapper(msg, text: str, *args, **kwargs):
            msg_id = id(msg)
            now = datetime.now(UTC)
            
            if msg_id in last_update:
                time_since_last = (now - last_update[msg_id]).total_seconds()
                if time_since_last < interval_seconds:
                    return True  # Пропускаем обновление
            
            last_update[msg_id] = now
            return await func(msg, text, *args, **kwargs)
        return wrapper
    return decorator


@throttle_updates(interval_seconds=1.0)
async def safe_edit_message(msg, text: str, retries: int = 3, **kwargs) -> bool:
    """Безопасное редактирование сообщения с retry логикой.
    
    Args:
        msg: Объект сообщения для редактирования
        text: Новый текст сообщения
        retries: Количество попыток
        **kwargs: Дополнительные параметры для edit_text
        
    Returns:
        bool: True если редактирование успешно, False иначе
    """
    for attempt in range(retries):
        try:
            await msg.edit_text(text, **kwargs)
            return True
        except TimedOut:
            if attempt < retries - 1:
                wait_time = 2 * (attempt + 1)  # Экспоненциальная задержка
                logger.warning(f"Telegram API timeout on edit_text, retrying in {wait_time}s (attempt {attempt + 1}/{retries})")
                await asyncio.sleep(wait_time)
            else:
                logger.error("Telegram API timeout on edit_text after all retries")
                return False
        except TelegramError as e:
            logger.error(f"Telegram error on edit_text: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error on edit_text: {e}")
            return False
    
    return False
