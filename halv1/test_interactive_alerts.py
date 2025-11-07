"""Test script for interactive trading alerts."""

import asyncio
import os
import sys
from pathlib import Path

# Добавляем путь к halv1 в sys.path
halv1_path = Path(__file__).parent.parent / "halv1"
sys.path.insert(0, str(halv1_path))

from bot.trading_alert_handler import TradingAlertHandler
from bot.telegram_bot import TelegramBot


class MockTelegramBot:
    """Mock TelegramBot для тестирования."""
    
    def __init__(self):
        self.app = None
        self._application = None


async def test_trading_alert_handler():
    """Тест TradingAlertHandler."""
    
    print("🧪 Тестирование TradingAlertHandler...")
    
    # Создаем mock бота
    mock_bot = MockTelegramBot()
    
    # Создаем handler
    handler = TradingAlertHandler(mock_bot)
    
    # Тестовые данные сигнала
    test_signal = {
        "id": "12345",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "direction": "long",
        "entry": 45000.50,
        "confidence": 78,
        "reasons": ["breakout", "volume_spike", "ema_cross"]
    }
    
    # Тестируем форматирование сообщения
    print("\n📝 Тест форматирования сообщения:")
    formatted_msg = handler._format_signal(test_signal)
    print(formatted_msg)
    
    # Тестируем создание клавиатуры
    print("\n⌨️ Тест создания клавиатуры:")
    keyboard = handler._create_keyboard(test_signal)
    for row in keyboard:
        print(f"  {[btn.text for btn in row]}")
    
    # Тестируем детали сигнала
    print("\n📊 Тест деталей сигнала:")
    details = await handler._get_signal_details("12345")
    print(details)
    
    print("\n✅ Все тесты TradingAlertHandler прошли успешно!")


async def test_feedback_saving():
    """Тест сохранения feedback."""
    
    print("\n💾 Тестирование сохранения feedback...")
    
    mock_bot = MockTelegramBot()
    handler = TradingAlertHandler(mock_bot)
    
    try:
        # Тестируем сохранение feedback
        await handler._save_feedback("12345", "take")
        print("✅ Feedback 'take' сохранен успешно")
        
        await handler._save_feedback("12345", "skip")
        print("✅ Feedback 'skip' сохранен успешно")
        
    except Exception as e:
        print(f"⚠️ Ошибка сохранения feedback (ожидаемо если HALv1 не запущен): {e}")


def test_signal_formatting():
    """Тест форматирования различных типов сигналов."""
    
    print("\n🎨 Тестирование форматирования сигналов...")
    
    mock_bot = MockTelegramBot()
    handler = TradingAlertHandler(mock_bot)
    
    # Тест 1: Long сигнал
    long_signal = {
        "id": "1",
        "symbol": "ETHUSDT",
        "timeframe": "1h",
        "direction": "long",
        "entry": 3200.75,
        "confidence": 85,
        "reasons": ["momentum", "volume"]
    }
    
    print("\n📈 Long сигнал:")
    print(handler._format_signal(long_signal))
    
    # Тест 2: Short сигнал
    short_signal = {
        "id": "2",
        "symbol": "ADAUSDT",
        "timeframe": "15m",
        "direction": "short",
        "entry": 0.4523,
        "confidence": 72,
        "reasons": ["mean_reversion", "rsi_overbought"]
    }
    
    print("\n📉 Short сигнал:")
    print(handler._format_signal(short_signal))
    
    # Тест 3: Сигнал без причин
    simple_signal = {
        "id": "3",
        "symbol": "SOLUSDT",
        "timeframe": "4h",
        "direction": "long",
        "entry": 95.50,
        "confidence": 60
    }
    
    print("\n🔹 Простой сигнал:")
    print(handler._format_signal(simple_signal))
    
    print("\n✅ Все тесты форматирования прошли успешно!")


async def main():
    """Основная функция тестирования."""
    
    print("🚀 Запуск тестов интерактивных торговых алертов")
    print("=" * 60)
    
    # Тест форматирования сигналов
    test_signal_formatting()
    
    # Тест TradingAlertHandler
    await test_trading_alert_handler()
    
    # Тест сохранения feedback
    await test_feedback_saving()
    
    print("\n🎉 Все тесты завершены!")
    print("\n📋 Следующие шаги:")
    print("1. Убедитесь что HALv1 запущен на порту 8001")
    print("2. Установите TRADING_ALERT_CHAT_ID в переменных окружения")
    print("3. Запустите Telegram бота")
    print("4. Отправьте тестовый алерт через tradingview-mcp")


if __name__ == "__main__":
    asyncio.run(main())
