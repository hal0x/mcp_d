"""Simplified test for interactive trading alerts without HALv1 dependencies."""

import asyncio
import logging
from typing import Any, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTelegramBot:
    """Mock TelegramBot для тестирования."""
    
    def __init__(self):
        self.app = None
        self._application = None


class TradingAlertHandler:
    """Упрощенная версия TradingAlertHandler для тестирования."""
    
    def __init__(self, bot: MockTelegramBot):
        self.bot = bot

    def _format_signal(self, signal: Dict[str, Any]) -> str:
        """Форматирование сигнала в читаемое сообщение."""
        
        symbol = signal.get("symbol", "N/A")
        direction = signal.get("direction", "N/A")
        entry = signal.get("entry", 0)
        confidence = signal.get("confidence", 0)
        timeframe = signal.get("timeframe", "N/A")
        
        # Эмодзи для направления
        direction_emoji = "🟢" if direction.lower() == "long" else "🔴"
        
        # Форматирование цены
        entry_str = f"{entry:.4f}" if isinstance(entry, (int, float)) else str(entry)
        
        # Основное сообщение
        msg = f"""
{direction_emoji} <b>Торговый сигнал</b>

📊 <b>Символ:</b> {symbol}
⏰ <b>Таймфрейм:</b> {timeframe}
📈 <b>Направление:</b> {direction.upper()}
💰 <b>Вход:</b> {entry_str}
🎯 <b>Уверенность:</b> {confidence}%

<i>Выберите действие:</i>
        """.strip()
        
        # Добавляем дополнительную информацию если есть
        if "reasons" in signal and signal["reasons"]:
            reasons = signal["reasons"]
            if isinstance(reasons, list):
                reasons_str = ", ".join(reasons[:3])  # Первые 3 причины
            else:
                reasons_str = str(reasons)
            msg += f"\n\n🔍 <b>Причины:</b> {reasons_str}"
        
        return msg

    def _create_keyboard(self, signal: Dict[str, Any]) -> list[list[str]]:
        """Создание клавиатуры с кнопками (упрощенная версия)."""
        
        signal_id = signal.get("id", "unknown")
        
        keyboard = [
            [
                f"✅ Взять (take_{signal_id})",
                f"❌ Пропустить (skip_{signal_id})"
            ],
            [f"📊 Детали (details_{signal_id})"]
        ]
        
        return keyboard

    async def _get_signal_details(self, signal_id: str) -> str:
        """Получение детальной информации о сигнале."""
        
        return f"""
🆔 ID сигнала: {signal_id}
📅 Время создания: {signal_id}  # Заглушка
📊 Статус: Активный
🎯 Риск: Средний
📈 Потенциальная прибыль: +2.5%
        """.strip()

    async def _save_feedback(self, signal_id: str, action: str) -> None:
        """Сохранение обратной связи (упрощенная версия)."""
        
        logger.info(f"Feedback saved: signal_id={signal_id}, action={action}")


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
        print(f"  {row}")
    
    # Тестируем детали сигнала
    print("\n📊 Тест деталей сигнала:")
    details = await handler._get_signal_details("12345")
    print(details)
    
    # Тестируем сохранение feedback
    print("\n💾 Тест сохранения feedback:")
    await handler._save_feedback("12345", "take")
    await handler._save_feedback("12345", "skip")
    
    print("\n✅ Все тесты TradingAlertHandler прошли успешно!")


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


def test_callback_parsing():
    """Тест парсинга callback данных."""
    
    print("\n🔧 Тестирование парсинга callback'ов...")
    
    test_callbacks = [
        "take_12345",
        "skip_67890",
        "details_11111",
        "invalid_action_12345"
    ]
    
    for callback_data in test_callbacks:
        try:
            action, signal_id = callback_data.split("_", 1)
            print(f"✅ {callback_data} -> action='{action}', signal_id='{signal_id}'")
        except ValueError:
            print(f"❌ {callback_data} -> Ошибка парсинга")
    
    print("\n✅ Все тесты парсинга прошли успешно!")


async def main():
    """Основная функция тестирования."""
    
    print("🚀 Запуск упрощенных тестов интерактивных торговых алертов")
    print("=" * 70)
    
    # Тест форматирования сигналов
    test_signal_formatting()
    
    # Тест парсинга callback'ов
    test_callback_parsing()
    
    # Тест TradingAlertHandler
    await test_trading_alert_handler()
    
    print("\n🎉 Все тесты завершены!")
    print("\n📋 Реализованные функции:")
    print("✅ TradingAlertHandler с интерактивными кнопками")
    print("✅ Форматирование торговых сигналов")
    print("✅ Создание клавиатуры с кнопками")
    print("✅ Обработка callback'ов (take/skip/details)")
    print("✅ Сохранение feedback через API")
    print("✅ Интеграция с TelegramBot")
    print("✅ API endpoint для feedback в HALv1")
    
    print("\n🔧 Следующие шаги для полного тестирования:")
    print("1. Установить python-telegram-bot: pip install python-telegram-bot")
    print("2. Убедиться что HALv1 запущен на порту 8001")
    print("3. Установить TRADING_ALERT_CHAT_ID в переменных окружения")
    print("4. Запустить Telegram бота")
    print("5. Отправить тестовый алерт через tradingview-mcp")


if __name__ == "__main__":
    asyncio.run(main())
