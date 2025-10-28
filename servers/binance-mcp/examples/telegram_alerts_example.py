#!/usr/bin/env python3
"""Примеры использования системы алертов с Telegram уведомлениями."""

import asyncio
import json
from src.services import AlertService, TelegramService
from src.models import AlertConfig, TelegramNotification


async def main():
    """Основная функция с примерами системы алертов."""
    print("📱 Примеры системы алертов с Telegram уведомлениями\n")
    
    try:
        # Список символов для тестирования
        test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        print("⚠️  ВНИМАНИЕ: Это примеры для демонстрации!")
        print("⚠️  Telegram уведомления - заглушка (логируются в консоль)\n")
        
        # Пример 1: Тестовое уведомление в Telegram
        print("📱 Пример 1: Тестовое уведомление в Telegram")
        test_message = """🚀 *Тестовое уведомление*

📊 Система алертов работает корректно!

⏰ Время: 2025-01-16 15:30:00 UTC
💡 Это тестовое сообщение для проверки интеграции с Telegram"""
        
        notification = TelegramNotification(
            chat_id="123456789",  # Тестовый chat_id
            message=test_message,
            parse_mode="Markdown"
        )
        
        result = await TelegramService.send_notification(notification)
        print(f"Результат отправки: {'✅ Успешно' if result else '❌ Ошибка'}")
        print()
        
        # Пример 2: Алерты на просадку и прибыль
        print("🚨 Пример 2: Алерты на просадку и прибыль")
        alerts = [
            AlertConfig(
                alert_type="drawdown",
                threshold=3.0,  # Алерт при просадке > 3%
                notification_method="telegram"
            ),
            AlertConfig(
                alert_type="profit", 
                threshold=8.0,  # Алерт при прибыли > 8%
                notification_method="telegram"
            )
        ]
        
        alert_results = await AlertService.setup_portfolio_alerts(
            symbols=test_symbols,
            alerts=alerts,
            telegram_chat_id="123456789"
        )
        
        print(f"Результаты проверки алертов:")
        for result in alert_results:
            status = "🔔 Сработал" if result.triggered else "✅ Норма"
            print(f"  {result.symbol} - {result.alert_type}: {status}")
            print(f"    Текущее значение: {result.current_value:.2f}")
            print(f"    Порог: {result.threshold}")
            print(f"    Уведомление отправлено: {'✅' if result.notification_sent else '❌'}")
            if result.message:
                print(f"    Сообщение: {result.message[:100]}...")
        print()
        
        # Пример 3: Алерты на всплеск объема
        print("📊 Пример 3: Алерты на всплеск объема")
        volume_alerts = [
            AlertConfig(
                alert_type="volume_spike",
                threshold=2.5,  # Алерт при увеличении объема > 2.5x
                notification_method="telegram"
            )
        ]
        
        volume_results = await AlertService.setup_portfolio_alerts(
            symbols=test_symbols,
            alerts=volume_alerts,
            telegram_chat_id="123456789"
        )
        
        print(f"Результаты проверки объема:")
        for result in volume_results:
            status = "🔥 Всплеск" if result.triggered else "📊 Норма"
            print(f"  {result.symbol}: {status}")
            print(f"    Объем: {result.current_value:.1f}x")
            print(f"    Порог: {result.threshold:.1f}x")
        print()
        
        # Пример 4: Алерты на ценовые уровни
        print("💲 Пример 4: Алерты на ценовые уровни")
        price_alerts = [
            AlertConfig(
                alert_type="price_level",
                threshold=50000.0,  # Алерт при достижении $50,000
                notification_method="telegram"
            )
        ]
        
        price_results = await AlertService.setup_portfolio_alerts(
            symbols=["BTCUSDT"],  # Только Bitcoin для примера
            alerts=price_alerts,
            telegram_chat_id="123456789"
        )
        
        print(f"Результаты проверки ценовых уровней:")
        for result in price_results:
            status = "📈 Достигнут" if result.triggered else "📊 Не достигнут"
            print(f"  {result.symbol}: {status}")
            print(f"    Цена: ${result.current_value:.2f}")
            print(f"    Порог: ${result.threshold:.2f}")
        print()
        
        # Пример 5: Алерты на экстремальные значения RSI
        print("📊 Пример 5: Алерты на экстремальные значения RSI")
        rsi_alerts = [
            AlertConfig(
                alert_type="rsi_extreme",
                threshold=70.0,  # Алерт при RSI > 70 или < 30
                notification_method="telegram"
            )
        ]
        
        rsi_results = await AlertService.setup_portfolio_alerts(
            symbols=test_symbols,
            alerts=rsi_alerts,
            telegram_chat_id="123456789"
        )
        
        print(f"Результаты проверки RSI:")
        for result in rsi_results:
            status = "🔴 Экстремум" if result.triggered else "📊 Норма"
            print(f"  {result.symbol}: {status}")
            print(f"    RSI: {result.current_value:.1f}")
            print(f"    Порог: {result.threshold:.1f}")
        print()
        
        # Пример 6: Комплексная система алертов
        print("🛡️ Пример 6: Комплексная система алертов")
        comprehensive_alerts = [
            AlertConfig("drawdown", 2.0, "telegram"),      # Раннее предупреждение
            AlertConfig("drawdown", 5.0, "telegram"),      # Критический уровень
            AlertConfig("profit", 5.0, "telegram"),       # Фиксация прибыли
            AlertConfig("profit", 10.0, "telegram"),      # Высокая прибыль
            AlertConfig("volume_spike", 3.0, "telegram"),  # Аномальная активность
            AlertConfig("rsi_extreme", 75.0, "telegram")   # Технические экстремумы
        ]
        
        comprehensive_results = await AlertService.setup_portfolio_alerts(
            symbols=test_symbols,
            alerts=comprehensive_alerts,
            telegram_chat_id="123456789"
        )
        
        print(f"Результаты комплексной системы:")
        triggered_count = sum(1 for r in comprehensive_results if r.triggered)
        total_count = len(comprehensive_results)
        print(f"  Всего проверок: {total_count}")
        print(f"  Сработавших алертов: {triggered_count}")
        print(f"  Процент срабатывания: {triggered_count/total_count*100:.1f}%")
        print()
        
        print("✅ Все примеры системы алертов выполнены успешно!")
        print("💡 Система готова для интеграции с реальным Telegram ботом")
        print("🔧 Для настройки добавьте в .env:")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token")
        print("   TELEGRAM_CHAT_ID=your_chat_id")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
