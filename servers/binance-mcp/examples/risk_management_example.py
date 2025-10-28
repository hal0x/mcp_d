#!/usr/bin/env python3
"""Примеры использования функций управления рисками Binance MCP сервера."""

import asyncio
import json
from src.services import RiskManagementService, AlertService
from src.models import SafetyRule, StopLossConfig, AlertConfig, RiskManagementRule


async def main():
    """Основная функция с примерами управления рисками."""
    print("🛡️ Примеры управления рисками Binance MCP сервера\n")
    
    try:
        # Список символов для тестирования
        test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        print("⚠️  ВНИМАНИЕ: Это примеры для демонстрации!")
        print("⚠️  Не используйте реальные деньги без тестирования!\n")
        
        # Пример 1: Проверка безопасности позиций
        print("🔍 Пример 1: Проверка безопасности позиций")
        safety_rules = SafetyRule(
            max_rsi_short=30.0,  # Не шортить при RSI < 30
            min_rsi_long=75.0,   # Не лонгить при RSI > 75
            min_adx=18.0,        # Минимальный ADX
            max_drawdown=5.0     # Максимальная просадка 5%
        )
        
        safety_results = await RiskManagementService.portfolio_safety_check(
            symbols=test_symbols,
            safety_rules=safety_rules,
            auto_close_unsafe=False  # Не закрываем автоматически в примере
        )
        
        print(f"Результаты проверки безопасности:")
        for result in safety_results:
            status = "✅ Безопасно" if result.is_safe else "❌ Небезопасно"
            print(f"  {result.symbol}: {status}")
            if result.violations:
                print(f"    Нарушения: {', '.join(result.violations)}")
            print(f"    Рекомендация: {result.recommendation}")
            if result.rsi:
                print(f"    RSI: {result.rsi:.1f}")
            if result.adx:
                print(f"    ADX: {result.adx:.1f}")
            if result.drawdown:
                print(f"    Просадка: {result.drawdown:.1f}%")
        print()
        
        # Пример 2: Управление стоп-лоссами
        print("🛑 Пример 2: Автоматическое управление стоп-лоссами")
        stop_loss_config = StopLossConfig(
            stop_loss_type="trailing",  # Трейлинг стоп-лосс
            trail_percentage=2.0,       # 2% трейлинг
            update_frequency="1h",      # Обновление каждый час
            max_loss_percent=5.0        # Максимальная потеря 5%
        )
        
        stop_loss_results = await RiskManagementService.manage_stop_losses(
            symbols=test_symbols,
            stop_loss_config=stop_loss_config
        )
        
        print(f"Результаты управления стоп-лоссами:")
        for result in stop_loss_results:
            print(f"  {result.symbol}: {result.action}")
            print(f"    Причина: {result.reason}")
            if result.stop_price:
                print(f"    Стоп-цена: {result.stop_price:.4f}")
        print()
        
        # Пример 3: Настройка алертов
        print("🚨 Пример 3: Настройка алертов портфеля")
        alerts = [
            AlertConfig(
                alert_type="drawdown",
                threshold=5.0,  # Алерт при просадке > 5%
                action="notify"
            ),
            AlertConfig(
                alert_type="profit",
                threshold=10.0,  # Алерт при прибыли > 10%
                action="take_profit"
            ),
            AlertConfig(
                alert_type="volume_spike",
                threshold=3.0,  # Алерт при всплеске объема > 3x
                action="analyze"
            )
        ]
        
        alert_results = await AlertService.setup_portfolio_alerts(
            symbols=test_symbols,
            alerts=alerts
        )
        
        print(f"Результаты проверки алертов:")
        for result in alert_results:
            status = "🔔 Сработал" if result.triggered else "✅ Норма"
            print(f"  {result.symbol} - {result.alert_type}: {status}")
            print(f"    Текущее значение: {result.current_value:.2f}")
            print(f"    Порог: {result.threshold}")
            print(f"    Действие: {result.action_taken}")
        print()
        
        # Пример 4: Автоматическое управление рисками
        print("🤖 Пример 4: Автоматическое управление рисками")
        risk_rules = RiskManagementRule(
            max_portfolio_loss=-10.0,  # Закрыть все при -10%
            max_position_loss=-5.0,    # Закрыть позицию при -5%
            profit_taking=15.0,         # Взять прибыль при +15%
            auto_close_on_loss=True     # Автоматически закрывать при убытках
        )
        
        risk_management_result = await AlertService.auto_risk_management(
            symbols=test_symbols,
            rules=risk_rules
        )
        
        print(f"Результат автоматического управления рисками:")
        print(f"  Общий PnL портфеля: {risk_management_result.get('total_pnl_percent', 0):.2f}%")
        print(f"  Примененные правила: {json.dumps(risk_management_result.get('rules_applied', {}), indent=2)}")
        
        actions_taken = risk_management_result.get('actions_taken', [])
        if actions_taken:
            print(f"  Выполненные действия:")
            for action in actions_taken:
                print(f"    - {action}")
        else:
            print(f"  Действия не требуются")
        print()
        
        print("✅ Все примеры управления рисками выполнены успешно!")
        print("💡 Используйте эти функции для автоматизации торговли и защиты капитала")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
