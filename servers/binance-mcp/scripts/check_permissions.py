#!/usr/bin/env python3
"""Проверка прав доступа API ключей."""

import asyncio
import json
from src.client import get_binance_client, get_config
from src.services import AccountService, OrderService

async def check_api_permissions():
    """Проверяет права доступа API ключей."""
    try:
        config = get_config()
        client = get_binance_client()
        
        print("🔍 Проверка прав доступа API ключей...")
        print(f"Режим: {'DEMO' if config.is_demo_mode else 'LIVE'}")
        print(f"API ключ: {config.effective_api_key[:8]}...")
        
        # Проверяем информацию об аккаунте
        print("\n1. Проверка информации об аккаунте...")
        try:
            account_info = await AccountService.get_account_info()
            print(f"✅ Аккаунт доступен: {account_info.account_type}")
            print(f"   Может торговать: {account_info.can_trade}")
            print(f"   Права: {account_info.permissions}")
        except Exception as e:
            print(f"❌ Ошибка аккаунта: {e}")
        
        # Проверяем баланс
        print("\n2. Проверка баланса...")
        try:
            balance = await AccountService.get_account_balance()
            print(f"✅ Баланс доступен: {len(balance)} активов")
            for asset in balance[:3]:  # Показываем первые 3
                print(f"   {asset.asset}: {asset.total}")
        except Exception as e:
            print(f"❌ Ошибка баланса: {e}")
        
        # Проверяем открытые ордера
        print("\n3. Проверка открытых ордеров...")
        try:
            orders = await OrderService.get_open_orders()
            print(f"✅ Открытые ордера доступны: {len(orders)} ордеров")
        except Exception as e:
            print(f"❌ Ошибка открытых ордеров: {e}")
        
        # Проверяем историю ордеров
        print("\n4. Проверка истории ордеров...")
        try:
            history = await OrderService.get_order_history("BTCUSDT", 5)
            print(f"✅ История ордеров доступна: {len(history)} ордеров")
        except Exception as e:
            print(f"❌ Ошибка истории ордеров: {e}")
        
        # Проверяем историю сделок
        print("\n5. Проверка истории сделок...")
        try:
            trades = await OrderService.get_trade_history("BTCUSDT", 5)
            print(f"✅ История сделок доступна: {len(trades)} сделок")
        except Exception as e:
            print(f"❌ Ошибка истории сделок: {e}")
        
        print("\n🎯 Рекомендации:")
        if config.is_demo_mode:
            print("   - Используется демо режим - безопасно для тестирования")
        else:
            print("   - Используется LIVE режим - будьте осторожны!")
            print("   - Убедитесь что API ключи имеют права на торговлю")
        
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(check_api_permissions())
