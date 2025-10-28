#!/usr/bin/env python3
"""Утилиты для тестирования и отладки."""

import os
import json
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()


def test_connection():
    """Тестирует подключение к Binance API."""
    try:
        # Проверяем режим работы
        demo_trading = os.getenv("BINANCE_DEMO_TRADING", "false").lower() in {"true", "1", "yes", "on"}
        
        if demo_trading:
            api_key = os.getenv("DEMO_BINANCE_API_KEY")
            api_secret = os.getenv("DEMO_BINANCE_API_SECRET")
            mode = "DEMO"
            base_endpoint = "https://demo-fapi.binance.com"
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            mode = "LIVE"
            base_endpoint = None
        
        if not api_key or not api_secret:
            print(f"❌ Ошибка: {'DEMO_' if demo_trading else ''}BINANCE_API_KEY и {'DEMO_' if demo_trading else ''}BINANCE_API_SECRET должны быть установлены")
            return False
        
        print(f"🔑 Режим: {mode}")
        print(f"🔑 Используем API ключ: {api_key[:8]}...")
        if base_endpoint:
            print(f"🌐 Endpoint: {base_endpoint}")
        
        if demo_trading:
            # Для демо режима нужно переопределить URL, так как testnet работает только для спота
            demo_futures_host = "https://demo-fapi.binance.com"
            Client.FUTURES_TESTNET_URL = f"{demo_futures_host}/fapi"
            Client.FUTURES_DATA_TESTNET_URL = f"{demo_futures_host}/futures/data"
            Client.FUTURES_COIN_TESTNET_URL = "https://demo-dapi.binance.com/dapi"
            Client.FUTURES_COIN_DATA_TESTNET_URL = "https://demo-dapi.binance.com/futures/data"
            client = Client(api_key, api_secret, testnet=True)
        else:
            client = Client(api_key, api_secret, testnet=False)
        
        # Тестируем подключение
        print("\n📊 Тестирование подключения...")
        server_time = client.get_server_time()
        print(f"✅ Время сервера: {server_time.get('serverTime', 'N/A')}")
        
        # Тестируем получение информации об аккаунте
        print("📊 Получение информации об аккаунте...")
        account = client.get_account()
        print(f"✅ Тип аккаунта: {account.get('accountType', 'UNKNOWN')}")
        print(f"✅ Может торговать: {account.get('canTrade', False)}")
        
        # Тестируем получение цены
        print("📈 Получение цены BTCUSDT...")
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        print(f"✅ Цена BTCUSDT: {ticker.get('price', 'N/A')}")
        
        print("\n🎉 Все тесты прошли успешно!")
        return True
        
    except BinanceAPIException as e:
        print(f"❌ Ошибка Binance API: {e}")
        return False
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
        return False


def show_balance():
    """Показывает баланс аккаунта."""
    try:
        # Проверяем режим работы
        demo_trading = os.getenv("BINANCE_DEMO_TRADING", "false").lower() in {"true", "1", "yes", "on"}
        
        if demo_trading:
            api_key = os.getenv("DEMO_BINANCE_API_KEY")
            api_secret = os.getenv("DEMO_BINANCE_API_SECRET")
            mode = "DEMO"
            base_endpoint = "https://demo-fapi.binance.com"
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            mode = "LIVE"
            base_endpoint = None
        
        if not api_key or not api_secret:
            print(f"❌ Ошибка: {'DEMO_' if demo_trading else ''}BINANCE_API_KEY и {'DEMO_' if demo_trading else ''}BINANCE_API_SECRET должны быть установлены")
            return
        
        print(f"🔑 Режим: {mode}")
        if base_endpoint:
            print(f"🌐 Endpoint: {base_endpoint}")
        
        if demo_trading:
            # Для демо режима нужно переопределить URL, так как testnet работает только для спота
            demo_futures_host = "https://demo-fapi.binance.com"
            Client.FUTURES_TESTNET_URL = f"{demo_futures_host}/fapi"
            Client.FUTURES_DATA_TESTNET_URL = f"{demo_futures_host}/futures/data"
            Client.FUTURES_COIN_TESTNET_URL = "https://demo-dapi.binance.com/dapi"
            Client.FUTURES_COIN_DATA_TESTNET_URL = "https://demo-dapi.binance.com/futures/data"
            client = Client(api_key, api_secret, testnet=True)
        else:
            client = Client(api_key, api_secret, testnet=False)
        account = client.get_account()
        
        balances = []
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked
            
            if total > 0:
                balances.append({
                    'asset': balance['asset'],
                    'free': free,
                    'locked': locked,
                    'total': total
                })
        
        balances.sort(key=lambda x: x['total'], reverse=True)
        
        print(f"\n💰 Баланс аккаунта:")
        print("=" * 50)
        for balance in balances:
            print(f"{balance['asset']:>8}: {balance['total']:>12.8f} (свободно: {balance['free']:>12.8f}, заблокировано: {balance['locked']:>12.8f})")
        
        if not balances:
            print("📭 Нет активов с ненулевым балансом")
            
    except BinanceAPIException as e:
        print(f"❌ Ошибка Binance API: {e}")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "balance":
        show_balance()
    else:
        test_connection()
