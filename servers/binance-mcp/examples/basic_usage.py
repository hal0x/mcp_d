#!/usr/bin/env python3
"""Примеры базового использования Binance MCP сервера."""

import asyncio
import json
from src.client import get_binance_client
from src.services import AccountService, MarketService, OrderService


async def main():
    """Основная функция с примерами использования."""
    print("🚀 Примеры использования Binance MCP сервера\n")
    
    try:
        # Получение информации об аккаунте
        print("📊 Получение информации об аккаунте...")
        account_info = await AccountService.get_account_info()
        print(f"Тип аккаунта: {account_info.accountType}")
        print(f"Может торговать: {account_info.canTrade}")
        print()
        
        # Получение баланса
        print("💰 Получение баланса...")
        balance = await AccountService.get_account_balance()
        non_zero_balances = [b for b in balance if float(b.free) > 0 or float(b.locked) > 0]
        print(f"Найдено {len(non_zero_balances)} ненулевых балансов")
        for b in non_zero_balances[:5]:  # Показываем первые 5
            print(f"  {b.asset}: {b.free} (заблокировано: {b.locked})")
        print()
        
        # Получение цены Bitcoin
        print("📈 Получение цены Bitcoin...")
        btc_price = await MarketService.get_ticker_price("BTCUSDT")
        print(f"Цена BTC/USDT: ${btc_price.price}")
        print()
        
        # Получение 24-часовой статистики
        print("📊 Получение 24-часовой статистики ETH...")
        eth_stats = await MarketService.get_24hr_ticker("ETHUSDT")
        print(f"ETH/USDT:")
        print(f"  Цена: ${eth_stats.lastPrice}")
        print(f"  Изменение: {eth_stats.priceChangePercent}%")
        print(f"  Объем: {eth_stats.volume}")
        print()
        
        # Получение книги ордеров
        print("📖 Получение книги ордеров BTCUSDT...")
        order_book = await MarketService.get_order_book("BTCUSDT", 5)
        print("Лучшие предложения:")
        for bid in order_book.bids[:3]:
            print(f"  {bid[0]} - {bid[1]}")
        print("Лучшие запросы:")
        for ask in order_book.asks[:3]:
            print(f"  {ask[0]} - {ask[1]}")
        print()
        
        # Получение свечей
        print("🕯️ Получение свечей BTCUSDT (1 час)...")
        klines = await MarketService.get_klines("BTCUSDT", "1h", 5)
        print(f"Получено {len(klines.klines)} свечей")
        for kline in klines.klines[:3]:
            print(f"  Время: {kline.openTime}, Открытие: {kline.open}, Закрытие: {kline.close}")
        print()
        
        # Получение открытых ордеров
        print("📋 Получение открытых ордеров...")
        open_orders = await OrderService.get_open_orders()
        print(f"Найдено {len(open_orders)} открытых ордеров")
        for order in open_orders[:3]:
            print(f"  {order.symbol}: {order.side} {order.type} - {order.origQty}")
        print()
        
        # Получение времени сервера
        print("⏰ Получение времени сервера...")
        server_time = await MarketService.get_server_time()
        print(f"Время сервера: {server_time.serverTime}")
        print()
        
        print("✅ Все примеры выполнены успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
