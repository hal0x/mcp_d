"""Генераторы тестовых данных"""
import json
import random
from typing import Dict, Any, List
from datetime import datetime, timedelta

class TestDataGenerator:
    """Генератор тестовых данных для различных сервисов"""
    
    @staticmethod
    def generate_binance_klines(symbol: str = "BTCUSDT", count: int = 100) -> List[Dict[str, Any]]:
        """Генерация исторических данных Binance"""
        base_price = 35000.0 if symbol == "BTCUSDT" else 2000.0
        klines = []
        
        for i in range(count):
            timestamp = int((datetime.now() - timedelta(hours=count-i)).timestamp() * 1000)
            price_change = random.uniform(-0.05, 0.05)  # ±5% изменение
            current_price = base_price * (1 + price_change)
            
            kline = {
                "symbol": symbol,
                "openTime": timestamp,
                "open": f"{current_price:.2f}",
                "high": f"{current_price * 1.02:.2f}",
                "low": f"{current_price * 0.98:.2f}",
                "close": f"{current_price * (1 + random.uniform(-0.01, 0.01)):.2f}",
                "volume": f"{random.uniform(50, 200):.1f}",
                "closeTime": timestamp + 3599999,
                "quoteAssetVolume": f"{current_price * random.uniform(100, 500):.2f}",
                "numberOfTrades": random.randint(100, 2000),
                "takerBuyBaseAssetVolume": f"{random.uniform(25, 100):.2f}",
                "takerBuyQuoteAssetVolume": f"{current_price * random.uniform(50, 250):.2f}",
                "ignore": "0"
            }
            klines.append(kline)
            base_price = float(kline["close"])
        
        return klines
    
    @staticmethod
    def generate_tradingview_alert(symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Генерация алерта TradingView"""
        return {
            "id": f"alert_{random.randint(1000, 9999)}",
            "symbol": symbol,
            "condition": random.choice([
                "price > 50000",
                "RSI > 70",
                "MACD cross",
                "volume > 1000000"
            ]),
            "message": f"{symbol} alert triggered",
            "created_at": datetime.now().isoformat() + "Z",
            "status": "active"
        }
    
    @staticmethod
    def generate_memory_message(chat_id: str = "test_chat") -> Dict[str, Any]:
        """Генерация сообщения для memory-mcp"""
        messages = [
            "Bitcoin достиг новых высот сегодня",
            "Ethereum показывает хорошую динамику",
            "Анализ рынка криптовалют",
            "Торговые сигналы на сегодня",
            "Обзор технических индикаторов"
        ]
        
        return {
            "chat_id": chat_id,
            "message_id": f"msg_{random.randint(1000, 9999)}",
            "text": random.choice(messages),
            "timestamp": datetime.now().isoformat() + "Z",
            "user_id": f"user_{random.randint(100, 999)}"
        }
    
    @staticmethod
    def generate_backtest_strategy() -> Dict[str, Any]:
        """Генерация стратегии для бэктестинга"""
        strategies = [
            {
                "name": "simple_ma_crossover",
                "description": "Simple Moving Average crossover strategy",
                "parameters": {
                    "short_period": random.randint(5, 15),
                    "long_period": random.randint(20, 50),
                    "symbol": random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"]),
                    "interval": random.choice(["1h", "4h", "1d"])
                }
            },
            {
                "name": "rsi_strategy",
                "description": "RSI-based trading strategy",
                "parameters": {
                    "rsi_period": random.randint(10, 20),
                    "oversold": random.randint(20, 40),
                    "overbought": random.randint(60, 80),
                    "symbol": random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"]),
                    "interval": random.choice(["1h", "4h", "1d"])
                }
            }
        ]
        
        return random.choice(strategies)
