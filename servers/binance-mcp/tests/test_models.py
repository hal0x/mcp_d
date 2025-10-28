"""Тесты для моделей данных."""

import pytest
from pydantic import ValidationError

from src.models import (
    AccountInfo,
    BalanceEntry,
    TickerPrice,
    Ticker24h,
    OrderBook,
    KlinesResponse,
    Order,
    Trade,
    ExchangeInfo,
    ServerTime,
    CreateOrderRequest,
    CreateOrderResponse,
    PortfolioSummary,
)


class TestAccountInfo:
    """Тесты для модели AccountInfo."""

    def test_valid_account_info(self):
        """Тест создания валидного AccountInfo."""
        data = {
            "makerCommission": 15,
            "takerCommission": 15,
            "buyerCommission": 0,
            "sellerCommission": 0,
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "updateTime": 1234567890000,
            "accountType": "SPOT",
            "balances": [],
            "permissions": ["SPOT"]
        }
        account = AccountInfo(**data)
        assert account.makerCommission == 15
        assert account.canTrade is True

    def test_invalid_account_info(self):
        """Тест создания невалидного AccountInfo."""
        with pytest.raises(ValidationError):
            AccountInfo(makerCommission="invalid")


class TestBalanceEntry:
    """Тесты для модели BalanceEntry."""

    def test_valid_balance_entry(self):
        """Тест создания валидного BalanceEntry."""
        data = {
            "asset": "BTC",
            "free": "1.00000000",
            "locked": "0.00000000"
        }
        balance = BalanceEntry(**data)
        assert balance.asset == "BTC"
        assert balance.free == "1.00000000"

    def test_balance_entry_with_zero_balance(self):
        """Тест создания BalanceEntry с нулевым балансом."""
        data = {
            "asset": "BTC",
            "free": "0.00000000",
            "locked": "0.00000000"
        }
        balance = BalanceEntry(**data)
        assert balance.asset == "BTC"
        assert balance.free == "0.00000000"


class TestTickerPrice:
    """Тесты для модели TickerPrice."""

    def test_valid_ticker_price(self):
        """Тест создания валидного TickerPrice."""
        data = {
            "symbol": "BTCUSDT",
            "price": "50000.00"
        }
        ticker = TickerPrice(**data)
        assert ticker.symbol == "BTCUSDT"
        assert ticker.price == "50000.00"


class TestCreateOrderRequest:
    """Тесты для модели CreateOrderRequest."""

    def test_valid_market_order(self):
        """Тест создания валидного рыночного ордера."""
        data = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 0.001
        }
        order = CreateOrderRequest(**data)
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.type == "MARKET"
        assert order.quantity == 0.001

    def test_valid_limit_order(self):
        """Тест создания валидного лимитного ордера."""
        data = {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "LIMIT",
            "quantity": 0.001,
            "price": 50000.0,
            "timeInForce": "GTC"
        }
        order = CreateOrderRequest(**data)
        assert order.symbol == "BTCUSDT"
        assert order.side == "SELL"
        assert order.type == "LIMIT"
        assert order.quantity == 0.001
        assert order.price == 50000.0
        assert order.timeInForce == "GTC"

    def test_order_with_futures_flags(self):
        """Тест создания ордера с фьючерсными флагами."""
        data = {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "MARKET",
            "reduce_only": True,
            "close_position": True,
            "price_protect": False,
        }
        order = CreateOrderRequest(**data)
        assert order.reduce_only is True
        assert order.close_position is True
        assert order.price_protect is False

    def test_invalid_order_side(self):
        """Тест создания ордера с невалидной стороной."""
        with pytest.raises(ValidationError):
            CreateOrderRequest(
                symbol="BTCUSDT",
                side="INVALID",
                type="MARKET",
                quantity=0.001
            )

    def test_invalid_order_type(self):
        """Тест создания ордера с невалидным типом."""
        with pytest.raises(ValidationError):
            CreateOrderRequest(
                symbol="BTCUSDT",
                side="BUY",
                type="INVALID",
                quantity=0.001
            )
