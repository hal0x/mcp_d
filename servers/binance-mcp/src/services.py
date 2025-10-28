"""Сервисы для работы с Binance API."""

import asyncio
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from typing_extensions import Literal

import structlog

from binance.exceptions import BinanceAPIException
from fastapi import HTTPException

from .client import get_binance_client
from .config import get_config, Config
from .cache import get_cached_json, set_cached_json
from .models import (
    AccountInfo,
    BalanceEntry,
    TickerPrice,
    Ticker24h,
    OrderBook,
    OrderBookLevel,
    KlinesResponse,
    Kline,
    Order,
    Trade,
    ExchangeInfo,
    ServerTime,
    CreateOrderRequest,
    CreateOrderResponse,
    Position,
    PortfolioMetrics,
    AccountSummary,
    PortfolioSummary,
    AveragePrice,
    RecentTrade,
    OrderDetails,
    CancelOrderResponse,
    CancelOrderRequest,
    TradeFee,
    RecentTradesBatchEntry,
    FuturesPosition,
    FuturesLeverageChange,
    FuturesMarginTypeChange,
    MarginAccount,
    MarginBalance,
    MarginOrder,
    MarginTrade,
    CreateMarginOrderRequest,
    CreateMarginOrderResponse,
    CancelMarginOrderRequest,
    CancelMarginOrderResponse,
    OCOOrder,
    OCOLeg,
    BatchResult,
    PortfolioOverview,
    TickerBatchItem,
    AvailablePair,
    TradingLimitsInfo,
    CreatePositionRequest,
    SafetyRule,
    SafetyCheckResult,
    StopLossConfig,
    StopLossResult,
    AlertConfig,
    AlertResult,
    RiskManagementRule,
    TelegramNotification,
    PortfolioRiskMetrics,
    PositionLimitUpdate,
    PositionLimitResponse,
    DerivativesContext,
)
from .storage import PostgresStorage
from .risk_tools import (
    PositionSizeRequest,
    compute_position_size,
    compute_rr,
    enforce_min_rr,
    should_halt,
)

logger = structlog.get_logger(__name__)

getcontext().prec = 28

_POSITION_LIMITS: Dict[str, Dict[str, Any]] = {}



async def call_binance(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Асинхронный вызов метода Binance API."""
    try:
        return await asyncio.to_thread(method, *args, **kwargs)
    except asyncio.CancelledError:
        logger.warning(
            "binance_call_cancelled", method=getattr(method, "__name__", repr(method))
        )
        raise
    except BinanceAPIException as exc:
        logger.warning(
            "binance_api_error",
            method=getattr(method, "__name__", repr(method)),
            error=str(exc),
            code=getattr(exc, "code", None),
            status=getattr(exc, "status_code", None),
        )
        detail: Dict[str, Any] = {
            "message": getattr(exc, "message", str(exc)),
            "code": getattr(exc, "code", None),
        }
        status = getattr(exc, "status_code", None) or 502
        response_payload = getattr(exc, "response", None)
        if response_payload:
            detail["response"] = response_payload
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        logger.exception(
            "binance_call_unexpected_error",
            method=getattr(method, "__name__", repr(method)),
        )
        raise HTTPException(
            status_code=500, detail="Неожиданная ошибка клиента Binance"
        ) from exc


class RiskService:
    """Расширенные функции управления рисками."""

    _storage: PostgresStorage | None = None

    @classmethod
    async def configure(cls, storage: PostgresStorage) -> None:
        cls._storage = storage
        records = await storage.fetch_position_limits()
        for record in records:
            symbol = record["symbol"]
            cls_timestamp = int(float(record.get("updated_at") or 0))
            _POSITION_LIMITS[symbol] = {
                "max_size": float(record.get("max_size") or 0.0),
                "updated_at": cls_timestamp,
            }
        logger.info("position_limits_loaded", count=len(_POSITION_LIMITS))

    @staticmethod
    async def get_portfolio_risk_metrics() -> PortfolioRiskMetrics:
        client = get_binance_client()
        config = get_config()
        total_equity = 0.0
        available_balance = 0.0
        unrealized_pnl = 0.0
        margin_ratio = None
        positions_count = 0
        largest_position: str | None = None

        if config.is_demo_mode:
            account = await call_binance(client.futures_account)
            total_equity = float(account.get("totalWalletBalance", 0.0))
            available_balance = float(account.get("availableBalance", 0.0))
            unrealized_pnl = float(account.get("totalUnrealizedProfit", 0.0))
            margin_ratio = (
                float(account.get("marginRatio"))
                if account.get("marginRatio") not in (None, "")
                else None
            )
            positions = [
                p
                for p in account.get("positions", [])
                if abs(float(p.get("positionAmt", 0.0))) > 0
            ]
            positions_count = len(positions)
            if positions:
                largest = max(
                    positions, key=lambda p: abs(float(p.get("notional", 0.0)))
                )
                largest_position = largest.get("symbol")
        else:
            account = await call_binance(client.get_account)
            balances = account.get("balances", [])
            total_equity = sum(
                float(b.get("free", 0)) + float(b.get("locked", 0)) for b in balances
            )
            available_balance = sum(float(b.get("free", 0)) for b in balances)
        metrics = PortfolioRiskMetrics(
            total_equity=total_equity,
            available_balance=available_balance,
            unrealized_pnl=unrealized_pnl,
            margin_ratio=margin_ratio,
            positions_count=positions_count,
            largest_position=largest_position,
        )
        if RiskService._storage is not None:
            await RiskService._storage.record_portfolio_metrics(
                metrics.model_dump(mode="json")
            )
        return metrics

    @staticmethod
    async def set_position_limit(update: PositionLimitUpdate) -> PositionLimitResponse:
        symbol = update.symbol.upper()
        updated_at = int(time())
        _POSITION_LIMITS[symbol] = {
            "max_size": update.max_size,
            "updated_at": updated_at,
        }
        if RiskService._storage is not None:
            await RiskService._storage.upsert_position_limit(
                symbol, update.max_size, updated_at
            )
        return PositionLimitResponse(
            symbol=symbol, max_size=update.max_size, updated_at=updated_at
        )

    @staticmethod
    async def get_position_limits() -> Dict[str, Any]:
        return {symbol: data.copy() for symbol, data in _POSITION_LIMITS.items()}

    @staticmethod
    async def get_derivatives_data(symbol: str) -> DerivativesContext:
        symbol_upper = symbol.upper()
        cache_key = f"derivatives_ctx:{symbol_upper}"
        cached = await get_cached_json(cache_key)
        if cached:
            return DerivativesContext.model_validate(cached)
        client = get_binance_client()
        open_interest = None
        funding_rate = None
        mark_price = None
        last_funding_time = None
        try:
            data = await call_binance(client.futures_open_interest, symbol=symbol_upper)
            open_interest = float(data.get("openInterest", 0.0))
        except Exception:  # noqa: BLE001
            logger.debug("Open interest not available for %s", symbol_upper)
        try:
            mark_data = await call_binance(
                client.futures_mark_price, symbol=symbol_upper
            )
            funding_rate = float(mark_data.get("lastFundingRate", 0.0))
            mark_price = float(mark_data.get("markPrice", 0.0))
            last_funding_time = int(mark_data.get("nextFundingTime", 0))
        except Exception:  # noqa: BLE001
            logger.debug("Funding/mark price not available for %s", symbol_upper)
        context = DerivativesContext(
            symbol=symbol_upper,
            open_interest=open_interest,
            funding_rate=funding_rate,
            mark_price=mark_price,
            last_funding_time=last_funding_time,
        )
        await set_cached_json(cache_key, context.model_dump(mode="json"), ttl=20)
        return context

    @staticmethod
    async def calculate_position_size(request: PositionSizeRequest) -> Dict[str, float]:
        result = compute_position_size(request)
        return {
            "quantity": result.quantity,
            "effective_entry": result.effective_entry,
            "risk_amount": result.risk_amount,
        }

    @staticmethod
    async def evaluate_trade(
        *,
        entry: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        side: str,
    ) -> Dict[str, float]:
        # Приводим side к правильному типу
        side_lower = side.lower()
        side_normalized: str = "long" if side_lower in ["buy", "long"] else "short"

        # Приводим к правильному Literal типу

        side_literal = cast(Literal["long", "short"], side_normalized)

        rr = compute_rr(entry, take_profit, stop_loss, side_literal)
        adjusted = enforce_min_rr(entry, take_profit, stop_loss, atr, side_literal)
        return {"rr": rr, "adjusted_tp": adjusted["tp"], "adjusted_rr": adjusted["rr"]}

    @staticmethod
    async def should_halt(total_pnl_pct_today: float, consecutive_losses: int) -> bool:
        return should_halt(total_pnl_pct_today, consecutive_losses)


class AccountService:
    """Сервис для работы с аккаунтом."""

    @staticmethod
    async def get_account_info() -> AccountInfo:
        """Получает информацию об аккаунте."""
        client = get_binance_client()
        config = get_config()

        if config.is_demo_mode:
            # Для демо режима используем futures API
            account = await call_binance(client.futures_account)
            return AccountInfo(
                account_type="FUTURES",
                can_trade=account.get("canTrade", False),
                can_withdraw=account.get("canWithdraw", False),
                can_deposit=account.get("canDeposit", False),
                update_time=account.get("updateTime", 0),
                balances_count=len(account.get("assets", [])),
                permissions=["FUTURES"],
            )
        else:
            # Для live режима используем spot API
            account = await call_binance(client.get_account)
            return AccountInfo(
                account_type=account.get("accountType", "UNKNOWN"),
                can_trade=account.get("canTrade", False),
                can_withdraw=account.get("canWithdraw", False),
                can_deposit=account.get("canDeposit", False),
                update_time=account.get("updateTime", 0),
                balances_count=len(account.get("balances", [])),
                permissions=account.get("permissions", []),
            )

    @staticmethod
    async def get_account_balance() -> List[BalanceEntry]:
        """Получает баланс аккаунта."""
        client = get_binance_client()
        config = get_config()

        if config.is_demo_mode:
            # Для демо режима используем futures API
            account = await call_binance(client.futures_account)
            balances: List[BalanceEntry] = []
            for asset in account.get("assets", []):
                wallet_balance = float(asset.get("walletBalance", 0))
                if wallet_balance > 0:
                    balances.append(
                        BalanceEntry(
                            asset=asset.get("asset", "UNKNOWN"),
                            free=wallet_balance,
                            locked=0.0,
                            total=wallet_balance,
                        )
                    )
        else:
            # Для live режима используем spot API
            account = await call_binance(client.get_account)
            spot_balances: List[BalanceEntry] = []
            for balance in account.get("balances", []):
                free = float(balance.get("free", 0))
                locked = float(balance.get("locked", 0))
                total = free + locked

                if total > 0:
                    spot_balances.append(
                        BalanceEntry(
                            asset=balance.get("asset", "UNKNOWN"),
                            free=free,
                            locked=locked,
                            total=total,
                        )
                    )

        # Объединяем балансы из обоих режимов
        all_balances = balances if config.is_demo_mode else spot_balances
        all_balances.sort(key=lambda entry: entry.total, reverse=True)
        return all_balances

    @staticmethod
    async def get_trade_fee(symbol: str) -> TradeFee:
        """Получает торговые комиссии для символа."""
        if not symbol:
            raise ValueError("Параметр symbol обязателен")
        client = get_binance_client()
        config = get_config()
        symbol_upper = symbol.upper()
        if config.is_demo_mode:
            # В демо/фьючерсном режиме используем futures endpoint
            futures_method = getattr(client, "futures_comission_rate", None)
            if futures_method is None:
                raise RuntimeError(
                    "futures_comission_rate не поддерживается текущей версией python-binance"
                )
            data = await call_binance(futures_method, symbol=symbol_upper)
            return TradeFee(
                symbol=data.get("symbol", symbol_upper),
                maker_fee=float(data.get("makerCommissionRate", 0.0)),
                taker_fee=float(data.get("takerCommissionRate", 0.0)),
            )
        else:
            data = await call_binance(client.get_trade_fee, symbol=symbol_upper)
            # API возвращает список словарей
            entry = data[0] if isinstance(data, list) and data else data
            if isinstance(entry, list):
                entry = entry[0] if entry else {}
            return TradeFee(
                symbol=entry.get("symbol", symbol_upper),
                maker_fee=float(entry.get("makerCommission", 0.0)),
                taker_fee=float(entry.get("takerCommission", 0.0)),
                maker_fee_bnb=(
                    float(entry.get("makerCommissionBnb"))
                    if entry.get("makerCommissionBnb") is not None
                    else None
                ),
                taker_fee_bnb=(
                    float(entry.get("takerCommissionBnb"))
                    if entry.get("takerCommissionBnb") is not None
                    else None
                ),
            )

    @staticmethod
    async def get_trade_fee_batch(symbols: List[str]) -> List[TradeFee]:
        """Получает торговые комиссии для набора символов."""
        if not symbols:
            raise ValueError("Parameter symbols must contain at least one item")

        normalized: List[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if not symbol:
                raise ValueError("Empty symbol provided in symbols list")
            upper = symbol.upper()
            if upper not in seen:
                seen.add(upper)
                normalized.append(upper)

        results: List[TradeFee] = []
        for symbol in normalized:
            fee = await AccountService.get_trade_fee(symbol)
            results.append(fee)
        return results


class MarketService:
    """Сервис для работы с рыночными данными."""

    @staticmethod
    async def get_ticker_price(symbol: str) -> TickerPrice:
        """Получает текущую цену символа."""
        symbol_upper = symbol.upper()
        cache_key = f"ticker_price:{symbol_upper}"
        cached = await get_cached_json(cache_key)
        if cached:
            return TickerPrice.model_validate(cached)
        client = get_binance_client()
        result = await call_binance(client.get_symbol_ticker, symbol=symbol_upper)
        payload = {"symbol": result["symbol"], "price": float(result["price"])}
        await set_cached_json(cache_key, payload, ttl=5)
        return TickerPrice.model_validate(payload)

    @staticmethod
    async def get_24hr_ticker(symbol: str) -> Ticker24h:
        """Получает статистику за 24 часа."""
        symbol_upper = symbol.upper()
        cache_key = f"ticker24h:{symbol_upper}"
        cached = await get_cached_json(cache_key)
        if cached:
            return Ticker24h.model_validate(cached)
        client = get_binance_client()
        ticker = await call_binance(client.get_ticker, symbol=symbol_upper)

        payload = dict(
            symbol=ticker["symbol"],
            price=float(ticker["lastPrice"]),
            price_change=float(ticker["priceChange"]),
            price_change_percent=float(ticker["priceChangePercent"]),
            high=float(ticker["highPrice"]),
            low=float(ticker["lowPrice"]),
            volume=float(ticker["volume"]),
            quote_volume=float(ticker["quoteVolume"]),
            trade_count=int(ticker["count"]),
        )
        await set_cached_json(cache_key, payload, ttl=10)
        return Ticker24h.model_validate(payload)

    @staticmethod
    async def get_order_book(symbol: str, limit: int = 100) -> OrderBook:
        """Получает книгу ордеров."""
        symbol_upper = symbol.upper()
        cache_key = f"order_book:{symbol_upper}:{limit}"
        cached = await get_cached_json(cache_key)
        if cached:
            return OrderBook.model_validate(cached)
        client = get_binance_client()
        book = await call_binance(
            client.get_order_book, symbol=symbol_upper, limit=limit
        )

        bids = [
            OrderBookLevel(price=float(price), quantity=float(quantity))
            for price, quantity in book.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(price), quantity=float(quantity))
            for price, quantity in book.get("asks", [])
        ]

        order_book = OrderBook(
            symbol=symbol_upper,
            last_update_id=book.get("lastUpdateId", 0),
            bids=bids,
            asks=asks,
            total_bids=len(bids),
            total_asks=len(asks),
        )
        await set_cached_json(cache_key, order_book.model_dump(mode="json"), ttl=5)
        return order_book

    @staticmethod
    async def get_klines(
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> KlinesResponse:
        """Получает данные свечей."""
        client = get_binance_client()
        symbol_upper = symbol.upper()
        cache_key = (
            f"klines:{symbol_upper}:{interval}:{limit}:"
            f"{start_time or 'none'}:{end_time or 'none'}"
        )
        cached = await get_cached_json(cache_key)
        if cached:
            return KlinesResponse.model_validate(cached)

        params: dict[str, object] = {
            "symbol": symbol_upper,
            "interval": interval,
            "limit": max(1, min(int(limit), 1000)),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        raw_klines = await call_binance(client.get_klines, **params)

        klines = [
            Kline(
                open_time=int(entry[0]),
                open=float(entry[1]),
                high=float(entry[2]),
                low=float(entry[3]),
                close=float(entry[4]),
                volume=float(entry[5]),
                close_time=int(entry[6]),
            )
            for entry in raw_klines
        ]

        response = KlinesResponse(
            symbol=symbol_upper,
            interval=interval,
            total_klines=len(raw_klines),
            recent_klines=klines,
        )
        await set_cached_json(cache_key, response.model_dump(mode="json"), ttl=20)
        return response

    @staticmethod
    async def get_recent_trades(symbol: str, limit: int = 50) -> List[Trade]:
        """Получает список последних сделок."""
        client = get_binance_client()
        symbol_upper = symbol.upper()
        limit = max(1, min(limit, 1000))
        cache_key = f"recent_trades:{symbol_upper}:{limit}"
        cached = await get_cached_json(cache_key)
        if cached:
            return [Trade.model_validate(item) for item in cached]
        trades_raw = await call_binance(
            client.get_recent_trades, symbol=symbol_upper, limit=limit
        )
        results: List[Trade] = []
        for item in trades_raw:
            results.append(
                Trade(
                    trade_id=item.get("id"),
                    symbol=symbol_upper,
                    order_id=item.get("orderId") or 0,
                    side="SELL" if item.get("isBuyerMaker") else "BUY",
                    quantity=float(item.get("qty", 0.0)),
                    price=float(item.get("price", 0.0)),
                    commission=float(item.get("commission", 0.0)),
                    commission_asset=item.get("commissionAsset", ""),
                    time=int(item.get("time", 0)),
                )
            )
        await set_cached_json(
            cache_key, [trade.model_dump(mode="json") for trade in results], ttl=3
        )
        return results

    @staticmethod
    async def get_all_tickers() -> List[TickerPrice]:
        """Получает цены всех символов."""
        client = get_binance_client()
        config = get_config()
        if config.is_demo_mode:
            tickers_raw = await call_binance(client.futures_symbol_ticker)
            if isinstance(tickers_raw, dict):
                tickers = [tickers_raw]
            else:
                tickers = tickers_raw
        else:
            tickers = await call_binance(client.get_all_tickers)
        return [
            TickerPrice(symbol=entry["symbol"], price=float(entry["price"]))
            for entry in tickers
        ]

    @staticmethod
    async def get_avg_price(symbol: str) -> AveragePrice:
        """Получает среднюю цену символа."""
        client = get_binance_client()
        config = get_config()
        symbol_upper = symbol.upper()
        if config.is_demo_mode:
            data = await call_binance(client.futures_mark_price, symbol=symbol_upper)
            price = float(data.get("markPrice", 0.0))
            minutes = 0
        else:
            data = await call_binance(client.get_avg_price, symbol=symbol_upper)
            price = float(data.get("price", 0.0))
            minutes = int(data.get("mins", 0))
        return AveragePrice(symbol=symbol_upper, minutes=minutes, price=price)

    @staticmethod
    async def get_public_trades(symbol: str, limit: int = 20) -> List[RecentTrade]:
        """Получает последние публичные сделки по символу."""
        client = get_binance_client()
        config = get_config()
        symbol_upper = symbol.upper()
        params = {"symbol": symbol_upper, "limit": limit}
        if config.is_demo_mode:
            trades_raw = await call_binance(client.futures_recent_trades, **params)
        else:
            trades_raw = await call_binance(client.get_recent_trades, **params)
        trades: List[RecentTrade] = []
        for entry in trades_raw:
            trades.append(
                RecentTrade(
                    id=int(entry.get("id", entry.get("a", 0))),
                    price=float(entry.get("price", 0.0)),
                    quantity=float(entry.get("qty", entry.get("quantity", 0.0))),
                    timestamp=int(entry.get("time", entry.get("T", 0))),
                    is_buyer_maker=bool(
                        entry.get("isBuyerMaker", entry.get("m", False))
                    ),
                )
            )
        return trades

    @staticmethod
    async def get_avg_price_batch(symbols: List[str]) -> List[AveragePrice]:
        """Получает средние цены для набора символов."""
        if not symbols:
            raise ValueError("Parameter symbols must contain at least one item")

        normalized: List[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if not symbol:
                raise ValueError("Empty symbol provided in symbols list")
            upper = symbol.upper()
            if upper not in seen:
                seen.add(upper)
                normalized.append(upper)

        result: List[AveragePrice] = []
        for symbol in normalized:
            price_info = await MarketService.get_avg_price(symbol)
            result.append(price_info)
        return result

    @staticmethod
    async def get_public_trades_batch(
        symbols: List[str], limit: int = 20
    ) -> List[RecentTradesBatchEntry]:
        """Получает последние публичные сделки для набора символов."""
        if not symbols:
            raise ValueError("Parameter symbols must contain at least one item")
        if limit <= 0:
            raise ValueError("Parameter limit must be greater than zero")

        normalized: List[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if not symbol:
                raise ValueError("Empty symbol provided in symbols list")
            upper = symbol.upper()
            if upper not in seen:
                seen.add(upper)
                normalized.append(upper)

        result: List[RecentTradesBatchEntry] = []
        for symbol in normalized:
            trades = await MarketService.get_public_trades(symbol, limit=limit)
            result.append(RecentTradesBatchEntry(symbol=symbol, trades=trades))
        return result

    @staticmethod
    async def get_order_book_batch(
        symbols: List[str], limit: int = 100
    ) -> List[OrderBook]:
        """Получает книги ордеров для набора символов."""
        if not symbols:
            raise ValueError("Parameter symbols must contain at least one item")
        if limit <= 0:
            raise ValueError("Parameter limit must be greater than zero")

        normalized: List[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if not symbol:
                raise ValueError("Empty symbol provided in symbols list")
            upper = symbol.upper()
            if upper not in seen:
                seen.add(upper)
                normalized.append(upper)

        result: List[OrderBook] = []
        for symbol in normalized:
            book = await MarketService.get_order_book(symbol, limit=limit)
            result.append(book)
        return result

    @staticmethod
    async def get_klines_batch(
        symbols: List[str],
        interval: str = "1h",
        limit: int = 100,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> List[KlinesResponse]:
        """Получает данные свечей для набора символов."""
        if not symbols:
            raise ValueError("Parameter symbols must contain at least one item")
        if limit <= 0:
            raise ValueError("Parameter limit must be greater than zero")

        normalized: List[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if not symbol:
                raise ValueError("Empty symbol provided in symbols list")
            upper = symbol.upper()
            if upper not in seen:
                seen.add(upper)
                normalized.append(upper)

        result: List[KlinesResponse] = []
        for symbol in normalized:
            klines = await MarketService.get_klines(
                symbol,
                interval=interval,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
            )
            result.append(klines)
        return result


class OrderService:
    """Сервис для работы с ордерами."""

    @staticmethod
    async def get_open_orders(symbol: Optional[str] = None) -> List[Order]:
        """Получает открытые ордера."""
        client = get_binance_client()
        config = get_config()

        if config.is_demo_mode:
            if symbol:
                orders_raw = await call_binance(
                    client.futures_get_open_orders, symbol=symbol.upper()
                )
            else:
                orders_raw = await call_binance(client.futures_get_open_orders)
        else:
            if symbol:
                orders_raw = await call_binance(
                    client.get_open_orders, symbol=symbol.upper()
                )
            else:
                orders_raw = await call_binance(client.get_open_orders)

        orders: List[Order] = []
        for order in orders_raw:
            price_str = order.get("price")
            price_value = (
                float(price_str)
                if price_str not in {None, "0", "0.00000000", ""}
                else None
            )
            time_key = "time"
            if config.is_demo_mode:
                time_key = "updateTime" if "updateTime" in order else "time"
            orders.append(
                Order(
                    order_id=int(order["orderId"]),
                    symbol=order["symbol"],
                    side=order.get("side", "UNKNOWN"),
                    type=order.get("type", order.get("origType", "UNKNOWN")),
                    quantity=float(
                        order.get("origQty", order.get("origQuantity", 0.0))
                    ),
                    price=price_value,
                    status=order["status"],
                    time=int(order.get(time_key, 0)),
                )
            )

        return orders

    @staticmethod
    async def get_order_history(symbol: str, limit: int = 10) -> List[Order]:
        """Получает историю ордеров."""
        client = get_binance_client()
        config = get_config()
        symbol_upper = symbol.upper()

        if config.is_demo_mode:
            orders_raw = await call_binance(
                client.futures_get_all_orders, symbol=symbol_upper, limit=limit
            )
        else:
            orders_raw = await call_binance(
                client.get_all_orders, symbol=symbol_upper, limit=limit
            )

        history: List[Order] = []
        for order in orders_raw:
            price_str = order.get("price")
            price_value = (
                float(price_str)
                if price_str not in {None, "0", "0.00000000", ""}
                else None
            )
            order_type = order.get("type", order.get("origType", "UNKNOWN"))
            time_key = "time"
            if config.is_demo_mode:
                time_key = "updateTime" if "updateTime" in order else "time"
            history.append(
                Order(
                    order_id=int(order["orderId"]),
                    symbol=order["symbol"],
                    side=order.get("side", "UNKNOWN"),
                    type=order_type,
                    quantity=float(
                        order.get("origQty", order.get("origQuantity", 0.0))
                    ),
                    price=price_value,
                    status=order["status"],
                    time=int(order.get(time_key, 0)),
                )
            )

        return history

    @staticmethod
    async def get_trade_history(symbol: str, limit: int = 10) -> List[Trade]:
        """Получает историю сделок."""
        client = get_binance_client()
        config = get_config()
        symbol_upper = symbol.upper()

        if config.is_demo_mode:
            trades_raw = await call_binance(
                client.futures_account_trades, symbol=symbol_upper, limit=limit
            )
        else:
            trades_raw = await call_binance(
                client.get_my_trades, symbol=symbol_upper, limit=limit
            )

        trades: List[Trade] = []
        for trade in trades_raw:
            if config.is_demo_mode:
                side = trade.get("side", "BUY")
                qty_key = "qty" if "qty" in trade else "origQty"
                commission_value = float(trade.get("commission", 0.0))
            else:
                side = "BUY" if trade.get("isBuyer") else "SELL"
                qty_key = "qty"
                commission_value = float(trade.get("commission", 0.0))

            trades.append(
                Trade(
                    trade_id=int(trade["id"]),
                    symbol=trade["symbol"],
                    order_id=int(trade["orderId"]),
                    side=side,
                    quantity=float(trade.get(qty_key, 0.0)),
                    price=float(trade["price"]),
                    commission=commission_value,
                    commission_asset=trade["commissionAsset"],
                    time=int(trade["time"]),
                )
            )

        return trades

    @staticmethod
    async def get_order(
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderDetails:
        """Получает детали ордера."""
        if order_id is None and client_order_id is None:
            raise ValueError("Необходимо указать order_id или client_order_id")
        client = get_binance_client()
        config = get_config()
        symbol_upper = symbol.upper()
        params = {"symbol": symbol_upper}
        if order_id is not None:
            params["orderId"] = str(order_id)
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        if config.is_demo_mode:
            order = await call_binance(client.futures_get_order, **params)
            executed = float(
                order.get("executedQty", order.get("executedQuantity", 0.0))
            )
        else:
            order = await call_binance(client.get_order, **params)
            executed = float(order.get("executedQty", 0.0))
        time_key = "updateTime" if "updateTime" in order else "time"
        price_str = order.get("price")
        price_value = (
            float(price_str) if price_str not in {None, "", "0", "0.00000000"} else 0.0
        )
        return OrderDetails(
            order_id=int(order.get("orderId", order.get("origOrderId", 0))),
            symbol=order.get("symbol", symbol_upper),
            status=order.get("status", "UNKNOWN"),
            side=order.get("side", "UNKNOWN"),
            type=order.get("type", order.get("origType", "UNKNOWN")),
            price=price_value,
            executed_quantity=executed,
            original_quantity=float(
                order.get("origQty", order.get("origQuantity", 0.0))
            ),
            time=int(order.get(time_key, 0)),
        )

    @staticmethod
    async def cancel_order(
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> CancelOrderResponse:
        """Отменяет ордер."""
        if order_id is None and client_order_id is None:
            raise ValueError("Необходимо указать order_id или client_order_id")
        client = get_binance_client()
        config = get_config()
        symbol_upper = symbol.upper()
        params = {"symbol": symbol_upper}
        if order_id is not None:
            params["orderId"] = str(order_id)
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        if config.is_demo_mode:
            result = await call_binance(client.futures_cancel_order, **params)
        else:
            result = await call_binance(client.cancel_order, **params)
        return CancelOrderResponse(
            order_id=int(result.get("orderId", 0)),
            symbol=result.get("symbol", symbol_upper),
            status=result.get("status", "CANCELED"),
            client_order_id=result.get(
                "clientOrderId", result.get("origClientOrderId", "")
            ),
            orig_client_order_id=result.get(
                "origClientOrderId", result.get("clientOrderId", "")
            ),
        )

    @staticmethod
    async def create_order(order_request: CreateOrderRequest) -> CreateOrderResponse:
        """Создает новый ордер."""
        client = get_binance_client()
        config = get_config()

        # Валидация типа ордера
        order_type = order_request.type.upper()
        if order_type not in ["MARKET", "LIMIT"]:
            raise ValueError(
                f"Неподдерживаемый тип ордера: {order_type}. Поддерживаются только MARKET и LIMIT"
            )

        close_entire_position = bool(
            order_request.close_position and config.is_demo_mode
        )

        # Подготавливаем параметры для создания ордера
        order_params = {
            "symbol": order_request.symbol.upper(),
            "side": order_request.side.upper(),
            "type": order_type,
            "newOrderRespType": order_request.new_order_resp_type,
        }

        # Добавляем timeInForce только для лимитных ордеров
        if order_type == "LIMIT" and order_request.time_in_force:
            order_params["timeInForce"] = order_request.time_in_force

        # Добавляем количество (обязательно для всех ордеров)
        if close_entire_position:
            # Для closePosition количество не передается
            quantity_for_api: Optional[float] = None
        else:
            quantity_for_api = order_request.quantity

        if quantity_for_api is not None:
            order_params["quantity"] = str(quantity_for_api)
        elif not close_entire_position:
            raise ValueError("Параметр quantity обязателен")

        # Добавляем цену (обязательно для лимитных ордеров)
        if order_type == "LIMIT":
            if order_request.price is not None:
                order_params["price"] = str(order_request.price)
            else:
                raise ValueError("Параметр price обязателен для лимитных ордеров")

        # Добавляем клиентский ID ордера
        if order_request.new_client_order_id is not None:
            order_params["newClientOrderId"] = order_request.new_client_order_id

        # Добавляем айсберг количество (только для лимитных ордеров)
        if order_request.iceberg_qty is not None and order_type == "LIMIT":
            order_params["icebergQty"] = str(order_request.iceberg_qty)

        # Проверяем минимальную сумму ордера (5 USDT)
        if order_type == "MARKET" and order_request.quantity is not None:
            # Получаем текущую цену для проверки минимальной суммы
            try:
                ticker = await MarketService.get_ticker_price(order_request.symbol)
                notional_value = order_request.quantity * ticker.price
                if notional_value < 5.0:
                    raise ValueError(
                        f"Минимальная сумма ордера 5 USDT. Текущая сумма: {notional_value:.2f} USDT"
                    )
            except Exception as e:
                logger.warning(f"Не удалось проверить минимальную сумму ордера: {e}")

        # Вызываем соответствующий API в зависимости от режима
        if config.is_demo_mode:
            # Для демо режима используем futures API
            if order_request.reduce_only is not None:
                order_params["reduceOnly"] = str(order_request.reduce_only).lower()
            elif close_entire_position:
                order_params["reduceOnly"] = "true"
            if order_request.close_position is not None:
                order_params["closePosition"] = str(
                    order_request.close_position
                ).lower()
            if order_request.price_protect is not None:
                order_params["priceProtect"] = str(order_request.price_protect).lower()
            result = await call_binance(client.futures_create_order, **order_params)
        else:
            # Для live режима используем spot API
            result = await call_binance(client.create_order, **order_params)

        # Преобразуем ответ в нашу модель с безопасным парсингом
        return CreateOrderResponse(
            order_id=int(result.get("orderId", 0)),
            symbol=result.get("symbol", order_request.symbol.upper()),
            side=result.get("side", order_request.side.upper()),
            type=result.get("type", order_type),
            quantity=float(result.get("origQty", order_request.quantity)),
            price=(
                float(result["price"])
                if result.get("price") not in {None, "0.00000000", ""}
                else None
            ),
            stop_price=None,  # Не используется в spot API
            time_in_force=result.get("timeInForce", order_request.time_in_force),
            status=result.get("status", "UNKNOWN"),
            client_order_id=result.get("clientOrderId", ""),
            transact_time=int(result.get("transactTime", 0)),
            fills=(
                result.get("fills", [])
                if order_request.new_order_resp_type == "FULL"
                else None
            ),
        )

    @staticmethod
    async def create_order_batch(
        orders: List[CreateOrderRequest],
    ) -> List[BatchResult]:
        """Создает несколько ордеров подряд."""
        if not orders:
            raise ValueError("Parameter orders must contain at least one item")

        async def _create(request: CreateOrderRequest) -> BatchResult:
            symbol = request.symbol.upper()
            try:
                response = await OrderService.create_order(request)
                return BatchResult(
                    success=True,
                    symbol=response.symbol,
                    data=response.model_dump(mode="json"),
                )
            except Exception as exc:
                logger.exception("create_order_batch failed", symbol=symbol)
                return BatchResult(success=False, symbol=symbol, error=str(exc))

        results = await asyncio.gather(*[_create(order) for order in orders])
        return list(results)

    @staticmethod
    async def cancel_orders_batch(
        requests: List[CancelOrderRequest],
    ) -> List[BatchResult]:
        """Отменяет несколько ордеров подряд."""
        if not requests:
            raise ValueError("Parameter orders must contain at least one item")

        async def _cancel(request: CancelOrderRequest) -> BatchResult:
            symbol = request.symbol.upper()
            try:
                response = await OrderService.cancel_order(
                    request.symbol,
                    order_id=request.order_id,
                    client_order_id=request.client_order_id,
                )
                return BatchResult(
                    success=True,
                    symbol=symbol,
                    data=response.model_dump(mode="json"),
                )
            except Exception as exc:
                logger.exception("cancel_orders_batch failed", symbol=symbol)
                return BatchResult(success=False, symbol=symbol, error=str(exc))

        results = await asyncio.gather(*[_cancel(req) for req in requests])
        return list(results)


class FuturesService:
    """Сервис для работы с фьючерсами."""

    @staticmethod
    async def get_positions(symbol: Optional[str] = None) -> List[FuturesPosition]:
        """Возвращает фьючерсные позиции."""
        client = get_binance_client()
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        result = await call_binance(client.futures_position_information, **params)
        positions: List[FuturesPosition] = []
        for entry in result:

            def _to_float(value: Any) -> float:
                try:
                    return float(value if value not in {None, ""} else 0.0)
                except (TypeError, ValueError):
                    return 0.0

            update_value = entry.get("updateTime", 0)
            try:
                update_time = int(update_value)
            except (TypeError, ValueError):
                update_time = 0
            positions.append(
                FuturesPosition(
                    symbol=entry.get("symbol", params.get("symbol", "UNKNOWN")),
                    position_amount=_to_float(entry.get("positionAmt")),
                    entry_price=_to_float(entry.get("entryPrice")),
                    mark_price=_to_float(entry.get("markPrice")),
                    unrealized_pnl=_to_float(entry.get("unRealizedProfit")),
                    leverage=_to_float(entry.get("leverage")),
                    margin_type=entry.get("marginType", "UNKNOWN"),
                    isolated_margin=_to_float(entry.get("isolatedMargin")),
                    notional=_to_float(entry.get("notional")),
                    liquidation_price=_to_float(entry.get("liquidationPrice")),
                    update_time=update_time,
                )
            )
        return positions

    @staticmethod
    async def change_leverage(symbol: str, leverage: int) -> FuturesLeverageChange:
        """Изменяет плечо для символа."""
        if leverage < 1:
            raise ValueError("Leverage должен быть положительным")
        client = get_binance_client()
        symbol_upper = symbol.upper()
        result = await call_binance(
            client.futures_change_leverage, symbol=symbol_upper, leverage=leverage
        )
        return FuturesLeverageChange(
            symbol=symbol_upper,
            leverage=int(result.get("leverage", leverage)),
            max_notional_value=float(result.get("maxNotionalValue", 0.0)),
        )

    @staticmethod
    async def change_margin_type(
        symbol: str, margin_type: str
    ) -> FuturesMarginTypeChange:
        """Меняет тип маржи (ISOLATED или CROSSED)."""
        if not margin_type:
            raise ValueError("Параметр margin_type обязателен")
        client = get_binance_client()
        symbol_upper = symbol.upper()
        margin_type_upper = margin_type.upper()
        result = await call_binance(
            client.futures_change_margin_type,
            symbol=symbol_upper,
            marginType=margin_type_upper,
        )
        if isinstance(result, dict):
            code = int(result.get("code", 200))
            message = result.get("msg", "success")
        else:
            code = 200
            message = "success"
        return FuturesMarginTypeChange(
            symbol=symbol_upper,
            margin_type=margin_type_upper,
            code=code,
            message=message,
        )


class MarginService:
    """Сервис для работы с маржинальными операциями."""

    @staticmethod
    def _ensure_margin_available() -> None:
        config = get_config()
        if config.is_demo_mode:
            raise HTTPException(
                status_code=400, detail="Маржинальные операции недоступны в демо режиме"
            )

    @staticmethod
    def _parse_margin_balance(raw: dict) -> MarginBalance:
        return MarginBalance(
            asset=raw.get("asset", "UNKNOWN"),
            free=float(raw.get("free", 0.0)),
            locked=float(raw.get("locked", 0.0)),
            borrowed=float(raw.get("borrowed", 0.0)),
            interest=float(raw.get("interest", 0.0)),
            net_asset=float(raw.get("netAsset", 0.0)),
        )

    @staticmethod
    async def get_margin_account(
        isolated: bool = False, symbol: Optional[str] = None
    ) -> MarginAccount:
        """Возвращает информацию о маржинальном аккаунте."""
        MarginService._ensure_margin_available()
        client = get_binance_client()

        if isolated:
            params: dict[str, Any] = {}
            if symbol:
                params["symbols"] = symbol.upper()
            data = await call_binance(client.get_isolated_margin_account, **params)
            balances: List[MarginBalance] = []
            for asset_entry in data.get("assets", []):
                base_asset = asset_entry.get("baseAsset")
                quote_asset = asset_entry.get("quoteAsset")
                if base_asset:
                    balances.append(MarginService._parse_margin_balance(base_asset))
                if quote_asset:
                    balances.append(MarginService._parse_margin_balance(quote_asset))
            margin_level = float(data.get("marginLevel", 0.0))
            return MarginAccount(
                account_type="ISOLATED",
                margin_level=margin_level,
                total_asset_of_btc=float(data.get("totalAssetOfBtc", 0.0)),
                total_liability_of_btc=float(data.get("totalLiabilityOfBtc", 0.0)),
                total_net_asset_of_btc=float(data.get("totalNetAssetOfBtc", 0.0)),
                trade_enabled=bool(data.get("tradeEnabled", True)),
                borrow_enabled=bool(data.get("borrowEnabled", True)),
                transfer_enabled=bool(data.get("transferEnabled", True)),
                balances=balances,
            )

        data = await call_binance(client.get_margin_account)
        margin_balances: List[MarginBalance] = []
        for entry in data.get("userAssets", []):
            margin_balances.append(MarginService._parse_margin_balance(entry))
        margin_level = float(data.get("marginLevel", 0.0))

        return MarginAccount(
            account_type="CROSS",
            margin_level=margin_level,
            total_asset_of_btc=float(data.get("totalAssetOfBtc", 0.0)),
            total_liability_of_btc=float(data.get("totalLiabilityOfBtc", 0.0)),
            total_net_asset_of_btc=float(data.get("totalNetAssetOfBtc", 0.0)),
            trade_enabled=bool(data.get("tradeEnabled", True)),
            borrow_enabled=bool(data.get("borrowEnabled", True)),
            transfer_enabled=bool(data.get("transferEnabled", True)),
            balances=margin_balances,
        )

    @staticmethod
    async def get_margin_orders(
        symbol: str, limit: int = 10, is_isolated: Optional[bool] = None
    ) -> List[MarginOrder]:
        """Возвращает историю маржинальных ордеров."""
        if not symbol:
            raise ValueError("Параметр symbol обязателен")
        MarginService._ensure_margin_available()
        client = get_binance_client()
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": limit,
        }
        if is_isolated is not None:
            params["isIsolated"] = "TRUE" if is_isolated else "FALSE"
        orders_raw = await call_binance(client.get_margin_orders, **params)
        orders: List[MarginOrder] = []
        for order in orders_raw:
            orders.append(
                MarginOrder(
                    order_id=int(order.get("orderId", 0)),
                    symbol=order.get("symbol", symbol.upper()),
                    is_isolated=order.get("isIsolated", False) in {True, "TRUE"},
                    side=order.get("side", "UNKNOWN"),
                    type=order.get("type", "UNKNOWN"),
                    status=order.get("status", "UNKNOWN"),
                    price=(
                        float(order["price"])
                        if order.get("price") not in {None, "", "0", "0.00000000"}
                        else None
                    ),
                    orig_qty=float(order.get("origQty", 0.0)),
                    executed_qty=float(order.get("executedQty", 0.0)),
                    client_order_id=order.get("clientOrderId", ""),
                    transact_time=int(order.get("updateTime", order.get("time", 0))),
                )
            )
        return orders

    @staticmethod
    async def create_margin_order(
        request: CreateMarginOrderRequest,
    ) -> CreateMarginOrderResponse:
        """Создает маржинальный ордер."""
        MarginService._ensure_margin_available()
        client = get_binance_client()

        params: dict[str, Any] = {
            "symbol": request.symbol.upper(),
            "side": request.side.upper(),
            "type": request.type.upper(),
        }
        if request.quantity is not None:
            params["quantity"] = request.quantity
        if request.quote_order_qty is not None:
            params["quoteOrderQty"] = request.quote_order_qty
        if request.price is not None:
            params["price"] = request.price
        if request.stop_price is not None:
            params["stopPrice"] = request.stop_price
        if request.time_in_force is not None:
            params["timeInForce"] = request.time_in_force
        if request.is_isolated is not None:
            params["isIsolated"] = "TRUE" if request.is_isolated else "FALSE"
        if request.side_effect_type is not None:
            params["sideEffectType"] = request.side_effect_type
        if request.new_client_order_id is not None:
            params["newClientOrderId"] = request.new_client_order_id
        if request.new_order_resp_type is not None:
            params["newOrderRespType"] = request.new_order_resp_type

        result = await call_binance(client.create_margin_order, **params)
        return CreateMarginOrderResponse(
            order_id=int(result.get("orderId", 0)),
            symbol=result.get("symbol", request.symbol.upper()),
            is_isolated=result.get("isIsolated", False) in {True, "TRUE"},
            status=result.get("status", "UNKNOWN"),
            client_order_id=result.get("clientOrderId", ""),
            transact_time=int(result.get("transactTime", 0)),
            price=(
                float(result["price"])
                if result.get("price") not in {None, "", "0", "0.00000000"}
                else None
            ),
            orig_qty=(
                float(result.get("origQty", 0.0))
                if result.get("origQty") is not None
                else None
            ),
            executed_qty=(
                float(result.get("executedQty", 0.0))
                if result.get("executedQty") is not None
                else None
            ),
            cummulative_quote_qty=(
                float(result.get("cummulativeQuoteQty", 0.0))
                if result.get("cummulativeQuoteQty") is not None
                else None
            ),
        )

    @staticmethod
    async def cancel_margin_order(
        request: CancelMarginOrderRequest,
    ) -> CancelMarginOrderResponse:
        """Отменяет маржинальный ордер."""
        MarginService._ensure_margin_available()
        if request.order_id is None and request.client_order_id is None:
            raise ValueError("Необходимо указать order_id или client_order_id")

        client = get_binance_client()
        params: dict[str, Any] = {"symbol": request.symbol.upper()}
        if request.is_isolated is not None:
            params["isIsolated"] = "TRUE" if request.is_isolated else "FALSE"
        if request.order_id is not None:
            params["orderId"] = request.order_id
        if request.client_order_id is not None:
            params["origClientOrderId"] = request.client_order_id
        if request.new_client_order_id is not None:
            params["newClientOrderId"] = request.new_client_order_id

        result = await call_binance(client.cancel_margin_order, **params)
        return CancelMarginOrderResponse(
            order_id=int(result.get("orderId", 0)),
            symbol=result.get("symbol", request.symbol.upper()),
            is_isolated=result.get("isIsolated", False) in {True, "TRUE"},
            status=result.get("status", "CANCELED"),
            client_order_id=result.get("clientOrderId", ""),
            orig_client_order_id=result.get("origClientOrderId", ""),
        )

    @staticmethod
    async def create_margin_order_batch(
        requests: List[CreateMarginOrderRequest],
    ) -> List[BatchResult]:
        """Создает несколько маржинальных ордеров подряд."""
        if not requests:
            raise ValueError("Parameter orders must contain at least one item")

        async def _create(request: CreateMarginOrderRequest) -> BatchResult:
            symbol = request.symbol.upper()
            try:
                response = await MarginService.create_margin_order(request)
                return BatchResult(
                    success=True,
                    symbol=response.symbol,
                    data=response.model_dump(mode="json"),
                )
            except Exception as exc:
                logger.exception("create_margin_order_batch failed", symbol=symbol)
                return BatchResult(success=False, symbol=symbol, error=str(exc))

        results = await asyncio.gather(*[_create(req) for req in requests])
        return list(results)

    @staticmethod
    async def cancel_margin_order_batch(
        requests: List[CancelMarginOrderRequest],
    ) -> List[BatchResult]:
        """Отменяет несколько маржинальных ордеров подряд."""
        if not requests:
            raise ValueError("Parameter orders must contain at least one item")

        async def _cancel(request: CancelMarginOrderRequest) -> BatchResult:
            symbol = request.symbol.upper()
            try:
                response = await MarginService.cancel_margin_order(request)
                return BatchResult(
                    success=True,
                    symbol=response.symbol,
                    data=response.model_dump(mode="json"),
                )
            except Exception as exc:
                logger.exception("cancel_margin_order_batch failed", symbol=symbol)
                return BatchResult(success=False, symbol=symbol, error=str(exc))

        results = await asyncio.gather(*[_cancel(req) for req in requests])
        return list(results)

    @staticmethod
    async def get_margin_trades(
        symbol: str, is_isolated: Optional[bool] = None, limit: int = 10
    ) -> List[MarginTrade]:
        """Возвращает историю маржинальных сделок."""
        if not symbol:
            raise ValueError("Параметр symbol обязателен")
        MarginService._ensure_margin_available()
        client = get_binance_client()
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": limit,
        }
        if is_isolated is not None:
            params["isIsolated"] = "TRUE" if is_isolated else "FALSE"
        trades_raw = await call_binance(client.get_margin_trades, **params)
        trades: List[MarginTrade] = []
        for trade in trades_raw:
            trades.append(
                MarginTrade(
                    trade_id=int(trade.get("id", 0)),
                    symbol=trade.get("symbol", symbol.upper()),
                    order_id=int(trade.get("orderId", 0)),
                    price=float(trade.get("price", 0.0)),
                    quantity=float(trade.get("qty", 0.0)),
                    commission=float(trade.get("commission", 0.0)),
                    commission_asset=trade.get("commissionAsset", ""),
                    is_isolated=trade.get("isIsolated", False) in {True, "TRUE"},
                    is_best_match=bool(trade.get("isBestMatch", False)),
                    time=int(trade.get("time", 0)),
                )
            )
        return trades

    @staticmethod
    async def create_margin_oco_order(params: Dict[str, Any]) -> OCOOrder:
        """Создает маржинальный OCO ордер."""
        MarginService._ensure_margin_available()
        client = get_binance_client()
        payload = params.copy()
        if "symbol" not in payload:
            raise ValueError("Параметр symbol обязателен")
        if "side" not in payload:
            raise ValueError("Параметр side обязателен")
        if payload.get("isIsolated") is not None and isinstance(
            payload["isIsolated"], bool
        ):
            payload["isIsolated"] = "TRUE" if payload["isIsolated"] else "FALSE"
        result = await call_binance(client.create_margin_oco_order, **payload)
        return MarginService._parse_oco_response(result)

    @staticmethod
    async def cancel_margin_oco_order(params: Dict[str, Any]) -> OCOOrder:
        """Отменяет маржинальный OCO ордер."""
        MarginService._ensure_margin_available()
        client = get_binance_client()
        payload = params.copy()
        if "symbol" not in payload:
            raise ValueError("Параметр symbol обязателен")
        if payload.get("isIsolated") is not None and isinstance(
            payload["isIsolated"], bool
        ):
            payload["isIsolated"] = "TRUE" if payload["isIsolated"] else "FALSE"
        result = await call_binance(client.cancel_margin_oco_order, **payload)
        return MarginService._parse_oco_response(result)

    @staticmethod
    async def get_open_margin_oco_orders(
        symbol: Optional[str] = None, is_isolated: Optional[bool] = None
    ) -> List[OCOOrder]:
        """Получает открытые маржинальные OCO ордера."""
        MarginService._ensure_margin_available()
        client = get_binance_client()
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        if is_isolated is not None:
            params["isIsolated"] = "TRUE" if is_isolated else "FALSE"
        result = await call_binance(client.get_open_margin_oco_orders, **params)
        return [MarginService._parse_oco_response(entry) for entry in result]

    @staticmethod
    async def get_margin_oco_order(params: Dict[str, Any]) -> OCOOrder:
        """Получает маржинальный OCO ордер по идентификаторам."""
        MarginService._ensure_margin_available()
        client = get_binance_client()
        payload = params.copy()
        if payload.get("isIsolated") is not None and isinstance(
            payload["isIsolated"], bool
        ):
            payload["isIsolated"] = "TRUE" if payload["isIsolated"] else "FALSE"
        result = await call_binance(client.get_margin_oco_order, **payload)
        return MarginService._parse_oco_response(result)

    @staticmethod
    def _parse_oco_response(data: dict) -> OCOOrder:
        legs: List[OCOLeg] = []
        reports = data.get("orderReports") or data.get("orders") or []
        for entry in reports:
            legs.append(
                OCOLeg(
                    symbol=entry.get("symbol", data.get("symbol", "")),
                    order_id=int(entry.get("orderId", entry.get("orderID", 0))),
                    client_order_id=entry.get("clientOrderId", ""),
                    price=(
                        float(entry.get("price", 0.0))
                        if entry.get("price") is not None
                        else None
                    ),
                    orig_qty=(
                        float(entry.get("origQty", 0.0))
                        if entry.get("origQty") is not None
                        else None
                    ),
                    executed_qty=(
                        float(entry.get("executedQty", 0.0))
                        if entry.get("executedQty") is not None
                        else None
                    ),
                    status=entry.get("status"),
                    side=entry.get("side"),
                    type=entry.get("type"),
                    stop_price=(
                        float(entry.get("stopPrice", 0.0))
                        if entry.get("stopPrice") not in {None, "", "0", "0.00000000"}
                        else None
                    ),
                )
            )
        return OCOOrder(
            order_list_id=int(data.get("orderListId", 0)),
            symbol=data.get("symbol", ""),
            status=data.get("status", data.get("listOrderStatus", "UNKNOWN")),
            list_status_type=data.get("listStatusType", "UNKNOWN"),
            list_order_status=data.get(
                "listOrderStatus", data.get("status", "UNKNOWN")
            ),
            list_client_order_id=data.get("listClientOrderId", ""),
            transact_time=int(data.get("transactionTime", data.get("transactTime", 0))),
            is_isolated=data.get("isIsolated"),
            legs=legs,
        )


class OCOService:
    """Сервис для работы с OCO ордерами (spot)."""

    @staticmethod
    def _ensure_spot_mode() -> None:
        config = get_config()
        if config.is_demo_mode:
            raise HTTPException(
                status_code=400, detail="OCO ордера доступны только в spot режиме"
            )

    @staticmethod
    async def create_oco_order(params: Dict[str, Any]) -> OCOOrder:
        """Создает spot OCO ордер."""
        OCOService._ensure_spot_mode()
        client = get_binance_client()
        if "symbol" not in params or "side" not in params:
            raise ValueError("Параметры symbol и side обязательны")
        result = await call_binance(client.create_oco_order, **params)
        return MarginService._parse_oco_response(result)

    @staticmethod
    async def cancel_oco_order(params: Dict[str, Any]) -> OCOOrder:
        """Отменяет spot OCO ордер."""
        OCOService._ensure_spot_mode()
        client = get_binance_client()
        if "symbol" not in params:
            raise ValueError("Параметр symbol обязателен")
        result = await call_binance(client._delete, "orderList", True, data=params)
        return MarginService._parse_oco_response(result)

    @staticmethod
    async def get_open_oco_orders() -> List[OCOOrder]:
        """Возвращает текущие OCO ордера."""
        OCOService._ensure_spot_mode()
        client = get_binance_client()
        result = await call_binance(client.get_open_oco_orders)
        return [MarginService._parse_oco_response(entry) for entry in result]


class ExchangeService:
    """Сервис для работы с биржей."""

    @staticmethod
    async def get_exchange_info(symbol: Optional[str] = None) -> ExchangeInfo:
        """Получает информацию о бирже."""
        client = get_binance_client()
        config = get_config()

        if symbol:
            if config.is_demo_mode:
                # В демо режиме для фьючерсов используем futures_exchange_info
                info = await call_binance(client.futures_exchange_info)
                symbols_info = info.get("symbols", [])
                symbol_info = next(
                    (s for s in symbols_info if s.get("symbol") == symbol.upper()), None
                )
                symbols_payload = [symbol_info.get("symbol", symbol.upper())] if symbol_info else [symbol.upper()]
            else:
                # В live режиме используем обычный get_exchange_info
                info = await call_binance(
                    client.get_exchange_info, symbols=[symbol.upper()]
                )
                symbols_payload = [
                    info.get("symbols", [{}])[0].get("symbol", symbol.upper())
                ]
        else:
            if config.is_demo_mode:
                # В демо режиме для фьючерсов используем futures_exchange_info
                info = await call_binance(client.futures_exchange_info)
                symbols_payload = [
                    entry.get("symbol") for entry in info.get("symbols", [])[:10]
                ]
            else:
                # В live режиме используем обычный get_exchange_info
                info = await call_binance(client.get_exchange_info)
                symbols_payload = [
                    entry.get("symbol") for entry in info.get("symbols", [])[:10]
                ]

        return ExchangeInfo(
            timezone=info.get("timezone", "UTC"),
            server_time=info.get("serverTime", 0),
            rate_limits=info.get("rateLimits", []),
            symbols_count=len(info.get("symbols", [])),
            symbols=[s for s in symbols_payload if s],
        )

    @staticmethod
    async def get_symbol_filters(symbol: str) -> Dict[str, Dict[str, Any]]:
        """Возвращает фильтры биржи для конкретного символа."""
        client = get_binance_client()
        config = get_config()
        symbol_upper = symbol.upper()

        if config.is_demo_mode:
            # В демо режиме для фьючерсов используем futures_exchange_info без параметра symbol
            try:
                exchange_info = await call_binance(client.futures_exchange_info)
                symbols_info = exchange_info.get("symbols", [])
                symbol_info = next(
                    (s for s in symbols_info if s.get("symbol") == symbol_upper), None
                )
                if not symbol_info:
                    logger.warning(f"Символ {symbol_upper} не найден в демо режиме")
                    return {}
            except Exception as e:
                logger.warning(f"Ошибка получения информации о символе в демо режиме: {e}")
                return {}
        else:
            symbol_info = await call_binance(
                client.get_symbol_info, symbol=symbol_upper
            )

        filters = symbol_info.get("filters", []) if symbol_info else []
        return {
            entry.get("filterType", ""): entry
            for entry in filters
            if entry.get("filterType")
        }

    @staticmethod
    async def get_server_time() -> ServerTime:
        """Получает время сервера."""
        client = get_binance_client()
        response = await call_binance(client.get_server_time)
        server_timestamp = response.get("serverTime", 0)

        return ServerTime(
            server_time=server_timestamp, epoch_seconds=server_timestamp / 1000
        )

    @staticmethod
    async def get_available_pairs(
        filters: Optional[Dict[str, str]] = None,
    ) -> List[AvailablePair]:
        """Получает доступные торговые пары с фильтрами."""
        client = get_binance_client()
        config = get_config()

        if config.is_demo_mode:
            # Для демо режима используем get_exchange_info
            info = await call_binance(client.get_exchange_info)
        else:
            info = await call_binance(client.get_exchange_info)

        pairs = []
        for symbol_info in info.get("symbols", []):
            symbol = symbol_info.get("symbol", "")
            status = symbol_info.get("status", "")
            base_asset = symbol_info.get("baseAsset", "")
            quote_asset = symbol_info.get("quoteAsset", "")

            # Применяем фильтры
            if filters:
                if "status" in filters and status != filters["status"]:
                    continue
                if "quoteAsset" in filters and quote_asset != filters["quoteAsset"]:
                    continue

            # Извлекаем фильтры символа
            symbol_filters = {}
            for filter_info in symbol_info.get("filters", []):
                filter_type = filter_info.get("filterType", "")
                symbol_filters[filter_type] = filter_info

            pairs.append(
                AvailablePair(
                    symbol=symbol,
                    status=status,
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    filters=symbol_filters,
                )
            )

        return pairs

    @staticmethod
    async def check_trading_limits(symbols: List[str]) -> List[TradingLimitsInfo]:
        """Проверяет лимиты и доступность для списка символов."""
        if len(symbols) > 20:
            raise ValueError("Максимум 20 символов в batch-операции")

        async def check_symbol_limits(symbol: str) -> TradingLimitsInfo:
            try:
                # Получаем позицию
                positions = await FuturesService.get_positions(symbol)
                position = positions[0] if positions else None

                # Получаем информацию о символе
                exchange_info = await ExchangeService.get_exchange_info(symbol)
                for sym in exchange_info.symbols:
                    if sym == symbol:
                        break

                # Получаем доступную маржу
                account = await call_binance(get_binance_client().futures_account)
                usdt_asset: Dict[str, Any] = next(
                    (
                        asset
                        for asset in account.get("assets", [])
                        if asset.get("asset") == "USDT"
                    ),
                    {},
                )
                margin_available = float(usdt_asset.get("availableBalance", 0.0))

                # Определяем доступность торговли
                available = position is None or abs(position.position_amount) < 1e-8

                limits = {
                    "max_position_size": 0,  # Будет заполнено из filters если нужно
                    "min_notional": 0,
                    "tick_size": 0,
                }

                return TradingLimitsInfo(
                    symbol=symbol,
                    available=available,
                    position=position,
                    margin_available=margin_available,
                    limits=limits,
                )
            except Exception as e:
                logger.warning(f"Ошибка проверки лимитов {symbol}: {e}")
                return TradingLimitsInfo(
                    symbol=symbol,
                    available=False,
                    position=None,
                    margin_available=0.0,
                    limits={},
                )

        results = await asyncio.gather(
            *[check_symbol_limits(symbol) for symbol in symbols], return_exceptions=True
        )

        return [result for result in results if isinstance(result, TradingLimitsInfo)]


class PortfolioService:
    """Сервис для работы с портфелем."""

    @staticmethod
    async def get_portfolio_summary() -> PortfolioSummary:
        """Получает сводку портфеля."""
        config = get_config()
        client = get_binance_client()

        # Получаем баланс аккаунта
        balances = []
        try:
            balances = await AccountService.get_account_balance()
        except Exception as e:
            logger.warning(f"Баланс недоступен: {e}")
            # Используем демо-баланс для демонстрации
            if config.is_demo_mode:
                balances = [
                    BalanceEntry(
                        asset="USDT", free=4999.92, locked=14.18, total=5014.1
                    ),
                    BalanceEntry(asset="USDC", free=0.0, locked=0.0, total=0.0),
                    BalanceEntry(asset="BTC", free=0.01, locked=0.0, total=0.01),
                ]

        def build_spot_account_summary() -> AccountSummary:
            total_balance_usdt = 0.0
            free_usdt = 0.0
            invested_usdt = 0.0

            for balance in balances:
                if balance.asset in {"USDT", "USDC", "BUSD"}:
                    total_balance_usdt += balance.total
                    free_usdt += balance.free
                    invested_usdt += balance.locked

            investment_percentage = (
                (invested_usdt / total_balance_usdt * 100)
                if total_balance_usdt > 0
                else 0.0
            )

            return AccountSummary(
                total_balance_usdt=total_balance_usdt,
                free_usdt=free_usdt,
                invested_usdt=invested_usdt,
                investment_percentage=investment_percentage,
            )

        async def build_spot_positions() -> List[Position]:
            positions: List[Position] = []
            quote_assets = ("USDT", "BUSD", "USDC", "BTC", "BNB", "ETH")

            for balance in balances:
                asset = balance.asset.upper()
                if asset in {"USDT", "USDC", "BUSD"}:
                    continue
                if balance.total <= 0:
                    continue

                trades: List[Trade] = []
                symbol: Optional[str] = None

                for quote_asset in quote_assets:
                    candidate_symbol = f"{asset}{quote_asset}"
                    try:
                        candidate_trades = await OrderService.get_trade_history(
                            candidate_symbol, limit=1000
                        )
                    except HTTPException as exc:
                        # Игнорируем ошибки с несуществующим символом
                        detail = str(exc.detail)
                        if "Invalid symbol" in detail or "No such symbol" in detail:
                            continue
                        logger.warning(
                            "Не удалось получить сделки %s: %s", candidate_symbol, exc
                        )
                        continue

                    if candidate_trades:
                        trades = sorted(candidate_trades, key=lambda trade: trade.time)
                        symbol = candidate_symbol
                        break

                if not trades or symbol is None:
                    logger.debug("Сделки для актива %s не найдены", asset)
                    continue

                remaining_qty = 0.0
                total_cost = 0.0

                for trade in trades:
                    if trade.side == "BUY":
                        remaining_qty += trade.quantity
                        total_cost += trade.quantity * trade.price
                    else:
                        if remaining_qty <= 0:
                            continue
                        avg_cost = (
                            total_cost / remaining_qty if remaining_qty > 0 else 0.0
                        )
                        sell_qty = min(trade.quantity, remaining_qty)
                        remaining_qty -= sell_qty
                        total_cost -= avg_cost * sell_qty

                if remaining_qty <= 0:
                    logger.debug(
                        "Открытая позиция по %s не найдена (remaining=0)", symbol
                    )
                    continue

                quantity = balance.total
                if quantity <= 0:
                    continue

                entry_price = (total_cost / remaining_qty) if remaining_qty > 0 else 0.0

                try:
                    ticker = await MarketService.get_ticker_price(symbol)
                    ticker_24h = await MarketService.get_24hr_ticker(symbol)
                except Exception as exc:
                    logger.warning(
                        "Не удалось получить рыночные данные %s: %s", symbol, exc
                    )
                    continue

                current_price = ticker.price
                market_value = quantity * current_price
                cost_basis = quantity * entry_price
                pnl_usdt = market_value - cost_basis
                pnl_percent = (pnl_usdt / cost_basis * 100) if cost_basis > 0 else 0.0

                positions.append(
                    Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=entry_price,
                        current_price=current_price,
                        market_value=market_value,
                        pnl_usdt=pnl_usdt,
                        pnl_percent=pnl_percent,
                        daily_change=ticker_24h.price_change_percent,
                        status="active",
                    )
                )

            return positions

        async def build_futures_summary() -> Tuple[AccountSummary, List[Position]]:
            account = await call_binance(client.futures_account)

            usdt_asset: Dict[str, Any] = next(
                (
                    asset
                    for asset in account.get("assets", [])
                    if asset.get("asset") == "USDT"
                ),
                {},
            )
            total_balance_usdt = float(usdt_asset.get("walletBalance", 0.0))
            free_usdt = float(usdt_asset.get("availableBalance", 0.0))
            invested_usdt = total_balance_usdt - free_usdt
            investment_percentage = (
                (invested_usdt / total_balance_usdt * 100)
                if total_balance_usdt
                else 0.0
            )

            positions: List[Position] = []
            for position in account.get("positions", []):
                position_amt = float(position.get("positionAmt", 0.0))
                if abs(position_amt) < 1e-8:
                    continue

                entry_price = float(position.get("entryPrice", 0.0))
                mark_price = float(position.get("markPrice", entry_price))
                unrealized_pnl = float(position.get("unRealizedProfit", 0.0))

                notional = abs(position_amt) * entry_price
                pnl_percent = (unrealized_pnl / notional * 100) if notional else 0.0

                daily_change = 0.0
                try:
                    ticker = await call_binance(
                        client.futures_ticker_24hr, symbol=position.get("symbol")
                    )
                    daily_change = float(ticker.get("priceChangePercent", 0.0))
                except Exception as exc:
                    logger.debug(
                        "Не удалось получить futures 24h для %s: %s",
                        position.get("symbol"),
                        exc,
                    )

                positions.append(
                    Position(
                        symbol=position.get("symbol", "UNKNOWN"),
                        quantity=position_amt,
                        entry_price=entry_price,
                        current_price=mark_price,
                        market_value=position_amt * mark_price,
                        pnl_usdt=unrealized_pnl,
                        pnl_percent=pnl_percent,
                        daily_change=daily_change,
                        status="long" if position_amt >= 0 else "short",
                    )
                )

            account_summary = AccountSummary(
                total_balance_usdt=total_balance_usdt,
                free_usdt=free_usdt,
                invested_usdt=invested_usdt,
                investment_percentage=investment_percentage,
            )

            return account_summary, positions

        if config.is_demo_mode:
            account_summary, positions = await build_futures_summary()
        else:
            account_summary = build_spot_account_summary()
            positions = await build_spot_positions()

        # Рассчитываем метрики портфеля
        total_pnl_usdt = sum(pos.pnl_usdt for pos in positions)
        total_invested = sum(abs(pos.quantity) * pos.entry_price for pos in positions)
        total_pnl_percent = (
            (total_pnl_usdt / total_invested * 100) if total_invested > 0 else 0.0
        )

        best_performer = (
            max(positions, key=lambda p: p.pnl_percent).symbol if positions else "N/A"
        )
        worst_performer = (
            min(positions, key=lambda p: p.pnl_percent).symbol if positions else "N/A"
        )

        # Оценка риска на основе диверсификации
        diversification_score = len(positions)
        if diversification_score >= 5:
            risk_score = "low"
        elif diversification_score >= 3:
            risk_score = "medium"
        else:
            risk_score = "high"

        portfolio_metrics = PortfolioMetrics(
            total_pnl_usdt=total_pnl_usdt,
            total_pnl_percent=total_pnl_percent,
            best_performer=best_performer,
            worst_performer=worst_performer,
            risk_score=risk_score,
            diversification_score=diversification_score,
        )

        # Генерируем рекомендации
        recommendations = []
        if not positions:
            recommendations.append("Портфель пуст - рассмотрите открытие позиций")
        else:
            if total_pnl_percent > 10:
                recommendations.append(
                    "Портфель показывает хорошую прибыль - рассмотрите фиксацию части прибыли"
                )
            elif total_pnl_percent < -10:
                recommendations.append("Портфель в убытке - рассмотрите стоп-лоссы")

            if diversification_score < 3:
                recommendations.append(
                    "Низкая диверсификация - рассмотрите добавление новых активов"
                )

            best_pos = next((p for p in positions if p.symbol == best_performer), None)
            if best_pos and best_pos.daily_change > 20:
                recommendations.append(
                    f"{best_performer} показывает сильный импульс +{best_pos.daily_change:.1f}%"
                )

            recommendations.append("Регулярно отслеживайте изменения в портфеле")

        return PortfolioSummary(
            account_summary=account_summary,
            positions=positions,
            portfolio_metrics=portfolio_metrics,
            recommendations=recommendations,
        )

    @staticmethod
    async def get_simple_balance() -> dict:
        """Получает простую сводку баланса (работает с ограниченными правами API)."""
        try:
            # Получаем только доступные данные
            balances = await AccountService.get_account_balance()

            # Рассчитываем общий баланс в USDT
            total_usdt = 0.0
            balance_details = []

            for balance in balances:
                if balance.total > 0:
                    # Получаем цену в USDT для каждого актива
                    try:
                        if balance.asset == "USDT":
                            usdt_value = balance.total
                        elif balance.asset == "USDC":
                            usdt_value = balance.total  # USDC ≈ USDT
                        else:
                            # Получаем цену в USDT
                            ticker = await MarketService.get_ticker_price(
                                f"{balance.asset}USDT"
                            )
                            usdt_value = balance.total * ticker.price

                        total_usdt += usdt_value
                        balance_details.append(
                            {
                                "asset": balance.asset,
                                "free": balance.free,
                                "locked": balance.locked,
                                "total": balance.total,
                                "usdt_value": usdt_value,
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Не удалось получить цену для {balance.asset}: {e}"
                        )
                        continue

            return {
                "total_balance_usdt": total_usdt,
                "balances": balance_details,
                "status": "success",
                "message": "Баланс успешно получен",
            }

        except Exception as e:
            logger.error(f"Ошибка при получении баланса: {e}")
            return {
                "total_balance_usdt": 0.0,
                "balances": [],
                "status": "error",
                "message": f"Ошибка получения баланса: {str(e)}",
            }


class BatchService:
    """Сервис для батчевых операций."""

    @staticmethod
    async def get_futures_positions_batch(symbols: List[str]) -> List[BatchResult]:
        """Батчевое получение фьючерсных позиций."""
        if len(symbols) > 20:
            raise ValueError("Максимум 20 символов в batch-операции")

        async def get_position_for_symbol(symbol: str) -> BatchResult:
            try:
                positions = await FuturesService.get_positions(symbol)
                position = positions[0] if positions else None
                return BatchResult(
                    success=True,
                    symbol=symbol,
                    data=position.dict() if position else None,
                )
            except Exception as e:
                logger.warning(f"Ошибка получения позиции {symbol}: {e}")
                return BatchResult(success=False, symbol=symbol, error=str(e))

        results = await asyncio.gather(
            *[get_position_for_symbol(symbol) for symbol in symbols],
            return_exceptions=True,
        )

        return [result for result in results if isinstance(result, BatchResult)]

    @staticmethod
    async def get_portfolio_overview() -> PortfolioOverview:
        """Комбинированный обзор портфолио."""
        config = get_config()

        # Получаем баланс и позиции параллельно
        balance_task = AccountService.get_account_balance()
        positions_task = (
            FuturesService.get_positions()
            if config.is_demo_mode
            else asyncio.create_task(asyncio.sleep(0, result=[]))
        )

        balance, positions = await asyncio.gather(balance_task, positions_task)

        # Рассчитываем статистику
        total_pnl = sum(pos.unrealized_pnl for pos in positions)
        total_notional = sum(
            abs(pos.position_amount) * pos.mark_price for pos in positions
        )

        statistics = {
            "total_positions": len(positions),
            "total_notional": total_notional,
            "avg_leverage": (
                sum(pos.leverage for pos in positions) / len(positions)
                if positions
                else 0
            ),
            "long_positions": len([p for p in positions if p.position_amount > 0]),
            "short_positions": len([p for p in positions if p.position_amount < 0]),
        }

        return PortfolioOverview(
            balance=balance,
            active_positions=positions,
            total_pnl=total_pnl,
            statistics=statistics,
        )

    @staticmethod
    async def get_tickers_batch(symbols: List[str]) -> List[BatchResult]:
        """Батчевое получение цен и 24h данных."""
        if len(symbols) > 20:
            raise ValueError("Максимум 20 символов в batch-операции")

        async def get_ticker_for_symbol(symbol: str) -> BatchResult:
            try:
                price_task = MarketService.get_ticker_price(symbol)
                ticker_24h_task = MarketService.get_24hr_ticker(symbol)

                price, ticker_24h = await asyncio.gather(price_task, ticker_24h_task)

                ticker_item = TickerBatchItem(
                    symbol=symbol, price=price.price, ticker_24h=ticker_24h
                )

                return BatchResult(success=True, symbol=symbol, data=ticker_item.dict())
            except Exception as e:
                logger.warning(f"Ошибка получения тикера {symbol}: {e}")
                return BatchResult(success=False, symbol=symbol, error=str(e))

        results = await asyncio.gather(
            *[get_ticker_for_symbol(symbol) for symbol in symbols],
            return_exceptions=True,
        )

        return [result for result in results if isinstance(result, BatchResult)]

    @staticmethod
    async def change_leverage_batch(
        symbol_leverage_map: Dict[str, int],
    ) -> List[BatchResult]:
        """Массовое изменение плеча."""
        if len(symbol_leverage_map) > 20:
            raise ValueError("Максимум 20 символов в batch-операции")

        async def change_leverage_for_symbol(symbol: str, leverage: int) -> BatchResult:
            try:
                result = await FuturesService.change_leverage(symbol, leverage)
                return BatchResult(success=True, symbol=symbol, data=result.dict())
            except Exception as e:
                logger.warning(f"Ошибка изменения плеча {symbol}: {e}")
                return BatchResult(success=False, symbol=symbol, error=str(e))

        results = await asyncio.gather(
            *[
                change_leverage_for_symbol(symbol, leverage)
                for symbol, leverage in symbol_leverage_map.items()
            ],
            return_exceptions=True,
        )

        return [result for result in results if isinstance(result, BatchResult)]

    @staticmethod
    async def create_positions_batch(
        positions: List[CreatePositionRequest],
    ) -> List[BatchResult]:
        """Массовое открытие позиций с автоматической установкой leverage."""
        if len(positions) > 20:
            raise ValueError("Максимум 20 позиций в batch-операции")

        async def create_position_for_request(
            pos_req: CreatePositionRequest,
        ) -> BatchResult:
            try:
                # Сначала устанавливаем leverage
                leverage_result = await FuturesService.change_leverage(
                    pos_req.symbol, pos_req.leverage
                )

                # Затем создаем MARKET ордер
                order_request = CreateOrderRequest(
                    symbol=pos_req.symbol,
                    side=pos_req.side,
                    type="MARKET",
                    quantity=pos_req.quantity,
                    new_order_resp_type="FULL",
                )

                order_result = await OrderService.create_order(order_request)

                return BatchResult(
                    success=True,
                    symbol=pos_req.symbol,
                    data={
                        "leverage_change": leverage_result.dict(),
                        "order": order_result.dict(),
                    },
                )
            except Exception as e:
                logger.warning(f"Ошибка создания позиции {pos_req.symbol}: {e}")
                return BatchResult(success=False, symbol=pos_req.symbol, error=str(e))

        results = await asyncio.gather(
            *[create_position_for_request(pos) for pos in positions],
            return_exceptions=True,
        )

        return [result for result in results if isinstance(result, BatchResult)]

    @staticmethod
    async def close_positions_batch(
        symbols: List[str], percentage: float = 100.0
    ) -> List[BatchResult]:
        """Массовое закрытие позиций с автоопределением направления."""
        if len(symbols) > 20:
            raise ValueError("Максимум 20 символов в batch-операции")

        config = get_config()

        async def close_position_for_symbol(symbol: str) -> BatchResult:
            try:
                # Получаем текущую позицию
                positions = await FuturesService.get_positions(symbol)
                if not positions:
                    return BatchResult(
                        success=False, symbol=symbol, error="Позиция не найдена"
                    )

                position = positions[0]
                if abs(position.position_amount) < 1e-8:
                    return BatchResult(
                        success=False, symbol=symbol, error="Позиция уже закрыта"
                    )

                # Получаем фильтры только если не в демо режиме
                filters = {}
                if not config.is_demo_mode:
                    filters = await ExchangeService.get_symbol_filters(symbol)

                # Определяем направление закрытия
                close_side = "SELL" if position.position_amount > 0 else "BUY"
                position_amount_dec = Decimal(str(abs(position.position_amount)))
                raw_quantity = (
                    position_amount_dec * Decimal(str(percentage))
                ) / Decimal("100")
                should_close_entire = (
                    percentage >= 99.9
                    or raw_quantity >= position_amount_dec * Decimal("0.999")
                )

                quantity_float: Optional[float] = None
                if not should_close_entire:
                    if config.is_demo_mode:
                        # В демо режиме используем простое округление до 3 знаков для BTC
                        quantity_float = round(float(raw_quantity), 3)
                    else:
                        market_filter = filters.get("MARKET_LOT_SIZE") or filters.get(
                            "LOT_SIZE"
                        )
                        step_size = BatchService._decimal_from_filter(
                            market_filter, "stepSize"
                        )
                        min_qty = BatchService._decimal_from_filter(market_filter, "minQty")
                        quantity = BatchService._round_to_step(
                            raw_quantity, step_size, ROUND_DOWN
                        )
                        if min_qty and quantity < min_qty:
                            quantity = BatchService._round_to_step(
                                min_qty, step_size, ROUND_UP
                            )
                        if quantity > position_amount_dec:
                            should_close_entire = True
                        elif quantity <= Decimal("0"):
                            return BatchResult(
                                success=False,
                                symbol=symbol,
                                error="Рассчитанное количество слишком мало для закрытия",
                            )
                        else:
                            quantity_float = float(quantity)

                # Создаем MARKET ордер для закрытия
                if should_close_entire:
                    # Для полного закрытия не указываем quantity
                    if config.is_demo_mode:
                        # В демо режиме используем простой MARKET ордер с количеством
                        quantity_float = round(float(position_amount_dec), 3)
                        order_request = CreateOrderRequest(
                            symbol=symbol,
                            side=close_side,
                            type="MARKET",
                            quantity=quantity_float,
                            new_order_resp_type="FULL",
                        )
                    else:
                        # В live режиме используем close_position
                        order_request = CreateOrderRequest(
                            symbol=symbol,
                            side=close_side,
                            type="MARKET",
                            new_order_resp_type="FULL",
                            reduce_only=True,
                            close_position=True,
                            price_protect=False,
                        )
                else:
                    # Для частичного закрытия указываем quantity
                    order_request = CreateOrderRequest(
                        symbol=symbol,
                        side=close_side,
                        type="MARKET",
                        quantity=quantity_float,
                        new_order_resp_type="FULL",
                        reduce_only=not config.is_demo_mode,  # Только в live режиме
                        close_position=False,
                        price_protect=False,
                    )

                fallback_used = False
                try:
                    order_result = await OrderService.create_order(order_request)
                except HTTPException as exc:
                    if BatchService._should_try_price_fallback(exc):
                        order_result = await BatchService._close_with_limit_fallback(
                            symbol,
                            close_side,
                            position,
                            percentage,
                            filters,
                            config,
                        )
                        fallback_used = True
                    else:
                        raise

                return BatchResult(
                    success=True,
                    symbol=symbol,
                    data={
                        "original_position": position.dict(),
                        "close_order": order_result.dict(),
                        "close_side": close_side,
                        "close_quantity": (
                            quantity_float
                            if quantity_float is not None
                            else float(position_amount_dec)
                        ),
                        **({"fallback": "limit"} if fallback_used else {}),
                    },
                )
            except HTTPException as exc:
                return BatchResult(
                    success=False,
                    symbol=symbol,
                    error=BatchService._format_http_error(exc),
                )
            except Exception as e:
                logger.warning(f"Ошибка закрытия позиции {symbol}: {e}")
                return BatchResult(success=False, symbol=symbol, error=str(e))

        results = await asyncio.gather(
            *[close_position_for_symbol(symbol) for symbol in symbols],
            return_exceptions=True,
        )

        return [result for result in results if isinstance(result, BatchResult)]

    @staticmethod
    def _decimal_from_filter(
        filter_data: Optional[Dict[str, Any]], key: str
    ) -> Optional[Decimal]:
        if not filter_data:
            return None
        value = filter_data.get(key)
        if value in (None, "", "0", "0.0"):
            return None
        try:
            return Decimal(str(value))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _round_to_step(
        value: Decimal, step: Optional[Decimal], rounding: str
    ) -> Decimal:
        if step is None or step == 0:
            return value
        if step <= 0:
            return value
        return (value / step).to_integral_value(rounding=rounding) * step

    @staticmethod
    def _should_try_price_fallback(exc: HTTPException) -> bool:
        detail: Union[str, Dict[str, Any]] = exc.detail
        if isinstance(detail, dict):
            message = str(detail.get("message", "")).upper()
            code = str(detail.get("code", "")).strip()
        else:
            message = str(detail).upper()
            code = ""
        return (
            "PERCENT_PRICE" in message or "PRICE_FILTER" in message or code in {"-1013"}
        )

    @staticmethod
    def _format_http_error(exc: HTTPException) -> str:
        detail: Union[str, Dict[str, Any]] = exc.detail
        if isinstance(detail, dict):
            message = detail.get("message") or detail.get("detail") or str(detail)
            code = detail.get("code")
            if code is not None:
                return f"{message} (code={code})"
            return str(message)
        else:
            return str(detail)

    @staticmethod
    async def _close_with_limit_fallback(
        symbol: str,
        close_side: str,
        position: FuturesPosition,
        percentage: float,
        filters: Dict[str, Any],
        config: Config,
    ) -> CreateOrderResponse:
        price_info = await MarketService.get_avg_price(symbol)
        price = Decimal(str(price_info.price))
        adjustment = Decimal("0.002")
        rounding = ROUND_DOWN if close_side == "SELL" else ROUND_UP
        tick_size = BatchService._decimal_from_filter(
            filters.get("PRICE_FILTER"), "tickSize"
        )
        price_multiplier = (
            Decimal("1") - adjustment
            if close_side == "SELL"
            else Decimal("1") + adjustment
        )
        price = BatchService._round_to_step(
            price * price_multiplier, tick_size, rounding
        )
        if price <= 0:
            raise ValueError("Не удалось определить цену для лимитного ордера")

        position_amount_dec = Decimal(str(abs(position.position_amount)))
        base_quantity = (position_amount_dec * Decimal(str(percentage))) / Decimal(
            "100"
        )

        lot_filter = filters.get("LOT_SIZE") or filters.get("MARKET_LOT_SIZE")
        step_size = BatchService._decimal_from_filter(lot_filter, "stepSize")
        min_qty = BatchService._decimal_from_filter(lot_filter, "minQty")

        quantity = BatchService._round_to_step(base_quantity, step_size, ROUND_DOWN)
        if min_qty and quantity < min_qty:
            quantity = BatchService._round_to_step(min_qty, step_size, ROUND_UP)
        if quantity > position_amount_dec:
            quantity = position_amount_dec
        if quantity <= Decimal("0"):
            raise ValueError("Количество для лимитного закрытия равно нулю")

        min_notional_filter = filters.get("MIN_NOTIONAL")
        min_notional = BatchService._decimal_from_filter(
            min_notional_filter, "notional"
        ) or BatchService._decimal_from_filter(min_notional_filter, "minNotional")
        if min_notional and quantity * price < min_notional:
            quantity = BatchService._round_to_step(
                (min_notional / price), step_size, ROUND_UP
            )
        if quantity <= Decimal("0"):
            raise ValueError(
                "Количество после корректировки минимального нотиона равно нулю"
            )

        order_request = CreateOrderRequest(
            symbol=symbol,
            side=close_side,
            type="LIMIT",
            quantity=float(quantity),
            price=float(price),
            time_in_force="GTC",
            new_order_resp_type="FULL",
            reduce_only=not config.is_demo_mode,  # Только в live режиме
            price_protect=False,
        )
        return await OrderService.create_order(order_request)


class RiskManagementService:
    """Сервис для управления рисками."""

    @staticmethod
    async def portfolio_safety_check(
        symbols: List[str],
        safety_rules: Optional[SafetyRule] = None,
        auto_close_unsafe: bool = False,
    ) -> List[SafetyCheckResult]:
        """Проверка безопасности позиций портфеля."""
        if len(symbols) > 20:
            raise ValueError("Максимум 20 символов в batch-операции")

        if safety_rules is None:
            safety_rules = SafetyRule()

        async def check_symbol_safety(symbol: str) -> SafetyCheckResult:
            try:
                # Получаем позицию
                positions = await FuturesService.get_positions(symbol)
                position = positions[0] if positions else None

                if not position or abs(position.position_amount) < 1e-8:
                    return SafetyCheckResult(
                        symbol=symbol,
                        is_safe=True,
                        violations=[],
                        recommendation="Нет активной позиции",
                    )

                violations = []

                # Получаем технические индикаторы (упрощенная версия)
                try:
                    # Получаем свечи для расчета RSI и ADX
                    klines = await MarketService.get_klines(symbol, "1h", 14)
                    if len(klines.recent_klines) >= 14:
                        closes = [kline.close for kline in klines.recent_klines]

                        # Упрощенный расчет RSI
                        rsi = RiskManagementService._calculate_rsi(closes)

                        # Проверка RSI правил
                        if position.position_amount > 0:  # Long позиция
                            if rsi > safety_rules.min_rsi_long:
                                violations.append(
                                    f"RSI слишком высокий: {rsi:.1f} > {safety_rules.min_rsi_long}"
                                )
                        else:  # Short позиция
                            if rsi < safety_rules.max_rsi_short:
                                violations.append(
                                    f"RSI слишком низкий: {rsi:.1f} < {safety_rules.max_rsi_short}"
                                )

                        # Упрощенный расчет ADX (требует больше данных)
                        adx = RiskManagementService._calculate_adx(closes)
                        if adx < safety_rules.min_adx:
                            violations.append(
                                f"ADX слишком низкий: {adx:.1f} < {safety_rules.min_adx}"
                            )

                        # Проверка просадки
                        drawdown = abs(
                            position.unrealized_pnl
                            / (position.position_amount * position.entry_price)
                            * 100
                        )
                        if drawdown > safety_rules.max_drawdown:
                            violations.append(
                                f"Просадка слишком большая: {drawdown:.1f}% > {safety_rules.max_drawdown}%"
                            )

                        is_safe = len(violations) == 0

                        if is_safe:
                            recommendation = "Позиция безопасна"
                        else:
                            recommendation = "Рекомендуется закрыть позицию или пересмотреть стратегию"

                        return SafetyCheckResult(
                            symbol=symbol,
                            is_safe=is_safe,
                            violations=violations,
                            rsi=rsi,
                            adx=adx,
                            drawdown=drawdown,
                            recommendation=recommendation,
                        )
                    else:
                        return SafetyCheckResult(
                            symbol=symbol,
                            is_safe=False,
                            violations=["Недостаточно данных для анализа"],
                            recommendation="Недостаточно исторических данных",
                        )

                except Exception as e:
                    logger.warning(f"Ошибка анализа {symbol}: {e}")
                    return SafetyCheckResult(
                        symbol=symbol,
                        is_safe=False,
                        violations=[f"Ошибка анализа: {str(e)}"],
                        recommendation="Требуется дополнительная проверка",
                    )

            except Exception as e:
                logger.warning(f"Ошибка проверки безопасности {symbol}: {e}")
                return SafetyCheckResult(
                    symbol=symbol,
                    is_safe=False,
                    violations=[f"Ошибка проверки: {str(e)}"],
                    recommendation="Требуется ручная проверка",
                )

        results = await asyncio.gather(
            *[check_symbol_safety(symbol) for symbol in symbols], return_exceptions=True
        )

        safety_results = [
            result for result in results if isinstance(result, SafetyCheckResult)
        ]

        # Автоматическое закрытие небезопасных позиций
        if auto_close_unsafe:
            unsafe_symbols = [r.symbol for r in safety_results if not r.is_safe]
            if unsafe_symbols:
                logger.info(
                    f"Автоматическое закрытие небезопасных позиций: {unsafe_symbols}"
                )
                await BatchService.close_positions_batch(unsafe_symbols, 100.0)

        return safety_results

    @staticmethod
    def _calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Упрощенный расчет RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_adx(prices: List[float], period: int = 14) -> float:
        """Упрощенный расчет ADX."""
        if len(prices) < period + 1:
            return 20.0

        # Упрощенная версия ADX
        highs = prices  # В реальности нужны отдельные high/low/close
        lows = prices

        tr_values = []
        for i in range(1, len(prices)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prices[i - 1]),
                abs(lows[i] - prices[i - 1]),
            )
            tr_values.append(tr)

        if len(tr_values) < period:
            return 20.0

        avg_tr = sum(tr_values[-period:]) / period
        return min(avg_tr / prices[-1] * 100, 50.0)  # Нормализованный ADX

    @staticmethod
    async def manage_stop_losses(
        symbols: List[str], stop_loss_config: Optional[StopLossConfig] = None
    ) -> List[StopLossResult]:
        """Автоматическое управление стоп-лоссами."""
        if len(symbols) > 20:
            raise ValueError("Максимум 20 символов в batch-операции")

        if stop_loss_config is None:
            stop_loss_config = StopLossConfig(stop_loss_type="trailing")

        async def manage_stop_loss_for_symbol(symbol: str) -> StopLossResult:
            try:
                # Получаем позицию
                positions = await FuturesService.get_positions(symbol)
                position = positions[0] if positions else None

                if not position or abs(position.position_amount) < 1e-8:
                    return StopLossResult(
                        symbol=symbol, action="no_action", reason="Нет активной позиции"
                    )

                # Проверяем существующие стоп-лоссы
                open_orders = await OrderService.get_open_orders(symbol)
                stop_orders = [o for o in open_orders if "STOP" in o.type]

                # Рассчитываем стоп-цену
                if stop_loss_config.stop_loss_type == "trailing":
                    if position.position_amount > 0:  # Long
                        stop_price = position.mark_price * (
                            1 - stop_loss_config.trail_percentage / 100
                        )
                    else:  # Short
                        stop_price = position.mark_price * (
                            1 + stop_loss_config.trail_percentage / 100
                        )
                else:  # fixed
                    if position.position_amount > 0:  # Long
                        stop_price = position.entry_price * (
                            1 - stop_loss_config.max_loss_percent / 100
                        )
                    else:  # Short
                        stop_price = position.entry_price * (
                            1 + stop_loss_config.max_loss_percent / 100
                        )

                # Проверяем, нужно ли создать/обновить стоп-лосс
                if not stop_orders:
                    # Создаем новый стоп-лосс
                    try:
                        order_request = CreateOrderRequest(
                            symbol=symbol,
                            side="SELL" if position.position_amount > 0 else "BUY",
                            type="STOP_MARKET",
                            quantity=abs(position.position_amount),
                            stop_price=stop_price,
                            new_order_resp_type="ACK",
                        )

                        await OrderService.create_order(order_request)

                        return StopLossResult(
                            symbol=symbol,
                            action="created",
                            stop_price=stop_price,
                            reason=f"Создан стоп-лосс на уровне {stop_price:.4f}",
                        )
                    except Exception as e:
                        return StopLossResult(
                            symbol=symbol,
                            action="no_action",
                            reason=f"Ошибка создания стоп-лосса: {str(e)}",
                        )
                else:
                    # Обновляем существующий стоп-лосс
                    existing_stop = stop_orders[0]
                    if (
                        existing_stop.price is not None
                        and abs(existing_stop.price - stop_price) / stop_price > 0.01
                    ):  # 1% разница
                        try:
                            # Отменяем старый ордер
                            await OrderService.cancel_order(
                                symbol, existing_stop.order_id
                            )

                            # Создаем новый
                            order_request = CreateOrderRequest(
                                symbol=symbol,
                                side="SELL" if position.position_amount > 0 else "BUY",
                                type="STOP_MARKET",
                                quantity=abs(position.position_amount),
                                stop_price=stop_price,
                                new_order_resp_type="ACK",
                            )

                            await OrderService.create_order(order_request)

                            return StopLossResult(
                                symbol=symbol,
                                action="updated",
                                stop_price=stop_price,
                                reason=f"Обновлен стоп-лосс на уровень {stop_price:.4f}",
                            )
                        except Exception as e:
                            return StopLossResult(
                                symbol=symbol,
                                action="no_action",
                                reason=f"Ошибка обновления стоп-лосса: {str(e)}",
                            )
                    else:
                        return StopLossResult(
                            symbol=symbol,
                            action="no_action",
                            reason="Стоп-лосс актуален",
                        )

            except Exception as e:
                logger.warning(f"Ошибка управления стоп-лоссом {symbol}: {e}")
                return StopLossResult(
                    symbol=symbol, action="no_action", reason=f"Ошибка: {str(e)}"
                )

        results = await asyncio.gather(
            *[manage_stop_loss_for_symbol(symbol) for symbol in symbols],
            return_exceptions=True,
        )

        return [result for result in results if isinstance(result, StopLossResult)]


class AlertService:
    """Сервис для мониторинга и алертов (только уведомления)."""

    @staticmethod
    async def setup_portfolio_alerts(
        symbols: List[str],
        alerts: List[AlertConfig],
        telegram_chat_id: Optional[str] = None,
    ) -> List[AlertResult]:
        """Настройка алертов для портфеля (только уведомления)."""
        if len(symbols) > 20:
            raise ValueError("Максимум 20 символов в batch-операции")

        config = get_config()
        chat_id = telegram_chat_id or config.telegram_chat_id

        async def check_alerts_for_symbol(symbol: str) -> List[AlertResult]:
            results = []

            try:
                # Получаем позицию и рыночные данные
                positions = await FuturesService.get_positions(symbol)
                position = positions[0] if positions else None

                ticker_24h = await MarketService.get_24hr_ticker(symbol)
                ticker_price = await MarketService.get_ticker_price(symbol)

                for alert in alerts:
                    triggered = False
                    current_value = 0.0
                    additional_info = {}

                    if alert.alert_type == "drawdown" and position:
                        # Проверка просадки
                        drawdown = abs(
                            position.unrealized_pnl
                            / (position.position_amount * position.entry_price)
                            * 100
                        )
                        current_value = drawdown
                        if drawdown > alert.threshold:
                            triggered = True
                            additional_info = {
                                "position_amount": position.position_amount,
                                "entry_price": position.entry_price,
                                "unrealized_pnl": position.unrealized_pnl,
                            }

                    elif alert.alert_type == "profit" and position:
                        # Проверка прибыли
                        profit = (
                            position.unrealized_pnl
                            / (position.position_amount * position.entry_price)
                            * 100
                        )
                        current_value = profit
                        if profit > alert.threshold:
                            triggered = True
                            additional_info = {
                                "position_amount": position.position_amount,
                                "entry_price": position.entry_price,
                                "unrealized_pnl": position.unrealized_pnl,
                            }

                    elif alert.alert_type == "volume_spike":
                        # Проверка всплеска объема (упрощенная)
                        volume_multiplier = (
                            ticker_24h.volume
                            / (ticker_24h.quote_volume / ticker_24h.price)
                            if ticker_24h.quote_volume > 0
                            else 1.0
                        )
                        current_value = volume_multiplier
                        if volume_multiplier > alert.threshold:
                            triggered = True
                            additional_info = {
                                "volume": ticker_24h.volume,
                                "quote_volume": ticker_24h.quote_volume,
                            }

                    elif alert.alert_type == "price_level":
                        # Проверка достижения ценового уровня
                        current_price = ticker_price.price
                        current_value = current_price
                        if current_price >= alert.threshold:
                            triggered = True
                            additional_info = {
                                "price": current_price,
                                "direction": 1.0,  # up = 1.0, down = -1.0
                            }

                    elif alert.alert_type == "rsi_extreme":
                        # Проверка экстремальных значений RSI
                        try:
                            klines = await MarketService.get_klines(symbol, "1h", 14)
                            if len(klines.recent_klines) >= 14:
                                closes = [kline.close for kline in klines.recent_klines]
                                rsi = RiskManagementService._calculate_rsi(closes)
                                current_value = rsi
                                additional_info = {"rsi": rsi}

                                if rsi > alert.threshold or rsi < (
                                    100 - alert.threshold
                                ):
                                    triggered = True
                        except Exception as e:
                            logger.warning(f"Ошибка расчета RSI для {symbol}: {e}")

                    # Отправляем уведомление если алерт сработал
                    notification_sent = False
                    message = ""

                    if triggered:
                        message = TelegramService.format_alert_message(
                            symbol,
                            alert.alert_type,
                            current_value,
                            alert.threshold,
                            additional_info,
                        )

                        if chat_id and alert.notification_method == "telegram":
                            notification = TelegramNotification(
                                chat_id=chat_id, message=message, parse_mode="Markdown"
                            )
                            notification_sent = await TelegramService.send_notification(
                                notification
                            )

                    results.append(
                        AlertResult(
                            symbol=symbol,
                            alert_type=alert.alert_type,
                            triggered=triggered,
                            current_value=current_value,
                            threshold=alert.threshold,
                            notification_sent=notification_sent,
                            message=message,
                            timestamp=datetime.utcnow().isoformat() + "Z",
                        )
                    )

            except Exception as e:
                logger.warning(f"Ошибка проверки алертов {symbol}: {e}")
                # Создаем результат с ошибкой
                for alert in alerts:
                    results.append(
                        AlertResult(
                            symbol=symbol,
                            alert_type=alert.alert_type,
                            triggered=False,
                            current_value=0.0,
                            threshold=alert.threshold,
                            notification_sent=False,
                            message=f"Ошибка: {str(e)}",
                            timestamp=datetime.utcnow().isoformat() + "Z",
                        )
                    )

            return results

        all_results = await asyncio.gather(
            *[check_alerts_for_symbol(symbol) for symbol in symbols],
            return_exceptions=True,
        )

        # Flatten results
        flat_results = []
        for symbol_results in all_results:
            if isinstance(symbol_results, list):
                flat_results.extend(symbol_results)

        return flat_results

    @staticmethod
    async def auto_risk_management(
        symbols: List[str], rules: Optional[RiskManagementRule] = None
    ) -> Dict[str, Any]:
        """Автоматическое управление рисками портфеля."""
        if len(symbols) > 20:
            raise ValueError("Максимум 20 символов в batch-операции")

        if rules is None:
            rules = RiskManagementRule()

        try:
            # Получаем обзор портфеля
            portfolio_overview = await BatchService.get_portfolio_overview()

            actions_taken = []
            total_pnl_percent = 0.0

            # Рассчитываем общий PnL портфеля
            if portfolio_overview.active_positions:
                total_invested = sum(
                    abs(pos.position_amount) * pos.entry_price
                    for pos in portfolio_overview.active_positions
                )
                total_pnl = sum(
                    pos.unrealized_pnl for pos in portfolio_overview.active_positions
                )
                total_pnl_percent = (
                    (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
                )

            # Проверка максимальной просадки портфеля
            if total_pnl_percent < rules.max_portfolio_loss:
                actions_taken.append(
                    f"Критическая просадка портфеля: {total_pnl_percent:.2f}%"
                )
                if rules.auto_close_on_loss:
                    await BatchService.close_positions_batch(symbols, 100.0)
                    actions_taken.append("Все позиции закрыты автоматически")

            # Проверка прибыли портфеля
            elif total_pnl_percent > rules.profit_taking:
                actions_taken.append(
                    f"Высокая прибыль портфеля: {total_pnl_percent:.2f}%"
                )
                await BatchService.close_positions_batch(symbols, 50.0)  # Закрываем 50%
                actions_taken.append("50% позиций закрыто для фиксации прибыли")

            # Проверка отдельных позиций
            for position in portfolio_overview.active_positions:
                position_pnl_percent = (
                    position.unrealized_pnl
                    / (abs(position.position_amount) * position.entry_price)
                    * 100
                )

                if position_pnl_percent < rules.max_position_loss:
                    actions_taken.append(
                        f"Критическая просадка {position.symbol}: {position_pnl_percent:.2f}%"
                    )
                    if rules.auto_close_on_loss:
                        await BatchService.close_positions_batch(
                            [position.symbol], 100.0
                        )
                        actions_taken.append(
                            f"Позиция {position.symbol} закрыта автоматически"
                        )

                elif position_pnl_percent > rules.profit_taking:
                    actions_taken.append(
                        f"Высокая прибыль {position.symbol}: {position_pnl_percent:.2f}%"
                    )
                    await BatchService.close_positions_batch([position.symbol], 50.0)
                    actions_taken.append(f"50% позиции {position.symbol} закрыто")

            return {
                "total_pnl_percent": total_pnl_percent,
                "actions_taken": actions_taken,
                "rules_applied": rules.dict(),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        except Exception as e:
            logger.error(f"Ошибка автоматического управления рисками: {e}")
            return {
                "error": str(e),
                "actions_taken": [],
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }


class TelegramService:
    """Сервис для отправки уведомлений в Telegram."""

    @staticmethod
    async def send_notification(notification: TelegramNotification) -> bool:
        """Отправляет уведомление в Telegram (заглушка)."""
        try:
            config = get_config()

            if not config.telegram_bot_token or not config.telegram_chat_id:
                logger.warning("Telegram не настроен - пропускаем уведомление")
                return False

            # ЗАГЛУШКА: В реальной реализации здесь будет HTTP запрос к Telegram API
            logger.info("📱 TELEGRAM УВЕДОМЛЕНИЕ:")
            logger.info(f"   Chat ID: {notification.chat_id}")
            logger.info(f"   Сообщение: {notification.message}")
            logger.info(f"   Parse Mode: {notification.parse_mode}")

            # В реальной реализации:
            # import aiohttp
            # async with aiohttp.ClientSession() as session:
            #     url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
            #     data = {
            #         "chat_id": notification.chat_id,
            #         "text": notification.message,
            #         "parse_mode": notification.parse_mode,
            #         "disable_notification": notification.disable_notification
            #     }
            #     async with session.post(url, json=data) as response:
            #         return response.status == 200

            return True

        except Exception as e:
            logger.error(f"Ошибка отправки Telegram уведомления: {e}")
            return False

    @staticmethod
    def format_alert_message(
        symbol: str,
        alert_type: str,
        current_value: float,
        threshold: float,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Форматирует сообщение для алерта."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Базовое сообщение
        if alert_type == "drawdown":
            message = f"🚨 *АЛЕРТ: Просадка {symbol}*\n\n"
            message += f"📉 Просадка: *{current_value:.2f}%*\n"
            message += f"⚠️ Порог: *{threshold:.2f}%*\n"
            message += f"⏰ Время: {timestamp}\n\n"
            message += "💡 *Рекомендация:* Рассмотрите закрытие позиции или установку стоп-лосса"

        elif alert_type == "profit":
            message = f"💰 *АЛЕРТ: Прибыль {symbol}*\n\n"
            message += f"📈 Прибыль: *{current_value:.2f}%*\n"
            message += f"🎯 Порог: *{threshold:.2f}%*\n"
            message += f"⏰ Время: {timestamp}\n\n"
            message += "💡 *Рекомендация:* Рассмотрите фиксацию прибыли или установку тейк-профита"

        elif alert_type == "volume_spike":
            message = f"📊 *АЛЕРТ: Всплеск объема {symbol}*\n\n"
            message += f"🔥 Объем: *{current_value:.1f}x*\n"
            message += f"⚠️ Порог: *{threshold:.1f}x*\n"
            message += f"⏰ Время: {timestamp}\n\n"
            message += "💡 *Рекомендация:* Проверьте новости и технический анализ"

        elif alert_type == "price_level":
            price = additional_info.get("price", 0) if additional_info else 0
            direction = (
                additional_info.get("direction", "up") if additional_info else "up"
            )
            emoji = "📈" if direction == "up" else "📉"
            message = f"{emoji} *АЛЕРТ: Ценовой уровень {symbol}*\n\n"
            message += f"💲 Цена: *${price:.4f}*\n"
            message += f"🎯 Направление: *{direction.upper()}*\n"
            message += f"⏰ Время: {timestamp}\n\n"
            message += "💡 *Рекомендация:* Проверьте технические уровни"

        elif alert_type == "rsi_extreme":
            rsi = additional_info.get("rsi", 0) if additional_info else 0
            level = "перекупленность" if rsi > 70 else "перепроданность"
            emoji = "🔴" if rsi > 70 else "🟢"
            message = f"{emoji} *АЛЕРТ: RSI {symbol}*\n\n"
            message += f"📊 RSI: *{rsi:.1f}*\n"
            message += f"📈 Уровень: *{level}*\n"
            message += f"⏰ Время: {timestamp}\n\n"
            message += "💡 *Рекомендация:* Рассмотрите разворот позиции"

        else:
            message = f"🔔 *АЛЕРТ: {symbol}*\n\n"
            message += f"📊 Тип: *{alert_type}*\n"
            message += f"📈 Значение: *{current_value:.2f}*\n"
            message += f"🎯 Порог: *{threshold:.2f}*\n"
            message += f"⏰ Время: {timestamp}"

        return message
