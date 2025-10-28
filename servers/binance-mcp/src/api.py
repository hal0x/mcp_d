"""FastAPI приложение с MCP интеграцией."""

import asyncio
from datetime import datetime
from importlib import metadata
from typing import Any, Dict, List, Optional, cast
from typing_extensions import Literal

from fastapi import (
    Body,
    FastAPI,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
import structlog
from fastapi_mcp import FastApiMCP

from .config import get_config
from .client import get_client_info
from .models import (
    AccountInfo,
    AlertConfig,
    AlertResult,
    AvailablePair,
    AveragePrice,
    BalanceEntry,
    BatchResult,
    CancelOrderResponse,
    ClosePositionsBatchRequest,
    CreateMarginOrderRequest,
    CreateMarginOrderResponse,
    CreatePositionRequest,
    CreatePositionsBatchRequest,
    DerivativesContext,
    ExchangeInfo,
    FuturesMarginTypeChange,
    KlinesResponse,
    MarginAccount,
    MarginOrder,
    MarginTrade,
    OCOOrder,
    Order,
    OrderBook,
    OrderDetails,
    PortfolioOverview,
    PortfolioRiskMetrics,
    PositionLimitResponse,
    PositionLimitUpdate,
    PositionSizeRequestModel,
    PositionSizeResponseModel,
    RiskEvaluationRequest,
    RiskEvaluationResponse,
    RiskManagementRule,
    HaltCheckRequest,
    SafetyCheckResult,
    SafetyRule,
    ServerTime,
    StopLossConfig,
    StopLossResult,
    Ticker24h,
    TickerPrice,
    TelegramNotification,
    Trade,
    TradeFee,
    TradingLimitsInfo,
)
from .services import (
    AccountService,
    AlertService,
    BatchService,
    ExchangeService,
    FuturesService,
    MarginService,
    MarketService,
    OCOService,
    OrderService,
    PortfolioService,
    RiskManagementService,
    RiskService,
    TelegramService,
)
from .risk_tools import PositionSizeRequest
from .cache import init_redis, close_redis
from .storage import init_postgres, close_postgres, get_postgres_storage
from .logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


class SymbolsRequest(BaseModel):
    symbols: List[str] = Field(..., description="Список торговых символов")


class ChangeLeverageRequest(BaseModel):
    symbol_leverage_map: Dict[str, int] = Field(
        ..., description="Словарь {символ: плечо}"
    )


class ClosePositionsRequest(BaseModel):
    symbols: List[str] = Field(..., description="Список торговых символов")
    percentage: float = Field(100.0, ge=1.0, le=100.0, description="Процент закрытия")


class ChangeMarginTypeRequest(BaseModel):
    symbol: str = Field(..., description="Торговая пара")
    margin_type: str = Field(..., description="Новый тип маржи (ISOLATED/CROSSED)")


class CancelOrderRequest(BaseModel):
    symbol: str = Field(..., description="Торговая пара")
    order_id: Optional[int] = Field(
        default=None, description="Идентификатор ордера Binance"
    )
    client_order_id: Optional[str] = Field(
        default=None, description="Клиентский идентификатор ордера"
    )


class ExchangeFiltersRequest(BaseModel):
    filters: Optional[Dict[str, str]] = Field(
        default=None, description="Фильтры для доступных пар"
    )


class PortfolioSafetyRequest(BaseModel):
    symbols: List[str] = Field(..., description="Список торговых символов")
    safety_rules: Optional[SafetyRule] = Field(
        default=None, description="Правила безопасности"
    )
    auto_close_unsafe: bool = Field(
        default=False, description="Автоматически закрывать опасные позиции"
    )


class ManageStopLossRequest(BaseModel):
    symbols: List[str] = Field(..., description="Список торговых символов")
    stop_loss_config: Optional[StopLossConfig] = Field(
        default=None, description="Конфигурация стоп-лоссов"
    )


class SetupAlertsRequest(BaseModel):
    symbols: List[str] = Field(..., description="Список торговых символов")
    alerts: List[AlertConfig] = Field(..., description="Конфигурация алертов")
    telegram_chat_id: Optional[str] = Field(
        default=None, description="ID Telegram-чата для уведомлений"
    )


class TestTelegramRequest(BaseModel):
    message: str = Field(..., description="Сообщение для отправки")
    chat_id: Optional[str] = Field(
        default=None, description="Необязательный ID Telegram-чата"
    )


class AutoRiskManagementRequest(BaseModel):
    symbols: List[str] = Field(..., description="Список торговых символов")
    rules: Optional[RiskManagementRule] = Field(
        default=None, description="Правила управления рисками"
    )


class TickersBatchRequest(BaseModel):
    symbols: List[str] = Field(..., description="Список торговых символов")


def create_app() -> FastAPI:
    """Создает и настраивает FastAPI приложение."""
    try:
        get_config()
        client_info = get_client_info()

        title = f"Binance Native MCP ({client_info['mode']})"
        description = (
            f"MCP сервер для работы с Binance API. Режим: {client_info['mode']}"
        )
    except Exception:
        title = "Binance Native MCP"
        description = "MCP сервер для работы с Binance API"

    app = FastAPI(
        title=title,
        version="1.0.0",
        description=description,
    )

    # Регистрируем эндпоинты
    register_routes(app)

    @app.on_event("startup")
    async def _startup() -> None:
        await init_redis()
        await init_postgres()
        storage = get_postgres_storage()
        if storage is not None:
            await RiskService.configure(storage)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await close_redis()
        await close_postgres()

    # Настраиваем MCP
    mcp = FastApiMCP(app, name="binance-mcp")
    mcp.mount_http()

    return app


def register_routes(app: FastAPI) -> None:
    """Регистрирует маршруты API."""

    @app.get("/healthz", tags=["internal"], include_in_schema=False)
    async def healthcheck() -> dict:
        """Проверка здоровья сервиса."""
        return {"status": "ok"}

    @app.get(
        "/meta/health",
        tags=["meta"],
        operation_id="health",
        summary="Проверить состояние MCP сервера",
    )
    async def mcp_health() -> Dict[str, Any]:
        """Расширенная проверка состояния сервиса."""
        status = "healthy"
        error: Optional[str] = None
        try:
            config = get_config()
        except Exception as exc:  # pragma: no cover
            status = "degraded"
            error = str(exc)
            config = None

        client_info = get_client_info()

        payload: Dict[str, Any] = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mode": client_info.get("mode", "UNKNOWN"),
            "services": {
                "account": True,
                "market": True,
                "orders": True,
                "portfolio": True,
            },
        }

        if config:
            payload["config"] = {
                "demo_trading": config.demo_trading,
                "host": config.host,
                "port": config.port,
                "api_key_present": bool(config.api_key),
                "demo_api_key_present": bool(config.demo_api_key),
            }
        else:
            payload["config"] = None

        if error:
            payload["error"] = error

        return payload

    @app.get(
        "/meta/version",
        tags=["meta"],
        operation_id="version",
        summary="Получить информацию о версии сервиса",
    )
    async def mcp_version() -> Dict[str, Any]:
        """Информация о версии и доступных возможностях сервера."""
        try:
            version_str = metadata.version("binance-mcp")
        except metadata.PackageNotFoundError:
            version_str = "0.0.0"

        client_info = get_client_info()
        return {
            "name": "binance-mcp",
            "version": version_str,
            "mode": client_info.get("mode", "UNKNOWN"),
            "features": [
                "account",
                "market",
                "orders",
                "portfolio",
                "futures",
                "margin",
                "oco",
                "batch",
            ],
        }

    # Аккаунт
    @app.get(
        "/account/info",
        response_model=AccountInfo,
        tags=["account"],
        operation_id="get_account_info",
    )
    async def account_info() -> AccountInfo:
        """Получить информацию об аккаунте."""
        return await AccountService.get_account_info()

    @app.get(
        "/account/balances",
        response_model=List[BalanceEntry],
        tags=["account"],
        operation_id="get_account_balance",
    )
    async def account_balances() -> List[BalanceEntry]:
        """Получить баланс аккаунта."""
        return await AccountService.get_account_balance()

    @app.get(
        "/account/trade-fee",
        response_model=TradeFee,
        tags=["account"],
        operation_id="get_trade_fee",
    )
    async def trade_fee(
        symbol: str = Query(..., description="Торговая пара, например BTCUSDT")
    ) -> TradeFee:
        """Получить торговые комиссии по символу."""
        return await AccountService.get_trade_fee(symbol)

    # Рыночные данные
    @app.get(
        "/market/ticker",
        response_model=TickerPrice,
        tags=["market"],
        operation_id="get_ticker_price",
    )
    async def ticker_price(
        symbol: str = Query(..., description="Торговая пара, например BTCUSDT")
    ) -> TickerPrice:
        """Получить текущую цену символа."""
        return await MarketService.get_ticker_price(symbol)

    @app.get(
        "/market/ticker/24h",
        response_model=Ticker24h,
        tags=["market"],
        operation_id="get_24hr_ticker",
    )
    async def ticker_24_hours(
        symbol: str = Query(..., description="Торговая пара, например BTCUSDT")
    ) -> Ticker24h:
        """Получить статистику за 24 часа."""
        return await MarketService.get_24hr_ticker(symbol)

    @app.get(
        "/market/order-book",
        response_model=OrderBook,
        tags=["market"],
        operation_id="get_order_book",
    )
    async def order_book(
        symbol: str = Query(..., description="Торговая пара, например BTCUSDT"),
        limit: int = Query(100, description="Глубина книги ордеров", ge=5, le=5000),
    ) -> OrderBook:
        """Получить книгу ордеров."""
        return await MarketService.get_order_book(symbol, limit)

    @app.get(
        "/market/klines",
        response_model=KlinesResponse,
        tags=["market"],
        operation_id="get_klines",
    )
    async def klines(
        symbol: str = Query(..., description="Торговая пара, например BTCUSDT"),
        interval: str = Query("1h", description="Интервал свечей"),
        limit: int = Query(
            100, description="Количество свечей (макс 1000)", ge=1, le=1000
        ),
    ) -> KlinesResponse:
        """Получить данные свечей."""
        return await MarketService.get_klines(symbol, interval, limit)

    @app.get(
        "/market/avg-price",
        response_model=AveragePrice,
        tags=["market"],
        operation_id="get_avg_price",
    )
    async def avg_price(
        symbol: str = Query(..., description="Торговая пара, например BTCUSDT")
    ) -> AveragePrice:
        """Получить усреднённую цену символа."""
        return await MarketService.get_avg_price(symbol)

    @app.get("/market/trades", tags=["market"], operation_id="get_recent_trades")
    async def recent_trades(
        symbol: str = Query(..., description="Торговая пара, например BTCUSDT"),
        limit: int = Query(50, description="Количество сделок", ge=1, le=1000),
    ) -> List[dict]:
        """Получить последние сделки по символу."""
        trades = await MarketService.get_recent_trades(symbol, limit)
        return [trade.model_dump(mode="json") for trade in trades]

    # Ордера
    @app.get(
        "/orders/open",
        response_model=List[Order],
        tags=["orders"],
        operation_id="get_open_orders",
    )
    async def open_orders(
        symbol: Optional[str] = Query(None, description="Опциональная торговая пара")
    ) -> List[Order]:
        """Получить открытые ордера."""
        return await OrderService.get_open_orders(symbol)

    @app.get(
        "/orders/history",
        response_model=List[Order],
        tags=["orders"],
        operation_id="get_order_history",
    )
    async def order_history(
        symbol: str = Query(..., description="Торговая пара"),
        limit: int = Query(10, description="Количество ордеров", ge=1, le=1000),
    ) -> List[Order]:
        """Получить историю ордеров."""
        return await OrderService.get_order_history(symbol, limit)

    @app.get(
        "/orders/trades",
        response_model=List[Trade],
        tags=["orders"],
        operation_id="get_trade_history",
    )
    async def trade_history(
        symbol: str = Query(..., description="Торговая пара"),
        limit: int = Query(10, description="Количество сделок", ge=1, le=1000),
    ) -> List[Trade]:
        """Получить историю сделок."""
        return await OrderService.get_trade_history(symbol, limit)

    # Маржинальная торговля
    @app.get(
        "/margin/account",
        response_model=MarginAccount,
        tags=["margin"],
        operation_id="get_margin_account",
    )
    async def margin_account(
        isolated: bool = Query(False, description="Режим изолированной маржи"),
        symbol: Optional[str] = Query(None, description="Опциональный символ"),
    ) -> MarginAccount:
        """Получить сведения о маржинальном аккаунте."""
        return await MarginService.get_margin_account(isolated=isolated, symbol=symbol)

    @app.get(
        "/margin/orders",
        response_model=List[MarginOrder],
        tags=["margin"],
        operation_id="get_margin_orders",
    )
    async def margin_orders(
        symbol: str = Query(..., description="Торговая пара"),
        limit: int = Query(10, ge=1, le=1000, description="Количество ордеров"),
        is_isolated: Optional[bool] = Query(
            None, description="Работа с изолированной маржой"
        ),
    ) -> List[MarginOrder]:
        """Получить историю маржинальных ордеров."""
        return await MarginService.get_margin_orders(
            symbol, limit=limit, is_isolated=is_isolated
        )

    @app.post(
        "/margin/orders",
        response_model=CreateMarginOrderResponse,
        tags=["margin"],
        operation_id="create_margin_order",
    )
    async def margin_create_order(
        payload: CreateMarginOrderRequest,
    ) -> CreateMarginOrderResponse:
        """Создать маржинальный ордер."""
        return await MarginService.create_margin_order(payload)

    @app.get(
        "/margin/trades",
        response_model=List[MarginTrade],
        tags=["margin"],
        operation_id="get_margin_trades",
    )
    async def margin_trades(
        symbol: str = Query(..., description="Торговая пара"),
        limit: int = Query(10, ge=1, le=1000, description="Количество сделок"),
        is_isolated: Optional[bool] = Query(
            None, description="Работа с изолированной маржой"
        ),
    ) -> List[MarginTrade]:
        """Получить историю маржинальных сделок."""
        return await MarginService.get_margin_trades(
            symbol, is_isolated=is_isolated, limit=limit
        )

    @app.post(
        "/margin/oco",
        response_model=OCOOrder,
        tags=["margin"],
        operation_id="create_margin_oco_order",
    )
    async def margin_create_oco(
        payload: Dict[str, Any] = Body(..., description="Параметры маржинального OCO")
    ) -> OCOOrder:
        """Создать маржинальный OCO ордер."""
        return await MarginService.create_margin_oco_order(payload)

    @app.post(
        "/margin/oco/cancel",
        response_model=OCOOrder,
        tags=["margin"],
        operation_id="cancel_margin_oco_order",
    )
    async def margin_cancel_oco(
        payload: Dict[str, Any] = Body(..., description="Параметры отмены OCO ордера")
    ) -> OCOOrder:
        """Отменить маржинальный OCO ордер."""
        return await MarginService.cancel_margin_oco_order(payload)

    @app.get(
        "/margin/oco/open",
        response_model=List[OCOOrder],
        tags=["margin"],
        operation_id="get_open_margin_oco_orders",
    )
    async def margin_open_oco(
        symbol: Optional[str] = Query(None, description="Торговая пара"),
        is_isolated: Optional[bool] = Query(
            None, description="Работа с изолированной маржой"
        ),
    ) -> List[OCOOrder]:
        """Получить открытые маржинальные OCO ордера."""
        return await MarginService.get_open_margin_oco_orders(
            symbol=symbol, is_isolated=is_isolated
        )

    @app.post(
        "/margin/oco/get",
        response_model=OCOOrder,
        tags=["margin"],
        operation_id="get_margin_oco_order",
    )
    async def margin_get_oco(
        payload: Dict[str, Any] = Body(..., description="Параметры поиска OCO ордера")
    ) -> OCOOrder:
        """Получить маржинальный OCO ордер по идентификаторам."""
        return await MarginService.get_margin_oco_order(payload)

    # Spot OCO ордера
    @app.post(
        "/spot/oco",
        response_model=OCOOrder,
        tags=["orders"],
        operation_id="create_oco_order",
    )
    async def spot_create_oco(
        payload: Dict[str, Any] = Body(..., description="Параметры OCO ордера")
    ) -> OCOOrder:
        """Создать spot OCO ордер."""
        return await OCOService.create_oco_order(payload)

    @app.post(
        "/spot/oco/cancel",
        response_model=OCOOrder,
        tags=["orders"],
        operation_id="cancel_oco_order",
    )
    async def spot_cancel_oco(
        payload: Dict[str, Any] = Body(..., description="Параметры отмены OCO ордера")
    ) -> OCOOrder:
        """Отменить spot OCO ордер."""
        return await OCOService.cancel_oco_order(payload)

    @app.get(
        "/spot/oco/open",
        response_model=List[OCOOrder],
        tags=["orders"],
        operation_id="get_open_oco_orders",
    )
    async def spot_open_oco() -> List[OCOOrder]:
        """Получить открытые spot OCO ордера."""
        return await OCOService.get_open_oco_orders()

    # Фьючерсы и ордера
    @app.post(
        "/futures/margin-type",
        response_model=FuturesMarginTypeChange,
        tags=["futures"],
        operation_id="change_futures_margin_type",
    )
    async def futures_change_margin_type(
        payload: ChangeMarginTypeRequest,
    ) -> FuturesMarginTypeChange:
        """Изменить тип маржи для фьючерсного символа."""
        return await FuturesService.change_margin_type(
            payload.symbol, payload.margin_type
        )

    @app.get(
        "/orders/details",
        response_model=OrderDetails,
        tags=["orders"],
        operation_id="get_order",
    )
    async def order_details(
        symbol: str = Query(..., description="Торговая пара"),
        order_id: Optional[int] = Query(None, description="Идентификатор ордера"),
        client_order_id: Optional[str] = Query(
            None, description="Клиентский идентификатор ордера"
        ),
    ) -> OrderDetails:
        """Получить детали ордера."""
        return await OrderService.get_order(
            symbol, order_id=order_id, client_order_id=client_order_id
        )

    @app.post(
        "/orders/cancel",
        response_model=CancelOrderResponse,
        tags=["orders"],
        operation_id="cancel_order",
    )
    async def cancel_order(
        payload: CancelOrderRequest,
    ) -> CancelOrderResponse:
        """Отменить ордер."""
        return await OrderService.cancel_order(
            payload.symbol,
            order_id=payload.order_id,
            client_order_id=payload.client_order_id,
        )

    # Биржа

    @app.post(
        "/risk/position-size",
        response_model=PositionSizeResponseModel,
        tags=["risk"],
        operation_id="calculate_position_size",
    )
    async def calculate_position_size_endpoint(
        payload: PositionSizeRequestModel,
    ) -> PositionSizeResponseModel:
        result = await RiskService.calculate_position_size(
            PositionSizeRequest(
                equity=payload.equity,
                entry=payload.entry,
                stop_loss=payload.stop_loss,
                side=cast(Literal["long", "short"], payload.side.lower()),
            )
        )
        return PositionSizeResponseModel(**result)

    @app.post(
        "/risk/evaluate",
        response_model=RiskEvaluationResponse,
        tags=["risk"],
        operation_id="evaluate_trade",
    )
    async def evaluate_trade_endpoint(
        payload: RiskEvaluationRequest,
    ) -> RiskEvaluationResponse:
        result = await RiskService.evaluate_trade(
            entry=payload.entry,
            stop_loss=payload.stop_loss,
            take_profit=payload.take_profit,
            atr=payload.atr,
            side=payload.side,
        )
        return RiskEvaluationResponse(**result)

    @app.post("/risk/halt-check", tags=["risk"], operation_id="halt_check")
    async def halt_check(payload: HaltCheckRequest) -> dict[str, bool]:
        decision = await RiskService.should_halt(
            payload.total_pnl_pct_today, payload.consecutive_losses
        )
        return {"halt": decision}

    @app.get(
        "/risk/portfolio",
        response_model=PortfolioRiskMetrics,
        tags=["risk"],
        operation_id="get_portfolio_risk",
    )
    async def get_portfolio_risk() -> PortfolioRiskMetrics:
        """Получить агрегированные метрики риска портфеля."""
        return await RiskService.get_portfolio_risk_metrics()

    @app.get(
        "/risk/position-limits", tags=["risk"], operation_id="list_position_limits"
    )
    async def list_position_limits() -> dict[str, dict[str, float]]:
        """Получить текущие лимиты позиций."""
        return await RiskService.get_position_limits()

    @app.post(
        "/risk/position-limits",
        response_model=PositionLimitResponse,
        tags=["risk"],
        operation_id="set_position_limit",
    )
    async def set_position_limit(update: PositionLimitUpdate) -> PositionLimitResponse:
        """Установить лимит позиции по символу."""
        return await RiskService.set_position_limit(update)

    @app.get(
        "/portfolio/simple-balance",
        tags=["portfolio"],
        operation_id="get_simple_balance",
    )
    async def simple_balance() -> Dict[str, Any]:
        """Получить упрощённую сводку баланса."""
        return await PortfolioService.get_simple_balance()

    @app.post(
        "/futures/positions/batch",
        response_model=List[BatchResult],
        tags=["futures"],
        operation_id="get_futures_positions_batch",
    )
    async def futures_positions_batch(
        payload: SymbolsRequest,
    ) -> List[BatchResult]:
        """Получить фьючерсные позиции для списка символов."""
        return await BatchService.get_futures_positions_batch(payload.symbols)

    @app.get(
        "/portfolio/overview",
        response_model=PortfolioOverview,
        tags=["portfolio"],
        operation_id="get_portfolio_overview",
    )
    async def portfolio_overview() -> PortfolioOverview:
        """Получить комбинированную сводку портфеля."""
        return await BatchService.get_portfolio_overview()

    @app.post(
        "/market/tickers/batch",
        response_model=List[BatchResult],
        tags=["market"],
        operation_id="get_tickers_batch",
    )
    async def tickers_batch(payload: TickersBatchRequest) -> List[BatchResult]:
        """Получить данные по списку символов."""
        return await BatchService.get_tickers_batch(payload.symbols)

    @app.post(
        "/futures/leverage/batch",
        response_model=List[BatchResult],
        tags=["futures"],
        operation_id="change_leverage_batch",
    )
    async def leverage_batch(
        payload: ChangeLeverageRequest,
    ) -> List[BatchResult]:
        """Изменить плечо для набора символов."""
        return await BatchService.change_leverage_batch(payload.symbol_leverage_map)

    @app.post(
        "/futures/positions/create",
        response_model=List[BatchResult],
        tags=["futures"],
        operation_id="create_positions_batch",
    )
    async def create_positions_batch(
        request: CreatePositionsBatchRequest,
    ) -> List[BatchResult]:
        """Создать несколько фьючерсных позиций."""
        return await BatchService.create_positions_batch(request.positions)

    @app.post(
        "/futures/positions/close",
        response_model=List[BatchResult],
        tags=["futures"],
        operation_id="close_positions_batch",
    )
    async def close_positions_batch(
        payload: ClosePositionsRequest,
    ) -> List[BatchResult]:
        """Закрыть фьючерсные позиции."""
        return await BatchService.close_positions_batch(
            payload.symbols, percentage=payload.percentage
        )

    @app.post(
        "/exchange/available-pairs",
        response_model=List[AvailablePair],
        tags=["exchange"],
        operation_id="get_available_pairs",
    )
    async def available_pairs(
        payload: ExchangeFiltersRequest,
    ) -> List[AvailablePair]:
        """Получить доступные торговые пары."""
        return await ExchangeService.get_available_pairs(filters=payload.filters)

    @app.post(
        "/exchange/check-limits",
        response_model=List[TradingLimitsInfo],
        tags=["exchange"],
        operation_id="check_trading_limits",
    )
    async def trading_limits(payload: SymbolsRequest) -> List[TradingLimitsInfo]:
        """Проверить торговые лимиты."""
        return await ExchangeService.check_trading_limits(payload.symbols)

    @app.post(
        "/risk/portfolio-check",
        response_model=List[SafetyCheckResult],
        tags=["risk"],
        operation_id="portfolio_safety_check",
    )
    async def portfolio_safety(
        payload: PortfolioSafetyRequest,
    ) -> List[SafetyCheckResult]:
        """Проверить безопасность портфеля."""
        return await RiskManagementService.portfolio_safety_check(
            payload.symbols, payload.safety_rules, payload.auto_close_unsafe
        )

    @app.post(
        "/risk/manage-stop-losses",
        response_model=List[StopLossResult],
        tags=["risk"],
        operation_id="manage_stop_losses",
    )
    async def manage_stop_losses(
        payload: ManageStopLossRequest,
    ) -> List[StopLossResult]:
        """Управление стоп-лоссами."""
        return await RiskManagementService.manage_stop_losses(
            payload.symbols, payload.stop_loss_config
        )

    @app.post(
        "/alerts/setup",
        response_model=List[AlertResult],
        tags=["alerts"],
        operation_id="setup_portfolio_alerts",
    )
    async def setup_alerts(payload: SetupAlertsRequest) -> List[AlertResult]:
        """Настроить уведомления по портфелю."""
        return await AlertService.setup_portfolio_alerts(
            payload.symbols, payload.alerts, payload.telegram_chat_id
        )

    @app.post(
        "/alerts/test-telegram",
        tags=["alerts"],
        operation_id="test_telegram_notification",
    )
    async def test_telegram(payload: TestTelegramRequest) -> Dict[str, Any]:
        """Отправить тестовое уведомление в Telegram."""
        notification = TelegramNotification(
            chat_id=payload.chat_id or "",
            message=payload.message,
            parse_mode="Markdown",
        )
        success = await TelegramService.send_notification(notification)
        return {"success": success, "message": payload.message}

    @app.post(
        "/alerts/auto-risk",
        tags=["alerts"],
        operation_id="auto_risk_management",
    )
    async def auto_risk(payload: AutoRiskManagementRequest) -> Dict[str, Any]:
        """Автоматическое управление рисками."""
        result = await AlertService.auto_risk_management(payload.symbols, payload.rules)
        return result

    @app.get(
        "/derivatives/context",
        response_model=DerivativesContext,
        tags=["market"],
        operation_id="get_derivatives_context",
    )
    async def derivatives_context(
        symbol: str = Query(..., description="Торговая пара")
    ) -> DerivativesContext:
        """Контекст деривативов для символа."""
        return await RiskService.get_derivatives_data(symbol)

    @app.get(
        "/exchange/info",
        response_model=ExchangeInfo,
        tags=["exchange"],
        operation_id="get_exchange_info",
    )
    async def exchange_info(
        symbol: Optional[str] = Query(None, description="Опциональная торговая пара")
    ) -> ExchangeInfo:
        """Получить информацию о бирже."""
        return await ExchangeService.get_exchange_info(symbol)

    @app.get(
        "/exchange/server-time",
        response_model=ServerTime,
        tags=["exchange"],
        operation_id="get_server_time",
    )
    async def server_time() -> ServerTime:
        """Получить время сервера."""
        return await ExchangeService.get_server_time()

    @app.websocket("/ws/ticker/{symbol}")
    async def ticker_stream(websocket: WebSocket, symbol: str) -> None:
        await websocket.accept()
        try:
            while True:
                data = await MarketService.get_ticker_price(symbol)
                await websocket.send_json(data.model_dump(mode="json"))
                await asyncio.sleep(1)
        except WebSocketDisconnect:  # pragma: no cover - client disconnect
            logger.info("ticker_ws_disconnect", symbol=symbol)
        except asyncio.CancelledError:
            logger.info("ticker_ws_cancelled", symbol=symbol)
            raise
        except Exception:  # pragma: no cover
            logger.exception("ticker_ws_error", symbol=symbol)
            await websocket.close(code=1011)

    @app.websocket("/ws/order-book/{symbol}")
    async def order_book_stream(
        websocket: WebSocket, symbol: str, limit: int = 50
    ) -> None:
        await websocket.accept()
        try:
            while True:
                book = await MarketService.get_order_book(symbol, limit)
                await websocket.send_json(book.model_dump(mode="json"))
                await asyncio.sleep(1)
        except WebSocketDisconnect:  # pragma: no cover
            logger.info("order_book_ws_disconnect", symbol=symbol)
        except asyncio.CancelledError:
            logger.info("order_book_ws_cancelled", symbol=symbol)
            raise
        except Exception:  # pragma: no cover
            logger.exception("order_book_ws_error", symbol=symbol)
            await websocket.close(code=1011)

    @app.websocket("/ws/trades/{symbol}")
    async def trades_stream(websocket: WebSocket, symbol: str, limit: int = 50) -> None:
        await websocket.accept()
        try:
            while True:
                trades = await MarketService.get_recent_trades(symbol, limit)
                await websocket.send_json(
                    [trade.model_dump(mode="json") for trade in trades]
                )
                await asyncio.sleep(1)
        except WebSocketDisconnect:  # pragma: no cover
            logger.info("trades_ws_disconnect", symbol=symbol)
        except asyncio.CancelledError:
            logger.info("trades_ws_cancelled", symbol=symbol)
            raise
        except Exception:  # pragma: no cover
            logger.exception("trades_ws_error", symbol=symbol)
            await websocket.close(code=1011)


# Создаем приложение
app = create_app()
