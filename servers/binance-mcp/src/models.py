"""Pydantic модели для API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AccountInfo(BaseModel):
    """Информация об аккаунте."""

    account_type: str
    can_trade: bool
    can_withdraw: bool
    can_deposit: bool
    update_time: int
    balances_count: int
    permissions: List[str]


class BalanceEntry(BaseModel):
    """Запись баланса."""

    asset: str
    free: float
    locked: float
    total: float


class TickerPrice(BaseModel):
    """Цена тикера."""

    symbol: str
    price: float


class Ticker24h(BaseModel):
    """24-часовая статистика тикера."""

    symbol: str
    price: float
    price_change: float
    price_change_percent: float
    high: float
    low: float
    volume: float
    quote_volume: float
    trade_count: int


class OrderBookLevel(BaseModel):
    """Уровень книги ордеров."""

    price: float
    quantity: float


class OrderBook(BaseModel):
    """Книга ордеров."""

    symbol: str
    last_update_id: int
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    total_bids: int
    total_asks: int


class Kline(BaseModel):
    """Свеча."""

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


class KlinesResponse(BaseModel):
    """Ответ с данными свечей."""

    symbol: str
    interval: str
    total_klines: int
    recent_klines: List[Kline]


class Order(BaseModel):
    """Ордер."""

    order_id: int
    symbol: str
    side: str
    type: str
    quantity: float
    price: Optional[float]
    status: str
    time: int


class Trade(BaseModel):
    """Сделка."""

    trade_id: int
    symbol: str
    order_id: int
    side: str
    quantity: float
    price: float
    commission: float
    commission_asset: str
    time: int


class TradeFee(BaseModel):
    """Комиссии для торговли символом."""

    symbol: str
    maker_fee: float
    taker_fee: float
    maker_fee_bnb: Optional[float] = None
    taker_fee_bnb: Optional[float] = None


class AveragePrice(BaseModel):
    """Средняя цена по символу."""

    symbol: str
    minutes: int
    price: float


class PortfolioRiskMetrics(BaseModel):
    """Метрики риска портфеля."""

    total_equity: float
    available_balance: float
    unrealized_pnl: float
    margin_ratio: float | None = None
    positions_count: int = 0
    largest_position: Optional[str] = None


class PositionLimitUpdate(BaseModel):
    """Запрос на обновление лимита позиции."""

    symbol: str
    max_size: float


class PositionLimitResponse(BaseModel):
    """Ответ после обновления лимита позиции."""

    symbol: str
    max_size: float
    updated_at: int


class DerivativesContext(BaseModel):
    """Контекст деривативов (фьючерсы)."""

    symbol: str
    open_interest: float | None = None
    funding_rate: float | None = None
    mark_price: float | None = None
    last_funding_time: int | None = None


class PositionSizeRequestModel(BaseModel):
    equity: float = Field(..., gt=0)
    entry: float = Field(..., gt=0)
    stop_loss: float = Field(...)
    side: str = Field(..., description="long/short")


class PositionSizeResponseModel(BaseModel):
    quantity: float
    effective_entry: float
    risk_amount: float


class RiskEvaluationRequest(BaseModel):
    entry: float
    stop_loss: float
    take_profit: float
    atr: float
    side: str


class RiskEvaluationResponse(BaseModel):
    rr: float
    adjusted_tp: float
    adjusted_rr: float


class HaltCheckRequest(BaseModel):
    total_pnl_pct_today: float
    consecutive_losses: int


class RecentTrade(BaseModel):
    """Недавняя публичная сделка."""

    id: int
    price: float
    quantity: float
    timestamp: int
    is_buyer_maker: bool


class RecentTradesBatchEntry(BaseModel):
    """Набор последних сделок для символа."""

    symbol: str
    trades: List[RecentTrade]


class OrderDetails(BaseModel):
    """Подробности ордера."""

    order_id: int
    symbol: str
    status: str
    side: str
    type: str
    price: float
    executed_quantity: float
    original_quantity: float
    time: int


class CancelOrderResponse(BaseModel):
    """Результат отмены ордера."""

    order_id: int
    symbol: str
    status: str
    client_order_id: str
    orig_client_order_id: str


class MarginBalance(BaseModel):
    """Баланс маржинального аккаунта по активу."""

    asset: str
    free: float
    locked: float
    borrowed: float
    interest: float
    net_asset: float


class MarginAccount(BaseModel):
    """Маржинальный аккаунт."""

    account_type: str
    margin_level: float
    total_asset_of_btc: float
    total_liability_of_btc: float
    total_net_asset_of_btc: float
    trade_enabled: bool
    borrow_enabled: bool
    transfer_enabled: bool
    balances: List[MarginBalance]


class MarginOrder(BaseModel):
    """Маржинальный ордер."""

    order_id: int
    symbol: str
    is_isolated: bool
    side: str
    type: str
    status: str
    price: Optional[float]
    orig_qty: float
    executed_qty: float
    client_order_id: str
    transact_time: int


class MarginTrade(BaseModel):
    """Маржинальная сделка."""

    trade_id: int
    symbol: str
    order_id: int
    price: float
    quantity: float
    commission: float
    commission_asset: str
    is_isolated: bool
    is_best_match: bool
    time: int


class CreateMarginOrderRequest(BaseModel):
    """Запрос на создание маржинального ордера."""

    symbol: str
    side: str
    type: str
    quantity: Optional[float] = None
    quote_order_qty: Optional[float] = None
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Optional[str] = None
    is_isolated: Optional[bool] = None
    side_effect_type: Optional[str] = None
    new_client_order_id: Optional[str] = None
    new_order_resp_type: Optional[str] = None


class CreateMarginOrderResponse(BaseModel):
    """Ответ создания маржинального ордера."""

    order_id: int
    symbol: str
    is_isolated: bool
    status: str
    client_order_id: str
    transact_time: int
    price: Optional[float]
    orig_qty: Optional[float]
    executed_qty: Optional[float]
    cummulative_quote_qty: Optional[float]


class CancelMarginOrderRequest(BaseModel):
    """Запрос на отмену маржинального ордера."""

    symbol: str
    order_id: Optional[int] = None
    client_order_id: Optional[str] = None
    new_client_order_id: Optional[str] = None
    is_isolated: Optional[bool] = None


class CancelMarginOrderResponse(BaseModel):
    """Ответ отмены маржинального ордера."""

    order_id: int
    symbol: str
    is_isolated: bool
    status: str
    client_order_id: str
    orig_client_order_id: str


class CreateMarginOrderBatchRequest(BaseModel):
    """Запрос на батчевое создание маржинальных ордеров."""

    orders: List[CreateMarginOrderRequest]


class CancelMarginOrderBatchRequest(BaseModel):
    """Запрос на батчевую отмену маржинальных ордеров."""

    orders: List[CancelMarginOrderRequest]


class OCOLeg(BaseModel):
    """Нога OCO ордера."""

    symbol: str
    order_id: int
    client_order_id: str
    price: Optional[float]
    orig_qty: Optional[float]
    executed_qty: Optional[float]
    status: Optional[str]
    side: Optional[str]
    type: Optional[str]
    stop_price: Optional[float] = None


class OCOOrder(BaseModel):
    """OCO ордер."""

    order_list_id: int
    symbol: str
    status: str
    list_status_type: str
    list_order_status: str
    list_client_order_id: str
    transact_time: int
    is_isolated: Optional[bool] = None
    legs: List[OCOLeg]


class ExchangeInfo(BaseModel):
    """Информация о бирже."""

    timezone: str
    server_time: int
    rate_limits: List[dict]
    symbols_count: int
    symbols: List[str]


class ServerTime(BaseModel):
    """Время сервера."""

    server_time: int
    epoch_seconds: float


class CreateOrderRequest(BaseModel):
    """Запрос на создание ордера."""

    symbol: str
    side: str  # BUY или SELL
    type: str  # MARKET, LIMIT, STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT, LIMIT_MAKER
    quantity: Optional[float] = None
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Optional[str] = "GTC"  # GTC, IOC, FOK
    new_client_order_id: Optional[str] = None
    iceberg_qty: Optional[float] = None
    new_order_resp_type: Optional[str] = "ACK"  # ACK, RESULT, FULL
    reduce_only: Optional[bool] = None
    close_position: Optional[bool] = None
    price_protect: Optional[bool] = None


class CreateOrderResponse(BaseModel):
    """Ответ на создание ордера."""

    order_id: int
    symbol: str
    side: str
    type: str
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str
    status: str
    client_order_id: str
    transact_time: int
    fills: Optional[List[dict]] = None


class CancelOrderRequest(BaseModel):
    """Запрос на отмену ордера."""

    symbol: str
    order_id: Optional[int] = None
    client_order_id: Optional[str] = None


class CreateOrderBatchRequest(BaseModel):
    """Запрос на батчевое создание ордеров."""

    orders: List[CreateOrderRequest]


class CancelOrderBatchRequest(BaseModel):
    """Запрос на батчевую отмену ордеров."""

    orders: List[CancelOrderRequest]


class Position(BaseModel):
    """Позиция в портфеле."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    pnl_usdt: float
    pnl_percent: float
    daily_change: float
    status: str


class PortfolioMetrics(BaseModel):
    """Метрики портфеля."""

    total_pnl_usdt: float
    total_pnl_percent: float
    best_performer: str
    worst_performer: str
    risk_score: str
    diversification_score: int


class AccountSummary(BaseModel):
    """Сводка аккаунта."""

    total_balance_usdt: float
    free_usdt: float
    invested_usdt: float
    investment_percentage: float


class PortfolioSummary(BaseModel):
    """Сводка портфеля."""

    account_summary: AccountSummary
    positions: List[Position]
    portfolio_metrics: PortfolioMetrics
    recommendations: List[str]


class FuturesPosition(BaseModel):
    """Позиция на фьючерсах."""

    symbol: str
    position_amount: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: float
    margin_type: str
    isolated_margin: float
    notional: float
    liquidation_price: float
    update_time: int


class FuturesLeverageChange(BaseModel):
    """Результат изменения плеча."""

    symbol: str
    leverage: int
    max_notional_value: float


class FuturesMarginTypeChange(BaseModel):
    """Результат изменения типа маржи."""

    symbol: str
    margin_type: str
    code: int
    message: str


# Batch operation models
class FuturesPositionsBatchRequest(BaseModel):
    """Запрос на батчевое получение позиций."""

    symbols: List[str]


class TickersBatchRequest(BaseModel):
    """Запрос на батчевое получение тикеров."""

    symbols: List[str]


class LeverageBatchRequest(BaseModel):
    """Запрос на батчевое изменение плеча."""

    symbol_leverage_map: Dict[str, int]


class CreatePositionRequest(BaseModel):
    """Запрос на создание позиции."""

    symbol: str
    side: str  # BUY or SELL
    quantity: float
    leverage: int


class CreatePositionsBatchRequest(BaseModel):
    """Запрос на батчевое создание позиций."""

    positions: List[CreatePositionRequest]


class ClosePositionsBatchRequest(BaseModel):
    """Запрос на батчевое закрытие позиций."""

    symbols: List[str]
    percentage: float = 100.0  # Процент закрытия позиции


class BatchResult(BaseModel):
    """Результат батчевой операции."""

    success: bool
    symbol: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PortfolioOverview(BaseModel):
    """Обзор портфолио."""

    balance: List[BalanceEntry]
    active_positions: List[FuturesPosition]
    total_pnl: float
    statistics: Dict[str, Any]


class TickerBatchItem(BaseModel):
    """Элемент батчевого тикера."""

    symbol: str
    price: float
    ticker_24h: Ticker24h


class AvailablePair(BaseModel):
    """Доступная торговая пара."""

    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    filters: Dict[str, Any]


class TradingLimitsInfo(BaseModel):
    """Информация о торговых лимитах."""

    symbol: str
    available: bool
    position: Optional[FuturesPosition]
    margin_available: float
    limits: Dict[str, Any]


# Risk Management Models
class SafetyRule(BaseModel):
    """Правило безопасности для позиции."""

    max_rsi_short: float = 30.0
    min_rsi_long: float = 75.0
    min_adx: float = 18.0
    max_drawdown: float = 5.0


class SafetyCheckResult(BaseModel):
    """Результат проверки безопасности позиции."""

    symbol: str
    is_safe: bool
    violations: List[str]
    rsi: Optional[float] = None
    adx: Optional[float] = None
    drawdown: Optional[float] = None
    recommendation: str


class StopLossConfig(BaseModel):
    """Конфигурация стоп-лосса."""

    stop_loss_type: str  # "fixed", "trailing", "atr"
    trail_percentage: float = 2.0
    update_frequency: str = "1h"
    max_loss_percent: float = 5.0


class StopLossResult(BaseModel):
    """Результат управления стоп-лоссом."""

    symbol: str
    action: str  # "created", "updated", "no_action"
    stop_price: Optional[float] = None
    reason: str


class AlertConfig(BaseModel):
    """Конфигурация алерта."""

    alert_type: (
        str  # "drawdown", "profit", "volume_spike", "price_level", "rsi_extreme"
    )
    threshold: float
    notification_method: str = "telegram"  # "telegram", "email", "webhook"
    message_template: Optional[str] = None  # Кастомный шаблон сообщения


class AlertResult(BaseModel):
    """Результат алерта."""

    symbol: str
    alert_type: str
    triggered: bool
    current_value: float
    threshold: float
    notification_sent: bool
    message: str
    timestamp: str


class TelegramNotification(BaseModel):
    """Уведомление в Telegram."""

    chat_id: str
    message: str
    parse_mode: str = "Markdown"
    disable_notification: bool = False


class RiskManagementRule(BaseModel):
    """Правило управления рисками."""

    max_portfolio_loss: float = -10.0
    max_position_loss: float = -5.0
    profit_taking: float = 15.0
    auto_close_on_loss: bool = True
