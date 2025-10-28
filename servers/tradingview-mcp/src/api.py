"""FastAPI приложение с MCP интеграцией для TradingView MCP."""

import asyncio
from datetime import datetime
from importlib import metadata
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from fastapi_mcp import FastApiMCP

from tradingview_mcp.config import get_settings

import logging
logger = logging.getLogger(__name__)

# Импортируем MCP server для интеграции
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mcp_server import server as mcp_server


# Pydantic модели для запросов
class ScannerRequest(BaseModel):
    """Модель запроса для сканеров с параметрами."""
    symbols: List[str] = Field(
        ...,
        description="Список торговых пар для анализа (например: ['BTCUSDT', 'ETHUSDT'])",
        example=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    )
    profile: str = Field(
        "balanced",
        description="Профиль риска: 'conservative', 'balanced', 'aggressive'",
        example="balanced"
    )
    exchange: Optional[str] = Field(
        "KUCOIN",
        description="Название биржи (BINANCE, KUCOIN, BYBIT и т.д.)",
        example="KUCOIN"
    )
    timeframe: Optional[str] = Field(
        "1h",
        description="Таймфрейм анализа (1m, 5m, 15m, 1h, 4h, 1d)",
        example="1h"
    )
    limit: Optional[int] = Field(
        50,
        description="Максимальное количество результатов",
        example=50
    )


class BatchRequest(BaseModel):
    """Модель запроса для batch операций."""
    requests: List[Dict[str, Any]] = Field(
        ...,
        description="Список запросов для batch обработки",
        example=[{"exchange": "KUCOIN", "limit": 10}]
    )


class BacktestRequest(BaseModel):
    """Модель запроса для backtest."""
    symbols: List[str] = Field(..., description="Список торговых пар для backtest")
    profile: str = Field("balanced", description="Профиль риска")
    timeframe: str = Field("15m", description="Таймфрейм для анализа")
    lookback_limit: int = Field(500, description="Количество свечей для анализа")
    strategy: str = Field("momentum", description="Тип стратегии: 'momentum', 'mean_revert', 'breakout', 'volume_profile'")


class SnapshotRequest(BaseModel):
    """Модель запроса для snapshot метрик."""
    symbols: List[str] = Field(..., description="Список торговых пар")
    scanner_type: str = Field("momentum", description="Тип сканера")
    profile: str = Field("balanced", description="Профиль риска")


def create_app() -> FastAPI:
    """Создает и настраивает FastAPI приложение."""
    try:
        settings = get_settings()
        title = "TradingView MCP"
        description = f"MCP сервер для анализа криптовалютного рынка. Режим: {settings.default_exchange}"
    except Exception:
        title = "TradingView MCP"
        description = "MCP сервер для анализа криптовалютного рынка"

    app = FastAPI(
        title=title,
        version="0.1.0",
        description=description,
    )

    # Регистрируем эндпоинты
    register_routes(app)

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("TradingView MCP FastAPI startup")
        # Инициализируем сервис сканеров
        from tradingview_mcp.server import _ensure_scanner_service, _shutdown_scanner_service
        await _ensure_scanner_service()
        
        # Запускаем планировщик сканеров
        from tradingview_mcp.scheduler import scheduler
        try:
            await scheduler.start()
            logger.info("Scanner scheduler started successfully")
        except Exception as exc:
            logger.error(f"Failed to start scheduler: {exc}")
    
    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("TradingView MCP FastAPI shutdown")
        # Останавливаем планировщик
        from tradingview_mcp.scheduler import scheduler
        try:
            await scheduler.stop()
            logger.info("Scanner scheduler stopped")
        except Exception as exc:
            logger.error(f"Failed to stop scheduler: {exc}")
        
        # Останавливаем сервис сканеров
        from tradingview_mcp.server import _shutdown_scanner_service
        await _shutdown_scanner_service()

    # Настраиваем MCP интеграцию  (создаст свой MCP server)
    mcp = FastApiMCP(app, name="tradingview-mcp")
    mcp.mount_http()

    return app


def register_routes(app: FastAPI) -> None:
    """Регистрирует все маршруты (endpoints) из оригинального server.py как FastAPI endpoints."""
    
    # Импортируем ВСЕ функции из оригинального server напрямую
    import tradingview_mcp.server as server
    
    # Meta tools
    @app.get("/health", operation_id="get_health")
    def health() -> dict:
        """Check MCP server health and configuration status."""
        return server.server_info()
    
    @app.get("/version", operation_id="get_version")
    def version() -> dict:
        """Return server version and enabled capabilities."""
        return server.server_info()
    
    @app.get("/exchanges_list", operation_id="list_exchanges")
    def exchanges_list() -> dict:
        """List available cryptocurrency exchanges."""
        exchanges = ["BINANCE", "KUCOIN", "BYBIT", "OKX", "COINBASE", "KRAKEN", "HUOBI"]
        return {"exchanges": exchanges}
    
    # Professional scanners
    @app.get("/pro_scanner_profiles", operation_id="list_profiles")
    def pro_profiles() -> list:
        """List professional scanner profiles and parameters."""
        return server.pro_scanner_profiles()
    
    @app.post("/pro_momentum_scan", operation_id="scan_momentum")
    async def pro_momentum(request: ScannerRequest) -> list:
        """Run momentum scanner for given symbols."""
        return await server.pro_momentum_scan(request.symbols or [], request.profile)
    
    @app.post("/pro_mean_revert_scan", operation_id="scan_mean_revert")
    async def pro_mean_revert(request: ScannerRequest) -> list:
        """Run mean reversion scanner for given symbols."""
        return await server.pro_mean_revert_scan(request.symbols or [], request.profile)
    
    @app.post("/pro_breakout_scan", operation_id="scan_breakout")
    async def pro_breakout(request: ScannerRequest) -> list:
        """Run breakout scanner for given symbols."""
        return await server.pro_breakout_scan(request.symbols or [], request.profile)
    
    @app.post("/pro_volume", operation_id="scan_volume_profile")
    async def volume(request: ScannerRequest) -> list:
        """Run volume profile scanner for given symbols."""
        return await server.pro_volume_profile_scan(request.symbols or [], request.profile)
    
    @app.post("/pro_backtest", operation_id="run_backtest")
    async def backtest(request: BacktestRequest) -> dict:
        """Run historical backtest on professional scanners."""
        return await server.pro_scanner_backtest(
            symbols=request.symbols,
            profile=request.profile,
            timeframe=request.timeframe,
            lookback_limit=request.lookback_limit,
            strategy=request.strategy
        )
    
    @app.get("/pro_metrics", operation_id="get_scanner_metrics")
    async def metrics(period_days: int = 14, strategy: Optional[str] = None) -> list:
        """Get professional scanner performance metrics."""
        return await server.pro_scanner_metrics(period_days, strategy)
    
    @app.post("/pro_snapshot", operation_id="get_metrics_snapshot")
    async def snapshot(request: SnapshotRequest) -> dict:
        """Get snapshot of scanner metrics for specific symbols."""
        return await server.pro_scanner_metrics_snapshot(
            request.symbols,
            request.scanner_type,
            request.profile
        )
    
    @app.get("/pro_signals", operation_id="get_recent_signals")
    async def signals(limit: int = 20) -> list:
        """Return most recent signals recorded by the pro scanners."""
        return await server.pro_scanner_recent_signals(limit)
    
    @app.post("/pro_feedback", operation_id="submit_feedback")
    async def feedback(signal_id: int, action: str) -> dict:
        """Record feedback on a professional scanner signal."""
        return await server.pro_scanner_feedback(signal_id, action)
    
    @app.delete("/pro_cache", operation_id="clear_cache")
    async def cache_clear(namespace: str = None) -> dict:
        """Clear professional scanner cache."""
        return await server.pro_scanner_cache_clear(namespace)
    
    @app.get("/pro_results", operation_id="get_recent_results")
    async def results(
        scanner_name: str = None,
        limit: int = 10
    ) -> list:
        """Get recently cached scanner results."""
        return await server.pro_scanner_get_recently_results(scanner_name, limit)
    
    @app.get("/pro_status", operation_id="get_scheduler_status")
    def status() -> dict:
        """Check professional scanner scheduler status."""
        return server.pro_scanner_scheduler_status()
    
    # Market analysis
    @app.get("/derivatives", operation_id="get_derivatives_context")
    async def derivatives(symbol: str) -> dict:
        """Get derivatives context for a symbol."""
        return await server.get_derivatives_context(symbol)
    
    @app.post("/top_gainers", operation_id="get_top_gainers")
    def gainers(request: BatchRequest) -> list:
        """Run multiple top_gainers presets in a single call."""
        return server.top_gainers_batch(request.requests)
    
    @app.post("/multi_changes", operation_id="get_multi_changes")
    def changes(request: BatchRequest) -> list:
        """Run multiple multi_changes presets in a single call."""
        return server.multi_changes_batch(request.requests)
    
    @app.post("/top_losers", operation_id="get_top_losers")
    def losers(request: BatchRequest) -> list:
        """Run multiple top_losers presets in a single call."""
        return server.top_losers_batch(request.requests)
    
    @app.post("/bollinger_batch", operation_id="scan_bollinger")
    def bollinger(request: BatchRequest) -> list:
        """Run multiple bollinger scans in a single call."""
        return server.bollinger_scan_batch(request.requests)
    
    @app.post("/coin_batch", operation_id="analyze_coins")
    def coins(request: BatchRequest) -> list:
        """Execute coin analysis for multiple symbols in one call."""
        return server.coin_analysis_batch(request.requests)
    
    @app.post("/candles_batch", operation_id="scan_candles")
    def candles(request: BatchRequest) -> list:
        """Run multiple consecutive candles scans in a single call."""
        return server.consecutive_candles_scan_batch(request.requests)
    
    @app.post("/patterns_batch", operation_id="scan_patterns")
    def patterns(request: BatchRequest) -> list:
        """Run multiple advanced candle pattern scans in a single call."""
        return server.advanced_candle_pattern_batch(request.requests)
    
    # Strategy tools
    @app.get("/trend_breakout", operation_id="scan_trend_breakout")
    def breakout(
        exchange: str = Query("KUCOIN", description="Название биржи (BINANCE, KUCOIN, BYBIT)"),
        timeframe: str = Query("1h", description="Таймфрейм анализа (1m, 5m, 15m, 1h, 4h, 1d)"),
        btc_symbol: str = Query("BINANCE:BTCUSDT", description="BTC пара для контекста"),
        quote: Optional[str] = Query(None, description="Фильтр по котируемой валюте (например: USDT)"),
        adx_min: float = Query(10.0, description="Минимальный ADX для тренда"),
        kc_mult: float = Query(1.5, description="Множитель Keltner Channel"),
        kc_min_width: float = Query(0.015, description="Минимальная ширина канала"),
        atr_k: float = Query(1.4, description="Коэффициент ATR"),
        pb_long_min: float = Query(0.95, description="Минимум для длинных позиций"),
        pb_short_max: float = Query(0.05, description="Максимум для коротких позиций"),
        sl_mult: float = Query(1.5, description="Множитель стоп-лосса"),
        tp1_rr: float = Query(1.75, description="Risk/Reward для первой цели"),
        trail_mult: float = Query(2.0, description="Множитель трейлинг-стопа"),
        add_step_atr: float = Query(1.0, description="Шаг добавления позиции в ATR"),
        max_adds: int = Query(2, description="Максимальное количество добавок"),
        add_fraction: float = Query(0.7, description="Доля размера при добавке"),
        equity: float = Query(10000.0, description="Размер капитала"),
        risk_pct: float = Query(0.02, description="Процент риска на сделку"),
        limit: int = Query(20, description="Максимальное количество результатов")
    ) -> list:
        """Trend breakout pyramiding strategy scanner."""
        return server.trend_breakout_pyramiding(
            exchange=exchange,
            timeframe=timeframe,
            btc_symbol=btc_symbol,
            quote=quote,
            adx_min=adx_min,
            kc_mult=kc_mult,
            kc_min_width=kc_min_width,
            atr_k=atr_k,
            pb_long_min=pb_long_min,
            pb_short_max=pb_short_max,
            sl_mult=sl_mult,
            tp1_rr=tp1_rr,
            trail_mult=trail_mult,
            add_step_atr=add_step_atr,
            max_adds=max_adds,
            add_fraction=add_fraction,
            equity=equity,
            risk_pct=risk_pct,
            limit=limit
        )
    
    @app.get("/pullback", operation_id="scan_pullback")
    def pullback(
        exchange: str = Query("KUCOIN", description="Название биржи"),
        timeframe: str = Query("1h", description="Таймфрейм анализа"),
        quote: Optional[str] = Query(None, description="Фильтр по котируемой валюте"),
        ema_fast: int = Query(20, description="Быстрая EMA"),
        ema_slow: int = Query(50, description="Медленная EMA"),
        btc_symbol: str = Query("BINANCE:BTCUSDT", description="BTC пара для контекста"),
        adx_min: float = Query(8.0, description="Минимальный ADX"),
        adx_max: float = Query(50.0, description="Максимальный ADX"),
        rsi_len: int = Query(14, description="Период RSI"),
        rsi_long_th: float = Query(45.0, description="Порог RSI для длинных позиций"),
        rsi_short_th: float = Query(55.0, description="Порог RSI для коротких позиций"),
        b_long_th: float = Query(0.40, description="Порог %b для длинных позиций"),
        b_short_th: float = Query(0.60, description="Порог %b для коротких позиций"),
        sl_mult: float = Query(1.5, description="Множитель стоп-лосса"),
        tp1_rr: float = Query(1.5, description="Risk/Reward"),
        trail_mult: float = Query(2.0, description="Множитель трейлинг-стопа"),
        enable_retest_add: bool = Query(True, description="Разрешить добавку на retest"),
        retest_fraction: float = Query(0.5, description="Доля при retest"),
        equity: float = Query(10000.0, description="Размер капитала"),
        risk_pct: float = Query(0.02, description="Процент риска"),
        limit: int = Query(20, description="Максимальное количество результатов")
    ) -> list:
        """Pullback engine scanner for trend continuation opportunities."""
        return server.pullback_engine(
            exchange=exchange,
            timeframe=timeframe,
            quote=quote,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            btc_symbol=btc_symbol,
            adx_min=adx_min,
            adx_max=adx_max,
            rsi_len=rsi_len,
            rsi_long_th=rsi_long_th,
            rsi_short_th=rsi_short_th,
            b_long_th=b_long_th,
            b_short_th=b_short_th,
            sl_mult=sl_mult,
            tp1_rr=tp1_rr,
            trail_mult=trail_mult,
            enable_retest_add=enable_retest_add,
            retest_fraction=retest_fraction,
            equity=equity,
            risk_pct=risk_pct,
            limit=limit
        )
    
    @app.get("/unified", operation_id="scan_unified")
    def unified(
        exchange: str = Query("KUCOIN", description="Название биржи"),
        timeframe: str = Query("1h", description="Таймфрейм анализа"),
        direction: str = Query("both", description="Направление: 'long', 'short', 'both'"),
        quote: Optional[str] = Query(None, description="Фильтр по котируемой валюте"),
        strategy_mode: str = Query("adaptive", description="Режим стратегии: 'adaptive', 'breakout', 'pullback', 'smart'"),
        min_volume: Optional[float] = Query(None, description="Минимальный объем"),
        min_rsi: Optional[float] = Query(None, description="Минимальный RSI"),
        max_rsi: Optional[float] = Query(None, description="Максимальный RSI"),
        min_adx: Optional[float] = Query(None, description="Минимальный ADX"),
        max_adx: Optional[float] = Query(None, description="Максимальный ADX"),
        max_bbw: Optional[float] = Query(None, description="Максимальная ширина Bollinger Bands"),
        adx_min: Optional[float] = Query(None, description="Минимальный ADX для пробоя"),
        kc_min_width: Optional[float] = Query(None, description="Минимальная ширина Keltner Channel"),
        atr_k: Optional[float] = Query(None, description="Коэффициент ATR"),
        pb_long_min: Optional[float] = Query(None, description="Минимум для длинных позиций"),
        pb_short_max: Optional[float] = Query(None, description="Максимум для коротких позиций"),
        include_breakout: bool = Query(True, description="Включать стратегию пробоя"),
        include_pullback: bool = Query(True, description="Включать стратегию отката"),
        limit: int = Query(30, description="Общий лимит результатов"),
        per_strategy_limit: int = Query(40, description="Лимит на стратегию")
    ) -> dict:
        """Unified scanner combining smart filters and strategy candidates."""
        return server.unified_scanner(
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            quote=quote,
            strategy_mode=strategy_mode,
            min_volume=min_volume,
            min_rsi=min_rsi,
            max_rsi=max_rsi,
            min_adx=min_adx,
            max_adx=max_adx,
            max_bbw=max_bbw,
            adx_min=adx_min,
            kc_min_width=kc_min_width,
            atr_k=atr_k,
            pb_long_min=pb_long_min,
            pb_short_max=pb_short_max,
            include_breakout=include_breakout,
            include_pullback=include_pullback,
            limit=limit,
            per_strategy_limit=per_strategy_limit
        )
    
    @app.get("/strategy", operation_id="find_strategy_candidates")
    def strategy(
        exchange: str = Query("KUCOIN", description="Название биржи"),
        timeframe: str = Query("1h", description="Таймфрейм анализа"),
        direction: str = Query("both", description="Направление: 'long', 'short', 'both'"),
        quote: Optional[str] = Query(None, description="Фильтр по котируемой валюте"),
        adx_min: Optional[float] = Query(None, description="Минимальный ADX"),
        kc_min_width: Optional[float] = Query(None, description="Минимальная ширина Keltner Channel"),
        atr_k: Optional[float] = Query(None, description="Коэффициент ATR"),
        pb_long_min: Optional[float] = Query(None, description="Минимум для длинных позиций"),
        pb_short_max: Optional[float] = Query(None, description="Максимум для коротких позиций"),
        include_breakout: bool = Query(True, description="Включать стратегию пробоя"),
        include_pullback: bool = Query(True, description="Включать стратегию отката"),
        limit: int = Query(30, description="Общий лимит результатов"),
        per_strategy_limit: int = Query(40, description="Лимит на стратегию")
    ) -> dict:
        """Find strategy candidates based on technical criteria."""
        return server.strategy_candidates(
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            quote=quote,
            adx_min=adx_min,
            kc_min_width=kc_min_width,
            atr_k=atr_k,
            pb_long_min=pb_long_min,
            pb_short_max=pb_short_max,
            include_breakout=include_breakout,
            include_pullback=include_pullback,
            limit=limit,
            per_strategy_limit=per_strategy_limit
        )
    
    @app.get("/scan_strategy", operation_id="scan_find_strategy_candidates")
    def scan(
        exchange: str = Query("BINANCE", description="Название биржи"),
        timeframe: str = Query("1h", description="Таймфрейм анализа"),
        limit: int = Query(50, description="Максимальное количество результатов"),
        strategy_type: str = Query("momentum", description="Тип стратегии: 'momentum', 'mean_revert', 'breakout'")
    ) -> dict:
        """Scan for strategy candidates with specific strategy type."""
        return server.scan_strategy_candidates(exchange, timeframe, limit, strategy_type)
    
    @app.get("/smart", operation_id="scan_smart")
    def smart(
        exchange: str = Query("KUCOIN", description="Название биржи"),
        timeframe: str = Query("1h", description="Таймфрейм анализа"),
        quote: Optional[str] = Query(None, description="Фильтр по котируемой валюте"),
        min_volume: Optional[float] = Query(None, description="Минимальный объем"),
        min_rsi: Optional[float] = Query(None, description="Минимальный RSI"),
        max_rsi: Optional[float] = Query(None, description="Максимальный RSI"),
        min_adx: Optional[float] = Query(None, description="Минимальный ADX"),
        max_adx: Optional[float] = Query(None, description="Максимальный ADX"),
        max_bbw: Optional[float] = Query(None, description="Максимальная ширина Bollinger Bands"),
        direction: str = Query("both", description="Направление: 'long', 'short', 'both'"),
        limit: int = Query(30, description="Максимальное количество результатов")
    ) -> list:
        """Smart scanner with adaptive parameters."""
        return server.smart_scanner(
            exchange=exchange,
            timeframe=timeframe,
            quote=quote,
            min_volume=min_volume,
            min_rsi=min_rsi,
            max_rsi=max_rsi,
            min_adx=min_adx,
            max_adx=max_adx,
            max_bbw=max_bbw,
            direction=direction,
            limit=limit
        )
    
    logger.info("Registered all FastAPI endpoints from original server.py successfully")


# Создаем приложение
app = create_app()
