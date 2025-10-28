"""FastAPI integration for backtesting-mcp HTTP server."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from fastapi_mcp import FastApiMCP
from .config import get_settings
from .services.backtesting import BacktestingService
from .services.health import HealthService
from .datasource import MarketDataSource

logger = logging.getLogger(__name__)

# Pydantic models for requests
class BacktestRequest(BaseModel):
    """Модель запроса для бэктеста."""
    strategy: str = Field(..., description="Название стратегии (ma_crossover)")
    symbol: str = Field(..., description="Торговая пара (BTCUSDT, ETHUSDT и т.д.)")
    timeframe: str = Field("1h", description="Таймфрейм (1m, 15m, 1h, 4h, 1d)")
    start: str = Field("2025-01-01T00:00:00", description="Начало периода (ISO 8601)")
    end: str = Field("2025-12-31T23:59:59", description="Конец периода (ISO 8601)")
    parameters: Optional[Dict[str, float]] = Field(None, description="Параметры стратегии")


class OptimizeRequest(BaseModel):
    """Модель запроса для оптимизации."""
    strategy: str = Field(..., description="Название стратегии")
    symbol: str = Field(..., description="Торговая пара")
    timeframe: str = Field("1h", description="Таймфрейм")
    start: str = Field("2025-01-01T00:00:00", description="Начало периода")
    end: str = Field("2025-12-31T23:59:59", description="Конец периода")
    parameter_space: Optional[Dict[str, List[int]]] = Field(None, description="Пространство параметров")
    objective: str = Field("return", description="Целевая функция (return/sharpe)")
    trials: int = Field(20, description="Количество попыток (1-1000)", ge=1, le=1000)


class CompareRequest(BaseModel):
    """Модель запроса для сравнения стратегий."""
    results: List[Dict[str, Any]] = Field(..., description="Список результатов бэктестов")


def _parse_datetime(value: str) -> datetime:
    """Parse ISO datetime string."""
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def create_app() -> FastAPI:
    """Create and configure FastAPI application for HTTP transport."""
    settings = get_settings()
    
    app = FastAPI(
        title="Backtesting MCP",
        version="0.2.0",
        description="MCP сервер для бэктестинга и оптимизации торговых стратегий"
    )
    
    # Initialize services
    data_source = MarketDataSource()
    optuna_storage_url = settings.get_optuna_storage_url()
    backtesting_service = BacktestingService(
        data_source,
        optuna_storage_url=optuna_storage_url,
        sampler_name=settings.optuna_sampler,
        pruner_name=settings.optuna_pruner,
        n_jobs=settings.optuna_n_jobs,
    )
    health_service = HealthService()
    
    # Register endpoints
    @app.post("/run_backtest", operation_id="run_backtest")
    async def run_backtest(request: BacktestRequest) -> dict:
        """Run a trading strategy backtest on historical market data."""
        if request.parameters is None:
            request.parameters = {}
        
        try:
            result = backtesting_service.run_backtest(
                strategy=request.strategy,
                symbol=request.symbol,
                timeframe=request.timeframe,
                start=_parse_datetime(request.start),
                end=_parse_datetime(request.end),
                parameters=request.parameters,
            )
            
            metrics = {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "trades": result.trades,
            }
            metrics.update(result.extra_metrics)
            
            return {
                "strategy": result.strategy,
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "period": [result.start.isoformat(), result.end.isoformat()],
                "parameters": result.params,
                "metrics": metrics,
                "equity_curve": result.equity_curve,
                "trade_log": result.trades_log,
            }
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise ValueError(f"Failed to run backtest: {e}")
    
    @app.post("/optimize", operation_id="optimize_params")
    async def optimize(request: OptimizeRequest) -> dict:
        """Optimize strategy parameters using Optuna."""
        if request.parameter_space is None:
            request.parameter_space = {"fast": [5, 10, 15], "slow": [20, 30, 40]}
        
        try:
            result = backtesting_service.optimize_parameters(
                strategy=request.strategy,
                symbol=request.symbol,
                timeframe=request.timeframe,
                start=_parse_datetime(request.start),
                end=_parse_datetime(request.end),
                parameter_space=request.parameter_space,
                objective=request.objective,
                trials=request.trials,
            )
            return result
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise ValueError(f"Failed to optimize parameters: {e}")
    
    @app.post("/compare", operation_id="compare_strategies")
    async def compare(request: CompareRequest) -> dict:
        """Compare multiple backtest results."""
        from .services.backtesting import BacktestResult
        
        try:
            bt_results = []
            for item in request.results:
                bt_results.append(
                    BacktestResult(
                        strategy=item["strategy"],
                        symbol=item["symbol"],
                        timeframe=item.get("timeframe", settings.default_timeframe),
                        start=_parse_datetime(item.get("start", "2025-01-01T00:00:00")),
                        end=_parse_datetime(item.get("end", "2025-12-31T23:59:59")),
                        params=item.get("parameters", {}),
                        total_return=float(item.get("total_return", item.get("metrics", {}).get("total_return", 0.0))),
                        annualized_return=float(item.get("annualized_return", item.get("metrics", {}).get("annualized_return", 0.0))),
                        sharpe=float(item.get("sharpe", item.get("metrics", {}).get("sharpe", 0.0))),
                        max_drawdown=float(item.get("max_drawdown", item.get("metrics", {}).get("max_drawdown", 0.0))),
                        trades=int(item.get("trades", item.get("metrics", {}).get("trades", 0))),
                        equity_curve=item.get("equity_curve", []),
                        extra_metrics=item.get("metrics", {}),
                        trades_log=item.get("trade_log", []),
                    )
                )
            
            result = backtesting_service.compare_strategies(bt_results)
            return result
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            raise ValueError(f"Failed to compare strategies: {e}")
    
    @app.get("/health_check", operation_id="check_health")
    async def health_check(check_data_sources: bool = Query(True, description="Проверять источники данных")) -> dict:
        """Check server health and external data source availability."""
        import time
        try:
            result = health_service.check_health(check_data_sources=check_data_sources)
            result["uptime"] = time.time() - time.time()
            return result
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    @app.get("/version", operation_id="get_version")
    async def version() -> dict:
        """Return server version information and configuration details."""
        masked_optuna_storage = settings.get_optuna_storage_url().replace(settings.optuna_db_password, "***")
        return {
            "version": "0.2.0",
            "package": "backtesting-mcp",
            "description": "Model Context Protocol server for running strategy backtests and optimizations",
            "config": {
                "default_timeframe": settings.default_timeframe,
                "preferred_source": settings.preferred_source,
                "optuna_sampler": settings.optuna_sampler,
                "optuna_pruner": settings.optuna_pruner,
                "optuna_storage": masked_optuna_storage,
            },
            "features": [
                "real_market_data_integration",
                "optuna_parameter_optimization",
                "persistent_optuna_storage",
                "strategy_comparison",
                "multiple_data_sources"
            ]
        }
    
    # Health endpoint
    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "name": "backtesting-mcp",
            "version": "0.2.0"
        }
    
    # Integrate MCP with FastAPI
    mcp = FastApiMCP(app, name="backtesting-mcp")
    mcp.mount_http()
    
    @app.on_event("startup")
    async def startup() -> None:
        logger.info("Backtesting MCP FastAPI startup")
    
    @app.on_event("shutdown")
    async def shutdown() -> None:
        logger.info("Backtesting MCP FastAPI shutdown")
    
    logger.info("FastAPI app created successfully")
    return app

