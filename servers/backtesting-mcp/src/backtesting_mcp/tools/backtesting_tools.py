"""Backtesting tools for MCP server."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Annotated

from pydantic import Field

from mcp.server.fastmcp import FastMCP

from ..services.backtesting import BacktestingService
from ..services.health import HealthService
from ..datasource import MarketDataSource
from ..config import get_settings

logger = logging.getLogger(__name__)


def _parse_datetime(value: str) -> datetime:
    """Parse ISO datetime string."""
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def register_backtesting_tools(mcp: FastMCP, start_time: float) -> None:
    """Register backtesting tools with the MCP server."""
    
    settings = get_settings()
    data_source = MarketDataSource()
    optuna_storage_url = settings.get_optuna_storage_url()
    masked_optuna_storage = optuna_storage_url.replace(settings.optuna_db_password, "***")
    backtesting_service = BacktestingService(
        data_source,
        optuna_storage_url=optuna_storage_url,
        sampler_name=settings.optuna_sampler,
        pruner_name=settings.optuna_pruner,
        n_jobs=settings.optuna_n_jobs,
    )
    health_service = HealthService()
    
    @mcp.tool(
        name="run_backtest",
        description="Run a trading strategy backtest on historical market data."
    )
    def run_backtest(
        strategy: Annotated[
            str,
            Field(
                description="Trading strategy name. Currently supported: 'ma_crossover'.",
                examples=["ma_crossover"],
            ),
        ],
        symbol: Annotated[
            str,
            Field(
                description="Trading symbol in base-quote format (uppercase).",
                examples=["BTCUSDT", "ETHUSDT"],
            ),
        ],
        timeframe: Annotated[
            str,
            Field(
                description="Candle timeframe (e.g. 1m, 15m, 1h, 4h, 1d).",
                examples=["1h"],
            ),
        ] = "1h",
        start: Annotated[
            str,
            Field(
                description="Start date/time in ISO 8601 format.",
                examples=["2025-01-01T00:00:00"],
            ),
        ] = "2025-01-01T00:00:00",
        end: Annotated[
            str,
            Field(
                description="End date/time in ISO 8601 format.",
                examples=["2025-12-31T23:59:59"],
            ),
        ] = "2025-12-31T23:59:59",
        parameters: Annotated[
            Dict[str, float] | None,
            Field(
                description="Strategy parameters, e.g. {'fast': 10, 'slow': 20}.",
            ),
        ] = None,
    ) -> Dict[str, Any]:
        """Run a backtest for a trading strategy on historical data.
        
        Executes a Moving Average Crossover strategy on the specified symbol and timeframe.
        Uses real market data from connected MCP servers (Binance, TradingView).
        
        Parameters:
        - strategy (str): Trading strategy name. Currently supported: 'ma_crossover' (Moving Average Crossover). Example: 'ma_crossover'
        - symbol (str): Trading symbol in base-quote format. Examples: 'BTCUSDT', 'ETHUSDT', 'ADAUSDT'. Must be uppercase.
        - timeframe (str): Candle timeframe. Supported values: '1m', '5m', '15m', '1h', '4h', '1d'. Default: '1h'
        - start (str): Start date for backtest in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). Example: '2025-01-01T00:00:00'
        - end (str): End date for backtest in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). Example: '2025-12-31T23:59:59'
        - parameters (Dict[str, float]): Strategy-specific parameters. For 'ma_crossover': {'fast': 10, 'slow': 20} where fast/slow are MA periods. Example: {'fast': 10, 'slow': 20}
        
        Returns:
        - Dict[str, Any]: Comprehensive backtest results including metrics, equity curve, and trade log
        """
        if parameters is None:
            parameters = {}
            
        logger.info(f"Running backtest: {strategy} on {symbol}")
        
        try:
            result = backtesting_service.run_backtest(
                strategy=strategy,
                symbol=symbol,
                timeframe=timeframe,
                start=_parse_datetime(start),
                end=_parse_datetime(end),
                parameters=parameters,
            )
            
            metrics = {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "trades": result.trades,
            }
            metrics.update(result.extra_metrics)
            
            response = {
                "strategy": result.strategy,
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "period": [result.start.isoformat(), result.end.isoformat()],
                "parameters": result.params,
                "metrics": metrics,
                "equity_curve": result.equity_curve,
                "trade_log": result.trades_log,
            }
            
            logger.info(f"Backtest completed: {result.total_return:.2%} return, {result.sharpe:.2f} Sharpe")
            return response
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise ValueError(f"Failed to run backtest: {e}")
    
    @mcp.tool(
        name="optimize_parameters",
        description="Optimize strategy parameters with an Optuna study."
    )
    def optimize_parameters(
        strategy: Annotated[
            str,
            Field(
                description="Trading strategy name. Currently supported: 'ma_crossover'.",
                examples=["ma_crossover"],
            ),
        ],
        symbol: Annotated[
            str,
            Field(
                description="Trading symbol in base-quote format (uppercase).",
                examples=["BTCUSDT", "ETHUSDT"],
            ),
        ],
        timeframe: Annotated[
            str,
            Field(
                description="Candle timeframe (e.g. 1m, 15m, 1h, 4h, 1d).",
                examples=["1h"],
            ),
        ] = "1h",
        start: Annotated[
            str,
            Field(
                description="Optimization start time in ISO 8601 format.",
                examples=["2025-01-01T00:00:00"],
            ),
        ] = "2025-01-01T00:00:00",
        end: Annotated[
            str,
            Field(
                description="Optimization end time in ISO 8601 format.",
                examples=["2025-12-31T23:59:59"],
            ),
        ] = "2025-12-31T23:59:59",
        parameter_space: Annotated[
            Dict[str, List[int]] | None,
            Field(
                description="Search space for parameters (e.g. {'fast': [5,10,15]}).",
            ),
        ] = None,
        objective: Annotated[
            str,
            Field(
                description="Optimization objective: 'return' or 'sharpe'.",
                examples=["return"],
            ),
        ] = "return",
        trials: Annotated[
            int,
            Field(
                description="Number of Optuna trials to run (1-1000).",
                ge=1,
                le=1000,
                examples=[20],
            ),
        ] = 20,
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using an Optuna-powered search.
        
        Runs an Optuna study (TPE sampler by default) over the provided parameter
        space and objective to identify the best-performing configuration. Results
        are persisted to the configured PostgreSQL storage so repeated calls reuse
        the same study history when possible.
        
        Parameters:
        - strategy (str): Trading strategy name. Currently supported: 'ma_crossover' (Moving Average Crossover). Example: 'ma_crossover'
        - symbol (str): Trading symbol in base-quote format. Examples: 'BTCUSDT', 'ETHUSDT', 'ADAUSDT'. Must be uppercase.
        - timeframe (str): Candle timeframe. Supported values: '1m', '5m', '15m', '1h', '4h', '1d'. Default: '1h'
        - start (str): Start date for optimization in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). Example: '2025-01-01T00:00:00'
        - end (str): End date for optimization in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). Example: '2025-12-31T23:59:59'
        - parameter_space (Dict[str, List[int]]): Parameter space for optimization. Example: {'fast': [5, 10, 15], 'slow': [20, 30, 40]}
        - objective (str): Optimization objective. Supported values: 'return' (total return), 'sharpe' (Sharpe ratio). Default: 'return'
        - trials (int): Number of optimization trials. Range: 1-1000. Default: 20
        Returns:
        - Dict[str, Any]: Optimization results including best parameters, score, trial history,
          and Optuna metadata (study name, sampler, storage details)
        """
        if parameter_space is None:
            parameter_space = {"fast": [5, 10, 15], "slow": [20, 30, 40]}
            
        logger.info(f"Optimizing parameters for {strategy} on {symbol}")
        
        try:
            result = backtesting_service.optimize_parameters(
                strategy=strategy,
                symbol=symbol,
                timeframe=timeframe,
                start=_parse_datetime(start),
                end=_parse_datetime(end),
                parameter_space=parameter_space,
                objective=objective,
                trials=trials,
            )
            
            best_score = result.get("best_score")
            if best_score is not None:
                logger.info(f"Optimization completed: best score {best_score:.4f}")
            else:
                logger.info("Optimization completed: no successful trials recorded.")
            return result
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            raise ValueError(f"Failed to optimize parameters: {e}")
    
    @mcp.tool(
        name="compare_strategies",
        description="Compare multiple backtest results and build a performance leaderboard."
    )
    def compare_strategies(
        results: Annotated[List[Dict[str, Any]], "List of backtest results to compare. Each result should contain at least 'strategy', 'symbol', and 'metrics' fields."]
    ) -> Dict[str, Any]:
        """Compare multiple backtest results and create a performance leaderboard.
        
        Analyzes a list of backtest results and ranks them by performance metrics.
        Handles results from run_backtest tool or external systems gracefully.
        Uses default values for missing metrics to ensure fair comparison.
        
        Parameters:
        - results (List[Dict[str, Any]]): List of backtest results to compare. Each result should contain at least 'strategy', 'symbol', and 'metrics' fields.
        
        Returns:
        - Dict[str, Any]: Comparison results including ranked leaderboard, winner, and metrics used
        """
        logger.info(f"Comparing {len(results)} strategy results")
        
        try:
            # Convert results to BacktestResult objects
            from ..services.backtesting import BacktestResult
            
            bt_results: List[BacktestResult] = []
            for item in results:
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
            logger.info("Strategy comparison completed")
            return result
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            raise ValueError(f"Failed to compare strategies: {e}")
    
    @mcp.tool(
        name="health",
        description="Check server health and connectivity to external data sources."
    )
    def health(
        check_data_sources: Annotated[bool, "Whether to check external data source availability. Default: True"] = True
    ) -> Dict[str, Any]:
        """Check server health and external data source availability.
        
        Performs comprehensive health check including server status, uptime,
        and connectivity to external MCP data sources (Binance, TradingView).
        
        Parameters:
        - check_data_sources (bool): Whether to check external data source availability. Default: True
        
        Returns:
        - Dict[str, Any]: Health status with server info, data source status, and error details
        """
        logger.info("Performing health check")
        
        try:
            result = health_service.check_health(check_data_sources=check_data_sources)
            # Add uptime to the result
            result["uptime"] = time.time() - start_time
            logger.info(f"Health check completed: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().timestamp(),
                "uptime": time.time() - start_time,
            }
    
    @mcp.tool(
        name="version",
        description="Return server version information and supported features."
    )
    def version() -> Dict[str, Any]:
        """Get server version information and configuration details.
        
        Returns comprehensive information about the backtesting MCP server including
        version number, package details, current configuration settings, and
        supported features.
        
        Returns:
        - Dict[str, Any]: Version information including config and features
        """
        logger.info("Getting version information")
        
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
