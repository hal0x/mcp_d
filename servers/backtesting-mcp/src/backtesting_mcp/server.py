"""Backtesting MCP server built on synthetic data generator."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .tools.backtesting_tools import register_backtesting_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Track server start time for uptime calculation
_start_time = time.time()


def create_server() -> FastMCP:
    """Create and configure the MCP server."""
    settings = get_settings()
    
    server = FastMCP(
        name="backtesting-mcp",
        instructions=(
            "Инструменты для бэктестинга и оптимизации торговых стратегий. "
            "Используйте run_backtest для запуска бэктестов, optimize_parameters для "
            "оптимизации параметров стратегии, compare_strategies для сравнения "
            "результатов, health для проверки состояния сервера и version для "
            "получения информации о версии."
        ),
    )
    
    # Register all tools
    register_backtesting_tools(server, _start_time)
    
    # Note: FastMCP doesn't expose tools directly, so we'll add uptime in the health tool itself
    
    logger.info("Backtesting MCP server created successfully")
    return server


def main() -> None:
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Run the Backtesting MCP server")
    parser.add_argument("--stdio", action="store_true", help="Run server over stdio (default)")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transport")
    parser.add_argument("--port", type=int, default=8082, help="Port for HTTP transport")
    parser.add_argument("--print-config", action="store_true", help="Print configuration and exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Print config if requested
    if args.print_config:
        settings = get_settings()
        print("Backtesting MCP Configuration:")
        print(f"  Default timeframe: {settings.default_timeframe}")
        print(f"  Preferred source: {settings.preferred_source}")
        print(f"  Seed: {settings.seed}")
        print(f"  Max candles: {settings.max_candles}")
        print(f"  Binance URL: {settings.binance_url}")
        print(f"  TradingView URL: {settings.tradingview_url}")
        print(f"  Request timeout: {settings.request_timeout}s")
        return
    
    # Create and run server
    server = create_server()
    
    transport = "stdio" if args.stdio else "streamable-http"

    try:
        if transport == "stdio":
            logger.info("Starting server in stdio mode")
            server.run()
        else:
            logger.info(
                "Starting server in HTTP mode on %s:%s", args.host, args.port
            )
            try:
                server.settings.host = args.host
                server.settings.port = args.port
            except Exception:
                logger.debug("Failed to set FastMCP host/port", exc_info=True)
            server.run(transport="streamable-http")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error("Server error: %s", e)
        raise


if __name__ == "__main__":
    main()
