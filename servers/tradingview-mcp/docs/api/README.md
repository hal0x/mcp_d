# 📡 API Reference - TradingView MCP Server

This section contains comprehensive API documentation for all TradingView MCP Server tools and endpoints.

## 🔧 Available Tools

### Meta Tools
- **[health](./api/health.md)** - Server health check and status
- **[version](./api/version.md)** - Package metadata and features
- **[pro_scanner_profiles](./api/pro_scanner_profiles.md)** - Scanner profiles and parameters

### Market Analysis Tools
- **[top_gainers](./api/top_gainers.md)** - Top performing cryptocurrencies
- **[top_gainers_batch](./api/top_gainers_batch.md)** - Batch top gainers analysis
- **[top_losers](./api/top_losers.md)** - Top declining cryptocurrencies
- **[top_losers_batch](./api/top_losers_batch.md)** - Batch top losers analysis
- **[multi_changes](./api/multi_changes.md)** - Multi-timeframe price changes
- **[multi_changes_batch](./api/multi_changes_batch.md)** - Batch multi-timeframe analysis
- **[coin_analysis](./api/coin_analysis.md)** - Detailed individual coin analysis
- **[coin_analysis_batch](./api/coin_analysis_batch.md)** - Batch coin analysis

### Technical Screening Tools
- **[bollinger_scan](./api/bollinger_scan.md)** - Bollinger Bands screening
- **[bollinger_scan_batch](./api/bollinger_scan_batch.md)** - Batch Bollinger Bands analysis
- **[consecutive_candles_scan](./api/consecutive_candles_scan.md)** - Candle pattern detection
- **[advanced_candle_pattern](./api/advanced_candle_pattern.md)** - Advanced candle patterns

### Strategy Tools
- **[trend_breakout_pyramiding](./api/trend_breakout_pyramiding.md)** - Trend breakout strategy
- **[pullback_engine](./api/pullback_engine.md)** - Pullback trading engine
- **[strategy_candidates](./api/strategy_candidates.md)** - Strategy candidate selection
- **[scan_strategy_candidates](./api/scan_strategy_candidates.md)** - Multi-timeframe strategy scanning
- **[smart_scanner](./api/smart_scanner.md)** - Adaptive market scanner

### Professional Scanner Tools
- **[pro_momentum_scan](./api/pro_momentum_scan.md)** - Professional momentum scanning
- **[pro_mean_revert_scan](./api/pro_mean_revert_scan.md)** - Professional mean reversion scanning
- **[pro_breakout_scan](./api/pro_breakout_scan.md)** - Professional breakout scanning
- **[pro_volume_profile_scan](./api/pro_volume_profile_scan.md)** - Professional volume profile scanning
- **[pro_scanner_backtest](./api/pro_scanner_backtest.md)** - Scanner backtesting
- **[pro_scanner_metrics](./api/pro_scanner_metrics.md)** - Scanner performance metrics
- **[pro_scanner_metrics_snapshot](./api/pro_scanner_metrics_snapshot.md)** - Live metrics snapshot
- **[pro_scanner_recent_signals](./api/pro_scanner_recent_signals.md)** - Recent scanner signals
- **[pro_scanner_feedback](./api/pro_scanner_feedback.md)** - Signal feedback system
- **[pro_scanner_cache_clear](./api/pro_scanner_cache_clear.md)** - Cache management

## 📊 Data Models

### Common Types
- **Exchange**: Supported exchange names (KUCOIN, BINANCE, BYBIT, etc.)
- **Timeframe**: Supported time intervals (5m, 15m, 1h, 4h, 1D, 1W, 1M)
- **Symbol**: Trading pair symbols (e.g., "BTCUSDT", "ETHUSDT")

### Response Formats
- **Market Data**: Price, volume, change percentage
- **Technical Indicators**: RSI, MACD, Bollinger Bands, etc.
- **Signals**: Buy/Sell recommendations with confidence levels
- **Metrics**: Performance statistics and analytics

## 🔍 Error Handling

All tools return structured error responses with:
- **Error Code**: Machine-readable error identifier
- **Error Message**: Human-readable error description
- **Context**: Additional debugging information

Common error codes:
- `INVALID_EXCHANGE`: Unsupported exchange name
- `INVALID_TIMEFRAME`: Unsupported timeframe
- `API_ERROR`: TradingView API error
- `RATE_LIMIT`: API rate limit exceeded
- `VALIDATION_ERROR`: Input validation failed

## 📈 Rate Limits

- **Standard Tools**: 100 requests/minute
- **Batch Tools**: 50 requests/minute
- **Professional Scanners**: 20 requests/minute
- **Cache**: 1000 requests/minute

## 🔐 Authentication

The server uses TradingView API credentials stored in environment variables:
- `TRADINGVIEW_API_KEY`: Your TradingView API key
- `TRADINGVIEW_API_SECRET`: Your TradingView API secret

## 📝 Examples

### Basic Usage
```python
# Health check
health()

# Top gainers
top_gainers(exchange="KUCOIN", timeframe="15m", limit=10)

# Coin analysis
coin_analysis(symbol="BTCUSDT", exchange="KUCOIN", timeframe="1h")
```

### Advanced Usage
```python
# Batch analysis
top_gainers_batch([
    {"exchange": "KUCOIN", "timeframe": "15m", "limit": 5},
    {"exchange": "BINANCE", "timeframe": "1h", "limit": 10}
])

# Professional scanning
pro_momentum_scan(
    symbols=["BTCUSDT", "ETHUSDT"],
    profile="balanced",
    leverage_range=[5, 20]
)
```

## 🚀 Getting Started

1. **Install the server**: See [Installation Guide](../installation/INSTALLATION.md)
2. **Configure Claude Desktop**: See [Launch Guide](../installation/LAUNCH_GUIDE.md)
3. **Try examples**: See [Usage Examples](../usage/EXAMPLES.md)
4. **Explore tools**: Use the health and version tools to get started

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/atilaahmettaner/tradingview-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/atilaahmettaner/tradingview-mcp/discussions)
- **Documentation**: [Main Documentation](../README.md)

---

*This API documentation is automatically generated and updated with each release.*
