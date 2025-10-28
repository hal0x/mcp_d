# 📈 TradingView MCP Server

A powerful Model Context Protocol (MCP) server that provides advanced cryptocurrency and stock market analysis using TradingView data. Perfect for traders, analysts, and AI assistants who need real-time market intelligence.


## ✨ Key Features

- 🚀 **Real-time Market Screening**: Find top gainers, losers, and trending stocks/crypto
- 📊 **Advanced Technical Analysis**: Bollinger Bands, RSI, MACD, and more indicators  
- 🎯 **Bollinger Band Intelligence**: Proprietary rating system (-3 to +3) for squeeze detection
- 🕯️ **Pattern Recognition**: Detect consecutive bullish/bearish candle formations
- 💎 **Multi-Market Support**: Crypto exchanges (KuCoin, Binance, Bybit) + Traditional markets (NASDAQ, BIST)
- ⏰ **Multi-Timeframe Analysis**: From 5-minute to monthly charts
- 🔍 **Individual Asset Deep-Dive**: Comprehensive technical analysis for any symbol

## 🔌 Network Topology

| Service | Port | Transport | Dependencies |
|---------|------|-----------|--------------|
| `binance-mcp` | `8000` | `stdio` | `redis`, `postgres` |
| `tradingview-mcp` | `8060` | `streamable-http` | `redis`, `postgres`, `binance-mcp`, `memory-mcp` |
| `memory-mcp` | `8050` | `stdio` | `redis` |
| `shell-mcp` | `8070` | `stdio` | `redis` |
| `backtesting-mcp` | `8082` | `stdio` | `postgres`, `binance-mcp`, `tradingview-mcp` |

> TradingView MCP listens on `0.0.0.0:8060` and is reachable by other services at `http://tradingview-mcp:8060`.

## 🚀 Quick Start

### Option 1: Claude Desktop (Recommended)

1. **Install UV Package Manager:**
   ```bash
   # macOS (Homebrew)
   brew install uv
   
   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # macOS/Linux (Direct)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Add to Claude Desktop Configuration:**
   
   **Config Path:**
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   
   ```json
   {
     "mcpServers": {
       "tradingview-mcp": {
         "command": "uv",
         "args": [
           "tool", "run", "--from",
           "git+https://github.com/atilaahmettaner/tradingview-mcp.git",
           "tradingview-mcp"
         ]
       }
     }
   }
   ```

3. **Restart Claude Desktop** - The server will be automatically available!

📋 **For detailed Windows instructions, see [INSTALLATION.md](INSTALLATION.md)**

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/atilaahmettaner/tradingview-mcp.git
cd tradingview-mcp

# Install dependencies
uv sync

# For local development, add to Claude Desktop:
```

**Windows Configuration Path:**
`%APPDATA%\Claude\claude_desktop_config.json`

**macOS Configuration Path:**
`~/Library/Application Support/Claude/claude_desktop_config.json`

**Configuration for Local Setup:**
```json
{
  "mcpServers": {
    "tradingview-mcp-local": {
      "command": "C:\\path\\to\\your\\tradingview-mcp\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\your\\tradingview-mcp\\src\\tradingview_mcp\\server.py"],
      "cwd": "C:\\path\\to\\your\\tradingview-mcp"
    }
  }
}
```

**macOS/Linux Configuration:**
```json
{
  "mcpServers": {
    "tradingview-mcp-local": {
      "command": "uv",
      "args": ["run", "python", "src/tradingview_mcp/server.py"],
      "cwd": "/path/to/your/tradingview-mcp"
    }
  }
}
```

## 🛠️ Available Tools

### 🧠 Professional Scanners
| Tool | Description | Example Usage |
|------|-------------|---------------|
| `pro_scanner_profiles` | List available professional profiles | `pro_scanner_profiles`
| `pro_momentum_scan` | Momentum scanner with risk filters | `pro_momentum_scan symbols=["BTCUSDT","ETHUSDT"] profile="balanced"`
| `pro_mean_revert_scan` | Mean-reversion scanner with Bollinger/RSI logic | `pro_mean_revert_scan symbols=["SOLUSDT"]`
| `pro_scanner_backtest` | Run historical evaluation for pro scanners | `pro_scanner_backtest symbols=["BTCUSDT"] strategy="mean_revert" lookback_limit=300`
| `pro_breakout_scan` | Detects volume-backed breakouts | `pro_breakout_scan symbols=["BTCUSDT"] profile="balanced"`
| `pro_volume_profile_scan` | Checks value area/POC context | `pro_volume_profile_scan symbols=["ETHUSDT"]`
| `pro_scanner_metrics` | Aggregated metrics from Postgres | `pro_scanner_metrics period_days=7 strategy="momentum"`
| `pro_scanner_recent_signals` | Latest persisted signals | `pro_scanner_recent_signals limit=10`
| `pro_scanner_feedback` | Record user feedback on a signal | `pro_scanner_feedback signal_id=123 action="take"`
| `pro_scanner_cache_clear` | Clear Redis cache namespace | `pro_scanner_cache_clear namespace="momentum"`

### 📈 Market Screening
| Tool | Description | Example Usage |
|------|-------------|---------------|
| `top_gainers` | Find highest performing assets | Top crypto gainers in 15m |
| `top_losers` | Find biggest declining assets | Worst performing stocks today |
| `bollinger_scan` | Find assets with tight Bollinger Bands | Coins ready for breakout |
| `rating_filter` | Filter by Bollinger Band rating | Strong buy signals (rating +2) |

### 🔍 Technical Analysis  
| Tool | Description | Example Usage |
|------|-------------|---------------|
| `coin_analysis` | Complete technical analysis | Analyze BTC with all indicators |
| `consecutive_candles_scan` | Find candlestick patterns | 3+ consecutive green candles |
| `advanced_candle_pattern` | Multi-timeframe pattern analysis | Complex pattern detection |

### 📋 Information
| Tool | Description |
|------|-------------|
| `exchanges://list` | List all supported exchanges and markets |

## 🧱 Infrastructure Setup

1. `cd tradingview-mcp` and start Redis/Postgres via `docker compose -f docker-compose.pro_scanners.yaml up -d`.
2. Apply the schema: `psql postgres://tradingview:tradingview@localhost:5432/tradingview_scanners -f sql/pro_scanners_schema.sql` (adjust credentials as needed).
3. Ensure `binance-mcp` server is running and reachable at the URL configured in `src/tradingview_mcp/configs/infrastructure.yaml` (or your `TRADINGVIEW_MCP_CONFIG_DIR` override).
4. Export `HALV1_ALERT_URL` or update `src/tradingview_mcp/configs/infrastructure.yaml` if HALv1 bot runs on a different host/port.

## 📝 Usage Examples

### Talk to Claude Like This:

**Basic Market Screening:**
```
"Show me the top 10 crypto gainers on KuCoin in the last 15 minutes"
"Find the biggest losers on Binance today"  
"Which Turkish stocks (BIST) are down more than 5% today?"
```

**Technical Analysis:**
```
"Analyze Bitcoin with all technical indicators"
"Find crypto coins with Bollinger Band squeeze (BBW < 0.05)"
"Show me coins with strong buy signals (rating +2)"
```

**Pattern Recognition:**
```
"Find coins with 3 consecutive bullish candles on Bybit"
"Scan for stocks showing growing candle patterns"
"Which assets have tight Bollinger Bands ready for breakout?"
```

**Advanced Queries:**
```
"Compare AAPL vs TSLA technical indicators"
"Find high-volume crypto with RSI below 30" 
"Show me NASDAQ stocks with strong momentum"
```

## 🎯 Understanding the Bollinger Band Rating System

Our proprietary rating system helps identify trading opportunities:

| Rating | Signal | Description |
|--------|---------|-------------|
| **+3** | 🔥 Strong Buy | Price above upper Bollinger Band |
| **+2** | ✅ Buy | Price in upper 50% of bands |
| **+1** | ⬆️ Weak Buy | Price above middle line |
| **0** | ➡️ Neutral | Price at middle line |
| **-1** | ⬇️ Weak Sell | Price below middle line |
| **-2** | ❌ Sell | Price in lower 50% of bands |
| **-3** | 🔥 Strong Sell | Price below lower Bollinger Band |

**Bollinger Band Width (BBW)**: Lower values indicate tighter bands → potential breakout coming!

## 🏢 Supported Markets & Exchanges

### 💰 Cryptocurrency Exchanges
- **KuCoin** (KUCOIN) - Primary recommendation
- **Binance** (BINANCE) - Largest crypto exchange  
- **Bybit** (BYBIT) - Derivatives focused
- **OKX** (OKX) - Global crypto exchange
- **Coinbase** (COINBASE) - US-regulated exchange
- **Gate.io** (GATEIO) - Altcoin specialist
- **Huobi** (HUOBI) - Asian market leader
- **Bitfinex** (BITFINEX) - Professional trading

### 📊 Traditional Markets
- **NASDAQ** (NASDAQ) - US tech stocks (AAPL, MSFT, TSLA)
- **BIST** (BIST) - Turkish stock market (Borsa İstanbul)
- More markets coming soon!

### ⏰ Supported Timeframes
`5m`, `15m`, `1h`, `4h`, `1D`, `1W`, `1M`

## 📊 Technical Indicators Included

- **Bollinger Bands** (20, 2) - Volatility and squeeze detection
- **RSI** (14) - Momentum oscillator  
- **Moving Averages** - SMA20, EMA50, EMA200
- **MACD** - Trend and momentum
- **ADX** - Trend strength measurement
- **Stochastic** - Overbought/oversold conditions
- **Volume Analysis** - Market participation
- **Price Action** - OHLC data with percentage changes

## 🚨 Troubleshooting

### Common Issues:

**1. "No data found" errors:**
- Try different exchanges (KuCoin usually works best)
- Use standard timeframes (15m, 1h, 1D)
- Check symbol format (e.g., "BTCUSDT" not "BTC")

**2. Empty arrays or rate limiting:**
- If you get empty results, you may have hit TradingView's rate limits
- Wait 5-10 minutes between query sessions
- The server automatically handles retries
- KuCoin and BIST have the most reliable data

**3. Claude Desktop not detecting the server:**
- Restart Claude Desktop after adding configuration
- Check that UV is installed: `uv --version`
- Verify the configuration JSON syntax

**4. Slow responses:**
- First request may be slower (warming up)
- Subsequent requests are much faster
- Consider using smaller limits (5-10 items)

## 🔧 Development & Customization

### Running in Development Mode:
```bash
# Clone and setup
git clone https://github.com/atilaahmettaner/tradingview-mcp.git
cd tradingview-mcp
uv sync

# Run with MCP Inspector for debugging
uv run mcp dev src/tradingview_mcp/server.py

# Test individual functions
uv run python test_api.py
```

### Adding New Exchanges:
The server is designed to be easily extensible. Check `src/tradingview_mcp/core/` for the modular architecture.

## 📚 Documentation

Complete documentation is available in the [`docs/`](./docs/) directory:

- **[📖 Documentation Index](./docs/README.md)** - Complete documentation overview
- **[🛠️ Installation & Setup](./docs/installation/)** - Installation and launch guides
- **[📖 Usage Examples](./docs/usage/)** - Examples and testing reports
- **[🔧 Development](./docs/development/)** - Contributing and development guides
- **[📡 API Reference](./docs/api/)** - API documentation (coming soon)

### Quick Links
- **[Installation Guide](./docs/installation/INSTALLATION.md)** - Detailed setup instructions
- **[Usage Examples](./docs/usage/EXAMPLES.md)** - Real-world usage scenarios  
- **[Launch Guide](./docs/installation/LAUNCH_GUIDE.md)** - Server startup instructions
- **[Contributing Guide](./docs/development/CONTRIBUTING.md)** - How to contribute
- **[Testing Report](./docs/usage/TESTING_REPORT.md)** - Test results and coverage

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./docs/development/CONTRIBUTING.md) for detailed instructions.

### Quick Start for Contributors:
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Follow** the [development guidelines](./docs/development/CONTRIBUTING.md#development-guidelines)
4. **Run tests** (`uv run pytest tests/`)
5. **Open** a Pull Request

### Ideas for Contributions:
- Add new exchanges or markets
- Implement additional technical indicators  
- Improve error handling and rate limiting
- Add more candlestick pattern recognition
- Create comprehensive test suite
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Support & Issues

- **Report bugs**: [GitHub Issues](https://github.com/atilaahmettaner/tradingview-mcp/issues)
- **Feature requests**: Open an issue with the "enhancement" label
- **Questions**: Check existing issues or open a new discussion

## 🌟 Star This Project

If you find this MCP server useful, please ⭐ star the repository to help others discover it!

---

**Built with ❤️ for traders and AI enthusiasts**

*Empowering intelligent trading decisions through advanced market analysis*
