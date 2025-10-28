from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import wraps
from importlib import metadata
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from mcp.server.fastmcp import FastMCP

from tradingview_mcp.config import Settings, get_settings
from tradingview_mcp.pro_scanners import ScannerProfiles, ScannerService
from tradingview_mcp.pro_scanners.models import BacktestRequest
from tradingview_mcp.pro_scanners.backtesting.metrics import MetricsTracker
from tradingview_mcp.scheduler import scheduler

# Глобальный кэш для результатов Professional Scanners
SCANNER_RESULTS_CACHE: Dict[str, Dict[str, Any]] = {}


async def _save_scanner_result_redis(scanner_name: str, symbols: List[str], profile: str, result: Any) -> str:
    """Сохраняет результат сканера в Redis кэш и возвращает уникальный ID запроса."""
    try:
        import redis
        import json
        
        # Подключаемся к Redis
        redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
        
        request_id = f"{scanner_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        cache_data = {
            "scanner_name": scanner_name,
            "symbols": symbols,
            "profile": profile,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed"
        }
        
        # Сохраняем в Redis с TTL 24 часа
        redis_client.setex(f"scanner_result:{request_id}", 86400, json.dumps(cache_data))
        
        # Добавляем в список последних результатов
        redis_client.lpush("scanner_results:recent", request_id)
        redis_client.ltrim("scanner_results:recent", 0, 99)  # Храним последние 100
        
        logger.info(f"Saved scanner result to Redis: {request_id} ({scanner_name}, {symbols}, {profile})")
        return request_id
        
    except Exception as exc:
        logger.error(f"Failed to save to Redis: {exc}")
        # Fallback to in-memory cache
        return _save_scanner_result(scanner_name, symbols, profile, result)


def _save_scanner_result(scanner_name: str, symbols: List[str], profile: str, result: Any) -> str:
    """Сохраняет результат сканера в кэш и возвращает уникальный ID запроса."""
    request_id = f"{scanner_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    SCANNER_RESULTS_CACHE[request_id] = {
        "scanner_name": scanner_name,
        "symbols": symbols,
        "profile": profile,
        "result": result,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "completed"
    }
    
    # Ограничиваем размер кэша (храним последние 100 результатов)
    if len(SCANNER_RESULTS_CACHE) > 100:
        oldest_key = min(SCANNER_RESULTS_CACHE.keys(), 
                        key=lambda k: SCANNER_RESULTS_CACHE[k]["timestamp"])
        del SCANNER_RESULTS_CACHE[oldest_key]
    
    logger.info(f"Saved scanner result: {request_id} ({scanner_name}, {symbols}, {profile})")
    return request_id

# Import bollinger band screener modules
from tradingview_mcp.core.services.indicators import compute_metrics
from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.utils.validators import sanitize_timeframe, sanitize_exchange, EXCHANGE_SCREENER, ALLOWED_TIMEFRAMES

try:
    from tradingview_ta import TA_Handler, get_multiple_analysis
    TRADINGVIEW_TA_AVAILABLE = True
except ImportError:
    TRADINGVIEW_TA_AVAILABLE = False

try:
    from tradingview_screener import Query
    from tradingview_screener.column import Column
    TRADINGVIEW_SCREENER_AVAILABLE = True
except ImportError:
    TRADINGVIEW_SCREENER_AVAILABLE = False


logger = logging.getLogger(__name__)

_BATCH_SIZE_TA = 5  # Уменьшен размер батча для избежания rate limiting
_TA_MAX_RETRIES = 2  # Уменьшено количество попыток
_TA_RETRY_DELAY = 3.0  # Увеличена задержка между попытками

_scanner_service: ScannerService | None = None
_scanner_service_started: bool = False
_scanner_service_lock: asyncio.Lock | None = None
_scanner_profiles: ScannerProfiles | None = None


def _safe_repr(value: Any, max_length: int = 120) -> str:
    """Return a trimmed representation suitable for logs."""
    try:
        rendered = repr(value)
    except Exception:
        rendered = f"<unreprable {type(value).__name__}>"
    if len(rendered) > max_length:
        return rendered[: max_length - 3] + "..."
    return rendered


def _summarize_result(value: Any) -> str:
    """Produce a concise description of tool output for logging."""
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    if value is None:
        return "None"
    return type(value).__name__


def _format_call_args(sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bind args to signature and return a log-friendly dict."""
    try:
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return {key: _safe_repr(val) for key, val in bound.arguments.items()}
    except Exception:
        return {"error": "failed to bind arguments"}


def log_tool(func):
    """Decorator that logs tool execution lifecycle."""
    sig = inspect.signature(func)
    tool_name = func.__name__

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.info("tool %s called with %s", tool_name, _format_call_args(sig, args, kwargs))
            try:
                result = await func(*args, **kwargs)
                logger.info("tool %s completed (%s)", tool_name, _summarize_result(result))
                return result
            except Exception:
                logger.exception("tool %s failed", tool_name)
                raise

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger.info("tool %s called with %s", tool_name, _format_call_args(sig, args, kwargs))
        try:
            result = func(*args, **kwargs)
            logger.info("tool %s completed (%s)", tool_name, _summarize_result(result))
            return result
        except Exception:
            logger.exception("tool %s failed", tool_name)
            raise

    return sync_wrapper


def _chunked(seq: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [seq]
    return [seq[i : i + size] for i in range(0, len(seq), size)]


async def _ensure_scanner_service() -> ScannerService:
    global _scanner_service, _scanner_service_started, _scanner_service_lock
    if _scanner_service is None:
        _scanner_service = ScannerService()
    if _scanner_service_started:
        return _scanner_service
    if _scanner_service_lock is None:
        _scanner_service_lock = asyncio.Lock()
    async with _scanner_service_lock:
        if not _scanner_service_started:
            try:
                await _scanner_service.startup()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to start ScannerService: %s", exc)
                # Don't raise immediately - allow partial functionality
                logger.warning("ScannerService started with limited functionality due to infrastructure issues")
            else:
                _scanner_service_started = True
    return _scanner_service


async def _shutdown_scanner_service() -> None:
    global _scanner_service_started, _scanner_service
    if _scanner_service and _scanner_service_started:
        try:
            await _scanner_service.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to shutdown ScannerService")
        finally:
            _scanner_service_started = False


def _scanner_lifespan(_: FastMCP):
    @asynccontextmanager
    async def _lifespan():
        # Startup
        await _ensure_scanner_service()
        
        # Запускаем планировщик сканеров
        try:
            await scheduler.start()
            logger.info("Scanner scheduler started successfully")
        except Exception as exc:
            logger.error(f"Failed to start scheduler: {exc}")
        
        try:
            yield
        finally:
            # Shutdown
            await _shutdown_scanner_service()
            # Останавливаем планировщик
            try:
                await scheduler.stop()
                logger.info("Scanner scheduler stopped")
            except Exception as exc:
                logger.error(f"Failed to stop scheduler: {exc}")

    return _lifespan()


def _get_scanner_profiles() -> ScannerProfiles:
    global _scanner_profiles
    if _scanner_profiles is None:
        _scanner_profiles = ScannerProfiles()
    return _scanner_profiles


def _batched_analysis(symbols: List[str], screener: str, interval: str) -> Dict[str, dict]:
    """Call tradingview_ta in small batches with retries to avoid rate limits."""
    aggregated: Dict[str, dict] = {}
    if not TRADINGVIEW_TA_AVAILABLE or not symbols:
        return aggregated

    for batch in _chunked(symbols, _BATCH_SIZE_TA):
        attempts = 0
        while attempts < _TA_MAX_RETRIES:
            try:
                data = get_multiple_analysis(screener=screener, interval=interval, symbols=batch)
                if data and isinstance(data, dict) and len(data) > 0:
                    for sym, analysis in data.items():
                        indicators = getattr(analysis, "indicators", None)
                        if indicators:
                            aggregated[sym] = indicators
                else:
                    logger.warning("Empty or invalid data received from TradingView API for batch %s", batch[:2])
                break
            except Exception as exc:
                attempts += 1
                if attempts >= _TA_MAX_RETRIES:
                    logger.warning(
                        "TradingView TA batch failed after %s attempts for %s… (%s)",
                        attempts,
                        batch[:2],
                        exc,
                    )
                    break
                sleep_for = _TA_RETRY_DELAY * attempts
                logger.warning(
                    "TradingView TA batch error (%s/%s) for %s…, retrying in %.1fs: %s",
                    attempts,
                    _TA_MAX_RETRIES,
                    batch[:2],
                    sleep_for,
                    exc,
                )
                time.sleep(sleep_for)
    return aggregated


def _check_consecutive_pattern(indicators: dict, pattern_type: str, candle_count: int, min_growth: float) -> bool:
    """
    Check if indicators show consecutive candle pattern.
    
    Args:
        indicators: Dictionary with indicator values (close, open, high, low)
        pattern_type: 'bullish' or 'bearish'
        candle_count: Number of consecutive candles
        min_growth: Minimum growth percentage
    
    Returns:
        True if pattern matches, False otherwise
    """
    if not indicators:
        return False
    
    close = indicators.get("close")
    open_price = indicators.get("open")
    high = indicators.get("high")
    low = indicators.get("low")
    
    if not all([close, open_price, high, low]):
        return False
    
    if pattern_type.lower() == "bullish":
        # Check for consecutive green candles
        if open_price and close:
            growth = ((close - open_price) / open_price) * 100 if open_price > 0 else 0
            return growth >= min_growth
    
    elif pattern_type.lower() == "bearish":
        # Check for consecutive red candles
        if open_price and close:
            growth = ((open_price - close) / open_price) * 100 if open_price > 0 else 0
            return growth >= min_growth
    
    return False


def _fetch_multi_timeframe_patterns(exchange: str, symbols: List[str], base_timeframe: str, pattern_length: int, min_size_increase: float) -> list:
    """
    Fetch multi-timeframe pattern data for symbols.
    
    Args:
        exchange: Exchange name
        symbols: List of symbols to analyze
        base_timeframe: Base timeframe for analysis
        pattern_length: Length of pattern to detect
        min_size_increase: Minimum size increase percentage
    
    Returns:
        List of pattern results
    """
    results = []
    
    try:
        # Use TradingView screener to get multi-timeframe data
        exchange_upper = exchange.upper()
        screener = EXCHANGE_SCREENER.get(exchange_upper, "crypto")
        
        # Get data for multiple timeframes
        timeframes = [base_timeframe, "1h", "4h"]
        
        for symbol in symbols[:20]:  # Limit to 20 symbols for performance
            symbol_results = []
            
            for tf in timeframes:
                try:
                    # Format symbol for TradingView
                    full_symbol = f"{exchange_upper}:{symbol}"
                    
                    # Get analysis data
                    analysis_data = get_multiple_analysis(
                        screener=screener,
                        interval=tf,
                        symbols=[full_symbol]
                    )
                    
                    if full_symbol in analysis_data and analysis_data[full_symbol]:
                        data = analysis_data[full_symbol]
                        indicators = data.indicators
                        
                        # Calculate size increase
                        close = indicators.get("close")
                        open_price = indicators.get("open")
                        
                        if close and open_price and open_price > 0:
                            size_increase = ((close - open_price) / open_price) * 100
                            
                            if abs(size_increase) >= min_size_increase:
                                symbol_results.append({
                                    "symbol": symbol,
                                    "timeframe": tf,
                                    "size_increase": size_increase,
                                    "close": close,
                                    "open": open_price
                                })
                
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol} for {tf}: {e}")
                    continue
            
            # Only add if multiple timeframes match
            if len(symbol_results) >= pattern_length:
                results.append({
                    "symbol": symbol,
                    "exchange": exchange,
                    "matches": len(symbol_results),
                    "results": symbol_results
                })
    
    except Exception as e:
        logger.exception(f"Failed to fetch multi-timeframe patterns: {e}")
    
    return results


def _fetch_via_screener(symbols: List[str], timeframe: str, exchange: str) -> Dict[str, dict]:
    """Fallback using tradingview_screener when tradingview_ta is unavailable or rate-limited."""
    if not TRADINGVIEW_SCREENER_AVAILABLE or not symbols:
        return {}

    resolution = _tf_to_tv_resolution(timeframe) or "60"
    base_cols = [
        "open",
        "close",
        "SMA20",
        "EMA20",
        "EMA50",
        "EMA100",
        "EMA200",
        "RSI",
        "ADX",
        "ATR",
        "BB.upper",
        "BB.lower",
        "volume",
    ]
    select_cols = [f"{col}|{resolution}" for col in base_cols]
    names = [sym.split(":", 1)[-1].upper() for sym in symbols]

    try:
        query = Query().set_markets("crypto").select(*select_cols)
        if exchange:
            query = query.where(Column("exchange") == exchange.upper())
        if names:
            query = query.where(Column("name").isin(names))
        query = query.limit(max(len(names), 50))
        _total, df = query.get_scanner_data()
    except Exception as exc:
        logger.warning("tradingview_screener fallback failed: %s", exc)
        return {}

    if df is None or df.empty:
        return {}

    out: Dict[str, dict] = {}

    def _coerce(val: Any) -> Optional[float]:
        try:
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                return None
            return float(val)
        except (TypeError, ValueError):
            return None

    for _, row in df.iterrows():
        ticker = row.get("ticker")
        if not ticker:
            continue
        indicators: Dict[str, Optional[float]] = {}
        for col in select_cols:
            base = col.split("|", 1)[0]
            indicators[base] = _coerce(row.get(col))
        out[ticker] = indicators

    return out


TOOL_FEATURES: List[str] = [
    "server_info",
    "top_gainers_batch",
    "multi_changes_batch",
    "top_losers_batch",
    "bollinger_scan_batch",
    "coin_analysis_batch",
    "consecutive_candles_scan_batch",
    "advanced_candle_pattern_batch",
    "trend_breakout_pyramiding",
    "pullback_engine",
    "unified_scanner",
    "scan_strategy_candidates",
    "pro_scanner_profiles",
    "pro_momentum_scan",
    "pro_mean_revert_scan",
    "pro_scanner_backtest",
    "pro_breakout_scan",
    "pro_volume_profile_scan",
    "pro_scanner_metrics",
    "pro_scanner_metrics_snapshot",
    "pro_scanner_recent_signals",
    "pro_scanner_feedback",
    "pro_scanner_cache_clear",
]


class IndicatorMap(TypedDict, total=False):
	open: Optional[float]
	close: Optional[float]
	SMA20: Optional[float]
	BB_upper: Optional[float]
	BB_lower: Optional[float]
	EMA50: Optional[float]
	RSI: Optional[float]
	volume: Optional[float]


class Row(TypedDict):
	symbol: str
	changePercent: float
	indicators: IndicatorMap


class MultiRow(TypedDict):
	symbol: str
	changes: dict[str, Optional[float]]
	base_indicators: IndicatorMap


def _map_indicators(raw: Dict[str, Any]) -> IndicatorMap:
	return IndicatorMap(
		open=raw.get("open"),
		close=raw.get("close"),
		SMA20=raw.get("SMA20"),
		BB_upper=raw.get("BB.upper") if "BB.upper" in raw else raw.get("BB_upper"),
		BB_lower=raw.get("BB.lower") if "BB.lower" in raw else raw.get("BB_lower"),
		EMA50=raw.get("EMA50"),
		RSI=raw.get("RSI"),
		volume=raw.get("volume"),
	)


def _percent_change(o: Optional[float], c: Optional[float]) -> Optional[float]:
	try:
		if o in (None, 0) or c is None:
			return None
		return (c - o) / o * 100
	except Exception:
		return None


def _tf_to_tv_resolution(tf: Optional[str]) -> Optional[str]:
	if not tf:
		return None
	return {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1D": "1D", "1W": "1W", "1M": "1M"}.get(tf)


def _fetch_bollinger_analysis(exchange: str, timeframe: str = "4h", limit: int = 50, bbw_filter: float = None) -> List[Row]:
    """Fetch analysis using tradingview_ta; fallback to tradingview_screener when rate limited."""

    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")

    # Limit symbols for performance (grab extra for filtering)
    symbols = symbols[: max(1, limit * 3)]

    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    analysis_map = _batched_analysis(symbols, screener, timeframe)
    if not analysis_map:
        analysis_map = _fetch_via_screener(symbols, timeframe, exchange)
        if not analysis_map:
            logger.warning("No indicator data retrieved for %s on %s", exchange, timeframe)
            return []

    rows: List[Row] = []

    for key, indicators in analysis_map.items():
        try:
            if indicators is None:
                continue

            metrics = compute_metrics(indicators)
            if not metrics or metrics.get("bbw") is None:
                continue

            bbw = metrics["bbw"]
            if bbw_filter is not None and (bbw is None or bbw <= 0 or bbw >= bbw_filter):
                continue

            if not (indicators.get("EMA50") and indicators.get("RSI")):
                continue

            rows.append(
                Row(
                    symbol=key,
                    changePercent=metrics.get("change", 0.0),
                    indicators=IndicatorMap(
                        open=indicators.get("open"),
                        close=indicators.get("close"),
                        SMA20=indicators.get("SMA20"),
                        BB_upper=indicators.get("BB.upper"),
                        BB_lower=indicators.get("BB.lower"),
                        EMA50=indicators.get("EMA50"),
                        RSI=indicators.get("RSI"),
                        volume=indicators.get("volume"),
                    ),
                )
            )
        except (TypeError, ZeroDivisionError, KeyError):
            continue

    rows.sort(key=lambda x: x["changePercent"], reverse=True)
    return rows[:limit]


def _fetch_trending_analysis(exchange: str, timeframe: str = "5m", filter_type: str = "", rating_filter: int = None, limit: int = 50) -> List[Row]:
    """Fetch trending coins analysis similar to the original app's trending endpoint."""
    if not TRADINGVIEW_TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is missing; run `uv sync`.")
    
    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")
    
    # Process symbols in batches due to TradingView API limits
    batch_size = 200  # Considering API limitations
    all_coins = []
    
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    
    # Process symbols in batches
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i + batch_size]
        
        try:
            analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=batch_symbols)
        except Exception as e:
            continue  # If this batch fails, move to the next one
            
        # Process coins in this batch
        for key, value in analysis.items():
            try:
                if value is None:
                    continue
                    
                indicators = value.indicators
                metrics = compute_metrics(indicators)
                
                if not metrics or metrics.get('bbw') is None:
                    continue
                
                # Apply rating filter if specified
                if filter_type == "rating" and rating_filter is not None:
                    if metrics['rating'] != rating_filter:
                        continue
                
                all_coins.append(Row(
                    symbol=key,
                    changePercent=metrics['change'],
                    indicators=IndicatorMap(
                        open=metrics.get('open'),
                        close=metrics.get('price'),
                        SMA20=indicators.get("SMA20"),
                        BB_upper=indicators.get("BB.upper"),
                        BB_lower=indicators.get("BB.lower"),
                        EMA50=indicators.get("EMA50"),
                        RSI=indicators.get("RSI"),
                        volume=indicators.get("volume"),
                    )
                ))
                
            except (TypeError, ZeroDivisionError, KeyError):
                continue
    
    # Sort all coins by change percentage
    all_coins.sort(key=lambda x: x["changePercent"], reverse=True)
    
    return all_coins[:limit]
def _fetch_multi_changes(exchange: str, timeframes: List[str] | None, base_timeframe: str = "4h", limit: int | None = None, cookies: Any | None = None) -> List[MultiRow]:
	try:
		from tradingview_screener import Query
		from tradingview_screener.column import Column
	except Exception as e:
		raise RuntimeError("tradingview-screener missing; run `uv sync`.") from e

	tfs = timeframes or ["15m", "1h", "4h", "1D"]
	suffix_map: dict[str, str] = {}
	for tf in tfs:
		s = _tf_to_tv_resolution(tf)
		if s:
			suffix_map[tf] = s
	if not suffix_map:
		suffix_map = {base_timeframe: _tf_to_tv_resolution(base_timeframe) or "240"}

	base_suffix = _tf_to_tv_resolution(base_timeframe) or next(iter(suffix_map.values()))
	cols: list[str] = []
	seen: set[str] = set()
	for tf, s in suffix_map.items():
		for c in (f"open|{s}", f"close|{s}"):
			if c not in seen:
				cols.append(c)
				seen.add(c)
	for c in (f"SMA20|{base_suffix}", f"BB.upper|{base_suffix}", f"BB.lower|{base_suffix}", f"volume|{base_suffix}"):
		if c not in seen:
			cols.append(c)
			seen.add(c)

	q = Query().set_markets("crypto").select(*cols)
	if exchange:
		q = q.where(Column("exchange") == exchange.upper())
	if limit:
		q = q.limit(int(limit))

	_total, df = q.get_scanner_data(cookies=cookies)
	if df is None or df.empty:
		return []

	out: List[MultiRow] = []
	for _, r in df.iterrows():
		symbol = r.get("ticker")
		changes: dict[str, Optional[float]] = {}
		for tf, s in suffix_map.items():
			o = r.get(f"open|{s}")
			c = r.get(f"close|{s}")
			changes[tf] = _percent_change(o, c)
		base_ind = IndicatorMap(
			open=r.get(f"open|{base_suffix}"),
			close=r.get(f"close|{base_suffix}"),
			SMA20=r.get(f"SMA20|{base_suffix}"),
			BB_upper=r.get(f"BB.upper|{base_suffix}"),
			BB_lower=r.get(f"BB.lower|{base_suffix}"),
			volume=r.get(f"volume|{base_suffix}"),
		)
		out.append(MultiRow(symbol=symbol, changes=changes, base_indicators=base_ind))
	return out


mcp = FastMCP(
    name="TradingView Screener",
    instructions=(
        "Crypto screener utilities backed by TradingView Screener. Tools: health, version, "
        "top_gainers, top_losers, bollinger_scan, coin_analysis, consecutive_candles_scan, "
        "advanced_candle_pattern, multi_changes, trend_breakout_pyramiding, pullback_engine, "
        "strategy_candidates, scan_strategy_candidates, smart_scanner, pro_scanner_profiles, "
        "pro_momentum_scan, pro_mean_revert_scan, pro_breakout_scan, pro_volume_profile_scan, "
        "pro_scanner_backtest, pro_scanner_metrics, pro_scanner_metrics_snapshot, "
        "pro_scanner_recent_signals, pro_scanner_feedback, pro_scanner_cache_clear, "
        "pro_scanner_get_recently_results, pro_scanner_scheduler_status."
    ),
    lifespan=_scanner_lifespan,
)


def _build_config_snapshot(settings: Settings) -> dict[str, Any]:
    """Return config payload shared across meta tools."""
    return {
        "host": settings.host,
        "port": settings.port,
        "log_level": settings.log_level,
        "default_transport": settings.default_transport,
        "debug_mode": settings.debug,
    }


@mcp.tool()
@log_tool
def server_info() -> dict:
    """Get comprehensive server information including health, version, and configuration."""
    settings = get_settings()
    
    # Get package version
    try:
        pkg_version = metadata.version("tradingview-mcp")
    except metadata.PackageNotFoundError:
        pkg_version = "0.0.0"
    
    # Check dependencies
    dependencies = {
        "tradingview_ta": TRADINGVIEW_TA_AVAILABLE,
        "tradingview_screener": TRADINGVIEW_SCREENER_AVAILABLE,
    }
    
    # Determine overall status
    status = "healthy" if all(dependencies.values()) else "degraded"
    
    return {
        "status": status,
        "name": "tradingview-mcp",
        "version": pkg_version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dependencies": dependencies,
        "config": _build_config_snapshot(settings),
        "features": TOOL_FEATURES,
        "capabilities": {
            "technical_analysis": TRADINGVIEW_TA_AVAILABLE,
            "market_screening": TRADINGVIEW_SCREENER_AVAILABLE,
            "professional_scanners": True,
            "batch_operations": True,
            "multi_timeframe_analysis": True
        }
    }


@mcp.tool()
@log_tool
def pro_scanner_profiles() -> List[dict[str, Any]]:
    """List professional scanner profiles and parameters."""
    profiles = _get_scanner_profiles()
    payload: List[dict[str, Any]] = []
    for name in profiles.list_profiles():
        cfg = profiles.get(name)
        payload.append(
            {
                "name": name,
                "params": cfg.params.model_dump(mode="json"),
                "risk": cfg.risk.model_dump(mode="json"),
                "cache_ttl": cfg.cache_ttl,
            }
        )
    return payload


@mcp.tool()
@log_tool
async def pro_momentum_scan(symbols: List[str], profile: str = "balanced") -> List[dict[str, Any]]:
    """Run momentum scanner for given symbols.
    
    Args:
        symbols: List of trading symbols (e.g., ["BTCUSDT", "ETHUSDT"])
        profile: Risk profile - "conservative", "balanced", or "aggressive"
        
    Returns:
        List of momentum signals with entry price, indicators, and risk parameters.
        Empty list if binance-mcp unavailable or errors occur.
        
    Note:
        Requires binance-mcp service at http://binance-mcp:8000. Check logs if empty results.
    """
    if isinstance(symbols, str):  # type: ignore[arg-type]
        symbols = [symbols]  # pragma: no cover - runtime type coercion
    symbols = [sym.upper() for sym in symbols if sym]
    if not symbols:
        raise ValueError("symbols must contain at least one ticker")

    profiles = _get_scanner_profiles()
    profiles.ensure_exists([profile])

    try:
        service = await _ensure_scanner_service()
        signals = await service.scan_momentum(symbols, profile)
        if not signals:
            logger.warning("Momentum scanner returned no signals for %s. Possible reasons: 1) No momentum detected, 2) Binance-mcp unavailable, 3) Market conditions don't match filters", symbols)
        
        result = [signal.model_dump(mode="json") for signal in signals]
        
        # Сохраняем результат в кэш
        request_id = await _save_scanner_result_redis("pro_momentum_scan", symbols, profile, result)
        logger.info(f"Momentum scanner completed with request_id: {request_id}")
        
        return result
    except RuntimeError as exc:
        # RuntimeError обычно означает что binance-mcp недоступен
        logger.error("Momentum scanner failed - infrastructure issue: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.exception("Momentum scanner failed with unexpected error: %s", exc)
        # Return empty results instead of crashing
        return []


@mcp.tool()
@log_tool
async def pro_mean_revert_scan(symbols: List[str], profile: str = "balanced") -> List[dict[str, Any]]:
    """Run mean reversion scanner for given symbols."""
    if isinstance(symbols, str):  # type: ignore[arg-type]
        symbols = [symbols]
    symbols = [sym.upper() for sym in symbols if sym]
    if not symbols:
        raise ValueError("symbols must contain at least one ticker")

    profiles = _get_scanner_profiles()
    profiles.ensure_exists([profile])

    try:
        service = await _ensure_scanner_service()
        signals = await service.scan_mean_revert(symbols, profile)
        
        result = [signal.model_dump(mode="json") for signal in signals]
        
        # Сохраняем результат в кэш
        request_id = _save_scanner_result("pro_mean_revert_scan", symbols, profile, result)
        logger.info(f"Mean revert scanner completed with request_id: {request_id}")
        
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Mean reversion scanner failed: %s", exc)
        # Return empty results instead of crashing
        return []


@mcp.tool()
@log_tool
async def pro_breakout_scan(symbols: List[str], profile: str = "balanced") -> List[dict[str, Any]]:
    if isinstance(symbols, str):
        symbols = [symbols]
    symbols = [sym.upper() for sym in symbols if sym]
    if not symbols:
        raise ValueError("symbols must contain at least one ticker")
    profiles = _get_scanner_profiles()
    profiles.ensure_exists([profile])
    try:
        service = await _ensure_scanner_service()
        signals = await service.scan_breakout(symbols, profile)
        
        result = [signal.model_dump(mode="json") for signal in signals]
        
        # Сохраняем результат в кэш
        request_id = _save_scanner_result("pro_breakout_scan", symbols, profile, result)
        logger.info(f"Breakout scanner completed with request_id: {request_id}")
        
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Breakout scanner failed: %s", exc)
        # Return empty results instead of crashing
        return []


@mcp.tool()
@log_tool
async def pro_volume_profile_scan(symbols: List[str], profile: str = "balanced") -> List[dict[str, Any]]:
    if isinstance(symbols, str):
        symbols = [symbols]
    symbols = [sym.upper() for sym in symbols if sym]
    if not symbols:
        raise ValueError("symbols must contain at least one ticker")
    profiles = _get_scanner_profiles()
    profiles.ensure_exists([profile])
    try:
        service = await _ensure_scanner_service()
        signals = await service.scan_volume_profile(symbols, profile)
        
        result = [signal.model_dump(mode="json") for signal in signals]
        
        # Сохраняем результат в кэш
        request_id = _save_scanner_result("pro_volume_profile_scan", symbols, profile, result)
        logger.info(f"Volume profile scanner completed with request_id: {request_id}")
        
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Volume profile scanner failed: %s", exc)
        # Return empty results instead of crashing
        return []


@mcp.tool()
@log_tool
async def pro_scanner_backtest(
    symbols: List[str],
    profile: str = "balanced",
    timeframe: str = "15m",
    lookback_limit: int = 500,
    strategy: str = "momentum",
) -> dict[str, Any]:
    """Execute a backtest for professional scanners."""
    if isinstance(symbols, str):  # type: ignore[arg-type]
        symbols = [symbols]
    symbols = [sym.upper() for sym in symbols if sym]
    if not symbols:
        raise ValueError("symbols must contain at least one ticker")

    profiles = _get_scanner_profiles()
    profiles.ensure_exists([profile])

    try:
        service = await _ensure_scanner_service()
        request = BacktestRequest(
            symbols=symbols,
            timeframe=timeframe,
            lookback_limit=lookback_limit,
            profile=profile,
            strategy=strategy,
        )
        result = await service.run_backtest(request)
        return result.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Backtest failed: %s", exc)
        # Return empty result instead of crashing
        return {
            "strategy": strategy,
            "profile": profile,
            "timeframe": timeframe,
            "symbols": symbols,
            "error": str(exc),
            "metrics": {},
            "signals": []
        }


@mcp.tool()
@log_tool
async def pro_scanner_metrics(period_days: int = 14, strategy: str | None = None) -> List[dict[str, Any]]:
    try:
        service = await _ensure_scanner_service()
        metrics = await service.get_metrics(period_days, strategy)
        return metrics
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to get metrics: %s", exc)
        # Return empty metrics instead of crashing
        return []


@mcp.tool()
@log_tool
async def pro_scanner_metrics_snapshot(
    symbols: List[str],
    strategy: str = "momentum",
    profile: str = "balanced",
    include_signals: bool = False,
) -> dict[str, Any]:
    """Compute live metrics for a strategy using current scanner signals."""
    if isinstance(symbols, str):  # type: ignore[arg-type]
        symbols = [symbols]
    symbols = [sym.upper() for sym in symbols if sym]
    if not symbols:
        raise ValueError("symbols must contain at least one ticker")

    strategy_key = (strategy or "momentum").lower()
    profiles = _get_scanner_profiles()
    profiles.ensure_exists([profile])

    try:
        service = await _ensure_scanner_service()
    except Exception as exc:
        raise RuntimeError(f"Professional scanners unavailable: {exc}") from exc

    strategy_map = {
        "momentum": service.scan_momentum,
        "mean_revert": service.scan_mean_revert,
        "breakout": service.scan_breakout,
        "volume_profile": service.scan_volume_profile,
    }
    scan_fn = strategy_map.get(strategy_key)
    if scan_fn is None:
        raise ValueError("strategy must be one of momentum, mean_revert, breakout, volume_profile")

    signals = await scan_fn(symbols, profile)
    metrics = MetricsTracker().calculate(signals)
    response: dict[str, Any] = {
        "strategy": strategy_key,
        "profile": profile,
        "symbols": symbols,
        "metrics": metrics,
        "total_signals": len(signals),
    }
    if include_signals:
        response["signals"] = [signal.model_dump(mode="json") for signal in signals]
    return response


@mcp.tool()
@log_tool
async def pro_scanner_recent_signals(limit: int = 20) -> List[dict[str, Any]]:
    """Return most recent signals recorded by the pro scanners.
    
    Args:
        limit: Maximum number of recent signals to return (default: 20)
        
    Returns:
        List of recent signals with metadata. Empty list if database unavailable.
        
    Note:
        Requires Postgres connection. Check logs if empty results returned.
    """
    try:
        service = await _ensure_scanner_service()
        signals = await service.get_recent_signals(limit)
        if not signals:
            logger.warning("No recent signals found. This could mean: 1) No signals generated yet, 2) Postgres unavailable, 3) Database empty")
        return signals
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to get recent signals: %s", exc)
        # Return empty signals instead of crashing
        return []


@mcp.tool()
@log_tool
async def pro_scanner_feedback(signal_id: int, action: str) -> dict[str, Any]:
    try:
        service = await _ensure_scanner_service()
        accepted = await service.submit_feedback(signal_id, action)
        return {"signal_id": signal_id, "action": action.lower(), "recorded": accepted}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to submit feedback: %s", exc)
        return {"signal_id": signal_id, "action": action.lower(), "recorded": False, "error": str(exc)}


@mcp.tool()
@log_tool
async def pro_scanner_cache_clear(namespace: str | None = None) -> dict[str, Any]:
    try:
        service = await _ensure_scanner_service()
        removed = await service.clear_cache(namespace)
        return {"namespace": namespace, "removed": removed}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to clear cache: %s", exc)
        return {"namespace": namespace, "removed": 0, "error": str(exc)}


@mcp.tool()
@log_tool
async def pro_scanner_get_recently_results(
    scanner_name: str | None = None,
    limit: int = 50
) -> List[dict[str, Any]]:
    """Получить историю последних сканирований Professional Scanners.
    
    Args:
        scanner_name: Имя сканера для фильтрации (pro_momentum_scan, pro_mean_revert_scan, etc.)
        limit: Максимальное количество результатов (по умолчанию 50)
        
    Returns:
        Список последних результатов сканирования с метаданными.
    """
    try:
        import redis
        import json
        
        # Подключаемся к Redis
        redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
        
        results = []
        
        # Получаем список последних результатов
        recent_ids = redis_client.lrange("scanner_results:recent", 0, limit - 1)
        
        for request_id in recent_ids:
            try:
                # Получаем данные из Redis
                cache_data = redis_client.get(f"scanner_result:{request_id}")
                if not cache_data:
                    continue
                    
                data = json.loads(cache_data)
                
                # Фильтруем по scanner_name если указан
                if scanner_name and data.get("scanner_name") != scanner_name:
                    continue
                
                results.append({
                    "request_id": request_id,
                    "scanner_name": data["scanner_name"],
                    "symbols": data["symbols"],
                    "profile": data["profile"],
                    "timestamp": data["timestamp"],
                    "status": data["status"],
                    "result": data["result"]
                })
                
            except Exception as exc:
                logger.warning(f"Failed to load cached result {request_id}: {exc}")
                continue
        
        logger.info(f"Retrieved {len(results)} recent scan results for scanner: {scanner_name or 'all'}")
        return results
        
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to get recent scan results: %s", exc)
        return []


@mcp.tool()
@log_tool
def pro_scanner_scheduler_status() -> dict[str, Any]:
    """Проверить статус планировщика сканеров.
    
    Returns:
        Статус планировщика включая время последнего запуска и конфигурацию.
    """
    try:
        status = scheduler.get_status()
        logger.info(f"Scheduler status: {status}")
        return status
        
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to get scheduler status: %s", exc)
        return {
            "is_running": False,
            "last_run_time": None,
            "config": {},
            "task_running": False,
            "error": str(exc)
        }


@mcp.tool()
@log_tool
async def get_derivatives_context(symbol: str) -> dict[str, Any]:
    """Compatibility helper that fetches derivative context via Binance MCP."""
    if not symbol:
        raise ValueError("symbol is required")
    try:
        service = await _ensure_scanner_service()
    except Exception as exc:
        raise RuntimeError(f"Professional scanners unavailable: {exc}") from exc
    snapshot = await service.adapter.fetch_derivatives(symbol.upper())
    return snapshot.model_dump(mode="json")


def _compute_top_gainers(exchange: str, timeframe: str, limit: int) -> tuple[str, str, list[dict[str, Any]]]:
    exchange_clean = sanitize_exchange(exchange, "KUCOIN")
    timeframe_clean = sanitize_timeframe(timeframe, "15m")
    limit_clamped = max(1, min(int(limit), 50))

    rows = _fetch_trending_analysis(exchange_clean, timeframe=timeframe_clean, limit=limit_clamped)
    payload = [
        {
            "symbol": row["symbol"],
            "changePercent": row["changePercent"],
            "indicators": dict(row["indicators"]),
        }
        for row in rows[:limit_clamped]
    ]
    return exchange_clean, timeframe_clean, payload


@mcp.tool()
@log_tool
def top_gainers_batch(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run multiple top_gainers presets in a single call."""
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")

    response: List[Dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            raise ValueError("Each request entry must be an object")
        exchange = item.get("exchange", "KUCOIN")
        timeframe = item.get("timeframe", "15m")
        limit = item.get("limit", 25)
        exchange_clean, timeframe_clean, results = _compute_top_gainers(
            exchange, timeframe, limit
        )
        response.append(
            {
                "exchange": exchange_clean,
                "timeframe": timeframe_clean,
                "results": results,
            }
        )
    return response


def _compute_multi_changes(
    exchange: str,
    timeframes: Optional[List[str]],
    base_timeframe: str,
    limit: int,
) -> tuple[str, List[str], str, List[dict[str, Any]]]:
    exchange_clean = sanitize_exchange(exchange, "KUCOIN")
    base_tf_clean = sanitize_timeframe(base_timeframe, "4h")

    tfs_clean: List[str] | None = None
    if timeframes:
        cleaned: List[str] = []
        for tf in timeframes:
            tf_s = (tf or "").strip()
            if tf_s in ALLOWED_TIMEFRAMES:
                cleaned.append(tf_s)
        tfs_clean = cleaned or None

    limit_clamped = max(1, min(200, int(limit)))

    rows = _fetch_multi_changes(
        exchange=exchange_clean,
        timeframes=tfs_clean,
        base_timeframe=base_tf_clean,
        limit=limit_clamped,
    )
    out: List[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "symbol": r["symbol"],
                "changes": {
                    k: (None if v is None else float(round(v, 3)))
                    for k, v in r["changes"].items()
                },
                "base_indicators": {
                    k: (None if v is None else float(v))
                    for k, v in r["base_indicators"].items()
                },
            }
        )

    if out:
        inferred_timeframes = sorted(out[0]["changes"].keys())
    else:
        inferred_timeframes = tfs_clean or []

    return exchange_clean, inferred_timeframes, base_tf_clean, out


@mcp.tool()
@log_tool
def multi_changes_batch(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run multiple multi_changes presets in a single call."""
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")

    response: List[Dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            raise ValueError("Each request entry must be an object")
        exchange = item.get("exchange", "KUCOIN")
        timeframes = item.get("timeframes")
        base_timeframe = item.get("base_timeframe", "4h")
        limit = item.get("limit", 50)
        exchange_clean, timeframes_used, base_tf_clean, results = _compute_multi_changes(
            exchange, timeframes, base_timeframe, limit
        )
        response.append(
            {
                "exchange": exchange_clean,
                "timeframes": timeframes_used,
                "base_timeframe": base_tf_clean,
                "results": results,
            }
        )
    return response


def _compute_top_losers(exchange: str, timeframe: str, limit: int) -> tuple[str, str, list[dict[str, Any]]]:
    exchange_clean = sanitize_exchange(exchange, "KUCOIN")
    timeframe_clean = sanitize_timeframe(timeframe, "15m")
    limit_clamped = max(1, min(int(limit), 50))

    rows = _fetch_trending_analysis(exchange_clean, timeframe=timeframe_clean, limit=limit_clamped)
    rows.sort(key=lambda x: x["changePercent"])
    payload = [
        {
            "symbol": row["symbol"],
            "changePercent": row["changePercent"],
            "indicators": dict(row["indicators"]),
        }
        for row in rows[:limit_clamped]
    ]
    return exchange_clean, timeframe_clean, payload

@mcp.tool()
@log_tool
def top_losers_batch(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run multiple top_losers presets in a single call."""
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")

    response: List[Dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            raise ValueError("Each request entry must be an object")
        exchange = item.get("exchange", "KUCOIN")
        timeframe = item.get("timeframe", "15m")
        limit = item.get("limit", 25)
        exchange_clean, timeframe_clean, results = _compute_top_losers(
            exchange, timeframe, limit
        )
        response.append(
            {
                "exchange": exchange_clean,
                "timeframe": timeframe_clean,
                "results": results,
            }
        )
    return response


def _compute_bollinger_scan(
    exchange: str, timeframe: str, bbw_threshold: float, limit: int
) -> tuple[str, str, float, list[dict[str, Any]]]:
    exchange_clean = sanitize_exchange(exchange, "KUCOIN")
    timeframe_clean = sanitize_timeframe(timeframe, "4h")
    try:
        bbw_value = float(bbw_threshold)
    except (TypeError, ValueError):
        bbw_value = 0.04
    limit_clamped = max(1, min(int(limit), 100))

    rows = _fetch_bollinger_analysis(
        exchange_clean,
        timeframe=timeframe_clean,
        bbw_filter=bbw_value,
        limit=limit_clamped,
    )
    payload = [
        {
            "symbol": row["symbol"],
            "changePercent": row["changePercent"],
            "indicators": dict(row["indicators"]),
        }
        for row in rows[:limit_clamped]
    ]
    return exchange_clean, timeframe_clean, bbw_value, payload


@mcp.tool()
@log_tool
def bollinger_scan_batch(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run multiple bollinger scans in a single call."""
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")

    response: List[Dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            raise ValueError("Each request entry must be an object")
        exchange = item.get("exchange", "KUCOIN")
        timeframe = item.get("timeframe", "4h")
        bbw_threshold = item.get("bbw_threshold", 0.04)
        limit = item.get("limit", 50)
        exchange_clean, timeframe_clean, bbw_value, results = _compute_bollinger_scan(
            exchange, timeframe, bbw_threshold, limit
        )
        response.append(
            {
                "exchange": exchange_clean,
                "timeframe": timeframe_clean,
                "bbw_threshold": bbw_value,
                "results": results,
            }
        )
    return response




@mcp.tool()
@log_tool
def coin_analysis_batch(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute coin analysis for multiple symbols in one call."""
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")

    def _safe_num(v: Any, ndigits: Optional[int] = None) -> Optional[float]:
        try:
            if v is None:
                return None
            x = float(v)
            if not math.isfinite(x):
                return None
            return round(x, ndigits) if isinstance(ndigits, int) else x
        except Exception:
            return None

    def _analyze_single_coin(symbol: str, exchange: str, timeframe: str) -> dict:
        try:
            exchange_clean = sanitize_exchange(exchange, "KUCOIN")
            timeframe_clean = sanitize_timeframe(timeframe, "15m")
            
            # Format symbol with exchange prefix
            if ":" not in symbol:
                full_symbol = f"{exchange_clean.upper()}:{symbol.upper()}"
            else:
                full_symbol = symbol.upper()
            
            screener = EXCHANGE_SCREENER.get(exchange_clean, "crypto")
            
            try:
                analysis = get_multiple_analysis(
                    screener=screener,
                    interval=timeframe_clean,
                    symbols=[full_symbol]
                )
                
                if full_symbol not in analysis or analysis[full_symbol] is None:
                    return {
                        "error": f"No data found for {symbol} on {exchange_clean}",
                        "symbol": symbol,
                        "exchange": exchange_clean,
                        "timeframe": timeframe_clean
                    }
                
                data = analysis[full_symbol]
                indicators = data.indicators
                
                # Calculate all metrics
                metrics = compute_metrics(indicators)
                if not metrics:
                    return {
                        "error": f"Could not compute metrics for {symbol}",
                        "symbol": symbol,
                        "exchange": exchange_clean,
                        "timeframe": timeframe_clean
                    }
                
                # Additional technical indicators (raw)
                macd = indicators.get("MACD.macd")
                macd_signal = indicators.get("MACD.signal")
                adx = indicators.get("ADX")
                stoch_k = indicators.get("Stoch.K")
                stoch_d = indicators.get("Stoch.D")
                
                # Volume analysis
                volume = indicators.get("volume")
                
                # Price levels
                high = indicators.get("high")
                low = indicators.get("low")
                open_price = indicators.get("open")
                close_price = indicators.get("close")
                
                # Sanitize/round values to ensure JSON-safe output
                cur_price = _safe_num(metrics.get('price'))
                chg = _safe_num(metrics.get('change'), 3)
                bbw = _safe_num(metrics.get('bbw'), 4)

                bb_u = _safe_num(indicators.get("BB.upper"), 6)
                bb_m = _safe_num(indicators.get("SMA20"), 6)
                bb_l = _safe_num(indicators.get("BB.lower"), 6)

                rsi_val = _safe_num(indicators.get("RSI"), 2)
                ema50_val = _safe_num(indicators.get("EMA50"), 6)
                ema200_val = _safe_num(indicators.get("EMA200"), 6)

                macd_val = _safe_num(macd, 6)
                macd_sig_val = _safe_num(macd_signal, 6)
                macd_div = _safe_num((float(macd) - float(macd_signal)) if (macd is not None and macd_signal is not None) else None, 6)
                adx_val = _safe_num(adx, 2)
                stoch_k_val = _safe_num(stoch_k, 2)
                stoch_d_val = _safe_num(stoch_d, 2)

                o = _safe_num(open_price, 6)
                h = _safe_num(high, 6)
                l = _safe_num(low, 6)
                c = _safe_num(close_price, 6)
                vol = _safe_num(volume)

                position = None
                try:
                    if c is not None and bb_u is not None and bb_l is not None:
                        position = "Above Upper" if c > bb_u else ("Below Lower" if c < bb_l else "Within Bands")
                except Exception:
                    position = None
                
                return {
                    "symbol": full_symbol,
                    "exchange": exchange_clean,
                    "timeframe": timeframe_clean,
                    "timestamp": "real-time",
                    "price_data": {
                        "current_price": cur_price,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "change_percent": chg,
                        "volume": vol
                    },
                    "bollinger_analysis": {
                        "rating": int(metrics['rating']) if isinstance(metrics.get('rating'), (int, float)) else metrics.get('rating'),
                        "signal": metrics['signal'],
                        "bbw": bbw,
                        "bb_upper": bb_u,
                        "bb_middle": bb_m,
                        "bb_lower": bb_l,
                        "position": position
                    },
                    "technical_indicators": {
                        "rsi": rsi_val,
                        "rsi_signal": ("Overbought" if (rsi_val is not None and rsi_val > 70) else ("Oversold" if (rsi_val is not None and rsi_val < 30) else "Neutral")),
                        "sma20": bb_m,
                        "ema50": ema50_val,
                        "ema200": ema200_val,
                        "macd": macd_val,
                        "macd_signal": macd_sig_val,
                        "macd_divergence": macd_div,
                        "adx": adx_val,
                        "trend_strength": ("Strong" if (adx_val is not None and adx_val > 25) else "Weak"),
                        "stoch_k": stoch_k_val,
                        "stoch_d": stoch_d_val
                    },
                    "market_sentiment": {
                        "overall_rating": int(metrics['rating']) if isinstance(metrics.get('rating'), (int, float)) else metrics.get('rating'),
                        "buy_sell_signal": metrics['signal'],
                        "volatility": ("High" if (bbw is not None and bbw > 0.05) else ("Medium" if (bbw is not None and bbw > 0.02) else "Low")),
                        "momentum": ("Bullish" if (chg is not None and chg > 0) else "Bearish")
                    }
                }
                
            except Exception as e:
                logger.exception("coin_analysis failed while computing indicators for %s", full_symbol)
                return {
                    "error": f"Analysis failed: {str(e)}",
                    "symbol": symbol,
                    "exchange": exchange_clean,
                    "timeframe": timeframe_clean
                }
                
        except Exception as e:
            logger.exception("coin_analysis encountered an unexpected error for %s", symbol)
            return {
                "error": f"Coin analysis failed: {str(e)}",
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }

    results: List[Dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            raise ValueError("Each request entry must be an object")
        symbol = item.get("symbol")
        if not symbol:
            results.append({"error": "symbol is required"})
            continue
        exchange = item.get("exchange", "KUCOIN")
        timeframe = item.get("timeframe", "15m")
        result = _analyze_single_coin(symbol=symbol, exchange=exchange, timeframe=timeframe)
        results.append(result)
    return results


@mcp.tool()
@log_tool
def consecutive_candles_scan_batch(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run multiple consecutive candles scans in a single call."""
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")

    def _scan_consecutive_candles(exchange: str, timeframe: str, pattern_type: str, candle_count: int, min_growth: float, limit: int) -> dict:
        try:
            exchange_clean = sanitize_exchange(exchange, "KUCOIN")
            timeframe_clean = sanitize_timeframe(timeframe, "15m")
            candle_count = max(2, min(5, candle_count))
            min_growth = max(0.5, min(20.0, min_growth))
            limit = max(1, min(50, limit))
            
            # Get symbols for the exchange
            symbols = load_symbols(exchange_clean)
            if not symbols:
                return {
                    "error": f"No symbols found for exchange: {exchange_clean}",
                    "exchange": exchange_clean,
                    "timeframe": timeframe_clean
                }
            
            # Use only well-known symbols to avoid API issues
            well_known_symbols = {
                "KUCOIN": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"],
                "BINANCE": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"],
                "BYBIT": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"]
            }
            
            exchange_upper = exchange_clean.upper()
            if exchange_upper in well_known_symbols:
                symbols = well_known_symbols[exchange_upper][:5]  # Use only 5 well-known symbols
            else:
                symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # Fallback to major coins
            
            # Format symbols for TradingView API
            formatted_symbols = [f"{exchange_upper}:{symbol}" for symbol in symbols]
            
            # Get analysis data
            analysis_data = get_multiple_analysis(
                screener=EXCHANGE_SCREENER.get(exchange_upper, "crypto"),
                interval=timeframe_clean,
                symbols=formatted_symbols
            )
            
            results = []
            for symbol in symbols:
                full_symbol = f"{exchange_upper}:{symbol}"
                if full_symbol in analysis_data and analysis_data[full_symbol]:
                    data = analysis_data[full_symbol]
                    indicators = data.indicators
                    
                    # Check for consecutive candle pattern
                    if _check_consecutive_pattern(indicators, pattern_type, candle_count, min_growth):
                        results.append({
                            "symbol": symbol,
                            "exchange": exchange_clean,
                            "timeframe": timeframe_clean,
                            "pattern_type": pattern_type,
                            "candle_count": candle_count,
                            "min_growth": min_growth,
                            "indicators": {
                                "close": indicators.get("close"),
                                "open": indicators.get("open"),
                                "high": indicators.get("high"),
                                "low": indicators.get("low"),
                                "volume": indicators.get("volume")
                            }
                        })
            
            return {
                "exchange": exchange_clean,
                "timeframe": timeframe_clean,
                "pattern_type": pattern_type,
                "candle_count": candle_count,
                "min_growth": min_growth,
                "results": results[:limit],
                "total_found": len(results)
            }
            
        except Exception as e:
            logger.exception("consecutive_candles_scan failed for %s: %s", exchange, e)
            return {
                "error": f"Scan failed: {str(e)}",
                "exchange": exchange,
                "timeframe": timeframe
            }

    response: List[Dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            raise ValueError("Each request entry must be an object")
        
        exchange = item.get("exchange", "KUCOIN")
        timeframe = item.get("timeframe", "15m")
        pattern_type = item.get("pattern_type", "bullish")
        candle_count = item.get("candle_count", 3)
        min_growth = item.get("min_growth", 2.0)
        limit = item.get("limit", 20)
        
        result = _scan_consecutive_candles(exchange, timeframe, pattern_type, candle_count, min_growth, limit)
        response.append(result)
    
    return response


@mcp.tool()
@log_tool
def advanced_candle_pattern_batch(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run multiple advanced candle pattern scans in a single call."""
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")

    def _scan_advanced_pattern(exchange: str, base_timeframe: str, pattern_length: int, min_size_increase: float, limit: int) -> dict:
        try:
            exchange_clean = sanitize_exchange(exchange, "KUCOIN")
            base_timeframe_clean = sanitize_timeframe(base_timeframe, "15m")
            pattern_length = max(2, min(4, pattern_length))
            min_size_increase = max(5.0, min(50.0, min_size_increase))
            limit = max(1, min(30, limit))
            
            # Get symbols
            symbols = load_symbols(exchange_clean)
            if not symbols:
                return {
                    "error": f"No symbols found for exchange: {exchange_clean}",
                    "exchange": exchange_clean
                }
            
            # Limit for performance
            symbols = symbols[:min(limit * 2, 100)]
            
            # Use tradingview-screener for multi-timeframe data if available
            if TRADINGVIEW_SCREENER_AVAILABLE:
                try:
                    # Get multiple timeframe data using screener
                    results = _fetch_multi_timeframe_patterns(
                        exchange_clean, symbols, base_timeframe_clean, pattern_length, min_size_increase
                    )
                    
                    return {
                        "exchange": exchange_clean,
                        "base_timeframe": base_timeframe_clean,
                        "pattern_length": pattern_length,
                        "min_size_increase": min_size_increase,
                        "results": results[:limit],
                        "total_found": len(results)
                    }
                    
                except Exception as e:
                    logger.exception("Multi-timeframe pattern analysis failed: %s", e)
                    return {
                        "error": f"Multi-timeframe analysis failed: {str(e)}",
                        "exchange": exchange_clean,
                        "base_timeframe": base_timeframe_clean
                    }
            else:
                # Fallback to single timeframe analysis
                return {
                    "error": "TradingView screener not available",
                    "exchange": exchange_clean,
                    "base_timeframe": base_timeframe_clean,
                    "suggestion": "Install tradingview-screener package for advanced pattern analysis"
                }
                
        except Exception as e:
            logger.exception("Advanced pattern analysis failed: %s", e)
            return {
                "error": f"Advanced pattern analysis failed: {str(e)}",
                "exchange": exchange,
                "base_timeframe": base_timeframe
            }

    response: List[Dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            raise ValueError("Each request entry must be an object")
        
        exchange = item.get("exchange", "KUCOIN")
        base_timeframe = item.get("base_timeframe", "15m")
        pattern_length = item.get("pattern_length", 3)
        min_size_increase = item.get("min_size_increase", 10.0)
        limit = item.get("limit", 15)
        
        result = _scan_advanced_pattern(exchange, base_timeframe, pattern_length, min_size_increase, limit)
        response.append(result)
    
    return response


@mcp.tool()
@log_tool
def trend_breakout_pyramiding(
    exchange: str = "KUCOIN",
    timeframe: str = "1h",
    # Context
    btc_symbol: str = "BINANCE:BTCUSDT",
    # Symbol filter
    quote: Optional[str] = None,  # e.g., "USDT" to keep only USDT pairs
    adx_min: float = 10.0,
    kc_mult: float = 1.5,
    kc_min_width: float = 0.015,
    # Entry triggers
    atr_k: float = 1.4,
    pb_long_min: float = 0.95,
    pb_short_max: float = 0.05,
    # Risk
    sl_mult: float = 1.5,
    tp1_rr: float = 1.75,
    trail_mult: float = 2.0,
    # Pyramiding
    add_step_atr: float = 1.0,
    max_adds: int = 2,
    add_fraction: float = 0.7,
    # Money/Risk
    equity: float = 10000.0,
    risk_pct: float = 0.02,
    # Scan
    limit: int = 20
) -> list[dict]:
    """Trend Breakout + Pyramiding screener using snapshot-friendly heuristics.

    Heuristics:
      - Breakout: close > BB.upper (or < BB.lower for short)
      - Micro-breakout: %b > 0.95 (or < 0.05)
      - ATR impulse: |close-open| > atr_k * ATR
      - Anti-flat: ADX >= adx_min and Keltner width >= kc_min_width (KC derived from ATR/EMA20 or BBW fallback)
      - Trend context (4h EMA200): allow long/short if BTC or local pair supports it
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        timeframe = sanitize_timeframe(timeframe, "1h")
        limit = max(1, min(50, int(limit)))
        max_scan = min(limit * 3, 120)

        syms = load_symbols(exchange)
        if quote:
            q = (quote or "").upper()
            syms = [s for s in syms if s.split(":", 1)[-1].upper().endswith(q)]
        syms = syms[:max_scan]
        if not syms:
            return []

        ind_map = _fetch_tv_indicators(syms, timeframe, exchange)
        if not ind_map:
            logger.warning("No indicator data available for trend_breakout_pyramiding on %s", exchange)
            return []

        pair_symbols = [sym.split(":", 1)[-1] for sym in ind_map.keys()]
        pair_symbols = list(dict.fromkeys(pair_symbols))
        context_map = _bulk_market_trend_context(exchange, pair_symbols, btc_symbol, trend_tf="4h")
        default_ctx = {
            "btc_bullish": None,
            "pair_bullish": None,
            "allow_long": False,
            "allow_short": False,
        }

        threshold_sets = [
            {
                "adx_min": adx_min,
                "kc_min_width": kc_min_width,
                "pb_long_min": pb_long_min,
                "pb_short_max": pb_short_max,
                "atr_k": atr_k,
                "note": None,
                "force_allow": False,
                "loose_trigger": False,
            },
            {
                "adx_min": max(6.0, adx_min * 0.75),
                "kc_min_width": kc_min_width * 0.8,
                "pb_long_min": max(0.85, pb_long_min - 0.05),
                "pb_short_max": min(0.15, pb_short_max + 0.05),
                "atr_k": max(0.8, atr_k * 0.85),
                "note": "Relaxed thresholds (level 1)",
                "force_allow": False,
                "loose_trigger": False,
            },
            {
                "adx_min": max(4.0, adx_min * 0.6),
                "kc_min_width": kc_min_width * 0.6,
                "pb_long_min": max(0.75, pb_long_min - 0.1),
                "pb_short_max": min(0.25, pb_short_max + 0.1),
                "atr_k": max(0.6, atr_k * 0.75),
                "note": "Relaxed thresholds (level 2)",
                "force_allow": True,
                "loose_trigger": False,
            },
            {
                "adx_min": max(2.5, adx_min * 0.5),
                "kc_min_width": kc_min_width * 0.4,
                "pb_long_min": max(0.7, pb_long_min - 0.15),
                "pb_short_max": min(0.3, pb_short_max + 0.15),
                "atr_k": max(0.4, atr_k * 0.6),
                "note": "Exploratory candidates (very relaxed)",
                "force_allow": True,
                "loose_trigger": True,
            },
        ]

        for scenario in threshold_sets:
            scenario_results: list[dict] = []
            adx_req = scenario["adx_min"]
            kc_req = scenario["kc_min_width"]
            pb_long_req = scenario["pb_long_min"]
            pb_short_req = scenario["pb_short_max"]
            atr_k_req = scenario["atr_k"]
            relaxed_note = scenario["note"]
            force_allow = scenario.get("force_allow", False)
            loose_trigger = scenario.get("loose_trigger", False)

            for full_sym, ind in ind_map.items():
                try:
                    open_p = ind.get("open"); close_p = ind.get("close")
                    bb_u = ind.get("BB.upper"); bb_l = ind.get("BB.lower")
                    rsi = ind.get("RSI"); adx = ind.get("ADX")
                    atr = ind.get("ATR")

                    k_width = _keltner_width(ind, kc_mult)
                    if adx is None or adx < adx_req:
                        continue
                    if k_width is None or k_width < kc_req:
                        continue

                    pair_symbol = full_sym.split(":", 1)[-1]
                    ctx = context_map.get(full_sym) or context_map.get(pair_symbol) or default_ctx

                    pb = _pct_b(ind)

                    breakout_long = (close_p is not None and bb_u is not None and close_p > bb_u)
                    breakout_short = (close_p is not None and bb_l is not None and close_p < bb_l)
                    micro_long = (pb is not None and pb >= pb_long_req)
                    micro_short = (pb is not None and pb <= pb_short_req)
                    atr_imp_long = (
                        open_p is not None
                        and close_p is not None
                        and atr not in (None, 0)
                        and (close_p - open_p) > atr_k_req * atr
                    )
                    atr_imp_short = (
                        open_p is not None
                        and close_p is not None
                        and atr not in (None, 0)
                        and (open_p - close_p) > atr_k_req * atr
                    )

                    if loose_trigger:
                        # Allow EMA proximity or RSI momentum to act as trigger when breakout not detected
                        ema20 = ind.get("EMA20")
                        ema_condition_long = ema20 and close_p and ((close_p - ema20) / ema20 >= 0.01)
                        ema_condition_short = ema20 and close_p and ((ema20 - close_p) / ema20 >= 0.01)
                        rsi = ind.get("RSI")
                        breakout_long = breakout_long or ema_condition_long or (rsi is not None and rsi >= 60)
                        breakout_short = breakout_short or ema_condition_short or (rsi is not None and rsi <= 40)

                    allow_long = ctx.get("allow_long", False) or (
                        ctx.get("btc_bullish") is None and ctx.get("pair_bullish") is None
                    )
                    allow_short = ctx.get("allow_short", False) or (
                        ctx.get("btc_bullish") is None and ctx.get("pair_bullish") is None
                    )
                    if force_allow:
                        allow_long = allow_short = True

                    long_ok = (breakout_long or micro_long or atr_imp_long) and allow_long
                    short_ok = (breakout_short or micro_short or atr_imp_short) and allow_short

                    if not long_ok and not short_ok:
                        continue

                    direction = "long" if long_ok and not short_ok else (
                        "short" if short_ok and not long_ok else ("long" if (pb or 0.5) >= 0.5 else "short")
                    )

                    price = close_p
                    if price in (None, 0):
                        continue
                    side_mult = 1 if direction == "long" else -1
                    sl = price - side_mult * (sl_mult * (atr or 0))
                    rr_units = sl_mult * (atr or 0)
                    tp1 = price + side_mult * (tp1_rr * rr_units)
                    pos = _position_sizing(equity, risk_pct, atr, sl_mult, price)

                    adds: list[dict] = []
                    step = add_step_atr * (atr or 0)
                    for i in range(max(0, int(max_adds))):
                        add_price = price + side_mult * (step * (i + 1))
                        adds.append({
                            "add_index": i + 1,
                            "price": round(add_price, 8),
                            "fraction": round(add_fraction, 3),
                        })

                    reasons = []
                    if relaxed_note:
                        reasons.append(relaxed_note)
                    if breakout_long or breakout_short:
                        reasons.append("Donchian/BB breakout proxy")
                    if micro_long or micro_short:
                        reasons.append("Micro-breakout (%b)")
                    if atr_imp_long or atr_imp_short:
                        reasons.append("ATR impulse")

                    scenario_results.append({
                        "symbol": full_sym,
                        "timeframe": timeframe,
                        "direction": direction,
                        "context": {
                            "btc_bullish": ctx.get("btc_bullish"),
                            "pair_bullish": ctx.get("pair_bullish"),
                            "adx": adx,
                            "keltner_width": k_width,
                        },
                        "entry_price": round(price, 8),
                        "sl": round(sl, 8),
                        "tp1": round(tp1, 8),
                        "trail_mult_atr": trail_mult,
                        "position": pos,
                        "pyramiding": {
                            "max_adds": int(max_adds),
                            "add_fraction": round(add_fraction, 3),
                            "add_step_atr": add_step_atr,
                            "adds": adds,
                        },
                        "indicators_used": {
                            "ATR": atr,
                            "RSI": rsi,
                            "BB_upper": bb_u,
                            "BB_lower": bb_l,
                            "%b": pb,
                        },
                        "reasons": reasons,
                    })
                except Exception:
                    continue

            if scenario_results:
                return scenario_results[:limit]

        return []
    except Exception:
        logger.exception("trend_breakout_pyramiding encountered an unexpected error")
        return []


@mcp.tool()
@log_tool
def pullback_engine(
    exchange: str = "KUCOIN",
    timeframe: str = "1h",
    # Symbol filter
    quote: Optional[str] = None,  # e.g., "USDT" to keep only USDT pairs
    # Trend filter
    ema_fast: int = 20,
    ema_slow: int = 50,
    btc_symbol: str = "BINANCE:BTCUSDT",
    adx_min: float = 8.0,
    adx_max: float = 50.0,
    # Confirmation
    rsi_len: int = 14,
    rsi_long_th: float = 45.0,
    rsi_short_th: float = 55.0,
    b_long_th: float = 0.40,
    b_short_th: float = 0.60,
    # Risk
    sl_mult: float = 1.5,
    tp1_rr: float = 1.5,
    trail_mult: float = 2.0,
    # Retest add
    enable_retest_add: bool = True,
    retest_fraction: float = 0.5,
    # Money
    equity: float = 10000.0,
    risk_pct: float = 0.02,
    # Scan
    limit: int = 20
) -> list[dict]:
    """Pullback Engine (EMA/RSI/%b) screener.

    Entry:
      - In-trend: EMA_fast > EMA_slow (or < for short) and BTC/local 4h context allows direction
      - Pullback zone: price near/under EMA_fast in uptrend (or over in downtrend)
      - Confirmation: RSI <= rsi_long_th and %b <= b_long_th for long (mirror for short)
      - Trigger: current bar crosses back over EMA_fast (open <= EMA_fast and close > EMA_fast) for long; reverse for short
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        timeframe = sanitize_timeframe(timeframe, "1h")
        limit = max(1, min(50, int(limit)))
        max_scan = min(limit * 3, 120)

        syms = load_symbols(exchange)
        if quote:
            q = (quote or "").upper()
            syms = [s for s in syms if s.split(":", 1)[-1].upper().endswith(q)]
        syms = syms[:max_scan]
        if not syms:
            return []

        ind_map = _fetch_tv_indicators(syms, timeframe, exchange)
        if not ind_map:
            logger.warning("No indicator data available for pullback_engine on %s", exchange)
            return []

        pair_symbols = [sym.split(":", 1)[-1] for sym in ind_map.keys()]
        pair_symbols = list(dict.fromkeys(pair_symbols))
        context_map = _bulk_market_trend_context(exchange, pair_symbols, btc_symbol, trend_tf="4h")
        default_ctx = {
            "btc_bullish": None,
            "pair_bullish": None,
            "allow_long": False,
            "allow_short": False,
        }

        scenarios = [
            {
                "adx_min": adx_min,
                "adx_max": adx_max,
                "rsi_long": rsi_long_th,
                "rsi_short": rsi_short_th,
                "b_long": b_long_th,
                "b_short": b_short_th,
                "zone_tol": 0.02,
                "note": None,
                "force_allow": False,
            },
            {
                "adx_min": max(5.0, adx_min * 0.8),
                "adx_max": adx_max + 10,
                "rsi_long": min(55.0, rsi_long_th + 5),
                "rsi_short": max(45.0, rsi_short_th - 5),
                "b_long": min(0.55, b_long_th + 0.1),
                "b_short": max(0.45, b_short_th - 0.1),
                "zone_tol": 0.03,
                "note": "Relaxed thresholds (level 1)",
                "force_allow": False,
            },
            {
                "adx_min": max(3.0, adx_min * 0.6),
                "adx_max": adx_max + 20,
                "rsi_long": min(60.0, rsi_long_th + 8),
                "rsi_short": max(40.0, rsi_short_th - 8),
                "b_long": min(0.65, b_long_th + 0.15),
                "b_short": max(0.35, b_short_th - 0.15),
                "zone_tol": 0.04,
                "note": "Relaxed thresholds (level 2)",
                "force_allow": True,
            },
        ]

        for scenario in scenarios:
            scenario_results: list[dict] = []
            adx_min_req = scenario["adx_min"]
            adx_max_req = scenario["adx_max"]
            rsi_long_req = scenario["rsi_long"]
            rsi_short_req = scenario["rsi_short"]
            b_long_req = scenario["b_long"]
            b_short_req = scenario["b_short"]
            zone_tol = scenario["zone_tol"]
            relaxed_note = scenario["note"]
            force_allow = scenario.get("force_allow", False)

            for full_sym, ind in ind_map.items():
                try:
                    open_p = ind.get("open"); close_p = ind.get("close")
                    ema_fast_val = ind.get(f"EMA{ema_fast}") or ind.get("EMA20")
                    ema_slow_val = ind.get(f"EMA{ema_slow}") or ind.get("EMA50")
                    rsi = ind.get("RSI")
                    adx = ind.get("ADX")
                    atr = ind.get("ATR")
                    bb_u = ind.get("BB.upper"); bb_l = ind.get("BB.lower")

                    if None in (open_p, close_p, ema_fast_val, ema_slow_val, rsi, adx):
                        continue

                    if not (adx_min_req <= adx <= adx_max_req):
                        continue

                    pb = _pct_b(ind)
                    if pb is None:
                        continue

                    pair_symbol = full_sym.split(":", 1)[-1]
                    ctx = context_map.get(full_sym) or context_map.get(pair_symbol) or default_ctx

                    uptrend = ema_fast_val > ema_slow_val
                    downtrend = ema_fast_val < ema_slow_val

                    long_zone = uptrend and (abs(close_p - ema_fast_val) / ema_fast_val <= zone_tol)
                    short_zone = downtrend and (abs(close_p - ema_fast_val) / ema_fast_val <= zone_tol)

                    long_conf = (rsi is not None and rsi <= rsi_long_req) or (pb is not None and pb <= b_long_req)
                    short_conf = (rsi is not None and rsi >= rsi_short_req) or (pb is not None and pb >= b_short_req)

                    allow_long = ctx.get("allow_long", False) or (
                        ctx.get("btc_bullish") is None and ctx.get("pair_bullish") is None
                    )
                    allow_short = ctx.get("allow_short", False) or (
                        ctx.get("btc_bullish") is None and ctx.get("pair_bullish") is None
                    )
                    if force_allow:
                        allow_long = allow_short = True

                    long_trigger = long_zone and long_conf and allow_long
                    short_trigger = short_zone and short_conf and allow_short

                    if not (long_trigger or short_trigger):
                        continue

                    direction = "long" if long_trigger else "short"
                    price = close_p
                    side_mult = 1 if direction == "long" else -1
                    sl_base = min(ema_slow_val, price) if direction == "long" else max(ema_slow_val, price)
                    sl = sl_base - side_mult * (sl_mult * (atr or 0))
                    rr_units = sl_mult * (atr or 0)
                    tp1 = price + side_mult * (tp1_rr * rr_units)
                    pos = _position_sizing(equity, risk_pct, atr, sl_mult, price)

                    retest = None
                    if enable_retest_add:
                        retest = {
                            "retest_level": round(ema_fast_val, 8),
                            "fraction": round(max(0.0, min(1.0, retest_fraction)), 3),
                        }

                    reasons = ["Trend filter OK", "Pullback confirmed via RSI/%b", "EMA_fast cross trigger"]
                    if relaxed_note:
                        reasons.insert(0, relaxed_note)

                    scenario_results.append({
                        "symbol": full_sym,
                        "timeframe": timeframe,
                        "direction": direction,
                        "context": {
                            "btc_bullish": ctx.get("btc_bullish"),
                            "pair_bullish": ctx.get("pair_bullish"),
                            "adx": adx,
                        },
                        "entry_price": round(price, 8),
                        "sl": round(sl, 8),
                        "tp1": round(tp1, 8),
                        "trail_mult_atr": trail_mult,
                        "position": pos,
                        "retest_add": retest,
                        "indicators_used": {
                            f"EMA{ema_fast}": ema_fast_val,
                            f"EMA{ema_slow}": ema_slow_val,
                            "ATR": atr,
                            "RSI": rsi,
                            "%b": pb,
                            "BB_upper": bb_u,
                            "BB_lower": bb_l,
                        },
                        "reasons": reasons,
                    })
                except Exception:
                    continue

            if scenario_results:
                return scenario_results[:limit]

        return []
    except Exception:
        logger.exception("pullback_engine encountered an unexpected error")
        return []


def _score_candidate(item: dict, strategy: str) -> float:
    try:
        s = 0.0
        reasons = item.get("reasons") or []
        s += 0.5 * len(reasons)
        ctx = item.get("context") or {}
        adx = ctx.get("adx")
        if isinstance(adx, (int, float)):
            s += min(5.0, max(0.0, adx / 10.0))
        # Reward better R multiple estimate if present
        entry = item.get("entry_price")
        sl = item.get("sl")
        tp1 = item.get("tp1")
        if all(isinstance(v, (int, float)) for v in (entry, sl, tp1)) and entry != sl:
            r_mult = abs(tp1 - entry) / abs(entry - sl)
            s += min(3.0, max(0.0, r_mult))
        # Small bias by strategy type
        if strategy == "breakout":
            s += 0.4
        elif strategy == "pullback":
            s += 0.2
        return float(s)
    except Exception:
        return 0.0


@mcp.tool()
@log_tool
def unified_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "1h",
    direction: str = "both",
    quote: Optional[str] = None,
    # Strategy selection
    strategy_mode: str = "adaptive",  # "adaptive" | "breakout" | "pullback" | "smart"
    # Smart scanner filters
    min_volume: Optional[float] = None,
    min_rsi: Optional[float] = None,
    max_rsi: Optional[float] = None,
    min_adx: Optional[float] = None,
    max_adx: Optional[float] = None,
    max_bbw: Optional[float] = None,
    # Breakout fine-tuning
    adx_min: Optional[float] = None,
    kc_min_width: Optional[float] = None,
    atr_k: Optional[float] = None,
    pb_long_min: Optional[float] = None,
    pb_short_max: Optional[float] = None,
    include_breakout: bool = True,
    include_pullback: bool = True,
    limit: int = 30,
    per_strategy_limit: int = 40,
) -> dict:
    """Unified scanner combining smart filters and strategy candidates.
    
    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D)
        direction: "long", "short", or "both"
        quote: Quote currency filter (e.g., "USDT")
        strategy_mode: "adaptive" (combines all), "breakout", "pullback", "smart"
        min_volume: Minimum volume filter
        min_rsi: Minimum RSI filter
        max_rsi: Maximum RSI filter
        min_adx: Minimum ADX filter
        max_adx: Maximum ADX filter
        max_bbw: Maximum Bollinger Band Width filter
        adx_min: Minimum ADX for breakout strategy
        kc_min_width: Minimum Keltner Channel width
        atr_k: ATR multiplier for breakout detection
        pb_long_min: Minimum %b for long positions
        pb_short_max: Maximum %b for short positions
        include_breakout: Include breakout strategy
        include_pullback: Include pullback strategy
        limit: Maximum results to return
        per_strategy_limit: Per-strategy limit before merging
    
    Returns:
        Unified scan results with strategy information
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        timeframe = sanitize_timeframe(timeframe, "1h")
        direction = (direction or "both").lower()
        if direction not in ("long", "short", "both"):
            direction = "both"
        limit = max(1, min(100, int(limit)))
        per_strategy_limit = max(1, min(100, int(per_strategy_limit)))
        
        results = {
            "exchange": exchange,
            "timeframe": timeframe,
            "direction": direction,
            "strategy_mode": strategy_mode,
            "results": [],
            "strategy_breakdown": {},
            "total_found": 0
        }
        
        if strategy_mode == "adaptive" or strategy_mode == "breakout":
            # Get strategy candidates
            try:
                strategy_results = strategy_candidates(
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
                    limit=per_strategy_limit,
                    per_strategy_limit=per_strategy_limit
                )
                
                if "results" in strategy_results:
                    results["strategy_breakdown"]["strategy_candidates"] = {
                        "count": len(strategy_results["results"]),
                        "breakout": strategy_results.get("breakout", []),
                        "pullback": strategy_results.get("pullback", [])
                    }
                    results["results"].extend(strategy_results["results"])
                    
            except Exception as e:
                logger.warning("Strategy candidates failed: %s", e)
                results["strategy_breakdown"]["strategy_candidates"] = {"error": str(e)}
        
        if strategy_mode == "adaptive" or strategy_mode == "smart":
            # Get smart scanner results
            try:
                smart_results = smart_scanner(
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
                    limit=per_strategy_limit
                )
                
                results["strategy_breakdown"]["smart_scanner"] = {
                    "count": len(smart_results)
                }
                
                # Merge smart scanner results
                for item in smart_results:
                    # Add strategy source
                    item["strategy_source"] = "smart_scanner"
                    results["results"].append(item)
                    
            except Exception as e:
                logger.warning("Smart scanner failed: %s", e)
                results["strategy_breakdown"]["smart_scanner"] = {"error": str(e)}
        
        # Remove duplicates and limit results
        seen_symbols = set()
        unique_results = []
        
        for item in results["results"]:
            symbol = item.get("symbol", "")
            if symbol and symbol not in seen_symbols:
                seen_symbols.add(symbol)
                unique_results.append(item)
                
                if len(unique_results) >= limit:
                    break
        
        results["results"] = unique_results
        results["total_found"] = len(unique_results)
        
        return results
        
    except Exception as e:
        logger.exception("Unified scanner failed: %s", e)
        return {
            "error": f"Unified scanner failed: {str(e)}",
            "exchange": exchange,
            "timeframe": timeframe,
            "strategy_mode": strategy_mode
        }


@mcp.tool()
@log_tool
def strategy_candidates(
    exchange: str = "KUCOIN",
    timeframe: str = "1h",
    direction: str = "both",  # long | short | both
    quote: Optional[str] = None,
    # Breakout fine-tuning
    adx_min: Optional[float] = None,
    kc_min_width: Optional[float] = None,
    atr_k: Optional[float] = None,
    pb_long_min: Optional[float] = None,
    pb_short_max: Optional[float] = None,
    include_breakout: bool = True,
    include_pullback: bool = True,
    limit: int = 30,
    per_strategy_limit: int = 40,
) -> dict:
    """Aggregate candidates from Trend Breakout + Pyramiding and Pullback Engine.

    Filters by direction and returns a merged, de-duplicated list (scored), plus per-strategy lists.
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        timeframe = sanitize_timeframe(timeframe, "1h")
        direction = (direction or "both").lower()
        if direction not in ("long", "short", "both"):
            direction = "both"
        limit = max(1, min(100, int(limit)))
        per_strategy_limit = max(1, min(100, int(per_strategy_limit)))

        breakout_list: list[dict] = []
        pullback_list: list[dict] = []

        if include_breakout:
            try:
                breakout_list = trend_breakout_pyramiding(
                    exchange=exchange,
                    timeframe=timeframe,
                    quote=quote,
                    adx_min=adx_min if isinstance(adx_min, (int, float)) else 14.0,
                    kc_min_width=kc_min_width if isinstance(kc_min_width, (int, float)) else 0.02,
                    atr_k=atr_k if isinstance(atr_k, (int, float)) else 1.4,
                    pb_long_min=pb_long_min if isinstance(pb_long_min, (int, float)) else 0.95,
                    pb_short_max=pb_short_max if isinstance(pb_short_max, (int, float)) else 0.05,
                    limit=per_strategy_limit,
                ) or []
            except Exception:
                breakout_list = []

        if include_pullback:
            try:
                pullback_list = pullback_engine(
                    exchange=exchange,
                    timeframe=timeframe,
                    quote=quote,
                    limit=per_strategy_limit,
                ) or []
            except Exception:
                pullback_list = []

        def _dir_ok(item: dict) -> bool:
            if direction == "both":
                return True
            return (item.get("direction") == direction)

        breakout_f = [x for x in breakout_list if _dir_ok(x)]
        pullback_f = [x for x in pullback_list if _dir_ok(x)]

        # Merge by (symbol, direction)
        merged_map: dict[tuple, dict] = {}
        for it in breakout_f:
            key = (it.get("symbol"), it.get("direction"))
            merged_map[key] = {
                **it,
                "strategy": "breakout",
                "sources": ["breakout"],
                "score": _score_candidate(it, "breakout"),
            }
        for it in pullback_f:
            key = (it.get("symbol"), it.get("direction"))
            if key in merged_map:
                # Combine
                combined = merged_map[key]
                combined["sources"] = list(sorted(set((combined.get("sources") or []) + ["pullback"])))
                combined["strategy"] = "both"
                combined["score"] = max(combined.get("score", 0.0), _score_candidate(it, "pullback"))
                # Optionally choose better R multiple (keep existing fields otherwise)
                try:
                    def r_mult(e, s, t):
                        return abs(t - e) / abs(e - s) if (e is not None and s is not None and t is not None and e != s) else 0
                    e1, s1, t1 = combined.get("entry_price"), combined.get("sl"), combined.get("tp1")
                    e2, s2, t2 = it.get("entry_price"), it.get("sl"), it.get("tp1")
                    if r_mult(e2, s2, t2) > r_mult(e1, s1, t1):
                        # Prefer pullback levels if R is better
                        for k in ("entry_price", "sl", "tp1"):
                            combined[k] = it.get(k)
                except Exception:
                    pass
            else:
                merged_map[key] = {
                    **it,
                    "strategy": "pullback",
                    "sources": ["pullback"],
                    "score": _score_candidate(it, "pullback"),
                }

        merged = list(merged_map.values())
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        merged = merged[:limit]

        return {
            "exchange": exchange,
            "timeframe": timeframe,
            "direction": direction,
            "counts": {
                "breakout_total": len(breakout_list),
                "pullback_total": len(pullback_list),
                "breakout_dir_filtered": len(breakout_f),
                "pullback_dir_filtered": len(pullback_f),
                "merged": len(merged),
            },
            "merged": merged,
            "by_strategy": {
                "breakout": breakout_f[:limit],
                "pullback": pullback_f[:limit],
            },
        }
    except Exception as e:
        logger.exception("strategy_candidates encountered an unexpected error")
        return {"error": f"strategy_candidates failed: {str(e)}"}


@mcp.tool()
@log_tool
def scan_strategy_candidates(
    exchange: str = "BINANCE",
    timeframes: Optional[List[str]] = None,
    quote: Optional[str] = "USDT",
    # Fine-tuning thresholds (breakout)
    adx_min: Optional[float] = None,
    kc_min_width: Optional[float] = None,
    atr_k: Optional[float] = None,
    pb_long_min: Optional[float] = None,
    pb_short_max: Optional[float] = None,
    include_breakout: bool = True,
    include_pullback: bool = True,
    limit: int = 30,
    per_strategy_limit: int = 40,
) -> dict:
    """Run strategy_candidates for multiple timeframes and both directions.

    Uses Trend Breakout + Pyramiding and Pullback Engine to find candidates
    for long and short across selected timeframes.

    Args:
        exchange: Exchange name (e.g., BINANCE, KUCOIN)
        timeframes: List of TFs (subset of 15m, 1h, 4h, 1D). Defaults to [15m, 1h, 4h, 1D]
        include_breakout: Include breakout strategy
        include_pullback: Include pullback strategy
        limit: Max merged results per direction
        per_strategy_limit: Per-strategy pre-merge cap

    Returns:
        { exchange, timeframes, results: { tf: { long: {...}, short: {...} } }, summary }
    """
    try:
        exchange = sanitize_exchange(exchange, "BINANCE")
        limit = max(1, min(100, int(limit)))
        per_strategy_limit = max(1, min(100, int(per_strategy_limit)))

        # Default TFs and sanitize
        default_tfs = ["15m", "1h", "4h", "1D"]
        tfs = timeframes or default_tfs
        cleaned_tfs: list[str] = []
        for tf in tfs:
            tf_s = sanitize_timeframe(tf, None) if tf else None
            if tf_s in ("5m", "15m", "1h", "4h", "1D", "1W", "1M"):
                cleaned_tfs.append(tf_s)
        if not cleaned_tfs:
            cleaned_tfs = default_tfs

        out: dict[str, Any] = {
            "exchange": exchange,
            "timeframes": cleaned_tfs,
            "results": {},
            "summary": {"total_timeframes": len(cleaned_tfs), "total_long": 0, "total_short": 0},
        }

        total_long = 0
        total_short = 0

        for tf in cleaned_tfs:
            try:
                long_res = strategy_candidates(
                    exchange=exchange,
                    timeframe=tf,
                    direction="long",
                    quote=quote,
                    adx_min=adx_min,
                    kc_min_width=kc_min_width,
                    atr_k=atr_k,
                    pb_long_min=pb_long_min,
                    pb_short_max=pb_short_max,
                    include_breakout=include_breakout,
                    include_pullback=include_pullback,
                    limit=limit,
                    per_strategy_limit=per_strategy_limit,
                ) or {}
            except Exception:
                long_res = {"error": "failed"}

            try:
                short_res = strategy_candidates(
                    exchange=exchange,
                    timeframe=tf,
                    direction="short",
                    quote=quote,
                    adx_min=adx_min,
                    kc_min_width=kc_min_width,
                    atr_k=atr_k,
                    pb_long_min=pb_long_min,
                    pb_short_max=pb_short_max,
                    include_breakout=include_breakout,
                    include_pullback=include_pullback,
                    limit=limit,
                    per_strategy_limit=per_strategy_limit,
                ) or {}
            except Exception:
                short_res = {"error": "failed"}

            out["results"][tf] = {
                "long": long_res,
                "short": short_res,
            }
            try:
                total_long += int((long_res.get("counts") or {}).get("merged", 0))
            except Exception:
                pass
            try:
                total_short += int((short_res.get("counts") or {}).get("merged", 0))
            except Exception:
                pass

        out["summary"]["total_long"] = total_long
        out["summary"]["total_short"] = total_short
        return out

    except Exception as e:
        logger.exception("scan_strategy_candidates encountered an unexpected error")
        return {"error": f"scan_strategy_candidates failed: {str(e)}"}


@mcp.tool()
@log_tool
def smart_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "1h",
    quote: Optional[str] = None,
    min_volume: Optional[float] = None,
    min_rsi: Optional[float] = None,
    max_rsi: Optional[float] = None,
    min_adx: Optional[float] = None,
    max_adx: Optional[float] = None,
    max_bbw: Optional[float] = None,
    direction: str = "both",
    limit: int = 30,
) -> list[dict]:
    """Adaptive TradingView screener combining volume, momentum and squeeze filters."""

    if not TRADINGVIEW_SCREENER_AVAILABLE:
        raise RuntimeError("tradingview-screener missing; run `uv sync`.")

    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "1h")
    direction = (direction or "both").lower()
    if direction not in {"long", "short", "both"}:
        direction = "both"
    limit = max(1, min(100, int(limit)))

    resolution = _tf_to_tv_resolution(timeframe) or "60"
    scan_limit = min(max(limit * 3, 60), 200)

    base_cols = [
        "open",
        "close",
        "SMA20",
        "EMA20",
        "EMA50",
        "EMA100",
        "EMA200",
        "RSI",
        "ADX",
        "ATR",
        "BB.upper",
        "BB.lower",
        "volume",
    ]
    select_cols = [Column("name")] + [Column(f"{col}|{resolution}") for col in base_cols]

    field = lambda name: f"{name}|{resolution}"

    try:
        query = Query().set_markets("crypto").select(*select_cols).where(Column("exchange") == exchange.upper())
        if min_volume is not None:
            query = query.where(Column(field("volume")) >= float(min_volume))
        if min_rsi is not None:
            query = query.where(Column(field("RSI")) >= float(min_rsi))
        if max_rsi is not None:
            query = query.where(Column(field("RSI")) <= float(max_rsi))
        if min_adx is not None:
            query = query.where(Column(field("ADX")) >= float(min_adx))
        if max_adx is not None:
            query = query.where(Column(field("ADX")) <= float(max_adx))
        query = query.limit(scan_limit)
        _total, df = query.get_scanner_data()
    except Exception as exc:
        raise RuntimeError(f"TradingView screener request failed: {exc}") from exc

    if df is None or df.empty:
        return []

    if quote:
        suffix = quote.upper()
        df = df[df["name"].str.upper().str.endswith(suffix)]
        if df.empty:
            return []

    def _num(val: Any) -> Optional[float]:
        try:
            if val is None:
                return None
            num = float(val)
            if not math.isfinite(num):
                return None
            return num
        except (TypeError, ValueError):
            return None

    results: list[dict] = []

    for _, row in df.iterrows():
        ticker = row.get("ticker")
        name = row.get("name")
        if not ticker or not name:
            continue

        indicators = {
            "open": _num(row.get(field("open"))),
            "close": _num(row.get(field("close"))),
            "SMA20": _num(row.get(field("SMA20"))),
            "BB.upper": _num(row.get(field("BB.upper"))),
            "BB.lower": _num(row.get(field("BB.lower"))),
            "EMA20": _num(row.get(field("EMA20"))),
            "EMA50": _num(row.get(field("EMA50"))),
            "EMA100": _num(row.get(field("EMA100"))),
            "EMA200": _num(row.get(field("EMA200"))),
            "RSI": _num(row.get(field("RSI"))),
            "ADX": _num(row.get(field("ADX"))),
            "ATR": _num(row.get(field("ATR"))),
            "volume": _num(row.get(field("volume"))),
        }

        metrics = compute_metrics(indicators) or {}
        bbw = metrics.get("bbw")
        if max_bbw is not None and (bbw is None or bbw > max_bbw):
            continue

        rsi = indicators.get("RSI")
        adx = indicators.get("ADX")
        volume = indicators.get("volume")
        close_price = indicators.get("close")
        ema20 = indicators.get("EMA20")
        price_val = metrics.get("price")
        if (price_val is None or price_val == 0) and close_price is not None:
            price_val = close_price

        inferred_direction = "long"
        if rsi is not None and rsi <= 45:
            inferred_direction = "short"
        elif ema20 and close_price and close_price < ema20:
            inferred_direction = "short"
        elif rsi is not None and rsi >= 55:
            inferred_direction = "long"

        if direction != "both" and inferred_direction != direction:
            continue

        score = 0.0
        if volume and volume > 0:
            score += math.log10(volume + 1)
        if adx is not None:
            score += min(5.0, max(0.0, adx / 10.0))
        if rsi is not None:
            score += max(0.0, (100 - abs(rsi - 50)) / 20.0)
        if bbw is not None:
            score += max(0.0, (0.2 - bbw) * 10.0)
        change = metrics.get("change")
        if isinstance(change, (int, float)):
            score += max(0.0, change / 5.0)

        results.append({
            "symbol": ticker,
            "name": name,
            "timeframe": timeframe,
            "direction": inferred_direction,
            "score": round(score, 3),
            "metrics": {
                "price": price_val,
                "change_percent": change,
                "bbw": bbw,
                "rsi": rsi,
                "adx": adx,
                "atr": indicators.get("ATR"),
                "volume": volume,
                "ema20": ema20,
                "ema50": indicators.get("EMA50"),
                "ema200": indicators.get("EMA200"),
            },
            "reasons": [
                reason for reason in [
                    "Volume filter" if min_volume and volume and volume >= min_volume else None,
                    "ADX filter" if min_adx and adx and adx >= min_adx else None,
                    "RSI momentum" if rsi and (rsi >= 60 or rsi <= 40) else None,
                    "BBW squeeze" if max_bbw and bbw is not None and bbw <= max_bbw else None,
                    "Trend direction long" if inferred_direction == "long" else None,
                    "Trend direction short" if inferred_direction == "short" else None,
                ] if reason
            ],
        })

    if not results:
        return []

    results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return results[:limit]


def main():
    """Main entry point for the TradingView MCP server."""
    import asyncio
    from mcp.server.fastmcp import FastMCP
    
    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()
