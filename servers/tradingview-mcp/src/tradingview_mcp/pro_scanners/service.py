"""Service layer orchestrating professional scanners."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Iterable, Sequence

from .adapters.binance_mcp import BinanceMCPAdapter
from .alerts.router import AlertRouter
from .alerts.callbacks import log_dispatch_result
from .config import InfrastructureConfig, load_infrastructure_config
from .filters.context import ContextFilter, ContextRules
from .filters.derivatives import DerivativesFilter, DerivativesThresholds
from .models import AlertRouteResult, BacktestRequest, BacktestResult, ScannerSignal
from .risk.calculator import RiskCalculator
from .risk.manager import RiskManager
from .scanners.mean_revert import MeanRevertScanner
from .scanners.momentum import MomentumScanner
from .scanners.breakout import BreakoutScanner
from .scanners.volume_profile import VolumeProfileScanner
from .storage.postgres_store import PostgresStore
from .storage.memory_bridge import TradingMemoryRecorder
from .storage.redis_cache import RedisCache
from .utils import make_cache_key


logger = logging.getLogger(__name__)


class ScannerService:
    """High level orchestrator that wires adapters, scanners, storage, and alerts."""

    def __init__(self, config: InfrastructureConfig | None = None):
        self._config = config or load_infrastructure_config()
        self.redis = RedisCache(self._config.redis)
        self.postgres = PostgresStore(self._config.postgres)
        self.adapter = BinanceMCPAdapter(
            http_config=self._config.binance_mcp,
            cache=self.redis,
        )
        filters_cfg = self._config.filters
        if filters_cfg:
            derivatives_thresholds = DerivativesThresholds(**filters_cfg.derivatives.model_dump())
            context_rules = ContextRules(**filters_cfg.context.model_dump())
        else:
            derivatives_thresholds = DerivativesThresholds()
            context_rules = ContextRules()
        self.derivatives_filter = DerivativesFilter(derivatives_thresholds)
        self.context_filter = ContextFilter(context_rules)
        self.risk_calculator = RiskCalculator()
        self.risk_manager = RiskManager()
        self.alert_router = AlertRouter(self._config.halv1_bot)
        self.alert_router.register_callback(log_dispatch_result)
        self.memory_recorder = TradingMemoryRecorder(self._config.memory_mcp)
        self._memory_lock = asyncio.Lock()
        self._memory_pending: dict[str, tuple[ScannerSignal, str | None, str | None]] = {}
        if self.memory_recorder.enabled:
            self.alert_router.register_callback(self._record_memory_callback)
        self.momentum_scanner = MomentumScanner(
            adapter=self.adapter,
            derivatives_filter=self.derivatives_filter,
            context_filter=self.context_filter,
            risk_calculator=self.risk_calculator,
            risk_manager=self.risk_manager,
            alert_router=self.alert_router,
            cache=self.redis,
        )
        self.mean_revert_scanner = MeanRevertScanner(
            adapter=self.adapter,
            derivatives_filter=self.derivatives_filter,
            context_filter=self.context_filter,
            risk_calculator=self.risk_calculator,
            risk_manager=self.risk_manager,
            alert_router=self.alert_router,
            cache=self.redis,
        )
        self.breakout_scanner = BreakoutScanner(
            adapter=self.adapter,
            derivatives_filter=self.derivatives_filter,
            context_filter=self.context_filter,
            risk_calculator=self.risk_calculator,
            risk_manager=self.risk_manager,
            alert_router=self.alert_router,
            cache=self.redis,
        )
        self.volume_profile_scanner = VolumeProfileScanner(
            adapter=self.adapter,
            derivatives_filter=self.derivatives_filter,
            context_filter=self.context_filter,
            risk_calculator=self.risk_calculator,
            risk_manager=self.risk_manager,
            alert_router=self.alert_router,
            cache=self.redis,
        )
        self._redis_ready = False
        self._postgres_ready = False
        self._adapter_ready = False
        self._alerts_ready = False

    async def startup(self) -> None:
        self._redis_ready = await self._safe_call("connect redis", self.redis.connect)
        self._postgres_ready = await self._safe_call("connect postgres", self.postgres.connect)
        self._adapter_ready = await self._safe_call("start binance adapter", self.adapter.startup)
        self._alerts_ready = await self._safe_call("start alert router", self.alert_router.startup)
        
        # Логируем статус каждой зависимости
        logger.info("Service startup status: Redis=%s, Postgres=%s, Binance=%s, Alerts=%s",
                    self._redis_ready, self._postgres_ready, self._adapter_ready, self._alerts_ready)
        
        if not self._adapter_ready:
            logger.error(
                "⚠️ Binance adapter unavailable at %s. Professional scanners will return empty results. "
                "To enable pro scanners, ensure binance-mcp service is running and accessible.",
                self._config.binance_mcp.url
            )
            # НЕ выбрасываем исключение - позволяем сервису работать с ограниченной функциональностью

    async def shutdown(self) -> None:
        if self._adapter_ready:
            await self._safe_call("shutdown binance adapter", self.adapter.shutdown)
            self._adapter_ready = False
        if self._alerts_ready:
            await self._safe_call("shutdown alert router", self.alert_router.shutdown)
            self._alerts_ready = False
        if self._postgres_ready:
            await self._safe_call("disconnect postgres", self.postgres.disconnect)
            self._postgres_ready = False
        if self._redis_ready:
            await self._safe_call("disconnect redis", self.redis.disconnect)
            self._redis_ready = False

    async def scan_momentum(self, symbols: Sequence[str], profile: str) -> list[ScannerSignal]:
        if not self._adapter_ready:
            logger.error(
                "❌ Cannot run momentum scanner: binance-mcp service not available at %s. "
                "Start binance-mcp to enable professional scanners.",
                self._config.binance_mcp.url
            )
            return []
            
        cache_key = make_cache_key("momentum", [profile, *symbols])
        cached = None
        if self._redis_ready:
            cached = await self.redis.get_cached_signals(cache_key)
        if cached:
            logger.info("Returning %d cached momentum signals for %s", len(cached), symbols)
            return cached
        
        try:
            logger.info("Running momentum scanner for %d symbols with profile %s", len(symbols), profile)
            # Добавляем общий таймаут для всего сканирования
            signals = await asyncio.wait_for(
                self.momentum_scanner.run(symbols, profile),
                timeout=30.0  # Увеличен до 30 секунд
            )
            logger.info("Momentum scanner completed: found %d signals", len(signals))
        except asyncio.TimeoutError:
            logger.error("Momentum scanner timed out after 30 seconds for symbols %s", symbols)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.exception("Momentum scanner failed for %s: %s", symbols, exc)
            return []
        
        await self._persist_and_route(
            signals,
            profile=profile,
            strategy=getattr(self.momentum_scanner, "strategy_name", "momentum"),
        )
        if self._redis_ready:
            await self.redis.set_cached_signals(cache_key, signals, ttl=self.momentum_scanner.get_cache_ttl(profile))
        return signals

    async def scan_mean_revert(self, symbols: Sequence[str], profile: str) -> list[ScannerSignal]:
        if not self._adapter_ready:
            logger.error(
                "❌ Cannot run mean revert scanner: binance-mcp service not available at %s. "
                "Start binance-mcp to enable professional scanners.",
                self._config.binance_mcp.url
            )
            return []
            
        cache_key = make_cache_key("mean_revert", [profile, *symbols])
        cached = None
        if self._redis_ready:
            cached = await self.redis.get_cached_signals(cache_key)
        if cached:
            return cached
        
        try:
            # Добавляем общий таймаут для всего сканирования
            signals = await asyncio.wait_for(
                self.mean_revert_scanner.run(symbols, profile),
                timeout=20.0  # 20 секунд максимум
            )
        except asyncio.TimeoutError:
            logger.warning("Mean revert scanner timed out after 20 seconds")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.exception("Mean revert scanner failed: %s", exc)
            return []
        
        await self._persist_and_route(
            signals,
            profile=profile,
            strategy=getattr(self.mean_revert_scanner, "strategy_name", "mean_revert"),
        )
        if self._redis_ready:
            await self.redis.set_cached_signals(cache_key, signals, ttl=self.mean_revert_scanner.get_cache_ttl(profile))
        return signals

    async def scan_breakout(self, symbols: Sequence[str], profile: str) -> list[ScannerSignal]:
        if not self._adapter_ready:
            logger.error(
                "❌ Cannot run breakout scanner: binance-mcp service not available at %s. "
                "Start binance-mcp to enable professional scanners.",
                self._config.binance_mcp.url
            )
            return []
            
        cache_key = make_cache_key("breakout", [profile, *symbols])
        cached = None
        if self._redis_ready:
            cached = await self.redis.get_cached_signals(cache_key)
        if cached:
            return cached
        
        try:
            # Добавляем общий таймаут для всего сканирования
            signals = await asyncio.wait_for(
                self.breakout_scanner.run(symbols, profile),
                timeout=20.0  # 20 секунд максимум
            )
        except asyncio.TimeoutError:
            logger.warning("Breakout scanner timed out after 20 seconds")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.exception("Breakout scanner failed: %s", exc)
            return []
        
        await self._persist_and_route(
            signals,
            profile=profile,
            strategy=getattr(self.breakout_scanner, "strategy_name", "breakout"),
        )
        if self._redis_ready:
            await self.redis.set_cached_signals(cache_key, signals, ttl=self.breakout_scanner.get_cache_ttl(profile))
        return signals

    async def scan_volume_profile(self, symbols: Sequence[str], profile: str) -> list[ScannerSignal]:
        if not self._adapter_ready:
            logger.error(
                "❌ Cannot run volume profile scanner: binance-mcp service not available at %s. "
                "Start binance-mcp to enable professional scanners.",
                self._config.binance_mcp.url
            )
            return []
            
        cache_key = make_cache_key("volume_profile", [profile, *symbols])
        cached = None
        if self._redis_ready:
            cached = await self.redis.get_cached_signals(cache_key)
        if cached:
            return cached
        
        try:
            # Добавляем общий таймаут для всего сканирования
            signals = await asyncio.wait_for(
                self.volume_profile_scanner.run(symbols, profile),
                timeout=20.0  # 20 секунд максимум
            )
        except asyncio.TimeoutError:
            logger.warning("Volume profile scanner timed out after 20 seconds")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.exception("Volume profile scanner failed: %s", exc)
            return []
        
        await self._persist_and_route(
            signals,
            profile=profile,
            strategy=getattr(self.volume_profile_scanner, "strategy_name", "volume_profile"),
        )
        if self._redis_ready:
            await self.redis.set_cached_signals(cache_key, signals, ttl=self.volume_profile_scanner.get_cache_ttl(profile))
        return signals

    async def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        if not self._adapter_ready:
            error_msg = (
                f"❌ Cannot run backtest: binance-mcp service not available at {self._config.binance_mcp.url}. "
                "Start binance-mcp to enable backtesting."
            )
            logger.error(error_msg)
            # Возвращаем пустой результат с информацией об ошибке
            return BacktestResult(
                signals=[],
                metrics={"error": 0.0},
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                profile=request.profile,
                timeframe=request.timeframe,
                symbol_universe=request.symbols,
                strategy=request.strategy or "momentum",
            )
            
        strategy = (request.strategy or "momentum").lower()
        if strategy == "mean_revert":
            engine = self.mean_revert_scanner.get_backtest_engine()
        elif strategy == "breakout":
            engine = self.breakout_scanner.get_backtest_engine()
        elif strategy == "volume_profile":
            engine = self.volume_profile_scanner.get_backtest_engine()
        else:
            engine = self.momentum_scanner.get_backtest_engine()
        result = await engine.execute(request)
        if self._postgres_ready:
            await self.postgres.record_backtest(result)
            await self.postgres.record_metrics_snapshot(result)
        return result

    async def get_recent_signals(self, limit: int = 20) -> list[dict[str, object]]:
        if not self._postgres_ready:
            logger.error("Cannot fetch recent signals: Postgres not available. Check database connection at %s:%s/%s",
                        self._config.postgres.host, self._config.postgres.port, self._config.postgres.database)
            return []
        try:
            logger.info("Fetching %d most recent signals from Postgres", limit)
            signals = await asyncio.wait_for(
                self.postgres.fetch_recent_signals(limit),
                timeout=10.0  # 10 секунд максимум для БД запроса
            )
            logger.info("Retrieved %d recent signals from database", len(signals))
            return signals
        except asyncio.TimeoutError:
            logger.error("Fetch recent signals timed out after 10 seconds. Database may be slow or unresponsive.")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch recent signals from Postgres: %s", exc)
            return []

    async def get_metrics(self, period_days: int = 14, strategy: str | None = None) -> list[dict[str, object]]:
        if not self._postgres_ready:
            logger.warning("Postgres not ready; returning empty metrics list")
            return []
        try:
            return await asyncio.wait_for(
                self.postgres.fetch_metrics(period_days, strategy),
                timeout=10.0  # 10 секунд максимум для БД запроса
            )
        except asyncio.TimeoutError:
            logger.warning("Fetch metrics timed out after 10 seconds")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch metrics: %s", exc)
            return []

    async def submit_feedback(self, signal_id: int, action: str) -> bool:
        if not self._postgres_ready:
            logger.warning("Postgres not ready; cannot record feedback")
            return False
        action_normalized = action.lower()
        if action_normalized not in {"take", "skip"}:
            raise ValueError("Action must be 'take' or 'skip'")
        try:
            await self.postgres.record_feedback(signal_id, action_normalized)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to record feedback: %s", exc)
            return False

    async def clear_cache(self, namespace: str | None = None) -> int:
        if not self._redis_ready:
            logger.warning("Redis not ready; cannot clear cache")
            return 0
        try:
            return await self.redis.clear_namespace(namespace)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to clear cache: %s", exc)
            return 0

    async def _persist_and_route(
        self,
        signals: Iterable[ScannerSignal],
        *,
        profile: str | None = None,
        strategy: str | None = None,
    ) -> None:
        for signal in signals:
            signal_key = self._signal_key(signal.symbol, signal.timeframe, signal.generated_at)
            if self.memory_recorder.enabled and self._alerts_ready:
                async with self._memory_lock:
                    self._memory_pending[signal_key] = (signal, profile, strategy)
            if self._postgres_ready:
                await self._safe_call("record signal", lambda s=signal: self.postgres.record_signal(s))
            if self._alerts_ready:
                await self._safe_call("route alert", lambda s=signal: self.alert_router.dispatch_signal(s))
            if self.memory_recorder.enabled and not self._alerts_ready:
                await self._safe_call(
                    "store trading memory",
                    lambda s=signal, p=profile, st=strategy: self.memory_recorder.store_signal(
                        s,
                        profile=p,
                        strategy=st,
                        alert_result=None,
                        signal_key=signal_key,
                    ),
                )

    async def _safe_call(self, name: str, func) -> bool:
        try:
            result = func()
            if hasattr(result, "__await__"):
                await result
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Failed to %s", name)
            return False

    async def _record_memory_callback(self, result: AlertRouteResult) -> None:
        if not self.memory_recorder.enabled:
            return
        metadata = result.payload.metadata or {}
        key = metadata.get("signal_key")
        if not key:
            logger.debug("Memory recorder: signal_key missing in alert metadata")
            return
        async with self._memory_lock:
            pending = self._memory_pending.pop(key, None)
        if not pending:
            logger.debug("Memory recorder: no pending signal for key %s", key)
            return
        signal, profile, strategy = pending
        await self._safe_call(
            "store trading memory",
            lambda s=signal, p=profile, st=strategy: self.memory_recorder.store_signal(
                s,
                profile=p,
                strategy=st,
                alert_result=result,
                signal_key=key,
            ),
        )

    @staticmethod
    def _signal_key(symbol: str, timeframe: str, generated_at: datetime) -> str:
        return f"{symbol.upper()}|{timeframe}|{generated_at.isoformat()}"


__all__ = ["ScannerService"]
