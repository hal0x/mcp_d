"""Supervisor MCP integration for TradingView server telemetry."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from tradingview_mcp.config import Settings, get_settings

_HTTP_TIMEOUT = 3.0


class _HttpPoster:
    """Lightweight, thread-safe HTTP poster reused across threads."""

    def __init__(self, base_url: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._lock = threading.Lock()

    def post(self, path: str, payload: Dict[str, Any]) -> None:
        url = f"{self._base_url}{path}"
        try:
            with self._lock:
                httpx.post(url, json=payload, timeout=self._timeout)
        except Exception:
            # Silent failure — telemetry is best-effort
            pass


class SupervisorReporter:
    """Fire-and-forget interface for Supervisor MCP telemetry."""

    _poster: Optional[_HttpPoster] = None
    _executor: Optional[threading.Thread] = None
    _enabled = False
    _actor = "tradingview-mcp"
    _lock = threading.Lock()

    @classmethod
    def configure(cls, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        if not cfg.supervisor_metrics_enabled or not cfg.supervisor_url:
            cls._enabled = False
            cls._poster = None
            return
        cls._enabled = True
        cls._actor = cfg.supervisor_actor or "tradingview-mcp"
        cls._poster = _HttpPoster(cfg.supervisor_url, cfg.supervisor_timeout or _HTTP_TIMEOUT)

    @classmethod
    def _dispatch(cls, path: str, payload: Dict[str, Any]) -> None:
        if not cls._enabled or cls._poster is None:
            return

        def _task() -> None:
            cls._poster.post(path, payload)

        thread = threading.Thread(target=_task, daemon=True)
        thread.start()

    @classmethod
    def emit_metric(cls, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        body = {
            "name": name,
            "value": value,
            "tags": tags or {},
            "ts": datetime.utcnow().isoformat(),
        }
        cls._dispatch("/ingest/metric", body)

    @classmethod
    def emit_fact(
        cls,
        kind: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> None:
        body = {
            "kind": kind,
            "payload": payload,
            "actor": cls._actor,
            "correlation_id": correlation_id or "",
            "ts": datetime.utcnow().isoformat(),
        }
        cls._dispatch("/ingest/fact", body)

    @classmethod
    def record_startup(cls, config_snapshot: Dict[str, Any]) -> None:
        cls.emit_fact("Fact:TradingViewServerStarted", config_snapshot)

    @classmethod
    def record_shutdown(cls, uptime_seconds: float) -> None:
        cls.emit_fact("Fact:TradingViewServerStopped", {"uptime_seconds": round(uptime_seconds, 2)})

    @classmethod
    def record_screener_latency(cls, latency_seconds: float, tool: str, exchange: str) -> None:
        cls.emit_metric(
            "tradingview.screener_latency",
            round(latency_seconds, 4),
            tags={"tool": tool, "exchange": exchange},
        )

    @classmethod
    def record_rate_limit(cls, tool: str, waited_seconds: float) -> None:
        cls.emit_metric(
            "tradingview.rate_limit_wait",
            round(waited_seconds, 4),
            tags={"tool": tool},
        )

    @classmethod
    def record_error(cls, tool: str, message: str) -> None:
        cls.emit_metric(
            "tradingview.error_rate",
            1.0,
            tags={"tool": tool, "message": message[:120]},
        )

    @classmethod
    def record_signals(cls, strategy: str, count: int) -> None:
        cls.emit_metric(
            "tradingview.signals_per_minute",
            float(count),
            tags={"strategy": strategy},
        )

    @classmethod
    def record_cache_stats(cls, hit_ratio: float) -> None:
        cls.emit_metric("tradingview.cache_hit_ratio", hit_ratio)

    @classmethod
    def record_dependency_state(cls, service: str, status: str, detail: Optional[str] = None) -> None:
        payload = {"service": service, "status": status}
        if detail:
            payload["detail"] = detail
        cls.emit_fact("Fact:TradingViewDependencyState", payload)


__all__ = ["SupervisorReporter"]
