"""Health check service for backtesting MCP."""

from __future__ import annotations

import time
from typing import Dict, Any

import httpx

from ..config import get_settings


class HealthService:
    """Service for health checks and system status."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def check_health(self, check_data_sources: bool = True) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        start_time = time.time()
        
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": 0,  # Will be set by server
            "version": "0.2.0",
            "data_sources": {},
            "checks": {
                "config_loaded": True,
                "data_sources_available": False,
            }
        }
        
        # Check data sources if requested
        if check_data_sources:
            data_sources_status = self._check_data_sources()
            health_status["data_sources"] = data_sources_status
            health_status["checks"]["data_sources_available"] = any(
                status["available"] for status in data_sources_status.values()
            )
        
        # Overall status
        if not health_status["checks"]["data_sources_available"] and check_data_sources:
            health_status["status"] = "degraded"  # Still functional with synthetic data
        
        health_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return health_status
    
    def _check_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Check availability of external data sources."""
        sources = {}
        
        # Check Binance MCP
        if self.settings.preferred_source.lower() in {"binance", "both"}:
            sources["binance"] = self._check_binance_source()
        
        # Check TradingView MCP
        if self.settings.preferred_source.lower() in {"tradingview", "both"}:
            sources["tradingview"] = self._check_tradingview_source()
        
        return sources
    
    def _check_binance_source(self) -> Dict[str, Any]:
        """Check Binance MCP server availability."""
        url = f"{self.settings.binance_url.rstrip('/')}/meta/health"
        
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                return {
                    "available": True,
                    "url": self.settings.binance_url,
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                }
        except Exception as e:
            return {
                "available": False,
                "url": self.settings.binance_url,
                "error": str(e),
            }
    
    def _check_tradingview_source(self) -> Dict[str, Any]:
        """Check TradingView MCP server availability."""
        url = f"{self.settings.tradingview_url.rstrip('/')}/health"
        
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                return {
                    "available": True,
                    "url": self.settings.tradingview_url,
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                }
        except Exception as e:
            return {
                "available": False,
                "url": self.settings.tradingview_url,
                "error": str(e),
            }
