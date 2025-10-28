
import asyncio

import httpx
import pytest

from tradingview_mcp.pro_scanners.alerts.router import AlertRouter
from tradingview_mcp.pro_scanners.config import HTTPServiceConfig, load_infrastructure_config
from tradingview_mcp.pro_scanners.filters.derivatives import DerivativesFilter, DerivativesThresholds
from tradingview_mcp.pro_scanners.models import (
    DerivativeSnapshot,
    IndicatorSnapshot,
    RiskAssessment,
    ScannerSignal,
    SignalDirection,
)


def _make_signal() -> ScannerSignal:
    return ScannerSignal(
        symbol="BTCUSDT",
        timeframe="15m",
        direction=SignalDirection.LONG,
        entry=100.0,
        indicators=IndicatorSnapshot(ema_fast=99, ema_slow=101, adx=25, volume_z=2),
        derivatives=DerivativeSnapshot(open_interest=1_000_000, funding_rate=0.01, cvd=10),
        risk=RiskAssessment(leverage=5, stop_loss=95.0, take_profit=(110.0,), position_size=100.0),
        confidence=70,
        reasons=("test",),
    )


@pytest.mark.asyncio
async def test_alert_router_dispatch_success() -> None:
    responses: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        responses.append(request)
        return httpx.Response(200, text="ok")

    transport = httpx.MockTransport(handler)
    router = AlertRouter(HTTPServiceConfig(url="http://example.com", timeout=5), max_retries=0)
    router._client = httpx.AsyncClient(transport=transport)
    try:
        result = await router.dispatch_signal(_make_signal())
    finally:
        await router._client.aclose()
    assert result.status == "sent"
    assert responses and responses[0].url == httpx.URL("http://example.com")


def test_derivatives_filter_thresholds() -> None:
    thresholds = DerivativesThresholds(min_open_interest=500_000, max_funding_abs=0.05, min_cvd_abs=5)
    filt = DerivativesFilter(thresholds)
    assert filt.validate(DerivativeSnapshot(open_interest=600_000, funding_rate=0.01, cvd=10))
    assert not filt.validate(DerivativeSnapshot(open_interest=100_000, funding_rate=0.01, cvd=10))


def test_load_infrastructure_filters_config() -> None:
    cfg = load_infrastructure_config()
    assert cfg.filters is not None
    assert cfg.filters.context.min_confidence >= 0
    assert cfg.filters.derivatives.max_funding_abs > 0

from tradingview_mcp.pro_scanners.models import BacktestRequest


def test_backtest_request_strategy_default():
    req = BacktestRequest(symbols=["BTC"], profile="balanced")
    assert req.strategy == "momentum"
