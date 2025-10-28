"""Tests for supervisor MCP server."""

from datetime import datetime
from functools import partial
from pathlib import Path
import sys

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "servers" / "supervisor-mcp" / "src"))

from supervisor_mcp.config import get_settings
from supervisor_mcp.db import dispose_engine, get_session, init_db
from supervisor_mcp.models import Metric, Fact, MCPInfo, AlertRule
from supervisor_mcp.services.registry import RegistryService
from supervisor_mcp.services.health import HealthService
from supervisor_mcp.services.metrics import MetricsService
from supervisor_mcp.services.alerts import AlertsService


async def _setup_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str) -> callable:
    db_path = tmp_path / filename
    monkeypatch.setenv("SUPERVISOR_DB_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("SUPERVISOR_DB_ECHO", "false")
    get_settings.cache_clear()
    settings = get_settings()
    await init_db(settings=settings)
    return partial(get_session, settings=settings)


@pytest.fixture
async def registry_service(tmp_path, monkeypatch):
    """Registry service fixture."""
    session_provider = await _setup_sqlite(tmp_path, monkeypatch, "registry.db")
    service = RegistryService(session_provider=session_provider)
    yield service
    await dispose_engine()
    get_settings.cache_clear()


@pytest.fixture
async def health_service(tmp_path, monkeypatch):  # pragma: no cover - placeholder for future tests
    """Health service fixture."""
    session_provider = await _setup_sqlite(tmp_path, monkeypatch, "health.db")
    service = HealthService(session_provider=session_provider)
    yield service
    await dispose_engine()
    get_settings.cache_clear()


@pytest.fixture
async def metrics_service(tmp_path, monkeypatch):
    """Metrics service fixture backed by SQLite."""
    session_provider = await _setup_sqlite(tmp_path, monkeypatch, "metrics.db")
    service = MetricsService(session_provider=session_provider, redis_client=None, cache_ttl_seconds=1)

    yield service

    await dispose_engine()
    get_settings.cache_clear()


@pytest.fixture
async def alerts_service(tmp_path, monkeypatch):
    """Alerts service fixture."""
    session_provider = await _setup_sqlite(tmp_path, monkeypatch, "alerts.db")
    service = AlertsService(session_provider=session_provider)
    yield service
    await dispose_engine()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_registry_service(registry_service):
    """Test registry service functionality."""
    mcp_info = MCPInfo(
        name="test-mcp",
        version="1.0.0",
        protocol="http",
        endpoint="http://localhost:8000",
        capabilities=["test_tool"],
        status="up"
    )
    
    await registry_service.register_mcp(mcp_info)
    
    registry = await registry_service.get_registry()
    assert len(registry) == 1
    assert registry[0].name == "test-mcp"
    
    capabilities = await registry_service.get_capabilities("test-mcp")
    assert "test_tool" in capabilities


@pytest.mark.asyncio
async def test_metrics_service(metrics_service):
    """Test metrics service functionality."""
    metric = Metric(
        name="test_metric",
        value=42.0,
        tags={"env": "test"},
        ts=datetime.now()
    )
    
    await metrics_service.ingest_metric(metric)
    
    metrics = await metrics_service.query_metrics(name="test_metric")
    assert len(metrics) == 1
    assert metrics[0].value == 42.0
    
    aggregation = await metrics_service.get_aggregation(kind="technical", window="7d")
    assert aggregation.window == "7d"
    assert aggregation.kind == "technical"


@pytest.mark.asyncio
async def test_facts_service(metrics_service):
    """Test facts service functionality."""
    fact = Fact(
        kind="Fact:Test",
        actor="test_actor",
        correlation_id="test_123",
        payload={"test": "data"},
        ts=datetime.now()
    )
    
    await metrics_service.ingest_fact(fact)
    
    facts = await metrics_service.query_facts(kind="Fact:Test")
    assert len(facts) == 1
    assert facts[0].actor == "test_actor"


@pytest.mark.asyncio
async def test_alerts_service(alerts_service):
    """Test alerts service functionality."""
    rule = AlertRule(
        id="test_rule",
        name="Test Alert",
        condition="error_rate > 0.1",
        severity="warning",
        enabled=True
    )
    
    await alerts_service.create_alert_rule(rule)
    
    rules = await alerts_service.get_alert_rules()
    assert len(rules) == 1
    assert rules[0].name == "Test Alert"
    
    # Test alert evaluation
    metrics_data = {"error_rate": 0.2}
    new_alerts = await alerts_service.evaluate_alerts(metrics_data)
    assert len(new_alerts) == 1
    assert new_alerts[0].rule_id == "test_rule"
