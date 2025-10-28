"""End-to-end tests for Learning ↔ Policy MCP integration."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Tuple

import httpx
import pytest

pytest.importorskip("fastmcp")
from fastmcp import FastMCP

# Ensure project modules are importable when tests run from repository root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "servers" / "learning-mcp" / "src"))
sys.path.append(str(ROOT / "servers" / "policy-mcp" / "src"))

from learning_mcp.services.pattern_analyzer import PatternAnalyzer
from learning_mcp.services.policy_client import PolicyClient
from learning_mcp.services.trainer import TrainerService
from learning_mcp.tools import register_tools
from policy_mcp.config import Settings as PolicySettings
from policy_mcp.db import dispose_engine, init_db
from policy_mcp.server import create_server as create_policy_server


@pytest.fixture
async def policy_http_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """Provide an httpx client backed by the Policy MCP FastAPI app."""
    db_path = tmp_path / "policy.db"
    settings = PolicySettings(DB_URL=f"sqlite+aiosqlite:///{db_path}", DB_ECHO=False)
    await init_db(settings=settings)
    policy_server = create_policy_server(settings=settings)

    async with httpx.AsyncClient(app=policy_server.http_app, base_url="http://policy") as client:
        yield client

    await dispose_engine()


@pytest.fixture
async def learning_mcp_app(
    policy_http_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Tuple[FastMCP, PolicyClient]]:
    """Configure Learning MCP with a PolicyClient pointing at the policy test app."""
    monkeypatch.setenv("LEARNING_POLICY_URL", "http://policy")
    monkeypatch.setenv("LEARNING_POLICY_TIMEOUT", "5")

    from learning_mcp.config import get_settings

    get_settings.cache_clear()

    mcp = FastMCP("learning-mcp-test")
    trainer_service = TrainerService(supervisor_url="http://localhost")
    pattern_analyzer = PatternAnalyzer()
    policy_client = PolicyClient(base_url="http://policy", timeout=5.0)
    policy_client._client = policy_http_client

    register_tools(
        mcp,
        trainer_service=trainer_service,
        pattern_analyzer=pattern_analyzer,
        policy_client=policy_client,
    )

    yield mcp, policy_client


@pytest.mark.asyncio
async def test_policy_health_and_profile_flow(
    learning_mcp_app: Tuple[FastMCP, PolicyClient],
) -> None:
    """Learning MCP tools should interact with Policy MCP REST API successfully."""
    mcp, _policy_client = learning_mcp_app

    health_tool = await mcp.get_tool("policy_health")
    health_result = await health_tool.call({})
    assert health_result["success"]
    assert health_result["health"]["status"] == "ok"
    assert health_result["health"]["active_profiles"] == 0

    profile_payload = {
        "profile_id": "integration-profile",
        "version": "1.0.0",
        "name": "Integration Profile",
        "description": "Generated during integration test",
        "weights": {"signal_a": 0.5, "signal_b": 0.5},
        "thresholds": {"min_confidence": 0.6},
        "risk_limits": {"max_drawdown": 0.1},
        "created_at": datetime.now(tz=timezone.utc),
        "trained_on_samples": 150,
        "confidence_score": 0.78,
        "performance_metrics": {"success_rate": 0.82},
    }

    propose_tool = await mcp.get_tool("propose_profile")
    proposal = await propose_tool.call({"profile_data": profile_payload, "activate": True})
    assert proposal["success"]
    assert proposal["status"] == "activated"
    assert proposal["profile_id"] == "integration-profile"

    active_tool = await mcp.get_tool("get_active_policy_profile")
    active = await active_tool.call({})
    assert active["success"]
    assert active["profile"]["profile_id"] == "integration-profile"
    assert active["profile"]["active"] is True

    list_tool = await mcp.get_tool("list_policy_profiles")
    listing = await list_tool.call({})
    assert listing["success"]
    assert listing["count"] == 1
    assert listing["profiles"][0]["profile_id"] == "integration-profile"
