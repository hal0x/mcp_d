"""MCP tools for Orchestrator MCP."""

from __future__ import annotations

from typing import Any, Dict, List

from fastmcp import FastMCP

from ..services import OrchestratorService


def register_tools(mcp: FastMCP, orchestrator_service: OrchestratorService) -> None:
    """Register orchestrator tools."""

    @mcp.tool()
    async def execute_plan(plan: List[Dict[str, Any]], dry_run: bool = False) -> Dict[str, Any]:
        """Execute a plan comprised of MCP steps."""
        return await orchestrator_service.execute_plan(plan, dry_run=dry_run)

    @mcp.tool()
    async def dry_run_plan(plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate plan execution without performing actions."""
        return await orchestrator_service.dry_run_plan(plan)
