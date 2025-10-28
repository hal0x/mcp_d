"""Example usage of supervisor MCP server."""

import asyncio
from datetime import datetime
from supervisor_mcp.server import create_server
from supervisor_mcp.models import Metric, Fact, MCPInfo


async def main():
    """Example usage of supervisor MCP server."""
    server = create_server()
    
    # Example: Register an MCP server
    mcp_info = MCPInfo(
        name="binance-mcp",
        version="1.0.0",
        protocol="http",
        endpoint="http://localhost:8000",
        capabilities=["create_order", "get_balance", "get_ticker"],
        status="up",
        last_seen=datetime.now()
    )
    
    # Example: Ingest metrics
    metric = Metric(
        name="rpc_latency_p95",
        value=150.5,
        tags={"svc": "binance-mcp", "op": "create_order"},
        ts=datetime.now()
    )
    
    # Example: Ingest facts
    fact = Fact(
        kind="Fact:Trade",
        actor="halv1",
        correlation_id="tsk_abc123",
        payload={
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.001,
            "price": 45000.0
        },
        ts=datetime.now()
    )
    
    print("Supervisor MCP server created successfully!")
    print(f"Registered MCP: {mcp_info.name}")
    print(f"Metric: {metric.name} = {metric.value}")
    print(f"Fact: {fact.kind} from {fact.actor}")


if __name__ == "__main__":
    asyncio.run(main())
