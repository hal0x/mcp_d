"""Registry service for MCP discovery and capabilities."""

from __future__ import annotations

from datetime import datetime
from typing import AsyncContextManager, Callable, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import MCPInfo
from ..models.orm import MCPRegistryORM

SessionProvider = Callable[[], AsyncContextManager[AsyncSession]]


class RegistryService:
    """Service for managing MCP registry and capabilities."""

    def __init__(self, session_provider: SessionProvider):
        self._session_provider = session_provider

    async def register_mcp(self, mcp_info: MCPInfo) -> None:
        """Register or update MCP server information."""
        async with self._session_provider() as session:
            stmt = select(MCPRegistryORM).where(MCPRegistryORM.name == mcp_info.name)
            existing = await session.scalar(stmt)

            if existing:
                existing.version = mcp_info.version
                existing.protocol = mcp_info.protocol
                existing.endpoint = mcp_info.endpoint
                existing.capabilities = list(mcp_info.capabilities)
                existing.status = mcp_info.status
                existing.last_seen = mcp_info.last_seen or datetime.utcnow()
            else:
                session.add(
                    MCPRegistryORM(
                        name=mcp_info.name,
                        version=mcp_info.version,
                        protocol=mcp_info.protocol,
                        endpoint=mcp_info.endpoint,
                        capabilities=list(mcp_info.capabilities),
                        status=mcp_info.status,
                        last_seen=mcp_info.last_seen or datetime.utcnow(),
                    )
                )

            await session.commit()

    async def get_registry(self) -> List[MCPInfo]:
        """Get all registered MCP servers."""
        async with self._session_provider() as session:
            result = await session.execute(select(MCPRegistryORM).order_by(MCPRegistryORM.name.asc()))
            records = result.scalars().all()
        return [self._to_model(record) for record in records]

    async def get_mcp_info(self, name: str) -> Optional[MCPInfo]:
        """Get information for specific MCP server."""
        async with self._session_provider() as session:
            record = await session.scalar(select(MCPRegistryORM).where(MCPRegistryORM.name == name))
        return self._to_model(record) if record else None

    async def get_capabilities(self, name: str) -> List[str]:
        """Get capabilities for specific MCP server."""
        async with self._session_provider() as session:
            record = await session.scalar(select(MCPRegistryORM.capabilities).where(MCPRegistryORM.name == name))
        return list(record or [])

    async def update_status(self, name: str, status: str) -> None:
        """Update status for specific MCP server."""
        async with self._session_provider() as session:
            record = await session.scalar(select(MCPRegistryORM).where(MCPRegistryORM.name == name))
            if record:
                record.status = status
                record.last_seen = datetime.utcnow()
                await session.commit()

    def _to_model(self, record: MCPRegistryORM) -> MCPInfo:
        return MCPInfo(
            name=record.name,
            version=record.version,
            protocol=record.protocol,
            endpoint=record.endpoint,
            capabilities=list(record.capabilities or []),
            status=record.status,
            last_seen=record.last_seen,
        )
