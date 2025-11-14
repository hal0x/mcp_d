"""Fixtures for testing the new MCP server."""

from pathlib import Path
from typing import Generator

import pytest

from memory_mcp.mcp.adapters import MemoryServiceAdapter
from memory_mcp.mcp.server import server


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path for testing."""
    return tmp_path / "test_memory.db"


@pytest.fixture
def mcp_server_adapter(temp_db_path: Path) -> Generator[MemoryServiceAdapter, None, None]:
    """Fixture for MemoryServiceAdapter with temporary database."""
    adapter = MemoryServiceAdapter(db_path=str(temp_db_path))
    try:
        yield adapter
    finally:
        adapter.close()


@pytest.fixture
def mcp_server():
    """Fixture for the MCP server instance."""
    return server

