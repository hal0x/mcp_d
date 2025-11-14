"""Pytest configuration and fixtures."""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

# Импортируем фикстуры для нового MCP сервера
from tests.fixtures.mcp_server import (  # noqa: F401
    mcp_server,
    mcp_server_adapter,
    temp_db_path,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_chat_data():
    """Sample chat data for testing."""
    return [
        {
            "id": 1,
            "text": "Привет! Как дела?",
            "date": "2025-01-27T10:00:00+00:00",
            "chat": "Test Chat",
        },
        {
            "id": 2,
            "text": "Все хорошо, спасибо! А у тебя?",
            "date": "2025-01-27T10:01:00+00:00",
            "chat": "Test Chat",
        },
        {
            "id": 3,
            "text": "Отлично! Обсуждаем криптовалюты",
            "date": "2025-01-27T10:02:00+00:00",
            "chat": "Test Chat",
        },
    ]


@pytest.fixture
def sample_json_file(temp_dir, sample_chat_data):
    """Create a sample JSON file for testing."""
    json_file = temp_dir / "test_chat.json"
    with open(json_file, "w", encoding="utf-8") as f:
        for message in sample_chat_data:
            f.write(f"{message}\n")
    return json_file
