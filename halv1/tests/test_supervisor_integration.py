"""Test supervisor MCP integration with HALv1."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from services.supervisor_client import SupervisorClient
from agent.core import AgentCore
from services.event_bus import AsyncEventBus
from events.models import MessageReceived


@pytest.fixture
def mock_supervisor_client():
    """Mock supervisor client."""
    client = AsyncMock(spec=SupervisorClient)
    client._session_id = "test_session"
    return client


@pytest.fixture
def mock_components():
    """Mock components for AgentCore."""
    bus = AsyncEventBus()
    planner = MagicMock()
    executor = MagicMock()
    search = MagicMock()
    memory = MagicMock()
    code_generator = MagicMock()
    
    return {
        "bus": bus,
        "planner": planner,
        "executor": executor,
        "search": search,
        "memory": memory,
        "code_generator": code_generator,
    }


@pytest.mark.asyncio
async def test_supervisor_integration(mock_supervisor_client, mock_components):
    """Test that HALv1 sends metrics and facts to supervisor."""
    
    # Создаем AgentCore с supervisor client
    core = AgentCore(
        bus=mock_components["bus"],
        planner=mock_components["planner"],
        executor=mock_components["executor"],
        search=mock_components["search"],
        memory=mock_components["memory"],
        code_generator=mock_components["code_generator"],
        supervisor_client=mock_supervisor_client,
    )
    
    # Настраиваем мок планировщика
    from planner import Plan, PlanStep, Tool
    mock_plan = Plan(steps=[
        PlanStep(
            id="step1",
            tool=Tool.CODE,
            content="print('hello')",
            expected_output="hello",
            is_final=True
        )
    ])
    mock_components["planner"].plan.return_value = mock_plan
    
    # Отправляем сообщение
    message = MessageReceived(chat_id=1, message_id=1, text="test task")
    await core.bus.publish("incoming", message)
    
    # Проверяем, что supervisor client был вызван
    assert mock_supervisor_client.send_metric.called
    assert mock_supervisor_client.send_planning_fact.called
    
    # Проверяем аргументы вызовов
    metric_calls = mock_supervisor_client.send_metric.call_args_list
    assert len(metric_calls) > 0
    assert metric_calls[0][0][0] == "planning_started"  # name
    assert metric_calls[0][0][1] == 1.0  # value
    
    fact_calls = mock_supervisor_client.send_planning_fact.call_args_list
    assert len(fact_calls) > 0
    assert fact_calls[0][0][0] == "test task"  # task
    assert fact_calls[0][0][1] == 1  # plan_steps


@pytest.mark.asyncio
async def test_supervisor_client_send_metric():
    """Test supervisor client metric sending."""
    client = SupervisorClient("http://localhost:8001")
    
    # Мокаем HTTP клиент
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    client._http_client = AsyncMock()
    client._http_client.post.return_value = mock_response
    
    # Отправляем метрику
    success = await client.send_metric("test_metric", 42.0, {"env": "test"})
    
    assert success is True
    assert client._http_client.post.called
    
    # Проверяем URL и данные
    call_args = client._http_client.post.call_args
    assert call_args[0][0] == "http://localhost:8001/ingest/metric"
    
    json_data = call_args[1]["json"]
    assert json_data["name"] == "test_metric"
    assert json_data["value"] == 42.0
    assert json_data["tags"] == {"env": "test"}


@pytest.mark.asyncio
async def test_supervisor_client_send_fact():
    """Test supervisor client fact sending."""
    client = SupervisorClient("http://localhost:8001")
    
    # Мокаем HTTP клиент
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    client._http_client = AsyncMock()
    client._http_client.post.return_value = mock_response
    
    # Отправляем факт
    success = await client.send_fact(
        kind="Fact:Test",
        actor="test_actor",
        correlation_id="test_123",
        payload={"test": "data"}
    )
    
    assert success is True
    assert client._http_client.post.called
    
    # Проверяем URL и данные
    call_args = client._http_client.post.call_args
    assert call_args[0][0] == "http://localhost:8001/ingest/fact"
    
    json_data = call_args[1]["json"]
    assert json_data["kind"] == "Fact:Test"
    assert json_data["actor"] == "test_actor"
    assert json_data["correlation_id"] == "test_123"
    assert json_data["payload"] == {"test": "data"}
