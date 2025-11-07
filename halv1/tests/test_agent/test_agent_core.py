from typing import Iterator
from unittest.mock import MagicMock

import pytest

from agent.core import AgentCore
from executor import SimpleCodeExecutor
from llm.base_client import LLMClient
from planner import PlanStep, Tool
from services.event_bus import AsyncEventBus
from memory import UnifiedMemory


class BadLLM(LLMClient):
    def generate(self, prompt: str) -> str:  # pragma: no cover - simple stub
        return "def f(:\n pass"

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - simple stub
        yield self.generate(prompt)


@pytest.mark.asyncio
async def test_syntax_error_from_llm_returns_message() -> None:
    """Тест: обработка синтаксической ошибки от LLM."""
    # Создаем реальный AgentCore с минимальными моками
    bus = AsyncEventBus(workers_per_topic=1)
    planner = MagicMock()  # Мок планировщика (LLM зависимость)
    executor = SimpleCodeExecutor()  # Реальный исполнитель
    search = MagicMock()  # Мок поиска (внешний API)
    memory = UnifiedMemory()  # Реальная память
    code_generator = MagicMock()  # Мок генератора кода (LLM зависимость)
    
    core = AgentCore(
        bus=bus,
        planner=planner,
        executor=executor,
        search=search,
        memory=memory,
        code_generator=code_generator
    )
    
    # Тестируем реальную логику обработки синтаксических ошибок
    try:
        # Попытка выполнить код с синтаксической ошибкой
        result = await core._execute_code(PlanStep(
            tool=Tool.CODE,
            content="def f(:\n pass"
        ))
        
        # Проверяем, что ошибка была обработана корректно
        assert "stderr" in result
        assert "SyntaxError" in result["stderr"] or "invalid syntax" in result["stderr"]
        
    except Exception as e:
        # Если ошибка не обработана, это проблема
        pytest.fail(f"Синтаксическая ошибка не была обработана: {e}")


@pytest.mark.asyncio
async def test_real_code_execution() -> None:
    """Тест: реальное выполнение корректного кода."""
    # Создаем реальный AgentCore
    bus = AsyncEventBus(workers_per_topic=1)
    planner = MagicMock()
    executor = SimpleCodeExecutor()
    search = MagicMock()
    memory = UnifiedMemory()
    code_generator = MagicMock()
    
    core = AgentCore(
        bus=bus,
        planner=planner,
        executor=executor,
        search=search,
        memory=memory,
        code_generator=code_generator
    )
    
    # Тестируем выполнение корректного кода
    result = await core._execute_code(PlanStep(
        tool=Tool.CODE,
        content="print('Hello, World!')"
    ))
    
    # Проверяем, что код выполнился успешно
    assert "stdout" in result
    assert "Hello, World!" in result["stdout"]
    assert "stderr" in result
    assert result["stderr"] == ""


@pytest.mark.asyncio
async def test_code_execution_with_error() -> None:
    """Тест: выполнение кода с ошибкой выполнения."""
    # Создаем реальный AgentCore
    bus = AsyncEventBus(workers_per_topic=1)
    planner = MagicMock()
    executor = SimpleCodeExecutor()
    search = MagicMock()
    memory = UnifiedMemory()
    code_generator = MagicMock()
    
    core = AgentCore(
        bus=bus,
        planner=planner,
        executor=executor,
        search=search,
        memory=memory,
        code_generator=code_generator
    )
    
    # Тестируем выполнение кода с ошибкой выполнения
    result = await core._execute_code(PlanStep(
        tool=Tool.CODE,
        content="print(undefined_variable)"
    ))
    
    # Проверяем, что ошибка была обработана
    assert "stderr" in result
    assert "NameError" in result["stderr"] or "undefined_variable" in result["stderr"]


@pytest.mark.asyncio
async def test_memory_integration() -> None:
    """Тест: интеграция с реальной памятью."""
    # Создаем реальный AgentCore
    bus = AsyncEventBus(workers_per_topic=1)
    planner = MagicMock()
    executor = SimpleCodeExecutor()
    search = MagicMock()
    memory = UnifiedMemory()
    code_generator = MagicMock()
    
    core = AgentCore(
        bus=bus,
        planner=planner,
        executor=executor,
        search=search,
        memory=memory,
        code_generator=code_generator
    )
    
    # Сохраняем данные в память
    test_data = "test memory data"
    memory.remember(test_data)
    
    # Проверяем, что данные действительно сохранены
    retrieved_data = memory.recall()
    assert len(retrieved_data) == 1
    assert retrieved_data[0] == test_data


@pytest.mark.asyncio
async def test_event_bus_integration() -> None:
    """Тест: интеграция с реальной шиной событий."""
    # Создаем реальный AgentCore
    bus = AsyncEventBus(workers_per_topic=1)
    planner = MagicMock()
    executor = SimpleCodeExecutor()
    search = MagicMock()
    memory = UnifiedMemory()
    code_generator = MagicMock()
    
    core = AgentCore(
        bus=bus,
        planner=planner,
        executor=executor,
        search=search,
        memory=memory,
        code_generator=code_generator
    )
    
    # Проверяем, что шина событий работает
    assert core.bus is not None
    assert hasattr(core.bus, 'publish')
    assert hasattr(core.bus, 'subscribe')
    
    # Тестируем подписку на события
    event_received = False
    
    async def test_handler(event):
        nonlocal event_received
        event_received = True
    
    await bus.subscribe("test_topic", test_handler)
    
    # Публикуем событие
    await bus.publish("test_topic", "test_event")
    
    # Даем время на обработку события
    import asyncio
    await asyncio.sleep(0.1)
    
    # Проверяем, что событие было обработано
    assert event_received, "Событие не было обработано"


@pytest.mark.asyncio
async def test_executor_integration() -> None:
    """Тест: интеграция с реальным исполнителем."""
    # Создаем реальный AgentCore
    bus = AsyncEventBus(workers_per_topic=1)
    planner = MagicMock()
    executor = SimpleCodeExecutor()
    search = MagicMock()
    memory = UnifiedMemory()
    code_generator = MagicMock()
    
    core = AgentCore(
        bus=bus,
        planner=planner,
        executor=executor,
        search=search,
        memory=memory,
        code_generator=code_generator
    )
    
    # Проверяем, что исполнитель работает
    assert core.executor is not None
    assert hasattr(core.executor, 'execute')
    
    # Тестируем выполнение простого кода через исполнитель
    result = executor.execute("print('executor test')")
    
    # Проверяем результат
    assert result is not None
    assert hasattr(result, 'stdout')
    assert hasattr(result, 'stderr')
    assert hasattr(result, 'files')
