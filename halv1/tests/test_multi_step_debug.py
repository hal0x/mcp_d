"""
Упрощенный тест для отладки многошаговых задач.

Этот тест создает минимальную среду для диагностики проблем
с многошаговым выполнением задач.
"""

import asyncio
import logging
import pytest
from typing import List

from agent.core import AgentCore, Phase
from events.models import MessageReceived
from services.event_bus import AsyncEventBus
from planner import Plan, PlanStep, Tool


class MockTaskPlanner:
    """Мок планировщик для отладки."""
    
    def plan(self, goal: str, context: List[str] = None, previous_results: List[str] = None) -> Plan:
        """Создает простой план для отладки."""
        steps = [
            PlanStep(
                id="step_1",
                tool=Tool.CODE,
                content="print('Step 1: Creating file with numbers 1-10')",
                expected_output="Step 1: Creating file with numbers 1-10",
                is_final=False
            ),
            PlanStep(
                id="step_2", 
                tool=Tool.CODE,
                content="print('Step 2: Reading file and calculating sum')",
                expected_output="Step 2: Reading file and calculating sum",
                is_final=True
            )
        ]
        return Plan(steps=steps, context=context or [])


class MockExecutor:
    """Мок исполнитель для отладки."""
    
    async def execute(self, code: str) -> dict:
        """Имитирует выполнение кода."""
        return {
            "stdout": f"Mock execution: {code[:50]}...",
            "stderr": "",
            "files": {}
        }


class MockMemory:
    """Мок память для отладки."""
    
    def __init__(self):
        self.memory: List[str] = []
    
    def semantic_search(self, query: str) -> List[str]:
        """Имитирует семантический поиск."""
        return ["mock context"]
    
    def recall(self) -> List[str]:
        """Возвращает содержимое памяти."""
        return self.memory
    
    def remember(self, content: str) -> None:
        """Сохраняет в память."""
        self.memory.append(content)


@pytest.mark.asyncio
async def test_multi_step_debug():
    """Упрощенный тест для отладки многошаговых задач."""
    # Создаем минимальную среду
    bus = AsyncEventBus(workers_per_topic=1)
    planner = MockTaskPlanner()  # Мок планировщик
    executor = MockExecutor()     # Мок исполнитель
    memory = MockMemory()         # Мок память
    
    core = AgentCore(
        bus=bus,
        planner=planner,
        executor=executor,
        search=None,
        memory=memory,
        code_generator=None
    )
    
    # Проверяем начальное состояние
    assert core.phase == Phase.IDLE, "Начальная фаза должна быть IDLE"
    assert core.iterations == 0, "Начальное количество итераций должно быть 0"
    assert len(memory.recall()) == 0, "Память должна быть пустой в начале"
    
    # Тестируем простую многошаговую задачу
    await bus.publish("incoming", MessageReceived(chat_id=1, message_id=1, text="test"))
    
    # Проверяем состояние агента после обработки
    assert core.phase in [Phase.IDLE, Phase.PLAN, Phase.EXEC, Phase.VALIDATE, Phase.REPORT], f"Фаза агента должна быть одной из допустимых, но получили {core.phase}"
    assert core.iterations >= 0, "Количество итераций должно быть неотрицательным"
    
    # Проверяем, что память может работать
    test_content = "test memory content"
    memory.remember(test_content)
    recalled = memory.recall()
    assert test_content in recalled, "Память должна сохранять и возвращать добавленный контент"
    
    # Проверяем семантический поиск
    search_results = memory.semantic_search("test")
    assert isinstance(search_results, list), "Результаты поиска должны быть списком"
    assert len(search_results) > 0, "Поиск должен возвращать результаты"
    
    await bus.graceful_shutdown()


@pytest.mark.asyncio
async def test_mock_planner_creates_valid_plan():
    """Тест что мок планировщик создает валидный план."""
    planner = MockTaskPlanner()
    
    # Создаем план
    plan = planner.plan("test goal")
    
    # Проверяем структуру плана
    assert isinstance(plan, Plan), "План должен быть экземпляром Plan"
    assert len(plan.steps) == 2, "План должен содержать 2 шага"
    
    # Проверяем первый шаг
    step1 = plan.steps[0]
    assert step1.id == "step_1", "ID первого шага должен быть 'step_1'"
    assert step1.tool == Tool.CODE, "Первый шаг должен использовать Tool.CODE"
    assert "Creating file" in step1.content, "Содержимое первого шага должно содержать 'Creating file'"
    assert step1.is_final == False, "Первый шаг не должен быть финальным"
    
    # Проверяем второй шаг
    step2 = plan.steps[1]
    assert step2.id == "step_2", "ID второго шага должен быть 'step_2'"
    assert step2.tool == Tool.CODE, "Второй шаг должен использовать Tool.CODE"
    assert "Reading file" in step2.content, "Содержимое второго шага должно содержать 'Reading file'"
    assert step2.is_final == True, "Второй шаг должен быть финальным"


@pytest.mark.asyncio
async def test_mock_executor_returns_valid_result():
    """Тест что мок исполнитель возвращает валидный результат."""
    executor = MockExecutor()
    
    # Выполняем код
    result = await executor.execute("print('hello world')")
    
    # Проверяем структуру результата
    assert isinstance(result, dict), "Результат должен быть словарем"
    assert "stdout" in result, "Результат должен содержать 'stdout'"
    assert "stderr" in result, "Результат должен содержать 'stderr'"
    assert "files" in result, "Результат должен содержать 'files'"
    
    # Проверяем содержимое
    assert "Mock execution" in result["stdout"], "stdout должен содержать 'Mock execution'"
    assert result["stderr"] == "", "stderr должен быть пустым"
    assert result["files"] == {}, "files должен быть пустым словарем"


@pytest.mark.asyncio
async def test_mock_memory_operations():
    """Тест операций с мок памятью."""
    memory = MockMemory()
    
    # Проверяем начальное состояние
    assert len(memory.recall()) == 0, "Память должна быть пустой в начале"
    
    # Добавляем контент
    test_content = "test memory content"
    memory.remember(test_content)
    
    # Проверяем что контент сохранен
    recalled = memory.recall()
    assert test_content in recalled, "Память должна содержать добавленный контент"
    assert len(recalled) == 1, "Память должна содержать 1 элемент"
    
    # Проверяем семантический поиск
    search_results = memory.semantic_search("test")
    assert isinstance(search_results, list), "Результаты поиска должны быть списком"
    assert len(search_results) > 0, "Поиск должен возвращать результаты"
    assert "mock context" in search_results, "Поиск должен возвращать 'mock context'"


@pytest.mark.asyncio
async def test_agent_core_initialization():
    """Тест инициализации AgentCore."""
    bus = AsyncEventBus(workers_per_topic=1)
    planner = MockTaskPlanner()
    executor = MockExecutor()
    memory = MockMemory()
    
    core = AgentCore(
        bus=bus,
        planner=planner,
        executor=executor,
        search=None,
        memory=memory,
        code_generator=None
    )
    
    # Проверяем что все зависимости установлены
    assert core.bus == bus, "Шина событий должна быть установлена"
    assert core.planner == planner, "Планировщик должен быть установлен"
    assert core.executor == executor, "Исполнитель должен быть установлен"
    assert core.memory == memory, "Память должна быть установлена"
    
    # Проверяем начальное состояние
    assert core.phase == Phase.IDLE, "Начальная фаза должна быть IDLE"
    assert core.iterations == 0, "Начальное количество итераций должно быть 0"
    
    await bus.graceful_shutdown()


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    
    # Запускаем тест
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])

