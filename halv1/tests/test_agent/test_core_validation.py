"""
Тесты для валидации результатов выполнения в AgentCore.

Проверяет логику validate_phase, включая повторные попытки
сохранения в память и обработку ошибок.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from agent.core import AgentCore, Phase
from planner import Plan, PlanStep, Tool
from services.event_bus import AsyncEventBus
from executor import SimpleCodeExecutor
from memory import UnifiedMemory
from core.goal_validator import GoalValidator
from events.models import ExecutionCompleted


class TestValidationPhase:
    """Тесты для validate_phase."""
    
    @pytest.fixture
    def real_core(self):
        """Создает реальный AgentCore для тестирования."""
        bus = AsyncEventBus(workers_per_topic=1)
        planner = Mock()  # Мок планировщика (LLM зависимость)
        executor = SimpleCodeExecutor()  # Реальный исполнитель
        search = Mock()  # Мок поиска (внешний API)
        memory = UnifiedMemory()  # Реальная память
        code_generator = Mock()  # Мок генератора кода (LLM зависимость)
        validator = GoalValidator()  # Реальный валидатор
        
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator,
            validator=validator
        )
        
        # Устанавливаем начальное состояние
        core.phase = Phase.EXEC
        core.goal = "test goal"
        core.iterations = 0
        core.max_iterations = 3
        
        return core
    
    @pytest.fixture
    def mock_execution_event(self):
        """Создает мок события ExecutionCompleted."""
        return ExecutionCompleted(
            results=["test result 1", "test result 2"],
            artifact={"stdout": "test", "stderr": "", "files": {}}
        )
    
    @pytest.mark.asyncio
    async def test_validation_phase_successful_save(self, real_core, mock_execution_event):
        """Тест: успешное сохранение результатов в память."""
        # Настраиваем валидатор для успешной валидации
        real_core.validator = Mock()
        real_core.validator.validate.return_value = (True, "Goal achieved")
        
        await real_core.validate_phase(mock_execution_event)
        
        # Проверяем, что результаты были сохранены в реальную память
        stored_results = real_core.memory.recall()
        assert len(stored_results) > 0
        
        # Проверяем, что фаза изменилась на REPORT
        assert real_core.phase == Phase.REPORT
        
        # Проверяем, что цель сброшена
        assert real_core.goal is None
    
    @pytest.mark.asyncio
    async def test_validation_phase_goal_not_satisfied(self, real_core, mock_execution_event):
        """Тест: цель не удовлетворена, требуется продолжение."""
        # Настраиваем валидатор для неуспешной валидации
        real_core.validator = Mock()
        real_core.validator.validate.return_value = (False, "Goal not achieved")
        
        # Настраиваем планировщик для генерации нового плана
        real_core.planner.refine.return_value = "refined goal"
        real_core.planner.plan.return_value = Plan(steps=[PlanStep(tool=Tool.CODE, content="test")], context=[])
        
        await real_core.validate_phase(mock_execution_event)
        
        # Проверяем, что результаты были сохранены в реальную память
        stored_results = real_core.memory.recall()
        assert len(stored_results) > 0
        
        # Проверяем, что фаза изменилась на PLAN
        assert real_core.phase == Phase.PLAN
        
        # Проверяем, что цель была уточнена
        assert real_core.goal == "refined goal"
        
        # Проверяем, что итерация увеличилась
        assert real_core.iterations == 1
    
    @pytest.mark.asyncio
    async def test_validation_phase_max_iterations_reached(self, real_core, mock_execution_event):
        """Тест: достигнут максимальный лимит итераций."""
        # Устанавливаем максимальное количество итераций
        real_core.iterations = 3
        real_core.max_iterations = 3
        
        # Настраиваем валидатор для неуспешной валидации
        real_core.validator = Mock()
        real_core.validator.validate.return_value = (False, "Goal not achieved")
        
        await real_core.validate_phase(mock_execution_event)
        
        # Проверяем, что результаты были сохранены в реальную память
        stored_results = real_core.memory.recall()
        assert len(stored_results) > 0
        
        # Проверяем, что фаза изменилась на REPORT
        assert real_core.phase == Phase.REPORT
        
        # Проверяем, что цель сброшена
        assert real_core.goal is None
        
        # Проверяем, что не было попытки продолжить выполнение
        real_core.planner.refine.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_validation_phase_no_new_results(self, real_core, mock_execution_event):
        """Тест: нет новых результатов для сохранения."""
        # Сначала сохраняем результаты
        real_core.memory.remember("test result 1")
        real_core.memory.remember("test result 2")
        
        # Настраиваем валидатор для успешной валидации
        real_core.validator = Mock()
        real_core.validator.validate.return_value = (True, "Goal achieved")
        
        await real_core.validate_phase(mock_execution_event)
        
        # Проверяем, что фаза изменилась на REPORT
        assert real_core.phase == Phase.REPORT
        
        # Проверяем, что цель сброшена
        assert real_core.goal is None
    
    @pytest.mark.asyncio
    async def test_validation_phase_no_goal(self, real_core, mock_execution_event):
        """Тест: валидация без установленной цели."""
        # Убираем цель
        real_core.goal = None
        
        # Настраиваем валидатор для успешной валидации
        real_core.validator = Mock()
        real_core.validator.validate.return_value = (True, "Goal achieved")
        
        await real_core.validate_phase(mock_execution_event)
        
        # Проверяем, что фаза изменилась на REPORT
        assert real_core.phase == Phase.REPORT
        
        # Проверяем, что не было попытки продолжить выполнение
        real_core.planner.refine.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_validation_phase_planner_no_new_steps(self, real_core, mock_execution_event):
        """Тест: планировщик не генерирует новые шаги."""
        # Настраиваем валидатор для неуспешной валидации
        real_core.validator = Mock()
        real_core.validator.validate.return_value = (False, "Goal not achieved")
        
        # Настраиваем планировщик для возврата пустого плана
        real_core.planner.refine.return_value = "refined goal"
        real_core.planner.plan.return_value = Plan(steps=[], context=[])
        
        await real_core.validate_phase(mock_execution_event)
        
        # Проверяем, что результаты были сохранены в реальную память
        stored_results = real_core.memory.recall()
        assert len(stored_results) > 0
        
        # Проверяем, что фаза изменилась на REPORT
        assert real_core.phase == Phase.REPORT
        
        # Проверяем, что цель сброшена
        assert real_core.goal is None
    
    @pytest.mark.asyncio
    async def test_validation_phase_phase_order_violation(self, real_core, mock_execution_event):
        """Тест: нарушение порядка фаз."""
        # Устанавливаем неправильную фазу
        real_core.phase = Phase.IDLE
        
        with pytest.raises(RuntimeError, match="validate_phase called out of order"):
            await real_core.validate_phase(mock_execution_event)
    
    @pytest.mark.asyncio
    async def test_validation_phase_empty_results(self, real_core):
        """Тест: валидация с пустыми результатами."""
        # Создаем событие с пустыми результатами
        empty_event = ExecutionCompleted(results=[], artifact={"stdout": "", "stderr": "", "files": {}})
        
        # Настраиваем валидатор для успешной валидации
        real_core.validator = Mock()
        real_core.validator.validate.return_value = (True, "Goal achieved")
        
        await real_core.validate_phase(empty_event)
        
        # Проверяем, что фаза изменилась на REPORT
        assert real_core.phase == Phase.REPORT
        
        # Проверяем, что цель сброшена
        assert real_core.goal is None
    
    @pytest.mark.asyncio
    async def test_validation_phase_memory_integration(self, real_core, mock_execution_event):
        """Тест: интеграция с реальной памятью."""
        # Настраиваем валидатор для успешной валидации
        real_core.validator = Mock()
        real_core.validator.validate.return_value = (True, "Goal achieved")
        
        # Проверяем, что память изначально пуста
        initial_results = real_core.memory.recall()
        assert len(initial_results) == 0
        
        await real_core.validate_phase(mock_execution_event)
        
        # Проверяем, что результаты действительно сохранены в памяти
        stored_results = real_core.memory.recall()
        assert len(stored_results) == 2
        assert "test result 1" in stored_results
        assert "test result 2" in stored_results
        
        # Проверяем, что фаза изменилась на REPORT
        assert real_core.phase == Phase.REPORT
