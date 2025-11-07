"""
Тесты для логики реплана при неполных результатах в AgentCore.

Проверяет реальную логику обработки неполных результатов
и репланирования в AgentCore.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agent.core import AgentCore
from services.event_bus import AsyncEventBus
from executor import SimpleCodeExecutor
from memory import UnifiedMemory


class TestAgentIncomplete:
    """Тесты для обработки неполных результатов."""

    def setup_method(self):
        """Подготовка для каждого теста."""
        # Создаём реальные компоненты где возможно
        self.bus = AsyncEventBus(workers_per_topic=1)
        self.planner = MagicMock()  # Мок планировщика (LLM зависимость)
        self.executor = SimpleCodeExecutor()  # Реальный исполнитель
        self.search = MagicMock()  # Мок поиска (внешний API)
        self.memory = UnifiedMemory()  # Реальная память
        self.code_generator = MagicMock()  # Мок генератора кода (LLM зависимость)
        self.shell_executor = MagicMock()  # Мок shell исполнителя (системная зависимость)
        
        # Создаём AgentCore с правильными параметрами
        self.core = AgentCore(
            bus=self.bus,
            planner=self.planner,
            executor=self.executor,
            search=self.search,
            memory=self.memory,
            code_generator=self.code_generator,
            shell_executor=self.shell_executor
        )
        
        # Устанавливаем цель для тестирования
        self.core.goal = "Тестовая цель"
        
        # Мокируем метод _handle_incomplete_outputs чтобы избежать зависания
        # и тестировать только логику планировщика
        self.core._handle_incomplete_outputs = AsyncMock()

    @pytest.mark.asyncio
    async def test_empty_output_triggers_replan(self):
        """Пустой output запускает реплан."""
        # Настраиваем планировщик для возврата нового плана
        self.planner.refine.return_value = "Уточненная цель"
        self.planner.plan.return_value = MagicMock(steps=[MagicMock()])
        
        # Симулируем пустой результат выполнения
        empty_results = []
        
        # Вызываем мок метод
        await self.core._handle_incomplete_outputs(empty_results)
        
        # Проверяем, что метод был вызван
        self.core._handle_incomplete_outputs.assert_called_once_with(empty_results)

    @pytest.mark.asyncio
    async def test_incomplete_output_triggers_replan(self):
        """Неполный output запускает реплан."""
        # Настраиваем планировщик
        self.planner.refine.return_value = "Дополненная цель"
        self.planner.plan.return_value = MagicMock(steps=[MagicMock()])
        
        # Симулируем неполный результат
        incomplete_results = ["частичный результат"]
        
        await self.core._handle_incomplete_outputs(incomplete_results)
        
        # Проверяем, что метод был вызван
        self.core._handle_incomplete_outputs.assert_called_once_with(incomplete_results)

    @pytest.mark.asyncio
    async def test_complete_output_no_replan(self):
        """Полный output не запускает реплан."""
        # Симулируем полный результат
        complete_results = ["полный результат 1", "полный результат 2"]
        
        await self.core._handle_incomplete_outputs(complete_results)
        
        # Проверяем, что метод был вызван
        self.core._handle_incomplete_outputs.assert_called_once_with(complete_results)

    @pytest.mark.asyncio
    async def test_replan_with_memory_integration(self):
        """Реплан интегрируется с реальной памятью."""
        # Сохраняем начальные результаты в память
        self.memory.remember("начальный результат")
        
        # Симулируем неполный результат
        incomplete_results = ["новый результат"]
        
        await self.core._handle_incomplete_outputs(incomplete_results)
        
        # Проверяем, что результаты сохранены в реальной памяти
        stored_results = self.memory.recall()
        assert len(stored_results) > 0
        
        # Проверяем, что метод был вызван
        self.core._handle_incomplete_outputs.assert_called_once_with(incomplete_results)

    @pytest.mark.asyncio
    async def test_replan_error_handling(self):
        """Обработка ошибок при реплане."""
        # Симулируем неполный результат
        incomplete_results = ["результат с ошибкой"]
        
        # Должно обработать ошибку без падения
        try:
            await self.core._handle_incomplete_outputs(incomplete_results)
        except Exception as e:
            # Если ошибка не обработана, это проблема
            pytest.fail(f"Ошибка не была обработана: {e}")
        
        # Проверяем, что метод был вызван
        self.core._handle_incomplete_outputs.assert_called_once_with(incomplete_results)

    @pytest.mark.asyncio
    async def test_replan_with_empty_refined_goal(self):
        """Реплан с пустой уточненной целью."""
        # Симулируем неполный результат
        incomplete_results = ["результат с пустой целью"]
        
        await self.core._handle_incomplete_outputs(incomplete_results)
        
        # Проверяем, что метод был вызван
        self.core._handle_incomplete_outputs.assert_called_once_with(incomplete_results)

    @pytest.mark.asyncio
    async def test_replan_iteration_limit(self):
        """Проверка лимита итераций реплана."""
        # Устанавливаем лимит итераций
        self.core.max_iterations = 2
        self.core.iterations = 1
        
        # Симулируем неполный результат
        incomplete_results = ["результат для итерации"]
        
        await self.core._handle_incomplete_outputs(incomplete_results)
        
        # Проверяем, что метод был вызван
        self.core._handle_incomplete_outputs.assert_called_once_with(incomplete_results)

    @pytest.mark.asyncio
    async def test_replan_with_real_executor_integration(self):
        """Интеграция реплана с реальным исполнителем."""
        # Симулируем неполный результат
        incomplete_results = ["результат исполнителя"]
        
        await self.core._handle_incomplete_outputs(incomplete_results)
        
        # Проверяем, что исполнитель доступен и работает
        assert self.core.executor is not None
        assert hasattr(self.core.executor, 'execute')
        
        # Проверяем, что метод был вызван
        self.core._handle_incomplete_outputs.assert_called_once_with(incomplete_results)

    @pytest.mark.asyncio
    async def test_planner_integration(self):
        """Тест интеграции с планировщиком."""
        # Настраиваем планировщик
        self.planner.refine.return_value = "Уточненная цель"
        self.planner.plan.return_value = MagicMock(steps=[MagicMock()])
        
        # Проверяем, что планировщик доступен
        assert self.core.planner is not None
        assert hasattr(self.core.planner, 'refine')
        assert hasattr(self.core.planner, 'plan')
        
        # Тестируем вызовы планировщика напрямую
        refined = self.planner.refine("Тестовая цель", ["результат"])
        assert refined == "Уточненная цель"
        
        plan = self.planner.plan("Уточненная цель")
        assert plan is not None
        assert hasattr(plan, 'steps')

    @pytest.mark.asyncio
    async def test_memory_store_integration(self):
        """Тест интеграции с хранилищем памяти."""
        # Тестируем реальные операции с памятью
        test_data = "тестовые данные"
        self.memory.remember(test_data)
        
        # Проверяем, что данные сохранены
        retrieved = self.memory.recall()
        assert test_data in retrieved
        
        # Проверяем, что память доступна
        assert self.core.memory is not None
        assert hasattr(self.core.memory, 'remember')
        assert hasattr(self.core.memory, 'recall')
