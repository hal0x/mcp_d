"""
Тесты для логики многошаговых задач в AgentCore.

Проверяет реальную логику _needs_continuation и _is_step_completed
в реальном AgentCore, а не в моках.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from agent.core import AgentCore, Phase
from planner import Plan, PlanStep, Tool
from executor import SimpleCodeExecutor
from memory import UnifiedMemory
from services.event_bus import AsyncEventBus


class TestMultiStepContinuationReal:
    """Тесты логики продолжения многошаговых задач с реальным AgentCore."""
    
    @pytest.fixture
    def real_core(self):
        """Создает реальный AgentCore для тестирования."""
        bus = AsyncEventBus(workers_per_topic=1)
        planner = Mock()  # Мок планировщика
        executor = SimpleCodeExecutor()
        memory = UnifiedMemory()
        
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=None,
            memory=memory,
            code_generator=None
        )
        
        return core
    
    def test_continuation_with_final_step_completed(self, real_core):
        """Тест: продолжение не нужно, если финальный шаг выполнен."""
        # Создаем план с выполненным финальным шагом
        plan = Plan(steps=[
            PlanStep(
                id="step_1",
                tool=Tool.CODE,
                content="print('Step 1')",
                expected_output="Step 1",
                is_final=False
            ),
            PlanStep(
                id="step_2",
                tool=Tool.CODE,
                content="print('Step 2')",
                expected_output="Step 2",
                is_final=True
            )
        ], context=["test context"])
        
        results = ["Step 1", "Step 2"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert not needs_continuation, "Продолжение не нужно, если финальный шаг выполнен"
    
    def test_continuation_without_final_step(self, real_core):
        """Тест: продолжение нужно, если нет финального шага."""
        # Создаем план без финальных шагов
        plan = Plan(steps=[
            PlanStep(
                id="step_1",
                tool=Tool.CODE,
                content="print('Step 1')",
                expected_output="Step 1",
                is_final=False
            )
        ], context=["test context"])
        
        results = ["Step 1"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert needs_continuation, "Продолжение нужно, если нет финального шага"
    
    def test_continuation_with_unfinished_final_step(self, real_core):
        """Тест: продолжение нужно, если финальный шаг не выполнен."""
        # Создаем план с невыполненным финальным шагом
        plan = Plan(steps=[
            PlanStep(
                id="step_1",
                tool=Tool.CODE,
                content="print('Step 1')",
                expected_output="Step 1",
                is_final=False
            ),
            PlanStep(
                id="step_2",
                tool=Tool.CODE,
                content="print('Step 2')",
                expected_output="Step 2",
                is_final=True
            )
        ], context=["test context"])
        
        results = ["Step 1", "Intermediate result"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert needs_continuation, "Продолжение нужно, если финальный шаг не выполнен"
    
    def test_multiple_final_steps_handling(self, real_core):
        """Тест: корректная обработка множественных финальных шагов."""
        # Создаем план с несколькими финальными шагами
        plan = Plan(steps=[
            PlanStep(
                id="step_1",
                tool=Tool.CODE,
                content="print('Step 1')",
                expected_output="Step 1",
                is_final=False
            ),
            PlanStep(
                id="step_2",
                tool=Tool.CODE,
                content="print('Step 2')",
                expected_output="Step 2",
                is_final=True
            ),
            PlanStep(
                id="step_3",
                tool=Tool.CODE,
                content="print('Step 3')",
                expected_output="Step 3",
                is_final=True
            )
        ], context=["test context"])
        
        results = ["Step 1", "Step 2"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert needs_continuation, "Продолжение нужно, если не все финальные шаги выполнены"
    
    def test_all_final_steps_completed(self, real_core):
        """Тест: продолжение не нужно, если все финальные шаги выполнены."""
        # Создаем план со всеми выполненными финальными шагами
        plan = Plan(steps=[
            PlanStep(
                id="step_1",
                tool=Tool.CODE,
                content="print('Step 1')",
                expected_output="Step 1",
                is_final=False
            ),
            PlanStep(
                id="step_2",
                tool=Tool.CODE,
                content="print('Step 2')",
                expected_output="Step 2",
                is_final=True
            ),
            PlanStep(
                id="step_3",
                tool=Tool.CODE,
                content="print('Step 3')",
                expected_output="Step 3",
                is_final=True
            )
        ], context=["test context"])
        
        results = ["Step 1", "Step 2", "Step 3"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert not needs_continuation, "Продолжение не нужно, если все финальные шаги выполнены"
    
    def test_empty_results_handling(self, real_core):
        """Тест: корректная обработка пустых результатов."""
        plan = Plan(steps=[
            PlanStep(
                id="step_1",
                tool=Tool.CODE,
                content="print('Step 1')",
                expected_output="Step 1",
                is_final=True
            )
        ], context=["test context"])
        
        results = []
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert not needs_continuation, "Продолжение не нужно при пустых результатах"
    
    def test_empty_plan_handling(self, real_core):
        """Тест: корректная обработка пустого плана."""
        plan = Plan(steps=[], context=["test context"])
        results = ["some_result"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert not needs_continuation, "Продолжение не нужно при пустом плане"
    
    def test_final_step_without_expected_output(self, real_core):
        """Тест: финальный шаг без expected_output считается выполненным."""
        plan = Plan(steps=[
            PlanStep(
                id="step_1",
                tool=Tool.CODE,
                content="print('Step 1')",
                expected_output="Step 1",
                is_final=False
            ),
            PlanStep(
                id="step_2",
                tool=Tool.CODE,
                content="print('Step 2')",
                expected_output=None,  # Без expected_output
                is_final=True
            )
        ], context=["test context"])
        
        results = ["Step 1", "any_result"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert not needs_continuation, "Финальный шаг без expected_output считается выполненным"
    
    def test_final_step_with_empty_expected_output(self, real_core):
        """Тест: финальный шаг с пустым expected_output считается выполненным."""
        plan = Plan(steps=[
            PlanStep(
                id="step_1",
                tool=Tool.CODE,
                content="print('Step 1')",
                expected_output="Step 1",
                is_final=False
            ),
            PlanStep(
                id="step_2",
                tool=Tool.CODE,
                content="print('Step 2')",
                expected_output="",  # Пустой expected_output
                is_final=True
            )
        ], context=["test context"])
        
        results = ["Step 1", "any_result"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert not needs_continuation, "Финальный шаг с пустым expected_output считается выполненным"


class TestStepCompletionReal:
    """Тесты для метода _is_step_completed с реальным AgentCore."""
    
    @pytest.fixture
    def real_core(self):
        """Создает реальный AgentCore для тестирования."""
        bus = AsyncEventBus(workers_per_topic=1)
        planner = Mock()
        executor = SimpleCodeExecutor()
        memory = UnifiedMemory()
        
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=None,
            memory=memory,
            code_generator=None
        )
        
        return core
    
    def test_step_without_expected_output_is_completed(self, real_core):
        """Тест: шаг без expected_output считается выполненным."""
        step = PlanStep(
            id="step_1",
            tool=Tool.CODE,
            content="print('test')",
            expected_output=None,
            is_final=False
        )
        results = ["any_result"]
        
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг без expected_output должен считаться выполненным"
    
    def test_step_with_empty_expected_output_is_completed(self, real_core):
        """Тест: шаг с пустым expected_output считается выполненным."""
        step = PlanStep(
            id="step_1",
            tool=Tool.CODE,
            content="print('test')",
            expected_output="",
            is_final=False
        )
        results = ["any_result"]
        
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг с пустым expected_output должен считаться выполненным"
    
    def test_step_with_matching_expected_output(self, real_core):
        """Тест: шаг с соответствующим expected_output считается выполненным."""
        step = PlanStep(
            id="step_1",
            tool=Tool.CODE,
            content="print('test')",
            expected_output="expected result",
            is_final=False
        )
        results = ["some text", "expected result", "other text"]
        
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг с соответствующим expected_output должен считаться выполненным"
    
    def test_step_with_partial_matching_expected_output(self, real_core):
        """Тест: шаг с частично соответствующим expected_output считается выполненным."""
        step = PlanStep(
            id="step_1",
            tool=Tool.CODE,
            content="print('test')",
            expected_output="expected result",
            is_final=False
        )
        results = ["some text", "this contains expected result inside", "other text"]
        
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг с частично соответствующим expected_output должен считаться выполненным"
    
    def test_step_without_matching_expected_output(self, real_core):
        """Тест: шаг без соответствующего expected_output не считается выполненным."""
        step = PlanStep(
            id="step_1",
            tool=Tool.CODE,
            content="print('test')",
            expected_output="expected result",
            is_final=False
        )
        results = ["some text", "different text", "other text"]
        
        is_completed = real_core._is_step_completed(step, results)
        assert not is_completed, "Шаг без соответствующего expected_output не должен считаться выполненным"
    
    def test_step_with_case_insensitive_matching(self, real_core):
        """Тест: шаг с регистронезависимым совпадением считается выполненным."""
        step = PlanStep(
            id="step_1",
            tool=Tool.CODE,
            content="print('test')",
            expected_output="Expected Result",
            is_final=False
        )
        results = ["some text", "expected result", "other text"]
        
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг с регистронезависимым совпадением должен считаться выполненным"
    
    def test_step_with_empty_results(self, real_core):
        """Тест: шаг с пустыми результатами не считается выполненным."""
        step = PlanStep(
            id="step_1",
            tool=Tool.CODE,
            content="print('test')",
            expected_output="expected result",
            is_final=False
        )
        results = []
        
        is_completed = real_core._is_step_completed(step, results)
        assert not is_completed, "Шаг с пустыми результатами не должен считаться выполненным"
    
    def test_step_with_non_string_results(self, real_core):
        """Тест: шаг с нестроковыми результатами обрабатывается корректно."""
        step = PlanStep(
            id="step_1",
            tool=Tool.CODE,
            content="print('test')",
            expected_output="expected result",
            is_final=False
        )
        results = ["string result", 42, {"key": "value"}, ["list", "item"]]
        
        # Должно работать без ошибок
        is_completed = real_core._is_step_completed(step, results)
        assert isinstance(is_completed, bool), "Результат должен быть булевым значением"
