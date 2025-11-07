"""
Тесты для исправленной логики многошагового выполнения в AgentCore.

Проверяет реальную функциональность _is_step_completed и _needs_continuation
в реальном AgentCore.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from agent.core import AgentCore, Phase
from planner import Plan, PlanStep, Tool
from executor import SimpleCodeExecutor
from memory import UnifiedMemory
from services.event_bus import AsyncEventBus


class TestFixedStepCompletionReal:
    """Тесты для исправленного метода _is_step_completed с реальным AgentCore."""
    
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
        """Тест: шаг без expected_output считается выполненным (согласно логике AgentCore)."""
        step = PlanStep(
            id="test_step",
            tool=Tool.CODE,
            content="print('hello')",
            expected_output="",  # Пустой expected_output
            is_final=False
        )
        
        results = ["hello"]
        
        # Проверяем реальную логику
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг без expected_output должен считаться выполненным согласно логике AgentCore"
    
    def test_step_with_none_expected_output_is_completed(self, real_core):
        """Тест: шаг с None expected_output считается выполненным (согласно логике AgentCore)."""
        step = PlanStep(
            id="test_step",
            tool=Tool.CODE,
            content="print('hello')",
            expected_output=None,  # None expected_output
            is_final=False
        )
        
        results = ["hello"]
        
        # Проверяем реальную логику
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг с None expected_output должен считаться выполненным согласно логике AgentCore"
    
    def test_step_with_whitespace_expected_output_is_completed(self, real_core):
        """Тест: шаг с пробелами в expected_output считается выполненным (согласно логике AgentCore)."""
        step = PlanStep(
            id="test_step",
            tool=Tool.CODE,
            content="print('hello')",
            expected_output="   ",  # Только пробелы
            is_final=False
        )
        
        results = ["hello"]
        
        # Проверяем реальную логику
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг с пробелами в expected_output должен считаться выполненным согласно логике AgentCore"
    
    def test_step_with_valid_expected_output_is_completed(self, real_core):
        """Тест: шаг с валидным expected_output считается выполненным при совпадении."""
        step = PlanStep(
            id="test_step",
            tool=Tool.CODE,
            content="print('hello')",
            expected_output="hello",
            is_final=False
        )
        
        results = ["hello world"]
        
        # Проверяем реальную логику
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг с соответствующим expected_output должен считаться выполненным"
    
    def test_step_with_partial_match_is_completed(self, real_core):
        """Тест: шаг считается выполненным при частичном совпадении."""
        step = PlanStep(
            id="test_step",
            tool=Tool.CODE,
            content="print('hello world')",
            expected_output="hello world",
            is_final=False
        )
        
        results = ["some text", "hello world", "other text"]
        
        # Проверяем реальную логику
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг должен считаться выполненным при точном совпадении"
    
    def test_step_with_word_based_match_is_completed(self, real_core):
        """Тест: шаг считается выполненным при совпадении по отдельным словам."""
        step = PlanStep(
            id="test_step",
            tool=Tool.CODE,
            content="print('hello world')",
            expected_output="hello world",
            is_final=False
        )
        
        results = ["some text", "this contains hello", "and also world", "other text"]
        
        # Проверяем реальную логику
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг должен считаться выполненным при совпадении по словам"
    
    def test_step_with_case_insensitive_match_is_completed(self, real_core):
        """Тест: шаг считается выполненным при регистронезависимом совпадении."""
        step = PlanStep(
            id="test_step",
            tool=Tool.CODE,
            content="print('Hello World')",
            expected_output="Hello World",
            is_final=False
        )
        
        results = ["some text", "hello world", "other text"]
        
        # Проверяем реальную логику
        is_completed = real_core._is_step_completed(step, results)
        assert is_completed, "Шаг должен считаться выполненным при регистронезависимом совпадении"
    
    def test_step_without_match_is_not_completed(self, real_core):
        """Тест: шаг не считается выполненным без совпадения."""
        step = PlanStep(
            id="test_step",
            tool=Tool.CODE,
            content="print('hello world')",
            expected_output="hello world",
            is_final=False
        )
        
        results = ["some text", "different text", "other text"]
        
        # Проверяем реальную логику
        is_completed = real_core._is_step_completed(step, results)
        assert not is_completed, "Шаг не должен считаться выполненным без совпадения"
    
    def test_step_with_empty_results_is_not_completed(self, real_core):
        """Тест: шаг не считается выполненным с пустыми результатами."""
        step = PlanStep(
            id="test_step",
            tool=Tool.CODE,
            content="print('hello world')",
            expected_output="hello world",
            is_final=False
        )
        
        results = []
        
        # Проверяем реальную логику
        is_completed = real_core._is_step_completed(step, results)
        assert not is_completed, "Шаг не должен считаться выполненным с пустыми результатами"
    
    def test_step_with_non_string_results_handles_gracefully(self, real_core):
        """Тест: шаг корректно обрабатывает нестроковые результаты."""
        step = PlanStep(
            id="test_step",
            tool=Tool.CODE,
            content="print('hello world')",
            expected_output="hello world",
            is_final=False
        )
        
        results = ["string result", 42, {"key": "value"}, ["list", "item"], None]
        
        # Должно работать без ошибок
        is_completed = real_core._is_step_completed(step, results)
        assert isinstance(is_completed, bool), "Результат должен быть булевым значением"


class TestFixedContinuationLogicReal:
    """Тесты для исправленной логики продолжения с реальным AgentCore."""
    
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
    
    def test_continuation_needed_when_final_step_not_completed(self, real_core):
        """Тест: продолжение нужно, когда финальный шаг не выполнен."""
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
        assert needs_continuation, "Продолжение должно быть нужно, когда финальный шаг не выполнен"
    
    def test_continuation_not_needed_when_final_step_completed(self, real_core):
        """Тест: продолжение не нужно, когда финальный шаг выполнен."""
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
        assert not needs_continuation, "Продолжение не должно быть нужно, когда финальный шаг выполнен"
    
    def test_continuation_needed_when_no_final_steps(self, real_core):
        """Тест: продолжение нужно, когда нет финальных шагов."""
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
                is_final=False
            )
        ], context=["test context"])
        
        results = ["Step 1", "Step 2"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert needs_continuation, "Продолжение должно быть нужно, когда нет финальных шагов"
    
    def test_continuation_not_needed_when_all_final_steps_completed(self, real_core):
        """Тест: продолжение не нужно, когда все финальные шаги выполнены."""
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
        assert not needs_continuation, "Продолжение не должно быть нужно, когда все финальные шаги выполнены"
    
    def test_continuation_handles_empty_plan(self, real_core):
        """Тест: продолжение корректно обрабатывает пустой план."""
        plan = Plan(steps=[], context=["test context"])
        results = ["any result"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert not needs_continuation, "Продолжение не должно быть нужно при пустом плане"
    
    def test_continuation_handles_empty_results(self, real_core):
        """Тест: продолжение корректно обрабатывает пустые результаты."""
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
        assert not needs_continuation, "Продолжение не должно быть нужно при пустых результатах"
    
    def test_continuation_with_final_step_no_expected_output(self, real_core):
        """Тест: продолжение не нужно для финального шага без expected_output."""
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
        
        results = ["Step 1", "any result"]
        
        # Проверяем реальную логику
        needs_continuation = real_core._needs_continuation(results, plan)
        assert not needs_continuation, "Продолжение не должно быть нужно для финального шага без expected_output"
