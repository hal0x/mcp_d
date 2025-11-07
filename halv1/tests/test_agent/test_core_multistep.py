"""
Тесты для логики многошаговых задач в AgentCore.

Проверяет исправленную логику _needs_continuation, которая теперь
основывается на строгом контракте вместо эвристики.
"""

import pytest
from unittest.mock import Mock
from agent.core import AgentCore
from planner import Plan, PlanStep, Tool
from services.event_bus import AsyncEventBus
from executor import SimpleCodeExecutor
from memory import UnifiedMemory


class TestMultiStepContinuation:
    """Тесты логики продолжения многошаговых задач."""
    
    @pytest.fixture
    def real_core(self):
        """Создает реальный AgentCore для тестирования."""
        bus = AsyncEventBus(workers_per_topic=1)
        planner = Mock()  # Мок планировщика (LLM зависимость)
        executor = SimpleCodeExecutor()  # Реальный исполнитель
        search = Mock()  # Мок поиска (внешний API)
        memory = UnifiedMemory()  # Реальная память
        code_generator = Mock()  # Мок генератора кода (LLM зависимость)
        
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator
        )
        return core
    
    def test_continuation_with_final_step_completed(self, real_core):
        """Тест: продолжение не нужно, если финальный шаг выполнен."""
        plan = Plan(
            steps=[
                PlanStep(tool=Tool.CODE, content="print('step1')", is_final=False),
                PlanStep(
                    tool=Tool.CODE, 
                    content="print('step2')", 
                    is_final=True, 
                    expected_output="step2"
                )
            ],
            context=[]
        )
        results = ["step1", "step2"]
        
        # Должно вернуть False - финальный шаг выполнен
        assert not real_core._needs_continuation(results, plan)
    
    def test_continuation_without_final_step(self, real_core):
        """Тест: продолжение нужно, если нет финального шага."""
        plan = Plan(
            steps=[
                PlanStep(tool=Tool.CODE, content="print('step1')", is_final=False)
            ],
            context=[]
        )
        results = ["step1"]
        
        # Должно вернуть True - нет финального шага
        assert real_core._needs_continuation(results, plan)
    
    def test_continuation_with_unfinished_final_step(self, real_core):
        """Тест: продолжение нужно, если финальный шаг не выполнен."""
        plan = Plan(
            steps=[
                PlanStep(tool=Tool.CODE, content="print('step1')", is_final=False),
                PlanStep(
                    tool=Tool.CODE, 
                    content="print('step2')", 
                    is_final=True, 
                    expected_output="final_result"
                )
            ],
            context=[]
        )
        results = ["step1", "intermediate_result"]
        
        # Должно вернуть True - финальный шаг не выполнен
        assert real_core._needs_continuation(results, plan)
    
    def test_multiple_final_steps_handling(self, real_core):
        """Тест: корректная обработка множественных финальных шагов."""
        plan = Plan(
            steps=[
                PlanStep(tool=Tool.CODE, content="print('step1')", is_final=False),
                PlanStep(
                    tool=Tool.CODE, 
                    content="print('step2')", 
                    is_final=True, 
                    expected_output="step2"
                ),
                PlanStep(
                    tool=Tool.CODE, 
                    content="print('step3')", 
                    is_final=True, 
                    expected_output="step3"
                )
            ],
            context=[]
        )
        results = ["step1", "step2"]
        
        # Должно вернуть True - второй финальный шаг не выполнен
        assert real_core._needs_continuation(results, plan)
    
    def test_all_final_steps_completed(self, real_core):
        """Тест: продолжение не нужно, если все финальные шаги выполнены."""
        plan = Plan(
            steps=[
                PlanStep(tool=Tool.CODE, content="print('step1')", is_final=False),
                PlanStep(
                    tool=Tool.CODE, 
                    content="print('step2')", 
                    is_final=True, 
                    expected_output="step2"
                ),
                PlanStep(
                    tool=Tool.CODE, 
                    content="print('step3')", 
                    is_final=True, 
                    expected_output="step3"
                )
            ],
            context=[]
        )
        results = ["step1", "step2", "step3"]
        
        # Должно вернуть False - все финальные шаги выполнены
        assert not real_core._needs_continuation(results, plan)
    
    def test_empty_results_handling(self, real_core):
        """Тест: корректная обработка пустых результатов."""
        plan = Plan(
            steps=[
                PlanStep(
                    tool=Tool.CODE, 
                    content="print('step1')", 
                    is_final=True
                )
            ],
            context=[]
        )
        results = []
        
        # Должно вернуть False - нет результатов для анализа
        assert not real_core._needs_continuation(results, plan)
    
    def test_empty_plan_handling(self, real_core):
        """Тест: корректная обработка пустого плана."""
        plan = Plan(steps=[], context=[])
        results = ["some_result"]
        
        # Должно вернуть False - нет шагов для выполнения
        assert not real_core._needs_continuation(results, plan)
    
    def test_final_step_without_expected_output(self, real_core):
        """Тест: финальный шаг без expected_output считается выполненным."""
        plan = Plan(
            steps=[
                PlanStep(tool=Tool.CODE, content="print('step1')", is_final=False),
                PlanStep(
                    tool=Tool.CODE, 
                    content="print('step2')", 
                    is_final=True
                    # Без expected_output
                )
            ],
            context=[]
        )
        results = ["step1", "any_result"]
        
        # Должно вернуть False - финальный шаг без expected_output считается выполненным
        assert not real_core._needs_continuation(results, plan)
    
    def test_final_step_with_empty_expected_output(self, real_core):
        """Тест: финальный шаг с пустым expected_output считается выполненным."""
        plan = Plan(
            steps=[
                PlanStep(tool=Tool.CODE, content="print('step1')", is_final=False),
                PlanStep(
                    tool=Tool.CODE, 
                    content="print('step2')", 
                    is_final=True, 
                    expected_output=""
                )
            ],
            context=[]
        )
        results = ["step1", "any_result"]
        
        # Должно вернуть False - финальный шаг с пустым expected_output считается выполненным
        assert not real_core._needs_continuation(results, plan)


class TestStepCompletion:
    """Тесты для метода _is_step_completed."""
    
    @pytest.fixture
    def real_core(self):
        """Создает реальный AgentCore для тестирования."""
        bus = AsyncEventBus(workers_per_topic=1)
        planner = Mock()  # Мок планировщика (LLM зависимость)
        executor = SimpleCodeExecutor()  # Реальный исполнитель
        search = Mock()  # Мок поиска (внешний API)
        memory = UnifiedMemory()  # Реальная память
        code_generator = Mock()  # Мок генератора кода (LLM зависимость)
        
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator
        )
        return core
    
    def test_step_without_expected_output_is_completed(self, real_core):
        """Тест: шаг без expected_output считается выполненным."""
        step = PlanStep(tool=Tool.CODE, content="print('test')")
        results = ["any_result"]
        
        assert real_core._is_step_completed(step, results)
    
    def test_step_with_empty_expected_output_is_completed(self, real_core):
        """Тест: шаг с пустым expected_output считается выполненным."""
        step = PlanStep(
            tool=Tool.CODE, 
            content="print('test')", 
            expected_output=""
        )
        results = ["any_result"]
        
        assert real_core._is_step_completed(step, results)
    
    def test_step_with_matching_expected_output(self, real_core):
        """Тест: шаг с соответствующим expected_output считается выполненным."""
        step = PlanStep(
            tool=Tool.CODE, 
            content="print('test')", 
            expected_output="expected result"
        )
        results = ["some text", "expected result", "other text"]
        
        assert real_core._is_step_completed(step, results)
    
    def test_step_with_partial_matching_expected_output(self, real_core):
        """Тест: шаг с частично соответствующим expected_output считается выполненным."""
        step = PlanStep(
            tool=Tool.CODE, 
            content="print('test')", 
            expected_output="expected result"
        )
        results = ["some text", "this contains expected result inside", "other text"]
        
        assert real_core._is_step_completed(step, results)
    
    def test_step_with_word_based_matching(self, real_core):
        """Тест: шаг с совпадением по отдельным словам считается выполненным."""
        step = PlanStep(
            tool=Tool.CODE, 
            content="print('test')", 
            expected_output="expected result"
        )
        results = ["some text", "this contains expected", "and also result", "other text"]
        
        assert real_core._is_step_completed(step, results)
    
    def test_step_without_matching_expected_output(self, real_core):
        """Тест: шаг без соответствующего expected_output не считается выполненным."""
        step = PlanStep(
            tool=Tool.CODE, 
            content="print('test')", 
            expected_output="expected result"
        )
        results = ["some text", "different text", "other text"]
        
        assert not real_core._is_step_completed(step, results)
    
    def test_step_with_case_insensitive_matching(self, real_core):
        """Тест: шаг с регистронезависимым совпадением считается выполненным."""
        step = PlanStep(
            tool=Tool.CODE, 
            content="print('test')", 
            expected_output="Expected Result"
        )
        results = ["some text", "expected result", "other text"]
        
        assert real_core._is_step_completed(step, results)
    
    def test_step_with_empty_results(self, real_core):
        """Тест: шаг с пустыми результатами не считается выполненным."""
        step = PlanStep(
            tool=Tool.CODE, 
            content="print('test')", 
            expected_output="expected result"
        )
        results = []
        
        assert not real_core._is_step_completed(step, results)
    
    def test_step_completion_integration(self, real_core):
        """Тест: интеграция логики завершения шагов с реальным AgentCore."""
        # Создаем план с несколькими шагами
        plan = Plan(
            steps=[
                PlanStep(
                    tool=Tool.CODE,
                    content="print('step1')",
                    expected_output="step1",
                    is_final=False
                ),
                PlanStep(
                    tool=Tool.CODE,
                    content="print('step2')",
                    expected_output="step2",
                    is_final=True
                )
            ],
            context=[]
        )
        
        # Тестируем первый шаг
        step1 = plan.steps[0]
        results1 = ["step1"]
        assert real_core._is_step_completed(step1, results1)
        
        # Тестируем второй шаг
        step2 = plan.steps[1]
        results2 = ["step1", "step2"]
        assert real_core._is_step_completed(step2, results2)
        
        # Проверяем логику продолжения
        assert not real_core._needs_continuation(results2, plan)
